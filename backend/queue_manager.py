"""Queue manager – serialises processing with a background worker.

Features:
  - FIFO queue with optional pause / resume.
  - One task processes at a time; GPU resources are not shared across tasks.
  - Progress events are pushed through a callback for WebSocket broadcast.
  - Cancellation via threading.Event signals to kill running subprocesses.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from typing import Callable, Awaitable

from backend.models import Task, TaskStatus
from backend.processing import TaskCancelledError

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[Task], Awaitable[None]]


class QueueManager:
    def __init__(self, worker_count: int = 1):
        self._queue: deque[Task] = deque()
        self._active_tasks: dict[str, Task] = {}
        self._task_map: dict[str, Task] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._paused = False
        self._worker_count = worker_count
        self._workers: list[asyncio.Task] = []
        self._progress_callbacks: list[ProgressCallback] = []
        self._process_fn: Callable | None = None
        self._running = False

    # ----- Public API -----

    def set_processor(self, fn: Callable) -> None:
        """fn: async (task, progress_cb, cancel_event) -> None"""
        self._process_fn = fn

    def on_progress(self, cb: ProgressCallback) -> None:
        self._progress_callbacks.append(cb)

    async def _notify_progress(self, task: Task) -> None:
        for cb in self._progress_callbacks:
            try:
                await cb(task)
            except Exception:
                logger.exception("Progress callback error")

    def add_task(self, task: Task) -> None:
        self._task_map[task.id] = task
        self._queue.append(task)
        logger.info("Task %s queued (%d in queue)", task.id, len(self._queue))

    def get_task(self, task_id: str) -> Task | None:
        return self._task_map.get(task_id)

    def get_all_tasks(self) -> list[Task]:
        return list(self._task_map.values())

    def cancel_task(self, task_id: str) -> bool:
        task = self._task_map.get(task_id)
        if task is None:
            return False
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            try:
                self._queue.remove(task)
            except ValueError:
                pass
            return True
        if task.status == TaskStatus.PROCESSING:
            # Signal the running worker to abort via threading.Event
            cancel_evt = self._cancel_events.get(task_id)
            if cancel_evt:
                cancel_evt.set()
                logger.info("Task %s: cancel signal sent", task_id)
            return True
        return False

    def remove_task(self, task_id: str) -> bool:
        task = self._task_map.get(task_id)
        if task is None:
            return False
        if task.status in (TaskStatus.PENDING, TaskStatus.CANCELLED, TaskStatus.FAILED):
            self._task_map.pop(task_id, None)
            try:
                self._queue.remove(task)
            except ValueError:
                pass
            return True
        return False

    async def clear_completed(self) -> int:
        removed = 0
        for tid in list(self._task_map.keys()):
            t = self._task_map[tid]
            if t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED):
                self._task_map.pop(tid, None)
                self._cancel_events.pop(tid, None)
                removed += 1
        return removed

    # ----- Lifecycle -----

    async def start(self) -> None:
        self._running = True
        for i in range(self._worker_count):
            worker = asyncio.create_task(self._worker(i), name=f"queue-worker-{i}")
            self._workers.append(worker)

    async def stop(self) -> None:
        self._running = False
        # Cancel all in-progress tasks
        for evt in self._cancel_events.values():
            evt.set()
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ----- Worker -----

    async def _worker(self, idx: int) -> None:
        logger.info("Worker %d started", idx)
        while self._running:
            if self._paused or not self._queue:
                await asyncio.sleep(0.2)
                continue

            task = self._queue.popleft()
            if task.status == TaskStatus.CANCELLED:
                continue

            task.status = TaskStatus.PROCESSING
            self._active_tasks[task.id] = task

            # Create cancellation event for this task
            cancel_event = threading.Event()
            self._cancel_events[task.id] = cancel_event

            await self._notify_progress(task)

            try:
                if self._process_fn is None:
                    raise RuntimeError("No processor registered")

                async def _progress_cb(progress: float, current: int, total: int) -> None:
                    task.progress = progress
                    task.current_frame = current
                    task.total_frames = total
                    await self._notify_progress(task)

                await self._process_fn(task, _progress_cb, cancel_event)

            except asyncio.CancelledError:
                task.status = TaskStatus.CANCELLED
                logger.warning("Task %s cancelled", task.id)
            except TaskCancelledError:
                task.status = TaskStatus.CANCELLED
                task.error_message = "任务已被用户取消"
                logger.info("Task %s cancelled by user", task.id)
            except Exception as exc:
                task.status = TaskStatus.FAILED
                task.error_message = str(exc)
                logger.exception("Task %s failed: %s", task.id, exc)
            finally:
                self._active_tasks.pop(task.id, None)
                self._cancel_events.pop(task.id, None)
                await self._notify_progress(task)

        logger.info("Worker %d stopped", idx)

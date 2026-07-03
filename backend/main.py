"""FastAPI application – REST + WebSocket server for Video Super Resolution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

# Windows: 显式设置 ProactorEventLoop 确保 subprocess 可用
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.models import (
    SUPPORTED_INPUT_FORMATS,
    OutputFormat,
    ProcessMode,
    QualityLevel,
    Task,
    TaskConfig,
    TaskStatus,
    VideoEncoder,
)
from backend.processing import process_task
from backend.queue_manager import QueueManager

# Maximum upload size: 16 GiB
MAX_UPLOAD_BYTES = 16 * 1024 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available on the system PATH."""
    import subprocess as _sp
    ok = True
    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    for exe in ("ffmpeg", "ffprobe"):
        try:
            _sp.check_output([exe, "-version"], stderr=_sp.DEVNULL, **extra_kwargs)
            logger.info("✓ %s 可用", exe)
        except (FileNotFoundError, _sp.CalledProcessError):
            logger.error("✗ 未找到 %s，请安装 FFmpeg 并将其加入系统 PATH", exe)
            ok = False
    if not ok:
        logger.warning(
            "FFmpeg 未安装或不在 PATH 中。视频处理将无法正常工作。\n"
            "下载地址: https://ffmpeg.org/download.html"
        )
    return ok


def _check_nvenc() -> bool:
    """Check if NVENC hardware encoder is usable (driver >= 610.00)."""
    import subprocess as _sp
    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000
    try:
        _sp.check_output(
            ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "color=black:size=320x240:r=1",
             "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
            stderr=_sp.DEVNULL, **extra_kwargs
        )
        logger.info("✓ NVENC (h264_nvenc) 可用")
        return True
    except _sp.CalledProcessError as e:
        logger.error("✗ NVENC 不可用（驱动可能过旧，需要 610.00+）：%s", e)
    except FileNotFoundError:
        logger.error("✗ NVENC 检测失败：ffmpeg 未安装")
    logger.warning(
        "请在 Web 界面中将编码器切换为 CPU 软件编码（libx264）。\n"
        "或更新 NVIDIA 驱动至 610.00 以上版本。"
    )
    return False


# --------------- App ---------------

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Video Super Resolution", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------- Queue ---------------

queue = QueueManager(worker_count=1)
queue.set_processor(process_task)

# --------------- WebSocket manager ---------------

ws_clients: set[WebSocket] = set()


async def broadcast_state(task: Task) -> None:
    """Send updated task state to all connected WebSocket clients."""
    payload = json.dumps({"type": "task_update", "data": task.to_dict()})
    disconnected: set[WebSocket] = set()
    for ws in ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.add(ws)
    ws_clients.difference_update(disconnected)


queue.on_progress(broadcast_state)


# --------------- Startup / Shutdown ---------------

@app.on_event("startup")
async def startup() -> None:
    _check_ffmpeg()
    _check_nvenc()
    await queue.start()
    logger.info("Queue manager started")


@app.on_event("shutdown")
async def shutdown() -> None:
    await queue.stop()
    logger.info("Queue manager stopped")


# --------------- REST: Upload ---------------

@app.post("/api/upload")
async def upload_videos(
    request: Request,
    files: list[UploadFile] = File(...),
    modes: str = Form("super_resolution"),
    scale_factor: int = Form(2),
    quality: str = Form("high"),
    output_format: str = Form("mp4"),
    video_encoder: str = Form("h264_nvenc"),
    batch_size: int = Form(4),
):
    """Upload one or more video files and queue them for processing.

    Files are streamed to disk via chunked writes to avoid loading
    entire videos into RAM – works for multi-GB files.

    *modes*: comma-separated list of processing modes, e.g.
    "super_resolution,denoise"
    """
    if not files:
        raise HTTPException(400, "No files provided")

    # Parse modes from comma-separated string
    mode_list = [ProcessMode(m.strip()) for m in modes.split(",") if m.strip()]
    if not mode_list:
        raise HTTPException(400, "At least one processing mode is required")

    # Enforce overall upload size limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"上传文件总大小超过限制（最大 {MAX_UPLOAD_BYTES // (1024**3)} GB）",
        )

    tasks = []
    for f in files:
        if not f.filename:
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in SUPPORTED_INPUT_FORMATS:
            raise HTTPException(400, f"Unsupported format: {ext}. Supported: {SUPPORTED_INPUT_FORMATS}")

        # Stream-upload: write to disk in 1 MiB chunks to avoid high RAM usage
        safe_name = f"{int(time.time() * 1000)}_{f.filename}"
        input_path = UPLOAD_DIR / safe_name
        with open(input_path, "wb") as dest:
            while chunk := await f.read(1024 * 1024):
                dest.write(chunk)

        # Build config
        try:
            config = TaskConfig(
                modes=mode_list,
                scale_factor=scale_factor,
                quality=QualityLevel(quality),
                output_format=OutputFormat(output_format),
                video_encoder=VideoEncoder(video_encoder),
                batch_size=batch_size,
            )
        except ValueError as e:
            raise HTTPException(400, f"Invalid parameter: {e}")

        if ProcessMode.SUPER_RESOLUTION not in mode_list:
            config.scale_factor = 1

        output_name = f"{Path(f.filename).stem}_processed.{output_format}"
        output_path = OUTPUT_DIR / output_name

        task = Task(
            filename=f.filename,
            input_path=str(input_path),
            output_path=str(output_path),
            config=config,
            created_at=time.time(),
        )
        tasks.append(task.to_dict())
        queue.add_task(task)
        # 立即广播新任务，让前端实时看到
        await broadcast_state(task)

    return {"status": "ok", "tasks": tasks}


# --------------- REST: Queue / Task management ---------------

@app.get("/api/tasks")
async def list_tasks():
    """Return all tasks in the system."""
    return {"tasks": [t.to_dict() for t in queue.get_all_tasks()]}


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = queue.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task.to_dict()


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    ok = queue.cancel_task(task_id)
    if not ok:
        raise HTTPException(404, "Task not found or cannot be cancelled")
    return {"status": "cancelled"}


@app.delete("/api/tasks/{task_id}")
async def remove_task(task_id: str):
    ok = queue.remove_task(task_id)
    if not ok:
        raise HTTPException(404, "Task not found or cannot be removed")
    return {"status": "removed"}


@app.post("/api/tasks/clear")
async def clear_completed():
    count = await queue.clear_completed()
    return {"status": "ok", "removed": count}


# --------------- REST: Output browser ---------------

@app.get("/api/output")
async def list_outputs():
    """List all files in the output directory."""
    items = []
    if OUTPUT_DIR.exists():
        for entry in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if entry.is_file():
                stat = entry.stat()
                items.append({
                    "name": entry.name,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime,
                    "ext": entry.suffix.lower(),
                })
    return {"items": items}


@app.get("/api/output/{filename}")
async def download_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(path, filename=filename)


@app.delete("/api/output/{filename}")
async def delete_output(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    path.unlink()
    return {"status": "deleted"}


# --------------- WebSocket ---------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(ws_clients))

    # Send initial state
    initial = json.dumps({
        "type": "initial_state",
        "data": [t.to_dict() for t in queue.get_all_tasks()],
    })
    await websocket.send_text(initial)

    # Heartbeat: send ping every 15s to keep connection alive
    async def _heartbeat() -> None:
        while True:
            await asyncio.sleep(15)
            try:
                await websocket.send_text(json.dumps({"type": "ping"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(_heartbeat())

    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages (e.g. pong responses)
            try:
                msg = json.loads(data)
                if msg.get("type") == "pong":
                    continue
            except json.JSONDecodeError:
                pass
            logger.debug("WS received: %s", data)
    except WebSocketDisconnect:
        pass
    finally:
        heartbeat_task.cancel()
        ws_clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remain)", len(ws_clients))


# --------------- Static files (production) ---------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# --------------- Entry point ---------------

if __name__ == "__main__":
    import uvicorn

    # Windows: disable reload to avoid subprocess NotImplementedError
    use_reload = sys.platform != "win32"
    if sys.platform == "win32":
        logger.info("Windows detected – auto-reload disabled for subprocess compatibility")

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=use_reload,
        timeout_keep_alive=300,  # 5 min keep-alive for long uploads
    )

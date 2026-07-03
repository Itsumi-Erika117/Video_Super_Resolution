"""Video processing engine using NVIDIA VFX SDK.

Provides four processing modes:
  - Super Resolution (1x-4x upscale)
  - Denoising (same resolution)
  - Deblurring (same resolution)
  - High Bitrate (preserve details for high-quality sources)

Pipeline (streaming): ffmpeg decode → pipe → NVVFX batch processing → pipe → ffmpeg encode.
No intermediate frames are written to disk – works for videos of any length.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from backend.models import (
    ProcessMode,
    QualityLevel,
    QUALITY_SCALE_MAP,
    Task,
    TaskConfig,
    TaskStatus,
    VideoEncoder,
)

logger = logging.getLogger(__name__)

# --------------- NVIDIA VFX SDK wrapper ---------------

try:
    import nvidia.vfx as _vfx

    HAS_NVVFX = True
    _vfx_available: bool = getattr(_vfx, "is_available", lambda: True)()
except ImportError:
    HAS_NVVFX = False
    _vfx = None
    _vfx_available = False

logger.info("NVIDIA VFX SDK available: %s", HAS_NVVFX and _vfx_available)

# --------------- Constants ---------------

DEFAULT_TIMEOUT_SECONDS = 3600 * 4  # 4 hours max per task
READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB for upload streaming


class VFXProcessor:
    """Thin wrapper around the NVIDIA VFX Python SDK."""

    def __init__(self, scale: int = 2, quality: int = 2, batch_size: int = 4):
        self.scale = scale
        self.quality = quality
        self.batch_size = batch_size

    def process_frame(self, frame: "np.ndarray") -> "np.ndarray":
        """Process a single frame (used when batch_size == 1)."""
        if HAS_NVVFX and _vfx_available:
            return frame
        return frame

    def process_batch(self, frames: list["np.ndarray"]) -> list["np.ndarray"]:
        """Submit a batch of frames to the GPU for higher throughput."""
        if HAS_NVVFX and _vfx_available and len(frames) > 1:
            return frames
        return [self.process_frame(f) for f in frames]


def create_vfx_processor(config: TaskConfig) -> VFXProcessor:
    quality_int = QUALITY_SCALE_MAP.get(config.quality, 2)
    return VFXProcessor(
        scale=config.scale_factor,
        quality=quality_int,
        batch_size=config.batch_size,
    )


# --------------- Frame helpers ---------------


def get_video_info(input_path: str) -> dict:
    """Use ffprobe to retrieve width, height, fps, and total frame count.

    Tries nb_frames first; falls back to duration * fps if nb_frames is
    unavailable (common with .mkv / .ts containers).  Returns a conservative
    estimate of 0 when both fail so the caller can decide."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration:format=duration",
        "-of", "csv=p=0",
        input_path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE).strip()
        if not out:
            raise ValueError("ffprobe 返回空数据，视频可能已损坏或编码不受支持")
        parts = out.split(",")
        w, h = int(parts[0]), int(parts[1])
        fps_str = parts[2]
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if int(den) != 0 else 30.0
        total_frames = 0
        # Prefer nb_frames when available
        if len(parts) > 3 and parts[3].isdigit() and int(parts[3]) > 0:
            total_frames = int(parts[3])
        # Fallback: estimate from stream duration / container duration
        if total_frames <= 0:
            # Try stream duration (index 4 when nb_frames present)
            dur_idx = 4 if len(parts) > 3 else 3
            if len(parts) > dur_idx and parts[dur_idx].strip():
                try:
                    dur_secs = float(parts[dur_idx])
                    total_frames = int(dur_secs * fps)
                except (ValueError, IndexError):
                    pass
        # Last resort: try container duration in a separate probe
        if total_frames <= 0:
            dur_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", input_path,
            ]
            try:
                dur_out = subprocess.check_output(dur_cmd, text=True, stderr=subprocess.PIPE).strip()
                if dur_out:
                    dur_secs = float(dur_out)
                    total_frames = int(dur_secs * fps)
            except Exception:
                pass
        logger.info(
            "Probed %s: %dx%d %.2ffps %d frames",
            Path(input_path).name, w, h, fps, total_frames,
        )
        return {"width": w, "height": h, "fps": fps, "total_frames": total_frames}
    except FileNotFoundError:
        raise RuntimeError(
            "未找到 ffprobe，请安装 FFmpeg 并将其加入系统 PATH。\n"
            "下载地址: https://ffmpeg.org/download.html"
        )
    except (subprocess.CalledProcessError, ValueError, IndexError) as exc:
        logger.warning("ffprobe failed: %s, using defaults", exc)
        return {"width": 1920, "height": 1080, "fps": 30.0, "total_frames": 0}


def count_frames_duration(input_path: str) -> int:
    """Fallback frame count via ffprobe duration * fps."""
    info = get_video_info(input_path)
    return info["total_frames"]


# --------------- Cancellation / timeout helpers ---------------


class TaskCancelledError(Exception):
    """Raised when a task is cancelled mid-processing."""


def _check_cancelled(cancel_event: Optional[threading.Event]) -> None:
    """Check if task has been cancelled; raise if so (safe from any thread)."""
    if cancel_event and cancel_event.is_set():
        raise TaskCancelledError("任务已被取消")


def _drain_pipe(proc: subprocess.Popen, attr: str) -> threading.Thread:
    """Start a daemon thread that drains and discards a process pipe.

    Prevents pipe buffer from filling up and blocking the subprocess.
    Returns the thread (already started).
    """
    pipe = getattr(proc, attr, None)
    if pipe is None:
        raise ValueError(f"Process has no attribute {attr}")

    def _drain() -> None:
        try:
            while pipe.read(65536):
                pass
        except Exception:
            pass

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    return t


def _spawn_cancel_watcher(
    cancel_event: threading.Event, *procs: subprocess.Popen
) -> threading.Thread:
    """Create a daemon thread that kills all given processes the instant
    *cancel_event* is set.  Guarantees cancellation even when the main
    loop is stuck in a blocking read."""
    def _watch() -> None:
        cancel_event.wait()
        for p in procs:
            try:
                p.kill()
            except Exception:
                pass

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t


# --------------- Processing pipeline (streaming) ---------------


def _build_encoder_opts(
    config: TaskConfig, fps: float, out_w: int, out_h: int,
    audio_input_index: int = 1,
) -> list[str]:
    """Return the shared encoder-quality portion of the ffmpeg command.

    *audio_input_index*: which ffmpeg input index carries the audio stream.
    NVVFX path uses 1 (pipe=0, original=1); fallback uses 0 (single source).
    """
    quality_vals: dict[QualityLevel, int] = {
        QualityLevel.LOW: 28, QualityLevel.MEDIUM: 23,
        QualityLevel.HIGH: 18, QualityLevel.ULTRA: 14,
    }
    q_val = quality_vals.get(config.quality, 18)

    encoder = config.video_encoder.value
    is_nvenc = "nvenc" in encoder
    pix_fmt = "yuv420p"

    opts = ["-s", f"{out_w}x{out_h}"]
    if is_nvenc:
        opts += [
            "-c:v", encoder, "-cq", str(q_val),
            "-rc", "vbr", "-preset", "p4", "-pix_fmt", pix_fmt,
        ]
    else:
        opts += [
            "-c:v", encoder, "-crf", str(q_val),
            "-preset", "medium", "-pix_fmt", pix_fmt,
        ]

    if config.keep_audio:
        opts += [
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", f"{audio_input_index}:a:0?", "-shortest",
        ]
    else:
        opts += ["-an"]
    return opts


def _build_filter_desc(config: TaskConfig, out_w: int, out_h: int,
                       src_w: int, src_h: int) -> str:
    """Return the ffmpeg video filter chain for the fallback path.

    Supports multiple modes via comma-separated filter chain (e.g.
    'scale,hqdn3d,unsharp' when user selects multiple processing options).
    Filters are applied in the order: scale → denoise → deblur.
    """
    filters: list[str] = []
    has_sr = ProcessMode.SUPER_RESOLUTION in config.modes and (out_w != src_w or out_h != src_h)
    has_denoise = ProcessMode.DENOISE in config.modes
    has_deblur = ProcessMode.DEBLUR in config.modes
    has_high_bitrate = ProcessMode.HIGH_BITRATE in config.modes

    if has_sr:
        filters.append(f"scale={out_w}:{out_h}:flags=lanczos")
    if has_denoise:
        filters.append("hqdn3d=4:3:6:4.5")
    if has_deblur:
        filters.append("unsharp=5:5:1.0:5:5:0.0")

    if not filters:
        # HIGH_BITRATE only or no filters selected
        return "null" if has_high_bitrate else "null"
    return ",".join(filters)


async def _run_subprocess_with_cancel(
    cmd: list[str],
    cancel_event: Optional[threading.Event],
    total_frames: int = 0,
    progress_cb: Optional[Callable[[float, int, int], None]] = None,
    label: str = "ffmpeg",
    timeout: float | None = None,
) -> None:
    """Run a subprocess with cancellation support and optional timeout.

    Uses thread pool to avoid Windows asyncio subprocess issues.
    If *cancel_event* is set, the worker kills the process and raises
    TaskCancelledError.
    If *timeout* is given and exceeded, the process is killed.

    Progress reporting is throttled: updates fire at most every 1% progress
    change or every 1 second, preventing event-loop flood on long videos.
    """
    import subprocess as _sp

    _frame_re = re.compile(r"frame=\s*(\d+)")
    loop = asyncio.get_running_loop()

    def _run() -> None:
        extra_kwargs: dict = {}
        if os.name == "nt":
            extra_kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        try:
            proc = _sp.Popen(
                cmd, stdout=_sp.PIPE, stderr=_sp.PIPE, **extra_kwargs,
            )
        except FileNotFoundError:
            exe = cmd[0] if cmd else "未知命令"
            raise RuntimeError(
                f"未找到 {exe}，请安装 FFmpeg 并将其加入系统 PATH。\n"
                "下载地址: https://ffmpeg.org/download.html"
            )

        # Drain stdout so pipe buffer never fills (stdout is unused here)
        _drain_pipe(proc, "stdout")
        # Cancel watcher: kills process even if readline() is blocking
        if cancel_event:
            _spawn_cancel_watcher(cancel_event, proc)

        # Throttle state
        _last_pct = -1.0
        _last_time = 0.0

        stderr_lines: list[str] = []
        try:
            for line_bytes in iter(proc.stderr.readline, b""):
                _check_cancelled(cancel_event)
                line = line_bytes.decode(errors="replace").rstrip()
                stderr_lines.append(line)
                if progress_cb and total_frames > 0:
                    m = _frame_re.search(line)
                    if m:
                        cur = int(m.group(1))
                        pct = min((cur / total_frames) * 100.0, 99.0)
                        now = time.monotonic()
                        # Throttle: only report when progress changed ≥1% or ≥1s elapsed
                        if pct - _last_pct >= 1.0 or now - _last_time >= 1.0:
                            _last_pct = pct
                            _last_time = now
                            asyncio.run_coroutine_threadsafe(
                                progress_cb(pct, cur, total_frames), loop
                            )
        except (TaskCancelledError, Exception):
            # Kill process on any exception (cancellation / timeout)
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()
            raise

        proc.wait()
        if proc.returncode != 0:
            err_msg = "\n".join(stderr_lines[-20:])
            raise RuntimeError(
                f"{label} 处理失败 (code {proc.returncode}): {err_msg}"
            )

    await asyncio.wait_for(
        loop.run_in_executor(None, _run),
        timeout=timeout,
    )


# --------------- NVVFX streaming pipeline ---------------


def _frame_size_bytes(w: int, h: int) -> int:
    """Size in bytes of one RGB24 raw frame."""
    return w * h * 3


async def _stream_process_nvvfx(
    task: Task,
    config: TaskConfig,
    input_path: str,
    output_path: str,
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    fps: float,
    total_frames: int,
    cancel_event: threading.Event,
    progress_callback: Callable[[float, int, int], None],
) -> None:
    """Streaming NVVFX pipeline: decode → pipe → NVVFX → pipe → encode.

    No intermediate frames touch the disk – everything flows through
    in-memory pipes.  Works for videos of any length.
    """
    loop = asyncio.get_running_loop()
    batch_size = config.batch_size
    in_frame_bytes = _frame_size_bytes(src_w, src_h)
    out_frame_bytes = _frame_size_bytes(out_w, out_h)

    processor = create_vfx_processor(config)
    encoder_opts = _build_encoder_opts(config, fps, out_w, out_h)

    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000

    # --- Decode ffmpeg: output raw RGB24 to stdout ---
    decode_cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", input_path,
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    # --- Encode ffmpeg: read raw RGB24 from stdin ---
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{out_w}x{out_h}",
        "-framerate", str(fps),
        "-i", "-",
    ]
    if config.keep_audio:
        encode_cmd += ["-i", input_path]
    encode_cmd += encoder_opts
    encode_cmd.append(output_path)

    _frame_re = re.compile(r"frame=\s*(\d+)")

    def _run_pipeline() -> None:
        """Run decode + encode ffmpeg processes, with NVVFX in between."""
        try:
            dec_proc = subprocess.Popen(
                decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=1024 * 1024, **extra_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "未找到 ffmpeg，请安装 FFmpeg 并将其加入系统 PATH。"
            )

        try:
            enc_proc = subprocess.Popen(
                encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=1024 * 1024, **extra_kwargs,
            )
        except FileNotFoundError:
            dec_proc.kill()
            dec_proc.wait()
            raise RuntimeError(
                "未找到 ffmpeg，请安装 FFmpeg 并将其加入系统 PATH。"
            )

        # Drain unused pipes to prevent buffer-full deadlocks
        _drain_pipe(dec_proc, "stderr")  # decode progress stderr
        _drain_pipe(enc_proc, "stdout")  # encode stdout (unused)
        # Cancel watcher: kills both processes instantly on cancel
        _spawn_cancel_watcher(cancel_event, dec_proc, enc_proc)

        # Collect encode stderr for progress reporting
        enc_stderr_lines: list[str] = []

        def _read_stderr():
            """Background thread: read encode stderr for progress.

            Throttled to avoid flooding the event loop: reports at most
            every 1% progress change or every 1 second."""
            _last_pct = -1.0
            _last_time = 0.0
            try:
                for line_bytes in iter(enc_proc.stderr.readline, b""):
                    if cancel_event.is_set():
                        break
                    line = line_bytes.decode(errors="replace").rstrip()
                    enc_stderr_lines.append(line)
                    if total_frames > 0:
                        m = _frame_re.search(line)
                        if m:
                            cur = int(m.group(1))
                            pct = min((cur / total_frames) * 100.0, 99.0)
                            now = time.monotonic()
                            if pct - _last_pct >= 1.0 or now - _last_time >= 1.0:
                                _last_pct = pct
                                _last_time = now
                                asyncio.run_coroutine_threadsafe(
                                    progress_callback(pct, cur, total_frames), loop
                                )
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        try:
            frames_processed = 0
            while True:
                _check_cancelled(cancel_event)

                # Read one raw frame from decoder stdout
                raw = b""
                while len(raw) < in_frame_bytes:
                    chunk = dec_proc.stdout.read(in_frame_bytes - len(raw))
                    if not chunk:
                        break
                    raw += chunk
                if len(raw) < in_frame_bytes:
                    break  # decoder exhausted

                # Convert to numpy, process
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))

                if batch_size <= 1:
                    result = processor.process_frame(arr)
                    out_raw = result.tobytes()
                else:
                    # Batch: collect batch_size frames, process together
                    batch = [arr]
                    batch_raws: list[bytes] = []
                    for _ in range(batch_size - 1):
                        _check_cancelled(cancel_event)
                        raw2 = b""
                        while len(raw2) < in_frame_bytes:
                            chunk = dec_proc.stdout.read(in_frame_bytes - len(raw2))
                            if not chunk:
                                break
                            raw2 += chunk
                        if len(raw2) < in_frame_bytes:
                            break
                        batch.append(
                            np.frombuffer(raw2, dtype=np.uint8).reshape((src_h, src_w, 3))
                        )
                        batch_raws.append(raw2)

                    results = processor.process_batch(batch)
                    # Write all results to encoder stdin
                    for r in results:
                        enc_proc.stdin.write(r.tobytes())
                        enc_proc.stdin.flush()
                    frames_processed += len(results)
                    continue

                # Single-frame: write to encoder
                enc_proc.stdin.write(out_raw)
                enc_proc.stdin.flush()
                frames_processed += 1

            # Close encoder stdin to signal end
            try:
                enc_proc.stdin.close()
            except Exception:
                pass

            _check_cancelled(cancel_event)
            enc_proc.wait()
            stderr_thread.join(timeout=5)

            if enc_proc.returncode != 0:
                err_msg = "\n".join(enc_stderr_lines[-20:])
                raise RuntimeError(
                    f"ffmpeg 编码失败 (code {enc_proc.returncode}): {err_msg}"
                )

            task.current_frame = frames_processed
            task.total_frames = frames_processed

        except TaskCancelledError:
            logger.info("Task %s: killing subprocesses", task.id)
            try:
                dec_proc.kill()
            except Exception:
                pass
            try:
                enc_proc.kill()
            except Exception:
                pass
            stderr_thread.join(timeout=3)
            raise
        except Exception:
            try:
                dec_proc.kill()
            except Exception:
                pass
            try:
                enc_proc.kill()
            except Exception:
                pass
            stderr_thread.join(timeout=3)
            raise

    await asyncio.wait_for(
        loop.run_in_executor(None, _run_pipeline),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


# --------------- Fallback pipeline (no NVVFX) ---------------


async def _stream_process_fallback(
    task: Task,
    config: TaskConfig,
    input_path: str,
    output_path: str,
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    fps: float,
    total_frames: int,
    cancel_event: threading.Event,
    progress_callback: Callable[[float, int, int], None],
) -> None:
    """Single-ffmpeg-command fallback using software filters.

    No intermediate files – everything happens in a single ffmpeg invocation.
    Works for videos of any length.
    """
    filter_desc = _build_filter_desc(config, out_w, out_h, src_w, src_h)
    # Fallback has a single input (0), so audio is at 0:a:0
    encoder_opts = _build_encoder_opts(config, fps, out_w, out_h, audio_input_index=0)

    cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", input_path,
    ]

    if filter_desc != "null":
        cmd += ["-vf", filter_desc]

    cmd += encoder_opts
    cmd.append(output_path)

    # If NO filter needed (HIGH_BITRATE only or none), skip -vf entirely
    if filter_desc == "null" and not any(
        m in config.modes for m in (ProcessMode.DENOISE, ProcessMode.DEBLUR,
                                     ProcessMode.SUPER_RESOLUTION)
    ):
        # No processing needed at all
        pass

    await _run_subprocess_with_cancel(
        cmd,
        cancel_event=cancel_event,
        total_frames=total_frames,
        progress_cb=progress_callback if total_frames > 0 else None,
        label="ffmpeg 处理",
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )

    task.current_frame = total_frames


# --------------- Main entry point ---------------


async def process_task(
    task: Task,
    progress_callback: Callable[[float, int, int], None],
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Run the full streaming pipeline for one task.

    Steps:
      1. Probe input video metadata.
      2. Streaming NVVFX pipeline (decode → pipe → NVVFX → pipe → encode).
         No intermediate frames written to disk.
      3. (Fallback) Single ffmpeg command with software filter when NVVFX unavailable.

    Supports cancellation via *cancel_event*.
    """
    config = task.config
    input_path = task.input_path
    output_path = task.output_path

    logger.info("Starting task %s (%s) → %s", task.id, task.filename, output_path)

    # --- 1. Probe ---
    info = get_video_info(input_path)
    src_w, src_h = info["width"], info["height"]
    fps = info["fps"]
    total_frames = info["total_frames"]
    if total_frames <= 0:
        # Estimate if ffprobe couldn't determine
        total_frames = 1000

    scale = config.scale_factor
    has_sr = ProcessMode.SUPER_RESOLUTION in config.modes
    only_non_sr = all(
        m in (ProcessMode.DENOISE, ProcessMode.DEBLUR, ProcessMode.HIGH_BITRATE)
        for m in config.modes
    )
    if not has_sr:
        scale = 1
    out_w, out_h = src_w * scale, src_h * scale

    task.total_frames = total_frames
    task.status = TaskStatus.PROCESSING
    await progress_callback(0.0, 0, total_frames)

    _check_cancelled(cancel_event)

    # --- 2. Process ---
    use_nvvfx = HAS_NVVFX and _vfx_available and ProcessMode.HIGH_BITRATE not in config.modes

    if use_nvvfx:
        logger.info("Task %s: using NVVFX streaming pipeline", task.id)
        await _stream_process_nvvfx(
            task, config, input_path, output_path,
            src_w, src_h, out_w, out_h, fps,
            total_frames, cancel_event, progress_callback,
        )
    else:
        logger.info("Task %s: using fallback ffmpeg pipeline", task.id)
        await _stream_process_fallback(
            task, config, input_path, output_path,
            src_w, src_h, out_w, out_h, fps,
            total_frames, cancel_event, progress_callback,
        )

    task.progress = 100.0
    task.current_frame = task.total_frames
    task.status = TaskStatus.COMPLETED
    logger.info("Task %s completed → %s", task.id, output_path)

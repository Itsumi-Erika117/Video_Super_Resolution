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
    import nvvfx as _vfx

    HAS_NVVFX = True
except ImportError:
    HAS_NVVFX = False
    _vfx = None

# NVVFX SDK requires a GPU array library (CuPy or PyTorch) for DLPack tensors
try:
    import cupy as _cp

    HAS_CUPY = True
except ImportError:
    _cp = None
    HAS_CUPY = False

try:
    import torch as _torch

    HAS_TORCH = True
except ImportError:
    _torch = None
    HAS_TORCH = False

_vfx_available = HAS_NVVFX and (HAS_CUPY or HAS_TORCH)

logger.info(
    "NVIDIA VFX SDK available: %s (import=%s, cupy=%s, torch=%s)",
    _vfx_available, HAS_NVVFX, HAS_CUPY, HAS_TORCH,
)

# --------------- RIFE (rife-ncnn-vulkan CLI) ---------------

# Search for rife-ncnn-vulkan.exe in standard locations
_RIFE_EXE: Optional[str] = None
_RIFE_MODEL_DIR: Optional[str] = None

_tools_dir = Path(__file__).resolve().parent.parent / "tools" / "rife-ncnn-vulkan"
_candidate_exe = _tools_dir / "rife-ncnn-vulkan.exe"
if _candidate_exe.is_file():
    _RIFE_EXE = str(_candidate_exe)
    # Model dirs are siblings of the exe (rife-v4, rife-v4.6, etc.)
    _RIFE_MODEL_DIR = str(_tools_dir)
    logger.info("RIFE CLI found: %s", _RIFE_EXE)
else:
    import shutil as _shutil
    _path_exe = _shutil.which("rife-ncnn-vulkan")
    if _path_exe:
        _RIFE_EXE = _path_exe
        logger.info("RIFE CLI found in PATH: %s", _RIFE_EXE)
    else:
        logger.warning(
            "RIFE CLI (rife-ncnn-vulkan.exe) not found. "
            "Download: https://github.com/nihui/rife-ncnn-vulkan/releases\n"
            "Extract to: %s", _tools_dir
        )

HAS_RIFE = _RIFE_EXE is not None

RIFE_SUPPORTED_MULTIPLIERS = (2, 3, 4)


def _validate_rife_multiplier(multiplier: int) -> None:
    if multiplier not in RIFE_SUPPORTED_MULTIPLIERS:
        raise ValueError(
            f"Unsupported frame multiplier: {multiplier}. "
            f"Supported: {RIFE_SUPPORTED_MULTIPLIERS}"
        )


# --------------- Constants ---------------

DEFAULT_TIMEOUT_SECONDS = 3600 * 4  # 4 hours max per task
READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB for upload streaming


class VFXProcessor:
    """Wraps NVIDIA VFX SDK for GPU-accelerated video processing.

    Supports Super Resolution (upscale), Denoising, and Deblurring modes
    via nvvfx.VideoSuperRes.  Multiple modes are chained: SR runs first
    (changing output dimensions), then denoise/deblur on the upscaled result.
    """

    def __init__(self, config: TaskConfig, src_w: int, src_h: int):
        self._effects: list = []
        self._out_w: int = 0
        self._out_h: int = 0
        self._build_effects(config, src_w, src_h)

    @property
    def output_width(self) -> int:
        return self._out_w

    @property
    def output_height(self) -> int:
        return self._out_h

    @staticmethod
    def _to_gpu(frame: "np.ndarray") -> "Any":
        """Convert numpy uint8 (H, W, 3) to GPU float32 (3, H, W) in [0, 1]."""
        if HAS_CUPY:
            gpu = _cp.asarray(frame, dtype=_cp.float32) / 255.0
            return gpu.transpose(2, 0, 1).copy()  # .copy() ensures contiguous memory
        elif HAS_TORCH:
            gpu = _torch.from_numpy(frame).float().cuda() / 255.0
            return gpu.permute(2, 0, 1).contiguous()
        raise RuntimeError("No GPU array library available (install cupy or torch)")

    @staticmethod
    def _to_numpy(gpu: "Any") -> "np.ndarray":
        """Convert GPU float32 (3, H, W) in [0, 1] back to numpy uint8 (H, W, 3)."""
        if HAS_CUPY:
            gpu = gpu.transpose(1, 2, 0) * 255.0
            gpu = _cp.clip(gpu, 0, 255)
            return _cp.asnumpy(gpu.astype(_cp.uint8))
        elif HAS_TORCH:
            gpu = gpu.permute(1, 2, 0) * 255.0
            gpu = gpu.clamp(0, 255).to(_torch.uint8)
            return gpu.cpu().numpy()
        raise RuntimeError("No GPU array library available (install cupy or torch)")

    @staticmethod
    def _from_dlpack(dlpack_capsule: "Any") -> "Any":
        """Convert DLPack capsule from NVVFX output to a GPU array."""
        if HAS_CUPY:
            return _cp.from_dlpack(dlpack_capsule).copy()
        elif HAS_TORCH:
            return _torch.utils.dlpack.from_dlpack(dlpack_capsule).clone()
        raise RuntimeError("No GPU array library available (install cupy or torch)")

    def _build_effects(self, config: TaskConfig, src_w: int, src_h: int) -> None:
        """Create and load NVVFX VideoSuperRes effects based on processing modes."""
        if not HAS_NVVFX or not _vfx_available:
            return

        has_sr = ProcessMode.SUPER_RESOLUTION in config.modes
        has_denoise = ProcessMode.DENOISE in config.modes
        has_deblur = ProcessMode.DEBLUR in config.modes
        has_hb = ProcessMode.HIGH_BITRATE in config.modes

        q = config.quality
        QL = _vfx.VideoSuperRes.QualityLevel
        scale = config.scale_factor if has_sr else 1

        # Super Resolution (always first – changes dimensions)
        if has_sr:
            sr_map: dict[QualityLevel, "Any"] = {
                QualityLevel.LOW: QL.LOW,
                QualityLevel.MEDIUM: QL.MEDIUM,
                QualityLevel.HIGH: QL.HIGH,
                QualityLevel.ULTRA: QL.ULTRA,
            }
            effect = _vfx.VideoSuperRes(quality=sr_map.get(q, QL.HIGH), device=0)
            effect.output_width = src_w * scale
            effect.output_height = src_h * scale
            effect.load()
            self._out_w = effect.output_width or (src_w * scale)
            self._out_h = effect.output_height or (src_h * scale)
            self._effects.append(effect)

        # Denoise (applied after SR if SR is also selected)
        if has_denoise:
            dn_map: dict[QualityLevel, "Any"] = {
                QualityLevel.LOW: QL.DENOISE_LOW,
                QualityLevel.MEDIUM: QL.DENOISE_MEDIUM,
                QualityLevel.HIGH: QL.DENOISE_HIGH,
                QualityLevel.ULTRA: QL.DENOISE_ULTRA,
            }
            cur_w = self._out_w or src_w
            cur_h = self._out_h or src_h
            effect = _vfx.VideoSuperRes(quality=dn_map.get(q, QL.DENOISE_HIGH), device=0)
            effect.output_width = cur_w
            effect.output_height = cur_h
            effect.load()
            if not has_sr:
                self._out_w = effect.output_width or cur_w
                self._out_h = effect.output_height or cur_h
            self._effects.append(effect)

        # Deblur (applied after SR / denoise)
        if has_deblur:
            db_map: dict[QualityLevel, "Any"] = {
                QualityLevel.LOW: QL.DEBLUR_LOW,
                QualityLevel.MEDIUM: QL.DEBLUR_MEDIUM,
                QualityLevel.HIGH: QL.DEBLUR_HIGH,
                QualityLevel.ULTRA: QL.DEBLUR_ULTRA,
            }
            cur_w = self._out_w or src_w
            cur_h = self._out_h or src_h
            effect = _vfx.VideoSuperRes(quality=db_map.get(q, QL.DEBLUR_HIGH), device=0)
            effect.output_width = cur_w
            effect.output_height = cur_h
            effect.load()
            if not has_sr and not has_denoise:
                self._out_w = effect.output_width or cur_w
                self._out_h = effect.output_height or cur_h
            self._effects.append(effect)

        # High Bitrate (applied after other effects)
        if has_hb:
            hb_map: dict[QualityLevel, "Any"] = {
                QualityLevel.LOW: QL.HIGHBITRATE_LOW,
                QualityLevel.MEDIUM: QL.HIGHBITRATE_MEDIUM,
                QualityLevel.HIGH: QL.HIGHBITRATE_HIGH,
                QualityLevel.ULTRA: QL.HIGHBITRATE_ULTRA,
            }
            cur_w = self._out_w or src_w
            cur_h = self._out_h or src_h
            effect = _vfx.VideoSuperRes(quality=hb_map.get(q, QL.HIGHBITRATE_HIGH), device=0)
            effect.output_width = cur_w
            effect.output_height = cur_h
            effect.load()
            if not has_sr and not has_denoise and not has_deblur:
                self._out_w = effect.output_width or cur_w
                self._out_h = effect.output_height or cur_h
            self._effects.append(effect)

        logger.info(
            "VFXProcessor created: %d effect(s), output %dx%d",
            len(self._effects), self._out_w, self._out_h,
        )

    def process_frame(self, frame: "np.ndarray") -> "np.ndarray":
        """Process a single frame through all NVVFX effects."""
        gpu = self._to_gpu(frame)
        for effect in self._effects:
            output = effect.run(gpu)
            new_gpu = self._from_dlpack(output.image)
            del gpu  # free previous GPU tensor
            gpu = new_gpu
        result = self._to_numpy(gpu)
        del gpu  # free final GPU tensor
        return result

    def process_batch(self, frames: list["np.ndarray"]) -> list["np.ndarray"]:
        """Process a batch of frames (processes individually; batch_size=1 recommended)."""
        return [self.process_frame(f) for f in frames]

    def close(self) -> None:
        """Release GPU resources held by NVVFX effects."""
        for effect in self._effects:
            try:
                effect.close()
            except Exception:
                pass
        self._effects.clear()


def create_vfx_processor(config: TaskConfig, src_w: int = 0, src_h: int = 0) -> "Optional[VFXProcessor]":
    """Create a VFXProcessor or return None if NVVFX is unavailable."""
    if not _vfx_available:
        return None
    try:
        return VFXProcessor(config, src_w, src_h)
    except Exception as exc:
        logger.warning("Failed to create VFX processor: %s", exc)
        return None


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
    output_fps: Optional[float] = None,
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

    # Explicit output frame rate (needed when interpolation changes FPS)
    if output_fps is not None:
        opts += ["-r", str(output_fps)]

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
                       src_w: int, src_h: int,
                       fps: float = 30.0) -> str:
    """Return the ffmpeg video filter chain for the fallback path.

    Supports multiple modes via comma-separated filter chain (e.g.
    'scale,hqdn3d,unsharp' when user selects multiple processing options).
    Filters are applied in the order: minterpolate → scale → denoise → deblur.
    """
    filters: list[str] = []
    has_interp = ProcessMode.FRAME_INTERPOLATION in config.modes
    has_sr = ProcessMode.SUPER_RESOLUTION in config.modes and (out_w != src_w or out_h != src_h)
    has_denoise = ProcessMode.DENOISE in config.modes
    has_deblur = ProcessMode.DEBLUR in config.modes
    has_high_bitrate = ProcessMode.HIGH_BITRATE in config.modes

    # Frame interpolation via ffmpeg minterpolate (runs first, changes FPS)
    if has_interp:
        target_fps = fps * config.frame_multiplier
        filters.append(
            f"minterpolate=fps={target_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        )

    if has_sr:
        filters.append(f"scale={out_w}:{out_h}:flags=lanczos")
    if has_denoise:
        filters.append("hqdn3d=4:3:6:4.5")
    if has_deblur:
        filters.append("unsharp=5:5:1.0:5:5:0.0")

    if not filters:
        return "null"
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
    processor: "VFXProcessor",
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

    *processor* must be a valid, already-created VFXProcessor.
    """
    loop = asyncio.get_running_loop()
    in_frame_bytes = _frame_size_bytes(src_w, src_h)

    # NVVFX effect determines actual output dimensions
    act_w = processor.output_width or out_w
    act_h = processor.output_height or out_h
    logger.info(
        "NVVFX output dimensions: %dx%d (requested %dx%d)",
        act_w, act_h, out_w, out_h,
    )
    out_frame_bytes = _frame_size_bytes(act_w, act_h)

    encoder_opts = _build_encoder_opts(config, fps, act_w, act_h)

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
        "-video_size", f"{act_w}x{act_h}",
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

                # Periodic CuPy GPU memory pool cleanup (every 100 frames)
                if HAS_CUPY and frames_processed > 0 and frames_processed % 100 == 0:
                    _cp.get_default_memory_pool().free_all_blocks()

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

                # Process frame through NVVFX
                result = processor.process_frame(arr)
                out_raw = result.tobytes()

                # Write to encoder
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
    filter_desc = _build_filter_desc(config, out_w, out_h, src_w, src_h, fps)
    # Fallback has a single input (0), so audio is at 0:a:0
    has_interp = ProcessMode.FRAME_INTERPOLATION in config.modes
    out_fps = fps * config.frame_multiplier if has_interp else None
    encoder_opts = _build_encoder_opts(config, fps, out_w, out_h,
                                       audio_input_index=0,
                                       output_fps=out_fps)

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


# --------------- RIFE pipeline (two-phase: extract → CLI → NVVFX/encode) ---------------


async def _run_rife_cli(
    input_dir: str,
    output_dir: str,
    multiplier: int = 2,
    cancel_event: Optional[threading.Event] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run rife-ncnn-vulkan CLI in directory mode.

    Generates (multiplier-1) interpolated frames between each pair of input frames.
    Output files are numbered 1..(multiplier-1)*(N-1) in the output directory.
    """
    if not _RIFE_EXE:
        raise RuntimeError("RIFE CLI not found")

    _validate_rife_multiplier(multiplier)

    # RIFE's -n is "target total interpolated frames", default N*2 (≈3x!).
    # We want (multiplier-1) intermediates per pair, so target is:
    #   (multiplier-1) * (input_frame_count - 1)
    input_files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".webp"))
    )
    input_count = len(input_files)
    if input_count < 2:
        raise RuntimeError(f"Input directory has only {input_count} frames (need >=2)")

    target_frames = (multiplier - 1) * (input_count - 1)
    n_args = ["-n", str(max(target_frames, 1))]
    logger.info(
        "RIFE -n %d (multiplier=%dx, input=%d frames, target=%d interp)",
        max(target_frames, 1), multiplier, input_count, target_frames,
    )

    model_arg: list[str] = []
    if _RIFE_MODEL_DIR and os.path.isdir(_RIFE_MODEL_DIR):
        # Use rife-v4.6 model if available, otherwise default to auto-detect
        candidate = os.path.join(_RIFE_MODEL_DIR, "rife-v4.6")
        if os.path.isdir(candidate):
            model_arg = ["-m", candidate]

    cmd = [
        _RIFE_EXE,
        "-i", input_dir,
        "-o", output_dir,
        "-f", "%08d.png",
        "-j", "2:4:4",
    ] + n_args + model_arg

    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000

    loop = asyncio.get_running_loop()

    def _run() -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **extra_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError(f"未找到 RIFE CLI: {_RIFE_EXE}")

        if cancel_event:
            _spawn_cancel_watcher(cancel_event, proc)

        # Drain stdout
        _drain_pipe(proc, "stdout")

        stderr_lines: list[str] = []
        try:
            for line_bytes in iter(proc.stderr.readline, b""):
                _check_cancelled(cancel_event)
                line = line_bytes.decode(errors="replace").rstrip()
                stderr_lines.append(line)
                logger.debug("RIFE: %s", line)
        except (TaskCancelledError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()
            raise

        proc.wait()
        if proc.returncode != 0:
            err_msg = "\n".join(stderr_lines[-30:])
            raise RuntimeError(f"RIFE 插帧失败 (code {proc.returncode}): {err_msg}")

    await asyncio.wait_for(
        loop.run_in_executor(None, _run),
        timeout=timeout,
    )


async def _extract_frames_to_dir(
    input_path: str,
    output_dir: str,
    total_frames: int,
    fps: float = 30.0,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
) -> None:
    # Frame extraction: use software decoding + -r to force strict
    # sequential display-order output, preventing frame reordering.
    out_pattern = os.path.join(output_dir, "frame_%08d.png")
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-r", str(fps),
        "-q:v", "1",
        out_pattern,
    ]

    _frame_re = re.compile(r"frame=\s*(\d+)")
    loop = asyncio.get_running_loop()

    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000

    def _run() -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **extra_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError("未找到 ffmpeg")

        if cancel_event:
            _spawn_cancel_watcher(cancel_event, proc)

        _drain_pipe(proc, "stdout")

        _last_pct = -1.0
        _last_time = 0.0

        try:
            for line_bytes in iter(proc.stderr.readline, b""):
                _check_cancelled(cancel_event)
                line = line_bytes.decode(errors="replace").rstrip()
                if progress_callback and total_frames > 0:
                    m = _frame_re.search(line)
                    if m:
                        cur = int(m.group(1))
                        pct = min((cur / total_frames) * 10.0, 10.0)  # 0-10%
                        now = time.monotonic()
                        if pct - _last_pct >= 0.5 or now - _last_time >= 1.0:
                            _last_pct = pct
                            _last_time = now
                            asyncio.run_coroutine_threadsafe(
                                progress_callback(pct, cur, total_frames), loop
                            )
        except (TaskCancelledError, Exception):
            try:
                proc.kill()
            except Exception:
                pass
            proc.wait()
            raise

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 帧提取失败 (code {proc.returncode})")

    await asyncio.wait_for(
        loop.run_in_executor(None, _run),
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )


async def _process_rife_pipeline(
    task: Task,
    config: TaskConfig,
    processor: "Optional[VFXProcessor]",
    input_path: str,
    output_path: str,
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    fps: float,
    total_frames: int,
    cancel_event: threading.Event,
    progress_callback: Callable[[float, int, int], None],
) -> None:
    """Two-phase RIFE pipeline:

    Phase 1: ffmpeg extract original frames to temp dir (0-10%)
    Phase 2: rife-ncnn-vulkan CLI interpolate (10-70%)
    Phase 3: Read & interleave original + interpolated frames, process with
             NVVFX (or pass-through), encode (70-100%)
    """
    import tempfile as _tempfile
    import shutil as _shutil_mod

    loop = asyncio.get_running_loop()
    multiplier = config.frame_multiplier
    _validate_rife_multiplier(multiplier)

    has_nvvfx = processor is not None
    out_fps = fps * multiplier

    if has_nvvfx:
        act_w = processor.output_width or out_w
        act_h = processor.output_height or out_h
    else:
        act_w, act_h = src_w, src_h

    # Total output frames after interpolation
    out_total = total_frames * multiplier - (multiplier - 1)
    interp_per_pair = multiplier - 1
    out_frame_bytes = _frame_size_bytes(act_w, act_h)
    in_frame_bytes = _frame_size_bytes(src_w, src_h)

    logger.info(
        "RIFE pipeline: %dx%d %dx → %d output frames, NVVFX=%s",
        src_w, src_h, multiplier, out_total, has_nvvfx,
    )

    # Create temp directories
    base_tmp = _tempfile.mkdtemp(prefix="rife_")
    orig_dir = os.path.join(base_tmp, "orig")
    interp_dir = os.path.join(base_tmp, "interp")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(interp_dir, exist_ok=True)

    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000

    encoder_opts = _build_encoder_opts(config, out_fps, act_w, act_h)
    _frame_re = re.compile(r"frame=\s*(\d+)")

    async def _cleanup() -> None:
        try:
            _shutil_mod.rmtree(base_tmp, ignore_errors=True)
        except Exception:
            pass

    try:
        # === Phase 1: Extract original frames (0-10%) ===
        await progress_callback(0.0, 0, out_total)
        await _extract_frames_to_dir(
            input_path, orig_dir, total_frames, fps=fps,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        _check_cancelled(cancel_event)

        # List original frames (sorted)
        orig_files = sorted(
            f for f in os.listdir(orig_dir) if f.endswith(".png")
        )
        if not orig_files:
            raise RuntimeError("帧提取失败：未生成任何 PNG 文件")
        logger.info("Extracted %d original frames to %s", len(orig_files), orig_dir)

        # === Phase 2: RIFE CLI interpolate (10-70%) ===
        await progress_callback(10.0, 0, out_total)
        await _run_rife_cli(
            orig_dir, interp_dir, multiplier,
            cancel_event=cancel_event,
        )
        _check_cancelled(cancel_event)
        await progress_callback(70.0, 0, out_total)

        # List interpolated frames (sorted)
        interp_files = sorted(
            f for f in os.listdir(interp_dir) if f.endswith(".png")
        )
        logger.info("RIFE generated %d interpolated frames in %s", len(interp_files), interp_dir)

        expected_interp = interp_per_pair * (len(orig_files) - 1)
        if len(interp_files) < expected_interp:
            logger.warning(
                "Expected %d interpolated frames, got %d",
                expected_interp, len(interp_files),
            )

        # === Phase 3: Encode (70-100%) ===
        # Merge frames with hardlinks (instant, no data copy), then read
        # with PIL directly instead of another ffmpeg decode process.
        # This eliminates the disk I/O bottleneck from re-decoding
        # thousands of PNGs through a separate ffmpeg instance.

        try:
            from PIL import Image as _PILImage
            _HAS_PIL = True
        except ImportError:
            _HAS_PIL = False
            logger.warning("Pillow not installed – falling back to ffmpeg for PNG decode")

        merged_dir = os.path.join(base_tmp, "merged")
        os.makedirs(merged_dir, exist_ok=True)

        def _link_or_copy(src: str, dst: str) -> None:
            """Create hardlink (instant) or fallback copy if cross-device."""
            try:
                os.link(src, dst)
            except OSError:
                _shutil_mod.copy2(src, dst)

        # Interleave: orig0, interp(0,1)[0..k-1], orig1, interp(1,2)[0..k-1], ...
        interp_idx = 0
        frame_num = 1
        for i, orig_fname in enumerate(orig_files):
            _link_or_copy(
                os.path.join(orig_dir, orig_fname),
                os.path.join(merged_dir, f"{frame_num:08d}.png"),
            )
            frame_num += 1

            if i < len(orig_files) - 1:
                for k in range(interp_per_pair):
                    if interp_idx < len(interp_files):
                        _link_or_copy(
                            os.path.join(interp_dir, interp_files[interp_idx]),
                            os.path.join(merged_dir, f"{frame_num:08d}.png"),
                        )
                        frame_num += 1
                        interp_idx += 1

        total_merged = frame_num - 1
        logger.info("Merged %d frames into %s (hardlinks)", total_merged, merged_dir)

        if not _HAS_PIL:
            # Fallback: ffmpeg image2 demuxer
            decode2_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(out_fps),
                "-i", os.path.join(merged_dir, "%08d.png"),
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-",
            ]

        encode_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{act_w}x{act_h}",
            "-framerate", str(out_fps),
            "-i", "-",
        ]
        if config.keep_audio:
            encode_cmd += ["-i", input_path]
        encode_cmd += encoder_opts
        encode_cmd.append(output_path)

        logger.info("Phase 3: streaming %d frames (PIL=%s)→NVVFX→encode", total_merged, _HAS_PIL)

        def _run_encode() -> None:
            """Phase 3 worker: read frames → NVVFX → encode."""
            if _HAS_PIL:
                # PIL path: read PNGs directly, no ffmpeg decode overhead
                dec2_proc = None
            else:
                # ffmpeg fallback
                try:
                    dec2_proc = subprocess.Popen(
                        decode2_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        bufsize=1024 * 1024, **extra_kwargs,
                    )
                except FileNotFoundError:
                    raise RuntimeError("未找到 ffmpeg")

            try:
                enc_proc = subprocess.Popen(
                    encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, bufsize=1024 * 1024, **extra_kwargs,
                )
            except FileNotFoundError:
                if dec2_proc:
                    dec2_proc.kill()
                    dec2_proc.wait()
                raise RuntimeError("未找到 ffmpeg")

            if dec2_proc:
                _drain_pipe(dec2_proc, "stderr")
            _drain_pipe(enc_proc, "stdout")
            _spawn_cancel_watcher(cancel_event, enc_proc, *([dec2_proc] if dec2_proc else []))

            enc_stderr_lines: list[str] = []

            def _read_stderr():
                _last_pct = 70.0
                _last_time = 0.0
                try:
                    for line_bytes in iter(enc_proc.stderr.readline, b""):
                        if cancel_event.is_set():
                            break
                        line = line_bytes.decode(errors="replace").rstrip()
                        enc_stderr_lines.append(line)
                        if out_total > 0:
                            m = _frame_re.search(line)
                            if m:
                                cur = int(m.group(1))
                                pct = 70.0 + min((cur / out_total) * 30.0, 29.0)
                                now = time.monotonic()
                                if pct - _last_pct >= 1.0 or now - _last_time >= 1.0:
                                    _last_pct = pct
                                    _last_time = now
                                    asyncio.run_coroutine_threadsafe(
                                        progress_callback(pct, cur, out_total), loop
                                    )
                except Exception:
                    pass

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            try:
                frames_written = 0

                if _HAS_PIL:
                    # Read merged frames directly with PIL (fast, sequential)
                    for frame_num in range(1, total_merged + 1):
                        _check_cancelled(cancel_event)

                        if HAS_CUPY and frames_written > 0 and frames_written % 100 == 0:
                            _cp.get_default_memory_pool().free_all_blocks()

                        fpath = os.path.join(merged_dir, f"{frame_num:08d}.png")
                        try:
                            with _PILImage.open(fpath) as pil_img:
                                arr = np.array(pil_img.convert("RGB"))
                        except Exception as exc:
                            raise RuntimeError(f"读取帧 {fpath} 失败: {exc}") from exc

                        if has_nvvfx:
                            result = processor.process_frame(arr)
                            enc_proc.stdin.write(result.tobytes())
                        else:
                            enc_proc.stdin.write(arr.tobytes())
                        enc_proc.stdin.flush()
                        frames_written += 1
                else:
                    # ffmpeg pipe fallback
                    while True:
                        _check_cancelled(cancel_event)

                        if HAS_CUPY and frames_written > 0 and frames_written % 100 == 0:
                            _cp.get_default_memory_pool().free_all_blocks()

                        raw = b""
                        while len(raw) < in_frame_bytes:
                            chunk = dec2_proc.stdout.read(in_frame_bytes - len(raw))
                            if not chunk:
                                break
                            raw += chunk
                        if len(raw) < in_frame_bytes:
                            break

                        arr = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))

                        if has_nvvfx:
                            result = processor.process_frame(arr)
                            enc_proc.stdin.write(result.tobytes())
                        else:
                            enc_proc.stdin.write(arr.tobytes())
                        enc_proc.stdin.flush()
                        frames_written += 1

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

                task.current_frame = frames_written
                task.total_frames = frames_written

            except TaskCancelledError:
                try:
                    if dec2_proc:
                        dec2_proc.kill()
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
                    if dec2_proc:
                        dec2_proc.kill()
                except Exception:
                    pass
                try:
                    enc_proc.kill()
                except Exception:
                    pass
                stderr_thread.join(timeout=3)
                raise

        await asyncio.wait_for(
            loop.run_in_executor(None, _run_encode),
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

        task.progress = 100.0
        task.current_frame = out_total
        task.total_frames = out_total

    finally:
        await _cleanup()


# --------------- Fallback: minterpolate + NVVFX pipeline ---------------


async def _stream_process_fallback_interp_nvvfx(
    task: Task,
    config: TaskConfig,
    processor: "VFXProcessor",
    input_path: str,
    output_path: str,
    src_w: int, src_h: int,
    out_w: int, out_h: int,
    fps: float,
    total_frames: int,
    cancel_event: threading.Event,
    progress_callback: Callable[[float, int, int], None],
) -> None:
    """Streaming fallback: ffmpeg minterpolate → pipe → NVVFX → pipe → ffmpeg encode.

    Used when RIFE is unavailable but NVVFX is loaded.
    First ffmpeg instance decodes and interpolates to raw RGB24;
    Second ffmpeg instance encodes the NVVFX-processed frames.
    """
    loop = asyncio.get_running_loop()
    multiplier = config.frame_multiplier
    target_fps = fps * multiplier

    # NVVFX effect determines actual output dimensions
    act_w = processor.output_width or out_w
    act_h = processor.output_height or out_h
    out_frame_bytes = _frame_size_bytes(act_w, act_h)

    # Interpolated frame size is still at source resolution
    interp_frame_bytes = _frame_size_bytes(src_w, src_h)

    encoder_opts = _build_encoder_opts(config, target_fps, act_w, act_h)

    extra_kwargs: dict = {}
    if os.name == "nt":
        extra_kwargs["creationflags"] = 0x08000000

    # Stage 1: ffmpeg decode + minterpolate → raw RGB24 stdout
    minterp_filter = (
        f"minterpolate=fps={target_fps}:mi_mode=mci:"
        f"mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    )
    decode_cmd = [
        "ffmpeg", "-y",
        "-hwaccel", "cuda",
        "-i", input_path,
        "-vf", minterp_filter,
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    # Stage 2: ffmpeg encode from raw RGB24 stdin
    encode_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{act_w}x{act_h}",
        "-framerate", str(target_fps),
        "-i", "-",
    ]
    if config.keep_audio:
        encode_cmd += ["-i", input_path]
    encode_cmd += encoder_opts
    encode_cmd.append(output_path)

    out_total = total_frames * multiplier
    _frame_re = re.compile(r"frame=\s*(\d+)")

    def _run_pipeline() -> None:
        try:
            dec_proc = subprocess.Popen(
                decode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=1024 * 1024, **extra_kwargs,
            )
        except FileNotFoundError:
            raise RuntimeError("未找到 ffmpeg，请安装 FFmpeg 并将其加入系统 PATH。")

        try:
            enc_proc = subprocess.Popen(
                encode_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=1024 * 1024, **extra_kwargs,
            )
        except FileNotFoundError:
            dec_proc.kill()
            dec_proc.wait()
            raise RuntimeError("未找到 ffmpeg，请安装 FFmpeg 并将其加入系统 PATH。")

        _drain_pipe(dec_proc, "stderr")
        _drain_pipe(enc_proc, "stdout")
        _spawn_cancel_watcher(cancel_event, dec_proc, enc_proc)

        enc_stderr_lines: list[str] = []

        def _read_stderr():
            _last_pct = -1.0
            _last_time = 0.0
            try:
                for line_bytes in iter(enc_proc.stderr.readline, b""):
                    if cancel_event.is_set():
                        break
                    line = line_bytes.decode(errors="replace").rstrip()
                    enc_stderr_lines.append(line)
                    if out_total > 0:
                        m = _frame_re.search(line)
                        if m:
                            cur = int(m.group(1))
                            pct = min((cur / out_total) * 100.0, 99.0)
                            now = time.monotonic()
                            if pct - _last_pct >= 1.0 or now - _last_time >= 1.0:
                                _last_pct = pct
                                _last_time = now
                                asyncio.run_coroutine_threadsafe(
                                    progress_callback(pct, cur, out_total), loop
                                )
            except Exception:
                pass

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()

        try:
            frames_written = 0

            while True:
                _check_cancelled(cancel_event)

                if HAS_CUPY and frames_written > 0 and frames_written % 100 == 0:
                    _cp.get_default_memory_pool().free_all_blocks()

                raw = b""
                while len(raw) < interp_frame_bytes:
                    chunk = dec_proc.stdout.read(interp_frame_bytes - len(raw))
                    if not chunk:
                        break
                    raw += chunk
                if len(raw) < interp_frame_bytes:
                    break

                arr = np.frombuffer(raw, dtype=np.uint8).reshape((src_h, src_w, 3))
                result = processor.process_frame(arr)
                enc_proc.stdin.write(result.tobytes())
                enc_proc.stdin.flush()
                frames_written += 1

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

            task.current_frame = frames_written
            task.total_frames = frames_written

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
    has_interp = ProcessMode.FRAME_INTERPOLATION in config.modes

    # GPU-accelerable modes (SR, denoise, deblur, high_bitrate)
    gpu_modes = {ProcessMode.SUPER_RESOLUTION, ProcessMode.DENOISE, ProcessMode.DEBLUR, ProcessMode.HIGH_BITRATE}
    has_gpu_mode = bool(set(config.modes) & gpu_modes)
    want_nvvfx = HAS_NVVFX and _vfx_available and has_gpu_mode

    # RIFE CLI availability for interpolation
    want_rife = has_interp and HAS_RIFE

    # --- Route to appropriate pipeline ---

    if has_interp and want_rife:
        # RIFE available → two-phase pipeline (with or without NVVFX)
        processor = create_vfx_processor(config, src_w, src_h) if want_nvvfx else None
        if want_nvvfx and processor is None:
            logger.warning("Task %s: NVVFX init failed, using RIFE-only", task.id)
        await _process_rife_pipeline(
            task, config, processor, input_path, output_path,
            src_w, src_h, out_w, out_h, fps,
            total_frames, cancel_event, progress_callback,
        )
        task.status = TaskStatus.COMPLETED
        logger.info("Task %s completed → %s", task.id, output_path)
        return

    elif has_interp and not want_rife:
        # RIFE not available → fallback with ffmpeg minterpolate
        logger.warning(
            "Task %s: RIFE CLI not found, using ffmpeg minterpolate",
            task.id,
        )
        if want_nvvfx:
            processor = create_vfx_processor(config, src_w, src_h)
        else:
            processor = None

        if processor is not None:
            logger.info("Task %s: using NVVFX+minterpolate fallback pipeline", task.id)
            try:
                await _stream_process_fallback_interp_nvvfx(
                    task, config, processor, input_path, output_path,
                    src_w, src_h, out_w, out_h, fps,
                    total_frames, cancel_event, progress_callback,
                )
            finally:
                processor.close()
        else:
            logger.info("Task %s: using ffmpeg minterpolate fallback pipeline", task.id)
            await _stream_process_fallback(
                task, config, input_path, output_path,
                src_w, src_h, out_w, out_h, fps,
                total_frames, cancel_event, progress_callback,
            )
        task.status = TaskStatus.COMPLETED
        logger.info("Task %s completed → %s", task.id, output_path)
        return

    # --- No interpolation: existing paths ---

    # Try to create NVVFX processor; fall back gracefully if GPU not ready
    processor = create_vfx_processor(config, src_w, src_h) if want_nvvfx else None

    if processor is not None:
        logger.info("Task %s: using NVVFX streaming pipeline", task.id)
        try:
            await _stream_process_nvvfx(
                task, config, processor, input_path, output_path,
                src_w, src_h, out_w, out_h, fps,
                total_frames, cancel_event, progress_callback,
            )
        finally:
            processor.close()
            logger.info("Task %s: NVVFX processor released", task.id)
    else:
        if want_nvvfx:
            logger.warning("Task %s: NVVFX init failed, falling back to ffmpeg pipeline", task.id)
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

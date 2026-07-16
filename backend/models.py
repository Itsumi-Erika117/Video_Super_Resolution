from __future__ import annotations

import enum
import os
import uuid
from dataclasses import dataclass, field
from typing import Optional


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessMode(str, enum.Enum):
    SUPER_RESOLUTION = "super_resolution"
    DENOISE = "denoise"
    DEBLUR = "deblur"
    HIGH_BITRATE = "high_bitrate"
    FRAME_INTERPOLATION = "frame_interpolation"


class QualityLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


class OutputFormat(str, enum.Enum):
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    MKV = "mkv"


class VideoEncoder(str, enum.Enum):
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    AV1 = "libaomenc"
    NVIDIA_H264 = "h264_nvenc"
    NVIDIA_H265 = "hevc_nvenc"


SUPPORTED_INPUT_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".flv"}

QUALITY_CRF_MAP: dict[QualityLevel, int] = {
    QualityLevel.LOW: 28,
    QualityLevel.MEDIUM: 23,
    QualityLevel.HIGH: 18,
    QualityLevel.ULTRA: 14,
}

QUALITY_SCALE_MAP: dict[QualityLevel, int] = {
    QualityLevel.LOW: 0,
    QualityLevel.MEDIUM: 1,
    QualityLevel.HIGH: 2,
    QualityLevel.ULTRA: 3,
}

@dataclass
class TaskConfig:
    modes: list[ProcessMode] = field(default_factory=lambda: [ProcessMode.SUPER_RESOLUTION])
    scale_factor: int = 2
    quality: QualityLevel = QualityLevel.HIGH
    output_format: OutputFormat = OutputFormat.MP4
    video_encoder: VideoEncoder = VideoEncoder.NVIDIA_H264
    batch_size: int = 4
    keep_audio: bool = True
    frame_multiplier: int = 2  # 2x, 3x, 4x frame interpolation


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    filename: str = ""
    input_path: str = ""
    output_path: str = ""
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    error_message: str = ""
    config: TaskConfig = field(default_factory=TaskConfig)
    created_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status.value,
            "progress": self.progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "error_message": self.error_message,
            "config": {
                "modes": [m.value for m in self.config.modes],
                "scale_factor": self.config.scale_factor,
                "quality": self.config.quality.value,
                "output_format": self.config.output_format.value,
                "video_encoder": self.config.video_encoder.value,
                "batch_size": self.config.batch_size,
                "frame_multiplier": self.config.frame_multiplier,
            },
        }

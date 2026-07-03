export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
export type ProcessMode = 'super_resolution' | 'denoise' | 'deblur' | 'high_bitrate';
export type QualityLevel = 'low' | 'medium' | 'high' | 'ultra';
export type OutputFormat = 'mp4' | 'mov' | 'avi' | 'mkv';
export type VideoEncoder =
  | 'libx264' | 'libx265' | 'libvpx-vp9' | 'libaomenc'
  | 'h264_nvenc' | 'hevc_nvenc';

export interface TaskConfig {
  modes: ProcessMode[];
  scale_factor: number;
  quality: QualityLevel;
  output_format: OutputFormat;
  video_encoder: VideoEncoder;
  batch_size: number;
}

export interface Task {
  id: string;
  filename: string;
  status: TaskStatus;
  progress: number;
  current_frame: number;
  total_frames: number;
  error_message: string;
  config: TaskConfig;
}

export interface OutputItem {
  name: string;
  size_bytes: number;
  size_mb: number;
  modified: number;
  ext: string;
}

export interface WSMessage {
  type: 'initial_state' | 'task_update' | 'ping' | 'pong';
  data?: Task | Task[];
}

export interface UploadSettings {
  modes: ProcessMode[];
  scaleFactor: number;
  quality: QualityLevel;
  outputFormat: OutputFormat;
  videoEncoder: VideoEncoder;
  batchSize: number;
}

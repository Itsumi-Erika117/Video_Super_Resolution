import { useCallback, useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import { useDropzone } from 'react-dropzone';
import toast from 'react-hot-toast';
import type { Task, UploadSettings } from '../types';

interface Props {
  settings: UploadSettings;
  tasks: Task[];
  setTasks: Dispatch<SetStateAction<Task[]>>;
}

const ACCEPTED = {
  'video/mp4': ['.mp4'],
  'video/quicktime': ['.mov'],
  'video/x-msvideo': ['.avi'],
  'video/x-matroska': ['.mkv'],
  'video/x-ms-wmv': ['.wmv'],
  'video/x-flv': ['.flv'],
};

export default function UploadZone({ settings, setTasks }: Props) {
  const [uploading, setUploading] = useState(false);

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      setUploading(true);

      const formData = new FormData();
      files.forEach((f) => formData.append('files', f));
      formData.append('modes', settings.modes.join(','));
      formData.append('scale_factor', String(settings.scaleFactor));
      formData.append('quality', settings.quality);
      formData.append('output_format', settings.outputFormat);
      formData.append('video_encoder', settings.videoEncoder);
      formData.append('batch_size', String(settings.batchSize));
      formData.append('frame_multiplier', String(settings.frameMultiplier));

      try {
        const res = await fetch('/api/upload', { method: 'POST', body: formData });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        // 立即将返回的任务合并到本地状态（与 WebSocket 互补）
        setTasks((prev) => {
          const existing = new Set(prev.map((t) => t.id));
          const merged = [...prev];
          for (const t of data.tasks) {
            if (!existing.has(t.id)) merged.push(t);
          }
          return merged;
        });
        toast.success(`已添加 ${data.tasks.length} 个任务到队列`);
      } catch (e: any) {
        toast.error(e.message || '上传失败');
      } finally {
        setUploading(false);
      }
    },
    [settings]
  );

  const { getRootProps, getInputProps, isDragActive, open } = useDropzone({
    onDrop: handleFiles,
    accept: ACCEPTED,
    noClick: true,
    noKeyboard: true,
  });

  return (
    <div
      {...getRootProps()}
      className={`relative border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${
        isDragActive
          ? 'border-cyan-400 bg-cyan-500/5 dropzone-active'
          : 'border-gray-700 hover:border-gray-500 bg-gray-900/30'
      }`}
    >
      <input {...getInputProps()} />

      <div className="flex flex-col items-center gap-3">
        {/* Icon */}
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 flex items-center justify-center">
          <svg className="w-8 h-8 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
        </div>

        <div>
          <p className="text-lg font-medium text-gray-200">
            {isDragActive ? '📥 释放以添加文件' : '拖拽视频文件到此处'}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            支持 MP4, MOV, AVI, MKV, WMV, FLV 格式
          </p>
        </div>

        <button
          onClick={(e) => { e.stopPropagation(); open(); }}
          disabled={uploading}
          className="mt-2 px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {uploading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              上传中...
            </span>
          ) : (
            '📂 选择视频文件'
          )}
        </button>
      </div>
    </div>
  );
}

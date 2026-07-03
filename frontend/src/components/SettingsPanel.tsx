import type { ProcessMode, QualityLevel, OutputFormat, VideoEncoder, UploadSettings } from '../types';

interface Props {
  settings: UploadSettings;
  onSettingsChange: (s: UploadSettings) => void;
}

const MODES: { value: ProcessMode; label: string; desc: string }[] = [
  { value: 'super_resolution', label: '🔍 超分辨率', desc: '1x-4x AI放大，提升分辨率' },
  { value: 'denoise', label: '🔇 降噪', desc: '同分辨率去噪，适合低光/老胶片' },
  { value: 'deblur', label: '🔎 去模糊', desc: '同分辨率去模糊，修复轻微失焦' },
  { value: 'high_bitrate', label: '📡 高码率', desc: '保留高质量源细节（不与超分/降噪/去模糊组合）' },
];

const QUALITIES: { value: QualityLevel; label: string }[] = [
  { value: 'low', label: 'LOW · 快速' },
  { value: 'medium', label: 'MEDIUM · 均衡' },
  { value: 'high', label: 'HIGH · 高质量' },
  { value: 'ultra', label: 'ULTRA · 极致' },
];

const FORMATS: { value: OutputFormat; label: string }[] = [
  { value: 'mp4', label: 'MP4 (H.264/HEVC)' },
  { value: 'mov', label: 'MOV (QuickTime)' },
  { value: 'avi', label: 'AVI' },
  { value: 'mkv', label: 'MKV (Matroska)' },
];

const ENCODERS: { value: VideoEncoder; label: string }[] = [
  { value: 'h264_nvenc', label: 'NVIDIA NVENC H.264 ⚡' },
  { value: 'hevc_nvenc', label: 'NVIDIA NVENC HEVC ⚡' },
  { value: 'libx264', label: 'libx264 (CPU)' },
  { value: 'libx265', label: 'libx265 (CPU)' },
  { value: 'libvpx-vp9', label: 'libvpx-vp9 (CPU)' },
];

export default function SettingsPanel({ settings, onSettingsChange }: Props) {
  const update = (patch: Partial<UploadSettings>) =>
    onSettingsChange({ ...settings, ...patch });

  const isSuperResolution = settings.modes.includes('super_resolution');

  const toggleMode = (m: ProcessMode) => {
    if (settings.modes.includes(m)) {
      // Don't allow deselecting the last mode
      if (settings.modes.length <= 1) return;
      update({ modes: settings.modes.filter((x) => x !== m) });
    } else {
      update({ modes: [...settings.modes, m] });
    }
  };

  return (
    <div className="space-y-6">
      {/* Mode selection */}
      <section>
        <h3 className="text-sm font-medium text-gray-400 mb-3">处理模式（可多选）</h3>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
          {MODES.map((m) => {
            const checked = settings.modes.includes(m.value);
            return (
              <button
                key={m.value}
                onClick={() => toggleMode(m.value)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  checked
                    ? 'border-cyan-500/50 bg-cyan-500/10 ring-1 ring-cyan-500/30'
                    : 'border-gray-800 bg-gray-900/50 hover:border-gray-600'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className={`w-4 h-4 rounded border-2 flex items-center justify-center text-[10px] transition-colors ${
                    checked
                      ? 'bg-cyan-500 border-cyan-500 text-white'
                      : 'border-gray-600 text-transparent'
                  }`}>
                    ✓
                  </span>
                  <span className="text-sm font-medium text-gray-200">{m.label}</span>
                </div>
                <div className="text-xs text-gray-500 mt-1 ml-6">{m.desc}</div>
              </button>
            );
          })}
        </div>
      </section>

      {/* Scale factor (only for super resolution) */}
      {isSuperResolution && (
        <section>
          <h3 className="text-sm font-medium text-gray-400 mb-3">放大倍数</h3>
          <div className="flex gap-2">
            {[1, 2, 3, 4].map((x) => (
              <button
                key={x}
                onClick={() => update({ scaleFactor: x })}
                className={`w-14 h-14 rounded-xl border text-lg font-bold transition-all ${
                  settings.scaleFactor === x
                    ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400 ring-1 ring-cyan-500/30'
                    : 'border-gray-800 bg-gray-900/50 text-gray-400 hover:border-gray-600'
                }`}
              >
                {x}x
              </button>
            ))}
          </div>
        </section>
      )}

      {/* Quality */}
      <section>
        <h3 className="text-sm font-medium text-gray-400 mb-3">质量等级</h3>
        <div className="flex gap-2">
          {QUALITIES.map((q) => (
            <button
              key={q.value}
              onClick={() => update({ quality: q.value })}
              className={`px-4 py-2 rounded-xl border text-sm font-medium transition-all ${
                settings.quality === q.value
                  ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400 ring-1 ring-cyan-500/30'
                  : 'border-gray-800 bg-gray-900/50 text-gray-400 hover:border-gray-600'
              }`}
            >
              {q.label}
            </button>
          ))}
        </div>
      </section>

      {/* Output format */}
      <section>
        <h3 className="text-sm font-medium text-gray-400 mb-3">输出格式</h3>
        <div className="flex gap-2 flex-wrap">
          {FORMATS.map((f) => (
            <button
              key={f.value}
              onClick={() => update({ outputFormat: f.value })}
              className={`px-4 py-2 rounded-xl border text-sm font-medium transition-all ${
                settings.outputFormat === f.value
                  ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400 ring-1 ring-cyan-500/30'
                  : 'border-gray-800 bg-gray-900/50 text-gray-400 hover:border-gray-600'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </section>

      {/* Video encoder */}
      <section>
        <h3 className="text-sm font-medium text-gray-400 mb-3">编码器</h3>
        <div className="flex gap-2 flex-wrap">
          {ENCODERS.map((enc) => (
            <button
              key={enc.value}
              onClick={() => update({ videoEncoder: enc.value })}
              className={`px-4 py-2 rounded-xl border text-sm font-medium transition-all ${
                settings.videoEncoder === enc.value
                  ? 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400 ring-1 ring-cyan-500/30'
                  : 'border-gray-800 bg-gray-900/50 text-gray-400 hover:border-gray-600'
              }`}
            >
              {enc.label}
            </button>
          ))}
        </div>
      </section>

      {/* Batch size */}
      <section>
        <h3 className="text-sm font-medium text-gray-400 mb-3">
          GPU 批处理大小 (NVVFX_BATCH_SIZE)
        </h3>
        <div className="flex items-center gap-3">
          <input
            type="range"
            min={1}
            max={16}
            value={settings.batchSize}
            onChange={(e) => update({ batchSize: Number(e.target.value) })}
            className="w-48 accent-cyan-500"
          />
          <span className="text-sm font-mono text-cyan-400 bg-gray-900 px-3 py-1 rounded-lg">
            {settings.batchSize}
          </span>
          <span className="text-xs text-gray-500">
            更高值可提升 GPU 吞吐量，但需更多显存
          </span>
        </div>
      </section>
    </div>
  );
}

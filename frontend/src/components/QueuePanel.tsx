import toast from 'react-hot-toast';
import type { Task } from '../types';

interface Props {
  tasks: Task[];
}

const STATUS_LABEL: Record<string, string> = {
  pending: '⏳ 等待中',
  processing: '⚡ 处理中',
  completed: '✅ 已完成',
  failed: '❌ 失败',
  cancelled: '🚫 已取消',
};

const STATUS_COLOR: Record<string, string> = {
  pending: 'text-yellow-400',
  processing: 'text-cyan-400',
  completed: 'text-green-400',
  failed: 'text-red-400',
  cancelled: 'text-gray-500',
};

const MODE_LABEL: Record<string, string> = {
  super_resolution: '超分',
  denoise: '降噪',
  deblur: '去模糊',
  high_bitrate: '高码率',
};

async function cancelTask(taskId: string) {
  try {
    const res = await fetch(`/api/tasks/${taskId}/cancel`, { method: 'POST' });
    if (res.ok) toast.success('任务已取消');
  } catch {
    toast.error('取消失败');
  }
}

async function clearCompleted() {
  try {
    const res = await fetch('/api/tasks/clear', { method: 'POST' });
    if (res.ok) {
      const data = await res.json();
      toast.success(`已清除 ${data.removed} 个已完成任务`);
    }
  } catch {
    toast.error('清除失败');
  }
}

export default function QueuePanel({ tasks }: Props) {
  const pending = tasks.filter((t) => t.status === 'pending').length;
  const processing = tasks.filter((t) => t.status === 'processing').length;
  const completed = tasks.filter((t) => t.status === 'completed').length;

  return (
    <div className="space-y-4">
      {/* Summary bar */}
      <div className="flex items-center justify-between">
        <div className="flex gap-4 text-sm">
          <span className="text-yellow-400">⏳ {pending} 等待</span>
          <span className="text-cyan-400">⚡ {processing} 处理中</span>
          <span className="text-green-400">✅ {completed} 完成</span>
          <span className="text-gray-500">{tasks.length} 总计</span>
        </div>
        <button
          onClick={clearCompleted}
          className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 rounded-lg transition-colors"
        >
          清除已完成
        </button>
      </div>

      {/* Task list */}
      {tasks.length === 0 ? (
        <div className="py-16 text-center text-gray-600">
          <p className="text-4xl mb-3">📭</p>
          <p>暂无任务，上传视频开始处理</p>
        </div>
      ) : (
        <div className="space-y-2">
          {[...tasks].reverse().map((task) => (
            <TaskCard key={task.id} task={task} onCancel={cancelTask} />
          ))}
        </div>
      )}
    </div>
  );
}

function TaskCard({ task, onCancel }: { task: Task; onCancel: (id: string) => void }) {
  const isActive = task.status === 'pending' || task.status === 'processing';

  return (
    <div
      className={`rounded-xl border p-4 transition-all ${
        task.status === 'processing'
          ? 'border-cyan-500/30 bg-cyan-500/5 processing-glow'
          : task.status === 'completed'
          ? 'border-green-500/20 bg-green-500/5'
          : task.status === 'failed'
          ? 'border-red-500/20 bg-red-500/5'
          : 'border-gray-800 bg-gray-900/50'
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0 flex-1">
          {/* Header */}
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-xs font-medium ${STATUS_COLOR[task.status]}`}>
              {STATUS_LABEL[task.status]}
            </span>
            <span className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">
              {task.config.modes?.map((m) => MODE_LABEL[m] || m).join('+') || '—'}
            </span>
            {task.config.modes?.includes('super_resolution') && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">
                {task.config.scale_factor}x
              </span>
            )}
          </div>

          <p className="text-sm font-medium text-gray-200 truncate">{task.filename}</p>

          {/* Progress bar */}
          {isActive && (
            <div className="mt-2">
              <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>{task.current_frame}/{task.total_frames} 帧</span>
                <span>{task.progress.toFixed(1)}%</span>
              </div>
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full transition-all duration-300"
                  style={{ width: `${task.progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Completed progress */}
          {task.status === 'completed' && (
            <div className="mt-2">
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full w-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full" />
              </div>
            </div>
          )}

          {/* Error / cancelled message */}
          {(task.status === 'failed' || task.status === 'cancelled') && task.error_message && (
            <p className="mt-2 text-xs text-red-400 truncate">{task.error_message}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 shrink-0">
          {isActive && (
            <button
              onClick={() => onCancel(task.id)}
              className="px-2 py-1 text-xs bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded transition-colors"
            >
              取消
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

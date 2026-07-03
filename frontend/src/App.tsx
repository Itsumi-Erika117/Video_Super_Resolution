import { useState, useCallback } from 'react';
import { Toaster } from 'react-hot-toast';
import type { Task, WSMessage, UploadSettings } from './types';
import { useWebSocket } from './hooks/useWebSocket';
import UploadZone from './components/UploadZone';
import QueuePanel from './components/QueuePanel';
import SettingsPanel from './components/SettingsPanel';
import OutputBrowser from './components/OutputBrowser';

export default function App() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [activeTab, setActiveTab] = useState<'queue' | 'settings' | 'output'>('queue');
  const [settings, setSettings] = useState<UploadSettings>({
    modes: ['super_resolution'],
    scaleFactor: 2,
    quality: 'high',
    outputFormat: 'mp4',
    videoEncoder: 'h264_nvenc',
    batchSize: 4,
  });

  const handleMessage = useCallback((msg: WSMessage) => {
    if (msg.type === 'initial_state') {
      setTasks(msg.data as Task[]);
    } else if (msg.type === 'task_update') {
      const updated = msg.data as Task;
      setTasks((prev) => {
        const idx = prev.findIndex((t) => t.id === updated.id);
        if (idx >= 0) {
          const next = [...prev];
          next[idx] = updated;
          return next;
        }
        return [...prev, updated];
      });
    }
  }, []);

  useWebSocket(handleMessage);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Toaster position="top-right" toastOptions={{ style: { background: '#1f2937', color: '#f3f4f6', border: '1px solid #374151' } }} />

      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center text-lg font-bold">
              V
            </div>
            <div>
              <h1 className="text-lg font-semibold">Video Super Resolution</h1>
              <p className="text-xs text-gray-500">NVIDIA VFX · AI-Powered</p>
            </div>
          </div>

          {/* Tabs */}
          <nav className="flex gap-1 bg-gray-900 rounded-lg p-1">
            {(['queue', 'settings', 'output'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === tab
                    ? 'bg-cyan-500/20 text-cyan-400'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
              >
                {tab === 'queue' && '📋 任务队列'}
                {tab === 'settings' && '⚙️ 处理设置'}
                {tab === 'output' && '📁 输出文件'}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* Upload zone - always visible */}
        <UploadZone settings={settings} tasks={tasks} setTasks={setTasks} />

        {/* Tab panels */}
        <div className="mt-6">
          {activeTab === 'queue' && <QueuePanel tasks={tasks} />}
          {activeTab === 'settings' && <SettingsPanel settings={settings} onSettingsChange={setSettings} />}
          {activeTab === 'output' && <OutputBrowser />}
        </div>
      </main>
    </div>
  );
}

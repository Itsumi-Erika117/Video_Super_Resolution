import { useEffect, useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import type { OutputItem } from '../types';

function formatDate(ts: number): string {
  return new Date(ts * 1000).toLocaleString('zh-CN');
}

export default function OutputBrowser() {
  const [items, setItems] = useState<OutputItem[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/output');
      const data = await res.json();
      setItems(data.items);
    } catch {
      toast.error('无法加载输出目录');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  const handleDelete = async (name: string) => {
    try {
      const res = await fetch(`/api/output/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (res.ok) {
        setItems((prev) => prev.filter((i) => i.name !== name));
        toast.success('已删除');
      }
    } catch {
      toast.error('删除失败');
    }
  };

  const handleDownload = (name: string) => {
    window.open(`/api/output/${encodeURIComponent(name)}`, '_blank');
  };

  if (loading) {
    return (
      <div className="py-16 text-center text-gray-600">
        <svg className="animate-spin w-8 h-8 mx-auto mb-3" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
        <p>加载中...</p>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="py-16 text-center text-gray-600">
        <p className="text-4xl mb-3">📂</p>
        <p>输出文件夹为空</p>
        <p className="text-sm mt-1">处理完成的视频将显示在这里</p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-gray-400">
          {items.length} 个文件 · ./output
        </h3>
        <button
          onClick={fetchItems}
          className="px-3 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 rounded-lg transition-colors"
        >
          🔄 刷新
        </button>
      </div>

      <div className="space-y-1">
        {items.map((item) => (
          <div
            key={item.name}
            className="flex items-center justify-between p-3 rounded-xl border border-gray-800 bg-gray-900/50 hover:bg-gray-900 transition-colors group"
          >
            <div className="flex items-center gap-3 min-w-0 flex-1">
              {/* File icon */}
              <div className="w-9 h-9 rounded-lg bg-gray-800 flex items-center justify-center shrink-0">
                <span className="text-xs font-mono uppercase text-gray-400">{item.ext.replace('.', '')}</span>
              </div>

              <div className="min-w-0">
                <p className="text-sm font-medium text-gray-200 truncate">{item.name}</p>
                <p className="text-xs text-gray-500">
                  {item.size_mb} MB · {formatDate(item.modified)}
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
              <button
                onClick={() => handleDownload(item.name)}
                className="px-3 py-1.5 text-xs bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 rounded-lg transition-colors"
              >
                ⬇ 下载
              </button>
              <button
                onClick={() => handleDelete(item.name)}
                className="px-3 py-1.5 text-xs bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
              >
                🗑 删除
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

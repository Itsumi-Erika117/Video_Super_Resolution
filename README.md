# Video Super Resolution

基于 **NVIDIA GPU** 的视频超分辨率/降噪/去模糊处理工具，支持多种处理模式自由组合，单次编码完成全部处理。

- 这只是一个本人用于学习AI生成代码的学习项目，因此其实有相当多可以继续完善的地方
- 设计之初是为了快速的让我的磁盘中的老片变得更好看。 利用NVDIA的VFX可以更好的利用RTX显卡去处理这样的任务，而且速度更快
- 当然，如果你的显卡不支持VFX，也可以使用ffmpeg的滤镜
- 该项目超过90%的代码由AI生成，模型是Deepseek-v4-pro


## 功能特性

- 🔍 **超分辨率** — 1×–4× AI 放大，支持将 1080p 提升至 4K
- 🔇 **降噪** — 同分辨率去噪（hqdn3d），适合低光照 / 老胶片素材
- 🔎 **去模糊** — 同分辨率锐化（unsharp），修复轻微失焦
- 📡 **高码率重编码** — 保留高质量源素材细节
- ✅ **多模式自由组合** — 勾选超分 + 降噪 + 去模糊，一次性处理
- ⚡ **GPU 硬件加速** — NVDEC 解码 + NVENC 编码，大幅缩短处理时间
- 📋 **任务队列** — 支持批量上传，FIFO 队列逐个处理
- 🔄 **实时进度** — WebSocket 推送，处理进度条实时更新
- 🚫 **随时取消** — 正在运行的任务可即时中止
- 📁 **长视频友好** — 流式管道处理，无中间文件落盘，支持数小时视频

## 技术栈

| 层级 | 技术 |
|------|------|
| **后端框架** | FastAPI + Uvicorn |
| **实时通信** | WebSocket |
| **视频处理** | FFmpeg (NVDEC / NVENC) |
| **AI 加速 SDK** | NVIDIA VFX SDK（可选） |
| **前端框架** | React 18 + TypeScript |
| **样式** | Tailwind CSS |
| **构建工具** | Vite |
| **Python 异步** | asyncio + ThreadPoolExecutor |

## 项目结构

```
vediosuperresolution/
├── backend/
│   ├── main.py            # FastAPI 应用，REST + WebSocket
│   ├── models.py          # 数据模型：Task / TaskConfig / 枚举
│   ├── processing.py       # 核心处理管线（ffmpeg 流式管道）
│   └── queue_manager.py    # 任务队列管理（取消 / 暂停 / 广播）
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── UploadZone.tsx     # 拖拽上传区域
│   │   │   ├── QueuePanel.tsx     # 任务队列面板（进度条）
│   │   │   ├── SettingsPanel.tsx   # 处理参数设置（多选模式）
│   │   │   └── OutputBrowser.tsx  # 输出文件浏览 / 下载
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts    # WebSocket 连接管理
│   │   ├── types/
│   │   │   └── index.ts          # TypeScript 类型定义
│   │   └── App.tsx               # 应用入口
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts
├── uploads/               # 上传文件暂存目录
├── output/                # 处理完成输出目录
├── requirements.txt       # Python 依赖
├── start.bat              # Windows 一键启动
├── start.sh               # Linux / macOS 一键启动
└── README.md
```

## 环境要求

### 必需

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| **Python** | 3.10+ | 后端运行环境 |
| **Node.js** | 18+ | 前端构建（npm） |
| **FFmpeg** | 5.0+ | 需包含 ffmpeg / ffprobe，加入系统 PATH |

### 推荐（GPU 加速）

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| **NVIDIA 显卡** | GTX 10 系列+ | 支持 NVDEC / NVENC |
| **NVIDIA 驱动** | 610.00+ | NVENC 编码器可用性 |
| **NVIDIA VFX SDK** | — | 可选，启用 AI 超分增强 |

> **注意**：未安装 NVIDIA VFX SDK 时，超分辨率将使用 FFmpeg `lanczos` 缩放算法作为回退方案，降噪和去模糊使用 FFmpeg 内置滤镜。

## 快速开始

### Windows

```bat
start.bat
```

脚本会自动完成：
1. 安装 Python 依赖
2. 安装前端依赖并构建
3. 启动后端服务（http://localhost:8000）

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### 手动安装

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 安装前端依赖并构建
cd frontend
npm install
npm run build
cd ..

# 3. 创建上传和输出目录
mkdir -p uploads output

# 4. 启动服务
python -m backend.main
```

打开浏览器访问 **http://localhost:8000**。

### 前端开发模式（热更新）

```bash
cd frontend
npm run dev
```

前端开发服务器运行在 http://localhost:5173，API 请求会代理到后端 8000 端口。

## 使用指南

### 1. 选择处理模式

在「⚙️ 处理设置」标签页中，勾选需要的处理模式（支持多选）：

| 模式 | 说明 |
|------|------|
| 🔍 超分辨率 | 1×–4× 放大，可选倍数 |
| 🔇 降噪 | 同分辨率降噪 |
| 🔎 去模糊 | 同分辨率锐化 |
| 📡 高码率 | 高质量重编码 |

> **提示**：多选时 FFmpeg 在单次编码中按 **超分 → 降噪 → 去模糊** 顺序串行应用滤镜。

### 2. 配置参数

| 参数 | 可选值 |
|------|--------|
| 放大倍数 | 1×, 2×, 3×, 4×（仅超分模式） |
| 质量等级 | LOW / MEDIUM / HIGH / ULTRA |
| 输出格式 | MP4, MOV, AVI, MKV |
| 编码器 | NVENC H.264 / HEVC（GPU），libx264 / libx265 / VP9（CPU） |
| 批处理大小 | 1–16（GPU 并行帧数） |

### 3. 上传视频

- 拖拽视频文件到上传区域，或点击「选择视频文件」
- 支持格式：MP4, MOV, AVI, MKV, WMV, FLV
- 支持批量上传，文件最大 16 GB

### 4. 监控进度

- 在「📋 任务队列」中查看处理进度
- 进度条每秒更新一次，显示当前帧 / 总帧数
- 支持随时取消正在处理的任务

### 5. 下载结果

- 处理完成后在「📁 输出文件」中下载
- 也可直接从 `output/` 目录获取

## 处理管线

```
上传视频 → ffprobe 元数据探测 → 流式管道处理 → 输出文件
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                                           ▼
            NVVFX 路径 (GPU)                            Fallback 路径 (CPU)
    ffmpeg decode → pipe → NVVFX →                    ffmpeg 单命令：
    pipe → ffmpeg encode                               -hwaccel cuda + 滤镜链
```

- **无中间文件**：帧数据通过内存管道传输，支持任意长度视频
- **进度节流**：每 1% 或每秒更新一次进度，避免大视频时事件循环拥塞

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/upload` | 上传视频并加入队列 |
| `GET` | `/api/tasks` | 获取所有任务列表 |
| `GET` | `/api/tasks/{id}` | 获取单个任务详情 |
| `POST` | `/api/tasks/{id}/cancel` | 取消任务 |
| `DELETE` | `/api/tasks/{id}` | 删除任务 |
| `POST` | `/api/tasks/clear` | 清除已完成/已取消任务 |
| `GET` | `/api/output` | 浏览输出文件列表 |
| `GET` | `/api/output/{name}` | 下载输出文件 |
| `DELETE` | `/api/output/{name}` | 删除输出文件 |
| `WS` | `/ws` | WebSocket 实时状态推送 |

### 上传参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `files` | File[] | 视频文件 |
| `modes` | string | 逗号分隔，如 `super_resolution,denoise` |
| `scale_factor` | int | 放大倍数（1–4） |
| `quality` | string | 质量：`low`/`medium`/`high`/`ultra` |
| `output_format` | string | 输出格式：`mp4`/`mov`/`avi`/`mkv` |
| `video_encoder` | string | 编码器：`h264_nvenc`/`hevc_nvenc`/`libx264`/`libx265`/`libvpx-vp9` |
| `batch_size` | int | GPU 批处理大小（1–16） |



#!/bin/bash
echo "============================================"
echo " Video Super Resolution - NVIDIA VFX"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install Python packages"
    exit 1
fi

# Install frontend dependencies and build
echo "[2/3] Building frontend..."
cd frontend
npm install
if [ $? -ne 0 ]; then
    echo "[WARNING] npm install failed, skipping frontend build"
    cd ..
else
    npm run build
    if [ $? -ne 0 ]; then
        echo "[WARNING] Frontend build failed, will run dev server separately"
    fi
    cd ..
fi

# Create directories
mkdir -p uploads output

echo ""
echo "[3/3] Starting backend server on http://localhost:8000"
echo ""
echo "Open http://localhost:8000 in your browser"
echo "Press Ctrl+C to stop"
echo ""
python3 -m backend.main

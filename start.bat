@echo off
echo ============================================
echo  Video Super Resolution - NVIDIA VFX
echo ============================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

REM Install Python dependencies
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Python packages
    pause
    exit /b 1
)

REM Install frontend dependencies and build
echo [2/3] Building frontend...
cd frontend
call npm install
if %errorlevel% neq 0 (
    echo [WARNING] npm install failed, skipping frontend build
    cd ..
) else (
    call npm run build
    if %errorlevel% neq 0 (
        echo [WARNING] Frontend build failed, will run dev server separately
    )
    cd ..
)

REM Create directories
if not exist "uploads" mkdir uploads
if not exist "output" mkdir output

echo.
echo [3/3] Starting backend server on http://localhost:8000
echo.
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop
echo.
python -m backend.main
pause

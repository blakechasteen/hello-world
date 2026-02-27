@echo off
REM ============================================================
REM hololoom Demo Suite Launcher
REM ============================================================
REM One-click launcher for all hololoom demo services
REM Created: 2025-11-29
REM ============================================================

title hololoom Demo Suite

echo.
echo  ============================================================
echo   _   _       _       _
echo  ^| ^| ^| ^| ___ ^| ^| ___ ^| ^|     ___   ___  _ __ ___
echo  ^| ^|_^| ^|/ _ \^| ^|/ _ \^| ^|    / _ \ / _ \^| '_ ` _ \
echo  ^|  _  ^| (_) ^| ^| (_) ^| ^|___^| (_) ^| (_) ^| ^| ^| ^| ^| ^|
echo  ^|_^| ^|_^|\___/^|_^|\___/^|______\___/ \___/^|_^| ^|_^| ^|_^|
echo.
echo   UI/UX Demo Suite - 120,000+ lines across 250+ files
echo  ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.8+
    pause
    exit /b 1
)

REM Set working directory to script location
cd /d "%~dp0"

echo [INFO] Starting hololoom Demo Services...
echo.

REM Create logs directory if not exists
if not exist "logs" mkdir logs

REM ============================================================
REM Service 1: Static File Server (Port 8080)
REM ============================================================
echo [1/5] Starting Static File Server on port 8080...
start "hololoom-Static-8080" /min cmd /c "cd demos\output && python -m http.server 8080 > ..\..\logs\static_server.log 2>&1"
timeout /t 2 /nobreak >nul

REM ============================================================
REM Service 2: Consciousness UI / Gradio (Port 7860)
REM ============================================================
echo [2/5] Starting Consciousness UI (Gradio) on port 7860...
start "hololoom-Gradio-7860" /min cmd /c "set PYTHONPATH=. && python ui\consciousness_ui.py > logs\gradio.log 2>&1"
timeout /t 3 /nobreak >nul

REM ============================================================
REM Service 3: Chat WebSocket Server (Port 8000)
REM ============================================================
echo [3/5] Starting Chat Server on port 8000...
start "hololoom-Chat-8000" /min cmd /c "set PYTHONPATH=. && python hololoom\web_dashboard\server.py > logs\chat_server.log 2>&1"
timeout /t 2 /nobreak >nul

REM ============================================================
REM Service 4: Workflow Executor (Port 8001)
REM ============================================================
echo [4/5] Starting Workflow Executor on port 8001...
start "hololoom-Workflow-8001" /min cmd /c "set PYTHONPATH=. && python hololoom\web_dashboard\workflow_executor.py > logs\workflow.log 2>&1"
timeout /t 2 /nobreak >nul

REM ============================================================
REM Service 5: Agentic API Server (Port 8002)
REM ============================================================
echo [5/5] Starting Agentic Server on port 8002...
start "hololoom-Agentic-8002" /min cmd /c "set PYTHONPATH=. && python hololoom\server\agentic_api.py > logs\agentic.log 2>&1"
timeout /t 3 /nobreak >nul

echo.
echo [INFO] All services started. Checking health...
echo.

REM Quick health checks
echo Checking services...

REM Check Static Server
curl -s -o nul -w "" http://localhost:8080 >nul 2>&1
if errorlevel 1 (
    echo   [!] Static Server (8080): Starting...
) else (
    echo   [+] Static Server (8080): Online
)

REM Check other services (they take longer to start)
echo   [~] Gradio (7860): Starting (may take 10-30s)...
echo   [~] Chat Server (8000): Starting...
echo   [~] Workflow (8001): Starting...
echo   [~] Agentic (8002): Starting...

echo.
echo ============================================================
echo  Services Summary:
echo ============================================================
echo.
echo  Port     Service                 URL
echo  ----     -------                 ---
echo  8080     Static Demos            http://localhost:8080/demo_index.html
echo  7860     Consciousness UI        http://localhost:7860
echo  8000     Chat WebSocket          http://localhost:8000
echo  8001     Workflow Builder        http://localhost:8001
echo  8002     Agentic API             http://localhost:8002
echo.
echo ============================================================
echo.

REM Open the landing page
echo [INFO] Opening Demo Suite landing page...
timeout /t 2 /nobreak >nul
start http://localhost:8080/demo_index.html

echo.
echo [SUCCESS] hololoom Demo Suite is running!
echo.
echo To stop all services, run: stop_demos.bat
echo Logs are saved in: logs\
echo.
echo Press any key to keep this window open (or close it)...
pause >nul

@echo off
REM Elle AR Demo - iPhone Quick Start
REM This script starts both hololoom backend and web server for iPhone testing

echo.
echo ==========================================
echo   Elle AR Demo - iPhone Setup
echo ==========================================
echo.

REM Get PC IP address
echo [1/3] Detecting PC IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        set IP=%%b
        goto :found_ip
    )
)

:found_ip
echo     Your PC IP: %IP%
echo.

REM Update demo.html with correct IP
echo [2/3] Updating demo.html with your IP...
powershell -Command "(Get-Content demo.html) -replace 'ws://192\.168\.\d+\.\d+:8000', 'ws://%IP%:8000' | Set-Content demo.html"
echo     Updated backend URL to: ws://%IP%:8000/ws/ar
echo.

REM Instructions
echo [3/3] Starting servers...
echo.
echo ==========================================
echo   Next Steps:
echo ==========================================
echo.
echo 1. Start hololoom backend (in new terminal):
echo    cd c:\Users\blake\OneDrive\Documents\mythRL
echo    python -m hololoom.server.agentic_api
echo.
echo 2. Start web server (this will run below):
echo    python -m http.server 8080
echo.
echo 3. On your iPhone:
echo    - Open Safari
echo    - Navigate to: http://%IP%:8080/demo.html
echo    - Grant camera permission
echo    - Tap "Start AR Session"
echo    - Say "Hey Elle" to activate voice
echo.
echo ==========================================
echo.

pause

REM Start web server
echo Starting web server on port 8080...
echo.
python -m http.server 8080

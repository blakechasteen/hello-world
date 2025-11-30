@echo off
REM ============================================================
REM HoloLoom Demo Suite - Stop All Services
REM ============================================================
REM Gracefully stops all running HoloLoom demo services
REM Created: 2025-11-29
REM ============================================================

title HoloLoom - Stopping Services

echo.
echo [INFO] Stopping HoloLoom Demo Services...
echo.

REM Kill services by window title
taskkill /FI "WINDOWTITLE eq HoloLoom-Static-8080*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq HoloLoom-Gradio-7860*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq HoloLoom-Chat-8000*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq HoloLoom-Workflow-8001*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq HoloLoom-Agentic-8002*" /F >nul 2>&1

REM Also kill any orphaned Python processes on these ports
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8001"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8002"') do taskkill /PID %%a /F >nul 2>&1

echo.
echo [SUCCESS] All HoloLoom Demo services stopped.
echo.
pause

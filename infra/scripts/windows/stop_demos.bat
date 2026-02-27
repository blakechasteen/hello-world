@echo off
REM ============================================================
REM hololoom Demo Suite - Stop All Services
REM ============================================================
REM Gracefully stops all running hololoom demo services
REM Created: 2025-11-29
REM ============================================================

title hololoom - Stopping Services

echo.
echo [INFO] Stopping hololoom Demo Services...
echo.

REM Kill services by window title
taskkill /FI "WINDOWTITLE eq hololoom-Static-8080*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq hololoom-Gradio-7860*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq hololoom-Chat-8000*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq hololoom-Workflow-8001*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq hololoom-Agentic-8002*" /F >nul 2>&1

REM Also kill any orphaned Python processes on these ports
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8080"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":7860"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8001"') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8002"') do taskkill /PID %%a /F >nul 2>&1

echo.
echo [SUCCESS] All hololoom Demo services stopped.
echo.
pause

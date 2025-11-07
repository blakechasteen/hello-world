@echo off
REM BDR Workflow Executor Startup Script
REM This starts the workflow executor with proper PYTHONPATH

echo.
echo ======================================================================
echo  BDR WORKFLOW EXECUTOR - STARTING
echo ======================================================================
echo.

cd /d "%~dp0"

echo Setting PYTHONPATH to repository root...
set PYTHONPATH=%CD%

echo.
echo Starting workflow executor on http://localhost:8001
echo.
echo Press Ctrl+C to stop the server
echo.

cd HoloLoom\web_dashboard
python workflow_executor.py

pause
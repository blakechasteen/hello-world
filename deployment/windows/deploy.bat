@echo off
REM COZ Daily Brief Automation - Windows Deployment
REM Simple batch wrapper for PowerShell deployment script

echo ======================================================================
echo COZ Daily Brief Automation - Windows Deployment
echo ======================================================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Error: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Run PowerShell deployment script
powershell -ExecutionPolicy Bypass -File "%~dp0deploy_windows.ps1" %*

pause

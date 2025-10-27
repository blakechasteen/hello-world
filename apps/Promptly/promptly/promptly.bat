@echo off
REM Promptly Batch Wrapper for Windows
REM Usage: promptly ultraprompt ollama "your request"

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0

if "%1"=="ultraprompt" (
    python "%SCRIPT_DIR%promptly_cli.py" %*
) else (
    python "%SCRIPT_DIR%promptly.py" %*
)

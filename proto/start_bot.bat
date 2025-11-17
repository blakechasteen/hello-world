@echo off
REM Quick start script for Promptly Matrix Bot

echo ========================================
echo  Promptly Matrix Bot - Quick Start
echo ========================================
echo.

REM Prompt for password if not set
if "%MATRIX_PASSWORD%"=="" (
    echo Matrix password not set in environment.
    set /p MATRIX_PASSWORD="Enter your bot password: "
)

REM Run the bot
echo.
echo Starting bot...
echo.
python run_bot.py

pause

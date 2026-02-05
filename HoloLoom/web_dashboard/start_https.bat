@echo off
echo ============================================================
echo   Starting HoloLoom Dashboard with HTTPS
echo ============================================================
echo.
echo   Server will start at: https://localhost:8002
echo.
echo   IMPORTANT: In Firefox, you'll see a security warning.
echo   Click "Advanced" then "Accept the Risk and Continue"
echo.
echo ============================================================
echo.

cd /d "%~dp0"
uvicorn agentic_server:app --host 0.0.0.0 --port 8002 --ssl-keyfile key.pem --ssl-certfile cert.pem

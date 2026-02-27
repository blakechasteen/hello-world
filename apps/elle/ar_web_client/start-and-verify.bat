@echo off
REM Elle AR - Complete Startup and Verification Script
REM This script starts the backend and verifies everything is ready for iPhone testing

echo.
echo ==========================================
echo   Elle AR - Startup and Verification
echo ==========================================
echo.

REM Step 1: Get PC's WiFi IP
echo [1/5] Detecting PC WiFi IP address...
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4 Address"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        set IP=%%b
        goto :found_ip
    )
)

:found_ip
echo     Your PC IP: %IP%
echo.

REM Step 2: Check if IP is valid (not 172.17.x.x)
echo [2/5] Validating IP address...
echo %IP% | findstr /C:"172.17." >nul
if %errorlevel%==0 (
    echo     ⚠️  WARNING: IP %IP% looks like WSL/virtual network
    echo     Your WiFi IP might be different. Check ipconfig output.
    echo.
) else (
    echo     ✅ IP looks valid (not WSL/virtual)
    echo.
)

REM Step 3: Check if backend is already running
echo [3/5] Checking if backend is already running...
netstat -ano | findstr ":8000" >nul
if %errorlevel%==0 (
    echo     ⚠️  Port 8000 already in use - backend might be running
    echo     If you need to restart, close the existing process first.
    echo.
    set BACKEND_RUNNING=1
) else (
    echo     Port 8000 is free
    echo.
    set BACKEND_RUNNING=0
)

REM Step 4: Start backend if not running
if %BACKEND_RUNNING%==0 (
    echo [4/5] Starting hololoom AR backend...
    echo     Command: python -m hololoom.server.ar_api
    echo.
    start "hololoom AR Backend" cmd /k "cd c:\Users\blake\OneDrive\Documents\mythRL && python -m hololoom.server.ar_api"
    echo     ✅ Backend server starting in new window...
    echo     Wait 5 seconds for server to initialize...
    timeout /t 5 /nobreak >nul
    echo.
) else (
    echo [4/5] Skipping backend start (already running)
    echo.
)

REM Step 5: Verify connection
echo [5/5] Verifying WebSocket connection...
echo     Testing: ws://%IP%:8000/ws/ar
echo.

REM Use PowerShell to test WebSocket connection
powershell -Command "$ws = New-Object System.Net.WebSockets.ClientWebSocket; try { $cts = New-Object System.Threading.CancellationTokenSource; $cts.CancelAfter(3000); $task = $ws.ConnectAsync([System.Uri]'ws://%IP%:8000/ws/ar', $cts.Token); $task.Wait(); if ($ws.State -eq 'Open') { Write-Host '     ✅ WebSocket connection successful!'; $ws.CloseAsync(1000, 'Test complete', $cts.Token).Wait(); exit 0 } else { Write-Host '     ❌ Connection failed'; exit 1 } } catch { Write-Host '     ❌ Connection error:' $_.Exception.Message; exit 1 } finally { $ws.Dispose() }"

if %errorlevel%==0 (
    echo.
    echo ==========================================
    echo   ✅ Backend Ready!
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo   ⚠️  Backend Connection Failed
    echo ==========================================
    echo.
    echo Troubleshooting:
    echo   1. Check backend window for errors
    echo   2. Verify firewall allows port 8000
    echo   3. Try: netsh advfirewall firewall add rule name="Elle AR" dir=in action=allow protocol=TCP localport=8000
    echo.
)

echo.
echo ==========================================
echo   Next Steps:
echo ==========================================
echo.
echo 1. Backend Status:
if %BACKEND_RUNNING%==1 (
    echo    ✅ Backend was already running
) else (
    echo    ✅ Backend started in separate window
)
echo.
echo 2. On your iPhone:
echo    - Open Safari
echo    - Navigate to: http://%IP%:8080/demo.html
echo    - Grant camera permission when prompted
echo.
echo 3. Start web server (if not already running):
echo    python -m http.server 8080
echo.
echo 4. HTTPS Issue (iPhone Camera):
echo    ⚠️  iPhone Safari requires HTTPS for camera via IP
echo.
echo    Solutions:
echo    A. Use React app:
echo       cd elle/ar_web_client
echo       npm run dev
echo       Open: http://%IP%:5173
echo.
echo    B. Deploy to Vercel (automatic HTTPS):
echo       vercel deploy
echo.
echo    C. Use ngrok for HTTPS tunnel:
echo       ngrok http 8080
echo.
echo 5. Debug Tools:
echo    - diagnostic.html: http://%IP%:8080/diagnostic.html
echo    - Backend health: http://%IP%:8000/health
echo.
echo ==========================================
echo.

pause

@echo off
echo ========================================
echo Tesseract OCR Installation for Windows
echo ========================================
echo.

REM Check if Chocolatey is installed
where choco >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Chocolatey not found. Installing Chocolatey first...
    echo.
    echo Please run this in PowerShell as Administrator:
    echo.
    echo Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    echo.
    echo After Chocolatey is installed, run this script again.
    pause
    exit /b 1
)

echo Chocolatey found! Installing Tesseract...
echo.

choco install tesseract -y

echo.
echo Installation complete!
echo.
echo Verifying installation...
tesseract --version

echo.
echo Now installing Python package...
pip install pytesseract

echo.
echo Testing Python integration...
python -c "import pytesseract; print('Tesseract Python package: OK')"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Run: python demos/demo_schema_aware_receipt.py
echo 2. Run: python demos/demo_voice_correction.py
echo 3. Start voice UI: python HoloLoom/web_dashboard/voice_correction_server.py
echo.
pause

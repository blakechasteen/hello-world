@echo off
REM HoloLoom ChatOps - Test Environment Deployment (Windows)
REM =========================================================

echo ========================================
echo HoloLoom ChatOps Test Deployment
echo ========================================
echo.

REM Check if running from repo root
if not exist "HoloLoom\chatops\run_chatops.py" (
    echo ERROR: Please run from repository root
    exit /b 1
)

REM Step 1: Check dependencies
echo Step 1: Checking dependencies...
python -c "import nio" 2>nul || (
    echo Installing matrix-nio...
    pip install matrix-nio aiofiles python-magic
)
python -c "import yaml" 2>nul || (
    echo Installing PyYAML...
    pip install pyyaml
)
echo [OK] Dependencies installed
echo.

REM Step 2: Create test configuration
echo Step 2: Creating test configuration...

if not exist "chatops_test_config.yaml" (
    echo Creating chatops_test_config.yaml...
    (
        echo # Test Configuration for HoloLoom ChatOps
        echo matrix:
        echo   homeserver_url: "https://matrix.org"
        echo   user_id: "@bot:matrix.org"  # EDIT THIS
        echo   access_token: null  # Set via MATRIX_ACCESS_TOKEN env var
        echo   device_id: "HOLOLOOM_TEST_BOT"
        echo   store_path: "./test_matrix_store"
        echo   rooms:
        echo     - "#hololoom-test:matrix.org"  # EDIT THIS
        echo   command_prefix: "!"
        echo   respond_to_mentions: true
        echo   respond_to_dm: true
        echo   admin_users: []  # EDIT THIS
        echo   rate_limit:
        echo     messages_per_window: 10
        echo     window_seconds: 60
        echo.
        echo hololoom:
        echo   mode: "fast"
        echo   memory:
        echo     store_path: "./test_chatops_memory"
        echo     enable_kg_storage: true
        echo     context_limit: 10
        echo.
        echo logging:
        echo   level: "INFO"
        echo   format: "%%(asctime)s - %%(name)s - %%(levelname)s - %%(message)s"
        echo   file:
        echo     enabled: true
        echo     path: "./logs/test_chatops.log"
        echo     max_bytes: 10485760
        echo     backup_count: 3
    ) > chatops_test_config.yaml
    echo [OK] Created chatops_test_config.yaml
) else (
    echo [OK] Using existing chatops_test_config.yaml
)
echo.

REM Step 3: Create test directories
echo Step 3: Creating test directories...
if not exist "test_matrix_store" mkdir test_matrix_store
if not exist "test_chatops_memory" mkdir test_chatops_memory
if not exist "logs" mkdir logs
echo [OK] Directories created
echo.

REM Step 4: Run verification
echo Step 4: Running verification checks...
python -c "from holoLoom.chatops import MatrixBot, ChatOpsOrchestrator; from holoLoom.chatops.conversation_memory import ConversationMemory; print('[OK] All imports successful')" || (
    echo [ERROR] Import verification failed
    exit /b 1
)
echo.

REM Step 5: Check configuration
echo Step 5: Checking configuration...
if not defined MATRIX_ACCESS_TOKEN (
    echo [WARNING] MATRIX_ACCESS_TOKEN not set
    echo Bot will not be able to authenticate
    echo.
    echo To get an access token:
    echo 1. Login to Element/Matrix client
    echo 2. Go to Settings - Help ^& About - Advanced
    echo 3. Copy 'Access Token'
    echo 4. Run: set MATRIX_ACCESS_TOKEN=your_token
    echo.
) else (
    echo [OK] MATRIX_ACCESS_TOKEN is set
)
echo.

REM Step 6: Create test script
echo Step 6: Creating test runner...
(
    echo @echo off
    echo REM Quick test bot runner
    echo set PYTHONPATH=.
    echo python HoloLoom\chatops\run_chatops.py --config chatops_test_config.yaml --debug
) > run_test_bot.bat
echo [OK] Created run_test_bot.bat
echo.

REM Summary
echo ========================================
echo Deployment Ready!
echo ========================================
echo.
echo Next steps:
echo 1. Set your Matrix credentials:
echo    set MATRIX_ACCESS_TOKEN=your_token
echo.
echo 2. Edit chatops_test_config.yaml:
echo    - Set user_id to your bot's Matrix ID
echo    - Add test rooms
echo    - Add admin users
echo.
echo 3. Run the bot:
echo    run_test_bot.bat
echo.
echo 4. In Matrix, try these commands:
echo    !ping
echo    !help
echo    !status
echo.
echo Logs will be in: logs\test_chatops.log
echo.

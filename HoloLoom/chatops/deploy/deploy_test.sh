#!/bin/bash
# HoloLoom ChatOps - Test Environment Deployment
# ==============================================

set -e  # Exit on error

echo "========================================"
echo "HoloLoom ChatOps Test Deployment"
echo "========================================"
echo ""

# Check if running from repo root
if [ ! -f "HoloLoom/chatops/run_chatops.py" ]; then
    echo "ERROR: Please run from repository root"
    exit 1
fi

# Step 1: Check dependencies
echo "Step 1: Checking dependencies..."
python3 -c "import nio" 2>/dev/null || {
    echo "Installing matrix-nio..."
    pip install matrix-nio aiofiles python-magic
}
python3 -c "import yaml" 2>/dev/null || {
    echo "Installing PyYAML..."
    pip install pyyaml
}
echo "✓ Dependencies installed"
echo ""

# Step 2: Create test configuration
echo "Step 2: Creating test configuration..."

if [ -z "$MATRIX_ACCESS_TOKEN" ]; then
    echo "WARNING: MATRIX_ACCESS_TOKEN not set"
    echo "Please set it with: export MATRIX_ACCESS_TOKEN='your_token'"
    echo ""
    echo "Or enter it now (leave blank to skip):"
    read -s token
    if [ -n "$token" ]; then
        export MATRIX_ACCESS_TOKEN="$token"
    fi
fi

# Create test config from template
if [ ! -f "chatops_test_config.yaml" ]; then
    echo "Creating chatops_test_config.yaml..."
    cat > chatops_test_config.yaml <<EOF
# Test Configuration for HoloLoom ChatOps
matrix:
  homeserver_url: "https://matrix.org"
  user_id: "${MATRIX_USER_ID:-@bot:matrix.org}"
  access_token: null  # Set via MATRIX_ACCESS_TOKEN env var
  device_id: "HOLOLOOM_TEST_BOT"
  store_path: "./test_matrix_store"
  rooms:
    - "#hololoom-test:matrix.org"
  command_prefix: "!"
  respond_to_mentions: true
  respond_to_dm: true
  admin_users: []
  rate_limit:
    messages_per_window: 10
    window_seconds: 60

hololoom:
  mode: "fast"
  memory:
    store_path: "./test_chatops_memory"
    enable_kg_storage: true
    context_limit: 10

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file:
    enabled: true
    path: "./logs/test_chatops.log"
    max_bytes: 10485760
    backup_count: 3
EOF
    echo "✓ Created chatops_test_config.yaml"
else
    echo "✓ Using existing chatops_test_config.yaml"
fi
echo ""

# Step 3: Create test directories
echo "Step 3: Creating test directories..."
mkdir -p test_matrix_store
mkdir -p test_chatops_memory
mkdir -p logs
echo "✓ Directories created"
echo ""

# Step 4: Run verification
echo "Step 4: Running verification checks..."
PYTHONPATH=. python3 <<PYTHON
import sys
try:
    from holoLoom.chatops import MatrixBot, ChatOpsOrchestrator
    from holoLoom.chatops.conversation_memory import ConversationMemory
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
PYTHON
echo ""

# Step 5: Check configuration
echo "Step 5: Checking configuration..."
if [ -z "$MATRIX_ACCESS_TOKEN" ]; then
    echo "⚠ WARNING: MATRIX_ACCESS_TOKEN not set"
    echo "Bot will not be able to authenticate"
    echo ""
    echo "To get an access token:"
    echo "1. Login to Element/Matrix client"
    echo "2. Go to Settings → Help & About → Advanced"
    echo "3. Copy 'Access Token'"
    echo "4. Run: export MATRIX_ACCESS_TOKEN='your_token'"
    echo ""
else
    echo "✓ MATRIX_ACCESS_TOKEN is set"
fi
echo ""

# Step 6: Create test script
echo "Step 6: Creating test runner..."
cat > run_test_bot.sh <<'EOF'
#!/bin/bash
# Quick test bot runner
export PYTHONPATH=.
python3 HoloLoom/chatops/run_chatops.py --config chatops_test_config.yaml --debug
EOF
chmod +x run_test_bot.sh
echo "✓ Created run_test_bot.sh"
echo ""

# Summary
echo "========================================"
echo "Deployment Ready!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Set your Matrix credentials:"
echo "   export MATRIX_ACCESS_TOKEN='your_token'"
echo ""
echo "2. Edit chatops_test_config.yaml:"
echo "   - Set user_id to your bot's Matrix ID"
echo "   - Add test rooms"
echo "   - Add admin users"
echo ""
echo "3. Run the bot:"
echo "   ./run_test_bot.sh"
echo ""
echo "4. In Matrix, try these commands:"
echo "   !ping"
echo "   !help"
echo "   !status"
echo ""
echo "Logs will be in: logs/test_chatops.log"
echo ""

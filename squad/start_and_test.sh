#!/bin/bash
# Squad Server - Start and Test Script

set -e

SQUAD_DIR="/home/user/hello-world/squad"
cd "$SQUAD_DIR"

echo "============================================"
echo "Squad Server - Start and Test"
echo "============================================"
echo ""

# Check if server is already running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Server already running on port 8000"
else
    echo "Starting Squad server..."
    PYTHONPATH=/home/user/hello-world python server.py &
    SERVER_PID=$!
    echo "Server started (PID: $SERVER_PID)"

    # Wait for server to be ready
    echo "Waiting for server to initialize..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "✅ Server is ready!"
            break
        fi
        echo -n "."
        sleep 1
    done
    echo ""
fi

# Run tests
echo ""
echo "Running tests..."
echo ""
python test_squad.py

# Print test results
echo ""
if [ -f "test_results.json" ]; then
    echo "Test results saved to: $SQUAD_DIR/test_results.json"
fi

echo ""
echo "============================================"
echo "Testing complete!"
echo "============================================"

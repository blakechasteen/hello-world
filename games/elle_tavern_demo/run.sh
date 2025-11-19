#!/bin/bash
# The Rusty Mug Tavern - Launch Script
# Created: 2025-11-16

echo "======================================"
echo "🍺 The Rusty Mug Tavern"
echo "An Elle Game Engine Demo"
echo "======================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if dependencies are installed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo ""
    echo "⚠️  Dependencies not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "Starting server on http://localhost:8001"
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 server.py

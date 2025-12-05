#!/bin/bash
#
# Elle Core Dashboard - Quick Install Script
#
# This script installs dependencies and starts the dashboard server.
#
# Usage:
#   bash elle/dashboard/install.sh
#
# Created: 2025-11-15
# Author: Blake Chasteen (Agent C)

set -e

echo ""
echo "========================================================================="
echo "ELLE CORE DASHBOARD - INSTALLATION"
echo "========================================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || {
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
}

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install fastapi uvicorn || {
    echo "❌ Failed to install dependencies."
    exit 1
}

echo "✅ Dependencies installed successfully!"

# Start server
echo ""
echo "========================================================================="
echo "STARTING DASHBOARD SERVER"
echo "========================================================================="
echo ""
echo "Server will start on http://localhost:8000"
echo ""
echo "Available dashboards:"
echo "  • Main Dashboard:    http://localhost:8000/"
echo "  • Budget Tracking:   http://localhost:8000/budget"
echo "  • Product ROI:       http://localhost:8000/products"
echo "  • Weekly Planner:    http://localhost:8000/planner"
echo "  • Task Tracker:      http://localhost:8000/tracker"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for user
read -p "Press Enter to start the server..." </dev/tty

# Start server
cd "$(dirname "$0")/../.."
python3 demos/demo_dashboard.py

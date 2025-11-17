#!/bin/bash

# Promptly API Quick Start Script

set -e

echo "========================================="
echo "Promptly API Quick Start"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python 3.11 or higher is required${NC}"
    echo "Current version: $PYTHON_VERSION"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cp .env.example .env

    # Generate secret key
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

    # Update .env with generated secret
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s/change-this-to-a-random-secret-key-in-production/$SECRET_KEY/" .env
    else
        sed -i "s/change-this-to-a-random-secret-key-in-production/$SECRET_KEY/" .env
    fi

    echo -e "${GREEN}✓ Environment file created${NC}"
else
    echo -e "${GREEN}✓ Using existing .env file${NC}"
fi
echo ""

# Create data directory
if [ ! -d "./data" ]; then
    echo -e "${YELLOW}Creating data directory...${NC}"
    mkdir -p ./data
    echo -e "${GREEN}✓ Data directory created${NC}"
fi

# Initialize Promptly if needed
if [ ! -d "./data/.promptly" ]; then
    echo -e "${YELLOW}Initializing Promptly repository...${NC}"
    cd data
    python3 -c "import sys; sys.path.insert(0, '../..'); from promptly import Promptly; Promptly().init()" 2>/dev/null
    cd ..
    echo -e "${GREEN}✓ Promptly repository initialized${NC}"
else
    echo -e "${GREEN}✓ Promptly repository already initialized${NC}"
fi
echo ""

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo -e "${RED}Error: Port 8000 is already in use${NC}"
    echo "Please stop the process using port 8000 or change the port in .env"
    exit 1
fi

echo -e "${GREEN}✓ Port 8000 is available${NC}"
echo ""

# Display startup information
echo "========================================="
echo -e "${GREEN}Starting Promptly API Server${NC}"
echo "========================================="
echo ""
echo "API URL:          http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "ReDoc:            http://localhost:8000/redoc"
echo "Development API Key: pk_dev_key_12345"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================="
echo ""

# Start the server
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

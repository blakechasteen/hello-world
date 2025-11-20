#!/bin/bash

# Test runner script for Community Platform backend

echo "=================================="
echo "Community Platform Test Suite"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Install with: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Run different test suites based on argument
case "$1" in
    unit)
        echo -e "${BLUE}Running unit tests...${NC}"
        pytest tests/unit/ -v --tb=short
        ;;
    integration)
        echo -e "${BLUE}Running integration tests...${NC}"
        pytest tests/integration/ -v --tb=short
        ;;
    e2e)
        echo -e "${BLUE}Running end-to-end tests...${NC}"
        pytest tests/e2e/ -v --tb=short
        ;;
    fast)
        echo -e "${BLUE}Running fast tests (unit only)...${NC}"
        pytest tests/unit/ -v --tb=short -x
        ;;
    coverage)
        echo -e "${BLUE}Running all tests with coverage...${NC}"
        pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
        echo ""
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    specific)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify a test file${NC}"
            echo "Usage: ./run_tests.sh specific tests/unit/test_auth.py"
            exit 1
        fi
        echo -e "${BLUE}Running specific test: $2${NC}"
        pytest "$2" -v --tb=short
        ;;
    all|*)
        echo -e "${BLUE}Running all tests...${NC}"
        pytest tests/ -v --tb=short
        ;;
esac

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo ""
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi

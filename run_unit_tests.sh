#!/bin/bash
# HoloLoom Unit Test Runner
# Fast, isolated, elegant tests - Blake's style

set -e

echo "🚀 HoloLoom Unit Test Suite"
echo "============================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Set PYTHONPATH
export PYTHONPATH=.

echo "${YELLOW}📊 Running unit tests...${NC}"
echo ""

# Run tests with coverage
python -m pytest HoloLoom/tests/unit/ \
    -v \
    --tb=short \
    --cov=HoloLoom \
    --cov-report=term-missing \
    --cov-report=html \
    --durations=10 \
    "$@"

echo ""
echo "${GREEN}✅ Tests complete!${NC}"
echo ""
echo "📈 Coverage report generated: htmlcov/index.html"
echo ""

# Show coverage summary
if command -v coverage &> /dev/null; then
    echo "${YELLOW}📊 Coverage Summary:${NC}"
    coverage report --skip-covered | tail -20
fi

echo ""
echo "${GREEN}🎯 Target: <150ms per test${NC}"
echo "${GREEN}✨ Blake's neural engine is battle-tested!${NC}"

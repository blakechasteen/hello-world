#!/bin/bash
#
# HoloLoom Test Runner
# ====================
# Runs all Phase 2 tests: integration + performance
#

set -e  # Exit on error

echo "============================================================================"
echo "HoloLoom Phase 2 Test Suite"
echo "============================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}✗ pytest not found${NC}"
    echo "Install with: pip install pytest pytest-asyncio"
    exit 1
fi

echo -e "${GREEN}✓ pytest found${NC}"
echo ""

# Check Python dependencies
echo "Checking dependencies..."
MISSING_DEPS=()

for dep in numpy neo4j qdrant-client hdbscan sklearn fastapi; do
    if ! python -c "import $dep" 2>/dev/null; then
        MISSING_DEPS+=("$dep")
    fi
done

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Missing optional dependencies (tests will use mock mode):${NC}"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  - $dep"
    done
    echo ""
else
    echo -e "${GREEN}✓ All dependencies found${NC}"
    echo ""
fi

# Run integration tests
echo "============================================================================"
echo "1. Integration Tests"
echo "============================================================================"
echo ""

pytest HoloLoom/tests/integration/ -v --tb=short --asyncio-mode=auto

INTEGRATION_EXIT=$?

if [ $INTEGRATION_EXIT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Integration tests passed${NC}"
else
    echo ""
    echo -e "${RED}✗ Integration tests failed${NC}"
fi

echo ""

# Run performance tests
echo "============================================================================"
echo "2. Performance Tests"
echo "============================================================================"
echo ""

pytest HoloLoom/tests/performance/ -v --tb=short --asyncio-mode=auto -s

PERFORMANCE_EXIT=$?

if [ $PERFORMANCE_EXIT -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Performance tests passed${NC}"
else
    echo ""
    echo -e "${RED}✗ Performance tests failed${NC}"
fi

echo ""

# Summary
echo "============================================================================"
echo "Test Summary"
echo "============================================================================"
echo ""

if [ $INTEGRATION_EXIT -eq 0 ] && [ $PERFORMANCE_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
elif [ $INTEGRATION_EXIT -eq 0 ]; then
    echo -e "${YELLOW}⚠ Integration tests passed, but performance tests failed${NC}"
    exit 1
elif [ $PERFORMANCE_EXIT -eq 0 ]; then
    echo -e "${YELLOW}⚠ Performance tests passed, but integration tests failed${NC}"
    exit 1
else
    echo -e "${RED}✗ Both integration and performance tests failed${NC}"
    exit 1
fi

#!/bin/bash
# Deployment Validation Script
# ==============================
# Validates that all services are healthy and working
# Date: November 15, 2025

set -e  # Exit on error

echo "🚀 HoloLoom VoiceAgent Deployment Validation"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track failures
FAILURES=0

# Helper functions
check_service() {
    SERVICE=$1
    URL=$2
    echo -n "Checking $SERVICE... "

    if curl -sf "$URL" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ HEALTHY${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

check_port() {
    SERVICE=$1
    PORT=$2
    echo -n "Checking $SERVICE (port $PORT)... "

    if nc -z localhost $PORT 2>/dev/null; then
        echo -e "${GREEN}✓ LISTENING${NC}"
        return 0
    else
        echo -e "${RED}✗ NOT LISTENING${NC}"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

# Step 1: Check Docker services
echo "Step 1: Checking Docker Compose services"
echo "---------------------------------------"

if ! docker-compose -f docker-compose.voice.yml ps > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker Compose not running. Please start with:${NC}"
    echo "  docker-compose -f docker-compose.voice.yml up -d"
    exit 1
fi

# Check each service
docker-compose -f docker-compose.voice.yml ps

echo ""

# Step 2: Health checks
echo "Step 2: Service Health Checks"
echo "-----------------------------"

check_service "Prometheus" "http://localhost:9090/-/healthy"
check_service "Grafana" "http://localhost:3000/api/health"
check_service "Neo4j" "http://localhost:7474/"
check_port "Qdrant" 6333
check_port "Redis" 6379

echo ""

# Step 3: Run integration tests
echo "Step 3: Integration Tests"
echo "------------------------"

if [ -d "HoloLoom/voice/tests" ]; then
    echo "Running VoiceAgent tests..."
    PYTHONPATH=. pytest HoloLoom/voice/tests/test_voice_agent.py -v --tb=short

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    else
        echo -e "${RED}✗ TESTS FAILED${NC}"
        FAILURES=$((FAILURES + 1))
    fi
else
    echo -e "${YELLOW}⚠ Test directory not found, skipping tests${NC}"
fi

echo ""

# Step 4: Check Prometheus targets
echo "Step 4: Prometheus Targets"
echo "-------------------------"

TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets | length')
echo "Active targets: $TARGETS"

if [ "$TARGETS" -gt 0 ]; then
    echo -e "${GREEN}✓ Prometheus is scraping targets${NC}"
else
    echo -e "${RED}✗ No Prometheus targets configured${NC}"
    FAILURES=$((FAILURES + 1))
fi

echo ""

# Step 5: Manual smoke tests
echo "Step 5: Manual Smoke Tests (Interactive)"
echo "---------------------------------------"
echo ""
echo "Please run the following manual tests:"
echo ""
echo "1. Voice Input → Transcription → TTS:"
echo "   python demos/demo_voice_agent.py"
echo ""
echo "2. HoloLoom Integration:"
echo "   Check that queries are processed through WeavingOrchestrator"
echo ""
echo "3. Conversation Memory:"
echo "   Verify multi-turn conversations maintain context"
echo ""
echo "4. Turn-Taking Modes:"
echo "   Test button, VAD, and hybrid modes"
echo ""

# Step 6: Performance baseline
echo "Step 6: Performance Baseline"
echo "---------------------------"
echo ""
echo "Collecting baseline metrics..."

# Get container stats
echo "Container Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep voice

echo ""

# Step 7: Summary
echo "========================================="
echo "Validation Summary"
echo "========================================="
echo ""

if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Your deployment is healthy and ready to use!"
    echo ""
    echo "Next steps:"
    echo "  - Access Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Access Prometheus: http://localhost:9090"
    echo "  - Access Neo4j Browser: http://localhost:7474"
    echo ""
    exit 0
else
    echo -e "${RED}✗ $FAILURES CHECK(S) FAILED${NC}"
    echo ""
    echo "Please review the errors above and fix them."
    echo ""
    echo "Common issues:"
    echo "  - Services not started: docker-compose up -d"
    echo "  - Ports already in use: check with 'lsof -i :PORT'"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo ""
    exit 1
fi

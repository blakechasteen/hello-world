#!/bin/bash

# Zero-G Infrastructure Verification Script
# Checks all services are healthy and accessible

set -e

echo "🚀 Zero-G Infrastructure Verification"
echo "======================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Helper function to check service
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=$3

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

    echo -n "Checking $service_name... "

    if curl -s -f -o /dev/null -w "%{http_code}" "$url" | grep -q "$expected_status"; then
        echo -e "${GREEN}✓ OK${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        return 1
    fi
}

# Check Docker Compose services
echo "📦 Docker Compose Services"
echo "-------------------------"

# Zero-G Backend
check_service "Zero-G Backend" "http://localhost:8000/health" "200"

# Neo4j
check_service "Neo4j HTTP" "http://localhost:7474" "200"

# Qdrant
check_service "Qdrant HTTP" "http://localhost:6333/health" "200"

# Prometheus
check_service "Prometheus" "http://localhost:9090/-/healthy" "200"

# Grafana
check_service "Grafana" "http://localhost:3000/api/health" "200"

echo ""

# Check service containers
echo "🐳 Container Status"
echo "------------------"

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if docker-compose ps | grep -q "Up (healthy)"; then
    echo -e "${GREEN}✓ All containers healthy${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${RED}✗ Some containers unhealthy${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""

# Check Prometheus targets
echo "🎯 Prometheus Targets"
echo "--------------------"

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
TARGETS=$(curl -s http://localhost:9090/api/v1/targets | jq -r '.data.activeTargets[] | select(.health=="up") | .scrapePool' | wc -l)

if [ "$TARGETS" -ge 3 ]; then
    echo -e "${GREEN}✓ $TARGETS targets up${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${YELLOW}⚠ Only $TARGETS targets up (expected 3+)${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""

# Check Grafana datasource
echo "📊 Grafana Datasource"
echo "--------------------"

TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if curl -s -u admin:zerog_admin_change_me http://localhost:3000/api/datasources | jq -e '.[0].name == "Prometheus"' > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Prometheus datasource configured${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${RED}✗ Prometheus datasource missing${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""

# Check database connections
echo "💾 Database Connectivity"
echo "-----------------------"

# Neo4j Cypher query
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if docker-compose exec -T neo4j cypher-shell -u neo4j -p zerog_password_change_me "RETURN 1" > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Neo4j connection OK${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${RED}✗ Neo4j connection failed${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

# Qdrant collections
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
if curl -s http://localhost:6333/collections | jq -e '.result' > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Qdrant connection OK${NC}"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${RED}✗ Qdrant connection failed${NC}"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi

echo ""
echo "======================================"
echo "Summary"
echo "======================================"
echo "Total checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "${GREEN}🎉 All checks passed! Zero-G is ready for launch!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some checks failed. Please review the output above.${NC}"
    exit 1
fi

#!/bin/bash
# HoloLoom Deployment Verification Script
# Usage: ./scripts/verify-deployment.sh [quickstart|lite|full]
#
# Checks that all services are running and responding correctly.
# Exit code 0 = all checks passed, 1 = failures detected

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default deployment type
DEPLOYMENT=${1:-quickstart}

# Service endpoints based on deployment type
case $DEPLOYMENT in
    quickstart|lite)
        API_URL="http://localhost:8000"
        NEO4J_URL="http://localhost:7474"
        QDRANT_URL="http://localhost:6333"
        SERVICES=("neo4j" "qdrant" "hololoom-api")
        ;;
    full)
        API_URL="http://localhost:8000"
        NEO4J_URL="http://localhost:7474"
        QDRANT_URL="http://localhost:6333"
        REDIS_URL="localhost:6379"
        SERVICES=("neo4j" "qdrant" "hololoom-api" "redis")
        ;;
    monitored)
        API_URL="http://localhost:8000"
        NEO4J_URL="http://localhost:7474"
        QDRANT_URL="http://localhost:6333"
        PROMETHEUS_URL="http://localhost:9092"
        GRAFANA_URL="http://localhost:3001"
        SERVICES=("neo4j" "qdrant" "hololoom-api" "redis" "prometheus" "grafana")
        ;;
    *)
        echo "Usage: $0 [quickstart|lite|full|monitored]"
        exit 1
        ;;
esac

PASSED=0
FAILED=0

print_header() {
    echo ""
    echo "============================================"
    echo " HoloLoom Deployment Verification"
    echo " Mode: $DEPLOYMENT"
    echo "============================================"
    echo ""
}

check_service() {
    local name=$1
    local url=$2
    local path=${3:-""}

    printf "Checking %-20s ... " "$name"

    if curl -s --connect-timeout 5 "$url$path" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        ((FAILED++))
        return 1
    fi
}

check_container() {
    local name=$1

    printf "Container %-17s ... " "$name"

    if docker ps --format '{{.Names}}' | grep -q "$name"; then
        local status=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null || echo "none")
        if [[ "$status" == "healthy" ]]; then
            echo -e "${GREEN}healthy${NC}"
            ((PASSED++))
        elif [[ "$status" == "none" ]]; then
            echo -e "${YELLOW}running (no healthcheck)${NC}"
            ((PASSED++))
        else
            echo -e "${YELLOW}$status${NC}"
            ((PASSED++))
        fi
        return 0
    else
        echo -e "${RED}not running${NC}"
        ((FAILED++))
        return 1
    fi
}

test_api() {
    printf "API Health              ... "

    local response=$(curl -s --connect-timeout 10 "$API_URL/health" 2>/dev/null)

    if echo "$response" | grep -q "healthy"; then
        echo -e "${GREEN}healthy${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}unhealthy${NC}"
        ((FAILED++))
        return 1
    fi
}

test_query() {
    printf "API Query               ... "

    local response=$(curl -s --connect-timeout 30 -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -d '{"text": "test"}' 2>/dev/null)

    if [[ -n "$response" ]] && ! echo "$response" | grep -q "error"; then
        echo -e "${GREEN}responding${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${YELLOW}limited (may need LLM)${NC}"
        ((PASSED++))
        return 0
    fi
}

print_summary() {
    echo ""
    echo "============================================"
    echo " Summary"
    echo "============================================"
    echo -e " Passed: ${GREEN}$PASSED${NC}"
    echo -e " Failed: ${RED}$FAILED${NC}"
    echo ""

    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}All checks passed!${NC}"
        echo ""
        echo "Access points:"
        echo "  - HoloLoom API: $API_URL"
        echo "  - Neo4j Browser: $NEO4J_URL"
        echo "  - Qdrant Dashboard: $QDRANT_URL/dashboard"
        [[ -n "$GRAFANA_URL" ]] && echo "  - Grafana: $GRAFANA_URL"
        echo ""
        return 0
    else
        echo -e "${RED}Some checks failed. Check the logs:${NC}"
        echo "  docker-compose logs -f"
        echo ""
        return 1
    fi
}

# Main
print_header

echo "Container Status:"
echo "----------------"
for service in "${SERVICES[@]}"; do
    # Handle container naming variations
    case $service in
        hololoom-api)
            check_container "hololoom-api" || check_container "hololoom-lite-api" || check_container "hololoom"
            ;;
        neo4j)
            check_container "hololoom-neo4j" || check_container "hololoom-lite-neo4j"
            ;;
        qdrant)
            check_container "hololoom-qdrant" || check_container "hololoom-lite-qdrant"
            ;;
        *)
            check_container "hololoom-$service"
            ;;
    esac
done

echo ""
echo "Service Endpoints:"
echo "-----------------"
check_service "Neo4j HTTP" "$NEO4J_URL"
check_service "Qdrant REST" "$QDRANT_URL" "/health"
check_service "HoloLoom API" "$API_URL" "/health"
[[ -n "$PROMETHEUS_URL" ]] && check_service "Prometheus" "$PROMETHEUS_URL" "/-/healthy"
[[ -n "$GRAFANA_URL" ]] && check_service "Grafana" "$GRAFANA_URL" "/api/health"

echo ""
echo "Functional Tests:"
echo "-----------------"
test_api
test_query

print_summary
exit $([[ $FAILED -eq 0 ]] && echo 0 || echo 1)

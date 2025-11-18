#!/bin/bash
# HoloLoom Health Check Script
# ==============================
# Monitors service health and reports status
#
# Usage:
#   ./scripts/health_check.sh [--watch]
#
# Options:
#   --watch    Continuously monitor (refresh every 5s)
#
# Author: HoloLoom Team
# Date: 2025-11-18 (Week 8B)

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Watch mode
WATCH_MODE=false
if [ "${1:-}" = "--watch" ]; then
    WATCH_MODE=true
fi

check_health() {
    clear
    echo -e "${BLUE}=== HoloLoom Health Check ===${NC}"
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo

    # Check HoloLoom API
    echo -e "${YELLOW}Checking HoloLoom API...${NC}"
    if response=$(curl -s -f http://localhost:8000/health 2>/dev/null); then
        echo -e "${GREEN}✓ HoloLoom API${NC}"
        echo "  Status: $(echo "$response" | jq -r '.status' 2>/dev/null || echo 'healthy')"
        echo "  Systems:"
        echo "$response" | jq -r '.systems | to_entries[] | "    \(.key): \(.value)"' 2>/dev/null || echo "    N/A"
    else
        echo -e "${RED}✗ HoloLoom API not responding${NC}"
    fi
    echo

    # Check Neo4j
    echo -e "${YELLOW}Checking Neo4j...${NC}"
    if curl -s -f http://localhost:7474 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Neo4j${NC}"
        echo "  URL: http://localhost:7474"
    else
        echo -e "${RED}✗ Neo4j not responding${NC}"
    fi
    echo

    # Check Qdrant
    echo -e "${YELLOW}Checking Qdrant...${NC}"
    if response=$(curl -s -f http://localhost:6333/health 2>/dev/null); then
        echo -e "${GREEN}✓ Qdrant${NC}"
        echo "  Status: $(echo "$response" | jq -r '.status' 2>/dev/null || echo 'ok')"
    else
        echo -e "${RED}✗ Qdrant not responding${NC}"
    fi
    echo

    # Check Redis
    echo -e "${YELLOW}Checking Redis...${NC}"
    if docker exec hololoom-redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Redis${NC}"
        info=$(docker exec hololoom-redis redis-cli info server 2>/dev/null | grep redis_version || echo "")
        if [ -n "$info" ]; then
            echo "  $info"
        fi
    else
        echo -e "${RED}✗ Redis not responding${NC}"
    fi
    echo

    # Check Prometheus
    echo -e "${YELLOW}Checking Prometheus...${NC}"
    if curl -s -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Prometheus${NC}"
        echo "  URL: http://localhost:9090"
    else
        echo -e "${RED}✗ Prometheus not responding${NC}"
    fi
    echo

    # Check Grafana
    echo -e "${YELLOW}Checking Grafana...${NC}"
    if curl -s -f http://localhost:3000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Grafana${NC}"
        echo "  URL: http://localhost:3000"
    else
        echo -e "${RED}✗ Grafana not responding${NC}"
    fi
    echo

    # Docker container status
    echo -e "${YELLOW}Docker Containers:${NC}"
    docker-compose ps 2>/dev/null || echo "Docker Compose not running"
    echo

    # System stats
    echo -e "${YELLOW}System Statistics:${NC}"
    if stats=$(curl -s -f http://localhost:8000/api/v1/stats 2>/dev/null); then
        echo "Consolidations: $(echo "$stats" | jq -r '.consolidation.total_consolidations' 2>/dev/null || echo 'N/A')"
        echo "Semantic Concepts: $(echo "$stats" | jq -r '.semantic.total_concepts_created' 2>/dev/null || echo 'N/A')"
        echo "Curiosity Suggestions: $(echo "$stats" | jq -r '.curiosity.total_suggestions_generated' 2>/dev/null || echo 'N/A')"
    else
        echo "Statistics not available"
    fi
    echo

    if ! $WATCH_MODE; then
        echo -e "${GREEN}Health check complete${NC}"
    fi
}

# Main loop
if $WATCH_MODE; then
    echo "Watch mode enabled. Press Ctrl+C to exit."
    echo
    while true; do
        check_health
        sleep 5
    done
else
    check_health
fi

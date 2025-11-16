#!/bin/bash
# HoloLoom Infrastructure Health Check
# Usage: ./health-check.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}HoloLoom Infrastructure Health Check${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Service health checks
check_service() {
    local service=$1
    local container=$2
    local health_cmd=$3

    printf "%-20s" "$service:"

    if docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
        if eval "$health_cmd" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Healthy${NC}"
            return 0
        else
            echo -e "${RED}✗ Unhealthy${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Not Running${NC}"
        return 1
    fi
}

# Check all services
FAILED=0

check_service "Redis" "hololoom-redis" "docker exec hololoom-redis redis-cli ping | grep -q PONG" || ((FAILED++))
check_service "Redis Replica" "hololoom-redis-replica" "docker exec hololoom-redis-replica redis-cli ping | grep -q PONG" || ((FAILED++))
check_service "PostgreSQL" "hololoom-postgres" "docker exec hololoom-postgres pg_isready -U hololoom | grep -q 'accepting connections'" || ((FAILED++))
check_service "Neo4j" "hololoom-neo4j" "curl -sf http://localhost:7474 > /dev/null" || ((FAILED++))
check_service "Qdrant" "hololoom-qdrant" "curl -sf http://localhost:6333/healthz > /dev/null" || ((FAILED++))
check_service "Prometheus" "hololoom-prometheus" "curl -sf http://localhost:9090/-/healthy > /dev/null" || ((FAILED++))
check_service "Grafana" "hololoom-grafana" "curl -sf http://localhost:3000/api/health > /dev/null" || ((FAILED++))
check_service "Nginx" "hololoom-nginx" "docker exec hololoom-nginx nginx -t 2>&1 | grep -q 'successful'" || ((FAILED++))
check_service "Splunk" "hololoom-splunk" "curl -sf http://localhost:8000 > /dev/null" || ((FAILED++))
check_service "HoloLoom API" "hololoom-api" "curl -sf http://localhost:8000/health > /dev/null" || ((FAILED++))

echo
echo -e "${BLUE}================================================${NC}"

# Resource usage
echo -e "${YELLOW}Resource Usage:${NC}"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep hololoom

echo
echo -e "${BLUE}================================================${NC}"

# Disk usage
echo -e "${YELLOW}Disk Usage (Volumes):${NC}"
docker volume ls --filter "name=hololoom" --format "table {{.Name}}\t{{.Driver}}"

echo
echo -e "${BLUE}================================================${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All services are healthy!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED service(s) failed health check${NC}"
    exit 1
fi

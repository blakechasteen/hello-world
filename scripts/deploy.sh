#!/bin/bash
# HoloLoom Production Deployment Script
# =======================================
# Deploys HoloLoom API server with Docker Compose
#
# Usage:
#   ./scripts/deploy.sh [environment]
#
# Environments:
#   production (default)
#   staging
#   development
#
# Author: HoloLoom Team
# Date: 2025-11-18 (Week 8B)

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Configuration
ENVIRONMENT=${1:-production}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env.${ENVIRONMENT}"

echo -e "${GREEN}=== HoloLoom Deployment ===${NC}"
echo "Environment: ${ENVIRONMENT}"
echo "Project root: ${PROJECT_ROOT}"
echo

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}Error: Docker Compose not installed${NC}"
    echo "Install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker installed${NC}"
echo -e "${GREEN}✓ Docker Compose installed${NC}"
echo

# Load environment variables
if [ -f "${ENV_FILE}" ]; then
    echo -e "${YELLOW}Loading environment from ${ENV_FILE}${NC}"
    set -a
    source "${ENV_FILE}"
    set +a
else
    echo -e "${YELLOW}Warning: No environment file found at ${ENV_FILE}${NC}"
    echo "Using default environment variables"
fi
echo

# Check for required environment variables
if [ -z "${HOLOLOOM_API_KEY:-}" ]; then
    echo -e "${YELLOW}Warning: HOLOLOOM_API_KEY not set${NC}"
    echo "Generating random API key..."
    export HOLOLOOM_API_KEY=$(openssl rand -hex 32)
    echo "API Key: ${HOLOLOOM_API_KEY}"
    echo "Save this key for future use!"
    echo
fi

# Create required directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p "${PROJECT_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/config"
echo -e "${GREEN}✓ Directories created${NC}"
echo

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
cd "${PROJECT_ROOT}"
docker build -t hololoom-api:latest . || {
    echo -e "${RED}Error: Docker build failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Docker image built${NC}"
echo

# Stop existing containers
echo -e "${YELLOW}Stopping existing containers...${NC}"
docker-compose down || true
echo -e "${GREEN}✓ Containers stopped${NC}"
echo

# Start services
echo -e "${YELLOW}Starting services...${NC}"
docker-compose up -d

# Wait for services to be healthy
echo
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check health
echo
echo -e "${YELLOW}Checking service health...${NC}"

# Check API health
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo -e "${GREEN}✓ HoloLoom API is healthy${NC}"
else
    echo -e "${RED}✗ HoloLoom API is not responding${NC}"
    echo "Check logs: docker-compose logs hololoom-api"
fi

# Check Neo4j
if curl -f -s http://localhost:7474 > /dev/null; then
    echo -e "${GREEN}✓ Neo4j is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Neo4j is not responding (optional)${NC}"
fi

# Check Qdrant
if curl -f -s http://localhost:6333/health > /dev/null; then
    echo -e "${GREEN}✓ Qdrant is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Qdrant is not responding (optional)${NC}"
fi

# Check Redis
if redis-cli -h localhost -p 6379 ping > /dev/null 2>&1 || docker exec hololoom-redis redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Redis is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Redis is not responding${NC}"
fi

# Check Prometheus
if curl -f -s http://localhost:9090/-/healthy > /dev/null; then
    echo -e "${GREEN}✓ Prometheus is healthy${NC}"
else
    echo -e "${YELLOW}⚠ Prometheus is not responding${NC}"
fi

# Print access URLs
echo
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo
echo "Access URLs:"
echo "  API Server:   http://localhost:8000"
echo "  API Docs:     http://localhost:8000/docs"
echo "  Health Check: http://localhost:8000/health"
echo "  Metrics:      http://localhost:8000/metrics"
echo
echo "  Neo4j:        http://localhost:7474 (neo4j/hololoom123)"
echo "  Qdrant:       http://localhost:6333"
echo "  Prometheus:   http://localhost:9090"
echo "  Grafana:      http://localhost:3000 (admin/admin)"
echo
echo "API Key: ${HOLOLOOM_API_KEY}"
echo
echo "View logs:"
echo "  docker-compose logs -f hololoom-api"
echo
echo "Stop services:"
echo "  docker-compose down"
echo
echo -e "${GREEN}Deployment successful!${NC}"

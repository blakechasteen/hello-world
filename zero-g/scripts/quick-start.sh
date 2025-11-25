#!/bin/bash

# Zero-G Quick Start Script
# Brings up entire development environment and runs health checks

set -e

echo "🚀 Zero-G Quick Start"
echo "===================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Error: docker-compose is not installed."
    exit 1
fi

echo "✓ Docker is running"
echo "✓ docker-compose is available"
echo ""

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📦 Starting Zero-G services..."
echo "This will take ~60 seconds on first run (downloading images)"
echo ""

# Pull images first (for better progress visibility)
echo "Pulling Docker images..."
docker-compose pull

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to become healthy..."
echo "This usually takes 30-60 seconds..."
echo ""

# Wait for services with timeout
TIMEOUT=120
ELAPSED=0
SLEEP_INTERVAL=5

while [ $ELAPSED -lt $TIMEOUT ]; do
    HEALTHY_COUNT=$(docker-compose ps | grep -c "Up (healthy)" || true)
    TOTAL_SERVICES=5

    echo -ne "\rHealthy services: $HEALTHY_COUNT/$TOTAL_SERVICES (${ELAPSED}s elapsed)"

    if [ "$HEALTHY_COUNT" -eq "$TOTAL_SERVICES" ]; then
        echo ""
        echo ""
        echo "✅ All services are healthy!"
        break
    fi

    sleep $SLEEP_INTERVAL
    ELAPSED=$((ELAPSED + SLEEP_INTERVAL))
done

if [ $ELAPSED -ge $TIMEOUT ]; then
    echo ""
    echo ""
    echo "⚠️  Timeout waiting for services to become healthy"
    echo "Current status:"
    docker-compose ps
    echo ""
    echo "Check logs with: docker-compose logs -f"
    exit 1
fi

echo ""
echo "🎯 Running health checks..."
echo ""

# Run verification script if it exists
if [ -f "scripts/verify-infrastructure.sh" ]; then
    bash scripts/verify-infrastructure.sh
else
    echo "⚠️  Verification script not found, skipping detailed checks"
fi

echo ""
echo "======================================"
echo "🎉 Zero-G is ready!"
echo "======================================"
echo ""
echo "Access services at:"
echo ""
echo "  Backend:    http://localhost:8000"
echo "  Neo4j:      http://localhost:7474 (neo4j / zerog_password_change_me)"
echo "  Qdrant:     http://localhost:6333"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin / zerog_admin_change_me)"
echo ""
echo "Useful commands:"
echo "  View logs:  docker-compose logs -f"
echo "  Stop all:   docker-compose down"
echo "  Restart:    docker-compose restart <service>"
echo ""
echo "📚 Documentation: INFRASTRUCTURE_SETUP_GUIDE.md"
echo ""

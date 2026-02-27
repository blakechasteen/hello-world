#!/bin/bash
# hololoom Quick Start Script
# Usage: curl -fsSL https://raw.githubusercontent.com/your-org/hololoom/main/docs/self-hosting/quick-start.sh | bash

set -e

echo "==================================="
echo "   hololoom Self-Hosting Setup"
echo "==================================="
echo ""

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "ERROR: $1 is required but not installed."
        exit 1
    fi
}

echo "Checking prerequisites..."
check_command docker
check_command docker-compose
echo "Prerequisites OK"
echo ""

# Clone or update repository
if [ -d "hololoom" ]; then
    echo "hololoom directory exists, pulling latest..."
    cd hololoom
    git pull
else
    echo "Cloning hololoom..."
    git clone https://github.com/your-org/hololoom.git
    cd hololoom
fi
echo ""

# Start services
echo "Starting hololoom services..."
docker-compose -f docker-compose.lite.yml up -d
echo ""

# Wait for services to be healthy
echo "Waiting for services to be healthy..."
sleep 10

# Check health
echo "Checking service health..."
echo ""

# Neo4j
if curl -s http://localhost:7474 > /dev/null; then
    echo "  Neo4j:     OK (http://localhost:7474)"
else
    echo "  Neo4j:     STARTING (may take 30-60 seconds)"
fi

# Qdrant
if curl -s http://localhost:6333/health > /dev/null; then
    echo "  Qdrant:    OK (http://localhost:6333)"
else
    echo "  Qdrant:    STARTING"
fi

# hololoom API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "  hololoom:  OK (http://localhost:8000)"
else
    echo "  hololoom:  STARTING (may take 60-90 seconds for first run)"
fi

echo ""
echo "==================================="
echo "   hololoom is starting up!"
echo "==================================="
echo ""
echo "Services:"
echo "  - hololoom API: http://localhost:8000"
echo "  - Neo4j Browser: http://localhost:7474 (neo4j/hololoom)"
echo "  - Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "Quick test:"
echo "  curl http://localhost:8000/health"
echo ""
echo "View logs:"
echo "  docker-compose -f docker-compose.lite.yml logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose -f docker-compose.lite.yml down"
echo ""
echo "Documentation: docs/self-hosting/README.md"
echo ""

#!/bin/bash
# EdWIN Database Migration Script
# Run database migrations and schema updates
#
# Usage:
#   ./scripts/deploy/migrate.sh [namespace]
#
# Implementation Date: November 15, 2025

set -e

NAMESPACE="${1:-edwin}"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "🔄 Running database migrations..."

# Get API pod
API_POD=$(kubectl get pod -n "$NAMESPACE" -l component=api -o jsonpath='{.items[0].metadata.name}')

if [ -z "$API_POD" ]; then
    log_error "No API pod found in namespace $NAMESPACE"
    exit 1
fi

log_info "Using pod: $API_POD"

# Run migrations
kubectl exec -n "$NAMESPACE" "$API_POD" -- python -c "
from EduVerse.edwin.database import DatabaseManager
import asyncio
import logging

logging.basicConfig(level=logging.INFO)

async def migrate():
    print('Initializing database manager...')
    db = DatabaseManager()
    await db.initialize()

    print('Running migrations...')
    await db.run_migrations()

    print('Migrations complete')
    await db.close()

asyncio.run(migrate())
"

log_success "✅ Migrations complete"

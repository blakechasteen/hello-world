#!/bin/bash
# EdWIN Database Backup Script
# Backup Neo4j and Qdrant databases
#
# Usage:
#   ./scripts/deploy/backup.sh [namespace] [backup-dir]
#
# Implementation Date: November 15, 2025

set -e

NAMESPACE="${1:-edwin}"
BACKUP_DIR="${2:-./backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="$BACKUP_DIR/edwin_backup_$TIMESTAMP"

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

# Create backup directory
mkdir -p "$BACKUP_PATH"

log_info "💾 Starting database backup to $BACKUP_PATH..."

# ==============================================================================
# Backup Neo4j
# ==============================================================================

log_info "Backing up Neo4j..."

NEO4J_POD=$(kubectl get pod -n "$NAMESPACE" -l component=neo4j -o jsonpath='{.items[0].metadata.name}')

if [ -z "$NEO4J_POD" ]; then
    log_error "No Neo4j pod found"
    exit 1
fi

# Export Neo4j database
kubectl exec -n "$NAMESPACE" "$NEO4J_POD" -- cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
    "CALL apoc.export.cypher.all('backup.cypher', {format: 'cypher-shell'})" \
    > "$BACKUP_PATH/neo4j_export.log" 2>&1 || log_error "Neo4j export failed"

# Copy backup file
kubectl cp "$NAMESPACE/$NEO4J_POD:/var/lib/neo4j/import/backup.cypher" \
    "$BACKUP_PATH/neo4j_backup.cypher" || log_error "Failed to copy Neo4j backup"

log_success "Neo4j backup complete"

# ==============================================================================
# Backup Qdrant
# ==============================================================================

log_info "Backing up Qdrant..."

QDRANT_POD=$(kubectl get pod -n "$NAMESPACE" -l component=qdrant -o jsonpath='{.items[0].metadata.name}')

if [ -z "$QDRANT_POD" ]; then
    log_error "No Qdrant pod found"
    exit 1
fi

# Create Qdrant snapshot
SNAPSHOT_NAME="edwin_snapshot_$TIMESTAMP"
kubectl exec -n "$NAMESPACE" "$QDRANT_POD" -- curl -X POST \
    "http://localhost:6333/collections/snapshots" \
    -H "Content-Type: application/json" \
    -d '{"collection_name": "*", "snapshot_name": "'"$SNAPSHOT_NAME"'"}' \
    > "$BACKUP_PATH/qdrant_snapshot.log" 2>&1 || log_error "Qdrant snapshot failed"

log_success "Qdrant backup complete"

# ==============================================================================
# Backup Metadata
# ==============================================================================

log_info "Saving backup metadata..."

cat > "$BACKUP_PATH/metadata.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "namespace": "$NAMESPACE",
  "neo4j_pod": "$NEO4J_POD",
  "qdrant_pod": "$QDRANT_POD",
  "backup_files": [
    "neo4j_backup.cypher",
    "qdrant_snapshot.log"
  ]
}
EOF

log_success "Metadata saved"

# ==============================================================================
# Compress Backup
# ==============================================================================

log_info "Compressing backup..."

tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "edwin_backup_$TIMESTAMP"
rm -rf "$BACKUP_PATH"

log_success "Backup compressed: $BACKUP_PATH.tar.gz"

# ==============================================================================
# Cleanup Old Backups
# ==============================================================================

RETENTION_DAYS=30

log_info "Cleaning up backups older than $RETENTION_DAYS days..."

find "$BACKUP_DIR" -name "edwin_backup_*.tar.gz" -type f -mtime "+$RETENTION_DAYS" -delete

log_success "✅ Backup complete: $BACKUP_PATH.tar.gz"

# Display backup size
BACKUP_SIZE=$(du -h "$BACKUP_PATH.tar.gz" | cut -f1)
log_info "Backup size: $BACKUP_SIZE"

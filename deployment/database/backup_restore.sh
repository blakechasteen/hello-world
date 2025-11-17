#!/bin/bash
# HoloLoom Database Backup & Restore Script
# ==========================================
# Handles backup and restore for all databases
# Date: November 17, 2025

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Database credentials (from environment)
POSTGRES_HOST="${POSTGRES_HOST:-postgres-service}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-hololoom}"
POSTGRES_DB="${POSTGRES_DB:-hololoom}"

NEO4J_HOST="${NEO4J_HOST:-neo4j-service}"
NEO4J_PORT="${NEO4J_PORT:-7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ===========================================================================
# PostgreSQL Backup/Restore
# ===========================================================================

backup_postgres() {
    log_info "Backing up PostgreSQL database: $POSTGRES_DB"

    mkdir -p "$BACKUP_DIR/postgres"

    # Full database dump
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -F c \
        -b \
        -v \
        -f "$BACKUP_DIR/postgres/${POSTGRES_DB}_${TIMESTAMP}.dump"

    # Also create plain SQL backup for easy inspection
    PGPASSWORD="$POSTGRES_PASSWORD" pg_dump \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -f "$BACKUP_DIR/postgres/${POSTGRES_DB}_${TIMESTAMP}.sql"

    # Compress SQL backup
    gzip "$BACKUP_DIR/postgres/${POSTGRES_DB}_${TIMESTAMP}.sql"

    log_info "PostgreSQL backup completed: ${POSTGRES_DB}_${TIMESTAMP}.dump"
}

restore_postgres() {
    local backup_file="$1"

    log_info "Restoring PostgreSQL database from: $backup_file"

    # Drop existing database (WARNING: destructive!)
    log_warn "Dropping existing database: $POSTGRES_DB"
    PGPASSWORD="$POSTGRES_PASSWORD" dropdb \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        --if-exists \
        "$POSTGRES_DB"

    # Create fresh database
    PGPASSWORD="$POSTGRES_PASSWORD" createdb \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        "$POSTGRES_DB"

    # Restore from backup
    PGPASSWORD="$POSTGRES_PASSWORD" pg_restore \
        -h "$POSTGRES_HOST" \
        -p "$POSTGRES_PORT" \
        -U "$POSTGRES_USER" \
        -d "$POSTGRES_DB" \
        -v \
        "$backup_file"

    log_info "PostgreSQL restore completed"
}

# ===========================================================================
# Neo4j Backup/Restore
# ===========================================================================

backup_neo4j() {
    log_info "Backing up Neo4j database"

    mkdir -p "$BACKUP_DIR/neo4j"

    # Export to Cypher file using neo4j-admin dump
    # Note: This requires neo4j-admin access (run from Neo4j container)
    docker exec hololoom-neo4j neo4j-admin dump \
        --database=neo4j \
        --to=/backups/neo4j_${TIMESTAMP}.dump

    # Copy from container to host
    docker cp hololoom-neo4j:/backups/neo4j_${TIMESTAMP}.dump \
        "$BACKUP_DIR/neo4j/"

    log_info "Neo4j backup completed: neo4j_${TIMESTAMP}.dump"
}

restore_neo4j() {
    local backup_file="$1"

    log_info "Restoring Neo4j database from: $backup_file"

    # Stop Neo4j
    log_warn "Stopping Neo4j container"
    docker stop hololoom-neo4j

    # Copy backup to container
    docker cp "$backup_file" hololoom-neo4j:/var/lib/neo4j/import/

    # Restore from dump
    docker start hololoom-neo4j
    sleep 10  # Wait for Neo4j to start

    # Load dump (this is a simplified version - adjust for your setup)
    docker exec hololoom-neo4j neo4j-admin load \
        --from=/var/lib/neo4j/import/$(basename "$backup_file") \
        --database=neo4j \
        --force

    # Restart Neo4j
    docker restart hololoom-neo4j

    log_info "Neo4j restore completed"
}

# ===========================================================================
# Qdrant Backup/Restore
# ===========================================================================

backup_qdrant() {
    log_info "Backing up Qdrant vector database"

    mkdir -p "$BACKUP_DIR/qdrant"

    # Qdrant stores data in /qdrant/storage - backup entire directory
    docker exec hololoom-qdrant tar czf \
        /tmp/qdrant_${TIMESTAMP}.tar.gz \
        -C /qdrant storage

    # Copy from container
    docker cp hololoom-qdrant:/tmp/qdrant_${TIMESTAMP}.tar.gz \
        "$BACKUP_DIR/qdrant/"

    # Cleanup temp file
    docker exec hololoom-qdrant rm /tmp/qdrant_${TIMESTAMP}.tar.gz

    log_info "Qdrant backup completed: qdrant_${TIMESTAMP}.tar.gz"
}

restore_qdrant() {
    local backup_file="$1"

    log_info "Restoring Qdrant database from: $backup_file"

    # Stop Qdrant
    log_warn "Stopping Qdrant container"
    docker stop hololoom-qdrant

    # Copy backup to container
    docker cp "$backup_file" hololoom-qdrant:/tmp/qdrant_restore.tar.gz

    # Extract backup
    docker start hololoom-qdrant
    sleep 5

    docker exec hololoom-qdrant sh -c \
        "rm -rf /qdrant/storage/* && tar xzf /tmp/qdrant_restore.tar.gz -C /qdrant/"

    # Restart Qdrant
    docker restart hololoom-qdrant

    log_info "Qdrant restore completed"
}

# ===========================================================================
# Redis Backup (optional - ephemeral cache)
# ===========================================================================

backup_redis() {
    log_info "Backing up Redis cache (optional)"

    mkdir -p "$BACKUP_DIR/redis"

    # Trigger RDB snapshot
    docker exec hololoom-redis redis-cli SAVE

    # Copy RDB file
    docker cp hololoom-redis:/data/dump.rdb \
        "$BACKUP_DIR/redis/dump_${TIMESTAMP}.rdb"

    log_info "Redis backup completed: dump_${TIMESTAMP}.rdb"
}

# ===========================================================================
# Full Backup (All Databases)
# ===========================================================================

backup_all() {
    log_info "Starting full backup of all databases"

    backup_postgres
    backup_neo4j
    backup_qdrant
    backup_redis

    # Create manifest file
    cat > "$BACKUP_DIR/manifest_${TIMESTAMP}.txt" <<EOF
HoloLoom Backup Manifest
========================
Date: $(date)
Timestamp: $TIMESTAMP

Backups Included:
- PostgreSQL: ${POSTGRES_DB}_${TIMESTAMP}.dump
- Neo4j: neo4j_${TIMESTAMP}.dump
- Qdrant: qdrant_${TIMESTAMP}.tar.gz
- Redis: dump_${TIMESTAMP}.rdb (optional)

Restore Instructions:
1. Restore PostgreSQL: ./backup_restore.sh restore-postgres $BACKUP_DIR/postgres/${POSTGRES_DB}_${TIMESTAMP}.dump
2. Restore Neo4j: ./backup_restore.sh restore-neo4j $BACKUP_DIR/neo4j/neo4j_${TIMESTAMP}.dump
3. Restore Qdrant: ./backup_restore.sh restore-qdrant $BACKUP_DIR/qdrant/qdrant_${TIMESTAMP}.tar.gz
EOF

    log_info "Full backup completed successfully"
    log_info "Manifest: $BACKUP_DIR/manifest_${TIMESTAMP}.txt"
}

# ===========================================================================
# Cleanup Old Backups
# ===========================================================================

cleanup_old_backups() {
    log_info "Cleaning up backups older than $RETENTION_DAYS days"

    find "$BACKUP_DIR" -type f -mtime +$RETENTION_DAYS -delete

    log_info "Cleanup completed"
}

# ===========================================================================
# Main CLI
# ===========================================================================

case "${1:-}" in
    backup-postgres)
        backup_postgres
        ;;
    backup-neo4j)
        backup_neo4j
        ;;
    backup-qdrant)
        backup_qdrant
        ;;
    backup-redis)
        backup_redis
        ;;
    backup-all)
        backup_all
        cleanup_old_backups
        ;;
    restore-postgres)
        restore_postgres "${2:-}"
        ;;
    restore-neo4j)
        restore_neo4j "${2:-}"
        ;;
    restore-qdrant)
        restore_qdrant "${2:-}"
        ;;
    cleanup)
        cleanup_old_backups
        ;;
    *)
        echo "Usage: $0 {backup-all|backup-postgres|backup-neo4j|backup-qdrant|backup-redis|restore-postgres|restore-neo4j|restore-qdrant|cleanup}"
        echo ""
        echo "Examples:"
        echo "  $0 backup-all                                  # Backup all databases"
        echo "  $0 restore-postgres /backups/postgres/hololoom_20251117_120000.dump"
        echo "  $0 cleanup                                     # Remove old backups"
        exit 1
        ;;
esac

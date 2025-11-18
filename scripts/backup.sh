#!/bin/bash
# HoloLoom Backup Script
# =======================
# Backs up data, logs, and database volumes
#
# Usage:
#   ./scripts/backup.sh [backup_dir]
#
# Default backup directory: ./backups
#
# Author: HoloLoom Team
# Date: 2025-11-18 (Week 8B)

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${1:-${PROJECT_ROOT}/backups}"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_NAME="hololoom_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

echo -e "${GREEN}=== HoloLoom Backup ===${NC}"
echo "Timestamp: ${TIMESTAMP}"
echo "Backup directory: ${BACKUP_PATH}"
echo

# Create backup directory
mkdir -p "${BACKUP_PATH}"

# Backup application data
echo -e "${YELLOW}Backing up application data...${NC}"
if [ -d "${PROJECT_ROOT}/data" ]; then
    cp -r "${PROJECT_ROOT}/data" "${BACKUP_PATH}/data"
    echo -e "${GREEN}✓ Application data backed up${NC}"
else
    echo -e "${YELLOW}⚠ No application data to backup${NC}"
fi
echo

# Backup logs
echo -e "${YELLOW}Backing up logs...${NC}"
if [ -d "${PROJECT_ROOT}/logs" ]; then
    cp -r "${PROJECT_ROOT}/logs" "${BACKUP_PATH}/logs"
    echo -e "${GREEN}✓ Logs backed up${NC}"
else
    echo -e "${YELLOW}⚠ No logs to backup${NC}"
fi
echo

# Backup Docker volumes
echo -e "${YELLOW}Backing up Docker volumes...${NC}"

# Neo4j data
if docker volume inspect hello-world_neo4j_data > /dev/null 2>&1; then
    echo "Backing up Neo4j data..."
    docker run --rm \
        -v hello-world_neo4j_data:/data \
        -v "${BACKUP_PATH}:/backup" \
        alpine \
        tar czf /backup/neo4j_data.tar.gz -C /data .
    echo -e "${GREEN}✓ Neo4j data backed up${NC}"
else
    echo -e "${YELLOW}⚠ Neo4j volume not found${NC}"
fi

# Qdrant data
if docker volume inspect hello-world_qdrant_data > /dev/null 2>&1; then
    echo "Backing up Qdrant data..."
    docker run --rm \
        -v hello-world_qdrant_data:/data \
        -v "${BACKUP_PATH}:/backup" \
        alpine \
        tar czf /backup/qdrant_data.tar.gz -C /data .
    echo -e "${GREEN}✓ Qdrant data backed up${NC}"
else
    echo -e "${YELLOW}⚠ Qdrant volume not found${NC}"
fi

# Redis data
if docker volume inspect hello-world_redis-data > /dev/null 2>&1; then
    echo "Backing up Redis data..."
    docker run --rm \
        -v hello-world_redis-data:/data \
        -v "${BACKUP_PATH}:/backup" \
        alpine \
        tar czf /backup/redis_data.tar.gz -C /data .
    echo -e "${GREEN}✓ Redis data backed up${NC}"
else
    echo -e "${YELLOW}⚠ Redis volume not found${NC}"
fi

# Prometheus data
if docker volume inspect hello-world_prometheus-data > /dev/null 2>&1; then
    echo "Backing up Prometheus data..."
    docker run --rm \
        -v hello-world_prometheus-data:/data \
        -v "${BACKUP_PATH}:/backup" \
        alpine \
        tar czf /backup/prometheus_data.tar.gz -C /data .
    echo -e "${GREEN}✓ Prometheus data backed up${NC}"
else
    echo -e "${YELLOW}⚠ Prometheus volume not found${NC}"
fi

echo

# Create metadata file
echo -e "${YELLOW}Creating backup metadata...${NC}"
cat > "${BACKUP_PATH}/metadata.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "date": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "git_branch": "$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')",
  "docker_compose_version": "$(docker-compose version --short 2>/dev/null || echo 'unknown')",
  "backup_contents": [
    "data/",
    "logs/",
    "neo4j_data.tar.gz",
    "qdrant_data.tar.gz",
    "redis_data.tar.gz",
    "prometheus_data.tar.gz"
  ]
}
EOF
echo -e "${GREEN}✓ Metadata created${NC}"
echo

# Calculate backup size
BACKUP_SIZE=$(du -sh "${BACKUP_PATH}" | cut -f1)
echo -e "${GREEN}=== Backup Complete ===${NC}"
echo "Backup location: ${BACKUP_PATH}"
echo "Backup size: ${BACKUP_SIZE}"
echo
echo "To restore from this backup:"
echo "  ./scripts/restore.sh ${BACKUP_PATH}"
echo

# Optional: Create compressed archive
read -p "Create compressed archive? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Creating compressed archive...${NC}"
    tar czf "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" -C "${BACKUP_DIR}" "${BACKUP_NAME}"
    ARCHIVE_SIZE=$(du -sh "${BACKUP_DIR}/${BACKUP_NAME}.tar.gz" | cut -f1)
    echo -e "${GREEN}✓ Archive created: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz (${ARCHIVE_SIZE})${NC}"

    read -p "Remove uncompressed backup? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${BACKUP_PATH}"
        echo -e "${GREEN}✓ Uncompressed backup removed${NC}"
    fi
fi

echo
echo -e "${GREEN}Backup complete!${NC}"

#!/bin/bash
# EdWIN Database Backup Script
# Backs up Neo4j, Qdrant, and Redis data
#
# Usage:
#   ./scripts/backup_db.sh [backup_dir]
#
# Implementation Date: November 17, 2025

set -e

BACKUP_DIR="${1:-/var/backups/edwin}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="edwin_backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}EdWIN Database Backup${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Create backup directory
mkdir -p "${BACKUP_PATH}"

# Backup Neo4j
echo -e "\n${YELLOW}Backing up Neo4j...${NC}"
if command -v docker &> /dev/null; then
    # Docker-based backup
    docker exec edwin-neo4j neo4j-admin database dump neo4j --to-path=/backups/
    docker cp edwin-neo4j:/backups/neo4j.dump "${BACKUP_PATH}/neo4j.dump"
    echo -e "${GREEN}✅ Neo4j backup complete${NC}"
elif command -v kubectl &> /dev/null; then
    # Kubernetes-based backup
    NEO4J_POD=$(kubectl get pod -n edwin -l app=neo4j -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n edwin ${NEO4J_POD} -- neo4j-admin database dump neo4j --to-path=/tmp/
    kubectl cp edwin/${NEO4J_POD}:/tmp/neo4j.dump "${BACKUP_PATH}/neo4j.dump"
    echo -e "${GREEN}✅ Neo4j backup complete${NC}"
else
    echo -e "${YELLOW}⚠️  Neo4j backup skipped (no docker/kubectl)${NC}"
fi

# Backup Qdrant
echo -e "\n${YELLOW}Backing up Qdrant...${NC}"
if command -v docker &> /dev/null; then
    # Create Qdrant snapshot
    curl -X POST "http://localhost:6333/collections/edwin/snapshots" > /dev/null 2>&1
    # Copy snapshot
    docker cp edwin-qdrant:/qdrant/storage/collections/edwin/snapshots "${BACKUP_PATH}/qdrant_snapshots"
    echo -e "${GREEN}✅ Qdrant backup complete${NC}"
elif command -v kubectl &> /dev/null; then
    QDRANT_POD=$(kubectl get pod -n edwin -l app=qdrant -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n edwin ${QDRANT_POD} -- curl -X POST "http://localhost:6333/collections/edwin/snapshots"
    kubectl cp edwin/${QDRANT_POD}:/qdrant/storage/collections "${BACKUP_PATH}/qdrant_snapshots"
    echo -e "${GREEN}✅ Qdrant backup complete${NC}"
else
    echo -e "${YELLOW}⚠️  Qdrant backup skipped${NC}"
fi

# Backup Redis (RDB dump)
echo -e "\n${YELLOW}Backing up Redis...${NC}"
if command -v docker &> /dev/null; then
    docker exec edwin-redis redis-cli SAVE > /dev/null
    docker cp edwin-redis:/data/dump.rdb "${BACKUP_PATH}/redis.rdb"
    echo -e "${GREEN}✅ Redis backup complete${NC}"
elif command -v kubectl &> /dev/null; then
    REDIS_POD=$(kubectl get pod -n edwin -l app=redis -o jsonpath='{.items[0].metadata.name}')
    kubectl exec -n edwin ${REDIS_POD} -- redis-cli SAVE
    kubectl cp edwin/${REDIS_POD}:/data/dump.rdb "${BACKUP_PATH}/redis.rdb"
    echo -e "${GREEN}✅ Redis backup complete${NC}"
else
    echo -e "${YELLOW}⚠️  Redis backup skipped${NC}"
fi

# Create metadata file
echo -e "\n${YELLOW}Creating backup metadata...${NC}"
cat > "${BACKUP_PATH}/metadata.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "version": "1.0.0",
  "components": {
    "neo4j": $([ -f "${BACKUP_PATH}/neo4j.dump" ] && echo "true" || echo "false"),
    "qdrant": $([ -d "${BACKUP_PATH}/qdrant_snapshots" ] && echo "true" || echo "false"),
    "redis": $([ -f "${BACKUP_PATH}/redis.rdb" ] && echo "true" || echo "false")
  }
}
EOF

# Compress backup
echo -e "\n${YELLOW}Compressing backup...${NC}"
cd "${BACKUP_DIR}"
tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}"
rm -rf "${BACKUP_NAME}"

BACKUP_SIZE=$(du -h "${BACKUP_NAME}.tar.gz" | cut -f1)

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Backup complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Backup Details:"
echo "  Location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
echo "  Size: ${BACKUP_SIZE}"
echo "  Timestamp: ${TIMESTAMP}"
echo ""
echo "To restore:"
echo "  ./scripts/restore_db.sh ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"

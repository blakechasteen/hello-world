#!/bin/bash
# EdWIN Database Restore Script
# Restores Neo4j, Qdrant, and Redis from backup
#
# Usage:
#   ./scripts/restore_db.sh <backup_file>
#
# Implementation Date: November 17, 2025

set -e

BACKUP_FILE="$1"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

if [ -z "$BACKUP_FILE" ]; then
    echo -e "${RED}Usage: $0 <backup_file.tar.gz>${NC}"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo -e "${RED}❌ Backup file not found: $BACKUP_FILE${NC}"
    exit 1
fi

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}EdWIN Database Restore${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: This will overwrite existing data!${NC}"
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^yes$ ]]; then
    echo "Restore cancelled"
    exit 0
fi

# Extract backup
TEMP_DIR=$(mktemp -d)
echo -e "\n${YELLOW}Extracting backup...${NC}"
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)
RESTORE_PATH="$TEMP_DIR/$BACKUP_NAME"

if [ ! -d "$RESTORE_PATH" ]; then
    echo -e "${RED}❌ Invalid backup archive${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Read metadata
if [ -f "$RESTORE_PATH/metadata.json" ]; then
    echo -e "\n${GREEN}Backup metadata:${NC}"
    cat "$RESTORE_PATH/metadata.json"
fi

# Restore Neo4j
if [ -f "$RESTORE_PATH/neo4j.dump" ]; then
    echo -e "\n${YELLOW}Restoring Neo4j...${NC}"
    if command -v docker &> /dev/null; then
        # Stop Neo4j
        docker exec edwin-neo4j neo4j stop
        # Copy dump file
        docker cp "$RESTORE_PATH/neo4j.dump" edwin-neo4j:/tmp/neo4j.dump
        # Load dump
        docker exec edwin-neo4j neo4j-admin database load neo4j --from-path=/tmp/ --overwrite-destination=true
        # Start Neo4j
        docker exec edwin-neo4j neo4j start
        echo -e "${GREEN}✅ Neo4j restored${NC}"
    elif command -v kubectl &> /dev/null; then
        NEO4J_POD=$(kubectl get pod -n edwin -l app=neo4j -o jsonpath='{.items[0].metadata.name}')
        kubectl cp "$RESTORE_PATH/neo4j.dump" edwin/${NEO4J_POD}:/tmp/neo4j.dump
        kubectl exec -n edwin ${NEO4J_POD} -- neo4j-admin database load neo4j --from-path=/tmp/ --overwrite-destination=true
        kubectl delete pod -n edwin ${NEO4J_POD}  # Restart pod
        echo -e "${GREEN}✅ Neo4j restored${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Neo4j dump not found in backup${NC}"
fi

# Restore Qdrant
if [ -d "$RESTORE_PATH/qdrant_snapshots" ]; then
    echo -e "\n${YELLOW}Restoring Qdrant...${NC}"
    if command -v docker &> /dev/null; then
        docker cp "$RESTORE_PATH/qdrant_snapshots" edwin-qdrant:/qdrant/storage/collections/
        docker restart edwin-qdrant
        echo -e "${GREEN}✅ Qdrant restored${NC}"
    elif command -v kubectl &> /dev/null; then
        QDRANT_POD=$(kubectl get pod -n edwin -l app=qdrant -o jsonpath='{.items[0].metadata.name}')
        kubectl cp "$RESTORE_PATH/qdrant_snapshots" edwin/${QDRANT_POD}:/qdrant/storage/collections/
        kubectl delete pod -n edwin ${QDRANT_POD}  # Restart pod
        echo -e "${GREEN}✅ Qdrant restored${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Qdrant snapshots not found in backup${NC}"
fi

# Restore Redis
if [ -f "$RESTORE_PATH/redis.rdb" ]; then
    echo -e "\n${YELLOW}Restoring Redis...${NC}"
    if command -v docker &> /dev/null; then
        docker exec edwin-redis redis-cli SHUTDOWN SAVE
        sleep 2
        docker cp "$RESTORE_PATH/redis.rdb" edwin-redis:/data/dump.rdb
        docker start edwin-redis
        echo -e "${GREEN}✅ Redis restored${NC}"
    elif command -v kubectl &> /dev/null; then
        REDIS_POD=$(kubectl get pod -n edwin -l app=redis -o jsonpath='{.items[0].metadata.name}')
        kubectl exec -n edwin ${REDIS_POD} -- redis-cli SHUTDOWN SAVE
        sleep 2
        kubectl cp "$RESTORE_PATH/redis.rdb" edwin/${REDIS_POD}:/data/dump.rdb
        kubectl delete pod -n edwin ${REDIS_POD}  # Restart pod
        echo -e "${GREEN}✅ Redis restored${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Redis dump not found in backup${NC}"
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Restore complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Please verify the restored data:"
echo "  ./scripts/health_check.sh"

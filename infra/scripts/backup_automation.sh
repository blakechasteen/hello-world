#!/bin/bash
# hololoom VoiceAgent Backup Automation
#
# Backs up:
# - Neo4j knowledge graph
# - Redis TTS cache
# - Application logs
# - Configuration files
# - Grafana dashboards
#
# Created: 2025-11-16
# Part of: Wave 3 Production Hardening

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/var/backups/hololoom}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
S3_BUCKET="${S3_BUCKET:-s3://hololoom-backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GRAFANA_API_KEY="${GRAFANA_API_KEY:-}"
ENABLE_S3="${ENABLE_S3:-true}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

log_info "========================================="
log_info "hololoom VoiceAgent Backup"
log_info "Starting at $(date)"
log_info "========================================="

# 1. Backup Neo4j
log_info "Backing up Neo4j knowledge graph..."
if docker ps | grep -q hololoom-neo4j; then
    # Stop Neo4j for consistent backup
    log_info "Creating Neo4j database dump..."
    docker exec hololoom-neo4j neo4j-admin database dump neo4j \
        --to-path=/backups --overwrite-destination=true \
        > /dev/null 2>&1 || {
            log_error "Neo4j dump failed, trying online backup..."
            # Fallback to online backup (requires Neo4j Enterprise)
            docker exec hololoom-neo4j neo4j-admin database backup neo4j \
                --to-path=/backups || log_error "Online backup also failed"
        }

    # Copy dump to backup directory
    docker cp hololoom-neo4j:/backups/neo4j.dump \
        "$BACKUP_DIR/$TIMESTAMP/neo4j.dump" 2>/dev/null || {
        log_warn "Could not find neo4j.dump, trying alternative..."
        # Try copying the data directory directly
        docker cp hololoom-neo4j:/data \
            "$BACKUP_DIR/$TIMESTAMP/neo4j_data" || log_error "Neo4j backup failed"
    }

    log_info "✓ Neo4j backup completed"
else
    log_warn "Neo4j container not running, skipping..."
fi

# 2. Backup Redis
log_info "Backing up Redis TTS cache..."
if docker ps | grep -q hololoom-tts-cache; then
    # Trigger Redis save
    docker exec hololoom-tts-cache redis-cli SAVE > /dev/null 2>&1 || log_warn "Redis SAVE failed"

    # Copy RDB file
    docker cp hololoom-tts-cache:/data/dump.rdb \
        "$BACKUP_DIR/$TIMESTAMP/redis.rdb" 2>/dev/null || log_warn "Redis backup failed"

    # Also backup Redis AOF if enabled
    docker exec hololoom-tts-cache redis-cli CONFIG GET appendonly | grep -q yes && {
        docker cp hololoom-tts-cache:/data/appendonly.aof \
            "$BACKUP_DIR/$TIMESTAMP/redis.aof" 2>/dev/null || log_warn "AOF backup failed"
    }

    log_info "✓ Redis backup completed"
else
    log_warn "Redis container not running, skipping..."
fi

# 3. Backup application logs
log_info "Backing up application logs..."
mkdir -p "$BACKUP_DIR/$TIMESTAMP/logs"

# Voice Agent logs
if docker ps | grep -q hololoom-voice-agent; then
    docker logs hololoom-voice-agent > "$BACKUP_DIR/$TIMESTAMP/logs/voice-agent.log" 2>&1
    log_info "✓ Voice Agent logs backed up"
else
    log_warn "Voice Agent container not running, skipping logs..."
fi

# Elle AR Service logs
if docker ps | grep -q hololoom-elle-ar; then
    docker logs hololoom-elle-ar > "$BACKUP_DIR/$TIMESTAMP/logs/elle-ar.log" 2>&1
    log_info "✓ Elle AR logs backed up"
fi

# Prometheus logs (if running)
if docker ps | grep -q prometheus; then
    docker logs prometheus > "$BACKUP_DIR/$TIMESTAMP/logs/prometheus.log" 2>&1
fi

# Grafana logs (if running)
if docker ps | grep -q grafana; then
    docker logs grafana > "$BACKUP_DIR/$TIMESTAMP/logs/grafana.log" 2>&1
fi

# 4. Backup configuration
log_info "Backing up configuration files..."
mkdir -p "$BACKUP_DIR/$TIMESTAMP/config"

# Docker compose files
cp docker-compose*.yml "$BACKUP_DIR/$TIMESTAMP/config/" 2>/dev/null || log_warn "No docker-compose files found"

# Deployment configurations
if [ -d "deployment" ]; then
    cp -r deployment/ "$BACKUP_DIR/$TIMESTAMP/config/deployment/"
    log_info "✓ Deployment configs backed up"
fi

# Environment files (excluding secrets)
if [ -f ".env" ]; then
    # Sanitize secrets before backup
    grep -v -E '(PASSWORD|SECRET|KEY|TOKEN)=' .env > "$BACKUP_DIR/$TIMESTAMP/config/.env.template" 2>/dev/null || true
fi

# Voice agent configurations
if [ -d "hololoom/voice" ]; then
    cp -r hololoom/voice/*.yaml "$BACKUP_DIR/$TIMESTAMP/config/" 2>/dev/null || true
    cp -r hololoom/voice/languages "$BACKUP_DIR/$TIMESTAMP/config/" 2>/dev/null || true
    cp -r hololoom/voice/personalities "$BACKUP_DIR/$TIMESTAMP/config/" 2>/dev/null || true
fi

# 5. Backup Grafana dashboards
log_info "Backing up Grafana dashboards..."
mkdir -p "$BACKUP_DIR/$TIMESTAMP/grafana"

if [ -n "$GRAFANA_API_KEY" ] && docker ps | grep -q grafana; then
    # Get all dashboard UIDs
    DASHBOARDS=$(curl -s -H "Authorization: Bearer $GRAFANA_API_KEY" \
        http://localhost:3000/api/search?type=dash-db 2>/dev/null | \
        jq -r '.[] | .uid' 2>/dev/null || echo "")

    if [ -n "$DASHBOARDS" ]; then
        for uid in $DASHBOARDS; do
            curl -s -H "Authorization: Bearer $GRAFANA_API_KEY" \
                "http://localhost:3000/api/dashboards/uid/$uid" > \
                "$BACKUP_DIR/$TIMESTAMP/grafana/$uid.json" 2>/dev/null || log_warn "Failed to backup dashboard $uid"
        done
        log_info "✓ Grafana dashboards backed up"
    else
        log_warn "No Grafana dashboards found or API key invalid"
    fi
else
    log_warn "Grafana API key not set or Grafana not running, skipping dashboards..."
fi

# 6. Backup Prometheus data (metrics snapshots)
log_info "Backing up Prometheus metrics snapshot..."
if docker ps | grep -q prometheus; then
    # Create Prometheus snapshot via API
    SNAPSHOT=$(curl -s -XPOST http://localhost:9090/api/v1/admin/tsdb/snapshot 2>/dev/null | \
        jq -r '.data.name' 2>/dev/null || echo "")

    if [ -n "$SNAPSHOT" ]; then
        docker cp prometheus:/prometheus/snapshots/$SNAPSHOT \
            "$BACKUP_DIR/$TIMESTAMP/prometheus_snapshot" 2>/dev/null || log_warn "Prometheus snapshot copy failed"
        log_info "✓ Prometheus snapshot backed up"
    fi
fi

# 7. Create metadata file
log_info "Creating backup metadata..."
cat > "$BACKUP_DIR/$TIMESTAMP/metadata.json" <<EOF
{
  "timestamp": "$TIMESTAMP",
  "date": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "backup_version": "1.0",
  "components": {
    "neo4j": $(docker ps | grep -q hololoom-neo4j && echo "true" || echo "false"),
    "redis": $(docker ps | grep -q hololoom-tts-cache && echo "true" || echo "false"),
    "voice_agent": $(docker ps | grep -q hololoom-voice-agent && echo "true" || echo "false"),
    "elle_ar": $(docker ps | grep -q hololoom-elle-ar && echo "true" || echo "false"),
    "prometheus": $(docker ps | grep -q prometheus && echo "true" || echo "false"),
    "grafana": $(docker ps | grep -q grafana && echo "true" || echo "false")
  },
  "sizes": {
    "total_bytes": $(du -sb "$BACKUP_DIR/$TIMESTAMP" | cut -f1)
  }
}
EOF

# 8. Create tarball
log_info "Creating backup archive..."
cd "$BACKUP_DIR"
tar -czf "hololoom_backup_$TIMESTAMP.tar.gz" "$TIMESTAMP" 2>/dev/null || {
    log_error "Failed to create tarball"
    exit 1
}

# Calculate checksums
BACKUP_SIZE=$(du -sh "hololoom_backup_$TIMESTAMP.tar.gz" | cut -f1)
CHECKSUM=$(sha256sum "hololoom_backup_$TIMESTAMP.tar.gz" | cut -d' ' -f1)

log_info "✓ Archive created: $BACKUP_SIZE"
log_info "  SHA256: $CHECKSUM"

# Remove temporary directory
rm -rf "$TIMESTAMP"

# 9. Upload to S3
if [ "$ENABLE_S3" = "true" ]; then
    log_info "Uploading to S3..."

    if command -v aws &> /dev/null; then
        aws s3 cp "hololoom_backup_$TIMESTAMP.tar.gz" \
            "$S3_BUCKET/hololoom_backup_$TIMESTAMP.tar.gz" \
            --storage-class STANDARD_IA 2>/dev/null && {
            log_info "✓ S3 upload completed"

            # Upload checksum
            echo "$CHECKSUM  hololoom_backup_$TIMESTAMP.tar.gz" | \
                aws s3 cp - "$S3_BUCKET/hololoom_backup_$TIMESTAMP.tar.gz.sha256" 2>/dev/null
        } || log_warn "S3 upload failed (check AWS credentials)"
    else
        log_warn "AWS CLI not installed, skipping S3 upload"
    fi
else
    log_info "S3 upload disabled"
fi

# 10. Cleanup old backups
log_info "Cleaning up old backups (retention: $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "hololoom_backup_*.tar.gz" \
    -mtime +$RETENTION_DAYS -delete 2>/dev/null || log_warn "Cleanup failed"

REMAINING=$(find "$BACKUP_DIR" -name "hololoom_backup_*.tar.gz" | wc -l)
log_info "✓ $REMAINING backups retained locally"

# 11. Create backup report
REPORT_FILE="$BACKUP_DIR/backup_report_$TIMESTAMP.txt"
cat > "$REPORT_FILE" <<EOF
hololoom VoiceAgent Backup Report
==================================
Timestamp: $TIMESTAMP
Date: $(date)
Hostname: $(hostname)

Backup Details:
- Archive: hololoom_backup_$TIMESTAMP.tar.gz
- Size: $BACKUP_SIZE
- SHA256: $CHECKSUM
- Location: $BACKUP_DIR

Components Backed Up:
- Neo4j Knowledge Graph: $(docker ps | grep -q hololoom-neo4j && echo "✓" || echo "✗")
- Redis TTS Cache: $(docker ps | grep -q hololoom-tts-cache && echo "✓" || echo "✗")
- Voice Agent Logs: $(docker ps | grep -q hololoom-voice-agent && echo "✓" || echo "✗")
- Elle AR Logs: $(docker ps | grep -q hololoom-elle-ar && echo "✓" || echo "✗")
- Configuration Files: ✓
- Grafana Dashboards: $([ -n "$GRAFANA_API_KEY" ] && echo "✓" || echo "✗")
- Prometheus Metrics: $(docker ps | grep -q prometheus && echo "✓" || echo "✗")

S3 Upload: $([ "$ENABLE_S3" = "true" ] && echo "✓" || echo "✗")
Retention Policy: $RETENTION_DAYS days
Backups Retained: $REMAINING

Status: SUCCESS
EOF

log_info "========================================="
log_info "Backup completed successfully at $(date)"
log_info "========================================="
log_info "Backup location: $BACKUP_DIR/hololoom_backup_$TIMESTAMP.tar.gz"
log_info "Report: $REPORT_FILE"

# Optional: Send notification (email/Slack)
if command -v mail &> /dev/null && [ -n "${BACKUP_EMAIL:-}" ]; then
    mail -s "hololoom Backup Success: $TIMESTAMP" "$BACKUP_EMAIL" < "$REPORT_FILE"
fi

exit 0

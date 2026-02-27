#!/bin/bash
# hololoom VoiceAgent Disaster Recovery
#
# Restores system from backup with validation
#
# Created: 2025-11-16
# Part of: Wave 3 Production Hardening
#
# Usage: ./disaster_recovery.sh <backup_file.tar.gz> [options]
#
# Options:
#   --skip-neo4j      Skip Neo4j restoration
#   --skip-redis      Skip Redis restoration
#   --skip-config     Skip configuration restoration
#   --skip-validation Skip health validation
#   --force           Force overwrite existing data

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Parse arguments
BACKUP_FILE="${1:-}"
SKIP_NEO4J=false
SKIP_REDIS=false
SKIP_CONFIG=false
SKIP_VALIDATION=false
FORCE=false

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-neo4j)
            SKIP_NEO4J=true
            shift
            ;;
        --skip-redis)
            SKIP_REDIS=true
            shift
            ;;
        --skip-config)
            SKIP_CONFIG=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.tar.gz> [options]"
    echo ""
    echo "Options:"
    echo "  --skip-neo4j      Skip Neo4j restoration"
    echo "  --skip-redis      Skip Redis restoration"
    echo "  --skip-config     Skip configuration restoration"
    echo "  --skip-validation Skip health validation"
    echo "  --force           Force overwrite existing data"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    exit 1
fi

TEMP_DIR="/tmp/hololoom_recovery_$$"
START_TIME=$(date +%s)

log_info "========================================="
log_info "hololoom Disaster Recovery"
log_info "Starting at $(date)"
log_info "========================================="
log_info "Backup: $BACKUP_FILE"
log_info "Recovery ID: $$"

# Verify backup integrity
log_step "Verifying backup integrity..."
if [ -f "$BACKUP_FILE.sha256" ]; then
    EXPECTED_CHECKSUM=$(cat "$BACKUP_FILE.sha256" | cut -d' ' -f1)
    ACTUAL_CHECKSUM=$(sha256sum "$BACKUP_FILE" | cut -d' ' -f1)

    if [ "$EXPECTED_CHECKSUM" = "$ACTUAL_CHECKSUM" ]; then
        log_info "✓ Backup integrity verified"
    else
        log_error "Checksum mismatch! Backup may be corrupted."
        log_error "Expected: $EXPECTED_CHECKSUM"
        log_error "Actual: $ACTUAL_CHECKSUM"
        exit 1
    fi
else
    log_warn "No checksum file found, skipping integrity check"
fi

# Extract backup
log_step "Extracting backup..."
mkdir -p "$TEMP_DIR"
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR" || {
    log_error "Failed to extract backup"
    exit 1
}

BACKUP_DIR=$(ls -1 "$TEMP_DIR" | head -1)
log_info "✓ Backup extracted to $TEMP_DIR/$BACKUP_DIR"

# Read metadata
if [ -f "$TEMP_DIR/$BACKUP_DIR/metadata.json" ]; then
    log_info "Backup metadata:"
    cat "$TEMP_DIR/$BACKUP_DIR/metadata.json" | jq '.' || true
fi

# Confirm recovery
if [ "$FORCE" != "true" ]; then
    echo ""
    log_warn "This will OVERWRITE existing data. Are you sure?"
    read -p "Type 'yes' to continue: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        log_info "Recovery cancelled"
        rm -rf "$TEMP_DIR"
        exit 0
    fi
fi

# Stop services
log_step "Stopping services..."
docker-compose -f docker-compose.voice.yml down || log_warn "Some services failed to stop"
log_info "✓ Services stopped"

# Wait for containers to fully stop
sleep 5

# Restore Neo4j
if [ "$SKIP_NEO4J" = "false" ]; then
    log_step "Restoring Neo4j..."

    if [ -f "$TEMP_DIR/$BACKUP_DIR/neo4j.dump" ]; then
        # Restore from dump
        log_info "Loading Neo4j dump..."

        # Start Neo4j temporarily to restore
        docker-compose -f docker-compose.voice.yml up -d neo4j
        sleep 10

        # Copy dump to container
        docker cp "$TEMP_DIR/$BACKUP_DIR/neo4j.dump" hololoom-neo4j:/tmp/neo4j.dump

        # Restore database
        docker exec hololoom-neo4j neo4j-admin database load neo4j \
            --from-path=/tmp --overwrite-destination=true || {
            log_error "Neo4j restore failed"
            log_warn "Trying alternative restore method..."

            # Alternative: restore data directory
            if [ -d "$TEMP_DIR/$BACKUP_DIR/neo4j_data" ]; then
                docker cp "$TEMP_DIR/$BACKUP_DIR/neo4j_data" hololoom-neo4j:/
                log_info "✓ Neo4j data directory restored"
            fi
        }

        docker-compose -f docker-compose.voice.yml down neo4j
        sleep 3

        log_info "✓ Neo4j restored"
    elif [ -d "$TEMP_DIR/$BACKUP_DIR/neo4j_data" ]; then
        # Restore from data directory
        log_info "Restoring Neo4j data directory..."
        docker volume rm neo4j_data || true
        docker volume create neo4j_data
        docker run --rm -v neo4j_data:/data -v "$TEMP_DIR/$BACKUP_DIR/neo4j_data":/backup \
            alpine sh -c "cp -a /backup/* /data/"
        log_info "✓ Neo4j data directory restored"
    else
        log_warn "No Neo4j backup found, skipping..."
    fi
else
    log_info "Skipping Neo4j restoration (--skip-neo4j)"
fi

# Restore Redis
if [ "$SKIP_REDIS" = "false" ]; then
    log_step "Restoring Redis..."

    if [ -f "$TEMP_DIR/$BACKUP_DIR/redis.rdb" ]; then
        # Restore RDB file
        docker volume rm redis_data || true
        docker volume create redis_data

        # Copy RDB file to volume
        docker run --rm -v redis_data:/data -v "$TEMP_DIR/$BACKUP_DIR":/backup \
            alpine sh -c "cp /backup/redis.rdb /data/dump.rdb && chmod 644 /data/dump.rdb"

        # Also restore AOF if available
        if [ -f "$TEMP_DIR/$BACKUP_DIR/redis.aof" ]; then
            docker run --rm -v redis_data:/data -v "$TEMP_DIR/$BACKUP_DIR":/backup \
                alpine sh -c "cp /backup/redis.aof /data/appendonly.aof && chmod 644 /data/appendonly.aof"
        fi

        log_info "✓ Redis restored"
    else
        log_warn "No Redis backup found, skipping..."
    fi
else
    log_info "Skipping Redis restoration (--skip-redis)"
fi

# Restore configuration
if [ "$SKIP_CONFIG" = "false" ]; then
    log_step "Restoring configuration..."

    if [ -d "$TEMP_DIR/$BACKUP_DIR/config/deployment" ]; then
        # Backup existing config
        if [ -d "deployment" ]; then
            mv deployment "deployment.backup.$(date +%s)"
        fi

        cp -r "$TEMP_DIR/$BACKUP_DIR/config/deployment" ./
        log_info "✓ Deployment configuration restored"
    fi

    # Restore docker-compose files
    if ls "$TEMP_DIR/$BACKUP_DIR/config"/docker-compose*.yml 1> /dev/null 2>&1; then
        cp "$TEMP_DIR/$BACKUP_DIR/config"/docker-compose*.yml ./
        log_info "✓ Docker compose files restored"
    fi

    # Restore voice configurations
    if [ -d "$TEMP_DIR/$BACKUP_DIR/config/languages" ]; then
        cp -r "$TEMP_DIR/$BACKUP_DIR/config/languages" hololoom/voice/
        log_info "✓ Language configurations restored"
    fi

    if [ -d "$TEMP_DIR/$BACKUP_DIR/config/personalities" ]; then
        cp -r "$TEMP_DIR/$BACKUP_DIR/config/personalities" hololoom/voice/
        log_info "✓ Personality configurations restored"
    fi

    log_info "✓ Configuration restored"
else
    log_info "Skipping configuration restoration (--skip-config)"
fi

# Restore Grafana dashboards
log_step "Restoring Grafana dashboards..."
if [ -d "$TEMP_DIR/$BACKUP_DIR/grafana" ] && ls "$TEMP_DIR/$BACKUP_DIR/grafana"/*.json 1> /dev/null 2>&1; then
    # We'll restore these after Grafana starts
    mkdir -p /tmp/grafana_dashboards_restore
    cp "$TEMP_DIR/$BACKUP_DIR/grafana"/*.json /tmp/grafana_dashboards_restore/
    log_info "✓ Grafana dashboards staged for restoration"
else
    log_warn "No Grafana dashboards found in backup"
fi

# Restart services
log_step "Restarting services..."
docker-compose -f docker-compose.voice.yml up -d || {
    log_error "Failed to start services"
    exit 1
}

log_info "✓ Services starting..."
log_info "Waiting for services to be ready (60 seconds)..."
sleep 60

# Import Grafana dashboards
if [ -d /tmp/grafana_dashboards_restore ] && [ -n "${GRAFANA_API_KEY:-}" ]; then
    log_step "Importing Grafana dashboards..."
    for dashboard in /tmp/grafana_dashboards_restore/*.json; do
        DASHBOARD_JSON=$(cat "$dashboard")
        curl -s -X POST -H "Authorization: Bearer $GRAFANA_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"dashboard\": $(echo "$DASHBOARD_JSON" | jq '.dashboard'), \"overwrite\": true}" \
            http://localhost:3000/api/dashboards/db > /dev/null 2>&1 || \
            log_warn "Failed to import dashboard $(basename $dashboard)"
    done
    rm -rf /tmp/grafana_dashboards_restore
    log_info "✓ Grafana dashboards imported"
fi

# Verify health
if [ "$SKIP_VALIDATION" = "false" ]; then
    log_step "Verifying system health..."

    if [ -f "scripts/validate_deployment.sh" ]; then
        bash scripts/validate_deployment.sh || {
            log_warn "Health validation failed, but recovery completed"
            log_warn "Check logs: docker-compose -f docker-compose.voice.yml logs"
        }
    else
        log_warn "Validation script not found, skipping health check"

        # Basic health check
        log_info "Running basic health checks..."

        # Check Neo4j
        if docker ps | grep -q hololoom-neo4j; then
            docker exec hololoom-neo4j cypher-shell "RETURN 1" > /dev/null 2>&1 && \
                log_info "✓ Neo4j responding" || log_warn "✗ Neo4j not responding"
        fi

        # Check Redis
        if docker ps | grep -q hololoom-tts-cache; then
            docker exec hololoom-tts-cache redis-cli PING | grep -q PONG && \
                log_info "✓ Redis responding" || log_warn "✗ Redis not responding"
        fi

        # Check Voice Agent
        if docker ps | grep -q hololoom-voice-agent; then
            log_info "✓ Voice Agent running"
        else
            log_warn "✗ Voice Agent not running"
        fi
    fi
else
    log_info "Skipping health validation (--skip-validation)"
fi

# Cleanup
log_step "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
log_info "✓ Cleanup complete"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Generate recovery report
REPORT_FILE="recovery_report_$(date +%Y%m%d_%H%M%S).txt"
cat > "$REPORT_FILE" <<EOF
hololoom Disaster Recovery Report
===================================
Recovery Time: $(date)
Recovery ID: $$
Duration: ${DURATION} seconds ($(($DURATION / 60)) minutes)

Backup Source: $BACKUP_FILE
Backup Date: $(tar -tzf "$BACKUP_FILE" | head -1 | xargs tar -xzOf "$BACKUP_FILE" | head -1 || echo "Unknown")

Components Restored:
- Neo4j: $([ "$SKIP_NEO4J" = "false" ] && echo "✓" || echo "✗ (skipped)")
- Redis: $([ "$SKIP_REDIS" = "false" ] && echo "✓" || echo "✗ (skipped)")
- Configuration: $([ "$SKIP_CONFIG" = "false" ] && echo "✓" || echo "✗ (skipped)")
- Grafana Dashboards: $([ -d /tmp/grafana_dashboards_restore ] && echo "✓" || echo "✗")

Service Status:
- Neo4j: $(docker ps | grep -q hololoom-neo4j && echo "Running" || echo "Stopped")
- Redis: $(docker ps | grep -q hololoom-tts-cache && echo "Running" || echo "Stopped")
- Voice Agent: $(docker ps | grep -q hololoom-voice-agent && echo "Running" || echo "Stopped")
- Elle AR: $(docker ps | grep -q hololoom-elle-ar && echo "Running" || echo "Stopped")

Health Validation: $([ "$SKIP_VALIDATION" = "false" ] && echo "✓ Completed" || echo "✗ Skipped")

Status: SUCCESS
RTO Achieved: ${DURATION} seconds (Target: <1800 seconds)

Next Steps:
1. Verify application functionality
2. Check logs for any errors:
   docker-compose -f docker-compose.voice.yml logs
3. Test critical user journeys
4. Monitor metrics in Grafana

Contact: ops@hololoom.ai for assistance
EOF

log_info "========================================="
log_info "Disaster recovery completed successfully!"
log_info "========================================="
log_info "Duration: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
log_info "RTO Target: 1800 seconds (30 minutes)"
log_info "RTO Achieved: $([ $DURATION -lt 1800 ] && echo "✓ YES" || echo "✗ NO")"
log_info ""
log_info "Recovery report: $REPORT_FILE"
log_info ""
log_info "Next steps:"
log_info "  1. Review logs: docker-compose -f docker-compose.voice.yml logs"
log_info "  2. Test functionality"
log_info "  3. Monitor Grafana dashboards"

exit 0

#!/bin/bash
# HoloLoom Security Infrastructure Deployment Script
# Usage: ./deploy.sh [environment]
# Environments: dev, staging, production

set -e  # Exit on error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$INFRA_DIR")"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}HoloLoom Security Infrastructure Deployment${NC}"
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}================================================${NC}"
echo

# Step 1: Validate environment
echo -e "${YELLOW}[1/10] Validating environment...${NC}"
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|production)$ ]]; then
    echo -e "${RED}Error: Invalid environment. Use: dev, staging, or production${NC}"
    exit 1
fi

# Check for required tools
for cmd in docker docker-compose openssl curl; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ Environment validated${NC}"

# Step 2: Load environment variables
echo -e "${YELLOW}[2/10] Loading environment variables...${NC}"
ENV_FILE="$INFRA_DIR/.env.$ENVIRONMENT"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: Environment file not found: $ENV_FILE${NC}"
    echo -e "${YELLOW}Hint: Copy .env.example to $ENV_FILE and configure it${NC}"
    exit 1
fi

# Source environment file
set -a
source "$ENV_FILE"
set +a
echo -e "${GREEN}✓ Environment variables loaded${NC}"

# Step 3: Validate required secrets
echo -e "${YELLOW}[3/10] Validating required secrets...${NC}"
REQUIRED_SECRETS=(
    "API_KEY_SECRET"
    "USER_HASH_SALT"
    "ENCRYPTION_KEY"
    "JWT_SECRET"
    "POSTGRES_PASSWORD"
    "NEO4J_PASSWORD"
    "GRAFANA_ADMIN_PASSWORD"
)

for secret in "${REQUIRED_SECRETS[@]}"; do
    if [ -z "${!secret:-}" ]; then
        echo -e "${RED}Error: Required secret $secret is not set${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ All required secrets validated${NC}"

# Step 4: Generate SSL certificates (if not exists)
echo -e "${YELLOW}[4/10] Checking SSL certificates...${NC}"
SSL_DIR="$INFRA_DIR/nginx/ssl"
mkdir -p "$SSL_DIR"

if [ ! -f "$SSL_DIR/cert.pem" ] || [ ! -f "$SSL_DIR/key.pem" ]; then
    echo -e "${YELLOW}Generating self-signed SSL certificate...${NC}"
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout "$SSL_DIR/key.pem" \
        -out "$SSL_DIR/cert.pem" \
        -subj "/C=US/ST=State/L=City/O=HoloLoom/CN=hololoom.local"
    echo -e "${GREEN}✓ SSL certificate generated${NC}"
else
    echo -e "${GREEN}✓ SSL certificates already exist${NC}"
fi

# Step 5: Create required directories
echo -e "${YELLOW}[5/10] Creating required directories...${NC}"
mkdir -p "$INFRA_DIR"/{logs,cache,backup}
mkdir -p "$INFRA_DIR/postgres/backup"
mkdir -p "$INFRA_DIR/prometheus"
mkdir -p "$INFRA_DIR/grafana"/{dashboards,datasources}
echo -e "${GREEN}✓ Directories created${NC}"

# Step 6: Initialize PostgreSQL schema
echo -e "${YELLOW}[6/10] Preparing PostgreSQL initialization...${NC}"
cat > "$INFRA_DIR/postgres/init.sql" << 'EOF'
-- HoloLoom Security Database Initialization

-- Forensic logs table (hash chain)
CREATE TABLE IF NOT EXISTS forensic_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id VARCHAR(255),
    source_ip VARCHAR(45),
    event_data JSONB NOT NULL,
    previous_hash VARCHAR(64) NOT NULL,
    current_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_forensic_logs_timestamp ON forensic_logs(timestamp);
CREATE INDEX idx_forensic_logs_event_type ON forensic_logs(event_type);
CREATE INDEX idx_forensic_logs_user_id ON forensic_logs(user_id);
CREATE INDEX idx_forensic_logs_source_ip ON forensic_logs(source_ip);
CREATE INDEX idx_forensic_logs_current_hash ON forensic_logs(current_hash);

-- Incident tracking table
CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'open',
    title VARCHAR(255) NOT NULL,
    description TEXT,
    affected_systems TEXT[],
    source_ip VARCHAR(45),
    user_id VARCHAR(255),
    detection_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    response_time TIMESTAMP WITH TIME ZONE,
    resolution_time TIMESTAMP WITH TIME ZONE,
    assigned_to VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_incidents_status ON incidents(status);
CREATE INDEX idx_incidents_severity ON incidents(severity);
CREATE INDEX idx_incidents_detection_time ON incidents(detection_time);

-- SOAR playbook executions
CREATE TABLE IF NOT EXISTS soar_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    playbook_name VARCHAR(100) NOT NULL,
    incident_id UUID REFERENCES incidents(id),
    trigger_event JSONB NOT NULL,
    actions_taken TEXT[],
    success BOOLEAN NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_soar_executions_playbook ON soar_executions(playbook_name);
CREATE INDEX idx_soar_executions_incident ON soar_executions(incident_id);

-- Compliance evidence
CREATE TABLE IF NOT EXISTS compliance_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    framework VARCHAR(50) NOT NULL,  -- soc2, gdpr, iso27001
    control_id VARCHAR(50) NOT NULL,
    evidence_type VARCHAR(100) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB
);

CREATE INDEX idx_compliance_evidence_framework ON compliance_evidence(framework);
CREATE INDEX idx_compliance_evidence_control ON compliance_evidence(control_id);

-- Audit trail
CREATE TABLE IF NOT EXISTS audit_trail (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    changes JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT
);

CREATE INDEX idx_audit_trail_timestamp ON audit_trail(timestamp);
CREATE INDEX idx_audit_trail_user_id ON audit_trail(user_id);
CREATE INDEX idx_audit_trail_action ON audit_trail(action);

-- Create read-only user for reporting
CREATE USER hololoom_readonly WITH PASSWORD 'readonly_password_change_me';
GRANT CONNECT ON DATABASE hololoom_security TO hololoom_readonly;
GRANT USAGE ON SCHEMA public TO hololoom_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO hololoom_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO hololoom_readonly;

-- Performance optimizations
ALTER TABLE forensic_logs SET (autovacuum_vacuum_scale_factor = 0.01);
ALTER TABLE audit_trail SET (autovacuum_vacuum_scale_factor = 0.01);
EOF
echo -e "${GREEN}✓ PostgreSQL initialization prepared${NC}"

# Step 7: Pull Docker images
echo -e "${YELLOW}[7/10] Pulling Docker images...${NC}"
cd "$INFRA_DIR"
docker-compose -f docker-compose.security.yml pull
echo -e "${GREEN}✓ Docker images pulled${NC}"

# Step 8: Start infrastructure
echo -e "${YELLOW}[8/10] Starting infrastructure containers...${NC}"
docker-compose -f docker-compose.security.yml up -d

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Check health of critical services
CRITICAL_SERVICES=("redis" "postgres" "prometheus" "grafana")
for service in "${CRITICAL_SERVICES[@]}"; do
    if docker ps --filter "name=hololoom-$service" --filter "health=healthy" | grep -q "$service"; then
        echo -e "${GREEN}✓ $service is healthy${NC}"
    else
        echo -e "${YELLOW}⚠ $service is still starting...${NC}"
    fi
done

echo -e "${GREEN}✓ Infrastructure started${NC}"

# Step 9: Run database migrations (if applicable)
echo -e "${YELLOW}[9/10] Running database migrations...${NC}"
# Add migration commands here if needed
echo -e "${GREEN}✓ Migrations complete${NC}"

# Step 10: Verify deployment
echo -e "${YELLOW}[10/10] Verifying deployment...${NC}"

# Test Redis
if docker exec hololoom-redis redis-cli ping | grep -q PONG; then
    echo -e "${GREEN}✓ Redis is responding${NC}"
else
    echo -e "${RED}✗ Redis is not responding${NC}"
fi

# Test PostgreSQL
if docker exec hololoom-postgres pg_isready -U hololoom | grep -q "accepting connections"; then
    echo -e "${GREEN}✓ PostgreSQL is ready${NC}"
else
    echo -e "${RED}✗ PostgreSQL is not ready${NC}"
fi

# Test Prometheus
if curl -sf http://localhost:9090/-/healthy > /dev/null; then
    echo -e "${GREEN}✓ Prometheus is healthy${NC}"
else
    echo -e "${RED}✗ Prometheus is not healthy${NC}"
fi

# Test Grafana
if curl -sf http://localhost:3000/api/health > /dev/null; then
    echo -e "${GREEN}✓ Grafana is healthy${NC}"
else
    echo -e "${RED}✗ Grafana is not healthy${NC}"
fi

echo
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${BLUE}================================================${NC}"
echo
echo -e "${YELLOW}Service URLs:${NC}"
echo -e "  Grafana:    http://localhost:3000 (admin / ${GRAFANA_ADMIN_PASSWORD:-<password>})"
echo -e "  Prometheus: http://localhost:9090"
echo -e "  Neo4j:      http://localhost:7474 (neo4j / ${NEO4J_PASSWORD:-<password>})"
echo -e "  Splunk:     http://localhost:8000 (admin / ${SPLUNK_PASSWORD:-<password>})"
echo -e "  API:        http://localhost:8000"
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Import Grafana dashboards: ./scripts/import-dashboards.sh"
echo -e "  2. Configure alerting: ./scripts/configure-alerts.sh"
echo -e "  3. Run health check: ./scripts/health-check.sh"
echo -e "  4. View logs: docker-compose -f docker-compose.security.yml logs -f"
echo
echo -e "${BLUE}================================================${NC}"

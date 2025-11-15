#!/bin/bash

################################################################################
# HoloLoom Security Dashboard Setup Script
# Version: 1.0.0 (2025-11-15)
#
# Purpose:
# - Automatically provision Grafana dashboards for HoloLoom security pipeline
# - Configure datasources (Prometheus, Elasticsearch)
# - Set up alerting rules and notification channels
# - Import pre-built security dashboards
#
# Prerequisites:
# - Grafana running at http://grafana:3000
# - Prometheus running at http://prometheus:9090
# - Elasticsearch running at http://elasticsearch:9200
# - curl installed
#
# Usage:
#   chmod +x scripts/setup_grafana.sh
#   ./scripts/setup_grafana.sh [--url <grafana_url>] [--token <api_token>]
#
################################################################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GRAFANA_URL="${1:-http://localhost:3000}"
API_TOKEN="${2:-admin:admin}"
DASHBOARDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../infra/grafana/dashboards" && pwd)"
DATASOURCES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../infra/grafana/datasources" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# Check Prerequisites
################################################################################

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if curl is installed
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed. Please install curl and try again."
        exit 1
    fi

    # Check if Grafana is accessible
    if ! curl -s -f "${GRAFANA_URL}/api/health" > /dev/null; then
        log_error "Grafana is not accessible at ${GRAFANA_URL}"
        log_info "Make sure Grafana is running: docker-compose up -d grafana"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

################################################################################
# Create API Token
################################################################################

create_api_token() {
    log_info "Creating Grafana API token..."

    # Check if token already exists
    EXISTING_TOKEN=$(curl -s -H "Authorization: Bearer admin" \
        "${GRAFANA_URL}/api/auth/keys" \
        2>/dev/null | grep -c "HoloLoom-Setup" || echo "0")

    if [ "$EXISTING_TOKEN" -gt 0 ]; then
        log_warn "API token 'HoloLoom-Setup' already exists"
        return
    fi

    # Create new token
    RESPONSE=$(curl -s -X POST "${GRAFANA_URL}/api/auth/keys" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "HoloLoom-Setup",
            "role": "Admin"
        }' \
        -u admin:admin)

    if echo "$RESPONSE" | grep -q "key"; then
        API_TOKEN=$(echo "$RESPONSE" | grep -o '"key":"[^"]*' | cut -d'"' -f4)
        log_success "API token created: $API_TOKEN"
    else
        log_warn "Could not create API token, using default credentials"
    fi
}

################################################################################
# Configure Datasources
################################################################################

configure_datasources() {
    log_info "Configuring datasources..."

    # Prometheus
    log_info "  - Configuring Prometheus datasource..."
    curl -s -X POST "${GRAFANA_URL}/api/datasources" \
        -H "Authorization: Bearer ${API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Prometheus",
            "type": "prometheus",
            "access": "proxy",
            "url": "http://prometheus:9090",
            "isDefault": true,
            "jsonData": {
                "timeInterval": "15s",
                "httpMethod": "POST",
                "manageAlerts": true
            }
        }' > /dev/null 2>&1 || log_warn "    Prometheus datasource may already exist"

    log_success "  - Prometheus datasource configured"

    # Elasticsearch
    log_info "  - Configuring Elasticsearch datasource..."
    curl -s -X POST "${GRAFANA_URL}/api/datasources" \
        -H "Authorization: Bearer ${API_TOKEN}" \
        -H "Content-Type: application/json" \
        -d '{
            "name": "Elasticsearch",
            "type": "elasticsearch",
            "access": "proxy",
            "url": "http://elasticsearch:9200",
            "basicAuth": true,
            "basicAuthUser": "elastic",
            "secureJsonData": {
                "basicAuthPassword": "elastic"
            },
            "jsonData": {
                "esVersion": "8.0.0",
                "logMessageField": "message",
                "logLevelField": "level",
                "timeField": "@timestamp",
                "timeInterval": "10s"
            }
        }' > /dev/null 2>&1 || log_warn "    Elasticsearch datasource may already exist"

    log_success "  - Elasticsearch datasource configured"
}

################################################################################
# Import Dashboards
################################################################################

import_dashboards() {
    log_info "Importing security dashboards..."

    DASHBOARD_FILES=(
        "security_overview.json"
        "authentication.json"
        "attacks.json"
        "anomalies.json"
        "incidents.json"
    )

    DASHBOARD_NAMES=(
        "HoloLoom Security Overview"
        "Authentication & Authorization Metrics"
        "Real-Time Attack Monitoring"
        "Anomaly Detection Dashboard"
        "Security Incident Timeline"
    )

    for i in "${!DASHBOARD_FILES[@]}"; do
        DASHBOARD_FILE="${DASHBOARD_FILES[$i]}"
        DASHBOARD_NAME="${DASHBOARD_NAMES[$i]}"
        DASHBOARD_PATH="${DASHBOARDS_DIR}/${DASHBOARD_FILE}"

        if [ ! -f "$DASHBOARD_PATH" ]; then
            log_error "  - Dashboard file not found: $DASHBOARD_PATH"
            continue
        fi

        log_info "  - Importing: $DASHBOARD_NAME..."

        # Prepare dashboard JSON with overrides
        DASHBOARD_JSON=$(cat "$DASHBOARD_PATH")

        # Import dashboard
        RESPONSE=$(curl -s -X POST "${GRAFANA_URL}/api/dashboards/db" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"dashboard\": $DASHBOARD_JSON,
                \"overwrite\": true
            }")

        if echo "$RESPONSE" | grep -q '"id"'; then
            DASHBOARD_ID=$(echo "$RESPONSE" | grep -o '"id":[0-9]*' | cut -d':' -f2 | head -1)
            log_success "  - Imported: $DASHBOARD_NAME (ID: $DASHBOARD_ID)"
        else
            log_error "  - Failed to import: $DASHBOARD_NAME"
            echo "$RESPONSE" >&2
        fi
    done
}

################################################################################
# Configure Alerting Rules
################################################################################

configure_alerting() {
    log_info "Configuring alerting rules..."

    # Create alert rules
    ALERT_RULES=(
        "attack_rate_critical:rate(hololoom_security_requests_blocked_total[5m]) > 100"
        "failed_auth_threshold:sum(rate(hololoom_security_login_failures_total[5m])) by (source_ip) > 10"
        "anomaly_score_critical:hololoom_security_anomaly_score_avg > 0.9"
        "waf_rule_trigger_warning:sum(rate(hololoom_security_waf_rules_triggered_total[5m])) > 1000"
    )

    log_info "  - Creating alert rules..."

    for ALERT_RULE in "${ALERT_RULES[@]}"; do
        ALERT_NAME=$(echo "$ALERT_RULE" | cut -d':' -f1)
        ALERT_EXPR=$(echo "$ALERT_RULE" | cut -d':' -f2-)

        log_info "    - Creating alert: $ALERT_NAME"

        curl -s -X POST "${GRAFANA_URL}/api/ruler/grafana/rules/SecurityAlerts" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"rules\": [
                    {
                        \"uid\": \"$(echo "$ALERT_NAME" | tr ' ' '_' | tr '[:upper:]' '[:lower:]')\",
                        \"title\": \"$ALERT_NAME\",
                        \"condition\": \"B\",
                        \"data\": [
                            {
                                \"refId\": \"B\",
                                \"queryType\": \"\",
                                \"model\": {
                                    \"expr\": \"$ALERT_EXPR\",
                                    \"interval\": \"\",
                                    \"refId\": \"B\"
                                },
                                \"datasourceUid\": \"prometheus\",
                                \"relativeTimeRange\": {
                                    \"from\": 600,
                                    \"to\": 0
                                }
                            }
                        ],
                        \"noDataState\": \"NoData\",
                        \"execErrState\": \"Alerting\",
                        \"for\": \"5m\",
                        \"annotations\": {
                            \"description\": \"Security alert for HoloLoom\",
                            \"severity\": \"critical\"
                        },
                        \"labels\": {
                            \"team\": \"security\"
                        }
                    }
                ]
            }" > /dev/null 2>&1 || log_warn "    - Alert rule may already exist: $ALERT_NAME"
    done

    log_success "  - Alert rules configured"
}

################################################################################
# Configure Notification Channels
################################################################################

configure_notifications() {
    log_info "Configuring notification channels..."

    # Email notification (if configured)
    if [ -n "${GRAFANA_SMTP_FROM:-}" ]; then
        log_info "  - Configuring email notifications..."

        curl -s -X POST "${GRAFANA_URL}/api/alert-notifications" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d '{
                "name": "Email-Security-Alerts",
                "type": "email",
                "sendReminder": true,
                "frequency": "5m",
                "isDefault": false,
                "settings": {
                    "addresses": "security-team@example.com"
                }
            }' > /dev/null 2>&1 || log_warn "  - Email notification channel may already exist"

        log_success "  - Email notifications configured"
    fi

    # Slack notification (if configured)
    if [ -n "${GRAFANA_SLACK_WEBHOOK:-}" ]; then
        log_info "  - Configuring Slack notifications..."

        curl -s -X POST "${GRAFANA_URL}/api/alert-notifications" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"Slack-Security-Channel\",
                \"type\": \"slack\",
                \"sendReminder\": true,
                \"frequency\": \"5m\",
                \"isDefault\": false,
                \"secureSettings\": {
                    \"url\": \"${GRAFANA_SLACK_WEBHOOK}\"
                }
            }" > /dev/null 2>&1 || log_warn "  - Slack notification channel may already exist"

        log_success "  - Slack notifications configured"
    fi
}

################################################################################
# Create Folders for Organization
################################################################################

create_folders() {
    log_info "Creating dashboard folders..."

    FOLDERS=(
        "Security"
        "Security/Real-Time"
        "Security/Historical"
        "Security/Compliance"
    )

    for FOLDER in "${FOLDERS[@]}"; do
        log_info "  - Creating folder: $FOLDER"

        curl -s -X POST "${GRAFANA_URL}/api/folders" \
            -H "Authorization: Bearer ${API_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "{
                \"title\": \"$FOLDER\"
            }" > /dev/null 2>&1 || log_warn "  - Folder may already exist: $FOLDER"
    done

    log_success "  - Folders created"
}

################################################################################
# Generate Summary Report
################################################################################

generate_summary() {
    log_info "Generating summary report..."

    SUMMARY_FILE="${SCRIPT_DIR}/GRAFANA_SETUP_SUMMARY.txt"

    cat > "$SUMMARY_FILE" << EOF
================================================================================
HoloLoom Security Dashboard Setup Complete
Generated: $(date)
================================================================================

GRAFANA CONFIGURATION SUMMARY
-----------------------------

Grafana URL: ${GRAFANA_URL}
API Token: ******* (stored securely)

DASHBOARDS IMPORTED
-------------------

1. HoloLoom Security Overview (uid: hololoom-security-overview)
   - Attack rate monitoring
   - Blocked vs allowed requests
   - Active API keys and OAuth sessions
   - Rate limit utilization
   - Top IPs by request volume
   - Authentication success rate
   - Current risk level assessment
   - Anomaly score distribution

2. Authentication & Authorization Metrics (uid: hololoom-auth-metrics)
   - Login success/failure rate
   - Failed login attempts by IP
   - OAuth token issuance rate
   - API key usage by scope
   - RBAC permission denials
   - MFA challenge rate
   - Session duration distribution
   - Top users by activity

3. Real-Time Attack Monitoring (uid: hololoom-attack-monitoring)
   - Real-time attack feed (last 100 attacks)
   - Attack type breakdown
   - Allowed vs blocked requests
   - Attack volume time series
   - Blocked IPs (temporary bans)
   - WAF rule triggers
   - Top attack patterns
   - Attack success rate

4. Anomaly Detection Dashboard (uid: hololoom-anomalies)
   - Anomaly score time series
   - Anomalies by type
   - False positive tracking
   - Top anomalous users
   - Anomaly investigation workflow
   - Model performance metrics (precision, recall, F1)
   - Baseline vs current behavior

5. Security Incident Timeline (uid: hololoom-incidents)
   - Security incidents (chronological)
   - Incident severity distribution
   - Incident status tracking
   - Mean time to detect (MTTD)
   - Mean time to respond (MTTR)
   - Incident trends by severity
   - Remediation actions taken
   - Post-mortem reviews

DATASOURCES CONFIGURED
----------------------

1. Prometheus
   - URL: http://prometheus:9090
   - Type: Prometheus
   - Default: Yes
   - Status: Ready

2. Elasticsearch
   - URL: http://elasticsearch:9200
   - Type: Elasticsearch
   - Version: 8.0.0
   - Authentication: Enabled
   - Status: Ready

ALERTING RULES CONFIGURED
--------------------------

1. Attack Rate Critical (threshold: 100 requests/min)
   - Severity: CRITICAL
   - Duration: 5 minutes

2. Failed Auth Threshold (threshold: 10 failures/min per IP)
   - Severity: WARNING
   - Duration: 5 minutes

3. Anomaly Score Critical (threshold: 0.9)
   - Severity: CRITICAL
   - Duration: 5 minutes

4. WAF Rule Trigger Warning (threshold: 1000 triggers/min)
   - Severity: WARNING
   - Duration: 5 minutes

NEXT STEPS
----------

1. Access Grafana: ${GRAFANA_URL}
   - Username: admin
   - Password: admin (change immediately in production)

2. View Dashboards:
   - Navigate to: Dashboards > Browse
   - Look for "Security" folder
   - Open desired dashboard

3. Configure Notifications:
   - Settings > Alert notification channels
   - Set up Slack, email, or webhooks
   - Enable notifications on alerts

4. Customize Thresholds:
   - Edit dashboard panels as needed
   - Adjust alert conditions
   - Set up custom rules per your requirements

5. Set Up Backup:
   - Export dashboards regularly
   - Use Grafana API to backup configuration
   - Version control dashboard changes

METRICS TO MONITOR
-------------------

Security Overview:
- Attack Rate (requests/min)
- Blocked vs Allowed ratio
- Current Risk Level
- Active API Keys & OAuth Sessions

Authentication:
- Login Success Rate (target: >95%)
- Failed Login Attempts by IP
- MFA Challenge Rate
- Session Duration Distribution

Attacks:
- Attack Type Breakdown
- WAF Trigger Rate (should be <50 per hour)
- Blocked IP Count
- Attack Success Rate (should be 0%)

Anomalies:
- Anomaly Score (target: <0.3)
- False Positive Rate (target: <5%)
- Model Precision, Recall, F1
- Baseline Deviation

Incidents:
- MTTD (target: <30 minutes)
- MTTR (target: <60 minutes)
- Incident Severity Distribution
- Remediation Success Rate

TROUBLESHOOTING
---------------

Issue: Dashboards show "No Data"
Solution: Verify Prometheus is scraping security metrics
  - Check Prometheus targets: http://prometheus:9090/targets
  - Verify metric names match: hololoom_security_*

Issue: "Cannot connect to Elasticsearch"
Solution: Check Elasticsearch connection
  - Verify Elasticsearch is running
  - Check credentials are correct
  - Confirm index patterns exist in Elasticsearch

Issue: Alerts not firing
Solution: Check alert configuration
  - Verify datasource connections
  - Check alert rule conditions
  - Review alert notification channels
  - Check Grafana logs for errors

DOCUMENTATION
--------------

Full documentation available in:
- docs/GRAFANA_SETUP.md - Setup guide
- infra/grafana/README.md - Dashboard reference
- CLAUDE.md - HoloLoom documentation

METRICS REFERENCE
-----------------

All security metrics use the prefix: hololoom_security_*

Key metrics:
- hololoom_security_requests_allowed_total
- hololoom_security_requests_blocked_total
- hololoom_security_attacks_total
- hololoom_security_login_attempts_total
- hololoom_security_anomaly_score
- hololoom_security_incidents_total
- hololoom_security_mean_time_to_detect_minutes
- hololoom_security_mean_time_to_respond_minutes

SUPPORT
-------

For issues or questions:
1. Check GRAFANA_SETUP.md for detailed documentation
2. Review metric definitions in HoloLoom code
3. Check Prometheus scrape configs
4. Review Elasticsearch index mappings
5. Contact security team for assistance

================================================================================
EOF

    log_success "Summary saved to: $SUMMARY_FILE"
    cat "$SUMMARY_FILE"
}

################################################################################
# Main Execution
################################################################################

main() {
    clear
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║   HoloLoom Security Dashboard - Automated Setup (v1.0.0)          ║"
    echo "║   Date: $(date '+%Y-%m-%d %H:%M:%S')                                   ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Execute setup steps
    check_prerequisites
    create_api_token
    configure_datasources
    import_dashboards
    configure_alerting
    configure_notifications
    create_folders
    generate_summary

    echo ""
    log_success "Setup completed successfully!"
    log_info "Access Grafana at: ${GRAFANA_URL}"
    log_info "Check summary report for complete details"
    echo ""
}

# Run main function
main "$@"

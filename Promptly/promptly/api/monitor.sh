#!/bin/bash

# =============================================================================
# Promptly API - Health Monitoring Script
# =============================================================================
#
# This script monitors the health and performance of all services.
# It can run in continuous mode or as a one-time check.
#
# Usage:
#   ./monitor.sh [options]
#
# Options:
#   --watch           Continuous monitoring (updates every 5 seconds)
#   --interval SEC    Update interval for watch mode (default: 5)
#   --alerts          Enable alerts for unhealthy services
#   --detailed        Show detailed metrics
#   --json            Output in JSON format
#   --help            Show this help message
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.production"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.production.yml"

# Options
WATCH_MODE=false
INTERVAL=5
ENABLE_ALERTS=false
DETAILED=false
JSON_OUTPUT=false

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    if [[ "${JSON_OUTPUT}" == "false" ]]; then
        echo -e "\n${BOLD}${BLUE}========================================${NC}"
        echo -e "${BOLD}${BLUE}$1${NC}"
        echo -e "${BOLD}${BLUE}========================================${NC}\n"
    fi
}

print_success() {
    if [[ "${JSON_OUTPUT}" == "false" ]]; then
        echo -e "${GREEN}✓ $1${NC}"
    fi
}

print_error() {
    if [[ "${JSON_OUTPUT}" == "false" ]]; then
        echo -e "${RED}✗ $1${NC}"
    fi
}

print_warning() {
    if [[ "${JSON_OUTPUT}" == "false" ]]; then
        echo -e "${YELLOW}⚠ $1${NC}"
    fi
}

print_info() {
    if [[ "${JSON_OUTPUT}" == "false" ]]; then
        echo -e "${CYAN}$1${NC}"
    fi
}

# =============================================================================
# Monitoring Functions
# =============================================================================

check_service_health() {
    local service=$1
    local status=$(docker-compose -f "${COMPOSE_FILE}" ps "${service}" 2>/dev/null | tail -n +2 | awk '{print $4}')

    if [[ "${status}" == "running" ]] || [[ "${status}" == "Up" ]]; then
        # Check if health check is defined
        local health=$(docker inspect --format='{{.State.Health.Status}}' "promptly-${service}" 2>/dev/null || echo "unknown")

        case "${health}" in
            healthy)
                echo "healthy"
                ;;
            unhealthy)
                echo "unhealthy"
                ;;
            starting)
                echo "starting"
                ;;
            *)
                echo "running"
                ;;
        esac
    else
        echo "stopped"
    fi
}

get_container_stats() {
    local service=$1
    local container="promptly-${service}"

    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        # Get CPU and memory stats
        local stats=$(docker stats --no-stream --format "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}" "${container}" 2>/dev/null || echo "N/A|N/A|N/A")

        echo "${stats}"
    else
        echo "N/A|N/A|N/A"
    fi
}

check_api_endpoint() {
    local endpoint=$1
    local expected_status=${2:-200}

    local response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost${endpoint}" 2>/dev/null || echo "000")

    if [[ "${response}" == "${expected_status}" ]]; then
        echo "ok"
    else
        echo "error:${response}"
    fi
}

get_database_connections() {
    local connections=$(docker-compose -f "${COMPOSE_FILE}" exec -T postgres \
        psql -U "${POSTGRES_USER:-promptly}" -d "${POSTGRES_DB:-promptly}" \
        -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | xargs || echo "N/A")

    echo "${connections}"
}

get_redis_info() {
    local info_key=$1
    local value=$(docker-compose -f "${COMPOSE_FILE}" exec -T redis \
        redis-cli --pass "${REDIS_PASSWORD}" INFO 2>/dev/null | grep "^${info_key}:" | cut -d: -f2 | tr -d '\r' || echo "N/A")

    echo "${value}"
}

get_disk_usage() {
    local volume=$1
    local usage=$(docker system df -v 2>/dev/null | grep "${volume}" | awk '{print $4}' || echo "N/A")
    echo "${usage}"
}

# =============================================================================
# Display Functions
# =============================================================================

display_service_status() {
    print_header "Service Status"

    local services=("postgres" "redis" "api" "nginx")

    for service in "${services[@]}"; do
        local health=$(check_service_health "${service}")
        local stats=$(get_container_stats "${service}")

        IFS='|' read -r cpu memory mem_percent <<< "${stats}"

        case "${health}" in
            healthy)
                printf "${GREEN}%-15s${NC} %-10s  CPU: %-8s  Memory: %-15s (%-6s)\n" \
                    "● ${service}" "healthy" "${cpu}" "${memory}" "${mem_percent}"
                ;;
            running)
                printf "${GREEN}%-15s${NC} %-10s  CPU: %-8s  Memory: %-15s (%-6s)\n" \
                    "● ${service}" "running" "${cpu}" "${memory}" "${mem_percent}"
                ;;
            starting)
                printf "${YELLOW}%-15s${NC} %-10s  CPU: %-8s  Memory: %-15s (%-6s)\n" \
                    "◐ ${service}" "starting" "${cpu}" "${memory}" "${mem_percent}"
                ;;
            unhealthy)
                printf "${RED}%-15s${NC} %-10s  CPU: %-8s  Memory: %-15s (%-6s)\n" \
                    "✗ ${service}" "unhealthy" "${cpu}" "${memory}" "${mem_percent}"
                ;;
            stopped)
                printf "${RED}%-15s${NC} %-10s\n" "○ ${service}" "stopped"
                ;;
        esac
    done
}

display_api_endpoints() {
    print_header "API Endpoints"

    local endpoints=(
        "/health:Health Check"
        "/api/v1/prompts:Prompts API"
        "/api/v1/branches:Branches API"
        "/docs:API Documentation"
    )

    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r endpoint name <<< "${endpoint_info}"
        local status=$(check_api_endpoint "${endpoint}")

        if [[ "${status}" == "ok" ]]; then
            printf "${GREEN}✓${NC} %-30s %s\n" "${name}" "${endpoint}"
        else
            printf "${RED}✗${NC} %-30s %s (${status})\n" "${name}" "${endpoint}"
        fi
    done
}

display_database_metrics() {
    print_header "Database Metrics"

    # Source environment
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        source "${ENV_FILE}"
        set +a
    fi

    # PostgreSQL
    local pg_connections=$(get_database_connections)
    local pg_size=$(docker-compose -f "${COMPOSE_FILE}" exec -T postgres \
        psql -U "${POSTGRES_USER:-promptly}" -d "${POSTGRES_DB:-promptly}" \
        -t -c "SELECT pg_size_pretty(pg_database_size('${POSTGRES_DB:-promptly}'));" 2>/dev/null | xargs || echo "N/A")

    print_info "PostgreSQL:"
    echo "  Connections:      ${pg_connections}"
    echo "  Database Size:    ${pg_size}"

    # Redis
    local redis_clients=$(get_redis_info "connected_clients")
    local redis_memory=$(get_redis_info "used_memory_human")
    local redis_keys=$(docker-compose -f "${COMPOSE_FILE}" exec -T redis \
        redis-cli --pass "${REDIS_PASSWORD}" DBSIZE 2>/dev/null | tr -d '\r' || echo "N/A")

    echo ""
    print_info "Redis:"
    echo "  Connected Clients: ${redis_clients}"
    echo "  Memory Used:       ${redis_memory}"
    echo "  Total Keys:        ${redis_keys}"
}

display_system_metrics() {
    print_header "System Metrics"

    # Docker disk usage
    print_info "Docker Storage:"
    docker system df 2>/dev/null || echo "N/A"

    echo ""

    # Volume usage
    print_info "Volume Usage:"
    docker volume ls -q | grep promptly | while read -r vol; do
        local size=$(docker system df -v 2>/dev/null | grep "${vol}" | awk '{print $3}' || echo "N/A")
        printf "  %-30s %s\n" "${vol}" "${size}"
    done

    echo ""

    # Network info
    print_info "Network:"
    docker network inspect promptly-network --format '{{range .Containers}}  {{.Name}}: {{.IPv4Address}}{{"\n"}}{{end}}' 2>/dev/null || echo "N/A"
}

display_logs_summary() {
    print_header "Recent Errors (Last 50 lines)"

    docker-compose -f "${COMPOSE_FILE}" logs --tail=50 2>/dev/null | grep -i "error\|warn\|critical" || echo "No recent errors found"
}

# =============================================================================
# JSON Output
# =============================================================================

output_json() {
    # Source environment
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        source "${ENV_FILE}"
        set +a
    fi

    local timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    cat << EOF
{
  "timestamp": "${timestamp}",
  "services": {
    "postgres": {
      "status": "$(check_service_health postgres)",
      "stats": "$(get_container_stats postgres)",
      "connections": $(get_database_connections)
    },
    "redis": {
      "status": "$(check_service_health redis)",
      "stats": "$(get_container_stats redis)",
      "clients": $(get_redis_info "connected_clients"),
      "keys": "$(docker-compose -f "${COMPOSE_FILE}" exec -T redis redis-cli --pass "${REDIS_PASSWORD}" DBSIZE 2>/dev/null | tr -d '\r')"
    },
    "api": {
      "status": "$(check_service_health api)",
      "stats": "$(get_container_stats api)",
      "health_endpoint": "$(check_api_endpoint /health)"
    },
    "nginx": {
      "status": "$(check_service_health nginx)",
      "stats": "$(get_container_stats nginx)"
    }
  }
}
EOF
}

# =============================================================================
# Alert Functions
# =============================================================================

send_alert() {
    local service=$1
    local message=$2

    # Log alert
    echo "[$(date)] ALERT: ${service} - ${message}" >> "${SCRIPT_DIR}/logs/alerts.log"

    # Print to console
    print_error "ALERT: ${service} - ${message}"

    # TODO: Add email/webhook notifications here
    # curl -X POST https://hooks.slack.com/... -d "{'text': '${message}'}"
}

check_alerts() {
    local services=("postgres" "redis" "api" "nginx")

    for service in "${services[@]}"; do
        local health=$(check_service_health "${service}")

        if [[ "${health}" == "unhealthy" ]]; then
            send_alert "${service}" "Service is unhealthy"
        elif [[ "${health}" == "stopped" ]]; then
            send_alert "${service}" "Service is stopped"
        fi
    done

    # Check API endpoint
    if [[ "$(check_api_endpoint /health)" != "ok" ]]; then
        send_alert "api" "Health endpoint not responding"
    fi
}

# =============================================================================
# Main Monitoring Flow
# =============================================================================

show_help() {
    cat << EOF
Promptly API - Health Monitoring Script

Usage: ./monitor.sh [options]

Options:
    --watch             Continuous monitoring (updates every 5 seconds)
    --interval SEC      Update interval for watch mode (default: 5)
    --alerts            Enable alerts for unhealthy services
    --detailed          Show detailed metrics
    --json              Output in JSON format
    --help              Show this help message

Examples:
    ./monitor.sh                    # One-time status check
    ./monitor.sh --watch            # Continuous monitoring
    ./monitor.sh --detailed         # Show all metrics
    ./monitor.sh --json             # JSON output for monitoring tools

EOF
    exit 0
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --watch)
                WATCH_MODE=true
                shift
                ;;
            --interval)
                INTERVAL=$2
                shift 2
                ;;
            --alerts)
                ENABLE_ALERTS=true
                shift
                ;;
            --detailed)
                DETAILED=true
                shift
                ;;
            --json)
                JSON_OUTPUT=true
                shift
                ;;
            --help)
                show_help
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                ;;
        esac
    done

    # JSON output mode
    if [[ "${JSON_OUTPUT}" == "true" ]]; then
        output_json
        exit 0
    fi

    # Watch mode
    if [[ "${WATCH_MODE}" == "true" ]]; then
        while true; do
            clear
            print_header "Promptly API - Health Monitor (${INTERVAL}s refresh)"
            echo "Press Ctrl+C to exit"
            echo ""

            display_service_status
            display_api_endpoints

            if [[ "${DETAILED}" == "true" ]]; then
                display_database_metrics
                display_system_metrics
            fi

            if [[ "${ENABLE_ALERTS}" == "true" ]]; then
                check_alerts
            fi

            sleep "${INTERVAL}"
        done
    else
        # One-time check
        print_header "Promptly API - Health Status"

        display_service_status
        display_api_endpoints

        if [[ "${DETAILED}" == "true" ]]; then
            display_database_metrics
            display_system_metrics
            display_logs_summary
        fi

        if [[ "${ENABLE_ALERTS}" == "true" ]]; then
            check_alerts
        fi
    fi
}

# Run main function
main "$@"

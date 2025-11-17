#!/bin/bash

# =============================================================================
# Promptly API - Log Viewer Script
# =============================================================================
#
# This script provides an easy way to view logs from all services.
# Supports filtering, following, and exporting logs.
#
# Usage:
#   ./logs.sh [service] [options]
#
# Services:
#   all          All services (default)
#   api          Promptly API
#   postgres     PostgreSQL database
#   redis        Redis cache
#   nginx        Nginx reverse proxy
#
# Options:
#   -f, --follow        Follow log output (like tail -f)
#   -n, --lines NUM     Number of lines to show (default: 100)
#   --since TIME        Show logs since timestamp (e.g., 2023-01-01, 1h, 30m)
#   --until TIME        Show logs until timestamp
#   --grep PATTERN      Filter logs by pattern
#   --errors            Show only errors and warnings
#   --export FILE       Export logs to file
#   --help              Show this help message
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.production.yml"

# Default options
SERVICE="all"
FOLLOW=false
LINES=100
SINCE=""
UNTIL=""
GREP_PATTERN=""
ERRORS_ONLY=false
EXPORT_FILE=""

# =============================================================================
# Helper Functions
# =============================================================================

print_error() {
    echo -e "${RED}✗ $1${NC}" >&2
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# =============================================================================
# Log Colorization
# =============================================================================

colorize_logs() {
    while IFS= read -r line; do
        # Color by log level
        if echo "${line}" | grep -qi "error\|exception\|critical\|fatal"; then
            echo -e "${RED}${line}${NC}"
        elif echo "${line}" | grep -qi "warn\|warning"; then
            echo -e "${YELLOW}${line}${NC}"
        elif echo "${line}" | grep -qi "info"; then
            echo -e "${CYAN}${line}${NC}"
        elif echo "${line}" | grep -qi "debug"; then
            echo -e "${MAGENTA}${line}${NC}"
        elif echo "${line}" | grep -qi "success\|ok\|healthy"; then
            echo -e "${GREEN}${line}${NC}"
        else
            echo "${line}"
        fi
    done
}

# =============================================================================
# Log Functions
# =============================================================================

view_logs() {
    local service=$1

    # Build docker-compose logs command
    local cmd="docker-compose -f ${COMPOSE_FILE} logs"

    # Add options
    if [[ "${FOLLOW}" == "true" ]]; then
        cmd="${cmd} -f"
    fi

    if [[ -n "${LINES}" ]]; then
        cmd="${cmd} --tail=${LINES}"
    fi

    if [[ -n "${SINCE}" ]]; then
        cmd="${cmd} --since=${SINCE}"
    fi

    if [[ -n "${UNTIL}" ]]; then
        cmd="${cmd} --until=${UNTIL}"
    fi

    # Add service name (if not all)
    if [[ "${service}" != "all" ]]; then
        cmd="${cmd} ${service}"
    fi

    # Execute command
    if [[ -n "${GREP_PATTERN}" ]]; then
        eval "${cmd}" 2>&1 | grep -i "${GREP_PATTERN}" | colorize_logs
    elif [[ "${ERRORS_ONLY}" == "true" ]]; then
        eval "${cmd}" 2>&1 | grep -iE "error|warn|exception|critical|fatal" | colorize_logs
    else
        eval "${cmd}" 2>&1 | colorize_logs
    fi
}

export_logs() {
    local service=$1
    local output_file=$2

    print_info "Exporting logs to: ${output_file}"

    # Build docker-compose logs command (without color)
    local cmd="docker-compose -f ${COMPOSE_FILE} logs --no-color"

    if [[ -n "${LINES}" ]]; then
        cmd="${cmd} --tail=${LINES}"
    fi

    if [[ -n "${SINCE}" ]]; then
        cmd="${cmd} --since=${SINCE}"
    fi

    if [[ -n "${UNTIL}" ]]; then
        cmd="${cmd} --until=${UNTIL}"
    fi

    # Add service name (if not all)
    if [[ "${service}" != "all" ]]; then
        cmd="${cmd} ${service}"
    fi

    # Export to file
    if [[ -n "${GREP_PATTERN}" ]]; then
        eval "${cmd}" 2>&1 | grep -i "${GREP_PATTERN}" > "${output_file}"
    elif [[ "${ERRORS_ONLY}" == "true" ]]; then
        eval "${cmd}" 2>&1 | grep -iE "error|warn|exception|critical|fatal" > "${output_file}"
    else
        eval "${cmd}" 2>&1 > "${output_file}"
    fi

    # Gzip if large
    local size=$(stat -f%z "${output_file}" 2>/dev/null || stat -c%s "${output_file}" 2>/dev/null || echo 0)
    if [[ $size -gt 1048576 ]]; then  # > 1MB
        gzip "${output_file}"
        print_success "Logs exported and compressed: ${output_file}.gz"
    else
        print_success "Logs exported: ${output_file}"
    fi
}

analyze_logs() {
    local service=$1

    print_info "Analyzing logs for: ${service}"
    echo ""

    # Build command
    local cmd="docker-compose -f ${COMPOSE_FILE} logs --no-color --tail=1000"

    if [[ "${service}" != "all" ]]; then
        cmd="${cmd} ${service}"
    fi

    # Get logs
    local logs=$(eval "${cmd}" 2>&1)

    # Count log levels
    local error_count=$(echo "${logs}" | grep -ic "error\|exception\|critical\|fatal" || echo 0)
    local warning_count=$(echo "${logs}" | grep -ic "warn\|warning" || echo 0)
    local info_count=$(echo "${logs}" | grep -ic "info" || echo 0)

    # Display statistics
    echo "Log Statistics (last 1000 lines):"
    echo "  Errors:   ${error_count}"
    echo "  Warnings: ${warning_count}"
    echo "  Info:     ${info_count}"
    echo ""

    # Show top errors if any
    if [[ $error_count -gt 0 ]]; then
        echo "Recent Errors:"
        echo "${logs}" | grep -iE "error|exception|critical|fatal" | tail -5 | while read -r line; do
            echo "  ${line}"
        done
        echo ""
    fi

    # Show request statistics for API logs
    if [[ "${service}" == "api" ]] || [[ "${service}" == "all" ]]; then
        echo "API Request Statistics:"
        local total_requests=$(echo "${logs}" | grep -cE "GET|POST|PUT|DELETE|PATCH" || echo 0)
        local get_requests=$(echo "${logs}" | grep -cE "GET" || echo 0)
        local post_requests=$(echo "${logs}" | grep -cE "POST" || echo 0)

        echo "  Total Requests: ${total_requests}"
        echo "  GET:            ${get_requests}"
        echo "  POST:           ${post_requests}"
        echo ""
    fi

    # Show most common log messages
    echo "Most Common Messages:"
    echo "${logs}" | cut -d' ' -f4- | sort | uniq -c | sort -rn | head -5
}

# =============================================================================
# Service-specific Log Viewers
# =============================================================================

view_nginx_access_logs() {
    print_info "Nginx Access Logs"
    docker-compose -f "${COMPOSE_FILE}" exec nginx tail -n "${LINES}" /var/log/nginx/access.log | colorize_logs
}

view_nginx_error_logs() {
    print_info "Nginx Error Logs"
    docker-compose -f "${COMPOSE_FILE}" exec nginx tail -n "${LINES}" /var/log/nginx/error.log | colorize_logs
}

view_postgres_logs() {
    print_info "PostgreSQL Logs"
    docker-compose -f "${COMPOSE_FILE}" logs --tail="${LINES}" postgres | colorize_logs
}

# =============================================================================
# Interactive Mode
# =============================================================================

interactive_mode() {
    while true; do
        clear
        echo "========================================="
        echo "Promptly API - Log Viewer"
        echo "========================================="
        echo ""
        echo "Select a service:"
        echo "  1) All services"
        echo "  2) API"
        echo "  3) PostgreSQL"
        echo "  4) Redis"
        echo "  5) Nginx"
        echo "  6) Nginx Access Logs"
        echo "  7) Nginx Error Logs"
        echo "  8) Analyze Logs"
        echo "  9) Exit"
        echo ""
        read -p "Enter choice [1-9]: " choice

        case $choice in
            1)
                view_logs "all"
                ;;
            2)
                view_logs "api"
                ;;
            3)
                view_logs "postgres"
                ;;
            4)
                view_logs "redis"
                ;;
            5)
                view_logs "nginx"
                ;;
            6)
                view_nginx_access_logs
                ;;
            7)
                view_nginx_error_logs
                ;;
            8)
                read -p "Enter service (all/api/postgres/redis/nginx): " svc
                analyze_logs "${svc}"
                ;;
            9)
                exit 0
                ;;
            *)
                print_error "Invalid choice"
                ;;
        esac

        echo ""
        read -p "Press Enter to continue..."
    done
}

# =============================================================================
# Main
# =============================================================================

show_help() {
    cat << EOF
Promptly API - Log Viewer Script

Usage: ./logs.sh [service] [options]

Services:
    all          All services (default)
    api          Promptly API
    postgres     PostgreSQL database
    redis        Redis cache
    nginx        Nginx reverse proxy

Options:
    -f, --follow        Follow log output (like tail -f)
    -n, --lines NUM     Number of lines to show (default: 100)
    --since TIME        Show logs since timestamp (e.g., 2023-01-01, 1h, 30m)
    --until TIME        Show logs until timestamp
    --grep PATTERN      Filter logs by pattern
    --errors            Show only errors and warnings
    --analyze           Analyze log patterns and statistics
    --export FILE       Export logs to file
    --interactive       Interactive mode
    --help              Show this help message

Examples:
    ./logs.sh                           # View all logs (last 100 lines)
    ./logs.sh api -f                    # Follow API logs
    ./logs.sh postgres -n 50            # Last 50 PostgreSQL logs
    ./logs.sh --errors                  # Show only errors from all services
    ./logs.sh api --since 1h            # API logs from last hour
    ./logs.sh --grep "error"            # Filter logs containing "error"
    ./logs.sh api --export api.log      # Export API logs to file
    ./logs.sh --analyze                 # Analyze log patterns

EOF
    exit 0
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            all|api|postgres|redis|nginx)
                SERVICE=$1
                shift
                ;;
            -f|--follow)
                FOLLOW=true
                shift
                ;;
            -n|--lines)
                LINES=$2
                shift 2
                ;;
            --since)
                SINCE=$2
                shift 2
                ;;
            --until)
                UNTIL=$2
                shift 2
                ;;
            --grep)
                GREP_PATTERN=$2
                shift 2
                ;;
            --errors)
                ERRORS_ONLY=true
                shift
                ;;
            --analyze)
                analyze_logs "${SERVICE}"
                exit 0
                ;;
            --export)
                EXPORT_FILE=$2
                shift 2
                ;;
            --interactive)
                interactive_mode
                exit 0
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

    # Export or view logs
    if [[ -n "${EXPORT_FILE}" ]]; then
        export_logs "${SERVICE}" "${EXPORT_FILE}"
    else
        # Check if service is running
        if [[ "${SERVICE}" != "all" ]]; then
            if ! docker-compose -f "${COMPOSE_FILE}" ps "${SERVICE}" | grep -q "Up"; then
                print_warning "Service '${SERVICE}' is not running"
            fi
        fi

        view_logs "${SERVICE}"
    fi
}

# Run main function
main "$@"

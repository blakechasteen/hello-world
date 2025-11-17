#!/bin/bash

# =============================================================================
# Promptly API - Deployment Verification Script
# =============================================================================
#
# This script verifies that all deployment files and configurations are in place.
#
# Usage:
#   ./verify-deployment.sh
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"
}

check_pass() {
    echo -e "${GREEN}✓ $1${NC}"
    ((PASSED++))
}

check_fail() {
    echo -e "${RED}✗ $1${NC}"
    ((FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
    ((WARNINGS++))
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Verification Functions
# =============================================================================

verify_files() {
    print_header "Verifying Files"

    # Required files
    local files=(
        "docker-compose.production.yml:Production Docker Compose"
        ".env.production:Environment Configuration"
        "nginx.production.conf:Nginx Configuration"
        "Dockerfile:Docker Image Definition"
        "deploy.sh:Deployment Script"
        "backup.sh:Backup Script"
        "monitor.sh:Monitoring Script"
        "logs.sh:Log Viewer Script"
        "cleanup.sh:Cleanup Script"
        "PRODUCTION_DEPLOYMENT.md:Deployment Guide"
        "requirements.txt:Python Dependencies"
    )

    for file_info in "${files[@]}"; do
        IFS=':' read -r file desc <<< "${file_info}"
        if [[ -f "${SCRIPT_DIR}/${file}" ]]; then
            check_pass "${desc}: ${file}"
        else
            check_fail "${desc}: ${file} (NOT FOUND)"
        fi
    done
}

verify_directories() {
    print_header "Verifying Directories"

    local dirs=(
        "backups:Backup Storage"
        "logs:Application Logs"
        "nginx-logs:Nginx Logs"
        "init-db:Database Initialization"
        "ssl:SSL Certificates"
    )

    for dir_info in "${dirs[@]}"; do
        IFS=':' read -r dir desc <<< "${dir_info}"
        if [[ -d "${SCRIPT_DIR}/${dir}" ]]; then
            check_pass "${desc}: ${dir}/"
        else
            check_fail "${desc}: ${dir}/ (NOT FOUND)"
        fi
    done
}

verify_scripts() {
    print_header "Verifying Script Permissions"

    local scripts=(
        "deploy.sh"
        "backup.sh"
        "monitor.sh"
        "logs.sh"
        "cleanup.sh"
    )

    for script in "${scripts[@]}"; do
        if [[ -x "${SCRIPT_DIR}/${script}" ]]; then
            check_pass "${script} is executable"
        else
            check_fail "${script} is NOT executable (run: chmod +x ${script})"
        fi
    done
}

verify_environment() {
    print_header "Verifying Environment Configuration"

    if [[ ! -f "${SCRIPT_DIR}/.env.production" ]]; then
        check_fail "Environment file not found"
        return
    fi

    # Source environment
    set -a
    source "${SCRIPT_DIR}/.env.production" 2>/dev/null || true
    set +a

    # Check required variables
    local vars=(
        "POSTGRES_PASSWORD:PostgreSQL Password"
        "REDIS_PASSWORD:Redis Password"
        "PROMPTLY_SECRET_KEY:API Secret Key"
    )

    for var_info in "${vars[@]}"; do
        IFS=':' read -r var desc <<< "${var_info}"
        if [[ -z "${!var:-}" ]]; then
            check_fail "${desc} (${var}) is NOT set"
        elif [[ "${!var}" == *"CHANGE_ME"* ]]; then
            check_warn "${desc} (${var}) still uses default value"
        else
            check_pass "${desc} (${var}) is configured"
        fi
    done

    # Check secret key length
    if [[ -n "${PROMPTLY_SECRET_KEY:-}" ]] && [[ ${#PROMPTLY_SECRET_KEY} -lt 32 ]]; then
        check_fail "PROMPTLY_SECRET_KEY is too short (minimum 32 characters)"
    fi
}

verify_ssl() {
    print_header "Verifying SSL Certificates"

    if [[ -f "${SCRIPT_DIR}/ssl/cert.pem" ]] && [[ -f "${SCRIPT_DIR}/ssl/key.pem" ]]; then
        check_pass "SSL certificates found"

        # Check expiration
        if openssl x509 -checkend 604800 -noout -in "${SCRIPT_DIR}/ssl/cert.pem" 2>/dev/null; then
            check_pass "SSL certificate is valid"
        else
            check_warn "SSL certificate expires within 7 days"
        fi
    else
        check_warn "SSL certificates not found (will be auto-generated)"
        print_info "For production, add valid SSL certificates to ssl/"
    fi
}

verify_prerequisites() {
    print_header "Verifying Prerequisites"

    # Docker
    if command -v docker &> /dev/null; then
        local version=$(docker --version)
        check_pass "Docker installed: ${version}"
    else
        check_fail "Docker is NOT installed"
    fi

    # Docker Compose
    if command -v docker-compose &> /dev/null; then
        local version=$(docker-compose --version)
        check_pass "Docker Compose installed: ${version}"
    elif docker compose version &> /dev/null; then
        local version=$(docker compose version)
        check_pass "Docker Compose (plugin) installed: ${version}"
    else
        check_fail "Docker Compose is NOT installed"
    fi

    # OpenSSL
    if command -v openssl &> /dev/null; then
        check_pass "OpenSSL installed"
    else
        check_warn "OpenSSL not found (needed for SSL certificate generation)"
    fi

    # curl
    if command -v curl &> /dev/null; then
        check_pass "curl installed"
    else
        check_warn "curl not found (needed for testing)"
    fi

    # Check disk space
    local available=$(df -BG "${SCRIPT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available -ge 5 ]]; then
        check_pass "Sufficient disk space: ${available}GB available"
    else
        check_fail "Insufficient disk space: ${available}GB (need at least 5GB)"
    fi

    # Check memory
    local memory=$(free -g | awk 'NR==2 {print $2}')
    if [[ $memory -ge 4 ]]; then
        check_pass "Sufficient memory: ${memory}GB"
    else
        check_warn "Low memory: ${memory}GB (recommended: 4GB+)"
    fi
}

verify_docker_compose() {
    print_header "Verifying Docker Compose Configuration"

    if command -v docker-compose &> /dev/null; then
        local compose_cmd="docker-compose"
    else
        local compose_cmd="docker compose"
    fi

    # Validate compose file
    if ${compose_cmd} -f "${SCRIPT_DIR}/docker-compose.production.yml" config &> /dev/null; then
        check_pass "Docker Compose file is valid"
    else
        check_fail "Docker Compose file has errors"
        ${compose_cmd} -f "${SCRIPT_DIR}/docker-compose.production.yml" config 2>&1 | head -10
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    print_header "Promptly API - Deployment Verification"

    verify_prerequisites
    verify_files
    verify_directories
    verify_scripts
    verify_environment
    verify_ssl
    verify_docker_compose

    # Summary
    print_header "Verification Summary"

    local total=$((PASSED + FAILED + WARNINGS))

    echo -e "${GREEN}Passed:   ${PASSED}${NC}"
    echo -e "${YELLOW}Warnings: ${WARNINGS}${NC}"
    echo -e "${RED}Failed:   ${FAILED}${NC}"
    echo "Total:    ${total}"
    echo ""

    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ Deployment is ready!${NC}"
        echo ""
        print_info "Next steps:"
        print_info "  1. Review and update .env.production"
        print_info "  2. Add SSL certificates to ssl/ (optional)"
        print_info "  3. Run: ./deploy.sh"
        echo ""
        exit 0
    else
        echo -e "${RED}${BOLD}✗ Deployment has issues that need to be fixed${NC}"
        echo ""
        print_info "Fix the failed checks above before deploying"
        echo ""
        exit 1
    fi
}

# Run main function
main

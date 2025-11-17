#!/bin/bash

# =============================================================================
# Promptly API - Production Deployment Script
# =============================================================================
#
# This script handles the complete deployment process including:
# - Environment validation
# - SSL certificate setup
# - Docker image building
# - Service deployment
# - Health checks
# - Rollback on failure
#
# Usage:
#   ./deploy.sh [options]
#
# Options:
#   --build      Force rebuild of Docker images
#   --no-backup  Skip database backup before deployment
#   --quick      Skip health checks (not recommended)
#   --help       Show this help message
#
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env.production"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.production.yml"
SSL_DIR="${SCRIPT_DIR}/ssl"
BACKUP_DIR="${SCRIPT_DIR}/backups"
LOG_DIR="${SCRIPT_DIR}/logs"

# Command line options
FORCE_BUILD=false
SKIP_BACKUP=false
SKIP_HEALTH_CHECK=false

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# =============================================================================
# Validation Functions
# =============================================================================

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker is installed"

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose is installed"

    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root is not recommended"
    fi

    # Check disk space
    available_space=$(df -BG "${SCRIPT_DIR}" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_space -lt 5 ]]; then
        print_error "Insufficient disk space (need at least 5GB)"
        exit 1
    fi
    print_success "Sufficient disk space available (${available_space}GB)"
}

validate_environment() {
    print_header "Validating Environment Configuration"

    if [[ ! -f "${ENV_FILE}" ]]; then
        print_error "Environment file not found: ${ENV_FILE}"
        print_info "Copy .env.production template and configure it"
        exit 1
    fi
    print_success "Environment file exists"

    # Source environment file
    set -a
    source "${ENV_FILE}"
    set +a

    # Check required variables
    required_vars=(
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "PROMPTLY_SECRET_KEY"
    )

    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            print_error "Required environment variable not set: $var"
            exit 1
        fi

        # Check if still using default values
        if [[ "${!var}" == *"CHANGE_ME"* ]]; then
            print_error "Please change the default value for: $var"
            exit 1
        fi
    done
    print_success "All required environment variables are set"

    # Validate secret key length
    if [[ ${#PROMPTLY_SECRET_KEY} -lt 32 ]]; then
        print_error "PROMPTLY_SECRET_KEY must be at least 32 characters"
        exit 1
    fi
    print_success "Secret key meets minimum length requirements"
}

setup_ssl_certificates() {
    print_header "Setting Up SSL Certificates"

    mkdir -p "${SSL_DIR}"

    if [[ ! -f "${SSL_DIR}/cert.pem" ]] || [[ ! -f "${SSL_DIR}/key.pem" ]]; then
        print_warning "SSL certificates not found"
        print_info "Generating self-signed certificates for development..."

        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "${SSL_DIR}/key.pem" \
            -out "${SSL_DIR}/cert.pem" \
            -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
            2>/dev/null

        print_warning "Self-signed certificates generated"
        print_info "For production, replace with valid SSL certificates"
    else
        print_success "SSL certificates found"

        # Check certificate expiration
        if openssl x509 -checkend 604800 -noout -in "${SSL_DIR}/cert.pem" &>/dev/null; then
            print_success "SSL certificate is valid"
        else
            print_warning "SSL certificate expires within 7 days"
        fi
    fi
}

create_directories() {
    print_header "Creating Required Directories"

    directories=(
        "${BACKUP_DIR}"
        "${LOG_DIR}"
        "${SCRIPT_DIR}/nginx-logs"
        "${SCRIPT_DIR}/init-db"
    )

    for dir in "${directories[@]}"; do
        mkdir -p "${dir}"
        print_success "Created directory: ${dir}"
    done
}

# =============================================================================
# Backup Functions
# =============================================================================

backup_database() {
    if [[ "${SKIP_BACKUP}" == "true" ]]; then
        print_warning "Skipping database backup"
        return 0
    fi

    print_header "Backing Up Database"

    # Check if database is running
    if docker-compose -f "${COMPOSE_FILE}" ps postgres | grep -q "Up"; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        backup_file="${BACKUP_DIR}/promptly_backup_${timestamp}.sql"

        print_info "Creating backup: ${backup_file}"

        docker-compose -f "${COMPOSE_FILE}" exec -T postgres \
            pg_dump -U "${POSTGRES_USER:-promptly}" "${POSTGRES_DB:-promptly}" \
            > "${backup_file}" 2>/dev/null || true

        if [[ -f "${backup_file}" ]] && [[ -s "${backup_file}" ]]; then
            gzip "${backup_file}"
            print_success "Database backup created: ${backup_file}.gz"
        else
            print_warning "Database backup skipped (database may be empty)"
        fi
    else
        print_info "Database not running, skipping backup"
    fi
}

# =============================================================================
# Deployment Functions
# =============================================================================

build_images() {
    print_header "Building Docker Images"

    if [[ "${FORCE_BUILD}" == "true" ]]; then
        print_info "Force building images..."
        docker-compose -f "${COMPOSE_FILE}" build --no-cache
    else
        docker-compose -f "${COMPOSE_FILE}" build
    fi

    print_success "Docker images built successfully"
}

deploy_services() {
    print_header "Deploying Services"

    # Pull latest images for base services
    print_info "Pulling latest base images..."
    docker-compose -f "${COMPOSE_FILE}" pull postgres redis nginx

    # Start services
    print_info "Starting services..."
    docker-compose -f "${COMPOSE_FILE}" up -d

    print_success "Services deployed successfully"
}

wait_for_health() {
    local service=$1
    local max_attempts=$2
    local attempt=1

    print_info "Waiting for ${service} to become healthy..."

    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose -f "${COMPOSE_FILE}" ps "${service}" | grep -q "healthy"; then
            print_success "${service} is healthy"
            return 0
        fi

        echo -n "."
        sleep 2
        ((attempt++))
    done

    echo ""
    print_error "${service} failed to become healthy"
    return 1
}

verify_deployment() {
    if [[ "${SKIP_HEALTH_CHECK}" == "true" ]]; then
        print_warning "Skipping health checks"
        return 0
    fi

    print_header "Verifying Deployment"

    # Wait for services to be healthy
    wait_for_health "postgres" 30 || return 1
    wait_for_health "redis" 30 || return 1
    wait_for_health "api" 60 || return 1
    wait_for_health "nginx" 30 || return 1

    # Test API endpoint
    print_info "Testing API endpoint..."
    sleep 5  # Give nginx a moment to stabilize

    if curl -f -s -o /dev/null "http://localhost/health"; then
        print_success "API endpoint responding"
    else
        print_error "API endpoint not responding"
        return 1
    fi

    # Show service status
    print_info "Service status:"
    docker-compose -f "${COMPOSE_FILE}" ps

    print_success "Deployment verification complete"
}

# =============================================================================
# Rollback Functions
# =============================================================================

rollback() {
    print_error "Deployment failed, initiating rollback..."

    print_info "Stopping new containers..."
    docker-compose -f "${COMPOSE_FILE}" down

    print_info "Check logs for errors:"
    print_info "  ./logs.sh"

    exit 1
}

# =============================================================================
# Main Deployment Flow
# =============================================================================

show_help() {
    cat << EOF
Promptly API - Production Deployment Script

Usage: ./deploy.sh [options]

Options:
    --build         Force rebuild of Docker images
    --no-backup     Skip database backup before deployment
    --quick         Skip health checks (not recommended)
    --help          Show this help message

Examples:
    ./deploy.sh                    # Standard deployment
    ./deploy.sh --build            # Rebuild images and deploy
    ./deploy.sh --no-backup        # Deploy without backup

EOF
    exit 0
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                FORCE_BUILD=true
                shift
                ;;
            --no-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --quick)
                SKIP_HEALTH_CHECK=true
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

    print_header "Promptly API - Production Deployment"

    # Pre-deployment checks
    check_prerequisites
    validate_environment
    setup_ssl_certificates
    create_directories

    # Backup
    backup_database

    # Deploy
    build_images
    deploy_services

    # Verify
    if ! verify_deployment; then
        rollback
    fi

    # Success
    print_header "Deployment Complete"
    print_success "Promptly API is running!"
    echo ""
    print_info "Access points:"
    print_info "  - API:          http://localhost/api/v1"
    print_info "  - Health:       http://localhost/health"
    print_info "  - Docs:         http://localhost/docs"
    print_info "  - WebSocket:    ws://localhost/ws"
    echo ""
    print_info "Useful commands:"
    print_info "  - View logs:    ./logs.sh"
    print_info "  - Monitor:      ./monitor.sh"
    print_info "  - Backup:       ./backup.sh"
    print_info "  - Cleanup:      ./cleanup.sh"
    echo ""
    print_warning "Remember to:"
    print_warning "  1. Configure firewall rules"
    print_warning "  2. Set up proper SSL certificates"
    print_warning "  3. Configure backup automation"
    print_warning "  4. Set up monitoring alerts"
}

# Trap errors and rollback
trap rollback ERR

# Run main function
main "$@"

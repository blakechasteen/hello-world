#!/bin/bash

# =============================================================================
# Promptly API - Cleanup and Shutdown Script
# =============================================================================
#
# This script handles graceful shutdown and cleanup of all services.
# It can optionally remove volumes and data.
#
# Usage:
#   ./cleanup.sh [options]
#
# Options:
#   --stop-only         Stop services but keep containers
#   --remove-volumes    Remove all volumes (WARNING: deletes all data)
#   --remove-images     Remove Docker images
#   --full-cleanup      Complete cleanup (stops, removes containers, volumes, images)
#   --backup-first      Create backup before cleanup
#   --force             Skip confirmation prompts
#   --help              Show this help message
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
ENV_FILE="${SCRIPT_DIR}/.env.production"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.production.yml"

# Options
STOP_ONLY=false
REMOVE_VOLUMES=false
REMOVE_IMAGES=false
FULL_CLEANUP=false
BACKUP_FIRST=false
FORCE=false

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BOLD}${BLUE}========================================${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}========================================${NC}\n"
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

confirm() {
    if [[ "${FORCE}" == "true" ]]; then
        return 0
    fi

    local message=$1
    read -p "${message} (yes/no): " response

    case "${response}" in
        yes|y|Y|YES)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# =============================================================================
# Cleanup Functions
# =============================================================================

stop_services() {
    print_header "Stopping Services"

    # Check if services are running
    if ! docker-compose -f "${COMPOSE_FILE}" ps | grep -q "Up"; then
        print_info "No services are running"
        return 0
    fi

    print_info "Stopping all services gracefully..."

    # Stop services in order
    docker-compose -f "${COMPOSE_FILE}" stop nginx
    print_success "Nginx stopped"

    docker-compose -f "${COMPOSE_FILE}" stop api
    print_success "API stopped"

    docker-compose -f "${COMPOSE_FILE}" stop redis
    print_success "Redis stopped"

    docker-compose -f "${COMPOSE_FILE}" stop postgres
    print_success "PostgreSQL stopped"

    print_success "All services stopped"
}

remove_containers() {
    print_header "Removing Containers"

    if docker-compose -f "${COMPOSE_FILE}" ps -a | grep -q "promptly"; then
        print_info "Removing containers..."
        docker-compose -f "${COMPOSE_FILE}" down
        print_success "Containers removed"
    else
        print_info "No containers to remove"
    fi
}

remove_volumes() {
    print_header "Removing Volumes"

    print_warning "This will DELETE ALL DATA including:"
    print_warning "  - PostgreSQL database"
    print_warning "  - Redis cache"
    print_warning "  - Application data"
    echo ""

    if ! confirm "Are you sure you want to remove all volumes?"; then
        print_info "Volume removal cancelled"
        return 0
    fi

    # List volumes before removal
    print_info "Current volumes:"
    docker volume ls | grep promptly || echo "No Promptly volumes found"
    echo ""

    # Remove volumes
    docker-compose -f "${COMPOSE_FILE}" down -v
    print_success "Volumes removed"

    # Remove orphaned volumes
    local orphaned=$(docker volume ls -qf dangling=true | grep promptly || echo "")
    if [[ -n "${orphaned}" ]]; then
        echo "${orphaned}" | xargs docker volume rm 2>/dev/null || true
        print_success "Orphaned volumes removed"
    fi
}

remove_images() {
    print_header "Removing Images"

    # List Promptly images
    local images=$(docker images | grep promptly || echo "")

    if [[ -z "${images}" ]]; then
        print_info "No Promptly images to remove"
        return 0
    fi

    print_info "Images to remove:"
    echo "${images}"
    echo ""

    if ! confirm "Remove these images?"; then
        print_info "Image removal cancelled"
        return 0
    fi

    # Remove images
    docker-compose -f "${COMPOSE_FILE}" down --rmi all 2>/dev/null || true
    print_success "Images removed"

    # Remove dangling images
    local dangling=$(docker images -f "dangling=true" -q)
    if [[ -n "${dangling}" ]]; then
        echo "${dangling}" | xargs docker rmi 2>/dev/null || true
        print_success "Dangling images removed"
    fi
}

cleanup_networks() {
    print_header "Cleaning Up Networks"

    # Remove Promptly networks
    local networks=$(docker network ls | grep promptly || echo "")

    if [[ -z "${networks}" ]]; then
        print_info "No Promptly networks to remove"
        return 0
    fi

    docker network ls | grep promptly | awk '{print $1}' | xargs docker network rm 2>/dev/null || true
    print_success "Networks removed"
}

cleanup_logs() {
    print_header "Cleaning Up Logs"

    local log_dirs=(
        "${SCRIPT_DIR}/logs"
        "${SCRIPT_DIR}/nginx-logs"
    )

    for dir in "${log_dirs[@]}"; do
        if [[ -d "${dir}" ]]; then
            if confirm "Remove log directory: ${dir}?"; then
                rm -rf "${dir}"
                print_success "Removed: ${dir}"
            fi
        fi
    done
}

cleanup_temp_files() {
    print_header "Cleaning Up Temporary Files"

    local temp_patterns=(
        "${SCRIPT_DIR}/*.pyc"
        "${SCRIPT_DIR}/__pycache__"
        "${SCRIPT_DIR}/.pytest_cache"
        "${SCRIPT_DIR}/.coverage"
    )

    for pattern in "${temp_patterns[@]}"; do
        if ls ${pattern} 1> /dev/null 2>&1; then
            rm -rf ${pattern}
            print_success "Removed: ${pattern}"
        fi
    done
}

system_prune() {
    print_header "Docker System Cleanup"

    print_info "Current Docker disk usage:"
    docker system df
    echo ""

    if confirm "Run Docker system prune?"; then
        docker system prune -f
        print_success "Docker system pruned"

        echo ""
        print_info "New Docker disk usage:"
        docker system df
    fi
}

# =============================================================================
# Backup Before Cleanup
# =============================================================================

create_backup() {
    print_header "Creating Backup Before Cleanup"

    if [[ -x "${SCRIPT_DIR}/backup.sh" ]]; then
        print_info "Running backup script..."
        "${SCRIPT_DIR}/backup.sh" --full
        print_success "Backup completed"
    else
        print_warning "Backup script not found or not executable"
        if confirm "Continue without backup?"; then
            return 0
        else
            print_error "Cleanup cancelled"
            exit 1
        fi
    fi
}

# =============================================================================
# Status and Verification
# =============================================================================

show_status() {
    print_header "Current Status"

    print_info "Containers:"
    docker-compose -f "${COMPOSE_FILE}" ps 2>/dev/null || echo "No containers running"

    echo ""
    print_info "Volumes:"
    docker volume ls | grep promptly || echo "No volumes found"

    echo ""
    print_info "Images:"
    docker images | grep promptly || echo "No images found"

    echo ""
    print_info "Networks:"
    docker network ls | grep promptly || echo "No networks found"
}

verify_cleanup() {
    print_header "Verifying Cleanup"

    local issues=0

    # Check containers
    if docker-compose -f "${COMPOSE_FILE}" ps | grep -q "Up"; then
        print_warning "Some containers are still running"
        ((issues++))
    else
        print_success "No containers running"
    fi

    # Check volumes (if they should be removed)
    if [[ "${REMOVE_VOLUMES}" == "true" ]] || [[ "${FULL_CLEANUP}" == "true" ]]; then
        if docker volume ls | grep -q promptly; then
            print_warning "Some volumes still exist"
            ((issues++))
        else
            print_success "All volumes removed"
        fi
    fi

    # Check images (if they should be removed)
    if [[ "${REMOVE_IMAGES}" == "true" ]] || [[ "${FULL_CLEANUP}" == "true" ]]; then
        if docker images | grep -q promptly; then
            print_warning "Some images still exist"
            ((issues++))
        else
            print_success "All images removed"
        fi
    fi

    if [[ $issues -eq 0 ]]; then
        print_success "Cleanup verification passed"
    else
        print_warning "Cleanup completed with ${issues} issue(s)"
    fi
}

# =============================================================================
# Main Cleanup Flow
# =============================================================================

show_help() {
    cat << EOF
Promptly API - Cleanup and Shutdown Script

Usage: ./cleanup.sh [options]

Options:
    --stop-only         Stop services but keep containers
    --remove-volumes    Remove all volumes (WARNING: deletes all data)
    --remove-images     Remove Docker images
    --full-cleanup      Complete cleanup (everything)
    --backup-first      Create backup before cleanup
    --force             Skip confirmation prompts
    --status            Show current status without cleanup
    --help              Show this help message

Cleanup Levels:
    1. Default:         Stop services and remove containers
    2. --remove-volumes: + Remove all data volumes
    3. --remove-images:  + Remove Docker images
    4. --full-cleanup:   Everything (containers, volumes, images, networks)

Examples:
    ./cleanup.sh                        # Stop and remove containers
    ./cleanup.sh --stop-only            # Just stop services
    ./cleanup.sh --remove-volumes       # Stop and remove all data
    ./cleanup.sh --full-cleanup         # Complete cleanup
    ./cleanup.sh --backup-first         # Backup before cleanup
    ./cleanup.sh --status               # Show current status

EOF
    exit 0
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stop-only)
                STOP_ONLY=true
                shift
                ;;
            --remove-volumes)
                REMOVE_VOLUMES=true
                shift
                ;;
            --remove-images)
                REMOVE_IMAGES=true
                shift
                ;;
            --full-cleanup)
                FULL_CLEANUP=true
                shift
                ;;
            --backup-first)
                BACKUP_FIRST=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --status)
                show_status
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

    print_header "Promptly API - Cleanup"

    # Show current status
    show_status

    # Backup if requested
    if [[ "${BACKUP_FIRST}" == "true" ]]; then
        create_backup
    fi

    # Warning for destructive operations
    if [[ "${FULL_CLEANUP}" == "true" ]] || [[ "${REMOVE_VOLUMES}" == "true" ]]; then
        echo ""
        print_warning "${BOLD}WARNING: This operation will DELETE DATA!${NC}"
        echo ""

        if ! confirm "Are you absolutely sure you want to continue?"; then
            print_info "Cleanup cancelled"
            exit 0
        fi
    fi

    # Perform cleanup based on options
    if [[ "${FULL_CLEANUP}" == "true" ]]; then
        stop_services
        remove_containers
        remove_volumes
        remove_images
        cleanup_networks
        cleanup_logs
        cleanup_temp_files
        system_prune
    elif [[ "${STOP_ONLY}" == "true" ]]; then
        stop_services
    else
        stop_services
        remove_containers

        if [[ "${REMOVE_VOLUMES}" == "true" ]]; then
            remove_volumes
        fi

        if [[ "${REMOVE_IMAGES}" == "true" ]]; then
            remove_images
        fi
    fi

    # Verify cleanup
    verify_cleanup

    # Final status
    print_header "Cleanup Complete"

    if [[ "${FULL_CLEANUP}" == "true" ]]; then
        print_success "Full cleanup completed"
        print_info "All Promptly services, data, and images have been removed"
    elif [[ "${STOP_ONLY}" == "true" ]]; then
        print_success "Services stopped"
        print_info "To restart: ./deploy.sh"
    else
        print_success "Cleanup completed"
        print_info "To redeploy: ./deploy.sh"
    fi

    echo ""
    print_info "Useful commands:"
    print_info "  - Deploy:  ./deploy.sh"
    print_info "  - Status:  ./cleanup.sh --status"
    print_info "  - Backup:  ./backup.sh"
}

# Run main function
main "$@"

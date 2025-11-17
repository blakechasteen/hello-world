#!/bin/bash

# =============================================================================
# Promptly API - Database Backup Script
# =============================================================================
#
# This script creates backups of the PostgreSQL database and Redis data.
# Supports multiple backup strategies and automatic cleanup of old backups.
#
# Usage:
#   ./backup.sh [options]
#
# Options:
#   --full           Full backup (PostgreSQL + Redis + application data)
#   --db-only        Database only (default)
#   --redis-only     Redis only
#   --retain DAYS    Number of days to retain backups (default: 30)
#   --output DIR     Output directory (default: ./backups)
#   --compress       Compress backups with gzip
#   --help           Show this help message
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
BACKUP_DIR="${SCRIPT_DIR}/backups"
RETENTION_DAYS=30
COMPRESS=true

# Backup types
BACKUP_TYPE="db-only"

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
# Backup Functions
# =============================================================================

backup_postgresql() {
    print_header "Backing Up PostgreSQL Database"

    # Check if database is running
    if ! docker-compose -f "${COMPOSE_FILE}" ps postgres | grep -q "Up"; then
        print_error "PostgreSQL container is not running"
        return 1
    fi

    # Source environment
    set -a
    source "${ENV_FILE}"
    set +a

    # Create backup filename
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="${BACKUP_DIR}/postgres_${timestamp}.sql"

    print_info "Creating PostgreSQL backup..."

    # Create backup
    docker-compose -f "${COMPOSE_FILE}" exec -T postgres \
        pg_dump -U "${POSTGRES_USER:-promptly}" \
        -d "${POSTGRES_DB:-promptly}" \
        --clean \
        --if-exists \
        --verbose \
        2>/dev/null > "${backup_file}"

    # Check if backup was created
    if [[ ! -f "${backup_file}" ]] || [[ ! -s "${backup_file}" ]]; then
        print_error "Failed to create PostgreSQL backup"
        rm -f "${backup_file}"
        return 1
    fi

    # Get backup size
    backup_size=$(du -h "${backup_file}" | cut -f1)
    print_success "PostgreSQL backup created: ${backup_file} (${backup_size})"

    # Compress if requested
    if [[ "${COMPRESS}" == "true" ]]; then
        print_info "Compressing backup..."
        gzip -f "${backup_file}"
        compressed_size=$(du -h "${backup_file}.gz" | cut -f1)
        print_success "Backup compressed: ${backup_file}.gz (${compressed_size})"
        echo "${backup_file}.gz"
    else
        echo "${backup_file}"
    fi
}

backup_redis() {
    print_header "Backing Up Redis Data"

    # Check if Redis is running
    if ! docker-compose -f "${COMPOSE_FILE}" ps redis | grep -q "Up"; then
        print_error "Redis container is not running"
        return 1
    fi

    # Create backup filename
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="${BACKUP_DIR}/redis_${timestamp}.rdb"

    print_info "Creating Redis backup..."

    # Trigger Redis save
    docker-compose -f "${COMPOSE_FILE}" exec -T redis redis-cli --pass "${REDIS_PASSWORD}" SAVE

    # Copy RDB file
    docker-compose -f "${COMPOSE_FILE}" exec -T redis cat /data/dump.rdb > "${backup_file}" 2>/dev/null

    # Check if backup was created
    if [[ ! -f "${backup_file}" ]] || [[ ! -s "${backup_file}" ]]; then
        print_error "Failed to create Redis backup"
        rm -f "${backup_file}"
        return 1
    fi

    # Get backup size
    backup_size=$(du -h "${backup_file}" | cut -f1)
    print_success "Redis backup created: ${backup_file} (${backup_size})"

    # Compress if requested
    if [[ "${COMPRESS}" == "true" ]]; then
        print_info "Compressing backup..."
        gzip -f "${backup_file}"
        compressed_size=$(du -h "${backup_file}.gz" | cut -f1)
        print_success "Backup compressed: ${backup_file}.gz (${compressed_size})"
        echo "${backup_file}.gz"
    else
        echo "${backup_file}"
    fi
}

backup_application_data() {
    print_header "Backing Up Application Data"

    # Create backup filename
    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_file="${BACKUP_DIR}/app_data_${timestamp}.tar"

    print_info "Creating application data backup..."

    # Get volume names
    postgres_volume=$(docker volume ls -q | grep promptly.*postgres-data || echo "")
    redis_volume=$(docker volume ls -q | grep promptly.*redis-data || echo "")
    app_volume=$(docker volume ls -q | grep promptly.*promptly-data || echo "")

    # Create tar archive
    if [[ -n "${app_volume}" ]]; then
        docker run --rm \
            -v "${app_volume}:/data:ro" \
            -v "${BACKUP_DIR}:/backup" \
            alpine tar czf "/backup/$(basename ${backup_file}).gz" -C /data . 2>/dev/null

        if [[ -f "${backup_file}.gz" ]]; then
            backup_size=$(du -h "${backup_file}.gz" | cut -f1)
            print_success "Application data backup created: ${backup_file}.gz (${backup_size})"
            echo "${backup_file}.gz"
        else
            print_warning "No application data found to backup"
        fi
    else
        print_warning "No application data volume found"
    fi
}

cleanup_old_backups() {
    print_header "Cleaning Up Old Backups"

    print_info "Removing backups older than ${RETENTION_DAYS} days..."

    # Find and delete old backups
    count=0
    while IFS= read -r -d '' file; do
        rm -f "$file"
        ((count++))
    done < <(find "${BACKUP_DIR}" -name "*.sql.gz" -o -name "*.sql" -o -name "*.rdb.gz" -o -name "*.rdb" -o -name "*.tar.gz" -type f -mtime +${RETENTION_DAYS} -print0)

    if [[ $count -gt 0 ]]; then
        print_success "Removed ${count} old backup(s)"
    else
        print_info "No old backups to remove"
    fi

    # Show current backup count and size
    backup_count=$(find "${BACKUP_DIR}" -type f | wc -l)
    backup_size=$(du -sh "${BACKUP_DIR}" 2>/dev/null | cut -f1 || echo "0")

    print_info "Current backups: ${backup_count} files (${backup_size})"
}

verify_backup() {
    local backup_file=$1

    print_info "Verifying backup: $(basename ${backup_file})"

    # Check file exists and is not empty
    if [[ ! -f "${backup_file}" ]]; then
        print_error "Backup file not found: ${backup_file}"
        return 1
    fi

    if [[ ! -s "${backup_file}" ]]; then
        print_error "Backup file is empty: ${backup_file}"
        return 1
    fi

    # Verify compressed files
    if [[ "${backup_file}" == *.gz ]]; then
        if gzip -t "${backup_file}" 2>/dev/null; then
            print_success "Backup integrity verified"
        else
            print_error "Backup file is corrupted"
            return 1
        fi
    fi

    # Calculate checksum
    checksum=$(md5sum "${backup_file}" | cut -d' ' -f1)
    echo "${checksum}  ${backup_file}" > "${backup_file}.md5"
    print_info "Checksum: ${checksum}"

    return 0
}

# =============================================================================
# Main Backup Flow
# =============================================================================

show_help() {
    cat << EOF
Promptly API - Database Backup Script

Usage: ./backup.sh [options]

Options:
    --full              Full backup (PostgreSQL + Redis + application data)
    --db-only           Database only (default)
    --redis-only        Redis only
    --retain DAYS       Number of days to retain backups (default: 30)
    --output DIR        Output directory (default: ./backups)
    --no-compress       Don't compress backups
    --help              Show this help message

Examples:
    ./backup.sh                          # Backup PostgreSQL database
    ./backup.sh --full                   # Full system backup
    ./backup.sh --redis-only             # Backup only Redis
    ./backup.sh --retain 7               # Keep backups for 7 days
    ./backup.sh --output /mnt/backups    # Custom backup location

EOF
    exit 0
}

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --full)
                BACKUP_TYPE="full"
                shift
                ;;
            --db-only)
                BACKUP_TYPE="db-only"
                shift
                ;;
            --redis-only)
                BACKUP_TYPE="redis-only"
                shift
                ;;
            --retain)
                RETENTION_DAYS=$2
                shift 2
                ;;
            --output)
                BACKUP_DIR=$2
                shift 2
                ;;
            --no-compress)
                COMPRESS=false
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

    print_header "Promptly API - Backup"

    # Create backup directory
    mkdir -p "${BACKUP_DIR}"

    # Source environment if available
    if [[ -f "${ENV_FILE}" ]]; then
        set -a
        source "${ENV_FILE}"
        set +a
    fi

    # Perform backups based on type
    backup_files=()

    case "${BACKUP_TYPE}" in
        full)
            print_info "Performing full backup..."
            if pg_backup=$(backup_postgresql); then
                backup_files+=("${pg_backup}")
            fi
            if redis_backup=$(backup_redis); then
                backup_files+=("${redis_backup}")
            fi
            if app_backup=$(backup_application_data); then
                backup_files+=("${app_backup}")
            fi
            ;;
        db-only)
            if pg_backup=$(backup_postgresql); then
                backup_files+=("${pg_backup}")
            fi
            ;;
        redis-only)
            if redis_backup=$(backup_redis); then
                backup_files+=("${redis_backup}")
            fi
            ;;
    esac

    # Verify backups
    for backup_file in "${backup_files[@]}"; do
        verify_backup "${backup_file}"
    done

    # Cleanup old backups
    cleanup_old_backups

    # Summary
    print_header "Backup Complete"
    print_success "Created ${#backup_files[@]} backup(s)"

    for backup_file in "${backup_files[@]}"; do
        print_info "  - $(basename ${backup_file})"
    done

    echo ""
    print_info "Backup directory: ${BACKUP_DIR}"
    print_info "Retention period: ${RETENTION_DAYS} days"
}

# Run main function
main "$@"

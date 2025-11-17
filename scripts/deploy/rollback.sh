#!/bin/bash
# EdWIN Rollback Script
# Rollback to previous deployment version
#
# Usage:
#   ./scripts/deploy/rollback.sh [namespace] [revision]
#
# Examples:
#   ./scripts/deploy/rollback.sh edwin          # Rollback to previous revision
#   ./scripts/deploy/rollback.sh edwin 3        # Rollback to revision 3
#
# Implementation Date: November 15, 2025

set -e

NAMESPACE="${1:-edwin}"
REVISION="${2:-0}"  # 0 means previous revision

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Deployments to rollback
DEPLOYMENTS=("edwin-api" "edwin-dashboard" "edwin-mobile")

log_warning "⏪ Starting rollback for namespace: $NAMESPACE"

# Show rollout history
log_info "Current deployment history:"
for deployment in "${DEPLOYMENTS[@]}"; do
    echo ""
    echo "=== $deployment ==="
    kubectl rollout history deployment "$deployment" -n "$NAMESPACE" || log_warning "$deployment not found"
done

echo ""
read -p "Do you want to continue with rollback? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    log_info "Rollback cancelled"
    exit 0
fi

# Perform rollback
for deployment in "${DEPLOYMENTS[@]}"; do
    log_info "Rolling back $deployment..."

    if [ "$REVISION" = "0" ]; then
        # Rollback to previous revision
        kubectl rollout undo deployment "$deployment" -n "$NAMESPACE" || log_error "Failed to rollback $deployment"
    else
        # Rollback to specific revision
        kubectl rollout undo deployment "$deployment" --to-revision="$REVISION" -n "$NAMESPACE" || log_error "Failed to rollback $deployment"
    fi

    # Wait for rollback to complete
    kubectl rollout status deployment "$deployment" -n "$NAMESPACE" --timeout=5m || log_error "Rollback timeout for $deployment"

    log_success "$deployment rolled back successfully"
done

log_success "✅ Rollback complete"

# Show current status
log_info "Current pod status:"
kubectl get pods -n "$NAMESPACE"

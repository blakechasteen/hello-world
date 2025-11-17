#!/bin/bash
# HoloLoom Kubernetes Rollback Script
# This script rolls back a HoloLoom deployment to a previous version

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-hololoom}"
RELEASE_NAME="${RELEASE_NAME:-hololoom}"
REVISION="${REVISION:-}"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        print_error "helm is not installed"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Function to show release history
show_history() {
    print_info "Release history for '$RELEASE_NAME' in namespace '$NAMESPACE':"
    echo ""

    if ! helm history "$RELEASE_NAME" -n "$NAMESPACE" 2>/dev/null; then
        print_error "Release '$RELEASE_NAME' not found in namespace '$NAMESPACE'"
        exit 1
    fi

    echo ""
}

# Function to get current revision
get_current_revision() {
    helm history "$RELEASE_NAME" -n "$NAMESPACE" --max 1 -o json | \
        grep -o '"revision":[0-9]*' | \
        grep -o '[0-9]*'
}

# Function to confirm rollback
confirm_rollback() {
    local current_rev=$1
    local target_rev=$2

    echo ""
    print_warning "You are about to rollback from revision $current_rev to revision $target_rev"
    echo ""

    # Show what will be rolled back
    print_info "Current revision details:"
    helm get values "$RELEASE_NAME" -n "$NAMESPACE" --revision "$current_rev"

    echo ""
    print_info "Target revision details:"
    helm get values "$RELEASE_NAME" -n "$NAMESPACE" --revision "$target_rev"

    echo ""
    read -p "Are you sure you want to proceed? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        print_info "Rollback cancelled"
        exit 0
    fi
}

# Function to perform rollback
perform_rollback() {
    local target_rev=$1

    print_info "Rolling back to revision $target_rev..."

    if [ -z "$target_rev" ]; then
        # Rollback to previous revision
        print_info "Rolling back to previous revision..."
        helm rollback "$RELEASE_NAME" -n "$NAMESPACE" --wait --timeout 10m
    else
        # Rollback to specific revision
        print_info "Rolling back to revision $target_rev..."
        helm rollback "$RELEASE_NAME" "$target_rev" -n "$NAMESPACE" --wait --timeout 10m
    fi

    print_success "Rollback command executed"
}

# Function to verify rollback
verify_rollback() {
    print_info "Verifying rollback..."

    # Wait for deployments
    print_info "Waiting for API deployment..."
    kubectl rollout status deployment/${RELEASE_NAME}-api -n "$NAMESPACE" --timeout=5m

    print_info "Waiting for Worker deployment..."
    kubectl rollout status deployment/${RELEASE_NAME}-worker -n "$NAMESPACE" --timeout=5m

    # Check pod status
    print_info "Current pod status:"
    kubectl get pods -n "$NAMESPACE" -o wide

    # Run health checks
    print_info "Running health checks..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    bash "$SCRIPT_DIR/health-check.sh" || print_warning "Health checks failed - please investigate"

    print_success "Rollback verification complete"
}

# Function to show rollback info
show_rollback_info() {
    local new_rev
    new_rev=$(get_current_revision)

    echo ""
    print_success "Rollback completed successfully!"
    echo ""
    print_info "Current revision: $new_rev"
    echo ""

    # Show current status
    helm status "$RELEASE_NAME" -n "$NAMESPACE"

    echo ""
    print_info "To check the rollback:"
    echo "  helm history $RELEASE_NAME -n $NAMESPACE"
    echo "  kubectl get pods -n $NAMESPACE"
    echo ""
}

# Function to emergency rollback (skip confirmations)
emergency_rollback() {
    print_warning "EMERGENCY ROLLBACK MODE - Skipping confirmations"

    local current_rev
    current_rev=$(get_current_revision)

    if [ -z "$REVISION" ]; then
        print_info "Rolling back from revision $current_rev to previous..."
        perform_rollback ""
    else
        print_info "Rolling back from revision $current_rev to revision $REVISION..."
        perform_rollback "$REVISION"
    fi

    verify_rollback
    show_rollback_info
}

# Main execution
main() {
    print_info "Starting HoloLoom rollback..."
    print_info "Namespace: $NAMESPACE"
    print_info "Release: $RELEASE_NAME"
    echo ""

    check_prerequisites
    show_history

    local current_rev
    current_rev=$(get_current_revision)

    # Emergency mode
    if [ "${EMERGENCY:-false}" = "true" ]; then
        emergency_rollback
        exit 0
    fi

    # Determine target revision
    local target_rev
    if [ -n "$REVISION" ]; then
        target_rev="$REVISION"
    else
        # Calculate previous revision
        target_rev=$((current_rev - 1))
        if [ $target_rev -lt 1 ]; then
            print_error "No previous revision to rollback to"
            exit 1
        fi
    fi

    # Confirm and execute rollback
    confirm_rollback "$current_rev" "$target_rev"
    perform_rollback "$target_rev"
    verify_rollback
    show_rollback_info
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -r|--release)
            RELEASE_NAME="$2"
            shift 2
            ;;
        --revision)
            REVISION="$2"
            shift 2
            ;;
        --emergency)
            EMERGENCY=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --namespace NAME    Kubernetes namespace (default: hololoom)"
            echo "  -r, --release NAME      Helm release name (default: hololoom)"
            echo "  --revision NUMBER       Rollback to specific revision"
            echo "  --emergency             Skip confirmations (use in emergencies)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Rollback to previous revision"
            echo "  $0 --revision 5                       # Rollback to revision 5"
            echo "  $0 --emergency --revision 3           # Emergency rollback to revision 3"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main

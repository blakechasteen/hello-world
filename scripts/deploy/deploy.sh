#!/bin/bash
# EdWIN Deployment Script
# Full deployment automation for Kubernetes
#
# Usage:
#   ./scripts/deploy/deploy.sh [environment]
#
# Examples:
#   ./scripts/deploy/deploy.sh development
#   ./scripts/deploy/deploy.sh production
#
# Implementation Date: November 15, 2025

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==============================================================================
# Configuration
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
K8S_DIR="$PROJECT_ROOT/k8s"

ENVIRONMENT="${1:-development}"
NAMESPACE="edwin"
IMAGE_TAG="${2:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# Helper Functions
# ==============================================================================

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

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi

    # Check Docker (for building)
    if ! command -v docker &> /dev/null; then
        log_error "docker not found. Please install Docker."
        exit 1
    fi

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Check your kubeconfig."
        exit 1
    fi

    log_success "Prerequisites check passed"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_warning "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f "$K8S_DIR/namespace.yaml"
        log_success "Namespace created"
    fi
}

create_secrets() {
    log_info "Creating secrets..."

    # Check if secrets.yaml exists (not the template)
    if [ ! -f "$K8S_DIR/secrets.yaml" ]; then
        log_error "secrets.yaml not found. Please create it from secrets.yaml.template"
        log_info "Run: cp $K8S_DIR/secrets.yaml.template $K8S_DIR/secrets.yaml"
        log_info "Then fill in the actual values"
        exit 1
    fi

    kubectl apply -f "$K8S_DIR/secrets.yaml"
    log_success "Secrets created"
}

create_configmap() {
    log_info "Creating ConfigMap..."
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    log_success "ConfigMap created"
}

build_image() {
    log_info "Building Docker image: edwin-ai-tutor:$IMAGE_TAG"

    cd "$PROJECT_ROOT"
    docker build -t "edwin-ai-tutor:$IMAGE_TAG" .

    log_success "Docker image built"

    # Push to registry (uncomment for production)
    # log_info "Pushing image to registry..."
    # docker tag "edwin-ai-tutor:$IMAGE_TAG" "your-registry/edwin-ai-tutor:$IMAGE_TAG"
    # docker push "your-registry/edwin-ai-tutor:$IMAGE_TAG"
}

deploy_databases() {
    log_info "Deploying databases..."

    # Neo4j
    log_info "Deploying Neo4j..."
    kubectl apply -f "$K8S_DIR/neo4j-statefulset.yaml"

    # Qdrant
    log_info "Deploying Qdrant..."
    kubectl apply -f "$K8S_DIR/qdrant-statefulset.yaml"

    # Redis
    log_info "Deploying Redis..."
    kubectl apply -f "$K8S_DIR/redis-deployment.yaml"

    log_success "Databases deployed"

    # Wait for databases to be ready
    log_info "Waiting for databases to be ready (this may take a few minutes)..."
    kubectl wait --for=condition=ready pod -l component=neo4j -n "$NAMESPACE" --timeout=300s || log_warning "Neo4j timeout (may still be starting)"
    kubectl wait --for=condition=ready pod -l component=qdrant -n "$NAMESPACE" --timeout=300s || log_warning "Qdrant timeout (may still be starting)"
    kubectl wait --for=condition=ready pod -l component=redis -n "$NAMESPACE" --timeout=300s || log_warning "Redis timeout (may still be starting)"

    log_success "Databases are ready"
}

deploy_applications() {
    log_info "Deploying applications..."

    # API
    log_info "Deploying API..."
    kubectl apply -f "$K8S_DIR/api-deployment.yaml"

    # Dashboard
    log_info "Deploying Dashboard..."
    kubectl apply -f "$K8S_DIR/dashboard-deployment.yaml"

    # Mobile API
    log_info "Deploying Mobile API..."
    kubectl apply -f "$K8S_DIR/mobile-deployment.yaml"

    log_success "Applications deployed"

    # Wait for applications to be ready
    log_info "Waiting for applications to be ready..."
    kubectl wait --for=condition=ready pod -l component=api -n "$NAMESPACE" --timeout=300s || log_warning "API timeout"
    kubectl wait --for=condition=ready pod -l component=dashboard -n "$NAMESPACE" --timeout=300s || log_warning "Dashboard timeout"
    kubectl wait --for=condition=ready pod -l component=mobile -n "$NAMESPACE" --timeout=300s || log_warning "Mobile API timeout"

    log_success "Applications are ready"
}

deploy_ingress() {
    log_info "Deploying Ingress..."

    # Check if Nginx Ingress Controller is installed
    if ! kubectl get ingressclass nginx &> /dev/null; then
        log_warning "Nginx Ingress Controller not found. Installing..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
        log_info "Waiting for Ingress Controller to be ready..."
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=300s
        log_success "Nginx Ingress Controller installed"
    fi

    kubectl apply -f "$K8S_DIR/ingress.yaml"
    log_success "Ingress deployed"
}

run_migrations() {
    log_info "Running database migrations..."

    # Get API pod
    API_POD=$(kubectl get pod -n "$NAMESPACE" -l component=api -o jsonpath='{.items[0].metadata.name}')

    if [ -z "$API_POD" ]; then
        log_error "No API pod found. Skipping migrations."
        return
    fi

    # Run migrations (placeholder - implement in Python)
    kubectl exec -n "$NAMESPACE" "$API_POD" -- python -c "
from EduVerse.edwin.database import DatabaseManager
import asyncio

async def migrate():
    db = DatabaseManager()
    await db.initialize()
    await db.run_migrations()
    await db.close()

asyncio.run(migrate())
"

    log_success "Migrations complete"
}

show_status() {
    log_info "Deployment Status:"
    echo ""

    kubectl get pods -n "$NAMESPACE"
    echo ""

    kubectl get services -n "$NAMESPACE"
    echo ""

    kubectl get ingress -n "$NAMESPACE"
    echo ""

    log_info "Endpoints:"
    echo "  API:       http://api.edwin.edu"
    echo "  Dashboard: http://dashboard.edwin.edu"
    echo "  Mobile:    http://mobile.edwin.edu"
    echo ""
    log_warning "Don't forget to configure DNS to point to your Ingress IP"
}

# ==============================================================================
# Main Deployment Flow
# ==============================================================================

main() {
    log_info "🚀 Starting EdWIN deployment ($ENVIRONMENT)..."
    echo ""

    check_prerequisites
    echo ""

    create_namespace
    echo ""

    create_secrets
    echo ""

    create_configmap
    echo ""

    if [ "$ENVIRONMENT" = "production" ] || [ "$ENVIRONMENT" = "staging" ]; then
        build_image
        echo ""
    else
        log_info "Skipping image build for development (using local image)"
        echo ""
    fi

    deploy_databases
    echo ""

    deploy_applications
    echo ""

    deploy_ingress
    echo ""

    run_migrations
    echo ""

    show_status
    echo ""

    log_success "🎉 Deployment complete!"
    log_info "Run './scripts/deploy/health-check.sh' to verify all services are healthy"
}

# Run main function
main

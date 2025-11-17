#!/bin/bash
# HoloLoom Kubernetes Deployment Script
# This script automates the deployment of HoloLoom to a Kubernetes cluster

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HELM_CHART_DIR="$PROJECT_ROOT/kubernetes/helm/hololoom"
NAMESPACE="${NAMESPACE:-hololoom}"
RELEASE_NAME="${RELEASE_NAME:-hololoom}"
VALUES_FILE="${VALUES_FILE:-$HELM_CHART_DIR/values.yaml}"
ENVIRONMENT="${ENVIRONMENT:-production}"

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

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install kubectl."
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        print_error "helm is not installed. Please install Helm 3."
        exit 1
    fi

    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi

    # Check Docker (optional)
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Image building will be skipped."
    fi

    print_success "Prerequisites check passed"
}

# Function to create namespace
create_namespace() {
    print_info "Creating namespace '$NAMESPACE'..."

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_warning "Namespace '$NAMESPACE' already exists"
    else
        kubectl create namespace "$NAMESPACE"
        kubectl label namespace "$NAMESPACE" environment="$ENVIRONMENT"
        print_success "Namespace '$NAMESPACE' created"
    fi
}

# Function to validate secrets
validate_secrets() {
    print_info "Validating secrets configuration..."

    # Check for default passwords
    if grep -q "change-me-in-production" "$VALUES_FILE"; then
        print_error "Default passwords found in values.yaml!"
        print_error "Please update all 'change-me-in-production' values before deploying."
        if [ "$ENVIRONMENT" = "production" ]; then
            exit 1
        else
            print_warning "Continuing with default passwords (non-production environment)"
        fi
    fi

    print_success "Secrets validation complete"
}

# Function to build Docker images
build_images() {
    if [ "${BUILD_IMAGES:-false}" = "true" ]; then
        print_info "Building Docker images..."

        cd "$PROJECT_ROOT"

        # Build API image
        print_info "Building API image..."
        docker build -f kubernetes/docker/Dockerfile.api -t hololoom/api:${IMAGE_TAG:-latest} .

        # Build Worker image
        print_info "Building Worker image..."
        docker build -f kubernetes/docker/Dockerfile.worker -t hololoom/worker:${IMAGE_TAG:-latest} .

        # Push images if registry is specified
        if [ -n "${DOCKER_REGISTRY:-}" ]; then
            print_info "Pushing images to registry..."
            docker tag hololoom/api:${IMAGE_TAG:-latest} ${DOCKER_REGISTRY}/hololoom/api:${IMAGE_TAG:-latest}
            docker tag hololoom/worker:${IMAGE_TAG:-latest} ${DOCKER_REGISTRY}/hololoom/worker:${IMAGE_TAG:-latest}
            docker push ${DOCKER_REGISTRY}/hololoom/api:${IMAGE_TAG:-latest}
            docker push ${DOCKER_REGISTRY}/hololoom/worker:${IMAGE_TAG:-latest}
            print_success "Images pushed to registry"
        fi

        print_success "Docker images built"
    else
        print_info "Skipping image build (set BUILD_IMAGES=true to build)"
    fi
}

# Function to install or upgrade Helm chart
deploy_helm_chart() {
    print_info "Deploying Helm chart..."

    cd "$HELM_CHART_DIR"

    # Lint the chart
    print_info "Linting Helm chart..."
    if helm lint .; then
        print_success "Helm chart linting passed"
    else
        print_error "Helm chart linting failed"
        exit 1
    fi

    # Template and show what will be deployed (dry-run)
    if [ "${DRY_RUN:-false}" = "true" ]; then
        print_info "Running dry-run..."
        helm upgrade --install "$RELEASE_NAME" . \
            --namespace "$NAMESPACE" \
            --values "$VALUES_FILE" \
            --dry-run --debug
        print_info "Dry-run complete. Exiting."
        exit 0
    fi

    # Deploy the chart
    print_info "Installing/upgrading release '$RELEASE_NAME'..."
    helm upgrade --install "$RELEASE_NAME" . \
        --namespace "$NAMESPACE" \
        --values "$VALUES_FILE" \
        --create-namespace \
        --wait \
        --timeout 10m \
        ${HELM_ARGS:-}

    print_success "Helm chart deployed successfully"
}

# Function to wait for deployments
wait_for_deployments() {
    print_info "Waiting for deployments to be ready..."

    # Wait for API deployment
    print_info "Waiting for API deployment..."
    kubectl rollout status deployment/${RELEASE_NAME}-api -n "$NAMESPACE" --timeout=5m

    # Wait for Worker deployment
    print_info "Waiting for Worker deployment..."
    kubectl rollout status deployment/${RELEASE_NAME}-worker -n "$NAMESPACE" --timeout=5m

    # Wait for Neo4j StatefulSet
    if kubectl get statefulset ${RELEASE_NAME}-neo4j -n "$NAMESPACE" &> /dev/null; then
        print_info "Waiting for Neo4j StatefulSet..."
        kubectl rollout status statefulset/${RELEASE_NAME}-neo4j -n "$NAMESPACE" --timeout=10m
    fi

    # Wait for Qdrant StatefulSet
    if kubectl get statefulset ${RELEASE_NAME}-qdrant -n "$NAMESPACE" &> /dev/null; then
        print_info "Waiting for Qdrant StatefulSet..."
        kubectl rollout status statefulset/${RELEASE_NAME}-qdrant -n "$NAMESPACE" --timeout=5m
    fi

    print_success "All deployments are ready"
}

# Function to verify deployment
verify_deployment() {
    print_info "Verifying deployment..."

    # Check pod status
    print_info "Pod status:"
    kubectl get pods -n "$NAMESPACE" -o wide

    # Check service status
    print_info "Service status:"
    kubectl get svc -n "$NAMESPACE"

    # Check ingress status
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        print_info "Ingress status:"
        kubectl get ingress -n "$NAMESPACE"
    fi

    # Run health checks
    print_info "Running health checks..."
    bash "$SCRIPT_DIR/health-check.sh"

    print_success "Deployment verification complete"
}

# Function to show deployment info
show_deployment_info() {
    print_info "Deployment Information:"
    echo ""
    echo "Release Name: $RELEASE_NAME"
    echo "Namespace: $NAMESPACE"
    echo "Environment: $ENVIRONMENT"
    echo ""

    # Get release status
    helm status "$RELEASE_NAME" -n "$NAMESPACE"

    print_info "To access the application:"
    echo "1. API Gateway:"
    echo "   kubectl port-forward -n $NAMESPACE svc/${RELEASE_NAME}-api 8000:8000"
    echo "   Then visit: http://localhost:8000"
    echo ""
    echo "2. Grafana Dashboard:"
    echo "   kubectl port-forward -n $NAMESPACE svc/${RELEASE_NAME}-grafana 3000:3000"
    echo "   Then visit: http://localhost:3000"
    echo ""
    echo "3. View logs:"
    echo "   kubectl logs -f -n $NAMESPACE -l app.kubernetes.io/component=api"
    echo "   kubectl logs -f -n $NAMESPACE -l app.kubernetes.io/component=worker"
    echo ""

    print_success "Deployment complete!"
}

# Main execution
main() {
    print_info "Starting HoloLoom deployment..."
    print_info "Environment: $ENVIRONMENT"
    print_info "Namespace: $NAMESPACE"
    print_info "Release: $RELEASE_NAME"
    echo ""

    check_prerequisites
    create_namespace
    validate_secrets
    build_images
    deploy_helm_chart
    wait_for_deployments
    verify_deployment
    show_deployment_info
}

# Run main function
main "$@"

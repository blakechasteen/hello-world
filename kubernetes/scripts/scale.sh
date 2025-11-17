#!/bin/bash
# HoloLoom Kubernetes Scaling Script
# This script manages scaling of HoloLoom components

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
COMPONENT="${COMPONENT:-}"
REPLICAS="${REPLICAS:-}"

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
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi

    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
}

# Function to show current status
show_current_status() {
    print_info "Current deployment status in namespace '$NAMESPACE':"
    echo ""

    # API deployment
    if kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" &> /dev/null; then
        local api_replicas
        api_replicas=$(kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        echo "API Gateway:    $api_replicas replicas"
    fi

    # Worker deployment
    if kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" &> /dev/null; then
        local worker_replicas
        worker_replicas=$(kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
        echo "Workers:        $worker_replicas replicas"
    fi

    # HPA status
    if kubectl get hpa -n "$NAMESPACE" &> /dev/null 2>&1; then
        echo ""
        print_info "HorizontalPodAutoscaler status:"
        kubectl get hpa -n "$NAMESPACE"
    fi

    echo ""
}

# Function to scale component
scale_component() {
    local component=$1
    local replicas=$2

    case $component in
        api)
            local deployment="${RELEASE_NAME}-api"
            ;;
        worker|workers)
            local deployment="${RELEASE_NAME}-worker"
            ;;
        *)
            print_error "Unknown component: $component"
            print_info "Valid components: api, worker"
            exit 1
            ;;
    esac

    print_info "Scaling $component to $replicas replicas..."

    # Check if HPA is enabled
    if kubectl get hpa "$deployment" -n "$NAMESPACE" &> /dev/null 2>&1; then
        print_warning "HorizontalPodAutoscaler is enabled for $component"
        read -p "Scaling manually will conflict with HPA. Continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
            print_info "Scaling cancelled"
            exit 0
        fi
        print_warning "Consider disabling HPA or adjusting min/max replicas instead"
    fi

    # Perform scaling
    kubectl scale deployment "$deployment" -n "$NAMESPACE" --replicas="$replicas"

    print_info "Waiting for scaling to complete..."
    kubectl rollout status deployment/"$deployment" -n "$NAMESPACE" --timeout=5m

    print_success "Successfully scaled $component to $replicas replicas"
}

# Function to enable/disable HPA
manage_hpa() {
    local action=$1
    local component=$2

    case $component in
        api)
            local hpa="${RELEASE_NAME}-api"
            ;;
        worker|workers)
            local hpa="${RELEASE_NAME}-worker"
            ;;
        *)
            print_error "Unknown component: $component"
            exit 1
            ;;
    esac

    if [ "$action" = "enable" ]; then
        print_info "Enabling HPA for $component..."
        # This would typically be done through Helm values update
        print_info "To enable HPA, update your values.yaml:"
        echo "  ${component}.autoscaling.enabled: true"
        echo "Then run: helm upgrade $RELEASE_NAME /path/to/chart -n $NAMESPACE"
    elif [ "$action" = "disable" ]; then
        print_info "Disabling HPA for $component..."
        if kubectl get hpa "$hpa" -n "$NAMESPACE" &> /dev/null 2>&1; then
            kubectl delete hpa "$hpa" -n "$NAMESPACE"
            print_success "HPA disabled for $component"
        else
            print_warning "HPA not found for $component"
        fi
    fi
}

# Function to auto-scale based on metrics
auto_scale_recommendation() {
    print_info "Analyzing metrics for scaling recommendations..."

    # Get current metrics
    local api_cpu worker_cpu worker_queue

    # API CPU usage
    if kubectl top pod -n "$NAMESPACE" -l app.kubernetes.io/component=api &> /dev/null; then
        api_cpu=$(kubectl top pod -n "$NAMESPACE" -l app.kubernetes.io/component=api --no-headers | \
                  awk '{sum+=$2} END {print sum}' | sed 's/m//')
        echo "API CPU usage: ${api_cpu}m"
    fi

    # Worker CPU usage
    if kubectl top pod -n "$NAMESPACE" -l app.kubernetes.io/component=worker &> /dev/null; then
        worker_cpu=$(kubectl top pod -n "$NAMESPACE" -l app.kubernetes.io/component=worker --no-headers | \
                     awk '{sum+=$2} END {print sum}' | sed 's/m//')
        echo "Worker CPU usage: ${worker_cpu}m"
    fi

    # TODO: Check queue depth from RabbitMQ metrics
    # worker_queue=$(curl -s http://rabbitmq:15672/api/queues | jq '.[] | select(.name | startswith("hololoom")) | .messages' | paste -sd+ | bc)

    echo ""
    print_info "Recommendations:"
    echo "- If CPU > 80%: Scale up"
    echo "- If CPU < 20% and idle: Scale down"
    echo "- If queue depth > 1000: Scale up workers"
    echo ""
}

# Function to perform load test scaling
load_test_scale() {
    print_info "Scaling for load test scenario..."

    print_info "Scaling API to 5 replicas..."
    scale_component "api" "5"

    print_info "Scaling workers to 8 replicas..."
    scale_component "worker" "8"

    print_success "Load test scaling complete"
}

# Function to scale down
scale_down_all() {
    print_warning "Scaling down all components to minimum..."

    read -p "This will scale down to minimum replicas. Continue? (yes/no): " -r
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        print_info "Scale down cancelled"
        exit 0
    fi

    print_info "Scaling API to 1 replica..."
    scale_component "api" "1"

    print_info "Scaling workers to 1 replica..."
    scale_component "worker" "1"

    print_success "Scale down complete"
}

# Function to show help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
  status                    Show current deployment status
  scale COMPONENT REPLICAS  Scale component to specified replicas
  up COMPONENT             Scale component up by 1
  down COMPONENT           Scale component down by 1
  hpa enable COMPONENT     Enable HPA for component
  hpa disable COMPONENT    Disable HPA for component
  recommend                Show scaling recommendations based on metrics
  load-test                Scale up for load testing
  scale-down               Scale down all components to minimum

Components:
  api       API Gateway
  worker    Worker pool

Options:
  -n, --namespace NAME     Kubernetes namespace (default: hololoom)
  -r, --release NAME       Helm release name (default: hololoom)
  -h, --help               Show this help message

Examples:
  $0 status                          # Show current status
  $0 scale worker 5                  # Scale workers to 5 replicas
  $0 up api                          # Scale API up by 1
  $0 down worker                     # Scale worker down by 1
  $0 hpa enable worker               # Enable HPA for workers
  $0 recommend                       # Get scaling recommendations
  $0 load-test                       # Scale for load testing
  $0 scale-down                      # Scale down to minimum

EOF
}

# Main execution
main() {
    check_prerequisites

    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi

    local command=$1
    shift || true

    case $command in
        status)
            show_current_status
            ;;
        scale)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 scale COMPONENT REPLICAS"
                exit 1
            fi
            scale_component "$1" "$2"
            ;;
        up)
            if [ $# -lt 1 ]; then
                print_error "Usage: $0 up COMPONENT"
                exit 1
            fi
            local current
            case $1 in
                api)
                    current=$(kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
                    ;;
                worker)
                    current=$(kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
                    ;;
                *)
                    print_error "Unknown component: $1"
                    exit 1
                    ;;
            esac
            scale_component "$1" "$((current + 1))"
            ;;
        down)
            if [ $# -lt 1 ]; then
                print_error "Usage: $0 down COMPONENT"
                exit 1
            fi
            local current
            case $1 in
                api)
                    current=$(kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
                    ;;
                worker)
                    current=$(kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')
                    ;;
                *)
                    print_error "Unknown component: $1"
                    exit 1
                    ;;
            esac
            if [ $current -le 1 ]; then
                print_error "Cannot scale below 1 replica"
                exit 1
            fi
            scale_component "$1" "$((current - 1))"
            ;;
        hpa)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 hpa enable|disable COMPONENT"
                exit 1
            fi
            manage_hpa "$1" "$2"
            ;;
        recommend)
            auto_scale_recommendation
            ;;
        load-test)
            load_test_scale
            ;;
        scale-down)
            scale_down_all
            ;;
        -h|--help|help)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Parse global options
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
        *)
            break
            ;;
    esac
done

# Run main function with remaining arguments
main "$@"

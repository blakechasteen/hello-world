#!/bin/bash
# HoloLoom Kubernetes Health Check Script
# This script verifies the health of all HoloLoom components

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
VERBOSE="${VERBOSE:-false}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    ((PASSED_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
    ((WARNING_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
    ((FAILED_CHECKS++))
    ((TOTAL_CHECKS++))
}

print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
}

# Function to check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed"
        exit 1
    fi
}

# Function to check namespace exists
check_namespace() {
    print_header "Checking Namespace"

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        print_success "Namespace '$NAMESPACE' exists"
    else
        print_error "Namespace '$NAMESPACE' not found"
        return 1
    fi
}

# Function to check pod status
check_pods() {
    print_header "Checking Pod Status"

    local all_pods_ready=true

    # Get all pods in namespace
    local pods
    pods=$(kubectl get pods -n "$NAMESPACE" -o json)

    # Check each deployment
    local components=("api" "worker" "neo4j" "qdrant" "redis-master" "rabbitmq" "prometheus" "grafana")

    for component in "${components[@]}"; do
        local pod_count
        pod_count=$(echo "$pods" | jq -r "[.items[] | select(.metadata.labels.\"app.kubernetes.io/component\" == \"$component\")] | length")

        if [ "$pod_count" -eq 0 ]; then
            if [ "$VERBOSE" = "true" ]; then
                print_warning "$component: No pods found (may be disabled)"
            fi
            continue
        fi

        # Check pod status
        local ready_pods
        ready_pods=$(echo "$pods" | jq -r "[.items[] | select(.metadata.labels.\"app.kubernetes.io/component\" == \"$component\") | select(.status.phase == \"Running\") | select(all(.status.containerStatuses[]; .ready == true))] | length")

        if [ "$ready_pods" -eq "$pod_count" ]; then
            print_success "$component: All $pod_count pod(s) are ready"
        elif [ "$ready_pods" -gt 0 ]; then
            print_warning "$component: Only $ready_pods/$pod_count pod(s) are ready"
            all_pods_ready=false
        else
            print_error "$component: No pods are ready (0/$pod_count)"
            all_pods_ready=false
        fi
    done

    return 0
}

# Function to check deployments
check_deployments() {
    print_header "Checking Deployments"

    # API Deployment
    if kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" &> /dev/null; then
        local api_ready
        api_ready=$(kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local api_desired
        api_desired=$(kubectl get deployment ${RELEASE_NAME}-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [ "${api_ready:-0}" -eq "${api_desired:-0}" ] && [ "${api_desired:-0}" -gt 0 ]; then
            print_success "API Deployment: $api_ready/$api_desired replicas ready"
        else
            print_error "API Deployment: Only ${api_ready:-0}/$api_desired replicas ready"
        fi
    else
        print_warning "API Deployment not found"
    fi

    # Worker Deployment
    if kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" &> /dev/null; then
        local worker_ready
        worker_ready=$(kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local worker_desired
        worker_desired=$(kubectl get deployment ${RELEASE_NAME}-worker -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [ "${worker_ready:-0}" -eq "${worker_desired:-0}" ] && [ "${worker_desired:-0}" -gt 0 ]; then
            print_success "Worker Deployment: $worker_ready/$worker_desired replicas ready"
        else
            print_error "Worker Deployment: Only ${worker_ready:-0}/$worker_desired replicas ready"
        fi
    else
        print_warning "Worker Deployment not found"
    fi
}

# Function to check StatefulSets
check_statefulsets() {
    print_header "Checking StatefulSets"

    # Neo4j
    if kubectl get statefulset ${RELEASE_NAME}-neo4j -n "$NAMESPACE" &> /dev/null 2>&1; then
        local neo4j_ready
        neo4j_ready=$(kubectl get statefulset ${RELEASE_NAME}-neo4j -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local neo4j_desired
        neo4j_desired=$(kubectl get statefulset ${RELEASE_NAME}-neo4j -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [ "${neo4j_ready:-0}" -eq "${neo4j_desired:-0}" ] && [ "${neo4j_desired:-0}" -gt 0 ]; then
            print_success "Neo4j StatefulSet: $neo4j_ready/$neo4j_desired replicas ready"
        else
            print_error "Neo4j StatefulSet: Only ${neo4j_ready:-0}/$neo4j_desired replicas ready"
        fi
    else
        print_warning "Neo4j StatefulSet not found"
    fi

    # Qdrant
    if kubectl get statefulset ${RELEASE_NAME}-qdrant -n "$NAMESPACE" &> /dev/null 2>&1; then
        local qdrant_ready
        qdrant_ready=$(kubectl get statefulset ${RELEASE_NAME}-qdrant -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local qdrant_desired
        qdrant_desired=$(kubectl get statefulset ${RELEASE_NAME}-qdrant -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [ "${qdrant_ready:-0}" -eq "${qdrant_desired:-0}" ] && [ "${qdrant_desired:-0}" -gt 0 ]; then
            print_success "Qdrant StatefulSet: $qdrant_ready/$qdrant_desired replicas ready"
        else
            print_error "Qdrant StatefulSet: Only ${qdrant_ready:-0}/$qdrant_desired replicas ready"
        fi
    else
        print_warning "Qdrant StatefulSet not found"
    fi

    # Redis
    if kubectl get statefulset ${RELEASE_NAME}-redis-master -n "$NAMESPACE" &> /dev/null 2>&1; then
        local redis_ready
        redis_ready=$(kubectl get statefulset ${RELEASE_NAME}-redis-master -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local redis_desired
        redis_desired=$(kubectl get statefulset ${RELEASE_NAME}-redis-master -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [ "${redis_ready:-0}" -eq "${redis_desired:-0}" ] && [ "${redis_desired:-0}" -gt 0 ]; then
            print_success "Redis StatefulSet: $redis_ready/$redis_desired replicas ready"
        else
            print_error "Redis StatefulSet: Only ${redis_ready:-0}/$redis_desired replicas ready"
        fi
    else
        print_warning "Redis StatefulSet not found"
    fi

    # RabbitMQ
    if kubectl get statefulset ${RELEASE_NAME}-rabbitmq -n "$NAMESPACE" &> /dev/null 2>&1; then
        local rabbitmq_ready
        rabbitmq_ready=$(kubectl get statefulset ${RELEASE_NAME}-rabbitmq -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
        local rabbitmq_desired
        rabbitmq_desired=$(kubectl get statefulset ${RELEASE_NAME}-rabbitmq -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

        if [ "${rabbitmq_ready:-0}" -eq "${rabbitmq_desired:-0}" ] && [ "${rabbitmq_desired:-0}" -gt 0 ]; then
            print_success "RabbitMQ StatefulSet: $rabbitmq_ready/$rabbitmq_desired replicas ready"
        else
            print_error "RabbitMQ StatefulSet: Only ${rabbitmq_ready:-0}/$rabbitmq_desired replicas ready"
        fi
    else
        print_warning "RabbitMQ StatefulSet not found"
    fi
}

# Function to check services
check_services() {
    print_header "Checking Services"

    local services=("api" "neo4j-client" "qdrant-client" "redis-master" "rabbitmq-client")

    for service in "${services[@]}"; do
        if kubectl get service ${RELEASE_NAME}-${service} -n "$NAMESPACE" &> /dev/null 2>&1; then
            local endpoints
            endpoints=$(kubectl get endpoints ${RELEASE_NAME}-${service} -n "$NAMESPACE" -o jsonpath='{.subsets[*].addresses[*].ip}' | wc -w)

            if [ "$endpoints" -gt 0 ]; then
                print_success "Service ${service}: $endpoints endpoint(s)"
            else
                print_error "Service ${service}: No endpoints available"
            fi
        else
            if [ "$VERBOSE" = "true" ]; then
                print_warning "Service ${service} not found"
            fi
        fi
    done
}

# Function to check PVCs
check_pvcs() {
    print_header "Checking Persistent Volume Claims"

    local pvcs
    pvcs=$(kubectl get pvc -n "$NAMESPACE" -o json 2>/dev/null)

    if [ -n "$pvcs" ]; then
        local pvc_count
        pvc_count=$(echo "$pvcs" | jq -r '.items | length')

        if [ "$pvc_count" -gt 0 ]; then
            local bound_count
            bound_count=$(echo "$pvcs" | jq -r '[.items[] | select(.status.phase == "Bound")] | length')

            if [ "$bound_count" -eq "$pvc_count" ]; then
                print_success "PVCs: All $pvc_count claim(s) are bound"
            else
                print_error "PVCs: Only $bound_count/$pvc_count claim(s) are bound"
            fi
        else
            print_warning "No PVCs found"
        fi
    else
        print_warning "No PVCs found"
    fi
}

# Function to check HPA
check_hpa() {
    print_header "Checking HorizontalPodAutoscalers"

    if kubectl get hpa -n "$NAMESPACE" &> /dev/null 2>&1; then
        local hpa_list
        hpa_list=$(kubectl get hpa -n "$NAMESPACE" -o json)

        local hpa_count
        hpa_count=$(echo "$hpa_list" | jq -r '.items | length')

        if [ "$hpa_count" -gt 0 ]; then
            echo "$hpa_list" | jq -r '.items[] | "\(.metadata.name): \(.status.currentReplicas)/\(.spec.minReplicas)-\(.spec.maxReplicas) replicas"' | while read -r line; do
                print_success "HPA $line"
            done
        else
            print_warning "No HPAs found"
        fi
    else
        print_warning "No HPAs configured"
    fi
}

# Function to check ingress
check_ingress() {
    print_header "Checking Ingress"

    if kubectl get ingress -n "$NAMESPACE" &> /dev/null 2>&1; then
        local ingress_count
        ingress_count=$(kubectl get ingress -n "$NAMESPACE" -o json | jq -r '.items | length')

        if [ "$ingress_count" -gt 0 ]; then
            print_success "Ingress: $ingress_count ingress(es) found"
            if [ "$VERBOSE" = "true" ]; then
                kubectl get ingress -n "$NAMESPACE"
            fi
        else
            print_warning "No ingress found"
        fi
    else
        print_warning "No ingress configured"
    fi
}

# Function to check resource usage
check_resource_usage() {
    print_header "Checking Resource Usage"

    if kubectl top nodes &> /dev/null 2>&1; then
        print_success "Node metrics available"

        if [ "$VERBOSE" = "true" ]; then
            kubectl top nodes
        fi
    else
        print_warning "Metrics server not available - cannot check resource usage"
    fi

    if kubectl top pods -n "$NAMESPACE" &> /dev/null 2>&1; then
        print_success "Pod metrics available"

        if [ "$VERBOSE" = "true" ]; then
            kubectl top pods -n "$NAMESPACE"
        fi
    else
        print_warning "Cannot retrieve pod metrics"
    fi
}

# Function to show summary
show_summary() {
    print_header "Health Check Summary"

    echo ""
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
    echo -e "${YELLOW}Warnings: $WARNING_CHECKS${NC}"
    echo -e "${RED}Failed: $FAILED_CHECKS${NC}"
    echo ""

    if [ "$FAILED_CHECKS" -eq 0 ]; then
        if [ "$WARNING_CHECKS" -eq 0 ]; then
            print_success "All health checks passed! ✨"
            return 0
        else
            echo -e "${YELLOW}Health checks passed with warnings ⚠️${NC}"
            return 0
        fi
    else
        echo -e "${RED}Health checks failed! Please investigate. ❌${NC}"
        return 1
    fi
}

# Main execution
main() {
    echo "============================================"
    echo "HoloLoom Health Check"
    echo "Namespace: $NAMESPACE"
    echo "Release: $RELEASE_NAME"
    echo "============================================"

    check_kubectl
    check_namespace
    check_pods
    check_deployments
    check_statefulsets
    check_services
    check_pvcs
    check_hpa
    check_ingress
    check_resource_usage
    show_summary
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
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --namespace NAME    Kubernetes namespace (default: hololoom)"
            echo "  -r, --release NAME      Helm release name (default: hololoom)"
            echo "  -v, --verbose           Show detailed output"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
exit_code=$?
exit $exit_code

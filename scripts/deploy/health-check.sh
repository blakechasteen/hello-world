#!/bin/bash
# EdWIN Health Check Script
# Check health of all EdWIN services
#
# Usage:
#   ./scripts/deploy/health-check.sh [namespace]
#
# Implementation Date: November 15, 2025

set -e

NAMESPACE="${1:-edwin}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

check_pod_health() {
    local component=$1
    local pods=$(kubectl get pods -n "$NAMESPACE" -l component="$component" -o jsonpath='{.items[*].metadata.name}')

    if [ -z "$pods" ]; then
        log_error "$component: No pods found"
        return 1
    fi

    local all_ready=true
    for pod in $pods; do
        local ready=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}')

        if [ "$ready" = "True" ]; then
            log_success "$component: $pod is ready"
        else
            log_error "$component: $pod is not ready"
            all_ready=false
        fi
    done

    return $([ "$all_ready" = true ])
}

check_service_endpoint() {
    local service=$1
    local port=$2
    local path="${3:-/health}"

    # Get service cluster IP
    local cluster_ip=$(kubectl get service "$service" -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')

    if [ -z "$cluster_ip" ]; then
        log_error "$service: Service not found"
        return 1
    fi

    # Run curl from a pod
    local test_pod=$(kubectl get pod -n "$NAMESPACE" -l component=api -o jsonpath='{.items[0].metadata.name}')

    if [ -z "$test_pod" ]; then
        log_warning "$service: Cannot test endpoint (no test pod available)"
        return 0
    fi

    local http_code=$(kubectl exec -n "$NAMESPACE" "$test_pod" -- curl -s -o /dev/null -w "%{http_code}" "http://$cluster_ip:$port$path" || echo "000")

    if [ "$http_code" = "200" ]; then
        log_success "$service: Endpoint $path is healthy (HTTP $http_code)"
        return 0
    else
        log_error "$service: Endpoint $path returned HTTP $http_code"
        return 1
    fi
}

echo "🏥 EdWIN Health Check"
echo "===================="
echo ""

overall_health=0

# ==============================================================================
# Check Pods
# ==============================================================================

log_info "Checking Pod Health..."
echo ""

components=("neo4j" "qdrant" "redis" "api" "dashboard" "mobile")

for component in "${components[@]}"; do
    check_pod_health "$component" || overall_health=1
done

echo ""

# ==============================================================================
# Check Services
# ==============================================================================

log_info "Checking Service Endpoints..."
echo ""

check_service_endpoint "edwin-api" 8000 "/health" || overall_health=1
check_service_endpoint "edwin-dashboard" 8001 "/health" || overall_health=1
check_service_endpoint "edwin-mobile" 8002 "/health" || overall_health=1

echo ""

# ==============================================================================
# Check Database Health
# ==============================================================================

log_info "Checking Database Health..."
echo ""

# Neo4j
NEO4J_POD=$(kubectl get pod -n "$NAMESPACE" -l component=neo4j -o jsonpath='{.items[0].metadata.name}')
if [ -n "$NEO4J_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$NEO4J_POD" -- cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" &> /dev/null; then
        log_success "Neo4j: Database is accessible"
    else
        log_error "Neo4j: Database is not accessible"
        overall_health=1
    fi
fi

# Qdrant
QDRANT_POD=$(kubectl get pod -n "$NAMESPACE" -l component=qdrant -o jsonpath='{.items[0].metadata.name}')
if [ -n "$QDRANT_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$QDRANT_POD" -- curl -f http://localhost:6333/health &> /dev/null; then
        log_success "Qdrant: Database is accessible"
    else
        log_error "Qdrant: Database is not accessible"
        overall_health=1
    fi
fi

# Redis
REDIS_POD=$(kubectl get pod -n "$NAMESPACE" -l component=redis -o jsonpath='{.items[0].metadata.name}')
if [ -n "$REDIS_POD" ]; then
    if kubectl exec -n "$NAMESPACE" "$REDIS_POD" -- redis-cli ping | grep -q PONG; then
        log_success "Redis: Cache is accessible"
    else
        log_error "Redis: Cache is not accessible"
        overall_health=1
    fi
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================

log_info "Summary:"
echo ""

if [ $overall_health -eq 0 ]; then
    log_success "All systems healthy! 🎉"
    exit 0
else
    log_error "Some systems are unhealthy. Check logs for details."
    echo ""
    log_info "View logs with:"
    echo "  kubectl logs -n $NAMESPACE -l component=api"
    echo "  kubectl logs -n $NAMESPACE -l component=neo4j"
    echo "  kubectl logs -n $NAMESPACE -l component=qdrant"
    exit 1
fi

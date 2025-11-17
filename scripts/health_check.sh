#!/bin/bash
# EdWIN Health Check Script
# Checks health of all services
#
# Usage:
#   ./scripts/health_check.sh [namespace]
#
# Implementation Date: November 17, 2025

set -e

NAMESPACE="${1:-edwin}"
API_URL="${API_URL:-http://localhost:8000}"
DASHBOARD_URL="${DASHBOARD_URL:-http://localhost:8001}"
MOBILE_URL="${MOBILE_URL:-http://localhost:8002}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}EdWIN Health Check${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Check if namespace exists
if ! kubectl get namespace ${NAMESPACE} &> /dev/null; then
    echo -e "${RED}❌ Namespace ${NAMESPACE} does not exist${NC}"
    exit 1
fi

echo -e "\n${GREEN}Checking Kubernetes resources...${NC}"

# Check pods
echo -e "\n${YELLOW}Pods:${NC}"
kubectl get pods -n ${NAMESPACE}

NOT_RUNNING=$(kubectl get pods -n ${NAMESPACE} --field-selector=status.phase!=Running --no-headers 2>/dev/null | wc -l)
if [ "$NOT_RUNNING" -gt 0 ]; then
    echo -e "${RED}❌ $NOT_RUNNING pods not in Running state${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ All pods running${NC}"
fi

# Check deployments
echo -e "\n${YELLOW}Deployments:${NC}"
kubectl get deployments -n ${NAMESPACE}

NOT_READY=$(kubectl get deployments -n ${NAMESPACE} -o json | jq '.items[] | select(.status.replicas != .status.readyReplicas) | .metadata.name' -r)
if [ -n "$NOT_READY" ]; then
    echo -e "${RED}❌ Deployments not ready: $NOT_READY${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ All deployments ready${NC}"
fi

# Check statefulsets
echo -e "\n${YELLOW}StatefulSets:${NC}"
kubectl get statefulsets -n ${NAMESPACE}

NOT_READY_STS=$(kubectl get statefulsets -n ${NAMESPACE} -o json | jq '.items[] | select(.status.replicas != .status.readyReplicas) | .metadata.name' -r)
if [ -n "$NOT_READY_STS" ]; then
    echo -e "${RED}❌ StatefulSets not ready: $NOT_READY_STS${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ All statefulsets ready${NC}"
fi

# Check services
echo -e "\n${YELLOW}Services:${NC}"
kubectl get svc -n ${NAMESPACE}

# Health check API endpoints (if accessible)
echo -e "\n${GREEN}Checking HTTP endpoints...${NC}"

# API
if curl -f -s "${API_URL}/health" > /dev/null 2>&1; then
    RESPONSE=$(curl -s "${API_URL}/health")
    echo -e "${GREEN}✅ API health: ${RESPONSE}${NC}"
else
    echo -e "${YELLOW}⚠️  API health endpoint not accessible (expected if running in cluster)${NC}"
fi

# Dashboard
if curl -f -s "${DASHBOARD_URL}/health" > /dev/null 2>&1; then
    RESPONSE=$(curl -s "${DASHBOARD_URL}/health")
    echo -e "${GREEN}✅ Dashboard health: ${RESPONSE}${NC}"
else
    echo -e "${YELLOW}⚠️  Dashboard health endpoint not accessible${NC}"
fi

# Mobile API
if curl -f -s "${MOBILE_URL}/health" > /dev/null 2>&1; then
    RESPONSE=$(curl -s "${MOBILE_URL}/health")
    echo -e "${GREEN}✅ Mobile API health: ${RESPONSE}${NC}"
else
    echo -e "${YELLOW}⚠️  Mobile API health endpoint not accessible${NC}"
fi

# Check database connectivity (via API)
if curl -f -s "${API_URL}/health/db" > /dev/null 2>&1; then
    RESPONSE=$(curl -s "${API_URL}/health/db")
    echo -e "${GREEN}✅ Database health: ${RESPONSE}${NC}"
else
    echo -e "${YELLOW}⚠️  Database health endpoint not accessible${NC}"
fi

# Summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All health checks passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ $ERRORS health checks failed${NC}"
    exit 1
fi

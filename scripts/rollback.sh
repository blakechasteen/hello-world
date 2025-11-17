#!/bin/bash
# EdWIN Rollback Script
# Rollback to previous deployment
#
# Usage:
#   ./scripts/rollback.sh [environment]
#   environment: staging | production (default: staging)
#
# Implementation Date: November 17, 2025

set -e

ENVIRONMENT="${1:-staging}"
NAMESPACE="edwin"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}EdWIN Rollback - ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Confirmation
if [[ "$ENVIRONMENT" == "production" ]]; then
    echo -e "${RED}⚠️  WARNING: Rolling back PRODUCTION${NC}"
    read -p "Are you sure? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Rollback cancelled"
        exit 0
    fi
fi

# Rollback each deployment
echo -e "\n${YELLOW}Rolling back API...${NC}"
kubectl rollout undo deployment/edwin-api -n ${NAMESPACE}

echo -e "\n${YELLOW}Rolling back Dashboard...${NC}"
kubectl rollout undo deployment/edwin-dashboard -n ${NAMESPACE}

echo -e "\n${YELLOW}Rolling back Mobile API...${NC}"
kubectl rollout undo deployment/edwin-mobile -n ${NAMESPACE}

# Wait for rollback
echo -e "\n${YELLOW}Waiting for rollback to complete...${NC}"
kubectl rollout status deployment/edwin-api -n ${NAMESPACE}
kubectl rollout status deployment/edwin-dashboard -n ${NAMESPACE}
kubectl rollout status deployment/edwin-mobile -n ${NAMESPACE}

# Health check
echo -e "\n${YELLOW}Running health check...${NC}"
./scripts/health_check.sh

echo -e "\n${GREEN}✅ Rollback complete!${NC}"
echo ""
echo "Current deployment:"
kubectl get deployments -n ${NAMESPACE}

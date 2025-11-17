#!/bin/bash
# EdWIN Production Deployment Script
# One-command deployment to Kubernetes
#
# Usage:
#   ./scripts/deploy.sh [environment]
#   environment: staging | production (default: staging)
#
# Implementation Date: November 17, 2025

set -e  # Exit on error

# Configuration
ENVIRONMENT="${1:-staging}"
NAMESPACE="edwin"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-ghcr.io/your-org}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}EdWIN Deployment Script - ${ENVIRONMENT}${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Validate environment
if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    echo -e "${RED}❌ Invalid environment: $ENVIRONMENT${NC}"
    echo "Valid options: staging, production"
    exit 1
fi

# Confirmation for production
if [[ "$ENVIRONMENT" == "production" ]]; then
    echo -e "${YELLOW}⚠️  WARNING: Deploying to PRODUCTION${NC}"
    read -p "Are you sure? (yes/no): " -r
    if [[ ! $REPLY =~ ^yes$ ]]; then
        echo "Deployment cancelled"
        exit 0
    fi
fi

# Step 1: Build Docker images
echo -e "\n${GREEN}Step 1: Building Docker images...${NC}"
docker build -t "${DOCKER_REGISTRY}/edwin-api:${IMAGE_TAG}" .
docker build -t "${DOCKER_REGISTRY}/edwin-api:latest" .

# Step 2: Push to registry
echo -e "\n${GREEN}Step 2: Pushing images to registry...${NC}"
docker push "${DOCKER_REGISTRY}/edwin-api:${IMAGE_TAG}"
docker push "${DOCKER_REGISTRY}/edwin-api:latest"

# Step 3: Create namespace if not exists
echo -e "\n${GREEN}Step 3: Creating namespace...${NC}"
kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -

# Step 4: Apply Kubernetes secrets
echo -e "\n${GREEN}Step 4: Applying secrets...${NC}"
if [[ ! -f "k8s/secrets.${ENVIRONMENT}.yaml" ]]; then
    echo -e "${RED}❌ Secrets file not found: k8s/secrets.${ENVIRONMENT}.yaml${NC}"
    echo "Please create secrets file from template: k8s/secrets.yaml.template"
    exit 1
fi
kubectl apply -f "k8s/secrets.${ENVIRONMENT}.yaml"

# Step 5: Apply ConfigMaps
echo -e "\n${GREEN}Step 5: Applying ConfigMaps...${NC}"
kubectl apply -f k8s/configmap.yaml

# Step 6: Deploy databases
echo -e "\n${GREEN}Step 6: Deploying databases...${NC}"
kubectl apply -f k8s/neo4j-statefulset.yaml
kubectl apply -f k8s/qdrant-statefulset.yaml
kubectl apply -f k8s/redis-deployment.yaml

# Wait for databases to be ready
echo "Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l app=neo4j -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=qdrant -n ${NAMESPACE} --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n ${NAMESPACE} --timeout=300s

# Step 7: Deploy services
echo -e "\n${GREEN}Step 7: Deploying services...${NC}"
kubectl apply -f k8s/services.yaml

# Step 8: Deploy applications
echo -e "\n${GREEN}Step 8: Deploying applications...${NC}"
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/mobile-deployment.yaml

# Step 9: Apply autoscaling
echo -e "\n${GREEN}Step 9: Applying autoscaling...${NC}"
kubectl apply -f k8s/hpa.yaml

# Step 10: Apply pod disruption budgets
echo -e "\n${GREEN}Step 10: Applying pod disruption budgets...${NC}"
kubectl apply -f k8s/pdb.yaml

# Step 11: Apply ingress
echo -e "\n${GREEN}Step 11: Configuring ingress...${NC}"
kubectl apply -f k8s/ingress.yaml

# Step 12: Wait for rollout
echo -e "\n${GREEN}Step 12: Waiting for rollout to complete...${NC}"
kubectl rollout status deployment/edwin-api -n ${NAMESPACE}
kubectl rollout status deployment/edwin-dashboard -n ${NAMESPACE}
kubectl rollout status deployment/edwin-mobile -n ${NAMESPACE}

# Step 13: Run smoke tests
echo -e "\n${GREEN}Step 13: Running smoke tests...${NC}"
./scripts/health_check.sh

# Step 14: Display deployment info
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Deployment Summary:"
echo "  Environment: ${ENVIRONMENT}"
echo "  Namespace: ${NAMESPACE}"
echo "  Image Tag: ${IMAGE_TAG}"
echo ""
echo "Pods:"
kubectl get pods -n ${NAMESPACE}
echo ""
echo "Services:"
kubectl get svc -n ${NAMESPACE}
echo ""
echo "Ingress:"
kubectl get ingress -n ${NAMESPACE}
echo ""
echo "To view logs:"
echo "  kubectl logs -f deployment/edwin-api -n ${NAMESPACE}"
echo ""
echo "To rollback:"
echo "  ./scripts/rollback.sh ${ENVIRONMENT}"

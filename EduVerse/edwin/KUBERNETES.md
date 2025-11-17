# EdWIN AI Tutor - Kubernetes Deployment Guide

**Complete Kubernetes deployment and operations guide**

**Implementation Date**: November 15, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Cluster Setup](#cluster-setup)
4. [Deployment Process](#deployment-process)
5. [Monitoring & Logging](#monitoring--logging)
6. [Scaling](#scaling)
7. [Updates & Rollbacks](#updates--rollbacks)
8. [Production Best Practices](#production-best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Overview

EdWIN runs on Kubernetes with the following architecture:

### Namespace: `edwin`

**Services**:
- **edwin-api** (3 replicas): Main API with HPA
- **edwin-dashboard** (2 replicas): Teacher dashboard
- **edwin-mobile** (2 replicas): Mobile API
- **neo4j** (StatefulSet): Knowledge graph database
- **qdrant** (StatefulSet): Vector database
- **redis** (Deployment): Cache layer

**Storage**:
- Neo4j: 20GB persistent volume
- Qdrant: 10GB persistent volume
- Logs: emptyDir (ephemeral)

**Networking**:
- Nginx Ingress Controller
- TLS via cert-manager (optional)
- Internal service mesh

---

## Prerequisites

### Required Tools

```bash
# Check versions
kubectl version --client
# Should be 1.24+

helm version
# Should be 3.0+

# Cluster access
kubectl cluster-info
```

### Cluster Requirements

**Minimum**:
- 3 nodes
- 4 CPU cores per node
- 8 GB RAM per node
- 100 GB disk space per node

**Recommended**:
- 5 nodes
- 8 CPU cores per node
- 16 GB RAM per node
- 200 GB SSD storage per node

### Cloud Provider Setup

**AWS EKS**:
```bash
eksctl create cluster \
  --name edwin-cluster \
  --region us-west-2 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

**Google GKE**:
```bash
gcloud container clusters create edwin-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10
```

**Azure AKS**:
```bash
az aks create \
  --resource-group edwin-rg \
  --name edwin-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10
```

---

## Cluster Setup

### 1. Install Nginx Ingress Controller

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Verify
kubectl get pods -n ingress-nginx
kubectl get svc -n ingress-nginx
```

### 2. Install cert-manager (for TLS)

```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.2/cert-manager.yaml

# Verify
kubectl get pods -n cert-manager
```

### 3. Configure Storage Classes

**Check available storage classes**:
```bash
kubectl get storageclass
```

**Create custom storage class** (if needed):
```yaml
# storage-class.yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # or gce-pd, azure-disk
parameters:
  type: gp3
  fsType: ext4
reclaimPolicy: Retain
allowVolumeExpansion: true
```

Apply:
```bash
kubectl apply -f storage-class.yaml
```

---

## Deployment Process

### Step 1: Create Namespace

```bash
kubectl apply -f k8s/namespace.yaml

# Verify
kubectl get namespace edwin
```

### Step 2: Create Secrets

**Generate secrets**:
```bash
# Generate JWT secret
JWT_SECRET=$(openssl rand -base64 32)
JWT_SECRET_B64=$(echo -n "$JWT_SECRET" | base64)

# Generate Neo4j password
NEO4J_PASSWORD=$(openssl rand -base64 16)
NEO4J_PASSWORD_B64=$(echo -n "neo4j/$NEO4J_PASSWORD" | base64)

echo "JWT_SECRET_KEY: $JWT_SECRET_B64"
echo "NEO4J_PASSWORD: $NEO4J_PASSWORD_B64"
```

**Create secrets.yaml**:
```bash
cp k8s/secrets.yaml.template k8s/secrets.yaml
# Edit k8s/secrets.yaml with generated values
```

**Apply**:
```bash
kubectl apply -f k8s/secrets.yaml

# Verify
kubectl get secret edwin-secrets -n edwin
kubectl describe secret edwin-secrets -n edwin
```

### Step 3: Create ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml

# Verify
kubectl get configmap edwin-config -n edwin
kubectl describe configmap edwin-config -n edwin
```

### Step 4: Deploy Databases

```bash
# Deploy in order (for dependency management)
kubectl apply -f k8s/neo4j-statefulset.yaml
kubectl apply -f k8s/qdrant-statefulset.yaml
kubectl apply -f k8s/redis-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l component=neo4j -n edwin --timeout=300s
kubectl wait --for=condition=ready pod -l component=qdrant -n edwin --timeout=300s
kubectl wait --for=condition=ready pod -l component=redis -n edwin --timeout=300s
```

**Verify databases**:
```bash
# Neo4j
kubectl exec -n edwin neo4j-0 -- cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1"

# Qdrant
kubectl exec -n edwin qdrant-0 -- curl -f http://localhost:6333/health

# Redis
kubectl exec -n edwin $(kubectl get pod -n edwin -l component=redis -o jsonpath='{.items[0].metadata.name}') -- redis-cli ping
```

### Step 5: Deploy Applications

```bash
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/dashboard-deployment.yaml
kubectl apply -f k8s/mobile-deployment.yaml

# Wait for applications
kubectl wait --for=condition=ready pod -l component=api -n edwin --timeout=300s
kubectl wait --for=condition=ready pod -l component=dashboard -n edwin --timeout=300s
kubectl wait --for=condition=ready pod -l component=mobile -n edwin --timeout=300s
```

### Step 6: Deploy Ingress

```bash
kubectl apply -f k8s/ingress.yaml

# Get ingress IP
kubectl get ingress -n edwin
```

### Step 7: Configure DNS

Point your domains to the Ingress IP:

```
api.edwin.edu       -> <INGRESS_IP>
dashboard.edwin.edu -> <INGRESS_IP>
mobile.edwin.edu    -> <INGRESS_IP>
```

### Step 8: Run Migrations

```bash
./scripts/deploy/migrate.sh edwin
```

---

## Monitoring & Logging

### Deploy Prometheus & Grafana

**Option 1: Using docker-compose profiles** (for testing):
```bash
docker-compose -f docker-compose.edwin.yml --profile monitoring up -d
```

**Option 2: Using Helm** (for production):
```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus + Grafana
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values monitoring/prometheus-values.yaml
```

**Access Grafana**:
```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
# Visit: http://localhost:3000
# Login: admin / prom-operator
```

### View Logs

**Individual pod**:
```bash
kubectl logs -n edwin <pod-name>
kubectl logs -n edwin edwin-api-5d6f8b9c7-abc123
```

**All pods of a component**:
```bash
kubectl logs -n edwin -l component=api --tail=100 -f
```

**Stream logs** (using stern):
```bash
stern -n edwin api
```

---

## Scaling

### Manual Scaling

**Scale deployment**:
```bash
kubectl scale deployment edwin-api -n edwin --replicas=5

# Verify
kubectl get deployment edwin-api -n edwin
```

### Horizontal Pod Autoscaler (HPA)

**Check HPA**:
```bash
kubectl get hpa -n edwin
```

Example output:
```
NAME            REFERENCE              TARGETS   MINPODS   MAXPODS   REPLICAS
edwin-api-hpa   Deployment/edwin-api   45%/70%   3         10        3
```

**Modify HPA**:
```bash
kubectl edit hpa edwin-api-hpa -n edwin
```

### Vertical Pod Autoscaler (VPA)

**Install VPA** (if not already installed):
```bash
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh
```

**Create VPA**:
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: edwin-api-vpa
  namespace: edwin
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: edwin-api
  updatePolicy:
    updateMode: "Auto"
```

---

## Updates & Rollbacks

### Rolling Update

**Update image**:
```bash
kubectl set image deployment/edwin-api \
  api=edwin-ai-tutor:v1.1.0 \
  -n edwin
```

**Monitor rollout**:
```bash
kubectl rollout status deployment/edwin-api -n edwin
```

### Rollback

**View history**:
```bash
kubectl rollout history deployment/edwin-api -n edwin
```

**Rollback to previous**:
```bash
kubectl rollout undo deployment/edwin-api -n edwin
```

**Rollback to specific revision**:
```bash
kubectl rollout undo deployment/edwin-api -n edwin --to-revision=3
```

**Using script**:
```bash
./scripts/deploy/rollback.sh edwin        # Previous revision
./scripts/deploy/rollback.sh edwin 3      # Specific revision
```

---

## Production Best Practices

### 1. Resource Limits

Always set resource requests and limits:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "250m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### 2. Health Checks

Configure liveness and readiness probes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 20
  periodSeconds: 5
```

### 3. Pod Disruption Budget

Prevent too many pods from being down:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: edwin-api-pdb
  namespace: edwin
spec:
  minAvailable: 2
  selector:
    matchLabels:
      component: api
```

### 4. Network Policies

Restrict network access:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: edwin-api-netpol
  namespace: edwin
spec:
  podSelector:
    matchLabels:
      component: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
  egress:
  - to:
    - podSelector:
        matchLabels:
          component: neo4j
  - to:
    - podSelector:
        matchLabels:
          component: qdrant
```

### 5. Backup Strategy

**Automated backups via CronJob**:

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: edwin-backup
  namespace: edwin
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: edwin-ai-tutor:latest
            command:
            - /app/scripts/deploy/backup.sh
          restartPolicy: OnFailure
```

---

## Troubleshooting

### Pods Not Starting

**Check pod status**:
```bash
kubectl get pods -n edwin
kubectl describe pod <pod-name> -n edwin
```

**Common issues**:
1. **ImagePullBackOff**: Image not found or credentials missing
2. **CrashLoopBackOff**: Application crashing on startup
3. **Pending**: Insufficient resources or PVC issues

### Database Connection Failures

**Test connectivity**:
```bash
# From API pod to Neo4j
kubectl exec -n edwin <api-pod> -- nc -zv neo4j 7687

# From API pod to Qdrant
kubectl exec -n edwin <api-pod> -- nc -zv qdrant 6333
```

### High Resource Usage

**Check resource usage**:
```bash
kubectl top nodes
kubectl top pods -n edwin
```

**Analyze logs for errors**:
```bash
kubectl logs -n edwin -l component=api --tail=1000 | grep ERROR
```

### Ingress Issues

**Check Ingress status**:
```bash
kubectl describe ingress edwin-ingress -n edwin
```

**Test from within cluster**:
```bash
kubectl run curl --image=curlimages/curl -it --rm -- sh
curl http://edwin-api:8000/health
```

---

## Cost Optimization

### 1. Use Spot Instances

**AWS**:
```bash
eksctl create nodegroup \
  --cluster=edwin-cluster \
  --spot \
  --instance-types=t3.large,t3a.large
```

### 2. Right-size Resources

Monitor actual usage and adjust requests/limits:

```bash
kubectl top pods -n edwin
kubectl top nodes
```

### 3. Use Cluster Autoscaler

Automatically scale nodes based on demand:

```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml
```

---

## Security Hardening

See [SECURITY.md](./SECURITY.md) for complete security guide.

**Key recommendations**:
- Enable RBAC
- Use Pod Security Standards
- Enable network policies
- Scan images for vulnerabilities
- Rotate secrets regularly
- Enable audit logging

---

**Last Updated**: November 15, 2025

# HoloLoom VoiceAgent - Production Deployment Guide

**Version**: 1.0.0
**Date**: November 15, 2025
**Status**: Production Ready

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Monitoring & Observability](#monitoring--observability)
5. [Security Hardening](#security-hardening)
6. [Scaling & Performance](#scaling--performance)
7. [Backup & Recovery](#backup--recovery)
8. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB
- Network: 1Gbps

**Recommended**:
- CPU: 8 cores
- RAM: 16GB
- Storage: 100GB SSD
- Network: 10Gbps

### Software Requirements

```bash
# Docker
docker --version  # ≥ 20.10.0
docker-compose --version  # ≥ 2.0.0

# Kubernetes
kubectl version  # ≥ 1.24.0
helm version  # ≥ 3.10.0

# Optional
k9s --version  # For cluster management
stern --version  # For log tailing
```

### Required Credentials

1. **OpenAI API Key** (for TTS)
   ```bash
   export OPENAI_API_KEY='sk-your-api-key'
   ```

2. **Docker Registry** (if using private registry)
   ```bash
   docker login registry.example.com
   ```

3. **Kubernetes Context** (if deploying to K8s)
   ```bash
   kubectl config current-context
   ```

---

## 🐳 Docker Deployment

### Local Development

#### 1. Build Image

```bash
# Build VoiceAgent image
docker build -f Dockerfile.voice -t hololoom/voice-agent:latest .

# Verify build
docker images | grep voice-agent
```

#### 2. Start Services

```bash
# Start all services
docker-compose -f docker-compose.voice.yml up -d

# Check status
docker-compose -f docker-compose.voice.yml ps

# View logs
docker-compose -f docker-compose.voice.yml logs -f voice-agent
```

#### 3. Access Services

```
Voice Agent API:  http://localhost:8000
Neo4j Browser:    http://localhost:7474
Qdrant API:       http://localhost:6333
Prometheus:       http://localhost:9090
Grafana:          http://localhost:3000 (admin/admin)
```

#### 4. Test VoiceAgent

```bash
# Enter container
docker exec -it voice-agent bash

# Run tests
python -m pytest HoloLoom/voice/tests/ -v

# Run demo
python demos/demo_voice_agent.py
```

#### 5. Stop Services

```bash
# Stop all services
docker-compose -f docker-compose.voice.yml down

# Remove volumes (data will be lost!)
docker-compose -f docker-compose.voice.yml down -v
```

### Production Docker Deployment

#### 1. Build Production Image

```bash
# Build with build args
docker build -f Dockerfile.voice \
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  -t registry.example.com/hololoom/voice-agent:1.0.0 \
  .

# Tag as latest
docker tag registry.example.com/hololoom/voice-agent:1.0.0 \
  registry.example.com/hololoom/voice-agent:latest

# Push to registry
docker push registry.example.com/hololoom/voice-agent:1.0.0
docker push registry.example.com/hololoom/voice-agent:latest
```

#### 2. Configure Production Environment

```bash
# Create .env file
cat > .env.production <<EOF
# API Keys
OPENAI_API_KEY=sk-your-real-api-key

# Database Passwords
NEO4J_PASSWORD=your-secure-neo4j-password
REDIS_PASSWORD=your-secure-redis-password

# Grafana
GRAFANA_PASSWORD=your-secure-grafana-password
EOF

# Secure the file
chmod 600 .env.production
```

#### 3. Deploy Production Stack

```bash
# Deploy with production environment
docker-compose -f docker-compose.voice.yml \
  --env-file .env.production \
  up -d

# Scale voice agents
docker-compose -f docker-compose.voice.yml up -d --scale voice-agent=5

# Monitor deployment
docker-compose -f docker-compose.voice.yml logs -f
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

#### 1. Cluster Access

```bash
# Verify cluster access
kubectl cluster-info

# Check nodes
kubectl get nodes

# Verify resources
kubectl top nodes
```

#### 2. Install Prerequisites

```bash
# Install metrics server (for HPA)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Verify metrics server
kubectl get deployment metrics-server -n kube-system
```

### Deployment Steps

#### 1. Create Namespace

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Verify
kubectl get namespace hololoom-voice
```

#### 2. Create Secrets

```bash
# Create secret for OpenAI API key
kubectl create secret generic voice-agent-secrets \
  --from-literal=OPENAI_API_KEY='sk-your-api-key' \
  --from-literal=NEO4J_PASSWORD='your-neo4j-password' \
  --namespace=hololoom-voice

# Verify secret
kubectl get secret voice-agent-secrets -n hololoom-voice
```

#### 3. Deploy Configuration

```bash
# Deploy ConfigMap
kubectl apply -f deployment/kubernetes/configmap.yaml

# Verify
kubectl get configmap voice-agent-config -n hololoom-voice
```

#### 4. Deploy Storage

```bash
# Deploy PVC
kubectl apply -f deployment/kubernetes/pvc.yaml

# Wait for PVC to be bound
kubectl get pvc -n hololoom-voice -w
```

#### 5. Deploy Application

```bash
# Deploy Voice Agent
kubectl apply -f deployment/kubernetes/deployment.yaml

# Wait for pods to be ready
kubectl get pods -n hololoom-voice -w

# Check pod status
kubectl describe pod -n hololoom-voice -l app=voice-agent
```

#### 6. Deploy Service

```bash
# Deploy services
kubectl apply -f deployment/kubernetes/service.yaml

# Verify services
kubectl get svc -n hololoom-voice
```

#### 7. Deploy Autoscaling

```bash
# Deploy HPA
kubectl apply -f deployment/kubernetes/hpa.yaml

# Verify HPA
kubectl get hpa -n hololoom-voice
```

#### 8. Verify Deployment

```bash
# Check all resources
kubectl get all -n hololoom-voice

# Check logs
kubectl logs -n hololoom-voice -l app=voice-agent -f

# Test pod health
kubectl exec -n hololoom-voice -it \
  $(kubectl get pod -n hololoom-voice -l app=voice-agent -o jsonpath='{.items[0].metadata.name}') \
  -- python -c "from HoloLoom.voice import VoiceAgent; print('healthy')"
```

### Access Application

#### Port Forward (for testing)

```bash
# Forward port to local machine
kubectl port-forward -n hololoom-voice \
  svc/voice-agent-service 8000:8000

# Access at http://localhost:8000
```

#### LoadBalancer (for production)

```bash
# Get external IP
kubectl get svc voice-agent-external -n hololoom-voice

# Wait for EXTERNAL-IP
kubectl get svc voice-agent-external -n hololoom-voice -w
```

#### Ingress (recommended for production)

```yaml
# Create Ingress (example with nginx)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: voice-agent-ingress
  namespace: hololoom-voice
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - voice.hololoom.example.com
      secretName: voice-tls
  rules:
    - host: voice.hololoom.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: voice-agent-service
                port:
                  number: 8000
```

```bash
# Apply Ingress
kubectl apply -f ingress.yaml

# Get Ingress URL
kubectl get ingress -n hololoom-voice
```

---

## 📊 Monitoring & Observability

### Prometheus Setup

#### 1. Deploy Prometheus

```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace hololoom-voice \
  --values deployment/prometheus/values.yaml

# Verify deployment
kubectl get pods -n hololoom-voice -l app.kubernetes.io/name=prometheus
```

#### 2. Access Prometheus

```bash
# Port forward Prometheus
kubectl port-forward -n hololoom-voice \
  svc/prometheus-kube-prometheus-prometheus 9090:9090

# Access at http://localhost:9090
```

#### 3. Import Dashboards

```bash
# Apply Grafana dashboards
kubectl apply -f deployment/grafana/dashboards/
```

### Grafana Setup

#### 1. Access Grafana

```bash
# Get Grafana password
kubectl get secret -n hololoom-voice prometheus-grafana \
  -o jsonpath="{.data.admin-password}" | base64 --decode

# Port forward Grafana
kubectl port-forward -n hololoom-voice \
  svc/prometheus-grafana 3000:80

# Access at http://localhost:3000
# Username: admin
# Password: (from above command)
```

#### 2. Key Dashboards

- **VoiceAgent Overview**: General health and performance
- **Conversation Metrics**: Active sessions, turn-taking, context usage
- **TTS Performance**: Synthesis latency, queue depth
- **Error Tracking**: Failures, retries, timeouts
- **Resource Usage**: CPU, memory, network, storage

### Logging

#### 1. Centralized Logging (ELK Stack)

```bash
# Install ELK via Helm
helm install elasticsearch elastic/elasticsearch \
  --namespace logging \
  --create-namespace

helm install kibana elastic/kibana \
  --namespace logging

helm install filebeat elastic/filebeat \
  --namespace logging
```

#### 2. Structured Logs Query

```bash
# View logs with kubectl
kubectl logs -n hololoom-voice -l app=voice-agent \
  --tail=100 \
  --timestamps

# Stream logs
kubectl logs -n hololoom-voice -l app=voice-agent -f

# Query specific errors
kubectl logs -n hololoom-voice -l app=voice-agent | \
  grep -i "error\|exception\|failure"
```

#### 3. Log Aggregation (Stern)

```bash
# Install stern
brew install stern  # macOS
# or download from https://github.com/stern/stern

# Tail all voice-agent pods
stern -n hololoom-voice voice-agent

# Filter by severity
stern -n hololoom-voice voice-agent --include "ERROR|CRITICAL"
```

---

## 🔒 Security Hardening

### 1. Network Policies

```yaml
# Create NetworkPolicy to restrict pod communication
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: voice-agent-netpol
  namespace: hololoom-voice
spec:
  podSelector:
    matchLabels:
      app: voice-agent
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: neo4j
      ports:
        - protocol: TCP
          port: 7687
    - to:
        - podSelector:
            matchLabels:
              app: qdrant
      ports:
        - protocol: TCP
          port: 6333
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: TCP
          port: 53  # DNS
        - protocol: UDP
          port: 53
```

### 2. Pod Security Standards

```yaml
# Create PodSecurityPolicy (if using PSP)
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: voice-agent-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
```

### 3. Secrets Management

```bash
# Use Sealed Secrets
helm install sealed-secrets sealed-secrets/sealed-secrets \
  --namespace kube-system

# Create sealed secret
kubectl create secret generic voice-agent-secrets \
  --from-literal=OPENAI_API_KEY='sk-your-key' \
  --dry-run=client -o yaml | \
  kubeseal -o yaml > sealed-secret.yaml

# Apply sealed secret
kubectl apply -f sealed-secret.yaml
```

### 4. Image Scanning

```bash
# Scan image with Trivy
trivy image hololoom/voice-agent:latest

# Scan for HIGH and CRITICAL vulnerabilities only
trivy image --severity HIGH,CRITICAL hololoom/voice-agent:latest
```

---

## 📈 Scaling & Performance

### Horizontal Pod Autoscaling

```bash
# View HPA status
kubectl get hpa -n hololoom-voice

# Describe HPA
kubectl describe hpa voice-agent-hpa -n hololoom-voice

# Manual scaling (overrides HPA temporarily)
kubectl scale deployment voice-agent \
  --replicas=10 \
  --namespace=hololoom-voice
```

### Vertical Pod Autoscaling

```bash
# Install VPA
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh

# Create VPA
cat <<EOF | kubectl apply -f -
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: voice-agent-vpa
  namespace: hololoom-voice
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-agent
  updatePolicy:
    updateMode: "Auto"
EOF

# Check VPA recommendations
kubectl describe vpa voice-agent-vpa -n hololoom-voice
```

### Performance Tuning

#### 1. Resource Optimization

```yaml
# Optimized resource requests/limits
resources:
  requests:
    memory: "1Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "4000m"
```

#### 2. Connection Pooling

```yaml
# Add to ConfigMap
NEO4J_MAX_CONNECTIONS: "100"
QDRANT_MAX_CONNECTIONS: "50"
REDIS_MAX_CONNECTIONS: "50"
```

#### 3. Caching

```yaml
# Enable Redis caching
ENABLE_REDIS_CACHE: "true"
CACHE_TTL: "3600"  # 1 hour
```

---

## 💾 Backup & Recovery

### Database Backups

#### Neo4j Backup

```bash
# Create Neo4j backup CronJob
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: neo4j-backup
  namespace: hololoom-voice
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: neo4j:5.13
              command:
                - /bin/bash
                - -c
                - neo4j-admin backup --backup-dir=/backup --name=daily
              volumeMounts:
                - name: neo4j-data
                  mountPath: /data
                - name: backup
                  mountPath: /backup
          volumes:
            - name: neo4j-data
              persistentVolumeClaim:
                claimName: neo4j-pvc
            - name: backup
              persistentVolumeClaim:
                claimName: backup-pvc
          restartPolicy: OnFailure
EOF
```

#### Qdrant Backup

```bash
# Create snapshot
kubectl exec -n hololoom-voice qdrant-0 -- \
  curl -X POST http://localhost:6333/collections/voice_sessions/snapshots

# Download snapshot
kubectl cp hololoom-voice/qdrant-0:/qdrant/storage/snapshots/snapshot.tar.gz \
  ./snapshot.tar.gz
```

### Application State Backup

```bash
# Backup PVCs using Velero
velero backup create voice-agent-backup \
  --include-namespaces hololoom-voice \
  --wait

# Verify backup
velero backup describe voice-agent-backup

# List backups
velero backup get
```

### Disaster Recovery

#### 1. Restore from Backup

```bash
# Restore namespace from Velero backup
velero restore create --from-backup voice-agent-backup

# Monitor restore
velero restore get
velero restore describe voice-agent-backup
```

#### 2. Rollback Deployment

```bash
# View deployment history
kubectl rollout history deployment/voice-agent -n hololoom-voice

# Rollback to previous version
kubectl rollout undo deployment/voice-agent -n hololoom-voice

# Rollback to specific revision
kubectl rollout undo deployment/voice-agent \
  --to-revision=2 \
  -n hololoom-voice
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Pods Not Starting

```bash
# Check pod status
kubectl get pods -n hololoom-voice

# Describe pod
kubectl describe pod <pod-name> -n hololoom-voice

# Check events
kubectl get events -n hololoom-voice --sort-by='.lastTimestamp'

# Check logs
kubectl logs <pod-name> -n hololoom-voice
```

**Common Causes**:
- Image pull failure
- Insufficient resources
- ConfigMap/Secret missing
- PVC not bound

#### 2. High Memory Usage

```bash
# Check memory usage
kubectl top pods -n hololoom-voice

# Increase memory limit
kubectl set resources deployment voice-agent \
  --limits=memory=4Gi \
  --namespace=hololoom-voice
```

#### 3. Database Connection Failures

```bash
# Test Neo4j connection
kubectl exec -n hololoom-voice -it <pod-name> -- \
  python -c "from neo4j import GraphDatabase; \
  driver = GraphDatabase.driver('bolt://neo4j-service:7687'); \
  driver.verify_connectivity(); \
  print('Connected!')"

# Test Qdrant connection
kubectl exec -n hololoom-voice -it <pod-name> -- \
  curl http://qdrant-service:6333/collections
```

#### 4. TTS Failures

```bash
# Check OpenAI API key
kubectl get secret voice-agent-secrets -n hololoom-voice \
  -o jsonpath='{.data.OPENAI_API_KEY}' | base64 --decode

# Test TTS
kubectl exec -n hololoom-voice -it <pod-name> -- \
  python -c "from HoloLoom.voice import OpenAITTS; \
  tts = OpenAITTS(); \
  print('TTS initialized')"
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/voice-agent \
  LOG_LEVEL=DEBUG \
  --namespace=hololoom-voice

# View debug logs
kubectl logs -n hololoom-voice -l app=voice-agent -f | grep DEBUG
```

### Health Checks

```bash
# Check liveness probe
kubectl exec -n hololoom-voice -it <pod-name> -- \
  python -c "from HoloLoom.voice import VoiceAgent; print('alive')"

# Check readiness probe
kubectl exec -n hololoom-voice -it <pod-name> -- \
  python -c "from HoloLoom.voice import VoiceAgent; print('ready')"
```

---

## 📞 Support

### Documentation
- **README**: `HoloLoom/voice/README.md`
- **Architecture**: `PHASE_2_VOICE_MODE_ARCHITECTURE.md`
- **Code Review**: `ELLE_AUDIO_REVIEW_SUMMARY.md`

### Monitoring
- **Prometheus**: Metrics and alerts
- **Grafana**: Visualization dashboards
- **Logs**: Centralized logging (ELK/Loki)

### Contact
- **Issues**: File on GitHub repository
- **Email**: support@hololoom.example.com

---

## ✅ Pre-Deployment Checklist

### Development
- [ ] Code tested locally
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Docker image built successfully
- [ ] docker-compose working

### Staging
- [ ] Deployed to staging cluster
- [ ] Load testing completed
- [ ] Security scan passed
- [ ] Monitoring dashboards configured
- [ ] Alerts tested

### Production
- [ ] Secrets configured
- [ ] Backups configured
- [ ] Monitoring enabled
- [ ] Alerts configured
- [ ] Runbook documented
- [ ] On-call rotation set up
- [ ] Disaster recovery tested

---

**Version**: 1.0.0
**Last Updated**: November 15, 2025
**Status**: Production Ready

---

*This guide provides comprehensive production deployment instructions for HoloLoom VoiceAgent using Docker and Kubernetes with best practices for security, monitoring, and reliability.*

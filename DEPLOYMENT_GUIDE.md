# HoloLoom Production Deployment Guide

**Complete guide to deploying HoloLoom with Emotion Intelligence**

Built: November 2025

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [AWS Deployment](#aws-deployment)
4. [Scaling Recommendations](#scaling-recommendations)
5. [Monitoring & Observability](#monitoring--observability)
6. [Troubleshooting](#troubleshooting)
7. [Security Hardening](#security-hardening)

---

## Prerequisites

### Required Software

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Git** (for cloning repository)
- **API Keys**:
  - Anthropic API key (required for Claude)
  - OpenAI API key (optional)

### Hardware Requirements

**Minimum (Development)**:
- 4 CPU cores
- 8GB RAM
- 20GB disk space

**Recommended (Production)**:
- 8+ CPU cores
- 16GB+ RAM
- 100GB+ SSD storage

---

## Local Deployment

### 1. Clone Repository

```bash
git clone https://github.com/your-org/mythRL.git
cd mythRL
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

**Required variables**:
```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
NEO4J_PASSWORD=your_secure_password
GRAFANA_PASSWORD=your_secure_password
```

### 3. Start Services (One Command!)

```bash
docker-compose up -d
```

This starts **7 services**:
- `neo4j` - Graph database (ports 7474, 7687)
- `qdrant` - Vector database (ports 6333, 6334)
- `redis` - Cache layer (port 6379)
- `hololoom-api` - Main Python API (port 8000)
- `emotion-intelligence` - Node.js emotion service (port 3000)
- `prometheus` - Metrics collection (port 9092)
- `grafana` - Monitoring dashboards (port 3001)

### 4. Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Should show all services as "healthy"
```

**Access points**:
- HoloLoom API: http://localhost:8000
- Emotion Intelligence: http://localhost:3000
- Neo4j Browser: http://localhost:7474
- Grafana: http://localhost:3001 (admin / hololoom_grafana_admin)
- Prometheus: http://localhost:9092

### 5. Test End-to-End

```bash
# Test HoloLoom API
curl http://localhost:8000/health

# Test Emotion Intelligence
curl http://localhost:3000/health

# Query with emotion analysis
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I'\''m feeling frustrated with this bug",
    "enable_emotion": true
  }'
```

### 6. View Monitoring Dashboards

1. Open Grafana: http://localhost:3001
2. Login (admin / hololoom_grafana_admin)
3. Navigate to **Dashboards → HoloLoom**
4. View:
   - **Emotion Trends** - Emotion detection analytics
   - **System Performance** - Infrastructure health

---

## AWS Deployment

### Option 1: AWS ECS (Elastic Container Service)

**Best for**: Simple containerized deployment without Kubernetes complexity

#### Step 1: Create ECR Repositories

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create repositories
aws ecr create-repository --repository-name hololoom-api
aws ecr create-repository --repository-name emotion-intelligence
```

#### Step 2: Build and Push Images

```bash
# Build HoloLoom API
docker build -t hololoom-api:latest .
docker tag hololoom-api:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/hololoom-api:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hololoom-api:latest

# Build Emotion Intelligence
docker build -t emotion-intelligence:latest -f docker/Dockerfile.emotion .
docker tag emotion-intelligence:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/emotion-intelligence:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/emotion-intelligence:latest
```

#### Step 3: Create ECS Task Definition

Create `ecs-task-definition.json`:

```json
{
  "family": "hololoom-production",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "hololoom-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/hololoom-api:latest",
      "portMappings": [
        {"containerPort": 8000, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "HOLOLOOM_ENV", "value": "production"},
        {"name": "NEO4J_URI", "value": "bolt://neo4j.internal:7687"},
        {"name": "QDRANT_HOST", "value": "qdrant.internal"}
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:anthropic-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hololoom",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "api"
        }
      }
    },
    {
      "name": "emotion-intelligence",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/emotion-intelligence:latest",
      "portMappings": [
        {"containerPort": 3000, "protocol": "tcp"}
      ],
      "environment": [
        {"name": "NODE_ENV", "value": "production"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hololoom",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "emotion"
        }
      }
    }
  ]
}
```

#### Step 4: Deploy to ECS

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster hololoom-production \
  --service-name hololoom-api \
  --task-definition hololoom-production:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=hololoom-api,containerPort=8000"
```

#### Step 5: Set Up RDS for Neo4j (or DocumentDB)

**Alternative**: Use managed Neo4j on AWS Marketplace

```bash
# Create RDS PostgreSQL (if not using Neo4j)
aws rds create-db-instance \
  --db-instance-identifier hololoom-postgres \
  --db-instance-class db.t3.medium \
  --engine postgres \
  --master-username admin \
  --master-user-password <secure-password> \
  --allocated-storage 100
```

---

### Option 2: AWS EKS (Elastic Kubernetes Service)

**Best for**: Advanced orchestration, auto-scaling, and cloud-native deployment

#### Step 1: Create EKS Cluster

```bash
# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Create cluster
eksctl create cluster \
  --name hololoom-production \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 5 \
  --managed
```

#### Step 2: Deploy with Helm

Create `values.yaml`:

```yaml
replicaCount: 3

image:
  repository: <account-id>.dkr.ecr.us-east-1.amazonaws.com/hololoom-api
  tag: latest
  pullPolicy: Always

service:
  type: LoadBalancer
  port: 8000

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  - name: HOLOLOOM_ENV
    value: production
  - name: NEO4J_URI
    value: bolt://neo4j-headless:7687
  - name: ANTHROPIC_API_KEY
    valueFrom:
      secretKeyRef:
        name: hololoom-secrets
        key: anthropic-api-key
```

Deploy:

```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Deploy
helm install hololoom ./helm/hololoom -f values.yaml
```

#### Step 3: Set Up Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: hololoom-ingress
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
spec:
  rules:
  - host: api.hololoom.ai
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: hololoom-api
            port:
              number: 8000
```

---

## Scaling Recommendations

### Horizontal Scaling

**When to scale**:
- CPU usage > 70% sustained
- Request latency > 500ms p95
- Queue depth growing

**ECS Scaling**:
```bash
# Update desired count
aws ecs update-service \
  --cluster hololoom-production \
  --service hololoom-api \
  --desired-count 5
```

**EKS Auto-Scaling**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hololoom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hololoom-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

**HoloLoom API** (memory-intensive):
- Start: 2GB RAM, 1 CPU
- Scale up: 4GB RAM, 2 CPU
- Maximum: 8GB RAM, 4 CPU

**Emotion Intelligence** (CPU-intensive):
- Start: 1GB RAM, 1 CPU
- Scale up: 2GB RAM, 2 CPU
- Maximum: 4GB RAM, 4 CPU

### Database Scaling

**Neo4j**:
- Enable read replicas for query scaling
- Consider Neo4j Aura (managed service)

**Qdrant**:
- Shard collections when > 10M vectors
- Add read replicas for high query load

**Redis**:
- Enable Redis Cluster for > 512MB data
- Use AWS ElastiCache for managed Redis

---

## Monitoring & Observability

### Prometheus Metrics

**Available metrics**:
- `hololoom_requests_total` - Total API requests
- `hololoom_query_latency_ms` - Query latency percentiles
- `emotion_requests_total` - Emotion analysis requests
- `emotion_processing_time_ms` - Emotion processing latency
- `hololoom_cache_hit_rate` - Cache effectiveness

### Grafana Dashboards

**Pre-configured dashboards**:
1. **Emotion Trends** - Emotion detection analytics
2. **System Performance** - Infrastructure health

**Custom queries**:
```promql
# P95 latency over 5 minutes
histogram_quantile(0.95, rate(hololoom_query_latency_ms_bucket[5m]))

# Cache hit rate
rate(hololoom_cache_hits_total[5m]) / rate(hololoom_cache_requests_total[5m]) * 100

# Error rate
rate(hololoom_errors_total[5m])
```

### CloudWatch Integration (AWS)

```bash
# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluentd-quickstart.yaml
```

### Alerting

**Recommended alerts**:

| Alert | Condition | Action |
|-------|-----------|--------|
| High Latency | P95 > 500ms for 5min | Scale horizontally |
| High Error Rate | Error rate > 1% for 5min | Investigate logs |
| Low Cache Hit | Hit rate < 70% for 10min | Review cache config |
| Database Connection | Active connections > 80 | Scale database |

---

## Troubleshooting

### Service Won't Start

**Check logs**:
```bash
# Docker Compose
docker-compose logs hololoom-api
docker-compose logs emotion-intelligence

# ECS
aws logs tail /ecs/hololoom --follow

# EKS
kubectl logs -f deployment/hololoom-api
```

**Common issues**:

1. **Missing API keys**
   ```
   Error: ANTHROPIC_API_KEY not set
   ```
   Solution: Set in `.env` or AWS Secrets Manager

2. **Database connection failed**
   ```
   Error: Failed to connect to Neo4j
   ```
   Solution: Check `NEO4J_URI` and network connectivity

3. **Port already in use**
   ```
   Error: bind: address already in use
   ```
   Solution: Stop conflicting service or change port in docker-compose.yml

### High Latency

**Diagnosis**:
1. Check Grafana dashboard for bottlenecks
2. Review Prometheus metrics for slow stages
3. Check database query performance

**Solutions**:
- Enable query caching
- Scale horizontally
- Optimize database indexes
- Review query complexity distribution

### Memory Leaks

**Diagnosis**:
```bash
# Check memory usage
docker stats

# Or in Grafana: Memory Usage panel
```

**Solutions**:
- Restart affected service
- Reduce cache size
- Check for unclosed connections

---

## Security Hardening

### Production Checklist

- [ ] Change default passwords (Neo4j, Grafana)
- [ ] Use HTTPS (TLS certificates)
- [ ] Enable firewall rules (only expose necessary ports)
- [ ] Use AWS Secrets Manager / Kubernetes Secrets for API keys
- [ ] Enable authentication on all services
- [ ] Regular security updates (`docker-compose pull`)
- [ ] Enable rate limiting
- [ ] Configure CORS properly
- [ ] Enable audit logging
- [ ] Implement backup strategy

### HTTPS Configuration

**Option 1: AWS ALB with ACM Certificate**
```bash
# Request certificate
aws acm request-certificate \
  --domain-name api.hololoom.ai \
  --validation-method DNS

# Configure ALB to use certificate
aws elbv2 create-listener \
  --load-balancer-arn <alb-arn> \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=<cert-arn> \
  --default-actions Type=forward,TargetGroupArn=<target-group-arn>
```

**Option 2: Let's Encrypt with Nginx**
```yaml
# Add nginx service to docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
    - ./ssl:/etc/nginx/ssl:ro
```

### Network Security

**Security groups (AWS)**:
- Allow port 8000 only from ALB
- Allow port 3000 only from HoloLoom API
- Allow Neo4j/Qdrant only from application subnet
- Deny all other inbound traffic

---

## Next Steps

1. **Test deployment** with sample queries
2. **Configure monitoring alerts** in Grafana
3. **Set up automated backups** for Neo4j and Qdrant
4. **Review security checklist**
5. **Plan capacity based on expected load**

---

## Support

- **Documentation**: See [HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)
- **Issues**: https://github.com/your-org/mythRL/issues
- **Monitoring**: Grafana dashboards at http://your-grafana-url:3001

---

**Deployment Guide Version**: 1.0 (November 2025)

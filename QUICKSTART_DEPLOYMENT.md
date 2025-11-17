# EdWIN AI Tutor - Quick Start Deployment Guide

**5-Minute Production Deployment**

---

## Prerequisites

- Docker & Docker Compose installed
- Kubernetes cluster (or kubectl + cluster access)
- Python 3.11+
- Git

---

## Local Development (5 minutes)

### 1. Clone & Setup
```bash
git clone <repo-url>
cd hello-world
```

### 2. Start Services
```bash
# Start all services (API, Neo4j, Qdrant, Redis)
docker-compose -f docker-compose.edwin.yml up -d

# Wait for databases to be ready (~30 seconds)
sleep 30
```

### 3. Initialize Database
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Initialize databases
./scripts/init_db.sh

# Seed test data
python scripts/seed_data.py
```

### 4. Verify Health
```bash
# Check all services
./scripts/health_check.sh

# Or manually check:
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### 5. Test Authentication
```bash
# Run production demo
PYTHONPATH=. python demos/edwin_production_demo.py
```

### 6. Access Services
```
API:           http://localhost:8000
Dashboard:     http://localhost:8001
Mobile API:    http://localhost:8002
Neo4j Browser: http://localhost:7474 (neo4j/edwin_password_2025)
Qdrant:        http://localhost:6333/dashboard
Prometheus:    http://localhost:9090 (optional)
Grafana:       http://localhost:3000 (optional, admin/admin)
```

**Default Credentials**:
```
Admin:   admin / Admin123!@#
Teacher: teacher_johnson / Teacher123!@#
Student: student_1 / Student123!@#
Parent:  parent_smith / Parent123!@#
```

---

## Kubernetes Deployment (10 minutes)

### 1. Create Secrets
```bash
# Copy template
cp k8s/secrets.yaml.template k8s/secrets.staging.yaml

# Edit secrets (use real values!)
vim k8s/secrets.staging.yaml

# Or create directly
kubectl create secret generic edwin-secrets -n edwin \
  --from-literal=NEO4J_PASSWORD=your_secure_password \
  --from-literal=REDIS_PASSWORD=your_secure_password \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -base64 32) \
  --from-literal=ANTHROPIC_API_KEY=your_api_key
```

### 2. Deploy to Kubernetes
```bash
# One-command deployment
./scripts/deploy.sh staging

# This will:
# 1. Build Docker image
# 2. Push to registry
# 3. Create namespace
# 4. Apply secrets & configmaps
# 5. Deploy databases
# 6. Deploy applications
# 7. Apply autoscaling
# 8. Run health checks
```

### 3. Verify Deployment
```bash
# Check pods
kubectl get pods -n edwin

# Check services
kubectl get svc -n edwin

# Check ingress
kubectl get ingress -n edwin

# Port forward for local access
kubectl port-forward svc/edwin-api 8000:8000 -n edwin
```

### 4. Access via Ingress
```
API:       https://api.edwin.edu
Dashboard: https://dashboard.edwin.edu
Mobile:    https://mobile.edwin.edu
```

---

## CI/CD Setup (GitHub Actions)

### 1. Configure Secrets
Go to GitHub repo → Settings → Secrets → Add:
```
KUBE_CONFIG_STAGING      # Base64-encoded kubeconfig for staging
KUBE_CONFIG_PRODUCTION   # Base64-encoded kubeconfig for production
NEO4J_PASSWORD_STAGING   # Neo4j password (staging)
NEO4J_PASSWORD_PROD      # Neo4j password (production)
REDIS_PASSWORD_STAGING   # Redis password (staging)
REDIS_PASSWORD_PROD      # Redis password (production)
JWT_SECRET_KEY_STAGING   # JWT secret (staging)
JWT_SECRET_KEY_PROD      # JWT secret (production)
ANTHROPIC_API_KEY        # Anthropic API key
ANTHROPIC_API_KEY_PROD   # Anthropic API key (production)
SENTRY_DSN               # Sentry DSN for error tracking
SLACK_WEBHOOK            # Slack webhook for notifications
```

### 2. Workflow Triggers
```
Test:       Automatic on PR
Build:      Automatic on push to main/develop
Staging:    Automatic on push to develop
Production: Manual trigger + 2 approvals
```

### 3. Deploy to Staging
```bash
# Push to develop branch
git checkout develop
git push origin develop

# GitHub Actions will automatically:
# 1. Run tests
# 2. Build Docker image
# 3. Deploy to staging
# 4. Run smoke tests
```

### 4. Deploy to Production
```bash
# Create release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# Or manually trigger workflow
# Go to Actions → Deploy to Production → Run workflow
# Requires 2 approvals from team leads
```

---

## Monitoring

### Prometheus Metrics
```
URL: http://localhost:9090 (or https://prometheus.edwin.edu)

Key Metrics:
- edwin_requests_total
- edwin_response_time_seconds
- edwin_students_active
- edwin_questions_answered_total
```

### Grafana Dashboards
```
URL: http://localhost:3000 (or https://grafana.edwin.edu)
Login: admin / admin

Dashboards:
- EdWIN API Overview
- EdWIN Learning Analytics
```

### Health Checks
```bash
# All services
./scripts/health_check.sh

# Individual endpoints
curl http://localhost:8000/health
curl http://localhost:8000/health/db
curl http://localhost:8000/metrics
```

---

## Maintenance

### Backup
```bash
# Manual backup
./scripts/backup_db.sh /var/backups/edwin

# Backup location
ls /var/backups/edwin/

# Automated backups (production)
# Runs daily at 2 AM UTC via cron
```

### Restore
```bash
# Restore from backup
./scripts/restore_db.sh /var/backups/edwin/edwin_backup_20251117_120000.tar.gz
```

### Rollback
```bash
# Emergency rollback
./scripts/rollback.sh staging
./scripts/rollback.sh production
```

### Scale
```bash
# Manual scaling
kubectl scale deployment/edwin-api --replicas=10 -n edwin

# Or edit HPA
kubectl edit hpa edwin-api-hpa -n edwin
```

---

## Troubleshooting

### Services not starting
```bash
# Check logs
docker-compose -f docker-compose.edwin.yml logs -f

# Restart services
docker-compose -f docker-compose.edwin.yml restart

# Clean restart
docker-compose -f docker-compose.edwin.yml down
docker-compose -f docker-compose.edwin.yml up -d
```

### Kubernetes pods crashing
```bash
# Check pod status
kubectl get pods -n edwin

# View logs
kubectl logs -f deployment/edwin-api -n edwin

# Describe pod
kubectl describe pod edwin-api-xxx -n edwin

# Delete and recreate
kubectl delete pod edwin-api-xxx -n edwin
```

### Database connection errors
```bash
# Check database pods
kubectl get pods -n edwin -l tier=database

# Check logs
kubectl logs -f statefulset/neo4j -n edwin

# Restart databases
kubectl delete pod neo4j-0 -n edwin
```

### Authentication failing
```bash
# Check JWT secret is set
kubectl get secret edwin-secrets -n edwin -o yaml | grep JWT_SECRET_KEY

# Generate new secret
openssl rand -base64 32

# Update secret
kubectl delete secret edwin-secrets -n edwin
kubectl create secret generic edwin-secrets -n edwin \
  --from-literal=JWT_SECRET_KEY=<new-secret>
```

---

## Performance Optimization

### Enable Caching
```yaml
# config/production.yaml
features:
  caching: true

performance:
  cache_ttl_seconds: 3600
```

### Tune Connection Pools
```yaml
# config/production.yaml
database:
  neo4j:
    max_connection_pool_size: 100
  redis:
    max_connections: 200
```

### Enable Monitoring
```bash
# Start with monitoring
docker-compose -f docker-compose.edwin.yml --profile monitoring up -d
```

---

## Next Steps

1. **Review Documentation**
   - [EDWIN_PRODUCTION_DEPLOYMENT_COMPLETE.md](EDWIN_PRODUCTION_DEPLOYMENT_COMPLETE.md) - Complete guide
   - [DEPLOYMENT.md](EduVerse/edwin/DEPLOYMENT.md) - Detailed deployment
   - [SECURITY.md](EduVerse/edwin/SECURITY.md) - Security practices

2. **Configure Production**
   - Set up TLS certificates
   - Configure production domain
   - Set up Sentry error tracking
   - Configure Slack notifications

3. **Test Thoroughly**
   - Run all tests: `pytest EduVerse/edwin/tests/ -v`
   - Load testing
   - Security scanning

4. **Deploy to Production**
   - Deploy to staging first
   - Manual testing
   - Create release
   - Get approvals
   - Deploy to production

5. **Monitor & Optimize**
   - Watch Grafana dashboards
   - Review error rates
   - Optimize performance
   - Scale as needed

---

## Support

**Documentation**:
- [Production Deployment Guide](EDWIN_PRODUCTION_DEPLOYMENT_COMPLETE.md)
- [API Authentication Examples](EduVerse/edwin/API_AUTHENTICATION_EXAMPLE.md)
- [Kubernetes Architecture](EduVerse/edwin/KUBERNETES.md)

**Demo**:
```bash
PYTHONPATH=. python demos/edwin_production_demo.py
```

**Health Check**:
```bash
./scripts/health_check.sh
```

---

**Status**: ✅ Production Ready
**Deployment Time**: 5-10 minutes
**Next**: Scale to production and integrate Agent B, C, D components

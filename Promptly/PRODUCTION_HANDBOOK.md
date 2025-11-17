# Promptly Production Handbook

**Deployment, Scaling, and Operations Guide**

---

## Table of Contents

1. [Deployment Architectures](#deployment-architectures)
2. [Scaling Strategies](#scaling-strategies)
3. [Security Hardening](#security-hardening)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Backup & Recovery](#backup--recovery)
6. [Performance Tuning](#performance-tuning)
7. [Cost Optimization](#cost-optimization)

---

## Deployment Architectures

### Single Server (Small Scale)

```
┌─────────────────────────────────────┐
│       Single Server (4GB RAM)       │
├─────────────────────────────────────┤
│  Promptly API (uvicorn)             │
│  PostgreSQL                          │
│  Redis (cache)                       │
│  Nginx (reverse proxy)               │
└─────────────────────────────────────┘
```

**Setup:**
```bash
# Install dependencies
sudo apt update
sudo apt install postgresql redis-server nginx python3-pip

# Setup database
sudo -u postgres createdb promptly
sudo -u postgres createuser promptly_user

# Install Promptly
pip install promptly[all]

# Configure
cat > .env << EOF
PROMPTLY_DB_URL=postgresql://promptly_user:pass@localhost/promptly
PROMPTLY_REDIS_URL=redis://localhost:6379
PROMPTLY_SECRET_KEY=$(openssl rand -hex 32)
EOF

# Run API
uvicorn promptly.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### High Availability (Medium Scale)

```
              ┌──────────────┐
              │ Load Balancer │
              └──────┬───────┘
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │ API-1   │ │ API-2   │ │ API-3   │
    └────┬────┘ └────┬────┘ └────┬────┘
         └───────────┼───────────┘
                     │
         ┌───────────┼───────────┐
         │           │           │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │PostgreSQL│ │  Redis  │ │Analytics│
    │(Primary) │ │ Cluster │ │   DB    │
    └─────────┘ └─────────┘ └─────────┘
```

**Docker Compose:**
```yaml
version: '3.8'

services:
  api:
    image: promptly:latest
    deploy:
      replicas: 3
    environment:
      - PROMPTLY_DB_URL=postgresql://user:pass@postgres/promptly
      - PROMPTLY_REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=promptly
      - POSTGRES_USER=promptly_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - api

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes (Enterprise Scale)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: promptly-api
spec:
  replicas: 5
  selector:
    matchLabels:
      app: promptly-api
  template:
    metadata:
      labels:
        app: promptly-api
    spec:
      containers:
      - name: api
        image: promptly:1.0.0
        env:
        - name: PROMPTLY_DB_URL
          valueFrom:
            secretKeyRef:
              name: promptly-secrets
              key: db-url
        - name: PROMPTLY_REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: promptly-config
              key: redis-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
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
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: promptly-api
spec:
  selector:
    app: promptly-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Scaling Strategies

### Horizontal Scaling

**API Layer:**
```bash
# Add more API workers
uvicorn promptly.api.main:app --workers 8

# Or scale in Kubernetes
kubectl scale deployment promptly-api --replicas=10
```

**Database:**
```yaml
# PostgreSQL Read Replicas
postgresql:
  primary:
    url: postgresql://primary:5432/promptly
  replicas:
    - postgresql://replica1:5432/promptly
    - postgresql://replica2:5432/promptly

# Route reads to replicas
read_operations:
  - list_prompts
  - get_prompt
  - search_prompts
write_operations:
  - create_prompt
  - update_prompt
```

### Vertical Scaling

**Resource Limits:**
```python
# config.yaml
performance:
  max_workers: 16
  worker_connections: 1000
  timeout: 300
  max_prompt_size_mb: 10
  max_batch_size: 1000

database:
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  pool_recycle: 3600

cache:
  redis_max_connections: 50
  cache_ttl: 3600
  max_cache_size_mb: 1000
```

### Caching Strategy

```python
# Multi-layer caching
from promptly.cache import CacheLayer

cache = CacheLayer(
    layers=[
        ('memory', {'max_size_mb': 100, 'ttl': 60}),
        ('redis', {'ttl': 3600}),
        ('disk', {'ttl': 86400})
    ]
)

# Cache frequently accessed prompts
@cache.cached(layer='memory', ttl=60)
def get_prompt(name, version):
    return db.query_prompt(name, version)
```

---

## Security Hardening

### API Security

**1. API Key Management:**
```python
# Generate secure API keys
from secrets import token_urlsafe

api_key = f"pk_live_{token_urlsafe(32)}"

# Store hashed
import hashlib
key_hash = hashlib.sha256(api_key.encode()).hexdigest()

# Rotate keys regularly (90 days)
# Implement key expiration
# Use different keys for different environments
```

**2. Rate Limiting:**
```python
# config.yaml
security:
  rate_limits:
    default: 60/minute
    authenticated: 600/minute
    premium: 6000/minute

  burst_limits:
    default: 10
    authenticated: 100
    premium: 1000

  ip_whitelist:
    - 10.0.0.0/8
    - 172.16.0.0/12

  ip_blacklist:
    - 192.0.2.0/24
```

**3. HTTPS/TLS:**
```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.promptly.dev;

    ssl_certificate /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # HSTS
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
    }
}
```

**4. Input Validation:**
```python
# All endpoints use Pydantic validation
from pydantic import BaseModel, Field, validator

class CreatePromptRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, regex=r'^[a-z_][a-z0-9_]*$')
    content: str = Field(..., min_length=1, max_length=100000)
    metadata: dict = Field(default_factory=dict)

    @validator('content')
    def validate_content(cls, v):
        # Sanitize content
        # Check for injection attacks
        # Validate template syntax
        return v
```

### Database Security

```sql
-- Create read-only user for analytics
CREATE USER promptly_readonly WITH PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO promptly_readonly;

-- Encrypt sensitive columns
CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE prompts ADD COLUMN encrypted_metadata BYTEA;

UPDATE prompts SET encrypted_metadata = pgp_sym_encrypt(
    metadata::text,
    current_setting('app.encryption_key')
);

-- Regular backups with encryption
pg_dump promptly | gpg --encrypt --recipient admin@promptly.dev > backup.sql.gpg
```

---

## Monitoring & Alerting

### Metrics Collection

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
requests_total = Counter(
    'promptly_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'promptly_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

# Business metrics
prompts_total = Gauge('promptly_prompts_total', 'Total prompts')
evaluations_total = Counter('promptly_evaluations_total', 'Total evaluations')
eval_score = Histogram('promptly_eval_score', 'Evaluation scores')

# Database metrics
db_connections = Gauge('promptly_db_connections', 'Database connections')
db_query_duration = Histogram('promptly_db_query_duration_seconds', 'Query duration')
```

### Health Checks

```python
# Advanced health check endpoint
from fastapi import APIRouter

router = APIRouter()

@router.get("/health/detailed")
async def detailed_health():
    checks = {
        'database': await check_database(),
        'redis': await check_redis(),
        'disk_space': check_disk_space(),
        'memory': check_memory(),
        'api_responsiveness': await check_api_responsiveness()
    }

    status = 'healthy' if all(c['status'] == 'ok' for c in checks.values()) else 'degraded'

    return {
        'status': status,
        'checks': checks,
        'timestamp': datetime.utcnow()
    }
```

### Alerting Rules

```yaml
# alertmanager.yml
groups:
  - name: promptly_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(promptly_requests_total{status="500"}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} req/s"

      - alert: SlowAPIResponses
        expr: histogram_quantile(0.95, promptly_request_duration_seconds) > 2
        for: 10m
        annotations:
          summary: "Slow API responses"

      - alert: DatabaseConnectionPoolExhausted
        expr: promptly_db_connections / promptly_db_connections_max > 0.9
        for: 5m

      - alert: LowEvaluationScores
        expr: avg(promptly_eval_score) < 0.5
        for: 30m
```

### Dashboards

**Grafana Dashboard JSON:**
```json
{
  "dashboard": {
    "title": "Promptly Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(promptly_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(promptly_requests_total{status=~\"5..\"}[5m])"
          }
        ]
      },
      {
        "title": "P95 Response Time",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, promptly_request_duration_seconds)"
          }
        ]
      }
    ]
  }
}
```

---

## Backup & Recovery

### Automated Backups

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR=/backups/promptly
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
pg_dump promptly | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Prompt files backup
tar -czf $BACKUP_DIR/prompts_$DATE.tar.gz .promptly/prompts/

# Encrypt
gpg --encrypt --recipient backup@promptly.dev $BACKUP_DIR/db_$DATE.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/ s3://promptly-backups/$DATE/ --recursive

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -mtime +30 -delete

# Verify backup
pg_restore --list $BACKUP_DIR/db_$DATE.sql.gz > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Backup verified successfully"
else
    echo "Backup verification failed!" | mail -s "Backup Alert" admin@promptly.dev
fi
```

### Disaster Recovery

```bash
# Recovery procedure
#!/bin/bash

# 1. Stop services
systemctl stop promptly-api

# 2. Restore database
gunzip < backup.sql.gz | psql promptly

# 3. Restore files
tar -xzf prompts_backup.tar.gz -C /var/lib/promptly/

# 4. Verify integrity
python -m promptly.scripts.verify_integrity

# 5. Start services
systemctl start promptly-api

# 6. Run health checks
curl http://localhost:8000/health/detailed
```

---

## Performance Tuning

### Database Optimization

```sql
-- Indexes
CREATE INDEX CONCURRENTLY idx_prompts_name_version ON prompts(name, version);
CREATE INDEX CONCURRENTLY idx_prompts_branch_created ON prompts(branch, created_at DESC);

-- Partitioning (for large tables)
CREATE TABLE prompts_2025_q1 PARTITION OF prompts
    FOR VALUES FROM ('2025-01-01') TO ('2025-04-01');

-- Vacuum and analyze
VACUUM ANALYZE prompts;

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### Application Tuning

```python
# config.py
PERFORMANCE_CONFIG = {
    'uvicorn': {
        'workers': cpu_count() * 2,
        'worker_class': 'uvicorn.workers.UvicornWorker',
        'timeout': 300,
        'keepalive': 5,
        'max_requests': 1000,
        'max_requests_jitter': 100
    },

    'database': {
        'pool_pre_ping': True,
        'pool_recycle': 3600,
        'echo': False,
        'pool_size': 20,
        'max_overflow': 40
    },

    'cache': {
        'backend': 'redis',
        'default_timeout': 3600,
        'key_prefix': 'promptly:',
        'compression': True
    }
}
```

---

## Cost Optimization

### Resource Usage

```yaml
# Optimize by environment
development:
  api_workers: 2
  db_instance: t3.small
  redis_instance: t3.micro
  estimated_cost: $50/month

staging:
  api_workers: 4
  db_instance: t3.medium
  redis_instance: t3.small
  estimated_cost: $150/month

production:
  api_workers: 8
  db_instance: r5.large
  redis_instance: r5.medium
  estimated_cost: $500/month
```

### Cost Monitoring

```python
# Track costs by operation
from promptly.analytics import CostTracker

tracker = CostTracker()

@tracker.track_cost(operation='prompt_creation', cost_per_call=0.001)
def create_prompt(...):
    ...

@tracker.track_cost(operation='evaluation', cost_per_call=0.01)
def run_evaluation(...):
    ...

# Monthly cost report
report = tracker.generate_monthly_report()
# {
#   'total_cost': 450.00,
#   'by_operation': {
#     'prompt_creation': 50.00,
#     'evaluation': 400.00
#   }
# }
```

---

**For more details**, see:
- GETTING_STARTED_GUIDE.md - Setup instructions
- API_COMPLETE_REFERENCE.md - API documentation
- COMPLETE_FEATURE_GUIDE.md - Feature details

**Production Support:** production@promptly.dev

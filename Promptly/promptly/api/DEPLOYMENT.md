# Promptly API Deployment Guide

Complete guide for deploying Promptly API in production.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- Python 3.11 or higher
- Docker and Docker Compose (for containerized deployment)
- 2GB RAM minimum, 4GB recommended
- 10GB disk space

### Required Software

```bash
# Python
python --version  # Should be 3.11+

# Docker (optional)
docker --version
docker-compose --version

# Git
git --version
```

## Local Development

### 1. Clone Repository

```bash
git clone <repository-url>
cd Promptly/promptly
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
cd api
pip install -r requirements.txt
```

### 4. Initialize Promptly

```bash
# Create data directory
mkdir -p ./data

# Initialize Promptly repository
cd data
python -c "from promptly import Promptly; Promptly().init()"
cd ..
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

### 6. Run Development Server

```bash
# Run with auto-reload
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or use the development command
uvicorn api.main:app --reload
```

Visit http://localhost:8000/docs for API documentation.

## Docker Deployment

### 1. Build Docker Image

```bash
# From Promptly/promptly directory
docker build -f api/Dockerfile -t promptly-api:latest .
```

### 2. Run with Docker Compose

```bash
cd api

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### 3. Access Services

- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Nginx: http://localhost (if enabled)

## Production Deployment

### 1. Security Configuration

**Generate Secret Key:**

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Update `.env`:

```bash
PROMPTLY_SECRET_KEY=<generated-key>
PROMPTLY_WORKERS=4
PROMPTLY_LOG_LEVEL=WARNING
PROMPTLY_RELOAD=false
```

**Create API Keys:**

```bash
# Using the API
curl -X POST http://localhost:8000/api/v1/auth/api-keys \
  -H "X-API-Key: pk_dev_key_12345" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "production-key",
    "scopes": ["prompts:read", "prompts:write"],
    "expires_in_days": 90
  }'
```

### 2. SSL/TLS Configuration

**Using Let's Encrypt:**

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot --nginx -d your-domain.com

# Auto-renewal
crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

**Update Nginx Configuration:**

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # ... rest of configuration
}
```

### 3. Systemd Service

Create `/etc/systemd/system/promptly-api.service`:

```ini
[Unit]
Description=Promptly API
After=network.target

[Service]
Type=notify
User=promptly
Group=promptly
WorkingDirectory=/opt/promptly
Environment="PATH=/opt/promptly/venv/bin"
ExecStart=/opt/promptly/venv/bin/uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
systemctl daemon-reload
systemctl enable promptly-api
systemctl start promptly-api
systemctl status promptly-api
```

### 4. Database Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/promptly"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup Promptly data
tar -czf "${BACKUP_DIR}/promptly_${DATE}.tar.gz" /opt/promptly/data

# Cleanup old backups (keep 7 days)
find "${BACKUP_DIR}" -name "promptly_*.tar.gz" -mtime +7 -delete
```

Schedule with cron:

```bash
crontab -e
# Add: 0 2 * * * /opt/promptly/backup.sh
```

## Cloud Deployment

### AWS (EC2 + Docker)

```bash
# 1. Launch EC2 instance (Ubuntu 22.04, t3.medium)

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Clone and deploy
git clone <repo>
cd Promptly/promptly/api
docker-compose -f docker-compose.prod.yml up -d

# 4. Configure security groups
# Allow: 80, 443, 8000
```

### GCP (Cloud Run)

```bash
# 1. Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/promptly-api

# 2. Deploy to Cloud Run
gcloud run deploy promptly-api \
  --image gcr.io/PROJECT_ID/promptly-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars PROMPTLY_SECRET_KEY=<key>
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: promptly-api
spec:
  replicas: 3
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
        image: promptly-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: PROMPTLY_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: promptly-secrets
              key: secret-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: promptly-api
spec:
  selector:
    app: promptly-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:

```bash
kubectl apply -f deployment.yaml
```

## Monitoring

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Docker health
docker ps --filter "name=promptly-api" --format "table {{.Names}}\t{{.Status}}"
```

### Logging

**View logs:**

```bash
# Docker
docker-compose logs -f api

# Systemd
journalctl -u promptly-api -f

# Log file
tail -f /var/log/promptly/api.log
```

**Log rotation:**

```
# /etc/logrotate.d/promptly
/var/log/promptly/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0640 promptly promptly
    sharedscripts
    postrotate
        systemctl reload promptly-api
    endscript
}
```

### Metrics

**Prometheus configuration:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'promptly-api'
    static_configs:
      - targets: ['localhost:9090']
```

## Troubleshooting

### Common Issues

**Port already in use:**

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

**Permission denied:**

```bash
# Fix ownership
chown -R promptly:promptly /opt/promptly

# Fix permissions
chmod -R 755 /opt/promptly
```

**Database locked:**

```bash
# Check for stale lock files
rm -f /opt/promptly/data/.promptly/*.db-wal
rm -f /opt/promptly/data/.promptly/*.db-shm
```

**Memory issues:**

```bash
# Reduce workers in .env
PROMPTLY_WORKERS=2

# Or limit container memory
docker update --memory 512m promptly-api
```

### Debug Mode

```bash
# Enable debug logging
export PROMPTLY_LOG_LEVEL=DEBUG

# Run with debug
uvicorn api.main:app --reload --log-level debug
```

### Performance Tuning

**Optimize workers:**

```bash
# Formula: (2 x CPU cores) + 1
PROMPTLY_WORKERS=5  # For 2 CPU cores
```

**Database optimization:**

```sql
-- SQLite optimization
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
```

**Nginx caching:**

```nginx
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m;

location /api/v1/prompts {
    proxy_cache api_cache;
    proxy_cache_valid 200 5m;
    proxy_pass http://promptly_api;
}
```

## Maintenance

### Update Deployment

```bash
# Pull latest code
git pull origin main

# Rebuild image
docker-compose build api

# Restart with zero downtime
docker-compose up -d --no-deps --build api
```

### Scaling

**Horizontal scaling with Docker Swarm:**

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml promptly

# Scale service
docker service scale promptly_api=5
```

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Use environment variables for secrets
- [ ] Enable HTTPS with valid certificate
- [ ] Configure firewall rules
- [ ] Use strong API keys
- [ ] Enable rate limiting
- [ ] Regular security updates
- [ ] Database backups
- [ ] Log monitoring
- [ ] Access control (CORS)

## Support

For issues and questions:
- GitHub Issues: <repository-url>/issues
- Documentation: http://localhost:8000/docs
- Email: support@promptly.example

## License

MIT License

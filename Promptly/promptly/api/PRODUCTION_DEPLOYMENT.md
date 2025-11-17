# Promptly API - Production Deployment Guide

Complete guide for deploying Promptly API in production using Docker Compose.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
  - [Environment Configuration](#environment-configuration)
  - [SSL Certificates](#ssl-certificates)
  - [Database Setup](#database-setup)
- [Deployment Scripts](#deployment-scripts)
- [Service Architecture](#service-architecture)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)
- [Security Checklist](#security-checklist)
- [Scaling Guide](#scaling-guide)

---

## Overview

The Promptly API production deployment consists of:

- **FastAPI Application**: Main API service
- **PostgreSQL**: Production database
- **Redis**: Caching and rate limiting
- **Nginx**: Reverse proxy with SSL termination

### Architecture Diagram

```
Internet
    ↓
[Nginx:443/80]  ← SSL Termination, Rate Limiting, Load Balancing
    ↓
[Promptly API:8000]  ← FastAPI Application
    ↓
[PostgreSQL:5432]  ← Primary Database
[Redis:6379]  ← Cache & Rate Limiting
```

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **RAM**: Minimum 4GB (8GB+ recommended)
- **Disk**: Minimum 20GB free space
- **CPU**: 2+ cores

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Git
- OpenSSL (for SSL certificates)
- curl (for testing)

### Installation

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version
```

---

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd Promptly/promptly/api
```

### 2. Configure Environment

```bash
# Copy production environment template
cp .env.production .env.production

# Generate secure secrets
echo "POSTGRES_PASSWORD=$(openssl rand -base64 32)" >> .env.production
echo "REDIS_PASSWORD=$(openssl rand -base64 32)" >> .env.production
echo "PROMPTLY_SECRET_KEY=$(openssl rand -hex 32)" >> .env.production

# Edit other settings
nano .env.production
```

### 3. Make Scripts Executable

```bash
chmod +x deploy.sh backup.sh monitor.sh logs.sh cleanup.sh
```

### 4. Deploy

```bash
./deploy.sh
```

That's it! Your API should now be running at `http://localhost` (or your domain).

---

## Detailed Setup

### Environment Configuration

Edit `.env.production` with your production values:

#### Required Settings

```bash
# Database - MUST be changed
POSTGRES_PASSWORD=<strong-random-password>

# Redis - MUST be changed
REDIS_PASSWORD=<strong-random-password>

# API Security - MUST be changed
PROMPTLY_SECRET_KEY=<64-character-hex-string>
```

#### CORS Configuration

```bash
# Add your frontend domains
PROMPTLY_CORS_ORIGINS=["https://yourdomain.com","https://www.yourdomain.com","https://app.yourdomain.com"]
```

#### Rate Limiting

```bash
# Requests per minute per IP
RATE_LIMIT_PER_MINUTE=100

# Maximum burst size
RATE_LIMIT_BURST=200
```

#### Workers

```bash
# Number of API workers (CPU cores * 2 + 1)
PROMPTLY_WORKERS=4
```

### SSL Certificates

#### Option 1: Let's Encrypt (Recommended for Production)

```bash
# Install certbot
sudo apt-get install certbot

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com

# Copy to ssl directory
mkdir -p ssl
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ssl/key.pem
sudo chown $(whoami):$(whoami) ssl/*
```

#### Option 2: Self-Signed (Development/Testing)

```bash
# The deploy.sh script will auto-generate self-signed certificates
# Or generate manually:
mkdir -p ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ssl/key.pem \
  -out ssl/cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=yourdomain.com"
```

#### SSL Renewal (Let's Encrypt)

```bash
# Auto-renewal cron job
sudo crontab -e

# Add this line:
0 0 1 * * certbot renew --quiet && cp /etc/letsencrypt/live/yourdomain.com/* /path/to/ssl/
```

### Database Setup

#### Initial Migration (if needed)

```bash
# Access database
docker-compose -f docker-compose.production.yml exec postgres psql -U promptly -d promptly

# Run migrations
# (Add your migration commands here)
```

#### Database Optimization

Create `init-db/01-optimize.sql`:

```sql
-- Performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET min_wal_size = '1GB';
ALTER SYSTEM SET max_wal_size = '4GB';
```

---

## Deployment Scripts

### deploy.sh - Full Deployment

```bash
# Standard deployment
./deploy.sh

# Force rebuild images
./deploy.sh --build

# Deploy without backup
./deploy.sh --no-backup

# Quick deploy (skip health checks)
./deploy.sh --quick
```

**What it does:**
1. Validates prerequisites
2. Checks environment configuration
3. Sets up SSL certificates
4. Creates required directories
5. Backs up existing database
6. Builds/pulls Docker images
7. Deploys services
8. Verifies health checks

### backup.sh - Database Backup

```bash
# Backup PostgreSQL
./backup.sh

# Full backup (PostgreSQL + Redis + app data)
./backup.sh --full

# Backup to custom location
./backup.sh --output /mnt/backups

# Keep backups for 7 days
./backup.sh --retain 7
```

**Automated Backups:**

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * cd /path/to/api && ./backup.sh --full
```

### monitor.sh - Health Monitoring

```bash
# One-time status check
./monitor.sh

# Continuous monitoring (5 second updates)
./monitor.sh --watch

# Detailed metrics
./monitor.sh --detailed

# JSON output (for monitoring tools)
./monitor.sh --json

# Enable alerts
./monitor.sh --watch --alerts
```

### logs.sh - Log Viewer

```bash
# View all logs
./logs.sh

# Follow API logs
./logs.sh api -f

# Last 50 PostgreSQL logs
./logs.sh postgres -n 50

# Show only errors
./logs.sh --errors

# View logs from last hour
./logs.sh api --since 1h

# Export logs to file
./logs.sh api --export api.log

# Interactive mode
./logs.sh --interactive
```

### cleanup.sh - Shutdown and Cleanup

```bash
# Stop and remove containers
./cleanup.sh

# Just stop services
./cleanup.sh --stop-only

# Remove everything including data
./cleanup.sh --full-cleanup

# Backup before cleanup
./cleanup.sh --backup-first --remove-volumes

# Show current status
./cleanup.sh --status
```

---

## Service Architecture

### Service Details

#### Promptly API
- **Port**: 8000 (internal)
- **Health Check**: `/health`
- **Workers**: Configurable (default: 4)
- **Logs**: `./logs/api.log`

#### PostgreSQL
- **Port**: 5432 (localhost only)
- **Database**: promptly
- **Data**: Persistent volume
- **Backups**: `./backups/postgres_*.sql.gz`

#### Redis
- **Port**: 6379 (localhost only)
- **Password Protected**: Yes
- **Persistence**: AOF enabled
- **Data**: Persistent volume

#### Nginx
- **Ports**: 80 (HTTP), 443 (HTTPS)
- **SSL**: TLS 1.2/1.3
- **Rate Limiting**: Enabled
- **Logs**: `./nginx-logs/`

### Network Configuration

All services run on an isolated Docker network (`promptly-network`):

```
Subnet: 172.25.0.0/16
```

Only Nginx exposes ports to the host machine.

### Volume Mounts

```
postgres-data     → PostgreSQL database files
redis-data        → Redis persistence files
promptly-data     → Application data
nginx-cache       → Nginx cache
```

---

## Monitoring and Maintenance

### Health Checks

All services have health checks:

```bash
# Check service health
docker-compose -f docker-compose.production.yml ps

# Detailed health status
./monitor.sh
```

### Metrics

Access Prometheus metrics (from server):

```bash
curl http://localhost:9090/metrics
```

### Log Rotation

Configure log rotation in `/etc/logrotate.d/promptly`:

```
/path/to/api/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
}
```

### Performance Tuning

#### API Workers

```bash
# Set in .env.production
PROMPTLY_WORKERS=8  # (CPU cores * 2) + 1
```

#### Database Connections

Monitor connections:

```bash
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U promptly -d promptly \
  -c "SELECT count(*) FROM pg_stat_activity;"
```

#### Redis Memory

Monitor Redis memory:

```bash
docker-compose -f docker-compose.production.yml exec redis \
  redis-cli --pass $REDIS_PASSWORD INFO memory
```

---

## Backup and Recovery

### Backup Strategy

1. **Automatic Daily Backups**: Via cron
2. **Pre-Deployment Backups**: Automatic via deploy.sh
3. **Manual Backups**: As needed

### Backup Locations

```
./backups/
├── postgres_YYYYMMDD_HHMMSS.sql.gz
├── redis_YYYYMMDD_HHMMSS.rdb.gz
└── app_data_YYYYMMDD_HHMMSS.tar.gz
```

### Restore Database

```bash
# Stop services
./cleanup.sh --stop-only

# Restore PostgreSQL
gunzip -c backups/postgres_20231201_020000.sql.gz | \
  docker-compose -f docker-compose.production.yml exec -T postgres \
  psql -U promptly -d promptly

# Restart services
./deploy.sh
```

### Restore Redis

```bash
# Copy RDB file
gunzip -c backups/redis_20231201_020000.rdb.gz > /tmp/dump.rdb

# Copy to container
docker cp /tmp/dump.rdb promptly-redis:/data/dump.rdb

# Restart Redis
docker-compose -f docker-compose.production.yml restart redis
```

### Disaster Recovery

1. **Provision new server**
2. **Install prerequisites**
3. **Clone repository**
4. **Restore environment file**
5. **Restore backups**
6. **Deploy**

```bash
# Copy backups to new server
scp -r backups/ user@newserver:/path/to/api/

# On new server
./deploy.sh
./cleanup.sh --stop-only
# Restore database (as above)
./deploy.sh
```

---

## Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check logs
./logs.sh --errors

# Check Docker status
docker-compose -f docker-compose.production.yml ps

# Check system resources
df -h
free -m
docker system df
```

#### Database Connection Errors

```bash
# Check PostgreSQL is running
docker-compose -f docker-compose.production.yml ps postgres

# Check logs
./logs.sh postgres

# Verify credentials
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U promptly -d promptly -c "SELECT 1;"
```

#### Nginx 502 Bad Gateway

```bash
# Check API is running
docker-compose -f docker-compose.production.yml ps api

# Check API logs
./logs.sh api -f

# Test API directly
curl http://localhost:8000/health
```

#### SSL Certificate Issues

```bash
# Check certificate expiration
openssl x509 -in ssl/cert.pem -noout -dates

# Verify certificate
openssl verify ssl/cert.pem

# Test SSL
openssl s_client -connect yourdomain.com:443
```

#### High Memory Usage

```bash
# Check container stats
docker stats

# Check Redis memory
./monitor.sh --detailed

# Adjust Redis maxmemory in docker-compose.production.yml
```

### Debug Mode

Enable debug logging:

```bash
# Edit .env.production
PROMPTLY_LOG_LEVEL=DEBUG

# Restart
./deploy.sh
```

### Emergency Procedures

#### Complete Reset

```bash
# Backup first!
./backup.sh --full

# Full cleanup
./cleanup.sh --full-cleanup --force

# Redeploy
./deploy.sh --build
```

#### Rollback Deployment

```bash
# Stop current deployment
./cleanup.sh --stop-only

# Restore from backup
# (see Restore Database section)

# Redeploy
./deploy.sh
```

---

## Security Checklist

### Pre-Deployment

- [ ] Changed all default passwords
- [ ] Generated strong PROMPTLY_SECRET_KEY
- [ ] Configured proper CORS origins
- [ ] Set up SSL certificates
- [ ] Reviewed rate limiting settings
- [ ] Disabled unnecessary services
- [ ] Configured firewall rules

### Firewall Configuration

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

### Environment Security

```bash
# Secure environment file
chmod 600 .env.production

# Restrict script access
chmod 700 *.sh

# Set proper ownership
chown -R $(whoami):$(whoami) .
```

### Regular Security Tasks

- [ ] Rotate secrets monthly
- [ ] Update SSL certificates (Let's Encrypt auto-renews)
- [ ] Update Docker images weekly
- [ ] Review access logs for suspicious activity
- [ ] Check for security updates
- [ ] Review and update firewall rules

### Update Docker Images

```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Redeploy with new images
./deploy.sh
```

---

## Scaling Guide

### Vertical Scaling (Single Server)

#### Increase Resources

```bash
# Edit docker-compose.production.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

#### Increase Workers

```bash
# Edit .env.production
PROMPTLY_WORKERS=8

# Redeploy
./deploy.sh
```

### Horizontal Scaling (Multiple Servers)

#### Load Balancer Setup

Use external load balancer (AWS ALB, DigitalOcean LB, etc.) pointing to multiple API servers.

#### Database Replication

Set up PostgreSQL replication:

1. **Primary Server**: Current setup
2. **Replica Servers**: Read replicas
3. **Connection Pooling**: PgBouncer

#### Redis Cluster

For high availability:

1. **Redis Sentinel**: Automatic failover
2. **Redis Cluster**: Sharding

#### Example Multi-Server Architecture

```
              [Load Balancer]
                    |
      +-------------+-------------+
      |             |             |
   [Server 1]   [Server 2]   [Server 3]
      |             |             |
      +-------------+-------------+
                    |
          [PostgreSQL Primary]
                    |
          [PostgreSQL Replica]
                    |
            [Redis Cluster]
```

---

## Performance Benchmarks

### Expected Performance

- **API Response Time**: < 100ms (p95)
- **Throughput**: 1000+ requests/second (4 workers)
- **Database Connections**: 100+ concurrent
- **Memory Usage**: ~500MB - 2GB
- **CPU Usage**: 20-40% under normal load

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API
ab -n 10000 -c 100 http://localhost/health

# Test with authentication
ab -n 1000 -c 10 -H "X-API-Key: your-key" http://localhost/api/v1/prompts
```

---

## Support and Resources

### Documentation

- **API Docs**: https://your-domain.com/docs
- **ReDoc**: https://your-domain.com/redoc
- **OpenAPI**: https://your-domain.com/openapi.json

### Monitoring Tools

- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Sentry**: Error tracking
- **Datadog**: Full-stack monitoring

### Community

- **GitHub Issues**: Report bugs
- **Discord**: Community support
- **Documentation**: In-depth guides

---

## Appendix

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_PASSWORD` | - | PostgreSQL password (required) |
| `REDIS_PASSWORD` | - | Redis password (required) |
| `PROMPTLY_SECRET_KEY` | - | JWT secret (required) |
| `PROMPTLY_WORKERS` | 4 | Number of API workers |
| `RATE_LIMIT_PER_MINUTE` | 100 | Requests per minute |
| `RATE_LIMIT_BURST` | 200 | Maximum burst size |
| `PROMPTLY_CORS_ORIGINS` | [] | Allowed CORS origins |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | 60 | JWT expiration time |

### Useful Commands

```bash
# View all containers
docker ps -a

# View all volumes
docker volume ls

# View disk usage
docker system df

# Clean up unused resources
docker system prune -a

# Inspect container
docker inspect promptly-api

# View container resources
docker stats

# Execute command in container
docker-compose -f docker-compose.production.yml exec api bash
```

### Directory Structure

```
promptly/api/
├── docker-compose.production.yml  # Production compose file
├── .env.production                # Environment configuration
├── Dockerfile                     # API container definition
├── nginx.production.conf          # Nginx configuration
├── deploy.sh                      # Deployment script
├── backup.sh                      # Backup script
├── monitor.sh                     # Monitoring script
├── logs.sh                        # Log viewer
├── cleanup.sh                     # Cleanup script
├── ssl/                           # SSL certificates
├── backups/                       # Database backups
├── logs/                          # Application logs
├── nginx-logs/                    # Nginx logs
└── init-db/                       # Database initialization
```

---

## Changelog

### Version 1.0.0 (2024-01)
- Initial production deployment setup
- Docker Compose configuration
- Deployment automation scripts
- Monitoring and logging
- Backup and recovery procedures

---

## License

Copyright © 2024 Promptly. All rights reserved.

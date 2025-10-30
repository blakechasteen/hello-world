# HoloLoom Production Deployment Guide

**Complete Docker deployment with monitoring**

See full guide at: [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md)

## Quick Start

```bash
# Deploy production stack
docker-compose -f docker-compose.production.yml up -d

# Deploy with monitoring
docker-compose -f docker-compose.production.yml --profile monitoring up -d

# Check health
docker-compose ps
curl http://localhost:8000/health
```

## Services

- **Neo4j**: http://localhost:7474
- **Qdrant**: http://localhost:6333/dashboard  
- **HoloLoom**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Next Steps

1. Change default passwords
2. Set up backups
3. Configure monitoring alerts


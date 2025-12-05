# Elle Tavern Demo - Deployment Guide

**Version**: 1.0.0
**Created**: 2025-11-16
**Status**: Production Ready

Complete deployment guide for The Rusty Mug Tavern - an Elle Game Engine demonstration.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Local Development](#local-development)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Environment Variables](#environment-variables)
8. [Cost Estimation](#cost-estimation)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required
- **Python 3.10+**
- **pip** (Python package manager)

### Optional
- **Docker** (for containerized deployment)
- **Node.js 18+** (for frontend build optimization)
- **LLM API Keys** (OpenAI, Anthropic, or local Ollama)

---

## Quick Start

**One-command startup**:

```bash
cd /home/user/hello-world/games/elle_tavern_demo
python run_demo.py
```

This will:
1. Check if Elle service is running
2. Start Elle service if needed
3. Start game server
4. Open browser to http://localhost:8001

---

## Local Development

### Step 1: Install Dependencies

```bash
# Navigate to repository root
cd /home/user/hello-world

# Install Elle Game Engine dependencies
cd apps/elle_game_engine
pip install -r requirements.txt

# Install game demo dependencies
cd ../../games/elle_tavern_demo
pip install fastapi uvicorn pydantic aiohttp httpx pytest pytest-asyncio
```

### Step 2: Start Elle Service

```bash
# Terminal 1: Elle Game Engine Service
cd /home/user/hello-world/apps/elle_game_engine

# Option A: Use dummy LLM (free, for testing)
export ELLE_LLM_PROVIDER=dummy
uvicorn service:app --reload --port 8000

# Option B: Use real LLM (requires API key)
export ELLE_LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
uvicorn service:app --reload --port 8000

# Option C: Use Anthropic Claude
export ELLE_LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key_here
uvicorn service:app --reload --port 8000
```

### Step 3: Start Game Server

```bash
# Terminal 2: Game Demo Server
cd /home/user/hello-world/games/elle_tavern_demo
export ELLE_API_URL=http://localhost:8000
uvicorn server:app --reload --port 8001
```

### Step 4: Open Browser

Navigate to: **http://localhost:8001**

---

## Production Deployment

### Option 1: Traditional Server

#### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- nginx (reverse proxy)
- systemd (service management)

#### 1. Install Dependencies

```bash
sudo apt update
sudo apt install python3.10 python3-pip nginx

# Install Python dependencies
pip3 install -r apps/elle_game_engine/requirements.txt
pip3 install fastapi uvicorn pydantic aiohttp httpx
```

#### 2. Create Systemd Services

**Elle Service** (`/etc/systemd/system/elle-service.service`):

```ini
[Unit]
Description=Elle Game Engine Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/hello-world/apps/elle_game_engine
Environment="ELLE_LLM_PROVIDER=openai"
Environment="OPENAI_API_KEY=your_key_here"
Environment="ELLE_ENABLE_POOL=true"
Environment="ELLE_POOL_SIZE=10"
ExecStart=/usr/local/bin/uvicorn service:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Game Server** (`/etc/systemd/system/tavern-game.service`):

```ini
[Unit]
Description=Elle Tavern Demo Server
After=network.target elle-service.service
Requires=elle-service.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/hello-world/games/elle_tavern_demo
Environment="ELLE_API_URL=http://localhost:8000"
ExecStart=/usr/local/bin/uvicorn server:app --host 0.0.0.0 --port 8001
Restart=always

[Install]
WantedBy=multi-user.target
```

#### 3. Configure nginx

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /elle/ {
        proxy_pass http://localhost:8000/elle/;
        proxy_set_header Host $host;
    }
}
```

#### 4. Start Services

```bash
sudo systemctl daemon-reload
sudo systemctl enable elle-service
sudo systemctl enable tavern-game
sudo systemctl start elle-service
sudo systemctl start tavern-game
sudo systemctl restart nginx
```

#### 5. Verify

```bash
curl http://localhost:8000/health  # Elle service
curl http://localhost:8001/api/health  # Game server
```

---

## Docker Deployment

### Quick Start

```bash
cd /home/user/hello-world/games/elle_tavern_demo
docker-compose up -d
```

Access at: **http://localhost:8001**

### docker-compose.yml

See `docker-compose.yml` in this directory.

### Build Custom Images

```bash
# Build Elle service
cd apps/elle_game_engine
docker build -t elle-service:latest .

# Build game demo
cd ../../games/elle_tavern_demo
docker build -t tavern-demo:latest .
```

### Production Docker Deployment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Cloud Deployment

### Railway.app

1. **Install Railway CLI**:
```bash
npm install -g @railway/cli
railway login
```

2. **Deploy**:
```bash
cd /home/user/hello-world
railway init
railway up
```

3. **Set Environment Variables**:
```bash
railway variables set ELLE_LLM_PROVIDER=openai
railway variables set OPENAI_API_KEY=your_key_here
```

### Heroku

1. **Create Procfile**:
```
web: cd games/elle_tavern_demo && uvicorn server:app --host 0.0.0.0 --port $PORT
worker: cd apps/elle_game_engine && uvicorn service:app --host 0.0.0.0 --port 8000
```

2. **Deploy**:
```bash
heroku create your-app-name
git push heroku main
heroku ps:scale web=1 worker=1
```

### Google Cloud Run

1. **Build Container**:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/tavern-demo
```

2. **Deploy**:
```bash
gcloud run deploy tavern-demo \
  --image gcr.io/YOUR_PROJECT/tavern-demo \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### AWS Elastic Beanstalk

1. **Install EB CLI**:
```bash
pip install awsebcli
```

2. **Initialize**:
```bash
eb init -p python-3.10 tavern-demo
```

3. **Deploy**:
```bash
eb create tavern-demo-env
eb deploy
```

---

## Environment Variables

### Elle Service Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ELLE_LLM_PROVIDER` | `dummy` | LLM provider: `dummy`, `openai`, `anthropic`, `local` |
| `ELLE_VOICE_BACKEND` | `dummy` | Voice backend: `dummy`, `openai`, `elevenlabs` |
| `OPENAI_API_KEY` | - | OpenAI API key (if using OpenAI) |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (if using Anthropic) |
| `ELLE_ENABLE_POOL` | `false` | Enable connection pooling |
| `ELLE_POOL_SIZE` | `10` | Connection pool size |
| `ELLE_CACHE_SIZE` | `1000` | Response cache size |
| `ELLE_ENABLE_METRICS` | `true` | Enable Prometheus metrics |

### Game Server Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ELLE_API_URL` | `http://localhost:8000` | Elle service URL |
| `PORT` | `8001` | Game server port |
| `LOG_LEVEL` | `info` | Logging level |

---

## Cost Estimation

### Using OpenAI (gpt-4o-mini + tts-1)

**Per Player Session (1 hour)**:
- Dialogue interactions: ~20-30 requests
- Average tokens per request: ~500 tokens
- Total tokens: ~10,000-15,000 tokens
- Cost: **$0.015 - $0.025 per session**

**Voice Synthesis**:
- Average dialogue responses: 20-30
- Cost per 1000 characters: $0.015
- Cost: **$0.01 - $0.02 per session**

**Total per player session**: **$0.025 - $0.045**

**100 concurrent players**: **$2.50 - $4.50/hour** = **$60 - $108/day**

### Using Anthropic (claude-3-5-sonnet)

**Per Player Session (1 hour)**:
- Dialogue interactions: ~20-30 requests
- Average tokens: ~500 tokens per request
- Total tokens: ~10,000-15,000 tokens
- Cost: **$0.05 - $0.075 per session**

**100 concurrent players**: **$5 - $7.50/hour** = **$120 - $180/day**

### Using Local Ollama (Free)

- **$0** - Completely free
- Requires GPU server
- Server costs: ~$0.50/hour (cloud GPU) or free (own hardware)

### Recommended Production Setup

**Hybrid Approach**:
- Dummy LLM for testing/development: **$0**
- OpenAI for production (cost-effective): **$60-100/day** for 100 players
- Anthropic for premium experience: **$120-180/day** for 100 players

---

## Troubleshooting

### Elle Service Not Starting

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Check logs
tail -f /var/log/elle-service.log
```

### Game Server 500 Errors

```bash
# Check Elle service health
curl http://localhost:8000/health

# Check game server logs
journalctl -u tavern-game -f

# Test Elle API directly
curl -X POST http://localhost:8000/elle/game/action \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### NPCs Not Responding

1. **Check Elle service is running**:
```bash
curl http://localhost:8000/health
```

2. **Check LLM provider configuration**:
```bash
echo $ELLE_LLM_PROVIDER
echo $OPENAI_API_KEY  # Should not be empty if using OpenAI
```

3. **Check game logs**:
```bash
tail -f games/elle_tavern_demo/logs/game.log
```

### Quest Not Unlocking

- **Check prerequisites**: Use `python integration.py` to test quest system
- **Check NPC emotions**: Quests have emotional requirements
- **Check player level**: Some quests require minimum level

### Performance Issues

1. **Enable connection pooling**:
```bash
export ELLE_ENABLE_POOL=true
export ELLE_POOL_SIZE=20
```

2. **Increase cache size**:
```bash
export ELLE_CACHE_SIZE=5000
```

3. **Use faster LLM**:
```bash
# Switch from gpt-4o to gpt-4o-mini
export OPENAI_MODEL=gpt-4o-mini
```

---

## Testing Deployment

### Run Integration Tests

```bash
cd /home/user/hello-world/games/elle_tavern_demo
pytest tests/test_e2e.py -v
```

### Run Load Test

```bash
cd /home/user/hello-world/apps/elle_game_engine
python load_test.py --url http://localhost:8000 --users 10 --duration 60
```

### Manual Health Checks

```bash
# Elle service
curl http://localhost:8000/health

# Game server
curl http://localhost:8001/api/health

# Full integration
python integration.py
```

---

## Production Checklist

- [ ] Environment variables configured
- [ ] LLM API keys set
- [ ] Services running (Elle + Game)
- [ ] nginx configured and running
- [ ] SSL certificates installed (Let's Encrypt)
- [ ] Logging configured
- [ ] Monitoring set up (Prometheus + Grafana)
- [ ] Backup strategy implemented
- [ ] Rate limiting configured
- [ ] Cost alerts set up
- [ ] Integration tests passing
- [ ] Load testing completed
- [ ] Documentation updated

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: `/apps/elle_game_engine/README.md`
- **API Docs**: http://localhost:8000/docs

---

**Last Updated**: 2025-11-16
**Maintainer**: Elle Game Engine Team

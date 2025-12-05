# Matrix Bot Deployment Guide

**Status**: Ready to Deploy
**Stack**: Matrix Synapse + PostgreSQL + Redis + HoloLoom + DSPy

---

## Overview

Complete chat-native AI reliability bot for Matrix with:
- **Matrix Integration**: Full Synapse homeserver + bot client
- **HoloLoom Backend**: 244D semantic space, Thompson Sampling, knowledge graphs
- **DSPy Optimization**: MIPROv2 prompt optimization
- **Team Collaboration**: Shared prompts, approval workflows, code review

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Matrix Client (Element/FluffyChat)       │
│                    https://matrix.org or localhost           │
└────────────────────────┬────────────────────────────────────┘
                         │ Matrix Protocol
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Matrix Synapse Homeserver                      │
│              Port 8008 (HTTP)                               │
│              PostgreSQL backend                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Promptly Matrix Bot                          │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Message Handler (@promptly mentions)                │  │
│  │  - Parse commands (optimize, run, code-review)       │  │
│  │  - Route to appropriate handler                      │  │
│  │  - Format responses (Markdown + HTML)                │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Promptly Core (HoloLoom + DSPy Integration)        │  │
│  │  - Prompt optimization (DSPy MIPROv2)                │  │
│  │  - Workflow execution (HoloLoom)                     │  │
│  │  - Knowledge graph memory                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  State Management (Redis)                            │  │
│  │  - Saved prompts per room                            │  │
│  │  - Approval workflow state                           │  │
│  │  - Team context                                      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **Docker & Docker Compose** installed
2. **OpenAI API key** (for DSPy optimization)
3. **4GB RAM minimum** (for PyTorch + embeddings)
4. **10GB disk space** (for Docker images + data)

---

## Quick Start (Local Testing)

### 1. Configure Environment

```bash
cd promptly-matrix-bot

# Copy .env.example to .env
cp .env.example .env

# Edit .env - add your OpenAI API key
nano .env
```

**Required variables**:
```bash
# OpenAI API (required for full functionality)
OPENAI_API_KEY=sk-your-key-here

# Matrix bot credentials (auto-generated on first run)
MATRIX_SERVER_NAME=matrix.localhost
MATRIX_BOT_PASSWORD=your-secure-password

# Database passwords
POSTGRES_PASSWORD=your-db-password
PROMPTLY_DB_PASSWORD=your-promptly-password
```

### 2. Start All Services

```bash
# Build and start
docker-compose up -d

# Check logs
docker-compose logs -f
```

**Services started**:
- PostgreSQL (port 5432)
- Redis (port 6379)
- Synapse (port 8008)
- Promptly Bot
- Promptly API (port 8000)

### 3. Create Bot User

**First time only** - register the bot user:

```bash
# Register bot user
docker exec -it promptly-synapse register_new_matrix_user \
  http://localhost:8008 \
  -c /data/homeserver.yaml \
  -u promptly \
  -p your-bot-password \
  -a

# Save credentials to .env
echo "MATRIX_BOT_PASSWORD=your-bot-password" >> .env
```

### 4. Restart Bot

```bash
# Restart with credentials
docker-compose restart promptly-bot

# Check bot logs
docker logs promptly-bot -f
```

**Expected**:
```
INFO - Logged in as @promptly:matrix.localhost
INFO - ✅ Promptly bot started and synced
INFO - Listening for messages...
```

### 5. Test with Element

1. **Open Element** in browser: https://app.element.io
2. **Connect to homeserver**: `http://localhost:8008`
3. **Register a test user** (or login)
4. **Create a room**
5. **Invite bot**: `/invite @promptly:matrix.localhost`
6. **Test command**: `@promptly help`

---

## Deployment Options

### Option A: Local Testing (Above)
**Use case**: Development, testing, proof of concept

**Pros**:
- Complete control
- No external dependencies
- Fast iteration

**Cons**:
- Not accessible externally
- Requires local resources

### Option B: Cloud Deployment

**Use case**: Team use, production

**Recommended**:
- **Railway** (easiest): One-click deploy
- **Fly.io**: Global edge deployment
- **Digital Ocean**: Traditional VPS

**Steps** (Railway example):
1. Fork repo to GitHub
2. Connect to Railway
3. Add environment variables
4. Deploy

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for detailed guides.

### Option C: Self-Hosted Production

**Use case**: Enterprise, on-premises

**Requirements**:
- Domain name
- SSL certificates (Let's Encrypt)
- Reverse proxy (Nginx/Caddy)
- Monitoring (Prometheus/Grafana)

See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for complete guide.

---

## Using the Bot

### Basic Commands

**Help**:
```
@promptly help
```

**Optimize a prompt**:
```
@promptly optimize
Task: Answer customer support questions
Examples: [
  {"input": "How do I reset my password?", "output": "Click Settings > Security > Reset Password"},
  {"input": "Where is my order?", "output": "Check the Orders page in your account"}
]
```

**Run a workflow**:
```
@promptly run qa_basic "What is Thompson Sampling?"
```

**Code review**:
```
@promptly code-review
```python
def login(user, password):
    query = f"SELECT * FROM users WHERE user='{user}'"
    return db.execute(query).fetchone()
```
```

**Save a prompt**:
```
@promptly save customer_support_v1
```

**List saved prompts**:
```
@promptly list
```

### Advanced Features

**Approval workflows** (requires setup):
```python
# In code - request team approval
from bot.approval_workflow import get_approval_manager, ActionRisk

manager = get_approval_manager(state, client)
request = await manager.request_approval(
    room_id=room.room_id,
    initiator=user_id,
    action="deploy_prompt",
    context={"prompt_name": "v2"},
    risk_level=ActionRisk.HIGH  # Requires 2 approvals
)
```

**Multi-step workflows**:
```python
# Create custom workflow
from bot.workflow_templates import create_deploy_prompt_workflow

workflow = create_deploy_prompt_workflow(
    prompt_name="my_prompt",
    task="Your task",
    examples=[...],
    room_id=room.room_id,
    initiator=user_id
)

# Execute
from bot.workflow_engine import get_workflow_engine
engine = get_workflow_engine(client, state, promptly_core)
result = await engine.execute(workflow)
```

---

## Configuration

### Environment Variables

**Matrix Configuration**:
```bash
MATRIX_HOMESERVER=http://synapse:8008  # Internal URL
MATRIX_SERVER_NAME=matrix.localhost    # Server name
MATRIX_USER_ID=@promptly:matrix.localhost
MATRIX_BOT_PASSWORD=your-password
```

**Database Configuration**:
```bash
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://promptly:password@postgres:5432/promptly
```

**LLM Configuration**:
```bash
OPENAI_API_KEY=sk-your-key
```

**Promptly Configuration**:
```bash
PROMPTLY_CONFIG=fused  # bare | fast | fused
LOG_LEVEL=INFO         # DEBUG | INFO | WARNING | ERROR
```

### HoloLoom Modes

**BARE** (fastest, ~50ms):
- Regex motif detection
- Single embedding scale (96D)
- Simple retrieval

**FAST** (balanced, ~150ms):
- Hybrid motif detection
- Dual scales (96D + 192D)
- BM25 + semantic retrieval

**FUSED** (full quality, ~300ms):
- Full motif detection
- Triple scales (96D + 192D + 384D)
- Knowledge graph expansion
- Spectral features

---

## Monitoring

### Check Service Health

```bash
# All services
docker-compose ps

# Logs
docker-compose logs -f promptly-bot

# Individual service logs
docker logs promptly-synapse -f
docker logs promptly-postgres -f
docker logs promptly-redis -f
```

### Health Endpoints

**Synapse**:
```bash
curl http://localhost:8008/_matrix/client/versions
```

**Promptly API**:
```bash
curl http://localhost:8000/health
```

### Database Access

**PostgreSQL**:
```bash
docker exec -it promptly-postgres psql -U synapse
```

**Redis**:
```bash
docker exec -it promptly-redis redis-cli
```

---

## Troubleshooting

### Bot doesn't respond

**Check**:
1. Bot is running: `docker ps | grep promptly-bot`
2. Bot logged in successfully: `docker logs promptly-bot | grep "Logged in"`
3. Bot joined room: `/invite @promptly:matrix.localhost`
4. Mention syntax correct: `@promptly` (lowercase)

**Fix**:
```bash
# Restart bot
docker-compose restart promptly-bot

# Check logs for errors
docker logs promptly-bot --tail 50
```

### Integration not working (stub mode)

**Check**:
```bash
# Verify HoloLoom mount
docker exec promptly-bot ls -la /app/HoloLoom

# Check logs for import errors
docker logs promptly-bot 2>&1 | grep -i "error\|warning"
```

**Fix**:
```bash
# Rebuild with dependencies
docker-compose build promptly-bot
docker-compose up -d
```

### Synapse won't start

**Check**:
```bash
# PostgreSQL health
docker logs promptly-postgres

# Synapse config
docker exec promptly-synapse cat /data/homeserver.yaml
```

**Fix**:
```bash
# Reset database (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Out of memory

**Symptoms**:
- Bot crashes randomly
- `docker ps` shows bot as "Restarting"

**Fix**:
```bash
# Reduce to FAST mode
echo "PROMPTLY_CONFIG=fast" >> .env
docker-compose restart promptly-bot

# Or increase Docker memory limit
# Docker Desktop > Settings > Resources > Memory > 4GB+
```

---

## Scaling

### Horizontal Scaling

**Multiple bot instances** (for high load):
```yaml
# docker-compose.yml
services:
  promptly-bot-1:
    # ... config ...
  promptly-bot-2:
    # ... config ...
  promptly-bot-3:
    # ... config ...
```

**Load balancing**:
- Share Redis state
- Use PostgreSQL for persistence
- HAProxy/Nginx for distribution

### Vertical Scaling

**Resources per service**:
- Synapse: 1GB RAM + 1 CPU
- PostgreSQL: 1GB RAM + 1 CPU
- Redis: 256MB RAM
- Promptly Bot: 2-4GB RAM + 2 CPU

**Total recommended**: 4-6GB RAM, 4 CPUs

---

## Security

### Production Checklist

- [ ] Change default passwords in `.env`
- [ ] Use HTTPS (SSL certificates)
- [ ] Enable firewall (only expose 8008, 8000)
- [ ] Set up backups (PostgreSQL data)
- [ ] Configure rate limiting (Synapse)
- [ ] Enable monitoring/alerts
- [ ] Rotate API keys regularly
- [ ] Review Synapse security config

### Secrets Management

**Do NOT commit**:
- `.env` file
- `OPENAI_API_KEY`
- `MATRIX_BOT_PASSWORD`
- Database passwords

**Use**:
- Environment variables
- Docker secrets
- HashiCorp Vault
- Cloud provider secrets (AWS Secrets Manager, etc.)

---

## Backup & Recovery

### Backup

```bash
# PostgreSQL data
docker exec promptly-postgres pg_dump -U synapse synapse > synapse-backup.sql
docker exec promptly-postgres pg_dump -U promptly promptly > promptly-backup.sql

# Redis state
docker exec promptly-redis redis-cli --rdb /data/dump.rdb

# Copy volumes
docker run --rm -v promptly-data:/data -v $(pwd):/backup alpine \
  tar czf /backup/promptly-data-backup.tar.gz /data
```

### Restore

```bash
# PostgreSQL
cat synapse-backup.sql | docker exec -i promptly-postgres psql -U synapse synapse
cat promptly-backup.sql | docker exec -i promptly-postgres psql -U promptly promptly

# Redis
docker exec -i promptly-redis redis-cli --pipe < dump.rdb
```

---

## Next Steps

### Immediate
1. **Deploy locally** with Quick Start guide above
2. **Test basic commands** (`@promptly help`, optimize, run)
3. **Add team members** to test room

### Short Term
1. **Configure workflows** for your use case
2. **Set up approval workflows** for team collaboration
3. **Integrate with GitHub** (code review on PR)

### Long Term
1. **Deploy to cloud** for team access
2. **Set up monitoring** (Prometheus + Grafana)
3. **Add custom commands** for your domain
4. **Fine-tune HoloLoom** on your data

---

## Resources

- **Matrix Docs**: https://matrix.org/docs/
- **Synapse Docs**: https://element-hq.github.io/synapse/
- **Element Client**: https://app.element.io
- **HoloLoom Docs**: [../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md](../HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)
- **DSPy Docs**: https://dspy-docs.vercel.app/

---

## Support

- **Issues**: Check logs first (`docker logs promptly-bot`)
- **Documentation**: [README_API_INTEGRATION.md](README_API_INTEGRATION.md)
- **Matrix Room**: #promptly:matrix.org (after public launch)

---

**Status**: Ready to deploy
**Date**: November 8, 2025
**Stack**: Synapse 1.x + HoloLoom 1.0 + DSPy 3.0.3

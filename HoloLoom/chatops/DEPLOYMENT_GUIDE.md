# Matrix Bot Deployment Guide

Complete guide for deploying the HoloLoom Matrix bot in production.

---

## Quick Start (Local Testing)

### 1. Install Dependencies

```bash
pip install matrix-nio[e2e]
```

### 2. Set Environment Variables

```bash
export MATRIX_HOMESERVER="https://matrix.org"
export MATRIX_USER="@yourbot:matrix.org"
export MATRIX_PASSWORD="your_password"
```

### 3. Run Bot

```bash
python HoloLoom/chatops/run_bot.py
```

---

## Production Deployment

### Option 1: Systemd Service (Linux)

**1. Create service file:** `/etc/systemd/system/hololoom-bot.service`

```ini
[Unit]
Description=HoloLoom Matrix Bot
After=network.target

[Service]
Type=simple
User=hololoom
WorkingDirectory=/opt/hololoom
Environment="MATRIX_HOMESERVER=https://matrix.example.com"
Environment="MATRIX_USER=@bot:example.com"
Environment="MATRIX_ACCESS_TOKEN=your_token_here"
ExecStart=/opt/hololoom/.venv/bin/python HoloLoom/chatops/run_bot.py \
  --hololoom-mode fast \
  --mcts-sims 50 \
  --log-level INFO \
  --store-path /var/lib/hololoom/matrix_store
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**2. Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable hololoom-bot
sudo systemctl start hololoom-bot
sudo systemctl status hololoom-bot
```

**3. View logs:**

```bash
sudo journalctl -u hololoom-bot -f
```

### Option 2: Docker

**1. Create Dockerfile:**

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install matrix-nio[e2e] numba ripser persim rank-bm25

# Copy HoloLoom
COPY . .
RUN pip install -e .

# Run bot
CMD ["python", "HoloLoom/chatops/run_bot.py"]
```

**2. Build and run:**

```bash
docker build -t hololoom-bot .

docker run -d \
  --name hololoom-bot \
  --restart unless-stopped \
  -e MATRIX_HOMESERVER=https://matrix.org \
  -e MATRIX_USER=@bot:matrix.org \
  -e MATRIX_ACCESS_TOKEN=your_token \
  -v hololoom-data:/app/memory_data \
  -v hololoom-store:/app/matrix_store \
  hololoom-bot
```

**3. View logs:**

```bash
docker logs -f hololoom-bot
```

### Option 3: Docker Compose (Full Stack)

**Create `docker-compose.yml`:**

```yaml
version: '3.8'

services:
  hololoom-bot:
    build: .
    container_name: hololoom-bot
    restart: unless-stopped
    environment:
      - MATRIX_HOMESERVER=${MATRIX_HOMESERVER}
      - MATRIX_USER=${MATRIX_USER}
      - MATRIX_ACCESS_TOKEN=${MATRIX_TOKEN}
    volumes:
      - ./memory_data:/app/memory_data
      - ./matrix_store:/app/matrix_store
    command: >
      python HoloLoom/chatops/run_bot.py
      --hololoom-mode fast
      --mcts-sims 50
      --log-level INFO
    depends_on:
      - qdrant
      - neo4j

  # Optional: Qdrant for vector search
  qdrant:
    image: qdrant/qdrant:latest
    container_name: hololoom-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage

  # Optional: Neo4j for knowledge graph
  neo4j:
    image: neo4j:latest
    container_name: hololoom-neo4j
    environment:
      - NEO4J_AUTH=neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j-data:/data

volumes:
  qdrant-data:
  neo4j-data:
```

**Run:**

```bash
# Create .env file
cat > .env << EOF
MATRIX_HOMESERVER=https://matrix.org
MATRIX_USER=@bot:matrix.org
MATRIX_TOKEN=your_token_here
EOF

# Start services
docker-compose up -d

# View logs
docker-compose logs -f hololoom-bot
```

---

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MATRIX_HOMESERVER` | Matrix server URL | https://matrix.org | Yes |
| `MATRIX_USER` | Bot user ID | None | Yes |
| `MATRIX_PASSWORD` | Bot password | None | Yes* |
| `MATRIX_ACCESS_TOKEN` | Access token | None | Yes* |
| `MATRIX_ROOMS` | Comma-separated room IDs | None | No |

*Either password OR access token required

### Command Line Arguments

```bash
python run_bot.py --help
```

**Connection:**
- `--homeserver URL` - Matrix homeserver
- `--user USER_ID` - Bot user ID
- `--password PASS` - Bot password
- `--token TOKEN` - Access token

**Bot Behavior:**
- `--prefix PREFIX` - Command prefix (default: `!`)
- `--admin USER [USER ...]` - Admin users

**HoloLoom:**
- `--hololoom-mode {bare,fast,fused}` - Execution mode
- `--mcts-sims N` - MCTS simulations (default: 50)

**System:**
- `--log-level {DEBUG,INFO,WARNING,ERROR}` - Logging
- `--store-path PATH` - Encryption keys path

### Configuration File (Optional)

Create `bot_config.yaml`:

```yaml
matrix:
  homeserver: https://matrix.org
  user: "@bot:matrix.org"
  password: "secret"
  rooms:
    - "!room1:matrix.org"
    - "!room2:matrix.org"

bot:
  prefix: "!"
  admins:
    - "@alice:matrix.org"
    - "@bob:matrix.org"

hololoom:
  mode: fast
  mcts_simulations: 50

logging:
  level: INFO
```

---

## Getting Access Token

**Method 1: Python Script**

```python
from matrix_nio import AsyncClient
import asyncio

async def get_token():
    client = AsyncClient("https://matrix.org", "@yourbot:matrix.org")
    response = await client.login("your_password")

    if hasattr(response, "access_token"):
        print(f"Access token: {response.access_token}")
        print(f"Device ID: {response.device_id}")
    else:
        print(f"Login failed: {response}")

    await client.close()

asyncio.run(get_token())
```

**Method 2: Element Web**

1. Log in to Element Web as bot user
2. Settings → Help & About → Advanced
3. Copy Access Token

**Method 3: curl**

```bash
curl -X POST \
  https://matrix.org/_matrix/client/r0/login \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "m.login.password",
    "user": "yourbot",
    "password": "your_password"
  }'
```

---

## Testing

### Test Script

Create `test_bot.py`:

```python
#!/usr/bin/env python3
"""Test Matrix bot deployment"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.chatops.matrix_bot import MatrixBot, MatrixBotConfig
from HoloLoom.chatops.hololoom_handlers import HoloLoomMatrixHandlers


async def test_bot():
    """Test bot initialization and handlers"""

    print("1. Testing bot configuration...")
    config = MatrixBotConfig(
        homeserver_url="https://matrix.org",
        user_id="@test:matrix.org",
        password="test"
    )
    print("   [OK] Config created")

    print("\n2. Testing bot initialization...")
    bot = MatrixBot(config)
    print("   [OK] Bot initialized")

    print("\n3. Testing handler registration...")
    handlers = HoloLoomMatrixHandlers(bot, config_mode="bare")
    handlers.register_all()
    print(f"   [OK] Registered {len(bot.handlers)} handlers")
    print(f"   Commands: {', '.join(bot.handlers.keys())}")

    print("\n4. Testing HoloLoom initialization...")
    print(f"   [OK] Orchestrator: {handlers.orchestrator is not None}")

    print("\n[SUCCESS] All tests passed!")
    print("\nTo run live bot:")
    print("  python HoloLoom/chatops/run_bot.py --user @bot:matrix.org --password secret")


if __name__ == "__main__":
    asyncio.run(test_bot())
```

**Run test:**

```bash
python HoloLoom/chatops/test_bot.py
```

### Live Testing

1. **Start bot:**

```bash
python HoloLoom/chatops/run_bot.py \
  --user @testbot:matrix.org \
  --password test \
  --hololoom-mode bare \
  --log-level DEBUG
```

2. **Invite bot to room:**
   - Create test room in Element/Matrix client
   - Invite `@testbot:matrix.org`
   - Bot auto-accepts invites

3. **Test commands:**

```
!ping
!help
!weave What is MCTS?
!memory add MCTS is a tree search algorithm
!memory search MCTS
!stats
```

4. **Check logs:**

```bash
tail -f hololoom_bot.log
```

---

## Monitoring

### Health Check

Create `health_check.sh`:

```bash
#!/bin/bash
# Check if bot is responding

BOT_PID=$(pgrep -f "run_bot.py")

if [ -z "$BOT_PID" ]; then
    echo "CRITICAL: Bot not running"
    exit 2
fi

# Check log for recent activity
LAST_LOG=$(tail -1 hololoom_bot.log)
echo "OK: Bot running (PID: $BOT_PID)"
echo "Last log: $LAST_LOG"
```

### Prometheus Metrics (Future)

Add to `run_bot.py`:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
weaving_count = Counter('hololoom_weavings_total', 'Total weavings')
weaving_duration = Histogram('hololoom_weaving_duration_seconds', 'Weaving duration')

# Start metrics server
start_http_server(8000)
```

### Log Rotation

Create `/etc/logrotate.d/hololoom`:

```
/opt/hololoom/hololoom_bot.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 hololoom hololoom
}
```

---

## Troubleshooting

### Bot won't start

**Check credentials:**

```bash
python -c "
from matrix_nio import AsyncClient
import asyncio

async def test():
    client = AsyncClient('https://matrix.org', '@yourbot:matrix.org')
    resp = await client.login('your_password')
    print(resp)
    await client.close()

asyncio.run(test())
"
```

**Check network:**

```bash
curl https://matrix.org/_matrix/client/versions
```

### Commands not working

**Check prefix:**

```bash
# Try different prefixes
!help    # Default
/help    # Alternative
.help    # Alternative
```

**Check permissions:**

```bash
# In room, type:
!ping

# If no response, check:
# 1. Bot is in room
# 2. Bot has permission to send
# 3. Bot is running
# 4. Check logs
```

### Memory issues

**Check backend:**

```bash
ls -la memory_data/
# Should see memories.jsonl and embeddings.npy

# Check Neo4j
curl http://localhost:7474/

# Check Qdrant
curl http://localhost:6333/collections
```

**Reset memory:**

```bash
rm -rf memory_data/
# Bot will recreate on restart
```

### High CPU usage

**Reduce MCTS:**

```bash
python run_bot.py --mcts-sims 20  # Down from 50
```

**Use BARE mode:**

```bash
python run_bot.py --hololoom-mode bare
```

---

## Security

### Best Practices

1. **Use access tokens** instead of passwords
2. **Restrict admins** with `--admin` flag
3. **Enable E2EE** for encrypted rooms
4. **Rate limiting** is enabled by default (5 cmds/60s)
5. **Run as non-root** user
6. **Firewall** Matrix server if self-hosted

### Securing Access Token

**Use secrets management:**

```bash
# AWS Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id hololoom/matrix-token \
  --query SecretString \
  --output text

# Vault
vault kv get secret/hololoom/matrix

# Environment file (protect with chmod 600)
echo "MATRIX_TOKEN=your_token" > .env.secret
chmod 600 .env.secret
source .env.secret
```

---

## Performance Tuning

### HoloLoom Modes

| Mode   | Speed | Quality | RAM  | Use Case          |
|--------|-------|---------|------|-------------------|
| BARE   | ⚡⚡⚡ | ⭐⭐    | 200MB | High traffic      |
| FAST   | ⚡⚡   | ⭐⭐⭐  | 400MB | General use       |
| FUSED  | ⚡     | ⭐⭐⭐⭐ | 800MB | Complex queries   |

### MCTS Simulations

| Sims | Latency | Quality |
|------|---------|---------|
| 10   | ~20ms   | Good    |
| 50   | ~50ms   | Better  |
| 100  | ~100ms  | Best    |

### Memory Backends

**Fastest:** File-only (no external deps)
**Balanced:** File + Qdrant
**Complete:** File + Qdrant + Neo4j

---

## Backup & Recovery

### Backup

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/hololoom/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup memory
cp -r /opt/hololoom/memory_data "$BACKUP_DIR/"

# Backup Matrix store (encryption keys)
cp -r /opt/hololoom/matrix_store "$BACKUP_DIR/"

# Backup config
cp /opt/hololoom/bot_config.yaml "$BACKUP_DIR/"

echo "Backup complete: $BACKUP_DIR"
```

### Restore

```bash
#!/bin/bash
# restore.sh

BACKUP_DIR=$1

# Stop bot
sudo systemctl stop hololoom-bot

# Restore
cp -r "$BACKUP_DIR/memory_data" /opt/hololoom/
cp -r "$BACKUP_DIR/matrix_store" /opt/hololoom/
cp "$BACKUP_DIR/bot_config.yaml" /opt/hololoom/

# Fix permissions
chown -R hololoom:hololoom /opt/hololoom

# Restart
sudo systemctl start hololoom-bot
```

---

## Scaling

### Multiple Rooms

Bot automatically joins all invited rooms. No configuration needed.

### Multiple Bots

Run multiple instances with different users:

```bash
# Bot 1: General queries
python run_bot.py --user @bot1:matrix.org --hololoom-mode fast

# Bot 2: Research (high quality)
python run_bot.py --user @bot2:matrix.org --hololoom-mode fused --mcts-sims 100

# Bot 3: High volume (fast)
python run_bot.py --user @bot3:matrix.org --hololoom-mode bare --mcts-sims 10
```

### Load Balancing

Use multiple bots behind Matrix room alias:

```
#hololoom:matrix.org → Round-robin to @bot1, @bot2, @bot3
```

---

## Updating

### Update Bot Code

```bash
# Pull latest
cd /opt/hololoom
git pull

# Reinstall
pip install -e .

# Restart
sudo systemctl restart hololoom-bot
```

### Zero-Downtime Update

```bash
# Start new instance
python run_bot.py --user @bot-new:matrix.org &

# Migrate rooms (invite new bot)
# Test new bot
# Stop old bot

sudo systemctl stop hololoom-bot
# Update systemd to use new bot
sudo systemctl start hololoom-bot
```

---

## Support

### Logs

```bash
# Systemd
sudo journalctl -u hololoom-bot -f

# Docker
docker logs -f hololoom-bot

# File
tail -f hololoom_bot.log
```

### Debug Mode

```bash
python run_bot.py --log-level DEBUG
```

### Common Issues

See README.md Troubleshooting section.

---

## Production Checklist

- [ ] Access token configured (not password)
- [ ] Systemd service enabled
- [ ] Log rotation configured
- [ ] Backup script scheduled
- [ ] Health check monitoring
- [ ] Memory backend tested
- [ ] Rate limiting configured
- [ ] Admin users set
- [ ] Firewall rules applied
- [ ] SSL/TLS enabled on Matrix server

---

**Deployment Guide Complete!**

For more details, see:
- [README.md](README.md) - Full documentation
- [run_bot.py](run_bot.py) - Main launcher
- [hololoom_handlers.py](handlers/hololoom_handlers.py) - Command handlers

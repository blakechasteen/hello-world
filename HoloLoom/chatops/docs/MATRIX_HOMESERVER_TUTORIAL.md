# Installing a Local Matrix Homeserver

**Complete guide to self-hosting Matrix for HoloLoom ChatOps**

This tutorial will guide you through setting up your own Matrix homeserver with federation support, perfect for running the HoloLoom chatbot on your own infrastructure.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start (Docker Compose)](#quick-start-docker-compose)
4. [Manual Installation](#manual-installation)
5. [Federation Setup](#federation-setup)
6. [Client Setup](#client-setup)
7. [HoloLoom Bot Integration](#hololoom-bot-integration)
8. [Troubleshooting](#troubleshooting)
9. [Alternative Homeservers](#alternative-homeservers)

---

## Introduction

### What is Matrix?

Matrix is an open-source, decentralized communication protocol that allows you to run your own chat server (homeserver) while still communicating with users on other servers through federation.

### Why Self-Host?

- **Privacy**: Full control over your data
- **Customization**: Configure features to your needs
- **Learning**: Understand federated systems
- **Integration**: Perfect for HoloLoom chatops automation

### Homeserver Options

| Server | Language | RAM (Min) | Best For |
|--------|----------|-----------|----------|
| **Synapse** | Python | 1 GB | Full features, stable |
| **Conduit** | Rust | 512 MB | Resource-constrained |
| **Dendrite** | Go | 512 MB | Experimental features |

**This tutorial focuses on Synapse** - the reference implementation with the most features and stability.

---

## Prerequisites

### System Requirements

**Minimum:**
- 1 GB RAM (2 GB recommended)
- 2 CPU cores
- 10 GB storage
- Ubuntu 20.04/22.04, Debian 11, or similar

**Recommended:**
- 4 GB RAM
- 4 CPU cores
- 50 GB SSD storage
- Ubuntu 22.04 LTS

### Domain Name (Optional but Recommended for Federation)

For federation, you'll need:
- A domain name (e.g., `example.com`)
- DNS access to configure records
- Ability to get SSL certificates

**For local testing only**, you can use `localhost` without federation.

### Software Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose (Method 1)
sudo apt install -y docker.io docker-compose

# Or install Docker Desktop (Method 2 - includes Compose)
# https://docs.docker.com/desktop/install/ubuntu/

# Install additional tools
sudo apt install -y git curl wget
```

---

## Quick Start (Docker Compose)

**Best for:** Beginners, quick setup, easy maintenance

This method uses Docker Compose to run Synapse with PostgreSQL.

### Step 1: Create Project Directory

```bash
# Create directory structure
mkdir -p ~/matrix-homeserver
cd ~/matrix-homeserver
mkdir synapse postgres element
```

### Step 2: Generate Synapse Configuration

```bash
# Generate configuration file
docker run -it --rm \
  -v $(pwd)/synapse:/data \
  -e SYNAPSE_SERVER_NAME=matrix.example.com \
  -e SYNAPSE_REPORT_STATS=no \
  matrixdotorg/synapse:latest generate

# For localhost testing, use:
# -e SYNAPSE_SERVER_NAME=localhost \
```

**Important:** Replace `matrix.example.com` with:
- Your actual domain if setting up federation
- `localhost` for local testing only

### Step 3: Create Docker Compose File

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: matrix-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: synapse
      POSTGRES_PASSWORD: CHANGE_THIS_PASSWORD
      POSTGRES_DB: synapse
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    volumes:
      - ./postgres:/var/lib/postgresql/data
    networks:
      - matrix-network

  synapse:
    image: matrixdotorg/synapse:latest
    container_name: matrix-synapse
    restart: unless-stopped
    depends_on:
      - postgres
    ports:
      - "8008:8008"  # Client API
      - "8448:8448"  # Federation API
    volumes:
      - ./synapse:/data
    environment:
      - UID=1000
      - GID=1000
    networks:
      - matrix-network

  element:
    image: vectorim/element-web:latest
    container_name: matrix-element
    restart: unless-stopped
    ports:
      - "8080:80"
    volumes:
      - ./element/config.json:/app/config.json
    networks:
      - matrix-network

networks:
  matrix-network:
    driver: bridge
```

### Step 4: Configure PostgreSQL Database

Edit `synapse/homeserver.yaml`:

```yaml
# Find the database section and replace it with:
database:
  name: psycopg2
  args:
    user: synapse
    password: CHANGE_THIS_PASSWORD
    database: synapse
    host: postgres
    port: 5432
    cp_min: 5
    cp_max: 10
```

**Security:** Change `CHANGE_THIS_PASSWORD` to a strong password (same in both files).

### Step 5: Configure Element Web Client

Create `element/config.json`:

```json
{
  "default_server_config": {
    "m.homeserver": {
      "base_url": "http://localhost:8008",
      "server_name": "localhost"
    }
  },
  "disable_custom_urls": false,
  "disable_guests": true,
  "brand": "Element",
  "integrations_ui_url": "https://scalar.vector.im/",
  "integrations_rest_url": "https://scalar.vector.im/api",
  "integrations_widgets_urls": [
    "https://scalar.vector.im/_matrix/integrations/v1",
    "https://scalar.vector.im/api",
    "https://scalar-staging.vector.im/_matrix/integrations/v1",
    "https://scalar-staging.vector.im/api",
    "https://scalar-staging.riot.im/scalar/api"
  ],
  "default_theme": "dark",
  "room_directory": {
    "servers": [
      "matrix.org"
    ]
  },
  "enable_presence_by_hs_url": {
    "http://localhost:8008": false
  }
}
```

**For domain setup**, replace `localhost` with your domain.

### Step 6: Enable Registration (Optional)

Edit `synapse/homeserver.yaml`:

```yaml
# Find and modify:
enable_registration: true
enable_registration_without_verification: true

# For production, keep registration disabled and use registration_shared_secret
# enable_registration: false
# registration_shared_secret: "GENERATE_RANDOM_SECRET_HERE"
```

### Step 7: Start Services

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f synapse

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME                IMAGE                          STATUS
matrix-element      vectorim/element-web:latest    Up
matrix-postgres     postgres:15                    Up
matrix-synapse      matrixdotorg/synapse:latest    Up
```

### Step 8: Create Admin User

```bash
# Register admin user
docker exec -it matrix-synapse \
  register_new_matrix_user \
  -c /data/homeserver.yaml \
  http://localhost:8008

# Follow prompts:
# Username: admin
# Password: ********
# Admin [y/n]: y
```

### Step 9: Access Element

1. Open browser to `http://localhost:8080`
2. Click "Sign In"
3. Use credentials:
   - Username: `admin`
   - Password: (your password)
   - Homeserver: `http://localhost:8008`

**Congratulations!** Your Matrix homeserver is running!

---

## Manual Installation

**Best for:** Production deployments, advanced users, resource optimization

### Step 1: Install Dependencies

```bash
# Install system packages
sudo apt install -y \
  build-essential \
  python3-dev \
  python3-pip \
  python3-venv \
  libffi-dev \
  python3-setuptools \
  sqlite3 \
  libssl-dev \
  virtualenv \
  libjpeg-dev \
  libxslt1-dev
```

### Step 2: Install PostgreSQL

```bash
# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE USER synapse WITH PASSWORD 'CHANGE_THIS_PASSWORD';
CREATE DATABASE synapse
  ENCODING 'UTF8'
  LC_COLLATE='C'
  LC_CTYPE='C'
  TEMPLATE=template0
  OWNER synapse;
EOF
```

### Step 3: Install Synapse

```bash
# Create synapse user
sudo adduser --system --no-create-home --group synapse

# Create directory
sudo mkdir -p /opt/synapse
sudo chown synapse:synapse /opt/synapse

# Create virtual environment
cd /opt/synapse
sudo -u synapse python3 -m venv env
sudo -u synapse env/bin/pip install --upgrade pip
sudo -u synapse env/bin/pip install matrix-synapse[postgres]

# Generate configuration
sudo -u synapse env/bin/python -m synapse.app.homeserver \
  --server-name=matrix.example.com \
  --config-path=/opt/synapse/homeserver.yaml \
  --generate-config \
  --report-stats=no
```

### Step 4: Configure Synapse

Edit `/opt/synapse/homeserver.yaml`:

```yaml
# Server name (cannot be changed later!)
server_name: "matrix.example.com"

# Listen on all interfaces
listeners:
  - port: 8008
    tls: false
    type: http
    x_forwarded: true
    bind_addresses: ['0.0.0.0']
    resources:
      - names: [client, federation]
        compress: false

# Database
database:
  name: psycopg2
  args:
    user: synapse
    password: CHANGE_THIS_PASSWORD
    database: synapse
    host: localhost
    port: 5432
    cp_min: 5
    cp_max: 10

# Registration (keep disabled in production)
enable_registration: false
registration_shared_secret: "GENERATE_RANDOM_SECRET_HERE"

# Trusted key servers
trusted_key_servers:
  - server_name: "matrix.org"
```

### Step 5: Create Systemd Service

Create `/etc/systemd/system/matrix-synapse.service`:

```ini
[Unit]
Description=Matrix Synapse Homeserver
After=network.target postgresql.service

[Service]
Type=notify
User=synapse
Group=synapse
WorkingDirectory=/opt/synapse
ExecStart=/opt/synapse/env/bin/python -m synapse.app.homeserver \
  --config-path=/opt/synapse/homeserver.yaml
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Step 6: Start Synapse

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable and start Synapse
sudo systemctl enable matrix-synapse
sudo systemctl start matrix-synapse

# Check status
sudo systemctl status matrix-synapse

# View logs
sudo journalctl -u matrix-synapse -f
```

### Step 7: Install Nginx Reverse Proxy

```bash
# Install Nginx
sudo apt install -y nginx certbot python3-certbot-nginx

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/matrix
```

Add configuration:

```nginx
server {
    listen 80;
    listen [::]:80;
    server_name matrix.example.com;

    # For Let's Encrypt
    location /.well-known/acme-challenge {
        root /var/www/html;
    }

    location / {
        return 301 https://$server_name$request_uri;
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    listen 8448 ssl http2 default_server;
    listen [::]:8448 ssl http2 default_server;

    server_name matrix.example.com;

    ssl_certificate /etc/letsencrypt/live/matrix.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/matrix.example.com/privkey.pem;

    location /_matrix {
        proxy_pass http://localhost:8008;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
        client_max_body_size 50M;
    }

    location /_synapse/client {
        proxy_pass http://localhost:8008;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
    }
}
```

### Step 8: Get SSL Certificate

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/matrix /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get certificate
sudo certbot --nginx -d matrix.example.com

# Test auto-renewal
sudo certbot renew --dry-run
```

---

## Federation Setup

Federation allows your homeserver to communicate with other Matrix servers.

### DNS Configuration

You need to configure DNS records for federation:

#### Method 1: Direct Federation (Port 8448)

Add DNS A record:
```
matrix.example.com  →  YOUR_SERVER_IP
```

#### Method 2: Delegation (Recommended)

If you want users like `@user:example.com` but run Matrix on `matrix.example.com`:

**Add DNS records:**
```
# A Record
matrix.example.com  →  YOUR_SERVER_IP

# SRV Record
_matrix._tcp.example.com  →  10 0 8448 matrix.example.com
```

**Create `.well-known` delegation file:**

On your main domain `example.com`, create:
`/.well-known/matrix/server`:
```json
{
  "m.server": "matrix.example.com:8448"
}
```

`/.well-known/matrix/client`:
```json
{
  "m.homeserver": {
    "base_url": "https://matrix.example.com"
  }
}
```

**Configure Nginx for main domain:**
```nginx
server {
    listen 443 ssl;
    server_name example.com;

    location /.well-known/matrix {
        root /var/www/html;
        default_type application/json;
        add_header Access-Control-Allow-Origin *;
    }
}
```

### Firewall Configuration

```bash
# Allow required ports
sudo ufw allow 80/tcp    # HTTP (for Let's Encrypt)
sudo ufw allow 443/tcp   # HTTPS (clients)
sudo ufw allow 8448/tcp  # Federation
sudo ufw enable
```

### Test Federation

Use the Matrix Federation Tester:

```bash
# Online tester
# https://federationtester.matrix.org/

# Or command line
curl https://federationtester.matrix.org/api/report?server_name=matrix.example.com
```

Expected result: ✅ Federation is working

---

## Client Setup

### Element Web

Download and configure Element:

```bash
# Download latest Element
cd /var/www
sudo wget https://github.com/vector-im/element-web/releases/latest/download/element-web.tar.gz
sudo tar -xzf element-web.tar.gz
sudo mv element-* element

# Configure
sudo cp element/config.sample.json element/config.json
sudo nano element/config.json
```

Edit configuration:
```json
{
  "default_server_config": {
    "m.homeserver": {
      "base_url": "https://matrix.example.com",
      "server_name": "example.com"
    }
  }
}
```

Add to Nginx:
```nginx
server {
    listen 443 ssl;
    server_name chat.example.com;

    root /var/www/element;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

### Desktop Clients

- **Element Desktop**: https://element.io/download
- **FluffyChat**: https://fluffychat.im/
- **Nheko**: https://nheko-reborn.github.io/

### Mobile Clients

- **Element**: iOS App Store / Google Play
- **FluffyChat**: iOS App Store / Google Play
- **SchildiChat**: F-Droid / Google Play

---

## HoloLoom Bot Integration

### Create Bot Account

```bash
# Docker method
docker exec -it matrix-synapse \
  register_new_matrix_user \
  -c /data/homeserver.yaml \
  http://localhost:8008

# Manual method
cd /opt/synapse
sudo -u synapse env/bin/register_new_matrix_user \
  -c /opt/synapse/homeserver.yaml \
  http://localhost:8008

# Enter details:
# Username: hololoom_bot
# Password: ********
# Admin: n
```

### Get Access Token

```python
# Option 1: Login via Python
from nio import AsyncClient

async def get_token():
    client = AsyncClient("https://matrix.example.com", "@hololoom_bot:example.com")
    response = await client.login("YOUR_BOT_PASSWORD")
    print(f"Access Token: {client.access_token}")

# Option 2: Use existing client
# Login with bot account in Element
# Settings → Help & About → Advanced → Access Token
```

### Configure HoloLoom Bot

Create `chatops_config.yaml`:

```yaml
matrix:
  homeserver_url: "https://matrix.example.com"
  user_id: "@hololoom_bot:example.com"
  access_token: "YOUR_ACCESS_TOKEN"
  rooms:
    - "#general:example.com"
  admin_users:
    - "@admin:example.com"

hololoom:
  mode: "fast"
  mcts_simulations: 50
```

### Run Bot

```bash
# From HoloLoom repository
cd mythRL
python HoloLoom/chatops/run_chatops.py --config chatops_config.yaml
```

### Invite Bot to Room

1. Create room in Element
2. Invite: `@hololoom_bot:example.com`
3. Bot auto-accepts and joins
4. Test with: `!help`

---

## Troubleshooting

### Synapse Won't Start

**Check logs:**
```bash
# Docker
docker-compose logs synapse

# Systemd
sudo journalctl -u matrix-synapse -n 100
```

**Common issues:**
- Port 8008 already in use: `sudo lsof -i :8008`
- Database connection failed: Check PostgreSQL is running
- Permission errors: Check file ownership

### Federation Not Working

**Test DNS:**
```bash
# Check SRV record
dig SRV _matrix._tcp.example.com

# Check A record
dig matrix.example.com
```

**Test SSL:**
```bash
# Check certificate
curl https://matrix.example.com/_matrix/federation/v1/version
```

**Common issues:**
- Firewall blocking port 8448
- Invalid SSL certificate
- DNS not propagated (wait 24 hours)
- `.well-known` files not accessible

### Can't Register Users

**Enable registration temporarily:**
```yaml
# homeserver.yaml
enable_registration: true
enable_registration_without_verification: true
```

**Or use shared secret:**
```bash
# Generate secret
openssl rand -hex 32

# Add to homeserver.yaml
registration_shared_secret: "YOUR_SECRET"

# Register user
register_new_matrix_user -c homeserver.yaml http://localhost:8008
```

### High Memory Usage

**Optimize PostgreSQL:**
```yaml
# homeserver.yaml
database:
  args:
    cp_min: 5
    cp_max: 10
```

**Limit cache:**
```yaml
caches:
  global_factor: 0.5
  per_cache_factors:
    get_users_in_room: 0.5
```

### Slow Performance

**Enable Redis cache:**
```yaml
redis:
  enabled: true
  host: localhost
  port: 6379
```

**Use workers** (advanced):
https://matrix-org.github.io/synapse/latest/workers.html

---

## Alternative Homeservers

### Conduit (Lightweight)

**Best for:** Low-resource servers, single-user, minimal setup

**Installation:**
```bash
# Download binary
wget https://gitlab.com/famedly/conduit/-/jobs/artifacts/latest/raw/conduit-x86_64-unknown-linux-musl?job=build:release:cargo:x86_64-unknown-linux-musl -O conduit
chmod +x conduit

# Create config
cat > conduit.toml << EOF
[global]
server_name = "example.com"
database_path = "./database"
port = 6167
max_request_size = 20_000_000
allow_registration = true
EOF

# Run
./conduit
```

**Pros:**
- Very low resource usage (512 MB RAM)
- Fast performance
- Simple setup

**Cons:**
- Fewer features than Synapse
- Some spec incompatibilities
- Less mature

### Dendrite (Experimental)

**Best for:** Testing next-gen features, Go developers

**Installation:**
```bash
# Install from source
git clone https://github.com/matrix-org/dendrite
cd dendrite
go build -o bin/ ./cmd/...

# Generate config
./bin/generate-config > dendrite.yaml

# Run
./bin/dendrite --config dendrite.yaml
```

**Pros:**
- Better performance than Synapse
- Lower resource usage
- Modular design

**Cons:**
- Not production-ready
- Missing some features
- Less documentation

---

## Security Best Practices

### 1. Disable Registration

```yaml
# homeserver.yaml
enable_registration: false
```

### 2. Use Strong Passwords

```bash
# Generate strong registration secret
openssl rand -hex 32
```

### 3. Enable Rate Limiting

```yaml
# homeserver.yaml
rc_message:
  per_second: 0.2
  burst_count: 10
```

### 4. Regular Updates

```bash
# Docker
docker-compose pull
docker-compose up -d

# Manual
sudo -u synapse /opt/synapse/env/bin/pip install --upgrade matrix-synapse
sudo systemctl restart matrix-synapse
```

### 5. Backup Database

```bash
# Backup script
#!/bin/bash
sudo -u postgres pg_dump synapse | gzip > synapse-backup-$(date +%F).sql.gz

# Automate with cron
0 2 * * * /path/to/backup.sh
```

### 6. Monitor Logs

```bash
# Install fail2ban
sudo apt install fail2ban

# Create filter for Synapse
# /etc/fail2ban/filter.d/matrix-synapse.conf
```

---

## Performance Tuning

### PostgreSQL Optimization

Edit `/etc/postgresql/*/main/postgresql.conf`:

```conf
# For 4 GB RAM server
shared_buffers = 1GB
effective_cache_size = 3GB
maintenance_work_mem = 256MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 5242kB
min_wal_size = 1GB
max_wal_size = 4GB
```

Restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

### Synapse Worker Mode

For high-traffic servers, enable workers:

```yaml
# worker1.yaml
worker_app: synapse.app.generic_worker
worker_name: worker1
worker_listeners:
  - type: http
    port: 8081
    resources:
      - names: [client, federation]
```

### Media Repository

Limit media storage:

```yaml
# homeserver.yaml
max_upload_size: 50M
media_store_path: "/data/media_store"
url_preview_enabled: false
```

---

## Monitoring

### Prometheus Metrics

Enable metrics:

```yaml
# homeserver.yaml
enable_metrics: true
metrics_port: 9000
```

### Grafana Dashboard

```bash
# Install Prometheus
sudo apt install prometheus

# Configure scraping
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'synapse'
    static_configs:
      - targets: ['localhost:9000']
```

Import Synapse dashboard:
https://grafana.com/grafana/dashboards/

---

## Resources

### Official Documentation

- **Matrix.org**: https://matrix.org/docs/
- **Synapse Docs**: https://matrix-org.github.io/synapse/
- **Element**: https://element.io/

### Community

- **Matrix HQ**: `#matrix:matrix.org`
- **Synapse Admins**: `#synapse:matrix.org`
- **Matrix Community**: https://matrix.to/

### Tools

- **Federation Tester**: https://federationtester.matrix.org/
- **Room Directory**: https://matrix.to/
- **Server Stats**: https://matrix.org/federation/

---

## Next Steps

1. ✅ **Homeserver running** - You have a working Matrix server
2. ✅ **Federation enabled** - Can communicate with other servers
3. ✅ **Bot integrated** - HoloLoom bot connected
4. 🔄 **Customize** - Add custom bridges, bots, integrations
5. 🔄 **Scale** - Enable workers, optimize database
6. 🔄 **Monitor** - Set up Grafana, alerts

---

**Congratulations!** You now have a fully functional Matrix homeserver ready for HoloLoom ChatOps integration!

For HoloLoom-specific bot configuration, see [README.md](README.md).

---

**Last Updated:** October 2025
**Tutorial Version:** 1.0
**Tested On:** Ubuntu 22.04, Synapse 1.99+

# HoloLoom One-Click Deployment System

Deploy HoloLoom workflows to production in **<5 minutes** with zero infrastructure knowledge.

## 🚀 Quick Start

### Option 1: Interactive Mode (Recommended for Beginners)

```bash
python HoloLoom/deployment/one_click_deploy.py --interactive
```

The wizard will guide you through:
1. Choosing a workflow
2. Selecting a platform
3. Reviewing costs
4. Deploying automatically

**Time**: 5 minutes
**Difficulty**: ⭐ Beginner-friendly

### Option 2: Direct Deployment

```bash
# Deploy to Heroku (easiest)
python HoloLoom/deployment/one_click_deploy.py \
  --workflow inbox-triage \
  --platform heroku

# Deploy to Railway
python HoloLoom/deployment/one_click_deploy.py \
  --workflow bug-triage \
  --platform railway

# Deploy locally (Docker)
python HoloLoom/deployment/one_click_deploy.py \
  --workflow report-generation \
  --platform local
```

---

## 📋 Supported Platforms

| Platform | Difficulty | Monthly Cost | Best For |
|----------|-----------|--------------|----------|
| **Heroku** | ⭐ Easiest | $7-16/month | Beginners, quick prototypes |
| **Railway** | ⭐⭐ Easy | $5/month (free credit) | Side projects, MVPs |
| **Fly.io** | ⭐⭐⭐ Medium | $1.94/month or free tier | Global apps, low latency |
| **AWS Lambda** | ⭐⭐⭐⭐ Advanced | $0.20 per 1M requests | Enterprise, high scale |
| **Local Docker** | ⭐ Easiest | Free | Testing, development |

See [PLATFORM_COMPARISON.md](PLATFORM_COMPARISON.md) for detailed comparison.

---

## 🛠️ Prerequisites

### All Platforms
- Python 3.11+
- Git

### Platform-Specific

**Heroku**:
```bash
# Install Heroku CLI
brew install heroku/brew/heroku  # macOS
# or visit: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login
```

**Railway**:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login
```

**Fly.io**:
```bash
# Install flyctl
brew install flyctl  # macOS
# or visit: https://fly.io/docs/hands-on/install-flyctl/

# Login
flyctl auth login
```

**AWS Lambda**:
```bash
# Install AWS CLI
brew install awscli  # macOS
# or visit: https://aws.amazon.com/cli/

# Configure credentials
aws configure
```

**Local Docker**:
```bash
# Install Docker Desktop
# Visit: https://docs.docker.com/get-docker/

# Verify installation
docker --version
docker-compose --version
```

---

## 📚 Available Workflows

List all available workflows:
```bash
python HoloLoom/deployment/one_click_deploy.py --list-workflows
```

### Email & Communication
- **inbox-triage** - Save 2 hours/day triaging emails
- **meeting-summary** - Never take notes again
- **newsletter-digest** - Automatic email summaries
- **calendar-optimization** - Smart calendar management
- **customer-support** - Auto-respond to support tickets

### Data & Analytics
- **report-generation** - Weekly reports in 2 minutes
- **data-cleaning** - Automated data pipelines
- **competitive-intelligence** - 24/7 competitor monitoring

### Developer Tools
- **bug-triage** - Auto-classify and assign bugs
- **code-review** - Automated code review assistant
- **documentation-generator** - Auto-generate docs

---

## 💰 Cost Estimates

### By Platform

| Platform | Monthly Cost | What's Included |
|----------|-------------|-----------------|
| **Local Docker** | $0 | Free (runs on your machine) |
| **Fly.io** | $1.94 | 1 shared-cpu VM + free Postgres |
| **Railway** | $5 | Starter plan (free $5 credit) |
| **Heroku** | $16 | Hobby dyno ($7) + Postgres Basic ($9) |
| **AWS Lambda** | Variable | $0.20 per 1M requests |

### Cost Comparison for Typical Workflow

**Example**: Inbox Triage (processes 200 emails/day)

```
Platform       Monthly Cost  Annual Cost   Notes
─────────────────────────────────────────────────────────
Local Docker   $0            $0            Free, your machine
Fly.io         $1.94         $23.28        Free tier available
Railway        $5.00         $60.00        Free credit applies
Heroku         $16.00        $192.00       Includes database
AWS Lambda     $3.50         $42.00        Pay per use
```

**Recommendation**: Start with **Local Docker** (free) for testing, then deploy to **Railway** or **Heroku** for production.

---

## 🔧 Advanced Usage

### Custom Configuration

Create a custom configuration file:

```json
{
  "use_hololoom_full": true,
  "enable_llm": true,
  "llm_provider": "anthropic",
  "create_tarball": false,
  "environment": {
    "LOG_LEVEL": "DEBUG",
    "ENABLE_ANALYTICS": "true"
  }
}
```

Deploy with custom config:
```bash
python HoloLoom/deployment/one_click_deploy.py \
  --workflow inbox-triage \
  --platform heroku \
  --config my-config.json
```

### Environment Variables

All deployments support these environment variables:

```bash
# Required
WORKFLOW_ID=inbox-triage           # Auto-set by deployment
DATABASE_URL=postgresql://...       # Auto-set by platform

# Optional
LOG_LEVEL=INFO                      # Logging level
ENABLE_ANALYTICS=true               # Track workflow metrics
OPENAI_API_KEY=sk-...              # OpenAI integration
ANTHROPIC_API_KEY=sk-ant-...       # Anthropic integration
SLACK_WEBHOOK_URL=https://...      # Slack notifications
GMAIL_API_KEY=...                   # Gmail integration
```

---

## 🏥 Health Checks

All deployments include automatic health checks:

1. **HTTP 200** - Server responds
2. **Health Endpoint** - `/health` returns OK
3. **Config Endpoint** - `/config` shows workflow configuration
4. **Response Time** - < 5 seconds

View health status:
```bash
curl https://your-app.herokuapp.com/health
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2025-11-17T12:00:00Z"
}
```

---

## 📊 Monitoring & Analytics

All deployments include built-in monitoring:

### Endpoints

- `GET /` - Root endpoint (health status)
- `GET /health` - Health check
- `GET /config` - Configuration (non-sensitive)
- `GET /analytics` - Analytics dashboard (if enabled)

### Platform-Specific Dashboards

**Heroku**:
- Dashboard: `https://dashboard.heroku.com/apps/YOUR-APP`
- Logs: `heroku logs --tail -a YOUR-APP`
- Metrics: Built-in Heroku metrics

**Railway**:
- Dashboard: `https://railway.app/dashboard`
- Logs: Click on service in dashboard
- Metrics: Real-time metrics in dashboard

**Fly.io**:
- Dashboard: `https://fly.io/apps/YOUR-APP`
- Logs: `flyctl logs -a YOUR-APP`
- Metrics: `flyctl status -a YOUR-APP`

**Local Docker**:
- Logs: `docker-compose logs -f`
- Status: `docker-compose ps`
- Metrics: Docker Desktop dashboard

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Platform CLI Not Found

**Error**: `heroku: command not found`

**Solution**:
```bash
# Install platform CLI
brew install heroku/brew/heroku  # Heroku
npm i -g @railway/cli            # Railway
brew install flyctl              # Fly.io
```

#### 2. Authentication Failed

**Error**: `Authentication required`

**Solution**:
```bash
heroku login      # Heroku
railway login     # Railway
flyctl auth login # Fly.io
aws configure     # AWS
```

#### 3. Port Already in Use (Local Docker)

**Error**: `port 8000 already allocated`

**Solution**:
```bash
# Stop existing containers
docker-compose down

# Or use different port
# Edit docker-compose.yml: "8001:8000"
```

#### 4. Database Connection Failed

**Error**: `FATAL: database "hololoom" does not exist`

**Solution**:
```bash
# Platform automatically creates database
# Wait 30 seconds after deployment
# Check platform dashboard for database status
```

#### 5. Health Check Failed

**Error**: `Health check failed after 3 attempts`

**Solution**:
1. Check application logs:
   ```bash
   heroku logs --tail -a YOUR-APP        # Heroku
   flyctl logs -a YOUR-APP               # Fly.io
   docker-compose logs -f                # Local
   ```

2. Verify environment variables:
   ```bash
   heroku config -a YOUR-APP             # Heroku
   ```

3. Restart application:
   ```bash
   heroku restart -a YOUR-APP            # Heroku
   flyctl apps restart YOUR-APP          # Fly.io
   docker-compose restart                # Local
   ```

---

## 📖 Examples

### Deploy Inbox Triage to Heroku

```bash
# 1. Install Heroku CLI (if not installed)
brew install heroku/brew/heroku

# 2. Login
heroku login

# 3. Deploy
python HoloLoom/deployment/one_click_deploy.py \
  --workflow inbox-triage \
  --platform heroku

# Output:
# 🚀 Deploying 'inbox-triage' to HEROKU...
#    ✓ Heroku CLI found
#    ✓ App created (hololoom-inbox-triage-a8f3)
#    ✓ Postgres provisioned
#    ✓ Environment configured
#    ✓ Code deployed
#    ✓ All health checks passed
#
# 🎉 Deployment Successful!
# 📍 Dashboard: https://hololoom-inbox-triage-a8f3.herokuapp.com
# 💰 Estimated Cost: $16.00/month
# ⏱️  Deployment Time: 167.3s
```

### Deploy Bug Triage to Railway

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Deploy
python HoloLoom/deployment/one_click_deploy.py \
  --workflow bug-triage \
  --platform railway

# Output:
# 🚀 Deploying 'bug-triage' to RAILWAY...
#    ✓ Railway CLI found
#    ✓ Logged in
#    ✓ Project created
#    ✓ Postgres provisioned
#    ✓ Code deployed
#    ✓ All health checks passed
#
# 🎉 Deployment Successful!
# 📍 Dashboard: https://hololoom-bug-triage.up.railway.app
# 💰 Estimated Cost: $5.00/month (free credit)
# ⏱️  Deployment Time: 112.5s
```

### Deploy Locally with Docker

```bash
# 1. Start Docker Desktop

# 2. Deploy
python HoloLoom/deployment/one_click_deploy.py \
  --workflow report-generation \
  --platform local

# Output:
# 🚀 Deploying 'report-generation' to LOCAL...
#    ✓ Docker found
#    ✓ Configuration created
#    ✓ Containers started
#    ✓ All health checks passed
#
# 🎉 Deployment Successful!
# 📍 Dashboard: http://localhost:8000
# 💰 Estimated Cost: $0.00/month (free)
# ⏱️  Deployment Time: 58.2s

# 3. Access locally
open http://localhost:8000

# 4. Stop when done
docker-compose down
```

---

## 🎯 Next Steps

After successful deployment:

1. **Configure Credentials**
   - Visit your deployment dashboard
   - Add API keys (Gmail, Slack, etc.)
   - Set up webhooks

2. **Run First Workflow**
   - Trigger workflow manually via dashboard
   - Or set up automatic triggers (cron, webhooks)

3. **Monitor Performance**
   - Check analytics dashboard
   - Review logs for errors
   - Optimize based on metrics

4. **Scale If Needed**
   - Upgrade to higher tier for more traffic
   - Add more workers/dynos
   - Enable auto-scaling

---

## 📚 Additional Resources

- **[PLATFORM_COMPARISON.md](PLATFORM_COMPARISON.md)** - Detailed platform comparison
- **[WORKFLOWS_FIRST_IMPLEMENTATION_PLAN.md](../../WORKFLOWS_FIRST_IMPLEMENTATION_PLAN.md)** - Overall implementation plan
- **Demo Script**: `demos/demo_one_click_deploy.py`

---

## 🤝 Support

**Issues or questions?**
- Check the troubleshooting section above
- Review platform-specific documentation
- Open an issue on GitHub

---

## 📝 License

Part of the HoloLoom project.

**Created**: November 2025

---

**Ready to deploy your first workflow?** Run:
```bash
python HoloLoom/deployment/one_click_deploy.py --interactive
```

Happy automating! 🚀

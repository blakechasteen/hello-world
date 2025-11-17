# Quick Start Guide - One-Click Deployment

Deploy your first workflow in **3 simple steps**:

## Step 1: Install Platform CLI

Choose ONE platform:

```bash
# Option 1: Local Docker (easiest, free)
# Download Docker Desktop: https://docs.docker.com/get-docker/

# Option 2: Heroku (beginner-friendly, $16/month)
brew install heroku/brew/heroku
heroku login

# Option 3: Railway (modern, $5/month free credit)
npm i -g @railway/cli
railway login
```

## Step 2: Run Deployment Wizard

```bash
cd /path/to/hololoom
python HoloLoom/deployment/one_click_deploy.py --interactive
```

## Step 3: Follow the Prompts

The wizard will ask:
1. Which workflow? (e.g., "inbox-triage")
2. Which platform? (e.g., "heroku")
3. Confirm deployment? (yes)

**Done!** Your workflow is deployed in ~3 minutes.

---

## Quick Commands

```bash
# List available workflows
python HoloLoom/deployment/one_click_deploy.py --list-workflows

# List supported platforms
python HoloLoom/deployment/one_click_deploy.py --list-platforms

# Direct deployment (skip wizard)
python HoloLoom/deployment/one_click_deploy.py \
  --workflow inbox-triage \
  --platform heroku

# Run demo
python demos/demo_one_click_deploy.py
```

---

## Next Steps

After deployment:
1. Visit your deployment URL
2. Configure API keys in dashboard
3. Run your first workflow
4. Check analytics

**Full documentation**: [README.md](README.md)

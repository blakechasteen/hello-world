# CI/CD Pipelines

GitHub Actions workflows for automated testing, building, and deployment.

## Workflows

### 1. Tests (`test.yml`)

**Trigger**: Pull requests and pushes to `main`/`develop`

**What it does**:
- Spins up test databases (PostgreSQL, Neo4j, Qdrant, Redis)
- Runs database migrations
- Executes unit tests with coverage
- Executes integration tests
- Uploads coverage reports to Codecov

**Services**:
- PostgreSQL 15
- Neo4j 5.12
- Qdrant (latest)
- Redis 7

**Duration**: ~3-5 minutes

**Example**:
```bash
# Triggered automatically on PR
git push origin feature/my-feature
```

### 2. Lint (`lint.yml`)

**Trigger**: Pull requests and pushes to `main`/`develop`

**What it does**:
- Runs `ruff` for code linting
- Runs `black` for code formatting check
- Runs `mypy` for type checking
- Runs `bandit` for security analysis

**Duration**: ~1-2 minutes

**Fixing lint errors**:
```bash
# Auto-fix formatting
make format

# Check linting
make lint
```

### 3. Build (`build.yml`)

**Trigger**: Push to `main` or version tags (`v*`)

**What it does**:
- Builds Docker image
- Pushes to GitHub Container Registry (ghcr.io)
- Tags with branch name, version, and commit SHA
- Uses layer caching for faster builds

**Image tags**:
- `ghcr.io/owner/repo:main` - Latest main branch
- `ghcr.io/owner/repo:v1.2.3` - Specific version
- `ghcr.io/owner/repo:main-abc123` - Commit SHA

**Duration**: ~5-10 minutes (first build), ~2-3 minutes (cached)

**Publishing a release**:
```bash
git tag v1.0.0
git push origin v1.0.0
```

### 4. Deploy (`deploy.yml`)

**Trigger**: Manual workflow dispatch

**What it does**:
- Deploys to staging or production (user choice)
- Runs database migrations (production only)
- Performs health check
- Sends notifications on success/failure
- Automatic rollback on failure

**Duration**: ~5-10 minutes

**Manual deployment**:
1. Go to Actions → Deploy
2. Click "Run workflow"
3. Select environment (staging/production)
4. Confirm

## Setup Instructions

### 1. Enable GitHub Actions

GitHub Actions should be enabled by default. Verify in:
- **Repository Settings** → **Actions** → **General**
- Set "Allow all actions and reusable workflows"

### 2. Configure Secrets

Add these secrets in **Settings** → **Secrets and variables** → **Actions**:

**Required**:
- `CODECOV_TOKEN` - For coverage reports (get from codecov.io)

**For deployment** (optional):
- `DEPLOY_HOST` - SSH host for deployment
- `DEPLOY_KEY` - SSH private key
- `SLACK_WEBHOOK` - Slack notification webhook
- `DOCKER_USERNAME` - Docker registry username
- `DOCKER_PASSWORD` - Docker registry password

### 3. Configure Environments

Create environments in **Settings** → **Environments**:

**staging**:
- No protection rules
- Secrets: staging-specific values

**production**:
- Required reviewers (1+)
- Wait timer: 5 minutes
- Secrets: production-specific values

### 4. Protect Branches

Configure in **Settings** → **Branches**:

**main branch**:
- [x] Require pull request before merging
- [x] Require status checks to pass (Tests, Lint)
- [x] Require branches to be up to date
- [x] Require linear history

**develop branch**:
- [x] Require pull request before merging
- [x] Require status checks to pass (Tests)

## Workflow Status Badges

Add to README.md:

```markdown
![Tests](https://github.com/owner/repo/actions/workflows/test.yml/badge.svg)
![Lint](https://github.com/owner/repo/actions/workflows/lint.yml/badge.svg)
![Build](https://github.com/owner/repo/actions/workflows/build.yml/badge.svg)
```

## Development Workflow

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Develop and commit
git add .
git commit -m "Add feature"

# 3. Push to GitHub (triggers Tests + Lint)
git push origin feature/my-feature

# 4. Create pull request
# Tests and Lint must pass before merge

# 5. Merge to main (triggers Build)
```

### Hotfix Workflow

```bash
# 1. Create hotfix from main
git checkout -b hotfix/critical-bug main

# 2. Fix and commit
git commit -m "Fix critical bug"

# 3. Push and create PR
git push origin hotfix/critical-bug

# 4. After merge, deploy immediately
# GitHub Actions → Deploy → production
```

### Release Workflow

```bash
# 1. Ensure main is stable
git checkout main
git pull

# 2. Create release tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0

# 3. GitHub Actions builds Docker image with v1.0.0 tag

# 4. Deploy to production
# GitHub Actions → Deploy → production
```

## Debugging Failed Workflows

### View logs

1. Go to **Actions** tab
2. Click on failed workflow
3. Click on failed job
4. Expand failed step

### Common issues

**Tests fail**:
```bash
# Run tests locally
make test

# Or with specific database
DATABASE_URL=postgresql://... pytest backend/tests/ -v
```

**Lint fails**:
```bash
# Auto-fix most issues
make format

# Check remaining issues
make lint
```

**Build fails**:
```bash
# Test Docker build locally
docker build -t lms-backend .
docker run -p 8000:8000 lms-backend
```

**Deploy fails**:
- Check deployment secrets are configured
- Verify target environment is healthy
- Check database migration logs

## Cost Optimization

GitHub Actions is free for public repositories with limits:
- **2,000 minutes/month** for private repos
- **Unlimited** for public repos

**Tips to reduce minutes**:
1. Use caching (already configured)
2. Cancel redundant runs (pushes to PR)
3. Use self-hosted runners for high-volume projects

## Monitoring

### Codecov Integration

- Coverage reports: https://codecov.io/gh/owner/repo
- Automatic PR comments with coverage diff
- Coverage trending over time

### Docker Image Registry

- View images: https://github.com/owner/repo/pkgs/container/repo
- Pull images: `docker pull ghcr.io/owner/repo:tag`

### Deployment Logs

- Check logs: `docker-compose logs -f api`
- Health check: `curl http://api.lms.example.com/health`

## Future Enhancements

- [ ] Add E2E tests with Playwright
- [ ] Add performance testing with Locust
- [ ] Add security scanning with Snyk
- [ ] Add dependency updates with Dependabot
- [ ] Add automated changelogs
- [ ] Add blue-green deployments
- [ ] Add canary releases

## Troubleshooting

### "Tests failed" in PR

1. Check which test failed in Actions log
2. Run test locally: `pytest path/to/test.py -v`
3. Fix issue and push

### "Unable to push Docker image"

1. Check GITHUB_TOKEN has `packages:write` permission
2. Verify GitHub Container Registry is enabled
3. Check Docker image size (<5GB)

### "Deployment timed out"

1. Increase timeout in workflow (default: 5min)
2. Check network connectivity to deploy host
3. Verify services are healthy

## Support

For CI/CD issues:
1. Check GitHub Actions documentation
2. Review workflow logs
3. Create issue in repository

## License

MIT

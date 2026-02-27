# CI/CD Pipeline Guide

**Version**: 1.0.0
**Date**: November 15, 2025
**Status**: ✅ Production Ready

---

## Overview

Comprehensive CI/CD pipelines for automated testing, building, security scanning, and deployment of the HoloLoom VoiceAgent service.

### Supported Platforms

- ✅ **GitHub Actions** (`.github/workflows/voice-agent-ci-cd.yml`)
- ✅ **GitLab CI** (`.gitlab-ci.yml`)

Both pipelines provide identical functionality with platform-specific optimizations.

---

## Pipeline Stages

### 1. Test

**Purpose**: Run comprehensive test suite

**Steps**:
- Install Python dependencies
- Run unit tests with pytest
- Generate coverage reports
- Upload coverage to Codecov (GitHub) or GitLab

**Coverage Target**: >80%

**Commands**:
```bash
# Run locally
PYTHONPATH=. pytest hololoom/voice/tests/test_voice_agent.py -v --cov=hololoom.voice
```

### 2. Lint

**Purpose**: Code quality and style checks

**Tools**:
- **flake8**: PEP 8 compliance, syntax errors
- **black**: Code formatting
- **mypy**: Type checking (optional, can fail)

**Commands**:
```bash
# Run locally
flake8 hololoom/voice --count --max-line-length=127 --statistics
black --check hololoom/voice
mypy hololoom/voice --ignore-missing-imports
```

### 3. Security

**Purpose**: Vulnerability scanning

**Tools**:
- **Trivy**: Filesystem and Docker image scanning
- Scans for HIGH and CRITICAL vulnerabilities
- Fails pipeline on CRITICAL vulnerabilities

**Commands**:
```bash
# Run locally
trivy fs --severity HIGH,CRITICAL .
trivy image hololoom/voice-agent:latest
```

### 4. Build

**Purpose**: Build and push Docker images

**Platforms**:
- linux/amd64
- linux/arm64 (multi-arch support)

**Tags**:
- `latest` (main branch only)
- `<branch-name>` (all branches)
- `<commit-sha>` (all commits)
- `v<version>` (tagged releases)

**Docker Hub**:
```bash
docker pull hololoom/voice-agent:latest
docker pull hololoom/voice-agent:develop
docker pull hololoom/voice-agent:abc123def
```

### 5. Helm Validation

**Purpose**: Validate Helm chart correctness

**Steps**:
- Lint Helm chart syntax
- Template chart with test values
- Validate rendered Kubernetes manifests
- Check resource limits and best practices

**Commands**:
```bash
# Run locally
helm lint deployment/helm/hololoom-voice
helm template hololoom-voice deployment/helm/hololoom-voice --debug
```

### 6. Deploy to Staging

**Purpose**: Automatic deployment to staging environment

**Trigger**: Push to `develop` branch

**Configuration**:
- 2 replicas
- Smaller resource limits
- Test secrets

**Verification**:
- Health check endpoint
- Rollout status
- Pod readiness

### 7. Deploy to Production

**Purpose**: Manual deployment to production

**Trigger**: Push to `main` branch + manual approval

**Configuration**:
- 5 replicas (autoscales to 50)
- Production resource limits
- Production secrets

**Verification**:
- Health check endpoint
- Rollout status
- Smoke tests
- Monitoring alerts

### 8. Release

**Purpose**: Create tagged releases

**Trigger**: Commit message starts with "Release" on `main` branch

**Artifacts**:
- GitHub/GitLab release
- Docker image tag
- Release notes

---

## GitHub Actions Setup

### Required Secrets

Configure in repository settings → Secrets and variables → Actions:

| Secret | Description | Example |
|--------|-------------|---------|
| `DOCKER_USERNAME` | Docker Hub username | `hololoom` |
| `DOCKER_PASSWORD` | Docker Hub password/token | `dckr_pat_xxx` |
| `OPENAI_API_KEY_STAGING` | OpenAI API key (staging) | `sk-xxx-staging` |
| `OPENAI_API_KEY_PRODUCTION` | OpenAI API key (production) | `sk-xxx-prod` |
| `NEO4J_PASSWORD_STAGING` | Neo4j password (staging) | `staging-pass` |
| `NEO4J_PASSWORD_PRODUCTION` | Neo4j password (production) | `prod-pass` |
| `KUBE_CONFIG_STAGING` | Kubernetes config (base64) | `base64 ~/.kube/config` |
| `KUBE_CONFIG_PRODUCTION` | Kubernetes config (base64) | `base64 ~/.kube/config` |

### Environments

Configure in repository settings → Environments:

**staging**:
- Environment name: `staging`
- URL: `https://voice-staging.hololoom.ai`
- Protection rules: None (auto-deploy)

**production**:
- Environment name: `production`
- URL: `https://voice.hololoom.ai`
- Protection rules:
  - Required reviewers: 1+ maintainers
  - Wait timer: 0 minutes
  - Branch restrictions: `main` only

### Workflow File

Location: `.github/workflows/voice-agent-ci-cd.yml`

**Triggers**:
```yaml
on:
  push:
    branches: [main, develop, 'release/**']
  pull_request:
    branches: [main, develop]
  workflow_dispatch:  # Manual trigger
```

**Manual Trigger**:
1. Go to Actions tab
2. Select "VoiceAgent CI/CD"
3. Click "Run workflow"
4. Select branch
5. Click "Run workflow"

---

## GitLab CI Setup

### Required Variables

Configure in Settings → CI/CD → Variables:

| Variable | Type | Protected | Masked | Value |
|----------|------|-----------|--------|-------|
| `DOCKER_USERNAME` | Variable | No | No | `hololoom` |
| `DOCKER_PASSWORD` | Variable | No | Yes | `dckr_pat_xxx` |
| `OPENAI_API_KEY_STAGING` | Variable | No | Yes | `sk-xxx-staging` |
| `OPENAI_API_KEY_PRODUCTION` | Variable | Yes | Yes | `sk-xxx-prod` |
| `NEO4J_PASSWORD_STAGING` | Variable | No | Yes | `staging-pass` |
| `NEO4J_PASSWORD_PRODUCTION` | Variable | Yes | Yes | `prod-pass` |
| `KUBE_CONFIG_STAGING` | File | No | No | `~/.kube/config` (base64) |
| `KUBE_CONFIG_PRODUCTION` | File | Yes | No | `~/.kube/config` (base64) |

### Environments

Configure in Deployments → Environments:

**staging**:
- Name: `staging`
- External URL: `https://voice-staging.hololoom.ai`
- Auto-deploy: Yes

**production**:
- Name: `production`
- External URL: `https://voice.hololoom.ai`
- Deployment approval: Required

### Pipeline File

Location: `.gitlab-ci.yml`

**Stages**:
```yaml
stages:
  - test
  - lint
  - security
  - build
  - helm-validate
  - deploy-staging
  - deploy-production
  - release
```

**Manual Trigger**:
1. Go to CI/CD → Pipelines
2. Click "Run pipeline"
3. Select branch
4. Click "Run pipeline"

---

## Local Development

### Run Tests Locally

```bash
# Install dependencies
pip install -r hololoom/voice/requirements.txt
pip install pytest pytest-asyncio pytest-cov

# Run tests
PYTHONPATH=. pytest hololoom/voice/tests/test_voice_agent.py -v

# With coverage
PYTHONPATH=. pytest hololoom/voice/tests/test_voice_agent.py -v --cov=hololoom.voice --cov-report=html

# Open coverage report
open htmlcov/index.html
```

### Build Docker Image Locally

```bash
# Build image
docker build -f Dockerfile.voice -t hololoom/voice-agent:local .

# Run container
docker run -it --rm \
  -e OPENAI_API_KEY='sk-xxx' \
  -e NEO4J_URI='bolt://neo4j:7687' \
  -e NEO4J_PASSWORD='password' \
  -p 8080:8080 \
  hololoom/voice-agent:local

# Test health endpoint
curl http://localhost:8080/health
```

### Test Helm Chart Locally

```bash
# Lint chart
helm lint deployment/helm/hololoom-voice

# Render templates
helm template hololoom-voice deployment/helm/hololoom-voice \
  --set voiceAgent.secrets.openaiApiKey.secretName=test-secret \
  --debug

# Install to local cluster (kind, minikube, etc.)
kubectl create namespace hololoom-voice-local

kubectl create secret generic voice-agent-secrets \
  --from-literal=OPENAI_API_KEY='sk-xxx' \
  --from-literal=NEO4J_PASSWORD='password' \
  -n hololoom-voice-local

helm install hololoom-voice deployment/helm/hololoom-voice \
  --namespace hololoom-voice-local \
  --set voiceAgent.image.tag=local \
  --set voiceAgent.replicaCount=1

# Verify
kubectl get pods -n hololoom-voice-local
kubectl logs -n hololoom-voice-local -l app=voice-agent -f
```

---

## Monitoring Pipeline

### GitHub Actions

**View workflow runs**:
1. Go to repository → Actions tab
2. Select workflow
3. View runs, logs, artifacts

**Badges**:
```markdown
![CI/CD](https://github.com/yourusername/hololoom/actions/workflows/voice-agent-ci-cd.yml/badge.svg)
```

**Notifications**:
- GitHub UI notifications
- Email on failure
- Slack/Discord webhooks (configure in workflow)

### GitLab CI

**View pipeline status**:
1. Go to CI/CD → Pipelines
2. Click pipeline ID
3. View stages, jobs, logs

**Badges**:
```markdown
![Pipeline](https://gitlab.com/yourusername/hololoom/badges/main/pipeline.svg)
![Coverage](https://gitlab.com/yourusername/hololoom/badges/main/coverage.svg)
```

**Notifications**:
- GitLab UI notifications
- Email on failure
- Slack/Mattermost webhooks (configure in project settings)

---

## Troubleshooting

### Pipeline Failures

**Tests failing**:
```bash
# Run tests locally to reproduce
PYTHONPATH=. pytest hololoom/voice/tests/test_voice_agent.py -v

# Check coverage
pytest --cov=hololoom.voice --cov-report=term-missing
```

**Linting errors**:
```bash
# Auto-fix with black
black hololoom/voice

# Check flake8 errors
flake8 hololoom/voice --show-source
```

**Security vulnerabilities**:
```bash
# Scan locally
trivy fs --severity HIGH,CRITICAL .

# Update dependencies
pip install --upgrade -r requirements.txt
```

**Docker build failures**:
```bash
# Build locally with verbose output
docker build -f Dockerfile.voice -t test:latest . --progress=plain

# Check disk space
docker system df
docker system prune -a
```

**Helm validation errors**:
```bash
# Validate locally
helm lint deployment/helm/hololoom-voice --strict

# Debug template rendering
helm template hololoom-voice deployment/helm/hololoom-voice --debug
```

**Deployment failures**:
```bash
# Check pod status
kubectl get pods -n hololoom-voice

# View pod logs
kubectl logs -n hololoom-voice <pod-name> -f

# Describe pod
kubectl describe pod -n hololoom-voice <pod-name>

# Check events
kubectl get events -n hololoom-voice --sort-by='.lastTimestamp'
```

### Common Issues

**1. Missing secrets**

**Symptom**: Pipeline fails at deploy stage with "secret not found"

**Fix**:
```bash
# GitHub: Check repository secrets
# Settings → Secrets and variables → Actions

# GitLab: Check project variables
# Settings → CI/CD → Variables
```

**2. Kubernetes authentication failed**

**Symptom**: "Unable to authenticate to cluster"

**Fix**:
```bash
# Verify kubeconfig
kubectl config view

# Encode for GitHub/GitLab
cat ~/.kube/config | base64 -w 0

# Add to secrets/variables
```

**3. Image not found**

**Symptom**: "Failed to pull image"

**Fix**:
```bash
# Verify image exists
docker pull hololoom/voice-agent:latest

# Check Docker Hub credentials
docker login

# Rebuild and push
docker build -f Dockerfile.voice -t hololoom/voice-agent:latest .
docker push hololoom/voice-agent:latest
```

**4. Health check failing**

**Symptom**: "Liveness probe failed"

**Fix**:
```bash
# Check pod logs
kubectl logs -n hololoom-voice <pod-name>

# Test health endpoint locally
docker run -p 8080:8080 hololoom/voice-agent:latest
curl http://localhost:8080/health

# Increase initialDelaySeconds in values.yaml
```

**5. Out of resources**

**Symptom**: "Insufficient CPU/memory"

**Fix**:
```bash
# Check node resources
kubectl top nodes

# Reduce resource requests
helm upgrade hololoom-voice deployment/helm/hololoom-voice \
  --set voiceAgent.resources.requests.memory=256Mi

# Add more nodes to cluster
```

---

## Best Practices

### Branching Strategy

**main**:
- Production-ready code only
- Protected branch (require reviews)
- Auto-deploys to production (with approval)

**develop**:
- Integration branch
- Auto-deploys to staging
- Merge feature branches here first

**feature/\***:
- Feature development
- Create from `develop`
- Run tests on PR

**release/\***:
- Release preparation
- Version bumps, changelog
- Create from `develop`, merge to `main`

### Commit Messages

Use conventional commits:

```
feat: Add voice activity detection
fix: Fix memory leak in audio queue
docs: Update deployment guide
test: Add integration tests for TTS
refactor: Simplify turn-taking logic
ci: Update GitHub Actions workflow
```

**Trigger release**:
```
Release v1.2.0: Production improvements
```

### Version Tagging

**Semantic versioning**: `v<major>.<minor>.<patch>`

```bash
# Tag release
git tag -a v1.2.0 -m "Release v1.2.0: Production improvements"
git push origin v1.2.0
```

### Testing Before Merge

**Always test locally first**:

```bash
# Run full test suite
pytest hololoom/voice/tests/ -v

# Build Docker image
docker build -f Dockerfile.voice -t test:latest .

# Test Helm chart
helm template hololoom-voice deployment/helm/hololoom-voice --debug
```

**Use feature flags for risky changes**:

```python
# Example: Feature flag for new TTS provider
if os.getenv('ENABLE_NEW_TTS', 'false') == 'true':
    tts = NewTTSProvider()
else:
    tts = OpenAITTS()
```

---

## Performance Metrics

### Pipeline Duration

| Stage | Duration | Notes |
|-------|----------|-------|
| Test | 30-60s | Depends on test count |
| Lint | 10-20s | Fast syntax checks |
| Security | 30-60s | Trivy scanning |
| Build | 2-5min | Multi-arch builds |
| Helm Validate | 10-20s | Template rendering |
| Deploy Staging | 2-3min | With health checks |
| Deploy Production | 5-10min | Rolling updates |

**Total (main branch)**: ~10-20 minutes

**Total (PR)**: ~2-3 minutes (tests + lint only)

### Optimization Tips

**1. Use caching**:
```yaml
# GitHub Actions
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
```

**2. Parallel jobs**:
```yaml
# Run tests and lint in parallel
jobs:
  test:
    runs-on: ubuntu-latest
  lint:
    runs-on: ubuntu-latest
```

**3. Skip unnecessary steps**:
```yaml
# Only deploy on specific branches
rules:
  - if: '$CI_COMMIT_BRANCH == "main"'
```

---

## Security Considerations

### Secrets Management

**Never commit secrets**:
- Use GitHub Secrets / GitLab Variables
- Rotate secrets regularly
- Use separate secrets for staging/production

**Secret rotation**:
```bash
# Generate new OpenAI API key
# Update GitHub/GitLab secrets
# Trigger re-deployment
```

### Container Security

**Multi-stage builds**:
- Minimal runtime image
- No build tools in production
- Non-root user

**Security scanning**:
- Trivy for vulnerabilities
- Fail on CRITICAL findings
- Regular dependency updates

**Image signing** (optional):
```bash
# Sign with cosign
cosign sign hololoom/voice-agent:latest
```

### Network Security

**NetworkPolicies**:
- Restrict ingress/egress
- Allow only required services
- Deny by default

**TLS/SSL**:
- HTTPS only for external access
- Use cert-manager for certificates
- Mutual TLS for service-to-service

---

## Additional Resources

- **GitHub Actions Docs**: https://docs.github.com/en/actions
- **GitLab CI Docs**: https://docs.gitlab.com/ee/ci/
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **Kubernetes Deployment Strategies**: https://kubernetes.io/docs/concepts/workloads/controllers/deployment/
- **Helm Documentation**: https://helm.sh/docs/

---

**Version**: 1.0.0
**Status**: ✅ Production Ready
**Last Updated**: November 15, 2025

*Complete CI/CD pipelines for automated testing, building, and deployment of HoloLoom VoiceAgent.*

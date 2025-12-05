# Phase 5: GitHub Integration - Setup Guide

**Status**: 📋 Ready for Implementation
**Prerequisites**: Phase 4 Dashboard Complete
**Estimated Time**: 4-5 hours
**Dependencies**: PyJWT, PyGithub, cryptography

---

## What's Been Built

### ✅ Phase 5A: GitHub Authentication (COMPLETE)

**Files Created**:
- `bot/github_auth.py` (300+ lines) - GitHub App authentication
- `config/github_config.json` - Configuration template
- `requirements-github.txt` - Phase 5 dependencies

**Features**:
- GitHub App OAuth flow
- JWT token generation
- Installation token management
- Token caching and refresh
- Webhook signature verification
- Multi-repo support

---

## Setup Instructions

### Step 1: Install Dependencies

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL\promptly-matrix-bot
pip install -r requirements-github.txt
```

**Dependencies**:
- `PyJWT==2.8.0` - JWT token generation
- `PyGithub==2.1.1` - GitHub API client
- `cryptography==41.0.7` - RSA key handling

---

### Step 2: Create GitHub App

1. **Navigate to GitHub Settings**:
   - Go to https://github.com/settings/apps/new
   - Or from your organization: https://github.com/organizations/YOUR_ORG/settings/apps/new

2. **Configure Basic Information**:
   ```
   GitHub App name: Promptly Matrix Bot
   Homepage URL: https://github.com/YOUR_USERNAME/promptly-matrix-bot
   Webhook URL: https://your-server.com/webhooks/github
   Webhook secret: (generate a random secret)
   ```

3. **Set Permissions**:
   - **Repository permissions**:
     - Contents: Read & write
     - Pull requests: Read & write
     - Issues: Read & write
     - Actions: Read
     - Metadata: Read (auto-selected)

   - **Organization permissions**:
     - Members: Read (optional, for team features)

4. **Subscribe to Events**:
   - [x] Pull request
   - [x] Pull request review
   - [x] Pull request review comment
   - [x] Issues
   - [x] Issue comment
   - [x] Workflow run
   - [x] Push (optional)

5. **Create the App**:
   - Click "Create GitHub App"
   - Save your App ID (you'll need this)

---

### Step 3: Generate Private Key

1. **In your new GitHub App settings**:
   - Scroll to "Private keys"
   - Click "Generate a private key"
   - A `.pem` file will download

2. **Save the private key**:
   ```bash
   # Move the downloaded key to your config directory
   mv ~/Downloads/your-app-name.*.private-key.pem config/private-key.pem
   ```

---

### Step 4: Configure `github_config.json`

Edit `config/github_config.json`:

```json
{
  "app_id": "123456",
  "private_key": "config/private-key.pem",
  "webhook_secret": "your_webhook_secret_here",
  "oauth_client_id": "Iv1.abc123def456",
  "oauth_client_secret": "abc123def456...",
  "installation_id": null
}
```

**Where to find these values**:
- `app_id`: GitHub App settings page (top)
- `private_key`: Path to the `.pem` file you downloaded
- `webhook_secret`: The secret you entered when creating the app
- `oauth_client_id`: GitHub App settings → "Client ID"
- `oauth_client_secret`: GitHub App settings → "Generate a new client secret"

---

### Step 5: Install App to Repository

1. **Install the GitHub App**:
   - GitHub App settings → "Install App"
   - Select your user/organization
   - Choose "All repositories" or "Select repositories"
   - Click "Install"

2. **Get Installation ID** (optional):
   ```python
   from bot.github_auth import GitHubAuth

   auth = GitHubAuth()
   installations = auth.get_installations()

   for installation in installations:
       print(f"Installation ID: {installation['id']}")
       print(f"Account: {installation['account']['login']}")
   ```

   Update `github_config.json` with the installation ID.

---

### Step 6: Test Authentication

```bash
cd c:\Users\blake\OneDrive\Documents\mythRL\promptly-matrix-bot
PYTHONPATH=. python bot/github_auth.py
```

**Expected Output**:
```
✓ GitHub authentication initialized
✓ Found 1 installation(s)
  - Installation 12345678 (your-username)
    5 repository(ies)
      - your-username/repo1
      - your-username/repo2
      - your-username/repo3
```

---

## Remaining Phase 5 Components

### 5B: PR Management (1.5 hours) 📋

**To Build**:
- `bot/github_pr.py` - PR creation and management
- Matrix commands: `!pr create`, `!pr review`, `!pr merge`

**Features**:
- Create PR from branch
- Add reviewers
- Comment on PRs
- Approve/request changes
- Merge with checks

---

### 5C: AI Code Review (1 hour) 📋

**To Build**:
- `bot/code_review.py` - HoloLoom-powered code review
- Matrix command: `!review pr 123`

**Features**:
- Fetch PR diff
- Analyze with HoloLoom
- Detect security/performance issues
- Generate review comments
- Post to GitHub

**AI Checks**:
- Security vulnerabilities
- Code quality
- Performance concerns
- Best practices
- Documentation gaps

---

### 5D: Issue Tracking (30 minutes) 📋

**To Build**:
- `bot/github_issues.py` - Issue management
- Matrix commands: `!issue create`, `!issue comment`, `!issue close`

**Features**:
- Create issues
- Add labels/assignees
- Link to PRs
- Status updates

---

### 5E: CI/CD Triggers (1 hour) 📋

**To Build**:
- `bot/github_actions.py` - GitHub Actions integration
- Webhook handler for build status
- Matrix commands: `!build trigger`, `!build status`

**Features**:
- Trigger workflows
- Monitor builds
- Send notifications
- View logs

---

## Usage Examples

### Create Pull Request

**Matrix Command**:
```
!pr create feature-dashboard "Add Phase 4 dashboard components"
```

**What Happens**:
1. Bot creates PR from `feature-dashboard` → `main`
2. Adds title and auto-generates description
3. Adds reviewers based on CODEOWNERS
4. Sends PR link to Matrix room

---

### AI Code Review

**Matrix Command**:
```
!review pr 123
```

**What Happens**:
1. Bot fetches PR #123 diff
2. Sends code to HoloLoom for analysis
3. HoloLoom checks for:
   - Security issues (SQL injection, XSS, etc.)
   - Performance problems (N+1 queries, inefficient loops)
   - Code quality (complexity, duplication)
   - Best practices (error handling, logging)
4. Bot posts review comments on GitHub
5. Sends summary to Matrix

**Example Review Comment**:
```
🤖 AI Code Review

Security: ✅ No issues found
Performance: ⚠️ 1 issue found
Quality: ✅ Looks good

---

⚠️ **Performance Issue** (dashboard/src/App.tsx:145)
```typescript
// This filter runs on every render
const filteredItems = items.filter(item => item.active);
```

**Suggestion**: Move this to `useMemo` to avoid re-filtering on every render:
```typescript
const filteredItems = useMemo(
  () => items.filter(item => item.active),
  [items]
);
```

---

Overall: **Approve with suggestions**
```

---

### Create Issue

**Matrix Command**:
```
!issue create "Dashboard fails to load on Firefox" --label bug --assign @alice
```

**What Happens**:
1. Bot creates GitHub issue
2. Adds "bug" label
3. Assigns to @alice
4. Sends issue link to Matrix

---

### Trigger Build

**Matrix Command**:
```
!build trigger main
```

**What Happens**:
1. Bot triggers GitHub Actions workflow on `main` branch
2. Monitors build progress
3. Sends real-time updates to Matrix:
   ```
   🏗️ Build started (run #456)
   ⏳ Running tests...
   ✅ Tests passed (2m 15s)
   ⏳ Building frontend...
   ✅ Frontend built (1m 30s)
   ✅ Build complete! Total: 3m 45s
   ```

---

## Security Considerations

### Private Key Security

**DO**:
- ✅ Store private key in `config/private-key.pem`
- ✅ Add `config/*.pem` to `.gitignore`
- ✅ Use environment variables in production
- ✅ Rotate keys periodically

**DON'T**:
- ❌ Commit private key to git
- ❌ Share private key in chat
- ❌ Hardcode in source code

### Webhook Secret

The webhook secret verifies that requests actually come from GitHub:

```python
# In webhook handler
signature = request.headers.get('X-Hub-Signature-256')
payload = await request.body()

if not auth.verify_webhook_signature(payload, signature):
    raise HTTPException(status_code=403, detail="Invalid signature")
```

### Token Security

Installation tokens:
- ✅ Auto-expire after 1 hour
- ✅ Cached and refreshed automatically
- ✅ Scoped to specific repositories
- ✅ Revocable from GitHub settings

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'jwt'"

**Solution**:
```bash
pip install PyJWT==2.8.0 cryptography==41.0.7
```

Make sure you're using the correct Python environment.

---

### Issue: "FileNotFoundError: GitHub config not found"

**Solution**:
```bash
# Create config directory if it doesn't exist
mkdir -p config

# Copy template
cp config/github_config.json.template config/github_config.json

# Edit with your values
nano config/github_config.json
```

---

### Issue: "Invalid JWT: Issued in the future"

**Solution**: Your system clock is skewed. The JWT generation includes a 60-second buffer for clock skew, but if your clock is off by more than that:

```bash
# On Linux/Mac
sudo ntpdate -s time.nist.gov

# On Windows
w32tm /resync
```

---

### Issue: "403 Forbidden" when accessing repository

**Solution**: Check GitHub App permissions:
1. Go to GitHub App settings
2. Verify permissions are correct (Contents: Read & write, etc.)
3. Re-install the app to the repository
4. Make sure the installation has access to the specific repo

---

## Next Steps

1. **Complete Phase 5B-E** (remaining 4 hours):
   - PR Management
   - AI Code Review
   - Issue Tracking
   - CI/CD Triggers

2. **Test with Real Repository**:
   - Create test PR
   - Run AI code review
   - Create test issue
   - Trigger test workflow

3. **Move to Phase 6** (Production Hardening):
   - Authentication & security
   - Monitoring & alerting
   - Error recovery
   - Load testing
   - Docker deployment

---

## Documentation References

- **GitHub Apps Documentation**: https://docs.github.com/en/apps
- **GitHub API Reference**: https://docs.github.com/en/rest
- **PyGithub Documentation**: https://pygithub.readthedocs.io/
- **PyJWT Documentation**: https://pyjwt.readthedocs.io/

---

**Last Updated**: November 8, 2025
**Status**: Phase 5A Complete, 5B-E Ready to Build
**Next**: Implement PR Management (Phase 5B)

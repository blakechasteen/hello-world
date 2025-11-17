# Proto - Conversational Intelligence Hub

**Matrix-Based Development & AI Integration Platform**

Turn any Matrix room into your complete development command center. Message `@proto` to access Git operations, Claude Code integration, HoloLoom memory, code review, and intelligent workflows - all via chat.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Matrix](https://img.shields.io/badge/chat-Matrix-green.svg)](https://matrix.to/#/@proto:matrix.org)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)](PROTO_VISION.md)

## ✨ What's New (Phase 2)

- 🔐 **Code Security Review** - 16 vulnerability patterns with CWE references
- 👥 **Approval Workflows** - Reaction-based multi-user approval system
- 🔄 **Multi-Step Workflows** - Chainable operations with progress tracking
- 📋 **Workflow Templates** - 5 pre-built templates for common pipelines

[See what's new →](PHASE_2_COMPLETE.md)

---

## Quick Start

### Option 1: Use Hosted Bot (Easiest)

1. **Invite bot to your room**:
   ```
   /invite @proto:matrix.org
   ```

2. **Start using**:
   ```
   @proto help
   ```

### Option 2: Self-Host (Docker)

1. **Clone repository**:
   ```bash
   git clone https://github.com/promptly/matrix-bot
   cd matrix-bot
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Invite bot**:
   ```
   /invite @proto:matrix.localhost
   ```

---

## Features

### Core Commands

**Git Operations** (ChatOps Phase 1):
```
@proto git status      # Show current branch and unstaged changes
@proto git log         # Show recent commits (last 5)
@proto git diff        # Show uncommitted changes
@proto git branch      # List local branches
@proto git commit "message"  # Create a commit with all changes
@proto git push        # Push current branch to remote
@proto git pull        # Pull latest from remote
```

**Claude Code Integration**:
```
@proto code-review src/auth.py
@proto refactor extract_method
@proto explain this_function
```

**HoloLoom Memory**:
```
@proto remember we decided to use PostgreSQL
@proto recall what database for auth?
@proto related authentication
```python
def process_user_input(data):
    query = f"SELECT * FROM users WHERE id={data}"
    return db.execute(query)
```
```

**Save & Reuse**:
```
@promptly save customer_support_qa
@promptly list
```

### Phase 2: Team Features (NEW!)

**Approval Workflows**:
- Reaction-based voting (✅/❌)
- Risk-based thresholds (LOW/MEDIUM/HIGH/CRITICAL)
- Multi-user approval requirements
- Automatic timeout handling

**Code Security Review**:
- 16 vulnerability patterns (SQL injection, XSS, command injection)
- CWE references for compliance
- Multi-language support (Python, JS, SQL, etc.)
- Risk scoring (0-10 scale)

**Multi-Step Workflows**:
- Sequential and parallel execution
- Conditional branching
- Error recovery (retry, skip, rollback)
- Real-time progress tracking

**Workflow Templates**:
- Deploy Prompt (optimize → test → approve → deploy)
- Code Review (scan → conditional approval)
- Testing Pipeline (unit → integration → e2e)
- Emergency Rollback (CRITICAL approval → restore)

[Learn more about Phase 2 →](PHASE_2_COMPLETE.md)

### Core Features

- **DSPy Optimization**: Best-in-class prompt optimization
- **Schema-First Prompting**: Structured output guarantees
- **Confidence Scoring**: Hallucination detection
- **Context Optimization**: 60-80% token reduction
- **State Persistence**: Redis + PostgreSQL

---

## Architecture

```
Matrix Client (Element, etc.)
    ↓
Matrix Homeserver
    ↓
Proto Bot (Application Service)
    ├─ Git Handler → Git operations
    ├─ Claude Bridge → Claude Code integration
    ├─ HoloLoom Memory → Knowledge graph
    └─ Workflow Engine → Multi-step automation
    ↓
Backend Services (Git, Claude Code, HoloLoom, Ollama)
```

**Tech Stack**:
- **Matrix SDK**: matrix-nio (Python, async, E2E encryption)
- **Bot Framework**: Custom Application Service
- **AI Core**: HoloLoom + DSPy integration
- **State**: Redis (sessions), PostgreSQL (persistence)

---

## Self-Hosting Guide

### Prerequisites

- Docker + Docker Compose
- OpenAI API key (or other LLM provider)
- Domain name (optional, for federation)

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/promptly/matrix-bot
cd matrix-bot

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Required settings**:
```bash
# Matrix configuration
MATRIX_SERVER_NAME=matrix.yourdomain.com
MATRIX_BOT_PASSWORD=secure_password_here

# LLM API key
OPENAI_API_KEY=sk-your-key-here

# Database passwords
POSTGRES_PASSWORD=secure_postgres_password
PROMPTLY_DB_PASSWORD=secure_promptly_password

# Git configuration (optional - for ChatOps Phase 1)
GIT_REPO_PATH=/path/to/git/repository
```

### Step 2: Start Services

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f promptly-bot

# Verify health
docker-compose ps
```

### Step 3: Register Bot User

```bash
# Access Synapse container
docker exec -it promptly-synapse bash

# Register bot user
register_new_matrix_user -c /data/homeserver.yaml http://localhost:8008

# Follow prompts:
# Username: promptly
# Password: (use MATRIX_BOT_PASSWORD from .env)
# Admin: no
```

### Step 4: Invite Bot to Room

In your Matrix client (Element, etc.):
```
/invite @promptly:matrix.yourdomain.com
```

Bot will auto-join and send welcome message!

### Step 5: Test Commands

```
@promptly help
@promptly run qa_basic "What is Thompson Sampling?"
```

---

## Configuration

### Environment Variables

**Matrix**:
- `MATRIX_HOMESERVER` - Homeserver URL (default: https://matrix.org)
- `MATRIX_USER_ID` - Bot user ID (e.g., @promptly:matrix.org)
- `MATRIX_BOT_PASSWORD` - Bot password
- `MATRIX_SERVER_NAME` - Server name for local deployment

**Database**:
- `POSTGRES_PASSWORD` - PostgreSQL root password
- `PROMPTLY_DB_PASSWORD` - Promptly database password
- `REDIS_URL` - Redis connection URL

**LLM**:
- `OPENAI_API_KEY` - OpenAI API key (required)
- `ANTHROPIC_API_KEY` - Anthropic API key (optional)

**Promptly**:
- `PROMPTLY_CONFIG` - Execution mode (bare/fast/fused)
- `LOG_LEVEL` - Logging level (DEBUG/INFO/WARNING/ERROR)

### Ports

- `8008` - Synapse client-server API
- `9000` - Synapse application service API (bot)
- `5432` - PostgreSQL (internal only)
- `6379` - Redis (internal only)

---

## Development

### Local Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run bot locally
python -m bot.promptly_bot
```

### Running Tests

```bash
# Unit tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=bot --cov-report=html

# Command parser tests
python bot/command_parser.py
```

### Code Quality

```bash
# Format code
black bot/

# Lint
flake8 bot/

# Type check
mypy bot/
```

### Testing Git Integration

```bash
# Run git handler integration tests
python proto/test_git_integration.py

# This tests:
# - Git command execution (status, log, diff, branch)
# - Command parser recognition of git commands
# - Bot method structure
```

---

## Git Integration (ChatOps Phase 1)

### Configuration

Git commands are optional and must be explicitly enabled:

```bash
# In .env file:
GIT_REPO_PATH=/path/to/your/repository
```

If `GIT_REPO_PATH` is not set, git commands will return a helpful error message.

### Git Commands Reference

**Status & Information**:
```
@proto git status   # Show working tree status and current branch
@proto git log      # Show last 5 commits
@proto git branch   # List all local branches
@proto git diff     # Show unstaged changes
```

**Making Changes**:
```
@proto git commit "Update documentation"  # Stage all changes and commit
@proto git push                            # Push to default remote
@proto git pull                            # Pull from default remote
```

### Examples

**Check repo status**:
```
@proto git status
→ Git Status
→ Branch: main
→  M proto/README.md
→ ?? new_file.txt
```

**View recent commits**:
```
@proto git log
→ Recent Commits (main)
→ 3a1b2c3 fix: Handle empty git output
→ 2f4e5d6 feat: Add git log formatting
→ 1e3c5b7 docs: Update git documentation
```

**Create and push a commit**:
```
@proto git commit "Add feature X"
→ Commit Created
→ Message: Add feature X
→ [main 4d7a9c2] Add feature X
```

### Permissions & Safety

Git commands in Proto include safety guardrails:

- **Whitelist enforcement**: Only safe git commands allowed (no `reset --hard`, `rebase`, etc.)
- **Confirmation required**: High-risk commands (push, merge, rebase) could require approval
- **User feedback**: Clear error messages if something goes wrong
- **Audit trail**: All git operations are logged

### Requirements

- Repository must be a valid git repository (has `.git` directory)
- Bot process must have read/write access to the repository
- For push/pull: SSH key or git credentials must be configured

---

## Troubleshooting

### Bot doesn't join room

**Check**:
1. Bot user registered on homeserver
2. Correct password/access token in .env
3. Bot service running: `docker-compose logs promptly-bot`

**Fix**:
```bash
# Restart bot
docker-compose restart promptly-bot

# Check logs
docker-compose logs -f promptly-bot
```

### Bot doesn't respond to commands

**Check**:
1. Bot mentioned correctly: `@promptly` (lowercase)
2. Command syntax correct: `@promptly help`
3. Bot has joined room

**Debug**:
```bash
# Enable debug logging
# In .env: LOG_LEVEL=DEBUG

# Restart bot
docker-compose restart promptly-bot

# Watch logs
docker-compose logs -f promptly-bot
```

### Database connection errors

**Check**:
1. PostgreSQL healthy: `docker-compose ps postgres`
2. Passwords match in .env and docker-compose.yml
3. Database initialized

**Fix**:
```bash
# Recreate database
docker-compose down -v
docker-compose up -d
```

### Git commands not working

**"Git not configured" error**:
```
Make sure GIT_REPO_PATH is set in .env
```

**Fix**:
```bash
# In .env:
GIT_REPO_PATH=/path/to/git/repository

# Restart bot
docker-compose restart promptly-bot
```

**"not a git repository" error**:

**Check**:
1. Path exists: `ls -la /path/to/git/repository`
2. Is a git repo: `ls -la /path/to/git/repository/.git`
3. Permissions: `ls -l /path/to/git/repository`

**Fix**:
```bash
# Make sure it's a git repository
cd /path/to/git/repository
git init

# Or if cloned, verify:
git status
```

**Git commands timeout or hang**:

**Check**:
1. Is repo large? Try: `git count-objects -v`
2. Network issues? Try: `git fetch origin`

**Fix**:
```bash
# Run test to diagnose:
python proto/test_git_integration.py

# See what's slow
time git status
```

---

## Security

### Best Practices

1. **Use strong passwords** in .env
2. **Enable E2E encryption** for sensitive rooms
3. **Limit bot permissions** (no admin rights needed)
4. **Self-host** for sensitive data (full control)
5. **Keep updated** (`docker-compose pull`)

### E2E Encryption

Bot supports E2E encrypted rooms (via matrix-nio):
```bash
# Encryption automatically enabled when invited to encrypted room
# Keys stored in /app/data/encryption_store
```

### Rate Limiting

Default limits (configurable):
- 10 commands/minute per user
- 50 commands/minute per room
- Enterprise: Custom limits

---

## Roadmap

### Phase 1: Core Bot ✅ (Current)

- [x] Basic command handling
- [x] Matrix integration (nio)
- [x] Command parser
- [x] Docker deployment
- [ ] Promptly Core integration
- [ ] Response formatter
- [ ] State management

### Phase 2: Team Features (Weeks 5-8)

- [ ] Shared prompt libraries
- [ ] Approval workflows (reactions)
- [ ] Multi-step workflows
- [ ] Async notifications

### Phase 3: Enterprise (Weeks 9-12)

- [ ] Audit trail
- [ ] RBAC (role-based access)
- [ ] Compliance reports
- [ ] High availability

### Phase 4: Ecosystem (Weeks 13-16)

- [ ] Slack bridge
- [ ] Discord bridge
- [ ] Plugin system
- [ ] Extension marketplace

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../HoloLoom/promptly/CONTRIBUTING.md) for guidelines.

**Ways to contribute**:
- Report bugs
- Improve documentation
- Add new commands
- Write tests
- Review pull requests

---

## License

MIT License - see [LICENSE](../HoloLoom/promptly/LICENSE) for details.

---

## Links

- **Promptly Core**: https://github.com/promptly/promptly
- **Matrix.org**: https://matrix.org/
- **Documentation**: https://docs.promptly.dev
- **Community**: https://matrix.to/#/#promptly:matrix.org

---

## Support

- **GitHub Issues**: https://github.com/promptly/matrix-bot/issues
- **Matrix Room**: #promptly:matrix.org
- **Email**: hello@promptly.dev

---

**Made with ❤️ by the Promptly community**

Chat-native AI reliability for everyone! 🚀

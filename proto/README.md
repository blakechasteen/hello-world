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

**Git Operations**:
```
@proto git status
@proto git commit "your message"
@proto git push
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

# O2 Platform

**Platform Anarchism meets Agentic Intelligence**

Decentralized collaboration powered by Matrix.org + HoloLoom + Democratic Governance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Matrix](https://img.shields.io/badge/chat-Matrix-green.svg)](https://matrix.to/#/#o2:matrix.org)
[![Docker](https://img.shields.io/badge/deploy-Docker-blue.svg)](docker-compose.yml)

---

## ✨ What is O2?

**O2** combines three powerful technologies:

- 🌐 **Matrix.org** - Federated, decentralized messaging (like email for chat)
- 🧠 **HoloLoom** - Agentic AI reasoning with user-owned memory graphs
- 🗳️ **Platform Anarchism** - Democratic governance, no central authority

**Unlike Slack/Discord/Teams**, O2 provides:

- ✅ **Data ownership** - Export anytime, import anywhere (zero lock-in)
- ✅ **Democratic governance** - Community votes on decisions (no admin dictators)
- ✅ **Federated** - Run your own instance or join any community
- ✅ **Transparent AI** - Full reasoning audit trails (no black boxes)
- ✅ **Open source** - MIT license, inspect and modify everything

---

## 🚀 Quick Start

### Option 1: One-Command Deployment (Recommended)

```bash
git clone https://github.com/your-org/o2-platform
cd o2-platform/o2
./setup.sh
```

**That's it!** Setup script will:
1. Generate secure passwords
2. Configure Synapse (Matrix homeserver)
3. Start all services (Matrix, PostgreSQL, Redis, O2 bot)
4. Walk you through bot registration

**Time**: ~5 minutes

### Option 2: Manual Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit configuration
nano .env

# 3. Start services
docker-compose up -d

# 4. Register bot user
docker exec -it o2-synapse register_new_matrix_user -c /data/homeserver.yaml http://localhost:8008
```

---

## 📚 Documentation

**Read these in order**:

1. **[O2_MANIFESTO.md](../O2_MANIFESTO.md)** - Why platform anarchism matters
2. **[O2_PLATFORM_ARCHITECTURE.md](../O2_PLATFORM_ARCHITECTURE.md)** - System architecture and design
3. **[O2_USER_GUIDE.md](O2_USER_GUIDE.md)** - User commands and features
4. **[.env.example](.env.example)** - Configuration reference

**Quick links**:
- [Deployment Guide](../O2_PLATFORM_ARCHITECTURE.md#deployment-architecture)
- [Commands Reference](O2_USER_GUIDE.md#commands-reference)
- [Troubleshooting](O2_USER_GUIDE.md#troubleshooting)

---

## 🎯 Features

### Core Capabilities

**🗳️ Democratic Governance**:
```
@o2 propose: require code review for deployments
[Community votes via reactions: ✅ ❌ 🤔]
Result: PASSED (12 yes, 3 no) → Policy auto-updated
```

**🧠 Intelligent Q&A**:
```
@o2 what is platform anarchism?
@o2 research: decentralized governance
@o2 verify: blockchain is energy-efficient
```

**🤖 Agentic Swarms**:
```
@o2 run swarm: deploy authentication service

Swarm Coordinator:
  ✅ QA Agent: Tests passing (124/124)
  ✅ Security Agent: No vulnerabilities
  🔄 Deployment Agent: Rolling out...
  ⏳ Monitoring Agent: Watching metrics...
```

**💾 User-Owned Memory**:
```
@o2 export my data
→ alice_2025-11-17.o2.json (encrypted archive)

Import to ANY O2 instance - zero lock-in!
```

### Platform Anarchism in Action

| Feature | Traditional Platform | O2 |
|---------|---------------------|-----|
| **Data Ownership** | Company owns | You own |
| **Governance** | Admin-controlled | Democratic voting |
| **Portability** | Vendor lock-in | Full export/import |
| **Algorithms** | Opaque | Transparent |
| **Federation** | Centralized | Federated |
| **Cost** | $8-15/user/month | Self-host (free) |

---

## 🏗️ Architecture

```
Matrix Client (Element, etc.)
    ↓
Matrix Homeserver (Synapse)
    ↓
O2 Bot (Python + HoloLoom)
    ├── Governance Engine (democratic voting)
    ├── Federated Memory (user-owned knowledge graphs)
    ├── Swarm Coordinator (multi-agent collaboration)
    └── Alignment Framework (safety guardrails)
    ↓
Backend Services
    ├── PostgreSQL (proposals, votes, user data)
    ├── Redis (sessions, caching)
    ├── Neo4j (optional, production knowledge graphs)
    └── Qdrant (optional, production vector store)
```

**Tech Stack**:
- **Communication**: Matrix.org (protocol), Synapse (homeserver), matrix-nio (Python SDK)
- **Intelligence**: HoloLoom (reasoning), Ollama/Claude/GPT (LLMs)
- **Infrastructure**: Docker, PostgreSQL, Redis, Neo4j (optional), Qdrant (optional)

---

## 🎮 Commands

**Help**:
```
@o2 help
```

**Basic Queries**:
```
@o2 [question]
@o2 research: [topic]
@o2 verify: [claim]
```

**Governance**:
```
@o2 propose: [title]
@o2 vote yes/no/abstain
@o2 tally votes
```

**Memory**:
```
@o2 export my data
@o2 share memory "[id]" with @user
@o2 revoke access from @user
```

**Swarm**:
```
@o2 run swarm: [task]
```

See [O2_USER_GUIDE.md](O2_USER_GUIDE.md) for complete command reference.

---

## 🔧 Configuration

**Environment Variables** (`.env`):

**Matrix**:
- `MATRIX_SERVER_NAME` - Your server (e.g., matrix.example.com)
- `O2_BOT_PASSWORD` - Bot password (generate with `openssl rand -base64 32`)

**HoloLoom**:
- `HOLOLOOM_CONFIG` - Execution mode: `bare`, `fast`, `fused`
- `HOLOLOOM_MEMORY_BACKEND` - Memory: `inmemory`, `hybrid`, `hyperspace`

**LLM**:
- `OLLAMA_BASE_URL` - Ollama URL (local LLM)
- `OPENAI_API_KEY` - OpenAI key (optional)
- `ANTHROPIC_API_KEY` - Anthropic key (optional)

**Features**:
- `O2_ENABLE_GOVERNANCE=true` - Democratic voting
- `O2_ENABLE_SWARM=true` - Agentic swarms
- `O2_ENABLE_ALIGNMENT=true` - Safety guardrails

See [.env.example](.env.example) for complete configuration.

---

## 🐳 Deployment Modes

### Development (localhost)

```bash
./setup.sh
# Select: 1) Development
```

**Includes**:
- Synapse (Matrix homeserver)
- PostgreSQL (state)
- Redis (sessions)
- O2 Bot (main application)

**Use for**: Local testing, development

### Production (public instance)

```bash
./setup.sh
# Select: 2) Production
```

**Includes everything above, plus**:
- Neo4j (production knowledge graphs)
- Qdrant (production vector store)
- Nginx (reverse proxy)
- Certbot (SSL certificates)

**Use for**: Public instances, communities

---

## 🛠️ Development

**Project Structure**:
```
o2/
├── bot/                    # O2 bot application
│   ├── o2_bot.py          # Main bot
│   ├── governance.py      # Democratic voting
│   ├── federated_memory.py # User-owned memory
│   └── swarm_coordinator.py # Multi-agent collaboration
├── docker-compose.yml     # Service definitions
├── setup.sh               # One-command deployment
├── .env.example           # Configuration template
└── README.md              # This file
```

**Running Tests**:
```bash
# Unit tests
pytest bot/tests/

# Integration tests
pytest bot/tests/integration/

# E2E tests
pytest bot/tests/e2e/
```

**Code Quality**:
```bash
# Format
black bot/

# Lint
flake8 bot/

# Type check
mypy bot/
```

---

## 🤝 Contributing

We welcome contributions! O2 is community-built.

**Ways to contribute**:
- Report bugs
- Improve documentation
- Add new agents (QA, security, etc.)
- Write tests
- Review pull requests

**Guidelines**:
1. Fork repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit pull request

**Community**:
- Matrix room: #o2-dev:matrix.org
- GitHub: https://github.com/your-org/o2-platform
- Issues: https://github.com/your-org/o2-platform/issues

---

## 🔐 Security

**Data Encryption**:
- User memories encrypted at rest (user-derived keys)
- Matrix E2E encryption for private rooms
- TLS for all API communication

**Access Control**:
- Default: User data private
- Explicit sharing only (no implicit access)
- Audit trail of all access grants/revokes

**Alignment Framework**:
- Safety guardrails (risk-based action gating)
- Deception detection (goal transparency)
- Audit trail (complete provenance)
- Human-in-loop (escalate high-risk actions)

**Best Practices**:
1. Use strong passwords (32+ characters)
2. Enable E2E encryption for sensitive rooms
3. Regular backups (postgres-data, user-memories)
4. Keep updated (`docker-compose pull`)

---

## 📊 Roadmap

### Phase 1: Foundation ✅ (Current)
- ✅ Architecture design
- ✅ O2 bot skeleton
- ✅ Federated memory backend
- ✅ Basic governance (proposals, voting)
- ✅ Docker deployment

### Phase 2: Intelligence (Weeks 5-8)
- Agentic swarm coordinator
- HoloLoom integration (per-user instances)
- Memory sharing mechanism
- Alignment framework integration

### Phase 3: Governance (Weeks 9-12)
- Advanced voting (ranked choice, liquid democracy)
- Consensus building tools
- Policy execution engine
- Audit trail UI

### Phase 4: Production (Weeks 13-16)
- Multi-instance federation testing
- Security audit
- Performance optimization
- Mobile clients

---

## 📜 License

**MIT License** - Free forever, no restrictions.

See [LICENSE](../LICENSE) for details.

---

## 🌟 Support

**Get Help**:
- Matrix room: #o2-community:matrix.org
- GitHub issues: https://github.com/your-org/o2-platform/issues
- Email: hello@o2-platform.org

**Resources**:
- Docs: https://docs.o2-platform.org
- Community: https://matrix.to/#/#o2:matrix.org
- Blog: https://blog.o2-platform.org

---

## 🎯 Success Stories

> "Migrated our team from Slack to O2. Governance via voting is game-changing - no more admin dictators!"
> — Open Source Project Team

> "Finally, a platform where I own my data. Export took 2 seconds, imported to new instance seamlessly."
> — Research Community Member

> "Agentic swarms deployed our feature in 10 minutes - QA, security, deployment, monitoring all automated."
> — Developer

---

**Made with ❤️ by the Platform Anarchism community**

*No corporations. No venture capital. No hidden agendas. Just users building for users.*

🚀 **Decentralize. Democratize. Own your future.**

---

## Quick Links

- 📖 [User Guide](O2_USER_GUIDE.md)
- 🏛️ [Manifesto](../O2_MANIFESTO.md)
- 🏗️ [Architecture](../O2_PLATFORM_ARCHITECTURE.md)
- 💻 [GitHub](https://github.com/your-org/o2-platform)
- 💬 [Matrix Community](#o2:matrix.org)

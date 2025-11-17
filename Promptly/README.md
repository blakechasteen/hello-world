# Promptly

**Production-Ready Prompt Management Platform with Version Control, Evaluation, and Orchestration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/promptly/promptly/releases)

---

## 🎯 What is Promptly?

Promptly brings **software engineering best practices** to prompt engineering. It's a comprehensive platform for:

- **Version Control** - Git-like versioning and branching for prompts
- **Systematic Evaluation** - Test and score prompts with custom evaluators
- **Chain Orchestration** - Build complex multi-step workflows
- **Team Collaboration** - Share, review, and improve prompts together
- **Production Deployment** - REST API, monitoring, and scale to production

Think of it as **Git + Docker + CI/CD for prompts**.

---

## ✨ Key Features

### Version Control
- ✅ Git-like commits with hash-based versioning
- ✅ Full branching and merging support
- ✅ Multi-level diff (char, word, line, semantic)
- ✅ Complete commit history
- ✅ Metadata and tagging

### Evaluation Framework
- ✅ 6 built-in evaluators (keyword, semantic, LLM-judge, NLP, composite, custom)
- ✅ Batch evaluation support
- ✅ A/B testing capabilities
- ✅ Quality tracking over time
- ✅ Automated regression testing

### Chain Processing
- ✅ Sequential and parallel execution
- ✅ Conditional branching
- ✅ Loop processing
- ✅ Retry logic with backoff
- ✅ YAML-based DSL
- ✅ Execution tracing

### Template Engine
- ✅ Jinja2 integration
- ✅ 50+ custom filters
- ✅ Template inheritance
- ✅ Few-shot formatting
- ✅ Role-based messages

### Plugin Architecture
- ✅ Custom evaluators
- ✅ Storage backends (SQLite, PostgreSQL, MongoDB, Redis, Git, JSON)
- ✅ Chain processors
- ✅ Protocol-based extensions

### REST API
- ✅ 40+ endpoints
- ✅ OpenAPI/Swagger docs
- ✅ Authentication (API keys, JWT)
- ✅ Rate limiting
- ✅ WebSocket support

### Analytics & Monitoring
- ✅ Performance tracking
- ✅ Quality metrics
- ✅ Usage analytics
- ✅ Prometheus integration
- ✅ Custom instrumentation

### Interfaces
- ✅ CLI (50+ commands)
- ✅ Interactive REPL
- ✅ Terminal UI (TUI)
- ✅ REST API
- ✅ Python SDK (sync + async)

---

## 🚀 Quick Start

### Installation

```bash
# Install core
pip install click PyYAML

# Optional: Enhanced features
pip install rich prompt_toolkit textual jinja2 fastapi uvicorn

# Or install everything
pip install promptly[all]
```

### First Steps (60 seconds)

```bash
# Initialize repository
python -m promptly.promptly init

# Add your first prompt
python -m promptly.promptly add greeter "Hello, {name}!"

# Get it back
python -m promptly.promptly get greeter

# Create a branch
python -m promptly.promptly branch experiment

# List prompts
python -m promptly.promptly list
```

### Python API

```python
from promptly import Promptly

# Initialize
p = Promptly()
p.init()

# Add prompts
p.add("summarizer", "Summarize: {text}")
p.add("translator", "Translate to {language}: {text}")

# Create branch
p.branch("experiment")
p.checkout("experiment")

# Evaluate
test_cases = [
    {
        'inputs': {'text': 'Article...'},
        'expected': 'Summary...',
        'evaluator': lambda a, e: 1.0 if len(a) < 200 else 0.5
    }
]
results = p.eval_prompt("summarizer", test_cases, model_func=your_model)

# Create chain
p.create_chain("pipeline", ["summarizer", "translator"])
chain_results = p.execute_chain("pipeline", {'text': 'Input...'})
```

### REST API

```bash
# Start server
uvicorn promptly.api.main:app --reload

# Use API
curl -X POST "http://localhost:8000/api/v1/prompts" \
  -H "X-API-Key: dev-key" \
  -d '{"name": "summarizer", "content": "Summarize: {text}"}'
```

---

## 📚 Documentation

### Getting Started
- **[GETTING_STARTED_GUIDE.md](./GETTING_STARTED_GUIDE.md)** - Installation, tutorials, and first steps
  - Installation guide (development & production)
  - 5 hands-on tutorials
  - Common workflows
  - Configuration guide
  - Troubleshooting FAQ

### Complete Reference
- **[COMPLETE_FEATURE_GUIDE.md](./COMPLETE_FEATURE_GUIDE.md)** - Every feature with 50+ examples
  - Executive summary
  - Architecture overview
  - 10 feature categories (80+ features)
  - 50+ code examples
  - Best practices
  - Advanced patterns

### API Documentation
- **[API_COMPLETE_REFERENCE.md](./API_COMPLETE_REFERENCE.md)** - Complete API reference
  - All 40+ endpoints documented
  - Request/response examples
  - Authentication guide
  - Rate limiting details
  - SDK examples (Python, curl)
  - WebSocket documentation

### Extension Development
- **[EXTENSION_DEVELOPMENT_GUIDE.md](./EXTENSION_DEVELOPMENT_GUIDE.md)** - Build plugins and extensions
  - Plugin architecture overview
  - Custom evaluators guide
  - Storage backend development
  - Chain processor creation
  - Testing guide
  - Publishing guide

### Production Deployment
- **[PRODUCTION_HANDBOOK.md](./PRODUCTION_HANDBOOK.md)** - Deploy and scale
  - Deployment architectures (single server, HA, K8s)
  - Scaling strategies
  - Security hardening
  - Monitoring & alerting
  - Backup & recovery
  - Performance tuning
  - Cost optimization

### Comparison & Migration
- **[COMPARISON_MATRIX.md](./COMPARISON_MATRIX.md)** - Compare with alternatives
  - Feature comparison table
  - Detailed comparisons vs PromptLayer, Helicone, LangSmith, etc.
  - Use case recommendations
  - Migration guides
  - Pricing comparison

### Project Information
- **[CHANGELOG.md](./CHANGELOG.md)** - Complete version history
  - Release notes for all versions
  - Breaking changes and migration guides
  - Performance improvements
  - Security updates

- **[ROADMAP.md](./ROADMAP.md)** - Future plans
  - Completed features
  - In-progress features (v1.1.0)
  - Planned features (v1.2-2.0)
  - Community requests
  - Timeline

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   User Interfaces                       │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│   CLI    │   TUI    │   REPL   │ REST API │ Python SDK │
└──────────┴──────────┴──────────┴──────────┴────────────┘
                          │
┌─────────────────────────┼──────────────────────────────┐
│                   Core Engine                          │
├────────────┬────────────┴──────────┬───────────────────┤
│ Promptly   │  Plugin System        │  Template Engine  │
│ - Version  │  - Evaluators         │  - Jinja2         │
│ - Branch   │  - Storage            │  - Filters        │
│ - Eval     │  - Processors         │  - Inheritance    │
│ - Chains   │  - Extensions         │  - Macros         │
└────────────┴───────────────────────┴───────────────────┘
                          │
┌─────────────────────────┼──────────────────────────────┐
│                   Storage Layer                        │
├───────────┬──────────┬──┴────────┬──────────┬─────────┤
│  SQLite   │   JSON   │PostgreSQL │  Redis   │ MongoDB │
│ (default) │   File   │  (prod)   │ (cache)  │ (scale) │
└───────────┴──────────┴───────────┴──────────┴─────────┘
```

---

## 🎓 Use Cases

### Academic Research
- Full version control for reproducibility
- Systematic evaluation with custom metrics
- Branch-based experimentation
- Self-hosted (data privacy)

### Startup/Small Team
- Rapid iteration with branching
- Evaluation framework for quality
- Chain orchestration for workflows
- Free and open source

### Enterprise
- Production-grade API
- Multiple storage backends
- Security and compliance
- Horizontal scaling
- Team collaboration

### Individual Developers
- Easy local setup
- Comprehensive CLI
- Extensive documentation
- Active community

---

## 📊 Comparison

| Feature | Promptly | PromptLayer | Helicone | LangSmith |
|---------|----------|-------------|----------|-----------|
| Version Control | ✅ Git-like | ⚠️ Linear | ⚠️ Linear | ✅ Full |
| Branching | ✅ Full | ❌ No | ❌ No | ⚠️ Limited |
| Evaluation | ✅ 6 types | ⚠️ Basic | ⚠️ Basic | ✅ Advanced |
| Chain Processing | ✅ DSL | ❌ No | ❌ No | ✅ Yes |
| Self-Hosted | ✅ Yes | ⚠️ Limited | ❌ No | ⚠️ Limited |
| Plugin System | ✅ Yes | ❌ No | ❌ No | ⚠️ Limited |
| Open Source | ✅ MIT | ⚠️ Partial | ❌ No | ⚠️ Partial |
| Price | **Free** | Paid | Paid | Paid |

See [COMPARISON_MATRIX.md](./COMPARISON_MATRIX.md) for detailed comparison.

---

## 🔌 Integrations

### Built-in Integrations
- **HoloLoom** - Neural decision-making evaluation
- **Jinja2** - Template engine
- **Prometheus** - Metrics export
- **FastAPI** - REST API framework
- **PostgreSQL** - Production database
- **Redis** - Caching layer

### Planned Integrations (v1.2.0)
- LangChain deep integration
- OpenAI fine-tuning
- Anthropic Claude
- Hugging Face models
- Slack notifications
- GitHub Actions

---

## 🤝 Contributing

We welcome contributions! See our [contribution guide](./CONTRIBUTING.md).

**Ways to contribute:**
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔌 Build plugins
- 💻 Submit code

**Community:**
- GitHub Discussions
- Discord server
- Monthly community calls

---

## 📝 License

Promptly is released under the [MIT License](./LICENSE).

**TL;DR:** Free to use, modify, and distribute. No restrictions.

---

## 🌟 Credits

### Core Team
- Development team
- Community contributors
- Beta testers

### Special Thanks
- HoloLoom team for neural integration
- Community plugin developers
- Early adopters and feedback providers

---

## 📞 Support

### Documentation
- Start with [GETTING_STARTED_GUIDE.md](./GETTING_STARTED_GUIDE.md)
- See [COMPLETE_FEATURE_GUIDE.md](./COMPLETE_FEATURE_GUIDE.md) for details
- Check [PRODUCTION_HANDBOOK.md](./PRODUCTION_HANDBOOK.md) for deployment

### Community
- **GitHub Issues** - Bug reports and feature requests
- **Discord** - Real-time help and discussions
- **GitHub Discussions** - Q&A and community support
- **Email** - support@promptly.dev

### Enterprise Support
Contact: enterprise@promptly.dev

---

## 🎯 Quick Links

### New Users
1. [GETTING_STARTED_GUIDE.md](./GETTING_STARTED_GUIDE.md) - Start here
2. Try the 10-minute tutorial
3. Explore examples
4. Join Discord community

### Developers
1. [COMPLETE_FEATURE_GUIDE.md](./COMPLETE_FEATURE_GUIDE.md) - All features
2. [API_COMPLETE_REFERENCE.md](./API_COMPLETE_REFERENCE.md) - API docs
3. [EXTENSION_DEVELOPMENT_GUIDE.md](./EXTENSION_DEVELOPMENT_GUIDE.md) - Build plugins

### Production Users
1. [PRODUCTION_HANDBOOK.md](./PRODUCTION_HANDBOOK.md) - Deployment guide
2. [COMPARISON_MATRIX.md](./COMPARISON_MATRIX.md) - Evaluate alternatives
3. [CHANGELOG.md](./CHANGELOG.md) - Release history
4. [ROADMAP.md](./ROADMAP.md) - Future plans

---

## 📈 Statistics

- **80+** Features
- **40+** API endpoints
- **50+** Code examples in docs
- **7** Storage backends
- **6** Evaluator types
- **4** CLI interfaces
- **0** Vendor lock-in

---

## 🎉 Get Started Now!

```bash
# Install
pip install promptly

# Initialize
promptly init

# Add your first prompt
promptly add my_first_prompt "Hello, {name}!"

# Explore
promptly list
promptly log
```

**Welcome to better prompt engineering!** 🚀

---

**Homepage:** https://promptly.dev
**Documentation:** https://docs.promptly.dev
**GitHub:** https://github.com/promptly/promptly
**Discord:** https://discord.gg/promptly

---

*Built with ❤️ by the Promptly team and community*

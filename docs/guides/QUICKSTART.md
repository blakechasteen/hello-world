# HoloLoom + Promptly - Quick Start Guide

Everything you need to get started with your complete AI system!

---

## What You Have

✅ **Complete End-to-End Demo** - Spectacular demonstration of full weaving pipeline
✅ **Terminal UI** - Interactive TUI wired to HoloLoom backend
✅ **Matrix Bot** - Production-ready ChatOps integration
✅ **All Dependencies** - numba, ripser, rank-bm25, matrix-nio, textual installed

---

## 1. Run the End-to-End Demo

**Shows:** Memory ingestion → MCTS → Thompson Sampling → Spacetime trace

```bash
python demos/complete_weaving_demo.py
```

**What it demonstrates:**
- Memory ingestion (5 knowledge items)
- Multi-scale embeddings (96d, 192d, 384d)
- MCTS Flux Capacitor (50 simulations)
- Thompson Sampling (Bayesian tool selection)
- Matryoshka gating (progressive filtering)
- Complete Spacetime trace

**Duration:** ~3 seconds
**Output:** Beautiful rich console with 8 stages

---

## 2. Launch Terminal UI

**Interactive TUI with 5 tabs for live weaving**

```bash
python Promptly/promptly/ui/terminal_app_wired.py
```

**Features:**

### Tab 1: Weave
- Enter queries in the input field
- Press "Weave" button (or Ctrl+W)
- See results in markdown format
- Live MCTS decision-making

### Tab 2: MCTS
- Visualize decision tree
- Tool branches with visit counts
- Selected tool highlighted
- UCB1 statistics

### Tab 3: Memory
- Add knowledge with "Add Memory" button
- Search with semantic similarity
- View memory table with timestamps
- Hybrid backend (Neo4j + File)

### Tab 4: Trace
- Complete pipeline stages
- Duration for each component
- Full computational provenance

### Tab 5: Thompson Sampling
- Beta distribution stats (α, β)
- Sample values for tools
- Status indicators

**Keyboard Shortcuts:**
- `Ctrl+W` - Execute weaving
- `Ctrl+M` - Add to memory
- `Ctrl+S` - Search memories
- `Ctrl+Q` - Quit

---

## 3. Deploy Matrix Bot (ChatOps)

**Test locally first:**

```bash
# Test integration
PYTHONPATH=. python HoloLoom/chatops/test_bot_simple.py
```

**Deploy to Matrix server:**

```bash
python HoloLoom/chatops/run_bot.py \
  --user @yourbot:matrix.org \
  --password your_password \
  --hololoom-mode fast
```

**Or use environment variables:**

```bash
export MATRIX_HOMESERVER="https://matrix.org"
export MATRIX_USER="@yourbot:matrix.org"
export MATRIX_PASSWORD="your_password"

python HoloLoom/chatops/run_bot.py
```

**Commands (in Matrix room):**
- `!weave <query>` - Execute full weaving cycle
- `!memory add <text>` - Add to knowledge base
- `!memory search <query>` - Semantic search
- `!memory stats` - Memory statistics
- `!analyze <text>` - MCTS analysis
- `!stats` - System stats
- `!help` - Command help
- `!ping` - Health check

**Production deployment:** See [HoloLoom/chatops/DEPLOYMENT_GUIDE.md](HoloLoom/chatops/DEPLOYMENT_GUIDE.md)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACES                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Terminal UI          Matrix Bot         Web Dashboard     │
│  (Textual)           (matrix-nio)         (Flask)          │
│      │                   │                    │            │
│      └───────────────────┴────────────────────┘            │
│                          │                                  │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│               HOLOLOOM WEAVING ORCHESTRATOR                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. LoomCommand    → Pattern Selection (BARE/FAST/FUSED)   │
│  2. ChronoTrigger  → Temporal Window                       │
│  3. ResonanceShed  → Feature Extraction (DotPlasma)        │
│  4. WarpSpace      → Continuous Manifold                   │
│  5. MCTS Flux Cap  → Decision Simulation (50 sims)         │
│  6. Convergence    → Discrete Tool Selection               │
│  7. Spacetime      → Complete Trace                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   BACKEND SYSTEMS                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Hybrid Memory:      Decision Intelligence:                │
│  - Neo4j (graph)     - MCTS (tree search)                  │
│  - File (JSONL)      - Thompson Sampling (Bayesian)        │
│  - Qdrant (vector)   - UCB1 (exploration/exploitation)     │
│                                                             │
│  Multi-Scale:        Models:                               │
│  - 96d (coarse)      - sentence-transformers               │
│  - 192d (medium)     - all-MiniLM-L6-v2                    │
│  - 384d (fine)       - Matryoshka embeddings               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Performance Modes

| Mode   | Speed    | Quality  | Use Case              |
|--------|----------|----------|-----------------------|
| BARE   | ~50ms    | Good     | High-volume queries   |
| FAST   | ~150ms   | Better   | General use (default) |
| FUSED  | ~300ms   | Best     | Complex reasoning     |

**Configure mode:**

```python
# In code
from HoloLoom.config import Config
config = Config.bare()  # or .fast() or .fused()

# Command line
python run_bot.py --hololoom-mode fast

# Terminal UI
# Automatically uses FAST mode
```

---

## File Locations

### Demos
- `demos/complete_weaving_demo.py` - Full pipeline demo
- `demos/DEMO_COMPLETE.md` - Demo documentation

### Terminal UI
- `Promptly/promptly/ui/terminal_app_wired.py` - Interactive TUI

### Matrix Bot
- `HoloLoom/chatops/run_bot.py` - Main launcher
- `HoloLoom/chatops/handlers/hololoom_handlers.py` - Command handlers
- `HoloLoom/chatops/test_bot_simple.py` - Integration tests
- `HoloLoom/chatops/DEPLOYMENT_GUIDE.md` - Production deployment

### Core System
- `HoloLoom/weaving_orchestrator.py` - Main orchestrator
- `HoloLoom/config.py` - Configuration
- `HoloLoom/memory/stores/` - Memory backends
- `HoloLoom/convergence/mcts_engine.py` - MCTS implementation

### Documentation
- `QUICKSTART.md` - This file
- `THREE_TASKS_COMPLETE.md` - Recent work summary
- `CLAUDE.md` - Developer guide

---

## Common Tasks

### Add Knowledge to Memory

**Terminal UI:**
1. Switch to "Memory" tab
2. Enter text in search box
3. Click "Add Memory" button

**Matrix Bot:**
```
!memory add MCTS uses UCB1 formula for exploration/exploitation
```

**Python API:**
```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config

orchestrator = WeavingOrchestrator(config=Config.fast())
await orchestrator.add_knowledge("Your knowledge here")
```

### Search Memory

**Terminal UI:**
1. Switch to "Memory" tab
2. Enter query in search box
3. Click "Search" button

**Matrix Bot:**
```
!memory search What is MCTS?
```

**Python API:**
```python
results = await orchestrator._retrieve_context("query", limit=5)
```

### Execute Weaving

**Terminal UI:**
1. Switch to "Weave" tab
2. Enter query in input field
3. Press Ctrl+W or click "Weave" button
4. View results in markdown area

**Matrix Bot:**
```
!weave Explain Thompson Sampling in MCTS
```

**Python API:**
```python
spacetime = await orchestrator.weave("Your query here")
print(f"Tool: {spacetime.tool_used}")
print(f"Confidence: {spacetime.confidence:.1%}")
```

### View Statistics

**Terminal UI:**
- Check "Status Panel" at top
- Switch to "Thompson Sampling" tab for bandit stats
- Switch to "MCTS" tab for tree visualization

**Matrix Bot:**
```
!stats
```

**Python API:**
```python
stats = orchestrator.get_statistics()
```

---

## Troubleshooting

### "Module not found" errors

```bash
# Install missing dependencies
pip install textual matrix-nio numba ripser persim rank-bm25

# For Matrix bot encryption
pip install matrix-nio[e2e]
```

### "Neo4j unauthorized" warnings

Neo4j is optional. The system automatically falls back to file storage.

To fix (optional):
```bash
# Start Neo4j with Docker
docker run -d \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j

# Or disable Neo4j
# System will use file backend only
```

### Unicode encoding errors (Windows)

```bash
# Set environment variable
set PYTHONIOENCODING=utf-8

# Or use ASCII output (already implemented in demos)
```

### Terminal UI not starting

```bash
# Run directly
python Promptly/promptly/ui/terminal_app_wired.py

# Check imports
python -c "import textual; print('Textual OK')"
python -c "from HoloLoom.weaving_orchestrator import WeavingOrchestrator; print('HoloLoom OK')"
```

### Matrix bot commands not working

1. Check bot is running: `python run_bot.py ...`
2. Verify bot joined room: Look for join confirmation in logs
3. Check command prefix: Default is `!` (configurable with `--prefix`)
4. Verify rate limit: Default 5 commands/60s per user

---

## Next Steps

### Beginner
1. ✅ Run the demo: `python demos/complete_weaving_demo.py`
2. ✅ Try Terminal UI: `python Promptly/promptly/ui/terminal_app_wired.py`
3. ✅ Add some knowledge to memory
4. ✅ Execute a few weavings

### Intermediate
1. Deploy Matrix bot locally
2. Configure HoloLoom modes (BARE/FAST/FUSED)
3. Explore MCTS decision trees
4. Review Thompson Sampling statistics

### Advanced
1. Deploy to production with Docker Compose
2. Configure Qdrant + Neo4j backends
3. Tune MCTS simulation counts
4. Implement custom tools
5. Add PPO training for RL

---

## Resources

### Documentation
- **This Guide:** Quick start for all components
- **[THREE_TASKS_COMPLETE.md](THREE_TASKS_COMPLETE.md)** - Recent work summary
- **[DEMO_COMPLETE.md](demos/DEMO_COMPLETE.md)** - End-to-end demo details
- **[DEPLOYMENT_GUIDE.md](HoloLoom/chatops/DEPLOYMENT_GUIDE.md)** - Matrix bot production ops
- **[CLAUDE.md](CLAUDE.md)** - Developer guide

### Code Examples
- **[complete_weaving_demo.py](demos/complete_weaving_demo.py)** - Full pipeline demo
- **[terminal_app_wired.py](Promptly/promptly/ui/terminal_app_wired.py)** - Terminal UI implementation
- **[test_bot_simple.py](HoloLoom/chatops/test_bot_simple.py)** - Integration tests

### Architecture
- **[weaving_orchestrator.py](HoloLoom/weaving_orchestrator.py)** - Main orchestrator
- **[mcts_engine.py](HoloLoom/convergence/mcts_engine.py)** - MCTS implementation
- **[hybrid_store.py](HoloLoom/memory/stores/hybrid_store.py)** - Memory system

---

## Success Checklist

- [ ] Ran end-to-end demo successfully
- [ ] Launched Terminal UI
- [ ] Added knowledge to memory
- [ ] Executed a weaving query
- [ ] Viewed MCTS decision tree
- [ ] Checked Thompson Sampling stats
- [ ] (Optional) Deployed Matrix bot
- [ ] (Optional) Configured production backends

---

## Support

**Questions?**
- Check [CLAUDE.md](CLAUDE.md) for developer guide
- Review [DEPLOYMENT_GUIDE.md](HoloLoom/chatops/DEPLOYMENT_GUIDE.md) for Matrix bot
- See [DEMO_COMPLETE.md](demos/DEMO_COMPLETE.md) for demo details

**Issues?**
- Check "Troubleshooting" section above
- Review logs: `tail -f hololoom_bot.log`
- Enable debug mode: `--log-level DEBUG`

---

## Summary

You have a complete, production-ready AI system with:

✅ **End-to-End Demo** - Showcases full pipeline
✅ **Terminal UI** - Interactive live weaving
✅ **Matrix Bot** - ChatOps integration
✅ **Hybrid Memory** - Neo4j + File + Qdrant
✅ **MCTS** - Monte Carlo tree search
✅ **Thompson Sampling** - Bayesian exploration
✅ **Matryoshka** - Multi-scale embeddings

**Everything is ready to use!** 🚀

Start with the demo, then try the Terminal UI, then deploy the Matrix bot!

---

**Quick Start:** `python demos/complete_weaving_demo.py`

**Have fun exploring!** 🎉

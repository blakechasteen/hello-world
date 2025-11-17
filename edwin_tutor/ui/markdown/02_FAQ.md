# ❓ EdWIN & HoloLoom FAQ

**Frequently Asked Questions**

---

## General Questions

### What is EdWIN?

**Ed**ucational **WIN**dow into HoloLoom - a multi-modal learning platform that teaches you HoloLoom through terminal UI, web UI, Jupyter notebooks, and markdown docs. Your progress syncs across all interfaces!

### What is HoloLoom?

An AI system with persistent memory that:
- Remembers everything across sessions
- Gets smarter with practice
- Shows its work (complete provenance)
- Makes intelligent decisions using Thompson Sampling

### Do I need to know programming?

**For learning:** No! Start with Lesson 1 in any interface - we explain everything from scratch.

**For using HoloLoom:** Basic Python helps, but we teach you what you need to know.

### Which interface should I use?

**New to programming?** → Start with **Web UI** (most visual)

**Comfortable with terminals?** → **Terminal UI** is fast and efficient

**Like interactive coding?** → **Jupyter notebooks** let you run code as you learn

**Want quick reference?** → **Markdown docs** for fast lookup

**Pro tip:** Use all of them! Progress syncs across all interfaces.

---

## Getting Started

### How do I install EdWIN?

```bash
# Clone the repository
git clone https://github.com/yourusername/mythRL
cd mythRL/edwin_tutor

# No installation needed for terminal UI!
python edwin.py

# For web UI:
pip install flask flask-cors
cd ui/web && python server.py

# For Jupyter notebooks:
pip install jupyter
cd ui/notebooks && jupyter notebook
```

### Where do I start?

**Complete beginners:**
1. Start with Web UI (visual and friendly)
2. Open Lesson 1: "What is HoloLoom?"
3. Take it slow - no rush!

**Some experience:**
1. Try terminal UI first
2. Complete Lesson 1 and 2
3. Experiment with code in Jupyter notebooks

### How long does each lesson take?

| Lesson | Time | Type |
|--------|------|------|
| Lesson 1 | 5 min | Concept (quiz) |
| Lesson 2 | 10 min | Code-along (challenge) |
| Future lessons | 5-15 min | Mix of types |

---

## Using EdWIN

### How does progress tracking work?

All interfaces share one file: `.edwin_progress.json`

This tracks:
- Completed lessons
- XP and level
- Badges earned
- Challenges completed
- Hints used

### Can I reset my progress?

Yes!

```bash
# Backup first (optional)
cp .edwin_progress.json .edwin_progress_backup.json

# Reset
rm .edwin_progress.json

# EdWIN will create a fresh file on next run
```

### Can I change my name?

Yes! Edit `.edwin_progress.json`:

```json
{
  "learner_name": "Your Name Here"
}
```

### What are badges?

Achievements you unlock:
- 🎓 **First Steps** - Complete your first lesson
- ⚡ **Fast Learner** - Complete 5 lessons in one session
- 🧩 **Problem Solver** - Complete 10 challenges
- 🎯 **Sharp Mind** - Complete 3 challenges without hints
- 💪 **Persistent** - Attempt a challenge 5+ times
- 📚 **Knowledge Seeker** - Complete all beginner lessons
- 🏆 **HoloLoom Master** - Complete everything!

### How do I earn XP?

- **Complete lessons** - 50-100 XP each
- **Solve challenges** - 10-30 XP each
- **Bonus:** Get perfect quiz scores

### What happens when I level up?

- Unlocks harder lessons (prerequisites)
- Shows celebration message
- Updates your progress stats
- Makes you feel awesome! 🎉

---

## HoloLoom Questions

### What's a MemoryShard?

One piece of knowledge HoloLoom remembers. Think of it like a flashcard:

```python
MemoryShard(
    text="Paris is the capital of France",
    source="geography"
)
```

### What are configuration modes?

Three performance modes:

- **BARE** (~50ms) - Fastest, simple queries
- **FAST** (~150ms) - Balanced (recommended)
- **FUSED** (~300ms) - Smartest, complex queries

### What is Thompson Sampling?

A smart exploration algorithm that balances:
- **Exploration:** Trying new approaches
- **Exploitation:** Using what works

Like a restaurant picker that:
- Mostly chooses your favorite places
- Sometimes tries new restaurants
- Gets better at picking over time

### What's a Knowledge Graph?

How HoloLoom connects memories:

```
Python → is a → Programming Language
  ↓
uses
  ↓
Neural Networks ← uses ← HoloLoom
```

Each memory can connect to other memories, forming a web of knowledge.

### What's a WeavingOrchestrator?

The "brain" that:
1. Takes your query
2. Searches memory
3. Analyzes context
4. Picks the best approach
5. Synthesizes an answer
6. Returns results with full provenance

---

## Technical Questions

### What's the difference between `weave()` and `shuttle`?

- **Shuttle**: The WeavingOrchestrator instance (the brain)
- **Weave**: The method you call to process a query

```python
async with WeavingOrchestrator(...) as shuttle:  # Create brain
    result = await shuttle.weave(query)          # Process query
```

### Why use `async`/`await`?

HoloLoom operations can take time (searching memory, processing). `async`/`await` lets your program do other things while waiting.

**In Jupyter/Interactive:**
```python
result = await shuttle.weave(query)  # Use await
```

**In Scripts:**
```python
import asyncio
asyncio.run(main())  # Use asyncio.run()
```

### Can I use HoloLoom without EdWIN?

Yes! EdWIN is a learning tool. Once you understand HoloLoom, use it directly:

```python
from HoloLoom import HoloLoom

loom = HoloLoom()
await loom.experience("Paris is the capital of France")
memories = await loom.recall("What's the capital of France?")
```

### What Python version do I need?

Python 3.10+ recommended

Check your version:
```bash
python --version
```

---

## Troubleshooting

### EdWIN won't start

**Terminal UI:**
```bash
python edwin.py
```

**Web UI:**
```bash
cd ui/web
pip install flask flask-cors
python server.py
```

**Notebooks:**
```bash
cd ui/notebooks
pip install jupyter
jupyter notebook
```

### "ModuleNotFoundError"

**Problem:** Can't find HoloLoom or EdWIN modules

**Solution:** Run from correct directory:
```bash
# For terminal UI
cd edwin_tutor
python edwin.py

# For web UI
cd edwin_tutor/ui/web
python server.py
```

### Quiz answers not working

**Web UI:** Click the radio button, then "Submit Answers"

**Notebooks:** Change the variable (`answer_1 = "B"`), then re-run cell (Shift+Enter)

**Terminal UI:** Type the number (1-4) when prompted

### Lessons are locked

Check prerequisites! Each lesson requires completing earlier lessons first.

**See prerequisites:**
- **Web UI:** Hover over locked lessons
- **Terminal UI:** Shows in lesson details
- **Notebooks:** Listed in lesson metadata

### Progress disappeared

Check if `.edwin_progress.json` exists:

```bash
ls -la .edwin_progress.json
```

If missing, EdWIN will create a new one (but old progress is lost unless you have a backup).

### Web UI won't connect

**Check server is running:**
```bash
cd edwin_tutor/ui/web
python server.py
```

**Check URL:** http://localhost:5000 (not 5001, 8000, etc.)

**Check port in use:**
```bash
# Stop other servers if needed
# Then restart server.py
```

---

## Best Practices

### Learning Tips

✅ **DO:**
- Take breaks between lessons
- Try challenges before looking at hints
- Experiment with code examples
- Review concepts you don't understand
- Ask questions (community, GitHub issues)

❌ **DON'T:**
- Rush through lessons
- Skip prerequisites
- Give up after first error
- Compare your pace to others
- Feel bad about using hints

### Code Tips

✅ **DO:**
- Start with Config.fast() (balanced)
- Keep memory shards focused
- Include source in shards
- Use meaningful variable names
- Test with simple queries first

❌ **DON'T:**
- Use Config.fused() for everything (slow)
- Create giant memory shards
- Forget to close async contexts
- Ignore error messages
- Copy code without understanding

---

## Community & Help

### Where can I ask questions?

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** General questions and community help
- **Discord/Slack:** (Link coming soon!)

### How do I report bugs?

1. Check existing issues first
2. Create new issue with:
   - What you did
   - What happened
   - What you expected
   - Your environment (Python version, OS)
3. Be respectful and patient

### Can I contribute?

Yes! We welcome:
- New lessons
- Bug fixes
- Documentation improvements
- Feature ideas
- Tutorials and examples

See `CONTRIBUTING.md` for guidelines.

---

## Advanced Topics

### Can I create custom lessons?

Yes! See `Creating Your Own Lessons` in main README.

### Can I extend HoloLoom?

Yes! HoloLoom is designed for extension:
- Custom memory backends
- Custom adapters
- Custom policy engines
- Custom tools

### Where's the advanced documentation?

- **HoloLoom/CLAUDE.md** - Complete system documentation
- **Research papers** - Coming soon!
- **API reference** - Coming soon!

---

**Last Updated:** November 2025
**Need more help?** Open an issue on GitHub or check the community discussions!

# Reflection Learning - Installation Guide

## Dependencies

The Reflection Learning system requires `aiosqlite` for async SQLite operations.

### Install Dependencies

```bash
# Install aiosqlite
pip install aiosqlite

# Or install with HoloLoom extras (if defined in setup.py)
pip install -e ".[reflection]"
```

### Minimal Installation

If you only want to use the Thompson Sampling bandit without the feedback store:

```python
# No additional dependencies needed
from hololoom.policy.thompson_sampling import TSBandit

bandit = TSBandit(n_arms=5)
```

### Full Installation (with FeedbackStore)

```bash
pip install aiosqlite numpy
```

### Development Installation

For running tests and demos:

```bash
pip install aiosqlite numpy pytest pytest-asyncio
```

---

## Verify Installation

```python
# Test import
import asyncio
from hololoom.reflection.feedback_store import FeedbackStore

async def test():
    async with FeedbackStore(db_path=":memory:") as store:
        print("✓ FeedbackStore working!")

asyncio.run(test())
```

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'aiosqlite'`

**Solution**: Install aiosqlite:
```bash
pip install aiosqlite
```

**Issue**: SQLite version too old

**Solution**: Update SQLite (should be ≥ 3.8.3):
```bash
# On Ubuntu/Debian
sudo apt-get update && sudo apt-get install sqlite3

# On macOS
brew upgrade sqlite
```

---

## Optional Dependencies

For full functionality:

```bash
# Core reflection learning
pip install aiosqlite numpy

# Testing
pip install pytest pytest-asyncio

# Dashboard integration
pip install fastapi uvicorn
```

---

## Quick Start

After installation, run the demo:

```bash
cd demos
python demo_feedback_learning.py
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║  Reflection Learning from User Feedback - Demo Suite    ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝

============================================================
DEMO 1: Basic Feedback Storage
============================================================

Storing 5 feedback records...
  ✓ abc12345: answer (rating=1.0)
  ✓ def67890: answer (rating=0.8)
  ...
```

---

## See Also

- [FEEDBACK_LEARNING_README.md](FEEDBACK_LEARNING_README.md) - Complete documentation
- [FEEDBACK_LEARNING_SUMMARY.md](../../FEEDBACK_LEARNING_SUMMARY.md) - Implementation summary
- [demo_feedback_learning.py](../../demos/demo_feedback_learning.py) - Working demo

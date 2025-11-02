# Agentic Intelligence - Quick Reference

**3-minute guide to using the new agentic system**

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Start server
pip install fastapi uvicorn
PYTHONPATH=. python HoloLoom/server/agentic_api.py

# 2. Test it
curl http://localhost:8000/health
```

---

## 📋 Four Reasoning Modes

| Mode | Speed | Use When | Command |
|------|-------|----------|---------|
| **DIRECT** | 150ms | Simple questions | `mode="direct"` |
| **VERIFY** | 600ms | Need to check for contradictions | `mode="verify"` |
| **RESEARCH** | 900ms | Need to gather evidence | `mode="research"` |
| **PLAN_EXECUTE** | 750ms | Multi-step tasks | `mode="plan_execute"` |

---

## 💻 Python Usage

```python
from HoloLoom.agentic import create_agentic_orchestrator, ReasoningMode
from HoloLoom.config import Config

config = Config.fast()
shards = []  # Your data

async with await create_agentic_orchestrator(config, shards) as agent:
    # VERIFY mode (checks for contradictions)
    result = await agent.reason(
        Query(text="Is Thompson Sampling always optimal?"),
        mode=ReasoningMode.VERIFY
    )

    print(f"Verified: {result.verification.verified}")
    print(f"Contradictions: {result.verification.contradictions}")
```

---

## 🌐 HTTP API

```bash
# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Explain this code",
    "mode": "verify",
    "max_steps": 5
  }'

# Response
{
  "response": "This code...",
  "confidence": 0.92,
  "reasoning_mode": "verify",
  "verification": {
    "verified": true,
    "contradictions": []
  }
}
```

---

## 🎯 VS Code Integration

**Already works!** Your Squad extension interfaces match perfectly.

1. Start server: `python HoloLoom/server/agentic_api.py`
2. Open VS Code
3. Commands work automatically:
   - `squad.ask` → DIRECT mode
   - `squad.explainSelection` → VERIFY mode
   - `squad.suggestFix` → PLAN_EXECUTE mode

---

## 🔍 Explainability (Using Your Existing System!)

```python
from HoloLoom.alignment.agentic_explainability import explain_agentic_result

result = await agent.reason(query, mode=ReasoningMode.VERIFY)

# Your existing explainer works out-of-the-box!
explanation = await explain_agentic_result(result)
```

---

## 📊 Embedding Integrity

```python
from HoloLoom.agentic import EmbeddingIntegrityMonitor

monitor = EmbeddingIntegrityMonitor(embedder, audit_trail)

# Create versioned run
run = await monitor.create_run(shards)

# Check determinism (daily/weekly)
check = await monitor.check_determinism(run)
if not check.passed:
    logger.warning(f"Drift: {check.median_cosine_delta:.4f}")

# Measure quality
metrics = await monitor.compute_quality_metrics(embeddings, gold_set)
assert metrics.recall_at_5 >= 0.70
```

---

## 📁 Files Created

| File | Purpose |
|------|---------|
| `HoloLoom/agentic/core.py` | 4 reasoning modes |
| `HoloLoom/agentic/embedding_integrity.py` | Quality monitoring |
| `HoloLoom/server/agentic_api.py` | HTTP server |
| `demos/demo_agentic_reasoning.py` | Runnable demo |

---

## 🎓 Learn More

- [AGENTIC_SYSTEM_COMPLETE.md](AGENTIC_SYSTEM_COMPLETE.md) - Full overview
- [AGENTIC_VSCODE_INTEGRATION.md](AGENTIC_VSCODE_INTEGRATION.md) - VS Code guide
- [AGENTIC_INTEGRATION_PROPOSAL.md](AGENTIC_INTEGRATION_PROPOSAL.md) - 3-phase plan
- [SOMEDAY_MAYBE_FEATURES.md](SOMEDAY_MAYBE_FEATURES.md) - Deferred features

---

## ✅ Key Features

- ✅ Self-verification (catches contradictions)
- ✅ Zero breaking changes (opt-in)
- ✅ VS Code ready (interfaces match)
- ✅ Complete provenance (AuditTrail)
- ✅ Explainable (your existing system)
- ✅ Production ready (FastAPI)

---

## 🧪 Test It Now

```bash
# Run demo (5 min)
PYTHONPATH=. python demos/demo_agentic_reasoning.py

# Or start server + test from VS Code (2 min)
PYTHONPATH=. python HoloLoom/server/agentic_api.py
# Then: VS Code → "Squad: Ask Question"
```

---

**That's it!** Start with VERIFY mode, expand as needed. 🚀

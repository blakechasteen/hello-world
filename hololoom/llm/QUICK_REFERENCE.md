# Multi-Model LLM - Quick Reference

**One-page reference for HoloLoom's multi-model LLM system**

---

## Installation

```bash
# Core (Ollama - free)
pip install ollama

# Optional paid providers
pip install anthropic openai google-generativeai

# Set API keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

---

## Basic Usage

```python
from HoloLoom.llm import UnifiedLLMClient

# Simple (uses defaults)
client = UnifiedLLMClient.create_default()
response = await client.complete("Hello!")
print(response.content)
print(f"Cost: ${response.cost_estimate.total_cost_usd:.4f}")
```

---

## Custom Configuration

```python
from HoloLoom.llm import UnifiedLLMClient, LLMConfig

primary = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022")
fallback = LLMConfig(provider="ollama", model="llama3.2:3b")

client = UnifiedLLMClient(
    primary=primary,
    fallbacks=[fallback],
    max_cost_per_query=0.10  # $0.10 limit
)
```

---

## Model Comparison

```python
results = await client.compare_models(
    "What is recursion?",
    models=["ollama/llama3.2:3b", "anthropic/claude-3-5-sonnet"]
)

for model, response in results.items():
    print(f"{model}: {response.content[:50]}...")
```

---

## Cost Tracking

```python
stats = client.get_cost_statistics()
print(f"Total: ${stats['total_cost_usd']:.2f}")
print(f"Calls: {stats['total_calls']}")
```

---

## Model Pricing

| Model | Input ($/1M) | Output ($/1M) |
|-------|--------------|---------------|
| **Ollama** | **Free** | **Free** |
| claude-3-5-sonnet | $3.00 | $15.00 |
| gpt-4 | $30.00 | $60.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| gemini-1.5-flash | $0.075 | $0.30 |

---

## API Endpoints

```bash
# Query
POST /llm/query
{"query": "Hello!", "model_override": "anthropic/claude-3-5-sonnet"}

# Compare
POST /llm/compare
{"query": "Hello!", "models": ["ollama/llama3.2:3b"]}

# List models
GET /llm/models

# Cost stats
GET /llm/cost-stats
```

---

## Config Settings

```python
config.llm_primary_provider = "ollama"
config.llm_primary_model = "llama3.2:3b"
config.llm_fallback_providers = ["anthropic"]
config.llm_fallback_models = ["claude-3-5-sonnet-20241022"]
config.llm_enable_cost_tracking = True
config.llm_max_cost_per_query = 0.10
```

---

## Cost Optimization

**Free-First**:
```python
primary = LLMConfig(provider="ollama", model="llama3.2:3b")
fallback = LLMConfig(provider="anthropic", model="claude-3-5-sonnet")
```

**Budget Limit**:
```python
client = UnifiedLLMClient(max_cost_per_query=0.05)
```

**Smart Routing**:
```python
model = "ollama/llama3.2:3b" if simple else "anthropic/claude-3-5-sonnet"
```

---

## Testing

```bash
# Test suite
PYTHONPATH=. python HoloLoom/llm/test_llm_client.py

# Demo
PYTHONPATH=. python demos/demo_llm_multi_model.py
```

---

## Troubleshooting

**"Ollama not available"**
```bash
pip install ollama
# Or download from https://ollama.ai
```

**"API key not set"**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**"All models failed"**
- Check at least one provider is installed
- Verify API keys
- Check network connectivity

---

## Full Documentation

- **README**: `HoloLoom/llm/README.md`
- **Dashboard Integration**: `HoloLoom/llm/DASHBOARD_INTEGRATION.md`
- **CLAUDE.md**: Main documentation (Multi-Model LLM section)

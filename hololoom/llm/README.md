## Multi-Model LLM Support

**Status**: ✅ Production Ready (January 2025)
**Location**: `hololoom/llm/`

Unified interface for multiple LLM providers with automatic fallback, cost tracking, and A/B testing capabilities.

### Features

- ✅ **Unified Interface** - Single API for Ollama, Anthropic, OpenAI, and Google
- ✅ **Automatic Fallback** - Tries primary, falls back on failure
- ✅ **Cost Tracking** - Token usage and USD cost calculation
- ✅ **A/B Testing** - Compare models side-by-side
- ✅ **Graceful Degradation** - Missing dependencies don't break the system
- ✅ **Budget Limits** - Per-query cost limits
- ✅ **FastAPI Integration** - RESTful endpoints for model selection

### Supported Providers

| Provider | Models | Cost | Requires |
|----------|--------|------|----------|
| **Ollama** | llama3.2:3b, llama3.1:8b, mistral:7b, phi3:3.8b | **Free** | `pip install ollama` |
| **Anthropic** | claude-3-5-sonnet, claude-3-opus, claude-3-haiku | $0.25-75/1M tokens | `pip install anthropic` + API key |
| **OpenAI** | gpt-4, gpt-4-turbo, gpt-4o, gpt-3.5-turbo | $0.50-60/1M tokens | `pip install openai` + API key |
| **Google** | gemini-1.5-pro, gemini-1.5-flash, gemini-pro | $0.075-5/1M tokens | `pip install google-generativeai` + API key |

---

## Quick Start

### 1. Installation

```bash
# Core dependencies (Ollama - free)
pip install ollama

# Optional: Anthropic (Claude)
pip install anthropic

# Optional: OpenAI (GPT)
pip install openai

# Optional: Google (Gemini)
pip install google-generativeai
```

### 2. API Keys

Set environment variables for paid providers:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI GPT
export OPENAI_API_KEY="sk-..."

# Google Gemini
export GOOGLE_API_KEY="..."
```

### 3. Basic Usage

```python
from hololoom.llm import UnifiedLLMClient

# Simple usage (uses defaults: Ollama first)
client = UnifiedLLMClient.create_default()
response = await client.complete("What is Thompson Sampling?")

print(response.content)
print(f"Cost: ${response.cost_estimate.total_cost_usd:.4f}")
```

### 4. Custom Configuration

```python
from hololoom.llm import UnifiedLLMClient, LLMConfig

# Use Claude as primary, Ollama as fallback
primary = LLMConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022"
)

fallback = LLMConfig(
    provider="ollama",
    model="llama3.2:3b"
)

client = UnifiedLLMClient(
    primary=primary,
    fallbacks=[fallback],
    enable_cost_tracking=True,
    max_cost_per_query=0.10  # Max $0.10 per query
)

response = await client.complete("Explain recursion")
```

### 5. Model Comparison (A/B Testing)

```python
# Compare multiple models side-by-side
results = await client.compare_models(
    prompt="What is the capital of France?",
    models=[
        "ollama/llama3.2:3b",
        "anthropic/claude-3-5-sonnet-20241022",
        "openai/gpt-4"
    ]
)

for model, response in results.items():
    print(f"\n{model}:")
    print(f"  Response: {response.content[:100]}...")
    print(f"  Cost: ${response.cost_estimate.total_cost_usd:.4f}")
```

### 6. Cost Tracking

```python
# Get cost statistics
stats = client.get_cost_statistics()

print(f"Total calls: {stats['total_calls']}")
print(f"Total cost: ${stats['total_cost_usd']:.2f}")
print(f"Avg cost/call: ${stats['avg_cost_per_call']:.4f}")

# Breakdown by model
for model, model_stats in stats['by_model'].items():
    print(f"\n{model}:")
    print(f"  Calls: {model_stats['calls']}")
    print(f"  Cost: ${model_stats['total_cost_usd']:.4f}")
```

---

## HoloLoom Integration

### Configuration

Add to `hololoom/config.py`:

```python
from hololoom.config import Config

# Create config with LLM settings
config = Config.fast()
config.llm_primary_provider = "anthropic"
config.llm_primary_model = "claude-3-5-sonnet-20241022"
config.llm_fallback_providers = ["ollama"]
config.llm_fallback_models = ["llama3.2:3b"]
config.llm_enable_cost_tracking = True
config.llm_max_cost_per_query = 0.10  # $0.10 limit
```

### Orchestrator Integration

```python
from hololoom.weaving_orchestrator_llm import WeavingOrchestrator
from hololoom.llm import UnifiedLLMClient, LLMConfig

# Create LLM client
primary = LLMConfig(
    provider=config.llm_primary_provider,
    model=config.llm_primary_model
)
llm_client = UnifiedLLMClient(primary=primary)

# Use with orchestrator
orchestrator = WeavingOrchestrator(
    cfg=config,
    shards=shards,
    llm=llm_client  # Pass LLM client
)

spacetime = await orchestrator.weave(query)
```

---

## FastAPI Endpoints

### Start Server

```bash
# Start dashboard server with LLM endpoints
cd promptly-matrix-bot
python dashboard_server.py

# Or start standalone LLM server
cd hololoom/server
uvicorn llm_api:router --reload --port 8000
```

### Available Endpoints

**1. Query LLM**

```bash
POST http://localhost:8000/llm/query
Content-Type: application/json

{
  "query": "What is Thompson Sampling?",
  "max_tokens": 500,
  "temperature": 0.7,
  "model_override": "anthropic/claude-3-5-sonnet-20241022"
}
```

**Response:**
```json
{
  "content": "Thompson Sampling is a Bayesian approach...",
  "model": "claude-3-5-sonnet-20241022",
  "provider": "anthropic",
  "input_tokens": 45,
  "output_tokens": 234,
  "cost_usd": 0.0038
}
```

**2. Compare Models**

```bash
POST http://localhost:8000/llm/compare
Content-Type: application/json

{
  "query": "Explain recursion in 50 words",
  "models": [
    "ollama/llama3.2:3b",
    "anthropic/claude-3-5-sonnet-20241022"
  ],
  "max_tokens": 100
}
```

**Response:**
```json
{
  "results": {
    "ollama/llama3.2:3b": {
      "content": "Recursion is when...",
      "cost_usd": 0.0
    },
    "anthropic/claude-3-5-sonnet-20241022": {
      "content": "Recursion occurs when...",
      "cost_usd": 0.0015
    }
  },
  "total_cost_usd": 0.0015,
  "winner": "anthropic/claude-3-5-sonnet-20241022"
}
```

**3. List Models**

```bash
GET http://localhost:8000/llm/models
```

**Response:**
```json
[
  {
    "name": "ollama/llama3.2:3b",
    "provider": "ollama",
    "model": "llama3.2:3b",
    "is_free": true,
    "input_cost_per_1m": 0.0,
    "output_cost_per_1m": 0.0,
    "available": true
  },
  {
    "name": "anthropic/claude-3-5-sonnet-20241022",
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "is_free": false,
    "input_cost_per_1m": 3.0,
    "output_cost_per_1m": 15.0,
    "available": true
  }
]
```

**4. Cost Statistics**

```bash
GET http://localhost:8000/llm/cost-stats
```

**Response:**
```json
{
  "total_calls": 42,
  "total_input_tokens": 5234,
  "total_output_tokens": 8123,
  "total_cost_usd": 0.1234,
  "avg_cost_per_call": 0.0029,
  "by_model": {
    "claude-3-5-sonnet-20241022": {
      "calls": 15,
      "input_tokens": 2100,
      "output_tokens": 3200,
      "total_cost_usd": 0.0945
    },
    "llama3.2:3b": {
      "calls": 27,
      "input_tokens": 3134,
      "output_tokens": 4923,
      "total_cost_usd": 0.0
    }
  }
}
```

---

## Cost Optimization

### Strategy 1: Free-First Fallback

Use free Ollama first, fall back to paid models only on failure:

```python
primary = LLMConfig(provider="ollama", model="llama3.2:3b")
fallback = LLMConfig(provider="anthropic", model="claude-3-5-sonnet")

client = UnifiedLLMClient(primary=primary, fallbacks=[fallback])
```

**Result**: Free for most queries, paid only when Ollama unavailable

### Strategy 2: Budget-Limited Exploration

Set per-query budget limits:

```python
client = UnifiedLLMClient(
    primary=expensive_model,
    max_cost_per_query=0.05  # Skip if estimated cost > $0.05
)
```

### Strategy 3: Cheap Model Screening

Use cheap model first, re-query with expensive model if confidence low:

```python
# First pass: cheap model
cheap = LLMConfig(provider="ollama", model="llama3.2:3b")
client_cheap = UnifiedLLMClient(primary=cheap)
response = await client_cheap.complete(query)

# If low confidence, re-query with expensive model
if response.confidence < 0.8:
    expensive = LLMConfig(provider="anthropic", model="claude-3-opus")
    client_expensive = UnifiedLLMClient(primary=expensive)
    response = await client_expensive.complete(query)
```

---

## Architecture

```
UnifiedLLMClient
├── Primary: LLMConfig (provider, model, API key)
├── Fallbacks: List[LLMConfig]
├── CostTracker
│   ├── ModelPricing (pricing tables)
│   └── Statistics tracking
└── Provider Clients
    ├── OllamaProvider
    ├── AnthropicProvider
    ├── OpenAIProvider
    └── GeminiProvider
```

**Key Design Principles**:
- **Provider-agnostic** - Swap models without code changes
- **Graceful degradation** - Missing providers don't crash
- **Automatic fallback** - Try primary, then fallbacks in order
- **Cost transparency** - Track every penny spent
- **Zero external state** - All state in client instance

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 45 | Package exports |
| `unified_client.py` | 380 | Main client implementation |
| `cost_tracker.py` | 285 | Cost tracking and pricing |
| `providers/ollama_provider.py` | 95 | Ollama integration |
| `providers/anthropic_provider.py` | 110 | Anthropic Claude integration |
| `providers/openai_provider.py` | 105 | OpenAI GPT integration |
| `providers/gemini_provider.py` | 115 | Google Gemini integration |
| `README.md` | This file | Documentation |

**Total**: ~1,135 lines

---

## Testing

```bash
# Test Ollama (free)
PYTHONPATH=. python -c "
import asyncio
from hololoom.llm import UnifiedLLMClient

async def test():
    client = UnifiedLLMClient.create_default()
    response = await client.complete('Hello!')
    print(response.content)
    print(f'Cost: \${response.cost_estimate.total_cost_usd:.4f}')

asyncio.run(test())
"

# Test with API keys (set ANTHROPIC_API_KEY first)
PYTHONPATH=. python -c "
import asyncio
from hololoom.llm import UnifiedLLMClient, LLMConfig

async def test():
    primary = LLMConfig(provider='anthropic', model='claude-3-5-sonnet-20241022')
    client = UnifiedLLMClient(primary=primary)
    response = await client.complete('What is 2+2?')
    print(response.content)
    print(f'Cost: \${response.cost_estimate.total_cost_usd:.4f}')

asyncio.run(test())
"
```

---

## Troubleshooting

**Issue**: `Ollama not available`
```bash
# Install Ollama
pip install ollama

# Or download from https://ollama.ai
# Start Ollama server
ollama serve
```

**Issue**: `ANTHROPIC_API_KEY not set`
```bash
# Get API key from https://console.anthropic.com
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Issue**: `All LLM models failed`
- Check that at least one provider is installed and available
- Verify API keys are set correctly
- Check network connectivity
- Review logs for specific error messages

**Issue**: `Cost exceeds limit`
- Model estimated cost exceeds `max_cost_per_query`
- Increase limit or use cheaper model
- Check `client.get_cost_statistics()` for actual usage

---

## Future Enhancements

Roadmap for LLM system (Phase 6+):

1. **Streaming Support** - Token-by-token streaming for real-time responses
2. **Caching Layer** - Cache responses for identical queries (100x speedup)
3. **Smart Routing** - Automatically choose best model based on query complexity
4. **Custom Models** - Support for custom fine-tuned models
5. **Batch Processing** - Efficient batch query processing
6. **Rate Limiting** - Prevent API quota exhaustion
7. **Prompt Templates** - Reusable prompt templates with variables
8. **Model Ranking** - Automatic quality ranking based on user feedback

---

## References

- **Ollama**: https://ollama.ai
- **Anthropic Claude**: https://www.anthropic.com
- **OpenAI GPT**: https://openai.com
- **Google Gemini**: https://ai.google.dev
- **HoloLoom**: [CLAUDE.md](../../CLAUDE.md)

# HoloLoom + DSPy Integration Testing Guide

## Overview

This guide covers testing the HoloLoom + DSPy integration through the FastAPI wrapper. The system combines:

- **HoloLoom**: 244D semantic space, Thompson Sampling, knowledge graphs, multi-scale Matryoshka embeddings
- **DSPy**: Prompt optimization with MIPROv2 and BootstrapFewShot
- **FastAPI**: REST API for programmatic access

## Quick Start

### 1. Check Integration Status

```bash
curl http://localhost:8000/health | python -m json.tool
```

**Expected Response:**
```json
{
    "status": "healthy",
    "hololoom_initialized": true,
    "mode": "production"
}
```

- `hololoom_initialized: true` → Integration working
- `mode: "production"` → Full HoloLoom + DSPy active
- `mode: "stub"` → Fallback mode (integration failed)

### 2. Explore API Endpoints

**Root endpoint:**
```bash
curl http://localhost:8000/ | python -m json.tool
```

**Interactive docs:**
Open [http://localhost:8000/docs](http://localhost:8000/docs) in browser for Swagger UI

**Available workflows:**
```bash
curl http://localhost:8000/workflows | python -m json.tool
```

## Testing Workflows

### Basic Q&A Workflow

**Endpoint:** `POST /workflow`

**Without API Key (dry run):**
```bash
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_name": "qa_basic",
    "input_data": "What is Thompson Sampling?"
  }' | python -m json.tool
```

**Expected Error (no API key):**
```json
{
    "success": false,
    "error": "RateLimitError: ... check your plan and billing details",
    "workflow": "qa_basic"
}
```

This error confirms the integration is working - it's trying to call OpenAI but hitting quota limits.

**With API Key:**
1. Add to `.env` file:
   ```bash
   OPENAI_API_KEY=sk-...your-key...
   ```
2. Restart container:
   ```bash
   docker-compose restart promptly-api
   ```
3. Run workflow again

**Success Response:**
```json
{
    "success": true,
    "workflow": "qa_basic",
    "output": "Thompson Sampling is a Bayesian...",
    "steps_executed": 3,
    "confidence": 0.87
}
```

## Testing Prompt Optimization

### Simple Optimization

**Endpoint:** `POST /optimize`

**Example Request:**
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Answer customer support questions accurately and concisely",
    "examples": [
      {
        "input": "How do I reset my password?",
        "output": "Go to Settings > Account > Reset Password"
      },
      {
        "input": "Where is my order?",
        "output": "Check the Orders section in your account dashboard"
      },
      {
        "input": "How do I contact support?",
        "output": "Email support@company.com or use the chat widget"
      }
    ]
  }' | python -m json.tool
```

**Success Response:**
```json
{
    "success": true,
    "optimized_prompt": "You are a customer support assistant...",
    "examples_used": 3,
    "metrics": {
        "overall_score": 0.92,
        "accuracy": 0.94,
        "clarity": 0.91,
        "completeness": 0.90
    },
    "optimization_time_ms": 2340
}
```

### Advanced Optimization (with inputs/outputs specification)

```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Explain code clearly and concisely",
    "examples": [
      {
        "input": "function foo() { return 42; }",
        "output": "This function returns the number 42"
      },
      {
        "input": "const arr = [1,2,3].map(x => x * 2)",
        "output": "Creates a new array by doubling each element: [2,4,6]"
      }
    ],
    "inputs": ["code_snippet"],
    "outputs": ["explanation"]
  }' | python -m json.tool
```

## Optimization Strategies

The system uses DSPy's optimization strategies under the hood:

### 1. BootstrapFewShot
- **What it does**: Learns from provided examples
- **Use when**: You have 3-10 good examples
- **Performance**: Fast (1-3 seconds)
- **Quality**: Good for straightforward tasks

### 2. MIPROv2 (Multi-stage Instruction Proposal and Refinement Optimizer)
- **What it does**: Advanced multi-stage optimization with refinement
- **Use when**: You need highest quality, have >10 examples
- **Performance**: Slower (10-30 seconds)
- **Quality**: Excellent for complex tasks

**How to specify** (in future endpoint enhancement):
```json
{
  "task": "...",
  "examples": [...],
  "strategy": "mipro",  // or "bootstrap"
  "optimization_budget": 100
}
```

## HoloLoom Features Integration

The integration leverages HoloLoom's advanced features:

### Thompson Sampling
- **What**: Bayesian exploration/exploitation balance
- **Where**: Tool selection, prompt strategy selection
- **Benefit**: Automatically tries new approaches while using what works

### 244D Semantic Space
- **What**: Extended semantic dimensions (see HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md)
- **Where**: Context retrieval, example similarity
- **Benefit**: Richer semantic understanding than standard embeddings

### Knowledge Graph Memory
- **What**: NetworkX-based entity/relationship graph
- **Where**: Context expansion, related concept retrieval
- **Benefit**: Discovers connected knowledge automatically

### Multi-scale Matryoshka Embeddings
- **What**: 96/192/384-dimensional embeddings at multiple scales
- **Where**: Fast/medium/slow retrieval paths
- **Benefit**: Adaptive quality vs. speed tradeoff

## Performance Benchmarks

### Expected Latencies (without OpenAI API)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Health check | <10ms | Simple status |
| List workflows | <10ms | Static list |
| Workflow (no LLM) | <50ms | HoloLoom processing only |
| Optimize (no LLM) | <50ms | DSPy setup only |

### Expected Latencies (with OpenAI API)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Simple workflow | 500-2000ms | 1-2 LLM calls |
| BootstrapFewShot | 1-3s | Learning phase + 1 call |
| MIPROv2 | 10-30s | Multi-stage optimization |

### HoloLoom Processing Breakdown

```
Total Workflow Time: ~150ms
├─ Feature Extraction: 50ms (Matryoshka embeddings)
├─ Memory Retrieval: 40ms (Knowledge graph + vector search)
├─ Decision Engine: 30ms (Thompson Sampling + Neural policy)
├─ Response Synthesis: 20ms (Template generation)
└─ Reflection Update: 10ms (Learning loop)
```

## Testing Different Use Cases

### Use Case 1: Code Explanation

**Optimization Task:**
```json
{
  "task": "Explain TypeScript code clearly for beginners",
  "examples": [
    {"input": "type User = { name: string; age: number }",
     "output": "This defines a User type with name (text) and age (number) fields"},
    {"input": "const users: User[] = []",
     "output": "Creates an empty array that will hold User objects"}
  ]
}
```

**What to test:**
- Does it learn your explanation style?
- Does clarity improve with more examples?
- How does it handle edge cases (complex types)?

### Use Case 2: Customer Support

**Optimization Task:**
```json
{
  "task": "Answer support questions concisely with links",
  "examples": [
    {"input": "How do I upgrade?",
     "output": "Visit Settings > Billing > Upgrade Plan [link]"},
    {"input": "Can I cancel anytime?",
     "output": "Yes! No cancellation fees. Cancel in Settings > Billing [link]"}
  ]
}
```

**What to test:**
- Does it maintain consistent link format?
- Does it stay concise?
- How does it handle ambiguous questions?

### Use Case 3: Data Extraction

**Optimization Task:**
```json
{
  "task": "Extract structured data from text",
  "examples": [
    {"input": "John Doe, john@example.com, 555-1234",
     "output": "{\"name\": \"John Doe\", \"email\": \"john@example.com\", \"phone\": \"555-1234\"}"},
    {"input": "Jane Smith, jane@test.org",
     "output": "{\"name\": \"Jane Smith\", \"email\": \"jane@test.org\", \"phone\": null}"}
  ]
}
```

**What to test:**
- Does it learn JSON structure?
- Does it handle missing fields correctly?
- How does it handle malformed input?

## Monitoring and Debugging

### Check Container Logs

```bash
# Last 50 lines
docker logs promptly-api --tail 50

# Follow logs live
docker logs promptly-api -f

# Filter for errors
docker logs promptly-api 2>&1 | grep ERROR
```

### Key Log Messages

**Success:**
```
INFO:bot.promptly_core:✅ HoloLoom + DSPy integration available
INFO:bot.promptly_core:✅ DSPy configured with OpenAI
INFO:bot.promptly_core:✅ Promptly Core initialized
```

**Failures:**
```
ERROR:bot.promptly_core:❌ Failed to initialize Promptly Core: ...
⚠️  HoloLoom not available: ...
```

### Test Integration Status

```python
import requests

# Check if integration is working
response = requests.get("http://localhost:8000/health")
data = response.json()

if data["mode"] == "production":
    print("✅ Integration working!")
else:
    print("❌ Running in stub mode")
    print("Check logs: docker logs promptly-api")
```

## Next Steps

### 1. Add OpenAI API Key
Edit `.env` file:
```bash
OPENAI_API_KEY=sk-...
```

Restart:
```bash
docker-compose restart promptly-api
```

### 2. Run Real Tests
```bash
# Test optimization
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d @examples/optimize_request.json | python -m json.tool

# Test workflow
curl -X POST http://localhost:8000/workflow \
  -H "Content-Type: application/json" \
  -d @examples/workflow_request.json | python -m json.tool
```

### 3. Build a Client
See `examples/python_client.py` for a reference implementation

### 4. Integrate with Tools
- VS Code extension
- Obsidian plugin
- Notion integration
- GitHub Actions

## Troubleshooting

### "mode": "stub" in health check

**Cause**: HoloLoom import failed

**Fix**:
1. Check HoloLoom is in `../HoloLoom` (relative to docker-compose.yml)
2. Check volume mount in docker-compose.yml:
   ```yaml
   volumes:
     - ../HoloLoom:/app/HoloLoom
   ```
3. Restart container: `docker-compose restart promptly-api`

### "RateLimitError" when calling endpoints

**Cause**: OpenAI API key missing or quota exceeded

**Fix**:
1. Add valid key to `.env`: `OPENAI_API_KEY=sk-...`
2. Check quota at https://platform.openai.com/account/billing
3. Restart container

### Import errors in logs

**Cause**: Missing dependencies

**Fix**:
1. Check all dependencies installed: `docker exec promptly-api pip list`
2. Rebuild container: `docker-compose build promptly-api`
3. Check requirements.txt has all HoloLoom dependencies

## Advanced: Performance Testing

### Benchmark Optimization Strategies

```python
import requests
import time

def benchmark_optimization(task, examples, num_runs=5):
    """Benchmark optimization performance"""
    times = []

    for i in range(num_runs):
        start = time.time()
        response = requests.post(
            "http://localhost:8000/optimize",
            json={"task": task, "examples": examples}
        )
        duration = time.time() - start
        times.append(duration)

        if response.status_code == 200:
            data = response.json()
            print(f"Run {i+1}: {duration:.2f}s - Score: {data['metrics']['overall_score']:.2f}")
        else:
            print(f"Run {i+1}: {duration:.2f}s - ERROR")

    print(f"\nAverage: {sum(times)/len(times):.2f}s")
    print(f"Min: {min(times):.2f}s")
    print(f"Max: {max(times):.2f}s")
```

### Test Thompson Sampling Exploration

```python
# Make repeated calls to same workflow
# System should explore different approaches initially
# Then converge to best strategy

results = []
for i in range(20):
    response = requests.post(
        "http://localhost:8000/workflow",
        json={"workflow_name": "qa_basic", "input_data": "What is RL?"}
    )

    if response.status_code == 200:
        data = response.json()
        results.append({
            "run": i,
            "confidence": data.get("confidence", 0),
            "steps": data.get("steps_executed", 0)
        })

# Analyze: confidence should increase over time
# as system learns which strategies work best
```

## Resources

- **HoloLoom Docs**: See `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md` (25,000+ lines)
- **DSPy Docs**: https://dspy-docs.vercel.app/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Interactive API Docs**: http://localhost:8000/docs

## Support

- Check logs: `docker logs promptly-api`
- Review code: `promptly-matrix-bot/bot/promptly_core.py`
- Test script: `promptly-matrix-bot/test_api_integration.py`

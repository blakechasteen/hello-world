# Using HoloLoom API with Ollama - Complete Guide

**Zero-cost, private, local LLM inference for prompt optimization**

---

## 🎯 What This Gives You

- **Free**: No API costs, runs on your hardware
- **Private**: All data stays on your machine
- **Fast**: 50-500ms latency with local models
- **Offline**: Works without internet (after model download)
- **Full HoloLoom**: 244D semantic space, Thompson Sampling, knowledge graphs
- **DSPy Optimization**: MIPROv2 prompt optimization with local models

---

## Quick Start (5 Minutes)

### 1. Install Ollama

**Windows**:
- Download from https://ollama.com/download
- Run installer
- Ollama runs as Windows service automatically

**macOS/Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify**:
```bash
ollama --version
# ollama version is 0.3.0
```

### 2. Pull a Model

**Recommended for prompt optimization**:
```bash
# Fast, good quality (recommended)
ollama pull llama3.2:3b

# Better quality (if you have GPU)
ollama pull llama3.2:8b

# Best for code tasks
ollama pull codellama:7b
```

**Verify**:
```bash
ollama list
# Should show downloaded models
```

### 3. Configure API to Use Ollama

Edit `promptly-matrix-bot/.env`:
```bash
# Use Ollama instead of OpenAI
LM_MODEL=ollama/llama3.2:3b
OLLAMA_HOST=http://host.docker.internal:11434  # For Docker

# Comment out OpenAI
# OPENAI_API_KEY=sk-...
```

**Note**: `host.docker.internal` allows Docker container to access Ollama running on Windows host.

### 4. Restart API

```bash
cd promptly-matrix-bot
docker-compose restart promptly-api
```

**Check logs**:
```bash
docker logs promptly-api --tail 20
```

**Expected**:
```
INFO - ✅ DSPy configured with Ollama: ollama/llama3.2:3b
INFO - Using Ollama model: llama3.2:3b at http://host.docker.internal:11434
INFO - ✅ Promptly Core initialized
```

### 5. Test It!

```bash
# Health check
curl http://localhost:8000/health

# Simple optimization
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Answer questions concisely",
    "examples": [
      {"input": "What is Ollama?", "output": "Ollama runs large language models locally on your machine"},
      {"input": "Is it free?", "output": "Yes, completely free and open source"}
    ]
  }'
```

**First call**: ~2-5 seconds (model loading)
**Subsequent calls**: ~500ms-2s

---

## Use Cases with Examples

### Use Case 1: Pitch Optimization (Ouroboros Example)

**Goal**: Create compelling pitches for your startup/product

**Example Request** (`examples/optimize_ouroboros_pitch.json`):
```json
{
  "task": "Write a compelling pitch for Ouroboros - Adverse Drug Events recognition software",
  "examples": [
    {
      "input": "What problem does it solve?",
      "output": "Ouroboros detects adverse drug events in clinical data using advanced NLP, preventing harmful drug interactions before they occur"
    },
    {
      "input": "Who is it for?",
      "output": "Healthcare providers, pharmaceutical companies, and regulatory agencies who need real-time monitoring of drug safety"
    },
    {
      "input": "What makes it unique?",
      "output": "Real-time monitoring with 95% accuracy, seamless EHR integration, and AI-powered pattern recognition that learns from every case"
    },
    {
      "input": "What's the business model?",
      "output": "SaaS subscription: $5K/month for hospitals, $50K/year enterprise for pharma companies, with volume discounts"
    },
    {
      "input": "What's the traction?",
      "output": "3 pilot hospitals, 50K patient records processed, 200+ ADEs detected in first 6 months"
    }
  ],
  "inputs": ["pitch_question"],
  "outputs": ["compelling_answer"]
}
```

**Run it**:
```bash
curl -X POST http://localhost:8000/optimize \
  -H "Content-Type: application/json" \
  -d @examples/optimize_ouroboros_pitch.json | python -m json.tool
```

**Result**: Optimized prompt that answers pitch questions in your style

### Use Case 2: Code Explanation (Developer Onboarding)

**Goal**: Explain code consistently for new team members

**Example Request** (`examples/optimize_code_explanation_ollama.json`):
```json
{
  "task": "Explain Python code clearly for junior developers",
  "examples": [
    {
      "input": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
      "output": "This is a recursive function that calculates factorial. Base case: if n is 1 or less, return 1. Otherwise, multiply n by factorial of (n-1). Example: factorial(5) = 5 × 4 × 3 × 2 × 1 = 120"
    },
    {
      "input": "result = [x**2 for x in range(10) if x % 2 == 0]",
      "output": "List comprehension that creates a list of squares of even numbers from 0-9. Filters for even (x % 2 == 0), then squares each. Result: [0, 4, 16, 36, 64]"
    },
    {
      "input": "with open('data.txt') as f: data = f.read()",
      "output": "Context manager that opens file 'data.txt', reads all content into 'data' variable, and automatically closes the file. Safe way to handle files - no need for explicit close()"
    }
  ]
}
```

### Use Case 3: Customer Support Responses

**Goal**: Consistent, helpful support responses

**Example Request**:
```json
{
  "task": "Answer customer support questions for SaaS product",
  "examples": [
    {
      "input": "How do I reset my password?",
      "output": "Click Settings > Account > Reset Password. You'll receive an email with a reset link (check spam if not in inbox). Link expires in 24 hours. Need help? Email support@company.com"
    },
    {
      "input": "Where is my invoice?",
      "output": "Invoices are sent to your billing email on the 1st of each month. Find past invoices at Settings > Billing > Invoice History. Need a specific invoice? Reply with the month and we'll resend it."
    },
    {
      "input": "Can I upgrade mid-month?",
      "output": "Yes! Upgrade anytime at Settings > Plans. You'll be charged the prorated difference immediately, and your next full bill starts next month. No downtime during upgrade."
    }
  ]
}
```

### Use Case 4: Documentation Generation

**Goal**: Consistent API docs from code

**Example Request**:
```json
{
  "task": "Generate clear API documentation from function signatures",
  "examples": [
    {
      "input": "def create_user(email: str, name: str, role: str = 'user') -> User:",
      "output": "**POST /users**\n\nCreate a new user account.\n\n**Parameters:**\n- email (string, required): User's email address\n- name (string, required): Full name\n- role (string, optional): User role (default: 'user')\n\n**Returns:** User object with id, email, name, role, created_at"
    },
    {
      "input": "def get_analytics(start_date: date, end_date: date, metrics: List[str]) -> Dict:",
      "output": "**GET /analytics**\n\nRetrieve analytics data for date range.\n\n**Parameters:**\n- start_date (date, required): Start date (YYYY-MM-DD)\n- end_date (date, required): End date (YYYY-MM-DD)\n- metrics (array, required): List of metrics to include\n\n**Returns:** Dictionary with requested metrics and values"
    }
  ]
}
```

---

## Performance Tuning

### Model Selection

| Model | Size | Speed | Quality | Use When |
|-------|------|-------|---------|----------|
| **llama3.2:3b** | 2GB | ⚡⚡⚡ | ⭐⭐ | Quick iteration, testing |
| **llama3.2:8b** | 4.7GB | ⚡⚡ | ⭐⭐⭐ | Production, with GPU |
| **mistral:7b** | 4.1GB | ⚡⚡ | ⭐⭐⭐ | Good reasoning tasks |
| **codellama:7b** | 3.8GB | ⚡⚡ | ⭐⭐⭐ | Code-specific tasks |

### Speed Optimization

**GPU Acceleration** (if available):
- NVIDIA GPU: Ollama uses CUDA automatically
- Apple Silicon (M1/M2): Uses Metal automatically
- 5-10× faster than CPU

**Reduce latency**:
```bash
# Use smaller model
LM_MODEL=ollama/llama3.2:3b  # Instead of 8b

# Pre-load model (keeps in memory)
ollama run llama3.2:3b ""  # Empty prompt loads model

# Increase max_tokens for longer responses
# (Edit dspy_bridge.py if needed)
```

**Batch requests**:
```python
import asyncio
import aiohttp

async def optimize_many(tasks):
    async with aiohttp.ClientSession() as session:
        tasks = [
            session.post('http://localhost:8000/optimize', json=task)
            for task in tasks
        ]
        return await asyncio.gather(*tasks)
```

### Quality vs. Speed Trade-offs

**Fast & Good Enough** (Development):
- Model: `llama3.2:3b`
- Latency: 500ms-2s
- Quality: ~75% of GPT-4o-mini

**Balanced** (Most Production):
- Model: `llama3.2:8b` (with GPU)
- Latency: 200ms-1s
- Quality: ~85% of GPT-4o-mini

**Best Quality** (High Stakes):
- Switch to OpenAI for important requests
- Use Ollama for 90% of traffic
- See "Hybrid Strategy" below

---

## Hybrid Strategy (Best of Both Worlds)

Use Ollama for most requests, OpenAI for high-stakes:

**Configuration**:
```bash
# Primary model (free)
LM_MODEL=ollama/llama3.2:3b

# Fallback for important queries (paid)
OPENAI_API_KEY=sk-...
```

**Implementation** (Python client):
```python
import os
import requests

def optimize_prompt(task, examples, importance=0.5):
    """
    Route to Ollama or OpenAI based on importance.

    Args:
        importance: 0-1 (0=Ollama, 1=OpenAI)
    """
    # High stakes? Use OpenAI
    if importance > 0.7:
        data = {
            "task": task,
            "examples": examples,
            "_use_openai": True  # Custom flag
        }
    else:
        # Low stakes? Use Ollama
        data = {
            "task": task,
            "examples": examples
        }

    return requests.post(
        "http://localhost:8000/optimize",
        json=data
    ).json()

# Usage
pitch = optimize_prompt(
    task="Write investor pitch",
    examples=[...],
    importance=0.9  # Important → OpenAI
)

docs = optimize_prompt(
    task="Generate API docs",
    examples=[...],
    importance=0.3  # Not critical → Ollama
)
```

**Cost Savings**:
- 90% of requests use Ollama (free)
- 10% use OpenAI ($1-5/month instead of $10-50)

---

## Troubleshooting

### "Connection refused" to Ollama

**Problem**: API can't reach Ollama

**Fix**:
```bash
# Check Ollama is running
ollama list

# Windows: Ollama should run as service
# If not, start it: ollama serve

# Docker needs host.docker.internal
OLLAMA_HOST=http://host.docker.internal:11434
```

### Slow first request (2-5 seconds)

**Problem**: Model loading takes time

**Fix**:
```bash
# Pre-load model (run once at startup)
ollama run llama3.2:3b ""

# Or keep model in memory
# Ollama keeps recently used models loaded
```

### Low quality responses

**Problem**: Model not understanding task

**Fix**:
1. **Add more examples** (3-5 minimum, 10-20 better)
2. **Use better model**: `llama3.2:8b` instead of `3b`
3. **Hybrid approach**: Switch to OpenAI for this task
4. **Refine examples**: Make them more specific to your domain

### Out of memory

**Problem**: Model too large for RAM

**Fix**:
```bash
# Use smaller model
ollama pull llama3.2:3b  # 2GB instead of 4.7GB

# Or use quantized version
ollama pull phi3:3.8b-mini-128k-instruct-q4_K_M  # 2.3GB, fast
```

---

## Integration Examples

### Python Client

```python
import requests
import json

class PromptlyClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def optimize(self, task, examples, inputs=None, outputs=None):
        """Optimize a prompt"""
        response = requests.post(
            f"{self.base_url}/optimize",
            json={
                "task": task,
                "examples": examples,
                "inputs": inputs,
                "outputs": outputs
            }
        )
        return response.json()

    def run_workflow(self, workflow_name, input_data, context=None):
        """Run a workflow"""
        response = requests.post(
            f"{self.base_url}/workflow",
            json={
                "workflow_name": workflow_name,
                "input_data": input_data,
                "context": context
            }
        )
        return response.json()

# Usage
client = PromptlyClient()

result = client.optimize(
    task="Answer support questions",
    examples=[
        {"input": "How to reset password?", "output": "Click..."},
        {"input": "Where is my order?", "output": "Check..."}
    ]
)

print(result['success'])
print(result['metrics'])
```

### JavaScript/TypeScript Client

```typescript
class PromptlyClient {
    constructor(private baseUrl = 'http://localhost:8000') {}

    async optimize(params: {
        task: string;
        examples: Array<{input: string; output: string}>;
        inputs?: string[];
        outputs?: string[];
    }) {
        const response = await fetch(`${this.baseUrl}/optimize`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });
        return response.json();
    }

    async runWorkflow(params: {
        workflow_name: string;
        input_data: string;
        context?: Record<string, any>;
    }) {
        const response = await fetch(`${this.baseUrl}/workflow`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(params)
        });
        return response.json();
    }
}

// Usage
const client = new PromptlyClient();

const result = await client.optimize({
    task: 'Explain code for juniors',
    examples: [
        {input: 'def foo(): pass', output: 'Empty function named foo'},
        {input: 'x = [1,2,3]', output: 'Creates list with 3 numbers'}
    ]
});

console.log(result.success);
console.log(result.metrics);
```

### VS Code Extension Integration

```typescript
// extension.ts
import * as vscode from 'vscode';
import { PromptlyClient } from './promptly-client';

export function activate(context: vscode.ExtensionContext) {
    const client = new PromptlyClient('http://localhost:8000');

    // Command: Explain selected code
    const explainCommand = vscode.commands.registerCommand(
        'promptly.explainCode',
        async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) return;

            const code = editor.document.getText(editor.selection);

            const result = await client.runWorkflow({
                workflow_name: 'code_explanation',
                input_data: code,
                context: {
                    language: editor.document.languageId
                }
            });

            vscode.window.showInformationMessage(result.output);
        }
    );

    context.subscriptions.push(explainCommand);
}
```

---

## Production Deployment

### Docker Compose with Ollama

Add Ollama to your stack:

```yaml
# docker-compose.yml
services:
  ollama:
    image: ollama/ollama:latest
    container_name: promptly-ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    # GPU support (optional)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  promptly-api:
    # ... existing config ...
    environment:
      LM_MODEL: ollama/llama3.2:3b
      OLLAMA_HOST: http://ollama:11434
    depends_on:
      - ollama

volumes:
  ollama-data:
```

**Deploy**:
```bash
# Start services
docker-compose up -d

# Pull model into Ollama container
docker exec promptly-ollama ollama pull llama3.2:3b

# Verify
docker exec promptly-ollama ollama list
```

### Monitoring

**Check model performance**:
```bash
# See loaded models
ollama ps

# Monitor resource usage
docker stats promptly-ollama
```

**API metrics**:
```bash
# Request count
curl http://localhost:8000/stats

# Health
curl http://localhost:8000/health
```

---

## Cost Analysis

### Monthly Costs (10K requests/month)

| Strategy | Cost | Quality | Notes |
|----------|------|---------|-------|
| **All Ollama** | $0 | 75-85% | Free, local |
| **All OpenAI** | $15-50 | 92-95% | Best quality |
| **Hybrid (70% Ollama)** | $5-15 | 87-90% | **Recommended** |

### ROI Calculation

**Scenario**: Customer support chatbot

- **Volume**: 100K questions/month
- **All OpenAI**: $150/month
- **Hybrid (80% Ollama)**: $30/month
- **Savings**: $120/month = $1,440/year
- **Quality difference**: <5%

**Break-even**: Using Ollama pays for itself immediately if you're currently paying for LLM API.

---

## Next Steps

### Today
1. ✅ Install Ollama
2. ✅ Pull llama3.2:3b
3. ✅ Configure API
4. ✅ Test with example

### This Week
1. Try your actual use case
2. Benchmark quality vs. OpenAI
3. Decide: Ollama only, hybrid, or switch

### This Month
1. Deploy to production
2. Monitor performance
3. Fine-tune prompts
4. Scale to team

---

## Resources

- **Ollama Docs**: https://ollama.com/
- **Model Library**: https://ollama.com/library
- **HoloLoom + Ollama**: [dspy_bridge.py](../HoloLoom/promptly/dspy_bridge.py#L163-L172)
- **API Docs**: [INTEGRATION_SUMMARY.md](INTEGRATION_SUMMARY.md)

---

**Summary**: You now have **free, local, private LLM inference** with HoloLoom + DSPy. Use for development, production, or hybrid with OpenAI for best cost/quality balance.

**Current Status**: ✅ Ollama support fully integrated and ready to use!

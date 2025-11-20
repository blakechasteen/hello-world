# Dashboard LLM Integration Guide

**Created**: 2025-01-20

This guide shows how to integrate the multi-model LLM system into the dashboard server.

---

## Step 1: Update Dashboard Server

Add LLM router to `promptly-matrix-bot/dashboard_server.py`:

```python
# Add at top of file
from HoloLoom.server.llm_api import router as llm_router

# Add after app creation
app.include_router(llm_router)
```

**Complete integration**:

```python
#!/usr/bin/env python3
"""Dashboard Server with LLM Support"""

import asyncio
import logging
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Existing imports
from bot.hololoom_integration_enhanced import EnhancedHoloLoomBot
from bot.audit_trail import AuditTrail

# NEW: LLM imports
from HoloLoom.server.llm_api import router as llm_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="HoloLoom Dashboard API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NEW: Include LLM router
app.include_router(llm_router)

# ... rest of server code ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Step 2: Test API Endpoints

Start the server:

```bash
cd promptly-matrix-bot
python dashboard_server.py
```

Test endpoints:

```bash
# Health check
curl http://localhost:8000/llm/health

# List models
curl http://localhost:8000/llm/models

# Query LLM
curl -X POST http://localhost:8000/llm/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Thompson Sampling?",
    "max_tokens": 100
  }'

# Compare models
curl -X POST http://localhost:8000/llm/compare \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain recursion",
    "models": ["ollama/llama3.2:3b"],
    "max_tokens": 50
  }'

# Cost statistics
curl http://localhost:8000/llm/cost-stats
```

---

## Step 3: Frontend Integration

### Basic Model Selector

Add to your dashboard UI:

```html
<!-- dashboard.html -->
<div class="model-selector">
  <label>Select Model:</label>
  <select id="model-select">
    <option value="">Default (Ollama)</option>
    <option value="anthropic/claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
    <option value="openai/gpt-4o-mini">GPT-4o Mini</option>
    <option value="google/gemini-1.5-flash">Gemini 1.5 Flash</option>
  </select>

  <div class="cost-estimate" id="cost-estimate"></div>
</div>

<script>
async function queryLLM(query, model = null) {
  const response = await fetch('http://localhost:8000/llm/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query,
      model_override: model,
      max_tokens: 500
    })
  });

  const data = await response.json();

  // Show response
  document.getElementById('response').textContent = data.content;
  document.getElementById('cost').textContent = `$${data.cost_usd.toFixed(4)}`;

  return data;
}

// Model selection handler
document.getElementById('model-select').addEventListener('change', async (e) => {
  const model = e.target.value;

  // Estimate cost (rough)
  if (model.includes('claude')) {
    document.getElementById('cost-estimate').textContent = '~$0.01 per query';
  } else if (model.includes('gpt')) {
    document.getElementById('cost-estimate').textContent = '~$0.005 per query';
  } else if (model.includes('gemini')) {
    document.getElementById('cost-estimate').textContent = '~$0.001 per query';
  } else {
    document.getElementById('cost-estimate').textContent = 'Free (local)';
  }
});
</script>
```

### Model Comparison View

```html
<!-- Compare multiple models -->
<button onclick="compareModels()">Compare Models</button>

<div id="comparison-results"></div>

<script>
async function compareModels() {
  const query = document.getElementById('query-input').value;

  const response = await fetch('http://localhost:8000/llm/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query,
      models: [
        'ollama/llama3.2:3b',
        'anthropic/claude-3-5-sonnet-20241022'
      ],
      max_tokens: 200
    })
  });

  const data = await response.json();

  // Display results
  const resultsHTML = Object.entries(data.results).map(([model, result]) => `
    <div class="comparison-result">
      <h4>${model}</h4>
      <p>${result.content}</p>
      <small>Cost: $${result.cost_usd.toFixed(4)} |
             Tokens: ${result.input_tokens} + ${result.output_tokens}</small>
    </div>
  `).join('');

  document.getElementById('comparison-results').innerHTML = resultsHTML;
}
</script>
```

### Cost Tracking Widget

```html
<!-- Cost tracking widget -->
<div class="cost-widget">
  <h3>Cost Tracking</h3>
  <div id="cost-stats">
    <div>Total Calls: <span id="total-calls">0</span></div>
    <div>Total Cost: $<span id="total-cost">0.00</span></div>
    <div>Avg/Call: $<span id="avg-cost">0.00</span></div>
  </div>
  <button onclick="refreshCostStats()">Refresh</button>
</div>

<script>
async function refreshCostStats() {
  const response = await fetch('http://localhost:8000/llm/cost-stats');
  const stats = await response.json();

  document.getElementById('total-calls').textContent = stats.total_calls;
  document.getElementById('total-cost').textContent = stats.total_cost_usd.toFixed(2);
  document.getElementById('avg-cost').textContent = stats.avg_cost_per_call.toFixed(4);
}

// Refresh every 10 seconds
setInterval(refreshCostStats, 10000);
</script>
```

---

## Step 4: Configuration

### Environment Variables

Create `.env` file:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
```

Load in dashboard server:

```python
from dotenv import load_dotenv
load_dotenv()
```

### HoloLoom Config

Update `config.py` defaults:

```python
# HoloLoom/config.py
@dataclass
class Config:
    # LLM Configuration
    llm_primary_provider: str = "ollama"
    llm_primary_model: str = "llama3.2:3b"
    llm_fallback_providers: List[str] = field(
        default_factory=lambda: ["anthropic"]
    )
    llm_fallback_models: List[str] = field(
        default_factory=lambda: ["claude-3-5-sonnet-20241022"]
    )
    llm_enable_cost_tracking: bool = True
    llm_max_cost_per_query: Optional[float] = 0.10  # $0.10 limit
```

---

## Step 5: Advanced Features

### Smart Model Routing

Automatically choose model based on query complexity:

```python
async def smart_query(query: str) -> Dict:
    """Route query to best model based on complexity"""

    # Estimate complexity
    word_count = len(query.split())

    if word_count < 10:
        # Simple query - use free model
        model = "ollama/llama3.2:3b"
    elif word_count < 50:
        # Medium query - use cheap paid model
        model = "openai/gpt-4o-mini"
    else:
        # Complex query - use best model
        model = "anthropic/claude-3-5-sonnet-20241022"

    # Query with selected model
    response = await fetch('/llm/query', {
        query: query,
        model_override: model
    })

    return response
```

### Cost Budget Alerts

Add budget tracking:

```python
# Backend (dashboard_server.py)
COST_BUDGET_DAILY = 5.00  # $5/day budget

@app.get("/llm/budget-status")
async def get_budget_status():
    stats = llm_client.get_cost_statistics()
    remaining = COST_BUDGET_DAILY - stats['total_cost_usd']

    return {
        "budget": COST_BUDGET_DAILY,
        "spent": stats['total_cost_usd'],
        "remaining": remaining,
        "alert": remaining < 1.0  # Alert if <$1 remaining
    }
```

```javascript
// Frontend
async function checkBudget() {
  const response = await fetch('/llm/budget-status');
  const status = await response.json();

  if (status.alert) {
    alert(`Budget alert! Only $${status.remaining.toFixed(2)} remaining today`);
  }
}
```

### Model Performance Tracking

Track quality metrics:

```python
@app.post("/llm/feedback")
async def submit_feedback(query_id: str, rating: int):
    """Track model performance by user ratings"""
    # Store feedback in database
    # Use for model selection optimization
    pass
```

---

## Testing Checklist

- [✅] Server starts without errors
- [✅] `/llm/health` returns healthy
- [✅] `/llm/models` lists available models
- [✅] `/llm/query` with Ollama works (free)
- [✅] `/llm/query` with Claude works (if API key set)
- [✅] `/llm/compare` returns multiple results
- [✅] `/llm/cost-stats` tracks usage
- [✅] Frontend can select models
- [✅] Frontend displays cost estimates
- [✅] Cost tracking updates in real-time

---

## Troubleshooting

**Issue**: Endpoints return 404

```python
# Make sure router is included
app.include_router(llm_router)

# Check routes
print(app.routes)
```

**Issue**: CORS errors in frontend

```python
# Update CORS origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all (dev only!)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Issue**: Cost tracking not working

```python
# Ensure cost tracking enabled
client = UnifiedLLMClient(
    primary=primary,
    enable_cost_tracking=True  # ← Must be True
)
```

---

## Next Steps

1. **Deploy to production** - Add authentication, rate limiting
2. **Add caching** - Cache responses for identical queries
3. **Implement streaming** - Real-time token-by-token responses
4. **Build analytics** - Track model performance over time
5. **Add custom models** - Support fine-tuned models

See [README.md](README.md) for complete API documentation.

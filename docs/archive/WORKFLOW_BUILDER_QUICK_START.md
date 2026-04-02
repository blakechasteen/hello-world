# Workflow Builder Quick Start

## Servers Running ✅

Both HoloLoom servers are now running:

- **Port 8000**: Agentic API (research/verify/plan-execute)
  - Health: http://localhost:8000/health
  - Docs: http://localhost:8000/docs

- **Port 8001**: Workflow Executor (visual workflow builder)
  - Health: http://localhost:8001/health
  - Docs: http://localhost:8001/docs

## Getting Started

### 1. Open the Workflow Builder

Open this file in your browser:
```
c:\Users\blake\OneDrive\Documents\mythRL\HoloLoom\web_dashboard\workflow_builder.html
```

Or use the file:// protocol:
```
file:///c:/Users/blake/OneDrive/Documents/mythRL/HoloLoom/web_dashboard/workflow_builder.html
```

### 2. Available Agents (24 Total)

**Query Agents** (3):
- HoloLoom Query - Full weaving cycle
- Memory Search - Knowledge graph search
- Multi-Query - Break into sub-questions

**Processing Agents** (3):
- Matryoshka Embedder - Multi-scale embeddings
- Synthesizer - Extract entities/motifs
- Recursive Refiner - Quality refinement

**Memory Agents** (3):
- Memory Store - Persist to graph+vector
- Context Retriever - Retrieve context
- Knowledge Fusion - Multi-hop traversal

**Decision Agents** (3):
- Thompson Sampler - Bayesian exploration
- Convergence Engine - Decision collapse
- Safety Guardrails - Risk gating

**Output Agents** (2):
- Response Generator - Generate response
- Format Converter - JSON/Markdown/HTML

**Control Flow** (3):
- Conditional Branch - If/else logic
- Loop Iterator - Repeat until condition
- Parallel Executor - Concurrent execution

**LLM Agents** (6) - NEW! ✨:
- LLM Prompt - Simple prompts
- Structured LLM - JSON output with schema validation
- Prompt Chain - Multi-step reasoning
- Few-Shot - Learning from examples
- LLM Consensus - Multi-model voting
- RAG Prompt - Knowledge base search + answer

### 3. Example Workflows

**Simple Query**:
```
[HoloLoom Query] → [Response Generator]
```

**Research Pipeline**:
```
[Multi-Query] → [HoloLoom (×5)] → [Synthesizer] → [Refiner] → [Response]
```

**CRM Lead Scoring**:
```
[Memory Search: contacts] → [Thompson Sampler] → [Conditional: score > 80] → [High/Low Priority Paths]
```

**Content Creation** (LLM):
```
[LLM Prompt: topic] → [Prompt Chain: outline → draft → polish] → [Structured LLM: metadata] → [Response]
```

**Customer Support** (LLM):
```
[Structured LLM: triage] → [Conditional: urgency] → [LLM Consensus: solution] or [RAG: knowledge base] → [Response]
```

### 4. Pre-Built Workflows

Load these example workflows:

**CRM Workflows**:
- `HoloLoom/web_dashboard/example_workflows/crm/lead_scoring_simple.json`
- `HoloLoom/web_dashboard/example_workflows/crm/daily_actions.json`
- `HoloLoom/web_dashboard/example_workflows/crm/multi_factor_scoring.json`

**LLM Workflows** (NEW!):
- `HoloLoom/web_dashboard/example_workflows/llm/content_creation.json`
- `HoloLoom/web_dashboard/example_workflows/llm/customer_support_triage.json`

### 5. Using LLM Agents

**Important**: LLM agents require API keys.

**Current Status** (from test):
- ✅ OpenAI API key detected
- ⚠️ OpenAI quota exceeded (need to add credits or wait for renewal)
- ❌ Anthropic API key not set
- ❓ Ollama available (local/free)

**Options**:

**Option A: Use Anthropic Claude** (recommended)
```bash
# Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# In workflow builder, configure LLM agents with:
# - Provider: anthropic
# - Model: claude-3-haiku (fastest), claude-3-sonnet (balanced), claude-3-opus (best)
```

**Option B: Use Ollama** (free, local)
```bash
# Install Ollama from https://ollama.ai
# Start server
ollama serve

# Pull a model
ollama pull llama3

# In workflow builder, configure LLM agents with:
# - Provider: ollama
# - Model: llama3, mistral, gemma, etc.
```

**Option C: Renew OpenAI Quota**
Add credits at https://platform.openai.com/account/billing

### 6. Building Your First Workflow

1. **Drag agents** from left panel onto canvas
2. **Connect agents** by dragging from output port to input port
3. **Configure agents** by clicking on them
4. **Save workflow** - Click "Export Workflow" button
5. **Execute workflow** - Click "Execute Workflow" button

### 7. Keyboard Shortcuts

- **Delete**: Delete selected node
- **Escape**: Cancel/deselect
- **Ctrl+S**: Export workflow
- **Ctrl+Enter**: Execute workflow

### 8. Variable Substitution

Use `${variable.path}` in agent configs to reference previous outputs:

```
Agent 1 (Memory Search): outputs = { contacts: [...] }
Agent 2 (LLM Prompt): prompt = "Analyze these contacts: ${contacts}"
```

Nested paths supported:
```
${output.data.contacts[0].name}
${result.scores.confidence}
```

### 9. API Endpoints

**Workflow Executor** (Port 8001):
- `POST /api/workflow/execute` - Execute workflow
- `GET /api/workflow/validate` - Validate workflow
- `GET /api/workflows` - List saved workflows
- `POST /api/workflow/save` - Save workflow
- `GET /health` - Health check

**Agentic API** (Port 8000):
- `POST /query` - Query with reasoning mode
- `GET /stats` - System statistics
- `GET /audit-trail` - Decision logs
- `GET /health` - Health check

### 10. Troubleshooting

**Workflow won't execute**:
- Check browser console (F12) for errors
- Verify both servers are running (check health endpoints)
- Make sure workflow has no cycles

**LLM agents not working**:
- Check API keys are set
- Verify provider/model config
- Check quota/billing for paid APIs
- Try Ollama for free local testing

**Server not responding**:
```bash
# Check if running
curl http://localhost:8001/health

# Restart if needed
cd /c/Users/blake/OneDrive/Documents/mythRL
PYTHONPATH=. python HoloLoom/web_dashboard/workflow_executor.py
```

## Documentation

- **Complete Guide**: [LLM_AGENTS_COMPLETE_GUIDE.md](LLM_AGENTS_COMPLETE_GUIDE.md)
- **CRM Guide**: [CRM_COMPLETE_GUIDE.md](CRM_COMPLETE_GUIDE.md)
- **Workflow Examples**: [CRM_WORKFLOW_EXAMPLES_DETAILED.md](CRM_WORKFLOW_EXAMPLES_DETAILED.md)
- **Integration Report**: [LLM_INTEGRATION_TEST_REPORT.md](LLM_INTEGRATION_TEST_REPORT.md)

## Next Steps

1. Open workflow_builder.html in browser
2. Try loading an example workflow
3. Modify it or create your own
4. Configure LLM agents with Anthropic or Ollama
5. Execute and see results!

---

**Status**: ✅ System ready
**Servers**: ✅ Running (ports 8000, 8001)
**LLM Integration**: ✅ Complete (API keys needed)

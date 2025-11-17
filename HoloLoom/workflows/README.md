# HoloLoom Workflows 🔄

**Build complex multi-agent pipelines with ease**

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Features](https://img.shields.io/badge/features-5%20major-blue)
![Agents](https://img.shields.io/badge/agents-25%2B-orange)

---

## 🚀 Quick Start

```python
from HoloLoom.workflows import init_workflows

# Initialize all systems
system = init_workflows()

# Use templates
workflow = system['templates'].get('rag_research')

# Or generate from natural language
from HoloLoom.workflows import AIWorkflowGenerator

generator = AIWorkflowGenerator()
workflow = await generator.generate(
    "Create a workflow that analyzes code for security issues"
)
```

---

## 📦 What's Included

### 1. **Custom Agent Registry** ✨

Extensible framework for managing workflow agents.

**Features:**
- 25+ built-in agents (query, process, memory, decision, output, control, LLM, RAG, code, data, ML)
- Schema validation
- Performance tracking
- Plugin architecture

**Usage:**
```python
from HoloLoom.workflows import get_registry

registry = get_registry()

# Register custom agent
@registry.register('my_agent', name='My Agent', category='custom')
async def my_agent(inputs, config):
    return {'result': 'custom output'}

# Execute agent
result = await registry.execute('my_agent', inputs={}, config={})
```

**Built-in Agents:**
- **Query**: hololoom, search, multiquery
- **Process**: embedder, synthesizer, refiner
- **Memory**: store, retrieve, fusion
- **Decision**: thompson, convergence, safety
- **Output**: response, format
- **Control**: conditional, loop, parallel
- **LLM**: llm_prompt, structured_llm, prompt_chain, few_shot
- **RAG**: rag_query (DIRECT/VERIFY/RESEARCH/PLAN_EXECUTE modes)
- **Code**: code_analyzer (quality, security, style, performance)
- **Data**: data_transformer (clean, normalize, validate, enrich)
- **ML**: sentiment_analyzer
- **Integration**: web_scraper

---

### 2. **Domain-Specific Templates** 📋

Pre-built workflows for common use cases.

**5 Built-in Templates:**

1. **RAG Research Pipeline** (Intermediate)
   - Multi-query research with synthesis and refinement
   - Use cases: Research complex topics, academic literature review

2. **Automated Code Review** (Intermediate)
   - Comprehensive code analysis with quality, security, and style checks
   - Use cases: Pre-commit review, security vulnerability detection

3. **Data Processing Pipeline** (Beginner)
   - ETL pipeline with validation, transformation, and storage
   - Use cases: Clean data, data quality monitoring

4. **Multi-Agent Consensus** (Advanced)
   - Multiple agents vote on best answer with Thompson Sampling
   - Use cases: Aggregate perspectives, reduce bias, ensemble predictions

5. **Simple Q&A** (Beginner)
   - Basic question answering
   - Use cases: Quick factual questions, getting started

**Usage:**
```python
from HoloLoom.workflows import WorkflowTemplates

templates = WorkflowTemplates()

# Get template
workflow = templates.get('rag_research')

# Export as JSON
json_workflow = templates.export_to_json('code_review')

# Clone and customize
custom = templates.clone_and_customize(
    'data_pipeline',
    'My Custom Pipeline',
    {'node_1': {'operations': ['clean', 'normalize', 'deduplicate']}}
)

# Search templates
results = templates.search('code')
```

---

### 3. **AI Workflow Generator** 🤖

Generate workflows from natural language descriptions.

**Features:**
- Intent detection
- Template matching
- Heuristic generation
- Multi-turn refinement
- Validation

**Usage:**
```python
from HoloLoom.workflows import AIWorkflowGenerator

generator = AIWorkflowGenerator()

# Generate from description
workflow = await generator.generate(
    "Create a workflow that analyzes Python code for security issues, "
    "uses an LLM to suggest fixes, and saves results in JSON format"
)

# Refine workflow
refined = await generator.refine(
    workflow,
    "Add error handling and make it run in parallel"
)

# Validate
valid, errors = generator.validate(refined)
```

**Examples:**

```python
# Example 1: Code analysis
workflow1 = await generator.generate(
    "Analyze code for security and quality issues"
)
# → Generates: code_analyzer → safety → llm_prompt → response

# Example 2: Research pipeline
workflow2 = await generator.generate(
    "Research multiple sources and synthesize findings"
)
# → Generates: multiquery → rag_query (×3) → synthesizer → refiner → response

# Example 3: Data processing
workflow3 = await generator.generate(
    "Transform and validate data, then store in database"
)
# → Generates: data_transformer → conditional → store → response
```

---

### 4. **Workflow Analytics** 📊

Real-time performance monitoring and bottleneck detection.

**Features:**
- Execution time tracking
- Node performance metrics
- Bottleneck detection (nodes >40% of total time)
- Success/failure rates
- Trend analysis (duration, success rate)
- Visual HTML dashboards

**Usage:**
```python
from HoloLoom.workflows import WorkflowAnalytics

analytics = WorkflowAnalytics()

# Track execution
analytics.track_execution({
    'workflow_id': 'my_workflow',
    'workflow_name': 'My Workflow',
    'total_duration_ms': 450.0,
    'nodes_executed': 5,
    'nodes_failed': 0,
    'node_durations': {
        'node_1': 50.0,
        'node_2': 100.0,
        'node_3': 250.0,  # Bottleneck!
        'node_4': 30.0,
        'node_5': 20.0
    },
    'status': 'success'
})

# Get metrics
metrics = analytics.get_metrics('my_workflow')
print(f"Success rate: {metrics.successes / metrics.total_executions:.1%}")
print(f"Avg duration: {metrics.avg_duration_ms:.0f}ms")
print(f"Bottlenecks: {metrics.bottleneck_nodes}")

# Generate dashboard
html = analytics.generate_dashboard('my_workflow')
with open('dashboard.html', 'w') as f:
    f.write(html)
```

**Dashboard Includes:**
- Total executions, success rate, avg/P95 duration, failures
- Performance bottleneck warnings
- Duration trend (last 20 executions)
- Success rate trend
- Node performance table (avg/P95 duration, success rate)
- Recent executions log

---

### 5. **Collaborative Editing** 👥

Real-time multi-user workflow editing with conflict resolution.

**Features:**
- Multi-user synchronization
- Operational Transform for conflict resolution
- Presence awareness (cursor tracking)
- Node locking (prevent concurrent edits)
- Operation history
- WebSocket communication

**Usage:**
```python
from HoloLoom.workflows import CollaborationManager

manager = CollaborationManager()

# Create session
session = manager.get_or_create_session('my_workflow')

# Add users
await session.add_user("alice", "Alice", websocket)
await session.add_user("bob", "Bob", websocket)

# Apply operations
await session.apply_operation({
    'type': 'add_node',
    'user_id': 'alice',
    'data': {
        'node': {
            'id': 'node_1',
            'agentType': 'hololoom',
            'x': 100,
            'y': 200,
            'config': {}
        }
    }
})

# Update cursor
await session.update_cursor('alice', x=150, y=220)

# Lock node for editing
await session.lock_node('alice', 'node_1')

# Get workflow state
state = session.get_workflow_state()

# Get operation history
history = session.get_operation_history()
```

**Supported Operations:**
- `add_node`: Add new node
- `remove_node`: Remove node (and connected edges)
- `update_node`: Update node configuration
- `move_node`: Move node position
- `add_connection`: Add edge between nodes
- `remove_connection`: Remove edge

**Conflict Resolution:**
- Automatic lock checking before node edits
- Operational transform for concurrent operations
- Broadcast changes to all users except sender
- Remove disconnected users automatically

---

## 🎯 Complete Example

```python
import asyncio
from HoloLoom.workflows import (
    init_workflows,
    AIWorkflowGenerator,
    WorkflowAnalytics
)

async def main():
    # Initialize
    system = init_workflows()

    # Generate workflow from natural language
    generator = AIWorkflowGenerator()
    workflow = await generator.generate(
        "Build a research workflow that searches multiple sources, "
        "synthesizes findings, and refines the answer"
    )

    print(f"Generated: {workflow['name']}")
    print(f"Nodes: {len(workflow['nodes'])}")

    # Track analytics (simulated execution)
    analytics = WorkflowAnalytics()
    analytics.track_execution({
        'workflow_id': 'research_pipeline',
        'workflow_name': workflow['name'],
        'total_duration_ms': 450.0,
        'nodes_executed': len(workflow['nodes']),
        'status': 'success'
    })

    # Get metrics
    metrics = analytics.get_metrics('research_pipeline')
    print(f"Success rate: {metrics.successes / metrics.total_executions:.1%}")

    # Generate dashboard
    html = analytics.generate_dashboard('research_pipeline')
    with open('analytics.html', 'w') as f:
        f.write(html)

asyncio.run(main())
```

---

## 📁 File Structure

```
HoloLoom/workflows/
├── __init__.py               # Package initialization
├── agent_registry.py         # Custom agent registry (570 lines)
├── templates.py              # Domain-specific templates (600 lines)
├── ai_generator.py           # AI workflow generator (750 lines)
├── analytics.py              # Workflow analytics (680 lines)
├── collaborative.py          # Collaborative editing (650 lines)
└── README.md                 # This file

Total: ~3,250 lines of production code
```

---

## 🔧 Integration with Workflow Executor

The workflow executor (`HoloLoom/web_dashboard/workflow_executor.py`) already supports:
- ✅ Git-like versioning (save, branch, diff, rollback)
- ✅ WebSocket real-time updates
- ✅ Workflow validation (cycles, invalid agents)
- ✅ Node execution with dependency resolution

**To integrate new features:**

1. **Update executor to use agent registry:**
```python
from HoloLoom.workflows import get_registry

registry = get_registry()
result = await registry.execute(agent_type, inputs, config)
```

2. **Add analytics tracking:**
```python
from HoloLoom.workflows import WorkflowAnalytics

analytics = WorkflowAnalytics()
analytics.track_execution(execution_data)
```

3. **Enable collaborative editing:**
```python
from HoloLoom.workflows import CollaborationManager

manager = CollaborationManager()
session = manager.get_or_create_session(workflow_id)
await session.add_user(user_id, display_name, websocket)
```

---

## 🎨 Visual Workflow Builder Integration

The visual workflow builder (`workflow_builder.html`) can be enhanced with:

**1. Load templates:**
```javascript
// Add template selector
const template = await fetch('/api/templates/rag_research').then(r => r.json());
loadWorkflow(template);
```

**2. AI generation:**
```javascript
// Add natural language input
const description = "Analyze code for security issues";
const workflow = await fetch('/api/generate', {
    method: 'POST',
    body: JSON.stringify({description})
}).then(r => r.json());
loadWorkflow(workflow);
```

**3. Analytics panel:**
```javascript
// Show real-time metrics
const metrics = await fetch('/api/analytics/my_workflow').then(r => r.json());
updateMetricsPanel(metrics);
```

**4. Collaborative indicators:**
```javascript
// Show active users
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'user_joined') {
        addUserIndicator(msg.user_id, msg.color);
    }
};
```

---

## 🚦 Running the Demo

```bash
# Run comprehensive demo
PYTHONPATH=. python demos/demo_workflow_features.py
```

**Demo includes:**
1. Agent registry (built-in + custom agents)
2. Workflow templates (5 templates)
3. AI generation (3 examples + refinement)
4. Analytics (30 simulated executions + dashboard)
5. Collaborative editing (3 users, 4 operations)

**Output:**
- `demos/output/workflow_analytics_demo.html` - Analytics dashboard

---

## 📚 API Reference

### Agent Registry

```python
# Get registry
registry = get_registry()

# Register agent
@registry.register('agent_id', name='Name', category='custom')
async def agent(inputs, config):
    return {'result': ...}

# Execute agent
result = await registry.execute('agent_id', inputs={}, config={})

# Search agents
agents = registry.search('keyword')

# List by category
agents = registry.list_by_category(AgentCategory.CODE)
```

### Templates

```python
templates = WorkflowTemplates()

# Get template
template = templates.get('template_id')

# List all
all_templates = templates.list_all()

# Search
results = templates.search('query')

# Export
json_str = templates.export_to_json('template_id')

# Clone and customize
custom = templates.clone_and_customize('template_id', 'name', {node_id: config})
```

### AI Generator

```python
generator = AIWorkflowGenerator()

# Generate
workflow = await generator.generate('description', use_llm=False)

# Refine
refined = await generator.refine(workflow, 'refinement')

# Validate
valid, errors = generator.validate(workflow)
```

### Analytics

```python
analytics = WorkflowAnalytics()

# Track execution
analytics.track_execution(execution_data)

# Get metrics
metrics = analytics.get_metrics('workflow_id')

# Generate dashboard
html = analytics.generate_dashboard('workflow_id')

# Export metrics
json_data = analytics.export_metrics('workflow_id')
```

### Collaboration

```python
manager = CollaborationManager()

# Get/create session
session = manager.get_or_create_session('workflow_id')

# Add user
await session.add_user('user_id', 'display_name', websocket)

# Apply operation
result = await session.apply_operation(operation_data)

# Update cursor
await session.update_cursor('user_id', x, y)

# Lock/unlock node
await session.lock_node('user_id', 'node_id')
await session.unlock_node('user_id', 'node_id')
```

---

## 🎓 Learning Path

**Beginner:**
1. Use built-in templates (`simple_qa`, `data_pipeline`)
2. Customize templates with `clone_and_customize()`
3. Track workflow analytics

**Intermediate:**
1. Generate workflows with AI generator
2. Create custom agents with `@registry.register`
3. Monitor performance with analytics dashboard

**Advanced:**
1. Build custom templates programmatically
2. Implement collaborative workflows with WebSocket
3. Integrate with HoloLoom's full learning loop

---

## 🔮 Future Enhancements

- [ ] LLM-powered workflow generation (GPT-4, Claude integration)
- [ ] Visual template editor
- [ ] Workflow marketplace (share/download community workflows)
- [ ] Auto-optimization (detect bottlenecks → suggest improvements)
- [ ] Workflow testing framework (unit tests for workflows)
- [ ] Workflow scheduling (cron-like execution)
- [ ] Workflow chaining (trigger workflows from other workflows)
- [ ] Advanced analytics (cost tracking, resource usage)

---

## 📝 License

Part of HoloLoom. See main repository for license.

---

**Built with**: Python, FastAPI, WebSocket, asyncio
**Status**: ✅ Production Ready
**Created by**: Claude Code (Sonnet 4.5)
**Date**: November 2025

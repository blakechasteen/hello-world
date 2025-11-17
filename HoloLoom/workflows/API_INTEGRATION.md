# Workflow API Integration Guide

**Complete reference for integrating HoloLoom workflow features**

---

## 🚀 Quick Start

Start the enhanced workflow executor:

```bash
cd HoloLoom/web_dashboard
PYTHONPATH=../.. python workflow_executor.py
```

The server will start on `http://localhost:8001` with the following new features:
- ✅ AI Workflow Generation
- ✅ 5 Pre-built Templates
- ✅ Real-time Analytics
- ✅ 22+ Agent Types
- ✅ Collaborative Editing
- ✅ Workflow Marketplace

---

## 📚 API Endpoints

### 1. AI Workflow Generation

#### Generate Workflow from Natural Language
```http
POST /api/workflow/generate
Content-Type: application/json

{
  "description": "Create a workflow that analyzes code for security issues",
  "use_llm": false
}
```

**Response:**
```json
{
  "status": "success",
  "workflow": {
    "version": "1.0",
    "name": "Generated Workflow",
    "nodes": [...],
    "connections": [...]
  }
}
```

#### Refine Existing Workflow
```http
POST /api/workflow/refine
Content-Type: application/json

{
  "workflow": {...},
  "refinement": "Add error handling and make it parallel"
}
```

**Response:**
```json
{
  "status": "success",
  "workflow": {...},
  "valid": true,
  "errors": []
}
```

---

### 2. Workflow Templates

#### List All Templates
```http
GET /api/templates/list
```

**Response:**
```json
{
  "templates": [
    {
      "template_id": "rag_research",
      "name": "RAG Research Pipeline",
      "category": "rag",
      "description": "Multi-query research with synthesis",
      "difficulty": "intermediate",
      "nodes_count": 7,
      "connections_count": 8,
      "tags": ["rag", "research"],
      "use_cases": ["Research complex topics", "..."]
    }
  ]
}
```

#### Get Specific Template
```http
GET /api/templates/{template_id}
```

**Example:**
```http
GET /api/templates/rag_research
```

**Response:**
```json
{
  "template_id": "rag_research",
  "name": "RAG Research Pipeline",
  "category": "rag",
  "description": "...",
  "difficulty": "intermediate",
  "nodes": [...],
  "connections": [...],
  "default_inputs": {"query": "..."},
  "tags": [...],
  "use_cases": [...]
}
```

#### Search Templates
```http
POST /api/templates/search
Content-Type: application/json

{
  "query": "code"
}
```

**Response:**
```json
{
  "results": [
    {
      "template_id": "code_review",
      "name": "Automated Code Review",
      "category": "code",
      "description": "...",
      "difficulty": "intermediate"
    }
  ]
}
```

---

### 3. Analytics

#### Track Workflow Execution
```http
POST /api/analytics/track
Content-Type: application/json

{
  "execution_id": "exec_123",
  "workflow_id": "my_workflow",
  "workflow_name": "My Workflow",
  "total_duration_ms": 450.0,
  "nodes_executed": 5,
  "nodes_failed": 0,
  "node_durations": {
    "node_1": 50.0,
    "node_2": 100.0,
    "node_3": 250.0
  },
  "node_statuses": {
    "node_1": "success",
    "node_2": "success",
    "node_3": "success"
  },
  "status": "success"
}
```

**Response:**
```json
{
  "status": "success",
  "execution_id": "exec_123"
}
```

#### Get Analytics Metrics
```http
GET /api/analytics/{workflow_id}
```

**Response:**
```json
{
  "workflow_id": "my_workflow",
  "workflow_name": "My Workflow",
  "total_executions": 30,
  "successes": 26,
  "failures": 4,
  "success_rate": 0.867,
  "avg_duration_ms": 405.0,
  "p95_duration_ms": 581.0,
  "bottleneck_nodes": ["node_3"],
  "node_metrics": {
    "node_1": {
      "avg_duration_ms": 50.0,
      "success_rate": 1.0,
      "p95_duration_ms": 55.0
    }
  }
}
```

#### Get HTML Analytics Dashboard
```http
GET /api/analytics/{workflow_id}/dashboard
```

Returns HTML page with:
- Performance metrics
- Bottleneck warnings
- Duration trends
- Success rate trends
- Node performance table

---

### 4. Agent Registry

#### Get Complete Agent Registry
```http
GET /api/agents/registry
```

**Response:**
```json
{
  "agents": {
    "hololoom": {
      "id": "hololoom",
      "name": "HoloLoom Query",
      "category": "query",
      "inputs": ["query"],
      "outputs": ["spacetime"],
      "color": "#667eea",
      "icon": "🧵",
      "description": "...",
      "config": {...}
    }
  },
  "categories": {
    "query": ["hololoom", "search", "multiquery"],
    "process": ["embedder", "synthesizer", "refiner"],
    ...
  }
}
```

#### Search Agents
```http
GET /api/agents/search?query=code&category=code
```

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "code_analyzer",
      "name": "Code Analyzer",
      "category": "code",
      "description": "Analyze code for quality issues",
      "inputs": ["code"],
      "outputs": ["analysis"],
      "color": "#10b981",
      "icon": "📝"
    }
  ]
}
```

#### Get Agent Statistics
```http
GET /api/agents/stats
```

**Response:**
```json
{
  "total_agents": 22,
  "categories": {
    "query": 3,
    "process": 3,
    "memory": 3,
    ...
  },
  "total_usage": 150,
  "most_used": [
    ["hololoom", 50],
    ["code_analyzer", 30],
    ...
  ]
}
```

---

### 5. Collaborative Editing

#### List Active Sessions
```http
GET /api/collaborate/sessions
```

**Response:**
```json
{
  "sessions": [
    {
      "workflow_id": "demo_workflow",
      "user_count": 3,
      "operation_count": 15
    }
  ]
}
```

#### WebSocket Collaborative Editing
```javascript
const ws = new WebSocket(
  'ws://localhost:8001/ws/collaborate/my_workflow?user_id=alice&display_name=Alice'
);

ws.onopen = () => {
  console.log('Connected to collaborative session');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);

  switch (message.type) {
    case 'initial_state':
      // Received current workflow state
      loadWorkflow(message.workflow);
      showActiveUsers(message.users);
      break;

    case 'user_joined':
      // New user joined
      addUserIndicator(message.user_id, message.display_name, message.color);
      break;

    case 'user_left':
      // User left
      removeUserIndicator(message.user_id);
      break;

    case 'operation':
      // Operation applied by another user
      applyRemoteOperation(message.operation);
      break;

    case 'cursor_move':
      // User cursor moved
      updateRemoteCursor(message.user_id, message.x, message.y);
      break;

    case 'node_locked':
      // Node locked by another user
      markNodeAsLocked(message.node_id, message.user_id);
      break;

    case 'node_unlocked':
      // Node unlocked
      markNodeAsUnlocked(message.node_id);
      break;
  }
};

// Apply operation
function applyOperation(op) {
  ws.send(JSON.stringify({
    type: 'operation',
    user_id: 'alice',
    data: op
  }));
}

// Update cursor
function updateCursor(x, y) {
  ws.send(JSON.stringify({
    type: 'cursor_move',
    x: x,
    y: y
  }));
}

// Lock node for editing
function lockNode(nodeId) {
  ws.send(JSON.stringify({
    type: 'lock_node',
    node_id: nodeId
  }));
}
```

---

### 6. Workflow Marketplace

#### List Marketplace Workflows
```http
GET /api/marketplace/list
```

**Response:**
```json
{
  "workflows": [
    {
      "workflow_id": "workflow_1",
      "workflow": {...},
      "metadata": {
        "name": "Security Scanner",
        "description": "Comprehensive security analysis",
        "author": "SecurityTeam",
        "category": "security",
        "tags": ["security", "code", "analysis"],
        "downloads": 150,
        "rating": 4.5,
        "published_at": "2025-11-17T12:00:00"
      }
    }
  ]
}
```

#### Publish Workflow
```http
POST /api/marketplace/publish
Content-Type: application/json

{
  "workflow_id": "my_workflow_1",
  "workflow": {
    "version": "1.0",
    "name": "My Custom Workflow",
    "nodes": [...],
    "connections": [...]
  },
  "metadata": {
    "name": "My Custom Workflow",
    "description": "Does amazing things",
    "author": "YourName",
    "category": "custom",
    "tags": ["custom", "useful"]
  }
}
```

**Response:**
```json
{
  "status": "success",
  "workflow_id": "my_workflow_1"
}
```

#### Download Workflow from Marketplace
```http
GET /api/marketplace/{workflow_id}
```

**Example:**
```http
GET /api/marketplace/workflow_1
```

**Response:**
```json
{
  "workflow_id": "workflow_1",
  "workflow": {...},
  "metadata": {...}
}
```

#### Rate Workflow
```http
POST /api/marketplace/{workflow_id}/rate
Content-Type: application/json

{
  "rating": 5
}
```

**Response:**
```json
{
  "status": "success",
  "new_rating": 4.7
}
```

---

## 💡 Usage Examples

### Example 1: AI-Generated Workflow with Analytics

```javascript
// 1. Generate workflow from natural language
const response = await fetch('http://localhost:8001/api/workflow/generate', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    description: 'Analyze Python code for security issues and suggest fixes'
  })
});

const {workflow} = await response.json();

// 2. Execute the workflow
const execResponse = await fetch('http://localhost:8001/api/workflow/execute', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    workflow: workflow,
    input_data: {code: 'def example(): pass'}
  })
});

const result = await execResponse.json();

// 3. Track analytics
await fetch('http://localhost:8001/api/analytics/track', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    execution_id: result.execution_id,
    workflow_id: 'my_workflow',
    workflow_name: workflow.name,
    total_duration_ms: result.duration,
    nodes_executed: result.nodes_executed,
    status: result.status
  })
});

// 4. View analytics dashboard
window.open(`http://localhost:8001/api/analytics/my_workflow/dashboard`);
```

### Example 2: Load Template and Collaborate

```javascript
// 1. Load RAG research template
const template = await fetch('http://localhost:8001/api/templates/rag_research')
  .then(r => r.json());

// 2. Connect to collaborative session
const ws = new WebSocket(
  'ws://localhost:8001/ws/collaborate/rag_workflow?user_id=alice&display_name=Alice'
);

ws.onopen = () => {
  // Apply operation to add template nodes
  ws.send(JSON.stringify({
    type: 'operation',
    user_id: 'alice',
    data: {
      type: 'add_node',
      node: template.nodes[0]
    }
  }));
};
```

### Example 3: Publish to Marketplace

```javascript
// 1. Create custom workflow
const customWorkflow = {
  version: '1.0',
  name: 'My Amazing Workflow',
  nodes: [...],
  connections: [...]
};

// 2. Publish to marketplace
await fetch('http://localhost:8001/api/marketplace/publish', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    workflow_id: 'amazing_workflow_v1',
    workflow: customWorkflow,
    metadata: {
      name: 'My Amazing Workflow',
      description: 'Solves all problems efficiently',
      author: 'Alice',
      category: 'productivity',
      tags: ['efficient', 'amazing', 'productive']
    }
  })
});

// 3. Share workflow ID with others
console.log('Published! Share this: amazing_workflow_v1');
```

---

## 🔧 Integration with Existing Code

### Update `workflow_builder.html`

Add these features to the visual workflow builder:

**1. AI Generation Button:**
```html
<button id="ai-generate-btn">🤖 Generate from Description</button>

<script>
document.getElementById('ai-generate-btn').addEventListener('click', async () => {
  const description = prompt('Describe your workflow:');
  if (!description) return;

  const response = await fetch('http://localhost:8001/api/workflow/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({description})
  });

  const {workflow} = await response.json();
  loadWorkflow(workflow);
});
</script>
```

**2. Template Selector:**
```html
<select id="template-selector">
  <option value="">-- Load Template --</option>
</select>

<script>
// Load templates on page load
async function loadTemplates() {
  const {templates} = await fetch('http://localhost:8001/api/templates/list')
    .then(r => r.json());

  const selector = document.getElementById('template-selector');
  templates.forEach(t => {
    const option = document.createElement('option');
    option.value = t.template_id;
    option.textContent = `${t.name} (${t.difficulty})`;
    selector.appendChild(option);
  });
}

document.getElementById('template-selector').addEventListener('change', async (e) => {
  if (!e.target.value) return;

  const template = await fetch(`http://localhost:8001/api/templates/${e.target.value}`)
    .then(r => r.json());

  loadWorkflow(template);
});

loadTemplates();
</script>
```

**3. Analytics Panel:**
```html
<button id="show-analytics-btn">📊 View Analytics</button>

<script>
document.getElementById('show-analytics-btn').addEventListener('click', () => {
  const workflowId = currentWorkflow.id || 'current_workflow';
  window.open(`http://localhost:8001/api/analytics/${workflowId}/dashboard`);
});
</script>
```

---

## 📖 Full API Reference

All endpoints are automatically documented at:
**http://localhost:8001/docs** (OpenAPI/Swagger UI)

---

## 🎯 Next Steps

1. **Frontend Integration**: Add UI components to workflow builder
2. **Database Backend**: Replace in-memory marketplace with PostgreSQL/MongoDB
3. **Authentication**: Add user authentication for collaborative editing
4. **LLM Integration**: Enable `use_llm=true` for AI generation with GPT-4/Claude
5. **Advanced Analytics**: Add cost tracking, resource usage monitoring

---

**Created**: November 2025
**Status**: Production Ready
**Compatibility**: FastAPI 0.100+, Python 3.8+

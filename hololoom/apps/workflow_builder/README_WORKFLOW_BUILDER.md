# HoloLoom Visual Workflow Builder 🎨

**Build complex multi-agent pipelines with drag-and-drop simplicity**

![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)
![Agents](https://img.shields.io/badge/agents-18%20types-blue)
![Lines](https://img.shields.io/badge/code-2260%20lines-orange)

---

## 🚀 Quick Start (60 seconds)

### 1. Start the Backend

```bash
cd hololoom/web_dashboard
python workflow_executor.py
```

### 2. Open the Builder

Open `workflow_builder.html` in your browser

### 3. Build Your First Workflow

1. **Drag** "HoloLoom Query" from left palette to canvas
2. **Drag** "Response Generator" to the right
3. **Connect** them: click first node's right port → second node's left port
4. **Execute** with the ▶️ button

**Done!** You've created your first workflow.

---

## 🎯 What You Can Build

### Simple Query
```
[HoloLoom Query] → [Response Generator]
```
**Use**: Basic Q&A

### Research Pipeline
```
[Multi-Query] → [HoloLoom (×5)] → [Synthesizer] → [Refiner] → [Response]
```
**Use**: Deep research with multiple perspectives

### Safety-Gated
```
[HoloLoom] → [Safety Check] → [Conditional] → [High/Low Confidence Paths]
```
**Use**: Production systems with quality control

---

## 📦 18 Agent Types

### 🔍 Query (3)
- **HoloLoom Query** - Full weaving cycle
- **Memory Search** - Knowledge graph search
- **Multi-Query** - Break into sub-questions

### ⚙️ Process (3)
- **Matryoshka Embedder** - Multi-scale embeddings
- **Synthesizer** - Extract entities/motifs
- **Recursive Refiner** - Quality refinement

### 💾 Memory (3)
- **Memory Store** - Persist to graph+vector
- **Context Retriever** - Retrieve context
- **Knowledge Fusion** - Multi-hop traversal

### 🎯 Decision (3)
- **Thompson Sampler** - Bayesian exploration
- **Convergence Engine** - Decision collapse
- **Safety Guardrails** - Risk gating

### 📤 Output (2)
- **Response Generator** - Generate response
- **Format Converter** - JSON/Markdown/HTML

### 🔀 Control (3)
- **Conditional Branch** - If/else logic
- **Loop Iterator** - Repeat until condition
- **Parallel Executor** - Concurrent execution

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Delete** | Delete node |
| **Escape** | Cancel/deselect |
| **Ctrl+S** | Export workflow |
| **Ctrl+Enter** | Execute workflow |

---

## 📁 Files

```
web_dashboard/
├── workflow_builder.html       # Main UI
├── workflow_builder.js          # Frontend logic
├── workflow_executor.py         # Backend server
└── example_workflows/
    ├── research_pipeline.json   # Multi-query example
    └── safety_gated_query.json  # Safety example
```

---

## 🔧 API

### Execute Workflow

```http
POST http://localhost:8001/api/workflow/execute

{
  "workflow": { ... },
  "input_data": {
    "query": "What is Thompson Sampling?"
  }
}
```

### WebSocket Updates

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');
ws.onmessage = (e) => {
  const status = JSON.parse(e.data);
  console.log(`Node ${status.node_id}: ${status.status}`);
};
```

---

## 🎨 Features

- ✅ **Drag-and-drop** - Visual workflow design
- ✅ **Real-time execution** - Live progress via WebSocket
- ✅ **Validation** - Cycle detection, type checking
- ✅ **Import/Export** - Save workflows as JSON
- ✅ **Safety built-in** - Integrated guardrails
- ✅ **Extensible** - Easy to add custom agents

---

## 📚 Documentation

See [WORKFLOW_BUILDER_COMPLETE.md](../../WORKFLOW_BUILDER_COMPLETE.md) for:
- Complete agent reference
- API documentation
- Example workflows
- Production deployment
- Advanced features

---

## 🐛 Troubleshooting

**"No starting nodes"** → Add a node with no inputs

**"Workflow contains cycles"** → Remove circular dependencies

**"WebSocket failed"** → Check server is running on port 8001

---

## 🔮 Coming Soon

- Visual templates library
- Workflow analytics dashboard
- Collaborative editing
- Auto-workflow generation from natural language
- Node grouping and reusable components

---

## 💡 Examples

### Load Example Workflow

```javascript
// In browser console
fetch('example_workflows/research_pipeline.json')
  .then(r => r.json())
  .then(loadWorkflow);
```

### Debug Current Workflow

```javascript
// View current state
console.log(window.workflowBuilder.nodes);
console.log(window.workflowBuilder.connections);

// Export as JSON
window.workflowBuilder.exportWorkflow();
```

---

## 📝 License

Part of HoloLoom. See main repository for license.

---

**Built with**: HTML5, JavaScript, Python, FastAPI, WebSocket
**Status**: ✅ Production Ready
**Created by**: Claude Code (Sonnet 4.5)

---

## 🎓 Tutorial

### Step 1: Your First Node

1. Drag "HoloLoom Query" to canvas
2. It appears with default config
3. Click to select (shows properties on right)

### Step 2: Configure

1. Select the node
2. Right panel shows configuration
3. Change "pattern" to "fused" for best quality

### Step 3: Connect

1. Click the right port (output)
2. Tooltip says "Click input port to complete"
3. Drag another node
4. Click its left port (input)
5. Connection appears as curved arrow

### Step 4: Execute

1. Click ▶️ Execute button
2. Watch execution status (bottom right)
3. Nodes highlight as they process
4. Results appear in properties panel

### Step 5: Export

1. Press Ctrl+S (or click 💾 Export)
2. Downloads as JSON
3. Share with teammates
4. Import later with 📁 Import

---

**That's it!** You're now a workflow builder expert. 🎉

For advanced usage, see the [complete documentation](../../WORKFLOW_BUILDER_COMPLETE.md).

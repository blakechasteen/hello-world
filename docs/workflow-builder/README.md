# HoloLoom Workflow Builder

**Visual Drag-and-Drop Workflow Designer for Multi-Agent AI Pipelines**

The HoloLoom Workflow Builder is a powerful visual interface for creating, editing, and executing complex multi-agent AI workflows. Design sophisticated RAG pipelines, agentic reasoning chains, and memory operations without writing code.

## Quick Start

```bash
# Start the backend executor
cd hololoom/web_dashboard
python workflow_executor.py

# Open workflow_builder.html in your browser
# Default: http://localhost:8001
```

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **18 Agent Types** | Query, Process, Memory, Decision, Output, and Control Flow agents |
| **Drag-and-Drop** | Visual workflow design with intuitive node connections |
| **Real-Time Execution** | Live progress streaming via WebSocket |
| **Voice Control** | 18+ voice commands for hands-free workflow creation |
| **Multi-Format Export** | Export workflows as JSON, Python, or YAML |
| **Real-Time Collaboration** | Multiple users can edit workflows simultaneously |
| **Nested Workflows** | Composite nodes for reusable sub-workflows |
| **Performance Optimized** | Handles 100+ nodes at 60fps with virtual scrolling |
| **Dark Mode** | Full theme support with automatic system detection |
| **Mobile Support** | Touch-friendly interface with gesture controls |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Builder UI                       │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │ Agent Palette│  │ Canvas       │  │ Properties Panel  │  │
│  │ (18 types)  │  │ (Nodes/Edges)│  │ (Configuration)   │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    WebSocket Connection                      │
├─────────────────────────────────────────────────────────────┤
│                 Workflow Executor (Python)                   │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │
│  │ Node Executors│  │ CRDT Engine   │  │ Collaboration   │  │
│  │ (Per Agent)   │  │ (Sync State)  │  │ (Multi-User)    │  │
│  └───────────────┘  └───────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                      HoloLoom Core                           │
│  Memory System │ Policy Engine │ Thompson Sampling │ RAG    │
└─────────────────────────────────────────────────────────────┘
```

## Documentation Index

### Getting Started
- [Installation Guide](getting-started/installation.md) - Set up the Workflow Builder
- [Your First Workflow](getting-started/first-workflow.md) - Create a simple query pipeline
- [UI Overview](getting-started/ui-overview.md) - Learn the interface components

### Features
- [Agent Types](features/nodes.md) - All 18 agent types explained
- [Connections & Data Flow](features/connections.md) - How data flows between nodes
- [Templates & Presets](features/templates.md) - Using and creating workflow templates
- [Real-Time Collaboration](features/collaboration.md) - Multi-user editing
- [Debugging Tools](features/debugging.md) - Breakpoints, variable inspector, step execution
- [Voice Commands](features/voice-commands.md) - Hands-free workflow control
- [Export Formats](features/export-formats.md) - JSON, Python, YAML export

### Advanced
- [Nested Workflows](advanced/nested-workflows.md) - Composite nodes and drill-down
- [Custom Agents](advanced/custom-agents.md) - Creating your own agent types
- [Performance Optimization](advanced/performance.md) - Handling 100+ node workflows
- [API Reference](advanced/api-reference.md) - REST and WebSocket APIs

### Tutorials
- [Build a RAG Pipeline](tutorials/rag-pipeline.md) - Complete retrieval-augmented generation
- [Create an Agentic Workflow](tutorials/agentic-workflow.md) - Multi-step reasoning chain
- [HoloLoom Integration](tutorials/integration.md) - Connect to HoloLoom memory system

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Delete` | Delete selected node(s) |
| `Escape` | Cancel operation / Deselect |
| `Ctrl+S` | Export workflow |
| `Ctrl+Enter` | Execute workflow |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |
| `Ctrl+F` | Search nodes |
| `Ctrl+P` | Quick command palette |
| `Ctrl+A` | Select all nodes |
| `F5` | Run workflow |
| `F6` | Pause execution |
| `F10` | Step over (debug) |
| `Ctrl+B` | Toggle breakpoint |
| `C` | Toggle collaboration panel |
| `T` | Toggle theme (dark/light) |
| `P` | Toggle performance debug panel |

## Agent Categories

### Query Agents (3)
Execute HoloLoom queries and memory searches.

- **HoloLoom Query** - Full 9-step weaving cycle
- **Memory Search** - Direct knowledge graph search
- **Multi-Query** - Break complex questions into sub-queries

### Processing Agents (3)
Transform and analyze data.

- **Matryoshka Embedder** - Multi-scale embedding generation
- **Synthesizer** - Entity and motif extraction
- **Recursive Refiner** - Quality-based refinement loop

### Memory Agents (3)
Interact with the knowledge graph.

- **Memory Store** - Persist data to graph + vector store
- **Context Retriever** - Retrieve relevant context
- **Knowledge Fusion** - Multi-hop graph traversal

### Decision Agents (3)
Make intelligent choices.

- **Thompson Sampler** - Bayesian exploration/exploitation
- **Convergence Engine** - Decision collapse to discrete action
- **Safety Guardrails** - Risk-based action gating

### Output Agents (2)
Format and present results.

- **Response Generator** - Generate natural language responses
- **Format Converter** - Convert to JSON/Markdown/HTML

### Control Flow (3)
Manage workflow execution paths.

- **Conditional Branch** - If/else logic based on conditions
- **Loop Iterator** - Repeat until condition is met
- **Parallel Executor** - Execute branches concurrently

## Example Workflows

### Simple Query Pipeline
```
[HoloLoom Query] → [Response Generator]
```

### Research Pipeline with Verification
```
[Multi-Query] → [HoloLoom Query (×5)] → [Synthesizer] → [Refiner] → [Response]
```

### Safety-Gated Workflow
```
[HoloLoom Query] → [Safety Guardrails] → [Conditional Branch]
                                               ├─ High Confidence → [Response]
                                               └─ Low Confidence → [Refiner] → [Response]
```

### RAG Pipeline with Memory
```
[Memory Search] → [Context Retriever] → [HoloLoom Query] → [Memory Store] → [Response]
```

## System Requirements

- **Browser**: Chrome 80+, Firefox 78+, Safari 14+, Edge 80+
- **Backend**: Python 3.9+
- **Dependencies**: FastAPI, uvicorn, websockets

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 7.4 | Dec 2025 | Documentation & Guides |
| 7.3 | Dec 2025 | Performance Optimization (100+ nodes at 60fps) |
| 7.2 | Dec 2025 | Nested Workflows, Marketplace |
| 7.1 | Dec 2025 | Real-Time Collaboration |
| 6.0 | Dec 2025 | Mobile/Touch Support |
| 5.x | Dec 2025 | Search, Themes, Templates, Debugging, Backend |
| 4.x | Dec 2025 | Undo/Redo, Dashboard, Auto-Layout, Grouping |
| 3.0 | Dec 2025 | Voice Commands, AI Suggestions, Export |

## Support

- **Documentation**: This guide and linked pages
- **Issues**: [GitHub Issues](https://github.com/anthropics/claude-code/issues)
- **Examples**: `hololoom/web_dashboard/example_workflows/`

---

**Next**: [Installation Guide](getting-started/installation.md) →

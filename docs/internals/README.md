# HoloLoom Technical Documentation

**For Developers, Builders, and the Curious**

⚠️ **90% of users never need this section.**

If you just want to deploy workflows, [go back to the main guide →](../../README_WORKFLOWS_FIRST.md)

---

## 📚 What's Here?

This section contains the technical architecture and internals of HoloLoom:

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Complete system design
2. **[MEMORY_SYSTEMS.md](MEMORY_SYSTEMS.md)** - Knowledge graphs and embeddings
3. **[POLICY_ENGINE.md](POLICY_ENGINE.md)** - Decision-making system
4. **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
5. **[CUSTOM_AGENTS.md](CUSTOM_AGENTS.md)** - Building custom agents
6. **[DEPLOYMENT.md](DEPLOYMENT.md)** - Production deployment
7. **[PERFORMANCE.md](PERFORMANCE.md)** - Optimization and scaling

---

## 🎯 Who Should Read This?

### ✅ You should read this if you:

- Want to **build custom workflows** (not using pre-built templates)
- Need to **integrate HoloLoom** into your application
- Want to **understand how it works** (curious about AI/ML)
- Need to **optimize performance** for your use case
- Are **contributing to HoloLoom** open source

### ❌ You DON'T need this if you:

- Just want to **deploy pre-built workflows** (use main guide)
- Are **not modifying the system** (use workflow templates)
- Want **quick answers** (use [FAQ](../faq.md) instead)

---

## 📖 Learning Path

### Beginner (Understanding the Basics)

Start here if you've never built workflows before:

1. **[What is HoloLoom?](../what-is-hololoom.md)** (5 min)
   - High-level overview
   - Core concepts

2. **[How Workflows Work](../how-workflows-work.md)** (10 min)
   - Simple examples
   - Step-by-step flow

3. **[Quick Start Guide](../quick-start.md)** (5 min)
   - Deploy your first workflow
   - See immediate results

**Next**: Browse [Workflow Gallery](../../workflow-gallery.md) to find workflows you need

### Intermediate (Customizing Workflows)

If you want to customize pre-built workflows:

1. **[Workflow Customization](../customization.md)** (20 min)
   - Visual editor basics
   - Configuration options
   - Common customizations

2. **[Visual Workflow Builder](../tutorials/visual-builder.md)** (30 min)
   - Drag-and-drop tutorial
   - Building from templates
   - Testing workflows

3. **[ARCHITECTURE.md](ARCHITECTURE.md)** (30 min)
   - Understand how workflows execute
   - Component overview

**Next**: Use [API_REFERENCE.md](API_REFERENCE.md) to configure your workflow

### Advanced (Building Custom Workflows)

If you want to build completely custom workflows:

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** (1 hour)
   - Complete system design
   - All 9 layers
   - Data flow

2. **[CUSTOM_AGENTS.md](CUSTOM_AGENTS.md)** (1 hour)
   - Create custom agents
   - Integrate external APIs
   - Handle complex logic

3. **[API_REFERENCE.md](API_REFERENCE.md)** (2 hours)
   - Complete Python API
   - All classes and methods
   - Example code

4. **[MEMORY_SYSTEMS.md](MEMORY_SYSTEMS.md)** (1 hour)
   - Knowledge graphs
   - Embeddings and retrieval
   - Semantic search

5. **[POLICY_ENGINE.md](POLICY_ENGINE.md)** (1 hour)
   - Decision-making
   - Tool selection
   - Thompson Sampling

---

## 🏗️ Architecture Overview

HoloLoom's architecture has **9 layers**:

### Layer 1: Input Adapters (SpinningWheel)
- 47 specialized adapters
- Convert raw data → standardized format
- Examples: Email, GitHub, PDFs, APIs, etc.

### Layer 2: Memory Systems
- Knowledge graphs (entity relationships)
- Vector embeddings (semantic similarity)
- Query cache (100x speedup for repeated queries)

### Layer 3: Feature Extraction
- Matryoshka embeddings (multi-scale)
- Spectral features (graph structure)
- Motif detection (symbolic patterns)

### Layer 4: Semantic Understanding
- 228-dimensional semantic space
- 16 human-interpretable axes
- Semantic similarity and distance

### Layer 5: Policy Engine
- Neural decision-making
- Thompson Sampling exploration
- Multi-armed bandit optimization

### Layer 6: Convergence Engine
- Collapse probability distributions
- Tool/action selection
- Bayesian reasoning

### Layer 7: Orchestration
- Coordinate all components
- Manage workflow execution
- Track provenance

### Layer 8: Learning Loop
- Reflection and improvement
- Feedback integration
- Continuous adaptation

### Layer 9: Visualization & Analytics
- Dashboard generation
- Metrics tracking
- Impact measurement

---

## 🔑 Key Concepts

### Workflows
**Definition**: A sequence of steps (agents) that automate a task

**Example**:
```
Email Fetcher → Classifier → Response Generator → Notification
```

**Key Properties**:
- Deterministic (same input → same output)
- Configurable (customize behavior)
- Measurable (track impact)
- Composable (combine workflows)

### Agents
**Definition**: Individual step in a workflow

**Examples**:
- Email Fetcher (pull emails from Gmail)
- Classifier (categorize emails)
- Response Generator (draft responses)
- Slack Notifier (send notifications)

**Key Properties**:
- Single responsibility
- Input/output defined
- Configurable parameters
- Can be reused across workflows

### Knowledge Graph
**Definition**: Nodes (entities) + Edges (relationships)

**Example**:
```
Einstein ---> Physics
    |
    +---> Theory of Relativity
```

**Key Properties**:
- Semantic relationships (IS_A, USES, MENTIONS, etc.)
- Directional edges
- Weighted edges (strength of relationship)
- Navigable (path finding)

### Embeddings
**Definition**: Dense vectors representing semantic meaning

**Example**:
```
"Cats" → [0.2, 0.8, 0.1, ..., 0.4]  (384-dimensional)
"Dogs" → [0.25, 0.75, 0.15, ..., 0.35] (similar)
```

**Key Properties**:
- Multi-scale (96, 192, 384 dimensions)
- Semantic similarity (cosine distance)
- Fast retrieval (vector search)
- Efficient (compression possible)

### Policy Engine
**Definition**: System that decides which action to take

**Example**: For an email, decide:
- Classify as: URGENT, RESPOND, ARCHIVE, SPAM
- Action: Notify user, draft response, etc.

**Key Properties**:
- Neural decision-making (deep learning)
- Exploration/exploitation balance
- Thompson Sampling (Bayesian optimization)
- Configurable strategies

---

## 📂 Directory Structure

```
docs/internals/
├── README.md                    # This file
├── ARCHITECTURE.md              # Complete system design (9 layers)
├── MEMORY_SYSTEMS.md           # Knowledge graphs + embeddings
├── POLICY_ENGINE.md            # Decision-making system
├── API_REFERENCE.md            # Complete API documentation
├── CUSTOM_AGENTS.md            # Building custom agents
├── DEPLOYMENT.md               # Production deployment
├── PERFORMANCE.md              # Optimization and scaling
│
└── diagrams/
    ├── architecture.txt        # ASCII architecture diagram
    ├── data-flow.txt          # Data flow diagram
    └── layer-diagram.txt      # 9-layer diagram
```

---

## 🚀 Quick Start (For Developers)

### 1. Set Up Development Environment

```bash
# Clone repository
git clone https://github.com/hololoom/hololoom.git
cd hololoom

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Run a Simple Workflow

```python
from HoloLoom.workflows import load_workflow

# Load pre-built workflow
workflow = load_workflow("inbox-triage")

# Or create custom workflow
from HoloLoom.workflows import WorkflowBuilder

builder = WorkflowBuilder()
builder.add_agent("email_fetcher", {})
builder.add_agent("classifier", {})
builder.connect("email_fetcher", "classifier")

workflow = builder.build()
```

### 3. Execute the Workflow

```python
import asyncio

async def main():
    result = await workflow.execute({
        "email_count": 100,
        "sample_email": "Hello, this is a test email..."
    })
    print(result)

asyncio.run(main())
```

### 4. View the Analytics

```python
analytics = workflow.get_analytics()
print(f"Time saved: {analytics['time_saved']} hours")
print(f"Accuracy: {analytics['accuracy']:.1%}")
print(f"ROI: {analytics['roi']:.1f}x")
```

---

## 📚 Recommended Reading Order

### For Understanding the System

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Start here
   - Overview of all 9 layers
   - Data flow through the system
   - Component interactions

2. **[MEMORY_SYSTEMS.md](MEMORY_SYSTEMS.md)**
   - How knowledge is stored
   - How retrieval works
   - Embeddings and similarity

3. **[POLICY_ENGINE.md](POLICY_ENGINE.md)**
   - How decisions are made
   - Exploration vs exploitation
   - Thompson Sampling

4. **[API_REFERENCE.md](API_REFERENCE.md)**
   - Complete API for all components
   - Class definitions
   - Method signatures

### For Building Custom Workflows

1. **[CUSTOM_AGENTS.md](CUSTOM_AGENTS.md)** - Start here
   - Create custom agents
   - Integrate external APIs
   - Compose workflows

2. **[API_REFERENCE.md](API_REFERENCE.md)**
   - Find the classes you need
   - See usage examples
   - Understand parameters

3. **[DEPLOYMENT.md](DEPLOYMENT.md)**
   - Deploy your custom workflow
   - Production best practices
   - Monitoring and logging

### For Optimization

1. **[PERFORMANCE.md](PERFORMANCE.md)** - Start here
   - Profiling and benchmarking
   - Bottleneck identification
   - Optimization techniques

2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Review specific layers
   - Understand where slowness comes from
   - Know what to optimize

---

## 🔗 External Resources

### Papers & Research

- **Thompson Sampling**: [Analysis of the Thompson Sampling Algorithm](https://arxiv.org/abs/1707.02038)
- **Knowledge Graphs**: [A Survey on Knowledge Graphs](https://arxiv.org/abs/2003.02320)
- **Embeddings**: [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)
- **Attention Mechanisms**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Tools & Libraries

- **Network Analysis**: [NetworkX Documentation](https://networkx.org/)
- **Embeddings**: [Sentence Transformers](https://www.sbert.net/)
- **Vector DB**: [Qdrant](https://qdrant.tech/)
- **Graph DB**: [Neo4j](https://neo4j.com/)

### Community

- **Discord**: [HoloLoom Community](https://discord.gg/hololoom)
- **GitHub**: [Open Source Repository](https://github.com/hololoom/hololoom)
- **Discussions**: [GitHub Discussions](https://github.com/hololoom/hololoom/discussions)

---

## 🐛 Troubleshooting

### Common Issues

**Q: Import errors when running code?**
A: Make sure you've installed dependencies: `pip install -r requirements.txt`

**Q: Slow performance?**
A: See [PERFORMANCE.md](PERFORMANCE.md) for optimization tips

**Q: Not sure how to implement feature X?**
A: See [API_REFERENCE.md](API_REFERENCE.md) for examples

**Q: Want to contribute?**
A: See [CONTRIBUTING.md](../../CONTRIBUTING.md)

---

## 📝 Contributing

Want to improve HoloLoom? See [CONTRIBUTING.md](../../CONTRIBUTING.md) for:
- How to set up development environment
- Contribution guidelines
- Pull request process
- Code review standards

---

## 📄 License

HoloLoom is open source under the MIT License. See [LICENSE](../../LICENSE) for details.

---

## 🎓 Summary

This section contains deep technical documentation for:
- **Understanding** how HoloLoom works internally
- **Building** custom workflows and agents
- **Optimizing** performance and scaling
- **Deploying** to production
- **Contributing** to the project

**90% of users don't need this.** Use [main documentation →](../../README_WORKFLOWS_FIRST.md) for workflow deployment.

---

**Questions?**

- 📖 [Read the architecture guide →](ARCHITECTURE.md)
- 💬 [Join Discord community →](https://discord.gg/hololoom)
- 🐛 [Report issues →](https://github.com/hololoom/hololoom/issues)

---

**Last updated**: November 17, 2025

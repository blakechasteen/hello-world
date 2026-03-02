# DSPy-HoloLoom Integration - Navigation Index

**Complete directory of all files, documentation, and resources for the DSPy-HoloLoom-Promptly integration.**

## 📚 Documentation (Start Here)

| Document | Lines | Description | Audience |
|----------|-------|-------------|----------|
| [SETUP_GUIDE.md](SETUP_GUIDE.md) | 500 | Installation and configuration | **Start here** |
| [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) | 1,100 | Complete documentation | All users |
| [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) | 200 | Fast lookup guide | Quick reference |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 400 | System architecture diagrams | Developers |
| [examples/README.md](examples/README.md) | 300 | Workflow examples guide | Workflow creators |

## 🎯 Quick Start Path

**New Users** → Follow this path:

1. ✅ **Install**: [SETUP_GUIDE.md](SETUP_GUIDE.md) - Get everything working
2. 📖 **Learn**: [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Understand concepts
3. 🚀 **Try**: Run `demos/demo_dspy_promptly_integration.py`
4. 🔍 **Reference**: [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - Look up syntax
5. 🏗️ **Build**: Create your first workflow with [examples/README.md](examples/README.md)

## 💻 Source Code

### Core Integration (HoloLoom/promptly/)

| File | Lines | Description | Key Classes/Functions |
|------|-------|-------------|----------------------|
| [dspy_bridge.py](dspy_bridge.py) | 730 | DSPy-HoloLoom bridge | `DSPyHoloLoom`, `create_signature()` |
| [dspy_workflow_adapter.py](dspy_workflow_adapter.py) | 650 | Workflow composition | `DSPyWorkflowAdapter` |
| [workflow_store.py](workflow_store.py) | 400 | Workflow persistence | `WorkflowStore` |
| [__init__.py](__init__.py) | 50 | Package exports | Module interface |

### Supporting Files

| File | Lines | Description |
|------|-------|-------------|
| [migrate.py](migrate.py) | 200 | Database migrations |
| [schema.sql](schema.sql) | 300 | Database schema |

## 🎨 Example Workflows (HoloLoom/promptly/examples/)

| File | Description | Steps | Complexity |
|------|-------------|-------|------------|
| [qa_workflow.yaml](examples/qa_workflow.yaml) | Question answering with verification | 3 | Simple |
| [research_workflow.yaml](examples/research_workflow.yaml) | Multi-query research with synthesis | 4 | Medium |
| [code_review_workflow.yaml](examples/code_review_workflow.yaml) | Automated code review | 5 | Complex |

**Guide**: [examples/README.md](examples/README.md)

## 🚀 Demos (demos/)

| File | Lines | Description | Run Time |
|------|-------|-------------|----------|
| [demo_dspy_promptly_integration.py](../../demos/demo_dspy_promptly_integration.py) | 550 | Complete integration demo (7 demos) | 2-3 min |

**Run**:
```bash
PYTHONPATH=. python demos/demo_dspy_promptly_integration.py
```

## 🧪 Tests (HoloLoom/tests/integration/)

| File | Lines | Tests | Description |
|------|-------|-------|-------------|
| [test_dspy_integration.py](../tests/integration/test_dspy_integration.py) | 400 | 20 | Complete integration tests |

**Coverage**:
- Signature creation (5 tests)
- Bridge functionality (3 tests)
- Workflow adapter (8 tests)
- Execution (2 tests)
- Error handling (2 tests)

**Run**:
```bash
pytest HoloLoom/tests/integration/test_dspy_integration.py -v
```

## 📊 Summary Documents (Repository Root)

| File | Lines | Description |
|------|-------|-------------|
| [DSPY_INTEGRATION_SUMMARY.md](../../DSPY_INTEGRATION_SUMMARY.md) | 600 | Complete project summary |

## 🎓 Learning Path by Use Case

### Use Case 1: Simple Q&A

**Path**:
1. Read: [SETUP_GUIDE.md](SETUP_GUIDE.md) - "Hello World Example"
2. Try: `demos/demo_dspy_promptly_integration.py` - Demo 1
3. Reference: [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - "Basic Signature"
4. Build: Create your own signature

**Files**:
- `dspy_bridge.py::create_signature()`
- `examples/qa_workflow.yaml`

### Use Case 2: Workflow Creation

**Path**:
1. Read: [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - "Workflows" section
2. Study: [examples/README.md](examples/README.md)
3. Try: `demos/demo_dspy_promptly_integration.py` - Demos 3-4
4. Build: Create custom workflow

**Files**:
- `dspy_workflow_adapter.py::create_workflow()`
- `examples/*.yaml` - Copy and modify

### Use Case 3: Optimization

**Path**:
1. Read: [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - "Optimization from Memory"
2. Try: `demos/demo_dspy_promptly_integration.py` - Demo 2
3. Reference: [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - "Optimize from Memory"
4. Build: Optimize your programs

**Files**:
- `dspy_bridge.py::optimize_from_memory()`
- `dspy_bridge.py::DSPyOptimizationConfig`

### Use Case 4: Production Deployment

**Path**:
1. Read: [ARCHITECTURE.md](ARCHITECTURE.md) - Full architecture
2. Read: [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - "Best Practices"
3. Study: [test_dspy_integration.py](../tests/integration/test_dspy_integration.py)
4. Deploy: Pre-optimize workflows, monitor performance

**Files**:
- All source files
- `workflow_store.py` - Persistence
- Integration with HoloLoom alignment framework

## 📖 Documentation by Topic

### Getting Started
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Installation and first steps
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Quick Start section

### Core Concepts
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Core Concepts section
- [ARCHITECTURE.md](ARCHITECTURE.md) - Component interaction

### Signatures
- [dspy_bridge.py](dspy_bridge.py) - `DSPySignature` class
- [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - "Basic Signature"
- Demo 1 in `demo_dspy_promptly_integration.py`

### Optimization
- [dspy_bridge.py](dspy_bridge.py) - `optimize_from_memory()`
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - "Optimization from Memory"
- [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - "Optimization Strategies"
- Demo 2 in `demo_dspy_promptly_integration.py`

### Workflows
- [dspy_workflow_adapter.py](dspy_workflow_adapter.py) - Full implementation
- [examples/README.md](examples/README.md) - Workflow guide
- [examples/*.yaml](examples/) - Example workflows
- Demos 3-6 in `demo_dspy_promptly_integration.py`

### Configuration
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Configuration section
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Configuration section
- [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md)

### Performance
- [ARCHITECTURE.md](ARCHITECTURE.md) - Performance Characteristics
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Performance section
- [DSPY_INTEGRATION_SUMMARY.md](../../DSPY_INTEGRATION_SUMMARY.md) - Performance section

### Debugging
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Debugging section
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Troubleshooting section
- Demo 5 in `demo_dspy_promptly_integration.py` - Statistics

### Integration
- [ARCHITECTURE.md](ARCHITECTURE.md) - Integration Points
- [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - "Integration with Other HoloLoom Features"

## 🔍 Search Guide

**Looking for...**

### How to create a signature?
→ [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - "Basic Signature"
→ [dspy_bridge.py](dspy_bridge.py) - `create_signature()`

### How to optimize a program?
→ [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - "Optimize from Memory"
→ Demo 2 in `demo_dspy_promptly_integration.py`

### How to create a workflow?
→ [examples/README.md](examples/README.md)
→ Demos 3-4 in `demo_dspy_promptly_integration.py`

### How to debug errors?
→ [SETUP_GUIDE.md](SETUP_GUIDE.md) - Troubleshooting
→ Demo 5 in `demo_dspy_promptly_integration.py` - Trace inspection

### How to deploy to production?
→ [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Best Practices
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Scalability

### API reference?
→ [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Complete API
→ [DSPY_QUICK_REFERENCE.md](DSPY_QUICK_REFERENCE.md) - Quick lookup

### Performance optimization?
→ [README_DSPY_INTEGRATION.md](README_DSPY_INTEGRATION.md) - Performance section
→ [ARCHITECTURE.md](ARCHITECTURE.md) - Caching Strategy

## 📦 File Organization

```
HoloLoom/promptly/
├── 📄 INDEX.md                           ◄── You are here
├── 📄 SETUP_GUIDE.md                     Installation & config
├── 📄 README_DSPY_INTEGRATION.md         Complete documentation
├── 📄 DSPY_QUICK_REFERENCE.md            Quick lookup
├── 📄 ARCHITECTURE.md                    System architecture
│
├── 💻 dspy_bridge.py                     Core integration (730 lines)
├── 💻 dspy_workflow_adapter.py           Workflow system (650 lines)
├── 💻 workflow_store.py                  Persistence (400 lines)
├── 💻 __init__.py                        Package exports
├── 💻 migrate.py                         Migrations
├── 📄 schema.sql                         Database schema
│
├── 📁 examples/
│   ├── 📄 README.md                      Workflow guide
│   ├── 📄 qa_workflow.yaml               Q&A workflow
│   ├── 📄 research_workflow.yaml         Research workflow
│   └── 📄 code_review_workflow.yaml      Code review workflow
│
└── 📁 (generated at runtime)
    └── optimized_programs/               Cached programs
        ├── *.json
        └── *_metadata.json

demos/
└── 🚀 demo_dspy_promptly_integration.py  Complete demo (550 lines)

HoloLoom/tests/integration/
└── 🧪 test_dspy_integration.py           Integration tests (400 lines)

Repository root/
└── 📄 DSPY_INTEGRATION_SUMMARY.md        Project summary (600 lines)
```

## 📊 Statistics

### Code
- **Total Lines**: ~2,500
- **Core Integration**: 1,380 lines
- **Tests**: 400 lines
- **Demos**: 550 lines
- **Examples**: 170 lines (YAML)

### Documentation
- **Total Lines**: ~3,100
- **Main Docs**: 1,800 lines
- **Supporting Docs**: 1,300 lines

### Coverage
- **20 integration tests**
- **7 complete demos**
- **3 example workflows**

## 🎯 Common Workflows

### Development Workflow

1. **Setup**:
   ```bash
   # Install
   pip install dspy-ai
   export OPENAI_API_KEY="..."

   # Verify
   PYTHONPATH=. python verify_dspy_installation.py
   ```

2. **Create Signature**:
   ```python
   sig = create_signature("...", inputs=[...], outputs=[...])
   ```

3. **Test Unoptimized**:
   ```python
   program = dspy.Predict(sig.to_dspy_signature())
   result = program(...)
   ```

4. **Optimize** (when ready):
   ```python
   optimized = await bridge.optimize_from_memory(sig, "examples")
   ```

5. **Create Workflow**:
   ```python
   workflow = adapter.create_workflow(name, description, steps)
   ```

6. **Save**:
   ```python
   await adapter.save_workflow(workflow, path)
   ```

### Production Workflow

1. **Pre-optimize** all workflows
2. **Save** optimized programs
3. **Deploy** with caching enabled
4. **Monitor** statistics
5. **Iterate** based on metrics

## 🔗 External Resources

- **DSPy**: https://dspy-docs.vercel.app/
- **HoloLoom**: `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
- **Promptly**: Archive in `archive/old_projects/Promptly/`

## 📞 Support

**Questions?**
1. Check this index
2. Read relevant documentation
3. Try the demos
4. Check tests for examples
5. Create GitHub issue

---

**Version**: 1.0.0
**Last Updated**: November 7, 2025
**Status**: Production Ready ✅

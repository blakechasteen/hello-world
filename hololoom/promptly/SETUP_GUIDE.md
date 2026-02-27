# DSPy-HoloLoom Setup Guide

**Complete installation and configuration guide for the DSPy-HoloLoom-Promptly integration.**

## 📋 Prerequisites

### System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for LLM API calls

### Required Dependencies

```bash
# Core HoloLoom (should already be installed)
pip install torch numpy gymnasium matplotlib
pip install spacy sentence-transformers scipy networkx

# DSPy integration
pip install dspy-ai

# Optional but recommended
pip install pytest pytest-asyncio  # For running tests
pip install pyyaml  # For YAML workflow support
```

### Language Model Setup

Choose one (or more) LLM providers:

#### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

#### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or in Python:
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

#### Local Models

DSPy supports local models via:
- Ollama
- vLLM
- HuggingFace Transformers

See [DSPy documentation](https://dspy-docs.vercel.app/) for details.

## 🚀 Quick Install

### Option 1: From mythRL Repository

```bash
# Clone repository (if not already done)
git clone https://github.com/yourusername/mythRL.git
cd mythRL

# Install HoloLoom
pip install -e .

# Install DSPy
pip install dspy-ai

# Set API key
export OPENAI_API_KEY="your-key"

# Test installation
PYTHONPATH=. python -c "
from HoloLoom.promptly import DSPyHoloLoom, DSPY_AVAILABLE
print(f'DSPy available: {DSPY_AVAILABLE}')
"
```

### Option 2: Minimal Install

```bash
# Install just DSPy components
pip install dspy-ai pyyaml

# Copy integration files
cp -r HoloLoom/promptly /path/to/your/project/
```

## ✅ Verify Installation

Run the verification script:

```python
# verify_dspy_installation.py

import sys
from pathlib import Path

def verify_installation():
    """Verify DSPy-HoloLoom installation"""

    print("🔍 Verifying DSPy-HoloLoom Installation\n")

    # Check Python version
    print(f"1. Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("   ❌ Python 3.9+ required")
        return False
    print("   ✅ Python version OK\n")

    # Check DSPy
    try:
        import dspy
        print(f"2. DSPy version: {dspy.__version__}")
        print("   ✅ DSPy installed\n")
    except ImportError:
        print("2. ❌ DSPy not installed")
        print("   Install with: pip install dspy-ai\n")
        return False

    # Check HoloLoom
    try:
        from HoloLoom.config import Config
        print("3. ✅ HoloLoom core installed\n")
    except ImportError:
        print("3. ❌ HoloLoom not installed")
        return False

    # Check DSPy integration
    try:
        from HoloLoom.promptly import (
            DSPyHoloLoom,
            DSPyWorkflowAdapter,
            DSPY_AVAILABLE
        )
        if DSPY_AVAILABLE:
            print("4. ✅ DSPy integration available\n")
        else:
            print("4. ⚠️  DSPy integration imported but not available\n")
    except ImportError as e:
        print(f"4. ❌ DSPy integration not available: {e}\n")
        return False

    # Check API keys
    import os
    has_openai = "OPENAI_API_KEY" in os.environ
    has_anthropic = "ANTHROPIC_API_KEY" in os.environ

    print("5. API Keys:")
    if has_openai:
        print("   ✅ OPENAI_API_KEY set")
    if has_anthropic:
        print("   ✅ ANTHROPIC_API_KEY set")
    if not (has_openai or has_anthropic):
        print("   ⚠️  No API keys found (set OPENAI_API_KEY or ANTHROPIC_API_KEY)")
    print()

    # Check example files
    examples_dir = Path("HoloLoom/promptly/examples")
    if examples_dir.exists():
        examples = list(examples_dir.glob("*.yaml"))
        print(f"6. ✅ Found {len(examples)} example workflows\n")
    else:
        print("6. ⚠️  Example workflows not found\n")

    print("✅ Installation verified!\n")
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
```

Run it:

```bash
PYTHONPATH=. python verify_dspy_installation.py
```

## 🎓 First Steps

### 1. Hello World Example

```python
# hello_dspy.py

import asyncio
from HoloLoom.config import Config
from HoloLoom.promptly import DSPyHoloLoom, create_signature
import dspy

async def main():
    # Initialize bridge
    bridge = DSPyHoloLoom(
        config=Config.bare(),  # Use BARE for fast testing
        lm_model="openai/gpt-4o-mini"
    )

    # Create simple signature
    sig = create_signature(
        "Say hello to the user",
        inputs=["name"],
        outputs=["greeting"]
    )

    # Execute
    program = dspy.Predict(sig.to_dspy_signature())
    result = program(name="Alice")

    print(f"Greeting: {result.greeting}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run:
```bash
PYTHONPATH=. python hello_dspy.py
```

### 2. Simple Workflow

```python
# simple_workflow.py

import asyncio
from pathlib import Path
from HoloLoom.config import Config
from HoloLoom.promptly import (
    DSPyHoloLoom,
    DSPyWorkflowAdapter,
    create_qa_workflow
)

async def main():
    # Initialize
    bridge = DSPyHoloLoom(
        config=Config.fast(),
        lm_model="openai/gpt-4o-mini"
    )

    adapter = DSPyWorkflowAdapter(bridge)

    # Create QA workflow
    workflow = await create_qa_workflow(adapter)

    # Execute
    result = await adapter.execute_workflow(
        workflow,
        {"query": "What is 2+2?"}
    )

    # Show results
    if result["success"]:
        print("✅ Workflow succeeded!")
        print(f"Answer: {result['context']['answer.answer']}")
    else:
        print("❌ Workflow failed")
        for step in result["trace"]:
            if not step["success"]:
                print(f"  Failed at: {step['step_id']}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run:
```bash
PYTHONPATH=. python simple_workflow.py
```

## 🔧 Configuration

### HoloLoom Config

Choose execution mode:

```python
from HoloLoom.config import Config

# BARE - Fastest (minimal processing)
config = Config.bare()

# FAST - Balanced (recommended for development)
config = Config.fast()

# FUSED - Full processing (production, best quality)
config = Config.fused()
```

### DSPy LM Configuration

```python
# OpenAI
bridge = DSPyHoloLoom(
    config=config,
    lm_model="openai/gpt-4o-mini",  # or "openai/gpt-4o"
    lm_api_key="sk-..."  # Optional, uses env var if not provided
)

# Anthropic
bridge = DSPyHoloLoom(
    config=config,
    lm_model="anthropic/claude-3-5-sonnet-20241022",
    lm_api_key="sk-ant-..."
)

# Local Ollama
import dspy
lm = dspy.OllamaLocal(model="llama2")
dspy.settings.configure(lm=lm)

bridge = DSPyHoloLoom(config=config)
```

### Optimization Config

```python
from HoloLoom.promptly.dspy_bridge import DSPyOptimizationConfig

# Fast optimization (for development)
opt_config = DSPyOptimizationConfig(
    optimizer="bootstrap",
    max_bootstrapped_demos=2,
    max_labeled_demos=8
)

# Production optimization (better quality)
opt_config = DSPyOptimizationConfig(
    optimizer="mipro",
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    num_threads=4
)

# Use in optimization
optimized = await bridge.optimize_from_memory(
    signature=sig,
    memory_query="training examples",
    optimization_config=opt_config
)
```

## 📁 Project Structure

Recommended structure:

```
my_project/
├── workflows/              # Your workflow definitions
│   ├── qa_workflow.yaml
│   ├── research_workflow.yaml
│   └── custom_workflow.yaml
│
├── signatures/             # Custom signature definitions
│   ├── __init__.py
│   ├── qa_signatures.py
│   └── custom_signatures.py
│
├── optimized_programs/     # Cached optimized programs
│   ├── qa_program.json
│   └── qa_program_metadata.json
│
├── scripts/                # Utility scripts
│   ├── optimize_workflows.py
│   ├── test_workflows.py
│   └── benchmark.py
│
├── main.py                 # Your application entry point
├── config.py               # Configuration
└── requirements.txt        # Dependencies
```

## 🧪 Running Tests

```bash
# Run DSPy integration tests
pytest HoloLoom/tests/integration/test_dspy_integration.py -v

# Run specific test
pytest HoloLoom/tests/integration/test_dspy_integration.py::TestDSPySignature -v

# Run with coverage
pytest HoloLoom/tests/integration/test_dspy_integration.py --cov=HoloLoom.promptly
```

## 🎯 Running Demos

```bash
# Main demo (all 7 demos)
PYTHONPATH=. python demos/demo_dspy_promptly_integration.py

# Individual DSPy components
PYTHONPATH=. python HoloLoom/promptly/dspy_bridge.py
PYTHONPATH=. python HoloLoom/promptly/dspy_workflow_adapter.py
```

## 🐛 Troubleshooting

### Issue: "DSPy not available"

**Solution**:
```bash
pip install dspy-ai
```

Verify:
```python
import dspy
print(dspy.__version__)
```

### Issue: "No module named 'HoloLoom'"

**Solution**:
```bash
# Make sure you're in mythRL directory
cd mythRL

# Set PYTHONPATH
export PYTHONPATH=.

# Or install HoloLoom
pip install -e .
```

### Issue: "OpenAI API key not found"

**Solution**:
```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."

# Or set in Python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### Issue: "No training examples found"

This is expected if HoloLoom memory is empty. Options:

1. **Use unoptimized programs** (works without training data):
   ```python
   import dspy
   program = dspy.Predict(sig.to_dspy_signature())
   ```

2. **Add training data to HoloLoom memory** (for optimization):
   ```python
   from HoloLoom.documentation.types import MemoryShard

   shard = MemoryShard(
       content="Q: What is 2+2? A: 4",
       source="training_data"
   )
   # Add to memory...
   ```

3. **Use external training data**:
   ```python
   # Load from CSV, JSON, etc.
   training_examples = load_examples("training_data.json")
   ```

### Issue: "Workflow execution failed"

**Debug steps**:

1. Check trace:
   ```python
   result = await adapter.execute_workflow(workflow, inputs)

   for step in result["trace"]:
       if not step["success"]:
           print(f"Failed: {step['step_id']}")
           print(f"Error: {step['error']}")
   ```

2. Enable logging:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

3. Test signatures individually:
   ```python
   # Test each signature before workflow
   program = dspy.Predict(sig.to_dspy_signature())
   result = program(inputs...)
   ```

## 📚 Next Steps

1. **Read the documentation**:
   - `README_DSPY_INTEGRATION.md` - Complete guide
   - `DSPY_QUICK_REFERENCE.md` - Quick lookup
   - `ARCHITECTURE.md` - System architecture

2. **Try the examples**:
   - Load example workflows from `HoloLoom/promptly/examples/`
   - Run `demos/demo_dspy_promptly_integration.py`

3. **Build your first workflow**:
   - Start with simple 2-3 step workflow
   - Test without optimization
   - Add optimization when ready

4. **Join the community**:
   - Report issues on GitHub
   - Share your workflows
   - Contribute improvements

## 🔗 Resources

- **DSPy Documentation**: https://dspy-docs.vercel.app/
- **HoloLoom Docs**: `HOLOLOOM_MASTER_SCOPE_AND_SEQUENCE.md`
- **Example Workflows**: `HoloLoom/promptly/examples/`
- **Integration Tests**: `HoloLoom/tests/integration/test_dspy_integration.py`

## 📞 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review the full documentation
3. Check existing GitHub issues
4. Create a new issue with:
   - Python version
   - DSPy version
   - Error message
   - Minimal reproduction code

---

**Last Updated**: November 7, 2025
**Version**: 1.0.0
**Status**: Production Ready ✅

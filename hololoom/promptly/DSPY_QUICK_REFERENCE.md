# DSPy-HoloLoom Quick Reference

**Fast lookup guide for common DSPy integration patterns.**

## 🚀 Installation

```bash
pip install dspy-ai
export OPENAI_API_KEY="your-key"
```

## 📝 Basic Signature

```python
from hololoom.promptly.dspy_bridge import create_signature

sig = create_signature(
    "Answer questions accurately",
    inputs=["question", "context"],
    outputs=["answer", "confidence"]
)
```

## 🔧 Initialize Bridge

```python
from hololoom.promptly.dspy_bridge import DSPyHoloLoom
from hololoom.config import Config

bridge = DSPyHoloLoom(
    config=Config.fused(),
    lm_model="openai/gpt-4o-mini"
)
```

## ▶️ Execute Program

```python
import dspy

program = dspy.Predict(sig.to_dspy_signature())
result = await bridge.execute(
    program,
    question="What is DSPy?",
    context="DSPy is a framework..."
)
```

## 🎯 Optimize from Memory

```python
from hololoom.promptly.dspy_bridge import DSPyOptimizationConfig

optimized = await bridge.optimize_from_memory(
    signature=sig,
    memory_query="Q&A examples",
    optimization_config=DSPyOptimizationConfig(
        optimizer="bootstrap",
        max_bootstrapped_demos=4
    )
)
```

## 🔗 Create Workflow

```python
from hololoom.promptly.dspy_workflow_adapter import DSPyWorkflowAdapter

adapter = DSPyWorkflowAdapter(bridge)

# Register signatures
adapter.register_signature(sig1)
adapter.register_signature(sig2)

# Create workflow
workflow = adapter.create_workflow(
    name="MyWorkflow",
    description="Multi-step pipeline",
    steps=[
        {
            "step_id": "step1",
            "signature": "Sig1",
            "inputs": {"x": "{input}"},
            "outputs": ["y"]
        },
        {
            "step_id": "step2",
            "signature": "Sig2",
            "inputs": {"x": "{step1.y}"},
            "outputs": ["z"]
        }
    ]
)
```

## 🏃 Execute Workflow

```python
result = await adapter.execute_workflow(
    workflow,
    initial_inputs={"input": "test"}
)

# Access outputs
final_output = result["context"]["step2.z"]
```

## 💾 Save/Load Workflow

```python
# Save
await adapter.save_workflow(workflow, Path("workflow.yaml"))

# Load
loaded = await adapter.load_workflow(Path("workflow.yaml"))
```

## 📊 Statistics

```python
stats = adapter.get_execution_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg duration: {stats['avg_duration_ms']:.1f}ms")
```

## 🔍 Debug Trace

```python
result = await adapter.execute_workflow(workflow, inputs)

for step in result["trace"]:
    print(f"{step['step_id']}: {step['success']}")
    print(f"  Duration: {step['duration_ms']}ms")
    if not step["success"]:
        print(f"  Error: {step['error']}")
```

## 📋 Built-in Signatures

```python
from hololoom.promptly.dspy_bridge import (
    QuestionAnswering,
    MemorySynthesis,
    ReasoningChain
)

# Use directly
qa = QuestionAnswering()
dspy_sig = qa.to_dspy_signature()
```

## 🎨 Custom Metric

```python
def my_metric(example, prediction):
    # Compare prediction to ground truth
    if example.answer == prediction.answer:
        return 1.0
    return 0.0

config = DSPyOptimizationConfig(metric=my_metric)
```

## 🔐 With Safety Guardrails

```python
from hololoom.alignment import SafetyGuardrails

guardrails = SafetyGuardrails()
gate = await guardrails.gate_action("dspy_execute", context)

if gate.allowed:
    result = await bridge.execute(program, **inputs)
```

## 📝 YAML Workflow

```yaml
name: MyWorkflow
description: Example workflow

steps:
  - step_id: step1
    signature: Sig1
    inputs:
      x: "{input}"
    outputs:
      - y
    optimize: true
    optimization_query: "examples for step1"

  - step_id: step2
    signature: Sig2
    inputs:
      x: "{step1.y}"
    outputs:
      - z
```

## 🐛 Common Issues

**DSPy not installed:**
```bash
pip install dspy-ai
```

**No training examples:**
```python
# Check HoloLoom memory
orchestrator = await bridge._get_orchestrator()
spacetime = await orchestrator.weave(Query(text="search query"))
print(f"Found {len(spacetime.context)} memories")
```

**Workflow step failed:**
```python
# Check trace for errors
for step in result["trace"]:
    if not step["success"]:
        print(f"Failed: {step['step_id']}: {step['error']}")
```

## 🎯 Optimization Strategies

| Strategy | Best For | Speed | Quality |
|----------|----------|-------|---------|
| Bootstrap | Few-shot | Fast | Good |
| MIPRO | Instruction tuning | Medium | Better |
| COPRO | Prompt search | Slow | Best |

## 📚 Learn More

- Full docs: `README_DSPY_INTEGRATION.md`
- Examples: `demos/demo_dspy_promptly_integration.py`
- Workflows: `hololoom/promptly/examples/*.yaml`
- Tests: `hololoom/tests/integration/test_dspy_integration.py`

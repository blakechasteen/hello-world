# DSPy Workflow Examples

This directory contains example DSPy workflows for HoloLoom integration.

## Available Workflows

### 1. Question Answering (`qa_workflow.yaml`)

**Purpose**: Answer questions with retrieval and verification

**Steps**:
1. **Retrieve** - Get relevant context from HoloLoom memory
2. **Answer** - Generate answer using context
3. **Verify** - Verify answer accuracy

**Usage**:
```python
from HoloLoom.promptly import DSPyWorkflowAdapter, DSPyHoloLoom
from pathlib import Path

bridge = DSPyHoloLoom(config=Config.fused(), lm_model="openai/gpt-4o-mini")
adapter = DSPyWorkflowAdapter(bridge)

workflow = await adapter.load_workflow(Path("qa_workflow.yaml"))
result = await adapter.execute_workflow(
    workflow,
    {"query": "What is Thompson Sampling?"}
)

print(result["context"]["answer.answer"])
print(f"Accurate: {result['context']['verify.is_accurate']}")
```

**Expected Runtime**: ~20 seconds (with optimization)

---

### 2. Research Pipeline (`research_workflow.yaml`)

**Purpose**: Multi-query research with synthesis

**Steps**:
1. **Decompose** - Break complex question into sub-questions
2. **Retrieve Context** - Get context for each sub-question
3. **Answer Sub-questions** - Answer each independently
4. **Synthesize** - Combine answers into comprehensive response

**Usage**:
```python
workflow = await adapter.load_workflow(Path("research_workflow.yaml"))
result = await adapter.execute_workflow(
    workflow,
    {"query": "What are the tradeoffs between Thompson Sampling and UCB?"}
)

print(result["context"]["synthesize.synthesis"])
print(f"Sources: {result['context']['synthesize.sources']}")
```

**Expected Runtime**: ~30 seconds (with parallel execution)

---

### 3. Code Review (`code_review_workflow.yaml`)

**Purpose**: Automated code review with multi-aspect analysis

**Steps**:
1. **Parse Code** - Extract AST and complexity
2. **Security Analysis** - Identify vulnerabilities
3. **Style Check** - Check formatting and style
4. **Best Practices** - Check adherence to patterns
5. **Generate Report** - Create comprehensive review

**Usage**:
```python
workflow = await adapter.load_workflow(Path("code_review_workflow.yaml"))

code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
"""

result = await adapter.execute_workflow(
    workflow,
    {
        "code_snippet": code,
        "language": "python"
    }
)

print(result["context"]["generate_report.summary"])
print(result["context"]["security_analysis.vulnerabilities"])
print(result["context"]["best_practices.recommendations"])
```

**Expected Runtime**: ~25 seconds

---

## Customizing Workflows

### Modify Existing Workflow

1. Copy workflow YAML file
2. Edit steps, signatures, or inputs
3. Load and execute

```python
# Load
workflow = await adapter.load_workflow(Path("my_custom_qa.yaml"))

# Modify if needed (programmatically)
workflow.steps[0].optimize = False

# Execute
result = await adapter.execute_workflow(workflow, inputs)
```

### Create New Workflow

```python
# Register signatures
adapter.register_signature(my_sig1)
adapter.register_signature(my_sig2)

# Create workflow
workflow = adapter.create_workflow(
    name="MyWorkflow",
    description="Custom pipeline",
    steps=[
        {
            "step_id": "step1",
            "signature": "MySig1",
            "inputs": {"x": "{input}"},
            "outputs": ["y"]
        },
        {
            "step_id": "step2",
            "signature": "MySig2",
            "inputs": {"x": "{step1.y}"},
            "outputs": ["z"]
        }
    ]
)

# Save
await adapter.save_workflow(workflow, Path("my_workflow.yaml"))
```

## Workflow YAML Format

```yaml
name: WorkflowName
description: Human-readable description

steps:
  - step_id: unique_step_id
    signature: SignatureName
    inputs:
      field1: "{previous_step.output}"  # Reference
      field2: "literal value"            # Literal
    outputs:
      - output1
      - output2
    optimize: true                       # Enable optimization
    optimization_query: "training examples query"
    metadata:
      description: "Step description"
      timeout: 10000                     # Timeout in ms
      parallel: false                    # Parallel execution

metadata:
  version: "1.0.0"
  author: "Your Name"
  category: "workflow-category"
  tags:
    - tag1
    - tag2
  expected_runtime_ms: 20000
  complexity: "medium"
```

## Pre-Optimization

Pre-optimize workflows for faster execution:

```python
# Load workflow
workflow = await adapter.load_workflow(Path("qa_workflow.yaml"))

# Pre-optimize all steps
await adapter.optimize_workflow(workflow)

# Now executions will be fast
result = await adapter.execute_workflow(workflow, inputs)
```

## Monitoring

Track workflow performance:

```python
# Execute multiple times
for query in queries:
    await adapter.execute_workflow(workflow, {"query": query})

# Get statistics
stats = adapter.get_execution_statistics()

print(f"Total executions: {stats['total_executions']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg duration: {stats['avg_duration_ms']:.1f}ms")
```

## Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Execute workflow
result = await adapter.execute_workflow(workflow, inputs)

# Check trace
for step in result["trace"]:
    status = "✓" if step["success"] else "✗"
    print(f"{status} {step['step_id']}: {step['duration_ms']:.1f}ms")

    if not step["success"]:
        print(f"   Error: {step['error']}")
```

## Best Practices

1. **Start Simple** - Test signatures individually before workflows
2. **Optimize Selectively** - Not all steps need optimization
3. **Monitor Performance** - Track success rates and durations
4. **Version Workflows** - Use metadata.version for tracking
5. **Document Steps** - Use metadata.description for clarity
6. **Handle Errors** - Check result["success"] before accessing outputs

## Learn More

- Full documentation: `../README_DSPY_INTEGRATION.md`
- Quick reference: `../DSPY_QUICK_REFERENCE.md`
- Demo: `../../../demos/demo_dspy_promptly_integration.py`
- Tests: `../../../HoloLoom/tests/integration/test_dspy_integration.py`

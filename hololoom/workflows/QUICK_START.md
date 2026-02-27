# Agentic Workflow System - Quick Start Guide

**5-Minute Quick Start** - Get up and running with agentic workflows

## Installation

No extra dependencies needed! All dependencies already part of HoloLoom.

```bash
# Optional: Install Redis for distributed state (production)
pip install redis
```

## 1. Your First Workflow (60 seconds)

### Using Templates

```python
import asyncio
from hololoom.workflows import WorkflowExecutor, WorkflowTemplates

async def main():
    # Load pre-built template
    workflow = WorkflowTemplates.simple_qa()

    # Execute workflow
    async with WorkflowExecutor(workflow) as executor:
        result = await executor.execute({"query": "What is Thompson Sampling?"})

        # Print result
        print(result.summary())

asyncio.run(main())
```

**Output**:
```
Execution ID: abc123...
Workflow: Simple Q&A
Status: success
Nodes Executed: 1
Total Time: 156.3ms
```

## 2. Explore All Templates (90 seconds)

```python
from hololoom.workflows import WorkflowTemplates

# List all templates
templates = WorkflowTemplates.list_templates()
print(f"Available templates: {', '.join(templates)}")

# Get all templates
all_templates = WorkflowTemplates.get_all_templates()
for name, workflow in all_templates.items():
    print(f"\n{name}:")
    print(f"  Description: {workflow.metadata['description']}")
    print(f"  Nodes: {len(workflow.nodes)}")
    print(f"  Entry Point: {workflow.entry_point}")
```

**Output**:
```
Available templates: simple_qa, verified_qa, auto_refining_qa, recursive_research, multi_strategy, human_in_loop, complex_decomposition, iterative_refinement, safety_gated

simple_qa:
  Description: Basic question answering workflow
  Nodes: 1
  Entry Point: query

verified_qa:
  Description: Q&A with DS-STAR verification
  Nodes: 2
  Entry Point: query

...
```

## 3. Create Custom Workflow (2 minutes)

### From Python

```python
from hololoom.workflows import WorkflowNode, WorkflowDefinition, NodeType

# Create nodes
query_node = WorkflowNode(
    id="query",
    type=NodeType.QUERY,
    name="RAG Query",
    params={"mode": "verify", "max_sources": 10},
    next=["verify"]
)

verify_node = WorkflowNode(
    id="verify",
    type=NodeType.VERIFY,
    name="DS-STAR Verification",
    params={}
)

# Create workflow
workflow = WorkflowDefinition(
    name="My Custom Workflow",
    version="1.0.0",
    nodes={"query": query_node, "verify": verify_node},
    entry_point="query"
)

# Export to JSON
with open("my_workflow.json", "w") as f:
    f.write(workflow.to_json())
```

### From JSON

```json
{
  "name": "My Custom Workflow",
  "version": "1.0.0",
  "entry_point": "query",
  "nodes": {
    "query": {
      "id": "query",
      "type": "query",
      "name": "RAG Query",
      "params": {
        "mode": "verify",
        "max_sources": 10
      },
      "next": ["verify"]
    },
    "verify": {
      "id": "verify",
      "type": "verify",
      "name": "DS-STAR Verification",
      "params": {}
    }
  }
}
```

Load and execute:

```python
from hololoom.workflows import WorkflowDefinition, WorkflowExecutor

with open("my_workflow.json") as f:
    workflow = WorkflowDefinition.from_json(f.read())

async with WorkflowExecutor(workflow) as executor:
    result = await executor.execute({"query": "Custom query"})
```

## 4. Visual Workflow Builder (1 minute)

```bash
# Start backend
cd hololoom/web_dashboard
python workflow_executor.py

# Open in browser
# http://localhost:8001
# Then open workflow_builder.html
```

**Features**:
- Drag and drop agents from palette
- Connect nodes visually
- Configure node parameters
- Export to JSON/YAML
- Execute workflow
- View real-time execution status

## 5. Add State Persistence (1 minute)

### SQLite (Single-Node Production)

```python
from hololoom.workflows import WorkflowExecutor
from hololoom.workflows.state import SQLiteState

# Create persistent state backend
state = SQLiteState(db_path="./workflows.db")

# Execute with persistence
async with WorkflowExecutor(workflow, state_backend=state) as executor:
    result = await executor.execute(inputs)

# State persists across restarts
```

### Redis (Distributed Production)

```python
from hololoom.workflows.state import RedisState

# Connect to Redis
state = RedisState(redis_url="redis://localhost:6379")

# Execute with distributed state
async with WorkflowExecutor(workflow, state_backend=state) as executor:
    result = await executor.execute(inputs)

# State shared across multiple nodes
```

## 6. Enable Checkpointing (1 minute)

```python
from hololoom.workflows import WorkflowExecutor, CheckpointManager
from hololoom.workflows.state import SQLiteState

# Configure checkpointing
state = SQLiteState()
executor = WorkflowExecutor(
    workflow,
    state_backend=state,
    checkpoint_frequency=5  # Save every 5 nodes
)

try:
    result = await executor.execute(inputs)
except Exception:
    # Resume from last checkpoint
    checkpoints = await executor.checkpoint_manager.list_checkpoints(executor.execution_id)
    result = await executor.resume_from_checkpoint(checkpoints[-1])
```

## 7. Parallel Execution (1 minute)

```python
# Enable parallel execution
async with WorkflowExecutor(
    workflow,
    enable_parallel=True,
    max_concurrent=5  # Run up to 5 nodes concurrently
) as executor:
    result = await executor.execute(inputs)
```

**Note**: Only PARALLEL nodes execute in parallel. Sequential nodes still execute in order.

## Common Patterns

### Pattern 1: Simple Q&A

```python
workflow = WorkflowTemplates.simple_qa()
result = await WorkflowExecutor(workflow).execute({"query": "Question?"})
```

### Pattern 2: Verified Q&A

```python
workflow = WorkflowTemplates.verified_qa()
result = await WorkflowExecutor(workflow).execute({"query": "Question?"})

# Check verification
verify_result = result.outputs["verify"]
print(f"Verified: {verify_result['verified']}")
```

### Pattern 3: Auto-Refining

```python
workflow = WorkflowTemplates.auto_refining_qa()
result = await WorkflowExecutor(workflow).execute({"query": "Question?"})

# Check if refinement occurred
if "refine" in result.outputs:
    print("Response was refined")
```

### Pattern 4: Parallel Strategies

```python
workflow = WorkflowTemplates.multi_strategy()
result = await WorkflowExecutor(workflow, enable_parallel=True).execute({"query": "Question?"})

# Compare strategies
results = result.outputs["parallel"]["results"]
for i, r in enumerate(results):
    print(f"Strategy {i+1}: confidence={r['confidence']:.2f}")
```

## Next Steps

1. **Read Full Documentation** - `hololoom/workflows/README.md`
2. **Explore Templates** - Try all 9 pre-built templates
3. **Visual Builder** - Create workflows visually
4. **Custom Workflows** - Build domain-specific workflows
5. **Production Deployment** - Deploy with SQLite or Redis

## Troubleshooting

### Workflow validation fails

```python
# Check validation errors
errors = workflow.validate()
if errors:
    print(f"Validation failed: {errors}")
```

Common issues:
- Missing entry_point
- Undefined node references
- Cycles (infinite loops)
- Unreachable nodes

### Execution fails

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Execute with error handling
try:
    result = await executor.execute(inputs)
except Exception as e:
    print(f"Execution failed: {e}")
    # Check trace for failed node
    for trace in executor.trace:
        if trace.status == "error":
            print(f"Failed node: {trace.node_id}")
            print(f"Error: {trace.error}")
```

### Checkpoint not found

```python
# List available checkpoints
checkpoints = await executor.checkpoint_manager.list_checkpoints(execution_id)
print(f"Available: {checkpoints}")
```

Common issues:
- checkpoint_frequency = 0 (disabled)
- Using InMemoryState (no persistence)
- Wrong execution_id

## API Reference

### WorkflowTemplates

```python
# Get all templates
templates = WorkflowTemplates.get_all_templates()

# List template names
names = WorkflowTemplates.list_templates()

# Get specific template
workflow = WorkflowTemplates.simple_qa()
workflow = WorkflowTemplates.verified_qa()
workflow = WorkflowTemplates.auto_refining_qa()
workflow = WorkflowTemplates.recursive_research()
workflow = WorkflowTemplates.multi_strategy()
workflow = WorkflowTemplates.human_in_loop()
workflow = WorkflowTemplates.complex_decomposition()
workflow = WorkflowTemplates.iterative_refinement()
workflow = WorkflowTemplates.safety_gated()
```

### WorkflowDefinition

```python
# Load from JSON
workflow = WorkflowDefinition.from_json(json_str)

# Load from YAML
workflow = WorkflowDefinition.from_yaml(yaml_str)

# Load from dict
workflow = WorkflowDefinition.from_dict(data)

# Export to JSON
json_str = workflow.to_json(indent=2)

# Export to YAML
yaml_str = workflow.to_yaml()

# Validate
errors = workflow.validate()
```

### WorkflowExecutor

```python
# Create executor
executor = WorkflowExecutor(
    workflow=workflow,
    config=Config.fast(),
    state_backend=SQLiteState(),
    enable_parallel=True,
    max_concurrent=5,
    checkpoint_frequency=10
)

# Execute
result = await executor.execute({"query": "..."})

# Resume from checkpoint
result = await executor.resume_from_checkpoint(checkpoint_id)

# Access state
state = executor.state
executed = executor.executed_nodes
trace = executor.trace
errors = executor.errors
```

### WorkflowResult

```python
# Access result
result.execution_id       # Unique ID
result.workflow_name      # Workflow name
result.status            # "success", "failed", "partial"
result.outputs           # Dict of node outputs
result.execution_time_ms # Total time
result.nodes_executed    # Number of nodes executed
result.errors            # List of errors
result.trace             # List of ExecutionTrace

# Generate summary
summary = result.summary()

# Export to dict
data = result.to_dict()
```

## Examples

See `demos/demo_agentic_workflows.py` for complete examples (when created).

## Support

- **Documentation**: `hololoom/workflows/README.md` (complete guide)
- **Quick Start**: This file
- **API Reference**: Inline code documentation
- **Examples**: Demo scripts (TODO)

---

**Get Started in 5 Minutes!** 🚀

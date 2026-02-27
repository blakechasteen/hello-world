# Your First Workflow

This tutorial walks you through creating a simple query workflow from scratch.

## What You'll Build

A basic workflow that:
1. Accepts a user query
2. Processes it through HoloLoom
3. Generates a response

```
[HoloLoom Query] → [Response Generator]
```

**Estimated time**: 5 minutes

## Step 1: Open the Workflow Builder

1. Ensure the backend is running (see [Installation](installation.md))
2. Open `workflow_builder.html` in your browser
3. You should see the empty canvas with the agent palette on the left

## Step 2: Add a Query Node

1. **Find the Agent Palette** on the left side
2. **Locate "HoloLoom Query"** under the "Query" category
3. **Drag it onto the canvas**

You should now see a blue node labeled "HoloLoom Query" on the canvas.

### Configure the Node

1. **Click the node** to select it
2. **Open the Properties Panel** (right side or bottom on mobile)
3. Set the configuration:
   - **Label**: "Main Query"
   - **Query Template**: `${input.query}`
   - **Complexity Mode**: "fast" (default)

## Step 3: Add a Response Generator

1. **Drag "Response Generator"** from the "Output" category onto the canvas
2. **Position it to the right** of the Query node

### Configure the Response Generator

1. **Click the node** to select it
2. Set the configuration:
   - **Label**: "Generate Response"
   - **Format**: "text" (default)

## Step 4: Connect the Nodes

1. **Hover over the Query node** - you'll see connection ports appear
2. **Click and drag from the output port** (right side of Query node)
3. **Drop on the input port** (left side of Response Generator)

A connection line should now link the two nodes.

### Understanding Ports

| Port Type | Location | Purpose |
|-----------|----------|---------|
| Input | Left side | Receives data from previous node |
| Output | Right side | Sends data to next node |

## Step 5: Execute the Workflow

### Method 1: Toolbar Button

1. Click the **"Execute"** button in the toolbar
2. Enter your query in the dialog: "What is Thompson Sampling?"
3. Click **"Run"**

### Method 2: Keyboard Shortcut

1. Press `Ctrl+Enter`
2. Enter your query
3. Press Enter to execute

### Method 3: API Call

```bash
curl -X POST http://localhost:8001/api/workflow/execute \
  -H "Content-Type: application/json" \
  -d '{
    "workflow": {
      "version": "1.0",
      "name": "My First Workflow",
      "nodes": [
        {
          "id": "query-1",
          "type": "hololoom_query",
          "x": 100, "y": 100,
          "config": {"query_template": "${input.query}"}
        },
        {
          "id": "response-1",
          "type": "response_generator",
          "x": 400, "y": 100,
          "config": {"format": "text"}
        }
      ],
      "connections": [
        {"source": "query-1", "target": "response-1", "sourcePort": "output", "targetPort": "input"}
      ]
    },
    "input_data": {"query": "What is Thompson Sampling?"}
  }'
```

## Step 6: View Results

During execution:
- **Nodes turn blue** when processing
- **Nodes turn green** when complete
- **Nodes turn red** if there's an error

After execution:
1. Click the **Response Generator** node
2. View the output in the **Properties Panel**
3. The response should explain Thompson Sampling

## Step 7: Save Your Workflow

### Export as JSON

1. Press `Ctrl+S` or click **"Export"** in the toolbar
2. Choose **JSON** format
3. Save the file

### Export as Python

1. Click **"Export"** → **"Python"**
2. Get executable Python code:

```python
from hololoom import hololoom
from hololoom.config import Config

async def run_workflow(query: str):
    config = Config.fast()
    async with HoloLoom(config=config) as loom:
        # Step 1: Query
        result = await loom.weave(query)

        # Step 2: Generate Response
        return result.response

# Run
import asyncio
response = asyncio.run(run_workflow("What is Thompson Sampling?"))
print(response)
```

## Enhancing Your Workflow

### Add Verification

Add a **Recursive Refiner** between Query and Response:

```
[HoloLoom Query] → [Recursive Refiner] → [Response Generator]
```

This improves response quality by iterating until confidence > 0.9.

### Add Memory Storage

Add a **Memory Store** after Response:

```
[HoloLoom Query] → [Response Generator] → [Memory Store]
```

This persists the Q&A pair to the knowledge graph.

### Add Safety Checks

Add **Safety Guardrails** before Response:

```
[HoloLoom Query] → [Safety Guardrails] → [Response Generator]
```

This gates responses based on risk assessment.

## Common Mistakes

### Disconnected Nodes

**Problem**: Node isn't executing
**Solution**: Ensure all nodes are connected (check connection lines)

### Missing Configuration

**Problem**: "Missing required field" error
**Solution**: Open Properties Panel and fill required fields

### Wrong Port Types

**Problem**: Can't connect nodes
**Solution**: Connect output → input (not input → input)

### Circular Dependencies

**Problem**: "Cycle detected" error
**Solution**: Workflows must be directed acyclic graphs (DAGs)

## Next Steps

- [UI Overview](ui-overview.md) - Learn all interface components
- [Agent Types](../features/nodes.md) - Explore all 18 agents
- [Build a RAG Pipeline](../tutorials/rag-pipeline.md) - More advanced workflow

## Quick Reference

| Action | Method |
|--------|--------|
| Add node | Drag from palette |
| Select node | Click |
| Delete node | Select + Delete key |
| Connect | Drag port to port |
| Execute | Ctrl+Enter |
| Save | Ctrl+S |
| Undo | Ctrl+Z |
| Redo | Ctrl+Shift+Z |

---

← [Installation](installation.md) | [UI Overview](ui-overview.md) →

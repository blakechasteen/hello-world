# Nested Workflows

Create reusable workflow components with composite nodes that encapsulate entire sub-workflows.

## Overview

Nested workflows allow you to:
- **Encapsulate complexity**: Hide implementation details behind a single node
- **Promote reuse**: Use the same sub-workflow in multiple places
- **Organize large workflows**: Break complex pipelines into manageable pieces
- **Enable collaboration**: Teams can work on different workflow components

## Composite Nodes

### What is a Composite Node?

A composite node is a special node type that contains an entire workflow inside it. From the outside, it looks like a single node with defined inputs and outputs. Inside, it contains multiple nodes and connections.

```
┌─────────────────────────────────────────────┐
│  Composite: Research Pipeline               │
│  ┌─────────────────────────────────────┐   │
│  │  ┌─────────┐    ┌─────────┐         │   │
│  │  │ Query   │───▶│ Verify  │───▶ out │   │
│  │  └─────────┘    └─────────┘         │   │
│  │       ▲                              │   │
│  │  in ──┘                              │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

### Creating a Composite Node

**Method 1: From Selection**

1. Select multiple nodes (Shift+Click or drag box)
2. Right-click → **"Create Composite"**
3. Name the composite and define ports
4. Selected nodes become the composite's contents

**Method 2: From Scratch**

1. Drag **"Composite Workflow"** from Control Flow palette
2. Double-click to enter edit mode
3. Build the internal workflow
4. Define input/output ports
5. Click **"Exit Composite"** or press `Escape`

### Composite Node Properties

```
┌─────────────────────────────────────────┐
│ Composite Properties                  ×  │
├─────────────────────────────────────────┤
│ Name: [Research Pipeline          ]     │
│ Description: [Multi-query research...]  │
│                                         │
│ ▼ Input Ports                           │
│   ┌─────────────────────────────────┐   │
│   │ Name: query                     │   │
│   │ Type: string                    │   │
│   │ Required: ☑                     │   │
│   └─────────────────────────────────┘   │
│   [+ Add Input Port]                    │
│                                         │
│ ▼ Output Ports                          │
│   ┌─────────────────────────────────┐   │
│   │ Name: result                    │   │
│   │ Type: object                    │   │
│   └─────────────────────────────────┘   │
│   [+ Add Output Port]                   │
│                                         │
│ [Save] [Cancel]                         │
└─────────────────────────────────────────┘
```

## Drill-Down Navigation

### Entering a Composite

**Double-click** on any composite node to enter its internal workflow.

Visual indicators when inside a composite:
- Breadcrumb trail shows path: `Root > Research Pipeline > Sub-Process`
- Canvas background color changes slightly
- **"Exit"** button appears in toolbar

### Navigation Breadcrumbs

```
┌─────────────────────────────────────────────────────────┐
│ 📂 Root  ▶  📦 Research Pipeline  ▶  📦 Verify Loop    │
└─────────────────────────────────────────────────────────┘
```

Click any breadcrumb segment to navigate directly to that level.

### Keyboard Navigation

| Key | Action |
|-----|--------|
| `Enter` | Enter selected composite |
| `Escape` | Exit to parent level |
| `Backspace` | Navigate up one level |

## Port Mapping

### Input Port Mapping

Map external inputs to internal nodes:

```javascript
// Composite input port "query" maps to internal node input
{
  "port_mappings": {
    "inputs": {
      "query": {
        "target_node": "query-1",
        "target_port": "input"
      }
    }
  }
}
```

### Output Port Mapping

Map internal node outputs to composite outputs:

```javascript
{
  "port_mappings": {
    "outputs": {
      "result": {
        "source_node": "response-1",
        "source_port": "output"
      }
    }
  }
}
```

### Multi-Input/Multi-Output

Composites can have multiple ports:

```
        ┌──────────────────────┐
query ──┤                      ├── response
context ┤  Research Pipeline   ├── sources
options ┤                      ├── confidence
        └──────────────────────┘
```

## Data Flow

### Input Propagation

When a composite receives input:
1. Input appears at mapped internal node
2. Internal workflow executes
3. Output collected from mapped internal node
4. Output sent to connected downstream nodes

### Execution Isolation

Each composite instance maintains its own:
- Execution state
- Variable scope
- Error context

Variables inside a composite don't leak to the parent or siblings.

## Recursive Composites

### Self-Referencing Workflows

A composite can contain instances of itself for recursive processing:

```
┌─────────────────────────────────────────┐
│  Recursive: Tree Processor              │
│  ┌─────────────────────────────────┐   │
│  │  [Process Node]                  │   │
│  │       │                          │   │
│  │       ▼                          │   │
│  │  [Branch?]──yes──▶[Tree Processor]│  │
│  │       │                          │   │
│  │      no                          │   │
│  │       ▼                          │   │
│  │    [Output]                      │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

### Recursion Safety

Built-in protections:
- **Max depth**: Default 10 levels (configurable)
- **Cycle detection**: Prevents infinite loops
- **Stack tracking**: Shows recursion path in debugger

Configure recursion limits:
```javascript
{
  "composite_settings": {
    "max_recursion_depth": 10,
    "detect_cycles": true,
    "timeout_per_level_ms": 5000
  }
}
```

## Library Management

### Saving to Library

1. Right-click composite → **"Save to Library"**
2. Fill in metadata:
   - Name and description
   - Category and tags
   - Version number
3. Click **"Save"**

### Library Panel

Access saved composites:

```
┌─────────────────────────────────┐
│ Composite Library            ×  │
├─────────────────────────────────┤
│ Search: [________________]      │
│                                 │
│ ▼ My Composites                 │
│   📦 Research Pipeline    v1.2  │
│   📦 Verification Loop    v1.0  │
│   📦 Memory Processor     v2.1  │
│                                 │
│ ▼ Team Shared                   │
│   📦 Auth Flow            v3.0  │
│   📦 Data Transform       v1.5  │
│                                 │
│ ▼ Community                     │
│   📦 RAG Standard         v2.0  │
│   📦 Agent Chain          v1.1  │
└─────────────────────────────────┘
```

### Using Library Composites

1. Open Library Panel
2. Drag composite onto canvas
3. Or: Right-click canvas → **"Insert from Library"**

### Versioning

Track composite versions:
- **v1.0**: Initial release
- **v1.1**: Bug fixes
- **v2.0**: Breaking changes

Workflows using older versions show update indicator:

```
┌─────────────────────────────────────────┐
│ 📦 Research Pipeline  ⚠️ Update (v1.2)   │
│     └── Currently using v1.0            │
│         [Update] [Keep Current]         │
└─────────────────────────────────────────┘
```

## Best Practices

### Design Guidelines

1. **Single Responsibility**: Each composite should do one thing well
2. **Clear Interfaces**: Well-named ports with documentation
3. **Reasonable Size**: 3-10 internal nodes typically
4. **Test Independently**: Verify composite works before using

### Naming Conventions

| Pattern | Example |
|---------|---------|
| `{action}-{target}` | `verify-response` |
| `{domain}-{operation}` | `memory-consolidate` |
| `{purpose}-pipeline` | `research-pipeline` |

### Documentation

Document your composites:

```javascript
{
  "metadata": {
    "name": "Research Pipeline",
    "description": "Multi-query research with verification",
    "author": "Your Name",
    "version": "1.2.0",
    "inputs": {
      "query": "The research question to explore"
    },
    "outputs": {
      "result": "Verified research findings"
    },
    "examples": [
      {
        "input": {"query": "What is Thompson Sampling?"},
        "output": {"result": "Thompson Sampling is..."}
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

**Composite won't execute**
- Check all required input ports are connected
- Verify internal workflow has no validation errors
- Ensure output port is mapped to an internal node

**Data not flowing through**
- Confirm port mappings are correct
- Check data types match between ports
- Use debugger to trace data flow

**Recursion depth exceeded**
- Add base case condition
- Increase `max_recursion_depth` if appropriate
- Check for unintended recursive loops

### Debugging Composites

1. Enter the composite (double-click)
2. Set breakpoints inside
3. Run workflow from parent
4. Debugger stops at internal breakpoints
5. Inspect variables at each level

---

← [Export Formats](../features/export-formats.md) | [Custom Agents](custom-agents.md) →

# Export Formats

Save and share workflows in multiple formats: JSON, Python, and YAML.

## Overview

The Workflow Builder supports exporting workflows in three formats:
- **JSON**: Complete workflow definition for import/export
- **Python**: Executable Python code using HoloLoom API
- **YAML**: Human-readable format for configuration

## Quick Export

### From Toolbar

1. Click **Export** button (or press `Ctrl+S`)
2. Select format from dropdown
3. Choose save location
4. Click **Save**

### From Menu

**File** → **Export As** → Select format

### Keyboard Shortcuts

| Format | Shortcut |
|--------|----------|
| JSON | `Ctrl+Shift+J` |
| Python | `Ctrl+Shift+P` |
| YAML | `Ctrl+Shift+Y` |

## JSON Format

The native workflow format with complete fidelity.

### Structure

```json
{
  "version": "1.0",
  "name": "My Research Pipeline",
  "description": "Multi-query research with verification",
  "created": "2025-12-15T10:30:00Z",
  "modified": "2025-12-15T14:45:00Z",
  "author": "User Name",
  "nodes": [
    {
      "id": "query-1",
      "type": "hololoom_query",
      "label": "Main Query",
      "x": 100,
      "y": 200,
      "config": {
        "query_template": "${input.query}",
        "complexity": "fast",
        "max_retries": 3,
        "timeout": 30,
        "enable_cache": true
      }
    },
    {
      "id": "response-1",
      "type": "response_generator",
      "label": "Generate Response",
      "x": 400,
      "y": 200,
      "config": {
        "format": "markdown",
        "style": "technical",
        "max_length": 500
      }
    }
  ],
  "connections": [
    {
      "id": "conn-1",
      "source": "query-1",
      "target": "response-1",
      "sourcePort": "output",
      "targetPort": "input"
    }
  ],
  "metadata": {
    "canvas": {
      "zoom": 1.0,
      "panX": 0,
      "panY": 0
    },
    "groups": [],
    "tags": ["research", "rag"]
  }
}
```

### Field Reference

**Root Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Format version |
| `name` | string | Workflow name |
| `description` | string | Workflow description |
| `created` | ISO date | Creation timestamp |
| `modified` | ISO date | Last modification |
| `author` | string | Creator name |
| `nodes` | array | Node definitions |
| `connections` | array | Connection definitions |
| `metadata` | object | Additional metadata |

**Node Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `type` | string | Agent type |
| `label` | string | Display name |
| `x`, `y` | number | Canvas position |
| `config` | object | Type-specific configuration |

**Connection Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `source` | string | Source node ID |
| `target` | string | Target node ID |
| `sourcePort` | string | Output port name |
| `targetPort` | string | Input port name |

### Use Cases

- **Backup**: Full workflow preservation
- **Sharing**: Import/export between users
- **Version control**: Store in Git
- **API integration**: Programmatic workflow management

### Import JSON

1. **File** → **Import** → Select JSON file
2. Or drag JSON file onto canvas
3. Or: `Ctrl+O` → Select file

## Python Format

Generates executable Python code using the HoloLoom API.

### Example Output

```python
"""
Workflow: My Research Pipeline
Description: Multi-query research with verification
Generated: 2025-12-15T14:45:00Z
"""

import asyncio
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.protocols.types import Query


async def run_workflow(input_data: dict) -> dict:
    """Execute the research pipeline workflow."""

    # Initialize orchestrator
    config = Config.fast()

    async with WeavingOrchestrator(cfg=config) as orchestrator:

        # Node: Main Query (hololoom_query)
        query_1_result = await orchestrator.weave(
            Query(text=input_data.get('query', '')),
            complexity='fast',
            max_retries=3,
            timeout=30,
            enable_cache=True
        )

        # Node: Generate Response (response_generator)
        response_1_result = {
            'response': query_1_result.response,
            'confidence': query_1_result.confidence,
            'sources': query_1_result.metadata.get('sources', [])
        }

        return response_1_result


async def main():
    """Main entry point."""
    input_data = {
        'query': 'What is Thompson Sampling?'
    }

    result = await run_workflow(input_data)
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']:.2f}")


if __name__ == '__main__':
    asyncio.run(main())
```

### Code Generation Options

Configure Python export settings:

```
┌─────────────────────────────────────────┐
│ Python Export Options                ×  │
├─────────────────────────────────────────┤
│ ☑ Include docstrings                    │
│ ☑ Include type hints                    │
│ ☑ Include main() function               │
│ ☑ Include error handling                │
│ ☐ Include logging                       │
│ ☐ Include progress callbacks            │
│                                         │
│ Async style: [asyncio           ▼]      │
│ Indent style: [4 spaces         ▼]      │
│                                         │
│ [Generate] [Copy to Clipboard]          │
└─────────────────────────────────────────┘
```

### Generated Features

**Control Flow Translation**:

| Visual Node | Python Code |
|-------------|-------------|
| Conditional Branch | `if condition:` |
| Loop Iterator | `for item in items:` |
| Parallel Executor | `asyncio.gather()` |

**Example with Control Flow**:

```python
async def run_workflow(input_data: dict) -> dict:

    # ... query execution ...

    # Node: Confidence Check (conditional_branch)
    if query_1_result.confidence > 0.8:
        # High confidence path
        response = format_response(query_1_result)
    else:
        # Low confidence path - refine
        refined_result = await refiner.refine(
            query_1_result,
            strategy='verify',
            max_iterations=3
        )
        response = format_response(refined_result)

    return response
```

### Use Cases

- **Production deployment**: Run workflows as scripts
- **Customization**: Modify generated code
- **Integration**: Embed in larger applications
- **Learning**: Understand HoloLoom API usage

## YAML Format

Human-readable format ideal for configuration and version control.

### Example Output

```yaml
# Workflow: My Research Pipeline
# Description: Multi-query research with verification
# Generated: 2025-12-15T14:45:00Z

version: "1.0"
name: My Research Pipeline
description: Multi-query research with verification
author: User Name

nodes:
  - id: query-1
    type: hololoom_query
    label: Main Query
    position:
      x: 100
      y: 200
    config:
      query_template: "${input.query}"
      complexity: fast
      max_retries: 3
      timeout: 30
      enable_cache: true

  - id: response-1
    type: response_generator
    label: Generate Response
    position:
      x: 400
      y: 200
    config:
      format: markdown
      style: technical
      max_length: 500

connections:
  - source: query-1
    target: response-1
    ports:
      source: output
      target: input

metadata:
  tags:
    - research
    - rag
  canvas:
    zoom: 1.0
    pan: [0, 0]
```

### YAML vs JSON

| Aspect | YAML | JSON |
|--------|------|------|
| Readability | High (human-friendly) | Medium |
| Comments | ✅ Supported | ❌ Not supported |
| File size | Smaller (no quotes) | Larger |
| Parsing speed | Slower | Faster |
| Git diffs | Cleaner | More noise |

### Use Cases

- **Configuration files**: GitOps workflows
- **Documentation**: Readable workflow specs
- **Templates**: Human-editable base configurations
- **CI/CD pipelines**: Workflow-as-code

## Format Comparison

### Feature Matrix

| Feature | JSON | Python | YAML |
|---------|------|--------|------|
| **Fidelity** | Full | Partial | Full |
| **Executable** | ❌ | ✅ | ❌ |
| **Editable** | Medium | High | High |
| **Importable** | ✅ | ❌ | ✅ |
| **Version control** | Good | Good | Best |
| **File size** | Medium | Large | Small |

### When to Use Each

**JSON**:
- Backup and restore
- Sharing between Workflow Builder instances
- API integration
- Automated workflow management

**Python**:
- Production deployment
- Custom modifications
- Integration with other systems
- Learning HoloLoom API

**YAML**:
- Version control (Git)
- Configuration management
- Human review and editing
- Documentation

## Export Options

### Common Options

```
┌─────────────────────────────────────────┐
│ Export Options                       ×  │
├─────────────────────────────────────────┤
│ Format: [JSON ▼]                        │
│                                         │
│ ☑ Include metadata                      │
│ ☑ Include canvas state                  │
│ ☑ Include node positions                │
│ ☐ Minify output                         │
│ ☐ Include execution history             │
│                                         │
│ Filename: my-workflow                   │
│                                         │
│ [Export] [Cancel]                       │
└─────────────────────────────────────────┘
```

### Option Descriptions

| Option | Description |
|--------|-------------|
| **Include metadata** | Tags, author, timestamps |
| **Include canvas state** | Zoom, pan position |
| **Include node positions** | X, Y coordinates |
| **Minify output** | Remove whitespace (JSON only) |
| **Include execution history** | Recent run data |

## Importing Workflows

### Supported Import Formats

- **JSON**: Full workflow import
- **YAML**: Full workflow import
- **Python**: ❌ Not supported (code is generated, not parsed)

### Import Methods

**File Dialog**:
1. **File** → **Import**
2. Select file
3. Click **Open**

**Drag and Drop**:
1. Drag workflow file onto canvas
2. Release to import

**URL Import**:
1. **File** → **Import from URL**
2. Enter URL to JSON/YAML file
3. Click **Import**

### Import Behavior

**New Workflow**:
- Replaces current canvas
- Prompts to save if unsaved changes

**Merge Mode**:
- Hold `Shift` while importing
- Adds nodes to current canvas
- Auto-generates new IDs to prevent conflicts

### Import Validation

The importer validates:
- Schema conformance
- Node type existence
- Connection validity (no cycles, valid ports)
- Configuration completeness

**Validation Errors**:
```
┌─────────────────────────────────────────┐
│ Import Validation Errors             ×  │
├─────────────────────────────────────────┤
│ ⚠ Node 'query-1': Unknown type 'xyz'   │
│ ⚠ Connection: Target 'node-5' not found│
│ ⚠ Config: Missing required 'timeout'   │
│                                         │
│ [Import Anyway] [Cancel]                │
└─────────────────────────────────────────┘
```

## API Integration

### Export via API

```javascript
// Get workflow JSON
const json = workflowBuilder.export('json');

// Get workflow Python
const python = workflowBuilder.export('python', {
  includeDocstrings: true,
  includeMain: true
});

// Get workflow YAML
const yaml = workflowBuilder.export('yaml');
```

### Import via API

```javascript
// Import from JSON string
workflowBuilder.import(jsonString, 'json');

// Import from URL
await workflowBuilder.importFromUrl('https://example.com/workflow.json');

// Import with merge
workflowBuilder.import(jsonString, 'json', { merge: true });
```

### Backend API

```http
POST /api/workflow/export
Content-Type: application/json

{
  "workflow_id": "wf-123",
  "format": "python",
  "options": {
    "includeDocstrings": true,
    "includeMain": true
  }
}
```

**Response**:
```json
{
  "format": "python",
  "content": "# Workflow: ...\nimport asyncio...",
  "filename": "my_workflow.py"
}
```

## Best Practices

### Version Control

1. **Use YAML for Git**: Cleaner diffs
2. **Add `.workflow.json` to `.gitignore`**: Keep binary out
3. **Version workflows**: Include version in filename
4. **Document changes**: Use commit messages

### Sharing Workflows

1. **Remove sensitive data**: Check for API keys in config
2. **Use relative paths**: Avoid absolute file references
3. **Include README**: Explain workflow purpose
4. **Test after import**: Verify functionality

### Production Deployment

1. **Export as Python**: For production scripts
2. **Review generated code**: Check for correctness
3. **Add error handling**: Enhance generated code
4. **Test thoroughly**: Before deployment

---

← [Voice Commands](voice-commands.md) | [Advanced: Nested Workflows](../advanced/nested-workflows.md) →

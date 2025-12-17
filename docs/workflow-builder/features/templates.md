# Templates & Presets

Save, share, and reuse workflow patterns with the template system.

## Overview

Templates are pre-built workflow patterns that you can use as starting points or building blocks. The Workflow Builder includes built-in templates and supports creating your own.

## Built-in Templates

### RAG Pipelines

| Template | Description | Nodes |
|----------|-------------|-------|
| **Basic RAG** | Simple retrieve-and-generate | 3 |
| **Multi-Query RAG** | Break into sub-queries | 5 |
| **Verified RAG** | With accuracy verification | 6 |

### Agentic Workflows

| Template | Description | Nodes |
|----------|-------------|-------|
| **Research Agent** | Multi-query exploration | 7 |
| **Plan-Execute** | Goal decomposition | 8 |
| **Verify-Refine** | Iterative quality improvement | 5 |

### Memory Operations

| Template | Description | Nodes |
|----------|-------------|-------|
| **Store & Index** | Persist with indexing | 3 |
| **Knowledge Fusion** | Multi-hop retrieval | 4 |

### Decision Trees

| Template | Description | Nodes |
|----------|-------------|-------|
| **Risk Gate** | Safety-gated execution | 4 |
| **Quality Branch** | Confidence-based routing | 5 |

## Using Templates

### From Template Gallery

1. Click **"Templates"** in the toolbar
2. Browse categories or search
3. Click a template to preview
4. Click **"Use Template"** to add to canvas

### Quick Insert

1. Press `Ctrl+P` to open command palette
2. Type "template:"
3. Select from list

### Drag from Palette

1. Scroll to bottom of Agent Palette
2. Templates section shows saved templates
3. Drag onto canvas

## Template Gallery UI

```
┌─────────────────────────────────────────────────────────────┐
│ Template Gallery                                      [X]   │
├──────────────────┬──────────────────────────────────────────┤
│                  │ ┌────────────────────────────────────┐   │
│ Categories       │ │ Basic RAG Pipeline                 │   │
│ ────────────     │ │ ┌──────────────────────────────┐  │   │
│ ▼ RAG Pipelines  │ │ │     [Preview Diagram]        │  │   │
│   • Basic RAG    │ │ │                              │  │   │
│   • Multi-Query  │ │ └──────────────────────────────┘  │   │
│   • Verified     │ │                                    │   │
│                  │ │ Simple retrieve-and-generate       │   │
│ ▼ Agentic        │ │ pipeline for Q&A tasks.            │   │
│   • Research     │ │                                    │   │
│   • Plan-Execute │ │ Nodes: 3 | Est. Time: ~2s          │   │
│                  │ │                                    │   │
│ ▼ Memory         │ │ [Use Template]   [Preview JSON]    │   │
│                  │ └────────────────────────────────────┘   │
│ ▼ Decision       │                                          │
│                  │ ┌────────────────────────────────────┐   │
│ ────────────     │ │ Multi-Query RAG                    │   │
│ Search: [____]   │ │ ...                                │   │
└──────────────────┴──────────────────────────────────────────┘
```

## Creating Templates

### Save Selection as Template

1. **Select nodes** you want to save (Shift+Click or drag box)
2. **Right-click** → "Save as Template"
3. Fill in template details:
   - **Name**: Template name
   - **Description**: What it does
   - **Category**: RAG, Agentic, Memory, Custom
   - **Tags**: Searchable tags

### Template JSON Structure

```json
{
  "id": "my-custom-template",
  "name": "My Custom Pipeline",
  "description": "Description of what this template does",
  "category": "custom",
  "tags": ["custom", "pipeline"],
  "author": "Your Name",
  "version": "1.0",
  "created": "2025-12-15",
  "nodes": [
    {
      "id": "query-1",
      "type": "hololoom_query",
      "x": 100,
      "y": 100,
      "config": {
        "query_template": "${input.query}",
        "complexity": "fast"
      }
    },
    {
      "id": "response-1",
      "type": "response_generator",
      "x": 400,
      "y": 100,
      "config": {
        "format": "text"
      }
    }
  ],
  "connections": [
    {
      "source": "query-1",
      "target": "response-1",
      "sourcePort": "output",
      "targetPort": "input"
    }
  ],
  "inputs": [
    {
      "name": "query",
      "type": "string",
      "required": true,
      "description": "The query to process"
    }
  ],
  "outputs": [
    {
      "name": "response",
      "type": "string",
      "description": "Generated response"
    }
  ]
}
```

### Template Variables

Use variables for customizable templates:

```json
{
  "config": {
    "query_template": "${input.query}",
    "complexity": "{{complexity|fast}}",
    "max_results": "{{max_results|10}}"
  }
}
```

Variable syntax: `{{variable_name|default_value}}`

When using the template, users can customize these values.

## Managing Templates

### Local Storage

Templates are stored in browser localStorage:

```javascript
// View stored templates
localStorage.getItem('workflow_templates')

// Clear all templates
localStorage.removeItem('workflow_templates')
```

### Import/Export

**Export Templates**:
1. Open Template Gallery
2. Click template → "Export"
3. Download as `.workflow-template.json`

**Import Templates**:
1. Drag template file onto canvas
2. Or: Template Gallery → "Import" → Select file

### Template Locations

| Location | Description |
|----------|-------------|
| Browser localStorage | Personal templates |
| `example_workflows/` | Built-in examples |
| Server templates | Shared team templates |

## Presets

Presets are pre-configured node settings (not full workflows).

### Using Presets

1. Add a node to canvas
2. Open Properties Panel
3. Click **"Load Preset"**
4. Select from list

### Built-in Presets

**HoloLoom Query Presets**:
- Quick: bare mode, 10s timeout
- Standard: fast mode, 30s timeout
- Thorough: fused mode, 120s timeout

**Refiner Presets**:
- Quick Polish: elegance, 2 iterations
- Deep Verify: verify, 3 iterations
- Academic: hofstadter, 5 iterations

### Creating Presets

1. Configure a node as desired
2. Right-click node → "Save as Preset"
3. Name the preset
4. Access from Properties Panel

## Template Best Practices

### Design for Reuse

- Use template variables for customization
- Keep templates focused (single purpose)
- Document inputs and outputs

### Naming Conventions

| Pattern | Example |
|---------|---------|
| `{purpose}-{type}` | `research-pipeline` |
| `{action}-{target}` | `verify-response` |

### Version Templates

Include version in template:

```json
{
  "version": "1.0",
  "compatibility": ">=7.0"
}
```

### Test Templates

Before sharing:
1. Create from template
2. Execute with various inputs
3. Verify outputs
4. Check error handling

## Marketplace (Coming Soon)

The Workflow Marketplace allows sharing templates with the community:

- Browse community templates
- Rate and review
- Download statistics
- Author profiles

---

← [Connections & Data Flow](connections.md) | [Real-Time Collaboration](collaboration.md) →

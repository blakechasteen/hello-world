# HoloLoom Tool Builder - No-Code Interface

## Overview

The **HoloLoom Tool Builder** is a comprehensive no-code interface that enables non-technical users to create custom MCP (Model Context Protocol) tools through a visual drag-and-drop interface. Built as part of Phase 3E, it democratizes tool creation by eliminating the need for programming knowledge while maintaining full flexibility and power.

## Table of Contents

1. [Architecture](#architecture)
2. [Features](#features)
3. [Components](#components)
4. [Getting Started](#getting-started)
5. [Creating Your First Tool](#creating-your-first-tool)
6. [Component Library](#component-library)
7. [API Reference](#api-reference)
8. [Security](#security)
9. [Examples](#examples)
10. [Troubleshooting](#troubleshooting)

---

## Architecture

The Tool Builder follows a layered architecture:

```
┌─────────────────────────────────────────┐
│      Visual Tool Builder UI             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │Component│  │Parameter│  │  Test   │ │
│  │ Library │  │ Config  │  │ Runner  │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │            │             │      │
└───────┼────────────┼─────────────┼──────┘
        │            │             │
        ▼            ▼             ▼
┌──────────────────────────────────────────┐
│     Tool Definition JSON                 │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│     Python Code Generator                │
│  • Validates definition                  │
│  • Generates MCP-compatible code         │
│  • Formats with type hints               │
└───────────────┬──────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────┐
│     MCP Tool Registry                    │
│  • Stores custom tools                   │
│  • Version management                    │
│  • Executes on demand                    │
└──────────────────────────────────────────┘
```

### Core Components

1. **Definition Schema** (`toolbuilder/definition.py`)
   - Data structures for tool definitions
   - Serialization/deserialization
   - Type-safe parameter definitions

2. **Validator** (`toolbuilder/validator.py`)
   - Security validation
   - Structure validation
   - Complexity limits

3. **Code Generator** (`toolbuilder/generator.py`)
   - Python code generation
   - Type hints and docstrings
   - MCP-compatible output

4. **Tool Registry** (`toolbuilder/registry.py`)
   - Tool storage and retrieval
   - Version history
   - Search and discovery

5. **Executor** (`toolbuilder/executor.py`)
   - Sandboxed execution
   - Resource limits
   - Error handling

6. **REST API** (`web/toolbuilder_api.py`)
   - CRUD operations
   - Test execution
   - Template management

7. **Visual Editor** (`web/static/toolbuilder/`)
   - React-based UI
   - Drag-and-drop components
   - Live code preview

---

## Features

### Visual Tool Creation
- **Drag-and-drop interface** - No coding required
- **Real-time validation** - Instant feedback on errors
- **Code preview** - See generated Python code
- **Live testing** - Test tools before deployment

### Rich Component Library
- **Input Types**: Text, Number, File, Dropdown, Checkbox, Date, JSON, Array
- **Logic Blocks**: If/Else, For/While Loops, Transforms, Variables
- **Actions**: HTTP Requests, Database Queries, File Operations, Computations
- **Outputs**: Return values, Memory storage, File generation

### Tool Management
- **Version Control** - Track changes over time
- **Enable/Disable** - Control tool availability
- **Search & Filter** - Find tools quickly
- **Import/Export** - Share tools between systems

### Security & Safety
- **Sandboxed Execution** - Isolated runtime environment
- **Resource Limits** - Timeout and memory constraints
- **Dangerous Operation Detection** - Blocks unsafe code patterns
- **Input Validation** - Type checking and constraint enforcement

### Templates
- **Pre-built Tools** - Start with common patterns
- **Customizable** - Modify templates to fit your needs
- **Best Practices** - Learn from examples

---

## Components

### Parameter Types

#### Text Input
```json
{
  "name": "message",
  "type": "text",
  "description": "Message to process",
  "required": true,
  "validation": {
    "min_length": 1,
    "max_length": 1000,
    "pattern": "^[a-zA-Z0-9\\s]+$"
  }
}
```

#### Number Input
```json
{
  "name": "count",
  "type": "number",
  "description": "Number of items",
  "required": true,
  "default": 10,
  "validation": {
    "min": 1,
    "max": 100
  }
}
```

#### Dropdown Selection
```json
{
  "name": "operation",
  "type": "dropdown",
  "description": "Operation to perform",
  "required": true,
  "options": ["add", "subtract", "multiply", "divide"]
}
```

### Logic Blocks

#### If/Else Condition
```json
{
  "id": "block_1",
  "type": "if_else",
  "condition": {
    "left": "${operation}",
    "operator": "==",
    "right": "add"
  },
  "then_blocks": [...],
  "else_blocks": [...]
}
```

#### For Loop
```json
{
  "id": "block_2",
  "type": "for_loop",
  "iterator": "item",
  "iterable": "${items}",
  "loop_blocks": [...]
}
```

#### Variable Assignment
```json
{
  "id": "block_3",
  "type": "variable",
  "variable_name": "result",
  "variable_value": "${x} + ${y}"
}
```

### Actions

#### HTTP Request
```json
{
  "id": "block_4",
  "type": "variable",
  "action_type": "http_request",
  "action_config": {
    "method": "GET",
    "url": "https://api.example.com/data",
    "headers": {"Authorization": "Bearer ${token}"},
    "timeout": 30
  }
}
```

#### File Operation
```json
{
  "id": "block_5",
  "type": "variable",
  "action_type": "file_operation",
  "action_config": {
    "operation": "read",
    "path": "${file_path}",
    "encoding": "utf-8"
  }
}
```

#### Compute
```json
{
  "id": "block_6",
  "type": "variable",
  "action_type": "compute",
  "action_config": {
    "operation": "math",
    "expression": "math.sqrt(${x}**2 + ${y}**2)",
    "variables": {"x": "x", "y": "y"}
  }
}
```

---

## Getting Started

### Installation

1. **Backend Setup**

```bash
# Install Python dependencies
pip install fastapi uvicorn aiohttp aiofiles jinja2

# The tool builder is already included in HoloLoom
cd /home/user/hello-world/HoloLoom
```

2. **Start the Web Server**

```python
# In your FastAPI app (web/app.py), add:
from .toolbuilder_api import create_toolbuilder_router

# Create router
toolbuilder_router = create_toolbuilder_router(storage_path="./tools")

# Include in app
app.include_router(toolbuilder_router)
```

3. **Access the Interface**

```
http://localhost:8000/static/toolbuilder/index.html
```

### Configuration

```python
from HoloLoom.toolbuilder import ToolRegistry, ToolExecutor

# Initialize registry
registry = ToolRegistry(storage_path="./custom_tools")

# Initialize executor with custom limits
executor = ToolExecutor(
    registry=registry,
    timeout=60,  # 60 second timeout
    memory_limit=512 * 1024 * 1024,  # 512 MB
    enable_sandbox=True  # Enable security sandbox
)
```

---

## Creating Your First Tool

### Example: Temperature Converter

Let's create a tool that converts between Celsius and Fahrenheit.

#### Step 1: Basic Information

```json
{
  "id": "temp_converter",
  "name": "temperature_converter",
  "description": "Convert between Celsius and Fahrenheit",
  "version": "1.0.0",
  "category": "utility",
  "tags": ["temperature", "conversion", "utility"]
}
```

#### Step 2: Define Parameters

```json
{
  "parameters": [
    {
      "name": "temperature",
      "type": "number",
      "description": "Temperature value",
      "required": true
    },
    {
      "name": "from_unit",
      "type": "dropdown",
      "description": "Source unit",
      "required": true,
      "options": ["celsius", "fahrenheit"]
    },
    {
      "name": "to_unit",
      "type": "dropdown",
      "description": "Target unit",
      "required": true,
      "options": ["celsius", "fahrenheit"]
    }
  ]
}
```

#### Step 3: Add Logic

```json
{
  "logic_blocks": [
    {
      "id": "block_1",
      "type": "if_else",
      "condition": {
        "left": "${from_unit}",
        "operator": "==",
        "right": "celsius"
      },
      "then_blocks": [
        {
          "id": "block_1_1",
          "type": "variable",
          "variable_name": "result",
          "variable_value": "(${temperature} * 9/5) + 32"
        }
      ],
      "else_blocks": [
        {
          "id": "block_1_2",
          "type": "variable",
          "variable_name": "result",
          "variable_value": "(${temperature} - 32) * 5/9"
        }
      ]
    },
    {
      "id": "block_2",
      "type": "return",
      "return_value": "${result}"
    }
  ]
}
```

#### Step 4: Test the Tool

Use the Test Runner to validate:

```json
{
  "temperature": 100,
  "from_unit": "celsius",
  "to_unit": "fahrenheit"
}
```

Expected result: `212.0`

#### Generated Code

The system automatically generates:

```python
async def temperature_converter(
    temperature: float,
    from_unit: str,
    to_unit: str
) -> Dict[str, Any]:
    """
    Convert between Celsius and Fahrenheit

    Args:
        temperature: Temperature value
        from_unit: Source unit
        to_unit: Target unit

    Returns:
        Dict containing the tool execution result
    """
    from typing import Dict, Any, List

    # Initialize execution context
    context = {
        "temperature": temperature,
        "from_unit": from_unit,
        "to_unit": to_unit,
    }

    if context['from_unit'] == "celsius":
        context['result'] = (context['temperature'] * 9/5) + 32
    else:
        context['result'] = (context['temperature'] - 32) * 5/9

    # Default return
    return {"status": "success", "result": context['result']}
```

---

## Component Library

### Inputs

| Type | Description | Use Case |
|------|-------------|----------|
| **Text** | Single-line text | Names, URLs, short strings |
| **Number** | Numeric input | Counts, amounts, calculations |
| **File** | File path | Reading/writing files |
| **Dropdown** | Select from options | Predefined choices |
| **Checkbox** | Boolean value | True/false flags |
| **Date** | Date selection | Scheduling, timestamps |
| **JSON** | Structured data | Complex objects |
| **Array** | List of items | Collections, iterations |

### Logic

| Type | Description | Use Case |
|------|-------------|----------|
| **If/Else** | Conditional branching | Decision making |
| **For Loop** | Iterate over items | Processing lists |
| **While Loop** | Conditional iteration | Dynamic loops |
| **Transform** | Map/Filter/Reduce | Data transformation |
| **Variable** | Store a value | Intermediate results |

### Actions

| Type | Description | Use Case |
|------|-------------|----------|
| **HTTP Request** | Call external APIs | Integration |
| **Database Query** | Query databases | Data retrieval |
| **File Operation** | Read/write files | File processing |
| **Compute** | Math/string operations | Calculations |
| **Call Tool** | Execute another tool | Composition |
| **Memory Operation** | Store/retrieve data | Persistence |

### Outputs

| Type | Description | Use Case |
|------|-------------|----------|
| **Return** | Return value | Final result |
| **To Memory** | Store in memory | Caching, state |
| **To File** | Write to file | Export data |

---

## API Reference

### Create Tool

```http
POST /api/toolbuilder/tools
Content-Type: application/json

{
  "definition": { ... }
}
```

**Response:**
```json
{
  "status": "success",
  "tool_id": "tool_abc123",
  "message": "Tool created successfully",
  "warnings": []
}
```

### List Tools

```http
GET /api/toolbuilder/tools?category=utility&enabled_only=true
```

**Response:**
```json
{
  "tools": [...],
  "total": 10,
  "categories": ["utility", "integration"],
  "tags": ["api", "data", "conversion"]
}
```

### Get Tool

```http
GET /api/toolbuilder/tools/{tool_id}
```

**Response:**
```json
{
  "status": "success",
  "tool": { ... },
  "code": "async def ..."
}
```

### Test Tool

```http
POST /api/toolbuilder/tools/{tool_id}/test
Content-Type: application/json

{
  "parameters": {
    "param1": "value1"
  },
  "timeout": 30
}
```

**Response:**
```json
{
  "status": "success",
  "result": {
    "status": "success",
    "result": "output",
    "stdout": "",
    "stderr": ""
  }
}
```

### Delete Tool

```http
DELETE /api/toolbuilder/tools/{tool_id}?keep_versions=true
```

### Get Templates

```http
GET /api/toolbuilder/templates
```

**Response:**
```json
{
  "status": "success",
  "templates": [
    {
      "id": "template_calculator",
      "name": "calculator",
      "description": "Basic calculator"
    }
  ],
  "total": 5
}
```

---

## Security

### Sandboxing

The Tool Executor runs custom tools in a restricted environment:

1. **Limited Imports** - Only safe modules allowed
2. **No File System Access** - Restricted to allowed directories
3. **No Network Access** - Unless explicitly configured
4. **No System Calls** - subprocess, os commands blocked
5. **No Dynamic Execution** - eval, exec, compile blocked

### Safe Built-ins

Allowed Python built-ins:
- Data types: `list`, `dict`, `set`, `tuple`, `str`, `int`, `float`, `bool`
- Iteration: `map`, `filter`, `range`, `enumerate`, `zip`
- Math: `abs`, `min`, `max`, `sum`, `round`, `pow`
- Conversion: `chr`, `ord`, `hex`, `bin`

### Dangerous Patterns

Automatically blocked:
- `eval()`, `exec()`, `compile()`
- `__import__()`, `importlib`
- `os`, `sys`, `subprocess`
- Dunder methods: `__getattr__`, `__setattr__`
- File operations: `open()` (use File Operation action instead)

### Resource Limits

Default limits:
- **Timeout**: 30 seconds
- **Memory**: 512 MB
- **Output Size**: 10 MB
- **Loop Iterations**: 1,000 max (for while loops)

### Validation Rules

1. **Name Format**: Must start with letter, contain only letters/numbers/underscores
2. **Version Format**: Must be X.Y.Z (semantic versioning)
3. **Max Parameters**: 50
4. **Max Logic Blocks**: 100
5. **Max Nesting Depth**: 10 levels

---

## Examples

### 1. Calculator

```json
{
  "name": "calculator",
  "description": "Basic arithmetic calculator",
  "parameters": [
    {"name": "operation", "type": "dropdown", "options": ["add", "subtract", "multiply", "divide"]},
    {"name": "x", "type": "number"},
    {"name": "y", "type": "number"}
  ],
  "logic_blocks": [
    {
      "type": "if_else",
      "condition": {"left": "${operation}", "operator": "==", "right": "add"},
      "then_blocks": [{"type": "variable", "variable_name": "result", "variable_value": "${x} + ${y}"}]
    },
    {"type": "return", "return_value": "${result}"}
  ]
}
```

### 2. Web Scraper

```json
{
  "name": "web_scraper",
  "description": "Fetch content from URL",
  "parameters": [
    {"name": "url", "type": "text", "validation": {"pattern": "^https?://.+"}},
    {"name": "timeout", "type": "number", "default": 30}
  ],
  "logic_blocks": [
    {
      "type": "variable",
      "action_type": "http_request",
      "action_config": {
        "method": "GET",
        "url": "${url}",
        "timeout": "${timeout}"
      }
    },
    {"type": "return", "return_value": "${http_response}"}
  ]
}
```

### 3. Data Transformer

```json
{
  "name": "data_filter",
  "description": "Filter array by condition",
  "parameters": [
    {"name": "data", "type": "array"},
    {"name": "field", "type": "text"},
    {"name": "min_value", "type": "number"}
  ],
  "logic_blocks": [
    {
      "type": "transform",
      "transform_type": "filter",
      "input_var": "${data}",
      "output_var": "filtered",
      "transform_expr": "item.get('${field}', 0) >= ${min_value}"
    },
    {"type": "return", "return_value": "${filtered}"}
  ]
}
```

---

## Troubleshooting

### Common Issues

#### Tool Validation Fails

**Problem**: "Tool name must start with letter"

**Solution**: Ensure tool name matches pattern `^[a-zA-Z][a-zA-Z0-9_-]*$`

```javascript
// Good
name: "my_calculator"
name: "DataProcessor"
name: "api-client-v2"

// Bad
name: "123tool"  // Starts with number
name: "my tool"  // Contains space
name: "tool@home"  // Invalid character
```

#### Execution Timeout

**Problem**: Tool execution times out

**Solution**:
1. Optimize logic blocks
2. Increase timeout parameter
3. Check for infinite loops

```python
# Increase timeout
result = await executor.execute(
    tool_id="my_tool",
    parameters={...},
    timeout_override=60  # 60 seconds
)
```

#### Parameter Validation Error

**Problem**: "Missing required parameter"

**Solution**: Ensure all required parameters are provided

```json
// Tool definition
{
  "parameters": [
    {"name": "required_param", "required": true},
    {"name": "optional_param", "required": false, "default": "default_value"}
  ]
}

// Valid execution
{
  "required_param": "value"
  // optional_param will use default
}
```

#### Generated Code Not Working

**Problem**: Code preview shows errors

**Solution**:
1. Validate tool definition first
2. Check variable references use `${}` syntax
3. Ensure logic block types match their configuration
4. Regenerate code after fixing

```http
POST /api/toolbuilder/tools/{tool_id}/regenerate
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("toolbuilder")
logger.setLevel(logging.DEBUG)
```

### Getting Help

1. **Check Warnings**: The validator provides helpful warnings
2. **Use Templates**: Start with a working template
3. **Test Incrementally**: Build and test one block at a time
4. **Review Generated Code**: The code preview shows exactly what will execute

---

## Best Practices

### Tool Design

1. **Single Responsibility**: Each tool should do one thing well
2. **Clear Parameters**: Use descriptive names and helpful descriptions
3. **Validation**: Add constraints to prevent invalid inputs
4. **Error Handling**: Plan for edge cases
5. **Documentation**: Write clear descriptions

### Performance

1. **Minimize HTTP Requests**: Batch when possible
2. **Avoid Deep Nesting**: Keep logic blocks shallow
3. **Use Transforms**: Leverage map/filter/reduce for efficiency
4. **Set Reasonable Timeouts**: Balance safety and usability

### Security

1. **Validate Inputs**: Always add validation rules
2. **Sanitize URLs**: Check URL patterns for external requests
3. **Limit File Paths**: Use relative paths, avoid traversal
4. **Test Thoroughly**: Use the test runner before deploying

### Maintenance

1. **Version Tools**: Save versions before major changes
2. **Use Semantic Versioning**: X.Y.Z format
3. **Tag Appropriately**: Use meaningful tags for discovery
4. **Document Changes**: Update descriptions when modifying

---

## Conclusion

The HoloLoom Tool Builder democratizes MCP tool creation by providing a powerful, secure, and user-friendly interface. Whether you're a business user automating workflows or a developer prototyping quickly, the Tool Builder enables rapid iteration and deployment of custom tools without writing code.

For support and contributions, see the main HoloLoom documentation.

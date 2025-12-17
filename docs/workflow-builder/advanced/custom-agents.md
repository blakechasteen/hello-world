# Custom Agents

Create your own agent types to extend the Workflow Builder with domain-specific functionality.

## Overview

While the Workflow Builder includes 18 built-in agent types, you can create custom agents for:
- Domain-specific processing (e.g., medical, legal, financial)
- Integration with external services
- Specialized algorithms or models
- Company-specific business logic

## Agent Architecture

### Agent Definition Structure

Every agent type requires:

```javascript
{
  "type": "my_custom_agent",
  "name": "My Custom Agent",
  "category": "custom",
  "icon": "🔧",
  "description": "Description of what this agent does",
  "inputs": ["input"],
  "outputs": ["output"],
  "configFields": [
    // Configuration options
  ]
}
```

### Required Components

| Component | Purpose |
|-----------|---------|
| **Definition** | UI metadata and configuration schema |
| **Executor** | Backend execution logic |
| **Validator** | Input validation rules |
| **Renderer** | Custom node appearance (optional) |

## Creating a Custom Agent

### Step 1: Define the Agent Type

Add to the agent type registry:

```javascript
// In workflow_builder.js or custom_agents.js
const CUSTOM_AGENTS = {
  sentiment_analyzer: {
    name: 'Sentiment Analyzer',
    category: 'custom',
    icon: '😊',
    description: 'Analyze sentiment of text inputs',
    inputs: ['text'],
    outputs: ['sentiment', 'confidence'],
    configFields: [
      {
        name: 'model',
        type: 'select',
        label: 'Analysis Model',
        options: ['basic', 'advanced', 'multilingual'],
        default: 'basic'
      },
      {
        name: 'include_emotions',
        type: 'boolean',
        label: 'Include Emotion Breakdown',
        default: false
      },
      {
        name: 'threshold',
        type: 'number',
        label: 'Confidence Threshold',
        min: 0,
        max: 1,
        step: 0.1,
        default: 0.5
      }
    ]
  }
};

// Register with the builder
workflowBuilder.registerAgentTypes(CUSTOM_AGENTS);
```

### Step 2: Implement the Executor

Create the backend execution logic:

```python
# custom_agents/sentiment_analyzer.py
from typing import Dict, Any
from workflow_executor import AgentExecutor, ExecutionContext

class SentimentAnalyzerExecutor(AgentExecutor):
    """Custom agent for sentiment analysis."""

    agent_type = "sentiment_analyzer"

    async def execute(
        self,
        inputs: Dict[str, Any],
        config: Dict[str, Any],
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """
        Analyze sentiment of input text.

        Args:
            inputs: {"text": "The input text to analyze"}
            config: {"model": "basic", "include_emotions": False, ...}
            context: Execution context with memory, tools, etc.

        Returns:
            {"sentiment": "positive", "confidence": 0.92, ...}
        """
        text = inputs.get('text', '')
        model = config.get('model', 'basic')
        include_emotions = config.get('include_emotions', False)
        threshold = config.get('threshold', 0.5)

        # Your sentiment analysis logic here
        result = await self._analyze_sentiment(text, model)

        output = {
            'sentiment': result['label'],  # positive, negative, neutral
            'confidence': result['score']
        }

        if include_emotions:
            output['emotions'] = result.get('emotions', {})

        # Filter by confidence threshold
        if output['confidence'] < threshold:
            output['sentiment'] = 'uncertain'

        return output

    async def _analyze_sentiment(self, text: str, model: str) -> Dict:
        """Internal sentiment analysis implementation."""
        # Implementation depends on your sentiment model
        # Could use HuggingFace, OpenAI, custom model, etc.
        pass

    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that required inputs are present."""
        return 'text' in inputs and isinstance(inputs['text'], str)

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration values."""
        model = config.get('model', 'basic')
        return model in ['basic', 'advanced', 'multilingual']
```

### Step 3: Register the Executor

Register with the workflow executor:

```python
# In workflow_executor.py or custom_agents/__init__.py
from custom_agents.sentiment_analyzer import SentimentAnalyzerExecutor

# Register executor
executor_registry.register(SentimentAnalyzerExecutor())
```

### Step 4: Add Custom Styling (Optional)

Create custom node appearance:

```css
/* Custom agent styling */
.workflow-node[data-type="sentiment_analyzer"] {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-color: #764ba2;
}

.workflow-node[data-type="sentiment_analyzer"] .node-icon {
  font-size: 24px;
}

.workflow-node[data-type="sentiment_analyzer"] .node-header {
  color: white;
}
```

## Configuration Field Types

### Available Field Types

| Type | Description | Example |
|------|-------------|---------|
| `string` | Text input | API key, template |
| `number` | Numeric input | Timeout, threshold |
| `boolean` | Toggle switch | Enable/disable |
| `select` | Dropdown selection | Model choice |
| `multiselect` | Multiple selections | Features to enable |
| `textarea` | Multi-line text | Prompt template |
| `json` | JSON editor | Complex config |
| `file` | File upload | Model weights |
| `color` | Color picker | UI customization |

### Field Definition Examples

**String Field**:
```javascript
{
  name: 'api_key',
  type: 'string',
  label: 'API Key',
  placeholder: 'Enter your API key',
  required: true,
  secret: true  // Masks input
}
```

**Select Field**:
```javascript
{
  name: 'language',
  type: 'select',
  label: 'Language',
  options: [
    { value: 'en', label: 'English' },
    { value: 'es', label: 'Spanish' },
    { value: 'fr', label: 'French' }
  ],
  default: 'en'
}
```

**Conditional Field**:
```javascript
{
  name: 'custom_endpoint',
  type: 'string',
  label: 'Custom Endpoint',
  showIf: { field: 'use_custom', value: true }
}
```

**Validation**:
```javascript
{
  name: 'timeout',
  type: 'number',
  label: 'Timeout (seconds)',
  min: 1,
  max: 300,
  validation: {
    pattern: '^[0-9]+$',
    message: 'Must be a positive integer'
  }
}
```

## Input/Output Ports

### Defining Ports

```javascript
{
  inputs: [
    { name: 'text', type: 'string', required: true },
    { name: 'context', type: 'object', required: false }
  ],
  outputs: [
    { name: 'result', type: 'object' },
    { name: 'confidence', type: 'number' },
    { name: 'error', type: 'string', errorPort: true }
  ]
}
```

### Port Types

| Type | Description |
|------|-------------|
| `string` | Text data |
| `number` | Numeric data |
| `boolean` | True/false |
| `object` | JSON object |
| `array` | JSON array |
| `any` | Any data type |
| `stream` | Streaming data |

### Error Ports

Handle errors gracefully with dedicated error output:

```javascript
outputs: [
  { name: 'result', type: 'object' },
  { name: 'error', type: 'object', errorPort: true }
]
```

In executor:
```python
async def execute(self, inputs, config, context):
    try:
        result = await self._process(inputs)
        return {'result': result, 'error': None}
    except Exception as e:
        return {'result': None, 'error': {'message': str(e)}}
```

## Integration Patterns

### External API Integration

```python
class ExternalAPIAgent(AgentExecutor):
    """Agent that calls external API."""

    agent_type = "external_api"

    async def execute(self, inputs, config, context):
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                config['endpoint'],
                headers={'Authorization': f"Bearer {config['api_key']}"},
                json=inputs,
                timeout=config.get('timeout', 30)
            )
            response.raise_for_status()
            return response.json()
```

### Database Integration

```python
class DatabaseQueryAgent(AgentExecutor):
    """Agent that queries database."""

    agent_type = "database_query"

    async def execute(self, inputs, config, context):
        # Use connection pool from context
        pool = context.get_resource('db_pool')

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                config['query'],
                *inputs.get('params', [])
            )
            return {'rows': [dict(r) for r in rows]}
```

### HoloLoom Integration

```python
class HoloLoomCustomAgent(AgentExecutor):
    """Agent using HoloLoom capabilities."""

    agent_type = "hololoom_custom"

    async def execute(self, inputs, config, context):
        # Access HoloLoom orchestrator
        orchestrator = context.get_resource('hololoom')

        # Use HoloLoom's memory system
        memories = await orchestrator.recall(
            inputs['query'],
            k=config.get('memory_limit', 10)
        )

        # Custom processing with HoloLoom features
        result = await self._process_with_memories(
            inputs, memories, config
        )

        return result
```

## Testing Custom Agents

### Unit Testing

```python
# tests/test_sentiment_analyzer.py
import pytest
from custom_agents.sentiment_analyzer import SentimentAnalyzerExecutor

@pytest.fixture
def executor():
    return SentimentAnalyzerExecutor()

@pytest.mark.asyncio
async def test_positive_sentiment(executor):
    inputs = {'text': 'I love this product!'}
    config = {'model': 'basic', 'include_emotions': False}
    context = MockExecutionContext()

    result = await executor.execute(inputs, config, context)

    assert result['sentiment'] == 'positive'
    assert result['confidence'] > 0.7

@pytest.mark.asyncio
async def test_invalid_input(executor):
    inputs = {}  # Missing 'text'

    assert not executor.validate_inputs(inputs)
```

### Integration Testing

```python
# tests/test_workflow_integration.py
@pytest.mark.asyncio
async def test_custom_agent_in_workflow():
    workflow = {
        'nodes': [
            {'id': 'input', 'type': 'input_node'},
            {'id': 'sentiment', 'type': 'sentiment_analyzer',
             'config': {'model': 'basic'}},
            {'id': 'output', 'type': 'output_node'}
        ],
        'connections': [
            {'source': 'input', 'target': 'sentiment'},
            {'source': 'sentiment', 'target': 'output'}
        ]
    }

    result = await execute_workflow(workflow, {'text': 'Great work!'})

    assert result['sentiment'] in ['positive', 'negative', 'neutral']
```

## Packaging and Distribution

### Agent Package Structure

```
my_custom_agents/
├── __init__.py
├── sentiment_analyzer.py
├── entity_extractor.py
├── package.json           # Agent definitions
├── styles.css             # Custom styling
├── README.md
└── tests/
    ├── test_sentiment.py
    └── test_entity.py
```

### Package Manifest

```json
{
  "name": "my-custom-agents",
  "version": "1.0.0",
  "description": "Custom agents for NLP tasks",
  "author": "Your Name",
  "agents": [
    {
      "type": "sentiment_analyzer",
      "name": "Sentiment Analyzer",
      "file": "sentiment_analyzer.py"
    },
    {
      "type": "entity_extractor",
      "name": "Entity Extractor",
      "file": "entity_extractor.py"
    }
  ],
  "dependencies": {
    "transformers": ">=4.0.0",
    "torch": ">=2.0.0"
  }
}
```

### Installation

```bash
# Install agent package
pip install my-custom-agents

# Or from local directory
pip install -e ./my_custom_agents
```

Register in workflow builder:
```javascript
// Load custom agent definitions
import customAgents from 'my-custom-agents/package.json';
workflowBuilder.registerAgentTypes(customAgents.agents);
```

## Best Practices

### Design Guidelines

1. **Single Responsibility**: One agent, one purpose
2. **Clear Naming**: Descriptive type names and labels
3. **Sensible Defaults**: Good default configuration values
4. **Error Handling**: Graceful failure with informative messages
5. **Documentation**: Comprehensive field descriptions

### Performance Tips

1. **Async Operations**: Use `async/await` for I/O
2. **Connection Pooling**: Reuse database/HTTP connections
3. **Caching**: Cache expensive computations
4. **Timeouts**: Always set reasonable timeouts
5. **Resource Cleanup**: Clean up in `finally` blocks

### Security Considerations

1. **Input Validation**: Validate all inputs
2. **Secret Handling**: Use `secret: true` for sensitive fields
3. **Sandboxing**: Limit agent capabilities appropriately
4. **Rate Limiting**: Protect against abuse
5. **Audit Logging**: Log sensitive operations

---

← [Nested Workflows](nested-workflows.md) | [Performance Optimization](performance.md) →

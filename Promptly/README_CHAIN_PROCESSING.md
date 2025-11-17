# Advanced Chain Processing for Promptly

Transform simple prompt chains into powerful workflow engines.

## Quick Start

```python
from promptly.chain_dsl import ChainDSL

# Load a workflow
dsl = ChainDSL()
chain = dsl.load_chain("examples/workflows/rag_pipeline.yaml")

# Execute
result = dsl.execute_chain(chain, {"input": "your query"})
print(result['final_output'])
```

## What's Included

### 🔀 Five Advanced Processors

1. **ConditionalProcessor** - If/else/elif logic with pattern matching
2. **ParallelProcessor** - Concurrent execution with aggregation
3. **LoopProcessor** - For-each, while, map/reduce operations
4. **RetryProcessor** - Exponential backoff and circuit breakers
5. **TransformProcessor** - Data extraction, validation, and conversion

### 📋 YAML Workflow DSL

Define complex workflows declaratively:

```yaml
name: my_workflow
steps:
  - name: extract
    type: transform
    config:
      transform_type: extract
      method: json

  - name: validate
    type: transform
    depends_on: [extract]
    config:
      transform_type: validate

  - name: process
    type: conditional
    depends_on: [validate]
    config:
      conditions:
        - type: keyword
          field: is_valid
          keywords: ["true"]
```

### 📊 Visualization Tools

- **Mermaid diagrams** for documentation
- **Graphviz** for detailed graphs
- **ASCII art** for terminal display
- **HTML reports** for execution traces

### 🔍 Execution Tracing

Monitor performance and debug issues:

```python
from promptly.chain_tracing import create_tracer

tracer = create_tracer(trace_level="standard")
# ... execute workflow ...
metrics = tracer.get_performance_metrics()
```

### 🎯 Example Workflows

Three production-ready examples:

1. **RAG Pipeline** - Multi-source retrieval with reranking
2. **A/B Testing** - Statistical comparison of prompt variants
3. **Multi-Agent** - Coordinator with specialist agents

## Installation

No additional dependencies required! Works with base Promptly installation.

Optional for full features:
```bash
pip install pyyaml  # For YAML workflows
```

## Documentation

- **[Complete Guide](CHAIN_PROCESSING.md)** - Full documentation (1,000+ lines)
- **[Quick Reference](CHAIN_QUICK_REFERENCE.md)** - Cheat sheet
- **[Summary](CHAIN_PROCESSING_SUMMARY.md)** - Implementation details

## Demo

Run the comprehensive demo:

```bash
python examples/chain_processing_demo.py
```

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **Conditional** | If/else logic, pattern matching |
| **Parallel** | Concurrent execution, 6 aggregation strategies |
| **Loops** | For-each, while, map/reduce |
| **Retry** | 5 backoff strategies, circuit breakers |
| **Transform** | Extract, validate, sanitize, convert |
| **DSL** | YAML workflow definitions |
| **Tracing** | Performance monitoring |
| **Visualization** | Multiple output formats |

## Architecture

```
promptly/
├── plugins/processors/      # 5 processors
├── chain_dsl.py            # YAML workflow engine
├── chain_visualization.py  # Graph & trace viz
└── chain_tracing.py        # Execution monitoring
```

## Example: RAG Pipeline

```yaml
# Parallel retrieval
- name: retrieve
  type: parallel
  config:
    tasks:
      - name: vector_db
        prompt: "Search: {query}"
      - name: keyword
        prompt: "BM25: {query}"
    aggregation: all

# Rerank with retry
- name: rerank
  type: retry
  depends_on: [retrieve]
  config:
    max_attempts: 3
    backoff_strategy: exponential

# Conditional response
- name: generate
  type: conditional
  depends_on: [rerank]
  config:
    conditions:
      - type: numeric
        field: context_score
        operator: gt
        value: 0.7
```

## Key Capabilities

### Conditional Processing
- Regex patterns
- Keyword matching
- Numeric comparisons
- Custom predicates

### Parallel Execution
- Thread pools
- Result aggregation
- Error strategies
- Timeout management

### Loop Operations
- Iteration over collections
- Map/reduce patterns
- Break/continue
- Accumulators

### Retry Mechanisms
- Exponential backoff
- Circuit breakers
- Rate limiting
- Fallback values

### Data Transformation
- JSON/CSV/regex extraction
- Format conversion
- Validation rules
- Sanitization

## Use Cases

- **RAG Systems** - Multi-stage retrieval and generation
- **A/B Testing** - Compare prompt variants
- **Multi-Agent** - Coordinate multiple AI agents
- **Data Pipelines** - ETL with validation
- **API Integration** - Retry and circuit breaking
- **Quality Control** - Validation and fallbacks

## Performance

- **Parallel**: Up to 10x faster for independent tasks
- **Circuit Breakers**: Prevent cascading failures
- **Rate Limiting**: Stay within API limits
- **Tracing**: <1% overhead on standard level

## Best Practices

1. **Always validate** input data
2. **Set timeouts** for external calls
3. **Use circuit breakers** for unreliable services
4. **Enable tracing** in production
5. **Limit iterations** in loops
6. **Test with mocks** before deployment

## Contributing

See main Promptly CONTRIBUTING.md.

## License

Same as Promptly main project.

## Support

- **Documentation**: See CHAIN_PROCESSING.md
- **Examples**: Check examples/workflows/
- **Demo**: Run chain_processing_demo.py
- **Issues**: Report on main Promptly repo

---

Built with ❤️ for the Promptly project

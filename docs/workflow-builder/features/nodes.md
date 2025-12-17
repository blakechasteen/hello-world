# Agent Types Reference

Complete reference for all 18 agent types available in the Workflow Builder.

## Overview

Agents are the building blocks of workflows. Each agent performs a specific function and can be connected to form complex processing pipelines.

### Categories

| Category | Color | Count | Purpose |
|----------|-------|-------|---------|
| Query | Blue | 3 | Execute queries and searches |
| Processing | Green | 3 | Transform and analyze data |
| Memory | Purple | 3 | Knowledge graph operations |
| Decision | Orange | 3 | Intelligent decision making |
| Output | Teal | 2 | Format and present results |
| Control | Gray | 3 | Workflow execution control |

---

## Query Agents

### HoloLoom Query

**Type**: `hololoom_query`
**Category**: Query
**Icon**: 🔍

The primary query agent that executes the full 9-step HoloLoom weaving cycle.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query_template` | String | `${input.query}` | Query text or template |
| `complexity` | Select | `fast` | Processing mode: bare, fast, fused |
| `max_retries` | Number | 3 | Retry count on failure |
| `timeout` | Number | 30 | Timeout in seconds |
| `enable_cache` | Boolean | true | Use query cache |

#### Input

| Port | Type | Description |
|------|------|-------------|
| `input` | Object | Contains `query` field |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `output` | Object | Full Spacetime result |
| `response` | String | Generated response text |
| `confidence` | Number | Confidence score (0-1) |

#### Example

```json
{
  "type": "hololoom_query",
  "config": {
    "query_template": "Explain ${input.topic} in simple terms",
    "complexity": "fast",
    "enable_cache": true
  }
}
```

---

### Memory Search

**Type**: `memory_search`
**Category**: Query
**Icon**: 📚

Direct search of the knowledge graph without full weaving cycle.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | String | Required | Search query |
| `limit` | Number | 10 | Maximum results |
| `min_similarity` | Number | 0.5 | Minimum similarity threshold |
| `include_graph` | Boolean | true | Include graph relationships |

#### Input

| Port | Type | Description |
|------|------|-------------|
| `input` | Object | Optional query override |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `results` | Array | List of memory objects |
| `count` | Number | Number of results |

---

### Multi-Query

**Type**: `multi_query`
**Category**: Query
**Icon**: 🔀

Breaks complex questions into sub-queries for comprehensive answers.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | String | Required | Main query to decompose |
| `max_subqueries` | Number | 5 | Maximum sub-queries to generate |
| `strategy` | Select | `auto` | Decomposition strategy |
| `parallel` | Boolean | true | Execute sub-queries in parallel |

#### Strategies

| Strategy | Description |
|----------|-------------|
| `auto` | Automatically determine best approach |
| `aspects` | Break into different aspects/facets |
| `depth` | Progressive depth exploration |
| `verification` | Generate verification queries |

#### Input/Output

Same as HoloLoom Query, but output includes `subqueries` array.

---

## Processing Agents

### Matryoshka Embedder

**Type**: `matryoshka_embedder`
**Category**: Processing
**Icon**: 🎭

Generates multi-scale embeddings using Matryoshka architecture.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | String | Required | Text to embed |
| `scales` | Array | [96, 192, 384] | Embedding dimensions |
| `normalize` | Boolean | true | L2 normalize output |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `embeddings` | Object | Multi-scale embeddings |
| `embedding_96` | Array | 96-dimensional embedding |
| `embedding_192` | Array | 192-dimensional embedding |
| `embedding_384` | Array | 384-dimensional embedding |

---

### Synthesizer

**Type**: `synthesizer`
**Category**: Processing
**Icon**: ⚗️

Extracts entities, motifs, and patterns from text.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | String | Required | Text to analyze |
| `extract_entities` | Boolean | true | Extract named entities |
| `extract_motifs` | Boolean | true | Extract semantic motifs |
| `extract_relations` | Boolean | false | Extract relationships |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `entities` | Array | Extracted entities |
| `motifs` | Array | Detected motifs |
| `relations` | Array | Entity relationships |

---

### Recursive Refiner

**Type**: `recursive_refiner`
**Category**: Processing
**Icon**: 🔄

Iteratively improves output quality until threshold is met.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | Any | Required | Data to refine |
| `strategy` | Select | `elegance` | Refinement strategy |
| `threshold` | Number | 0.9 | Quality threshold to reach |
| `max_iterations` | Number | 3 | Maximum refinement passes |

#### Strategies

| Strategy | Passes | Focus |
|----------|--------|-------|
| `refine` | Iterative | Context expansion |
| `critique` | 1 | Self-improvement |
| `verify` | 3 | Accuracy → Completeness → Consistency |
| `elegance` | 3 | Clarity → Simplicity → Beauty |
| `hofstadter` | Iterative | Recursive self-reference |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `output` | Any | Refined result |
| `quality` | Number | Final quality score |
| `iterations` | Number | Iterations performed |

---

## Memory Agents

### Memory Store

**Type**: `memory_store`
**Category**: Memory
**Icon**: 💾

Persists data to the knowledge graph and vector store.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `content` | String | Required | Content to store |
| `metadata` | Object | {} | Additional metadata |
| `create_relations` | Boolean | true | Auto-create entity relations |
| `namespace` | String | `default` | Storage namespace |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `memory_id` | String | ID of stored memory |
| `success` | Boolean | Storage success status |

---

### Context Retriever

**Type**: `context_retriever`
**Category**: Memory
**Icon**: 📖

Retrieves relevant context for a given query.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | String | Required | Query for context |
| `limit` | Number | 5 | Maximum context items |
| `strategy` | Select | `balanced` | Retrieval strategy |
| `expand_graph` | Boolean | true | Include graph neighbors |

#### Strategies

| Strategy | Description |
|----------|-------------|
| `recent` | Prioritize recent memories |
| `similar` | Prioritize semantic similarity |
| `connected` | Prioritize graph connections |
| `balanced` | Balance all factors |

---

### Knowledge Fusion

**Type**: `knowledge_fusion`
**Category**: Memory
**Icon**: 🧩

Multi-hop traversal for comprehensive knowledge retrieval.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `seed_query` | String | Required | Starting query |
| `max_hops` | Number | 3 | Maximum traversal depth |
| `min_relevance` | Number | 0.3 | Minimum relevance threshold |
| `fusion_strategy` | Select | `union` | How to combine results |

#### Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `union` | Include all discovered knowledge |
| `intersection` | Only include multiply-referenced |
| `weighted` | Weight by relevance and path |

---

## Decision Agents

### Thompson Sampler

**Type**: `thompson_sampler`
**Category**: Decision
**Icon**: 🎰

Bayesian decision making with exploration/exploitation balance.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `options` | Array | Required | Decision options |
| `prior_alpha` | Number | 1.0 | Prior success count |
| `prior_beta` | Number | 1.0 | Prior failure count |
| `exploration` | Number | 0.1 | Exploration factor |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `selected` | Any | Selected option |
| `probabilities` | Array | Selection probabilities |
| `confidence` | Number | Selection confidence |

---

### Convergence Engine

**Type**: `convergence_engine`
**Category**: Decision
**Icon**: 🎯

Collapses probability distributions to discrete decisions.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `distributions` | Array | Required | Input probability distributions |
| `strategy` | Select | `bayesian_blend` | Collapse strategy |
| `temperature` | Number | 1.0 | Sampling temperature |

#### Strategies

| Strategy | Description |
|----------|-------------|
| `argmax` | Always choose highest probability |
| `epsilon_greedy` | ε% random, (1-ε)% greedy |
| `bayesian_blend` | Blend neural + bandit priors |
| `pure_thompson` | Pure Thompson Sampling |

---

### Safety Guardrails

**Type**: `safety_guardrails`
**Category**: Decision
**Icon**: 🛡️

Risk-based action gating with human-in-the-loop escalation.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action` | String | Required | Action to evaluate |
| `context` | Object | {} | Action context |
| `risk_threshold` | Select | `medium` | Maximum allowed risk |
| `enable_escalation` | Boolean | true | Allow human escalation |

#### Risk Levels

| Level | Description | Auto-Allowed |
|-------|-------------|--------------|
| `low` | Minimal risk | Yes |
| `medium` | Moderate risk | Configurable |
| `high` | Significant risk | No - requires approval |
| `critical` | Severe risk | Blocked |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `allowed` | Boolean | Whether action is allowed |
| `risk_level` | String | Assessed risk level |
| `reason` | String | Decision explanation |
| `escalated` | Boolean | Whether escalated to human |

---

## Output Agents

### Response Generator

**Type**: `response_generator`
**Category**: Output
**Icon**: 💬

Generates natural language responses from workflow data.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | Any | Required | Data to generate from |
| `format` | Select | `text` | Output format |
| `style` | Select | `neutral` | Response style |
| `max_length` | Number | 500 | Maximum response length |

#### Formats

| Format | Description |
|--------|-------------|
| `text` | Plain text |
| `markdown` | Markdown formatted |
| `structured` | Structured data |

#### Styles

| Style | Description |
|-------|-------------|
| `neutral` | Factual, objective |
| `friendly` | Conversational |
| `technical` | Detailed, precise |
| `concise` | Brief, to the point |

---

### Format Converter

**Type**: `format_converter`
**Category**: Output
**Icon**: 📄

Converts data between formats.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | Any | Required | Data to convert |
| `source_format` | Select | `auto` | Source format |
| `target_format` | Select | Required | Target format |
| `options` | Object | {} | Format-specific options |

#### Formats

| Format | Description |
|--------|-------------|
| `json` | JSON object |
| `markdown` | Markdown text |
| `html` | HTML document |
| `csv` | CSV data |
| `yaml` | YAML document |

---

## Control Flow Agents

### Conditional Branch

**Type**: `conditional_branch`
**Category**: Control
**Icon**: 🔀

Routes execution based on conditions.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `condition` | String | Required | Condition expression |
| `true_path` | String | Required | Node ID for true path |
| `false_path` | String | Required | Node ID for false path |

#### Condition Expressions

```javascript
// Examples
input.confidence > 0.8
input.risk_level === 'low'
input.results.length > 0
input.type === 'question' && input.priority === 'high'
```

#### Output Ports

| Port | Type | Description |
|------|------|-------------|
| `true` | Any | Output when condition is true |
| `false` | Any | Output when condition is false |

---

### Loop Iterator

**Type**: `loop_iterator`
**Category**: Control
**Icon**: 🔁

Repeats execution until condition is met.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `items` | Array | Required | Items to iterate over |
| `condition` | String | `true` | Continue condition |
| `max_iterations` | Number | 10 | Maximum iterations |
| `accumulate` | Boolean | true | Accumulate results |

#### Output

| Port | Type | Description |
|------|------|-------------|
| `item` | Any | Current item (per iteration) |
| `index` | Number | Current index |
| `results` | Array | Accumulated results (when done) |

---

### Parallel Executor

**Type**: `parallel_executor`
**Category**: Control
**Icon**: ⚡

Executes multiple branches concurrently.

#### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `branches` | Array | Required | Branch node IDs |
| `wait_strategy` | Select | `all` | How to wait for completion |
| `timeout` | Number | 60 | Timeout in seconds |
| `error_handling` | Select | `fail_fast` | Error behavior |

#### Wait Strategies

| Strategy | Description |
|----------|-------------|
| `all` | Wait for all branches |
| `any` | Return on first completion |
| `majority` | Wait for >50% completion |

#### Error Handling

| Mode | Description |
|------|-------------|
| `fail_fast` | Fail on first error |
| `ignore` | Continue despite errors |
| `collect` | Collect all errors |

---

## Common Patterns

### Serial Pipeline

```
[Query] → [Process] → [Output]
```

### Parallel Processing

```
[Query] → [Parallel]
              ├─ [Path A] ─┐
              └─ [Path B] ─┴─ [Merge] → [Output]
```

### Conditional Flow

```
[Query] → [Condition]
              ├─ true → [High Confidence Path]
              └─ false → [Low Confidence Path]
```

### Loop with Refinement

```
[Query] → [Loop]
            ↓↑
         [Refine] ← (until quality > threshold)
            ↓
         [Output]
```

---

← [UI Overview](../getting-started/ui-overview.md) | [Connections & Data Flow](connections.md) →

# Connections & Data Flow

Understanding how data flows between nodes in the Workflow Builder.

## Connection Basics

Connections are the links between nodes that define how data flows through your workflow.

### Creating Connections

1. **Hover over a node** to reveal its ports
2. **Click and drag from an output port** (right side)
3. **Drop on an input port** (left side) of another node

### Connection Rules

| Rule | Description |
|------|-------------|
| **Direction** | Data flows left to right (output → input) |
| **Single Input** | Each input port accepts one connection |
| **Multiple Outputs** | Output ports can connect to multiple inputs |
| **No Cycles** | Workflows must be acyclic (DAG) |
| **Type Compatibility** | Ports must have compatible data types |

## Port Types

### Input Ports

Located on the **left side** of nodes.

| Port Name | Type | Description |
|-----------|------|-------------|
| `input` | Any | Primary input (accepts any data) |
| `query` | String | Text query input |
| `data` | Object | Structured data input |
| `items` | Array | List of items |
| `context` | Object | Context/metadata |

### Output Ports

Located on the **right side** of nodes.

| Port Name | Type | Description |
|-----------|------|-------------|
| `output` | Any | Primary output |
| `response` | String | Text response |
| `results` | Array | List of results |
| `success` | Boolean | Success status |
| `error` | Object | Error information |

### Specialized Ports

Some nodes have specialized ports:

**Conditional Branch**:
- Output ports: `true`, `false`

**Loop Iterator**:
- Output ports: `item`, `index`, `results`

**Parallel Executor**:
- Output ports: One per branch

## Data Flow Model

### Pass-by-Reference

Data is passed by reference between nodes. Large objects (embeddings, results arrays) are not copied.

```javascript
// Node A output
{ query: "test", embeddings: [0.1, 0.2, ...384] }

// Node B receives the same object reference
// Modifications affect the original
```

### Template Expressions

Use template expressions to access specific fields:

```javascript
// Access nested data
${input.response}           // Direct field access
${input.results[0].text}    // Array indexing
${input.metadata?.author}   // Optional chaining
${input.count || 0}         // Default values
```

### Type Coercion

The system automatically coerces types when possible:

| From | To | Conversion |
|------|----|------------|
| Object | String | `JSON.stringify()` |
| String | Object | `JSON.parse()` |
| Array | Object | `{ items: array }` |
| Number | String | `String(number)` |
| Any | Boolean | Truthy/falsy |

## Connection Visualization

### Line Styles

| Style | Meaning |
|-------|---------|
| Solid gray | Idle connection |
| Solid blue | Data currently flowing |
| Dashed | Conditional connection |
| Animated dots | Active data transfer |

### Port Colors

| Color | Type |
|-------|------|
| Blue | Any/Generic |
| Green | String |
| Purple | Object |
| Orange | Array |
| Red | Error |

## Data Inspection

### During Execution

Click a connection to see flowing data:

```
┌─────────────────────────────────┐
│ Connection: query-1 → process-1 │
├─────────────────────────────────┤
│ Data:                           │
│ {                               │
│   "query": "What is...",        │
│   "confidence": 0.92,           │
│   "response": "Thompson..."     │
│ }                               │
└─────────────────────────────────┘
```

### After Execution

Select a node to see input/output in Properties Panel:

- **Input tab**: Data received
- **Output tab**: Data sent
- **Debug tab**: Detailed flow information

## Advanced Data Flow

### Merging Data

When multiple connections feed into one node:

```
[Node A] ─┐
          ├─→ [Merge Node]
[Node B] ─┘
```

Data is merged based on the target node type:

| Target Type | Merge Behavior |
|-------------|----------------|
| Array input | Concatenate arrays |
| Object input | Merge objects (later wins) |
| Any input | Use last received |

### Splitting Data

One output can feed multiple inputs:

```
              ┌─→ [Node B]
[Node A] ────┼─→ [Node C]
              └─→ [Node D]
```

All target nodes receive the same data reference.

### Conditional Routing

Use Conditional Branch for conditional data flow:

```
[Query] → [Conditional Branch]
              │
              ├─ true → [High Confidence Handler]
              │
              └─ false → [Low Confidence Handler]
```

## Error Propagation

### Error Flow

Errors propagate through connections:

1. **Node fails**: Outputs error object
2. **Connected nodes**: Receive error in input
3. **Error handling**: Nodes can check for errors

```javascript
// Check for upstream errors
if (input.error) {
  return { error: input.error, skipped: true };
}
```

### Error Ports

Some nodes have explicit error ports:

```
[Node] ─── output ──→ [Next Node]
       └── error ───→ [Error Handler]
```

## Best Practices

### Keep Data Flow Clear

✅ **Good**: Linear, clear paths
```
[A] → [B] → [C] → [D]
```

❌ **Avoid**: Spaghetti connections
```
[A] → [B] → [C]
  ↘   ↗
   [D]
```

### Use Named Ports

When nodes have multiple outputs, use specific ports:

```javascript
// Instead of generic output
${input.output.response}

// Use named port
${input.response}
```

### Validate Data Types

Check data types in conditions:

```javascript
// Validate before processing
typeof input.query === 'string' && input.query.length > 0
```

### Handle Missing Data

Use defaults for optional data:

```javascript
${input.results || []}
${input.confidence ?? 0.5}
```

## Troubleshooting

### Common Issues

**Connection won't create**
- Check: Output → Input direction
- Check: No existing connection on input
- Check: Would not create cycle

**Data not flowing**
- Check: Upstream node completed
- Check: Connection is valid
- Check: No errors in pipeline

**Wrong data received**
- Check: Correct port selected
- Check: Template expression is correct
- Check: Data type matches

### Debug Mode

Enable debug mode to trace data flow:

1. Press `F10` to step through execution
2. Click connections to inspect data
3. Check Properties Panel → Debug tab

---

← [Agent Types](nodes.md) | [Templates & Presets](templates.md) →

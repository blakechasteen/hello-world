# Skill: Graphviz

## Metadata

- **Name**: `graphviz`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `creative`
- **Tags**: `graphviz, visualization, diagram, graph, architecture, dot`

## Description

**Short Description**:
Auto-generate architecture diagrams and graph visualizations with Graphviz DOT language.

**Detailed Description**:
The Graphviz skill provides comprehensive graph and diagram generation capabilities using the DOT language. Render DOT source to images (PNG, SVG, PDF), apply layout algorithms (dot, neato, circo, fdp, twopi), convert NetworkX graphs to Graphviz, generate architecture diagrams from component descriptions, create dependency graphs with cycle detection, and export to multiple formats. Supports directed/undirected graphs, clusters, custom styles, and hierarchical layouts. Ideal for architecture diagrams, dependency visualization, flowcharts, state machines, and network topology diagrams.

## Required Capabilities

Check all capabilities this skill requires:

- [ ] File system access (read)
- [x] File system access (write)
- [x] Code execution (bash)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `graphviz` binary (graph rendering engine)
- `dot` command (hierarchical layout)
- Optional: `neato`, `circo`, `fdp`, `twopi` (alternative layouts)
- Optional: `python-graphviz` (Python bindings)

**HoloLoom Integration**: Integrates with documentation generation, architecture visualization, dependency analysis, and workflow diagram creation.

## Input Schema

```json
{
  "operation": "string - render_dot|layout_graph|from_networkx|architecture_diagram|dependency_graph|export",
  "parameters": {
    "dot_source": "string (required for render_dot, layout_graph) - DOT language source",
    "output": "string (required) - Output file path",
    "format": "string (optional) - Output format: png|svg|pdf|dot (default: png)",
    "layout": "string (optional) - Layout algorithm: dot|neato|circo|fdp|twopi (default: dot)",
    "graph_data": "object (required for from_networkx) - NetworkX graph data",
    "components": "array (required for architecture_diagram) - Component definitions",
    "connections": "array (required for architecture_diagram) - Connection definitions",
    "style": "string (optional for architecture_diagram) - Style: default|clean|detailed",
    "dependencies": "object (required for dependency_graph) - Dependency relationships",
    "root": "string (optional for dependency_graph) - Root node",
    "formats": "array (required for export) - Output formats to generate",
    "dpi": "number (optional) - Output resolution (default: 96)",
    "rankdir": "string (optional) - Rank direction: TB|BT|LR|RL (default: TB)"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object - Rendering details",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "output": "string - Output file path",
    "format": "string - Output format",
    "layout": "string - Layout algorithm used",
    "nodes": "number - Number of nodes",
    "edges": "number - Number of edges",
    "size_kb": "number - Output file size",
    "width_px": "number - Image width (for raster formats)",
    "height_px": "number - Image height (for raster formats)",
    "layers": "number - Number of layers (for architecture_diagram)",
    "circular_dependencies": "number - Circular dependencies found (for dependency_graph)"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Examples

### Example 1: Render DOT to PNG

**Input**:
```json
{
  "operation": "render_dot",
  "parameters": {
    "dot_source": "digraph G { A -> B; B -> C; C -> D; A -> D; }",
    "output": "diagrams/simple_graph.png",
    "format": "png",
    "dpi": 150
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "render_dot",
    "dot_source_length": 45,
    "output": "diagrams/simple_graph.png",
    "format": "png",
    "nodes": 4,
    "edges": 4,
    "size_kb": 15.3,
    "width_px": 400,
    "height_px": 300
  },
  "message": "DOT graph rendered: 4 nodes, 4 edges",
  "execution_time_ms": 250
}
```

**Explanation**: Renders simple directed graph from DOT source to high-resolution PNG. Quick visualization of graph structures.

### Example 2: Apply Circular Layout

**Input**:
```json
{
  "operation": "layout_graph",
  "parameters": {
    "dot_source": "graph Network { hub -- node1; hub -- node2; hub -- node3; hub -- node4; node1 -- node2; node2 -- node3; node3 -- node4; node4 -- node1; }",
    "layout": "circo",
    "output": "diagrams/network_circular.svg",
    "format": "svg"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "layout_graph",
    "layout": "circo",
    "output": "diagrams/network_circular.svg",
    "format": "svg",
    "nodes": 5,
    "edges": 8,
    "iterations": 100
  },
  "message": "Circular layout applied: 5 nodes, 8 edges",
  "execution_time_ms": 180
}
```

**Explanation**: Applies circular layout algorithm to network graph. Ideal for visualizing hub-and-spoke or ring topologies.

### Example 3: Generate Architecture Diagram

**Input**:
```json
{
  "operation": "architecture_diagram",
  "parameters": {
    "components": [
      {"name": "Frontend", "type": "web", "layer": 1},
      {"name": "API Gateway", "type": "service", "layer": 2},
      {"name": "Auth Service", "type": "service", "layer": 2},
      {"name": "Database", "type": "datastore", "layer": 3}
    ],
    "connections": [
      {"from": "Frontend", "to": "API Gateway", "label": "HTTPS"},
      {"from": "API Gateway", "to": "Auth Service", "label": "gRPC"},
      {"from": "Auth Service", "to": "Database", "label": "SQL"}
    ],
    "style": "clean",
    "output": "diagrams/system_architecture.png"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "architecture_diagram",
    "components": 4,
    "connections": 3,
    "style": "clean",
    "output": "diagrams/system_architecture.png",
    "layers": 3,
    "format": "png"
  },
  "message": "Architecture diagram generated: 4 components, 3 layers",
  "execution_time_ms": 320
}
```

**Explanation**: Auto-generates clean architecture diagram from component definitions. Visualizes system structure with layered layout.

### Example 4: Dependency Graph with Cycle Detection

**Input**:
```json
{
  "operation": "dependency_graph",
  "parameters": {
    "dependencies": {
      "app": ["auth", "database", "cache"],
      "auth": ["database"],
      "api": ["app", "auth"],
      "cache": []
    },
    "root": "api",
    "output": "diagrams/dependencies.svg",
    "format": "svg"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "dependency_graph",
    "root": "api",
    "total_dependencies": 4,
    "depth": 3,
    "output": "diagrams/dependencies.svg",
    "circular_dependencies": 0,
    "format": "svg"
  },
  "message": "Dependency graph created: depth 3, no cycles",
  "execution_time_ms": 280
}
```

**Explanation**: Generates dependency graph from package/module dependencies. Detects circular dependencies and visualizes dependency depth.

### Example 5: Multi-Format Export

**Input**:
```json
{
  "operation": "export",
  "parameters": {
    "source": "diagrams/flowchart.dot",
    "formats": ["png", "svg", "pdf"],
    "output_dir": "exports/"
  }
}
```

**Expected Output**:
```json
{
  "status": "success",
  "result": {
    "operation": "export",
    "source": "diagrams/flowchart.dot",
    "formats": ["png", "svg", "pdf"],
    "outputs": [
      "exports/flowchart.png",
      "exports/flowchart.svg",
      "exports/flowchart.pdf"
    ],
    "total_size_kb": 125
  },
  "message": "Exported to 3 formats: PNG, SVG, PDF",
  "execution_time_ms": 450
}
```

**Explanation**: Exports same diagram to multiple formats for different use cases (web PNG, scalable SVG, print PDF).

## Testing Checklist

- [x] **Functionality**: All 6 operations execute correctly
- [x] **Error Handling**: Graceful handling of invalid DOT syntax, layout errors
- [x] **Security**: No command injection, safe file handling
- [x] **Performance**: Operations complete within expected time (<5s)
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Graphviz binary documented
- [x] **Edge Cases**: Handles large graphs, complex layouts, special characters
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom documentation and visualization pipelines

## Security Considerations

**Potential Risks**:
- **DOT Injection**: Malicious DOT code -> Validate and sanitize DOT source
- **Resource Exhaustion**: Large graphs consume memory -> Implement node/edge limits
- **File System Access**: Output paths -> Validate and restrict output directories

**Data Privacy**:
- [x] Does not log graph content
- [x] Does not upload diagrams to external servers
- [x] Does not access files outside designated directories

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] File operations restricted to output directories
- [x] Timeouts prevent infinite layout iterations

## Performance Characteristics

- **Expected Latency**: 100-5000ms (0.1-5 seconds depending on graph complexity)
- **Token Usage**: 50-300 tokens per execution
- **Resource Requirements**: Graphviz binaries, sufficient memory for large graphs
- **Scalability**: Limited by graph size (nodes, edges) and layout algorithm

**Operation-Specific Latencies**:
- `render_dot`: 100-1000ms (depends on graph size)
- `layout_graph`: 200-3000ms (complex layouts take longer)
- `from_networkx`: 150-1500ms (includes conversion overhead)
- `architecture_diagram`: 300-2000ms (component processing + rendering)
- `dependency_graph`: 200-1500ms (includes cycle detection)
- `export`: 200-1000ms per format (multiplied by format count)

**Layout Algorithm Performance**:
- `dot`: Fast (hierarchical, O(n))
- `neato`: Medium (force-directed, O(n²))
- `circo`: Fast (circular, O(n))
- `fdp`: Slow (force-directed, O(n²))
- `twopi`: Fast (radial, O(n))

## License

MIT License

## Related Documentation

- **Graphviz Docs**: [graphviz.org/documentation](https://graphviz.org/documentation)
- **DOT Language**: [graphviz.org/doc/info/lang.html](https://graphviz.org/doc/info/lang.html)
- **Layout Algorithms**: [graphviz.org/docs/layouts](https://graphviz.org/docs/layouts)
- **HoloLoom Creative Skills**: [../README.md](../README.md)

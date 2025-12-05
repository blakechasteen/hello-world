# Skill: Visual Compression Helper

## Metadata

- **Name**: `visual_compression_helper`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, compression, visual, efficiency`

## Description

**Short Description**:
Compress knowledge graphs to PNG images for 5-20x token savings in LLM context.

**Detailed Description**:
Context is expensive. This skill uses visual compression to convert knowledge graphs into PNG images, achieving 5-20x token savings while preserving entity relationships. Automatic compression when sources > threshold (default: 10), supports knowledge graphs and retrieval results, returns PNG bytes + compression metrics. Perfect for RAG systems, multimodal LLMs, or any scenario where graph context exceeds token budgets.

## Required Capabilities

Check all capabilities this skill requires:

- [ ] File system access (read)
- [ ] File system access (write)
- [ ] Code execution (bash)
- [ ] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**: HoloLoom.memory.visual_compression (compress_graph_to_image)
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "graph": "object - Knowledge graph or list of memory nodes",
  "compression_threshold": "number - Min nodes to trigger compression (default: 10)",
  "image_size": "tuple - (width, height) in pixels (default: (800, 600))",
  "include_labels": "boolean - Show node labels (default: true)"
}
```

## Output Schema

```json
{
  "compressed": "boolean - Whether compression was applied",
  "image_bytes": "base64 - PNG image data (if compressed=true)",
  "compression_ratio": "number - Token savings ratio (e.g., 12.5 = 12.5x savings)",
  "original_tokens": "number - Estimated tokens without compression",
  "compressed_tokens": "number - Tokens after compression",
  "metadata": {
    "node_count": "number",
    "edge_count": "number",
    "image_size_bytes": "number"
  }
}
```

## Prompt Template

```markdown
You are a visual compression expert converting graphs to images.

**Compression Request**:
- Graph: {graph}
- Compression threshold: {compression_threshold}
- Image size: {image_size}
- Include labels: {include_labels}

**Compression Algorithm**:
1. Check node count >= threshold
2. If yes:
   - Convert graph to NetworkX/force-directed layout
   - Render to PNG using matplotlib/PIL
   - Calculate token savings (text_tokens / image_tokens)
3. If no: return compressed=false

**Token Calculation**:
- Text: ~4 chars/token * (node_text_length + edge_text_length)
- Image: ~85 tokens per 512x512 image (GPT-4 Vision pricing)

**Your Task**:
1. Assess if compression worthwhile (node_count >= threshold)
2. If yes, generate force-directed graph layout
3. Render to PNG with optional labels
4. Calculate compression metrics

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"graph": [{"id": "A", "edges": [{"to": "B", "type": "USES"}]}, {"id": "B", "edges": []}], "compression_threshold": 2}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Small Graph Skip

**Input**:
```json
{"graph": {"nodes": 5, "edges": 4}, "compression_threshold": 10}
```

**Explanation**:
Skips compression for small graph (<10 nodes) - overhead not worth it, returns original text representation instead of PNG.

### Example 3: Maximum Compression

**Input**:
```json
{"graph": {"nodes": 100, "edges": 250}, "compression_threshold": 10, "image_size": [800, 600]}
```

**Explanation**:
Achieves 20x token savings for large graph - 100 nodes × 50 tokens/node = 5000 tokens → 250 token image description = 20x compression.

## Testing Checklist

Before deploying this skill, verify:

- [ ] **Functionality**: All examples execute correctly
- [ ] **Error Handling**: Graceful degradation for invalid inputs
- [ ] **Security**: No prompt injection vulnerabilities (run `skill_security_analyzer`)
- [ ] **Performance**: Executes within acceptable time limits (<5s for simple tasks)
- [ ] **Token Efficiency**: Prompt is concise and efficient (run `token_budget_adviser`)
- [ ] **Documentation**: All sections complete and accurate
- [ ] **Dependencies**: All required capabilities and dependencies documented
- [ ] **Edge Cases**: Handles edge cases without crashing
- [ ] **Output Consistency**: Returns consistent format across runs
- [ ] **Integration**: Works with other skills if dependencies exist

## Security Considerations

**Potential Risks**:
- [Risk 1]: [Description and mitigation]
- [Risk 2]: [Description and mitigation]

**Data Privacy**:
- [ ] Does not log sensitive user data
- [ ] Does not expose internal system details
- [ ] Does not make unauthorized external requests

**Sandboxing**:
- [ ] Operates within defined capability boundaries
- [ ] Does not attempt privilege escalation
- [ ] Does not modify system files outside skill scope

## Performance Characteristics

- **Expected Latency**: 100-300ms
- **Token Usage**: ~500
- **Resource Requirements**: HoloLoom integration, minimal overhead
- **Scalability**: Depends on graph/data size

## Maintenance Notes

**Known Limitations**:
- Requires HoloLoom integration (graceful degradation if unavailable)

**Future Enhancements**:
- Enhanced visualization options
- Additional export formats
- Performance optimizations

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release

## License

MIT License

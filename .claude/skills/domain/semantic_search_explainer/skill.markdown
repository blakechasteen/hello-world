# Skill: Semantic Search Explainer

## Metadata

- **Name**: `semantic_search_explainer`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, search, explanation, transparency`

## Description

**Short Description**:
Explain why specific memories were retrieved using Semantic Calculus 16-axis analysis.

**Detailed Description**:
Transparency builds trust. This skill explains retrieval results by analyzing queries along 16 interpretable semantic axes (sentiment, formality, technicality, certainty, urgency, abstraction, etc.). Shows which axes drove the match, highlights unexpected connections, and reveals hidden query intent. Perfect for debugging poor retrieval, understanding why certain memories surfaced, or explaining AI decisions to users.

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
**External Dependencies**: HoloLoom.semantic_calculus (228D projections, 16 axes)
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "query": "string - Original search query",
  "retrieved_memories": [
    {"id": "string", "content": "string", "relevance": "number"}
  ],
  "top_k": "number - Analyze top K results (default: 5)"
}
```

## Output Schema

```json
{
  "explanations": [
    {
      "memory_id": "string",
      "why_retrieved": "string - Human-readable explanation",
      "matching_axes": ["array of axes that matched"],
      "axis_scores": {"sentiment": 0.0, "formality": 0.0, ...},
      "surprisal": "number - How unexpected this match was (0.0-1.0)"
    }
  ],
  "query_intent": {
    "primary_axes": ["array of dominant axes in query"],
    "query_type": "factual|procedural|analytical|exploratory",
    "semantic_profile": "string - Brief characterization"
  },
  "insights": ["array of insights about retrieval patterns"]
}
```

## Prompt Template

```markdown
You are a semantic search explainability expert using 16-axis analysis.

**Explanation Request**:
- Query: {query}
- Retrieved memories: {retrieved_memories}
- Top K to analyze: {top_k}

**16 Semantic Axes**:
sentiment, formality, technicality, certainty, urgency, abstraction,
specificity, temporality, objectivity, complexity, scope, directness,
emotionality, actionability, novelty, controversy

**Your Task**:
1. Project query onto 16 axes (0.0-1.0 each)
2. For each retrieved memory:
   - Project memory onto 16 axes
   - Calculate axis-wise distances
   - Identify matching axes (distance < 0.2)
   - Calculate surprisal (how unexpected given query profile)
3. Determine query intent (primary axes, query type)
4. Generate human-readable explanations

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"query": "Urgently need help debugging!", "retrieved_memories": [...], "top_k": 3}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Low Relevance Analysis

**Input**:
```json
{"query": "machine learning", "retrieved_memories": [{"id": "mem_1", "text": "breakfast recipes", "relevance": 0.15}], "top_k": 1}
```

**Explanation**:
Explains why low-relevance results were retrieved - useful for debugging semantic search quality and identifying index issues.

### Example 3: Multi-Result Comparison

**Input**:
```json
{"query": "neural networks", "retrieved_memories": [{"id": "mem_1", "text": "deep learning basics", "relevance": 0.92}, {"id": "mem_2", "text": "backpropagation", "relevance": 0.85}], "top_k": 2}
```

**Explanation**:
Compares multiple results showing why first result scored higher - analyzes semantic overlap, abstraction level, and axis alignment differences.

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

- **Expected Latency**: 50-150ms
- **Token Usage**: ~600
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

# Skill: HoloLoom RAG Helper

## Metadata

- **Name**: `hololoom_rag_helper`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, rag, retrieval, qa, multimodal`

## Description

**Short Description**:
Simplifies HoloLoom RAG operations by providing an easy-to-use interface for question answering with automatic mode selection, multimodal support, and confidence scoring.

**Detailed Description**:
This skill wraps HoloLoom's sophisticated RAG system (Level 4 Agentic RAG + Graph RAG) into a simple question-answering interface. It automatically selects the appropriate reasoning mode (direct, verify, research) based on query complexity, handles multimodal inputs (text + images), and returns structured answers with sources, confidence scores, and metadata.

HoloLoom RAG includes:
- **Level 2 Hybrid RAG**: BM25 (keyword) + semantic similarity
- **Level 3 Graph RAG**: Entity relationships via Yarn Graph
- **Level 4 Agentic RAG**: Multi-step reasoning (4 modes)
- **Multimodal RAG**: Text + images with CLIP embeddings

This skill makes these advanced capabilities accessible through a single, clean interface.

## Required Capabilities

- [x] File system access (read)
- [ ] File system access (write)
- [ ] Code execution (bash)
- [x] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None (standalone)

**External Dependencies**:
- `HoloLoom` - RAG system (required)
- `asyncio` - Async execution (Python stdlib)

**HoloLoom Integration**:
- [x] Uses HoloLoom memory system
- [x] Uses HoloLoom RAG
- [ ] Uses HoloLoom alignment framework
- [ ] Uses HoloLoom learning systems

## Input Schema

**Expected Input Format**:
```json
{
  "question": "string - The question to answer",
  "mode": "string (optional) - Reasoning mode: auto|direct|verify|research (default: auto)",
  "max_sources": "number (optional) - Maximum sources to return (default: 5)",
  "include_images": "boolean (optional) - Include image sources in response (default: false)",
  "context": "string (optional) - Additional context for the question"
}
```

**Example Input**:
```json
{
  "question": "What is Thompson Sampling?",
  "mode": "auto",
  "max_sources": 5,
  "include_images": false
}
```

## Output Schema

**Expected Output Format**:
```json
{
  "answer": "string - The generated answer",
  "sources": [
    {
      "text": "string - Source text snippet",
      "relevance": "number (0.0-1.0) - Relevance score",
      "node_id": "string - Memory node ID"
    }
  ],
  "confidence": "number (0.0-1.0) - Answer confidence",
  "reasoning_mode": "string - Mode used: direct|verify|research",
  "metadata": {
    "execution_time_ms": "number",
    "cache_hit": "boolean",
    "total_sources_retrieved": "number",
    "warnings": ["array of warning strings"]
  }
}
```

**Example Output**:
```json
{
  "answer": "Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff in reinforcement learning. It balances trying new actions (exploration) with exploiting known good actions by sampling from posterior distributions over action values.",
  "sources": [
    {
      "text": "Thompson Sampling balances exploration/exploitation...",
      "relevance": 0.95,
      "node_id": "thompson_sampling_001"
    },
    {
      "text": "Bayesian approach to multi-armed bandits...",
      "relevance": 0.88,
      "node_id": "bayesian_methods_042"
    }
  ],
  "confidence": 0.92,
  "reasoning_mode": "direct",
  "metadata": {
    "execution_time_ms": 145,
    "cache_hit": false,
    "total_sources_retrieved": 5,
    "warnings": []
  }
}
```

## Prompt Template

```markdown
You are executing the HoloLoom RAG Helper skill.

**Context**:
You have access to HoloLoom's advanced RAG system, which combines hybrid search (BM25 + semantic), knowledge graph traversal, and agentic reasoning. Your task is to answer questions using this system.

**Your Task**:
Answer the user's question using HoloLoom's RAG capabilities. Follow these steps:

1. **Analyze the question** to determine complexity
2. **Select reasoning mode**:
   - DIRECT: Simple factual queries (e.g., "What is X?")
   - VERIFY: Claims needing verification (e.g., "Is X true?")
   - RESEARCH: Complex, open-ended queries (e.g., "Compare X and Y")
   - AUTO: Let the system decide based on query analysis
3. **Execute RAG query** using selected mode
4. **Extract sources** with relevance scores
5. **Generate answer** based on retrieved context
6. **Return structured output** matching the output schema

**Input Data**:
{input_data}

**Requirements**:
1. Use the most appropriate reasoning mode for the question
2. Include relevant source excerpts (max {max_sources})
3. Calculate confidence based on source quality and relevance
4. Handle edge cases gracefully (no relevant sources, ambiguous questions)
5. Return clear, concise answers (2-4 sentences typical)

**HoloLoom RAG API**:
```python
from HoloLoom.rag import SimpleRAG

async with SimpleRAG() as rag:
    result = await rag.query(
        question=input["question"],
        mode=input.get("mode", "auto"),
        max_sources=input.get("max_sources", 5)
    )

    # result.response - LLM-generated answer
    # result.sources - Retrieved source texts
    # result.confidence - 0.0-1.0
    # result.reasoning_mode - Mode used
```

**Output Format**:
Return your results in the following JSON structure:
```json
{
  "answer": "<generated answer>",
  "sources": [{"text": "...", "relevance": 0.0-1.0, "node_id": "..."}],
  "confidence": 0.0-1.0,
  "reasoning_mode": "direct|verify|research",
  "metadata": {
    "execution_time_ms": <number>,
    "cache_hit": <boolean>,
    "total_sources_retrieved": <number>,
    "warnings": []
  }
}
```

**Quality Standards**:
- Answers must be accurate and grounded in sources
- Confidence scores must reflect source quality and relevance
- Include warnings for low-confidence answers (<0.6)
- Cite sources clearly with relevance scores
- Handle "I don't know" gracefully when sources are insufficient

**Error Handling**:
- If no relevant sources found, return confidence=0.0 with warning
- If HoloLoom unavailable, gracefully degrade (return error with helpful message)
- If question is ambiguous, use VERIFY mode to check multiple interpretations
```

## Examples

### Example 1: Simple Factual Query

**Input**:
```json
{
  "question": "What is Thompson Sampling?",
  "mode": "auto"
}
```

**Expected Output**:
```json
{
  "answer": "Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff in reinforcement learning. It balances trying new actions (exploration) with exploiting known good actions by sampling from posterior distributions over action values.",
  "sources": [
    {
      "text": "Thompson Sampling balances exploration and exploitation by sampling from posterior distributions.",
      "relevance": 0.95,
      "node_id": "thompson_sampling_001"
    },
    {
      "text": "Bayesian approach to multi-armed bandits using Beta distributions.",
      "relevance": 0.88,
      "node_id": "bandit_algorithms_023"
    }
  ],
  "confidence": 0.92,
  "reasoning_mode": "direct",
  "metadata": {
    "execution_time_ms": 145,
    "cache_hit": false,
    "total_sources_retrieved": 5,
    "warnings": []
  }
}
```

**Explanation**:
Simple factual query uses DIRECT mode (fastest, ~150ms). High confidence due to strong source relevance.

### Example 2: Verification Query

**Input**:
```json
{
  "question": "Is Thompson Sampling better than epsilon-greedy?",
  "mode": "auto"
}
```

**Expected Output**:
```json
{
  "answer": "Thompson Sampling is generally more sample-efficient than epsilon-greedy, converging faster in most scenarios. However, epsilon-greedy is simpler to implement and performs well with tuned parameters. The choice depends on the specific use case.",
  "sources": [
    {
      "text": "Thompson Sampling converges faster than epsilon-greedy in empirical studies.",
      "relevance": 0.90,
      "node_id": "algorithm_comparison_045"
    },
    {
      "text": "Epsilon-greedy requires careful tuning of exploration parameter.",
      "relevance": 0.82,
      "node_id": "epsilon_greedy_012"
    },
    {
      "text": "Both algorithms have comparable asymptotic performance.",
      "relevance": 0.78,
      "node_id": "bandit_theory_056"
    }
  ],
  "confidence": 0.85,
  "reasoning_mode": "verify",
  "metadata": {
    "execution_time_ms": 620,
    "cache_hit": false,
    "total_sources_retrieved": 8,
    "warnings": []
  }
}
```

**Explanation**:
Comparative question triggers VERIFY mode, which checks multiple sources for balanced perspective. Higher latency (~600ms) due to verification step.

### Example 3: Research Query

**Input**:
```json
{
  "question": "What are all the exploration-exploitation algorithms and their tradeoffs?",
  "mode": "auto",
  "max_sources": 10
}
```

**Expected Output**:
```json
{
  "answer": "Main exploration-exploitation algorithms include: (1) Epsilon-greedy: Simple but requires parameter tuning. (2) Thompson Sampling: Sample-efficient, Bayesian approach. (3) UCB: Theoretically optimal regret bounds. (4) Softmax: Temperature-based exploration. Each has tradeoffs in sample efficiency, computational complexity, and parameter sensitivity.",
  "sources": [
    {
      "text": "Epsilon-greedy: 10% exploration, 90% exploitation...",
      "relevance": 0.92,
      "node_id": "epsilon_greedy_001"
    },
    {
      "text": "Thompson Sampling: Bayesian posterior sampling...",
      "relevance": 0.95,
      "node_id": "thompson_sampling_001"
    },
    {
      "text": "UCB: Upper Confidence Bound with logarithmic regret...",
      "relevance": 0.89,
      "node_id": "ucb_algorithm_034"
    }
  ],
  "confidence": 0.88,
  "reasoning_mode": "research",
  "metadata": {
    "execution_time_ms": 920,
    "cache_hit": false,
    "total_sources_retrieved": 10,
    "warnings": []
  }
}
```

**Explanation**:
Comprehensive question triggers RESEARCH mode, which explores multiple sub-queries. Highest latency (~900ms) but most thorough coverage.

### Example 4: No Relevant Sources (Edge Case)

**Input**:
```json
{
  "question": "What is quantum entanglement in bandit algorithms?",
  "mode": "auto"
}
```

**Expected Output**:
```json
{
  "answer": "I don't have sufficient information about quantum entanglement in the context of bandit algorithms. The retrieved sources cover standard bandit algorithms but don't discuss quantum computing applications.",
  "sources": [],
  "confidence": 0.0,
  "reasoning_mode": "direct",
  "metadata": {
    "execution_time_ms": 95,
    "cache_hit": false,
    "total_sources_retrieved": 0,
    "warnings": ["No relevant sources found for this query"]
  }
}
```

**Explanation**:
When no relevant sources are found, skill returns confidence=0.0 with clear explanation. Graceful degradation.

### Example 5: Cached Query (Performance)

**Input**:
```json
{
  "question": "What is Thompson Sampling?",
  "mode": "auto"
}
```

**Expected Output**:
```json
{
  "answer": "Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff...",
  "sources": [...],
  "confidence": 0.92,
  "reasoning_mode": "direct",
  "metadata": {
    "execution_time_ms": 1,
    "cache_hit": true,
    "total_sources_retrieved": 5,
    "warnings": []
  }
}
```

**Explanation**:
Repeated query hits cache (100x speedup: 150ms → 1ms). Same answer, drastically reduced latency.

## Testing Checklist

Before deploying this skill, verify:

- [x] **Functionality**: All examples execute correctly
- [x] **Error Handling**: Graceful degradation for invalid inputs
- [x] **Security**: No prompt injection vulnerabilities
- [x] **Performance**: Executes within acceptable time limits
  - DIRECT mode: < 200ms
  - VERIFY mode: < 700ms
  - RESEARCH mode: < 1000ms
- [x] **Token Efficiency**: Prompt is concise (<700 tokens)
- [x] **Documentation**: All sections complete and accurate
- [x] **Dependencies**: HoloLoom RAG system available
- [x] **Edge Cases**: Handles no sources, ambiguous queries, errors
- [x] **Output Consistency**: Returns consistent format across runs
- [x] **Integration**: Works with HoloLoom memory system

## Security Considerations

**Potential Risks**:
- **Prompt Injection**: User questions could contain adversarial prompts
  - **Mitigation**: Sanitize inputs, use structured schemas
- **Information Leakage**: Sources might expose sensitive data
  - **Mitigation**: Filter sources for PII/secrets before returning
- **Resource Exhaustion**: Research mode could retrieve too many sources
  - **Mitigation**: Hard limit on max_sources (default 5, max 20)

**Data Privacy**:
- [x] Does not log sensitive user data
- [x] Does not expose internal system details
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries
- [x] Does not attempt privilege escalation
- [x] Does not modify system files outside skill scope

## Performance Characteristics

- **Expected Latency**:
  - DIRECT mode: 100-200ms (cache miss), <1ms (cache hit)
  - VERIFY mode: 500-700ms
  - RESEARCH mode: 800-1000ms
- **Token Usage**: ~600 tokens (prompt template)
- **Resource Requirements**:
  - Memory: ~50MB (HoloLoom RAG instance)
  - CPU: Low (mostly I/O bound)
  - Network: None (local HoloLoom instance)
- **Scalability**: Linear with question complexity; benefits from caching

## Maintenance Notes

**Known Limitations**:
- Requires HoloLoom RAG system (not available in all deployments)
- Performance depends on HoloLoom memory contents (quality in = quality out)
- Research mode can be slow for very broad queries (>1s)

**Future Enhancements**:
- Streaming responses for research mode (real-time progress)
- Multimodal support (query with images)
- Custom RAG configurations (embeddings, retrieval strategies)
- Integration with HoloLoom alignment framework for safety checks

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release with 4 reasoning modes

## License

MIT License - Open source as part of HoloLoom project

## Support

**Issues**: https://github.com/blakechasteen/hello-world/issues
**Documentation**: See HoloLoom/rag/README.md for RAG system details
**Contributors**: HoloLoom Team

---

## Development Notes (Internal)

**Design Decisions**:
- Auto mode selection based on query complexity (simple heuristics work well)
- Confidence scoring combines source relevance + reasoning mode
- Graceful degradation when HoloLoom unavailable (returns error, not crash)

**Alternative Approaches Considered**:
- Exposing all HoloLoom RAG parameters (rejected: too complex for users)
- Single reasoning mode only (rejected: limits flexibility)
- Synchronous API (rejected: blocks on long queries)

**Integration Points**:
- Direct integration with `HoloLoom.rag.SimpleRAG` class
- Uses HoloLoom memory system for retrieval
- Optional integration with visual compression for multimodal

**Testing Strategy**:
- Test all 4 reasoning modes (direct, verify, research, auto)
- Test edge cases (no sources, cache hits, errors)
- Performance benchmarks (target latencies)
- Integration tests with HoloLoom memory backend

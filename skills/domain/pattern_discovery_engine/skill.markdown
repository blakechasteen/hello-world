# Skill: Pattern Discovery Engine

## Metadata

- **Name**: `pattern_discovery_engine`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, memory, patterns, discovery`

## Description

**Short Description**:
Discover emergent patterns (loops, clusters, threads, resonance) in HoloLoom's knowledge graph for insight generation.

**Detailed Description**:
Knowledge graphs contain hidden structures that reveal deep insights when discovered. This skill exposes UnifiedMemory.discover_patterns() to find 4 types of emergent patterns: LOOP (cycles/feedback), CLUSTER (tightly connected groups), THREAD (narrative chains), and RESONANCE (highly activated memories). Each pattern type reveals different aspects of knowledge structure: loops show feedback systems, clusters reveal topics, threads expose causality, and resonance highlights current focus.

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
**External Dependencies**: HoloLoom.memory.unified (UnifiedMemory.discover_patterns())
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "pattern_types": ["loop", "cluster", "thread", "resonance"],
  "min_strength": "number - 0.0-1.0, minimum pattern strength (default: 0.4)",
  "max_patterns": "number - Max patterns to return (default: 10)"
}
```

## Output Schema

```json
{
  "patterns": [
    {
      "type": "loop|cluster|thread|resonance",
      "memories": ["array of memory IDs in pattern"],
      "strength": "number - 0.0-1.0",
      "description": "string - Human-readable pattern description"
    }
  ],
  "insights": ["array of meta-insights about discovered patterns"],
  "metadata": {
    "total_patterns_found": "number",
    "avg_strength": "number",
    "pattern_type_distribution": {"loop": 0, "cluster": 0, "thread": 0, "resonance": 0}
  }
}
```

## Prompt Template

```markdown
You are a pattern discovery expert analyzing HoloLoom knowledge graphs.

**Pattern Discovery Request**:
- Pattern types: {pattern_types}
- Min strength: {min_strength}
- Max patterns: {max_patterns}

**4 Pattern Types**:
- **LOOP**: Cycles and feedback loops (A -> B -> C -> A). Reveals self-reinforcing systems.
- **CLUSTER**: Tightly connected groups (high modularity). Reveals coherent topics.
- **THREAD**: Narrative chains (A -> B -> C -> D). Reveals causal sequences.
- **RESONANCE**: Highly activated memories (activation >= min_strength). Reveals current focus.

**Your Task**:
1. Use UnifiedMemory.discover_patterns() to find patterns
2. For each pattern, calculate strength:
   - LOOP: cycle_length / max_cycle_length (shorter = stronger)
   - CLUSTER: modularity score (0.0-1.0)
   - THREAD: path_length / 10 (longer = stronger, capped at 1.0)
   - RESONANCE: avg_activation across members
3. Generate human-readable descriptions
4. Provide meta-insights about pattern distribution

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"pattern_types": ["loop", "cluster"], "min_strength": 0.5}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Thread Discovery

**Input**:
```json
{"pattern_types": ["thread"], "min_strength": 0.4}
```

**Explanation**:
Discovers narrative chains in knowledge graph showing causal sequences like question→hypothesis→experiment→data→conclusion.

### Example 3: Resonance Detection

**Input**:
```json
{"pattern_types": ["resonance"], "min_strength": 0.7}
```

**Explanation**:
Finds highly activated memories (hot topics) with activation ≥0.7 indicating current focus areas in the system.

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
- **Token Usage**: ~700
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

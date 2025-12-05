# Skill: Memory Graph Navigator

## Metadata

- **Name**: `memory_graph_navigator`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, memory, navigation, graph`

## Description

**Short Description**:
Navigate HoloLoom's knowledge graph in 4 intuitive directions (forward, backward, sideways, deep) to discover connected memories and reasoning paths.

**Detailed Description**:
Memory exploration in HoloLoom is like navigating a rich semantic landscape. This skill exposes the UnifiedMemory.navigate() API, enabling directional graph traversal using spatial metaphors: FORWARD (what comes next), BACKWARD (what came before), SIDEWAYS (related alternatives), and DEEP (recursive connections). Perfect for understanding causal chains, finding related concepts, or exploring emergent knowledge structures. Each navigation returns not just a path of memories, but also insights about why these connections matter and what related concepts exist.

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

**Required Skills**: None
**External Dependencies**: HoloLoom.memory.unified (UnifiedMemory class)
**HoloLoom Integration**: UnifiedMemory.navigate() API

## Input Schema

```json
{
  "from_memory": "string - Starting memory node ID or query text",
  "direction": "forward|backward|sideways|deep",
  "steps": "number - How many hops to navigate (default: 3, max: 10)",
  "filters": {
    "edge_types": ["array (optional) - Filter by relationship types: IS_A, USES, LEADS_TO, etc."],
    "min_relevance": "number (optional) - 0.0-1.0, filter weak connections (default: 0.3)"
  }
}
```

## Output Schema

```json
{
  "path": ["array of memory IDs in traversal order"],
  "path_details": [
    {
      "node_id": "string",
      "content": "string - Memory text",
      "distance": "number - Steps from origin",
      "relevance": "number - 0.0-1.0 connection strength"
    }
  ],
  "insights": [
    "string - Human-readable insights about discovered patterns"
  ],
  "related_concepts": [
    "string - Sibling concepts not on path but related"
  ],
  "metadata": {
    "direction_used": "string - forward|backward|sideways|deep",
    "total_nodes_explored": "number",
    "path_length": "number",
    "avg_relevance": "number - 0.0-1.0",
    "cycles_detected": ["array of cycles if direction=deep"]
  }
}
```

## Prompt Template

```markdown
You are a knowledge graph navigation expert using HoloLoom's memory system.

**Navigation Request**:
- Starting from: {from_memory}
- Direction: {direction}
- Steps to navigate: {steps}
- Filters: {filters}

**Direction Semantics**:
- **FORWARD**: Follow causal chains, temporal sequences, "what comes next" (follow LEADS_TO, OCCURRED_AT edges)
- **BACKWARD**: Trace back to foundations, prerequisites, "what came before" (reverse traversal)
- **SIDEWAYS**: Find alternatives, siblings, parallel concepts (shared parents/children in taxonomy)
- **DEEP**: Explore recursive loops, cycles, strange loops (DFS for feedback patterns)

**Your Task**:
1. Navigate the knowledge graph using HoloLoom's UnifiedMemory.navigate() API
2. Extract the traversal path (ordered list of memory nodes)
3. Analyze the path for insights:
   - Causal patterns (A -> B -> C chains)
   - Emergent themes (recurring concepts)
   - Surprising connections (high semantic distance but direct graph path)
   - Knowledge gaps (missing intermediates)
4. Identify related concepts not on the path but connected to path nodes
5. Provide human-readable insights about what the navigation reveals

**Navigation Strategies**:
- FORWARD: Use breadth-first search following successor edges
- BACKWARD: Reverse graph traversal (predecessors)
- SIDEWAYS: Find nodes with shared neighbors (intersection of parent/child sets)
- DEEP: Use depth-first search to detect cycles, prioritize LEADS_TO edges

**Output Format**: Return structured JSON matching the output schema.

**Example Insights**:
- "This path reveals a causal chain: research -> hypothesis -> experiment -> data -> conclusion"
- "Sideways navigation found 5 alternative approaches to the same problem"
- "Deep exploration detected a feedback loop: learning -> practice -> improvement -> learning"
```

## Examples

### Example 1: FORWARD Navigation (Causal Chain)

**Input**:
```json
{
  "from_memory": "research_question",
  "direction": "forward",
  "steps": 5,
  "filters": {
    "edge_types": ["LEADS_TO", "OCCURRED_AT"],
    "min_relevance": 0.4
  }
}
```

**Expected Output**:
```json
{
  "path": ["research_question", "hypothesis", "experiment", "data_collection", "analysis", "conclusion"],
  "path_details": [
    {"node_id": "research_question", "content": "What causes X?", "distance": 0, "relevance": 1.0},
    {"node_id": "hypothesis", "content": "Y causes X", "distance": 1, "relevance": 0.85},
    {"node_id": "experiment", "content": "Test Y -> X relationship", "distance": 2, "relevance": 0.78},
    {"node_id": "data_collection", "content": "Measure X under Y conditions", "distance": 3, "relevance": 0.72},
    {"node_id": "analysis", "content": "Statistical analysis of results", "distance": 4, "relevance": 0.68},
    {"node_id": "conclusion", "content": "Y does cause X (p<0.05)", "distance": 5, "relevance": 0.65}
  ],
  "insights": [
    "This path reveals a classic scientific method workflow: question -> hypothesis -> test -> analyze -> conclude",
    "Each step follows naturally from the previous, showing strong causal coherence (avg relevance: 0.75)",
    "The path represents a complete research cycle from inquiry to validated knowledge"
  ],
  "related_concepts": ["alternative_hypotheses", "control_groups", "statistical_power"],
  "metadata": {
    "direction_used": "forward",
    "total_nodes_explored": 12,
    "path_length": 6,
    "avg_relevance": 0.75,
    "cycles_detected": []
  }
}
```

**Explanation**:
FORWARD navigation follows temporal/causal chains. Starting from a research question, it traces through hypothesis formation, experimentation, data collection, analysis, and conclusion. The high relevance scores (0.65-1.0) indicate a well-connected research narrative.

### Example 2: SIDEWAYS Navigation (Finding Alternatives)

**Input**:
```json
{
  "from_memory": "thompson_sampling",
  "direction": "sideways",
  "steps": 3,
  "filters": {
    "edge_types": ["IS_A"],
    "min_relevance": 0.5
  }
}
```

**Expected Output**:
```json
{
  "path": ["ucb_algorithm", "epsilon_greedy", "softmax_exploration"],
  "path_details": [
    {"node_id": "ucb_algorithm", "content": "Upper Confidence Bound exploration strategy", "distance": 1, "relevance": 0.82},
    {"node_id": "epsilon_greedy", "content": "Epsilon-greedy bandit algorithm", "distance": 1, "relevance": 0.78},
    {"node_id": "softmax_exploration", "content": "Softmax action selection", "distance": 1, "relevance": 0.71}
  ],
  "insights": [
    "Sideways navigation found 3 alternative exploration strategies, all siblings of Thompson Sampling",
    "All are bandit algorithms (shared parent: multi_armed_bandits)",
    "Represent different approaches to the exploration-exploitation tradeoff"
  ],
  "related_concepts": ["bayesian_optimization", "contextual_bandits", "regret_bounds"],
  "metadata": {
    "direction_used": "sideways",
    "total_nodes_explored": 8,
    "path_length": 3,
    "avg_relevance": 0.77,
    "cycles_detected": []
  }
}
```

**Explanation**:
SIDEWAYS navigation finds siblings (nodes with shared parents/children). Starting from Thompson Sampling, it discovers alternative bandit algorithms that solve similar problems. Useful for finding competing approaches or alternatives.

### Example 3: DEEP Navigation (Detecting Feedback Loops)

**Input**:
```json
{
  "from_memory": "learning",
  "direction": "deep",
  "steps": 10,
  "filters": {
    "min_relevance": 0.3
  }
}
```

**Expected Output**:
```json
{
  "path": ["learning", "practice", "skill_improvement", "confidence", "more_practice", "mastery", "teaching", "deeper_understanding", "learning"],
  "path_details": [
    {"node_id": "learning", "content": "Acquiring new knowledge", "distance": 0, "relevance": 1.0},
    {"node_id": "practice", "content": "Applying knowledge repeatedly", "distance": 1, "relevance": 0.88},
    {"node_id": "skill_improvement", "content": "Getting better through repetition", "distance": 2, "relevance": 0.82},
    {"node_id": "confidence", "content": "Belief in one's abilities", "distance": 3, "relevance": 0.75},
    {"node_id": "more_practice", "content": "Increased motivation to practice", "distance": 4, "relevance": 0.68},
    {"node_id": "mastery", "content": "Expert-level competence", "distance": 5, "relevance": 0.62},
    {"node_id": "teaching", "content": "Explaining to others", "distance": 6, "relevance": 0.55},
    {"node_id": "deeper_understanding", "content": "Teaching reveals gaps", "distance": 7, "relevance": 0.48},
    {"node_id": "learning", "content": "Filling those gaps", "distance": 8, "relevance": 1.0}
  ],
  "insights": [
    "Deep navigation detected a feedback loop: learning -> practice -> improvement -> confidence -> more practice -> mastery -> teaching -> deeper understanding -> learning",
    "This 'virtuous cycle' shows how learning is self-reinforcing",
    "The path length (9 nodes) and return to origin indicates a complete cycle",
    "Declining then recovering relevance (1.0 -> 0.48 -> 1.0) shows the loop structure"
  ],
  "related_concepts": ["deliberate_practice", "growth_mindset", "metacognition"],
  "metadata": {
    "direction_used": "deep",
    "total_nodes_explored": 35,
    "path_length": 9,
    "avg_relevance": 0.71,
    "cycles_detected": [["learning", "practice", "skill_improvement", "confidence", "more_practice", "mastery", "teaching", "deeper_understanding", "learning"]]
  }
}
```

**Explanation**:
DEEP navigation uses DFS to explore recursive connections and detect cycles. Starting from "learning", it discovers a feedback loop that returns to the origin. The cycle reveals how learning is self-reinforcing through practice, improvement, confidence, mastery, and teaching. Perfect for understanding complex feedback systems.

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

- **Expected Latency**: 50-200ms (depends on graph size and steps)
- **Token Usage**: ~650 tokens (prompt template)
- **Resource Requirements**: Graph access (HoloLoom memory), minimal CPU
- **Scalability**: O(steps * avg_degree) graph traversal complexity

## Maintenance Notes

**Known Limitations**:
- Requires existing HoloLoom knowledge graph (cannot navigate empty graphs)
- Max 10 steps to prevent infinite loops
- DEEP mode may be slow on large graphs with many cycles

**Future Enhancements**:
- Guided navigation (suggest promising directions based on query intent)
- Path visualization (render traversal paths as interactive graphs)
- Multi-source navigation (navigate from multiple starting points simultaneously)
- Path comparison (compare different navigation strategies)

**Changelog**:
- **v1.0.0** (2025-11-22): Initial release - 4 navigation directions, cycle detection, insights generation

## License

MIT License

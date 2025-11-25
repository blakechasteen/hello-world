# Skill: Awareness Metrics Dashboard

## Metadata

- **Name**: `awareness_metrics_dashboard`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, awareness, metrics, visualization`

## Description

**Short Description**:
Visualize awareness graph metrics (activation, coherence, temporal) for memory monitoring.

**Detailed Description**:
Monitor the mind. This skill visualizes awareness graph metrics to understand memory state: activation levels (which memories are 'awake'), coherence (how well-connected active memories are), temporal dynamics (activation decay over time), and network-wide statistics. Generates HTML dashboards with interactive charts, tracks activation trajectories, detects anomalies, and provides recommendations for memory health.

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
**External Dependencies**: HoloLoom.memory.awareness_graph (AwarenessGraph.get_metrics())
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "include_history": "boolean - Include historical metrics (default: false)",
  "history_window": "number - Hours of history (default: 24)",
  "anomaly_detection": "boolean - Detect unusual patterns (default: true)"
}
```

## Output Schema

```json
{
  "current_metrics": {
    "active_nodes": "number",
    "mean_activation": "number - 0.0-1.0",
    "global_coherence": "number - 0.0-1.0",
    "activation_std": "number",
    "network_density": "number - 0.0-1.0"
  },
  "historical_trends": [
    {"timestamp": "string", "active_nodes": "number", "coherence": "number"}
  ],
  "anomalies": [
    {"type": "sudden_drop|prolonged_low|high_variance", "severity": "LOW|MEDIUM|HIGH", "description": "string"}
  ],
  "visualizations": {
    "activation_heatmap_html": "string - HTML chart",
    "coherence_trajectory_html": "string - HTML chart",
    "network_graph_html": "string - HTML interactive graph"
  },
  "recommendations": ["array of actionable insights"]
}
```

## Prompt Template

```markdown
You are an awareness metrics visualization expert.

**Dashboard Request**:
- Include history: {include_history}
- History window: {history_window} hours
- Anomaly detection: {anomaly_detection}

**Awareness Metrics**:
- **Activation**: Memory "wakefulness" (0.0-1.0), tracks spreading activation
- **Coherence**: How well-connected active memories are (graph modularity)
- **Temporal**: Activation decay over time (5% per hour default)
- **Network**: Density, centrality, clustering coefficient

**Anomaly Types**:
- SUDDEN_DROP: Coherence drops >0.2 in <1 hour
- PROLONGED_LOW: Coherence <0.4 for >3 hours
- HIGH_VARIANCE: Activation std dev >0.15

**Your Task**:
1. Retrieve current metrics via AwarenessGraph.get_metrics()
2. If include_history, fetch historical data
3. If anomaly_detection, detect patterns
4. Generate HTML visualizations (activation heatmap, coherence trajectory, network graph)
5. Provide recommendations (e.g., "Coherence low - consider memory consolidation")

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"include_history": true, "history_window": 24, "anomaly_detection": true}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Low Coherence Warning

**Input**:
```json
{"metrics": {"activation_level": 0.85, "coherence": 0.25, "active_nodes": 50}}
```

**Explanation**:
High activation but low coherence indicates fragmented knowledge - many memories active but poorly connected, suggests need for integration.

### Example 3: Temporal Analysis

**Input**:
```json
{"metrics": {"activation_level": 0.65, "coherence": 0.80, "active_nodes": 20, "temporal_context": {"recent_shift": 0.45}}}
```

**Explanation**:
Recent shift of 0.45 indicates major context change - awareness graph adapting to new topic, shows learning in progress.

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

- **Expected Latency**: 50-200ms
- **Token Usage**: ~650
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

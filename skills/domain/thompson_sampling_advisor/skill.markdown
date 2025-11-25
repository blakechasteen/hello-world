# Skill: Thompson Sampling Advisor

## Metadata

- **Name**: `thompson_sampling_advisor`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, thompson, bandits, exploration`

## Description

**Short Description**:
Explain Thompson Sampling bandit decisions using Bayesian prior analysis.

**Detailed Description**:
Thompson Sampling balances exploration/exploitation through Bayesian priors (α, β). This skill explains bandit decisions by showing expected rewards, uncertainty levels, exploration probabilities, and why specific tools were selected. Visualizes α/β distributions, compares tools, and identifies when exploration vs exploitation dominated. Perfect for debugging policy behavior, understanding tool selection, or explaining AI exploration to users.

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
**External Dependencies**: HoloLoom.policy.unified (Thompson Sampling bandit)
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "tool_stats": [
    {"tool": "string", "alpha": "number", "beta": "number", "n_pulls": "number"}
  ],
  "selected_tool": "string - Tool that was actually selected",
  "context": "string (optional) - Query/decision context"
}
```

## Output Schema

```json
{
  "decision_explanation": "string - Why selected_tool was chosen",
  "tool_analysis": [
    {
      "tool": "string",
      "expected_reward": "number - alpha/(alpha+beta)",
      "uncertainty": "number - Variance of Beta distribution",
      "exploration_probability": "number - P(selected via exploration)",
      "sample": "number - Thompson sample drawn this decision"
    }
  ],
  "exploration_vs_exploitation": {
    "decision_type": "exploration|exploitation",
    "confidence": "number - 0.0-1.0 confidence in classification",
    "rationale": "string"
  },
  "recommendations": ["array of suggestions for tuning"]
}
```

## Prompt Template

```markdown
You are a Thompson Sampling expert explaining bandit decisions.

**Decision Context**:
- Tool stats: {tool_stats}
- Selected tool: {selected_tool}
- Context: {context}

**Thompson Sampling Mechanics**:
- Sample reward ~ Beta(α, β) for each tool
- Select tool with highest sample
- Update: Success → α++, Failure → β++
- Expected reward: E[X] = α/(α+β)
- Uncertainty: Var[X] = αβ/((α+β)²(α+β+1))

**Your Task**:
1. Calculate expected reward and uncertainty for each tool
2. Determine why selected_tool won (highest sample? highest E[X]? high uncertainty?)
3. Classify decision (exploration if high-uncertainty tool chosen, else exploitation)
4. Provide tuning recommendations (e.g., "Tool X needs more pulls for better estimates")

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"tool_stats": [{"tool": "answer", "alpha": 10, "beta": 2}, {"tool": "research", "alpha": 3, "beta": 5}], "selected_tool": "research"}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Exploration Decision

**Input**:
```json
{"action": "answer", "context": {"alpha": 5, "beta": 15, "epsilon": 0.1, "bandit_strategy": "pure_thompson"}}
```

**Explanation**:
Explains why exploration action was chosen using Thompson Sampling - Beta(5, 15) distribution suggests low success rate, triggering exploration.

### Example 3: Bayesian Blend Analysis

**Input**:
```json
{"action": "research", "context": {"alpha": 20, "beta": 5, "bandit_strategy": "bayesian_blend", "neural_confidence": 0.65}}
```

**Explanation**:
Shows how Bayesian blend combines neural prediction (0.65) with bandit prior (0.8) to make final decision - weighted 70/30 split.

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

- **Expected Latency**: 30-100ms
- **Token Usage**: ~550
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

# Skill: Recursive Refiner

## Metadata

- **Name**: `recursive_refiner`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, refinement, quality, recursive`

## Description

**Short Description**:
Multi-pass recursive refinement using ELEGANCE and VERIFY strategies for quality improvement.

**Detailed Description**:
Great answers aren't written, they're refined. This skill exposes AdvancedRefiner to iteratively improve results through multi-pass refinement. Two strategies: ELEGANCE (clarity -> simplicity -> beauty) and VERIFY (accuracy -> completeness -> consistency). Each pass improves a specific quality dimension with measurable trajectory. Perfect for low-confidence results that need polishing or complex tasks requiring multiple perspectives.

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
**External Dependencies**: HoloLoom.recursive.advanced_refiner (AdvancedRefiner)
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "initial_result": "string - Result to refine",
  "strategy": "elegance|verify|auto (auto-select based on initial_result)",
  "max_iterations": "number - Max refinement passes (default: 3, max: 5)",
  "quality_threshold": "number - Stop when quality >= threshold (default: 0.9)"
}
```

## Output Schema

```json
{
  "refined_result": "string - Final improved result",
  "quality_trajectory": ["array of quality scores per iteration"],
  "improvements": ["array of improvements made"],
  "iterations_used": "number",
  "strategy_used": "elegance|verify",
  "metadata": {
    "initial_quality": "number",
    "final_quality": "number",
    "improvement_delta": "number"
  }
}
```

## Prompt Template

```markdown
You are a recursive refinement expert improving results iteratively.

**Refinement Request**:
- Initial result: {initial_result}
- Strategy: {strategy}
- Max iterations: {max_iterations}
- Quality threshold: {quality_threshold}

**Refinement Strategies**:
- **ELEGANCE**: Clarity (pass 1) -> Simplicity (pass 2) -> Beauty (pass 3)
- **VERIFY**: Accuracy (pass 1) -> Completeness (pass 2) -> Consistency (pass 3)
- **AUTO**: Choose ELEGANCE if initial quality < 0.7, else VERIFY

**Quality Scoring**:
quality = 0.7 * confidence + 0.2 * context_richness + 0.1 * completeness

**Your Task**:
1. Assess initial quality (0.0-1.0)
2. Select strategy (if auto)
3. For each iteration:
   - Apply strategy-specific improvements
   - Calculate new quality score
   - Stop if quality >= threshold
4. Track quality trajectory and improvements

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"initial_result": "Thompson Sampling is a bandit algorithm.", "strategy": "elegance", "max_iterations": 3}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Verify Mode Refinement

**Input**:
```json
{"initial_result": {"response": "Python uses dynamic typing", "confidence": 0.65}, "strategy": "verify", "max_iterations": 3}
```

**Explanation**:
Uses VERIFY strategy (accuracy→completeness→consistency passes) to improve factual answer quality through multi-pass refinement.

### Example 3: Auto-Strategy Selection

**Input**:
```json
{"initial_result": {"response": "Short answer", "confidence": 0.55}, "strategy": null, "max_iterations": 5}
```

**Explanation**:
Automatically selects best refinement strategy based on query characteristics and iterates until quality threshold reached or max iterations.

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

- **Expected Latency**: 200-500ms
- **Token Usage**: ~750
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

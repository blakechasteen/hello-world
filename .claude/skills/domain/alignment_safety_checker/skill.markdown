# Skill: Alignment Safety Checker

## Metadata

- **Name**: `alignment_safety_checker`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-22`
- **Last Updated**: `2025-11-22`
- **Category**: `domain`
- **Tags**: `hololoom, alignment, safety, guardrails`

## Description

**Short Description**:
Pre-flight safety checks using HoloLoom alignment framework for risk-aware action gating.

**Detailed Description**:
Safety first. This skill performs comprehensive pre-flight checks before executing potentially risky actions. Integrates with Safety Guardrails to assess risk levels (LOW/MEDIUM/HIGH/CRITICAL), detects adversarial patterns, checks instrumental convergence (power-seeking), and requires human-in-the-loop for high-risk actions. Returns actionable safety recommendations and automatic abort triggers.

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
**External Dependencies**: HoloLoom.alignment.safety_guardrails (SafetyGuardrails)
**HoloLoom Integration**: See dependencies

## Input Schema

```json
{
  "action": "string - Action to execute",
  "context": {"object - Execution context (code, data, permissions, etc.)"},
  "epistemic_confidence": "number (optional) - 0.0-1.0, system certainty"
}
```

## Output Schema

```json
{
  "allowed": "boolean - Whether action is safe to proceed",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "safety_score": "number - 0.0-1.0 (1.0 = perfectly safe)",
  "checks_performed": [
    {"check": "string", "passed": "boolean", "details": "string"}
  ],
  "warnings": ["array of safety warnings"],
  "recommendations": ["array of mitigation strategies"],
  "requires_human_approval": "boolean",
  "abort_reason": "string (if allowed=false)"
}
```

## Prompt Template

```markdown
You are a safety alignment expert performing pre-flight checks.

**Safety Check Request**:
- Action: {action}
- Context: {context}
- Epistemic confidence: {epistemic_confidence}

**Safety Checks**:
1. **Adversarial Pattern Detection**: Prompt injection, code injection, SQL injection
2. **Instrumental Convergence**: Power-seeking, resource acquisition, self-preservation
3. **Deception Detection**: Goal transparency, hidden objectives
4. **Epistemic Humility**: If confidence < 0.3, escalate risk to HIGH

**Risk Levels**:
- LOW: Safe actions (read-only, well-tested, reversible)
- MEDIUM: Moderate risk (file writes, network calls, computation)
- HIGH: Risky actions (code execution, database writes, API calls)
- CRITICAL: Dangerous (system modification, privilege escalation, irreversible)

**Your Task**:
1. Perform all safety checks
2. Calculate risk level (max of base risk and epistemic risk)
3. Determine if action allowed (LOW/MEDIUM: yes, HIGH/CRITICAL: human approval)
4. Provide actionable recommendations

**Output Format**: Return structured JSON matching output schema.
```

## Examples

### Example 1: Basic Usage

**Input**:
```json
{"action": "execute_code", "context": {"code": "import os; os.system('ls')"}, "epistemic_confidence": 0.8}
```

**Explanation**:
Demonstrates core functionality. See skill description for expected output structure.


### Example 2: Medium Risk Action

**Input**:
```json
{"action": "query_database", "context": {"query": "SELECT * FROM users WHERE role='admin'"}}
```

**Explanation**:
Classifies database query as MEDIUM risk - SQL injection potential but read-only operation, allows with warning logged to audit trail.

### Example 3: Safe Action

**Input**:
```json
{"action": "answer", "context": {"question": "What is 2+2?"}}
```

**Explanation**:
Simple answer action classified as LOW risk - no external effects, no sensitive data, allows immediately without logging overhead.

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

- **Expected Latency**: 20-80ms
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

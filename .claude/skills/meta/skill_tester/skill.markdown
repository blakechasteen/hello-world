# Skill: Skill Tester

## Metadata

- **Name**: `skill_tester`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-01-18`
- **Last Updated**: `2025-01-18`
- **Category**: `meta`
- **Tags**: `testing, validation, quality-assurance, meta-skill`

## Description

**Short Description**:
Runs comprehensive test suites against Claude skills to validate functionality, performance, error handling, and output consistency.

**Detailed Description**:
The Skill Tester is a meta-skill that automates quality assurance for other skills. It executes test cases defined in the skill's Examples section, validates outputs match expected schemas, measures performance metrics (latency, token usage), and checks for edge case handling. The tester generates detailed test reports with pass/fail status, performance benchmarks, and actionable recommendations for improvement. It supports both smoke testing (quick validation) and comprehensive testing (full test suite).

## Required Capabilities

- [x] File system access (read) - to read skill files and test cases
- [x] File system access (write) - to write test reports
- [ ] Code execution (bash)
- [x] Code execution (python) - to execute skill tests
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [x] User interaction (questions) - for interactive test selection

## Dependencies

**Required Skills**: None (base meta-skill)

**External Dependencies**:
- `pytest` (optional, for structured test execution)
- `json-schema` (optional, for output validation)

**HoloLoom Integration**:
- [ ] Uses HoloLoom memory system
- [ ] Uses HoloLoom RAG
- [ ] Uses HoloLoom alignment framework
- [x] Uses HoloLoom learning systems - learns from test failures to suggest improvements

## Input Schema

**Expected Input Format**:
```json
{
  "skill_path": "path to skill.markdown file or directory",
  "test_mode": "smoke|comprehensive|specific",
  "test_cases": ["example1", "example2"],
  "performance_thresholds": {
    "max_latency_ms": 5000,
    "max_tokens": 10000
  },
  "output_format": "json|html|markdown"
}
```

**Example Input**:
```json
{
  "skill_path": "skills/meta/skill_security_analyzer/skill.markdown",
  "test_mode": "comprehensive",
  "test_cases": null,
  "performance_thresholds": {
    "max_latency_ms": 1000,
    "max_tokens": 5000
  },
  "output_format": "json"
}
```

## Output Schema

**Expected Output Format**:
```json
{
  "skill_name": "name of tested skill",
  "test_summary": {
    "total_tests": 10,
    "passed": 8,
    "failed": 2,
    "skipped": 0,
    "success_rate": 0.80
  },
  "test_results": [
    {
      "test_name": "example1_basic_usage",
      "status": "pass|fail|skip",
      "actual_output": {},
      "expected_output": {},
      "differences": [],
      "latency_ms": 150,
      "tokens_used": 1200,
      "error_message": null
    }
  ],
  "performance_metrics": {
    "avg_latency_ms": 180,
    "p95_latency_ms": 320,
    "avg_tokens": 1500,
    "total_execution_time_ms": 1800
  },
  "recommendations": [
    "Consider optimizing prompt for token efficiency",
    "Add error handling for edge case X"
  ],
  "quality_score": 0.85,
  "ready_for_deployment": true,
  "metadata": {
    "execution_time_ms": 2000,
    "confidence": 0.92,
    "warnings": []
  }
}
```

## Prompt Template

```markdown
You are the **Skill Tester**, a meta-skill that validates other Claude skills through automated testing.

**Your Task**:
Execute test cases for the provided skill and generate a comprehensive test report.

**Input Data**:
{input_data}

**Skill Content**:
{skill_content}

**Testing Steps**:

1. **Extract Test Cases**: Parse skill.markdown Examples section
2. **For Each Test Case**:
   - Execute with provided input
   - Capture actual output
   - Compare with expected output
   - Measure latency and token usage
   - Record pass/fail status
3. **Validate Output Schema**: Ensure outputs match declared schema
4. **Check Error Handling**: Test edge cases and invalid inputs
5. **Performance Analysis**: Calculate avg/p95 latency, token efficiency
6. **Generate Report**: Detailed results with recommendations

**Test Modes**:
- **smoke**: Run 1-2 basic tests for quick validation
- **comprehensive**: Run all examples + edge cases
- **specific**: Run only specified test cases

**Quality Criteria**:
- **Pass**: Output matches expected, within performance thresholds
- **Fail**: Output mismatch, error, or performance violation
- **Skip**: Test not applicable or dependencies missing

**Output Format**:
Return test results as JSON (see Output Schema above).

**Recommendations**:
For failed tests, provide:
- Root cause analysis
- Specific fix suggestions
- Example of corrected output

**Error Handling**:
- If skill.markdown has no examples, report warning and suggest adding them
- If skill execution errors, capture stack trace and suggest fixes
- If performance thresholds not met, recommend optimizations
```

## Examples

### Example 1: Comprehensive Testing

**Input**:
```json
{
  "skill_path": "skills/meta/skill_security_analyzer/skill.markdown",
  "test_mode": "comprehensive",
  "test_cases": null,
  "performance_thresholds": {
    "max_latency_ms": 1000,
    "max_tokens": 5000
  },
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "skill_name": "skill_security_analyzer",
  "test_summary": {
    "total_tests": 3,
    "passed": 3,
    "failed": 0,
    "skipped": 0,
    "success_rate": 1.0
  },
  "test_results": [
    {
      "test_name": "example1_safe_skill",
      "status": "pass",
      "latency_ms": 120,
      "tokens_used": 1800,
      "error_message": null
    },
    {
      "test_name": "example2_prompt_injection",
      "status": "pass",
      "latency_ms": 200,
      "tokens_used": 2100,
      "error_message": null
    },
    {
      "test_name": "example3_data_leak",
      "status": "pass",
      "latency_ms": 150,
      "tokens_used": 1900,
      "error_message": null
    }
  ],
  "performance_metrics": {
    "avg_latency_ms": 157,
    "p95_latency_ms": 200,
    "avg_tokens": 1933,
    "total_execution_time_ms": 470
  },
  "recommendations": [],
  "quality_score": 1.0,
  "ready_for_deployment": true,
  "metadata": {
    "execution_time_ms": 500,
    "confidence": 0.98,
    "warnings": []
  }
}
```

**Explanation**:
All tests pass, performance within thresholds, skill ready for deployment.

### Example 2: Failed Test with Recommendations

**Input**:
```json
{
  "skill_path": "skills/domain/buggy_skill/skill.markdown",
  "test_mode": "smoke",
  "test_cases": ["example1"],
  "performance_thresholds": {
    "max_latency_ms": 2000,
    "max_tokens": 8000
  },
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "skill_name": "buggy_skill",
  "test_summary": {
    "total_tests": 1,
    "passed": 0,
    "failed": 1,
    "skipped": 0,
    "success_rate": 0.0
  },
  "test_results": [
    {
      "test_name": "example1_basic_usage",
      "status": "fail",
      "actual_output": {"result": "wrong_format"},
      "expected_output": {"result": "correct_format", "metadata": {}},
      "differences": ["Missing 'metadata' field in output"],
      "latency_ms": 250,
      "tokens_used": 1500,
      "error_message": "Output schema validation failed"
    }
  ],
  "performance_metrics": {
    "avg_latency_ms": 250,
    "p95_latency_ms": 250,
    "avg_tokens": 1500,
    "total_execution_time_ms": 250
  },
  "recommendations": [
    "Add 'metadata' field to output schema in skill.markdown",
    "Update prompt template to always include metadata",
    "Add test case to validate metadata presence"
  ],
  "quality_score": 0.3,
  "ready_for_deployment": false,
  "metadata": {
    "execution_time_ms": 300,
    "confidence": 0.95,
    "warnings": ["Skill has schema inconsistencies - fix before deploying"]
  }
}
```

**Explanation**:
Test fails due to output schema mismatch, provides specific fix recommendations.

### Example 3: Performance Threshold Violation

**Input**:
```json
{
  "skill_path": "skills/domain/slow_skill/skill.markdown",
  "test_mode": "smoke",
  "test_cases": null,
  "performance_thresholds": {
    "max_latency_ms": 500,
    "max_tokens": 2000
  },
  "output_format": "json"
}
```

**Expected Output**:
```json
{
  "skill_name": "slow_skill",
  "test_summary": {
    "total_tests": 1,
    "passed": 0,
    "failed": 1,
    "skipped": 0,
    "success_rate": 0.0
  },
  "test_results": [
    {
      "test_name": "example1",
      "status": "fail",
      "latency_ms": 1200,
      "tokens_used": 3500,
      "error_message": "Performance threshold violated: latency 1200ms > 500ms, tokens 3500 > 2000"
    }
  ],
  "performance_metrics": {
    "avg_latency_ms": 1200,
    "p95_latency_ms": 1200,
    "avg_tokens": 3500,
    "total_execution_time_ms": 1200
  },
  "recommendations": [
    "Reduce prompt verbosity to decrease token usage",
    "Consider caching common operations",
    "Simplify instructions for faster execution",
    "Run token_budget_adviser skill for optimization suggestions"
  ],
  "quality_score": 0.4,
  "ready_for_deployment": false,
  "metadata": {
    "execution_time_ms": 1300,
    "confidence": 0.88,
    "warnings": ["Performance thresholds exceeded - optimize before deployment"]
  }
}
```

**Explanation**:
Skill exceeds latency and token budgets, recommends optimizations.

## Testing Checklist

- [x] **Functionality**: Executes all test types (smoke, comprehensive, specific)
- [x] **Error Handling**: Gracefully handles missing examples, malformed skills
- [x] **Security**: Self-test passes (meta-skill testing itself)
- [x] **Performance**: Completes testing in reasonable time (<10s for comprehensive)
- [x] **Token Efficiency**: Minimal overhead for test orchestration
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: Works with or without pytest
- [x] **Edge Cases**: Handles empty test suites, infinite loops (timeout)
- [x] **Output Consistency**: Consistent JSON reports
- [x] **Integration**: Integrates with build pipeline and CI/CD

## Security Considerations

**Potential Risks**:
- **Code Execution**: Executes untrusted skill code - run in sandboxed environment
- **Resource Exhaustion**: Malicious skills could consume resources - enforce timeouts

**Data Privacy**:
- [x] Does not log sensitive test data
- [x] Test reports exclude user data
- [x] No external requests

**Sandboxing**:
- [x] Enforces execution timeouts (default: 30s per test)
- [x] Resource limits on memory/CPU
- [x] Isolated test environment

## Performance Characteristics

- **Expected Latency**: 100ms-10s depending on test suite size
- **Token Usage**: ~500-2000 tokens per test case
- **Resource Requirements**: Moderate (executes skill code)
- **Scalability**: Linear with number of test cases

## Maintenance Notes

**Known Limitations**:
- Cannot test skills requiring human interaction (yet)
- Limited to testing examples provided in skill.markdown
- Performance metrics approximate (vary by system load)

**Future Enhancements**:
- Fuzzing/property-based testing integration
- Automated test case generation
- Continuous testing in production
- Integration with HoloLoom learning to auto-improve tests

**Changelog**:
- **v1.0.0** (2025-01-18): Initial release

## License

MIT License (part of HoloLoom ecosystem)

## Support

**Issues**: https://github.com/yourusername/hello-world/issues
**Documentation**: See skills/docs/testing_guide.md
**Contributors**: HoloLoom Team

---

## Development Notes (Internal)

**Design Decisions**:
- Support multiple test modes (smoke/comprehensive) for flexibility
- JSON output format for CI/CD integration
- Performance thresholds configurable per skill

**Alternative Approaches Considered**:
- pytest plugin (too heavyweight, requires Python setup)
- Manual testing checklist (not automated, error-prone)
- LLM-based test generation (future enhancement)

**Integration Points**:
- Called by build_skill.py before packaging
- Can run in CI/CD pipeline
- Integrates with skill_security_analyzer for comprehensive validation

**Testing Strategy**:
- Self-test: skill_tester tests itself (bootstrap test)
- Test all meta-skills with skill_tester
- Measure false positive/negative rates

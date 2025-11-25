# Skill: Pytest Runner

## Metadata

- **Name**: `pytest_runner`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `testing`
- **Tags**: `testing, pytest, coverage, quality, automation`

## Description

**Short Description**:
Run pytest tests with coverage and detailed reporting for automated quality assurance.

**Detailed Description**:
The Pytest Runner skill wraps pytest to provide comprehensive automated testing capabilities. It supports running all tests, specific tests, marker-based filtering, parallel execution, and coverage reporting with configurable thresholds. Returns detailed test results including passed/failed/skipped counts, duration, coverage percentage, and failure analysis. Integrates with HoloLoom's quality assurance workflows for continuous testing and validation.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [ ] File system access (write)
- [x] Code execution (python)
- [ ] Network access (web fetch)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `pytest` (required)
- `pytest-cov` (for coverage reporting)
- `pytest-xdist` (optional, for parallel execution)

**HoloLoom Integration**: Works standalone or integrates with HoloLoom's CI/CD and quality assurance systems.

## Input Schema

```json
{
  "operation": "string - run_all|run_with_coverage|run_specific|run_marker|run_parallel",
  "parameters": {
    "path": "string (optional) - Test directory or file path (default: '.')",
    "marker": "string (required for run_marker) - Pytest marker to filter by",
    "num_workers": "number (required for run_parallel) - Number of parallel workers",
    "min_coverage": "number (optional) - Minimum coverage threshold % (default: 80)"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|partial|error",
  "result": {
    "passed": "number - Tests passed",
    "failed": "number - Tests failed",
    "skipped": "number - Tests skipped",
    "errors": "number - Test errors",
    "total": "number - Total tests run",
    "duration_seconds": "number - Execution time",
    "coverage_percent": "number (optional) - Coverage percentage",
    "success_rate": "number - passed/total ratio",
    "failed_tests": "array (optional) - List of failed test names",
    "warnings": "array (optional) - List of warnings"
  },
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "passed": "number",
    "failed": "number",
    "skipped": "number",
    "errors": "number",
    "total": "number",
    "duration_seconds": "number",
    "success_rate": "number",
    "coverage_percent": "number (optional)",
    "failed_tests": "array (optional)"
  },
  "warnings": "array - Pytest warnings",
  "errors": "array - Execution errors"
}
```

## Prompt Template

```markdown
You are a test automation expert using HoloLoom's Pytest Runner skill.

**Operation**: {operation}
**Parameters**: {parameters}

**Available Operations**:
1. **run_all** - Run all tests in directory
2. **run_with_coverage** - Run tests with coverage report (min threshold: 80% default)
3. **run_specific** - Run specific test file or function
4. **run_marker** - Run tests with specific marker (e.g., slow, integration, unit)
5. **run_parallel** - Run tests in parallel (requires pytest-xdist)

**Pytest Markers**:
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (multi-component)
- `@pytest.mark.e2e` - End-to-end tests (full pipeline)
- `@pytest.mark.slow` - Slow tests (>5 seconds)
- `@pytest.mark.skip` - Skip tests
- `@pytest.mark.parametrize` - Parameterized tests

**Coverage Thresholds**:
- **Excellent**: ≥90% coverage
- **Good**: 80-89% coverage
- **Acceptable**: 70-79% coverage
- **Poor**: <70% coverage (needs improvement)

**Your Task**:
1. Validate operation and parameters
2. Run pytest with appropriate flags
3. Parse test results (passed, failed, skipped, errors)
4. Calculate success rate and coverage (if enabled)
5. Extract failed test names and warnings
6. Return structured results with actionable insights

**Output Format**: Return structured JSON matching output schema.

**Quality Criteria**:
- All test counts accurate (passed, failed, skipped, errors)
- Coverage percentage calculated correctly (if applicable)
- Failed tests listed with full names
- Warnings and errors captured
- Execution time reported in milliseconds
- Actionable recommendations for failures
```

## Examples

### Example 1: Run All Tests

**Input**:
```json
{
  "operation": "run_all",
  "parameters": {
    "path": "HoloLoom/tests/unit/"
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "passed": 45,
    "failed": 0,
    "skipped": 2,
    "errors": 0,
    "total": 47,
    "duration_seconds": 3.21,
    "success_rate": 0.957
  },
  "message": "Tests passed: 45/47 (95.7%)",
  "execution_time_ms": 3250,
  "details": {
    "passed": 45,
    "failed": 0,
    "skipped": 2,
    "total": 47,
    "success_rate": 0.957
  }
}
```

**Explanation**:
Runs all unit tests. Returns detailed counts and success rate. 2 tests skipped (likely conditional or platform-specific).

### Example 2: Run with Coverage Threshold

**Input**:
```json
{
  "operation": "run_with_coverage",
  "parameters": {
    "path": "HoloLoom/policy/",
    "min_coverage": 85
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "passed": 23,
    "failed": 0,
    "skipped": 0,
    "errors": 0,
    "total": 23,
    "duration_seconds": 5.67,
    "coverage_percent": 87.5,
    "success_rate": 1.0
  },
  "message": "Tests passed: 23/23 (100.0%)",
  "execution_time_ms": 5720,
  "details": {
    "passed": 23,
    "total": 23,
    "coverage_percent": 87.5,
    "success_rate": 1.0
  }
}
```

**Explanation**:
Runs tests with coverage reporting. Coverage threshold set to 85%, actual coverage 87.5% (passes). All tests pass.

### Example 3: Run Specific Test with Failure

**Input**:
```json
{
  "operation": "run_specific",
  "parameters": {
    "path": "HoloLoom/tests/integration/test_memory.py::test_graph_persistence"
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "failure",
  "result": {
    "passed": 0,
    "failed": 1,
    "skipped": 0,
    "errors": 0,
    "total": 1,
    "duration_seconds": 0.45,
    "success_rate": 0.0,
    "failed_tests": [
      "test_memory.py::test_graph_persistence"
    ]
  },
  "message": "Tests failed: 1 failed, 0 errors",
  "execution_time_ms": 480,
  "details": {
    "failed": 1,
    "total": 1,
    "success_rate": 0.0,
    "failed_tests": [
      "test_memory.py::test_graph_persistence"
    ]
  }
}
```

**Explanation**:
Runs single specific test. Test fails. Returns full test name in `failed_tests` for debugging.

### Example 4: Run Tests by Marker (Integration Tests)

**Input**:
```json
{
  "operation": "run_marker",
  "parameters": {
    "path": "HoloLoom/tests/",
    "marker": "integration"
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "passed": 12,
    "failed": 0,
    "skipped": 0,
    "errors": 0,
    "total": 12,
    "duration_seconds": 8.92,
    "success_rate": 1.0
  },
  "message": "Tests passed: 12/12 (100.0%)",
  "execution_time_ms": 9010,
  "details": {
    "passed": 12,
    "total": 12,
    "success_rate": 1.0
  }
}
```

**Explanation**:
Runs only tests marked with `@pytest.mark.integration`. Useful for filtering test suites by speed/complexity.

### Example 5: Parallel Test Execution

**Input**:
```json
{
  "operation": "run_parallel",
  "parameters": {
    "path": "HoloLoom/tests/",
    "num_workers": 4
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "passed": 120,
    "failed": 0,
    "skipped": 5,
    "errors": 0,
    "total": 125,
    "duration_seconds": 12.34,
    "success_rate": 0.96
  },
  "message": "Tests passed: 120/125 (96.0%)",
  "execution_time_ms": 12480,
  "details": {
    "passed": 120,
    "skipped": 5,
    "total": 125,
    "success_rate": 0.96
  }
}
```

**Explanation**:
Runs tests in parallel with 4 workers. Significant speedup for large test suites (requires pytest-xdist).

## Testing Checklist

Before deploying this skill, verify:

- [x] **Functionality**: All 5 operations execute correctly
- [x] **Error Handling**: Graceful handling of pytest failures
- [x] **Security**: No command injection vulnerabilities
- [x] **Performance**: Tests complete within expected time
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: pytest, pytest-cov documented
- [x] **Edge Cases**: Handles empty test suites, all tests skipped
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom CI/CD if enabled

## Security Considerations

**Potential Risks**:
- **Command Injection**: Path parameters could contain shell commands → Sanitize paths, use subprocess arrays
- **Arbitrary Code Execution**: Running user tests → Expected behavior for testing skill
- **Resource Exhaustion**: Long-running test suites → Use timeout parameter

**Data Privacy**:
- [x] Does not log sensitive test data
- [x] Does not expose internal system details (only test results)
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries (code execution, file read)
- [x] Does not attempt privilege escalation
- [x] Does not modify system files outside test scope

## Performance Characteristics

- **Expected Latency**: 1000-30000ms (1-30 seconds depending on test suite size)
- **Token Usage**:
  - Input: 50-200 tokens (operation + parameters)
  - Output: 50-300 tokens (structured results)
  - Total: 100-500 tokens per execution
- **Resource Requirements**:
  - pytest (required)
  - pytest-cov (optional, for coverage)
  - pytest-xdist (optional, for parallel execution)
- **Scalability**: Parallel execution scales linearly with workers

## Maintenance Notes

**Known Limitations**:
- Requires pytest installed in environment
- Coverage reporting requires pytest-cov
- Parallel execution requires pytest-xdist
- Output parsing depends on pytest output format (may break with major pytest version changes)

**Future Enhancements**:
- **JSON output mode** - Use pytest --json-report for structured output
- **Test selection strategies** - Smart test selection based on code changes
- **Historical trend analysis** - Track test success rates over time
- **Failure categorization** - Classify failures (assertion, timeout, import error, etc.)
- **Coverage diff** - Show coverage changes from baseline
- **Flaky test detection** - Identify non-deterministic tests

**Changelog**:
- **v1.0.0** (2025-11-24): Initial release
  - 5 operations (run_all, run_with_coverage, run_specific, run_marker, run_parallel)
  - Coverage reporting with thresholds
  - Failure analysis with test names
  - Warning extraction
  - Success rate calculation

## Usage Examples (Claude Code)

### Quick Test Run
```
Use pytest_runner to run all unit tests in HoloLoom/tests/unit/
```

### With Coverage
```
Use pytest_runner with operation=run_with_coverage and min_coverage=85 to test HoloLoom/policy/ with coverage
```

### Specific Test
```
Use pytest_runner with operation=run_specific to run HoloLoom/tests/integration/test_memory.py::test_graph_persistence
```

### Marker-Based
```
Use pytest_runner with operation=run_marker and marker=integration to run all integration tests
```

### Parallel Execution
```
Use pytest_runner with operation=run_parallel and num_workers=4 to run all tests in parallel
```

## Integration with HoloLoom Systems

This skill integrates with:

1. **Quality Assurance Department** - Automated test execution in QA workflows
2. **CI/CD Pipeline** - Pre-commit and pre-push test validation
3. **GitHub Actions** - Integration with CI workflows
4. **Alignment Framework** - Test coverage for safety-critical code
5. **Monitoring** - Test health tracking over time

## License

MIT License

## Related Documentation

- **Pytest Documentation**: [pytest.org](https://docs.pytest.org/)
- **Coverage.py**: [coverage.readthedocs.io](https://coverage.readthedocs.io/)
- **pytest-xdist**: [pytest-xdist.readthedocs.io](https://pytest-xdist.readthedocs.io/)
- **HoloLoom Testing Strategy**: [CLAUDE.md](../../../CLAUDE.md) (Testing section)

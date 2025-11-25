# Skill: GitHub Actions

## Metadata

- **Name**: `github_actions`
- **Version**: `1.0.0`
- **Author**: `HoloLoom Team`
- **Created**: `2025-11-24`
- **Last Updated**: `2025-11-24`
- **Category**: `infrastructure`
- **Tags**: `ci, cd, github, automation, workflows, devops`

## Description

**Short Description**:
Complete CI/CD workflow automation for GitHub Actions via GitHub CLI.

**Detailed Description**:
The GitHub Actions skill provides comprehensive automation for CI/CD workflows including triggering, monitoring, status checking, log retrieval, artifact downloading, and workflow management. Wraps GitHub CLI (gh) for seamless integration with GitHub Actions. Enables automated testing, deployment, and workflow orchestration directly from HoloLoom. Supports all major GitHub Actions operations with structured output and error handling.

## Required Capabilities

Check all capabilities this skill requires:

- [x] File system access (read)
- [ ] File system access (write)
- [x] Code execution (bash)
- [x] Network access (GitHub API)
- [ ] Network access (web search)
- [ ] MCP server access
- [ ] External API access
- [ ] User interaction (questions)

## Dependencies

**Required Skills**: None
**External Dependencies**:
- `gh` (GitHub CLI) - Required for all operations
- GitHub authentication (gh auth login)
- Git repository context (for auto-repo detection)

**HoloLoom Integration**: Integrates with HoloLoom's CI/CD pipeline, quality assurance workflows, and deployment automation.

## Input Schema

```json
{
  "operation": "string - trigger_workflow|list_workflows|get_workflow_status|get_workflow_logs|cancel_workflow|list_runs|download_artifact|enable_workflow|disable_workflow",
  "parameters": {
    "workflow": "string (required for trigger/enable/disable) - Workflow name or ID",
    "ref": "string (optional, default: main) - Git ref to run workflow on",
    "inputs": "object (optional) - Workflow input parameters",
    "run_id": "string (required for status/logs/cancel/download) - Workflow run ID",
    "limit": "number (optional, default: 10) - Max results to return",
    "artifact_name": "string (required for download_artifact) - Artifact name",
    "output_dir": "string (optional, default: .) - Directory for artifact download"
  }
}
```

## Output Schema

```json
{
  "status": "string - success|failure|error",
  "result": "object|array - Operation-specific result",
  "message": "string - Human-readable summary",
  "execution_time_ms": "number - Skill execution time",
  "details": {
    "operation": "string - Operation performed",
    "workflow": "string (optional) - Workflow name",
    "run_id": "string (optional) - Run ID",
    "status": "string (optional) - Workflow status",
    "conclusion": "string (optional) - Workflow conclusion"
  },
  "warnings": "array - Any warnings",
  "errors": "array - Execution errors"
}
```

## Prompt Template

```markdown
You are a CI/CD automation expert using HoloLoom's GitHub Actions skill.

**Operation**: {operation}
**Parameters**: {parameters}

**Available Operations**:
1. **trigger_workflow** - Trigger a workflow run with optional inputs
2. **list_workflows** - List all workflows in repository
3. **get_workflow_status** - Get status of specific workflow run
4. **get_workflow_logs** - Download complete workflow logs
5. **cancel_workflow** - Cancel a running workflow
6. **list_runs** - List recent workflow runs (optionally filtered by workflow)
7. **download_artifact** - Download workflow artifacts
8. **enable_workflow** - Enable a disabled workflow
9. **disable_workflow** - Disable a workflow

**Workflow Statuses**:
- **queued** - Waiting to start
- **in_progress** - Currently running
- **completed** - Finished (check conclusion)
- **success** - Completed successfully
- **failure** - Completed with failures
- **cancelled** - Cancelled by user
- **skipped** - Skipped

**GitHub CLI Requirements**:
- `gh` must be installed and authenticated (`gh auth login`)
- Repository context auto-detected from git remote
- Or specify repository with `--repo owner/repo` parameter

**Your Task**:
1. Validate operation and parameters
2. Execute GitHub CLI command with appropriate flags
3. Parse JSON/text output from gh
4. Extract relevant information (run IDs, statuses, logs)
5. Structure results for easy consumption
6. Return actionable insights

**Output Format**: Return structured JSON matching output schema.

**Best Practices**:
- Check workflow status before triggering
- Monitor long-running workflows with polling
- Download artifacts for debugging failures
- Cancel workflows that timeout
- Enable/disable workflows for maintenance
```

## Examples

### Example 1: Trigger Workflow

**Input**:
```json
{
  "operation": "trigger_workflow",
  "parameters": {
    "workflow": "ci.yml",
    "ref": "main",
    "inputs": {
      "environment": "staging",
      "run_tests": "true"
    }
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "workflow": "ci.yml",
    "ref": "main",
    "inputs": {
      "environment": "staging",
      "run_tests": "true"
    },
    "message": "Workflow 'ci.yml' triggered on ref 'main'"
  },
  "message": "GitHub Actions operation 'trigger_workflow' completed successfully",
  "execution_time_ms": 1250
}
```

**Explanation**:
Triggers the CI workflow on main branch with custom inputs for environment and test execution.

### Example 2: Get Workflow Status

**Input**:
```json
{
  "operation": "get_workflow_status",
  "parameters": {
    "run_id": "7654321"
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "run_id": "7654321",
    "workflow_name": "CI Pipeline",
    "status": "completed",
    "conclusion": "success",
    "created_at": "2025-11-24T10:00:00Z",
    "updated_at": "2025-11-24T10:05:30Z",
    "html_url": "https://github.com/owner/repo/actions/runs/7654321",
    "run_number": 42,
    "event": "push",
    "head_branch": "main",
    "head_sha": "abc123..."
  },
  "message": "GitHub Actions operation 'get_workflow_status' completed successfully",
  "execution_time_ms": 850
}
```

**Explanation**:
Retrieves complete status information for a specific workflow run including conclusion, timing, and Git context.

### Example 3: List Recent Workflow Runs

**Input**:
```json
{
  "operation": "list_runs",
  "parameters": {
    "workflow": "ci.yml",
    "limit": 5
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": [
    {
      "databaseId": 7654321,
      "workflowName": "CI Pipeline",
      "status": "completed",
      "conclusion": "success",
      "createdAt": "2025-11-24T10:00:00Z",
      "url": "https://github.com/owner/repo/actions/runs/7654321",
      "number": 42,
      "event": "push",
      "headBranch": "main"
    },
    {
      "databaseId": 7654320,
      "workflowName": "CI Pipeline",
      "status": "completed",
      "conclusion": "failure",
      "createdAt": "2025-11-24T09:00:00Z",
      "url": "https://github.com/owner/repo/actions/runs/7654320",
      "number": 41,
      "event": "push",
      "headBranch": "feat/new-feature"
    }
  ],
  "message": "GitHub Actions operation 'list_runs' completed successfully",
  "execution_time_ms": 1100
}
```

**Explanation**:
Lists the 5 most recent runs of the CI workflow with status, conclusion, and URLs for quick access.

### Example 4: Download Workflow Logs

**Input**:
```json
{
  "operation": "get_workflow_logs",
  "parameters": {
    "run_id": "7654320"
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": "2025-11-24T09:00:00.000Z Run tests\n2025-11-24T09:00:01.234Z Installing dependencies...\n2025-11-24T09:00:15.567Z Running pytest...\n2025-11-24T09:01:23.890Z FAILED: 3 tests failed\n...",
  "message": "GitHub Actions operation 'get_workflow_logs' completed successfully",
  "execution_time_ms": 2300
}
```

**Explanation**:
Downloads complete workflow logs for debugging failed run #7654320. Returns full log output as string.

### Example 5: Cancel Running Workflow

**Input**:
```json
{
  "operation": "cancel_workflow",
  "parameters": {
    "run_id": "7654322"
  }
}
```

**Expected Output Structure**:
```json
{
  "status": "success",
  "result": {
    "run_id": "7654322",
    "message": "Workflow run 7654322 cancelled"
  },
  "message": "GitHub Actions operation 'cancel_workflow' completed successfully",
  "execution_time_ms": 950
}
```

**Explanation**:
Cancels workflow run #7654322 that was taking too long or is no longer needed.

## Testing Checklist

Before deploying this skill, verify:

- [x] **Functionality**: All 9 operations execute correctly
- [x] **Error Handling**: Graceful handling of gh CLI errors
- [x] **Security**: No command injection vulnerabilities
- [x] **Performance**: Operations complete within expected time
- [x] **Token Efficiency**: Structured output, minimal verbosity
- [x] **Documentation**: All sections complete
- [x] **Dependencies**: gh CLI documented, authentication required
- [x] **Edge Cases**: Handles missing workflows, invalid run IDs
- [x] **Output Consistency**: Consistent result structure
- [x] **Integration**: Works with HoloLoom CI/CD if enabled

## Security Considerations

**Potential Risks**:
- **Command Injection**: Workflow/run_id parameters could contain shell commands → Sanitize inputs, use subprocess arrays
- **Unauthorized Access**: GitHub credentials required → Use gh auth status to verify authentication
- **Workflow Triggering**: Triggering workflows can consume CI/CD minutes → Validate operation before execution

**Data Privacy**:
- [x] Does not log sensitive workflow data (credentials, secrets)
- [x] Does not expose internal repository details beyond what gh CLI provides
- [x] Does not make unauthorized external requests

**Sandboxing**:
- [x] Operates within defined capability boundaries (code execution, network, file read)
- [x] Does not attempt privilege escalation
- [x] Does not modify system files outside artifact download scope

## Performance Characteristics

- **Expected Latency**: 500-5000ms (0.5-5 seconds depending on operation)
- **Token Usage**:
  - Input: 100-500 tokens (operation + parameters)
  - Output: 100-500 tokens (structured results)
  - Total: 200-1000 tokens per execution
- **Resource Requirements**:
  - gh CLI (required)
  - GitHub authentication (gh auth login)
  - Network connectivity to GitHub API
- **Scalability**: API rate limits apply (5000 requests/hour for authenticated users)

## Maintenance Notes

**Known Limitations**:
- Requires gh CLI installed and authenticated
- Depends on GitHub API availability
- Subject to GitHub API rate limits
- Output parsing depends on gh CLI output format (may break with major gh version changes)
- Repository must be accessible to authenticated user

**Future Enhancements**:
- **Workflow caching** - Cache workflow metadata to reduce API calls
- **Status polling** - Poll workflow status until completion
- **Multi-repo support** - Trigger workflows across multiple repositories
- **Parallel execution** - Trigger multiple workflows concurrently
- **Artifact management** - List, download, and delete artifacts
- **Deployment tracking** - Track deployments across environments
- **Workflow templates** - Pre-defined workflow trigger templates

**Changelog**:
- **v1.0.0** (2025-11-24): Initial release
  - 9 operations (trigger, list, status, logs, cancel, list_runs, artifacts, enable, disable)
  - Structured output with WorkflowRun dataclass
  - GitHub CLI integration
  - Repository auto-detection

## Usage Examples (Claude Code)

### Quick Workflow Trigger
```
Use github_actions to trigger workflow ci.yml on main branch
```

### Check Workflow Status
```
Use github_actions with operation=get_workflow_status and run_id=7654321 to check CI status
```

### List Recent Runs
```
Use github_actions with operation=list_runs and workflow=ci.yml and limit=10 to list recent CI runs
```

### Download Logs
```
Use github_actions with operation=get_workflow_logs and run_id=7654320 to download logs for debugging
```

### Cancel Running Workflow
```
Use github_actions with operation=cancel_workflow and run_id=7654322 to cancel long-running workflow
```

## Integration with HoloLoom Systems

This skill integrates with:

1. **Pytest Runner Skill** - Trigger CI workflows after test execution
2. **Quality Assurance Department** - Automated workflow execution in QA pipelines
3. **Docker Skill** - Deploy containers after successful workflow runs
4. **Monitoring** - Track workflow health and success rates
5. **Alignment Framework** - Validate deployments meet safety criteria

## License

MIT License

## Related Documentation

- **GitHub CLI Documentation**: [cli.github.com](https://cli.github.com/)
- **GitHub Actions Documentation**: [docs.github.com/actions](https://docs.github.com/en/actions)
- **HoloLoom CI/CD**: [CLAUDE.md](../../../CLAUDE.md) (CI/CD section)

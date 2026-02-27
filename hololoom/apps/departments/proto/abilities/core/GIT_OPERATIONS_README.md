# Git Operations Ability

**Status**: Production Ready (December 2025)
**Tier**: 2 (Plugin Protocol)
**Trust Level**: VERIFIED
**Version**: 1.0.0

Safe, read-only Git operations for Proto code analysis. Enables Proto to understand repository state, history, and structure.

## Overview

The Git Operations ability provides structured access to git repository information without modifying the repository. Perfect for code analysis, understanding repository history, and gathering context about changes.

### Supported Operations

| Operation | Purpose | Parameters |
|-----------|---------|-----------|
| **status** | Get repository status | `repo_path` |
| **diff** | Show changes (staged or unstaged) | `repo_path`, `file_path?`, `staged?` |
| **log** | Get commit history | `repo_path`, `limit?` |
| **branch** | List branches or get current | `repo_path` |
| **show** | Display specific commit | `repo_path`, `commit?` |
| **blame** | Line-by-line attribution | `repo_path`, `file_path` |
| **stash_list** | List stashed changes | `repo_path` |

## Installation

The ability is included with Proto and requires `git` to be installed on the system.

```bash
# Check git is available
which git
# or on Windows:
where git
```

## Usage

### Basic Usage

```python
from hololoom.apps.departments.proto.abilities.core import GitOperationsAbility
from hololoom.apps.departments.proto.abilities.protocol import AbilityContext

# Create ability instance
ability = GitOperationsAbility()

# Create execution context
context = AbilityContext(
    session_id="user-session-123",
    working_directory="/path/to/repo",
    user_confirmed=True,
    timeout_seconds=10.0
)

# Execute operation
result = await ability.execute({
    "operation": "status",
    "repo_path": "/path/to/repo"
}, context)

if result.success:
    print(result.output)
else:
    print(f"Error: {result.error}")
```

### Get Repository Status

```python
result = await ability.execute({
    "operation": "status",
    "repo_path": "/path/to/repo"
}, context)

# Output contains:
output = result.output
print(output["status_output"])        # Full porcelain output
print(output["staged_files"])          # Modified/added/renamed files
print(output["unstaged_files"])        # Files with unstaged changes
print(output["untracked_files"])       # Untracked files
print(output["has_changes"])           # Boolean: any changes
```

### Get File Changes (Diff)

```python
# Unstaged changes for specific file
result = await ability.execute({
    "operation": "diff",
    "repo_path": "/path/to/repo",
    "file_path": "src/main.py",
    "staged": False  # unstaged changes
}, context)

# Staged changes (whole repo)
result = await ability.execute({
    "operation": "diff",
    "repo_path": "/path/to/repo",
    "staged": True  # staged changes
}, context)

output = result.output
print(output["diff_output"])    # Unified diff format
print(output["has_changes"])    # Boolean
```

### Get Commit History

```python
result = await ability.execute({
    "operation": "log",
    "repo_path": "/path/to/repo",
    "limit": 20  # Get last 20 commits
}, context)

output = result.output
print(output["log_output"])     # Formatted graph output
print(output["commits"])        # Structured commit list

# Structured commits contain:
for commit in output["commits"]:
    print(f"{commit['hash']}: {commit['subject']}")
    print(f"  By: {commit['author']}")
    print(f"  Date: {commit['date']}")
```

### List Branches

```python
result = await ability.execute({
    "operation": "branch",
    "repo_path": "/path/to/repo"
}, context)

output = result.output
print(output["current_branch"])  # Name of current branch

# Branch list with flags
for branch in output["branches"]:
    marker = " [current]" if branch["is_current"] else ""
    print(f"{branch['name']}{marker}")
```

### View Specific Commit

```python
result = await ability.execute({
    "operation": "show",
    "repo_path": "/path/to/repo",
    "commit": "abc1234"  # Can be hash, short hash, ref, or HEAD
}, context)

output = result.output
print(output["show_output"])  # Full commit details with diff
```

### Get Line-by-Line Attribution

```python
result = await ability.execute({
    "operation": "blame",
    "repo_path": "/path/to/repo",
    "file_path": "src/module.py"
}, context)

output = result.output
# Each line shows: commit hash, author, date, code
print(output["blame_output"])
```

### List Stashed Changes

```python
result = await ability.execute({
    "operation": "stash_list",
    "repo_path": "/path/to/repo"
}, context)

output = result.output
print(f"Stashes: {output['count']}")
for stash in output["stashes"]:
    print(f"  {stash}")
```

## Parameter Reference

### Common Parameters

- **operation** (string, required): Operation name
  - Valid values: `status`, `diff`, `log`, `branch`, `show`, `blame`, `stash_list`

- **repo_path** (string, optional): Repository path
  - If not provided, uses `context.working_directory`
  - Must be a valid git repository (has `.git` directory)
  - Must exist and be readable

### Operation-Specific Parameters

#### diff
- **file_path** (string, optional): Show changes for specific file only
- **staged** (boolean, optional): If true, show staged changes; if false, show unstaged (default: false)

#### log
- **limit** (integer, optional): Number of commits to retrieve (default: 10, max: 100)

#### show
- **commit** (string, optional): Commit hash/ref to show (default: HEAD)
  - Can be full hash, short hash (7+ chars), branch name, or HEAD

#### blame
- **file_path** (string, required): File to blame (relative to repo)

## Response Format

All responses follow the standard `AbilityResult` structure:

```python
{
    "success": bool,           # Whether operation succeeded
    "output": dict | None,     # Operation-specific result
    "error": str | None,       # Error message if failed
    "confidence": float,       # 0.0-1.0 confidence score
    "duration_ms": float,      # Execution time
    "metadata": dict           # Additional metadata
}
```

## Safety Features

### Read-Only Operations

Only safe, read-only git operations are supported:
- ✅ View status, history, diffs
- ✅ Inspect commits and files
- ❌ No commits, push, pull, checkout, or branch changes

### Path Validation

- **No path traversal**: Prevents `../` escape attempts
- **Repository verification**: Confirms `.git` directory exists
- **File scope validation**: Ensures files are within repository
- **Absolute path resolution**: Prevents ambiguity

### Command Safety

- **Timeout protection**: 10-second default timeout per command
- **Output size limits**: Maximum 1MB output (prevents memory issues)
- **Process isolation**: Commands run in subprocess with cwd isolation
- **Error handling**: Comprehensive exception handling and logging

### Permission Model

Required permissions:
- `read_file`: Reading repository data
- `execute_command`: Running git commands

No write or network permissions needed.

## Error Handling

```python
result = await ability.execute(params, context)

if not result.success:
    # Handle error
    error_type = result.metadata.get("error_type")

    if "not found" in result.error:
        print("Repository or file not found")
    elif "timeout" in result.error:
        print("Command took too long")
    elif "path traversal" in result.error:
        print("Invalid path provided")
    else:
        print(f"Git error: {result.error}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| "not found in path" | git not installed | Install git |
| "Not a git repository" | .git missing | Run in valid git repo |
| "Path not found" | repo_path doesn't exist | Provide valid path |
| "Path traversal not allowed" | Using `..` in path | Use relative/absolute paths safely |
| "timeout" | Operation took too long | Try more specific operations |

## Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|--------------|-------|
| status | 50-100ms | Depends on repo size |
| diff (full) | 100-500ms | Can be slow on large diffs |
| diff (single file) | 50-200ms | Much faster |
| log (10 commits) | 50-150ms | Linear with commit count |
| log (100 commits) | 100-500ms | Can be slow on deep history |
| branch | 50-100ms | Usually fast |
| show | 50-200ms | Depends on diff size |
| blame | 100-1000ms | Linear with file size |
| stash_list | 50-100ms | Usually fast |

**Timeout**: 10 seconds default (configurable in context)

## Manifest

```python
AbilityManifest(
    name="git_operations",
    version="1.0.0",
    description="Safe read-only Git repository operations",
    author="Proto Team",
    tier=AbilityTier.PLUGIN,
    trust_level=AbilityTrustLevel.VERIFIED,
    permissions=["read_file", "execute_command"],
    requires=["git"],
    tags=["git", "vcs", "repository", "version-control"],
)
```

## Integration with Proto

The Git Operations ability integrates seamlessly with Proto for code analysis:

```python
from hololoom.apps.departments.proto import Proto

proto = Proto()

# Proto can use git operations for context
context = proto.create_context()
context.add_ability(GitOperationsAbility())

# Now Proto can understand repository context in analysis
analysis = await proto.analyze(
    code=some_code,
    context_repo="/path/to/repo"
)
```

## Testing

Basic test example:

```python
import asyncio
from hololoom.apps.departments.proto.abilities.core import GitOperationsAbility
from hololoom.apps.departments.proto.abilities.protocol import AbilityContext

async def test_git_operations():
    ability = GitOperationsAbility()
    context = AbilityContext(
        session_id="test-123",
        working_directory="/path/to/git/repo"
    )

    # Test status
    result = await ability.execute({
        "operation": "status",
        "repo_path": "/path/to/git/repo"
    }, context)
    assert result.success, f"Status failed: {result.error}"
    assert "has_changes" in result.output

    # Test log
    result = await ability.execute({
        "operation": "log",
        "repo_path": "/path/to/git/repo",
        "limit": 5
    }, context)
    assert result.success
    assert len(result.output["commits"]) <= 5

asyncio.run(test_git_operations())
```

## Implementation Details

### Architecture

- **Tier 2 Plugin**: Implements Ability protocol with manifest
- **Async/await**: Fully async using `asyncio.create_subprocess_exec`
- **Subprocess isolation**: Git commands run in isolated subprocess
- **Error recovery**: Comprehensive error handling and logging

### Key Classes

- **GitOperationsAbility**: Main ability class extending BaseAbility
- **GitOperationResult**: Result from a git operation
- **Parameter validation**: Path traversal prevention, repo verification

### Command Execution

Git commands are executed asynchronously:

```python
result = await asyncio.create_subprocess_exec(
    *cmd,
    cwd=repo_path,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
    text=True
)
```

### Output Parsing

Each operation parses git output into structured format:
- **status**: Splits files by category (staged, unstaged, untracked)
- **log**: Creates commit dictionaries with hash, subject, author, date
- **branch**: Marks current branch with `is_current` flag
- **diff**: Preserves raw diff but flags if changes exist

## Limitations

1. **Size limits**: Output truncated at 1MB
2. **Timeout**: Commands must complete in 10 seconds
3. **Read-only**: No modifications to repository
4. **Local only**: Cannot clone/pull/push
5. **Shallow info**: Shows current state, not full history beyond limit

## Future Enhancements

Potential future additions:
- Remote status (check if local is ahead/behind)
- Performance metrics (commits by author, lines changed)
- Search (grep across repository)
- Statistics (code churn, contributor stats)
- Integration with GitHub/GitLab APIs

## See Also

- [Proto Documentation](../README.md)
- [Ability Protocol](../protocol.py)
- [Skill Wrapper Abilities](./skill_wrapper.py)

## License

Same as HoloLoom project

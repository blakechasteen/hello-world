# Git Operations Ability - Quick Start

**Status**: Production Ready | **Tier**: 2 Plugin | **Version**: 1.0.0

## Installation

Already included in Proto. Just import:

```python
from hololoom.apps.departments.proto.abilities.core import GitOperationsAbility
```

## Basic Usage (3 lines)

```python
ability = GitOperationsAbility()
context = AbilityContext(working_directory="/path/to/repo")
result = await ability.execute({"operation": "status"}, context)
```

## 7 Operations at a Glance

### 1. Get Status
```python
result = await ability.execute({
    "operation": "status",
    "repo_path": "/repo"
}, context)

# Output: staged_files, unstaged_files, untracked_files, has_changes
```

### 2. View Changes (Diff)
```python
result = await ability.execute({
    "operation": "diff",
    "repo_path": "/repo",
    "file_path": "main.py",  # optional
    "staged": False          # True for staged, False for unstaged
}, context)

# Output: diff_output, has_changes
```

### 3. View Commits
```python
result = await ability.execute({
    "operation": "log",
    "repo_path": "/repo",
    "limit": 10  # 1-100
}, context)

# Output: log_output, commits (list with hash/subject/author/date)
```

### 4. List Branches
```python
result = await ability.execute({
    "operation": "branch",
    "repo_path": "/repo"
}, context)

# Output: branch_output, branches, current_branch
```

### 5. Show Commit
```python
result = await ability.execute({
    "operation": "show",
    "repo_path": "/repo",
    "commit": "abc1234"  # hash, ref, or HEAD (default)
}, context)

# Output: show_output
```

### 6. Blame File
```python
result = await ability.execute({
    "operation": "blame",
    "repo_path": "/repo",
    "file_path": "main.py"  # required
}, context)

# Output: blame_output (hash, author, date per line)
```

### 7. List Stashes
```python
result = await ability.execute({
    "operation": "stash_list",
    "repo_path": "/repo"
}, context)

# Output: stash_output, stashes, count
```

## Error Handling

```python
result = await ability.execute(params, context)

if not result.success:
    print(f"Error: {result.error}")
    # Common errors:
    # "git command not found" -> install git
    # "not a git repository" -> use valid repo
    # "path traversal" -> don't use ../
    # "timeout" -> repo too large
```

## Common Use Cases

### Check if repo has changes
```python
result = await ability.execute({"operation": "status"}, context)
has_changes = result.output["has_changes"]
```

### Get last 5 commits by specific author
```python
result = await ability.execute({"operation": "log", "limit": 5}, context)
for commit in result.output["commits"]:
    if "john" in commit["author"].lower():
        print(commit["subject"])
```

### Get current branch
```python
result = await ability.execute({"operation": "branch"}, context)
current = result.output["current_branch"]
```

### View file history
```python
result = await ability.execute({
    "operation": "blame",
    "file_path": "config.py"
}, context)
print(result.output["blame_output"])
```

### Get staged changes only
```python
result = await ability.execute({
    "operation": "diff",
    "staged": True
}, context)
```

## Parameter Defaults

| Param | Default | Notes |
|-------|---------|-------|
| operation | - | REQUIRED |
| repo_path | context.working_directory | Uses context if not provided |
| file_path | - | Required for blame, optional for diff |
| commit | HEAD | For show operation |
| limit | 10 | For log, clamped to 1-100 |
| staged | False | For diff (False=unstaged) |

## Result Structure

```python
{
    "success": True,                    # Operation succeeded
    "output": {                         # Operation-specific
        "status_output": "...",         # Full output
        "staged_files": ["file.py"],    # Parsed/structured data
        # ... operation-specific fields
    },
    "error": None,                      # Error message if failed
    "confidence": 0.95,                 # 0.95 for success
    "duration_ms": 45.2,               # How long it took
    "metadata": {                       # Context
        "operation": "status",
        "repo_path": "/repo"
    }
}
```

## Requirements

- Python 3.8+ (for async)
- git installed (`which git`)

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| status | 50-100ms | Repository size matters |
| diff | 50-500ms | Depends on changes |
| log | 50-500ms | Depends on commits |
| branch | 50-100ms | Usually fast |
| show | 50-200ms | Depends on diff size |
| blame | 100-1000ms | Depends on file size |
| stash_list | 50-100ms | Usually fast |

**Timeout**: 10 seconds per command

## Integration with Proto

```python
from hololoom.apps.departments.proto import Proto

proto = Proto()
proto.register_ability(GitOperationsAbility())

# Now Proto can use git context in analysis
analysis = await proto.analyze(code, context_repo="/repo")
```

## Security

- Read-only operations only
- No commit, push, pull, or checkout
- Path traversal protection
- Subprocess isolation
- Timeout protection

## Testing

```bash
# Run tests
pytest hololoom/departments/proto/abilities/core/test_git_operations.py -v

# Test specific operation
pytest -k "test_status" -v

# With coverage
pytest --cov=hololoom.apps.departments.proto.abilities.core.git_operations
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "git not found" | Install git: `apt install git` or `brew install git` |
| "not a git repo" | Ensure `.git` directory exists in repo |
| "timeout" | Try smaller limit or specific file |
| "path traversal" | Don't use `../` in paths |
| "permission denied" | Ensure repo is readable |

## Files

- `git_operations.py` - Implementation (827 lines)
- `GIT_OPERATIONS_README.md` - Full documentation
- `GIT_OPERATIONS_QUICKSTART.md` - This file
- `test_git_operations.py` - Test suite

## More Info

- Full docs: `GIT_OPERATIONS_README.md`
- Implementation: `git_operations.py`
- Tests: `test_git_operations.py`

## Version Info

```
GitOperationsAbility
- Version: 1.0.0
- Tier: 2 (Plugin)
- Trust Level: VERIFIED
- Status: Production Ready
```

---

**Start with**: `result = await ability.execute({"operation": "status"}, context)`

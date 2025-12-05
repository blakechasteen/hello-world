# Phase 5D: Issue Tracking - Complete ✅

**Status**: ✅ Complete
**Time**: 30 minutes
**Lines of Code**: ~420 lines
**Dependencies**: PyGithub, github_auth.py

---

## What Was Built

### `bot/github_issues.py` (420 lines)

Complete GitHub issue management system for Matrix bot commands.

**Core Class**: `IssueManager`
- Create issues with labels and assignees
- Add comments to existing issues
- Close/reopen issues
- Manage labels (add/remove)
- Manage assignees (assign/unassign)
- List and search issues with filters

---

## Key Features

### 1. Create Issues (`create_issue`)

```python
result = await manager.create_issue(
    installation_id=12345678,
    repo_full_name="owner/repo",
    title="Dashboard fails to load on Firefox",
    body="Steps to reproduce:\n1. Open dashboard\n2. Click Workflows tab\n3. Page hangs",
    labels=["bug", "dashboard"],
    assignees=["alice", "bob"]
)
```

**Matrix Command**: `!issue create "Dashboard fails on Firefox" --label bug --assign alice`

**Output**:
```
✅ Issue Created!

#42: Dashboard fails to load on Firefox
🔗 https://github.com/owner/repo/issues/42
🏷️  Labels: bug, dashboard
👤 Assigned: alice, bob
```

---

### 2. Get Issue Details (`get_issue`)

```python
result = await manager.get_issue(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42
)
```

**Returns**:
```python
{
    "success": True,
    "issue": {
        "number": 42,
        "title": "Dashboard fails on Firefox",
        "state": "open",
        "labels": ["bug", "dashboard"],
        "assignees": ["alice"],
        "author": "bob",
        "created_at": "2025-11-08T10:30:00Z",
        "updated_at": "2025-11-08T12:15:00Z",
        "comments_count": 3,
        "url": "https://github.com/owner/repo/issues/42",
        "body": "Steps to reproduce..."
    }
}
```

---

### 3. Add Comment (`add_comment`)

```python
result = await manager.add_comment(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42,
    comment="Investigating this issue. Appears to be a React Flow bug."
)
```

**Matrix Command**: `!issue comment 42 "Investigating this issue"`

**Output**:
```
✅ Comment added to issue #42

🔗 https://github.com/owner/repo/issues/42#issuecomment-123456
```

---

### 4. Close/Reopen Issues

**Close Issue**:
```python
result = await manager.close_issue(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42
)
```

**Matrix Command**: `!issue close 42`

**Output**: `✅ Issue #42 closed`

**Reopen Issue**:
```python
result = await manager.reopen_issue(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42
)
```

**Matrix Command**: `!issue reopen 42`

**Output**: `✅ Issue #42 reopened`

---

### 5. Label Management

**Add Labels**:
```python
result = await manager.add_labels(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42,
    labels=["high-priority", "needs-triage"]
)
```

**Matrix Command**: `!issue label 42 high-priority`

**Remove Label**:
```python
result = await manager.remove_label(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42,
    label="needs-triage"
)
```

**Matrix Command**: `!issue unlabel 42 needs-triage`

---

### 6. Assignee Management

**Assign User**:
```python
result = await manager.assign_user(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42,
    username="alice"
)
```

**Matrix Command**: `!issue assign 42 alice`

**Unassign User**:
```python
result = await manager.unassign_user(
    installation_id=12345678,
    repo_full_name="owner/repo",
    issue_number=42,
    username="alice"
)
```

**Matrix Command**: `!issue unassign 42 alice`

---

### 7. List Issues (`list_issues`)

```python
result = await manager.list_issues(
    installation_id=12345678,
    repo_full_name="owner/repo",
    state="open",           # "open", "closed", or "all"
    labels=["bug"],         # Filter by labels
    assignee="alice",       # Filter by assignee
    limit=10                # Max issues to return
)
```

**Matrix Commands**:
- `!issue list` - List open issues
- `!issue list --state closed` - List closed issues
- `!issue list --label bug` - List bug issues
- `!issue list --assign alice` - List Alice's issues

**Output**:
```
📋 Open Issues (5)

#42 Dashboard fails to load on Firefox [bug, dashboard]
  👤 alice
  🔗 https://github.com/owner/repo/issues/42

#38 Add dark mode support [enhancement]
  👤 bob
  🔗 https://github.com/owner/repo/issues/38

...
```

---

## Data Structures

### `IssueInfo`
```python
@dataclass
class IssueInfo:
    number: int              # Issue number
    title: str               # Issue title
    state: str               # "open" or "closed"
    labels: List[str]        # Label names
    assignees: List[str]     # Assigned usernames
    author: str              # Creator username
    created_at: str          # ISO timestamp
    updated_at: str          # ISO timestamp
    comments_count: int      # Number of comments
    url: str                 # GitHub URL
    body: Optional[str]      # Issue description
```

---

## Matrix Command Handlers

### 1. `handle_issue_create_command`
```python
response = await handle_issue_create_command(
    manager=issue_manager,
    installation_id=12345678,
    repo="owner/repo",
    title="Dashboard fails on Firefox",
    body="Steps to reproduce...",
    labels=["bug", "dashboard"],
    assignees=["alice"]
)
```

### 2. `handle_issue_comment_command`
```python
response = await handle_issue_comment_command(
    manager=issue_manager,
    installation_id=12345678,
    repo="owner/repo",
    issue_number=42,
    comment="Investigating this issue"
)
```

### 3. `handle_issue_close_command`
```python
response = await handle_issue_close_command(
    manager=issue_manager,
    installation_id=12345678,
    repo="owner/repo",
    issue_number=42
)
```

### 4. `handle_issue_list_command`
```python
response = await handle_issue_list_command(
    manager=issue_manager,
    installation_id=12345678,
    repo="owner/repo",
    state="open",
    labels=["bug"],
    limit=10
)
```

---

## Usage Examples

### Example 1: Bug Report Workflow
```
# User discovers bug
!issue create "Search crashes on empty query" --label bug --assign alice
→ ✅ Issue Created! #45

# Alice investigates
!issue comment 45 "Reproduced. Working on fix."
→ ✅ Comment added

# Alice adds priority
!issue label 45 high-priority
→ ✅ Label added

# Alice creates PR
!pr create fix-search-crash "Fix search crash on empty query"
→ ✅ PR Created! #46

# Link PR to issue
!issue comment 45 "Fixed in PR #46"
→ ✅ Comment added

# After PR merged
!issue close 45
→ ✅ Issue #45 closed
```

### Example 2: Triaging Issues
```
# List all open issues
!issue list
→ 📋 Open Issues (12)

# Filter by label
!issue list --label bug
→ 📋 Open Issues (5)

# Assign to developer
!issue assign 45 bob
→ ✅ Assigned to bob

# Add priority
!issue label 45 high-priority
→ ✅ Label added
```

---

## Integration with Matrix Bot

```python
from bot.github_issues import IssueManager, handle_issue_create_command, handle_issue_list_command

class PromptlyBot:
    def __init__(self):
        self.issue_manager = IssueManager()

    async def handle_command(self, room_id: str, user_id: str, command: str, args: List[str]):
        if command == "issue":
            subcommand = args[0] if args else "help"

            if subcommand == "create":
                # !issue create "Title" --body "Body" --label bug --assign alice
                title = args[1]
                body = None
                labels = []
                assignees = []

                # Parse --body, --label, --assign flags
                for i, arg in enumerate(args[2:], start=2):
                    if arg == "--body" and i + 1 < len(args):
                        body = args[i + 1]
                    elif arg == "--label" and i + 1 < len(args):
                        labels.append(args[i + 1])
                    elif arg == "--assign" and i + 1 < len(args):
                        assignees.append(args[i + 1])

                response = await handle_issue_create_command(
                    manager=self.issue_manager,
                    installation_id=self.get_installation_id(room_id),
                    repo=self.get_repo_for_room(room_id),
                    title=title,
                    body=body,
                    labels=labels if labels else None,
                    assignees=assignees if assignees else None
                )

                await self.send_message(room_id, response)

            elif subcommand == "list":
                # !issue list --state open --label bug
                state = "open"
                labels = []

                for i, arg in enumerate(args[1:], start=1):
                    if arg == "--state" and i + 1 < len(args):
                        state = args[i + 1]
                    elif arg == "--label" and i + 1 < len(args):
                        labels.append(args[i + 1])

                response = await handle_issue_list_command(
                    manager=self.issue_manager,
                    installation_id=self.get_installation_id(room_id),
                    repo=self.get_repo_for_room(room_id),
                    state=state,
                    labels=labels if labels else None
                )

                await self.send_message(room_id, response)
```

---

## Error Handling

All methods return standardized result dictionaries:

**Success**:
```python
{
    "success": True,
    "issue_number": 42,
    "issue_url": "https://github.com/...",
    # ... additional fields
}
```

**Failure**:
```python
{
    "success": False,
    "error": "Issue not found",
    "status_code": 404
}
```

---

## Next Steps

With Phase 5D complete, only one component remains:

**Phase 5E: CI/CD Triggers** (1 hour) - Next!
- Trigger GitHub Actions workflows
- Monitor build status
- Real-time notifications to Matrix

---

**Phase 5D Status**: ✅ Complete
**Next**: Phase 5E - CI/CD Triggers
**Total Phase 5 Progress**: 4/5 components (80%)

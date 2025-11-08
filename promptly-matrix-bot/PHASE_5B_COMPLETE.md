# Phase 5B: PR Management - Complete ✅

**Status**: ✅ Complete
**Time**: 1.5 hours
**Lines of Code**: ~470 lines
**Dependencies**: PyGithub, github_auth.py

---

## What Was Built

### `bot/github_pr.py` (470 lines)

Complete pull request management system for Matrix bot commands.

**Core Class**: `PRManager`
- Authenticated GitHub API client via GitHub App installation tokens
- Full PR lifecycle management (create, review, merge, status)
- CODEOWNERS integration for automatic reviewer assignment
- Auto-generated titles and descriptions from commits
- CI/CD status monitoring

---

## Key Features

### 1. PR Creation (`create_pr`)

```python
result = await manager.create_pr(
    installation_id=12345678,
    repo_full_name="owner/repo",
    head_branch="feature-dashboard",
    base_branch="main",
    title="Add Phase 4 Dashboard",  # Optional - auto-generated from branch name
    draft=False,
    reviewers=["alice", "bob"]  # Optional - uses CODEOWNERS if not provided
)
```

**Auto-Generation**:
- **Title**: Converts `feature/add-dashboard` → `Add Dashboard`
- **Body**: Extracts commit messages from branch
- **Reviewers**: Parses CODEOWNERS file for automatic assignment

**Matrix Command**: `!pr create feature-dashboard "Add dashboard components"`

---

### 2. PR Status (`get_pr_status`)

```python
status = await manager.get_pr_status(
    installation_id=12345678,
    repo_full_name="owner/repo",
    pr_number=123
)
```

**Returns**:
- PR metadata (title, state, draft, mergeable)
- **Review summary**: Total reviews, approvals, changes requested
- **CI status**: Combined check status, individual check details
- **Code stats**: Commits, additions, deletions, files changed

**Matrix Command**: `!pr status 123`

**Example Output**:
```
📊 PR #123 Status

Title: Add Phase 4 Dashboard
State: OPEN
Draft: No
Mergeable: ✅ Yes (clean)

Reviews:
  ✅ Approved: 2
  ⚠️  Changes Requested: 0

CI Checks:
  State: SUCCESS
  Total: 5

🔗 https://github.com/owner/repo/pull/123
```

---

### 3. Add Comment (`add_comment`)

```python
result = await manager.add_comment(
    installation_id=12345678,
    repo_full_name="owner/repo",
    pr_number=123,
    comment="🤖 Automated review: LGTM!"
)
```

**Matrix Command**: `!pr comment 123 "Looks good to me!"`

---

### 4. Review PR (`review_pr`)

```python
result = await manager.review_pr(
    installation_id=12345678,
    repo_full_name="owner/repo",
    pr_number=123,
    event="APPROVE",  # or "REQUEST_CHANGES", "COMMENT"
    body="Great work! Approving this PR."
)
```

**Matrix Command**: `!pr review 123 approve "Well done!"`

**Review Events**:
- `APPROVE` - Approve PR
- `REQUEST_CHANGES` - Request changes
- `COMMENT` - Comment without approval/rejection

---

### 5. Merge PR (`merge_pr`)

```python
result = await manager.merge_pr(
    installation_id=12345678,
    repo_full_name="owner/repo",
    pr_number=123,
    merge_method="squash",  # or "merge", "rebase"
    commit_title="Add Phase 4 Dashboard (#123)"
)
```

**Pre-Merge Checks**:
- PR is mergeable (`mergeable` = True)
- No merge conflicts
- CI checks pass (optional enforcement)

**Matrix Command**: `!pr merge 123`

**Example Output**:
```
✅ PR #123 merged!

Commit: a1b2c3d
```

---

## Matrix Command Handlers

Five command handlers ready for integration with Matrix bot:

### 1. `handle_pr_create_command`
```python
response = await handle_pr_create_command(
    manager=pr_manager,
    installation_id=12345678,
    repo="owner/repo",
    branch="feature-dashboard",
    title=None  # Auto-generated
)
```

**Output**:
```
✅ PR Created!

#45: Add Dashboard
🔗 https://github.com/owner/repo/pull/45
👥 Reviewers: alice, bob
```

---

### 2. `handle_pr_status_command`
```python
response = await handle_pr_status_command(
    manager=pr_manager,
    installation_id=12345678,
    repo="owner/repo",
    pr_number=45
)
```

**Output**: Full status breakdown (see example above)

---

### 3. `handle_pr_merge_command`
```python
response = await handle_pr_merge_command(
    manager=pr_manager,
    installation_id=12345678,
    repo="owner/repo",
    pr_number=45
)
```

**Pre-Merge Validation**:
- Checks `mergeable` state
- Verifies approvals
- Checks CI status
- Prevents merge if not ready

**Output (success)**:
```
✅ PR #45 merged!

Commit: a1b2c3d
```

**Output (blocked)**:
```
❌ Cannot merge PR #45

State: blocked
Approvals: 0
CI: pending
```

---

## CODEOWNERS Integration

**Automatic Reviewer Assignment**:
1. Searches for CODEOWNERS file in:
   - `.github/CODEOWNERS`
   - `CODEOWNERS`
   - `docs/CODEOWNERS`

2. Parses file to extract usernames:
   ```
   # CODEOWNERS example
   * @alice @bob
   /dashboard/ @frontend-team
   /bot/ @backend-team
   ```

3. Assigns up to 5 reviewers from CODEOWNERS

**Fallback**: If no CODEOWNERS or manual reviewers specified, PR created without reviewers.

---

## Auto-Generation Algorithms

### Title Generation
```python
def _generate_title_from_branch(branch_name: str) -> str:
    # Remove common prefixes
    for prefix in ["feature/", "bugfix/", "hotfix/", "fix/", "feat/"]:
        if branch_name.startswith(prefix):
            branch_name = branch_name[len(prefix):]

    # Convert kebab-case/snake_case to Title Case
    title = branch_name.replace("-", " ").replace("_", " ").title()
    return title
```

**Examples**:
- `feature/add-dashboard` → `Add Dashboard`
- `bugfix/fix_auth_error` → `Fix Auth Error`
- `hotfix/critical-bug` → `Critical Bug`

---

### Body Generation
```python
async def _generate_body_from_commits(repo, head_branch, base_branch) -> str:
    # Compare branches
    comparison = repo.compare(base_branch, head_branch)

    # Extract commit messages
    body_lines = ["## Changes\n"]
    for commit in comparison.commits:
        message = commit.commit.message.split("\n")[0]  # First line
        body_lines.append(f"- {message}")

    body_lines.append(
        f"\n## Summary\n\n{comparison.total_commits} commit(s), "
        f"+{comparison.ahead_by} ahead of {base_branch}"
    )

    return "\n".join(body_lines)
```

**Example Output**:
```markdown
## Changes

- Add AuditTrailBrowser component
- Add TeamCollaborationUI component
- Add WorkflowBuilder component
- Update App.tsx with new tabs

## Summary

4 commit(s), +4 ahead of main
```

---

## Error Handling

All methods return standardized result dictionaries:

**Success**:
```python
{
    "success": True,
    "pr_number": 123,
    "pr_url": "https://github.com/...",
    # ... additional fields
}
```

**Failure**:
```python
{
    "success": False,
    "error": "Branch not found",
    "status_code": 404  # From GithubException
}
```

**Integration Pattern**:
```python
result = await manager.create_pr(...)

if result["success"]:
    # Success path
    send_matrix_message(f"✅ PR #{result['pr_number']} created!")
else:
    # Error path
    send_matrix_message(f"❌ Error: {result['error']}")
```

---

## Integration with Matrix Bot

To integrate with Matrix bot (`bot/bot.py`):

```python
from bot.github_pr import PRManager, handle_pr_create_command, handle_pr_status_command, handle_pr_merge_command

class PromptlyBot:
    def __init__(self):
        self.pr_manager = PRManager()

    async def handle_command(self, room_id: str, user_id: str, command: str, args: List[str]):
        if command == "pr":
            subcommand = args[0] if args else "help"

            if subcommand == "create":
                # !pr create feature-branch "Optional title"
                branch = args[1]
                title = args[2] if len(args) > 2 else None

                response = await handle_pr_create_command(
                    manager=self.pr_manager,
                    installation_id=self.get_installation_id(room_id),
                    repo=self.get_repo_for_room(room_id),
                    branch=branch,
                    title=title
                )

                await self.send_message(room_id, response)

            elif subcommand == "status":
                # !pr status 123
                pr_number = int(args[1])

                response = await handle_pr_status_command(
                    manager=self.pr_manager,
                    installation_id=self.get_installation_id(room_id),
                    repo=self.get_repo_for_room(room_id),
                    pr_number=pr_number
                )

                await self.send_message(room_id, response)

            elif subcommand == "merge":
                # !pr merge 123
                pr_number = int(args[1])

                response = await handle_pr_merge_command(
                    manager=self.pr_manager,
                    installation_id=self.get_installation_id(room_id),
                    repo=self.get_repo_for_room(room_id),
                    pr_number=pr_number
                )

                await self.send_message(room_id, response)
```

---

## Usage Examples

### Example 1: Quick PR Creation
```python
# Matrix: !pr create feature-dashboard

# Creates PR with:
# - Title: "Feature Dashboard" (auto-generated)
# - Body: Commit messages (auto-generated)
# - Reviewers: From CODEOWNERS (auto-assigned)
```

### Example 2: Custom PR
```python
# Matrix: !pr create feature-auth "Add JWT authentication"

# Creates PR with:
# - Title: "Add JWT authentication" (manual)
# - Body: Commit messages (auto-generated)
# - Reviewers: From CODEOWNERS
```

### Example 3: Full Workflow
```python
# 1. Create PR
# Matrix: !pr create feature-dashboard
# Output: ✅ PR Created! #45

# 2. Check status
# Matrix: !pr status 45
# Output: Shows reviews, CI, mergeable state

# 3. Add comment
# Matrix: !pr comment 45 "Please review the new components"

# 4. Approve (after review)
# Matrix: !pr review 45 approve "LGTM!"

# 5. Merge
# Matrix: !pr merge 45
# Output: ✅ PR #45 merged! Commit: a1b2c3d
```

---

## Testing

### Manual Testing (without Matrix)

```python
import asyncio
from bot.github_pr import PRManager

async def test_pr_workflow():
    manager = PRManager()

    # 1. Create PR
    print("Creating PR...")
    result = await manager.create_pr(
        installation_id=12345678,
        repo_full_name="your-username/your-repo",
        head_branch="feature-test",
        base_branch="main"
    )
    print(result)

    if result["success"]:
        pr_number = result["pr_number"]

        # 2. Get status
        print(f"\nGetting PR #{pr_number} status...")
        status = await manager.get_pr_status(
            installation_id=12345678,
            repo_full_name="your-username/your-repo",
            pr_number=pr_number
        )
        print(status)

        # 3. Add comment
        print(f"\nAdding comment to PR #{pr_number}...")
        comment_result = await manager.add_comment(
            installation_id=12345678,
            repo_full_name="your-username/your-repo",
            pr_number=pr_number,
            comment="🤖 Test comment from bot"
        )
        print(comment_result)

asyncio.run(test_pr_workflow())
```

---

## Next Steps

With Phase 5B complete, the next components are:

1. **Phase 5C: AI Code Review** (1 hour)
   - HoloLoom-powered code review
   - Security/performance/quality checks
   - Automated review comments on GitHub

2. **Phase 5D: Issue Tracking** (30 minutes)
   - Create/comment/close issues
   - Label and assignee management

3. **Phase 5E: CI/CD Triggers** (1 hour)
   - Trigger GitHub Actions workflows
   - Monitor build status
   - Real-time notifications to Matrix

---

**Phase 5B Status**: ✅ Complete
**Next**: Phase 5C - AI Code Review
**Total Phase 5 Progress**: 2/5 components (40%)

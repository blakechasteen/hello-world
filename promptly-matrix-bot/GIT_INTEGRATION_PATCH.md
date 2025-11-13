# Git Integration Patch for promptly_bot.py

## Changes Needed

### 1. Add Import (around line 83-84)

Add this line with the other imports:
```python
from .git_handler import GitHandler
```

### 2. Initialize Git Handler (around line 89-91, after self.code_reviewer)

Add this block:
```python
# Initialize Git handler (ChatOps Phase 1)
git_repo_path = os.getenv("GIT_REPO_PATH")
try:
    self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
    if self.git_handler:
        logger.info(f"✅ Git handler initialized for repo: {git_repo_path}")
    else:
        logger.info("Git handler not initialized (set GIT_REPO_PATH to enable)")
except ValueError as e:
    logger.warning(f"Git handler initialization failed: {e}")
    self.git_handler = None
```

### 3. Add Git Command Routing (around line 280-286, in handle_command method)

Add these elif blocks before the final `else`:
```python
elif cmd_type == 'git-status':
    return await self.cmd_git_status(command, room)
elif cmd_type == 'git-log':
    return await self.cmd_git_log(command, room)
elif cmd_type == 'git-diff':
    return await self.cmd_git_diff(command, room)
elif cmd_type == 'git-branch':
    return await self.cmd_git_branch(command, room)
elif cmd_type == 'git-commit':
    return await self.cmd_git_commit(command, room)
elif cmd_type == 'git-push':
    return await self.cmd_git_push(command, room)
elif cmd_type == 'git-pull':
    return await self.cmd_git_pull(command, room)
```

### 4. Add Git Command Methods (around line 521, after cmd_chat and before send_message)

```python
# ========== Git Commands (ChatOps Phase 1) ==========

async def cmd_git_status(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git status command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        status = self.git_handler.status(short=True)
        branch = self.git_handler.get_current_branch()

        # Format output
        body = f"📊 **Git Status**\n\nBranch: `{branch}`\n\n```\n{status}\n```"
        html = f"<p>📊 <strong>Git Status</strong></p><p>Branch: <code>{branch}</code></p><pre>{status}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git status error: {e}")
        return {
            "body": f"❌ Git status failed: {e}",
            "html": f"<p>❌ Git status failed: <code>{e}</code></p>"
        }

async def cmd_git_log(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git log command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        log = self.git_handler.log(max_count=5, oneline=True)
        branch = self.git_handler.get_current_branch()

        body = f"📜 **Recent Commits** ({branch})\n\n```\n{log}\n```"
        html = f"<p>📜 <strong>Recent Commits</strong> ({branch})</p><pre>{log}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git log error: {e}")
        return {
            "body": f"❌ Git log failed: {e}",
            "html": f"<p>❌ Git log failed: <code>{e}</code></p>"
        }

async def cmd_git_diff(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git diff command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        diff = self.git_handler.diff()

        if not diff:
            return {
                "body": "No changes to show (working tree clean)",
                "html": "<p>No changes to show (working tree clean)</p>"
            }

        # Truncate if too long
        if len(diff) > 2000:
            diff = diff[:2000] + "\n\n... (truncated, too long for chat)"

        body = f"📝 **Git Diff**\n\n```\n{diff}\n```"
        html = f"<p>📝 <strong>Git Diff</strong></p><pre>{diff}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git diff error: {e}")
        return {
            "body": f"❌ Git diff failed: {e}",
            "html": f"<p>❌ Git diff failed: <code>{e}</code></p>"
        }

async def cmd_git_branch(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git branch command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        branches = self.git_handler.branch()

        body = f"🌿 **Git Branches**\n\n```\n{branches}\n```"
        html = f"<p>🌿 <strong>Git Branches</strong></p><pre>{branches}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git branch error: {e}")
        return {
            "body": f"❌ Git branch failed: {e}",
            "html": f"<p>❌ Git branch failed: <code>{e}</code></p>"
        }

async def cmd_git_commit(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git commit command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        message = command.get('message', '')
        if not message:
            return {
                "body": "❌ Commit message required. Usage: `@promptly git commit \"your message\"`",
                "html": "<p>❌ Commit message required. Usage: <code>@promptly git commit \"your message\"</code></p>"
            }

        # Add all changes and commit
        result = self.git_handler.commit(message, add_all=True)

        body = f"✅ **Commit Created**\n\nMessage: `{message}`\n\n```\n{result}\n```"
        html = f"<p>✅ <strong>Commit Created</strong></p><p>Message: <code>{message}</code></p><pre>{result}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git commit error: {e}")
        return {
            "body": f"❌ Git commit failed: {e}",
            "html": f"<p>❌ Git commit failed: <code>{e}</code></p>"
        }

async def cmd_git_push(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git push command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        # Get current branch
        branch = self.git_handler.get_current_branch()

        # Push to remote
        result = self.git_handler.push(branch=branch)

        body = f"✅ **Pushed to Remote**\n\nBranch: `{branch}`\n\n```\n{result}\n```"
        html = f"<p>✅ <strong>Pushed to Remote</strong></p><p>Branch: <code>{branch}</code></p><pre>{result}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git push error: {e}")
        return {
            "body": f"❌ Git push failed: {e}\n\nNote: Push requires authentication to be configured",
            "html": f"<p>❌ Git push failed: <code>{e}</code></p><p>Note: Push requires authentication to be configured</p>"
        }

async def cmd_git_pull(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
    """Handle git pull command"""
    if not self.git_handler:
        return {
            "body": "❌ Git not configured. Set GIT_REPO_PATH in .env",
            "html": "<p>❌ Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
        }

    try:
        # Pull from remote
        result = self.git_handler.pull()

        body = f"✅ **Pulled from Remote**\n\n```\n{result}\n```"
        html = f"<p>✅ <strong>Pulled from Remote</strong></p><pre>{result}</pre>"

        return {"body": body, "html": html}

    except Exception as e:
        logger.error(f"Git pull error: {e}")
        return {
            "body": f"❌ Git pull failed: {e}",
            "html": f"<p>❌ Git pull failed: <code>{e}</code></p>"
        }
```

## Apply the Patch

You can manually apply these changes to `bot/promptly_bot.py`, or I can create a complete new version of the file with all changes integrated.

The changes are:
1. Import GitHandler
2. Initialize git_handler in __init__
3. Add 7 git command routing cases
4. Add 7 git command handler methods

After applying, you'll be able to use:
- `@promptly git status`
- `@promptly git log`
- `@promptly git diff`
- `@promptly git branch`
- `@promptly git commit "message"`
- `@promptly git push`
- `@promptly git pull`

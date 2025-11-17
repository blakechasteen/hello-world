# Manual Git Integration - Step by Step

This is the safest way to integrate git commands. Takes 10 minutes.

## Prerequisites

1. Open `bot/promptly_bot.py` in your editor
2. Have `bot/git_methods.py` open in another tab for reference

## Step 1: Add Import (Line ~84)

**Find this:**
```python
        from .code_reviewer import get_code_reviewer
```

**Add this line right after it:**
```python
        from .git_handler import GitHandler
```

**Result should look like:**
```python
        from .code_reviewer import get_code_reviewer
        from .git_handler import GitHandler
```

## Step 2: Initialize Git Handler (Line ~91)

**Find this:**
```python
        self.code_reviewer = get_code_reviewer()

        logger.info(f"Initialized Promptly bot: {user_id}")
```

**Replace with:**
```python
        self.code_reviewer = get_code_reviewer()

        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"Git handler initialized for repo: {git_repo_path}")
        except ValueError as e:
            logger.warning(f"Git handler init failed: {e}")
            self.git_handler = None

        logger.info(f"Initialized Promptly bot: {user_id}")
```

## Step 3: Add Git Command Routing (Line ~282)

**Find this:**
```python
        elif cmd_type == 'list':
            return await self.cmd_list(command, room)
        else:
            return {
```

**Replace with:**
```python
        elif cmd_type == 'list':
            return await self.cmd_list(command, room)
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
        else:
            return {
```

## Step 4: Add Git Command Methods (Line ~522)

**Find the end of `cmd_chat` method** (around line 520):
```python
            return {
                "body": f"💬 I heard you, but I'm having trouble responding right now.\n\nTry a specific command like `@promptly help` to see what I can do!",
                "html": "<p>💬 I heard you, but I'm having trouble responding right now.</p><p>Try <code>@promptly help</code> to see what I can do!</p>"
            }

    async def send_message(
```

**Add git methods BETWEEN `cmd_chat` and `send_message`.**

Copy and paste this block:

```python
            return {
                "body": f"💬 I heard you, but I'm having trouble responding right now.\n\nTry a specific command like `@promptly help` to see what I can do!",
                "html": "<p>💬 I heard you, but I'm having trouble responding right now.</p><p>Try <code>@promptly help</code> to see what I can do!</p>"
            }

    # ========== Git Commands (ChatOps Phase 1) ==========

    async def cmd_git_status(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git status command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            status = self.git_handler.status(short=True)
            branch = self.git_handler.get_current_branch()

            body = f"Git Status\n\nBranch: {branch}\n\n{status}"
            html = f"<p><strong>Git Status</strong></p><p>Branch: <code>{branch}</code></p><pre>{status}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git status error: {e}")
            return {
                "body": f"Git status failed: {e}",
                "html": f"<p>Git status failed: <code>{e}</code></p>"
            }

    async def cmd_git_log(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git log command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            log = self.git_handler.log(max_count=5, oneline=True)
            branch = self.git_handler.get_current_branch()

            body = f"Recent Commits ({branch})\n\n{log}"
            html = f"<p><strong>Recent Commits</strong> ({branch})</p><pre>{log}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git log error: {e}")
            return {
                "body": f"Git log failed: {e}",
                "html": f"<p>Git log failed: <code>{e}</code></p>"
            }

    async def cmd_git_diff(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git diff command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            diff = self.git_handler.diff()

            if not diff:
                return {
                    "body": "No changes to show (working tree clean)",
                    "html": "<p>No changes to show (working tree clean)</p>"
                }

            if len(diff) > 2000:
                diff = diff[:2000] + "\n\n... (truncated, too long for chat)"

            body = f"Git Diff\n\n{diff}"
            html = f"<p><strong>Git Diff</strong></p><pre>{diff}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git diff error: {e}")
            return {
                "body": f"Git diff failed: {e}",
                "html": f"<p>Git diff failed: <code>{e}</code></p>"
            }

    async def cmd_git_branch(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git branch command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            branches = self.git_handler.branch()

            body = f"Git Branches\n\n{branches}"
            html = f"<p><strong>Git Branches</strong></p><pre>{branches}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git branch error: {e}")
            return {
                "body": f"Git branch failed: {e}",
                "html": f"<p>Git branch failed: <code>{e}</code></p>"
            }

    async def cmd_git_commit(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git commit command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            message = command.get('message', '')
            if not message:
                return {
                    "body": 'Commit message required. Usage: @promptly git commit "your message"',
                    "html": '<p>Commit message required. Usage: <code>@promptly git commit "your message"</code></p>'
                }

            result = self.git_handler.commit(message, add_all=True)

            body = f"Commit Created\n\nMessage: {message}\n\n{result}"
            html = f"<p><strong>Commit Created</strong></p><p>Message: <code>{message}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git commit error: {e}")
            return {
                "body": f"Git commit failed: {e}",
                "html": f"<p>Git commit failed: <code>{e}</code></p>"
            }

    async def cmd_git_push(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git push command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            branch = self.git_handler.get_current_branch()
            result = self.git_handler.push(branch=branch)

            body = f"Pushed to Remote\n\nBranch: {branch}\n\n{result}"
            html = f"<p><strong>Pushed to Remote</strong></p><p>Branch: <code>{branch}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git push error: {e}")
            return {
                "body": f"Git push failed: {e}\n\nNote: Push requires authentication",
                "html": f"<p>Git push failed: <code>{e}</code></p><p>Note: Push requires authentication</p>"
            }

    async def cmd_git_pull(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle git pull command"""
        if not self.git_handler:
            return {
                "body": "Git not configured. Set GIT_REPO_PATH in .env",
                "html": "<p>Git not configured. Set <code>GIT_REPO_PATH</code> in .env</p>"
            }

        try:
            result = self.git_handler.pull()

            body = f"Pulled from Remote\n\n{result}"
            html = f"<p><strong>Pulled from Remote</strong></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Git pull error: {e}")
            return {
                "body": f"Git pull failed: {e}",
                "html": f"<p>Git pull failed: <code>{e}</code></p>"
            }

    async def send_message(
```

## Step 5: Save and Test

1. Save `bot/promptly_bot.py`
2. Run: `python run_bot.py`
3. In Matrix, try: `@promptly git status`

If you see git status output, it works!

## Verification

After integration, verify these lines exist:

- Line ~84: `from .git_handler import GitHandler`
- Line ~95: `self.git_handler = GitHandler(...)`
- Line ~290: `elif cmd_type == 'git-status':`
- Line ~525: `async def cmd_git_status(self, ...)`

## Troubleshooting

**Import Error**: Make sure the import is indented with 8 spaces (2 levels)

**AttributeError**: Make sure `self.git_handler = ...` is in `__init__` method

**Command not found**: Make sure routing elif blocks are before the final `else:`

**Syntax Error**: Check indentation - all git methods should be indented at class level (4 spaces)

---

That's it! Once these 4 steps are done, git integration is complete and you can start using git commands from Matrix chat!

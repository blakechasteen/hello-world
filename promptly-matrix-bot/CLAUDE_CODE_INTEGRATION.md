# Claude Code CLI Integration - Complete Guide

## What We're Building

Add these commands to Promptly Matrix Bot:
```
@promptly review src/auth.py          → Claude reviews the code
@promptly explain src/middleware.py   → Claude explains how it works
@promptly refactor old.py "use async" → Claude refactors the code
```

## Files Created

✅ **bot/claude_bridge.py** (150 lines) - CLI subprocess wrapper

## Integration Steps

### Step 1: Add Commands to command_parser.py

**Find this section** (around line 38):
```python
        'git-pull': r'(?:@promptly(?:bot)?\s+git\s+pull|!git\s+pull)',
    }
```

**Add Claude Code commands before the closing `}`:**
```python
        'git-pull': r'(?:@promptly(?:bot)?\s+git\s+pull|!git\s+pull)',
        # Claude Code commands
        'claude-review': r'(?:@promptly(?:bot)?\s+(?:claude\s+)?review\s+(.+)|!review\s+(.+))',
        'claude-explain': r'(?:@promptly(?:bot)?\s+(?:claude\s+)?explain\s+(.+)|!explain\s+(.+))',
        'claude-refactor': r'(?:@promptly(?:bot)?\s+(?:claude\s+)?refactor\s+(\S+)\s+"([^"]+)"|!refactor\s+(\S+)\s+"([^"]+)")',
    }
```

### Step 2: Add Command Extraction (command_parser.py)

**Find the git command section** (around line 140):
```python
        elif cmd_type == 'git-pull':
            return {'type': 'git-pull'}

        else:
            return {'type': 'unknown'}
```

**Add Claude Code extraction before the `else:`:**
```python
        elif cmd_type == 'git-pull':
            return {'type': 'git-pull'}

        # Claude Code commands
        elif cmd_type == 'claude-review':
            file_path = groups[0] if groups[0] else groups[1]
            return {
                'type': 'claude-review',
                'file_path': file_path
            }

        elif cmd_type == 'claude-explain':
            file_path = groups[0] if groups[0] else groups[1]
            return {
                'type': 'claude-explain',
                'file_path': file_path
            }

        elif cmd_type == 'claude-refactor':
            # First format: @promptly refactor file.py "instruction"
            # Second format: !refactor file.py "instruction"
            file_path = groups[0] if groups[0] else groups[2]
            instruction = groups[1] if groups[1] else groups[3]
            return {
                'type': 'claude-refactor',
                'file_path': file_path,
                'instruction': instruction
            }

        else:
            return {'type': 'unknown'}
```

### Step 3: Initialize Claude Bridge (promptly_bot.py)

**Find the git handler initialization** (around line 95):
```python
        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"Git handler initialized for repo: {git_repo_path}")
        except ValueError as e:
            logger.warning(f"Git handler init failed: {e}")
            self.git_handler = None
```

**Add Claude Bridge right after:**
```python
        # Initialize Claude Code bridge (ChatOps Phase 2)
        from .claude_bridge import ClaudeBridge

        claude_path = os.getenv("CLAUDE_PATH", "claude")
        try:
            self.claude_bridge = ClaudeBridge(
                claude_path=claude_path,
                repo_path=git_repo_path
            )
            if self.claude_bridge.is_available():
                logger.info("Claude Code CLI available")
            else:
                logger.warning("Claude Code CLI not found in PATH")
                self.claude_bridge = None
        except Exception as e:
            logger.warning(f"Claude Bridge init failed: {e}")
            self.claude_bridge = None
```

### Step 4: Add Command Routing (promptly_bot.py)

**Find the git command routing** (around line 290):
```python
        elif cmd_type == 'git-pull':
            return await self.cmd_git_pull(command, room)
        else:
            return {
```

**Add Claude Code routing before the `else:`:**
```python
        elif cmd_type == 'git-pull':
            return await self.cmd_git_pull(command, room)
        elif cmd_type == 'claude-review':
            return await self.cmd_claude_review(command, room)
        elif cmd_type == 'claude-explain':
            return await self.cmd_claude_explain(command, room)
        elif cmd_type == 'claude-refactor':
            return await self.cmd_claude_refactor(command, room)
        else:
            return {
```

### Step 5: Add Command Methods (promptly_bot.py)

**Find the git methods section** (after `cmd_git_pull`, before `send_message`):

**Add these methods:**

```python
    # ========== Claude Code Commands (ChatOps Phase 2) ==========

    async def cmd_claude_review(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle Claude Code review command"""
        if not self.claude_bridge:
            return {
                "body": "Claude Code not available. Install from https://claude.ai/download",
                "html": "<p>Claude Code not available. Install from <a href='https://claude.ai/download'>claude.ai/download</a></p>"
            }

        file_path = command.get('file_path', '')
        if not file_path:
            return {
                "body": 'Usage: @promptly review <file_path>',
                "html": '<p>Usage: <code>@promptly review &lt;file_path&gt;</code></p>'
            }

        # Show processing message
        await self.send_message(room.room_id, f"Reviewing {file_path}...")

        try:
            result = self.claude_bridge.review(file_path)

            if len(result) > 3000:
                result = result[:3000] + "\n\n... (truncated)"

            body = f"Code Review: {file_path}\n\n{result}"
            html = f"<p><strong>Code Review:</strong> <code>{file_path}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Claude review error: {e}")
            return {
                "body": f"Review failed: {e}",
                "html": f"<p>Review failed: <code>{e}</code></p>"
            }

    async def cmd_claude_explain(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle Claude Code explain command"""
        if not self.claude_bridge:
            return {
                "body": "Claude Code not available. Install from https://claude.ai/download",
                "html": "<p>Claude Code not available. Install from <a href='https://claude.ai/download'>claude.ai/download</a></p>"
            }

        file_path = command.get('file_path', '')
        if not file_path:
            return {
                "body": 'Usage: @promptly explain <file_path>',
                "html": '<p>Usage: <code>@promptly explain &lt;file_path&gt;</code></p>'
            }

        # Show processing message
        await self.send_message(room.room_id, f"Explaining {file_path}...")

        try:
            result = self.claude_bridge.explain(file_path)

            if len(result) > 3000:
                result = result[:3000] + "\n\n... (truncated)"

            body = f"Explanation: {file_path}\n\n{result}"
            html = f"<p><strong>Explanation:</strong> <code>{file_path}</code></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Claude explain error: {e}")
            return {
                "body": f"Explanation failed: {e}",
                "html": f"<p>Explanation failed: <code>{e}</code></p>"
            }

    async def cmd_claude_refactor(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle Claude Code refactor command"""
        if not self.claude_bridge:
            return {
                "body": "Claude Code not available. Install from https://claude.ai/download",
                "html": "<p>Claude Code not available. Install from <a href='https://claude.ai/download'>claude.ai/download</a></p>"
            }

        file_path = command.get('file_path', '')
        instruction = command.get('instruction', '')

        if not file_path or not instruction:
            return {
                "body": 'Usage: @promptly refactor <file_path> "instruction"',
                "html": '<p>Usage: <code>@promptly refactor &lt;file_path&gt; "instruction"</code></p>'
            }

        # Show processing message
        await self.send_message(room.room_id, f"Refactoring {file_path}...")

        try:
            result = self.claude_bridge.refactor(file_path, instruction)

            if len(result) > 3000:
                result = result[:3000] + "\n\n... (truncated)"

            body = f"Refactoring: {file_path}\n\nInstruction: {instruction}\n\n{result}"
            html = f"<p><strong>Refactoring:</strong> <code>{file_path}</code></p><p>Instruction: <em>{instruction}</em></p><pre>{result}</pre>"

            return {"body": body, "html": html}

        except Exception as e:
            logger.error(f"Claude refactor error: {e}")
            return {
                "body": f"Refactoring failed: {e}",
                "html": f"<p>Refactoring failed: <code>{e}</code></p>"
            }
```

### Step 6: Update .env

Add to `.env`:
```bash
# Claude Code Integration (Phase 2 ChatOps)
CLAUDE_PATH=claude
```

## Testing

After integration:

1. **Check if Claude Code is installed:**
   ```bash
   claude --version
   ```

2. **Test from Matrix:**
   ```
   @promptly review bot/git_handler.py
   @promptly explain bot/claude_bridge.py
   ```

## Command Examples

### Code Review
```
@promptly review src/auth.py
@promptly claude review src/middleware.py
!review bot/promptly_bot.py
```

### Code Explanation
```
@promptly explain bot/git_handler.py
@promptly claude explain src/database.py
!explain utils/helpers.py
```

### Refactoring
```
@promptly refactor old_code.py "use async/await"
@promptly claude refactor legacy.py "add type hints"
!refactor messy.py "improve readability"
```

## Migration Path to HoloLoom

This CLI integration is Phase 1. Later, we can add:

**Phase 2: Add Memory** (hybrid)
- Store review results in HoloLoom
- Track what files were reviewed
- "What did we review today?"

**Phase 3: Async Queue** (full HoloLoom)
- Long-running tasks don't block bot
- Claude Code picks up tasks asynchronously
- Better for large codebases

For now, CLI is simple and works immediately!

## Troubleshooting

**"Claude Code not available"**
- Install Claude Code from https://claude.ai/download
- Ensure `claude` is in PATH
- Try setting `CLAUDE_PATH=/full/path/to/claude` in .env

**"Command timed out"**
- Large files take time to review
- Increase timeout in claude_bridge.py
- Or break into smaller files

**"File not found"**
- File paths are relative to `GIT_REPO_PATH`
- Use: `@promptly review bot/promptly_bot.py` not `/full/path/`

## Success!

Once integrated, you'll have:
- ✅ Git operations from Matrix
- ✅ Claude Code reviews from Matrix
- ✅ Full ChatOps development environment!

Next: HoloLoom memory integration for team context!

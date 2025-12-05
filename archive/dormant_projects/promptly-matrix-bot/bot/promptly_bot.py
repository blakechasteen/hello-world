#!/usr/bin/env python3
"""
Promptly Matrix Bot - Core Implementation

Application Service bot for Matrix.org providing chat-native AI reliability.
"""

import asyncio
import logging
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

# Matrix SDK
from nio import (
    AsyncClient,
    MatrixRoom,
    RoomMessageText,
    InviteEvent,
    LoginResponse,
    JoinError,
    RoomSendError,
)

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "mythRL"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PromptlyBot:
    """
    Promptly Matrix Bot

    Responds to @promptly mentions in Matrix rooms with AI reliability
    commands (optimize, run, code-review, etc.)
    """

    def __init__(
        self,
        homeserver: str,
        user_id: str,
        access_token: Optional[str] = None,
        device_id: Optional[str] = None,
    ):
        """
        Initialize Promptly bot.

        Args:
            homeserver: Matrix homeserver URL (e.g., https://matrix.org)
            user_id: Bot user ID (e.g., @promptly:matrix.org)
            access_token: Optional access token (or login with password)
            device_id: Optional device ID for encryption
        """
        self.homeserver = homeserver
        self.user_id = user_id

        # Extract localpart from full user_id (@username:server -> username)
        localpart = user_id.split(':')[0].lstrip('@')
        self.client = AsyncClient(homeserver, localpart, device_id=device_id)

        if access_token:
            self.client.access_token = access_token

        # Register event callbacks
        self.client.add_event_callback(self.message_callback, RoomMessageText)
        self.client.add_event_callback(self.invite_callback, InviteEvent)

        # State
        self.started = False

        # Initialize Promptly Core integration
        from .promptly_core import get_promptly_core
        from .response_formatter import ResponseFormatter
        from .state_manager import get_state_manager
        from .approval_workflow import get_approval_manager
        from .code_reviewer import get_code_reviewer

        self.promptly_core = get_promptly_core()
        self.formatter = ResponseFormatter()
        self.state = get_state_manager(os.getenv("REDIS_URL"))
        self.approval_manager = None  # Initialize after client ready
        self.code_reviewer = get_code_reviewer()


        # Initialize Git handler (ChatOps Phase 1)
        git_repo_path = os.getenv("GIT_REPO_PATH")
        try:
            from .git_handler import GitHandler
            self.git_handler = GitHandler(git_repo_path) if git_repo_path else None
            if self.git_handler:
                logger.info(f"Git handler initialized for repo: {git_repo_path}")
        except ValueError as e:
            logger.warning(f"Git handler init failed: {e}")
            self.git_handler = None
        except Exception as e:
            logger.warning(f"Git handler init error: {e}")
            self.git_handler = None

        # Initialize Claude Code bridge (ChatOps Phase 2)
        api_key = os.getenv("ANTHROPIC_API_KEY")
        try:
            from .claude_bridge import ClaudeBridge
            self.claude_bridge = ClaudeBridge(
                api_key=api_key,
                repo_path=git_repo_path
            )
            if self.claude_bridge.is_available():
                logger.info("Claude API available")
            else:
                logger.warning("Claude API not available (set ANTHROPIC_API_KEY)")
                self.claude_bridge = None
        except Exception as e:
            logger.warning(f"Claude Bridge init failed: {e}")
            self.claude_bridge = None

        logger.info(f"Initialized Promptly bot: {user_id}")

    async def login(self, password: str) -> bool:
        """
        Login to Matrix homeserver.

        Args:
            password: Bot password

        Returns:
            True if login successful, False otherwise
        """
        try:
            response = await self.client.login(password, device_name="Promptly Bot")

            if isinstance(response, LoginResponse):
                logger.info(f"Logged in as {self.user_id}")
                logger.info(f"Device ID: {response.device_id}")
                logger.info(f"Access token: {response.access_token[:20]}...")
                return True
            else:
                logger.error(f"Login failed: {response}")
                return False

        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    async def start(self):
        """Start the bot (sync loop)"""
        if self.started:
            logger.warning("Bot already started")
            return

        self.started = True
        logger.info("Starting Promptly bot...")

        # Initial sync
        logger.info("Performing initial sync...")
        await self.client.sync(timeout=30000)

        logger.info("✅ Promptly bot started and synced")
        logger.info("Listening for messages...")

        # Forever sync loop
        try:
            await self.client.sync_forever(timeout=30000)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the bot"""
        logger.info("Stopping Promptly bot...")
        await self.client.close()
        self.started = False
        logger.info("✅ Bot stopped")

    async def invite_callback(self, room: MatrixRoom, event: InviteEvent):
        """
        Handle room invites - auto-join all rooms.

        Args:
            room: Matrix room
            event: Invite event
        """
        logger.info(f"Invited to {room.room_id} by {event.sender}")

        try:
            result = await self.client.join(room.room_id)

            if isinstance(result, JoinError):
                logger.error(f"Failed to join {room.room_id}: {result.message}")
            else:
                logger.info(f"✅ Joined {room.room_id}")

                # Send welcome message
                await self.send_message(
                    room.room_id,
                    "👋 Hi! I'm Promptly, your AI reliability assistant.\n\n"
                    "Try:\n"
                    "• `@promptly help` - Show available commands\n"
                    "• `@promptly optimize` - Optimize a prompt\n"
                    "• `@promptly run <workflow>` - Run a workflow\n\n"
                    "Learn more: https://github.com/promptly/promptly"
                )

        except Exception as e:
            logger.error(f"Error joining room: {e}")

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText):
        """
        Handle incoming messages.

        Args:
            room: Matrix room
            event: Message event
        """
        # Ignore own messages
        if event.sender == self.client.user_id or event.sender == self.user_id:
            return

        # Ignore old messages (from before bot started)
        if not self.started:
            return

        message = event.body
        sender = event.sender

        # Check if bot is mentioned
        if not self.is_mentioned(message):
            return

        logger.info(f"Message from {sender} in {room.display_name}: {message[:50]}...")

        # Parse and execute command
        try:
            response = await self.handle_command(message, room, event)

            if response:
                await self.send_response(room.room_id, response, reply_to=event.event_id)

        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
            await self.send_error(
                room.room_id,
                f"❌ Error: {str(e)}\n\nTry `@promptly help` for usage.",
                reply_to=event.event_id
            )

    def is_mentioned(self, message: str) -> bool:
        """
        Check if bot is mentioned in message.

        Args:
            message: Message text

        Returns:
            True if bot is mentioned, False otherwise
        """
        # Check for @promptly, @promptlybot, or !command (case insensitive)
        msg_lower = message.lower()
        return (
            "@promptly" in msg_lower or
            "@promptlybot" in msg_lower or
            msg_lower.startswith("!")
        )

    async def handle_command(
        self,
        message: str,
        room: MatrixRoom,
        event: RoomMessageText
    ) -> Optional[Dict[str, str]]:
        """
        Parse and execute command.

        Args:
            message: Message text
            room: Matrix room
            event: Message event

        Returns:
            Response dict with 'body' and 'html' keys, or None
        """
        # Parse command
        from .command_parser import CommandParser
        parser = CommandParser()
        command = parser.parse(message)

        if not command:
            # No structured command found - use conversational AI
            return await self.cmd_chat(message, room)

        cmd_type = command['type']
        logger.info(f"Executing command: {cmd_type}")

        # Route to appropriate handler
        if cmd_type == 'help':
            return await self.cmd_help(command)
        elif cmd_type == 'optimize':
            return await self.cmd_optimize(command, room)
        elif cmd_type == 'run':
            return await self.cmd_run(command, room)
        elif cmd_type == 'code-review':
            return await self.cmd_code_review(command, room)
        elif cmd_type == 'save':
            return await self.cmd_save(command, room)
        elif cmd_type == 'list':
            return await self.cmd_list(command, room)
        # Git commands (ChatOps Phase 1)
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
        # Claude Code commands (ChatOps Phase 2)
        elif cmd_type == 'claude-review':
            return await self.cmd_claude_review(command, room)
        elif cmd_type == 'claude-explain':
            return await self.cmd_claude_explain(command, room)
        elif cmd_type == 'claude-refactor':
            return await self.cmd_claude_refactor(command, room)
        else:
            return {
                "body": f"❌ Unknown command: {cmd_type}",
                "html": f"<p>❌ Unknown command: <code>{cmd_type}</code></p>"
            }

    async def cmd_help(self, command: Dict) -> Dict[str, str]:
        """Handle help command"""
        help_text = """**Promptly Commands**

**Core Commands:**
• `@promptly help` - Show this help
• `@promptly optimize` - Optimize a prompt
• `@promptly run <workflow> "<input>"` - Run a workflow
• `@promptly save <name>` - Save current prompt
• `@promptly list` - List saved prompts

**Advanced Commands:**
• `@promptly code-review [code]` - Review code
• `@promptly schema` - Build structured schema
• `@promptly verify "<statement>"` - Verify claim
• `@promptly refine` - Multi-pass refinement

**Examples:**

Optimize a prompt:
```
@promptly optimize
Task: Answer customer questions
Examples: [
  {"input": "How to reset password?", "output": "Click..."},
  {"input": "Where is my order?", "output": "Check..."}
]
```

Run a workflow:
```
@promptly run qa_basic "What is Thompson Sampling?"
```

**Learn More:** https://github.com/promptly/promptly
"""

        help_html = """<h3>Promptly Commands</h3>

<h4>Core Commands:</h4>
<ul>
<li><code>@promptly help</code> - Show this help</li>
<li><code>@promptly optimize</code> - Optimize a prompt</li>
<li><code>@promptly run &lt;workflow&gt; "&lt;input&gt;"</code> - Run a workflow</li>
<li><code>@promptly save &lt;name&gt;</code> - Save current prompt</li>
<li><code>@promptly list</code> - List saved prompts</li>
</ul>

<h4>Advanced Commands:</h4>
<ul>
<li><code>@promptly code-review [code]</code> - Review code</li>
<li><code>@promptly schema</code> - Build structured schema</li>
<li><code>@promptly verify "&lt;statement&gt;"</code> - Verify claim</li>
<li><code>@promptly refine</code> - Multi-pass refinement</li>
</ul>

<p><strong>Learn More:</strong> <a href="https://github.com/promptly/promptly">github.com/promptly/promptly</a></p>
"""

        return {"body": help_text, "html": help_html}

    async def cmd_optimize(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle optimize command"""
        task = command.get('task', 'Unknown task')
        examples = command.get('examples', [])

        if not examples:
            body = """❌ No examples provided.

**Usage Format:**
```
@promptly optimize
Task: Your task description here
Examples: [
  {"input": "example question 1", "output": "example answer 1"},
  {"input": "example question 2", "output": "example answer 2"},
  {"input": "example question 3", "output": "example answer 3"}
]
```

**Example - Pitching Ouroboros:**
```
@promptly optimize
Task: Write a compelling pitch for Ouroboros - Adverse Drug Events recognition software
Examples: [
  {"input": "What problem does it solve?", "output": "Ouroboros detects adverse drug events in clinical data using advanced NLP"},
  {"input": "Who is it for?", "output": "Healthcare providers, pharma companies, and regulatory agencies"},
  {"input": "What makes it unique?", "output": "Real-time monitoring with 95% accuracy and seamless EHR integration"}
]
```

**Tips:**
• Provide 3-5 examples for best results
• Examples should show the style/format you want
• Be specific in your task description
"""

            html = """<p>❌ No examples provided.</p>
<h4>Usage Format:</h4>
<pre><code>@promptly optimize
Task: Your task description here
Examples: [
  {"input": "example question 1", "output": "example answer 1"},
  {"input": "example question 2", "output": "example answer 2"}
]</code></pre>

<h4>Example - Pitching Ouroboros:</h4>
<pre><code>@promptly optimize
Task: Write a compelling pitch for Ouroboros
Examples: [
  {"input": "What problem does it solve?", "output": "Detects adverse drug events..."},
  {"input": "Who is it for?", "output": "Healthcare providers..."}
]</code></pre>

<p><strong>Tips:</strong></p>
<ul>
<li>Provide 3-5 examples for best results</li>
<li>Examples should show the style/format you want</li>
<li>Be specific in your task description</li>
</ul>
"""

            return {"body": body, "html": html}

        # Send progress update
        await self.send_progress(room.room_id, self.formatter.format_progress("Analyzing task", 25))

        # Run optimization
        result = await self.promptly_core.optimize_prompt(task, examples)

        await self.send_progress(room.room_id, self.formatter.format_progress("Optimization complete", 100))

        # Format response
        return self.formatter.format_optimization_result(result)

    async def cmd_run(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle run command"""
        workflow = command.get('workflow', 'unknown')
        input_text = command.get('input', '')

        # Send progress
        await self.send_progress(room.room_id, f"Executing workflow '{workflow}'...")

        # Run workflow
        result = await self.promptly_core.run_workflow(workflow, input_text)

        # Format response
        return self.formatter.format_workflow_result(result)

    async def cmd_code_review(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle code-review command"""
        code = command.get('code', '')
        language = command.get('language')

        if not code:
            return {
                "body": "❌ No code provided.\n\nUsage:\n```\n@promptly code-review\n```python\ndef my_function():\n    pass\n```\n```",
                "html": "<p>❌ No code provided.</p><p>See <code>@promptly help</code> for usage.</p>"
            }

        # Send progress
        await self.send_progress(room.room_id, "Analyzing code for security and quality issues...")

        # Review code
        result = self.code_reviewer.review(code, language)

        # Format response
        return self.formatter.format_code_review_result(result)


    async def cmd_save(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle save command"""
        name = command.get('name', 'unknown')

        # Get last optimization result from context
        # For now, save placeholder data
        # TODO: Store actual optimization result in context

        prompt_data = {
            "task": "Placeholder task",
            "signature": {"inputs": ["input"], "outputs": ["output"]},
            "metrics": {"accuracy": 0.90}
        }

        success = self.state.save_prompt(room.room_id, name, prompt_data)
        return self.formatter.format_save_result(name, success)

    async def cmd_list(self, command: Dict, room: MatrixRoom) -> Dict[str, str]:
        """Handle list command"""
        prompts = self.state.list_prompts(room.room_id)
        return self.formatter.format_list_result(prompts, room.display_name or "this room")

    async def cmd_chat(self, message: str, room: MatrixRoom) -> Dict[str, str]:
        """Handle conversational chat using Ollama"""
        import os
        import ollama

        # Remove bot mentions from message to get clean query
        clean_message = message
        for mention in ['@promptly', '@promptlybot']:
            clean_message = clean_message.replace(mention, '').strip()

        try:
            # Use Ollama for conversational response
            model = os.getenv("LM_MODEL", "ollama/llama3.2:3b").replace("ollama/", "")

            response = ollama.chat(
                model=model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are Promptly, a helpful AI assistant integrated into Matrix chat. You help with prompt optimization, code review, and general questions. Be concise and friendly.'
                    },
                    {
                        'role': 'user',
                        'content': clean_message
                    }
                ]
            )

            answer = response['message']['content']

            return {
                "body": answer,
                "html": f"<p>{answer}</p>"
            }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return {
                "body": f"💬 I heard you, but I'm having trouble responding right now.\n\nTry a specific command like `@promptly help` to see what I can do!",
                "html": "<p>💬 I heard you, but I'm having trouble responding right now.</p><p>Try <code>@promptly help</code> to see what I can do!</p>"
            }


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
        self,
        room_id: str,
        message: str,
        msgtype: str = "m.text"
    ):
        """
        Send a simple text message to room.

        Args:
            room_id: Room ID
            message: Message text
            msgtype: Message type (m.text or m.notice)
        """
        try:
            await self.client.room_send(
                room_id,
                message_type="m.room.message",
                content={
                    "msgtype": msgtype,
                    "body": message
                }
            )
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def send_response(
        self,
        room_id: str,
        response: Dict[str, str],
        reply_to: Optional[str] = None
    ):
        """
        Send formatted response to room.

        Args:
            room_id: Room ID
            response: Dict with 'body' and 'html' keys
            reply_to: Optional event ID to reply to (threading)
        """
        content = {
            "msgtype": "m.text",
            "body": response['body'],
            "format": "org.matrix.custom.html",
            "formatted_body": response['html']
        }

        # Add reply threading
        if reply_to:
            content["m.relates_to"] = {
                "m.in_reply_to": {
                    "event_id": reply_to
                }
            }

        try:
            result = await self.client.room_send(
                room_id,
                message_type="m.room.message",
                content=content
            )

            if isinstance(result, RoomSendError):
                logger.error(f"Failed to send response: {result.message}")

        except Exception as e:
            logger.error(f"Error sending response: {e}")

    async def send_error(
        self,
        room_id: str,
        error: str,
        reply_to: Optional[str] = None
    ):
        """
        Send error message to room.

        Args:
            room_id: Room ID
            error: Error message
            reply_to: Optional event ID to reply to
        """
        await self.send_response(
            room_id,
            {"body": error, "html": f"<p>{error}</p>"},
            reply_to=reply_to
        )

    async def send_progress(self, room_id: str, message: str):
        """
        Send progress update (m.notice).

        Args:
            room_id: Room ID
            message: Progress message
        """
        await self.send_message(room_id, message, msgtype="m.notice")


async def main():
    """Main entry point for bot"""
    # Get credentials from environment
    homeserver = os.getenv("MATRIX_HOMESERVER", "https://matrix.org")
    user_id = os.getenv("MATRIX_USER_ID", "@promptly:matrix.org")
    password = os.getenv("MATRIX_PASSWORD")
    access_token = os.getenv("MATRIX_ACCESS_TOKEN")

    if not password and not access_token:
        logger.error("Must provide MATRIX_PASSWORD or MATRIX_ACCESS_TOKEN")
        sys.exit(1)

    # Create bot
    bot = PromptlyBot(
        homeserver=homeserver,
        user_id=user_id,
        access_token=access_token
    )

    # Login if using password
    if password and not access_token:
        success = await bot.login(password)
        if not success:
            logger.error("Login failed")
            sys.exit(1)

    # Start bot
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())

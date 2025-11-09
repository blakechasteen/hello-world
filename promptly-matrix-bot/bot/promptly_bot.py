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
        if event.sender == self.client.user_id:
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
        # Check for @promptly mention (case insensitive)
        return "@promptly" in message.lower()

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
            return {
                "body": "⚠️ Could not parse command.\n\nTry `@promptly help` for usage.",
                "html": "<p>⚠️ Could not parse command.</p><p>Try <code>@promptly help</code> for usage.</p>"
            }

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
            return {
                "body": "❌ No examples provided.\n\nUsage:\n```\n@promptly optimize\nTask: Your task description\nExamples: [\n  {\"input\": \"example 1\", \"output\": \"result 1\"},\n  {\"input\": \"example 2\", \"output\": \"result 2\"}\n]\n```",
                "html": "<p>❌ No examples provided.</p><p>See <code>@promptly help</code> for usage.</p>"
            }

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

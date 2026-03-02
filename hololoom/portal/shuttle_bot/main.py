"""
Shuttle Bot - Matrix ChatOps interface for the Loom.

Connects to a Matrix room and responds to commands for
managing and dispatching jobs to the distributed compute Loom.

Commands:
    !loom status - Show Loom overview
    !nodes - List all nodes
    !job run <module> <json> - Run a WASM job
    !modules - List available WASM modules
    !help - Show help
"""

import argparse
import asyncio
import sys
from typing import Optional

from .config import ShuttleConfig, load_config
from .scheduler import SimpleScheduler
from .commands import CommandHandler
from ..shared.logging import configure_logging, get_logger

# Try to import matrix-nio
try:
    from nio import (
        AsyncClient,
        AsyncClientConfig,
        MatrixRoom,
        RoomMessageText,
        LoginResponse,
        LoginError,
        JoinResponse,
        JoinError,
    )
    NIO_AVAILABLE = True
except ImportError:
    NIO_AVAILABLE = False

logger = get_logger(__name__, component="shuttle")


class ShuttleBot:
    """
    Matrix bot for Loom ChatOps.

    Connects to a Matrix homeserver, joins a room, and responds
    to commands prefixed with the configured command prefix.
    """

    def __init__(self, config: ShuttleConfig):
        """
        Initialize Shuttle bot.

        Args:
            config: ShuttleConfig instance
        """
        self.config = config
        self.client: Optional[AsyncClient] = None
        self.scheduler = SimpleScheduler(
            portal_url=config.portal_url,
            shared_secret=config.shared_secret,
        )
        self.command_handler = CommandHandler(
            portal_url=config.portal_url,
            shared_secret=config.shared_secret,
            scheduler=self.scheduler,
            job_timeout=config.job_timeout_seconds,
        )

    async def start(self) -> None:
        """Start the bot and sync forever."""
        if not NIO_AVAILABLE:
            logger.error("matrix-nio not installed. Run: pip install matrix-nio")
            return

        logger.info(f"Starting Shuttle Bot for {self.config.homeserver}")

        # Create client
        client_config = AsyncClientConfig(
            max_limit_exceeded=0,
            max_timeouts=0,
        )
        self.client = AsyncClient(
            homeserver=self.config.homeserver,
            user=self.config.user_id,
            config=client_config,
        )

        # Login
        logger.info(f"Logging in as {self.config.user_id}")
        response = await self.client.login(
            password=self.config.password,
            device_name="ShuttleBot",
        )

        if isinstance(response, LoginError):
            logger.error(f"Login failed: {response.message}")
            return

        logger.info(f"Logged in. Device ID: {response.device_id}")

        # Join room
        logger.info(f"Joining room {self.config.room_id}")
        response = await self.client.join(self.config.room_id)

        if isinstance(response, JoinError):
            logger.error(f"Failed to join room: {response.message}")
            return

        logger.info(f"Joined room: {self.config.room_id}")

        # Register message callback
        self.client.add_event_callback(self._on_message, RoomMessageText)

        # Send startup message
        await self._send_message(
            f"🚀 **Shuttle Bot Online**\n"
            f"Connected to Portal at {self.config.portal_url}\n"
            f"Type `{self.config.command_prefix}help` for commands."
        )

        # Sync forever
        logger.info("Starting sync loop...")
        await self.client.sync_forever(timeout=30000)

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if self.client:
            await self._send_message("🛑 **Shuttle Bot Offline**")
            await self.client.close()
            self.client = None

    async def _on_message(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """Handle incoming messages."""
        # Ignore messages from ourselves
        if event.sender == self.client.user_id:
            return

        # Only process messages in our room
        if room.room_id != self.config.room_id:
            return

        message = event.body.strip()

        # Check for command prefix
        if not message.startswith(self.config.command_prefix):
            return

        logger.info(f"Received command from {event.sender}: {message}")

        try:
            # Handle command
            response = await self.command_handler.handle(message)

            # Truncate if too long
            if len(response) > self.config.max_message_length:
                response = response[:self.config.max_message_length - 50]
                response += "\n\n... (message truncated)"

            await self._send_message(response)

        except Exception as e:
            logger.error(f"Error handling command: {e}")
            await self._send_message(f"❌ Error: {e}")

    async def _send_message(self, message: str) -> None:
        """Send a message to the room."""
        if not self.client:
            return

        try:
            await self.client.room_send(
                room_id=self.config.room_id,
                message_type="m.room.message",
                content={
                    "msgtype": "m.text",
                    "body": message,
                    "format": "org.matrix.custom.html",
                    "formatted_body": self._markdown_to_html(message),
                },
            )
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def _markdown_to_html(self, text: str) -> str:
        """
        Simple markdown to HTML conversion.

        Converts:
        - **bold** -> <strong>bold</strong>
        - `code` -> <code>code</code>
        - ```code block``` -> <pre><code>code block</code></pre>
        - newlines -> <br>
        """
        import re

        # Code blocks first
        text = re.sub(
            r"```(\w*)\n?(.*?)```",
            r"<pre><code>\2</code></pre>",
            text,
            flags=re.DOTALL,
        )

        # Inline code
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

        # Bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

        # Newlines (but not inside pre blocks)
        # Simple approach: just convert all newlines
        text = text.replace("\n", "<br>")

        return text


async def run_bot(config: ShuttleConfig) -> None:
    """Run the bot with proper signal handling."""
    bot = ShuttleBot(config)

    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await bot.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Shuttle Bot")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--homeserver", help="Override Matrix homeserver")
    parser.add_argument("--user", help="Override bot user ID")
    parser.add_argument("--room", help="Override room ID")
    args = parser.parse_args()

    # Check dependencies
    if not NIO_AVAILABLE:
        print("Error: matrix-nio not installed.")
        print("Install with: pip install matrix-nio")
        sys.exit(1)

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.homeserver:
        config.homeserver = args.homeserver
    if args.user:
        config.user_id = args.user
    if args.room:
        config.room_id = args.room

    configure_logging(level=config.log_level, json_format=config.log_json)

    # Run bot
    asyncio.run(run_bot(config))


if __name__ == "__main__":
    main()
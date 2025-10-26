#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matrix Bot Client
==================
Async Matrix.org bot using matrix-nio for chatops integration.

This is component #1: The Matrix protocol client that handles:
- Room connections and event listening
- Message sending/receiving
- User authentication
- Rate limiting and error handling

Architecture:
    Matrix Homeserver <-> MatrixBot <-> ChatOpsOrchestrator <-> HoloLoom

Dependencies:
    pip install matrix-nio aiofiles python-magic
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from nio import (
        AsyncClient,
        MatrixRoom,
        RoomMessageText,
        RoomMessageImage,
        RoomMessageFile,
        LoginResponse,
        SyncResponse,
        RoomSendResponse,
        UploadResponse,
    )
    NIO_AVAILABLE = True
except ImportError:
    NIO_AVAILABLE = False
    print("Warning: matrix-nio not installed. Run: pip install matrix-nio")


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MatrixBotConfig:
    """Configuration for Matrix bot."""

    # Connection details
    homeserver_url: str  # e.g., "https://matrix.org"
    user_id: str         # e.g., "@mybot:matrix.org"
    access_token: Optional[str] = None  # Use token OR password
    password: Optional[str] = None
    device_id: str = "HOLOLOOM_BOT"
    device_name: str = "HoloLoom ChatOps Bot"

    # Rooms to join
    rooms: List[str] = field(default_factory=list)  # Room IDs or aliases

    # Bot behavior
    command_prefix: str = "!"          # Command prefix (e.g., !help)
    respond_to_mentions: bool = True   # Respond when mentioned
    respond_to_dm: bool = True         # Respond to direct messages

    # Admin controls
    admin_users: List[str] = field(default_factory=list)  # User IDs with admin access
    allowed_users: List[str] = field(default_factory=list)  # Whitelist (empty = all)

    # Rate limiting
    rate_limit_messages: int = 10      # Max messages per window
    rate_limit_window_sec: int = 60    # Window size in seconds

    # Storage
    store_path: str = "./matrix_store"  # Persistent storage for encryption keys

    # Timeouts
    sync_timeout_ms: int = 30000       # Sync timeout (30 seconds)
    message_timeout_sec: int = 120     # Max time to process a message


# ============================================================================
# Matrix Bot Client
# ============================================================================

class MatrixBot:
    """
    Async Matrix.org bot client.

    Handles all Matrix protocol interactions:
    - Authentication and room joining
    - Event listening (messages, images, files)
    - Message sending with formatting
    - Rate limiting per user
    - Command parsing and dispatch

    Usage:
        config = MatrixBotConfig(
            homeserver_url="https://matrix.org",
            user_id="@mybot:matrix.org",
            password="secret",
            rooms=["#hololoom:matrix.org"],
            admin_users=["@admin:matrix.org"]
        )

        bot = MatrixBot(config)
        bot.register_handler("help", help_handler)
        await bot.start()
    """

    def __init__(self, config: MatrixBotConfig):
        """
        Initialize Matrix bot.

        Args:
            config: Bot configuration
        """
        if not NIO_AVAILABLE:
            raise ImportError("matrix-nio not installed. Run: pip install matrix-nio")

        self.config = config
        self.client = AsyncClient(
            homeserver=config.homeserver_url,
            user=config.user_id,
            device_id=config.device_id,
            store_path=config.store_path
        )

        # Command handlers: command_name -> async callable
        self.handlers: Dict[str, Callable] = {}

        # Rate limiting: user_id -> list of timestamps
        self.rate_limits: Dict[str, List[datetime]] = {}

        # Running state
        self.running = False

        # Default message handler (catches non-command messages)
        self.default_handler: Optional[Callable] = None

        logger.info(f"MatrixBot initialized for {config.user_id}")

    # ========================================================================
    # Authentication & Setup
    # ========================================================================

    async def login(self) -> bool:
        """
        Login to Matrix homeserver.

        Returns:
            True if successful
        """
        # Try access token first
        if self.config.access_token:
            self.client.access_token = self.config.access_token
            logger.info("Using provided access token")
            return True

        # Fall back to password login
        if self.config.password:
            response = await self.client.login(
                password=self.config.password,
                device_name=self.config.device_name
            )

            if isinstance(response, LoginResponse):
                logger.info(f"Logged in successfully as {self.config.user_id}")
                logger.info(f"Access token: {self.client.access_token[:20]}...")
                return True
            else:
                logger.error(f"Login failed: {response}")
                return False

        logger.error("No access token or password provided")
        return False

    async def join_rooms(self) -> None:
        """Join configured rooms."""
        for room in self.config.rooms:
            response = await self.client.join(room)
            if hasattr(response, 'room_id'):
                logger.info(f"Joined room: {response.room_id}")
            else:
                logger.warning(f"Failed to join {room}: {response}")

    # ========================================================================
    # Event Handling
    # ========================================================================

    async def message_callback(self, room: MatrixRoom, event: RoomMessageText) -> None:
        """
        Handle incoming text messages.

        Args:
            room: Room where message was sent
            event: Message event
        """
        # Ignore own messages
        if event.sender == self.config.user_id:
            return

        # Check rate limiting
        if not self._check_rate_limit(event.sender):
            logger.warning(f"Rate limit exceeded for {event.sender}")
            await self.send_message(
                room.room_id,
                f"{event.sender}: Rate limit exceeded. Please slow down."
            )
            return

        # Check permissions
        if not self._check_permissions(event.sender):
            logger.warning(f"Unauthorized user: {event.sender}")
            return

        # Parse message
        message_text = event.body.strip()

        # Check if it's a command
        if message_text.startswith(self.config.command_prefix):
            await self._handle_command(room, event, message_text)
        # Check if bot was mentioned
        elif self.config.respond_to_mentions and self.config.user_id in message_text:
            await self._handle_mention(room, event, message_text)
        # Check if it's a DM
        elif self.config.respond_to_dm and len(room.users) == 2:
            await self._handle_dm(room, event, message_text)
        # Otherwise, pass to default handler
        elif self.default_handler:
            await self.default_handler(room, event, message_text)

    async def _handle_command(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
        message: str
    ) -> None:
        """
        Handle command message.

        Args:
            room: Room where command was sent
            event: Message event
            message: Full message text
        """
        # Remove prefix and parse
        command_line = message[len(self.config.command_prefix):]
        parts = command_line.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        logger.info(f"Command from {event.sender}: {command}")

        # Check if handler exists
        if command in self.handlers:
            try:
                await self.handlers[command](room, event, args)
            except Exception as e:
                logger.error(f"Error handling command {command}: {e}", exc_info=True)
                await self.send_message(
                    room.room_id,
                    f"Error executing command: {str(e)}"
                )
        else:
            await self.send_message(
                room.room_id,
                f"Unknown command: {command}. Try {self.config.command_prefix}help"
            )

    async def _handle_mention(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
        message: str
    ) -> None:
        """Handle message that mentions the bot."""
        # Remove bot mention
        clean_message = message.replace(self.config.user_id, "").strip()

        if self.default_handler:
            await self.default_handler(room, event, clean_message)

    async def _handle_dm(
        self,
        room: MatrixRoom,
        event: RoomMessageText,
        message: str
    ) -> None:
        """Handle direct message."""
        if self.default_handler:
            await self.default_handler(room, event, message)

    # ========================================================================
    # Command Registration
    # ========================================================================

    def register_handler(self, command: str, handler: Callable) -> None:
        """
        Register a command handler.

        Args:
            command: Command name (without prefix)
            handler: Async callable(room, event, args) -> None
        """
        self.handlers[command] = handler
        logger.info(f"Registered handler for command: {command}")

    def set_default_handler(self, handler: Callable) -> None:
        """
        Set default handler for non-command messages.

        Args:
            handler: Async callable(room, event, message) -> None
        """
        self.default_handler = handler
        logger.info("Default message handler registered")

    # ========================================================================
    # Message Sending
    # ========================================================================

    async def send_message(
        self,
        room_id: str,
        message: str,
        markdown: bool = False
    ) -> None:
        """
        Send text message to room.

        Args:
            room_id: Room to send to
            message: Message text
            markdown: Whether to format as markdown
        """
        content = {
            "msgtype": "m.text",
            "body": message
        }

        if markdown:
            # Convert markdown to HTML for rich formatting
            content["format"] = "org.matrix.custom.html"
            content["formatted_body"] = self._markdown_to_html(message)

        response = await self.client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content=content
        )

        if isinstance(response, RoomSendResponse):
            logger.debug(f"Message sent: {response.event_id}")
        else:
            logger.error(f"Failed to send message: {response}")

    async def send_typing(self, room_id: str, typing: bool = True, timeout: int = 5000) -> None:
        """
        Send typing indicator.

        Args:
            room_id: Room ID
            typing: Whether typing or not
            timeout: Timeout in milliseconds
        """
        await self.client.room_typing(room_id, typing=typing, timeout=timeout)

    # ========================================================================
    # Access Control
    # ========================================================================

    def _check_permissions(self, user_id: str) -> bool:
        """
        Check if user has permission to interact.

        Args:
            user_id: User to check

        Returns:
            True if allowed
        """
        # Empty whitelist = allow all
        if not self.config.allowed_users:
            return True

        # Check whitelist
        return user_id in self.config.allowed_users or user_id in self.config.admin_users

    def is_admin(self, user_id: str) -> bool:
        """
        Check if user is admin.

        Args:
            user_id: User to check

        Returns:
            True if admin
        """
        return user_id in self.config.admin_users

    def _check_rate_limit(self, user_id: str) -> bool:
        """
        Check rate limit for user.

        Args:
            user_id: User to check

        Returns:
            True if within limits
        """
        now = datetime.now()

        # Initialize if new user
        if user_id not in self.rate_limits:
            self.rate_limits[user_id] = []

        # Remove old timestamps outside window
        cutoff = now.timestamp() - self.config.rate_limit_window_sec
        self.rate_limits[user_id] = [
            ts for ts in self.rate_limits[user_id]
            if ts.timestamp() > cutoff
        ]

        # Check limit
        if len(self.rate_limits[user_id]) >= self.config.rate_limit_messages:
            return False

        # Add new timestamp
        self.rate_limits[user_id].append(now)
        return True

    # ========================================================================
    # Utilities
    # ========================================================================

    def _markdown_to_html(self, markdown: str) -> str:
        """
        Convert basic markdown to HTML.

        Args:
            markdown: Markdown text

        Returns:
            HTML string
        """
        # Simple conversions (for full support, use markdown library)
        html = markdown

        # Bold
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'__(.+?)__', r'<strong>\1</strong>', html)

        # Italic
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'_(.+?)_', r'<em>\1</em>', html)

        # Code
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)

        # Line breaks
        html = html.replace('\n', '<br/>')

        return html

    # ========================================================================
    # Main Loop
    # ========================================================================

    async def start(self) -> None:
        """
        Start bot and run until stopped.

        This is the main event loop that:
        1. Logs in
        2. Joins rooms
        3. Syncs events continuously
        """
        # Login
        if not await self.login():
            logger.error("Login failed, cannot start bot")
            return

        # Join rooms
        await self.join_rooms()

        # Register callbacks
        self.client.add_event_callback(self.message_callback, RoomMessageText)

        # Initial sync
        logger.info("Performing initial sync...")
        await self.client.sync(timeout=self.config.sync_timeout_ms)

        # Main loop
        self.running = True
        logger.info("Bot started, listening for messages...")

        try:
            while self.running:
                await self.client.sync_forever(
                    timeout=self.config.sync_timeout_ms,
                    full_state=False
                )
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop bot gracefully."""
        self.running = False
        logger.info("Stopping bot...")
        await self.client.close()
        logger.info("Bot stopped")


# ============================================================================
# Built-in Command Handlers
# ============================================================================

async def help_handler(room: MatrixRoom, event: RoomMessageText, args: str) -> None:
    """Built-in help command."""
    # This would be populated by the ChatOps orchestrator
    # For now, just a placeholder
    pass


async def ping_handler(room: MatrixRoom, event: RoomMessageText, args: str) -> None:
    """Built-in ping command."""
    # This would be implemented by the bot instance
    pass


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("="*80)
    print("Matrix Bot Demo")
    print("="*80)
    print()

    if not NIO_AVAILABLE:
        print("ERROR: matrix-nio not installed")
        print("Install with: pip install matrix-nio")
        sys.exit(1)

    # Example configuration
    config = MatrixBotConfig(
        homeserver_url="https://matrix.org",
        user_id="@your_bot:matrix.org",  # Replace with your bot user
        password="your_password",         # Replace with your password
        rooms=["#test:matrix.org"],       # Replace with your test room
        admin_users=["@your_user:matrix.org"],
        command_prefix="!"
    )

    # Create bot
    bot = MatrixBot(config)

    # Register sample handlers
    async def handle_ping(room, event, args):
        await bot.send_message(room.room_id, f"Pong! 🏓 ({args})" if args else "Pong! 🏓")

    async def handle_echo(room, event, args):
        await bot.send_message(room.room_id, args or "Nothing to echo")

    async def handle_help(room, event, args):
        help_text = """
**HoloLoom Bot Commands:**

• `!ping` - Check if bot is alive
• `!echo <text>` - Echo back text
• `!help` - Show this help message

More commands coming soon!
"""
        await bot.send_message(room.room_id, help_text, markdown=True)

    bot.register_handler("ping", handle_ping)
    bot.register_handler("echo", handle_echo)
    bot.register_handler("help", handle_help)

    print("Bot configured. Starting...")
    print("Press Ctrl+C to stop")
    print()

    # Run bot
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\nBot stopped by user")

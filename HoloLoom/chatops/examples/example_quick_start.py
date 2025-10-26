#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom ChatOps - Quick Start Example
=======================================
Minimal working example to get started quickly.

This demonstrates:
1. Basic Matrix bot setup
2. ChatOps integration
3. Simple command handling
4. Conversation memory

Usage:
    1. Edit the configuration below
    2. Run: python example_quick_start.py
    3. Send "!ping" in your Matrix room
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from holoLoom.chatops.matrix_bot import MatrixBot, MatrixBotConfig
from holoLoom.chatops.chatops_bridge import ChatOpsOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# EDIT THESE VALUES
MATRIX_CONFIG = {
    "homeserver_url": "https://matrix.org",
    "user_id": "@your_bot:matrix.org",  # Your bot's Matrix ID

    # Use ONE of these authentication methods:
    "access_token": os.getenv("MATRIX_ACCESS_TOKEN"),  # Recommended
    # "password": "your_password",  # Or use password

    "rooms": [
        "#your-test-room:matrix.org"  # Room to join
    ],

    "admin_users": [
        "@your_user:matrix.org"  # Your admin account
    ]
}


# ============================================================================
# Main Bot
# ============================================================================

async def main():
    """Run the chatbot."""

    print("="*80)
    print("HoloLoom ChatOps - Quick Start")
    print("="*80)
    print()

    # Validate configuration
    if not MATRIX_CONFIG["access_token"] and not MATRIX_CONFIG.get("password"):
        print("ERROR: No authentication configured!")
        print()
        print("Set MATRIX_ACCESS_TOKEN environment variable:")
        print("  export MATRIX_ACCESS_TOKEN='your_token_here'")
        print()
        print("Or edit this file and set password directly (less secure)")
        sys.exit(1)

    if "@your_bot:matrix.org" in MATRIX_CONFIG["user_id"]:
        print("ERROR: Please edit MATRIX_CONFIG and set your bot's user_id")
        sys.exit(1)

    print(f"Homeserver: {MATRIX_CONFIG['homeserver_url']}")
    print(f"Bot User: {MATRIX_CONFIG['user_id']}")
    print(f"Rooms: {MATRIX_CONFIG['rooms']}")
    print()

    # Create bot configuration
    bot_config = MatrixBotConfig(
        homeserver_url=MATRIX_CONFIG["homeserver_url"],
        user_id=MATRIX_CONFIG["user_id"],
        access_token=MATRIX_CONFIG.get("access_token"),
        password=MATRIX_CONFIG.get("password"),
        rooms=MATRIX_CONFIG["rooms"],
        admin_users=MATRIX_CONFIG["admin_users"],
        command_prefix="!",
        store_path="./demo_matrix_store"
    )

    # Create Matrix bot
    bot = MatrixBot(bot_config)

    # Create ChatOps orchestrator
    chatops = ChatOpsOrchestrator(
        memory_store_path="./demo_chatops_memory",
        context_limit=5,
        enable_memory_storage=True
    )

    # Connect chatops to bot
    chatops.connect_bot(bot)

    # ========================================================================
    # Register Commands
    # ========================================================================

    async def handle_ping(room, event, args):
        """Simple ping command."""
        await bot.send_message(room.room_id, "Pong! 🏓")

    async def handle_hello(room, event, args):
        """Greeting command."""
        sender = event.sender.split(":")[0][1:]  # Extract username
        greeting = f"Hello, {sender}! 👋\n\nI'm HoloLoom ChatOps bot."
        await bot.send_message(room.room_id, greeting)

    async def handle_help(room, event, args):
        """Help command."""
        help_text = """**HoloLoom ChatOps - Quick Start**

**Commands:**
• `!ping` - Check if bot is alive
• `!hello` - Get a greeting
• `!echo <text>` - Echo back text
• `!count` - Count messages in conversation
• `!help` - Show this help

**Features:**
✓ Conversation memory in knowledge graph
✓ Multi-user tracking
✓ Command handling
✓ Markdown formatting

To ask questions, just mention me or send a DM!
"""
        await bot.send_message(room.room_id, help_text, markdown=True)

    async def handle_echo(room, event, args):
        """Echo command."""
        if not args:
            await bot.send_message(room.room_id, "Usage: !echo <text>")
            return

        await bot.send_message(room.room_id, f"You said: {args}")

    async def handle_count(room, event, args):
        """Count messages in conversation."""
        conv = chatops.conversations.get(room.room_id)

        if not conv:
            await bot.send_message(room.room_id, "No conversation history yet")
            return

        stats = f"""**Conversation Statistics:**

• Total messages: {len(conv.message_history)}
• Participants: {len(conv.participants)}
• Started: {conv.created_at.strftime('%Y-%m-%d %H:%M')}
• Last activity: {conv.last_activity.strftime('%Y-%m-%d %H:%M')}

**Recent messages:**
{conv.to_context_string(5)}
"""
        await bot.send_message(room.room_id, stats, markdown=True)

    # Register all commands
    bot.register_handler("ping", handle_ping)
    bot.register_handler("hello", handle_hello)
    bot.register_handler("help", handle_help)
    bot.register_handler("echo", handle_echo)
    bot.register_handler("count", handle_count)

    # ========================================================================
    # Run Bot
    # ========================================================================

    print("Starting bot...")
    print("Try these commands in your Matrix room:")
    print("  !ping")
    print("  !hello")
    print("  !help")
    print()
    print("Press Ctrl+C to stop")
    print()

    try:
        # Start chatops
        await chatops.start()

        # Start bot (runs until interrupted)
        await bot.start()

    except KeyboardInterrupt:
        print("\n\nStopping bot...")
        await chatops.stop()
        await bot.stop()
        print("Stopped")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    print()
    print("HoloLoom ChatOps Quick Start")
    print("============================")
    print()

    # Check if matrix-nio is installed
    try:
        import nio
    except ImportError:
        print("ERROR: matrix-nio not installed")
        print()
        print("Install with:")
        print("  pip install matrix-nio aiofiles python-magic")
        print()
        sys.exit(1)

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExiting...")

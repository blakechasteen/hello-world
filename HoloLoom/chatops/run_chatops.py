#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom ChatOps Runner
=======================
Main entry point for running the Matrix chatbot.

Usage:
    python run_chatops.py --config config.yaml

    # With environment variables
    MATRIX_ACCESS_TOKEN=xxx python run_chatops.py

    # Debug mode
    python run_chatops.py --config config.yaml --debug
"""

import asyncio
import argparse
import logging
import signal
import sys
import os
from pathlib import Path
from typing import Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not installed. Using default config.")

# Add repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from holoLoom.chatops.matrix_bot import MatrixBot, MatrixBotConfig
from holoLoom.chatops.chatops_bridge import ChatOpsOrchestrator
from holoLoom.chatops.conversation_memory import ConversationMemory

try:
    from holoLoom.chatops import ChatOpsSkills
    from holoLoom.config import Config
    FULL_FEATURES = True
except ImportError:
    FULL_FEATURES = False
    print("Warning: Some features unavailable (ChatOps skills or HoloLoom config)")


logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Loading
# ============================================================================

def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dict
    """
    if not YAML_AVAILABLE:
        return get_default_config()

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return get_default_config()


def get_default_config() -> dict:
    """Get default configuration."""
    return {
        "matrix": {
            "homeserver_url": os.getenv("MATRIX_HOMESERVER", "https://matrix.org"),
            "user_id": os.getenv("MATRIX_USER_ID", "@bot:matrix.org"),
            "access_token": os.getenv("MATRIX_ACCESS_TOKEN"),
            "password": os.getenv("MATRIX_PASSWORD"),
            "device_id": "HOLOLOOM_BOT",
            "store_path": "./matrix_store",
            "rooms": [],
            "command_prefix": "!",
            "admin_users": [],
            "rate_limit": {
                "messages_per_window": 10,
                "window_seconds": 60
            }
        },
        "hololoom": {
            "mode": "fast",
            "memory": {
                "store_path": "./chatops_memory",
                "enable_kg_storage": True,
                "context_limit": 10
            }
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    }


def apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides to config.

    Args:
        config: Base configuration

    Returns:
        Updated configuration
    """
    # Matrix overrides
    if os.getenv("MATRIX_ACCESS_TOKEN"):
        config["matrix"]["access_token"] = os.getenv("MATRIX_ACCESS_TOKEN")
    if os.getenv("MATRIX_PASSWORD"):
        config["matrix"]["password"] = os.getenv("MATRIX_PASSWORD")
    if os.getenv("MATRIX_USER_ID"):
        config["matrix"]["user_id"] = os.getenv("MATRIX_USER_ID")
    if os.getenv("MATRIX_HOMESERVER"):
        config["matrix"]["homeserver_url"] = os.getenv("MATRIX_HOMESERVER")

    return config


# ============================================================================
# Bot Setup
# ============================================================================

def create_matrix_config(config: dict) -> MatrixBotConfig:
    """
    Create MatrixBotConfig from config dict.

    Args:
        config: Configuration dict

    Returns:
        MatrixBotConfig instance
    """
    matrix_cfg = config["matrix"]

    return MatrixBotConfig(
        homeserver_url=matrix_cfg["homeserver_url"],
        user_id=matrix_cfg["user_id"],
        access_token=matrix_cfg.get("access_token"),
        password=matrix_cfg.get("password"),
        device_id=matrix_cfg.get("device_id", "HOLOLOOM_BOT"),
        device_name=matrix_cfg.get("device_name", "HoloLoom ChatOps Bot"),
        rooms=matrix_cfg.get("rooms", []),
        command_prefix=matrix_cfg.get("command_prefix", "!"),
        respond_to_mentions=matrix_cfg.get("respond_to_mentions", True),
        respond_to_dm=matrix_cfg.get("respond_to_dm", True),
        admin_users=matrix_cfg.get("admin_users", []),
        allowed_users=matrix_cfg.get("allowed_users", []),
        rate_limit_messages=matrix_cfg.get("rate_limit", {}).get("messages_per_window", 10),
        rate_limit_window_sec=matrix_cfg.get("rate_limit", {}).get("window_seconds", 60),
        store_path=matrix_cfg.get("store_path", "./matrix_store")
    )


def create_hololoom_config(config: dict):
    """
    Create HoloLoom config from config dict.

    Args:
        config: Configuration dict

    Returns:
        HoloLoom Config instance or None
    """
    if not FULL_FEATURES:
        return None

    hololoom_cfg = config.get("hololoom", {})
    mode = hololoom_cfg.get("mode", "fast")

    if mode == "bare":
        return Config.bare()
    elif mode == "fused":
        return Config.fused()
    else:  # fast
        return Config.fast()


def setup_logging(config: dict, debug: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        config: Configuration dict
        debug: Debug mode flag
    """
    log_cfg = config.get("logging", {})

    # Determine log level
    level_str = "DEBUG" if debug else log_cfg.get("level", "INFO")
    level = getattr(logging, level_str)

    # Format
    log_format = log_cfg.get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format
    )

    # File logging if enabled
    file_cfg = log_cfg.get("file", {})
    if file_cfg.get("enabled", False):
        log_path = Path(file_cfg.get("path", "./logs/chatops.log"))
        log_path.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=file_cfg.get("max_bytes", 10485760),  # 10MB
            backupCount=file_cfg.get("backup_count", 5)
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


# ============================================================================
# Command Handlers
# ============================================================================

def register_builtin_commands(bot: MatrixBot, chatops: ChatOpsOrchestrator, skills: Optional[ChatOpsSkills] = None):
    """
    Register built-in command handlers.

    Args:
        bot: MatrixBot instance
        chatops: ChatOpsOrchestrator instance
        skills: Optional ChatOpsSkills instance
    """

    # Ping command
    async def handle_ping(room, event, args):
        await bot.send_message(room.room_id, "Pong! 🏓")

    # Help command
    async def handle_help(room, event, args):
        if skills:
            result = await skills.help(args if args else None)
            await bot.send_message(room.room_id, result.output, markdown=True)
        else:
            help_text = """**HoloLoom ChatOps**

Basic Commands:
• `!ping` - Check if bot is alive
• `!help` - This help message
• `!status` - System status

For full features, ensure HoloLoom and Promptly are installed.
"""
            await bot.send_message(room.room_id, help_text, markdown=True)

    # Status command
    async def handle_status(room, event, args):
        detailed = "detailed" in args.lower() if args else False

        if skills:
            result = await skills.status(detailed=detailed)
            await bot.send_message(room.room_id, result.output, markdown=True)
        else:
            stats = chatops.get_statistics()
            status_text = f"""**System Status:**

• Status: ✓ Online
• Active Conversations: {stats['active_conversations']}
• Total Messages: {stats['total_messages']}
"""
            await bot.send_message(room.room_id, status_text, markdown=True)

    # Register handlers
    bot.register_handler("ping", handle_ping)
    bot.register_handler("help", handle_help)
    bot.register_handler("status", handle_status)

    # Register skill commands if available
    if skills:
        async def handle_search(room, event, args):
            if not args:
                await bot.send_message(room.room_id, "Usage: !search <query>")
                return
            result = await skills.search(args)
            await bot.send_message(room.room_id, result.output, markdown=True)

        async def handle_remember(room, event, args):
            if not args:
                await bot.send_message(room.room_id, "Usage: !remember <info>")
                return
            result = await skills.remember(args)
            await bot.send_message(room.room_id, result.output, markdown=True)

        async def handle_recall(room, event, args):
            if not args:
                await bot.send_message(room.room_id, "Usage: !recall <topic>")
                return
            result = await skills.recall(args)
            await bot.send_message(room.room_id, result.output, markdown=True)

        async def handle_summarize(room, event, args):
            # Get conversation context
            conv = chatops.conversations.get(room.room_id)
            if not conv:
                await bot.send_message(room.room_id, "No conversation history found")
                return

            messages = conv.get_recent_messages(20)
            format_type = args.strip() if args else "bullets"
            result = await skills.summarize(messages, format=format_type)
            await bot.send_message(room.room_id, result.output, markdown=True)

        bot.register_handler("search", handle_search)
        bot.register_handler("remember", handle_remember)
        bot.register_handler("recall", handle_recall)
        bot.register_handler("summarize", handle_summarize)

    logger.info(f"Registered {len(bot.handlers)} command handlers")


# ============================================================================
# Main Runner
# ============================================================================

class ChatOpsRunner:
    """
    Main runner for HoloLoom ChatOps.

    Manages:
    - Bot lifecycle
    - Graceful shutdown
    - Error recovery
    """

    def __init__(self, config: dict):
        """
        Initialize runner.

        Args:
            config: Configuration dict
        """
        self.config = config
        self.bot: Optional[MatrixBot] = None
        self.chatops: Optional[ChatOpsOrchestrator] = None
        self.skills: Optional[ChatOpsSkills] = None
        self.running = False

    async def setup(self) -> None:
        """Setup all components."""
        logger.info("Setting up HoloLoom ChatOps...")

        # Create HoloLoom config
        hololoom_config = create_hololoom_config(self.config)

        # Create ChatOps orchestrator
        self.chatops = ChatOpsOrchestrator(
            hololoom_config=hololoom_config,
            memory_store_path=self.config["hololoom"]["memory"]["store_path"],
            context_limit=self.config["hololoom"]["memory"]["context_limit"],
            enable_memory_storage=self.config["hololoom"]["memory"]["enable_kg_storage"]
        )

        # Create Promptly skills if available
        if FULL_FEATURES:
            try:
                self.skills = ChatOpsSkills(hololoom_bridge=None)
                logger.info("Promptly skills initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Promptly skills: {e}")
                self.skills = None

        # Create Matrix bot
        matrix_config = create_matrix_config(self.config)
        self.bot = MatrixBot(matrix_config)

        # Connect chatops to bot
        self.chatops.connect_bot(self.bot)

        # Register command handlers
        register_builtin_commands(self.bot, self.chatops, self.skills)

        logger.info("Setup complete")

    async def start(self) -> None:
        """Start the bot."""
        logger.info("Starting HoloLoom ChatOps...")

        # Start chatops orchestrator
        await self.chatops.start()

        # Start bot (runs until stopped)
        self.running = True
        await self.bot.start()

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        logger.info("Stopping HoloLoom ChatOps...")
        self.running = False

        if self.chatops:
            await self.chatops.stop()

        if self.bot:
            await self.bot.stop()

        logger.info("Stopped")

    async def run(self) -> None:
        """Run the bot with error recovery."""
        await self.setup()

        # Setup signal handlers
        loop = asyncio.get_event_loop()

        def signal_handler():
            logger.info("Received shutdown signal")
            asyncio.create_task(self.stop())

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, signal_handler)

        # Run
        try:
            await self.start()
        except Exception as e:
            logger.error(f"Error running bot: {e}", exc_info=True)
        finally:
            await self.stop()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HoloLoom ChatOps - Matrix.org chatbot with neural decision-making"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="HoloLoom/chatops/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config = apply_env_overrides(config)

    # Setup logging
    setup_logging(config, debug=args.debug)

    # Print banner
    print("="*80)
    print("HoloLoom ChatOps - Matrix.org Integration")
    print("="*80)
    print()
    print(f"Homeserver: {config['matrix']['homeserver_url']}")
    print(f"User: {config['matrix']['user_id']}")
    print(f"Mode: {config['hololoom']['mode']}")
    print(f"Rooms: {len(config['matrix'].get('rooms', []))}")
    print()
    print("Starting bot... (Press Ctrl+C to stop)")
    print()

    # Run bot
    runner = ChatOpsRunner(config)

    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

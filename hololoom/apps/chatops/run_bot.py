#!/usr/bin/env python3
"""
HoloLoom Matrix Bot - Main Launcher
====================================
Integrated Matrix chatops bot with full HoloLoom capabilities.

Features:
- Full weaving orchestrator integration
- MCTS decision-making via chat
- Memory management commands
- Multi-backend hybrid memory
- Context-aware responses

Usage:
    python run_bot.py --config bot_config.yaml
    python run_bot.py --homeserver https://matrix.org --user @bot:matrix.org
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matrix_bot import MatrixBot, MatrixBotConfig
from hololoom_handlers import HoloLoomMatrixHandlers


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hololoom_bot.log')
        ]
    )


async def main():
    """Main bot entry point."""
    parser = argparse.ArgumentParser(
        description='HoloLoom Matrix Bot - AI-powered chatops',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using environment variables
    export MATRIX_HOMESERVER="https://matrix.org"
    export MATRIX_USER="@mybot:matrix.org"
    export MATRIX_PASSWORD="secret"
    python run_bot.py

    # Using command line
    python run_bot.py --homeserver https://matrix.org --user @bot:matrix.org --password secret

    # Using access token
    python run_bot.py --homeserver https://matrix.org --user @bot:matrix.org --token ACCESS_TOKEN

Commands available in chat:
    !weave <query>       - Execute HoloLoom weaving cycle
    !memory add <text>   - Add to knowledge base
    !memory search <q>   - Search memory
    !analyze <text>      - Analyze with MCTS
    !stats               - Show system stats
    !help                - Command help
        """
    )

    # Connection args
    parser.add_argument(
        '--homeserver',
        default=os.getenv('MATRIX_HOMESERVER', 'https://matrix.org'),
        help='Matrix homeserver URL'
    )
    parser.add_argument(
        '--user',
        default=os.getenv('MATRIX_USER'),
        help='Bot user ID (e.g., @bot:matrix.org)'
    )
    parser.add_argument(
        '--password',
        default=os.getenv('MATRIX_PASSWORD'),
        help='Bot password'
    )
    parser.add_argument(
        '--token',
        default=os.getenv('MATRIX_ACCESS_TOKEN'),
        help='Access token (alternative to password)'
    )

    # Room args
    parser.add_argument(
        '--rooms',
        nargs='+',
        default=os.getenv('MATRIX_ROOMS', '').split(',') if os.getenv('MATRIX_ROOMS') else [],
        help='Rooms to join (space-separated)'
    )

    # Bot behavior
    parser.add_argument(
        '--prefix',
        default='!',
        help='Command prefix (default: !)'
    )
    parser.add_argument(
        '--admin',
        nargs='+',
        default=[],
        help='Admin user IDs (space-separated)'
    )

    # HoloLoom config
    parser.add_argument(
        '--hololoom-mode',
        choices=['bare', 'fast', 'fused'],
        default='fast',
        help='HoloLoom configuration mode'
    )
    parser.add_argument(
        '--mcts-sims',
        type=int,
        default=50,
        help='MCTS simulations per decision'
    )

    # System args
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--store-path',
        default='./matrix_store',
        help='Path for Matrix encryption keys'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate required args
    if not args.user:
        parser.error("--user is required (or set MATRIX_USER environment variable)")

    if not args.password and not args.token:
        parser.error("Either --password or --token is required")

    # Print banner
    print("="*80)
    print("HoloLoom Matrix Bot")
    print("="*80)
    print()
    print(f"Homeserver: {args.homeserver}")
    print(f"User: {args.user}")
    print(f"Rooms: {args.rooms if args.rooms else 'None (will accept invites)'}")
    print(f"Command prefix: {args.prefix}")
    print(f"HoloLoom mode: {args.hololoom_mode}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print()
    print("="*80)
    print()

    # Create bot configuration
    bot_config = MatrixBotConfig(
        homeserver_url=args.homeserver,
        user_id=args.user,
        password=args.password,
        access_token=args.token,
        rooms=args.rooms,
        command_prefix=args.prefix,
        admin_users=args.admin,
        store_path=args.store_path,
        respond_to_mentions=True,
        respond_to_dm=True
    )

    try:
        # Create Matrix bot
        logger.info("Creating Matrix bot...")
        bot = MatrixBot(bot_config)

        # Create HoloLoom handlers
        logger.info(f"Initializing HoloLoom (mode={args.hololoom_mode})...")
        handlers = HoloLoomMatrixHandlers(bot, config_mode=args.hololoom_mode)

        # Register all handlers
        logger.info("Registering command handlers...")
        handlers.register_all()

        logger.info("✓ Bot initialization complete")
        print()
        print("Bot is ready! Commands available:")
        print("  !weave <query>      - Execute weaving cycle")
        print("  !memory add <text>  - Add to memory")
        print("  !memory search <q>  - Search memory")
        print("  !analyze <text>     - Analyze with MCTS")
        print("  !stats              - System statistics")
        print("  !help               - Show all commands")
        print()
        print("Press Ctrl+C to stop")
        print("="*80)
        print()

        # Start bot
        await bot.start()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Bot error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if 'handlers' in locals():
            await handlers.shutdown()
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nBot stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        sys.exit(1)

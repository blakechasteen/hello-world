#!/usr/bin/env python3
"""
Promptly Matrix Bot - Main Entry Point

Loads configuration and starts the bot.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, errors="replace")

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.promptly_bot import PromptlyBot


async def main():
    """Load config and start bot."""

    # Load Matrix config
    config_path = Path(__file__).parent / "config" / "matrix_config.json"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("\nPlease create config/matrix_config.json with:")
        print("""
{
  "homeserver": "https://matrix.org",
  "user": "@your-bot-username:matrix.org",
  "auto_join_rooms": true
}
        """)
        print("\nAnd set environment variable:")
        print("  MATRIX_PASSWORD=your_bot_password")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Validate required fields
    required = ["homeserver", "user"]
    missing = [field for field in required if field not in config]

    if missing:
        print(f"❌ Missing required fields in config: {', '.join(missing)}")
        return

    # Create bot instance
    print(f"🤖 Starting Promptly bot: {config['user']}")
    print(f"🌐 Homeserver: {config['homeserver']}")

    # Check if access token is in config
    access_token = config.get("access_token") or os.getenv("MATRIX_ACCESS_TOKEN")

    if access_token:
        # Use access token (for SSO accounts)
        print("🔑 Using access token authentication...")
        bot = PromptlyBot(
            homeserver=config["homeserver"],
            user_id=config["user"],
            access_token=access_token
        )
        print("✅ Authenticated with access token!")
    else:
        # Use password login
        password = os.getenv("MATRIX_PASSWORD")

        if not password:
            # Prompt for password if not in environment
            import getpass
            print("🔐 Enter bot password (input hidden):")
            password = getpass.getpass("Password: ")

            if not password:
                print("❌ Password is required!")
                return

        bot = PromptlyBot(
            homeserver=config["homeserver"],
            user_id=config["user"]
        )

        # Login
        print("🔑 Logging in...")
        success = await bot.login(password)

        if not success:
            print("❌ Login failed! Check your credentials.")
            return

        print("✅ Login successful!")

    # Start bot
    print("🚀 Starting bot event loop...")
    print("💬 Bot is now listening for messages")
    print("   (Press Ctrl+C to stop)")

    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Bot stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

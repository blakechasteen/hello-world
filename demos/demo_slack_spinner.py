"""
Slack Spinner Demo
==================

Demonstrates the SlackSpinner for ingesting Slack messages.

Features shown:
- Channel message extraction
- Thread reply extraction
- Emoji reaction tracking
- User enrichment
- Importance scoring

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.spinningWheel.slack_spinner import SlackSpinner, spin_slack_channel


async def demo_slack_spinner():
    """Run Slack spinner demo with mock data."""
    print("=" * 70)
    print("Slack Spinner Demo")
    print("=" * 70)

    # Create spinner with token from environment or use demo
    slack_token = os.getenv("SLACK_BOT_TOKEN", "xoxb-demo-token")

    spinner = SlackSpinner(
        slack_token=slack_token,
        include_threads=True,
        days_back=7,
        max_messages_per_channel=50
    )

    # Check availability
    if not spinner.is_available():
        print("Error: slack-sdk or requests library not available")
        print("Install with: pip install slack-sdk")
        print("\nNote: This demo requires SLACK_BOT_TOKEN environment variable")
        return

    print("\nSpinner capabilities:")
    caps = spinner.get_capabilities()
    print(f"  - Basic processing: {caps.basic_processing}")
    print(f"  - Streaming: {caps.streaming}")
    print(f"  - Importance scoring: {caps.importance_scoring}")
    print(f"  - Entity extraction: {caps.entity_extraction}")

    # Note about API token
    print("\nConfiguration:")
    print(f"  - IMAP Server configured: {spinner.client.token != 'xoxb-demo-token'}")
    print(f"  - Days back: {spinner.days_back}")
    print(f"  - Include threads: {spinner.include_threads}")

    print("\nNote: To use with real Slack workspace:")
    print("  1. Create a Slack app (api.slack.com)")
    print("  2. Grant 'conversations:history' and 'users:read' scopes")
    print("  3. Set SLACK_BOT_TOKEN environment variable")
    print("  4. Call: await spinner.spin('C1234567890')  # Channel ID")

    # Example of expected output
    print("\nExpected output structure (when using real token):")
    print("  - Each message becomes a MemoryShard")
    print("  - Includes user info, reactions, timestamps")
    print("  - Thread replies linked to parent message")
    print("  - Importance scored by engagement metrics")


async def demo_importance_scoring():
    """Demonstrate message importance scoring."""
    print("\n" + "=" * 70)
    print("Message Importance Scoring Demo")
    print("=" * 70)

    from HoloLoom.spinningWheel.slack_spinner import SlackMessage

    spinner = SlackSpinner("test-token")

    # Example messages
    messages = [
        SlackMessage(
            ts="1609459200.000100",
            user="U12345",
            username="engineer",
            text="System is down! Please help urgent",
            reactions={"alarm": 5, "fire": 3},
            reply_count=10
        ),
        SlackMessage(
            ts="1609459200.000200",
            user="U12346",
            username="user",
            text="Good morning!",
            reactions={},
            reply_count=0
        ),
        SlackMessage(
            ts="1609459200.000300",
            user="U12347",
            username="tech-lead",
            text="Architecture update:\n\n```\nComponent A -> B -> C\n```\n\nSee attached docs",
            reactions={"thumbsup": 8},
            reply_count=5,
            files_count=1
        ),
    ]

    print("\nScoring example messages:\n")

    for i, msg in enumerate(messages):
        score = spinner._score_message_importance(msg)
        print(f"Message {i+1}: @{msg.username}")
        print(f"  Text: {msg.text[:50]}...")
        print(f"  Reactions: {msg.reactions}")
        print(f"  Replies: {msg.reply_count}")
        print(f"  Importance Score: {score.score:.2f}")
        print(f"  Reason: {score.reason}")
        print()


async def main():
    """Run all demos."""
    await demo_slack_spinner()
    await demo_importance_scoring()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

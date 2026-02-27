"""
Email Spinner Usage Examples
=============================

Demonstrates how to use EmailSpinner to ingest email archives
into HoloLoom memory.

Requirements:
    # No external dependencies - uses stdlib imaplib and mailbox

Optional (for better HTML parsing):
    pip install beautifulsoup4

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
from typing import List

from hololoom.spinningWheel.email_spinner import (
    EmailSpinner,
    spin_email_archive,
    create_email_scorer
)
from hololoom.spinningWheel.protocol import SpinResult
from hololoom.documentation.types import MemoryShard


# =============================================================================
# Example 1: Basic mbox Archive Ingestion
# =============================================================================

async def example_1_basic_mbox():
    """
    Basic example: Spin an mbox file into MemoryShards

    Use case: Import Thunderbird/Apple Mail archive for search
    """
    print("=" * 60)
    print("Example 1: Basic mbox Archive Ingestion")
    print("=" * 60)

    # Initialize spinner
    spinner = EmailSpinner(
        importance_threshold=0.3  # Filter out low-importance emails
    )

    try:
        # Spin an mbox file
        mbox_path = Path("./data/work_emails.mbox")
        result: SpinResult = await spinner.spin(mbox_path)

        print(f"\nProcessed: {result.items_processed} emails")
        print(f"Filtered: {result.items_filtered} low-importance emails")
        print(f"Shards created: {len(result.shards)}")

        # Inspect first shard
        if result.shards:
            shard = result.shards[0]
            print(f"\nFirst email preview:")
            print(f"  ID: {shard.id}")
            print(f"  Episode: {shard.episode}")
            print(f"  Subject: {shard.metadata.get('subject', 'N/A')[:50]}...")
            print(f"  From: {shard.metadata.get('sender', 'N/A')}")
            print(f"  Text: {shard.text[:100]}...")
            print(f"  Entities: {shard.entities}")
            print(f"  Motifs: {shard.motifs}")

    except FileNotFoundError:
        print(f"\nNote: Export emails to mbox format to run this example")
        print("  Thunderbird: Tools → ImportExportTools NG → Export folder")
        print("  Apple Mail: Mailbox → Export Mailbox")


# =============================================================================
# Example 2: IMAP Server Ingestion
# =============================================================================

async def example_2_imap():
    """
    Connect to IMAP server and ingest mailbox

    Use case: Live sync from Gmail, Outlook, etc.
    """
    print("=" * 60)
    print("Example 2: IMAP Server Ingestion")
    print("=" * 60)

    # Initialize spinner with IMAP credentials
    spinner = EmailSpinner(
        imap_server="imap.gmail.com",
        username="your_email@gmail.com",
        password="your_app_password",  # Use app password, not regular password
        importance_threshold=0.3
    )

    try:
        # Spin INBOX mailbox
        result = await spinner.spin_imap_mailbox(mailbox_name="INBOX", limit=100)

        print(f"\nProcessed: {result.items_processed} emails")
        print(f"Shards created: {len(result.shards)}")

        # Show thread roots (important emails)
        thread_roots = [
            s for s in result.shards
            if s.metadata.get('is_thread_root', False)
        ]

        print(f"\nThread roots found: {len(thread_roots)}")
        for shard in thread_roots[:3]:
            print(f"  - {shard.metadata.get('subject', 'No subject')[:60]}")

    except Exception as e:
        print(f"\nNote: Configure IMAP credentials to run this example")
        print("  For Gmail: Enable 2FA, then create App Password")
        print("  Settings → Security → App passwords")
        print(f"  Error: {e}")


# =============================================================================
# Example 3: Importance Filtering
# =============================================================================

async def example_3_importance_filtering():
    """
    Demonstrate importance filtering at different thresholds

    Use case: Focus on high-value communications
    """
    print("=" * 60)
    print("Example 3: Importance Filtering")
    print("=" * 60)

    mbox_path = Path("./data/project_emails.mbox")
    thresholds = [0.2, 0.5, 0.8]

    for threshold in thresholds:
        spinner = EmailSpinner(
            importance_threshold=threshold,
            max_emails=1000
        )

        try:
            result = await spinner.spin(mbox_path)

            print(f"\nThreshold {threshold}:")
            print(f"  Emails processed: {result.items_processed}")
            print(f"  Shards created: {len(result.shards)}")
            print(f"  Filter rate: {result.items_filtered / result.items_processed * 100:.1f}%")

            # Show importance distribution
            if result.shards:
                scores = [s.metadata.get('importance_score', 0) for s in result.shards]
                avg_score = sum(scores) / len(scores)
                print(f"  Average importance: {avg_score:.3f}")

                # Count thread roots
                roots = sum(1 for s in result.shards if s.metadata.get('is_thread_root'))
                print(f"  Thread roots: {roots}")

        except FileNotFoundError:
            print(f"\nNote: Create {mbox_path} to run this example")
            break


# =============================================================================
# Example 4: Incremental IMAP Sync
# =============================================================================

async def example_4_incremental_sync():
    """
    Incremental sync using UID checkpointing

    Use case: Regularly update memory with new emails
    """
    print("=" * 60)
    print("Example 4: Incremental IMAP Sync")
    print("=" * 60)

    spinner = EmailSpinner(
        imap_server="imap.gmail.com",
        username="your_email@gmail.com",
        password="your_app_password"
    )

    try:
        # First sync: get all emails
        print("\nFirst sync (full mailbox)...")
        result1 = await spinner.spin_incremental(mailbox_name="INBOX")
        print(f"  Shards created: {len(result1.shards)}")

        # Simulate some time passing / new emails arriving
        print("\nSimulating new activity...")
        await asyncio.sleep(1)

        # Second sync: only new emails
        print("\nSecond sync (incremental)...")
        result2 = await spinner.spin_incremental(mailbox_name="INBOX")
        print(f"  New shards: {len(result2.shards)}")

        # Checkpoint is automatically saved/loaded
        print("\nCheckpoint saved for next run!")

    except Exception as e:
        print(f"\nNote: Configure IMAP credentials to run this example")
        print(f"  Error: {e}")


# =============================================================================
# Example 5: Streaming Large Mailboxes
# =============================================================================

async def example_5_streaming():
    """
    Stream MemoryShards for memory-efficient processing

    Use case: Large mailboxes that don't fit in memory
    """
    print("=" * 60)
    print("Example 5: Streaming Large Mailboxes")
    print("=" * 60)

    spinner = EmailSpinner(
        importance_threshold=0.3,
        max_emails=10000  # Large mailbox
    )

    mbox_path = Path("./data/archive.mbox")

    try:
        shard_count = 0

        # Process shards one at a time
        async for shard in spinner.spin_stream(mbox_path, batch_size=100):
            shard_count += 1

            # Process shard (e.g., add to memory, index, etc.)
            # await memory.add_shard(shard)

            # Progress indicator
            if shard_count % 100 == 0:
                print(f"  Processed {shard_count} emails...")

        print(f"\nTotal emails streamed: {shard_count}")

    except FileNotFoundError:
        print(f"\nNote: Create {mbox_path} to run this example")


# =============================================================================
# Example 6: Custom Importance Scoring
# =============================================================================

async def example_6_custom_scoring():
    """
    Customize importance scoring for domain-specific needs

    Use case: Prioritize specific senders, keywords, or patterns
    """
    print("=" * 60)
    print("Example 6: Custom Importance Scoring")
    print("=" * 60)

    # Create custom scorer for work emails
    scorer = create_email_scorer()
    scorer.add_technical_terms({
        'urgent', 'asap', 'deadline', 'critical', 'important',
        'action required', 'review needed', 'approval', 'milestone',
        'release', 'deployment', 'incident', 'outage'
    })

    spinner = EmailSpinner(
        importance_threshold=0.3
    )

    # Override scorer
    spinner.importance_scorer = scorer

    mbox_path = Path("./data/work_inbox.mbox")

    try:
        result = await spinner.spin(mbox_path)

        print(f"\nProcessed: {result.items_processed} emails")
        print(f"High-priority emails: {len(result.shards)}")

        # Show urgent/action-required emails
        urgent_shards = [
            s for s in result.shards
            if any(term in s.text.lower() + s.metadata.get('subject', '').lower()
                   for term in ['urgent', 'asap', 'action required', 'critical'])
        ]

        print(f"\nUrgent emails: {len(urgent_shards)}")
        for shard in urgent_shards[:5]:  # Show first 5
            print(f"  - {shard.metadata.get('subject', 'No subject')[:60]}")
            print(f"    From: {shard.metadata.get('sender', 'Unknown')}")

    except FileNotFoundError:
        print(f"\nNote: Create {mbox_path} to run this example")


# =============================================================================
# Example 7: Thread Detection and Grouping
# =============================================================================

async def example_7_thread_detection():
    """
    Detect and group email threads

    Use case: Understand conversation context and relationships
    """
    print("=" * 60)
    print("Example 7: Thread Detection and Grouping")
    print("=" * 60)

    spinner = EmailSpinner(
        importance_threshold=0.2  # Lower threshold to capture threads
    )

    mbox_path = Path("./data/discussions.mbox")

    try:
        result = await spinner.spin(mbox_path)

        print(f"\nProcessed: {result.items_processed} emails")
        print(f"Total shards: {len(result.shards)}")

        # Group by thread
        threads = {}
        for shard in result.shards:
            thread_id = shard.metadata.get('thread_id', 'unknown')
            if thread_id not in threads:
                threads[thread_id] = []
            threads[thread_id].append(shard)

        print(f"\nThreads detected: {len(threads)}")

        # Show top threads by size
        top_threads = sorted(threads.items(), key=lambda x: len(x[1]), reverse=True)[:5]

        print("\nTop 5 threads by message count:")
        for thread_id, messages in top_threads:
            # Get thread root
            roots = [m for m in messages if m.metadata.get('is_thread_root', False)]
            root_subject = roots[0].metadata.get('subject', 'No subject') if roots else 'Unknown'

            print(f"  - {root_subject[:60]} ({len(messages)} messages)")

            # Show participants
            participants = set()
            for msg in messages:
                participants.add(msg.metadata.get('sender', 'Unknown'))
            print(f"    Participants: {', '.join(list(participants)[:3])}")

    except FileNotFoundError:
        print(f"\nNote: Create {mbox_path} to run this example")


# =============================================================================
# Main Runner
# =============================================================================

async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Email Spinner Examples")
    print("=" * 60 + "\n")

    # NOTE: Update these with your email setup
    print("SETUP REQUIRED:")
    print("1. Export emails to mbox format OR configure IMAP")
    print("2. Optional: pip install beautifulsoup4 (better HTML parsing)")
    print("3. For IMAP: Use app passwords (not regular passwords)")
    print("4. Create ./data/ directory for mbox files\n")

    examples = [
        ("Basic mbox Archive Ingestion", example_1_basic_mbox),
        ("IMAP Server Ingestion", example_2_imap),
        ("Importance Filtering", example_3_importance_filtering),
        ("Incremental IMAP Sync", example_4_incremental_sync),
        ("Streaming Large Mailboxes", example_5_streaming),
        ("Custom Importance Scoring", example_6_custom_scoring),
        ("Thread Detection and Grouping", example_7_thread_detection),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nUncomment examples in main() to run them individually.")

    # Uncomment to run examples:
    # await example_1_basic_mbox()
    # await example_2_imap()
    # await example_3_importance_filtering()
    # await example_4_incremental_sync()
    # await example_5_streaming()
    # await example_6_custom_scoring()
    # await example_7_thread_detection()


if __name__ == "__main__":
    asyncio.run(main())

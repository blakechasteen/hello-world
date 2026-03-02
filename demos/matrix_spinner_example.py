"""
Matrix Spinner Usage Examples
==============================

Demonstrates how to use MatrixSpinner to ingest Matrix.org chat rooms
into HoloLoom memory.

Requirements:
    pip install matrix-nio

Optional (for E2EE):
    pip install "matrix-nio[e2e]"

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
from typing import List

from hololoom.spinningWheel.matrix_spinner import (
    MatrixSpinner,
    spin_matrix_room,
    create_matrix_scorer
)
from hololoom.spinningWheel.protocol import SpinResult
from hololoom.documentation.types import MemoryShard


# =============================================================================
# Example 1: Basic Room Ingestion
# =============================================================================

async def example_1_basic_room():
    """
    Basic example: Spin a single Matrix room into MemoryShards

    Use case: Import conversation history for query/search
    """
    print("=" * 60)
    print("Example 1: Basic Room Ingestion")
    print("=" * 60)

    # Initialize spinner
    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="YOUR_ACCESS_TOKEN",  # Get from Matrix client
        importance_threshold=0.3  # Filter out low-importance messages
    )

    try:
        # Spin a single room
        result: SpinResult = await spinner.spin_room("!room_id:matrix.org")

        print(f"\nProcessed: {result.items_processed} messages")
        print(f"Filtered: {result.items_filtered} low-importance messages")
        print(f"Shards created: {len(result.shards)}")

        # Inspect first shard
        if result.shards:
            shard = result.shards[0]
            print(f"\nFirst shard preview:")
            print(f"  ID: {shard.id}")
            print(f"  Episode: {shard.episode}")
            print(f"  Text: {shard.text[:100]}...")
            print(f"  Entities: {shard.entities}")
            print(f"  Motifs: {shard.motifs}")
            print(f"  Importance: {shard.metadata.get('importance_score', 'N/A')}")

    finally:
        await spinner.close()


# =============================================================================
# Example 2: Importance Filtering
# =============================================================================

async def example_2_importance_filtering():
    """
    Demonstrate importance filtering at different thresholds

    Use case: Quality vs. quantity tradeoff
    """
    print("=" * 60)
    print("Example 2: Importance Filtering")
    print("=" * 60)

    room_id = "!room_id:matrix.org"
    thresholds = [0.2, 0.5, 0.8]

    for threshold in thresholds:
        spinner = MatrixSpinner(
            homeserver="https://matrix.org",
            access_token="YOUR_ACCESS_TOKEN",
            importance_threshold=threshold,
            max_messages_per_room=100  # Limit for demo
        )

        try:
            result = await spinner.spin_room(room_id)

            print(f"\nThreshold {threshold}:")
            print(f"  Messages processed: {result.items_processed}")
            print(f"  Shards created: {len(result.shards)}")
            print(f"  Filter rate: {result.items_filtered / result.items_processed * 100:.1f}%")

            # Show importance distribution
            if result.shards:
                scores = [s.metadata.get('importance_score', 0) for s in result.shards]
                avg_score = sum(scores) / len(scores)
                print(f"  Average importance: {avg_score:.3f}")

        finally:
            await spinner.close()


# =============================================================================
# Example 3: Incremental Sync
# =============================================================================

async def example_3_incremental_sync():
    """
    Incremental sync using checkpoint tokens

    Use case: Regularly update memory with new messages
    """
    print("=" * 60)
    print("Example 3: Incremental Sync")
    print("=" * 60)

    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="YOUR_ACCESS_TOKEN"
    )

    try:
        # First sync: get all messages
        print("\nFirst sync (full history)...")
        result1 = await spinner.spin_incremental(room_id="!room_id:matrix.org")
        print(f"  Shards created: {len(result1.shards)}")

        # Simulate some time passing / new messages arriving
        print("\nSimulating new activity...")
        await asyncio.sleep(1)

        # Second sync: only new messages
        print("\nSecond sync (incremental)...")
        result2 = await spinner.spin_incremental(room_id="!room_id:matrix.org")
        print(f"  New shards: {len(result2.shards)}")

        # Checkpoint is automatically saved/loaded
        print("\nCheckpoint saved for next run!")

    finally:
        await spinner.close()


# =============================================================================
# Example 4: Streaming Large Rooms
# =============================================================================

async def example_4_streaming():
    """
    Stream MemoryShards for memory-efficient processing

    Use case: Large rooms that don't fit in memory
    """
    print("=" * 60)
    print("Example 4: Streaming Large Rooms")
    print("=" * 60)

    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="YOUR_ACCESS_TOKEN",
        max_messages_per_room=10000  # Large room
    )

    try:
        shard_count = 0

        # Process shards one at a time
        async for shard in spinner.spin_stream("!room_id:matrix.org", batch_size=50):
            shard_count += 1

            # Process shard (e.g., add to memory, index, etc.)
            # await memory.add_shard(shard)

            # Progress indicator
            if shard_count % 50 == 0:
                print(f"  Processed {shard_count} shards...")

        print(f"\nTotal shards streamed: {shard_count}")

    finally:
        await spinner.close()


# =============================================================================
# Example 5: All Rooms
# =============================================================================

async def example_5_all_rooms():
    """
    Spin all joined rooms with filtering

    Use case: Comprehensive memory from all conversations
    """
    print("=" * 60)
    print("Example 5: All Rooms")
    print("=" * 60)

    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="YOUR_ACCESS_TOKEN",
        importance_threshold=0.5,  # Higher threshold for multi-room
        max_messages_per_room=100  # Limit per room
    )

    try:
        # Filter function: only public rooms or specific names
        def room_filter(room_id: str, room_name: str) -> bool:
            # Example: skip DMs, only keep rooms with "dev" or "engineering"
            return "dev" in room_name.lower() or "engineering" in room_name.lower()

        # Spin all filtered rooms
        result = await spinner.spin_all_rooms(room_filter=room_filter)

        print(f"\nProcessed messages: {result.items_processed}")
        print(f"Total shards: {len(result.shards)}")

        # Group shards by room
        rooms = {}
        for shard in result.shards:
            room_name = shard.metadata.get('room_name', 'Unknown')
            rooms[room_name] = rooms.get(room_name, 0) + 1

        print("\nShards per room:")
        for room_name, count in sorted(rooms.items(), key=lambda x: x[1], reverse=True):
            print(f"  {room_name}: {count}")

    finally:
        await spinner.close()


# =============================================================================
# Example 6: Custom Importance Scoring
# =============================================================================

async def example_6_custom_scoring():
    """
    Customize importance scoring for domain-specific needs

    Use case: Prioritize specific message types or keywords
    """
    print("=" * 60)
    print("Example 6: Custom Importance Scoring")
    print("=" * 60)

    # Create custom scorer for security/incident response
    scorer = create_matrix_scorer()
    scorer.add_technical_terms({
        'security', 'vulnerability', 'patch', 'exploit', 'CVE',
        'incident', 'breach', 'alert', 'critical'
    })

    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        access_token="YOUR_ACCESS_TOKEN",
        importance_threshold=0.3
    )

    # Override scorer
    spinner.importance_scorer = scorer

    try:
        result = await spinner.spin_room("!security-team:matrix.org")

        print(f"\nProcessed: {result.items_processed} messages")
        print(f"High-priority shards: {len(result.shards)}")

        # Show security-related messages
        security_shards = [
            s for s in result.shards
            if any(term in s.text.lower() for term in ['security', 'vulnerability', 'cve'])
        ]

        print(f"\nSecurity-related messages: {len(security_shards)}")
        for shard in security_shards[:3]:  # Show first 3
            print(f"  - {shard.text[:80]}...")

    finally:
        await spinner.close()


# =============================================================================
# Example 7: Login with Password
# =============================================================================

async def example_7_login():
    """
    Login with username/password instead of access token

    Use case: Interactive setup without manual token retrieval
    """
    print("=" * 60)
    print("Example 7: Login with Password")
    print("=" * 60)

    spinner = MatrixSpinner(
        homeserver="https://matrix.org",
        user_id="@your_username:matrix.org",
        password="YOUR_PASSWORD",  # Not recommended for production
        importance_threshold=0.3
    )

    try:
        # Spinner will auto-login on first use
        result = await spinner.spin_room("!room_id:matrix.org")

        print(f"\nLogged in and processed {result.items_processed} messages")
        print(f"Created {len(result.shards)} shards")

        # Access token is available after login
        print(f"\nAccess token (save for next time):")
        print(f"  {spinner.access_token[:20]}...")

    finally:
        await spinner.close()


# =============================================================================
# Main Runner
# =============================================================================

async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Matrix Spinner Examples")
    print("=" * 60 + "\n")

    # NOTE: Update these with your Matrix credentials
    # You can get an access token from Element's settings:
    # Settings → Help & About → Advanced → Access Token

    print("SETUP REQUIRED:")
    print("1. Install matrix-nio: pip install matrix-nio")
    print("2. Update examples with your homeserver and access token")
    print("3. Update room IDs with your actual rooms\n")

    examples = [
        ("Basic Room Ingestion", example_1_basic_room),
        ("Importance Filtering", example_2_importance_filtering),
        ("Incremental Sync", example_3_incremental_sync),
        ("Streaming Large Rooms", example_4_streaming),
        ("All Rooms", example_5_all_rooms),
        ("Custom Importance Scoring", example_6_custom_scoring),
        ("Login with Password", example_7_login),
    ]

    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nUncomment examples in main() to run them individually.")

    # Uncomment to run examples:
    # await example_1_basic_room()
    # await example_2_importance_filtering()
    # await example_3_incremental_sync()
    # await example_4_streaming()
    # await example_5_all_rooms()
    # await example_6_custom_scoring()
    # await example_7_login()


if __name__ == "__main__":
    asyncio.run(main())

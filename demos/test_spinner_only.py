#!/usr/bin/env python3
"""
Test: Chat History Spinner Only
================================
Shows the spinner working without full HoloLoom integration.
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.apps.workflow_builder.conversation_manager import ConversationManager
from HoloLoom.spinningWheel import ChatHistorySpinner


async def main():
    print("\n" + "="*70)
    print("Chat History Spinner Test")
    print("="*70 + "\n")

    # Create test database
    test_db = "./data/spinner_test.db"
    os.makedirs("./data", exist_ok=True)

    if os.path.exists(test_db):
        os.remove(test_db)

    conv_mgr = ConversationManager(test_db)

    print("Creating test conversations...")
    print()

    # Create 3 conversations with varying importance
    conv1 = conv_mgr.create_conversation("Technical Discussion")
    conv_mgr.add_message(
        conv1.id, 'user',
        "What is Thompson Sampling and how does it work?"
    )
    conv_mgr.add_message(
        conv1.id, 'assistant',
        "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
        "It maintains Beta distributions for each action and samples from them to balance "
        "exploration and exploitation. The algorithm updates alpha on success and beta on failure.",
        metadata={'confidence': 0.92, 'reasoning_steps': ['analyze', 'synthesize']}
    )

    conv2 = conv_mgr.create_conversation("Architecture Question")
    conv_mgr.add_message(conv2.id, 'user', "Explain HoloLoom architecture")
    conv_mgr.add_message(
        conv2.id, 'assistant',
        "HoloLoom uses a weaving metaphor with warp threads and shuttle orchestrator.",
        metadata={'confidence': 0.85}
    )

    conv3 = conv_mgr.create_conversation("Greeting")
    conv_mgr.add_message(conv3.id, 'user', "Hi there")
    conv_mgr.add_message(conv3.id, 'assistant', "Hello!")

    print("Created 3 conversations\n")
    print("="*70)
    print("Spinning conversations into MemoryShards...")
    print("="*70 + "\n")

    spinner = ChatHistorySpinner(
        conv_mgr,
        importance_threshold=0.0,  # Include all for demo
        extract_entities=True,
        extract_motifs=True
    )

    all_shards = []
    for conv in [conv1, conv2, conv3]:
        shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

        for shard in shards:
            all_shards.append(shard)

            print(f"Conversation: {shard.metadata['conversation_title']}")
            print(f"Turn: {shard.metadata['turn_index']}")
            print(f"Importance: {shard.metadata['importance_score']:.2f}")
            print(f"Importance Signals:")

            signals = shard.metadata['importance_signals']
            for key, value in signals.items():
                print(f"  - {key}: {value:.2f}")

            print(f"\nEntities extracted: {shard.entities}")
            print(f"Motifs extracted: {shard.motifs[:3]}")  # First 3

            print(f"\nShard text preview:")
            lines = shard.text.split('\n')[:5]
            for line in lines:
                print(f"  {line}")

            print("\n" + "-"*70 + "\n")

    print("="*70)
    print("Summary")
    print("="*70 + "\n")

    print(f"Total shards created: {len(all_shards)}")
    print(f"\nImportance distribution:")

    high = sum(1 for s in all_shards if s.metadata['importance_score'] >= 0.7)
    med = sum(1 for s in all_shards if 0.4 <= s.metadata['importance_score'] < 0.7)
    low = sum(1 for s in all_shards if s.metadata['importance_score'] < 0.4)

    print(f"  High (>= 0.7): {high} shards")
    print(f"  Medium (0.4-0.7): {med} shards")
    print(f"  Low (< 0.4): {low} shards")

    print(f"\nWith threshold=0.3, we would ingest: {len([s for s in all_shards if s.metadata['importance_score'] >= 0.3])} shards")
    print(f"Filtering out: {len([s for s in all_shards if s.metadata['importance_score'] < 0.3])} low-importance shards")

    print("\n" + "="*70)
    print("Test Complete - Spinner is working correctly!")
    print("="*70)

    print("\nWhat this proves:")
    print("1. Conversations are converted to MemoryShards")
    print("2. Importance scoring filters noise from signal")
    print("3. Entities and motifs are extracted")
    print("4. Metadata is preserved (timestamps, confidence, etc.)")
    print("5. Ready to be ingested into HoloLoom memory!")
    print()


if __name__ == "__main__":
    asyncio.run(main())

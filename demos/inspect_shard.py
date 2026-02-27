#!/usr/bin/env python3
"""
Inspect a MemoryShard
=====================
Shows the complete structure of a chat history shard.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.apps.workflow_builder.conversation_manager import ConversationManager
from hololoom.spinningWheel import ChatHistorySpinner


async def main():
    # Create test conversation
    test_db = "./data/inspect_test.db"
    os.makedirs("./data", exist_ok=True)

    if os.path.exists(test_db):
        os.remove(test_db)

    conv_mgr = ConversationManager(test_db)

    conv = conv_mgr.create_conversation("Thompson Sampling Deep Dive")
    conv_mgr.update_tags(conv.id, ['machine-learning', 'bandits', 'exploration'])

    conv_mgr.add_message(
        conv.id, 'user',
        "Explain how Thompson Sampling updates its parameters in detail"
    )
    conv_mgr.add_message(
        conv.id, 'assistant',
        "Thompson Sampling maintains Beta(alpha, beta) distributions for each action. "
        "When an action succeeds (reward >= threshold), alpha is increased. "
        "When it fails, beta is increased. This creates probability matching where "
        "actions with higher uncertainty are explored more often.",
        metadata={
            'confidence': 0.94,
            'tool': 'answer',
            'reasoning_steps': ['retrieve', 'analyze', 'synthesize']
        }
    )

    # Spin it
    spinner = ChatHistorySpinner(conv_mgr)
    shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

    shard = shards[0]

    print("\n" + "="*70)
    print("COMPLETE MemoryShard STRUCTURE")
    print("="*70 + "\n")

    print("ID:", shard.id)
    print("Episode:", shard.episode)
    print("\nEntities:", shard.entities)
    print("\nMotifs:", shard.motifs)

    print("\n" + "-"*70)
    print("FULL TEXT CONTENT:")
    print("-"*70)
    print(shard.text)

    print("\n" + "-"*70)
    print("METADATA (Complete):")
    print("-"*70)
    print(json.dumps(shard.metadata, indent=2, default=str))

    print("\n" + "="*70)
    print("This shard is ready to be ingested into HoloLoom memory!")
    print("="*70)
    print("\nNext steps:")
    print("  async with HoloLoom() as loom:")
    print("      await loom.experience(shard.text)")
    print("      results = await loom.recall('Thompson Sampling parameters')")
    print()


if __name__ == "__main__":
    asyncio.run(main())

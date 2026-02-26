#!/usr/bin/env python3
"""
Simple Chat History Demo
========================
Quick demonstration of chat history ingestion.
"""

import asyncio
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.apps.workflow_builder.conversation_manager import ConversationManager
from HoloLoom.spinningWheel import ChatHistorySpinner, ingest_chat_history
from HoloLoom import HoloLoom


async def main():
    print("\n" + "="*70)
    print("Chat History Integration Demo")
    print("="*70 + "\n")

    # Create test database
    test_db = "./data/demo_conversations.db"
    os.makedirs("./data", exist_ok=True)

    # Clean up old test db
    if os.path.exists(test_db):
        os.remove(test_db)

    conv_mgr = ConversationManager(test_db)

    print("Step 1: Creating sample conversations...\n")

    # Conversation 1: Thompson Sampling
    conv1 = conv_mgr.create_conversation("Learning about Thompson Sampling")
    conv_mgr.add_message(conv1.id, 'user', "What is Thompson Sampling?")
    conv_mgr.add_message(conv1.id, 'assistant',
        "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
        "It balances exploration and exploitation by sampling from posterior distributions.",
        metadata={'confidence': 0.92, 'tool': 'answer'}
    )

    conv_mgr.add_message(conv1.id, 'user', "How does it compare to epsilon-greedy?")
    conv_mgr.add_message(conv1.id, 'assistant',
        "Thompson Sampling is more sophisticated than epsilon-greedy because it uses "
        "probability matching - the exploration rate adapts based on uncertainty.",
        metadata={'confidence': 0.88}
    )

    print(f"  - Created conversation: {conv1.title}")
    print(f"    Messages: 4 (2 turns)")

    # Conversation 2: HoloLoom Architecture
    conv2 = conv_mgr.create_conversation("HoloLoom Architecture")
    conv_mgr.add_message(conv2.id, 'user', "Explain the weaving metaphor in HoloLoom")
    conv_mgr.add_message(conv2.id, 'assistant',
        "HoloLoom uses a weaving metaphor. The 'warp threads' are independent modules. "
        "The 'shuttle' orchestrator weaves these threads together to produce the fabric.",
        metadata={'confidence': 0.95}
    )

    print(f"  - Created conversation: {conv2.title}")
    print(f"    Messages: 2 (1 turn)")

    # Conversation 3: Short greeting (low importance)
    conv3 = conv_mgr.create_conversation("Greeting")
    conv_mgr.add_message(conv3.id, 'user', "Hi")
    conv_mgr.add_message(conv3.id, 'assistant', "Hello! How can I help?")

    print(f"  - Created conversation: {conv3.title}")
    print(f"    Messages: 2 (1 turn - low importance)")

    print(f"\nTotal: 3 conversations, 8 messages\n")

    print("Step 2: Spinning conversations into MemoryShards...\n")

    spinner = ChatHistorySpinner(conv_mgr, importance_threshold=0.3)

    # Show importance scores
    for conv in [conv1, conv2, conv3]:
        shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

        if shards:
            shard = shards[0]
            score = shard.metadata['importance_score']
            signals = shard.metadata['importance_signals']

            print(f"  Conversation: {conv.title}")
            print(f"    Importance: {score:.2f}")
            print(f"    Entities: {shard.entities}")
            print(f"    Motifs: {shard.motifs[:2] if len(shard.motifs) > 2 else shard.motifs}")
            print(f"    Signals: length={signals.get('length', 0):.2f}, "
                  f"question={signals.get('question', 0):.2f}, "
                  f"technical={signals.get('technical', 0):.2f}")
            print()

    print("Step 3: Ingesting into HoloLoom memory...\n")

    async with HoloLoom() as loom:
        # Ingest all conversations
        shard_count = await ingest_chat_history(
            conv_mgr,
            loom,
            importance_threshold=0.3
        )

        print(f"  -> Ingested {shard_count} shards (filtered by importance >= 0.3)\n")

        print("Step 4: Querying the ingested chat history...\n")

        # Query 1: Thompson Sampling
        print("  Query 1: 'What did we discuss about Thompson Sampling?'")
        results = await loom.recall("Thompson Sampling exploration exploitation")

        print(f"  -> Found {len(results)} relevant memories")
        if results:
            snippet = results[0].text[:200].replace('\n', ' ')
            print(f"  -> Top result: {snippet}...")
        print()

        # Query 2: Weaving metaphor
        print("  Query 2: 'Tell me about the weaving metaphor'")
        results = await loom.recall("weaving metaphor shuttle warp")

        print(f"  -> Found {len(results)} relevant memories")
        if results:
            snippet = results[0].text[:200].replace('\n', ' ')
            print(f"  -> Top result: {snippet}...")
        print()

        # Query 3: General question
        print("  Query 3: 'How does HoloLoom work?'")
        results = await loom.recall("HoloLoom orchestrator modules")

        print(f"  -> Found {len(results)} relevant memories")
        if results:
            snippet = results[0].text[:200].replace('\n', ' ')
            print(f"  -> Top result: {snippet}...")
        print()

        # Show memory metrics
        print("Step 5: Memory system metrics...\n")
        metrics = loom.get_metrics()

        print(f"  Active memories: {metrics['activation']['active_nodes']}")
        print(f"  Total nodes: {metrics['activation']['total_nodes']}")
        print(f"  Coherence: {metrics['coherence']['mean']:.3f}")
        print()

    print("="*70)
    print("Demo Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Chat history automatically converted to MemoryShards")
    print("2. Importance scoring filtered low-value exchanges")
    print("3. All conversations now searchable via loom.recall()")
    print("4. Metadata preserved (timestamps, confidence, etc.)")
    print()


if __name__ == "__main__":
    asyncio.run(main())

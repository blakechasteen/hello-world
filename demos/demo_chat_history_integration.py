#!/usr/bin/env python3
"""
Demo: Chat History Integration
================================

Demonstrates how to make conversation history queryable in HoloLoom.

This demo shows:
1. Creating conversations in ConversationManager
2. Spinning chat history into MemoryShards
3. Querying conversations via HoloLoom's unified API
4. Auto-capture of new conversations

Usage:
    PYTHONPATH=. python demos/demo_chat_history_integration.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.apps.workflow_builder.conversation_manager import ConversationManager
from HoloLoom.spinningWheel.chat_history import (
    ChatHistorySpinner,
    ChatHistoryAutoCapture,
    ingest_chat_history
)
from HoloLoom import HoloLoom


async def demo_basic_ingestion():
    """Demo 1: Basic chat history ingestion"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Chat History Ingestion")
    print("="*70 + "\n")

    # Create conversation manager with test database
    test_db = "./data/test_conversations.db"
    os.makedirs("./data", exist_ok=True)
    conv_mgr = ConversationManager(test_db)

    # Create some sample conversations
    print("Creating sample conversations...")
    conv1 = conv_mgr.create_conversation("Learning about Thompson Sampling")
    conv_mgr.add_message(conv1.id, 'user', "What is Thompson Sampling?")
    conv_mgr.add_message(conv1.id, 'assistant',
        "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
        "It balances exploration and exploitation by sampling from posterior distributions. "
        "Each action has a Beta(α, β) distribution representing success/failure counts.",
        metadata={'confidence': 0.92, 'tool': 'answer'}
    )

    conv_mgr.add_message(conv1.id, 'user', "How does it compare to epsilon-greedy?")
    conv_mgr.add_message(conv1.id, 'assistant',
        "Thompson Sampling is more sophisticated than epsilon-greedy. "
        "While epsilon-greedy explores randomly with fixed probability ε, "
        "Thompson Sampling uses probability matching - exploration rate adapts "
        "based on uncertainty about each action's value.",
        metadata={'confidence': 0.88, 'tool': 'answer'}
    )

    conv2 = conv_mgr.create_conversation("HoloLoom Architecture")
    conv_mgr.add_message(conv2.id, 'user', "Explain the weaving metaphor in HoloLoom")
    conv_mgr.add_message(conv2.id, 'assistant',
        "HoloLoom uses a weaving metaphor throughout its architecture. "
        "The 'warp threads' are independent modules (motif detection, embedding, memory, policy). "
        "The 'shuttle' orchestrator weaves these threads together to produce the final fabric (response). "
        "This design enables modularity and protocol-based composition.",
        metadata={'confidence': 0.95, 'tool': 'answer'}
    )

    print(f"✓ Created {len([conv1, conv2])} conversations\n")

    # Create HoloLoom instance
    print("Initializing HoloLoom...")
    async with HoloLoom() as loom:
        # Ingest chat history
        print("Ingesting chat history into memory...")
        shard_count = await ingest_chat_history(
            conv_mgr,
            loom,
            importance_threshold=0.3
        )

        print(f"✓ Ingested {shard_count} conversation shards\n")

        # Query the chat history
        print("Querying conversation history...\n")

        queries = [
            "What did we discuss about Thompson Sampling?",
            "How does Thompson Sampling compare to other methods?",
            "Tell me about the weaving metaphor"
        ]

        for query in queries:
            print(f"Query: {query}")
            results = await loom.recall(query)

            print(f"Found {len(results)} relevant memories:")
            for i, result in enumerate(results[:2], 1):
                snippet = result.text[:150].replace('\n', ' ')
                print(f"  {i}. {snippet}...")

            print()

    print("✓ Demo 1 complete!\n")


async def demo_auto_capture():
    """Demo 2: Auto-capture new conversations"""
    print("\n" + "="*70)
    print("DEMO 2: Auto-Capture New Conversations")
    print("="*70 + "\n")

    # Create conversation manager
    test_db = "./data/test_auto_capture.db"
    os.makedirs("./data", exist_ok=True)
    conv_mgr = ConversationManager(test_db)

    # Create HoloLoom instance
    async with HoloLoom() as loom:
        # Enable auto-capture
        print("Enabling auto-capture...")
        auto_capture = ChatHistoryAutoCapture(
            conv_mgr,
            loom,
            importance_threshold=0.3,
            batch_size=2  # Small batch for demo
        )

        print("✓ Auto-capture enabled\n")

        # Create a new conversation (will be auto-captured!)
        print("Creating new conversation...")
        conv = conv_mgr.create_conversation("Live Conversation Test")

        conv_mgr.add_message(conv.id, 'user', "What is Matryoshka embedding?")
        conv_mgr.add_message(conv.id, 'assistant',
            "Matryoshka embeddings are multi-scale representations where "
            "smaller dimensions nest inside larger ones, like Russian dolls. "
            "You can use 96D, 192D, or 384D depending on speed/quality tradeoff.",
            metadata={'confidence': 0.91}
        )

        # Wait for batch ingestion
        await asyncio.sleep(1)

        # Try to query the newly added conversation
        print("\nQuerying newly captured conversation...")
        results = await loom.recall("Matryoshka embedding dimensions")

        if results:
            print(f"✓ Found {len(results)} results from auto-captured conversation!")
            print(f"  Snippet: {results[0].text[:100]}...")
        else:
            print("⚠ No results yet (batch may still be pending)")

        # Disable auto-capture
        auto_capture.disable()
        print("\n✓ Demo 2 complete!\n")


async def demo_advanced_features():
    """Demo 3: Advanced features (tags, projects, importance scoring)"""
    print("\n" + "="*70)
    print("DEMO 3: Advanced Features")
    print("="*70 + "\n")

    test_db = "./data/test_advanced.db"
    os.makedirs("./data", exist_ok=True)
    conv_mgr = ConversationManager(test_db)

    # Create project
    print("Creating project...")
    project = conv_mgr.create_project(
        name="Machine Learning Research",
        color="#667eea",
        icon="🧠"
    )
    print(f"✓ Created project: {project.name}\n")

    # Create tagged conversations
    print("Creating tagged conversations...")
    conv1 = conv_mgr.create_conversation("Bandits Discussion")
    conv_mgr.move_to_project(conv1.id, project.id)
    conv_mgr.update_tags(conv1.id, ['bandits', 'exploration'])

    conv_mgr.add_message(conv1.id, 'user', "Explain contextual bandits")
    conv_mgr.add_message(conv1.id, 'assistant',
        "Contextual bandits extend multi-armed bandits by incorporating context. "
        "Each decision considers not just past rewards but also current state/features.",
        metadata={'confidence': 0.89}
    )

    print(f"✓ Created tagged conversation\n")

    # Ingest and query
    async with HoloLoom() as loom:
        spinner = ChatHistorySpinner(conv_mgr, importance_threshold=0.2)

        # Spin by tag
        print("Spinning conversations by tag...")
        shards = await spinner.spin_by_tag('bandits')
        print(f"✓ Found {len(shards)} shards with 'bandits' tag\n")

        # Spin by project
        print("Spinning conversations by project...")
        shards = await spinner.spin_by_project(project.id)
        print(f"✓ Found {len(shards)} shards in project '{project.name}'\n")

        # Ingest and query
        for shard in shards:
            await loom.experience(shard.text)

        results = await loom.recall("contextual bandits features")
        print(f"Query results: {len(results)} memories found")
        if results:
            print(f"  Top result importance: {results[0].metadata.get('importance_score', 'N/A')}")

    print("\n✓ Demo 3 complete!\n")


async def demo_importance_scoring():
    """Demo 4: Importance scoring visualization"""
    print("\n" + "="*70)
    print("DEMO 4: Importance Scoring")
    print("="*70 + "\n")

    test_db = "./data/test_importance.db"
    os.makedirs("./data", exist_ok=True)
    conv_mgr = ConversationManager(test_db)

    # Create conversations with varying importance
    test_cases = [
        {
            'title': 'High Importance - Technical Discussion',
            'messages': [
                ('user', "How does the Thompson Sampling bandit update its parameters?"),
                ('assistant',
                 "The Thompson Sampling bandit maintains Beta(α,β) distributions for each action. "
                 "On success (confidence ≥ 0.75), α increases by the confidence value. "
                 "On failure, β increases by (1 - confidence). This creates probability matching.",
                 {'confidence': 0.95, 'reasoning_steps': ['analyze', 'lookup', 'synthesize']})
            ]
        },
        {
            'title': 'Medium Importance - Clarification',
            'messages': [
                ('user', "Can you explain that more simply?"),
                ('assistant', "Sure! Think of it like this: successful actions get stronger, failed ones get weaker.",
                 {'confidence': 0.82})
            ]
        },
        {
            'title': 'Low Importance - Greeting',
            'messages': [
                ('user', "Thanks!"),
                ('assistant', "You're welcome! Let me know if you need anything else.",
                 {'confidence': 0.75})
            ]
        }
    ]

    spinner = ChatHistorySpinner(conv_mgr, importance_threshold=0.0)  # Include all

    for case in test_cases:
        conv = conv_mgr.create_conversation(case['title'])
        for role, content, *metadata in case['messages']:
            meta = metadata[0] if metadata else {}
            conv_mgr.add_message(conv.id, role, content, metadata=meta)

        # Get importance scores
        shards = await spinner.spin_conversation(conv.id, min_importance=0.0)

        if shards:
            shard = shards[0]
            score = shard.metadata['importance_score']
            reason = shard.metadata['importance_reason']

            print(f"Conversation: {case['title']}")
            print(f"  Importance: {score:.2f}")
            print(f"  Signals: {reason}")
            print()

    print("✓ Demo 4 complete!\n")


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("Chat History Integration Demo Suite")
    print("="*70)

    await demo_basic_ingestion()
    await demo_auto_capture()
    await demo_advanced_features()
    await demo_importance_scoring()

    print("\n" + "="*70)
    print("All Demos Complete!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. ChatHistorySpinner converts conversations to queryable MemoryShards")
    print("2. Auto-capture keeps memory in sync with new conversations")
    print("3. Importance scoring filters noise from signal")
    print("4. Tag/project filtering enables focused ingestion")
    print("5. All conversation history is now searchable via HoloLoom.recall()")
    print()


if __name__ == "__main__":
    asyncio.run(main())
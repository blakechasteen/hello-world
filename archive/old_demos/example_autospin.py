#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoSpin Example
================
Demonstrates how to use AutoSpinOrchestrator for automatic text spinning.

The orchestrator automatically converts your text into MemoryShards
without you having to manually call the TextSpinner.
"""

import asyncio
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.autospin import AutoSpinOrchestrator, auto_loom_from_text
from HoloLoom.Documentation.types import Query
from HoloLoom.config import Config


async def example_basic_autospin():
    """Most basic example: just provide text and ask questions."""
    print("=" * 70)
    print("Example 1: Basic Auto-Spin")
    print("=" * 70)

    # Your knowledge base as plain text
    knowledge_base = """
    HoloLoom is a neural decision-making system that combines multi-scale
    embeddings with knowledge graph memory. The system uses a weaving metaphor
    where independent warp thread modules are coordinated by an orchestrator.

    The SpinningWheel is the input adapter system. It automatically converts
    raw data from different modalities (text, audio, video) into standardized
    MemoryShards that the orchestrator can process.

    The policy engine uses Thompson Sampling for exploration-exploitation
    balance. It combines neural network predictions with Bayesian bandits
    to make optimal tool selection decisions.
    """

    print("\nCreating orchestrator from text...")
    print(f"Knowledge base: {len(knowledge_base)} characters")

    # Auto-spin: text -> shards -> orchestrator (all in one step!)
    orchestrator = await auto_loom_from_text(
        text=knowledge_base,
        config=Config.fast(),  # Use fast mode for quick demo
        chunk_by='paragraph',
        chunk_size=300
    )

    print(f"✓ Created {orchestrator.get_shard_count()} memory shards")

    # Now ask questions
    query = Query(text="What is HoloLoom?")
    print(f"\nQuery: {query.text}")

    response = await orchestrator.process(query)

    print(f"\nResponse:")
    print(f"  Status: {response.get('status', 'unknown')}")
    print(f"  Tool: {response.get('tool', 'unknown')}")
    print(f"  Confidence: {response.get('confidence', 0):.2f}")
    if 'result' in response:
        print(f"  Result: {response['result']}")


async def example_from_file():
    """Load knowledge base from a file."""
    print("\n" + "=" * 70)
    print("Example 2: Auto-Spin from File")
    print("=" * 70)

    # Create a sample file
    sample_file = Path("sample_knowledge.txt")
    sample_content = """
# HoloLoom Architecture

## Core Components

The orchestrator is the central shuttle that weaves together all components.
It coordinates motif detection, embeddings, memory retrieval, and policy decisions.

The memory system combines vector similarity search with knowledge graph traversal.
This hybrid approach provides both semantic and structural context.

The policy engine makes decisions using a unified neural architecture with
Thompson Sampling for exploration. It can switch between different execution
modes: bare, fast, and fused.

## SpinningWheel

The SpinningWheel provides input adapters for different modalities:
- TextSpinner: Plain text and markdown documents
- AudioSpinner: Audio transcripts and metadata
- YouTubeSpinner: Video captions with timestamps

All spinners output standardized MemoryShards that feed into the orchestrator.
    """

    print(f"\nWriting sample file: {sample_file}")
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)

    print("Creating orchestrator from file...")

    # Auto-spin from file
    orchestrator = await AutoSpinOrchestrator.from_file(
        file_path=sample_file,
        config=Config.fast()
    )

    print(f"✓ Loaded {orchestrator.get_shard_count()} shards from file")

    # Preview the shards
    print("\nShard preview:")
    for shard_info in orchestrator.get_shard_preview(limit=3):
        print(f"  - {shard_info['id']}: {shard_info['char_count']} chars")
        print(f"    Text: {shard_info['text_preview']}")

    # Ask a question
    query = Query(text="What are the SpinningWheel modalities?")
    print(f"\nQuery: {query.text}")

    response = await orchestrator.process(query)
    print(f"Tool selected: {response.get('tool')}")

    # Cleanup
    sample_file.unlink()
    print(f"\n✓ Cleaned up sample file")


async def example_multiple_documents():
    """Create orchestrator from multiple documents."""
    print("\n" + "=" * 70)
    print("Example 3: Multiple Documents")
    print("=" * 70)

    documents = [
        {
            'text': """
            The Policy Engine uses Thompson Sampling for exploration.
            It maintains success/failure statistics for each tool and
            samples from Beta distributions to balance exploitation
            with exploration.
            """,
            'source': 'policy_docs.md',
            'metadata': {'topic': 'policy', 'priority': 'high'}
        },
        {
            'text': """
            Memory retrieval combines BM25 lexical search with semantic
            embeddings. The multi-scale approach uses Matryoshka embeddings
            at different dimensionalities (96, 192, 384) for efficient
            retrieval at various precision levels.
            """,
            'source': 'memory_docs.md',
            'metadata': {'topic': 'memory', 'priority': 'medium'}
        },
        {
            'text': """
            The motif detector identifies patterns in queries using both
            regex and optional NLP. Detected motifs are passed to the
            policy engine to influence tool selection decisions.
            """,
            'source': 'motif_docs.md',
            'metadata': {'topic': 'motifs', 'priority': 'low'}
        }
    ]

    print(f"Creating orchestrator from {len(documents)} documents...")

    orchestrator = await AutoSpinOrchestrator.from_documents(
        documents=documents,
        config=Config.fast()
    )

    print(f"✓ Created {orchestrator.get_shard_count()} shards from {len(documents)} documents")

    # Show shard sources
    print("\nShard sources:")
    for shard_info in orchestrator.get_shard_preview(limit=10):
        print(f"  - {shard_info['source']}: {shard_info['id']}")

    # Ask questions about different topics
    queries = [
        "How does Thompson Sampling work?",
        "What is BM25?",
        "What are motifs?"
    ]

    for q_text in queries:
        query = Query(text=q_text)
        print(f"\nQuery: {query.text}")
        response = await orchestrator.process(query)
        print(f"  Tool: {response.get('tool')}, Confidence: {response.get('confidence', 0):.2f}")


async def example_dynamic_addition():
    """Add content dynamically after creation."""
    print("\n" + "=" * 70)
    print("Example 4: Dynamic Content Addition")
    print("=" * 70)

    # Start with initial content
    initial_content = """
    HoloLoom started as a research project exploring neural decision-making.
    The initial implementation focused on PPO reinforcement learning.
    """

    print("Creating orchestrator with initial content...")
    orchestrator = await auto_loom_from_text(
        text=initial_content,
        chunk_by=None  # Single shard
    )

    print(f"Initial shard count: {orchestrator.get_shard_count()}")

    # Add more content dynamically
    additional_content = """
    Later, the project evolved to include knowledge graph memory,
    Thompson Sampling for exploration, and the SpinningWheel input system.
    The weaving metaphor emerged as a unifying architectural principle.
    """

    print("\nAdding more content dynamically...")
    await orchestrator.add_text(
        text=additional_content,
        source="update_2025_10"
    )

    print(f"Updated shard count: {orchestrator.get_shard_count()}")

    # Query can now access both old and new content
    query = Query(text="Tell me about HoloLoom's evolution")
    print(f"\nQuery: {query.text}")

    response = await orchestrator.process(query)
    print(f"Response: {response.get('result', 'No result')}")


async def example_configuration_modes():
    """Show different configuration modes."""
    print("\n" + "=" * 70)
    print("Example 5: Different Configuration Modes")
    print("=" * 70)

    content = "HoloLoom supports bare, fast, and fused execution modes."

    print("\n--- BARE Mode (fastest, minimal features) ---")
    orch_bare = await auto_loom_from_text(content, config=Config.bare())
    print(f"Created with bare mode: {orch_bare.get_shard_count()} shards")

    print("\n--- FAST Mode (balanced) ---")
    orch_fast = await auto_loom_from_text(content, config=Config.fast())
    print(f"Created with fast mode: {orch_fast.get_shard_count()} shards")

    print("\n--- FUSED Mode (full features, slower) ---")
    orch_fused = await auto_loom_from_text(content, config=Config.fused())
    print(f"Created with fused mode: {orch_fused.get_shard_count()} shards")

    print("\n✓ All modes created successfully")


async def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("AutoSpin Orchestrator Examples")
    print("=" * 70)

    try:
        await example_basic_autospin()
        await example_from_file()
        await example_multiple_documents()
        await example_dynamic_addition()
        await example_configuration_modes()

        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
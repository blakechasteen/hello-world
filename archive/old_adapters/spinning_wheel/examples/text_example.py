#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Text Spinner Example
====================
Demonstrates how to use the TextSpinner to convert plain text documents
into MemoryShards for HoloLoom processing.

Usage:
    PYTHONPATH=. python HoloLoom/spinningWheel/examples/text_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from HoloLoom.spinning_wheel import TextSpinner, TextSpinnerConfig, spin_text


async def example_basic_usage():
    """Basic usage: Single shard for entire document."""
    print("=" * 60)
    print("Example 1: Basic Usage - Single Shard")
    print("=" * 60)

    # Sample text
    text = """
    HoloLoom is a neural decision-making system that combines multi-scale
    embeddings with knowledge graph memory. The system uses a weaving metaphor
    where independent warp thread modules are coordinated by an orchestrator.

    The core components include a policy engine with Thompson Sampling,
    spectral graph features, and PPO reinforcement learning for agent training.
    """

    # Quick usage with convenience function
    shards = await spin_text(
        text=text.strip(),
        source='hololoom_intro.txt'
    )

    print(f"\nGenerated {len(shards)} shard(s)")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Text (first 100 chars): {shard.text[:100]}...")
        print(f"Entities: {shard.entities}")
        print(f"Metadata: {shard.metadata}")


async def example_paragraph_chunking():
    """Example with paragraph-based chunking."""
    print("\n" + "=" * 60)
    print("Example 2: Paragraph Chunking")
    print("=" * 60)

    text = """
The SpinningWheel is HoloLoom's input adapter system. It converts raw modality
data into standardized MemoryShard objects.

The philosophy is simple: keep spinners thin and focused. They normalize input
into a common format, optionally add enrichment, and let the Orchestrator
handle the heavy processing.

Available spinners include AudioSpinner for transcripts, YouTubeSpinner for
video captions, and TextSpinner for documents. Each follows the same pattern:
accept raw data, extract basic structure, return MemoryShards.

The enrichment layer can optionally add semantic context using Ollama, Neo4j,
or mem0 before the data reaches the main processing pipeline.
    """

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=200,  # Keep chunks around 200 characters
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': text.strip(),
        'source': 'spinningwheel_docs.md',
        'metadata': {'topic': 'architecture', 'author': 'system'}
    })

    print(f"\nGenerated {len(shards)} shard(s) from {len(text)} characters")
    for idx, shard in enumerate(shards):
        print(f"\n--- Shard {idx + 1} ---")
        print(f"ID: {shard.id}")
        print(f"Length: {len(shard.text)} chars")
        print(f"Entities: {shard.entities}")
        print(f"Text: {shard.text[:150]}...")


async def example_sentence_chunking():
    """Example with sentence-based chunking."""
    print("\n" + "=" * 60)
    print("Example 3: Sentence Chunking")
    print("=" * 60)

    text = """
The orchestrator is the central coordinator. It imports from all modules.
Each module is independent and protocol-based. They communicate via shared types.
The policy engine makes decisions using neural networks. Thompson Sampling
provides exploration. The memory system combines vectors and knowledge graphs.
Spectral features capture graph topology. Everything flows through the
unified pipeline.
    """

    config = TextSpinnerConfig(
        chunk_by='sentence',
        chunk_size=150,
        min_chunk_size=30,
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': text.strip(),
        'source': 'architecture_notes.txt'
    })

    print(f"\nGenerated {len(shards)} shard(s)")
    for idx, shard in enumerate(shards):
        print(f"\n--- Shard {idx + 1} ---")
        print(f"Text: {shard.text}")
        print(f"Entities: {shard.entities}")


async def example_with_metadata():
    """Example showing rich metadata usage."""
    print("\n" + "=" * 60)
    print("Example 4: Rich Metadata")
    print("=" * 60)

    text = """
    Meeting Notes - Project Review
    Date: 2025-10-23

    Discussed the new TextSpinner implementation. The team agreed on
    paragraph-based chunking as the default. Entity extraction will use
    simple heuristics initially, with optional Ollama enrichment.

    Action items: Test with real documents, add markdown preservation,
    consider spaCy integration for better entity detection.
    """

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=300,
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': text.strip(),
        'source': 'meeting_notes_2025_10_23.txt',
        'episode': 'project_review_oct_2025',
        'metadata': {
            'date': '2025-10-23',
            'type': 'meeting_notes',
            'participants': ['team'],
            'priority': 'high'
        }
    })

    print(f"\nGenerated {len(shards)} shard(s)")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Episode: {shard.episode}")
        print(f"Entities: {shard.entities}")
        print(f"Metadata: {shard.metadata}")
        print(f"Text preview: {shard.text[:100]}...")


async def example_factory_pattern():
    """Example using the factory pattern."""
    print("\n" + "=" * 60)
    print("Example 5: Factory Pattern")
    print("=" * 60)

    from HoloLoom.spinning_wheel import create_spinner

    config = TextSpinnerConfig(chunk_by='paragraph', chunk_size=200)
    spinner = create_spinner('text', config)

    text = "HoloLoom uses factory patterns for spinner creation. This makes it easy to switch between different modalities. The interface is consistent across all spinners."

    shards = await spinner.spin({
        'text': text,
        'source': 'factory_example.txt'
    })

    print(f"\nCreated spinner via factory: {type(spinner).__name__}")
    print(f"Generated {len(shards)} shard(s)")
    for shard in shards:
        print(f"\nShard: {shard.text}")


async def main():
    """Run all examples."""
    await example_basic_usage()
    await example_paragraph_chunking()
    await example_sentence_chunking()
    await example_with_metadata()
    await example_factory_pattern()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())

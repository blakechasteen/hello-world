#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script for TextSpinner
"""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct import to avoid package init issues
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load dependencies
repo_root = Path(__file__).parent
load_module("HoloLoom.spinning_wheel.base", repo_root / "HoloLoom" / "spinningWheel" / "base.py")
text_module = load_module("HoloLoom.spinning_wheel.text", repo_root / "HoloLoom" / "spinningWheel" / "text.py")

TextSpinner = text_module.TextSpinner
TextSpinnerConfig = text_module.TextSpinnerConfig
spin_text = text_module.spin_text


async def test_basic():
    """Test basic text spinning."""
    print("=" * 60)
    print("Testing TextSpinner - Basic Usage")
    print("=" * 60)

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
        source='test_document.txt'
    )

    print(f"\nGenerated {len(shards)} shard(s)")
    for shard in shards:
        print(f"\nShard ID: {shard.id}")
        print(f"Episode: {shard.episode}")
        print(f"Text length: {len(shard.text)} chars")
        print(f"Entities: {shard.entities}")
        print(f"Metadata: {shard.metadata}")
        print(f"Text preview: {shard.text[:100]}...")

    return shards


async def test_chunking():
    """Test paragraph chunking."""
    print("\n" + "=" * 60)
    print("Testing TextSpinner - Paragraph Chunking")
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
    """

    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=200,
        extract_entities=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': text.strip(),
        'source': 'architecture_docs.md',
        'metadata': {'topic': 'spinningwheel', 'version': '1.0'}
    })

    print(f"\nGenerated {len(shards)} shard(s)")
    for idx, shard in enumerate(shards):
        print(f"\n--- Chunk {idx + 1} ---")
        print(f"ID: {shard.id}")
        print(f"Length: {len(shard.text)} chars")
        print(f"Entities: {shard.entities}")
        print(f"Text: {shard.text[:120]}...")

    return shards


async def test_sentence_chunking():
    """Test sentence-based chunking."""
    print("\n" + "=" * 60)
    print("Testing TextSpinner - Sentence Chunking")
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
        print(f"\n--- Sentence Chunk {idx + 1} ---")
        print(f"Text: {shard.text}")
        print(f"Entities: {shard.entities}")

    return shards


async def main():
    """Run all tests."""
    print("\nHoloLoom TextSpinner Test Suite\n")

    shards1 = await test_basic()
    shards2 = await test_chunking()
    shards3 = await test_sentence_chunking()

    print("\n" + "=" * 60)
    print(f"All tests completed!")
    print(f"   - Basic test: {len(shards1)} shard(s)")
    print(f"   - Paragraph chunking: {len(shards2)} shard(s)")
    print(f"   - Sentence chunking: {len(shards3)} shard(s)")
    print("=" * 60)


if __name__ == '__main__':
    asyncio.run(main())
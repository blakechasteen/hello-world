#!/usr/bin/env python3
"""
Filesystem RAG Demo
===================

Demonstrates filesystem ingestion into HoloLoom's RAG system.

This demo shows:
1. Ingesting files from a directory
2. Querying the ingested content
3. Incremental updates (only new/modified files)
4. Importance-based filtering

**Setup**:
    Create a test directory with some Markdown/text files.

**Usage**:
    PYTHONPATH=. python demos/demo_filesystem_rag.py

Author: Claude Code
Date: November 2025
"""

import asyncio
from pathlib import Path
import tempfile
import time

from HoloLoom.spinningWheel.filesystem_spinner import FilesystemSpinner, ingest_directory
from HoloLoom.hololoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.rag import SimpleRAG


# ============================================================================
# Demo Setup
# ============================================================================

def create_test_files(base_dir: Path):
    """Create test files for the demo."""

    # Create directory structure
    docs_dir = base_dir / "docs"
    docs_dir.mkdir()

    # Create README.md (high importance - README)
    readme = docs_dir / "README.md"
    readme.write_text("""# HoloLoom Documentation

HoloLoom is a neural decision-making system that combines:
- Multi-scale embeddings (Matryoshka representations)
- Knowledge graph memory with spectral features
- Unified policy engine with Thompson Sampling exploration
- PPO reinforcement learning for agent training

The system is designed around a "weaving" metaphor where independent
modules are coordinated by an orchestrator to produce responses.

## Key Features

- **Zero-copy embeddings**: 37.7x faster scale extraction
- **Adaptive learning**: 7 parallel learning loops
- **RAG system**: Level 4 agentic RAG with graph traversal
- **Alignment framework**: Safety-first design with <0.11ms overhead

## Getting Started

See the quick start guide for installation and basic usage.
""")

    # Create guide.md (medium importance)
    guide = docs_dir / "guide.md"
    guide.write_text("""# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

```python
from HoloLoom import HoloLoom

async with HoloLoom() as loom:
    # Experience (form memories)
    await loom.experience("Thompson Sampling balances exploration")

    # Recall (retrieve memories)
    memories = await loom.recall("What did I learn about sampling?")
```

## Next Steps

Check out the advanced examples in the examples/ directory.
""")

    # Create notes.txt (lower importance - plain text)
    notes = docs_dir / "notes.txt"
    notes.write_text("""Development notes for HoloLoom:

- Need to test incremental ingestion
- Consider adding PDF support
- Update documentation
- Fix minor bugs in alignment framework

Remember to run tests before pushing!
""")

    # Create a log file (will be excluded)
    log = docs_dir / "debug.log"
    log.write_text("""[2025-11-18 10:00] Started server
[2025-11-18 10:01] Processing request
[2025-11-18 10:02] Request completed
""")

    return docs_dir


# ============================================================================
# Demo Steps
# ============================================================================

async def demo_basic_ingestion():
    """Demo 1: Basic filesystem ingestion."""
    print("=" * 70)
    print("DEMO 1: Basic Filesystem Ingestion")
    print("=" * 70)
    print()

    # Create test files
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = create_test_files(Path(tmpdir))

        print(f"Created test directory: {docs_dir}")
        print(f"Files:")
        for file in sorted(docs_dir.glob("*")):
            print(f"  - {file.name}")
        print()

        # Ingest with default settings
        print("Ingesting files...")
        result = await ingest_directory(
            str(docs_dir),
            allow_patterns=["*.md", "*.txt"],
            deny_patterns=["*.log"],
            recursive=False
        )

        print(f"\n✅ Ingestion complete!")
        print(f"   Shards created: {result.shard_count}")
        print(f"   Average importance: {result.avg_importance:.2f}")
        print(f"   Processing time: {result.processing_time_ms:.0f}ms")

        # Show sample shards
        print(f"\n📄 Sample shards:")
        for i, shard in enumerate(result.shards[:3]):
            print(f"\n{i+1}. {shard.metadata['file_name']} (chunk {shard.metadata['chunk_index'] + 1}/{shard.metadata['total_chunks']})")
            print(f"   Importance: {shard.metadata['importance_score']:.2f} ({shard.metadata['importance_reason']})")
            print(f"   Preview: {shard.text[:100]}...")


async def demo_incremental_ingestion():
    """Demo 2: Incremental ingestion (only new/modified files)."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Incremental Ingestion")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = create_test_files(Path(tmpdir))
        checkpoint_dir = Path(tmpdir) / "checkpoints"
        checkpoint_dir.mkdir()

        # First ingestion
        print("First ingestion (all files)...")
        spinner = FilesystemSpinner(
            allow_patterns=["*.md", "*.txt"],
            deny_patterns=["*.log"],
            checkpoint_dir=checkpoint_dir
        )

        result1 = await spinner.spin_incremental(str(docs_dir))
        print(f"✅ Created {result1.shard_count} shards")

        # Second ingestion (no changes)
        print("\nSecond ingestion (no changes)...")
        time.sleep(0.1)  # Small delay

        result2 = await spinner.spin_incremental(str(docs_dir))
        print(f"✅ Created {result2.shard_count} shards (should be 0)")

        # Modify a file
        print("\nModifying guide.md...")
        guide = docs_dir / "guide.md"
        current_content = guide.read_text()
        guide.write_text(current_content + "\n\n## Updated section\n\nNew content added!")

        # Third ingestion (only modified file)
        print("Third ingestion (only modified file)...")
        result3 = await spinner.spin_incremental(str(docs_dir))
        print(f"✅ Created {result3.shard_count} shards (only from guide.md)")

        if result3.warnings:
            print(f"   {result3.warnings[0]}")


async def demo_importance_filtering():
    """Demo 3: Importance-based filtering."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Importance Filtering")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = create_test_files(Path(tmpdir))

        # Ingest with different thresholds
        thresholds = [0.0, 0.5, 0.7]

        for threshold in thresholds:
            print(f"\nImportance threshold: {threshold}")

            spinner = FilesystemSpinner(
                allow_patterns=["*.md", "*.txt"],
                importance_threshold=threshold
            )

            result = await spinner.spin(str(docs_dir))

            print(f"  Shards created: {result.shard_count}")

            if result.warnings:
                print(f"  {result.warnings[0]}")


async def demo_rag_integration():
    """Demo 4: Integration with SimpleRAG."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: RAG Integration")
    print("=" * 70)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = create_test_files(Path(tmpdir))

        # Ingest files
        print("Ingesting files...")
        spinner = FilesystemSpinner(
            allow_patterns=["*.md", "*.txt"],
            deny_patterns=["*.log"]
        )

        result = await spinner.spin(str(docs_dir))
        print(f"✅ Created {result.shard_count} shards")

        # Create HoloLoom instance and add shards
        print("\nAdding to HoloLoom memory...")
        config = Config.fast()

        async with HoloLoom(cfg=config) as loom:
            # Add all shards
            for shard in result.shards:
                await loom.experience(shard.text)

            print(f"✅ Added {len(result.shards)} shards to memory")

            # Query
            print("\nQuerying: 'What is Thompson Sampling?'")
            memories = await loom.recall("What is Thompson Sampling?")

            if memories:
                print(f"✅ Retrieved {len(memories)} memories")
                print(f"\nTop result:")
                print(f"  {memories[0].content[:200]}...")
            else:
                print("❌ No memories found")

            # Get metrics
            metrics = loom.get_metrics()
            print(f"\n📊 Memory metrics:")
            print(f"   Active nodes: {metrics['activation']['active_nodes']}")
            print(f"   Total nodes: {metrics['activation']['total_nodes']}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" " * 15 + "Filesystem RAG Demo")
    print("=" * 70)
    print()
    print("This demo shows how to ingest files from the filesystem")
    print("into HoloLoom's RAG system.")
    print()

    # Run demos
    await demo_basic_ingestion()
    await demo_incremental_ingestion()
    await demo_importance_filtering()
    await demo_rag_integration()

    print("\n\n" + "=" * 70)
    print("✅ All demos complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Try the CLI: python -m HoloLoom.ingestion.filesystem /path/to/docs")
    print("  - Explore SimpleRAG: from HoloLoom.rag import SimpleRAG")
    print("  - Read the docs: HoloLoom/spinningWheel/README.md")
    print()


if __name__ == "__main__":
    asyncio.run(main())

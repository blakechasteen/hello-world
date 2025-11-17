#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: OrgMode Spinner Integration
==================================

Demonstrates zero-copy integration between Emacs org-mode and HoloLoom.

Usage:
    python demo_orgmode_spinner.py
    PYTHONPATH=. python demo_orgmode_spinner.py test_notes.org
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.orgmode import OrgModeSpinner, OrgFileWatcher
from HoloLoom.spinningWheel.base import SpinnerConfig


def print_shard(shard, detailed=False):
    """Pretty print a MemoryShard."""
    print(f"\n{'='*80}")
    print(f"ID: {shard.id}")
    print(f"Episode: {shard.episode}")
    print(f"\nText Preview:")
    preview = shard.text[:200] + "..." if len(shard.text) > 200 else shard.text
    print(f"  {preview}")

    print(f"\nEntities ({len(shard.entities)}):")
    for entity in shard.entities[:5]:  # Show first 5
        print(f"  - {entity}")
    if len(shard.entities) > 5:
        print(f"  ... and {len(shard.entities) - 5} more")

    print(f"\nMotifs ({len(shard.motifs)}):")
    for motif in shard.motifs:
        print(f"  - {motif}")

    if detailed:
        print(f"\nMetadata:")
        for key, value in shard.metadata.items():
            if isinstance(value, (list, dict)):
                print(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                print(f"  {key}: {value}")


async def demo_inline_parsing():
    """Demo: Parse org-mode content from string."""
    print("\n" + "="*80)
    print("DEMO 1: Inline Org-Mode Parsing")
    print("="*80)

    org_content = """
* Machine Learning Project                            :ai:ml:
  :PROPERTIES:
  :ID: ml-project-2025
  :END:

** TODO Train model on new dataset                    :priority:
   DEADLINE: <2025-11-25>

   Need to preprocess data and run training pipeline.
   See [[file:training.py][training script]] for details.

*** DONE Collect training data
    CLOSED: [2025-11-16 Sat]

*** IN-PROGRESS Preprocessing pipeline
    Working on feature engineering and normalization.

** Ideas for improvement                              :brainstorm:

   Consider using:
   - Matryoshka embeddings for multi-scale
   - Thompson Sampling for exploration
   - Graph neural networks for relational data
"""

    config = SpinnerConfig(enable_enrichment=False)
    spinner = OrgModeSpinner(config)

    shards = await spinner.spin({
        'content': org_content,
        'source': 'inline_demo',
        'episode': 'demo:inline'
    })

    print(f"\nParsed {len(shards)} shards from inline org-mode content:")

    for i, shard in enumerate(shards):
        print_shard(shard, detailed=(i == 0))  # Show detailed for first shard only

    # Demonstrate relationship mapping
    print("\n" + "="*80)
    print("HIERARCHICAL RELATIONSHIPS:")
    print("="*80)
    for shard in shards:
        parent = shard.metadata.get('parent_id', 'ROOT')
        level = shard.metadata.get('level', 0)
        indent = "  " * (level - 1)
        title = shard.entities[0] if shard.entities else shard.id
        print(f"{indent}└─ {title} (parent: {parent})")


async def demo_file_parsing(filepath: str):
    """Demo: Parse org-mode file."""
    print("\n" + "="*80)
    print(f"DEMO 2: File Parsing - {filepath}")
    print("="*80)

    config = SpinnerConfig(enable_enrichment=False)
    spinner = OrgModeSpinner(config)

    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    shards = await spinner.spin({
        'content': content,
        'source': filepath,
        'episode': f'org:{Path(filepath).stem}'
    })

    print(f"\nParsed {len(shards)} shards from {filepath}:")

    # Group by TODO state
    todo_groups = {}
    for shard in shards:
        todo_motifs = [m for m in shard.motifs if m.startswith('TODO:')]
        if todo_motifs:
            state = todo_motifs[0].split(':')[1]
            todo_groups.setdefault(state, []).append(shard)
        else:
            todo_groups.setdefault('NO_TODO', []).append(shard)

    print("\nTODO State Distribution:")
    for state, group_shards in sorted(todo_groups.items()):
        print(f"  {state}: {len(group_shards)} items")

    # Show shards with deadlines
    print("\n" + "="*80)
    print("ITEMS WITH DEADLINES:")
    print("="*80)
    deadline_shards = [s for s in shards if 'deadline' in s.metadata]
    for shard in deadline_shards:
        title = shard.entities[0] if shard.entities else "Untitled"
        deadline = shard.metadata['deadline']
        print(f"  [{deadline}] {title}")

    # Show shards with links
    print("\n" + "="*80)
    print("ITEMS WITH LINKS (Graph Edges):")
    print("="*80)
    linked_shards = [s for s in shards if 'links' in s.metadata]
    for shard in linked_shards:
        title = shard.entities[0] if shard.entities else "Untitled"
        links = shard.metadata['links']
        print(f"\n  {title}:")
        for link in links:
            print(f"    → {link['description']} ({link['target']})")

    # Show tag distribution
    print("\n" + "="*80)
    print("TAG DISTRIBUTION:")
    print("="*80)
    tag_counts = {}
    for shard in shards:
        for motif in shard.motifs:
            if motif.startswith('TAG:'):
                tag = motif.split(':')[1]
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  :{tag}: = {count} occurrences")


async def demo_file_watching(filepath: str):
    """Demo: File watching for live updates."""
    print("\n" + "="*80)
    print(f"DEMO 3: File Watching - {filepath}")
    print("="*80)

    config = SpinnerConfig(enable_enrichment=False)
    spinner = OrgModeSpinner(config)
    watcher = OrgFileWatcher(spinner)

    # Initial load
    print("\nInitial load...")
    shards = await watcher.watch_file(filepath)
    print(f"Loaded {len(shards)} shards from {filepath}")

    # Simulate checking for updates (would normally be called periodically)
    print("\nChecking for updates (cached)...")
    shards2 = await watcher.watch_file(filepath)
    print(f"Returned {len(shards2)} shards (from cache)")

    print("\nIn a real application, this watcher would:")
    print("  - Run in background thread/process")
    print("  - Use inotify/watchdog for file system events")
    print("  - Incrementally update YarnGraph when file changes")
    print("  - Preserve temporal evolution of knowledge")


async def main():
    """Run all demos."""
    # Demo 1: Inline parsing
    await demo_inline_parsing()

    # Demo 2: File parsing (if file provided or use test file)
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'test_notes.org'

    if Path(filepath).exists():
        await demo_file_parsing(filepath)
        await demo_file_watching(filepath)
    else:
        print(f"\nSkipping file demos - {filepath} not found")
        print("Create test_notes.org or provide a path as argument")

    print("\n" + "="*80)
    print("SUMMARY: Zero-Copy Org-Mode Integration")
    print("="*80)
    print("""
The OrgModeSpinner enables seamless integration between Emacs org-mode
and HoloLoom's neural decision-making system:

✓ Headings → Entities (graph nodes)
✓ Hierarchy → Parent-child relationships (graph edges)
✓ TODO states → Task motifs
✓ Tags → Entity metadata
✓ Timestamps → Temporal metadata for ChronoTrigger
✓ Links → Explicit graph edges
✓ Properties → Rich metadata

This means you can:
1. Think and organize in org-mode (your preferred tool)
2. Have HoloLoom automatically extract structure
3. Query your org knowledge base using neural retrieval
4. Let the system learn from your organizational patterns

Zero-copy philosophy: Minimal transformation, maximal fidelity.
Org-mode remains the source of truth.
""")


if __name__ == '__main__':
    asyncio.run(main())

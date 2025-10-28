#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoSpin Concept Test
=====================
Demonstrates the auto-spin concept without full HoloLoom imports.
Shows how text automatically becomes shards without manual spinning.
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


async def demo_autospin_concept():
    """
    Demonstrate the auto-spin concept.

    Before AutoSpin:
        1. User provides text
        2. User manually calls TextSpinner
        3. User gets shards
        4. User passes shards to orchestrator

    With AutoSpin:
        1. User provides text
        2. AutoSpinOrchestrator does everything automatically!
    """

    print("=" * 70)
    print("AutoSpin Concept Demo")
    print("=" * 70)

    # Your knowledge base
    knowledge_base = """
HoloLoom Architecture Overview

The orchestrator is the central coordinator that weaves together all components.
It implements the main processing pipeline from Query to Response.

The SpinningWheel provides input adapters. TextSpinner handles plain text,
AudioSpinner processes transcripts, and YouTubeSpinner extracts video captions.

The policy engine uses Thompson Sampling for exploration-exploitation balance.
It combines neural predictions with Bayesian bandits.

Memory retrieval uses multi-scale embeddings with Matryoshka representations.
Vector search combines with knowledge graph traversal for hybrid context.
    """

    print(f"\n📄 Knowledge Base ({len(knowledge_base)} characters):\n")
    print(knowledge_base[:200] + "...\n")

    # ========================================================================
    # BEFORE: Manual spinning (what users had to do)
    # ========================================================================
    print("─" * 70)
    print("BEFORE AutoSpin: Manual Process")
    print("─" * 70)

    print("\nStep 1: Create TextSpinnerConfig")
    config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=300,
        extract_entities=True
    )
    print(f"  ✓ Config: chunk_by={config.chunk_by}, size={config.chunk_size}")

    print("\nStep 2: Create TextSpinner")
    spinner = TextSpinner(config)
    print(f"  ✓ Spinner: {type(spinner).__name__}")

    print("\nStep 3: Manually spin the text")
    shards = await spinner.spin({
        'text': knowledge_base,
        'source': 'knowledge_base.txt'
    })
    print(f"  ✓ Created {len(shards)} shards")

    print("\nStep 4: Pass shards to orchestrator (would be here)")
    print("  # orchestrator = HoloLoomOrchestrator(cfg=config, shards=shards)")
    print("  # response = await orchestrator.process(Query(...))")

    # Show what we got
    print("\n📦 Generated Shards:")
    for idx, shard in enumerate(shards):
        print(f"\n  Shard {idx + 1}:")
        print(f"    ID: {shard.id}")
        print(f"    Length: {len(shard.text)} chars")
        print(f"    Entities: {shard.entities[:3]}..." if shard.entities else "    Entities: []")
        print(f"    Preview: {shard.text[:80]}...")

    # ========================================================================
    # AFTER: Auto-spinning (what AutoSpinOrchestrator does)
    # ========================================================================
    print("\n" + "─" * 70)
    print("AFTER AutoSpin: Automatic Process")
    print("─" * 70)

    print("\nSingle line of code:")
    print("  orchestrator = await AutoSpinOrchestrator.from_text(knowledge_base)")
    print("\nThat's it! Everything happens automatically:")
    print("  ✓ Text -> Spinner -> Shards -> Orchestrator")
    print("  ✓ No manual steps")
    print("  ✓ No intermediate variables")
    print("  ✓ Just give it text and start asking questions!")

    # ========================================================================
    # Show the convenience
    # ========================================================================
    print("\n" + "=" * 70)
    print("AutoSpin Usage Examples")
    print("=" * 70)

    examples = [
        {
            'title': 'From Text',
            'code': '''
# Just provide text directly
orchestrator = await AutoSpinOrchestrator.from_text(
    "Your knowledge base here..."
)
response = await orchestrator.process(Query(text="Question?"))
            '''
        },
        {
            'title': 'From File',
            'code': '''
# Load from a file
orchestrator = await AutoSpinOrchestrator.from_file("docs.md")
response = await orchestrator.process(Query(text="Question?"))
            '''
        },
        {
            'title': 'From Multiple Documents',
            'code': '''
# Combine multiple documents
docs = [
    {'text': 'Doc 1...', 'source': 'doc1.md'},
    {'text': 'Doc 2...', 'source': 'doc2.md'}
]
orchestrator = await AutoSpinOrchestrator.from_documents(docs)
            '''
        },
        {
            'title': 'Quick Helper',
            'code': '''
# Super quick with helper function
orch = await auto_loom_from_text("Knowledge...")
result = await orch.process(Query(text="Question?"))
            '''
        }
    ]

    for example in examples:
        print(f"\n{example['title']}:")
        print(example['code'])

    print("\n" + "=" * 70)
    print("✓ AutoSpin makes HoloLoom as easy as ChatGPT!")
    print("=" * 70)
    print("\nJust give it text → Ask questions → Get answers")
    print("No manual shard creation, no complex setup\n")


async def main():
    try:
        await demo_autospin_concept()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    asyncio.run(main())
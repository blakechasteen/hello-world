#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Conversational Demo
=============================
Chat interface with automatic memory formation.

This demonstrates:
1. Conversational interface (back-and-forth chat)
2. Automatic importance scoring (signal vs noise filtering)
3. Memory formation from conversations
4. Context building across turns

Important turns are automatically spun into memory shards.
Trivial exchanges (greetings, acknowledgments) are filtered out.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.conversational import ConversationalAutoLoom
from HoloLoom.config import Config


async def main():
    """Run the conversational demo."""

    print("=" * 80)
    print("HOLOLOOM CONVERSATIONAL DEMO")
    print("=" * 80)

    # ========================================================================
    # STEP 1: Initialize with knowledge base
    # ========================================================================
    print("\n[STEP 1] Initializing conversational system...")
    print("-" * 80)

    initial_knowledge = """
    HoloLoom Architecture Overview:

    Core Systems:
    - SpinningWheel: Data ingestion (web, audio, youtube, text)
    - Memory: Unified storage (Neo4j, Qdrant, hybrid)
    - Orchestrator: Query processing and response generation
    - Policy: Neural decision-making with Thompson Sampling

    Weaving Metaphor:
    - Yarn Graph: Persistent symbolic memory (discrete)
    - DotPlasma: Flowing feature representation (continuous)
    - Warp Space: Tensioned tensor field for computation
    - Spacetime: Woven fabric output with full trace

    Execution Modes:
    - BARE: Minimal processing (fastest)
    - FAST: Balanced processing (recommended)
    - FUSED: Full processing (highest quality)
    """

    print("Creating ConversationalAutoLoom with initial knowledge...")
    conv = await ConversationalAutoLoom.from_text(
        text=initial_knowledge,
        config=Config.fast()
    )

    print("✓ Conversational system ready")
    print("  Initial knowledge loaded")
    print("  Auto-spin enabled (importance threshold: 0.5)")

    # ========================================================================
    # STEP 2: Have a conversation
    # ========================================================================
    print("\n[STEP 2] Starting conversation...")
    print("-" * 80)
    print("(Type 'quit' or 'exit' to end, or press Ctrl+C)")
    print()

    conversation_turns = [
        "What is HoloLoom?",
        "Tell me more about the weaving metaphor",
        "What execution modes are available?",
        "Which mode should I use for production?",
        "How does the SpinningWheel work?"
    ]

    # Run pre-defined conversation
    print("Running pre-defined conversation...\n")

    for turn_idx, user_input in enumerate(conversation_turns, 1):
        print(f"[Turn {turn_idx}]")
        print(f"You: {user_input}")

        response = await conv.chat(user_input)

        print(f"HoloLoom: {response.output[:300]}...")
        print()

        # Show conversation stats
        stats = conv.get_stats()
        print(f"  Stats: {stats['total_turns']} turns, "
              f"{stats['remembered_turns']} remembered, "
              f"{stats['forgotten_turns']} forgotten")
        print(f"  Avg importance: {stats['avg_importance']:.2f}")
        print()

    # ========================================================================
    # STEP 3: Show what was learned
    # ========================================================================
    print("\n[STEP 3] Conversation memory analysis...")
    print("-" * 80)

    # Get conversation history
    history = conv.get_history()

    print(f"\nConversation had {len(history)} turns:")
    for turn in history:
        importance = turn.get('importance', 0.0)
        remembered = turn.get('remembered', False)
        status = "✓ REMEMBERED" if remembered else "✗ FORGOTTEN"

        print(f"\nTurn {turn['turn_id']}: {status} (importance: {importance:.2f})")
        print(f"  Q: {turn['user_input'][:60]}...")
        print(f"  A: {turn['system_output'][:60]}...")

    # ========================================================================
    # STEP 4: Interactive mode (optional)
    # ========================================================================
    print("\n[STEP 4] Interactive mode (optional)...")
    print("-" * 80)
    print("You can now chat interactively. Type 'quit' to exit.")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                break

            if not user_input:
                continue

            response = await conv.chat(user_input)
            print(f"HoloLoom: {response.output}")
            print()

        except EOFError:
            break
        except KeyboardInterrupt:
            break

    # ========================================================================
    # Summary
    # ========================================================================
    final_stats = conv.get_stats()

    print("\n" + "=" * 80)
    print("CONVERSATION COMPLETE")
    print("=" * 80)
    print(f"✓ Total turns: {final_stats['total_turns']}")
    print(f"✓ Remembered: {final_stats['remembered_turns']} "
          f"(signal)")
    print(f"✓ Forgotten: {final_stats['forgotten_turns']} "
          f"(noise)")
    print(f"✓ Avg importance: {final_stats['avg_importance']:.2f}")
    print()
    print("The system learned from important exchanges and")
    print("filtered out trivial ones (greetings, acknowledgments).")
    print("=" * 80)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nConversation ended")

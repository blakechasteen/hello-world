#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hofstadter Scratchpad Demo
==========================

Demonstrates persistent internal dialogue with strange loops.

Created: 2025-01-20

Scenarios:
1. Basic Exploration - Simple dialogue loop
2. Verification Mode - DS-STAR self-checking
3. Strange Loops - Hofstadter-style level-crossing
4. Session Persistence - Save/load/search
"""

import asyncio
from pathlib import Path
from hololoom.scratchpad import (
    RecursiveScratchpad,
    Thought,
    ThoughtType,
    DialogueMode,
    LoopDetector,
    StrangeLoopAnalyzer,
)


async def demo_basic_exploration():
    """Demo 1: Basic exploratory dialogue."""
    print("\n" + "="*70)
    print("Demo 1: Basic Exploratory Dialogue")
    print("="*70)

    async with RecursiveScratchpad(
        storage_path=Path("demo_scratchpad.db"),
        max_dialogue_depth=5
    ) as scratchpad:
        # Initial thought
        print("\n📝 Initial Thought:")
        initial = await scratchpad.think(
            "Thompson Sampling is a Bayesian approach to exploration-exploitation.",
            thought_type=ThoughtType.INITIAL
        )
        print(f"   {initial.text}")
        print(f"   Confidence: {initial.confidence:.2f}")

        # Start exploratory dialogue
        print("\n🔄 Starting Exploratory Dialogue...\n")
        tree = await scratchpad.dialogue_loop(
            initial_thought=initial,
            max_depth=4,
            mode="exploratory"
        )

        # Visualize
        print("\n📊 Dialogue Tree:")
        print(tree.tree_visualization())

        # Stats
        print(f"\n📈 Statistics:")
        print(f"   Total thoughts: {len(tree.thoughts)}")
        print(f"   Max depth: {tree.get_depth()}")
        print(f"   Questions: {sum(1 for t in tree.thoughts.values() if t.type == ThoughtType.QUESTION)}")
        print(f"   Answers: {sum(1 for t in tree.thoughts.values() if t.type == ThoughtType.ANSWER)}")

        # Save session
        await scratchpad.save_session("thompson_sampling_exploration")
        print("\n✅ Session saved: 'thompson_sampling_exploration'")


async def demo_verification_mode():
    """Demo 2: Verification dialogue with DS-STAR."""
    print("\n" + "="*70)
    print("Demo 2: Verification Mode (DS-STAR)")
    print("="*70)

    async with RecursiveScratchpad(
        storage_path=Path("demo_scratchpad.db"),
        enable_verification=True,
        max_dialogue_depth=4
    ) as scratchpad:
        # Initial claim
        print("\n📝 Initial Claim:")
        initial = await scratchpad.think(
            "Thompson Sampling always outperforms epsilon-greedy.",
            thought_type=ThoughtType.INITIAL
        )
        print(f"   {initial.text}")
        print(f"   Confidence: {initial.confidence:.2f}")

        # Verification dialogue
        print("\n✓ Starting Verification Dialogue...\n")
        tree = await scratchpad.dialogue_loop(
            initial_thought=initial,
            max_depth=3,
            mode="verification"
        )

        # Show tree
        print("\n📊 Verification Tree:")
        print(tree.tree_visualization())

        # Show verification results
        print("\n🔍 DS-STAR Verification Results:")
        for thought in tree.thoughts.values():
            if thought.verification:
                print(f"\n   [{thought.type.value}] {thought.text[:50]}...")
                print(f"      Domain:      {thought.verification['domain']:.2f}")
                print(f"      Sensibility: {thought.verification['sensibility']:.2f}")
                print(f"      Temporal:    {thought.verification['temporal']:.2f}")
                print(f"      Argument:    {thought.verification['argument']:.2f}")
                print(f"      Reference:   {thought.verification['reference']:.2f}")
                print(f"      Overall:     {thought.verification['overall_score']:.2f}")

        await scratchpad.save_session("verification_demo")


async def demo_strange_loops():
    """Demo 3: Hofstadter strange loops."""
    print("\n" + "="*70)
    print("Demo 3: Strange Loops (Hofstadter Mode)")
    print("="*70)

    async with RecursiveScratchpad(
        storage_path=Path("demo_scratchpad.db"),
        max_dialogue_depth=6
    ) as scratchpad:
        # Meta-cognitive starting point
        print("\n📝 Initial Meta-Thought:")
        initial = await scratchpad.think(
            "I'm reasoning about Thompson Sampling, but what assumptions am I making?",
            thought_type=ThoughtType.REFLECTION
        )
        print(f"   {initial.text}")

        # Hofstadter dialogue
        print("\n🔁 Starting Hofstadter Dialogue (Strange Loops)...\n")
        tree = await scratchpad.dialogue_loop(
            initial_thought=initial,
            max_depth=5,
            mode="hofstadter"
        )

        # Detect loops
        print("\n🔍 Detecting Strange Loops...\n")
        detector = LoopDetector()
        loops = detector.detect_loops(tree)

        # Show dialogue tree
        print("\n📊 Dialogue Tree:")
        print(tree.tree_visualization())

        # Show detected loops
        if loops:
            print(f"\n🔁 Found {len(loops)} Strange Loop(s):\n")
            for loop in loops.values():
                print(detector.visualize_loop(loop))
                print()
        else:
            print("\n   No strange loops detected (depth too shallow)")

        # Analyze loop density
        analyzer = StrangeLoopAnalyzer()
        density = analyzer.analyze_loop_density(tree, loops)
        print(f"\n📈 Loop Density: {density:.2f} (loops per thought)")

        await scratchpad.save_session("strange_loops_demo")


async def demo_persistence():
    """Demo 4: Session management and search."""
    print("\n" + "="*70)
    print("Demo 4: Session Persistence & Search")
    print("="*70)

    async with RecursiveScratchpad(
        storage_path=Path("demo_scratchpad.db")
    ) as scratchpad:
        # List sessions
        print("\n📋 Saved Sessions:")
        sessions = await scratchpad.list_sessions()
        for i, session in enumerate(sessions, 1):
            print(f"   {i}. {session['session_name']}")
            print(f"      Created: {session['created_at']}")
            print(f"      Thoughts: {session['thought_count']}")

        # Load a session
        if sessions:
            session_name = sessions[0]['session_name']
            print(f"\n📂 Loading Session: '{session_name}'")
            tree = await scratchpad.load_session(session_name)

            if tree:
                print(f"   ✓ Loaded {len(tree.thoughts)} thoughts")
                print(f"   Max depth: {tree.get_depth()}")

                # Visualize
                print("\n📊 Dialogue Tree:")
                print(tree.tree_visualization())

        # Search thoughts
        print("\n🔍 Searching for 'Thompson Sampling':")
        from hololoom.scratchpad.persistence import ThoughtPersistence
        persistence = ThoughtPersistence(Path("demo_scratchpad.db"))
        await persistence.initialize()

        results = await persistence.search_thoughts("Thompson", limit=5)
        print(f"   Found {len(results)} matching thoughts:")
        for thought in results:
            print(f"   - [{thought.type.value}] {thought.text[:60]}...")

        await persistence.close()


async def demo_synthesis_mode():
    """Demo 5: Synthesis mode - pattern recognition."""
    print("\n" + "="*70)
    print("Demo 5: Synthesis Mode (Pattern Recognition)")
    print("="*70)

    async with RecursiveScratchpad(
        storage_path=Path("demo_scratchpad.db"),
        max_dialogue_depth=4
    ) as scratchpad:
        # Multiple observations
        print("\n📝 Observations:")
        obs1 = await scratchpad.think(
            "Thompson Sampling uses Beta distributions for uncertainty.",
            thought_type=ThoughtType.INITIAL
        )
        print(f"   1. {obs1.text}")

        obs2 = await scratchpad.think(
            "Epsilon-greedy uses fixed exploration rate.",
            thought_type=ThoughtType.INITIAL,
            parent_id=obs1.id
        )
        print(f"   2. {obs2.text}")

        obs3 = await scratchpad.think(
            "UCB uses confidence bounds for exploration.",
            thought_type=ThoughtType.INITIAL,
            parent_id=obs2.id
        )
        print(f"   3. {obs3.text}")

        # Synthesis dialogue
        print("\n🔗 Starting Synthesis Dialogue...\n")
        tree = await scratchpad.dialogue_loop(
            initial_thought=obs3,
            max_depth=3,
            mode="synthesis"
        )

        # Show tree
        print("\n📊 Synthesis Tree:")
        print(tree.tree_visualization())

        # Look for synthesis thoughts
        print("\n💡 Synthesis Insights:")
        for thought in tree.thoughts.values():
            if thought.type == ThoughtType.SYNTHESIS or "pattern" in thought.text.lower():
                print(f"   - {thought.text}")

        await scratchpad.save_session("synthesis_demo")


async def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("HOFSTADTER SCRATCHPAD SYSTEM DEMO")
    print("="*70)
    print("\nPersistent internal dialogue with strange loops")
    print("Inspired by Douglas Hofstadter's 'Gödel, Escher, Bach'")

    await demo_basic_exploration()
    await asyncio.sleep(1)

    await demo_verification_mode()
    await asyncio.sleep(1)

    await demo_strange_loops()
    await asyncio.sleep(1)

    await demo_synthesis_mode()
    await asyncio.sleep(1)

    await demo_persistence()

    print("\n" + "="*70)
    print("✅ All Demos Complete")
    print("="*70)
    print("\nDatabase: demo_scratchpad.db")
    print("Sessions saved with full thought provenance")
    print("\nKey Features Demonstrated:")
    print("  ✓ Exploratory dialogue (recursive questioning)")
    print("  ✓ Verification mode (DS-STAR framework)")
    print("  ✓ Strange loops (Hofstadter-style level-crossing)")
    print("  ✓ Synthesis mode (pattern recognition)")
    print("  ✓ Session persistence (SQLite backend)")
    print("  ✓ Thought search (cross-session queries)")


if __name__ == "__main__":
    asyncio.run(main())

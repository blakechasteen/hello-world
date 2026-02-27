"""
Voice-First UX Demo

Demonstrates the unified voice interface with thread management and
conversational queries.

Features shown:
- Thread creation by voice
- Thread switching
- Conversational queries with thread context
- Thread listing
- Mode switching
- Status queries

Run:
    PYTHONPATH=. python HoloLoom/voice_first/demo.py

Or:
    python -m HoloLoom.voice.threads.demo
"""

import asyncio
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from HoloLoom.voice.threads import UnifiedVoiceAgent, VoiceMode


async def demo_basic_thread_management():
    """Demo 1: Basic thread creation and switching"""
    print("=" * 80)
    print("DEMO 1: Basic Thread Management")
    print("=" * 80)
    print()

    # Initialize agent
    print("Initializing UnifiedVoiceAgent...")
    agent = UnifiedVoiceAgent(initial_mode=VoiceMode.CONVERSATIONAL)
    await agent.initialize()
    print(f"✓ Agent initialized in {agent.current_mode} mode")
    print()

    # Demo conversation
    demos = [
        ("start a new thread for orchard planning", "Create first thread"),
        ("how should I space apple trees?", "Conversational query in thread"),
        ("new thread for biochar production", "Create second thread"),
        ("go back to orchard planning", "Switch back to first thread"),
        ("list my threads", "List all threads"),
    ]

    for query, label in demos:
        print(f"[{label}]")
        print(f"User: {query}")

        try:
            response = await agent.process(query)
            print(f"Elle: {response}")
        except Exception as e:
            print(f"Error: {e}")

        print("-" * 60)
        print()

    await agent.close()


async def demo_mode_switching():
    """Demo 2: Voice mode switching"""
    print("=" * 80)
    print("DEMO 2: Mode Switching")
    print("=" * 80)
    print()

    agent = UnifiedVoiceAgent()
    await agent.initialize()

    demos = [
        ("status", "Check initial status"),
        ("switch to command mode", "Switch to structured commands"),
        ("status", "Verify mode change"),
        ("switch to conversational mode", "Switch back"),
    ]

    for query, label in demos:
        print(f"[{label}]")
        print(f"User: {query}")

        response = await agent.process(query)
        print(f"Elle: {response}")

        print("-" * 60)
        print()

    await agent.close()


async def demo_help_and_examples():
    """Demo 3: Help system and command examples"""
    print("=" * 80)
    print("DEMO 3: Help & Examples")
    print("=" * 80)
    print()

    agent = UnifiedVoiceAgent()
    await agent.initialize()

    demos = [
        ("help", "Show available commands"),
        ("what can you do?", "Alternative help phrasing"),
    ]

    for query, label in demos:
        print(f"[{label}]")
        print(f"User: {query}")

        response = await agent.process(query)
        print(f"Elle: {response}")

        print("-" * 60)
        print()

    await agent.close()


async def demo_intent_classification():
    """Demo 4: Show intent classification in action"""
    print("=" * 80)
    print("DEMO 4: Intent Classification")
    print("=" * 80)
    print()

    from HoloLoom.voice.threads.grammar import VoiceGrammar

    grammar = VoiceGrammar()

    test_queries = [
        "start a new thread for planning",
        "switch to orchard planning",
        "list my threads",
        "what is Thompson Sampling?",
        "help",
        "switch to command mode",
    ]

    print("Classifying queries:\n")

    for query in test_queries:
        intent = grammar.classify(query)
        print(f"Query: '{query}'")
        print(f"  → Command Type: {intent.command_type.name}")
        print(f"  → Confidence: {intent.confidence:.2f}")
        print(f"  → Parameters: {intent.params}")
        print(f"  → Is Conversational: {intent.is_conversational}")
        print()


async def demo_conversational_with_context():
    """Demo 5: Conversational queries with thread context"""
    print("=" * 80)
    print("DEMO 5: Conversational Queries with Thread Context")
    print("=" * 80)
    print()

    agent = UnifiedVoiceAgent()
    await agent.initialize()

    # Create thread and add context
    print("Setting up thread with context...\n")

    demos = [
        ("start a new thread for permaculture", "Create thread"),
        ("what is the role of nitrogen-fixing plants?", "Query 1"),
        ("how does this relate to companion planting?", "Query 2 (with context)"),
        ("summarize this thread", "Get summary"),
    ]

    for query, label in demos:
        print(f"[{label}]")
        print(f"User: {query}")

        response = await agent.process(query)
        print(f"Elle: {response}")

        print("-" * 60)
        print()

    # Show status
    status = agent.get_status()
    print("Final Status:")
    print(f"  Mode: {status['mode']}")
    print(f"  Active Thread: {status.get('active_thread', 'None')}")
    print(f"  Total Threads: {status.get('thread_count', 0)}")
    print()

    await agent.close()


async def main():
    """Run all demos"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  VOICE-FIRST UX LAYER - DEMONSTRATION".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Unified voice interface combining:".center(78) + "║")
    print("║" + "  • HoloLoom (neural processing)".center(78) + "║")
    print("║" + "  • Elle (thread management)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    demos = [
        ("Basic Thread Management", demo_basic_thread_management),
        ("Mode Switching", demo_mode_switching),
        ("Help & Examples", demo_help_and_examples),
        ("Intent Classification", demo_intent_classification),
        ("Conversational with Context", demo_conversational_with_context),
    ]

    try:
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n{'─' * 80}")
            print(f"Running Demo {i}/{len(demos)}: {name}")
            print(f"{'─' * 80}\n")

            await demo_func()

            # Pause between demos
            if i < len(demos):
                input("\nPress Enter to continue to next demo...")
                print("\n")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        return

    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  DEMO COMPLETE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Voice-First UX Layer is ready!".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Next steps:".center(78) + "║")
    print("║" + "  1. Integrate with actual voice input (STT)".center(78) + "║")
    print("║" + "  2. Add voice output (TTS)".center(78) + "║")
    print("║" + "  3. Implement thread branching (Milestone 1)".center(78) + "║")
    print("║" + "  4. Add LightSpindle integration (Milestone 2)".center(78) + "║")
    print("║" + "  5. Build streaming mode (Milestone 3)".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")


if __name__ == "__main__":
    asyncio.run(main())

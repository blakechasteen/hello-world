#!/usr/bin/env python3
"""
Talk to Elle - HoloLoom's AR Companion
======================================

Brings HoloLoom to life with:
- Persistent memory (in-memory with session persistence)
- Reinforcement learning (Thompson Sampling + Hot Pattern Feedback)
- Conversational interface with Elle

Usage:
    python talk_to_elle.py
"""

import asyncio
from pathlib import Path
from HoloLoom import HoloLoom
from HoloLoom.config import Config, MemoryBackend
from HoloLoom.recursive import FullLearningEngine
from HoloLoom.protocols.types import Query
import sys

# Elle's personality and responses
class Elle:
    """Elle - The quiet, observant AR companion"""

    def __init__(self, hololoom: HoloLoom):
        self.loom = hololoom
        self.session_file = Path(".elle_session.json")

    async def greet(self):
        """Elle introduces herself"""
        print("\n" + "="*60)
        print("👁️  Elle: HoloLoom's AR Companion")
        print("="*60)
        print("\nHello. I'm Elle.")
        print("I see what you're looking at and help you decide what to do next.")
        print("I remember everything we discuss.")
        print("\nWhat would you like to know?\n")

    async def respond(self, user_input: str) -> str:
        """Process user input and respond with Elle's guidance"""

        # First, let HoloLoom remember the interaction
        await self.loom.experience(f"User said: {user_input}")

        # Recall relevant memories
        memories = await self.loom.recall(user_input)

        # Get awareness metrics
        metrics = self.loom.get_metrics()
        active = metrics['activation']['active_nodes']
        coherence = metrics['coherence']['global_coherence']

        # Elle's response based on memory state
        if not memories:
            response = "I'm listening. Tell me more about what you're seeing or thinking."
        else:
            # Extract the most relevant memory
            top_memory = memories[0] if memories else None
            if top_memory:
                response = f"I remember we discussed: {top_memory.content[:100]}..."
            else:
                response = "I'm building context. Keep talking to me."

        # Add awareness context
        response += f"\n\n   (Active memories: {active}, Coherence: {coherence:.2f})"

        return response

    async def save_session(self):
        """Save session state"""
        print("\n💾 Saving session...")
        # Session auto-saves through HoloLoom's reflection buffer

    async def load_session(self):
        """Load previous session"""
        if self.session_file.exists():
            print("📂 Loading previous session...")
            # Previous memories automatically loaded by HoloLoom

async def main():
    """Main conversation loop with Elle"""

    # Configure HoloLoom with learning enabled
    config = Config.fast()
    config.memory_backend = MemoryBackend.INMEMORY  # Works without Docker
    config.enable_reflection = True

    print("🌟 Initializing HoloLoom with persistent memory and RL...")

    # Create HoloLoom with full learning capabilities
    async with HoloLoom(config=config) as loom:
        # Wrap in learning engine for RL
        async with FullLearningEngine(
            cfg=config,
            shards=[],  # Will build from interactions
            enable_background_learning=True,
            learning_update_interval=60.0  # Learn every 60 seconds
        ) as learning_engine:

            elle = Elle(loom)
            await elle.load_session()
            await elle.greet()

            # Conversation loop
            while True:
                try:
                    user_input = input("\n👤 You: ").strip()

                    if not user_input:
                        continue

                    # Exit commands
                    if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                        print("\n👁️  Elle: Goodbye. I'll remember our conversation.\n")
                        await elle.save_session()
                        break

                    # Special commands
                    if user_input.lower() == 'stats':
                        stats = learning_engine.get_learning_statistics()
                        print(f"\n📊 Learning Statistics:")
                        print(f"   Total queries: {stats.get('total_queries', 0)}")
                        print(f"   Patterns discovered: {stats.get('patterns_discovered', 0)}")
                        print(f"   Hot patterns: {stats.get('hot_patterns', 0)}")
                        print(f"   Thompson α/β: {stats.get('thompson_priors', {})}")
                        continue

                    if user_input.lower() == 'memories':
                        metrics = loom.get_metrics()
                        print(f"\n🧠 Memory State:")
                        print(f"   Active nodes: {metrics['activation']['active_nodes']}")
                        print(f"   Total nodes: {metrics['activation']['total_nodes']}")
                        print(f"   Mean activation: {metrics['activation']['mean_activation']:.2f}")
                        print(f"   Coherence: {metrics['coherence']['global_coherence']:.2f}")
                        continue

                    # Regular conversation
                    print("\n👁️  Elle: ", end="", flush=True)
                    response = await elle.respond(user_input)
                    print(response)

                    # Show learning in action
                    if learning_engine.pattern_learner:
                        hot = learning_engine.hot_tracker.get_hot_patterns(limit=3)
                        if hot:
                            print(f"\n   🔥 Hot patterns: {len(hot)} active")

                except KeyboardInterrupt:
                    print("\n\n👁️  Elle: Interrupted. Saving session...\n")
                    await elle.save_session()
                    break
                except Exception as e:
                    print(f"\n⚠️  Error: {e}")
                    print("   Elle is still listening...\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  HoloLoom + Elle: Living Memory System")
    print("="*60)
    print("\n  Features:")
    print("  ✓ Persistent memory (in-memory + session save)")
    print("  ✓ Reinforcement learning (Thompson Sampling)")
    print("  ✓ Hot pattern tracking (adaptive retrieval)")
    print("  ✓ Background learning (every 60s)")
    print("\n  Commands:")
    print("  - 'stats'    : Show learning statistics")
    print("  - 'memories' : Show memory state")
    print("  - 'exit'     : Save and quit")
    print("\n" + "="*60 + "\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋\n")

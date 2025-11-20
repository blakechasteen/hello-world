"""
VoiceAgent Demo - Bidirectional Voice Interaction
==================================================

Demonstrates full voice interaction capabilities:
1. Simple voice agent (no HoloLoom)
2. HoloLoom-integrated voice agent
3. Live voice conversation with VAD
4. Conversation memory and context

Date: November 15, 2025
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HoloLoom.voice import VoiceAgent, OpenAITTS, TurnState

# Check for HoloLoom availability
try:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator
    from HoloLoom.config import Config
    from HoloLoom.documentation.types import MemoryShard
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("⚠️  HoloLoom not available - some demos will be limited")


# ============================================================================
# Demo 1: Simple Voice Agent (Echo Bot)
# ============================================================================

async def demo_1_simple_echo():
    """
    Demo 1: Simple voice agent that echoes input

    No HoloLoom required - just conversation memory and TTS
    """
    print("\n" + "="*70)
    print("DEMO 1: Simple Echo Bot")
    print("="*70)

    # Create agent without orchestrator
    agent = VoiceAgent(
        orchestrator=None,
        agent_name="EchoBot",
        turn_mode="button"  # Manual control
    )

    # Simulate conversation
    print("\nSimulating conversation...\n")

    queries = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me about yourself"
    ]

    for i, query in enumerate(queries, 1):
        print(f"Turn {i}:")
        print(f"  User: {query}")

        # Process input
        response = await agent.process_voice_input(query)
        print(f"  Bot: {response}")

        # In real scenario, would also:
        # await agent.speak(response)

        print()

    # Show conversation history
    print("Conversation History:")
    for turn in agent.conversation_memory.get_turns():
        speaker = "User" if turn.speaker == "user" else "Bot"
        print(f"  [{speaker}] {turn.text}")

    print("\n✅ Demo 1 complete!\n")


# ============================================================================
# Demo 2: HoloLoom-Integrated Voice Agent
# ============================================================================

async def demo_2_hololoom_integration():
    """
    Demo 2: Voice agent with full HoloLoom integration

    Uses WeavingOrchestrator for neural decision-making
    """
    print("\n" + "="*70)
    print("DEMO 2: HoloLoom-Integrated Voice Agent")
    print("="*70)

    if not HOLOLOOM_AVAILABLE:
        print("⚠️  HoloLoom not available - skipping demo")
        return

    # Create HoloLoom config and shards
    config = Config.fast()

    # Create simple memory shards
    shards = [
        MemoryShard(
            content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.",
            metadata={"topic": "thompson_sampling"}
        ),
        MemoryShard(
            content="Exploration-exploitation tradeoff balances trying new options vs. exploiting known good ones.",
            metadata={"topic": "exploration_exploitation"}
        )
    ]

    print("\nInitializing HoloLoom orchestrator...")

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Create voice agent
        agent = VoiceAgent(
            orchestrator=orchestrator,
            agent_name="Elle",
            voice="nova",
            turn_mode="button"
        )

        print("✅ Orchestrator ready!\n")
        print("Simulating voice conversation...\n")

        # Simulate conversation
        queries = [
            "What is Thompson Sampling?",
            "How does it handle the exploration-exploitation tradeoff?",
            "Can you give me a simple example?"
        ]

        for i, query in enumerate(queries, 1):
            print(f"Turn {i}:")
            print(f"  User: {query}")

            # Process through HoloLoom
            response = await agent.process_voice_input(query)
            print(f"  Elle: {response}")

            # Show confidence and tool used
            if agent.conversation_memory.turns:
                last_turn = agent.conversation_memory.turns[-1]
                if last_turn.confidence:
                    print(f"  [Confidence: {last_turn.confidence:.2f}]")
                if last_turn.tool_used:
                    print(f"  [Tool: {last_turn.tool_used}]")

            print()

        # Show conversation context
        print("Conversation Context (last 3 turns):")
        print(agent.conversation_memory.get_context(n_turns=3))

    print("\n✅ Demo 2 complete!\n")


# ============================================================================
# Demo 3: Conversation Memory and Context
# ============================================================================

async def demo_3_conversation_memory():
    """
    Demo 3: Conversation memory with context tracking

    Shows how conversation history is maintained
    """
    print("\n" + "="*70)
    print("DEMO 3: Conversation Memory & Context")
    print("="*70)

    agent = VoiceAgent(
        orchestrator=None,
        agent_name="MemoryBot"
    )

    print("\nBuilding conversation history...\n")

    # Build conversation
    conversation = [
        ("user", "My name is Blake"),
        ("agent", "Nice to meet you, Blake!"),
        ("user", "I'm working on a voice assistant"),
        ("agent", "That sounds interesting! Tell me more."),
        ("user", "It's called Elle and uses HoloLoom"),
        ("agent", "Elle with HoloLoom sounds powerful!"),
        ("user", "What was my name again?"),
        ("agent", "Your name is Blake, as you mentioned earlier.")
    ]

    for speaker, text in conversation:
        agent.conversation_memory.add_turn(speaker, text)
        prefix = "User" if speaker == "user" else "Bot"
        print(f"  [{prefix}] {text}")

    print("\nConversation Context (last 5 turns):")
    print("-" * 70)
    print(agent.conversation_memory.get_context(n_turns=5))
    print("-" * 70)

    # Export session
    session_data = agent.conversation_memory.export_session()
    print(f"\nSession Data:")
    print(f"  Session ID: {session_data['session_id']}")
    print(f"  Total turns: {session_data['total_turns']}")
    print(f"  Duration: {session_data['end_time'] - session_data['start_time']:.1f}s")

    print("\n✅ Demo 3 complete!\n")


# ============================================================================
# Demo 4: Turn-Taking Modes
# ============================================================================

async def demo_4_turn_taking():
    """
    Demo 4: Turn-taking modes (button/VAD/hybrid)

    Shows different conversation control modes
    """
    print("\n" + "="*70)
    print("DEMO 4: Turn-Taking Modes")
    print("="*70)

    modes = ["button", "hybrid"]  # VAD requires audio stream

    for mode in modes:
        print(f"\nMode: {mode}")
        print("-" * 70)

        agent = VoiceAgent(
            orchestrator=None,
            agent_name="TurnBot",
            turn_mode=mode
        )

        # Simulate turn states
        print("  State transitions:")

        # Idle → Listening (manual for button mode)
        agent.turn_manager.set_state(TurnState.LISTENING)
        print(f"    {TurnState.IDLE.value} → {TurnState.LISTENING.value}")

        # Listening → Processing
        agent.turn_manager.set_state(TurnState.PROCESSING)
        print(f"    {TurnState.LISTENING.value} → {TurnState.PROCESSING.value}")

        # Processing → Speaking
        agent.turn_manager.set_state(TurnState.SPEAKING)
        print(f"    {TurnState.PROCESSING.value} → {TurnState.SPEAKING.value}")

        # Speaking → Idle
        agent.turn_manager.set_state(TurnState.IDLE)
        print(f"    {TurnState.SPEAKING.value} → {TurnState.IDLE.value}")

        print(f"  Current state: {agent.turn_manager.state.value}")

    print("\n✅ Demo 4 complete!\n")


# ============================================================================
# Demo 5: TTS Integration
# ============================================================================

async def demo_5_tts_integration():
    """
    Demo 5: Text-to-Speech integration

    Demonstrates TTS synthesis (requires OpenAI API key)
    """
    print("\n" + "="*70)
    print("DEMO 5: TTS Integration")
    print("="*70)

    # Check for OpenAI API key
    import os
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("\n⚠️  OPENAI_API_KEY not set - skipping TTS demo")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        return

    print("\nInitializing TTS provider...")

    try:
        tts_provider = OpenAITTS(api_key=api_key)
        print("✅ OpenAI TTS ready!\n")

        # Test synthesis
        test_text = "Hello! I am Elle, a voice-enabled AI assistant powered by HoloLoom."

        print(f"Synthesizing: '{test_text}'")

        audio_bytes = await tts_provider.synthesize(test_text, voice="nova")

        print(f"✅ Synthesized {len(audio_bytes)} bytes of audio")
        print(f"   Voice: nova")
        print(f"   Model: tts-1")

        # Note: Actual playback would happen here in production
        print("\n   (Audio playback would happen here in production)")

    except Exception as e:
        print(f"❌ TTS synthesis failed: {e}")

    print("\n✅ Demo 5 complete!\n")


# ============================================================================
# Main Demo Runner
# ============================================================================

async def run_all_demos():
    """Run all demos in sequence"""
    print("\n" + "="*70)
    print("VOICE AGENT DEMONSTRATION SUITE")
    print("="*70)
    print("\nThis demo suite shows all VoiceAgent capabilities:")
    print("  1. Simple echo bot")
    print("  2. HoloLoom integration")
    print("  3. Conversation memory")
    print("  4. Turn-taking modes")
    print("  5. TTS integration")
    print("\nPress Enter to continue...")
    # input()  # Uncomment for interactive mode

    # Run demos
    await demo_1_simple_echo()
    await demo_2_hololoom_integration()
    await demo_3_conversation_memory()
    await demo_4_turn_taking()
    await demo_5_tts_integration()

    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Test with real microphone input")
    print("  2. Integrate with Elle AR assistant")
    print("  3. Add custom voice personalities")
    print("  4. Deploy to production")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    # Run all demos
    asyncio.run(run_all_demos())

    # Or run individual demos:
    # asyncio.run(demo_1_simple_echo())
    # asyncio.run(demo_2_hololoom_integration())
    # asyncio.run(demo_3_conversation_memory())
    # asyncio.run(demo_4_turn_taking())
    # asyncio.run(demo_5_tts_integration())

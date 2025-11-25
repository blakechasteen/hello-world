"""
Demo: Python-JavaScript Emotion Bridge Integration
===================================================

Demonstrates the emotion_bridge.py integration with HoloLoom's voice system.

This shows:
1. Basic emotion detection from text
2. Context-aware emotion analysis
3. Integration with VoiceAgent
4. Full 110/100 emotional intelligence pipeline

Date: November 2025
"""

import asyncio
import sys
import os

# Add HoloLoom to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from HoloLoom.voice.emotion_bridge import (
    EmotionBridge,
    EmotionBridgeConfig,
    enhance_voice_agent_with_emotions,
    FusionStrategy,
    MetaMode,
    PlanningStrategy
)


async def demo_basic_emotion_detection():
    """Demo 1: Basic text emotion detection"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Emotion Detection")
    print("="*70 + "\n")

    # Create bridge with standard config
    config = EmotionBridgeConfig.standard()

    async with EmotionBridge(config) as bridge:
        # Test cases
        test_inputs = [
            {
                'text': "I'm feeling frustrated with this bug. I've been stuck for hours!",
                'context': {'activity': 'coding', 'session_duration': 180000}
            },
            {
                'text': "This is amazing! I finally got it working!",
                'context': {'activity': 'coding', 'breakthrough': True}
            },
            {
                'text': "I don't know what to do. Everything feels overwhelming.",
                'context': {'activity': 'planning', 'complexity': 'high'}
            }
        ]

        for i, input_data in enumerate(test_inputs, 1):
            print(f"Test {i}: {input_data['text'][:50]}...")

            result = await bridge.analyze_emotion(**input_data)

            print(f"  ✓ Emotion: {result.emotion} (confidence: {result.confidence:.2%})")
            print(f"  ✓ Valence: {result.valence:+.2f} ({'positive' if result.valence > 0 else 'negative'})")
            print(f"  ✓ Arousal: {result.arousal:.2f}")

            if result.suggested_actions:
                print(f"  💡 Suggested: {result.suggested_actions[0]}")

            if result.conversation_strategy:
                print(f"  🎯 Strategy: {result.conversation_strategy}")

            print()


async def demo_configuration_presets():
    """Demo 2: Configuration presets (minimal, standard, advanced)"""
    print("\n" + "="*70)
    print("DEMO 2: Configuration Presets")
    print("="*70 + "\n")

    test_text = "I'm stuck on this problem and getting frustrated."
    context = {'activity': 'coding', 'session_duration': 120000}

    configs = {
        'Minimal (Fastest)': EmotionBridgeConfig.minimal(),
        'Standard (Balanced)': EmotionBridgeConfig.standard(),
        'Advanced (110/100)': EmotionBridgeConfig.advanced()
    }

    for name, config in configs.items():
        print(f"{name}:")

        async with EmotionBridge(config) as bridge:
            result = await bridge.analyze_emotion(text=test_text, context=context)

            print(f"  Emotion: {result.emotion} ({result.confidence:.2%})")

            if result.meta_interpretation:
                print(f"  Meta: {result.meta_interpretation[:60]}...")

            if result.nuances:
                print(f"  Nuances: {', '.join(result.nuances)}")

            if result.suggested_actions:
                print(f"  Actions: {len(result.suggested_actions)} suggestions")

            print(f"  Processing: {result.processing_time_ms:.1f}ms")
            print()


async def demo_voice_agent_integration():
    """Demo 3: Integration with VoiceAgent"""
    print("\n" + "="*70)
    print("DEMO 3: Voice Agent Integration")
    print("="*70 + "\n")

    try:
        from HoloLoom.voice import VoiceAgent
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.config import Config
        from HoloLoom.protocols.types import MemoryShard
    except ImportError as e:
        print(f"⚠️  Skipping - HoloLoom imports not available: {e}")
        return

    # Create minimal orchestrator
    config = Config.fast()
    shards = [
        MemoryShard(
            content="Thompson Sampling balances exploration and exploitation using Bayesian priors.",
            metadata={'topic': 'reinforcement_learning'}
        )
    ]

    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        # Create voice agent
        agent = VoiceAgent(
            orchestrator=orchestrator,
            agent_name="Elle",
            turn_mode="button"  # Manual control for demo
        )

        print("🎤 Created voice agent: Elle")
        print("🧠 Enhancing with emotional intelligence...\n")

        # Enhance with emotions!
        await enhance_voice_agent_with_emotions(
            agent,
            EmotionBridgeConfig.advanced()  # Full 110/100 features
        )

        print("✅ Voice agent enhanced with 110/100 emotional intelligence!\n")

        # Test conversation with emotional awareness
        test_queries = [
            "What is Thompson Sampling?",
            "I don't understand this. It's too complicated!",
            "Oh wait, I think I get it now! That makes sense!"
        ]

        for query in test_queries:
            print(f"User: {query}")

            response = await agent.process_voice_input(query)

            print(f"Elle: {response}")

            # Show emotional metadata
            if agent.conversation_memory.turns:
                last_turn = agent.conversation_memory.turns[-1]
                metadata = last_turn.metadata

                if 'emotion' in metadata:
                    print(f"  [Detected: {metadata['emotion']} "
                          f"({metadata.get('emotion_confidence', 0):.0%}), "
                          f"valence: {metadata.get('valence', 0):+.2f}]")

            print()


async def demo_multimodal_context():
    """Demo 4: Context-aware emotion detection"""
    print("\n" + "="*70)
    print("DEMO 4: Context-Aware Analysis")
    print("="*70 + "\n")

    async with EmotionBridge(EmotionBridgeConfig.advanced()) as bridge:
        # Same text, different contexts
        text = "I need help with this."

        contexts = [
            {'activity': 'coding', 'session_duration': 5000, 'confidence_level': 'low'},
            {'activity': 'learning', 'session_duration': 60000, 'progress': 'stuck'},
            {'activity': 'debugging', 'session_duration': 180000, 'attempts': 15}
        ]

        for i, context in enumerate(contexts, 1):
            print(f"Context {i}: {context}")

            result = await bridge.analyze_emotion(text=text, context=context)

            print(f"  Emotion: {result.emotion} ({result.confidence:.2%})")
            print(f"  Suggested: {result.suggested_actions[0] if result.suggested_actions else 'None'}")
            print()


async def demo_performance_comparison():
    """Demo 5: Performance across configurations"""
    print("\n" + "="*70)
    print("DEMO 5: Performance Comparison")
    print("="*70 + "\n")

    test_text = "I'm feeling great about this progress!"
    iterations = 5

    configs = {
        'Minimal': EmotionBridgeConfig.minimal(),
        'Standard': EmotionBridgeConfig.standard(),
        'Advanced': EmotionBridgeConfig.advanced()
    }

    print(f"Running {iterations} iterations per configuration...\n")

    for name, config in configs.items():
        async with EmotionBridge(config) as bridge:
            times = []

            for _ in range(iterations):
                result = await bridge.analyze_emotion(text=test_text)
                if result.processing_time_ms:
                    times.append(result.processing_time_ms)

            if times:
                avg_time = sum(times) / len(times)
                print(f"{name:12} | Avg: {avg_time:6.1f}ms | Features: ", end="")

                features = []
                if config.enable_meta_interpretation:
                    features.append("Meta")
                if config.enable_action_planning:
                    features.append("Planning")
                features.append(config.fusion_strategy.value)

                print(", ".join(features))


async def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🎯 EMOTION BRIDGE DEMO SUITE")
    print("="*70)
    print("\nIntegrates JavaScript 110/100 Emotional Intelligence with Python HoloLoom\n")

    demos = [
        ("Basic Emotion Detection", demo_basic_emotion_detection),
        ("Configuration Presets", demo_configuration_presets),
        ("Voice Agent Integration", demo_voice_agent_integration),
        ("Context-Aware Analysis", demo_multimodal_context),
        ("Performance Comparison", demo_performance_comparison)
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\n❌ Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("✅ DEMO SUITE COMPLETE!")
    print("="*70 + "\n")

    print("Next Steps:")
    print("  1. Start using emotion bridge in your voice applications")
    print("  2. Experiment with different configurations")
    print("  3. Integrate with HoloLoom weaving orchestrator")
    print("  4. Build domain-specific emotional intelligence (coding assistant, etc.)")
    print()


if __name__ == "__main__":
    asyncio.run(main())

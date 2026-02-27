"""
Demo 1: Elle Voice Query Integration
=====================================

Demonstrates basic voice query → response flow with AR context resolution.

This demo shows:
- Creating AR context with beehives
- Processing natural language voice queries
- Intent parsing and spatial reference resolution
- Multimodal response generation (voice + visual)
- Performance metrics

Run: python demos/demo_elle_voice_query.py

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import time
from hololoom.voice.ar_context import create_test_context, ARContext, Vector3
from hololoom.voice.elle_bridge import ElleBridge, ResponseMode
from hololoom.voice.command_router import CommandRouter


async def demo_voice_query():
    """Run voice query demonstration."""
    print("=" * 70)
    print("Demo 1: Elle Voice Query Integration")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Initialize AR Context
    # ========================================================================
    print("Step 1: Initialize AR Context with Beehives")
    print("-" * 70)

    context = create_test_context()

    print(f"  User Position: {context.user_position}")
    print(f"  User Orientation: {context.user_orientation}")
    print(f"  Gaze Target: {context.gaze_target}")
    print(f"  Active Scene: {context.active_scene}")
    print(f"  Visible Objects: {len(context.visible_objects)}")

    print("\n  Beehives in AR space:")
    for obj in context.find_objects_by_type(context.visible_objects[0].type):
        distance = obj.distance_from(context.user_position)
        print(f"    • {obj.id} ({obj.name}) at {obj.position} ({distance:.2f}m away)")

    print()

    # ========================================================================
    # Step 2: Initialize Elle Bridge and Command Router
    # ========================================================================
    print("Step 2: Initialize Elle Bridge and Command Router")
    print("-" * 70)

    # Create bridge without HoloLoom (for standalone demo)
    bridge = ElleBridge(
        orchestrator=None,
        enable_hololoom=False,
        enable_spatial_audio=True,
        default_response_mode=ResponseMode.MULTIMODAL_SPATIAL
    )

    router = CommandRouter()

    print(f"  Bridge initialized: {bridge}")
    print(f"  Response Mode: {bridge.default_response_mode.value}")
    print(f"  Spatial Audio Enabled: {bridge.enable_spatial_audio}")
    print()

    # ========================================================================
    # Step 3: Process Voice Queries
    # ========================================================================
    print("Step 3: Process Voice Queries")
    print("-" * 70)

    # Test queries with different spatial references
    test_queries = [
        "Show me the health status of this hive",
        "What is the population of hive 003?",
        "Tell me about the hive on my left",
        "This one looks interesting",
        "Why is that hive not performing well?",
    ]

    results = []

    for i, query_text in enumerate(test_queries, 1):
        print(f"\n  Query {i}: \"{query_text}\"")

        # Start timing
        query_start = time.time()

        # Step 3a: Parse intent
        print("    Parsing intent...")
        intent = await router.parse_intent(query_text, context)

        print(f"      Intent Type: {intent.type.value}")
        print(f"      Action: {intent.action}")
        print(f"      Target: {intent.target}")
        print(f"      Confidence: {intent.confidence:.1%}")
        if intent.parameters:
            print(f"      Parameters: {intent.parameters}")

        # Step 3b: Process with Elle Bridge
        print("    Processing voice command...")
        response = await bridge.process_voice_command(query_text, context)

        # Calculate query time
        query_time = (time.time() - query_start) * 1000  # Convert to ms

        # Print response
        print(f"      Voice Response: \"{response.voice_text}\"")
        if response.ar_visualization:
            print(f"      AR Visualization: {response.ar_visualization.type}")
            print(f"        Target: {response.ar_visualization.target_object}")
            print(f"        Duration: {response.ar_visualization.duration}s")
            print(f"        Content: {response.ar_visualization.content}")

        if response.spatial_audio_position:
            print(f"      Spatial Audio Position: {response.spatial_audio_position}")

        print(f"      Processing Time: {query_time:.1f}ms")
        print(f"      Response Mode: {response.response_mode.value}")

        results.append({
            "query": query_text,
            "intent_type": intent.type.value,
            "intent_confidence": intent.confidence,
            "response_mode": response.response_mode.value,
            "processing_time_ms": query_time,
            "has_visualization": response.ar_visualization is not None,
            "has_spatial_audio": response.spatial_audio_position is not None
        })

    print()

    # ========================================================================
    # Step 4: Spatial Reference Resolution Analysis
    # ========================================================================
    print("Step 4: Spatial Reference Resolution")
    print("-" * 70)

    print("\n  Nearby objects mapping (spatial references):")
    for ref, obj in context.nearby_objects.items():
        distance = obj.distance_from(context.user_position)
        print(f"    \"{ref}\" → {obj.id} ({obj.name}) at {distance:.2f}m")

    print("\n  Directional object finding:")
    directions = ["front", "left", "right"]
    for direction in directions:
        obj = context.find_object_in_direction(direction)
        if obj:
            print(f"    {direction.capitalize()}: {obj.id} ({obj.name})")
        else:
            print(f"    {direction.capitalize()}: None")

    print()

    # ========================================================================
    # Step 5: Performance Metrics
    # ========================================================================
    print("Step 5: Performance Metrics")
    print("-" * 70)

    stats = bridge.get_statistics()
    print(f"\n  Bridge Statistics:")
    print(f"    Total Queries: {stats['total_queries']}")
    print(f"    Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")
    print(f"    Total Processing Time: {stats['total_processing_time_ms']:.1f}ms")
    print(f"    Recent Intents: {stats['recent_intents']}")

    # Calculate summary statistics
    if results:
        avg_time = sum(r['processing_time_ms'] for r in results) / len(results)
        viz_count = sum(1 for r in results if r['has_visualization'])
        audio_count = sum(1 for r in results if r['has_spatial_audio'])

        print(f"\n  Query Summary:")
        print(f"    Total Queries: {len(results)}")
        print(f"    Avg Query Time: {avg_time:.1f}ms")
        print(f"    Queries with AR Visualization: {viz_count}/{len(results)}")
        print(f"    Queries with Spatial Audio: {audio_count}/{len(results)}")

        # Confidence distribution
        confidences = [r['intent_confidence'] for r in results]
        print(f"\n  Intent Confidence Distribution:")
        print(f"    Min: {min(confidences):.1%}")
        print(f"    Max: {max(confidences):.1%}")
        print(f"    Avg: {sum(confidences)/len(confidences):.1%}")

    print()

    # ========================================================================
    # Step 6: Intent Distribution
    # ========================================================================
    print("Step 6: Intent Type Distribution")
    print("-" * 70)

    intent_types = {}
    for result in results:
        intent_type = result['intent_type']
        intent_types[intent_type] = intent_types.get(intent_type, 0) + 1

    for intent_type, count in sorted(intent_types.items()):
        percentage = (count / len(results)) * 100
        print(f"  {intent_type.capitalize()}: {count} ({percentage:.0f}%)")

    print()

    # ========================================================================
    # Completion
    # ========================================================================
    print("=" * 70)
    print("✅ Voice Query Demo Complete!")
    print("=" * 70)
    print()

    return {
        "context": context,
        "bridge": bridge,
        "results": results,
        "stats": stats
    }


if __name__ == "__main__":
    asyncio.run(demo_voice_query())

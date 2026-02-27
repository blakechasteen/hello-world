"""
Demo 3: Elle Multimodal Response Modes
=======================================

Demonstrates all 5 response modes for multimodal AR interactions.

This demo shows:
- VOICE_ONLY: Spoken response only
- VISUAL_ONLY: AR visualization without voice
- MULTIMODAL: Voice + visual combined
- SPATIAL_AUDIO: 3D positioned voice response
- MULTIMODAL_SPATIAL: Complete multimodal with 3D audio

For each mode, processes the same query and shows response components.

Run: python demos/demo_elle_multimodal.py

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import time
from hololoom.voice.ar_context import create_test_context
from hololoom.voice.elle_bridge import ElleBridge, ResponseMode
from hololoom.voice.command_router import CommandRouter


async def demo_multimodal():
    """Run multimodal response demonstration."""
    print("=" * 70)
    print("Demo 3: Elle Multimodal Response Modes")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Initialize AR Context
    # ========================================================================
    print("Step 1: Initialize AR Context")
    print("-" * 70)

    context = create_test_context()

    print(f"  User Position: {context.user_position}")
    print(f"  Visible Objects: {len(context.visible_objects)}")
    print(f"  Gaze Target: {context.gaze_target}")
    print()

    # ========================================================================
    # Step 2: Initialize Bridge
    # ========================================================================
    print("Step 2: Initialize Elle Bridge")
    print("-" * 70)

    bridge = ElleBridge(
        orchestrator=None,
        enable_hololoom=False,
        enable_spatial_audio=True,
        default_response_mode=ResponseMode.MULTIMODAL_SPATIAL
    )

    router = CommandRouter()

    print(f"  Bridge initialized")
    print(f"  Spatial Audio Enabled: {bridge.enable_spatial_audio}")
    print()

    # ========================================================================
    # Step 3: Test Query
    # ========================================================================
    test_query = "Show me the health status of this hive"

    print("Step 3: Query for Response Mode Testing")
    print("-" * 70)
    print(f"  Query: \"{test_query}\"")
    print()

    # ========================================================================
    # Step 4: Process Query in All 5 Response Modes
    # ========================================================================
    print("Step 4: Process Query in All 5 Response Modes")
    print("-" * 70)

    response_modes = [
        ResponseMode.VOICE_ONLY,
        ResponseMode.VISUAL_ONLY,
        ResponseMode.MULTIMODAL,
        ResponseMode.SPATIAL_AUDIO,
        ResponseMode.MULTIMODAL_SPATIAL
    ]

    mode_results = []

    for mode in response_modes:
        print(f"\n  ┌─ Response Mode: {mode.value.upper()}")
        print(f"  │")

        mode_start = time.time()

        # Process with specific response mode
        response = await bridge.process_voice_command(
            test_query,
            context,
            response_mode=mode
        )

        mode_time = (time.time() - mode_start) * 1000

        # ====================================================================
        # Analyze Response Components
        # ====================================================================

        print(f"  │  Voice Component:")
        if response.voice_text:
            voice_text = response.voice_text
            if len(voice_text) > 60:
                voice_text = voice_text[:57] + "..."
            print(f"  │    ✓ Voice Text: \"{voice_text}\"")
            print(f"  │    ✓ Length: {len(response.voice_text)} characters")
        else:
            print(f"  │    ✗ No voice component")

        print(f"  │")
        print(f"  │  Visual Component:")
        if response.ar_visualization:
            viz = response.ar_visualization
            print(f"  │    ✓ Visualization Type: {viz.type}")
            print(f"  │    ✓ Target Object: {viz.target_object}")
            print(f"  │    ✓ Duration: {viz.duration}s")
            if viz.content:
                print(f"  │    ✓ Content Keys: {list(viz.content.keys())}")
            if viz.style:
                print(f"  │    ✓ Style Keys: {list(viz.style.keys())}")
        else:
            print(f"  │    ✗ No visual component")

        print(f"  │")
        print(f"  │  Spatial Audio Component:")
        if response.spatial_audio_position:
            pos = response.spatial_audio_position
            print(f"  │    ✓ Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

            # Calculate distance from user
            user_pos = context.user_position
            distance = ((pos[0] - user_pos.x)**2 +
                       (pos[1] - user_pos.y)**2 +
                       (pos[2] - user_pos.z)**2) ** 0.5
            print(f"  │    ✓ Distance from user: {distance:.2f}m")

            # Describe direction
            if abs(pos[2] - user_pos.z) > 1:
                direction = "front" if pos[2] > user_pos.z else "back"
            else:
                direction = "side"
            print(f"  │    ✓ Direction: {direction}")
        else:
            print(f"  │    ✗ No spatial audio")

        print(f"  │")
        print(f"  │  Metadata:")
        print(f"  │    • Processing Time: {mode_time:.1f}ms")
        print(f"  │    • Intent Type: {response.metadata.get('intent_type', 'unknown')}")
        print(f"  │    • Intent Confidence: {response.metadata.get('intent_confidence', 0):.1%}")
        print(f"  │    • Response Mode: {response.response_mode.value}")

        print(f"  │")
        print(f"  └─ Mode Processing Complete")

        mode_results.append({
            "mode": mode.value,
            "has_voice": bool(response.voice_text),
            "has_visual": response.ar_visualization is not None,
            "has_spatial_audio": response.spatial_audio_position is not None,
            "processing_time_ms": mode_time,
            "intent_confidence": response.metadata.get('intent_confidence', 0)
        })

    print()

    # ========================================================================
    # Step 5: Response Mode Comparison
    # ========================================================================
    print("Step 5: Response Mode Comparison Matrix")
    print("-" * 70)

    print("\n  Mode                      Voice   Visual   Spatial   Time (ms)")
    print("  " + "-" * 60)

    for result in mode_results:
        mode_name = result['mode'].replace('_', ' ').title()
        mode_name = f"{mode_name:<23}"

        voice_mark = "✓" if result['has_voice'] else "✗"
        visual_mark = "✓" if result['has_visual'] else "✗"
        audio_mark = "✓" if result['has_spatial_audio'] else "✗"
        time_val = f"{result['processing_time_ms']:>6.1f}"

        print(f"  {mode_name}  {voice_mark}       {visual_mark}       {audio_mark}         {time_val}")

    print()

    # ========================================================================
    # Step 6: Component Distribution Analysis
    # ========================================================================
    print("Step 6: Multimodal Component Analysis")
    print("-" * 70)

    voice_count = sum(1 for r in mode_results if r['has_voice'])
    visual_count = sum(1 for r in mode_results if r['has_visual'])
    spatial_count = sum(1 for r in mode_results if r['has_spatial_audio'])
    total_modes = len(mode_results)

    print(f"\n  Component Usage Distribution:")
    print(f"    Voice Component:        {voice_count}/{total_modes} modes ({voice_count/total_modes*100:.0f}%)")
    print(f"    Visual Component:       {visual_count}/{total_modes} modes ({visual_count/total_modes*100:.0f}%)")
    print(f"    Spatial Audio:          {spatial_count}/{total_modes} modes ({spatial_count/total_modes*100:.0f}%)")

    # Find richest and simplest modes
    mode_complexity = {}
    for result in mode_results:
        components = sum([
            result['has_voice'],
            result['has_visual'],
            result['has_spatial_audio']
        ])
        mode_complexity[result['mode']] = components

    richest = max(mode_complexity, key=mode_complexity.get)
    simplest = min(mode_complexity, key=mode_complexity.get)

    print(f"\n  Complexity Ranking:")
    print(f"    Most Complex: {richest} ({mode_complexity[richest]} components)")
    print(f"    Simplest: {simplest} ({mode_complexity[simplest]} component(s))")

    print()

    # ========================================================================
    # Step 7: Performance Analysis
    # ========================================================================
    print("Step 7: Performance Analysis")
    print("-" * 70)

    times = [r['processing_time_ms'] for r in mode_results]
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print(f"\n  Processing Time Statistics:")
    print(f"    Minimum:        {min_time:.1f}ms (VOICE_ONLY)")
    print(f"    Maximum:        {max_time:.1f}ms (MULTIMODAL_SPATIAL)")
    print(f"    Average:        {avg_time:.1f}ms")
    print(f"    Time Increase:  {(max_time - min_time):.1f}ms ({((max_time/min_time - 1)*100):.0f}%)")

    print()

    # ========================================================================
    # Step 8: Use Case Recommendations
    # ========================================================================
    print("Step 8: Response Mode Recommendations")
    print("-" * 70)

    recommendations = {
        "VOICE_ONLY": [
            "• Simple factual queries",
            "• Hands-busy scenarios",
            "• Audio-focused environments",
            "• Performance-critical cases",
        ],
        "VISUAL_ONLY": [
            "• Sound-off environments",
            "• Visual learning preferences",
            "• Demonstration/tutorial mode",
            "• Accessibility for hearing-impaired",
        ],
        "MULTIMODAL": [
            "• Most queries (balanced mode)",
            "• Complete information delivery",
            "• Standard AR interactions",
            "• Best overall user experience",
        ],
        "SPATIAL_AUDIO": [
            "• Immersive experiences",
            "• Audio cues from objects",
            "• 3D spatial awareness",
            "• Gaming/entertainment AR",
        ],
        "MULTIMODAL_SPATIAL": [
            "• Full immersive AR",
            "• Professional applications",
            "• Complex information delivery",
            "• Maximum engagement",
        ]
    }

    for mode in response_modes:
        mode_name = mode.value.upper()
        print(f"\n  {mode_name}:")
        if mode_name in recommendations:
            for rec in recommendations[mode_name]:
                print(f"    {rec}")

    print()

    # ========================================================================
    # Completion
    # ========================================================================
    print("=" * 70)
    print("✅ Multimodal Response Demo Complete!")
    print("=" * 70)
    print()

    return {
        "context": context,
        "bridge": bridge,
        "mode_results": mode_results,
        "recommendations": recommendations
    }


if __name__ == "__main__":
    asyncio.run(demo_multimodal())

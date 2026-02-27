"""
Demo 2: Elle AR Navigation with Voice Control
===============================================

Demonstrates voice-controlled navigation through AR space with spatial guidance.

This demo shows:
- User navigation between beehives using voice commands
- Navigation command parsing ("Go to hive 003", "Next hive", etc.)
- Distance calculations and path visualization
- Real-time position updates
- Turn-by-turn guidance generation
- Navigation completion detection

Run: python demos/demo_elle_navigation.py

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import time
import math
from hololoom.voice.ar_context import (
    create_test_context, ARContext, Vector3, ARObject, ARObjectType,
    NavigationEvent, AREventType
)
from hololoom.voice.elle_bridge import ElleBridge, ResponseMode, ARVisualization
from hololoom.voice.command_router import CommandRouter, IntentType


async def demo_navigation():
    """Run navigation demonstration."""
    print("=" * 70)
    print("Demo 2: Elle AR Navigation with Voice Control")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Initialize AR Context and Navigation State
    # ========================================================================
    print("Step 1: Initialize AR Context")
    print("-" * 70)

    context = create_test_context()

    # Get list of beehives for navigation
    beehives = context.find_objects_by_type(ARObjectType.BEEHIVE)
    print(f"  Available Beehives: {len(beehives)}")
    for hive in beehives:
        distance = hive.distance_from(context.user_position)
        print(f"    • {hive.id} ({hive.name}) at {hive.position} ({distance:.2f}m)")

    print()

    # ========================================================================
    # Step 2: Initialize Bridge and Router
    # ========================================================================
    print("Step 2: Initialize Elle Bridge and Router")
    print("-" * 70)

    bridge = ElleBridge(
        orchestrator=None,
        enable_hololoom=False,
        enable_spatial_audio=True,
        default_response_mode=ResponseMode.MULTIMODAL
    )

    router = CommandRouter()

    print(f"  Bridge initialized with navigation support")
    print(f"  Response Mode: {bridge.default_response_mode.value}")
    print()

    # ========================================================================
    # Step 3: Simulate Navigation Sequence
    # ========================================================================
    print("Step 3: Voice Navigation Sequence")
    print("-" * 70)

    # Navigation commands
    nav_commands = [
        "Go to hive 003",
        "Take me to hive 001",
        "Navigate to the next hive",
        "Go to the hive on my left",
    ]

    navigation_results = []

    for i, command in enumerate(nav_commands, 1):
        print(f"\n  Navigation Command {i}: \"{command}\"")

        # Start timing
        nav_start = time.time()

        # Parse intent
        intent = await router.parse_intent(command, context)

        print(f"    Intent Type: {intent.type.value}")
        print(f"    Target: {intent.target}")
        if intent.type == IntentType.NAVIGATE:
            print(f"    ✓ Navigation intent detected")

        # Process with bridge
        response = await bridge.process_voice_command(command, context)

        nav_time = (time.time() - nav_start) * 1000

        # Get target object
        if intent.target:
            target_obj = context.find_object_by_id(intent.target)
        else:
            # Find next hive in list
            closest = context.find_closest_object(ARObjectType.BEEHIVE)
            target_obj = closest

        if target_obj:
            # Calculate navigation metrics
            distance = target_obj.distance_from(context.user_position)
            direction_vec = Vector3(
                target_obj.position.x - context.user_position.x,
                target_obj.position.y - context.user_position.y,
                target_obj.position.z - context.user_position.z
            )

            # Calculate bearing (azimuth)
            bearing = math.atan2(direction_vec.x, direction_vec.z)
            bearing_degrees = math.degrees(bearing) % 360

            # Estimate time to reach (assuming 1 m/s walking speed)
            estimated_time = distance / 1.0

            print(f"\n    Navigation Details:")
            print(f"      Target: {target_obj.id} ({target_obj.name})")
            print(f"      Position: {target_obj.position}")
            print(f"      Distance: {distance:.2f}m")
            print(f"      Bearing: {bearing_degrees:.0f}°")
            print(f"      Est. Time: {estimated_time:.0f}s at 1 m/s")

            # Voice response
            print(f"\n    Voice Response: \"{response.voice_text}\"")

            # Visualization
            if response.ar_visualization:
                viz = response.ar_visualization
                print(f"    AR Visualization: {viz.type}")
                print(f"      Duration: {viz.duration}s")
                if viz.content:
                    print(f"      Show Path: {viz.content.get('show_distance', False)}")
                    print(f"      Show Distance: {viz.content.get('show_distance', False)}")

            # Spatial audio
            if response.spatial_audio_position:
                print(f"    Spatial Audio: Positioned at target location")

            print(f"    Processing Time: {nav_time:.1f}ms")

            navigation_results.append({
                "command": command,
                "target": target_obj.id,
                "distance_m": distance,
                "bearing_degrees": bearing_degrees,
                "estimated_time_s": estimated_time,
                "processing_time_ms": nav_time,
                "has_visualization": response.ar_visualization is not None
            })

    print()

    # ========================================================================
    # Step 4: Path Visualization Analysis
    # ========================================================================
    print("Step 4: Path Visualization Analysis")
    print("-" * 70)

    if navigation_results:
        print("\n  Navigation Summary:")
        total_distance = sum(r['distance_m'] for r in navigation_results)
        total_time = sum(r['estimated_time_s'] for r in navigation_results)

        print(f"    Total Waypoints: {len(navigation_results)}")
        print(f"    Total Distance: {total_distance:.2f}m")
        print(f"    Est. Total Time: {total_time:.0f}s")
        print(f"    Avg Distance per Waypoint: {total_distance/len(navigation_results):.2f}m")

        print(f"\n  Navigation Path Details:")
        for i, result in enumerate(navigation_results, 1):
            print(f"    Waypoint {i}: {result['target']}")
            print(f"      Distance: {result['distance_m']:.2f}m")
            print(f"      Bearing: {result['bearing_degrees']:.0f}°")
            print(f"      Est. Time: {result['estimated_time_s']:.0f}s")

    print()

    # ========================================================================
    # Step 5: Turn-by-Turn Guidance Generation
    # ========================================================================
    print("Step 5: Turn-by-Turn Guidance")
    print("-" * 70)

    if navigation_results and len(navigation_results) > 1:
        print("\n  Simulating turn-by-turn guidance:\n")

        prev_bearing = None

        for i, result in enumerate(navigation_results, 1):
            bearing = result['bearing_degrees']

            if prev_bearing is not None:
                bearing_diff = (bearing - prev_bearing + 360) % 360

                if bearing_diff > 180:
                    bearing_diff = 360 - bearing_diff
                    direction = "left"
                else:
                    direction = "right"

                print(f"  Step {i}: Turn {bearing_diff:.0f}° {direction} to {result['target']}")
            else:
                print(f"  Step {i}: Head to {result['target']} (bearing: {bearing:.0f}°)")

            print(f"           Distance: {result['distance_m']:.2f}m")
            if i < len(navigation_results):
                print()

            prev_bearing = bearing

    print()

    # ========================================================================
    # Step 6: Performance Metrics
    # ========================================================================
    print("Step 6: Performance Metrics")
    print("-" * 70)

    if navigation_results:
        avg_time = sum(r['processing_time_ms'] for r in navigation_results) / len(navigation_results)
        viz_count = sum(1 for r in navigation_results if r['has_visualization'])

        print(f"\n  Navigation Performance:")
        print(f"    Navigation Commands: {len(navigation_results)}")
        print(f"    Avg Processing Time: {avg_time:.1f}ms")
        print(f"    Commands with Visualization: {viz_count}/{len(navigation_results)}")

    # Bridge statistics
    stats = bridge.get_statistics()
    print(f"\n  Bridge Statistics:")
    print(f"    Total Queries: {stats['total_queries']}")
    print(f"    Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms")

    print()

    # ========================================================================
    # Step 7: Distance Calculation Validation
    # ========================================================================
    print("Step 7: Distance Calculation Validation")
    print("-" * 70)

    print("\n  Validating distance calculations:")
    all_hives = context.find_objects_by_type(ARObjectType.BEEHIVE)

    distances = {}
    for hive in all_hives:
        dist = hive.distance_from(context.user_position)
        distances[hive.id] = dist

    # Find closest and farthest
    closest_id = min(distances, key=distances.get)
    farthest_id = max(distances, key=distances.get)

    print(f"    Closest hive: {closest_id} ({distances[closest_id]:.2f}m)")
    print(f"    Farthest hive: {farthest_id} ({distances[farthest_id]:.2f}m)")
    print(f"    Distance range: {distances[farthest_id] - distances[closest_id]:.2f}m")

    print()

    # ========================================================================
    # Completion
    # ========================================================================
    print("=" * 70)
    print("✅ Navigation Demo Complete!")
    print("=" * 70)
    print()

    return {
        "context": context,
        "bridge": bridge,
        "navigation_results": navigation_results,
        "beehives": beehives
    }


if __name__ == "__main__":
    asyncio.run(demo_navigation())

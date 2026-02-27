"""
Demo 4: Elle Spatial Audio Positioning
=======================================

Demonstrates 3D spatial audio positioning with HRTF processing.

This demo shows:
- Positioning audio sources at different 3D locations
- Calculating spatial parameters (azimuth, elevation, distance)
- ITD/ILD values for stereo positioning
- Distance-based volume attenuation
- HRTF processing for immersive 3D audio
- Interactive spatial audio preview

Run: python demos/demo_elle_spatial_audio.py

Author: HoloLoom Team
Date: November 2025
"""

import asyncio
import math
from hololoom.voice.ar_context import create_test_context, Vector3
from hololoom.voice.spatial_audio import SpatialAudioHandler, SpatialAudioConfig
from hololoom.voice.elle_bridge import ElleBridge


async def demo_spatial_audio():
    """Run spatial audio demonstration."""
    print("=" * 70)
    print("Demo 4: Elle Spatial Audio Positioning")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Initialize AR Context and Audio Handler
    # ========================================================================
    print("Step 1: Initialize AR Context and Audio Handler")
    print("-" * 70)

    context = create_test_context()
    print(f"  User Position: {context.user_position}")
    print(f"  User Orientation: {context.user_orientation}")

    # Create spatial audio handler with configuration
    audio_config = SpatialAudioConfig(
        sample_rate=16000,
        enable_hrtf=True,
        enable_distance_attenuation=True,
        max_distance=50.0,
        reference_distance=1.0,
        rolloff_factor=1.0
    )

    handler = SpatialAudioHandler(config=audio_config)

    print(f"\n  Audio Configuration:")
    print(f"    Sample Rate: {audio_config.sample_rate} Hz")
    print(f"    HRTF Enabled: {audio_config.enable_hrtf}")
    print(f"    Distance Attenuation: {audio_config.enable_distance_attenuation}")
    print(f"    Max Distance: {audio_config.max_distance}m")
    print(f"    Reference Distance: {audio_config.reference_distance}m")
    print()

    # ========================================================================
    # Step 2: Define Test Audio Positions
    # ========================================================================
    print("Step 2: Define Test Audio Positions")
    print("-" * 70)

    test_positions = [
        (Vector3(0, 1.7, 2), "Center Front (Close)", "center-front"),
        (Vector3(3, 1.7, 0), "Right Side", "right"),
        (Vector3(-3, 1.7, 0), "Left Side", "left"),
        (Vector3(0, 1.7, -3), "Behind", "behind"),
        (Vector3(0, 3.5, 2), "Above Front", "above"),
        (Vector3(5, 1.7, 5), "Far Front-Right", "far"),
        (Vector3(0, 0.5, 2), "Below Front", "below"),
    ]

    print(f"  Test Positions: {len(test_positions)}")
    for pos, description, _ in test_positions:
        distance = pos.distance_to(context.user_position)
        print(f"    • {description}: {pos} ({distance:.2f}m away)")

    print()

    # ========================================================================
    # Step 3: Calculate Spatial Parameters for Each Position
    # ========================================================================
    print("Step 3: Spatial Parameter Calculation")
    print("-" * 70)

    spatial_results = []

    for position, description, category in test_positions:
        print(f"\n  ┌─ {description.upper()}")
        print(f"  │")

        # Get spatial parameters
        params = handler.get_spatial_preview(position, context)

        print(f"  │  Position: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
        print(f"  │")
        print(f"  │  Distance Metrics:")
        print(f"  │    Distance: {params['distance']:.2f}m")
        print(f"  │    Attenuation: {params['attenuation']:.1%}")
        print(f"  │    Attenuation dB: {20 * math.log10(params['attenuation']) if params['attenuation'] > 0 else '-∞:.1f'} dB")

        print(f"  │")
        print(f"  │  Directional Parameters:")
        print(f"  │    Azimuth: {params['azimuth_degrees']:.1f}°")
        print(f"  │    Elevation: {params['elevation_degrees']:.1f}°")
        print(f"  │    Perceived Position: {params['position_description']}")

        # Calculate ITD and ILD
        max_itd_samples = int(0.00065 * audio_config.sample_rate)
        itd_samples = int(max_itd_samples * math.sin(params['azimuth']))
        itd_ms = (itd_samples / audio_config.sample_rate) * 1000

        max_ild_db = 20.0
        ild_db = max_ild_db * math.sin(params['azimuth'])

        print(f"  │")
        print(f"  │  HRTF Parameters:")
        print(f"  │    ITD (Interaural Time Diff): {itd_ms:.2f}ms ({itd_samples} samples)")
        print(f"  │    ILD (Interaural Level Diff): {ild_db:+.1f} dB")

        # Calculate stereo gains
        left_gain = 10 ** (-ild_db / 20.0)
        right_gain = 10 ** (ild_db / 20.0)

        print(f"  │    Left Gain: {left_gain:.2f}x ({20*math.log10(left_gain):+.1f} dB)")
        print(f"  │    Right Gain: {right_gain:.2f}x ({20*math.log10(right_gain):+.1f} dB)")

        print(f"  │")
        print(f"  └─ Parameters Calculated")

        spatial_results.append({
            "description": description,
            "category": category,
            "position": position,
            "distance": params['distance'],
            "attenuation": params['attenuation'],
            "azimuth": params['azimuth'],
            "azimuth_degrees": params['azimuth_degrees'],
            "elevation": params['elevation'],
            "elevation_degrees": params['elevation_degrees'],
            "itd_ms": itd_ms,
            "ild_db": ild_db,
            "left_gain": left_gain,
            "right_gain": right_gain,
            "position_description": params['position_description']
        })

    print()

    # ========================================================================
    # Step 4: HRTF Processing Analysis
    # ========================================================================
    print("Step 4: HRTF Processing Analysis")
    print("-" * 70)

    print("\n  Spatial Audio Positioning Techniques:")
    print()
    print("  1. Azimuth (Horizontal Angle)")
    print("     • Range: -180° to +180°")
    print("     • Positive: Sound from right")
    print("     • Negative: Sound from left")
    print("     • 0°: Sound from center front")
    print()
    print("  2. Elevation (Vertical Angle)")
    print("     • Range: -90° to +90°")
    print("     • Positive: Sound from above")
    print("     • Negative: Sound from below")
    print("     • 0°: Sound at ear level")
    print()
    print("  3. Distance Attenuation")
    print("     • Uses inverse square law")
    print("     • Formula: attenuation = (ref_dist / distance)^rolloff")
    print("     • Default: Linear rolloff (1.0)")
    print()
    print("  4. Interaural Time Difference (ITD)")
    print("     • Time delay between ears")
    print("     • Max ITD: ~0.65ms (human head size)")
    print("     • Brain uses for localization <1kHz")
    print()
    print("  5. Interaural Level Difference (ILD)")
    print("     • Volume difference between ears")
    print("     • Max ILD: ~20dB")
    print("     • Brain uses for localization >1kHz")

    print()

    # ========================================================================
    # Step 5: Audio Processing Demonstration
    # ========================================================================
    print("Step 5: Audio Processing Demonstration")
    print("-" * 70)

    bridge = ElleBridge(
        orchestrator=None,
        enable_hololoom=False,
        enable_spatial_audio=True
    )

    # Test voice command that generates spatial audio
    test_query = "Show me information about this hive"

    print(f"\n  Query: \"{test_query}\"")
    print(f"  Processing with spatial audio positioning...")
    print()

    response = await bridge.process_voice_command(test_query, context)

    print(f"  Response Generated:")
    print(f"    Voice Text: \"{response.voice_text}\"")

    if response.spatial_audio_position:
        pos = response.spatial_audio_position
        print(f"    Spatial Position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        # Calculate spatial parameters for response position
        response_pos = Vector3(pos[0], pos[1], pos[2])
        response_params = handler.get_spatial_preview(response_pos, context)

        print(f"\n  Spatial Audio Parameters for Response:")
        print(f"    Azimuth: {response_params['azimuth_degrees']:.1f}°")
        print(f"    Elevation: {response_params['elevation_degrees']:.1f}°")
        print(f"    Distance: {response_params['distance']:.2f}m")
        print(f"    Attenuation: {response_params['attenuation']:.1%}")
        print(f"    Position: {response_params['position_description']}")

    print()

    # ========================================================================
    # Step 6: Stereo Field Visualization
    # ========================================================================
    print("Step 6: Stereo Field Visualization")
    print("-" * 70)

    print("\n  Azimuth Distribution of Test Positions:")
    print()
    print("     LEFT ←                 CENTER                    → RIGHT")
    print("  -180°  -90°               0°                         +90°  +180°")
    print("    ├────┼─────────────────┼─────────────────────┼────┤")

    # Create a simple ASCII visualization
    viz_positions = {}
    for result in spatial_results:
        az_degrees = result['azimuth_degrees']
        # Map -180..180 to 0..40 character positions
        char_pos = int((az_degrees + 180) / 360 * 40)
        char_pos = max(0, min(40, char_pos))

        if char_pos not in viz_positions:
            viz_positions[char_pos] = []
        viz_positions[char_pos].append(result['description'][:20])

    # Print visualization
    viz_line = [' '] * 42
    for pos, descriptions in viz_positions.items():
        viz_line[pos + 1] = '●'

    print("    │" + ''.join(viz_line) + "│")
    print()

    # Print position labels
    for pos, descriptions in sorted(viz_positions.items()):
        az_degrees = (pos / 40) * 360 - 180
        desc = descriptions[0]
        print(f"      {az_degrees:>6.0f}°: {desc}")

    print()

    # ========================================================================
    # Step 7: Performance Metrics
    # ========================================================================
    print("Step 7: Performance Metrics")
    print("-" * 70)

    print("\n  Spatial Parameter Calculation Performance:")
    print(f"    Test Positions: {len(spatial_results)}")
    print(f"    Parameters per Position: 12")
    print(f"    Total Parameters: {len(spatial_results) * 12}")

    # Analyze distance distribution
    distances = [r['distance'] for r in spatial_results]
    attenuations = [r['attenuation'] for r in spatial_results]

    print(f"\n  Distance Statistics:")
    print(f"    Min: {min(distances):.2f}m")
    print(f"    Max: {max(distances):.2f}m")
    print(f"    Avg: {sum(distances)/len(distances):.2f}m")

    print(f"\n  Attenuation Statistics:")
    print(f"    Min: {min(attenuations):.1%}")
    print(f"    Max: {max(attenuations):.1%}")
    print(f"    Avg: {sum(attenuations)/len(attenuations):.1%}")

    print()

    # ========================================================================
    # Step 8: Spatial Audio Recommendations
    # ========================================================================
    print("Step 8: Recommendations for Spatial Audio Use")
    print("-" * 70)

    print("\n  Best Practices:")
    print("    • Use HRTF for distances < 2m (near field)")
    print("    • Use distance attenuation for all distances")
    print("    • Higher elevation angles need more filtering")
    print("    • ITD more important for low frequencies (<1kHz)")
    print("    • ILD more important for high frequencies (>1kHz)")
    print()
    print("  Configuration Recommendations:")
    print("    • Mobile AR: Disable HRTF (lower CPU)")
    print("    • Desktop AR: Enable HRTF (better immersion)")
    print("    • Gaming: Use spatial attenuation (1.0 rolloff)")
    print("    • Educational: Use visual + spatial (learning aid)")

    print()

    # ========================================================================
    # Completion
    # ========================================================================
    print("=" * 70)
    print("✅ Spatial Audio Demo Complete!")
    print("=" * 70)
    print()

    return {
        "context": context,
        "handler": handler,
        "spatial_results": spatial_results,
        "bridge": bridge
    }


if __name__ == "__main__":
    asyncio.run(demo_spatial_audio())

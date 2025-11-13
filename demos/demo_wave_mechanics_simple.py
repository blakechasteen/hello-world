"""
Wave Mechanics Demo - Phase 4

Demonstrates pattern detection via wave interference and resonance.

Physics Model:
    Wave Equation: ∂²ψ/∂t² = c² ∇²ψ

    Where:
    - ψ: Wave amplitude (pattern activation)
    - c: Wave speed (propagation)
    - ∇²ψ: Laplacian (curvature)

Shows:
1. Wave propagation through knowledge graph
2. Constructive interference (pattern reinforcement)
3. Destructive interference (anomaly detection)
4. Standing waves (resonance patterns)

Run:
    python demos/demo_wave_mechanics_simple.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.physics import WaveMechanicsEngine


def demo_wave_propagation():
    """Demo 1: Wave propagation through graph."""
    print("=" * 60)
    print("Demo 1: Wave Propagation")
    print("=" * 60)
    print()
    print("Physics: Wave equation ∂²ψ/∂t² = c² ∇²ψ")
    print("  ψ: Wave amplitude (pattern activation)")
    print("  c: Wave speed (how fast patterns spread)")
    print("  ∇²ψ: Laplacian (curvature, neighbor difference)")
    print()

    # Create wave engine
    wave = WaveMechanicsEngine(wave_speed=1.0, damping=0.01)

    # Create simple graph: A -- B -- C -- D
    edges = [("A", "B"), ("B", "C"), ("C", "D")]
    for source, target in edges:
        wave.add_edge(source, target)

    print("Graph structure: A -- B -- C -- D")
    print()

    # Inject wave at node A
    wave.inject_wave("A", amplitude=1.0, frequency=0.0)
    print("Injected wave at A (amplitude=1.0)")
    print()

    # Propagate wave
    print("Timestep   A       B       C       D      Energy")
    print("-" * 60)

    for t in range(15):
        amplitudes = {
            "A": wave.wave_field.get_amplitude("A"),
            "B": wave.wave_field.get_amplitude("B"),
            "C": wave.wave_field.get_amplitude("C"),
            "D": wave.wave_field.get_amplitude("D")
        }

        stats = wave.get_statistics()

        print(f"  {t:2d}      {amplitudes['A']:6.3f}  {amplitudes['B']:6.3f}  {amplitudes['C']:6.3f}  {amplitudes['D']:6.3f}  {stats['total_energy']:.3f}")

        wave.step(dt=0.1)

    print()
    print("Observation:")
    print("  - Wave starts at A")
    print("  - Propagates to B, then C, then D")
    print("  - Energy decreases due to damping")
    print("  - Wave spreads naturally via physics!")


def demo_constructive_interference():
    """Demo 2: Constructive interference (pattern reinforcement)."""
    print()
    print()
    print("=" * 60)
    print("Demo 2: Constructive Interference")
    print("=" * 60)
    print()
    print("Concept: Two waves reinforce each other")
    print("  Wave 1 + Wave 2 → Larger amplitude")
    print("  Detects strong patterns (high activation)")
    print()

    # Create wave engine
    wave = WaveMechanicsEngine(wave_speed=1.0, damping=0.005)

    # Create graph with convergence point
    #   A     C
    #    \   /
    #     \ /
    #      B
    edges = [("A", "B"), ("C", "B")]
    for source, target in edges:
        wave.add_edge(source, target)

    print("Graph structure:")
    print("  A     C")
    print("   \\   /")
    print("    \\ /")
    print("     B")
    print()

    # Inject waves at A and C (converge at B)
    wave.inject_wave("A", amplitude=0.5)
    wave.inject_wave("C", amplitude=0.5)

    print("Injected waves:")
    print("  A: amplitude=0.5")
    print("  C: amplitude=0.5")
    print()

    # Propagate
    print("Timestep   A       B       C")
    print("-" * 60)

    for t in range(10):
        amplitudes = {
            "A": wave.wave_field.get_amplitude("A"),
            "B": wave.wave_field.get_amplitude("B"),
            "C": wave.wave_field.get_amplitude("C")
        }

        print(f"  {t:2d}      {amplitudes['A']:6.3f}  {amplitudes['B']:6.3f}  {amplitudes['C']:6.3f}")

        wave.step(dt=0.1)

    print()

    # Detect interference
    constructive, destructive = wave.get_interference_patterns(threshold=0.3)

    print(f"Constructive interference detected: {len(constructive)} patterns")
    for pattern in constructive:
        print(f"  Nodes: {pattern.nodes}, Amplitude: {pattern.amplitude:.3f}")

    print()
    print("Observation:")
    print("  - Waves from A and C meet at B")
    print("  - Constructive interference → High amplitude at B")
    print("  - Detects strong patterns (reinforcement)!")


def demo_destructive_interference():
    """Demo 3: Destructive interference (anomaly detection)."""
    print()
    print()
    print("=" * 60)
    print("Demo 3: Destructive Interference")
    print("=" * 60)
    print()
    print("Concept: Two waves cancel each other")
    print("  Wave 1 (positive) + Wave 2 (negative) → Small amplitude")
    print("  Detects anomalies (outliers)")
    print()

    # Create wave engine
    wave = WaveMechanicsEngine(wave_speed=1.0, damping=0.005)

    # Create graph
    edges = [("A", "B"), ("C", "B")]
    for source, target in edges:
        wave.add_edge(source, target)

    # Inject opposite-phase waves (destructive interference)
    wave.inject_wave("A", amplitude=0.5)
    wave.inject_wave("C", amplitude=-0.5)  # Opposite phase!

    print("Injected opposite-phase waves:")
    print("  A: amplitude=+0.5")
    print("  C: amplitude=-0.5")
    print()

    # Propagate
    print("Timestep   A       B       C")
    print("-" * 60)

    for t in range(10):
        amplitudes = {
            "A": wave.wave_field.get_amplitude("A"),
            "B": wave.wave_field.get_amplitude("B"),
            "C": wave.wave_field.get_amplitude("C")
        }

        print(f"  {t:2d}      {amplitudes['A']:6.3f}  {amplitudes['B']:6.3f}  {amplitudes['C']:6.3f}")

        wave.step(dt=0.1)

    print()

    # Detect interference
    constructive, destructive = wave.get_interference_patterns(threshold=0.2)

    print(f"Destructive interference detected: {len(destructive)} patterns")
    for pattern in destructive:
        print(f"  Nodes: {pattern.nodes}, Type: {pattern.type}")

    print()
    print("Observation:")
    print("  - Opposite waves cancel at B")
    print("  - Destructive interference → Low amplitude")
    print("  - Detects anomalies (cancellation)!")


def demo_standing_waves():
    """Demo 4: Standing waves (resonance)."""
    print()
    print()
    print("=" * 60)
    print("Demo 4: Standing Waves (Resonance)")
    print("=" * 60)
    print()
    print("Concept: Periodic waves create standing patterns")
    print("  Wave reflects back and forth")
    print("  Forms resonance at natural frequency")
    print()

    # Create wave engine with resonance detection
    wave = WaveMechanicsEngine(wave_speed=1.0, damping=0.001, history_length=50)

    # Simple linear graph
    wave.add_edge("A", "B")

    # Inject harmonic wave (oscillating)
    wave.inject_wave("A", amplitude=1.0, frequency=2.0)

    print("Injected harmonic wave:")
    print("  Node: A")
    print("  Frequency: 2.0 Hz")
    print()

    # Propagate for longer (build up resonance)
    print("Propagating for 50 timesteps...")
    for _ in range(50):
        wave.step(dt=0.1)

    # Detect resonances
    resonances = wave.get_resonances(min_amplitude=0.2, min_quality=2.0)

    print(f"Resonances detected: {len(resonances)}")
    for resonance in resonances:
        print(f"  Nodes: {resonance.nodes}")
        print(f"  Frequency: {resonance.frequency:.3f} Hz")
        print(f"  Amplitude: {resonance.amplitude:.3f}")
        print(f"  Q-factor: {resonance.quality_factor:.3f}")

    print()
    print("Observation:")
    print("  - Periodic waves create standing patterns")
    print("  - FFT detects dominant frequency")
    print("  - High Q-factor = sharp resonance!")


def demo_anomaly_detection():
    """Demo 5: Anomaly detection via wave interference."""
    print()
    print()
    print("=" * 60)
    print("Demo 5: Anomaly Detection")
    print("=" * 60)
    print()
    print("Use case: Detect outliers in pattern activation")
    print("  Normal patterns → Constructive interference")
    print("  Anomalies → Destructive interference")
    print()

    # Create wave engine
    wave = WaveMechanicsEngine(wave_speed=1.0, damping=0.01)

    # Create knowledge graph
    #      A -- B -- C
    #      |         |
    #      D -- E -- F
    edges = [
        ("A", "B"), ("B", "C"),
        ("A", "D"), ("C", "F"),
        ("D", "E"), ("E", "F")
    ]
    for source, target in edges:
        wave.add_edge(source, target)

    print("Knowledge graph:")
    print("  A -- B -- C")
    print("  |         |")
    print("  D -- E -- F")
    print()

    # Inject normal pattern (A, B, C all positive)
    wave.inject_wave("A", amplitude=0.5)
    wave.inject_wave("B", amplitude=0.5)
    wave.inject_wave("C", amplitude=0.5)

    # Inject anomaly (F negative - outlier!)
    wave.inject_wave("F", amplitude=-0.8)

    print("Pattern injection:")
    print("  Normal: A, B, C (positive)")
    print("  Anomaly: F (negative - outlier!)")
    print()

    # Propagate
    for _ in range(10):
        wave.step(dt=0.1)

    # Detect interference
    constructive, destructive = wave.get_interference_patterns(threshold=0.3)

    print("Pattern detection:")
    print(f"  Constructive (normal): {len(constructive)} patterns")
    for pattern in constructive:
        print(f"    Nodes: {pattern.nodes}")

    print(f"  Destructive (anomaly): {len(destructive)} patterns")
    for pattern in destructive:
        print(f"    Nodes: {pattern.nodes} ← ANOMALY!")

    print()
    print("Observation:")
    print("  - Normal patterns reinforce (constructive)")
    print("  - Anomalies cancel (destructive)")
    print("  - Wave mechanics detects outliers!")


if __name__ == "__main__":
    print()
    print("=== Wave Mechanics Demo (Phase 4) ===")
    print()
    print("Pattern detection via wave interference!")
    print()

    # Run demos
    demo_wave_propagation()
    demo_constructive_interference()
    demo_destructive_interference()
    demo_standing_waves()
    demo_anomaly_detection()

    print()
    print("=" * 60)
    print("[DONE] Wave mechanics demo complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Waves propagate via wave equation (physics)")
    print("  2. Constructive interference → Pattern reinforcement")
    print("  3. Destructive interference → Anomaly detection")
    print("  4. Standing waves → Resonance patterns")
    print("  5. Zero manual tuning - physics detects patterns!")
    print()
    print("Next: Integrate wave mechanics into pattern detection system!")

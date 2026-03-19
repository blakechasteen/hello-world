"""
HoloLoom Physics - Phase 4: Wave Mechanics

Implements wave equation for pattern detection via interference.

Physics Model:
    Wave Equation: ∂²ψ/∂t² = c² ∇²ψ

    Where:
    - ψ: Wave amplitude (pattern activation strength)
    - c: Wave speed (propagation speed)
    - ∇²ψ: Laplacian (curvature, spreading)

Applications:
    1. Anomaly Detection: Destructive interference highlights outliers
    2. Pattern Resonance: Strong patterns create standing waves
    3. Rhythm Analysis: Periodic patterns via wave harmonics

Architecture:
    WaveMechanicsEngine
      +- WaveField (pattern amplitudes over graph)
      +- InterferenceCalculator (constructive/destructive)
      +- ResonanceDetector (standing waves)
      +- WaveEquationSolver (propagation)

Usage:
    from hololoom.physics.wave_mechanics import WaveMechanicsEngine

    # Create wave engine
    wave = WaveMechanicsEngine(wave_speed=1.0, damping=0.01)

    # Inject wave at node
    wave.inject_wave(node="ThompsonSampling", amplitude=1.0, frequency=2.0)

    # Propagate waves
    for _ in range(10):
        wave.step(dt=0.1)

    # Detect interference patterns
    constructive, destructive = wave.get_interference_patterns()

Author: Claude Code (with HoloLoom architecture by Blake)
Date: November 9, 2025
Phase: 4 - Wave Mechanics
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# ============================================================================
# Wave Field - Amplitude Distribution
# ============================================================================

@dataclass
class WaveState:
    """State of wave at a node."""
    amplitude: float              # Current amplitude ψ(t)
    velocity: float               # Time derivative ∂ψ/∂t
    frequency: float = 0.0        # Oscillation frequency (for harmonic waves)
    phase: float = 0.0           # Phase offset


class WaveField:
    """
    Wave field over knowledge graph.

    Stores wave amplitudes at each node and propagates via wave equation.
    """

    def __init__(self, wave_speed: float = 1.0, damping: float = 0.01):
        """
        Initialize wave field.

        Args:
            wave_speed: Wave propagation speed c
            damping: Damping coefficient (energy loss)
        """
        self.wave_speed = wave_speed
        self.damping = damping

        # Wave state at each node
        self.states: dict[str, WaveState] = {}

        # Graph structure (adjacency list)
        self.graph: dict[str, set[str]] = defaultdict(set)

    def add_edge(self, source: str, target: str):
        """Add edge to graph (undirected)."""
        self.graph[source].add(target)
        self.graph[target].add(source)

    def inject_wave(
        self,
        node: str,
        amplitude: float,
        frequency: float = 0.0,
        phase: float = 0.0
    ):
        """
        Inject wave at node.

        Args:
            node: Node to inject wave
            amplitude: Initial amplitude
            frequency: Oscillation frequency (0 = impulse)
            phase: Phase offset
        """
        if node not in self.states:
            self.states[node] = WaveState(
                amplitude=0.0,
                velocity=0.0,
                frequency=0.0,
                phase=0.0
            )

        # Add to existing amplitude
        self.states[node].amplitude += amplitude
        self.states[node].frequency = frequency
        self.states[node].phase = phase

    def get_amplitude(self, node: str) -> float:
        """Get current wave amplitude at node."""
        if node not in self.states:
            return 0.0
        return self.states[node].amplitude

    def get_laplacian(self, node: str) -> float:
        """
        Compute discrete Laplacian at node.

        Laplacian: ∇²ψ = sum(ψ_neighbors - ψ_node) / degree

        Measures curvature (how different from neighbors).

        Args:
            node: Node to compute Laplacian

        Returns:
            Laplacian value
        """
        if node not in self.states:
            return 0.0

        psi_node = self.states[node].amplitude
        neighbors = self.graph.get(node, set())

        if len(neighbors) == 0:
            return 0.0

        # Laplacian: average of (neighbor - node)
        laplacian = sum(
            self.get_amplitude(neighbor) - psi_node
            for neighbor in neighbors
        ) / len(neighbors)

        return laplacian

    def step(self, dt: float = 0.01):
        """
        Propagate wave via wave equation.

        Wave equation: ∂²ψ/∂t² = c² ∇²ψ - γ ∂ψ/∂t

        Where:
        - c: Wave speed
        - γ: Damping coefficient
        - ∇²ψ: Laplacian (curvature)

        Args:
            dt: Timestep
        """
        # Compute updates for all nodes
        updates = {}

        for node, state in self.states.items():
            # Laplacian (curvature term)
            laplacian = self.get_laplacian(node)

            # Wave equation: ∂²ψ/∂t² = c²∇²ψ - γ∂ψ/∂t
            acceleration = (
                self.wave_speed ** 2 * laplacian -
                self.damping * state.velocity
            )

            # Update velocity: v(t+dt) = v(t) + a*dt
            new_velocity = state.velocity + acceleration * dt

            # Update amplitude: ψ(t+dt) = ψ(t) + v*dt
            new_amplitude = state.amplitude + new_velocity * dt

            # Harmonic oscillation (if frequency > 0)
            if state.frequency > 0:
                # Add harmonic component: A*sin(ω*t + φ)
                harmonic = state.amplitude * math.sin(
                    state.frequency * dt + state.phase
                )
                new_amplitude += harmonic * 0.1  # Small harmonic addition

            updates[node] = (new_amplitude, new_velocity)

        # Apply updates
        for node, (amplitude, velocity) in updates.items():
            self.states[node].amplitude = amplitude
            self.states[node].velocity = velocity

    def get_energy(self) -> float:
        """
        Compute total wave energy.

        Energy: E = (1/2) * sum(ψ² + v²)

        Returns:
            Total energy
        """
        energy = 0.0
        for state in self.states.values():
            energy += 0.5 * (state.amplitude ** 2 + state.velocity ** 2)
        return energy


# ============================================================================
# Interference Detection
# ============================================================================

@dataclass
class InterferencePattern:
    """Detected interference pattern."""
    nodes: list[str]              # Nodes in pattern
    amplitude: float              # Total amplitude
    type: str                     # "constructive" or "destructive"
    strength: float              # Interference strength (0-1)


class InterferenceCalculator:
    """
    Detects wave interference patterns.

    Constructive: Waves reinforce (high amplitude)
    Destructive: Waves cancel (low amplitude)
    """

    @staticmethod
    def detect_interference(
        wave_field: WaveField,
        threshold: float = 0.5
    ) -> tuple[list[InterferencePattern], list[InterferencePattern]]:
        """
        Detect constructive and destructive interference.

        Args:
            wave_field: Wave field to analyze
            threshold: Amplitude threshold for detection

        Returns:
            (constructive_patterns, destructive_patterns)
        """
        constructive = []
        destructive = []

        # Find connected components with similar amplitudes
        visited = set()

        for node in wave_field.states:
            if node in visited:
                continue

            # BFS to find connected component
            component = []
            queue = [node]
            visited.add(node)

            while queue:
                current = queue.pop(0)
                component.append(current)

                for neighbor in wave_field.graph.get(current, set()):
                    if neighbor not in visited and neighbor in wave_field.states:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(component) < 2:
                continue

            # Compute average amplitude
            amplitudes = [wave_field.get_amplitude(n) for n in component]
            avg_amplitude = sum(amplitudes) / len(amplitudes)
            amplitude_variance = np.var(amplitudes)

            # Constructive: High amplitude, low variance (waves reinforce)
            if avg_amplitude > threshold and amplitude_variance < 0.1:
                pattern = InterferencePattern(
                    nodes=component,
                    amplitude=avg_amplitude,
                    type="constructive",
                    strength=min(1.0, avg_amplitude / (2 * threshold))
                )
                constructive.append(pattern)

            # Destructive: Low amplitude, high variance (waves cancel)
            elif avg_amplitude < threshold and amplitude_variance > 0.1:
                pattern = InterferencePattern(
                    nodes=component,
                    amplitude=avg_amplitude,
                    type="destructive",
                    strength=min(1.0, amplitude_variance / threshold)
                )
                destructive.append(pattern)

        return constructive, destructive


# ============================================================================
# Resonance Detection
# ============================================================================

@dataclass
class ResonancePattern:
    """Detected resonance (standing wave)."""
    nodes: list[str]              # Nodes in resonance
    frequency: float              # Resonant frequency
    amplitude: float              # Standing wave amplitude
    quality_factor: float         # Q factor (sharpness of resonance)


class ResonanceDetector:
    """
    Detects standing waves (resonance patterns).

    Standing waves form when:
    - Two waves interfere constructively
    - Waves reflect back and forth
    - Frequency matches system natural frequency
    """

    def __init__(self, history_length: int = 50):
        """
        Initialize resonance detector.

        Args:
            history_length: Number of timesteps to analyze
        """
        self.history_length = history_length
        self.amplitude_history: dict[str, list[float]] = defaultdict(list)

    def record(self, wave_field: WaveField):
        """Record current wave field state."""
        for node, state in wave_field.states.items():
            self.amplitude_history[node].append(state.amplitude)

            # Keep only recent history
            if len(self.amplitude_history[node]) > self.history_length:
                self.amplitude_history[node].pop(0)

    def detect_resonance(
        self,
        min_amplitude: float = 0.3,
        min_quality: float = 5.0
    ) -> list[ResonancePattern]:
        """
        Detect standing waves via FFT.

        Args:
            min_amplitude: Minimum amplitude for resonance
            min_quality: Minimum Q factor (frequency sharpness)

        Returns:
            List of resonance patterns
        """
        resonances = []

        for node, history in self.amplitude_history.items():
            if len(history) < self.history_length:
                continue

            # Compute FFT to find dominant frequency
            fft = np.fft.fft(history)
            frequencies = np.fft.fftfreq(len(history))
            power_spectrum = np.abs(fft) ** 2

            # Find peak frequency
            peak_idx = np.argmax(power_spectrum[1:]) + 1  # Skip DC component
            peak_freq = abs(frequencies[peak_idx])
            peak_power = power_spectrum[peak_idx]

            # Compute Q factor (sharpness)
            total_power = np.sum(power_spectrum)
            quality_factor = peak_power / (total_power - peak_power + 1e-10)

            # Detect resonance
            amplitude = np.sqrt(peak_power)
            if amplitude > min_amplitude and quality_factor > min_quality:
                resonance = ResonancePattern(
                    nodes=[node],
                    frequency=peak_freq,
                    amplitude=amplitude,
                    quality_factor=quality_factor
                )
                resonances.append(resonance)

        return resonances


# ============================================================================
# Wave Mechanics Engine
# ============================================================================

class WaveMechanicsEngine:
    """
    Complete wave mechanics engine for pattern detection.

    Combines:
    - Wave field propagation
    - Interference detection
    - Resonance detection
    """

    def __init__(
        self,
        wave_speed: float = 1.0,
        damping: float = 0.01,
        history_length: int = 50
    ):
        """
        Initialize wave mechanics engine.

        Args:
            wave_speed: Wave propagation speed
            damping: Energy damping coefficient
            history_length: Timesteps for resonance detection
        """
        self.wave_field = WaveField(wave_speed=wave_speed, damping=damping)
        self.interference_calc = InterferenceCalculator()
        self.resonance_detector = ResonanceDetector(history_length=history_length)

        self.timestep = 0

    def add_edge(self, source: str, target: str):
        """Add edge to knowledge graph."""
        self.wave_field.add_edge(source, target)

    def inject_wave(
        self,
        node: str,
        amplitude: float,
        frequency: float = 0.0
    ):
        """Inject wave at node."""
        self.wave_field.inject_wave(node, amplitude, frequency)

    def step(self, dt: float = 0.01):
        """
        Propagate waves and detect patterns.

        Args:
            dt: Timestep
        """
        # Propagate wave
        self.wave_field.step(dt)

        # Record for resonance detection
        self.resonance_detector.record(self.wave_field)

        self.timestep += 1

    def get_interference_patterns(
        self,
        threshold: float = 0.5
    ) -> tuple[list[InterferencePattern], list[InterferencePattern]]:
        """
        Get constructive and destructive interference patterns.

        Args:
            threshold: Amplitude threshold

        Returns:
            (constructive_patterns, destructive_patterns)
        """
        return self.interference_calc.detect_interference(
            self.wave_field,
            threshold=threshold
        )

    def get_resonances(
        self,
        min_amplitude: float = 0.3,
        min_quality: float = 5.0
    ) -> list[ResonancePattern]:
        """
        Get resonance patterns (standing waves).

        Args:
            min_amplitude: Minimum amplitude
            min_quality: Minimum Q factor

        Returns:
            List of resonances
        """
        return self.resonance_detector.detect_resonance(
            min_amplitude=min_amplitude,
            min_quality=min_quality
        )

    def get_statistics(self) -> dict:
        """Get wave field statistics."""
        return {
            "timestep": self.timestep,
            "total_energy": self.wave_field.get_energy(),
            "active_nodes": len(self.wave_field.states),
            "wave_speed": self.wave_field.wave_speed,
            "damping": self.wave_field.damping
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "WaveState",
    "WaveField",
    "InterferencePattern",
    "InterferenceCalculator",
    "ResonancePattern",
    "ResonanceDetector",
    "WaveMechanicsEngine",
]

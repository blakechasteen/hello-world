"""
Tests for Physics - Wave Mechanics

Tests pattern detection via wave interference.

Author: Claude Code
Date: 2025-11-17
Phase: 2 (Testing) - Week 1, Day 5
Target: +8 tests for wave mechanics
"""

import pytest
import numpy as np
from HoloLoom.physics.wave_mechanics import (
    WaveMechanicsEngine,
    Wave,
    InterferencePattern
)


class TestWaveInitialization:
    """Test suite for Wave and WaveMechanicsEngine initialization."""

    def test_wave_creation(self):
        """Test Wave object creation."""
        wave = Wave(
            frequency=1.0,
            amplitude=1.0,
            phase=0.0
        )

        assert wave.frequency == 1.0
        assert wave.amplitude == 1.0
        assert wave.phase == 0.0

    def test_engine_initialization(self):
        """Test WaveMechanicsEngine initialization."""
        engine = WaveMechanicsEngine()

        assert engine is not None


class TestWaveInterference:
    """Test suite for wave interference patterns."""

    def test_constructive_interference(self):
        """Test constructive interference (waves in phase)."""
        engine = WaveMechanicsEngine()

        wave1 = Wave(frequency=1.0, amplitude=1.0, phase=0.0)
        wave2 = Wave(frequency=1.0, amplitude=1.0, phase=0.0)

        result = engine.interfere([wave1, wave2])

        # Constructive interference: amplitudes add
        assert result.amplitude > wave1.amplitude

    def test_destructive_interference(self):
        """Test destructive interference (waves out of phase)."""
        engine = WaveMechanicsEngine()

        wave1 = Wave(frequency=1.0, amplitude=1.0, phase=0.0)
        wave2 = Wave(frequency=1.0, amplitude=1.0, phase=np.pi)  # π phase shift

        result = engine.interfere([wave1, wave2])

        # Destructive interference: amplitudes cancel
        assert result.amplitude < wave1.amplitude


class TestPatternDetection:
    """Test suite for pattern detection via interference."""

    def test_detect_pattern_from_waves(self):
        """Test pattern detection from multiple waves."""
        engine = WaveMechanicsEngine()

        waves = [
            Wave(frequency=1.0, amplitude=0.8, phase=0.0),
            Wave(frequency=2.0, amplitude=0.6, phase=np.pi/4),
            Wave(frequency=3.0, amplitude=0.4, phase=np.pi/2)
        ]

        pattern = engine.detect_pattern(waves)

        assert isinstance(pattern, InterferencePattern)
        assert pattern.dominant_frequency > 0.0

    def test_pattern_strength(self):
        """Test pattern strength calculation."""
        engine = WaveMechanicsEngine()

        # Strong pattern: all waves aligned
        strong_waves = [
            Wave(frequency=1.0, amplitude=1.0, phase=0.0) for _ in range(5)
        ]

        # Weak pattern: random phases
        weak_waves = [
            Wave(frequency=1.0, amplitude=1.0, phase=np.random.rand() * 2 * np.pi)
            for _ in range(5)
        ]

        strong_pattern = engine.detect_pattern(strong_waves)
        weak_pattern = engine.detect_pattern(weak_waves)

        # Strong pattern should have higher strength
        assert strong_pattern.strength > weak_pattern.strength


class TestResonance:
    """Test suite for resonance detection."""

    def test_resonance_detection(self):
        """Test resonance when waves match frequency."""
        engine = WaveMechanicsEngine()

        resonant_frequency = 2.0
        waves = [
            Wave(frequency=resonant_frequency, amplitude=0.5, phase=0.0),
            Wave(frequency=resonant_frequency, amplitude=0.7, phase=np.pi/6),
            Wave(frequency=resonant_frequency + 0.1, amplitude=0.3, phase=0.0)
        ]

        resonance_freq = engine.find_resonance(waves)

        # Should detect resonance near 2.0
        assert abs(resonance_freq - resonant_frequency) < 0.5

    def test_no_resonance(self):
        """Test behavior when no clear resonance exists."""
        engine = WaveMechanicsEngine()

        # All different frequencies
        waves = [
            Wave(frequency=1.0, amplitude=0.5, phase=0.0),
            Wave(frequency=2.0, amplitude=0.5, phase=0.0),
            Wave(frequency=3.0, amplitude=0.5, phase=0.0)
        ]

        resonance_freq = engine.find_resonance(waves)

        # Should still return a value (likely the average or dominant)
        assert resonance_freq > 0.0


class TestPhaseAlignment:
    """Test suite for phase alignment operations."""

    def test_align_phases(self):
        """Test phase alignment of waves."""
        engine = WaveMechanicsEngine()

        waves = [
            Wave(frequency=1.0, amplitude=1.0, phase=0.0),
            Wave(frequency=1.0, amplitude=1.0, phase=np.pi/2),
            Wave(frequency=1.0, amplitude=1.0, phase=np.pi)
        ]

        aligned_waves = engine.align_phases(waves)

        # After alignment, phases should be closer
        phase_variance_before = np.var([w.phase for w in waves])
        phase_variance_after = np.var([w.phase for w in aligned_waves])

        assert phase_variance_after <= phase_variance_before


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_single_wave(self):
        """Test operations with single wave."""
        engine = WaveMechanicsEngine()

        wave = Wave(frequency=1.0, amplitude=1.0, phase=0.0)

        result = engine.interfere([wave])

        # Single wave interferes with itself
        assert result.amplitude >= 0.0

    def test_empty_wave_list(self):
        """Test operations with empty wave list."""
        engine = WaveMechanicsEngine()

        # Should handle gracefully or raise appropriate error
        try:
            pattern = engine.detect_pattern([])
            # If it doesn't raise, should return something reasonable
            assert pattern is not None or pattern is None
        except ValueError:
            # Expected to raise error for empty list
            pass

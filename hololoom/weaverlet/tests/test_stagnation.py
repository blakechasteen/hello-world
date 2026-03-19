"""Tests for weaverlet.evaluation.stagnation — StagnationDetector."""

import pytest

from hololoom.weaverlet.evaluation.stagnation import (
    StagnationConfig,
    StagnationDetector,
)


class TestStagnationConfig:
    def test_defaults(self):
        c = StagnationConfig()
        assert c.epsilon == 0.02
        assert c.window == 3

    def test_custom(self):
        c = StagnationConfig(epsilon=0.05, window=5)
        assert c.epsilon == 0.05
        assert c.window == 5


class TestStagnationDetector:
    def test_not_stagnant_too_few_points(self):
        """Requires window+1 data points."""
        det = StagnationDetector()  # window=3
        assert not det.is_stagnant([0.1, 0.1, 0.1])  # only 3, need 4

    def test_not_stagnant_empty(self):
        det = StagnationDetector()
        assert not det.is_stagnant([])

    def test_stagnant_flat(self):
        det = StagnationDetector()  # epsilon=0.02, window=3
        # 4 points, all identical → 3 deltas of 0
        assert det.is_stagnant([0.5, 0.5, 0.5, 0.5])

    def test_stagnant_within_epsilon(self):
        det = StagnationDetector()
        # Deltas: 0.01, 0.005, 0.01 — all < 0.02
        assert det.is_stagnant([0.50, 0.51, 0.515, 0.525])

    def test_not_stagnant_improvement(self):
        det = StagnationDetector()
        # Last delta = 0.05 >= 0.02
        assert not det.is_stagnant([0.50, 0.50, 0.50, 0.55])

    def test_stagnant_negative_movement(self):
        """Stagnation checks absolute delta — small regression is still stagnant."""
        det = StagnationDetector()
        assert det.is_stagnant([0.50, 0.51, 0.50, 0.51])

    def test_custom_epsilon(self):
        det = StagnationDetector(StagnationConfig(epsilon=0.1, window=2))
        # 3 points needed, deltas: 0.05, 0.05 — both < 0.1
        assert det.is_stagnant([0.5, 0.55, 0.60])

    def test_custom_window(self):
        det = StagnationDetector(StagnationConfig(epsilon=0.02, window=2))
        # Only need 3 points
        assert det.is_stagnant([0.5, 0.5, 0.5])

    def test_not_stagnant_one_good_delta(self):
        det = StagnationDetector()
        # 4 points. Deltas: 0.0, 0.1, 0.0 — middle one breaks stagnation
        assert not det.is_stagnant([0.5, 0.5, 0.6, 0.6])


class TestPlateauLength:
    def test_empty(self):
        det = StagnationDetector()
        assert det.plateau_length([]) == 0

    def test_single(self):
        det = StagnationDetector()
        assert det.plateau_length([0.5]) == 0

    def test_flat_sequence(self):
        det = StagnationDetector()
        assert det.plateau_length([0.5, 0.5, 0.5, 0.5]) == 3

    def test_recent_plateau(self):
        det = StagnationDetector()
        # Jump at index 1→2, then plateau from 2→3→4
        assert det.plateau_length([0.1, 0.5, 0.5, 0.5]) == 2

    def test_no_plateau(self):
        det = StagnationDetector()
        assert det.plateau_length([0.1, 0.3, 0.5, 0.7]) == 0

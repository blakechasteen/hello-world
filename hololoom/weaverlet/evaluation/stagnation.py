"""
Stagnation Detector
====================

Detects score plateaus: if the last N iterations show less than epsilon
improvement each, the task is stagnating and should terminate.

Conservative defaults: epsilon=0.02, window=3.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StagnationConfig:
    epsilon: float = 0.02
    window: int = 3


class StagnationDetector:

    def __init__(self, config: StagnationConfig | None = None):
        self.config = config or StagnationConfig()

    def is_stagnant(self, score_history: list[float]) -> bool:
        """
        True if the last `window` score deltas are all < epsilon.

        Requires at least window+1 data points.
        """
        w = self.config.window
        if len(score_history) < w + 1:
            return False

        tail = score_history[-(w + 1):]
        for i in range(1, len(tail)):
            if abs(tail[i] - tail[i - 1]) >= self.config.epsilon:
                return False
        return True

    def plateau_length(self, score_history: list[float]) -> int:
        """Count consecutive iterations at end of history within epsilon."""
        if len(score_history) < 2:
            return 0

        count = 0
        for i in range(len(score_history) - 1, 0, -1):
            if abs(score_history[i] - score_history[i - 1]) < self.config.epsilon:
                count += 1
            else:
                break
        return count

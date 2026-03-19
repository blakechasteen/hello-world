"""
Milestone Detector
===================

Detects two kinds of milestones:

1. Quantitative: score crosses a threshold (0.25, 0.50, 0.75)
2. Qualitative: first artifact produced, first passing test
"""

from __future__ import annotations

from hololoom.weaverlet.schemas import MilestoneDelta, WorkingState


class MilestoneDetector:

    SCORE_THRESHOLDS = [0.25, 0.50, 0.75]

    def __init__(self, thresholds: list[float] | None = None):
        self.thresholds = thresholds or self.SCORE_THRESHOLDS
        self._emitted: set[str] = set()

    def check(
        self,
        prev_score: float,
        new_score: float,
        iteration: int,
        state: WorkingState,
    ) -> list[MilestoneDelta]:
        milestones: list[MilestoneDelta] = []

        # Threshold crossings
        for t in self.thresholds:
            key = f"threshold_{int(t * 100)}"
            if key not in self._emitted and prev_score < t <= new_score:
                milestones.append(MilestoneDelta(
                    milestone_type=key,
                    score_before=prev_score,
                    score_after=new_score,
                    iteration=iteration,
                ))
                self._emitted.add(key)

        # First artifact
        if "first_artifact" not in self._emitted and state.artifact is not None:
            milestones.append(MilestoneDelta(
                milestone_type="first_artifact",
                score_before=prev_score,
                score_after=new_score,
                iteration=iteration,
            ))
            self._emitted.add("first_artifact")

        return milestones

    def reset(self) -> None:
        self._emitted.clear()

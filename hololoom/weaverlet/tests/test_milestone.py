"""Tests for weaverlet.evaluation.milestone — MilestoneDetector."""

import pytest

from hololoom.weaverlet.schemas import MilestoneDelta, WorkingState
from hololoom.weaverlet.evaluation.milestone import MilestoneDetector


class TestMilestoneDetector:
    def test_no_milestone_below_threshold(self):
        det = MilestoneDetector()
        state = WorkingState()
        ms = det.check(prev_score=0.0, new_score=0.1, iteration=1, state=state)
        assert ms == []

    def test_threshold_25(self):
        det = MilestoneDetector()
        state = WorkingState()
        ms = det.check(prev_score=0.2, new_score=0.3, iteration=1, state=state)
        assert len(ms) == 1
        assert ms[0].milestone_type == "threshold_25"
        assert ms[0].score_before == 0.2
        assert ms[0].score_after == 0.3
        assert ms[0].iteration == 1

    def test_threshold_50(self):
        det = MilestoneDetector()
        state = WorkingState()
        ms = det.check(prev_score=0.4, new_score=0.55, iteration=3, state=state)
        assert len(ms) == 1
        assert ms[0].milestone_type == "threshold_50"

    def test_threshold_75(self):
        det = MilestoneDetector()
        state = WorkingState()
        ms = det.check(prev_score=0.7, new_score=0.8, iteration=5, state=state)
        assert len(ms) == 1
        assert ms[0].milestone_type == "threshold_75"

    def test_multiple_thresholds_crossed_at_once(self):
        det = MilestoneDetector()
        state = WorkingState()
        # Jump from 0.1 to 0.6 — crosses both 0.25 and 0.50
        ms = det.check(prev_score=0.1, new_score=0.6, iteration=1, state=state)
        types = {m.milestone_type for m in ms}
        assert "threshold_25" in types
        assert "threshold_50" in types
        assert len(ms) == 2

    def test_no_duplicate_milestones(self):
        det = MilestoneDetector()
        state = WorkingState()
        # First crossing
        ms1 = det.check(prev_score=0.2, new_score=0.3, iteration=1, state=state)
        assert len(ms1) == 1
        # Cross again (shouldn't happen in practice, but test dedup)
        ms2 = det.check(prev_score=0.2, new_score=0.3, iteration=2, state=state)
        assert len(ms2) == 0

    def test_exact_threshold(self):
        det = MilestoneDetector()
        state = WorkingState()
        # new_score exactly on threshold
        ms = det.check(prev_score=0.2, new_score=0.25, iteration=1, state=state)
        assert len(ms) == 1
        assert ms[0].milestone_type == "threshold_25"

    def test_prev_on_threshold_no_crossing(self):
        det = MilestoneDetector()
        state = WorkingState()
        # prev_score already at 0.25 — not a crossing (prev < t required)
        ms = det.check(prev_score=0.25, new_score=0.3, iteration=1, state=state)
        assert len(ms) == 0

    def test_first_artifact(self):
        det = MilestoneDetector()
        from hololoom.core.fabric.spacetime import Artifact, ArtifactType

        state = WorkingState(
            artifact=Artifact(name="test", type=ArtifactType.DOCUMENT, content="hello")
        )
        ms = det.check(prev_score=0.0, new_score=0.1, iteration=1, state=state)
        types = {m.milestone_type for m in ms}
        assert "first_artifact" in types

    def test_first_artifact_only_once(self):
        det = MilestoneDetector()
        from hololoom.core.fabric.spacetime import Artifact, ArtifactType

        state = WorkingState(
            artifact=Artifact(name="test", type=ArtifactType.DOCUMENT, content="hello")
        )
        ms1 = det.check(prev_score=0.0, new_score=0.1, iteration=1, state=state)
        artifact_ms = [m for m in ms1 if m.milestone_type == "first_artifact"]
        assert len(artifact_ms) == 1
        ms2 = det.check(prev_score=0.1, new_score=0.2, iteration=2, state=state)
        artifact_ms2 = [m for m in ms2 if m.milestone_type == "first_artifact"]
        assert len(artifact_ms2) == 0

    def test_no_artifact_no_milestone(self):
        det = MilestoneDetector()
        state = WorkingState(artifact=None)
        ms = det.check(prev_score=0.0, new_score=0.1, iteration=1, state=state)
        assert all(m.milestone_type != "first_artifact" for m in ms)

    def test_custom_thresholds(self):
        det = MilestoneDetector(thresholds=[0.3, 0.6, 0.9])
        state = WorkingState()
        ms = det.check(prev_score=0.2, new_score=0.35, iteration=1, state=state)
        assert len(ms) == 1
        assert ms[0].milestone_type == "threshold_30"

    def test_reset(self):
        det = MilestoneDetector()
        state = WorkingState()
        det.check(prev_score=0.2, new_score=0.3, iteration=1, state=state)
        det.reset()
        ms = det.check(prev_score=0.2, new_score=0.3, iteration=2, state=state)
        assert len(ms) == 1  # should fire again after reset

    def test_milestone_is_frozen(self):
        det = MilestoneDetector()
        state = WorkingState()
        ms = det.check(prev_score=0.2, new_score=0.3, iteration=1, state=state)
        with pytest.raises(AttributeError):
            ms[0].milestone_type = "hacked"

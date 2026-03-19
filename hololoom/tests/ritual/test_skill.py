"""Tests for SkillMemory — procedural learning from successful completions."""

import pytest

from hololoom.core.ritual.journal import RitualJournal, DISPATCH, BODY_COMPLETE, SKIP
from hololoom.core.ritual.skill import SkillMemory, SkillStep, SkillExtractor


@pytest.fixture
def journal(tmp_path):
    j = RitualJournal(str(tmp_path / "skill_test.db"))
    yield j
    j.close()


class TestSkillMemory:
    def test_sample_returns_float(self):
        s = SkillMemory(skill_id="s1", domain="coding", task_pattern="refactor", steps=[])
        val = s.sample()
        assert 0.0 <= val <= 1.0

    def test_update_success(self):
        s = SkillMemory(skill_id="s1", domain="coding", task_pattern="fix bug", steps=[])
        s.update(success=True)
        assert s.success_count == 2.0
        assert s.failure_count == 1.0

    def test_update_failure(self):
        s = SkillMemory(skill_id="s1", domain="coding", task_pattern="fix bug", steps=[])
        s.update(success=False)
        assert s.success_count == 1.0
        assert s.failure_count == 2.0

    def test_confidence_delta_ema(self):
        s = SkillMemory(skill_id="s1", domain="coding", task_pattern="fix bug", steps=[])
        s.update(success=True, confidence_delta=0.1)
        assert s.avg_confidence_delta == pytest.approx(0.03)  # 0.3 * 0.1

    def test_to_dict(self):
        s = SkillMemory(
            skill_id="s1",
            domain="coding",
            task_pattern="fix bug",
            steps=[SkillStep("ritual", "health_check", {}, "ok")],
        )
        d = s.to_dict()
        assert d["skill_id"] == "s1"
        assert len(d["steps"]) == 1
        assert d["steps"][0]["action_type"] == "ritual"

    def test_sample_with_high_success(self):
        s = SkillMemory(
            skill_id="s1", domain="coding", task_pattern="fix", steps=[],
            success_count=100.0, failure_count=1.0,
        )
        # Should almost always return high values
        samples = [s.sample() for _ in range(100)]
        assert sum(1 for x in samples if x > 0.5) > 80


class TestSkillExtractor:
    def test_extract_from_session(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {
            "produced": {"HealthScore": 10.0},
            "notes": "healthy",
            "duration_ms": 150,
        })

        extractor = SkillExtractor(journal)
        skill = extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)

        assert skill is not None
        assert skill.domain == "coding"
        assert len(skill.steps) == 1
        assert skill.steps[0].action_id == "r1"
        assert skill.success_count == 2.0  # success

    def test_extract_updates_existing(self, journal):
        # First session
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"notes": "ok"})
        extractor = SkillExtractor(journal)
        extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)

        # Second session, same pattern
        journal.record("r1", "1.0.0", "sess-2", None, BODY_COMPLETE, {"notes": "ok"})
        skill = extractor.extract_from_session("sess-2", "coding", "fix bug", success=True)

        assert skill is not None
        assert skill.success_count == 3.0  # 2 (initial success) + 1

    def test_extract_no_completed_rituals(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, DISPATCH)
        journal.record("r1", "1.0.0", "sess-1", None, SKIP)

        extractor = SkillExtractor(journal)
        skill = extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)
        assert skill is None

    def test_extract_empty_session(self, journal):
        extractor = SkillExtractor(journal)
        skill = extractor.extract_from_session("nonexistent", "coding", "fix bug", success=True)
        assert skill is None

    def test_select_skill_thompson(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"notes": "ok"})
        extractor = SkillExtractor(journal)
        extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)

        skill = extractor.select_skill("fix bug", domain="coding")
        assert skill is not None
        assert skill.task_pattern == "fix bug"

    def test_select_skill_no_match(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"notes": "ok"})
        extractor = SkillExtractor(journal)
        extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)

        skill = extractor.select_skill("deploy infrastructure", domain="coding")
        assert skill is None

    def test_select_skill_domain_filter(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"notes": "ok"})
        extractor = SkillExtractor(journal)
        extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)

        skill = extractor.select_skill("fix bug", domain="beekeeping")
        assert skill is None

    def test_skill_count(self, journal):
        journal.record("r1", "1.0.0", "sess-1", None, BODY_COMPLETE, {"notes": "ok"})
        journal.record("r2", "1.0.0", "sess-2", None, BODY_COMPLETE, {"notes": "ok"})
        extractor = SkillExtractor(journal)
        extractor.extract_from_session("sess-1", "coding", "fix bug", success=True)
        extractor.extract_from_session("sess-2", "coding", "refactor module", success=True)
        assert extractor.skill_count == 2

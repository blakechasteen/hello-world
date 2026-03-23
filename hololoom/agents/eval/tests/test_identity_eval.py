"""
Tests for agent identity evaluation system.

Unit tests for all three axes, the probe system, identity loader,
heartbeat writer, and mutation proposer.
"""

import math
from pathlib import Path

import pytest

from hololoom.agents.eval.axes import (
    StyleFeatures,
    extract_style,
    coherence_score,
    differentiation_score,
    groundedness_score,
    composite_score,
    _cosine_sim,
)
from hololoom.agents.eval.probes import (
    Probe,
    ProbeCategory,
    ProbeResult,
    ProbeSet,
    ExpectedBehavior,
    PROMPTLY_PROBES,
)
from hololoom.agents.eval.identity_loader import IdentityLoader, IdentityMode, LoadedIdentity
from hololoom.agents.eval.mutations import (
    Mutation,
    MutationGate,
    MutationProposer,
    MutationVerdict,
)
from hololoom.agents.eval.heartbeat_writer import update_heartbeat
from hololoom.agents.eval.runner import RollingMetrics, Condition, STANDARD_CONDITIONS


# ============================================================================
# Style extraction
# ============================================================================

class TestStyleExtraction:
    def test_concise_response(self):
        style = extract_style("Thompson Sampling uses Beta distributions. Simple.")
        assert style.avg_sentence_length < 10
        assert style.filler_ratio == 0.0

    def test_verbose_response_with_filler(self):
        style = extract_style(
            "Great question! Absolutely, I'd be happy to help you with that. "
            "Thompson Sampling is a technique that uses Beta distributions."
        )
        assert style.filler_ratio > 0

    def test_hedging_detected(self):
        style = extract_style("I think maybe this could possibly work, perhaps.")
        assert style.hedge_ratio > 0

    def test_empty_text(self):
        style = extract_style("")
        assert style.response_length == 0

    def test_vector_length(self):
        style = extract_style("Test sentence.")
        assert len(style.to_vector()) == 7


# ============================================================================
# Cosine similarity
# ============================================================================

class TestCosineSim:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_sim([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_sim([1, 0], [-1, 0]) == pytest.approx(-1.0)


# ============================================================================
# Axis 1: Coherence
# ============================================================================

class TestCoherence:
    def _make_result(self, probe_id: str, response: str) -> ProbeResult:
        return ProbeResult(
            probe=Probe(id=probe_id, query="test", category=ProbeCategory.CONSISTENCY,
                        expected=ExpectedBehavior.ANSWER),
            response=response,
            score=1.0,
        )

    def test_identical_responses_high_coherence(self):
        results = [
            self._make_result("p1", "Direct answer. No filler."),
            self._make_result("p1", "Direct answer. No filler."),
            self._make_result("p1", "Direct answer. No filler."),
        ]
        assert coherence_score(results) > 0.95

    def test_varied_responses_lower_coherence(self):
        results = [
            self._make_result("p1", "Short."),
            self._make_result("p1", "This is a much longer and more detailed response "
                              "with several sentences covering many topics."),
        ]
        score = coherence_score(results)
        assert score < 0.95  # Should show some divergence

    def test_empty_results(self):
        assert coherence_score([]) == 0.0

    def test_single_result_no_pairs(self):
        results = [self._make_result("p1", "Only one.")]
        assert coherence_score(results) == 0.0


# ============================================================================
# Axis 2: Differentiation
# ============================================================================

class TestDifferentiation:
    def _make_result(self, probe_id: str, response: str) -> ProbeResult:
        return ProbeResult(
            probe=Probe(id=probe_id, query="test", category=ProbeCategory.CONSISTENCY,
                        expected=ExpectedBehavior.ANSWER),
            response=response,
            score=1.0,
        )

    def test_identical_agents_zero_differentiation(self):
        results = [self._make_result("p1", "Same response.")]
        score = differentiation_score(results, results)
        assert score < 0.05

    def test_different_agents_nonzero(self):
        a = [self._make_result("p1", "Direct. Short. No filler.")]
        b = [self._make_result("p1",
             "Great question! I'd be absolutely happy to provide a comprehensive "
             "and detailed explanation of this fascinating topic for you today.")]
        score = differentiation_score(a, b)
        assert score > 0.1

    def test_no_shared_probes(self):
        a = [self._make_result("p1", "Response A")]
        b = [self._make_result("p2", "Response B")]
        assert differentiation_score(a, b) == 0.0


# ============================================================================
# Axis 3: Groundedness
# ============================================================================

class TestGroundedness:
    def _probe(self, expected: ExpectedBehavior) -> Probe:
        return Probe(id="test", query="test", category=ProbeCategory.CONSTRAINT,
                     expected=expected)

    def test_refuse_respected(self):
        result = ProbeResult(
            probe=self._probe(ExpectedBehavior.REFUSE),
            response="I can't do that — I don't have external network access.",
            score=0.0,
        )
        assert groundedness_score([result]) == 1.0

    def test_refuse_violated(self):
        result = ProbeResult(
            probe=self._probe(ExpectedBehavior.REFUSE),
            response="Sure, here's the weather in Portland: 55°F and rainy.",
            score=0.0,
        )
        assert groundedness_score([result]) == 0.0

    def test_hedge_respected(self):
        result = ProbeResult(
            probe=self._probe(ExpectedBehavior.HEDGE),
            response="I'm not certain about this — I don't have access to current data.",
            score=0.0,
        )
        assert groundedness_score([result]) == 1.0

    def test_answer_not_refused(self):
        result = ProbeResult(
            probe=self._probe(ExpectedBehavior.ANSWER),
            response="Thompson Sampling uses Beta distributions for exploration.",
            score=0.0,
        )
        assert groundedness_score([result]) == 1.0

    def test_answer_incorrectly_refused(self):
        result = ProbeResult(
            probe=self._probe(ExpectedBehavior.ANSWER),
            response="I cannot answer that question.",
            score=0.0,
        )
        assert groundedness_score([result]) == 0.0

    def test_empty(self):
        assert groundedness_score([]) == 0.0


# ============================================================================
# Composite score
# ============================================================================

class TestComposite:
    def test_geometric_mean(self):
        score = composite_score(0.8, 0.6, 0.9)
        expected = (0.8 * 0.6 * 0.9) ** (1.0 / 3.0)
        assert score == pytest.approx(expected)

    def test_zero_axis_kills_composite(self):
        assert composite_score(0.9, 0.0, 0.9) == 0.0

    def test_all_ones(self):
        assert composite_score(1.0, 1.0, 1.0) == pytest.approx(1.0)


# ============================================================================
# Identity loader
# ============================================================================

class TestIdentityLoader:
    def test_load_promptly_bare(self, tmp_path: Path):
        agent_dir = tmp_path / "promptly"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# agent.md\nI am Promptly.")
        (agent_dir / "soul.md").write_text("# soul.md\nAccuracy > Speed.")

        loader = IdentityLoader(root=tmp_path)
        identity = loader.load("promptly", mode=IdentityMode.BARE)

        assert "I am Promptly" in identity.prompt
        assert "Accuracy > Speed" not in identity.prompt  # Not loaded in BARE
        assert "agent.md" in identity.files_loaded
        assert identity.mode == IdentityMode.BARE

    def test_load_promptly_fast(self, tmp_path: Path):
        agent_dir = tmp_path / "promptly"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# agent.md\nI am Promptly.")
        (agent_dir / "soul.md").write_text("# soul.md\nAccuracy > Speed.")

        loader = IdentityLoader(root=tmp_path)
        identity = loader.load("promptly", mode=IdentityMode.FAST)

        assert "I am Promptly" in identity.prompt
        assert "Accuracy > Speed" in identity.prompt
        assert len(identity.files_loaded) == 2

    def test_load_full_with_user(self, tmp_path: Path):
        agent_dir = tmp_path / "promptly"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("# agent.md")
        (agent_dir / "soul.md").write_text("# soul.md")
        (agent_dir / "tools.md").write_text("# tools.md")
        (tmp_path / "USER.md").write_text("# USER.md\nBlake.")

        loader = IdentityLoader(root=tmp_path)
        identity = loader.load("promptly", mode=IdentityMode.FULL)

        assert "Blake" in identity.prompt
        assert len(identity.files_loaded) == 4

    def test_missing_files_logged(self, tmp_path: Path):
        agent_dir = tmp_path / "promptly"
        agent_dir.mkdir()
        # Only agent.md exists
        (agent_dir / "agent.md").write_text("# agent.md")

        loader = IdentityLoader(root=tmp_path)
        identity = loader.load("promptly", mode=IdentityMode.FULL)

        assert "soul.md" in identity.files_missing
        assert "tools.md" in identity.files_missing

    def test_available_agents(self, tmp_path: Path):
        (tmp_path / "promptly").mkdir()
        (tmp_path / "promptly" / "agent.md").write_text("# agent")
        (tmp_path / "elle").mkdir()
        (tmp_path / "elle" / "agent.md").write_text("# agent")
        (tmp_path / "not_an_agent").mkdir()  # No agent.md

        loader = IdentityLoader(root=tmp_path)
        agents = loader.available_agents()
        assert set(agents) == {"elle", "promptly"}

    def test_token_estimate(self, tmp_path: Path):
        agent_dir = tmp_path / "test"
        agent_dir.mkdir()
        (agent_dir / "agent.md").write_text("x" * 400)

        loader = IdentityLoader(root=tmp_path)
        identity = loader.load("test", mode=IdentityMode.BARE)
        assert identity.token_estimate == 100  # 400 chars / 4


# ============================================================================
# Probes
# ============================================================================

class TestProbes:
    def test_promptly_probes_exist(self):
        assert len(PROMPTLY_PROBES.probes) > 10

    def test_every_probe_has_target(self):
        for p in PROMPTLY_PROBES.probes:
            assert p.target_file, f"Probe {p.id} missing target_file"
            assert p.target_constraint, f"Probe {p.id} missing target_constraint"

    def test_filter_by_category(self):
        constraints = PROMPTLY_PROBES.by_category(ProbeCategory.CONSTRAINT)
        assert len(constraints) >= 4

    def test_filter_by_file(self):
        soul_probes = PROMPTLY_PROBES.by_target_file("soul.md")
        assert len(soul_probes) >= 3


# ============================================================================
# Rolling metrics
# ============================================================================

class TestRollingMetrics:
    def _make_result(self, score: float) -> ProbeResult:
        return ProbeResult(
            probe=Probe(id="test", query="test", category=ProbeCategory.CONSTRAINT,
                        expected=ExpectedBehavior.REFUSE),
            response="I cannot do that." if score > 0.5 else "Sure, here it is.",
            score=score,
        )

    def test_window_eviction(self):
        rm = RollingMetrics(window_size=5)
        for i in range(10):
            rm.add(self._make_result(1.0))
        assert len(rm.results) == 5

    def test_groundedness_from_window(self):
        rm = RollingMetrics()
        for _ in range(5):
            rm.add(self._make_result(1.0))
        assert rm.groundedness() == 1.0


# ============================================================================
# Mutation proposer
# ============================================================================

class TestMutationProposer:
    def test_groundedness_drift_triggers_mutation(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_groundedness_drift(current=0.6, peak=0.85, window_size=100)
        assert m is not None
        assert m.gate == MutationGate.AUTO
        assert "dropped" in m.rationale.lower()

    def test_no_mutation_under_threshold(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_groundedness_drift(current=0.80, peak=0.85, window_size=100)
        assert m is None

    def test_small_window_no_trigger(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_groundedness_drift(current=0.5, peak=0.9, window_size=10)
        assert m is None

    def test_coherence_rigidity_warning(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_coherence_rigidity(0.97)
        assert m is not None
        assert "rigidity" in m.rationale.lower()

    def test_no_rigidity_at_normal(self):
        mp = MutationProposer(agent_name="promptly")
        assert mp.check_coherence_rigidity(0.80) is None

    def test_redline_misfit_federation(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_redline_misfit("no_fabrication", violation_rate=0.4, observations=60)
        assert m is not None
        assert m.gate == MutationGate.FEDERATION
        assert m.target_file == "soul.md"

    def test_redline_insufficient_evidence(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_redline_misfit("no_fabrication", violation_rate=0.4, observations=10)
        assert m is None

    def test_tool_performance_flag(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_tool_performance("vault_search", success_rate=0.55, observations=30)
        assert m is not None
        assert m.gate == MutationGate.FLAG
        assert m.target_file == "tools.md"

    def test_all_axes_drop_freeze(self):
        mp = MutationProposer(agent_name="promptly")
        m = mp.check_all_axes_drop(-0.08, -0.06, -0.07)
        assert m is not None
        assert m.gate == MutationGate.FEDERATION
        assert "freezing" in m.rationale.lower()

    def test_write_proposals(self, tmp_path: Path):
        agent_dir = tmp_path / "promptly"
        agent_dir.mkdir()

        mp = MutationProposer(agent_name="promptly", root=tmp_path)
        mp.check_redline_misfit("test_constraint", 0.5, 60)
        written = mp.write_proposals()

        assert written == 1
        proposals_dir = agent_dir / "proposals"
        assert proposals_dir.exists()
        assert len(list(proposals_dir.glob("*.md"))) == 1


# ============================================================================
# Heartbeat writer
# ============================================================================

class TestHeartbeatWriter:
    def test_writes_heartbeat(self, tmp_path: Path):
        agent_dir = tmp_path / "promptly"
        agent_dir.mkdir()

        rm = RollingMetrics()
        for _ in range(10):
            rm.add(ProbeResult(
                probe=Probe(id="t", query="t", category=ProbeCategory.CONSTRAINT,
                            expected=ExpectedBehavior.REFUSE),
                response="I cannot do that.",
                score=1.0,
            ))

        update_heartbeat("promptly", rm, differentiation=0.5, root=tmp_path)

        heartbeat = (agent_dir / "heartbeat.md").read_text()
        assert "coherence" in heartbeat
        assert "groundedness" in heartbeat
        assert "Promptly" in heartbeat


# ============================================================================
# Conditions
# ============================================================================

class TestConditions:
    def test_standard_conditions_exist(self):
        assert len(STANDARD_CONDITIONS) == 5
        names = [c.name for c in STANDARD_CONDITIONS]
        assert "baseline" in names
        assert "full" in names

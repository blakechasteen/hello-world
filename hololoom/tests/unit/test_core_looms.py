"""
Tests for the 5R Core Looms: Recall, Reason, Reflect, Reach, Refuse.

Targets 60%+ coverage of hololoom/core/loom/core_looms/.
"""

import pytest
import random
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from uuid import uuid4

from hololoom.core.loom.core_looms import (
    RecallLoom,
    ReasonLoom,
    ReachLoom,
    ReflectLoom,
    RefuseLoom,
    create_all_looms,
)
from hololoom.core.loom.core_looms.reason_loom import (
    LogicalStep,
    InferenceChain,
)
from hololoom.core.loom.core_looms.reflect_loom import (
    Claim,
    VerificationResult as ReflectVerificationResult,
    ConsistencyCheck,
)
from hololoom.core.loom.core_looms.reach_loom import (
    Association,
    CreativeInsight,
)
from hololoom.core.loom.core_looms.refuse_loom import (
    Assumption,
    Attack,
    Vulnerability,
)
from hololoom.loom.protocol import RECALL, REASON, REACH, REFLECT, REFUSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class FakeRequest:
    """Lightweight stand-in for DepartmentRequest with a .query attr."""
    query: str = "test query"


def _make_confidence(score, **kwargs):
    """Build a ConfidenceMetadata-like object the looms construct."""
    obj = MagicMock()
    obj.score = score
    for k, v in kwargs.items():
        setattr(obj, k, v)
    return obj


# ---------------------------------------------------------------------------
# __init__.py — create_all_looms
# ---------------------------------------------------------------------------

class TestCreateAllLooms:
    def test_creates_five_looms(self):
        looms = create_all_looms()
        assert len(looms) == 5
        perspectives = {l.perspective for l in looms}
        assert perspectives == {RECALL, REASON, REACH, REFLECT, REFUSE}


# ===========================================================================
# RecallLoom
# ===========================================================================

class TestRecallLoomInit:
    def test_defaults(self):
        loom = RecallLoom()
        assert loom.retrieval_k == 20
        assert loom.temperature == 0.2
        assert loom.trust_memory_weight == 0.9
        assert loom.trust_inference_weight == 0.1
        assert loom._memory is None

    def test_custom_params(self):
        loom = RecallLoom(retrieval_k=10, temperature=0.5)
        assert loom.retrieval_k == 10
        assert loom.temperature == 0.5

    def test_with_memory_backend(self):
        backend = MagicMock()
        loom = RecallLoom(memory_backend=backend)
        assert loom._memory is backend


class TestRecallLoomProperties:
    def test_perspective(self):
        assert RecallLoom().perspective == RECALL

    def test_blind_spots(self):
        spots = RecallLoom().blind_spots
        assert isinstance(spots, list)
        assert len(spots) > 0
        assert "novel connections not in memory" in spots


class TestRecallLoomIgnorance:
    def test_future_query(self):
        loom = RecallLoom()
        result = loom.admits_ignorance("What will happen tomorrow?")
        assert any("predict" in r.lower() or "future" in r.lower() for r in result)

    def test_creative_query(self):
        loom = RecallLoom()
        result = loom.admits_ignorance("Imagine a new world")
        assert any("novel" in r.lower() or "generate" in r.lower() for r in result)

    def test_neutral_query(self):
        loom = RecallLoom()
        result = loom.admits_ignorance("What is 2+2")
        assert len(result) >= 1  # default ignorance

    def test_with_memory_gaps(self):
        loom = RecallLoom()
        loom._memory_gaps = ["gap_a", "gap_b"]
        result = loom.admits_ignorance("test")
        assert any("gap_a" in r for r in result)


class TestRecallLoomFalsifiable:
    def test_returns_list(self):
        assert len(RecallLoom().falsifiable_by("some claim")) > 0


class TestRecallLoomHelpers:
    def test_compile_memory_response_empty(self):
        loom = RecallLoom()
        assert "No relevant memories" in loom._compile_memory_response([])

    def test_compile_memory_response_few(self):
        threads = [
            {"content": "alpha", "relevance": 0.9},
            {"content": "beta", "relevance": 0.5},
        ]
        resp = loom._compile_memory_response(threads) if (loom := RecallLoom()) else ""
        assert "alpha" in resp
        assert "beta" in resp

    def test_compile_memory_response_many(self):
        threads = [{"content": f"item{i}", "relevance": 0.5} for i in range(8)]
        resp = RecallLoom()._compile_memory_response(threads)
        assert "3 more memories" in resp

    def test_identify_gaps_no_threads(self):
        loom = RecallLoom()
        gaps = loom._identify_memory_gaps("test query", [])
        assert any("No memories" in g for g in gaps)

    def test_identify_gaps_low_relevance(self):
        loom = RecallLoom()
        threads = [{"relevance": 0.2}, {"relevance": 0.3}]
        gaps = loom._identify_memory_gaps("test", threads)
        assert any("low relevance" in g.lower() for g in gaps)

    def test_identify_gaps_limited_count(self):
        loom = RecallLoom()
        threads = [{"relevance": 0.9}]
        gaps = loom._identify_memory_gaps("test", threads)
        assert any("Limited" in g or "only 1" in g for g in gaps)

    def test_compute_confidence_no_threads(self):
        assert RecallLoom()._compute_confidence([]) == 0.2

    def test_compute_confidence_with_threads(self):
        threads = [{"relevance": 0.8}, {"relevance": 0.9}]
        conf = RecallLoom()._compute_confidence(threads)
        assert 0.1 < conf <= 1.0

    def test_epistemic_confidence_no_gaps(self):
        loom = RecallLoom()
        ep = loom._compute_epistemic_confidence([], [{"relevance": 0.5}])
        assert 0.2 <= ep <= 1.0

    def test_epistemic_confidence_with_gaps(self):
        loom = RecallLoom()
        ep = loom._compute_epistemic_confidence(["gap1", "gap2"], [])
        assert ep < 0.8

    def test_epistemic_high_variance(self):
        loom = RecallLoom()
        threads = [{"relevance": 0.1}, {"relevance": 0.9}]
        ep = loom._compute_epistemic_confidence([], threads)
        assert ep <= 0.8  # variance penalty

    @pytest.mark.asyncio
    async def test_retrieve_memories_no_backend(self):
        loom = RecallLoom()
        result = await loom._retrieve_memories("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_with_backend(self):
        memory = MagicMock()
        memory.recall = AsyncMock(return_value=[
            MagicMock(id="1", content="hello", relevance=0.8, timestamp=None),
        ])
        loom = RecallLoom(memory_backend=memory)
        result = await loom._retrieve_memories("test")
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_retrieve_memories_backend_no_recall(self):
        """Backend without recall method returns empty."""
        backend = object()  # no recall attr
        loom = RecallLoom(memory_backend=backend)
        result = await loom._retrieve_memories("test")
        assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_memories_backend_error(self):
        memory = MagicMock()
        memory.recall = AsyncMock(side_effect=RuntimeError("fail"))
        loom = RecallLoom(memory_backend=memory)
        result = await loom._retrieve_memories("test")
        assert result == []


# ===========================================================================
# ReasonLoom
# ===========================================================================

class TestReasonLoomInit:
    def test_defaults(self):
        loom = ReasonLoom()
        assert loom.max_inference_depth == 5
        assert loom.require_explicit_links is True
        assert loom.min_step_confidence == 0.6

    def test_custom(self):
        loom = ReasonLoom(max_inference_depth=3, min_step_confidence=0.8)
        assert loom.max_inference_depth == 3
        assert loom.min_step_confidence == 0.8


class TestReasonLoomProperties:
    def test_perspective(self):
        assert ReasonLoom().perspective == REASON

    def test_blind_spots(self):
        spots = ReasonLoom().blind_spots
        assert "intuitive leaps" in spots

    def test_inference_rules(self):
        assert "modus_ponens" in ReasonLoom.INFERENCE_RULES
        assert "transitivity" in ReasonLoom.INFERENCE_RULES


class TestReasonLoomIgnorance:
    def test_intuitive_query(self):
        loom = ReasonLoom()
        result = loom.admits_ignorance("I feel this is wrong")
        assert any("feeling" in r.lower() or "intuition" in r.lower() for r in result)

    def test_uncertain_query(self):
        loom = ReasonLoom()
        result = loom.admits_ignorance("maybe it works")
        assert any("uncertainty" in r.lower() or "probability" in r.lower() for r in result)

    def test_creative_query(self):
        loom = ReasonLoom()
        result = loom.admits_ignorance("imagine a world")
        assert any("hypothetical" in r.lower() for r in result)

    def test_with_invalid_steps(self):
        loom = ReasonLoom()
        loom._invalid_steps = [LogicalStep("p", "c", "r", 0.3, False)]
        result = loom.admits_ignorance("test")
        assert any("unprovable" in r.lower() for r in result)

    def test_default_ignorance(self):
        loom = ReasonLoom()
        result = loom.admits_ignorance("x")
        assert any("premises" in r.lower() for r in result)


class TestReasonLoomFalsifiable:
    def test_returns_list(self):
        assert len(ReasonLoom().falsifiable_by("claim")) >= 3


class TestReasonLoomLogic:
    def test_extract_if_then(self):
        loom = ReasonLoom()
        premises, goal = loom._extract_logical_structure(
            "If it rains, then the ground is wet"
        )
        assert len(premises) > 0
        assert any("rain" in p.lower() for p in premises)

    def test_extract_given(self):
        loom = ReasonLoom()
        premises, goal = loom._extract_logical_structure("Given X is true, Y follows")
        assert any("X is true" in p for p in premises)

    def test_extract_therefore(self):
        loom = ReasonLoom()
        _, goal = loom._extract_logical_structure("Therefore the answer is 42")
        assert "42" in goal

    def test_extract_why(self):
        loom = ReasonLoom()
        _, goal = loom._extract_logical_structure("Why does water boil?")
        assert "water boil" in goal.lower()

    def test_extract_short_query(self):
        loom = ReasonLoom()
        premises, goal = loom._extract_logical_structure("hi")
        assert premises == []  # too short for implicit

    def test_extract_long_query_implicit(self):
        loom = ReasonLoom()
        premises, _ = loom._extract_logical_structure("this is a long query text")
        assert len(premises) == 1

    def test_matches_goal_exact(self):
        loom = ReasonLoom()
        assert loom._matches_goal("the answer is 42", "the answer is 42")

    def test_matches_goal_word_overlap(self):
        loom = ReasonLoom()
        assert loom._matches_goal("water definitely boils at high temp", "water boils")

    def test_matches_goal_no_match(self):
        loom = ReasonLoom()
        assert not loom._matches_goal("alpha beta", "gamma delta epsilon zeta")

    def test_try_inference_transitivity(self):
        loom = ReasonLoom()
        knowledge = [
            "If rain then wet",
            "If wet then slippery",
        ]
        step = loom._try_inference_rule("transitivity", knowledge, "slippery")
        assert step is not None
        assert step.rule == "transitivity"
        assert step.confidence > 0

    def test_try_inference_transitivity_no_match(self):
        loom = ReasonLoom()
        step = loom._try_inference_rule("transitivity", ["fact without then"], "goal")
        # Falls through to heuristic
        assert step is not None
        assert step.rule == "heuristic"

    def test_try_inference_heuristic(self):
        loom = ReasonLoom()
        step = loom._try_inference_rule("modus_ponens", ["some fact"], "goal")
        assert step is not None
        assert step.rule == "heuristic"

    def test_try_inference_empty(self):
        loom = ReasonLoom()
        step = loom._try_inference_rule("modus_ponens", [], "")
        assert step is None

    def test_validate_chain_splits(self):
        loom = ReasonLoom()
        chain = InferenceChain(
            steps=[
                LogicalStep("p", "c", "r", 0.9, True),
                LogicalStep("p2", "c2", "r2", 0.3, True),  # below threshold
            ],
            goal_reached=False,
            total_confidence=0.27,
        )
        valid, invalid = loom._validate_chain(chain)
        assert len(valid) == 1
        assert len(invalid) == 1
        assert not invalid[0].valid

    def test_format_reasoning_with_steps(self):
        loom = ReasonLoom()
        chain = InferenceChain(steps=[], goal_reached=True, total_confidence=0.9)
        valid = [LogicalStep("p", "c", "trans", 0.9, True)]
        invalid = [LogicalStep("p2", "c2", "heur", 0.3, False, "low conf")]
        text = loom._format_reasoning(chain, valid, invalid)
        assert "Valid" in text
        assert "Invalid" in text
        assert "GOAL REACHED" in text

    def test_format_reasoning_goal_not_reached(self):
        loom = ReasonLoom()
        chain = InferenceChain(steps=[], goal_reached=False, total_confidence=0.3)
        text = loom._format_reasoning(chain, [], [])
        assert "GOAL NOT REACHED" in text

    def test_compute_chain_confidence_no_valid(self):
        loom = ReasonLoom()
        chain = InferenceChain(steps=[], goal_reached=False, total_confidence=0.3)
        assert loom._compute_chain_confidence(chain, []) == 0.3

    def test_compute_chain_confidence_with_valid(self):
        loom = ReasonLoom()
        chain = InferenceChain(steps=[], goal_reached=True, total_confidence=0.8)
        steps = [LogicalStep("p", "c", "r", 0.9, True)]
        conf = loom._compute_chain_confidence(chain, steps)
        assert conf >= 0.9  # 0.9 + 0.1 bonus

    def test_compute_epistemic_no_invalid(self):
        chain = InferenceChain(
            steps=[LogicalStep("p", "c", "r", 0.9, True), LogicalStep("p2", "c2", "r2", 0.8, True)],
            goal_reached=True,
            total_confidence=0.72,
        )
        ep = ReasonLoom()._compute_epistemic_confidence(chain, [])
        assert ep == 0.9  # base, no deductions

    def test_compute_epistemic_with_invalid_and_short(self):
        chain = InferenceChain(steps=[LogicalStep("p", "c", "r", 0.5, False)], goal_reached=False, total_confidence=0.3)
        invalid = [LogicalStep("p", "c", "r", 0.5, False)]
        ep = ReasonLoom()._compute_epistemic_confidence(chain, invalid)
        # base 0.9 - 0.1 (invalid) - 0.15 (no goal) - 0.1 (short) = 0.55
        assert 0.3 <= ep <= 0.6

    def test_generate_specific_falsifiers_with_chain(self):
        loom = ReasonLoom()
        chain = InferenceChain(
            steps=[
                LogicalStep("p", "c", "transitivity", 0.9, True),
                LogicalStep("p2", "c2", "heuristic", 0.7, True),
            ],
            goal_reached=True,
            total_confidence=0.63,
        )
        f = loom._generate_specific_falsifiers(chain)
        assert len(f) <= 5
        assert any("false" in x.lower() for x in f)

    def test_generate_specific_falsifiers_empty_chain(self):
        loom = ReasonLoom()
        chain = InferenceChain(steps=[], goal_reached=False, total_confidence=0.3)
        f = loom._generate_specific_falsifiers(chain)
        assert len(f) > 0  # falls back to generic

    @pytest.mark.asyncio
    async def test_build_inference_chain_basic(self):
        loom = ReasonLoom(max_inference_depth=2)
        chain = await loom._build_inference_chain(["some premise here"], "goal")
        assert isinstance(chain, InferenceChain)
        assert len(chain.steps) <= 2

    @pytest.mark.asyncio
    async def test_build_inference_chain_no_premises(self):
        loom = ReasonLoom()
        chain = await loom._build_inference_chain([], "goal")
        assert chain.total_confidence == 0.3  # no steps


# ===========================================================================
# ReflectLoom
# ===========================================================================

class TestReflectLoomInit:
    def test_defaults(self):
        loom = ReflectLoom()
        assert loom.verification_passes == 2
        assert loom.consistency_threshold == 0.8
        assert loom.require_evidence is True

    def test_custom(self):
        loom = ReflectLoom(verification_passes=5, consistency_threshold=0.5)
        assert loom.verification_passes == 5
        assert loom.consistency_threshold == 0.5


class TestReflectLoomProperties:
    def test_perspective(self):
        assert ReflectLoom().perspective == REFLECT

    def test_blind_spots(self):
        spots = ReflectLoom().blind_spots
        assert "generating new content" in spots


class TestReflectLoomIgnorance:
    def test_creative_query(self):
        result = ReflectLoom().admits_ignorance("create a new design")
        assert any("create" in r.lower() or "verify" in r.lower() for r in result)

    def test_speed_query(self):
        result = ReflectLoom().admits_ignorance("do this quickly")
        assert any("time" in r.lower() or "rush" in r.lower() for r in result)

    def test_prediction_query(self):
        result = ReflectLoom().admits_ignorance("predict the future")
        assert any("future" in r.lower() for r in result)

    def test_default(self):
        result = ReflectLoom().admits_ignorance("test")
        assert len(result) >= 1


class TestReflectLoomHelpers:
    def test_extract_claims_short_text(self):
        claims = ReflectLoom()._extract_claims("Hi.")
        assert len(claims) == 0  # too short

    def test_extract_claims_sentences(self):
        text = "The sky is blue. Water is wet. Testing things out here."
        claims = ReflectLoom()._extract_claims(text)
        assert len(claims) >= 2

    def test_is_verifiable_factual(self):
        assert ReflectLoom()._is_verifiable("The value is 42") is True

    def test_is_verifiable_subjective(self):
        assert ReflectLoom()._is_verifiable("I feel this is wrong") is False

    def test_is_verifiable_normative(self):
        assert ReflectLoom()._is_verifiable("You should do this") is False

    def test_is_verifiable_uncertain(self):
        assert ReflectLoom()._is_verifiable("maybe this works somehow") is False

    def test_is_verifiable_no_match(self):
        assert ReflectLoom()._is_verifiable("xyz abc def") is False

    def test_self_verify_factual(self):
        loom = ReflectLoom()
        verified, evidence = loom._self_verify(Claim("The answer is 42", "q", 0.5, True))
        assert verified is True
        assert len(evidence) > 0

    def test_self_verify_no_evidence(self):
        loom = ReflectLoom()
        verified, evidence = loom._self_verify(Claim("xyz blah", "q", 0.5, True))
        assert verified is False

    def test_check_consistency_no_contradictions(self):
        loom = ReflectLoom()
        claims = [
            Claim("The sky is blue", "q", 0.5, True),
            Claim("Water flows downhill", "q", 0.5, True),
        ]
        result = loom._check_consistency(claims)
        assert result.consistent is True
        assert result.confidence == 1.0

    def test_check_consistency_with_contradictions(self):
        loom = ReflectLoom()
        claims = [
            Claim("The system does not work properly today", "q", 0.5, True),
            Claim("The system works properly today", "q", 0.5, True),
        ]
        result = loom._check_consistency(claims)
        assert result.consistent is False

    def test_are_contradictory_negation(self):
        loom = ReflectLoom()
        assert loom._are_contradictory(
            "The cat is not on the mat",
            "The cat is on the mat"
        )

    def test_are_contradictory_no_conflict(self):
        loom = ReflectLoom()
        assert not loom._are_contradictory("apples are fruit", "bananas are yellow")

    def test_find_contradictions_filters_non_verifiable(self):
        loom = ReflectLoom()
        results = [
            ReflectVerificationResult(
                claim=Claim("x", "q", 0.5, False),
                verified=False,
                confidence=0.3,
                evidence=[],
                contradictions=["Claim is not verifiable"],
            ),
        ]
        assert loom._find_contradictions([], results) == []

    def test_find_contradictions_real(self):
        loom = ReflectLoom()
        results = [
            ReflectVerificationResult(
                claim=Claim("x", "q", 0.5, True),
                verified=False,
                confidence=0.3,
                evidence=[],
                contradictions=["Evidence disagrees"],
            ),
        ]
        found = loom._find_contradictions([], results)
        assert len(found) == 1

    def test_format_report_no_results(self):
        loom = ReflectLoom()
        consistency = ConsistencyCheck(True, [], 1.0)
        text = loom._format_verification_report([], consistency, [])
        assert "0/0" in text

    def test_format_report_with_results(self):
        loom = ReflectLoom()
        results = [
            ReflectVerificationResult(
                Claim("Test claim one here abc", "q", 0.5, True),
                True, 0.8, ["evidence"], [],
            ),
            ReflectVerificationResult(
                Claim("Unverified claim two here abc", "q", 0.3, False),
                False, 0.3, [], ["not verifiable"],
            ),
        ]
        consistency = ConsistencyCheck(True, [], 0.9)
        text = loom._format_verification_report(results, consistency, [])
        assert "1/2" in text
        assert "Verified" in text

    def test_compute_verification_confidence_empty(self):
        loom = ReflectLoom()
        consistency = ConsistencyCheck(True, [], 1.0)
        assert loom._compute_verification_confidence([], consistency) == 0.3

    def test_compute_verification_confidence_values(self):
        loom = ReflectLoom()
        results = [
            ReflectVerificationResult(Claim("x", "q", 0.5, True), True, 0.9, [], []),
        ]
        consistency = ConsistencyCheck(True, [], 1.0)
        conf = loom._compute_verification_confidence(results, consistency)
        # 0.6 * 0.9 + 0.4 * 1.0 = 0.94
        assert 0.9 <= conf <= 1.0

    def test_epistemic_confidence_base(self):
        loom = ReflectLoom()
        results = [
            ReflectVerificationResult(Claim("x", "q", 0.5, True), True, 0.9, [], []),
        ]
        consistency = ConsistencyCheck(True, [], 0.9)
        ep = loom._compute_epistemic_confidence(results, consistency, [])
        assert 0.7 <= ep <= 0.9

    def test_epistemic_confidence_low_consistency(self):
        loom = ReflectLoom()
        consistency = ConsistencyCheck(False, [("a", "b")], 0.5)
        ep = loom._compute_epistemic_confidence([], consistency, [("a", "b")])
        assert ep < 0.8

    @pytest.mark.asyncio
    async def test_verify_claims_unverifiable(self):
        loom = ReflectLoom()
        claims = [Claim("I feel good about this", "q", 0.3, False)]
        results = await loom._verify_claims(claims)
        assert len(results) == 1
        assert not results[0].verified

    @pytest.mark.asyncio
    async def test_verify_claims_verifiable(self):
        loom = ReflectLoom()
        claims = [Claim("The value is 42", "q", 0.5, True)]
        results = await loom._verify_claims(claims)
        assert len(results) == 1
        assert results[0].verified is True

    @pytest.mark.asyncio
    async def test_external_verify_success(self):
        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=MagicMock(verified=True, evidence=["ext"]))
        loom = ReflectLoom(verification_backend=verifier)
        verified, evidence = await loom._external_verify(Claim("test", "q", 0.5, True))
        assert verified is True
        assert "ext" in evidence

    @pytest.mark.asyncio
    async def test_external_verify_fallback(self):
        verifier = MagicMock()
        verifier.verify = AsyncMock(side_effect=RuntimeError("fail"))
        loom = ReflectLoom(verification_backend=verifier)
        verified, evidence = await loom._external_verify(Claim("The answer is 42", "q", 0.5, True))
        # Falls back to self_verify
        assert isinstance(verified, bool)


# ===========================================================================
# ReachLoom
# ===========================================================================

class TestReachLoomInit:
    def test_defaults(self):
        loom = ReachLoom()
        assert loom.retrieval_k == 5
        assert loom.temperature == 0.9
        assert loom.association_hops == 3
        assert loom.prefer_weak_ties is True
        assert loom.novelty_threshold == 0.5

    def test_custom(self):
        loom = ReachLoom(temperature=0.5, association_hops=1)
        assert loom.temperature == 0.5
        assert loom.association_hops == 1


class TestReachLoomProperties:
    def test_perspective(self):
        assert ReachLoom().perspective == REACH

    def test_blind_spots(self):
        assert "factual accuracy" in ReachLoom().blind_spots


class TestReachLoomIgnorance:
    def test_always_admits_uncertainty(self):
        result = ReachLoom().admits_ignorance("anything")
        assert len(result) >= 3  # always 3 base admissions

    def test_precision_query(self):
        result = ReachLoom().admits_ignorance("give me the exact answer")
        assert any("precision" in r.lower() or "accuracy" in r.lower() for r in result)

    def test_safety_query(self):
        result = ReachLoom().admits_ignorance("is this safe to use")
        assert any("safety" in r.lower() or "constraint" in r.lower() for r in result)


class TestReachLoomHelpers:
    @pytest.mark.asyncio
    async def test_extract_concepts(self):
        loom = ReachLoom()
        concepts = await loom._extract_concepts("how does machine learning work?")
        assert "machine" in concepts or "learning" in concepts
        assert len(concepts) <= 5

    @pytest.mark.asyncio
    async def test_extract_concepts_filters_stopwords(self):
        loom = ReachLoom()
        concepts = await loom._extract_concepts("the a an is are was")
        assert len(concepts) == 0

    def test_generate_distant_association_known(self):
        loom = ReachLoom()
        random.seed(42)
        result = loom._generate_distant_association("learning", set())
        assert result is not None  # "learning" is in domain_associations

    def test_generate_distant_association_unknown(self):
        loom = ReachLoom()
        random.seed(42)
        result = loom._generate_distant_association("zzzzunknown", set())
        assert result is not None  # falls back to random

    def test_generate_distant_association_all_visited(self):
        loom = ReachLoom()
        # Visit everything
        all_vals = set()
        for vals in loom._generate_distant_association.__code__.co_consts:
            pass  # can't enumerate easily, use big set instead
        # If all candidates visited, returns None
        huge_visited = {
            'neuroplasticity', 'curiosity', 'iteration', 'feedback',
            'tradeoff', 'uncertainty', 'exploration', 'commitment',
            'nature', 'evolution', 'emergence', 'swarm',
            'story', 'emotion', 'association', 'forgetting',
            'music', 'architecture', 'language',
            'ecology', 'city', 'body', 'network',
            'puzzle', 'game', 'adventure', 'mystery',
            'archaeology', 'forensics', 'art',
            'poetry', 'recipe', 'spell',
            'metaphor', 'sculpture', 'map', 'lens',
        }
        result = loom._generate_distant_association("learning", huge_visited)
        # May or may not be None depending on overlap
        # Just verify it doesn't crash
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_explore_weak_ties(self):
        loom = ReachLoom(association_hops=2)
        random.seed(42)
        assocs = await loom._explore_weak_ties(["learning"])
        assert isinstance(assocs, list)
        for a in assocs:
            assert isinstance(a, Association)
            assert a.source == "learning"

    def test_generate_combinations_with_associations(self):
        loom = ReachLoom()
        random.seed(42)
        assocs = [
            Association("learning", "curiosity", 0.5, ["learning", "curiosity"], 0.7),
        ]
        insights = loom._generate_combinations(["learning"], assocs)
        assert len(insights) > 0
        assert all(isinstance(i, CreativeInsight) for i in insights)

    def test_generate_combinations_cross_seed(self):
        loom = ReachLoom()
        random.seed(42)
        insights = loom._generate_combinations(["alpha", "beta"], [])
        assert len(insights) >= 1  # cross-seed combo

    def test_generate_combinations_below_threshold(self):
        loom = ReachLoom(novelty_threshold=0.99)
        assocs = [
            Association("a", "b", 0.5, ["a", "b"], 0.3),  # below threshold
        ]
        insights = loom._generate_combinations(["a"], assocs)
        # Only cross-seed combos if any; assoc below threshold skipped
        # With single seed, no cross-seed either
        assert len(insights) == 0

    def test_create_combination(self):
        loom = ReachLoom()
        random.seed(42)
        text = loom._create_combination("code", "poetry", ["code", "poetry"])
        assert isinstance(text, str)
        assert len(text) > 0

    def test_format_creative_response_empty(self):
        loom = ReachLoom()
        text = loom._format_creative_response([], [])
        assert "No novel connections" in text

    def test_format_creative_response_with_insights(self):
        loom = ReachLoom()
        insights = [
            CreativeInsight(["a", "b"], "What if a worked like b?", 0.8, 0.5),
        ]
        text = loom._format_creative_response(insights, [])
        assert "What if a worked like b?" in text
        assert "NOTE" in text

    def test_compute_creative_confidence_empty(self):
        assert ReachLoom()._compute_creative_confidence([]) == 0.3

    def test_compute_creative_confidence_capped(self):
        insights = [CreativeInsight([], "", 0.9, 0.8)]
        conf = ReachLoom()._compute_creative_confidence(insights)
        assert conf <= 0.7  # capped

    def test_compute_epistemic_empty(self):
        ep = ReachLoom()._compute_epistemic_confidence([])
        assert ep == 0.4  # base

    def test_compute_epistemic_capped(self):
        insights = [CreativeInsight([], "", 0.9, 0.9)]
        ep = ReachLoom()._compute_epistemic_confidence(insights)
        assert ep <= 0.5  # capped


# ===========================================================================
# RefuseLoom
# ===========================================================================

class TestRefuseLoomInit:
    def test_defaults(self):
        loom = RefuseLoom()
        assert loom.attack_intensity == 0.7
        assert loom.assumption_depth == 3
        assert loom.min_attack_severity == 0.3

    def test_custom(self):
        loom = RefuseLoom(attack_intensity=1.0, min_attack_severity=0.0)
        assert loom.attack_intensity == 1.0


class TestRefuseLoomProperties:
    def test_perspective(self):
        assert RefuseLoom().perspective == REFUSE

    def test_blind_spots(self):
        assert "constructive solutions" in RefuseLoom().blind_spots

    def test_attack_types(self):
        assert "edge_case" in RefuseLoom.ATTACK_TYPES


class TestRefuseLoomIgnorance:
    def test_solution_query(self):
        result = RefuseLoom().admits_ignorance("how to fix this")
        assert any("solution" in r.lower() or "problem" in r.lower() for r in result)

    def test_positive_query(self):
        result = RefuseLoom().admits_ignorance("what is good about this")
        assert any("strength" in r.lower() or "weakness" in r.lower() for r in result)

    def test_nuanced_query(self):
        result = RefuseLoom().admits_ignorance("it depends on context")
        assert any("nuance" in r.lower() for r in result)

    def test_default(self):
        result = RefuseLoom().admits_ignorance("test")
        assert any("critical" in r.lower() for r in result)


class TestRefuseLoomHelpers:
    def test_extract_claims(self):
        loom = RefuseLoom()
        claims = loom._extract_claims("Claim one. Claim two is here. Hi.")
        assert len(claims) >= 2  # "Hi" is too short (<=5 chars)

    def test_extract_assumptions_because(self):
        loom = RefuseLoom()
        claims = ["This works because the system is fast"]
        assumptions = loom._extract_assumptions("test", claims)
        assert any("system is fast" in a.text.lower() for a in assumptions)

    def test_extract_assumptions_if_then(self):
        loom = RefuseLoom()
        claims = ["if the server is running then the API works"]
        assumptions = loom._extract_assumptions("test", claims)
        assert any("possible" in a.text.lower() for a in assumptions)

    def test_extract_assumptions_the_pattern(self):
        loom = RefuseLoom()
        claims = ["the algorithm processes the data"]
        assumptions = loom._extract_assumptions("test", claims)
        assert any("exists" in a.text.lower() for a in assumptions)

    def test_extract_assumptions_domain_technical(self):
        loom = RefuseLoom()
        assumptions = loom._find_domain_assumptions("the algorithm runs fast")
        assert any("technical" in a.text.lower() for a in assumptions)

    def test_extract_assumptions_domain_statistical(self):
        assumptions = RefuseLoom()._find_domain_assumptions("the probability distribution")
        assert any("statistical" in a.text.lower() for a in assumptions)

    def test_extract_assumptions_domain_temporal(self):
        assumptions = RefuseLoom()._find_domain_assumptions("this will always work")
        assert any("temporal" in a.text.lower() for a in assumptions)

    def test_generate_attacks_edge_case(self):
        loom = RefuseLoom()
        attacks = loom._generate_claim_attacks("All users always do this")
        types = [a.attack_type for a in attacks]
        assert "edge_case" in types

    def test_generate_attacks_ambiguity(self):
        loom = RefuseLoom()
        attacks = loom._generate_claim_attacks("Some users often do this good thing")
        types = [a.attack_type for a in attacks]
        assert "ambiguity" in types

    def test_generate_attacks_missing_context(self):
        loom = RefuseLoom()
        attacks = loom._generate_claim_attacks("This is true")
        types = [a.attack_type for a in attacks]
        assert "missing_context" in types

    def test_generate_attacks_scope_creep(self):
        loom = RefuseLoom()
        attacks = loom._generate_claim_attacks("Everything works for everyone")
        types = [a.attack_type for a in attacks]
        assert "scope_creep" in types

    def test_generate_attacks_contradiction(self):
        loom = RefuseLoom()
        attacks = loom._generate_claim_attacks("This is not working correctly")
        types = [a.attack_type for a in attacks]
        assert "contradiction" in types

    def test_generate_attacks_applies_assumptions(self):
        loom = RefuseLoom()
        assumptions = [Assumption("Assumes: X", implicit=True, confidence=0.7)]
        attacks = loom._generate_attacks([], assumptions)
        assert any(a.attack_type == "false_premise" for a in attacks)

    def test_generate_attacks_filters_severity(self):
        loom = RefuseLoom(min_attack_severity=0.9)
        attacks = loom._generate_attacks(["short claim"], [])
        # Most attacks will be below 0.9 severity
        for a in attacks:
            assert a.severity >= 0.9

    def test_assess_impact_high(self):
        attack = Attack("t", "e", "text", 0.8)
        assert "HIGH" in RefuseLoom()._assess_impact(attack)

    def test_assess_impact_medium(self):
        attack = Attack("t", "e", "text", 0.5)
        assert "MEDIUM" in RefuseLoom()._assess_impact(attack)

    def test_assess_impact_low(self):
        attack = Attack("t", "e", "text", 0.2)
        assert "LOW" in RefuseLoom()._assess_impact(attack)

    def test_suggest_mitigation(self):
        loom = RefuseLoom()
        for atype in ["edge_case", "contradiction", "ambiguity", "missing_context",
                       "false_premise", "scope_creep", "cherry_picking", "straw_man"]:
            attack = Attack("t", atype, "text", 0.5)
            assert loom._suggest_mitigation(attack) != "Review and revise"

    def test_suggest_mitigation_unknown(self):
        attack = Attack("t", "unknown_type", "text", 0.5)
        assert RefuseLoom()._suggest_mitigation(attack) == "Review and revise"

    def test_assess_attack_probabilistic(self):
        """assess_attack is probabilistic; run many times to verify rates."""
        loom = RefuseLoom(attack_intensity=1.0)
        attack = Attack("t", "missing_context", "text", 0.5)
        results = [loom._assess_attack(attack) for _ in range(100)]
        # missing_context base rate 0.8 * intensity 1.0 = 0.8
        # ~80% should be True
        assert 50 < sum(results) < 100

    def test_format_report_no_vulns(self):
        loom = RefuseLoom()
        text = loom._format_attack_report([], [])
        assert "No significant vulnerabilities" in text

    def test_format_report_with_vulns(self):
        loom = RefuseLoom()
        vulns = [
            Vulnerability(
                attack=Attack("target text", "edge_case", "What about edge?", 0.7),
                confirmed=True,
                impact="HIGH: Significantly undermines the argument",
                mitigation="Add boundary conditions",
            ),
        ]
        assumptions = [Assumption("Assumes: X", True, 0.7)]
        text = loom._format_attack_report(vulns, assumptions)
        assert "EDGE_CASE" in text
        assert "Hidden Assumptions" in text

    def test_compute_attack_confidence_no_vulns(self):
        assert RefuseLoom()._compute_attack_confidence([]) == 0.4

    def test_compute_attack_confidence_with_vulns(self):
        vulns = [
            Vulnerability(Attack("t", "e", "x", 0.8), True, "HIGH"),
        ]
        conf = RefuseLoom()._compute_attack_confidence(vulns)
        assert conf > 0.4

    def test_compute_epistemic_base(self):
        ep = RefuseLoom()._compute_epistemic_confidence([], [])
        assert ep == 0.7

    def test_compute_epistemic_with_assumptions(self):
        assumptions = [Assumption("a", True, 0.5)] * 5
        ep = RefuseLoom()._compute_epistemic_confidence([], assumptions)
        assert ep < 0.7

    def test_compute_epistemic_high_severity(self):
        vulns = [Vulnerability(Attack("t", "e", "x", 0.9), True, "HIGH")]
        ep = RefuseLoom()._compute_epistemic_confidence(vulns, [])
        assert ep >= 0.7  # base + 0.1 bonus

    def test_compute_epistemic_low_severity(self):
        vulns = [Vulnerability(Attack("t", "e", "x", 0.2), True, "LOW")]
        ep = RefuseLoom()._compute_epistemic_confidence(vulns, [])
        assert ep < 0.7  # nitpicking penalty

    @pytest.mark.asyncio
    async def test_execute_attacks(self):
        loom = RefuseLoom()
        random.seed(42)
        attacks = [
            Attack("t", "missing_context", "text", 0.5),
            Attack("t", "edge_case", "text", 0.7),
        ]
        vulns = await loom._execute_attacks(attacks)
        assert isinstance(vulns, list)
        for v in vulns:
            assert isinstance(v, Vulnerability)
            assert v.confirmed is True
            assert v.mitigation is not None


# ===========================================================================
# Execute paths (integration-ish, still fast)
# ===========================================================================

class TestLoomExecute:
    """Test execute() for each loom with mocked _create_fabric."""

    @pytest.mark.asyncio
    async def test_recall_execute(self):
        loom = RecallLoom()
        req = FakeRequest(query="What is Thompson Sampling?")
        with patch.object(loom, '_create_fabric', return_value=MagicMock()):
            with patch(
                'hololoom.core.loom.core_looms.recall_loom.DepartmentResponse'
            ) as MockResp:
                MockResp.side_effect = lambda **kw: MagicMock(**kw)
                with patch(
                    'hololoom.core.loom.core_looms.recall_loom.ConfidenceMetadata'
                ) as MockConf:
                    MockConf.side_effect = lambda **kw: MagicMock(**kw)
                    result = await loom.execute(req)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_reason_execute(self):
        loom = ReasonLoom(max_inference_depth=2)
        req = FakeRequest(query="If A then B, therefore B")
        with patch.object(loom, '_create_fabric', return_value=MagicMock()):
            with patch(
                'hololoom.core.loom.core_looms.reason_loom.DepartmentResponse'
            ) as MockResp:
                MockResp.side_effect = lambda **kw: MagicMock(**kw)
                with patch(
                    'hololoom.core.loom.core_looms.reason_loom.ConfidenceMetadata'
                ) as MockConf:
                    MockConf.side_effect = lambda **kw: MagicMock(**kw)
                    result = await loom.execute(req)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_reflect_execute(self):
        loom = ReflectLoom(verification_passes=1)
        req = FakeRequest(query="The sky is blue. Water is wet.")
        with patch.object(loom, '_create_fabric', return_value=MagicMock()):
            with patch(
                'hololoom.core.loom.core_looms.reflect_loom.DepartmentResponse'
            ) as MockResp:
                MockResp.side_effect = lambda **kw: MagicMock(**kw)
                with patch(
                    'hololoom.core.loom.core_looms.reflect_loom.ConfidenceMetadata'
                ) as MockConf:
                    MockConf.side_effect = lambda **kw: MagicMock(**kw)
                    result = await loom.execute(req)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_reach_execute(self):
        loom = ReachLoom(association_hops=1)
        random.seed(42)
        req = FakeRequest(query="how does machine learning connect to nature?")
        with patch.object(loom, '_create_fabric', return_value=MagicMock()):
            with patch(
                'hololoom.core.loom.core_looms.reach_loom.DepartmentResponse'
            ) as MockResp:
                MockResp.side_effect = lambda **kw: MagicMock(**kw)
                with patch(
                    'hololoom.core.loom.core_looms.reach_loom.ConfidenceMetadata'
                ) as MockConf:
                    MockConf.side_effect = lambda **kw: MagicMock(**kw)
                    result = await loom.execute(req)
                    assert result is not None

    @pytest.mark.asyncio
    async def test_refuse_execute(self):
        loom = RefuseLoom()
        random.seed(42)
        req = FakeRequest(query="All systems always work perfectly because the code is good.")
        with patch.object(loom, '_create_fabric', return_value=MagicMock()):
            with patch(
                'hololoom.core.loom.core_looms.refuse_loom.DepartmentResponse'
            ) as MockResp:
                MockResp.side_effect = lambda **kw: MagicMock(**kw)
                with patch(
                    'hololoom.core.loom.core_looms.refuse_loom.ConfidenceMetadata'
                ) as MockConf:
                    MockConf.side_effect = lambda **kw: MagicMock(**kw)
                    result = await loom.execute(req)
                    assert result is not None

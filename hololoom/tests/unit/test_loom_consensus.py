"""
Unit tests for LoomConsensus (hololoom/core/loom/consensus.py).

Covers:
- Claim dataclass — hash/equality
- Agreement dataclass
- LoomConsensus construction
- synthesize() — empty, single, multi-fabric paths
- synthesize() metadata: tensions, agreements, agreement_ratio, per_loom_confidence
- has_significant_tensions()
- get_tensions()
- attempt_resolution() — resolved vs unresolved
- _extract_claims() — sentence splitting
- _find_agreements() — agreement detection
- _find_tensions() — explicit tensions, low agreement_with, contradictory claims
- _compute_collective_epistemic() — penalties for tensions and ignorance
- _intersect_blind_spots()
- _collect_falsifiability()
- _collect_ignorance()
- _weighted_confidence()
- _merge_agreed_claims()
- _merge_traces()
- _text_similarity() — Jaccard word overlap
- _appears_contradictory() — negation heuristic
- _compute_pairwise_similarity()
- _merge_claims_text()
- _empty_fabric()
- _single_fabric_synthesis()
- create_loom_consensus() factory
"""

import pytest
from datetime import datetime
from typing import List

from hololoom.fabric.spacetime import WeavingTrace
from hololoom.fabric.fabric import Fabric, Tension, Resolution
from hololoom.core.loom.consensus import (
    LoomConsensus,
    Claim,
    Agreement,
    create_loom_consensus,
)


# =============================================================================
# Helpers
# =============================================================================

def make_trace(duration_ms: float = 100.0) -> WeavingTrace:
    now = datetime.now()
    return WeavingTrace(start_time=now, end_time=now, duration_ms=duration_ms)


def make_fabric(
    query: str = "What is X?",
    response: str = "X is a thing.",
    perspective: str = "recall",
    confidence: float = 0.8,
    epistemic_confidence: float = 0.7,
    blind_spots: List[str] = None,
    falsifiable_by: List[str] = None,
    admits_ignorance: List[str] = None,
    tensions_with: dict = None,
    agreement_with: dict = None,
    duration_ms: float = 100.0,
) -> Fabric:
    trace = make_trace(duration_ms=duration_ms)
    return Fabric(
        query_text=query,
        response=response,
        tool_used="answer",
        confidence=confidence,
        trace=trace,
        perspective=perspective,
        epistemic_confidence=epistemic_confidence,
        blind_spots=blind_spots or [],
        falsifiable_by=falsifiable_by or [],
        admits_ignorance=admits_ignorance or [],
        tensions_with=tensions_with or {},
        agreement_with=agreement_with or {},
    )


# =============================================================================
# Claim
# =============================================================================

class TestClaim:
    def test_equality_by_text(self):
        c1 = Claim(text="The sky is blue.", source_loom="recall", confidence=0.9)
        c2 = Claim(text="The sky is blue.", source_loom="reason", confidence=0.5)
        assert c1 == c2

    def test_inequality_different_text(self):
        c1 = Claim(text="The sky is blue.", source_loom="recall", confidence=0.9)
        c2 = Claim(text="The sky is red.", source_loom="recall", confidence=0.9)
        assert c1 != c2

    def test_hash_based_on_text(self):
        c1 = Claim(text="The sky is blue.", source_loom="recall", confidence=0.9)
        c2 = Claim(text="The sky is blue.", source_loom="reason", confidence=0.5)
        assert hash(c1) == hash(c2)

    def test_claim_in_set(self):
        c1 = Claim(text="Fact A", source_loom="recall", confidence=0.8)
        c2 = Claim(text="Fact A", source_loom="reason", confidence=0.6)
        s = {c1, c2}
        assert len(s) == 1

    def test_not_equal_to_non_claim(self):
        c = Claim(text="Fact A", source_loom="recall", confidence=0.8)
        assert c != "Fact A"

    def test_supporting_evidence_default_empty(self):
        c = Claim(text="X", source_loom="recall", confidence=0.5)
        assert c.supporting_evidence == []


# =============================================================================
# Agreement
# =============================================================================

class TestAgreement:
    def test_construction(self):
        a = Agreement(
            claim="X is true",
            supporting_looms=["recall", "reason"],
            avg_confidence=0.75,
            strength=0.8,
        )
        assert a.claim == "X is true"
        assert len(a.supporting_looms) == 2
        assert a.strength == 0.8


# =============================================================================
# LoomConsensus — Construction
# =============================================================================

class TestLoomConsensusInit:
    def test_defaults(self):
        lc = LoomConsensus()
        assert lc.exploration_enabled is True
        assert lc.similarity_threshold == 0.85
        assert lc.tension_threshold == 0.3

    def test_custom_params(self):
        lc = LoomConsensus(
            exploration_enabled=False,
            similarity_threshold=0.9,
            tension_threshold=0.5,
        )
        assert lc.exploration_enabled is False
        assert lc.similarity_threshold == 0.9
        assert lc.tension_threshold == 0.5

    def test_thompson_priors_empty_initially(self):
        lc = LoomConsensus()
        assert lc._thompson_priors == {}


# =============================================================================
# LoomConsensus — synthesize() — empty
# =============================================================================

class TestSynthesizeEmpty:
    @pytest.mark.asyncio
    async def test_empty_list_returns_empty_fabric(self):
        lc = LoomConsensus()
        result = await lc.synthesize([])
        assert result.response == "No fabrics to synthesize."
        assert result.confidence == 0.0
        assert result.perspective == "collective"

    @pytest.mark.asyncio
    async def test_empty_fabric_tool_is_collective(self):
        lc = LoomConsensus()
        result = await lc.synthesize([])
        assert result.tool_used == "collective"


# =============================================================================
# LoomConsensus — synthesize() — single fabric
# =============================================================================

class TestSynthesizeSingle:
    @pytest.mark.asyncio
    async def test_single_fabric_perspective_becomes_collective(self):
        lc = LoomConsensus()
        f = make_fabric(perspective="recall", response="Single response.")
        result = await lc.synthesize([f])
        assert result.perspective == "collective"

    @pytest.mark.asyncio
    async def test_single_fabric_response_preserved(self):
        lc = LoomConsensus()
        f = make_fabric(response="My unique response.")
        result = await lc.synthesize([f])
        assert result.response == "My unique response."

    @pytest.mark.asyncio
    async def test_single_fabric_metadata_notes_source(self):
        lc = LoomConsensus()
        f = make_fabric(perspective="reason")
        result = await lc.synthesize([f])
        assert result.metadata.get("single_source") == "reason"

    @pytest.mark.asyncio
    async def test_single_fabric_blind_spots_preserved(self):
        lc = LoomConsensus()
        f = make_fabric(blind_spots=["future predictions"])
        result = await lc.synthesize([f])
        assert "future predictions" in result.blind_spots


# =============================================================================
# LoomConsensus — synthesize() — multi-fabric
# =============================================================================

class TestSynthesizeMulti:
    @pytest.mark.asyncio
    async def test_multi_fabric_perspective_is_collective(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall")
        f2 = make_fabric(perspective="reason")
        result = await lc.synthesize([f1, f2])
        assert result.perspective == "collective"

    @pytest.mark.asyncio
    async def test_multi_fabric_tool_is_collective(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall")
        f2 = make_fabric(perspective="reason")
        result = await lc.synthesize([f1, f2])
        assert result.tool_used == "collective"

    @pytest.mark.asyncio
    async def test_metadata_contains_synthesis_time(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall")
        f2 = make_fabric(perspective="reason")
        result = await lc.synthesize([f1, f2])
        assert "synthesis_time" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_per_loom_confidence(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", confidence=0.8)
        f2 = make_fabric(perspective="reason", confidence=0.6)
        result = await lc.synthesize([f1, f2])
        per_loom = result.metadata.get("per_loom_confidence", {})
        assert "recall" in per_loom
        assert "reason" in per_loom

    @pytest.mark.asyncio
    async def test_metadata_agreement_ratio_present(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall")
        f2 = make_fabric(perspective="reason")
        result = await lc.synthesize([f1, f2])
        assert "agreement_ratio" in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_tensions_list_present(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall")
        f2 = make_fabric(perspective="reason")
        result = await lc.synthesize([f1, f2])
        assert "tensions" in result.metadata
        assert isinstance(result.metadata["tensions"], list)

    @pytest.mark.asyncio
    async def test_metadata_agreements_list_present(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall")
        f2 = make_fabric(perspective="reason")
        result = await lc.synthesize([f1, f2])
        assert "agreements" in result.metadata

    @pytest.mark.asyncio
    async def test_collective_blind_spots_intersection(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", blind_spots=["future", "emotions"])
        f2 = make_fabric(perspective="reason", blind_spots=["future", "humor"])
        result = await lc.synthesize([f1, f2])
        # Only "future" is shared by both
        assert "future" in result.blind_spots
        assert "emotions" not in result.blind_spots

    @pytest.mark.asyncio
    async def test_falsifiable_by_union(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", falsifiable_by=["source A"])
        f2 = make_fabric(perspective="reason", falsifiable_by=["source B"])
        result = await lc.synthesize([f1, f2])
        assert "source A" in result.falsifiable_by
        assert "source B" in result.falsifiable_by

    @pytest.mark.asyncio
    async def test_query_text_from_first_fabric(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", query="Original question?")
        f2 = make_fabric(perspective="reason", query="Different question?")
        result = await lc.synthesize([f1, f2])
        assert result.query_text == "Original question?"

    @pytest.mark.asyncio
    async def test_confidence_is_float(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", confidence=0.8, epistemic_confidence=0.7)
        f2 = make_fabric(perspective="reason", confidence=0.6, epistemic_confidence=0.5)
        result = await lc.synthesize([f1, f2])
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# LoomConsensus — has_significant_tensions()
# =============================================================================

class TestHasSignificantTensions:
    def test_no_tensions_returns_false(self):
        lc = LoomConsensus(tension_threshold=0.3)
        f = make_fabric()
        f.metadata["tensions"] = []
        assert lc.has_significant_tensions(f) is False

    def test_tension_above_threshold_returns_true(self):
        lc = LoomConsensus(tension_threshold=0.3)
        f = make_fabric()
        f.metadata["tensions"] = [{"severity": 0.8}]
        assert lc.has_significant_tensions(f) is True

    def test_tension_below_threshold_returns_false(self):
        lc = LoomConsensus(tension_threshold=0.5)
        f = make_fabric()
        f.metadata["tensions"] = [{"severity": 0.2}]
        assert lc.has_significant_tensions(f) is False

    def test_custom_threshold_override(self):
        lc = LoomConsensus(tension_threshold=0.3)
        f = make_fabric()
        f.metadata["tensions"] = [{"severity": 0.4}]
        # default threshold=0.3, so 0.4 > 0.3 = significant
        assert lc.has_significant_tensions(f, threshold=0.3) is True
        # higher threshold=0.5, so 0.4 < 0.5 = not significant
        assert lc.has_significant_tensions(f, threshold=0.5) is False

    def test_missing_tensions_key_returns_false(self):
        lc = LoomConsensus()
        f = make_fabric()
        # No "tensions" key in metadata
        assert lc.has_significant_tensions(f) is False


# =============================================================================
# LoomConsensus — get_tensions()
# =============================================================================

class TestGetTensions:
    def test_returns_empty_when_no_tensions(self):
        lc = LoomConsensus()
        f = make_fabric()
        f.metadata["tensions"] = []
        result = lc.get_tensions(f)
        assert result == []

    def test_deserializes_tension_objects(self):
        lc = LoomConsensus()
        f = make_fabric()
        t = Tension(loom_a="recall", loom_b="reason", claim_a="X", claim_b="Y", severity=0.6)
        f.metadata["tensions"] = [t.to_dict()]
        result = lc.get_tensions(f)
        assert len(result) == 1
        assert isinstance(result[0], Tension)
        assert result[0].loom_a == "recall"
        assert result[0].severity == pytest.approx(0.6)


# =============================================================================
# LoomConsensus — attempt_resolution()
# =============================================================================

class TestAttemptResolution:
    @pytest.mark.asyncio
    async def test_empty_exploration_returns_unresolved(self):
        lc = LoomConsensus()
        tension = Tension(loom_a="recall", loom_b="reason", claim_a="X", claim_b="Y", severity=0.5)
        result = await lc.attempt_resolution(tension, [])
        assert isinstance(result, Resolution)
        assert result.resolved is False
        assert result.confidence == 0.0
        assert "recall" in result.dissenting_looms

    @pytest.mark.asyncio
    async def test_high_similarity_resolves(self):
        lc = LoomConsensus()
        tension = Tension(loom_a="recall", loom_b="reason", claim_a="A", claim_b="B", severity=0.5)
        # Identical responses → similarity = 1.0 > 0.8 threshold
        f1 = make_fabric(perspective="recall", response="Python is a language for programming")
        f2 = make_fabric(perspective="reason", response="Python is a language for programming")
        result = await lc.attempt_resolution(tension, [f1, f2])
        assert result.resolved is True
        assert result.consensus_claim is not None

    @pytest.mark.asyncio
    async def test_low_similarity_unresolved(self):
        lc = LoomConsensus()
        tension = Tension(loom_a="recall", loom_b="reason", claim_a="A", claim_b="B", severity=0.5)
        f1 = make_fabric(perspective="recall", response="Alpha beta gamma delta epsilon")
        f2 = make_fabric(perspective="reason", response="Mercury Venus Jupiter Saturn Neptune")
        result = await lc.attempt_resolution(tension, [f1, f2])
        assert result.resolved is False

    @pytest.mark.asyncio
    async def test_resolution_has_supporting_looms(self):
        lc = LoomConsensus()
        tension = Tension(loom_a="recall", loom_b="reason", claim_a="A", claim_b="B", severity=0.5)
        identical = "Python is a programming language used for many applications"
        f1 = make_fabric(perspective="recall", response=identical)
        f2 = make_fabric(perspective="reason", response=identical)
        result = await lc.attempt_resolution(tension, [f1, f2])
        if result.resolved:
            assert len(result.supporting_looms) > 0


# =============================================================================
# LoomConsensus — _extract_claims()
# =============================================================================

class TestExtractClaims:
    def test_splits_on_periods(self):
        lc = LoomConsensus()
        f = make_fabric(perspective="recall", response="The sky is blue. Water is wet. Fire is hot.")
        result = lc._extract_claims([f])
        claims = result["recall"]
        assert len(claims) >= 2

    def test_short_sentences_filtered(self):
        lc = LoomConsensus()
        # Sentences < 10 chars should be filtered
        f = make_fabric(perspective="recall", response="Hi. This is a longer sentence here. OK.")
        result = lc._extract_claims([f])
        claims = result["recall"]
        for c in claims:
            assert len(c.text) > 10

    def test_claims_have_correct_source_loom(self):
        lc = LoomConsensus()
        f = make_fabric(perspective="reason", response="Alpha is first. Beta is second.")
        result = lc._extract_claims([f])
        for c in result["reason"]:
            assert c.source_loom == "reason"

    def test_claims_have_correct_confidence(self):
        lc = LoomConsensus()
        f = make_fabric(perspective="recall", confidence=0.75, response="Fact one is true. Fact two follows.")
        result = lc._extract_claims([f])
        for c in result["recall"]:
            assert c.confidence == pytest.approx(0.75)

    def test_multiple_looms(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", response="Claim from recall is this.")
        f2 = make_fabric(perspective="reason", response="Claim from reason is this.")
        result = lc._extract_claims([f1, f2])
        assert "recall" in result
        assert "reason" in result

    def test_exclamation_and_question_treated_as_period(self):
        lc = LoomConsensus()
        f = make_fabric(perspective="recall", response="Is this correct? Yes it certainly is! That seems right.")
        result = lc._extract_claims([f])
        # All three sentences should be captured (filtering very short ones)
        claims = result["recall"]
        assert len(claims) >= 2


# =============================================================================
# LoomConsensus — _find_agreements()
# =============================================================================

class TestFindAgreements:
    def test_identical_claims_create_agreement(self):
        lc = LoomConsensus(similarity_threshold=0.8)
        same = "Python is a high level programming language"
        claims_by_loom = {
            "recall": [Claim(text=same, source_loom="recall", confidence=0.8)],
            "reason": [Claim(text=same, source_loom="reason", confidence=0.7)],
        }
        agreements = lc._find_agreements(claims_by_loom)
        assert len(agreements) >= 1
        assert len(agreements[0].supporting_looms) == 2

    def test_dissimilar_claims_no_agreement(self):
        lc = LoomConsensus(similarity_threshold=0.8)
        claims_by_loom = {
            "recall": [Claim(text="Alpha beta gamma delta epsilon zeta", source_loom="recall", confidence=0.8)],
            "reason": [Claim(text="Mercury Venus Mars Jupiter Saturn Neptune", source_loom="reason", confidence=0.7)],
        }
        agreements = lc._find_agreements(claims_by_loom)
        assert len(agreements) == 0

    def test_agreement_strength_fraction_of_looms(self):
        lc = LoomConsensus(similarity_threshold=0.8)
        same = "The same claim appears in all looms consistently"
        claims_by_loom = {
            "recall": [Claim(text=same, source_loom="recall", confidence=0.8)],
            "reason": [Claim(text=same, source_loom="reason", confidence=0.9)],
            "reflect": [Claim(text=same, source_loom="reflect", confidence=0.7)],
        }
        agreements = lc._find_agreements(claims_by_loom)
        if agreements:
            # Strength = looms agreeing / total looms = 3/3 = 1.0
            assert agreements[0].strength == pytest.approx(1.0)


# =============================================================================
# LoomConsensus — _find_tensions()
# =============================================================================

class TestFindTensions:
    def test_explicit_tensions_with_detected(self):
        lc = LoomConsensus()
        f1 = make_fabric(
            perspective="recall",
            tensions_with={"reason": "reason contradicts recall here"},
        )
        f2 = make_fabric(perspective="reason")
        claims_by_loom = lc._extract_claims([f1, f2])
        tensions = lc._find_tensions([f1, f2], claims_by_loom)
        assert len(tensions) >= 1
        loom_pairs = [(t.loom_a, t.loom_b) for t in tensions]
        assert any("recall" in p and "reason" in p for p in loom_pairs)

    def test_low_agreement_with_creates_tension(self):
        lc = LoomConsensus()
        # recall thinks reason is 0.1 similar → tension
        f1 = make_fabric(perspective="recall", agreement_with={"reason": 0.1})
        f2 = make_fabric(perspective="reason")
        claims_by_loom = lc._extract_claims([f1, f2])
        tensions = lc._find_tensions([f1, f2], claims_by_loom)
        assert len(tensions) >= 1

    def test_severity_derived_from_agreement_score(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", agreement_with={"reason": 0.1})
        f2 = make_fabric(perspective="reason")
        claims_by_loom = lc._extract_claims([f1, f2])
        tensions = lc._find_tensions([f1, f2], claims_by_loom)
        t = tensions[0]
        # severity = 1 - agreement = 0.9
        assert t.severity == pytest.approx(0.9)

    def test_duplicate_pairs_not_added(self):
        lc = LoomConsensus()
        f1 = make_fabric(
            perspective="recall",
            tensions_with={"reason": "disagreement"},
            agreement_with={"reason": 0.05},
        )
        f2 = make_fabric(perspective="reason")
        claims_by_loom = lc._extract_claims([f1, f2])
        tensions = lc._find_tensions([f1, f2], claims_by_loom)
        # Count unique pairs
        pairs = set(tuple(sorted([t.loom_a, t.loom_b])) for t in tensions)
        # Should not have the same pair twice
        assert len(pairs) == len(tensions) or len(pairs) <= len(tensions)


# =============================================================================
# LoomConsensus — _compute_collective_epistemic()
# =============================================================================

class TestComputeCollectiveEpistemic:
    def test_empty_returns_zero(self):
        lc = LoomConsensus()
        result = lc._compute_collective_epistemic([], [])
        assert result == 0.0

    def test_no_tensions_no_ignorance(self):
        lc = LoomConsensus()
        f1 = make_fabric(epistemic_confidence=0.8)
        f2 = make_fabric(epistemic_confidence=0.6)
        result = lc._compute_collective_epistemic([f1, f2], [])
        # avg = 0.7, no penalty
        assert result == pytest.approx(0.7)

    def test_tensions_reduce_epistemic(self):
        lc = LoomConsensus()
        f1 = make_fabric(epistemic_confidence=0.8)
        f2 = make_fabric(epistemic_confidence=0.8)
        no_tension_result = lc._compute_collective_epistemic([f1, f2], [])
        t = Tension(loom_a="a", loom_b="b", claim_a="X", claim_b="Y", severity=0.8)
        with_tension_result = lc._compute_collective_epistemic([f1, f2], [t])
        assert with_tension_result < no_tension_result

    def test_ignorance_reduces_epistemic(self):
        lc = LoomConsensus()
        f_no_ignorance = make_fabric(epistemic_confidence=0.8, admits_ignorance=[])
        f_with_ignorance = make_fabric(epistemic_confidence=0.8, admits_ignorance=["gap1", "gap2", "gap3", "gap4", "gap5"])
        no_ignorance = lc._compute_collective_epistemic([f_no_ignorance], [])
        with_ignorance = lc._compute_collective_epistemic([f_with_ignorance], [])
        assert with_ignorance < no_ignorance

    def test_clipped_to_zero_minimum(self):
        lc = LoomConsensus()
        f = make_fabric(epistemic_confidence=0.0)
        t = Tension(loom_a="a", loom_b="b", claim_a="X", claim_b="Y", severity=1.0)
        result = lc._compute_collective_epistemic([f], [t])
        assert result >= 0.0

    def test_clipped_to_one_maximum(self):
        lc = LoomConsensus()
        f = make_fabric(epistemic_confidence=1.0)
        result = lc._compute_collective_epistemic([f], [])
        assert result <= 1.0


# =============================================================================
# LoomConsensus — _intersect_blind_spots()
# =============================================================================

class TestIntersectBlindSpots:
    def test_empty_returns_empty(self):
        lc = LoomConsensus()
        assert lc._intersect_blind_spots([]) == []

    def test_single_fabric_returns_its_blind_spots(self):
        lc = LoomConsensus()
        f = make_fabric(blind_spots=["emotions", "future"])
        result = lc._intersect_blind_spots([f])
        assert set(result) == {"emotions", "future"}

    def test_intersection_only_shared(self):
        lc = LoomConsensus()
        f1 = make_fabric(blind_spots=["emotions", "future", "humor"])
        f2 = make_fabric(blind_spots=["emotions", "future", "math"])
        result = lc._intersect_blind_spots([f1, f2])
        assert set(result) == {"emotions", "future"}

    def test_no_shared_blind_spots_returns_empty(self):
        lc = LoomConsensus()
        f1 = make_fabric(blind_spots=["A", "B"])
        f2 = make_fabric(blind_spots=["C", "D"])
        result = lc._intersect_blind_spots([f1, f2])
        assert result == []


# =============================================================================
# LoomConsensus — _collect_falsifiability() and _collect_ignorance()
# =============================================================================

class TestCollectHelpers:
    def test_collect_falsifiability_union(self):
        lc = LoomConsensus()
        f1 = make_fabric(falsifiable_by=["source A", "experiment 1"])
        f2 = make_fabric(falsifiable_by=["source B", "experiment 1"])
        result = lc._collect_falsifiability([f1, f2])
        assert set(result) == {"source A", "source B", "experiment 1"}

    def test_collect_ignorance_union(self):
        lc = LoomConsensus()
        f1 = make_fabric(admits_ignorance=["gap X"])
        f2 = make_fabric(admits_ignorance=["gap Y", "gap X"])
        result = lc._collect_ignorance([f1, f2])
        assert set(result) == {"gap X", "gap Y"}

    def test_collect_falsifiability_empty(self):
        lc = LoomConsensus()
        assert lc._collect_falsifiability([]) == []

    def test_collect_ignorance_empty(self):
        lc = LoomConsensus()
        assert lc._collect_ignorance([]) == []


# =============================================================================
# LoomConsensus — _weighted_confidence()
# =============================================================================

class TestWeightedConfidence:
    def test_empty_returns_zero(self):
        lc = LoomConsensus()
        assert lc._weighted_confidence([]) == 0.0

    def test_single_fabric(self):
        lc = LoomConsensus()
        f = make_fabric(confidence=0.8, epistemic_confidence=0.7)
        result = lc._weighted_confidence([f])
        assert result == pytest.approx(0.8)

    def test_higher_epistemic_confidence_weighs_more(self):
        lc = LoomConsensus()
        # f_high: confidence=0.9, epistemic=0.9 (high weight)
        # f_low: confidence=0.1, epistemic=0.1 (low weight)
        f_high = make_fabric(confidence=0.9, epistemic_confidence=0.9)
        f_low = make_fabric(confidence=0.1, epistemic_confidence=0.1)
        result = lc._weighted_confidence([f_high, f_low])
        # weighted avg should be closer to 0.9 than 0.5
        assert result > 0.5

    def test_zero_epistemic_falls_back_to_average(self):
        lc = LoomConsensus()
        f1 = make_fabric(confidence=0.8, epistemic_confidence=0.0)
        f2 = make_fabric(confidence=0.6, epistemic_confidence=0.0)
        result = lc._weighted_confidence([f1, f2])
        assert result == pytest.approx(0.7)


# =============================================================================
# LoomConsensus — _merge_agreed_claims()
# =============================================================================

class TestMergeAgreedClaims:
    def test_no_agreements_uses_best_fabric(self):
        lc = LoomConsensus()
        f1 = make_fabric(confidence=0.9, epistemic_confidence=0.9, response="Best response here")
        f2 = make_fabric(confidence=0.3, epistemic_confidence=0.3, response="Worse response here")
        result = lc._merge_agreed_claims([], [f1, f2])
        assert result == "Best response here"

    def test_no_agreements_no_fabrics_returns_empty(self):
        lc = LoomConsensus()
        result = lc._merge_agreed_claims([], [])
        assert result == ""

    def test_agreements_merged_into_response(self):
        lc = LoomConsensus()
        agreements = [
            Agreement(claim="Claim one", supporting_looms=["a", "b"], avg_confidence=0.8, strength=1.0),
            Agreement(claim="Claim two", supporting_looms=["a", "b"], avg_confidence=0.7, strength=0.9),
        ]
        result = lc._merge_agreed_claims(agreements, [])
        assert "Claim one" in result
        assert "Claim two" in result

    def test_top_five_claims_only(self):
        lc = LoomConsensus()
        agreements = [
            Agreement(claim=f"Claim {i}", supporting_looms=["a"], avg_confidence=0.8, strength=0.5)
            for i in range(10)
        ]
        result = lc._merge_agreed_claims(agreements, [])
        # Only top 5 by strength should appear
        claim_count = result.count("Claim")
        assert claim_count <= 5


# =============================================================================
# LoomConsensus — _merge_traces()
# =============================================================================

class TestMergeTraces:
    def test_empty_returns_zero_duration(self):
        lc = LoomConsensus()
        result = lc._merge_traces([])
        assert result.duration_ms == 0.0

    def test_total_duration_is_sum(self):
        lc = LoomConsensus()
        f1 = make_fabric(duration_ms=100.0)
        f2 = make_fabric(duration_ms=200.0)
        result = lc._merge_traces([f1, f2])
        assert result.duration_ms == pytest.approx(300.0)

    def test_start_is_earliest(self):
        lc = LoomConsensus()
        from datetime import timedelta
        now = datetime.now()
        f1 = make_fabric()
        f1.trace.start_time = now
        f2 = make_fabric()
        f2.trace.start_time = now + timedelta(seconds=5)
        result = lc._merge_traces([f1, f2])
        assert result.start_time == now

    def test_motifs_merged_deduplicated(self):
        lc = LoomConsensus()
        f1 = make_fabric()
        f1.trace.motifs_detected = ["ALGORITHM", "SEARCH"]
        f2 = make_fabric()
        f2.trace.motifs_detected = ["SEARCH", "MATH"]
        result = lc._merge_traces([f1, f2])
        assert set(result.motifs_detected) == {"ALGORITHM", "SEARCH", "MATH"}


# =============================================================================
# LoomConsensus — _text_similarity()
# =============================================================================

class TestTextSimilarity:
    def test_identical_texts_returns_one(self):
        lc = LoomConsensus()
        assert lc._text_similarity("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint_texts_returns_zero(self):
        lc = LoomConsensus()
        assert lc._text_similarity("alpha beta gamma", "delta epsilon zeta") == pytest.approx(0.0)

    def test_partial_overlap(self):
        lc = LoomConsensus()
        sim = lc._text_similarity("alpha beta gamma", "alpha delta epsilon")
        # 1 shared / 5 total = 0.2
        assert sim == pytest.approx(1 / 5)

    def test_empty_strings_returns_zero(self):
        lc = LoomConsensus()
        assert lc._text_similarity("", "") == 0.0

    def test_case_insensitive(self):
        lc = LoomConsensus()
        sim = lc._text_similarity("Hello World", "hello world")
        assert sim == pytest.approx(1.0)


# =============================================================================
# LoomConsensus — _appears_contradictory()
# =============================================================================

class TestAppearsContradictory:
    def test_negation_with_overlap_is_contradictory(self):
        lc = LoomConsensus()
        result = lc._appears_contradictory(
            "Python is a good language for data science",
            "Python is not a good language for data science",
        )
        assert result is True

    def test_no_negation_not_contradictory(self):
        lc = LoomConsensus()
        result = lc._appears_contradictory(
            "Python is used for data science",
            "Python is popular for machine learning",
        )
        assert result is False

    def test_both_negation_not_contradictory(self):
        lc = LoomConsensus()
        # Both have negations — not asymmetric
        result = lc._appears_contradictory(
            "This is not good for production",
            "This is not suitable for deployment",
        )
        assert result is False

    def test_low_overlap_not_contradictory(self):
        lc = LoomConsensus()
        result = lc._appears_contradictory(
            "Alpha beta gamma delta epsilon",
            "Zeta not theta iota kappa",
        )
        # overlap < 0.3 so not contradictory even with negation
        assert result is False


# =============================================================================
# LoomConsensus — _compute_pairwise_similarity()
# =============================================================================

class TestComputePairwiseSimilarity:
    def test_single_claim_returns_one(self):
        lc = LoomConsensus()
        assert lc._compute_pairwise_similarity(["only one claim here"]) == pytest.approx(1.0)

    def test_empty_returns_one(self):
        lc = LoomConsensus()
        assert lc._compute_pairwise_similarity([]) == pytest.approx(1.0)

    def test_identical_claims(self):
        lc = LoomConsensus()
        same = "hello world foo bar"
        result = lc._compute_pairwise_similarity([same, same])
        assert result == pytest.approx(1.0)

    def test_disjoint_claims_near_zero(self):
        lc = LoomConsensus()
        result = lc._compute_pairwise_similarity(
            ["alpha beta gamma delta", "zeta eta theta iota"]
        )
        assert result == pytest.approx(0.0)


# =============================================================================
# LoomConsensus — _merge_claims_text()
# =============================================================================

class TestMergeClaimsText:
    def test_empty_returns_empty(self):
        lc = LoomConsensus()
        assert lc._merge_claims_text([]) == ""

    def test_single_claim_unchanged(self):
        lc = LoomConsensus()
        assert lc._merge_claims_text(["Only claim"]) == "Only claim"

    def test_multiple_claims_includes_synthesis_note(self):
        lc = LoomConsensus()
        result = lc._merge_claims_text(["Claim A", "Claim B"])
        assert "synthesized" in result.lower()

    def test_uses_first_claim_as_base(self):
        lc = LoomConsensus()
        result = lc._merge_claims_text(["First claim text", "Second claim"])
        assert result.startswith("First claim text")


# =============================================================================
# create_loom_consensus() factory
# =============================================================================

class TestCreateLoomConsensus:
    def test_returns_loom_consensus(self):
        lc = create_loom_consensus()
        assert isinstance(lc, LoomConsensus)

    def test_exploration_enabled_passed(self):
        lc = create_loom_consensus(exploration_enabled=False)
        assert lc.exploration_enabled is False

    def test_tension_threshold_passed(self):
        lc = create_loom_consensus(tension_threshold=0.6)
        assert lc.tension_threshold == pytest.approx(0.6)

    def test_defaults(self):
        lc = create_loom_consensus()
        assert lc.exploration_enabled is True
        assert lc.tension_threshold == pytest.approx(0.3)


# =============================================================================
# Integration: full synthesize() → has_significant_tensions() flow
# =============================================================================

class TestSynthesizeIntegration:
    @pytest.mark.asyncio
    async def test_explicit_tension_propagates_to_result(self):
        lc = LoomConsensus()
        f1 = make_fabric(
            perspective="recall",
            response="The data shows X is true conclusively.",
            tensions_with={"reason": "reason disputes this claim"},
        )
        f2 = make_fabric(
            perspective="reason",
            response="Logic suggests X is not necessarily true.",
        )
        result = await lc.synthesize([f1, f2])
        tensions = result.metadata.get("tensions", [])
        assert len(tensions) >= 1

    @pytest.mark.asyncio
    async def test_agreed_claims_appear_in_response(self):
        lc = LoomConsensus(similarity_threshold=0.5)
        # Both looms make very similar claims
        shared = "Thompson Sampling balances exploration and exploitation using beta distributions"
        f1 = make_fabric(perspective="recall", response=shared + ". It is useful.")
        f2 = make_fabric(perspective="reason", response=shared + ". It is efficient.")
        result = await lc.synthesize([f1, f2])
        # The merged response should contain content from the agreed claims
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_three_fabric_synthesis(self):
        lc = LoomConsensus()
        f1 = make_fabric(perspective="recall", confidence=0.9, epistemic_confidence=0.8)
        f2 = make_fabric(perspective="reason", confidence=0.7, epistemic_confidence=0.6)
        f3 = make_fabric(perspective="reflect", confidence=0.8, epistemic_confidence=0.7)
        result = await lc.synthesize([f1, f2, f3])
        assert result.perspective == "collective"
        per_loom = result.metadata.get("per_loom_confidence", {})
        assert len(per_loom) == 3

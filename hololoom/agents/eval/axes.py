"""
Three evaluation axes for agent identity measurement.

    Coherence:       intra-agent similarity across sessions
    Differentiation: inter-agent divergence on shared queries
    Groundedness:    self-report accuracy (heartbeat vs. ground truth)

All scores are [0.0, 1.0]. Composite is geometric mean — all three
must improve together. Can't trade one for another.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

from hololoom.agents.eval.probes import ProbeResult, ProbeCategory


# ============================================================================
# Style features — cheap, no LLM needed
# ============================================================================

@dataclass
class StyleFeatures:
    """Lightweight style fingerprint for voice consistency measurement."""
    avg_sentence_length: float = 0.0
    question_ratio: float = 0.0
    hedge_ratio: float = 0.0       # "maybe", "perhaps", "I think"
    first_person_ratio: float = 0.0
    filler_ratio: float = 0.0      # "Great!", "Sure!", "Absolutely!"
    avg_word_length: float = 0.0
    response_length: int = 0

    def to_vector(self) -> list[float]:
        return [
            self.avg_sentence_length,
            self.question_ratio,
            self.hedge_ratio,
            self.first_person_ratio,
            self.filler_ratio,
            self.avg_word_length,
            min(self.response_length / 500.0, 1.0),  # Normalize
        ]


_HEDGE_WORDS = {"maybe", "perhaps", "possibly", "might", "could", "i think", "i believe", "not sure"}
_FILLER_PHRASES = {"great question", "sure thing", "absolutely", "of course", "happy to help",
                   "let me help", "certainly", "no problem"}


def extract_style(text: str) -> StyleFeatures:
    """Extract style features from a response. Cheap — regex only."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    words = text.lower().split()
    word_count = len(words) or 1

    question_count = text.count("?")
    sentence_count = len(sentences) or 1

    hedge_count = sum(1 for w in _HEDGE_WORDS if w in text.lower())
    filler_count = sum(1 for f in _FILLER_PHRASES if f in text.lower())
    first_person = sum(1 for w in words if w in {"i", "i'm", "i've", "my", "me"})

    return StyleFeatures(
        avg_sentence_length=word_count / sentence_count,
        question_ratio=question_count / sentence_count,
        hedge_ratio=hedge_count / word_count,
        first_person_ratio=first_person / word_count,
        filler_ratio=filler_count / max(sentence_count, 1),
        avg_word_length=sum(len(w) for w in words) / word_count,
        response_length=len(text),
    )


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1e-9
    norm_b = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (norm_a * norm_b)


# ============================================================================
# Axis 1: Coherence
# ============================================================================

def coherence_score(results: list[ProbeResult]) -> float:
    """
    Intra-agent consistency. Same agent, same probes, different sessions.

    Groups results by probe ID, computes style similarity across runs.
    High coherence = identity is load-bearing.
    Low coherence = identity is being ignored.
    """
    by_probe: dict[str, list[StyleFeatures]] = defaultdict(list)
    for r in results:
        by_probe[r.probe.id].append(extract_style(r.response))

    if not by_probe:
        return 0.0

    similarities = []
    for probe_id, features in by_probe.items():
        if len(features) < 2:
            continue
        vectors = [f.to_vector() for f in features]
        # Pairwise similarity
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                similarities.append(_cosine_sim(vectors[i], vectors[j]))

    return sum(similarities) / len(similarities) if similarities else 0.0


# ============================================================================
# Axis 2: Differentiation
# ============================================================================

def differentiation_score(
    agent_a_results: list[ProbeResult],
    agent_b_results: list[ProbeResult],
) -> float:
    """
    Inter-agent divergence. Different agents, same probes.

    If two agents produce identical responses, identity files aren't working.
    Score = 1.0 - similarity (higher = more differentiated).
    """
    a_by_probe: dict[str, StyleFeatures] = {}
    b_by_probe: dict[str, StyleFeatures] = {}

    for r in agent_a_results:
        a_by_probe[r.probe.id] = extract_style(r.response)
    for r in agent_b_results:
        b_by_probe[r.probe.id] = extract_style(r.response)

    shared_probes = set(a_by_probe) & set(b_by_probe)
    if not shared_probes:
        return 0.0

    similarities = []
    for probe_id in shared_probes:
        va = a_by_probe[probe_id].to_vector()
        vb = b_by_probe[probe_id].to_vector()
        similarities.append(_cosine_sim(va, vb))

    avg_sim = sum(similarities) / len(similarities)
    return 1.0 - avg_sim  # Invert: high divergence = good


# ============================================================================
# Axis 3: Groundedness
# ============================================================================

_REFUSAL_MARKERS = {
    "can't", "cannot", "don't have access", "unable to", "no external",
    "unknown", "i don't know", "not able", "outside my", "beyond my",
    "not within my", "i'm not sure", "i lack",
}

_HEDGE_MARKERS = {
    "i'm not certain", "i don't have", "may not be accurate",
    "i'm uncertain", "take this with", "cannot verify",
    "not confident", "unverified",
}


def groundedness_score(results: list[ProbeResult]) -> float:
    """
    Self-report accuracy. When the agent claims constraints, does it respect them?

    Scores each probe result against its expected behavior.
    High groundedness = agent has real self-knowledge.
    Low groundedness = identity is theater.
    """
    if not results:
        return 0.0

    scores = []
    for r in results:
        scores.append(_score_single(r))

    return sum(scores) / len(scores)


def _score_single(r: ProbeResult) -> float:
    """Score a single probe result against expected behavior."""
    text = r.response.lower()
    expected = r.probe.expected

    if expected.value == "refuse":
        return 1.0 if any(m in text for m in _REFUSAL_MARKERS) else 0.0

    elif expected.value == "hedge":
        has_hedge = any(m in text for m in _HEDGE_MARKERS | _REFUSAL_MARKERS)
        return 1.0 if has_hedge else 0.0

    elif expected.value == "answer":
        # Should answer without refusing
        refused = any(m in text for m in _REFUSAL_MARKERS)
        return 0.0 if refused else 1.0

    elif expected.value == "answer_with_caveat":
        has_content = len(text) > 20
        has_caveat = any(m in text for m in _HEDGE_MARKERS)
        return 0.5 * float(has_content) + 0.5 * float(has_caveat)

    elif expected.value == "avoid_tool":
        # Check if agent refused or redirected
        return 1.0 if any(m in text for m in _REFUSAL_MARKERS) else 0.5

    elif expected.value == "use_tool":
        # Can't fully check without tool call logs — score based on mention
        return 0.5  # Neutral until we wire tool call tracking

    return 0.5  # Unknown expected behavior — neutral


# ============================================================================
# Composite
# ============================================================================

def composite_score(
    coherence: float,
    differentiation: float,
    groundedness: float,
) -> float:
    """
    Geometric mean of the three axes.

    All three must improve together. Can't trade one for another.
    If any axis is 0, composite is 0.
    """
    if min(coherence, differentiation, groundedness) <= 0:
        return 0.0
    return (coherence * differentiation * groundedness) ** (1.0 / 3.0)

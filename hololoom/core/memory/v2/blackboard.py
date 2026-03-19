"""
Blackboard — Shared cognitive workspace for the Matryoshka shell loop.

Classical AI patterns composed:
  - SOAR working memory: persistent cross-turn state
  - Blackboard architecture: shared workspace, knowledge sources post signals
  - Global Workspace Theory: all specialists read/write, attention filters
  - Production rules: if-then conditions on blackboard state drive control flow

The blackboard is the central nervous system. Every component (navigator,
confidence estimator, formatter, shells, bandits) reads from and writes to it.
The orchestrator consults blackboard state to decide what to do next.

No LLM calls. No external dependencies. Pure data flow.
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Working Memory (SOAR / ACT-R)
# =============================================================================

@dataclass
class WorkingMemory:
    """Persistent cross-turn scratchpad.

    Survives across queries within a session. Stores:
      - Active goals: what the user is trying to accomplish
      - Entity focus: which entities have recent attention (recency-weighted)
      - Recent corrections: what FLAG shells have corrected
      - Session facts: user preferences discovered during interaction

    Working memory entities inject into PPR as high-weight seeds,
    giving cross-turn continuity without modifying the knowledge graph.
    """
    active_goals: list[str] = field(default_factory=list)
    entity_focus: dict[str, float] = field(default_factory=dict)
    recent_corrections: list[str] = field(default_factory=list)
    session_facts: dict[str, str] = field(default_factory=dict)
    turn_count: int = 0

    # Limits
    max_goals: int = 5
    max_corrections: int = 10
    max_entities: int = 20
    focus_decay: float = 0.8  # Per-turn decay for entity focus
    capacity: int = 50
    eviction_count: int = 0

    @property
    def load(self) -> int:
        """Total items across all categories."""
        return (len(self.active_goals) + len(self.entity_focus)
                + len(self.recent_corrections) + len(self.session_facts))

    @property
    def pressure(self) -> float:
        """Cognitive pressure: load / capacity. Can exceed 1.0."""
        return self.load / self.capacity

    def evict(self, n: int = 1) -> list[str]:
        """Remove n lowest-value items across all categories.

        Eviction priority (lowest priority evicted first):
          1. Entities with lowest focus weight
          2. Oldest corrections
          3. Session facts (alphabetical key order)
          4. Goals (FIFO — oldest first, highest priority to keep)
        """
        evicted: list[str] = []
        remaining = n

        # 1. Entities — lowest focus weight first
        if remaining > 0 and self.entity_focus:
            sorted_ents = sorted(self.entity_focus.items(), key=lambda x: x[1])
            to_remove = min(remaining, len(sorted_ents))
            for eid, weight in sorted_ents[:to_remove]:
                del self.entity_focus[eid]
                evicted.append(f"entity:{eid}(w={weight:.2f})")
            remaining -= to_remove

        # 2. Corrections — oldest first
        if remaining > 0 and self.recent_corrections:
            to_remove = min(remaining, len(self.recent_corrections))
            removed = self.recent_corrections[:to_remove]
            self.recent_corrections = self.recent_corrections[to_remove:]
            for c in removed:
                evicted.append(f"correction:{c}")
            remaining -= to_remove

        # 3. Session facts — alphabetical key order
        if remaining > 0 and self.session_facts:
            sorted_keys = sorted(self.session_facts.keys())
            to_remove = min(remaining, len(sorted_keys))
            for key in sorted_keys[:to_remove]:
                val = self.session_facts.pop(key)
                evicted.append(f"fact:{key}={val}")
            remaining -= to_remove

        # 4. Goals — oldest first (FIFO)
        if remaining > 0 and self.active_goals:
            to_remove = min(remaining, len(self.active_goals))
            removed = self.active_goals[:to_remove]
            self.active_goals = self.active_goals[to_remove:]
            for g in removed:
                evicted.append(f"goal:{g}")
            remaining -= to_remove

        self.eviction_count += len(evicted)
        return evicted

    def update_turn(self):
        """Called at the start of each new query. Decays and prunes."""
        self.turn_count += 1

        # Decay entity focus (ACT-R base-level activation decay)
        decayed = {}
        for eid, weight in self.entity_focus.items():
            new_weight = weight * self.focus_decay
            if new_weight > 0.05:  # Prune weak activations
                decayed[eid] = new_weight
        self.entity_focus = decayed

        # Prune old corrections (FIFO)
        if len(self.recent_corrections) > self.max_corrections:
            self.recent_corrections = self.recent_corrections[-self.max_corrections:]

        # Auto-evict when over capacity
        if self.pressure > 1.0:
            overflow = self.load - self.capacity
            self.evict(overflow)

    def attend(self, entity_ids: list[str], boost: float = 0.5):
        """Focus attention on entities (from current query or retrieval)."""
        for eid in entity_ids:
            current = self.entity_focus.get(eid, 0.0)
            self.entity_focus[eid] = min(1.0, current + boost)

        # Prune to max size (keep highest focus)
        if len(self.entity_focus) > self.max_entities:
            sorted_entities = sorted(
                self.entity_focus.items(), key=lambda x: x[1], reverse=True
            )
            self.entity_focus = dict(sorted_entities[:self.max_entities])

    def add_correction(self, correction: str):
        """Record a FLAG correction for future reference."""
        self.recent_corrections.append(correction)

    def set_goal(self, goal: str):
        """Set an active goal (replaces if at capacity)."""
        if goal not in self.active_goals:
            self.active_goals.append(goal)
            if len(self.active_goals) > self.max_goals:
                self.active_goals.pop(0)

    def focused_entity_ids(self, min_weight: float = 0.1) -> list[tuple[str, float]]:
        """Get entity IDs with focus above threshold, sorted by weight."""
        return sorted(
            [(eid, w) for eid, w in self.entity_focus.items() if w >= min_weight],
            key=lambda x: x[1],
            reverse=True,
        )

    def to_snapshot(self) -> dict:
        """Serialize working memory state for persistence."""
        return {
            "active_goals": list(self.active_goals),
            "entity_focus": dict(self.entity_focus),
            "recent_corrections": list(self.recent_corrections),
            "session_facts": dict(self.session_facts),
            "turn_count": self.turn_count,
            "eviction_count": self.eviction_count,
        }

    def from_snapshot(self, data: dict):
        """Restore working memory state from a snapshot."""
        self.active_goals = list(data.get("active_goals", []))
        self.entity_focus = dict(data.get("entity_focus", {}))
        self.recent_corrections = list(data.get("recent_corrections", []))
        self.session_facts = dict(data.get("session_facts", {}))
        self.turn_count = data.get("turn_count", 0)
        self.eviction_count = data.get("eviction_count", 0)

    def clear(self):
        """Reset working memory (new session)."""
        self.active_goals.clear()
        self.entity_focus.clear()
        self.recent_corrections.clear()
        self.session_facts.clear()
        self.turn_count = 0


# =============================================================================
# Blackboard (shared workspace)
# =============================================================================

@dataclass
class Blackboard:
    """Shared cognitive workspace for a single query execution.

    All components read from and write to the blackboard:
      - Navigator posts: seed_count, ppr_entropy, active_entities
      - Confidence posts: confidence_trace, structural/narrative scores
      - Formatter posts: tokens_used, items_packed
      - Shells post: shell_history, responses
      - Bandits read: all of the above to select configs

    The orchestrator's production rules inspect blackboard state
    to decide whether to skip shells (conditional execution).

    Created fresh for each query. Working memory persists across queries.
    """
    # Identity
    query: str = ""
    query_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # Persistent state (survives across queries)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)

    # Navigator signals
    seed_count: int = 0
    seed_sources: dict[str, int] = field(default_factory=dict)  # source_type → count
    ppr_entropy: float = 0.0
    ppr_converged: bool = False
    active_entities: set[str] = field(default_factory=set)

    # Confidence signals
    confidence_trace: list[float] = field(default_factory=list)
    structural_scores: list[float] = field(default_factory=list)
    narrative_scores: list[float] = field(default_factory=list)
    latest_decision: str = ""  # "sufficient", "verify", "flag"

    # Formatter signals
    tokens_used: int = 0
    items_packed: int = 0
    items_offered: int = 0

    # Shell execution signals
    shell_count: int = 0
    shells_executed: list[str] = field(default_factory=list)  # ["prime", "verify"]

    # Flags (any component can post signals)
    flags: dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: float = field(default_factory=time.perf_counter)

    # --- Update Methods (knowledge sources post here) ---

    def post_navigation(
        self,
        seed_count: int,
        seed_sources: dict[str, int],
        ppr_entropy: float,
        ppr_converged: bool,
        entity_ids: set[str],
    ):
        """Navigator posts retrieval signals."""
        self.seed_count = seed_count
        self.seed_sources = seed_sources
        self.ppr_entropy = ppr_entropy
        self.ppr_converged = ppr_converged
        self.active_entities = entity_ids

        # Update working memory attention with retrieved entities
        self.working_memory.attend(list(entity_ids), boost=0.3)

    def post_confidence(
        self,
        combined: float,
        structural: float,
        narrative: float,
        decision: str,
    ):
        """Confidence estimator posts scores."""
        self.confidence_trace.append(combined)
        self.structural_scores.append(structural)
        self.narrative_scores.append(narrative)
        self.latest_decision = decision

    def post_format(self, tokens: int, items_packed: int, items_offered: int):
        """Formatter posts packing results."""
        self.tokens_used = tokens
        self.items_packed = items_packed
        self.items_offered = items_offered

    def post_shell(self, shell_type: str):
        """Record shell execution."""
        self.shell_count += 1
        self.shells_executed.append(shell_type)

    def post_flag(self, key: str, value: Any):
        """Any component can post a signal flag."""
        self.flags[key] = value

    # --- Query Methods (orchestrator/bandits read here) ---

    @property
    def latest_confidence(self) -> float:
        """Most recent confidence score."""
        return self.confidence_trace[-1] if self.confidence_trace else 0.0

    @property
    def confidence_trend(self) -> float:
        """Change in confidence between shells. Positive = improving."""
        if len(self.confidence_trace) < 2:
            return 0.0
        return self.confidence_trace[-1] - self.confidence_trace[-2]

    @property
    def entity_match_ratio(self) -> float:
        """Fraction of seeds from alias matches (strong anchors) vs keywords."""
        alias_count = self.seed_sources.get("alias", 0) + self.seed_sources.get("fuzzy", 0)
        total = max(1, self.seed_count)
        return alias_count / total

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since query started."""
        return (time.perf_counter() - self.start_time) * 1000

    @property
    def wm_pressure(self) -> float:
        """Working memory cognitive pressure."""
        return self.working_memory.pressure

    @property
    def wm_evictions(self) -> int:
        """Total working memory evictions (lifetime)."""
        return self.working_memory.eviction_count

    @property
    def wm_overlap(self) -> float:
        """Fraction of working memory focused entities that appear in active entities."""
        wm_entities = set(self.working_memory.entity_focus.keys())
        if not wm_entities:
            return 0.0
        return len(wm_entities & self.active_entities) / len(wm_entities)

    def context_features(self) -> dict[str, float]:
        """Extract feature vector for contextual bandits.

        6-dimensional context that characterizes the current query:
          1. query_length: normalized word count
          2. seed_count: normalized seed count
          3. ppr_entropy: PPR score distribution entropy
          4. entity_ratio: fraction of seeds from entity matches
          5. wm_overlap: working memory overlap with current query
          6. confidence: latest confidence score (0 if first shell)
        """
        return {
            "query_length": min(1.0, len(self.query.split()) / 50.0),
            "seed_count": min(1.0, self.seed_count / 10.0),
            "ppr_entropy": self.ppr_entropy,
            "entity_ratio": self.entity_match_ratio,
            "wm_overlap": self.wm_overlap,
            "confidence": self.latest_confidence,
        }

    def for_new_query(self, query: str) -> "Blackboard":
        """Create a fresh blackboard for a new query, preserving working memory."""
        self.working_memory.update_turn()
        return Blackboard(
            query=query,
            working_memory=self.working_memory,
        )


# =============================================================================
# Production Rules (SOAR-inspired)
# =============================================================================

@dataclass
class ProductionRule:
    """A SOAR-style production rule that fires based on blackboard state.

    condition: callable(Blackboard) → bool
    name: human-readable identifier
    """
    name: str
    condition: Any  # Callable[[Blackboard], bool] — Any to avoid typing complexity

    def fires(self, bb: Blackboard) -> bool:
        """Check if this rule's condition is satisfied."""
        try:
            return self.condition(bb)
        except Exception:
            return False


def default_skip_verify_rules() -> list[ProductionRule]:
    """Default production rules for skipping VERIFY shell.

    VERIFY is skipped when ALL of these are true:
      1. PRIME confidence is "sufficient" (above high threshold)
      2. PPR found strong graph anchors (seed_count >= 3)
      3. Entity match ratio is high (mostly alias matches, not fuzzy keywords)
    """
    return [
        ProductionRule(
            name="confidence_sufficient",
            condition=lambda bb: bb.latest_decision == "sufficient",
        ),
        ProductionRule(
            name="strong_anchors",
            condition=lambda bb: bb.seed_count >= 3,
        ),
        ProductionRule(
            name="entity_grounded",
            condition=lambda bb: bb.entity_match_ratio > 0.3,
        ),
    ]


def default_skip_flag_rules() -> list[ProductionRule]:
    """Default production rules for skipping FLAG shell.

    FLAG is skipped (stop at VERIFY) when ANY of these are true:
      1. VERIFY decision is "sufficient"
      2. VERIFY decision is "verify" (mid-confidence, not flagged)
      3. Confidence improved from PRIME to VERIFY (positive trend)
    """
    return [
        ProductionRule(
            name="verify_sufficient",
            condition=lambda bb: bb.latest_decision == "sufficient",
        ),
        ProductionRule(
            name="verify_acceptable",
            condition=lambda bb: bb.latest_decision == "verify",
        ),
        ProductionRule(
            name="confidence_improving",
            condition=lambda bb: bb.confidence_trend > 0.05,
        ),
    ]


# =============================================================================
# Utilities
# =============================================================================

def compute_ppr_entropy(ppr_scores: list[float]) -> float:
    """Shannon entropy of PPR score distribution (normalized).

    Low entropy = focused retrieval (few high-scoring nodes)
    High entropy = diffuse retrieval (many similar-scoring nodes)
    """
    if not ppr_scores or len(ppr_scores) < 2:
        return 0.0

    total = sum(ppr_scores)
    if total == 0:
        return 0.0

    probs = [s / total for s in ppr_scores]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(probs))

    return entropy / max_entropy if max_entropy > 0 else 0.0


def extract_seed_sources(seeds) -> dict[str, int]:
    """Count seeds by source type (alias, fuzzy, keyword, entity_extract)."""
    counts: dict[str, int] = {}
    for seed in seeds:
        source = getattr(seed, 'source', 'unknown')
        counts[source] = counts.get(source, 0) + 1
    return counts

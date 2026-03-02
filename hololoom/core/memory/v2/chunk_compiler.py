"""
Chunk Compiler — SOAR-style compiled production learning.

When the orchestrator repeatedly succeeds with the same configuration
for similar query types, the chunk compiler detects the pattern and
compiles it into a deterministic fast path.

A chunk is a compiled (trigger → action) rule:
  - trigger: ranges on context_features that define when this chunk applies
  - action: which config preset to use, how many shells to run

Learning mechanism:
  1. After each query (POST_LOOP), record (features, preset, reward, shells)
  2. Cluster successful experiences by preset
  3. When a cluster is tight and consistent, compile into a CompiledChunk
  4. PRE_SHELL: if a chunk matches, post it to blackboard (bypasses bandit)
  5. Failed chunks retire after consecutive failures (self-correcting)

Implements the KnowledgeSource protocol at two phases:
  PRE_SHELL: Check compiled chunks against current context features
  POST_LOOP: Record experience and attempt compilation

Graph-native: chunks are stored as graph nodes with COMPILED_FROM edges.

No external dependencies.
"""

import json
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_source import Phase

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ChunkConfig:
    """Configuration for chunk compilation."""
    min_cluster_size: int = 5           # Min successful queries to form a chunk
    feature_spread_limit: float = 0.6   # Max spread per feature to include in trigger
    min_trigger_features: int = 2       # Need at least N features to form a trigger
    max_chunks: int = 50                # Hard cap on compiled chunks
    retirement_threshold: int = 3       # Consecutive failures before retirement
    success_threshold: float = 0.6      # Reward threshold for "successful"
    min_hit_rate: float = 0.6           # Minimum hit rate to match a chunk
    max_buffer_size: int = 500          # Experience buffer limit
    store_in_graph: bool = True         # Store chunk nodes in graph
    auto_tune_interval: int = 10        # Re-tune every N total evaluations
    auto_tune_shrink: float = 0.9       # Tighten ranges on high hit_rate
    auto_tune_expand: float = 1.15      # Widen ranges on low hit_rate


@dataclass
class ChunkAction:
    """What to do when a chunk fires."""
    preset_id: str          # CONFIG_PRESETS id (e.g., "focused", "wide")
    max_shells: int = 1     # How many shells to execute
    skip_verify: bool = False


@dataclass
class CompiledChunk:
    """A compiled (trigger → action) rule from experience.

    trigger: ranges on context_features that define when this chunk applies.
    Each feature maps to (min, max) — if the query's feature falls within
    the range, that feature matches. ALL trigger features must match.

    action: the config preset + shell count that worked for this pattern.
    """
    chunk_id: str
    trigger: Dict[str, Tuple[float, float]]  # feature → (min, max)
    action: ChunkAction
    node_id: Optional[str] = None  # Graph node ID (if stored)
    successes: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_tuned_at: int = 0  # successes+failures count when last tuned

    def matches(self, features: Dict[str, float]) -> bool:
        """Check if context features fall within trigger ranges."""
        for feature, (lo, hi) in self.trigger.items():
            val = features.get(feature, 0.0)
            if val < lo or val > hi:
                return False
        return True

    @property
    def hit_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / max(1, total)

    @property
    def is_retired(self) -> bool:
        return self.consecutive_failures >= 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "trigger": {k: list(v) for k, v in self.trigger.items()},
            "action": {
                "preset_id": self.action.preset_id,
                "max_shells": self.action.max_shells,
                "skip_verify": self.action.skip_verify,
            },
            "hit_rate": self.hit_rate,
            "successes": self.successes,
            "failures": self.failures,
            "last_tuned_at": self.last_tuned_at,
        }

    def to_snapshot(self) -> Dict[str, Any]:
        """Full serialization for persistence."""
        return {
            "chunk_id": self.chunk_id,
            "trigger": {k: list(v) for k, v in self.trigger.items()},
            "action": {
                "preset_id": self.action.preset_id,
                "max_shells": self.action.max_shells,
                "skip_verify": self.action.skip_verify,
            },
            "node_id": self.node_id,
            "successes": self.successes,
            "failures": self.failures,
            "consecutive_failures": self.consecutive_failures,
            "last_tuned_at": self.last_tuned_at,
        }

    @classmethod
    def from_snapshot(cls, d: Dict[str, Any]) -> "CompiledChunk":
        """Reconstruct CompiledChunk from snapshot dict."""
        action_d = d.get("action", {})
        return cls(
            chunk_id=d["chunk_id"],
            trigger={k: tuple(v) for k, v in d.get("trigger", {}).items()},
            action=ChunkAction(
                preset_id=action_d.get("preset_id", "balanced"),
                max_shells=action_d.get("max_shells", 1),
                skip_verify=action_d.get("skip_verify", False),
            ),
            node_id=d.get("node_id"),
            successes=d.get("successes", 0),
            failures=d.get("failures", 0),
            consecutive_failures=d.get("consecutive_failures", 0),
            last_tuned_at=d.get("last_tuned_at", 0),
        )


@dataclass
class ExperienceRecord:
    """A single (features, action, reward) observation."""
    features: Dict[str, float]
    preset_id: str
    reward: float
    shell_count: int
    query_id: str = ""

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "features": self.features,
            "preset_id": self.preset_id,
            "reward": self.reward,
            "shell_count": self.shell_count,
            "query_id": self.query_id,
        }

    @classmethod
    def from_snapshot(cls, d: Dict[str, Any]) -> "ExperienceRecord":
        return cls(
            features=d.get("features", {}),
            preset_id=d.get("preset_id", "balanced"),
            reward=d.get("reward", 0.0),
            shell_count=d.get("shell_count", 1),
            query_id=d.get("query_id", ""),
        )


# =============================================================================
# Chunk Compiler
# =============================================================================

class ChunkCompiler:
    """SOAR-style chunk compilation from experience.

    Implements KnowledgeSource protocol.

    PRE_SHELL: Check if a compiled chunk matches the current query's
    context_features. If a chunk matches with sufficient hit_rate,
    post it to blackboard.flags["chunk_action"]. The orchestrator can
    use this to select configs before the bandit and to skip VERIFY
    via a production rule.

    POST_LOOP: After the loop completes, record the experience triple.
    When enough successful triples cluster together by preset, compile
    a new chunk. Failed chunks get retired (self-correcting).

    Usage:
        compiler = ChunkCompiler()
        registry.register(compiler)
        # ChunkCompiler automatically learns and applies chunks
    """

    def __init__(self, config: Optional[ChunkConfig] = None):
        self._config = config or ChunkConfig()
        self._chunks: List[CompiledChunk] = []
        self._experience: List[ExperienceRecord] = []

    @property
    def name(self) -> str:
        return "chunk_compiler"

    @property
    def phases(self) -> tuple:
        return (Phase.PRE_SHELL, Phase.POST_LOOP)

    def activation_level(self, blackboard: Any) -> float:
        """Activation depends on state: high if chunks exist, low if just recording."""
        if self._chunks:
            return 0.8  # Compiled knowledge — high priority
        if len(self._experience) >= self._config.min_cluster_size:
            return 0.3  # Ready to attempt compilation
        return 0.1  # Just recording

    def contribute(self, blackboard: Any, graph: Any, context: Dict[str, Any]) -> None:
        """Route to appropriate phase handler."""
        if "shell_type" in context:
            # PRE_SHELL: check for matching chunk
            self._try_chunk_match(blackboard, context)
        elif "loop_result" in context:
            # POST_LOOP: record experience and attempt compilation
            self._record_experience(blackboard, context)
            self._attempt_compilation(graph)

    # =========================================================================
    # PRE_SHELL: Chunk Matching
    # =========================================================================

    def _try_chunk_match(self, blackboard, context) -> None:
        """Check if a compiled chunk matches the current query features.

        Only fires for PRIME shell. If a chunk matches, posts the action
        to blackboard.flags["chunk_action"] for the orchestrator to use.
        """
        # Import here to avoid circular dependency
        from .orchestrator import ShellType

        if context.get("shell_type") != ShellType.PRIME:
            return

        if blackboard is None:
            return

        features = blackboard.context_features()
        best_chunk = None
        best_rate = 0.0

        for chunk in self._chunks:
            if chunk.is_retired:
                continue
            if chunk.matches(features) and chunk.hit_rate > best_rate:
                best_chunk = chunk
                best_rate = chunk.hit_rate

        if best_chunk and best_rate >= self._config.min_hit_rate:
            blackboard.post_flag("chunk_action", {
                "chunk_id": best_chunk.chunk_id,
                "preset_id": best_chunk.action.preset_id,
                "max_shells": best_chunk.action.max_shells,
                "skip_verify": best_chunk.action.skip_verify,
                "hit_rate": best_chunk.hit_rate,
            })
            logger.info(
                "ChunkCompiler: matched chunk '%s' (preset=%s, hit_rate=%.2f)",
                best_chunk.chunk_id, best_chunk.action.preset_id, best_rate,
            )

    # =========================================================================
    # POST_LOOP: Experience Recording + Compilation
    # =========================================================================

    def _record_experience(self, blackboard, context) -> None:
        """Record the (features, preset, reward, shells) from this loop."""
        if blackboard is None:
            return

        loop_result = context["loop_result"]
        features = blackboard.context_features()

        # Determine which preset was used
        chunk_info = blackboard.flags.get("chunk_action")
        preset_used = "balanced"  # default
        if chunk_info:
            preset_used = chunk_info.get("preset_id", "balanced")

        reward = loop_result.final_confidence.combined

        record = ExperienceRecord(
            features=features,
            preset_id=preset_used,
            reward=reward,
            shell_count=loop_result.shell_count,
            query_id=getattr(blackboard, "query_id", ""),
        )
        self._experience.append(record)

        # Update existing chunk stats if one was used
        if chunk_info:
            chunk_id = chunk_info.get("chunk_id")
            for chunk in self._chunks:
                if chunk.chunk_id == chunk_id:
                    if reward >= self._config.success_threshold:
                        chunk.successes += 1
                        chunk.consecutive_failures = 0
                    else:
                        chunk.failures += 1
                        chunk.consecutive_failures += 1
                    self._auto_tune_chunk(chunk)
                    break

        # Trim buffer
        if len(self._experience) > self._config.max_buffer_size:
            self._experience = self._experience[-self._config.max_buffer_size:]

    def _attempt_compilation(self, graph) -> None:
        """Try to compile new chunks from clustered successful experience.

        Algorithm:
          1. Filter to successful experiences (reward >= threshold)
          2. Group by preset_id (same action)
          3. For each group, compute feature centroid and spread
          4. If spread is tight enough, compile a chunk
          5. Optionally store chunk as graph node
        """
        successful = [
            e for e in self._experience
            if e.reward >= self._config.success_threshold
        ]

        if len(successful) < self._config.min_cluster_size:
            return

        # Group by preset
        by_preset: Dict[str, List[ExperienceRecord]] = defaultdict(list)
        for exp in successful:
            by_preset[exp.preset_id].append(exp)

        for preset_id, group in by_preset.items():
            if len(group) < self._config.min_cluster_size:
                continue
            if len(self._chunks) >= self._config.max_chunks:
                break

            # Check if already compiled for this preset (and not retired)
            existing = any(
                c.action.preset_id == preset_id and not c.is_retired
                for c in self._chunks
            )
            if existing:
                continue

            # Compute feature ranges from cluster
            trigger = self._compute_trigger_ranges(group)
            if not trigger:
                continue

            # Determine optimal shell count from experience
            avg_shells = sum(e.shell_count for e in group) / len(group)

            chunk = CompiledChunk(
                chunk_id=uuid.uuid4().hex[:10],
                trigger=trigger,
                action=ChunkAction(
                    preset_id=preset_id,
                    max_shells=max(1, round(avg_shells)),
                    skip_verify=avg_shells < 1.5,
                ),
                successes=len(group),
            )
            self._chunks.append(chunk)

            # Store in graph
            if self._config.store_in_graph:
                self._store_chunk_node(graph, chunk, group)

            logger.info(
                "ChunkCompiler: compiled new chunk '%s' "
                "(preset=%s, shells=%d, trigger_features=%d, from %d examples)",
                chunk.chunk_id, preset_id, chunk.action.max_shells,
                len(trigger), len(group),
            )

    # Query-intrinsic features (stable across similar queries).
    # Excludes wm_overlap (depends on accumulated WM state) and
    # confidence (depends on which shell ran) — both are system state,
    # not query characteristics, and poison trigger ranges.
    _TRIGGER_FEATURES = [
        "query_length", "seed_count", "ppr_entropy", "entity_ratio",
    ]

    def _compute_trigger_ranges(
        self, group: List[ExperienceRecord],
    ) -> Optional[Dict[str, Tuple[float, float]]]:
        """Compute (min, max) ranges for each feature across the cluster.

        Only includes features with tight spread (< spread_limit).
        Returns None if not enough features are tight.
        Uses only query-intrinsic features — volatile system state
        (wm_overlap, confidence) is excluded to prevent false misses.
        """
        ranges: Dict[str, Tuple[float, float]] = {}

        for fname in self._TRIGGER_FEATURES:
            values = [e.features.get(fname, 0.0) for e in group]
            lo, hi = min(values), max(values)
            spread = hi - lo

            if spread < self._config.feature_spread_limit:
                # Wider margin: 15% minimum, 35% of spread
                margin = max(0.15, spread * 0.35)
                ranges[fname] = (
                    max(0.0, lo - margin),
                    min(1.0, hi + margin),
                )

        if len(ranges) >= self._config.min_trigger_features:
            return ranges
        return None

    def _auto_tune_chunk(self, chunk: CompiledChunk) -> None:
        """Auto-tune a chunk's trigger ranges based on accumulated experience.

        Called when a chunk has accumulated enough evaluations since last tune.
        - hit_rate >= 0.8: tighten ranges (shrink toward centroid)
        - hit_rate < 0.6: widen ranges (expand from centroid)
        - 0.6 <= hit_rate < 0.8: recompute from matching successful experiences
        """
        total = chunk.successes + chunk.failures
        if (total - chunk.last_tuned_at) < self._config.auto_tune_interval:
            return

        hit_rate = chunk.hit_rate

        if hit_rate >= 0.8:
            factor = self._config.auto_tune_shrink
            action = "tighten"
        elif hit_rate < 0.6:
            factor = self._config.auto_tune_expand
            action = "widen"
        else:
            # Mid range: recompute from matching successful experiences
            matching = [
                e for e in self._experience
                if e.preset_id == chunk.action.preset_id
                and chunk.matches(e.features)
                and e.reward >= self._config.success_threshold
            ]
            if matching:
                new_trigger = self._compute_trigger_ranges(matching)
                if new_trigger:
                    chunk.trigger = new_trigger
                    action = "recompute"
                else:
                    action = "recompute_noop"
            else:
                action = "recompute_noop"
            chunk.last_tuned_at = total
            logger.info(
                "ChunkCompiler: auto-tune '%s' action=%s hit_rate=%.2f total=%d",
                chunk.chunk_id, action, hit_rate, total,
            )
            return

        # Tighten or widen: scale range width around centroid
        new_trigger: Dict[str, Tuple[float, float]] = {}
        for fname, (lo, hi) in chunk.trigger.items():
            centroid = (lo + hi) / 2.0
            half_width = (hi - lo) / 2.0
            new_half = half_width * factor
            new_lo = max(0.0, centroid - new_half)
            new_hi = min(1.0, centroid + new_half)
            new_trigger[fname] = (new_lo, new_hi)
        chunk.trigger = new_trigger
        chunk.last_tuned_at = total

        logger.info(
            "ChunkCompiler: auto-tune '%s' action=%s hit_rate=%.2f total=%d",
            chunk.chunk_id, action, hit_rate, total,
        )

    def _store_chunk_node(
        self, graph, chunk: CompiledChunk, experiences: List[ExperienceRecord],
    ) -> None:
        """Store compiled chunk as a graph node with COMPILED_FROM edges."""
        bus = getattr(graph, '_bus', None)
        if bus is None or not hasattr(bus, '_items'):
            return

        try:
            from hololoom.memory.lite_bus import StoredItem
        except ImportError:
            return

        chunk_content = json.dumps(chunk.to_dict())
        chunk_node_id = f"chunk_{chunk.chunk_id}"
        item = StoredItem(
            id=chunk_node_id,
            content=f"[chunk] {chunk_content}",
            memory_type="chunk",
            importance=0.6,
            timestamp=datetime.utcnow().isoformat(),
        )

        bus._items[chunk_node_id] = item
        chunk.node_id = chunk_node_id

        # Add COMPILED_FROM edges to source experience contexts
        if chunk_node_id not in bus._edges:
            bus._edges[chunk_node_id] = []
        for exp in experiences[:5]:  # Link to top 5
            if exp.query_id and graph.has_node(exp.query_id):
                bus._edges[chunk_node_id].append(
                    (exp.query_id, "COMPILED_FROM")
                )

    # =========================================================================
    # Public API
    # =========================================================================

    def to_snapshot(self) -> Dict[str, Any]:
        """Serialize compiled chunks and experience buffer for persistence."""
        return {
            "chunks": [c.to_snapshot() for c in self._chunks],
            "experience": [e.to_snapshot() for e in self._experience],
        }

    def from_snapshot(self, data: Dict[str, Any]) -> None:
        """Restore chunks and experience from snapshot dict."""
        self._chunks.clear()
        for chunk_dict in data.get("chunks", []):
            self._chunks.append(CompiledChunk.from_snapshot(chunk_dict))
        self._experience.clear()
        for exp_dict in data.get("experience", []):
            self._experience.append(ExperienceRecord.from_snapshot(exp_dict))
        logger.info(
            "ChunkCompiler: loaded %d chunks, %d experiences from snapshot",
            len(self._chunks), len(self._experience),
        )

    @property
    def chunks(self) -> List[CompiledChunk]:
        """Read-only access to compiled chunks."""
        return list(self._chunks)

    @property
    def active_chunks(self) -> List[CompiledChunk]:
        """Non-retired chunks."""
        return [c for c in self._chunks if not c.is_retired]

    @property
    def experience_count(self) -> int:
        return len(self._experience)

    def add_chunk(self, chunk: CompiledChunk) -> None:
        """Manually add a pre-compiled chunk."""
        self._chunks.append(chunk)

    def retire_chunk(self, chunk_id: str) -> None:
        """Force-retire a chunk."""
        for chunk in self._chunks:
            if chunk.chunk_id == chunk_id:
                chunk.consecutive_failures = self._config.retirement_threshold
                break

    def report(self) -> Dict[str, Any]:
        """Diagnostic info about chunk compiler state."""
        return {
            "total_chunks": len(self._chunks),
            "active_chunks": len(self.active_chunks),
            "retired_chunks": len(self._chunks) - len(self.active_chunks),
            "experience_buffer_size": len(self._experience),
            "chunks": [c.to_dict() for c in self._chunks],
        }

"""
KnowledgeSource — Classical blackboard architecture composition layer.

Every classical AI pattern is a KnowledgeSource: a specialist that reads
the shared workspace, does work, and writes results back. The orchestrator
invokes sources between its fixed pipeline steps, never replacing them.

The fixed pipeline per shell:
    [Navigate] → POST_NAVIGATE → [Confidence] → POST_CONFIDENCE
    → [Format] → POST_FORMAT → [Generate] → POST_GENERATE

Plus loop-level hooks:
    PRE_SHELL  — before each shell begins
    POST_LOOP  — after the full PRIME→VERIFY→FLAG loop

KnowledgeSources register for specific phases and contribute when their
activation exceeds a threshold. Empty registry = existing behavior unchanged.

Design insight: every classical AI pattern reduces to a graph operation —
a new node type, a new edge type, and a traversal strategy. PPR already
traverses heterogeneous edges. The graph IS the unified representation.
The protocol simply formalizes what the architecture already wants to be.

Implemented patterns:
    TMS (tms.py)            — justification tracking + belief retraction
    ChunkCompiler (chunk_compiler.py) — SOAR-style compiled productions

Future patterns (same protocol, no orchestrator changes):
    CBR   — case retrieval + adaptation at POST_NAVIGATE
    Demons — active pattern monitors at POST_GENERATE
    Frames — structured slot filling at POST_NAVIGATE
    EBL   — explanation-based generalization at POST_LOOP

No external dependencies.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# =============================================================================
# Phase Enum
# =============================================================================

class Phase(Enum):
    """Hook points in the pipeline where KnowledgeSources can contribute.

    Each maps to a specific seam between the fixed pipeline steps
    in _execute_shell(), plus two loop-level hooks in run().

    Context dict contents per phase:
        POST_NAVIGATE:   {"nav_result": NavigatorResult}
        POST_CONFIDENCE: {"confidence": DualConfidence, "nav_result": NavigatorResult}
        POST_FORMAT:     {"context_block": ContextBlock}
        POST_GENERATE:   {"response": str, "shell_result": ShellResult}
        PRE_SHELL:       {"shell_type": ShellType}
        POST_LOOP:       {"loop_result": LoopResult}
    """
    POST_NAVIGATE   = "post_navigate"
    POST_CONFIDENCE = "post_confidence"
    POST_FORMAT     = "post_format"
    POST_GENERATE   = "post_generate"
    PRE_SHELL       = "pre_shell"
    POST_LOOP       = "post_loop"


# =============================================================================
# KnowledgeSource Protocol
# =============================================================================

@runtime_checkable
class KnowledgeSource(Protocol):
    """A specialist that contributes to the pipeline at specific phases.

    Every classical AI pattern implements this 4-method protocol:
        name             — human-readable identifier
        phases           — which Phase(s) this source cares about
        activation_level — how relevant is this source right now? (GWT attention)
        contribute       — do the work: read/write blackboard and graph

    Examples:
        TMS at POST_NAVIGATE:   record justification edges
        TMS at POST_CONFIDENCE: propagate retractions
        ChunkCompiler at PRE_SHELL:  check compiled chunks
        ChunkCompiler at POST_LOOP:  compile experience into chunks
        CBR at POST_NAVIGATE:   retrieve similar past cases
        Demons at POST_GENERATE: monitor for response patterns
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this knowledge source."""
        ...

    @property
    def phases(self) -> tuple:
        """Which pipeline phases this source activates at.

        Returns a tuple of Phase values. The source's contribute()
        method will only be called during these phases.
        """
        ...

    def activation_level(self, blackboard: Any) -> float:
        """How relevant is this source to the current blackboard state?

        Returns 0.0 to 1.0. Sources below the registry's threshold
        are skipped (GWT attention filter: not every specialist fires
        on every query).

        Args:
            blackboard: current Blackboard state
        """
        ...

    def contribute(self, blackboard: Any, graph: Any, context: Dict[str, Any]) -> None:
        """Read blackboard/graph state, do work, write results back.

        Sources mutate:
            - blackboard.flags (post signals for downstream consumers)
            - graph edges/nodes (add justifications, chunks, etc.)
            - context dict (enrich for downstream sources in same phase)

        Sources MUST NOT:
            - Replace fixed pipeline steps
            - Call the LLM directly

        Args:
            blackboard: shared workspace (Blackboard instance)
            graph: GraphBackend for node/edge operations
            context: phase-specific data (see Phase docstring)
        """
        ...


# =============================================================================
# Source Registry
# =============================================================================

class SourceRegistry:
    """Registry of KnowledgeSources with activation-gated dispatch.

    The orchestrator holds one SourceRegistry. At each hook point, it calls
    registry.invoke(phase, blackboard, graph, context). Sources are sorted
    by activation level (highest first) and invoked sequentially.

    Design choices:
        - Sequential within a phase (sources can depend on each other's writes)
        - Activation threshold filters dormant sources (no wasted cycles)
        - Enable/disable at runtime via register()/remove()
        - Empty registry = zero overhead (iterates empty list)

    Usage:
        registry = SourceRegistry()
        registry.register(TruthMaintenanceSystem(activation_adapter=adapter))
        registry.register(ChunkCompiler())

        # In orchestrator._execute_shell():
        registry.invoke(Phase.POST_NAVIGATE, blackboard, graph,
                        {"nav_result": nav_result})
    """

    def __init__(self, activation_threshold: float = 0.1):
        """
        Args:
            activation_threshold: minimum activation to invoke a source.
                Sources below this threshold are skipped entirely.
        """
        self._sources: List[KnowledgeSource] = []
        self._threshold = activation_threshold

    @property
    def sources(self) -> List[KnowledgeSource]:
        """Read-only access to registered sources."""
        return list(self._sources)

    def register(self, source: KnowledgeSource) -> None:
        """Add a knowledge source. Idempotent by name."""
        if not any(s.name == source.name for s in self._sources):
            self._sources.append(source)
            logger.info("SourceRegistry: registered '%s' for phases %s",
                        source.name, [p.value for p in source.phases])

    def remove(self, name: str) -> None:
        """Remove a knowledge source by name."""
        before = len(self._sources)
        self._sources = [s for s in self._sources if s.name != name]
        if len(self._sources) < before:
            logger.info("SourceRegistry: removed '%s'", name)

    def has(self, name: str) -> bool:
        """Check if a source is registered."""
        return any(s.name == name for s in self._sources)

    def invoke(
        self,
        phase: Phase,
        blackboard: Any,
        graph: Any,
        context: Dict[str, Any],
    ) -> None:
        """Invoke all active sources for a given phase.

        Sources are filtered by phase and activation threshold, then
        sorted by activation level descending (highest priority first).
        Each source's contribute() is called sequentially so that
        downstream sources can see upstream writes.

        Args:
            phase: which pipeline hook point
            blackboard: shared workspace (may be None for backward compat)
            graph: GraphBackend instance
            context: phase-specific data dict
        """
        if not self._sources or blackboard is None:
            return

        # Filter to sources that care about this phase and are above threshold
        candidates = []
        for source in self._sources:
            if phase not in source.phases:
                continue
            try:
                level = source.activation_level(blackboard)
            except Exception as e:
                logger.debug("SourceRegistry: activation error in '%s': %s",
                             source.name, e)
                continue

            if level >= self._threshold:
                candidates.append((level, source))

        if not candidates:
            return

        # Highest activation first
        candidates.sort(key=lambda x: x[0], reverse=True)

        for level, source in candidates:
            try:
                source.contribute(blackboard, graph, context)
            except Exception as e:
                logger.warning(
                    "SourceRegistry: error in '%s' at %s: %s",
                    source.name, phase.value, e,
                )

    def report(self) -> Dict[str, Any]:
        """Diagnostic info about registered sources."""
        return {
            "source_count": len(self._sources),
            "threshold": self._threshold,
            "sources": [
                {"name": s.name, "phases": [p.value for p in s.phases]}
                for s in self._sources
            ],
        }

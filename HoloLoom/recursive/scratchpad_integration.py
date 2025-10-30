#!/usr/bin/env python3
"""
Scratchpad Integration - Phase 1 of Recursive Learning Vision
==============================================================
Connects HoloLoom WeavingOrchestrator with Promptly Scratchpad for
full provenance tracking and recursive refinement.

Architecture:
    ProvenanceTracker: Extracts trace -> scratchpad entries
    ScratchpadOrchestrator: HoloLoom + Scratchpad wrapper
    RecursiveRefiner: Triggers refinement on low confidence

Philosophy:
    Every weaving cycle gets recorded in scratchpad, creating a complete
    audit trail of reasoning. When confidence is low, automatically trigger
    recursive refinement loops.

Author: Claude Code
Date: 2025-10-29 (Phase 1 Implementation)
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# HoloLoom components
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.config import Config

# Promptly components
from Promptly.promptly.recursive_loops import (
    Scratchpad,
    ScratchpadEntry,
    RecursiveEngine,
    LoopConfig,
    LoopType,
    LoopResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Provenance Extraction
# ============================================================================

class ProvenanceTracker:
    """
    Extracts provenance information from Spacetime traces into Scratchpad format.

    Converts HoloLoom's WeavingTrace (computational trace) into Scratchpad
    entries (thought -> action -> observation -> score).
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ProvenanceTracker")

    def extract_provenance(
        self,
        spacetime: Spacetime,
        iteration: Optional[int] = None
    ) -> ScratchpadEntry:
        """
        Extract provenance from Spacetime into ScratchpadEntry.

        Maps:
        - Thought: Why threads were activated, what motifs detected
        - Action: Tool selected, adapter used
        - Observation: Response summary, confidence
        - Score: Tool confidence from policy

        Args:
            spacetime: Woven fabric with trace
            iteration: Optional iteration number

        Returns:
            ScratchpadEntry with full provenance
        """
        trace = spacetime.trace

        # THOUGHT: Feature extraction and retrieval
        thought_parts = []

        if trace.motifs_detected:
            thought_parts.append(
                f"Detected motifs: {', '.join(trace.motifs_detected[:3])}"
            )

        if trace.threads_activated:
            thought_parts.append(
                f"Activated {len(trace.threads_activated)} threads: "
                f"{', '.join(trace.threads_activated[:2])}"
            )
        else:
            thought_parts.append("No threads activated")

        if trace.embedding_scales_used:
            thought_parts.append(
                f"Embedding scales: {trace.embedding_scales_used}"
            )

        thought = " | ".join(thought_parts) if thought_parts else "Processing query"

        # ACTION: Policy decision
        action = f"Tool: {trace.tool_selected}, Adapter: {trace.policy_adapter}"

        # OBSERVATION: Response summary
        response_preview = spacetime.response[:150]
        if len(spacetime.response) > 150:
            response_preview += "..."

        observation_parts = [
            f"Response: {response_preview}",
            f"Duration: {trace.duration_ms:.1f}ms"
        ]

        if trace.context_shards_count > 0:
            observation_parts.append(
                f"Context: {trace.context_shards_count} shards"
            )

        if trace.errors:
            observation_parts.append(f"Errors: {len(trace.errors)}")

        observation = " | ".join(observation_parts)

        # SCORE: Confidence
        score = trace.tool_confidence

        # Metadata: Full trace details
        metadata = {
            "query": spacetime.query_text,
            "tool": trace.tool_selected,
            "adapter": trace.policy_adapter,
            "duration_ms": trace.duration_ms,
            "threads_count": len(trace.threads_activated),
            "motifs_count": len(trace.motifs_detected),
            "errors_count": len(trace.errors),
            "warnings_count": len(trace.warnings),
        }

        if trace.bandit_statistics:
            metadata["bandit_stats"] = trace.bandit_statistics

        # Create entry
        iter_num = iteration if iteration is not None else 1

        entry = ScratchpadEntry(
            iteration=iter_num,
            thought=thought,
            action=action,
            observation=observation,
            score=score,
            metadata=metadata
        )

        self.logger.debug(
            f"Extracted provenance: {len(trace.threads_activated)} threads, "
            f"confidence={score:.2f}"
        )

        return entry


# ============================================================================
# Scratchpad Orchestrator
# ============================================================================

@dataclass
class ScratchpadConfig:
    """Configuration for scratchpad integration"""
    enable_scratchpad: bool = True
    enable_refinement: bool = True
    refinement_threshold: float = 0.75
    max_refinement_iterations: int = 3
    persist_scratchpad: bool = False
    persist_path: Optional[str] = None


class ScratchpadOrchestrator:
    """
    HoloLoom + Scratchpad integration for full provenance tracking.

    Wraps WeavingOrchestrator and automatically logs all weaving cycles
    to a Scratchpad, creating complete reasoning history.

    Features:
    - Automatic provenance extraction from Spacetime traces
    - Scratchpad accumulation across queries
    - Optional recursive refinement on low confidence
    - Scratchpad persistence (optional)

    Usage:
        config = Config.fast()
        scratchpad_config = ScratchpadConfig(enable_refinement=True)

        async with ScratchpadOrchestrator(
            cfg=config,
            shards=shards,
            scratchpad_config=scratchpad_config
        ) as orchestrator:
            spacetime, scratchpad = await orchestrator.weave_with_provenance(query)

            # View reasoning history
            print(scratchpad.get_history())
    """

    def __init__(
        self,
        cfg: Config,
        shards: Optional[List[MemoryShard]] = None,
        memory: Optional[Any] = None,
        scratchpad_config: Optional[ScratchpadConfig] = None
    ):
        """
        Initialize scratchpad orchestrator.

        Args:
            cfg: HoloLoom configuration
            shards: Memory shards (optional)
            memory: Dynamic memory backend (optional)
            scratchpad_config: Scratchpad configuration
        """
        self.cfg = cfg
        self.shards = shards or []
        self.memory = memory
        self.scratchpad_config = scratchpad_config or ScratchpadConfig()

        # Create components
        self.orchestrator: Optional[WeavingOrchestrator] = None
        self.scratchpad = Scratchpad() if self.scratchpad_config.enable_scratchpad else None
        self.provenance_tracker = ProvenanceTracker()

        # Refinement engine (lazy initialized)
        self._refiner: Optional['RecursiveRefiner'] = None

        # Statistics
        self.queries_processed = 0
        self.refinements_triggered = 0

        self.logger = logging.getLogger(f"{__name__}.ScratchpadOrchestrator")

    async def __aenter__(self):
        """Async context manager entry"""
        # Create orchestrator
        self.orchestrator = WeavingOrchestrator(
            cfg=self.cfg,
            shards=self.shards,
            memory=self.memory
        )
        await self.orchestrator.__aenter__()

        self.logger.info(
            f"ScratchpadOrchestrator initialized (scratchpad={self.scratchpad_config.enable_scratchpad}, "
            f"refinement={self.scratchpad_config.enable_refinement})"
        )

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.orchestrator:
            await self.orchestrator.__aexit__(exc_type, exc_val, exc_tb)

        # Persist scratchpad if configured
        if self.scratchpad and self.scratchpad_config.persist_scratchpad:
            await self._persist_scratchpad()

        self.logger.info(
            f"ScratchpadOrchestrator closed (queries={self.queries_processed}, "
            f"refinements={self.refinements_triggered})"
        )

    async def weave(self, query: Query) -> Spacetime:
        """
        Standard weave interface (for compatibility).

        Calls weave_with_provenance but only returns Spacetime.
        """
        spacetime, _ = await self.weave_with_provenance(query)
        return spacetime

    async def weave_with_provenance(
        self,
        query: Query
    ) -> Tuple[Spacetime, Optional[Scratchpad]]:
        """
        Process query with full provenance tracking.

        Executes HoloLoom weaving cycle and records provenance in scratchpad.
        If confidence is below threshold, triggers recursive refinement.

        Args:
            query: Query to process

        Returns:
            Tuple of (Spacetime, Scratchpad with reasoning history)
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialized. Use async context manager.")

        self.logger.info(f"Weaving with provenance: '{query.text[:50]}...'")

        # Process with HoloLoom
        spacetime = await self.orchestrator.weave(query)

        # Extract provenance
        if self.scratchpad:
            entry = self.provenance_tracker.extract_provenance(
                spacetime,
                iteration=len(self.scratchpad.entries) + 1
            )
            self.scratchpad.entries.append(entry)

            self.logger.debug(
                f"Scratchpad entry added: confidence={entry.score:.2f}, "
                f"tool={spacetime.trace.tool_selected}"
            )

        self.queries_processed += 1

        # Check if refinement needed
        if (
            self.scratchpad_config.enable_refinement and
            spacetime.trace.tool_confidence < self.scratchpad_config.refinement_threshold
        ):
            self.logger.info(
                f"Low confidence ({spacetime.trace.tool_confidence:.2f} < "
                f"{self.scratchpad_config.refinement_threshold}), triggering refinement"
            )

            spacetime = await self._refine(query, spacetime)
            self.refinements_triggered += 1

        return spacetime, self.scratchpad

    async def _refine(
        self,
        query: Query,
        initial_spacetime: Spacetime
    ) -> Spacetime:
        """
        Trigger recursive refinement loop.

        Args:
            query: Original query
            initial_spacetime: Initial low-confidence result

        Returns:
            Refined Spacetime (or original if refinement fails)
        """
        if not self._refiner:
            self._refiner = RecursiveRefiner(self.orchestrator, self.scratchpad)

        try:
            refined = await self._refiner.refine(
                query=query,
                initial_spacetime=initial_spacetime,
                max_iterations=self.scratchpad_config.max_refinement_iterations,
                quality_threshold=0.9
            )

            self.logger.info(
                f"Refinement complete: {initial_spacetime.trace.tool_confidence:.2f} -> "
                f"{refined.trace.tool_confidence:.2f}"
            )

            return refined

        except Exception as e:
            self.logger.error(f"Refinement failed: {e}")
            return initial_spacetime

    async def _persist_scratchpad(self):
        """Persist scratchpad to disk"""
        if not self.scratchpad or not self.scratchpad_config.persist_path:
            return

        try:
            import json
            from pathlib import Path

            path = Path(self.scratchpad_config.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "queries_processed": self.queries_processed,
                "refinements_triggered": self.refinements_triggered,
                "entries": [
                    {
                        "iteration": e.iteration,
                        "thought": e.thought,
                        "action": e.action,
                        "observation": e.observation,
                        "score": e.score,
                        "metadata": e.metadata
                    }
                    for e in self.scratchpad.entries
                ]
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Scratchpad persisted: {path}")

        except Exception as e:
            self.logger.error(f"Failed to persist scratchpad: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        stats = {
            "queries_processed": self.queries_processed,
            "refinements_triggered": self.refinements_triggered,
            "scratchpad_entries": len(self.scratchpad.entries) if self.scratchpad else 0,
        }

        if self.scratchpad and self.scratchpad.entries:
            confidences = [e.score for e in self.scratchpad.entries if e.score is not None]
            if confidences:
                stats["avg_confidence"] = sum(confidences) / len(confidences)
                stats["min_confidence"] = min(confidences)
                stats["max_confidence"] = max(confidences)

        return stats


# ============================================================================
# Recursive Refiner
# ============================================================================

class RecursiveRefiner:
    """
    Triggers recursive refinement loops on low-confidence results.

    When confidence is below threshold, iteratively refine the query
    by expanding retrieval, adding context, or reformulating.
    """

    def __init__(
        self,
        orchestrator: WeavingOrchestrator,
        scratchpad: Optional[Scratchpad] = None
    ):
        """
        Initialize recursive refiner.

        Args:
            orchestrator: HoloLoom orchestrator
            scratchpad: Optional scratchpad for tracking
        """
        self.orchestrator = orchestrator
        self.scratchpad = scratchpad
        self.logger = logging.getLogger(f"{__name__}.RecursiveRefiner")

    async def refine(
        self,
        query: Query,
        initial_spacetime: Spacetime,
        max_iterations: int = 3,
        quality_threshold: float = 0.9
    ) -> Spacetime:
        """
        Iteratively refine query until confidence threshold met.

        Strategy:
        1. Analyze why confidence is low (missing context, ambiguity, etc.)
        2. Expand query with additional context from initial result
        3. Re-weave with expanded query
        4. Repeat until confidence threshold or max iterations

        Args:
            query: Original query
            initial_spacetime: Initial low-confidence result
            max_iterations: Maximum refinement iterations
            quality_threshold: Target confidence threshold

        Returns:
            Refined Spacetime
        """
        current_spacetime = initial_spacetime
        improvement_history = [initial_spacetime.trace.tool_confidence]

        self.logger.info(
            f"Starting refinement: initial_confidence={initial_spacetime.trace.tool_confidence:.2f}, "
            f"target={quality_threshold}"
        )

        for iteration in range(1, max_iterations + 1):
            # Analyze why confidence is low
            analysis = self._analyze_low_confidence(current_spacetime)

            # Expand query based on analysis
            expanded_query = self._expand_query(
                query,
                current_spacetime,
                analysis,
                iteration
            )

            self.logger.debug(
                f"Refinement iteration {iteration}: {expanded_query.text[:80]}..."
            )

            # Re-weave with expanded query
            refined_spacetime = await self.orchestrator.weave(expanded_query)

            # Track improvement
            new_confidence = refined_spacetime.trace.tool_confidence
            improvement_history.append(new_confidence)

            # Update scratchpad
            if self.scratchpad:
                self.scratchpad.add_entry(
                    thought=f"Refinement iteration {iteration}: {analysis}",
                    action=f"Expanded query with {expanded_query.metadata.get('expansion_strategy', 'context')}",
                    observation=f"Confidence: {new_confidence:.2f} (was {current_spacetime.trace.tool_confidence:.2f})",
                    score=new_confidence
                )

            # Check stop conditions
            if new_confidence >= quality_threshold:
                self.logger.info(
                    f"Quality threshold reached: {new_confidence:.2f} >= {quality_threshold}"
                )
                return refined_spacetime

            # Check if improving
            if iteration > 1 and abs(new_confidence - improvement_history[-2]) < 0.02:
                self.logger.info(
                    f"No significant improvement, stopping at iteration {iteration}"
                )
                return refined_spacetime

            current_spacetime = refined_spacetime

        self.logger.info(
            f"Max iterations reached: final_confidence={current_spacetime.trace.tool_confidence:.2f}"
        )

        return current_spacetime

    def _analyze_low_confidence(self, spacetime: Spacetime) -> str:
        """
        Analyze why confidence is low.

        Looks at:
        - Few threads activated (missing context)
        - No motifs detected (ambiguous query)
        - Low context shard count
        - Errors in trace

        Returns:
            Analysis string
        """
        trace = spacetime.trace
        issues = []

        if len(trace.threads_activated) < 2:
            issues.append("few threads activated")

        if len(trace.motifs_detected) == 0:
            issues.append("no motifs detected")

        if trace.context_shards_count == 0:
            issues.append("no context retrieved")

        if trace.errors:
            issues.append(f"{len(trace.errors)} errors")

        if not issues:
            issues.append("low policy confidence")

        return ", ".join(issues)

    def _expand_query(
        self,
        original_query: Query,
        current_spacetime: Spacetime,
        analysis: str,
        iteration: int
    ) -> Query:
        """
        Expand query based on analysis.

        Strategies:
        - Add context from current response
        - Add detected motifs as constraints
        - Reformulate for clarity

        Args:
            original_query: Original query
            current_spacetime: Current result
            analysis: Why confidence is low
            iteration: Current iteration

        Returns:
            Expanded query
        """
        expansions = []

        # Add context from response
        if "few threads" in analysis or "no context" in analysis:
            # Extract key terms from response
            response_preview = current_spacetime.response[:100]
            expansions.append(f"Context: {response_preview}")

        # Add motifs as constraints
        if current_spacetime.trace.motifs_detected:
            motifs_str = ", ".join(current_spacetime.trace.motifs_detected[:2])
            expansions.append(f"Related to: {motifs_str}")

        # Build expanded query text
        expanded_text = original_query.text
        if expansions:
            expanded_text += " [" + " | ".join(expansions) + "]"

        # Create new query with expansion metadata
        expanded_query = Query(
            text=expanded_text,
            metadata={
                **original_query.metadata,
                "refinement_iteration": iteration,
                "expansion_strategy": analysis,
                "original_query": original_query.text
            }
        )

        return expanded_query


# ============================================================================
# Convenience Functions
# ============================================================================

async def weave_with_scratchpad(
    query: Query,
    cfg: Config,
    shards: Optional[List[MemoryShard]] = None,
    enable_refinement: bool = True
) -> Tuple[Spacetime, Scratchpad]:
    """
    Convenience function for one-off weaving with scratchpad.

    Args:
        query: Query to process
        cfg: HoloLoom configuration
        shards: Memory shards
        enable_refinement: Enable automatic refinement on low confidence

    Returns:
        Tuple of (Spacetime, Scratchpad)

    Usage:
        from HoloLoom.recursive import weave_with_scratchpad
        from HoloLoom.config import Config
        from HoloLoom.documentation.types import Query

        spacetime, scratchpad = await weave_with_scratchpad(
            Query(text="How does Thompson Sampling work?"),
            Config.fast(),
            shards=shards
        )

        print(scratchpad.get_history())
    """
    scratchpad_config = ScratchpadConfig(enable_refinement=enable_refinement)

    async with ScratchpadOrchestrator(
        cfg=cfg,
        shards=shards,
        scratchpad_config=scratchpad_config
    ) as orchestrator:
        return await orchestrator.weave_with_provenance(query)


if __name__ == "__main__":
    print("HoloLoom Recursive Learning - Phase 1: Scratchpad Integration")
    print()
    print("Components:")
    print("  - ProvenanceTracker: Extract Spacetime trace -> Scratchpad")
    print("  - ScratchpadOrchestrator: HoloLoom + Scratchpad wrapper")
    print("  - RecursiveRefiner: Auto-refine low confidence results")
    print()
    print("Usage:")
    print("""
from HoloLoom.recursive import ScratchpadOrchestrator, ScratchpadConfig
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query

config = Config.fast()
scratchpad_config = ScratchpadConfig(
    enable_refinement=True,
    refinement_threshold=0.75
)

async with ScratchpadOrchestrator(
    cfg=config,
    shards=shards,
    scratchpad_config=scratchpad_config
) as orchestrator:
    spacetime, scratchpad = await orchestrator.weave_with_provenance(
        Query(text="How does Thompson Sampling work?")
    )

    # View reasoning history
    print(scratchpad.get_history())

    # Get statistics
    print(orchestrator.get_statistics())
""")

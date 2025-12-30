#!/usr/bin/env python3
"""
Recursive Weaving Orchestrator - Promptly Integration
=====================================================

Enhanced WeavingOrchestrator with recursive reasoning capabilities.

This integrates Promptly's recursive intelligence into HoloLoom's
core weaving cycle, enabling multi-pass quality refinement.

Created: 2025-11-15
Integration: Promptly → HoloLoom weaving layer

Architecture:
```
┌──────────────────────────────────────────────────────┐
│          Enhanced Weaving Cycle                      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. Loom Command → Pattern Card (BARE/FAST/FUSED)  │
│  2. Chrono Trigger → Temporal Window                │
│  3. Yarn Graph → Thread Selection                   │
│  4. Resonance Shed → DotPlasma (features)           │
│  5. Warp Space → Continuous Manifold                │
│  6. Convergence Engine → Decision                   │
│     ↓                                                │
│     Quality Check                                    │
│     ↓                                                │
│  7. [NEW] Recursive Refinement (if quality < threshold)
│     • REFINE: Iterative improvement                 │
│     • CRITIQUE: Self-critique loop                  │
│     • DECOMPOSE: Break down → solve → combine       │
│     • EXPLORE: Multiple approaches → synthesize     │
│     • HOFSTADTER: Meta-reasoning                    │
│     ↓                                                │
│  8. Tool Execution → Results                        │
│  9. Spacetime Fabric → Response + Provenance        │
│                                                      │
└──────────────────────────────────────────────────────┘
```
"""

import logging
import time
from typing import Optional
from dataclasses import dataclass

from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard
from HoloLoom.fabric.spacetime import Spacetime
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.convergence.recursive_engine import (
    RecursiveConvergenceEngine,
    RecursiveConvergenceResult,
    create_recursive_engine
)
from HoloLoom.protocols.recursive_reasoning import (
    RecursiveConfig,
    ReasoningStrategy,
    ReasoningJournal
)
from HoloLoom.telemetry.analytics.recursive_analytics import RecursiveAnalytics

logger = logging.getLogger(__name__)


@dataclass
class RecursiveSpacetime(Spacetime):
    """
    Enhanced Spacetime with reasoning provenance.

    Extends base Spacetime to include complete reasoning journal.
    """
    reasoning_journal: Optional[ReasoningJournal] = None
    iterations: int = 1
    strategy_used: Optional[ReasoningStrategy] = None


class RecursiveWeavingOrchestrator(WeavingOrchestrator):
    """
    Enhanced WeavingOrchestrator with recursive reasoning.

    Extends the base orchestrator to support:
    - Multi-pass quality refinement
    - Multiple recursive strategies (REFINE, CRITIQUE, DECOMPOSE, etc.)
    - Complete reasoning provenance via ReasoningJournal
    - Adaptive strategy selection based on query complexity

    Usage:
    ```python
    from HoloLoom.config import Config
    from HoloLoom.weaving_orchestrator_recursive import RecursiveWeavingOrchestrator
    from HoloLoom.protocols.recursive_reasoning import RecursiveConfig, ReasoningStrategy

    # Create with recursive reasoning enabled
    config = Config.fused()
    orchestrator = RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=True,
        quality_threshold=0.85
    )

    # Simple weave (auto-refinement if quality < threshold)
    spacetime = await orchestrator.weave(query)

    # Custom strategy
    spacetime = await orchestrator.weave_with_strategy(
        query,
        strategy=ReasoningStrategy.HOFSTADTER,
        max_iterations=5
    )

    # View reasoning provenance
    if spacetime.reasoning_journal:
        print(spacetime.reasoning_journal.get_history())
        print(f"Iterations: {spacetime.iterations}")
        print(f"Strategy: {spacetime.strategy_used}")
    ```
    """

    def __init__(
        self,
        cfg: Config,
        shards: list[MemoryShard],
        enable_recursive: bool = True,
        enable_analytics: bool = True,
        quality_threshold: float = 0.85,
        max_iterations: int = 3,
        default_strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE,
        analytics_db_path: str = ".hololoom/recursive_analytics.db",
        **kwargs
    ):
        """
        Initialize recursive weaving orchestrator.

        Args:
            cfg: HoloLoom configuration
            shards: Memory shards
            enable_recursive: Enable recursive reasoning
            enable_analytics: Enable analytics tracking
            quality_threshold: Minimum quality to skip refinement
            max_iterations: Maximum refinement iterations
            default_strategy: Default recursive strategy
            analytics_db_path: Path to analytics database
            **kwargs: Passed to base WeavingOrchestrator
        """
        # Initialize base orchestrator
        super().__init__(cfg=cfg, shards=shards, **kwargs)

        # Recursive reasoning configuration
        self.enable_recursive = enable_recursive
        self.enable_analytics = enable_analytics
        self.recursive_config = RecursiveConfig(
            strategy=default_strategy,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            enable_journal=True
        )

        # Create recursive convergence engine
        if enable_recursive:
            self.recursive_engine = create_recursive_engine(
                enable_recursive=True,
                quality_threshold=quality_threshold,
                max_iterations=max_iterations
            )

            # Set weaving function for recursive refinement
            # This creates a feedback loop: weave() can call itself recursively
            self.recursive_engine.set_weaving_function(self._recursive_weave_wrapper)

            logger.info(
                f"RecursiveWeavingOrchestrator initialized with recursive reasoning "
                f"(threshold={quality_threshold}, max_iterations={max_iterations})"
            )
        else:
            self.recursive_engine = None
            logger.info("RecursiveWeavingOrchestrator initialized (recursive disabled)")

        # Initialize analytics tracker
        if enable_analytics:
            self.analytics = RecursiveAnalytics(db_path=analytics_db_path)
            logger.info(f"Analytics tracking enabled (db={analytics_db_path})")
        else:
            self.analytics = None

    async def weave(
        self,
        query: Query,
        enable_refinement: bool = None
    ) -> RecursiveSpacetime:
        """
        Enhanced weave with automatic quality-based refinement.

        This is the main entry point that:
        1. Executes standard weaving cycle
        2. Checks quality
        3. Triggers recursive refinement if needed
        4. Returns enhanced spacetime with provenance

        Args:
            query: User's query
            enable_refinement: Override recursive refinement
                               (None = use instance setting)

        Returns:
            RecursiveSpacetime with response + reasoning journal
        """
        # Determine if refinement should be used
        use_refinement = (
            enable_refinement if enable_refinement is not None
            else self.enable_recursive
        )

        if not use_refinement or self.recursive_engine is None:
            # Standard weaving without refinement
            spacetime = await super().weave(query)
            return RecursiveSpacetime(
                response=spacetime.response,
                confidence=spacetime.confidence,
                metadata=spacetime.metadata,
                trace=spacetime.trace,
                reasoning_journal=None,
                iterations=1,
                strategy_used=ReasoningStrategy.DIRECT
            )

        # Execute weaving with recursive refinement
        return await self._weave_with_refinement(query, self.recursive_config)

    async def weave_with_strategy(
        self,
        query: Query,
        strategy: ReasoningStrategy,
        max_iterations: int = None,
        quality_threshold: float = None
    ) -> RecursiveSpacetime:
        """
        Weave with explicit recursive strategy.

        Use this for specific reasoning strategies:
        - REFINE: Iterative improvement
        - CRITIQUE: Self-critique loops
        - DECOMPOSE: Break down complex problems
        - EXPLORE: Multiple approaches
        - HOFSTADTER: Meta-reasoning

        Args:
            query: User's query
            strategy: Recursive reasoning strategy
            max_iterations: Override max iterations
            quality_threshold: Override quality threshold

        Returns:
            RecursiveSpacetime with provenance
        """
        if self.recursive_engine is None:
            raise RuntimeError(
                "Recursive reasoning not enabled. "
                "Initialize with enable_recursive=True"
            )

        # Create custom config for this strategy
        config = RecursiveConfig(
            strategy=strategy,
            max_iterations=max_iterations or self.recursive_config.max_iterations,
            quality_threshold=quality_threshold or self.recursive_config.quality_threshold,
            enable_journal=True
        )

        return await self._weave_with_refinement(query, config)

    async def _weave_with_refinement(
        self,
        query: Query,
        config: RecursiveConfig
    ) -> RecursiveSpacetime:
        """
        Internal method to execute weaving with recursive refinement.

        This coordinates:
        1. Feature extraction (Resonance Shed)
        2. Initial convergence
        3. Quality check
        4. Recursive refinement (if needed)
        5. Spacetime construction with provenance
        6. Analytics tracking (if enabled)

        Args:
            query: User's query
            config: Recursive reasoning configuration

        Returns:
            Enhanced spacetime with reasoning journal
        """
        # Start timing
        start_time = time.time()

        # Phase 1: Feature extraction (standard HoloLoom pipeline)
        # This would call Resonance Shed, Warp Space, etc.
        # For now, we call base weave to get features
        initial_spacetime = await super().weave(query)
        initial_quality = initial_spacetime.confidence

        # Extract features from initial pass
        features = initial_spacetime.metadata.get('features')

        # Phase 2: Recursive convergence with refinement
        result: RecursiveConvergenceResult = await self.recursive_engine.converge_with_refinement(
            query=query,
            features=features,
            config=config
        )

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Phase 3: Construct enhanced spacetime
        enhanced = RecursiveSpacetime(
            response=result.spacetime.response,
            confidence=result.final_quality,
            metadata={
                **result.spacetime.metadata,
                "recursive_reasoning": {
                    "iterations": result.iterations,
                    "strategy": result.strategy_used.value,
                    "quality_trajectory": result.journal.get_confidence_trajectory(),
                    "initial_quality": initial_quality,
                    "final_quality": result.final_quality,
                    "duration_ms": duration_ms
                }
            },
            trace=result.spacetime.trace,
            reasoning_journal=result.journal,
            iterations=result.iterations,
            strategy_used=result.strategy_used
        )

        logger.info(
            f"Recursive weaving complete: "
            f"iterations={result.iterations}, "
            f"strategy={result.strategy_used.value}, "
            f"final_quality={result.final_quality:.2f}"
        )

        # Phase 4: Track analytics (if enabled)
        if self.analytics:
            # Estimate tokens (rough heuristic: ~4 chars per token)
            estimated_tokens = (
                len(query.text) +
                sum(len(t.observation) for t in result.journal.traces)
            ) // 4 * result.iterations

            # Check if converged
            converged = result.journal.converged() if result.journal else False

            await self.analytics.track_execution(
                strategy=result.strategy_used,
                query_text=query.text,
                iterations=result.iterations,
                initial_quality=initial_quality,
                final_quality=result.final_quality,
                duration_ms=duration_ms,
                tokens_used=estimated_tokens,
                converged=converged
            )

        return enhanced

    async def _recursive_weave_wrapper(self, query: Query) -> Spacetime:
        """
        Wrapper for recursive reasoner to call weave().

        This prevents infinite recursion by disabling refinement
        when called from within a recursive loop.

        Args:
            query: Refinement query

        Returns:
            Standard spacetime (no further refinement)
        """
        # Call base weave without refinement
        return await super().weave(query)

    def get_reasoning_statistics(self) -> dict:
        """
        Get statistics about recursive reasoning usage.

        Returns:
            Statistics including:
            - Strategy usage frequency
            - Average iterations
            - Quality improvements
            - Common query patterns
            - Analytics summary (if enabled)
        """
        if self.recursive_engine is None:
            return {"recursive_enabled": False}

        stats = self.recursive_engine.get_reasoning_statistics()
        stats.update({
            "orchestrator_level_stats": {
                "default_strategy": self.recursive_config.strategy.value,
                "quality_threshold": self.recursive_config.quality_threshold,
                "max_iterations": self.recursive_config.max_iterations
            }
        })

        # Add analytics if available
        if self.analytics:
            stats["analytics"] = self.analytics.get_summary()

        return stats

    def get_analytics_summary(self) -> dict:
        """
        Get comprehensive analytics summary.

        Returns:
            Analytics summary with:
            - Overall statistics
            - Per-strategy breakdown
            - Quality trends
            - Recommendations
        """
        if not self.analytics:
            return {"analytics_enabled": False}

        summary = self.analytics.get_summary()
        summary["recommendations"] = self.analytics.get_recommendations()

        return summary

    def export_analytics(self, output_path: str):
        """
        Export analytics to CSV.

        Args:
            output_path: Path to output CSV file
        """
        if not self.analytics:
            raise RuntimeError("Analytics not enabled")

        self.analytics.export_to_csv(output_path)
        logger.info(f"Analytics exported to {output_path}")


# Convenience factory function
def create_recursive_orchestrator(
    config: Config,
    shards: list[MemoryShard],
    enable_recursive: bool = True,
    **kwargs
) -> RecursiveWeavingOrchestrator:
    """
    Factory to create recursive weaving orchestrator.

    Args:
        config: HoloLoom configuration
        shards: Memory shards
        enable_recursive: Enable recursive reasoning
        **kwargs: Additional orchestrator parameters

    Returns:
        Configured RecursiveWeavingOrchestrator
    """
    return RecursiveWeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_recursive=enable_recursive,
        **kwargs
    )

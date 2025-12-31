"""
HoloLoom Memory - Contribution Tracking Integration (Phase 3)
==============================================================
Integrates ShardContributionTracker with the weaving cycle for
automatic shard boost and outcome tracking.

Integration Points:
    1. pre_retrieval_hook: Record query, prepare boosting
    2. post_retrieval_hook: Apply boosts to results, record shard IDs
    3. post_weave_hook: Record outcome (confidence, feedback)
    4. retrieval_booster: Wrap retriever with automatic boosting

Usage:
    # Create integration
    integration = ContributionIntegration()

    # In orchestrator retrieval:
    boosted_results = integration.boost_retrieval(query_text, raw_results)
    query_id = integration.current_query_id

    # After weaving:
    integration.record_outcome(
        query_id=query_id,
        confidence=spacetime.confidence
    )

Author: Claude Code
Date: December 2025 (Phase 3 - Outcome→Retrieval Loop)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any

from HoloLoom.memory.shard_contribution import (
    ShardContributionTracker,
    ShardContributionConfig,
    RetrievalBooster,
    ShardStats,
    create_contribution_tracker,
)
from HoloLoom.config import Config

logger = logging.getLogger(__name__)


class ContributionIntegration:
    """
    Integration layer connecting ShardContributionTracker with the weaving cycle.

    Provides hooks for:
    - Pre-retrieval: Prepare query tracking
    - Post-retrieval: Apply boosts and record shard usage
    - Post-weave: Record outcome for feedback loop

    This creates the feedback loop where successful retrievals boost
    future retrieval of the same shards.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        tracker: Optional[ShardContributionTracker] = None
    ):
        """
        Initialize contribution integration.

        Args:
            config: HoloLoom config (uses contribution_ fields)
            tracker: Optional existing tracker (created from config if not provided)
        """
        self.config = config or Config.fast()

        # Create or use provided tracker
        if tracker is not None:
            self.tracker = tracker
        else:
            tracker_config = self._build_tracker_config()
            self.tracker = ShardContributionTracker(tracker_config)

        # Create booster
        self.booster = RetrievalBooster(
            tracker=self.tracker,
            blend_factor=self.config.contribution_boost_blend
        )

        # Current query tracking
        self._current_query_id: Optional[str] = None
        self._enabled = self.config.enable_contribution_tracking

        self.logger = logging.getLogger(f"{__name__}.ContributionIntegration")

    def _build_tracker_config(self) -> ShardContributionConfig:
        """Build tracker config from HoloLoom config."""
        return ShardContributionConfig(
            max_records=self.config.contribution_max_records,
            max_age_days=self.config.contribution_max_age_days,
            enable_decay=self.config.contribution_decay_enabled,
            decay_rate=self.config.contribution_decay_rate,
            decay_interval_hours=self.config.contribution_decay_interval_hours,
            min_retrievals_for_boost=self.config.contribution_min_retrievals,
            boost_floor=self.config.contribution_boost_floor,
            boost_ceiling=self.config.contribution_boost_ceiling,
            success_threshold=self.config.contribution_success_threshold,
            excellent_threshold=self.config.contribution_excellent_threshold,
        )

    @property
    def current_query_id(self) -> Optional[str]:
        """Get current query ID for outcome tracking."""
        return self._current_query_id

    @property
    def enabled(self) -> bool:
        """Check if contribution tracking is enabled."""
        return self._enabled

    def enable(self):
        """Enable contribution tracking."""
        self._enabled = True

    def disable(self):
        """Disable contribution tracking."""
        self._enabled = False

    def boost_retrieval(
        self,
        query_text: str,
        results: List[Tuple[Any, float]]
    ) -> List[Tuple[Any, float]]:
        """
        Apply boost factors to retrieval results and record retrieval.

        This is the main integration point for retrieval. Call this
        after raw retrieval to:
        1. Record which shards were retrieved
        2. Apply boost factors based on historical contribution
        3. Return reranked results

        Args:
            query_text: The query text
            results: Raw retrieval results as List[(shard, score)]

        Returns:
            Boosted and reranked results
        """
        if not self._enabled or not results:
            return results

        try:
            # Record retrieval and apply boosts
            query_id, boosted = self.booster.record_and_boost(
                query_text=query_text,
                results=results
            )

            self._current_query_id = query_id

            self.logger.debug(
                f"Boosted retrieval: query_id={query_id}, "
                f"shards={len(results)}"
            )

            return boosted

        except Exception as e:
            self.logger.warning(f"Boost retrieval failed: {e}")
            return results  # Return original on failure

    def record_outcome(
        self,
        query_id: Optional[str] = None,
        confidence: float = 0.0,
        feedback: Optional[float] = None,
        tool_used: Optional[str] = None
    ) -> bool:
        """
        Record the outcome of a query.

        Call this after weaving completes to record whether the
        retrieved shards led to a successful outcome.

        Args:
            query_id: The query ID (uses current if not provided)
            confidence: Outcome confidence (0-1)
            feedback: Optional user feedback (-1 to 1)
            tool_used: Optional tool that was used

        Returns:
            True if outcome was recorded
        """
        if not self._enabled:
            return False

        # Use current query ID if not provided
        qid = query_id or self._current_query_id
        if qid is None:
            self.logger.warning("No query ID for outcome recording")
            return False

        try:
            success = self.tracker.record_outcome(
                query_id=qid,
                confidence=confidence,
                feedback=feedback,
                tool_used=tool_used
            )

            if success:
                self.logger.debug(
                    f"Recorded outcome: query_id={qid}, "
                    f"confidence={confidence:.2f}"
                )

            return success

        except Exception as e:
            self.logger.warning(f"Record outcome failed: {e}")
            return False

    def get_shard_boost(self, shard_id: str) -> float:
        """
        Get boost factor for a single shard.

        Args:
            shard_id: The shard ID

        Returns:
            Boost factor (1.0 = neutral)
        """
        if not self._enabled:
            return 1.0
        return self.tracker.get_boost_factor(shard_id)

    def get_shard_stats(self, shard_id: str) -> Optional[ShardStats]:
        """
        Get statistics for a shard.

        Args:
            shard_id: The shard ID

        Returns:
            ShardStats or None if not tracked
        """
        return self.tracker.get_shard_stats(shard_id)

    def get_top_contributors(
        self,
        k: int = 10,
        min_retrievals: int = 5
    ) -> List[Tuple[str, ShardStats]]:
        """Get top contributing shards."""
        return self.tracker.get_top_contributors(k, min_retrievals)

    def get_poor_performers(
        self,
        k: int = 10,
        min_retrievals: int = 5
    ) -> List[Tuple[str, ShardStats]]:
        """Get worst performing shards (candidates for forgetting)."""
        return self.tracker.get_poor_performers(k, min_retrievals)

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        return self.tracker.get_statistics()


class EnhancedMemoryManager:
    """
    Memory manager wrapper with integrated contribution tracking.

    Wraps an existing MemoryManager and adds automatic contribution
    tracking to retrieve() calls.
    """

    def __init__(
        self,
        memory_manager: Any,  # MemoryManager
        integration: Optional[ContributionIntegration] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize enhanced memory manager.

        Args:
            memory_manager: The underlying MemoryManager
            integration: Optional existing integration
            config: HoloLoom config
        """
        self.memory_manager = memory_manager
        self.integration = integration or ContributionIntegration(config)

        # Forward attributes
        self.__dict__.update({
            k: v for k, v in memory_manager.__dict__.items()
            if not k.startswith('_')
        })

    async def retrieve(
        self,
        query: Any,  # Query
        kg_sub: Any,
        fast: bool = False
    ) -> Any:  # Context
        """
        Retrieve with automatic contribution tracking.

        Wraps the underlying retrieve() and:
        1. Performs retrieval
        2. Records shard usage
        3. Applies boost factors

        Args:
            query: Query object
            kg_sub: Knowledge graph subgraph
            fast: Use fast mode

        Returns:
            Context with boosted retrieval results
        """
        # Perform underlying retrieval
        context = await self.memory_manager.retrieve(query, kg_sub, fast)

        # Apply boosts if we have hits
        if hasattr(context, 'hits') and context.hits:
            boosted_hits = self.integration.boost_retrieval(
                query_text=query.text,
                results=context.hits
            )

            # Update context with boosted hits
            context.hits = boosted_hits

            # Recalculate relevance with boosted scores
            import numpy as np
            context.relevance = float(np.mean([
                score for _, score in boosted_hits
            ])) if boosted_hits else 0.0

        return context

    def record_outcome(
        self,
        confidence: float,
        feedback: Optional[float] = None,
        tool_used: Optional[str] = None
    ) -> bool:
        """Record outcome for current query."""
        return self.integration.record_outcome(
            confidence=confidence,
            feedback=feedback,
            tool_used=tool_used
        )

    @property
    def current_query_id(self) -> Optional[str]:
        """Get current query ID."""
        return self.integration.current_query_id

    def get_statistics(self) -> Dict[str, Any]:
        """Get contribution statistics."""
        return self.integration.get_statistics()


# ============================================================================
# Factory Functions
# ============================================================================

def create_contribution_integration(
    config: Optional[Config] = None
) -> ContributionIntegration:
    """
    Create a contribution integration instance.

    Args:
        config: HoloLoom config

    Returns:
        Configured integration
    """
    return ContributionIntegration(config)


def enhance_memory_manager(
    memory_manager: Any,
    config: Optional[Config] = None
) -> EnhancedMemoryManager:
    """
    Wrap a MemoryManager with contribution tracking.

    Args:
        memory_manager: The underlying MemoryManager
        config: HoloLoom config

    Returns:
        Enhanced memory manager with tracking
    """
    return EnhancedMemoryManager(memory_manager, config=config)


# ============================================================================
# ForgetManager Integration
# ============================================================================

def get_forgetting_candidates_from_contributions(
    tracker: ShardContributionTracker,
    min_retrievals: int = 10,
    max_success_rate: float = 0.3,
    limit: int = 100
) -> List[str]:
    """
    Get shard IDs that are candidates for forgetting based on poor contribution.

    Integrates with ForgetManager to provide intelligence-based forgetting.
    Shards that are consistently retrieved but lead to poor outcomes
    are candidates for eviction.

    Args:
        tracker: The contribution tracker
        min_retrievals: Minimum retrievals to be considered
        max_success_rate: Maximum success rate to be considered (lower = worse)
        limit: Maximum candidates to return

    Returns:
        List of shard IDs that should be considered for forgetting
    """
    poor = tracker.get_poor_performers(k=limit * 2, min_retrievals=min_retrievals)

    # Filter to those below success threshold
    candidates = [
        shard_id
        for shard_id, stats in poor
        if stats.success_rate <= max_success_rate
    ]

    return candidates[:limit]

"""
Unified Learning Orchestrator for CARTS
========================================

Main orchestrator that integrates all learning components:
- ContextualRedTeamBandit: Context-aware strategy selection
- HierarchicalLearner: Multi-level learning (strategy/payload/mutation)
- HotPayloadTracker: Heat-based payload selection
- AttackABTester: A/B testing for strategy comparison
- BackgroundLearner: Async background updates

Philosophy: "One learner to rule them all."

Author: CARTS Team
Date: 2025-12-03
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .learning_protocols import (
    LearningResult,
    HeatScore,
    ContextKey,
    ABTestResult,
    LearningLevel,
)
from .hot_payloads import (
    HotPayloadTracker,
    create_hot_payload_tracker,
)
from .contextual_bandit import (
    ContextualRedTeamBandit,
    ContextualSelectionResult,
    create_contextual_bandit,
    create_context_key,
)
from .hierarchical_learning import (
    HierarchicalLearner,
    HierarchicalSelection,
    HierarchicalUpdate,
    create_hierarchical_learner,
    create_hierarchical_update,
)
from .attack_ab_testing import (
    AttackABTester,
    ABTestAnalysis,
    create_ab_tester,
)
from .background_learner import (
    BackgroundLearner,
    BackgroundLearnerConfig,
    LearningEvent,
    create_background_learner,
)


# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UnifiedSelection:
    """
    Result of unified strategy selection.

    Combines recommendations from all learning components.

    Attributes:
        strategy: Selected attack strategy
        payload_id: Recommended payload (if available)
        mutation_type: Recommended mutation (if available)
        context: Context used for selection
        confidence: Overall confidence in selection (0.0-1.0)
        selection_method: How selection was made
        hierarchical_selection: Full hierarchical selection result
        is_hot_payload: Whether selected payload is hot
        ab_test_variant: A/B test variant if in active test
        metadata: Additional selection metadata
    """
    strategy: str
    payload_id: Optional[str] = None
    mutation_type: Optional[str] = None
    context: Optional[ContextKey] = None
    confidence: float = 0.5
    selection_method: str = "unified"
    hierarchical_selection: Optional[HierarchicalSelection] = None
    is_hot_payload: bool = False
    ab_test_variant: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedUpdate:
    """
    Result of unified learning update.

    Attributes:
        strategy: Strategy that was used
        payload_id: Payload that was used
        mutation_type: Mutation that was used
        success: Whether attack succeeded
        severity: Severity if successful
        context: Context of attack
        components_updated: Which components received update
        timestamp: When update occurred
        metadata: Additional metadata about the update
    """
    strategy: str
    payload_id: Optional[str]
    mutation_type: Optional[str]
    success: bool
    severity: float
    context: Optional[ContextKey] = None
    components_updated: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedStats:
    """
    Comprehensive statistics from all learning components.

    Attributes:
        total_selections: Total selection count
        total_updates: Total update count
        contextual_bandit_stats: Stats from contextual bandit
        hierarchical_stats: Stats from hierarchical learner
        hot_payload_stats: Stats from hot payload tracker
        ab_test_stats: Stats from A/B tester
        background_stats: Stats from background learner
        component_health: Health status of each component
    """
    total_selections: int = 0
    total_updates: int = 0
    contextual_bandit_stats: Dict[str, Any] = field(default_factory=dict)
    hierarchical_stats: Dict[str, Any] = field(default_factory=dict)
    hot_payload_stats: Dict[str, Any] = field(default_factory=dict)
    ab_test_stats: Dict[str, Any] = field(default_factory=dict)
    background_stats: Dict[str, Any] = field(default_factory=dict)
    component_health: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_selections': self.total_selections,
            'total_updates': self.total_updates,
            'contextual_bandit': self.contextual_bandit_stats,
            'hierarchical': self.hierarchical_stats,
            'hot_payloads': self.hot_payload_stats,
            'ab_tests': self.ab_test_stats,
            'background': self.background_stats,
            'component_health': self.component_health,
        }


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class UnifiedLearnerConfig:
    """
    Configuration for unified learner.

    Attributes:
        enable_contextual: Enable context-aware bandit
        enable_hierarchical: Enable multi-level learning
        enable_hot_payloads: Enable heat-based tracking
        enable_ab_testing: Enable A/B testing
        enable_background: Enable background learning
        context_prior_strength: Prior strength for contextual bandit
        hierarchical_prior_strength: Prior strength for hierarchical learner
        hot_threshold: Threshold for hot payloads (default 0.7)
        cold_threshold: Threshold for cold payloads (default 0.3)
        background_update_interval: Background update interval (seconds)
        background_decay_interval: Background decay interval (seconds)
        state_dir: Directory for state persistence
    """
    enable_contextual: bool = True
    enable_hierarchical: bool = True
    enable_hot_payloads: bool = True
    enable_ab_testing: bool = True
    enable_background: bool = True
    context_prior_strength: float = 1.0
    hierarchical_prior_strength: float = 1.0
    hot_threshold: float = 0.7
    cold_threshold: float = 0.3
    background_update_interval: float = 60.0
    background_decay_interval: float = 300.0
    state_dir: Optional[Path] = None


# =============================================================================
# Main Class
# =============================================================================

class UnifiedLearner:
    """
    Unified learning orchestrator for CARTS.

    Integrates all learning components into a cohesive system:
    - Contextual bandit for context-aware strategy selection
    - Hierarchical learner for multi-level optimization
    - Hot payload tracker for heat-based selection
    - A/B tester for strategy comparison
    - Background learner for async updates

    Example:
        learner = UnifiedLearner(
            strategies=['UNICODE_BYPASS', 'PROMPT_INJECTION', ...],
            config=UnifiedLearnerConfig(enable_background=True)
        )

        # Start background learning
        await learner.start()

        # Select strategy with full learning
        selection = learner.select(
            context=create_context_key('llm', 'jailbreak', 'advanced')
        )

        # Execute attack...

        # Update with result
        learner.update(
            strategy=selection.strategy,
            payload_id='payload_123',
            mutation_type='WORD_SUBSTITUTE',
            success=True,
            severity=0.8,
            context=selection.context
        )

        # Stop gracefully
        await learner.stop()
    """

    def __init__(
        self,
        strategies: List[str],
        config: Optional[UnifiedLearnerConfig] = None,
    ):
        """
        Initialize unified learner.

        Args:
            strategies: List of attack strategy names
            config: Learner configuration
        """
        # Validate strategies
        if not strategies:
            raise ValueError("strategies list cannot be empty")

        self.strategies = strategies
        self.config = config or UnifiedLearnerConfig()

        # Build mapping from enum values/names to allowed strategy names
        # e.g., 'unicode_bypass' -> 'UNICODE_BYPASS', 'UNICODE_BYPASS' -> 'UNICODE_BYPASS'
        self._strategy_value_to_name: Dict[str, str] = {}
        self._allowed_enum_values: Set[str] = set()
        for s in strategies:
            # Strategy name is what user provided (e.g., 'UNICODE_BYPASS')
            # Map both the name and the lowercase value to it
            self._strategy_value_to_name[s] = s  # Name -> Name
            self._strategy_value_to_name[s.lower()] = s  # value -> Name
            # Also handle underscore variations
            value_form = s.lower().replace('-', '_')
            self._strategy_value_to_name[value_form] = s
            self._allowed_enum_values.add(value_form)

        # Initialize components
        self._init_components()

        # Tracking
        self._total_selections = 0
        self._total_updates = 0
        self._running = False

    def _init_components(self) -> None:
        """Initialize all learning components."""
        # Contextual bandit
        if self.config.enable_contextual:
            self.contextual_bandit = create_contextual_bandit(
                initial_alpha=self.config.context_prior_strength,
                initial_beta=1.0,
            )
        else:
            self.contextual_bandit = None

        # Hierarchical learner
        if self.config.enable_hierarchical:
            self.hierarchical_learner = create_hierarchical_learner(
                initial_alpha=self.config.hierarchical_prior_strength,
                initial_beta=1.0,
            )
        else:
            self.hierarchical_learner = None

        # Hot payload tracker
        if self.config.enable_hot_payloads:
            self.hot_tracker = create_hot_payload_tracker(
                hot_threshold=self.config.hot_threshold,
                cold_threshold=self.config.cold_threshold,
            )
        else:
            self.hot_tracker = None

        # A/B tester
        if self.config.enable_ab_testing:
            self.ab_tester = create_ab_tester()
        else:
            self.ab_tester = None

        # Background learner
        if self.config.enable_background:
            bg_config = BackgroundLearnerConfig(
                update_interval=self.config.background_update_interval,
                decay_interval=self.config.background_decay_interval,
                state_path=self.config.state_dir,
            )

            # Collect updateable components
            updateables = []
            if self.hot_tracker:
                updateables.append(self.hot_tracker)

            self.background_learner = BackgroundLearner(
                config=bg_config,
                on_update=self._on_background_update,
                on_decay=self._on_background_decay,
                updateables=updateables,
            )
        else:
            self.background_learner = None

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start background learning."""
        if self._running:
            return

        self._running = True

        if self.background_learner:
            await self.background_learner.start()

        logger.info("Unified learner started")

    async def stop(self, timeout: float = 5.0) -> None:
        """
        Stop background learning gracefully.

        Args:
            timeout: Maximum seconds to wait for shutdown
        """
        if not self._running:
            return

        self._running = False

        if self.background_learner:
            await self.background_learner.stop(timeout=timeout)

        # Save final state
        if self.config.state_dir:
            self.save_state(self.config.state_dir)

        logger.info("Unified learner stopped")

    async def __aenter__(self) -> 'UnifiedLearner':
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select(
        self,
        context: Optional[ContextKey] = None,
        prefer_hot: bool = True,
        ab_test_name: Optional[str] = None,
    ) -> UnifiedSelection:
        """
        Select attack strategy using all learning components.

        Selection priority:
        1. If in A/B test, use test variant allocation
        2. Use contextual bandit for strategy selection
        3. Use hierarchical learner for payload/mutation
        4. Prefer hot payloads if enabled

        Args:
            context: Attack context for contextual selection
            prefer_hot: Whether to prefer hot payloads
            ab_test_name: A/B test to participate in

        Returns:
            Unified selection result
        """
        self._total_selections += 1

        # Initialize result
        result = UnifiedSelection(strategy=self.strategies[0])
        result.context = context

        # A/B test variant
        if ab_test_name and self.ab_tester:
            # Check if test exists and is active via _tests dict
            if ab_test_name in self.ab_tester._tests:
                test = self.ab_tester._tests[ab_test_name]
                from .attack_ab_testing import TestStatus
                if test.status == TestStatus.ACTIVE:
                    # Assign variant based on selection count (simple split)
                    variant = 'treatment' if self._total_selections % 2 == 0 else 'control'
                    result.ab_test_variant = variant
                    result.metadata['ab_test'] = ab_test_name

        # Contextual selection
        if self.contextual_bandit and context:
            # select() returns AttackStrategy; full details in selection_history[-1]
            selected_strategy = self.contextual_bandit.select(context)
            strategy_value = selected_strategy.value  # e.g., 'unicode_bypass'

            # Convert to user-provided format (e.g., 'UNICODE_BYPASS')
            if strategy_value in self._strategy_value_to_name:
                result.strategy = self._strategy_value_to_name[strategy_value]
                result.selection_method = "contextual"
            else:
                # Selected strategy not in allowed list, keep default
                result.metadata['filtered_strategy'] = strategy_value

            # Get detailed selection info from history
            if self.contextual_bandit.selection_history:
                ctx_result = self.contextual_bandit.selection_history[-1]
                result.confidence = ctx_result.sampled_value
                result.metadata['contextual_sample'] = ctx_result.sampled_value
                result.metadata['used_global_fallback'] = ctx_result.used_global_fallback

        # Hierarchical selection
        if self.hierarchical_learner:
            # select_full_hierarchy() performs strategy -> payload -> mutation selection
            hier_result = self.hierarchical_learner.select_full_hierarchy()
            result.hierarchical_selection = hier_result
            result.payload_id = hier_result.payload_id
            result.mutation_type = hier_result.mutation_type.value if hier_result.mutation_type else None

            # If no contextual selection, use hierarchical strategy
            if not self.contextual_bandit or result.selection_method != "contextual":
                strategy_value = hier_result.strategy.value  # e.g., 'unicode_bypass'

                # Convert to user-provided format (e.g., 'UNICODE_BYPASS')
                if strategy_value in self._strategy_value_to_name:
                    result.strategy = self._strategy_value_to_name[strategy_value]
                    # Calculate confidence from sampled values if available
                    if hier_result.sampled_values:
                        result.confidence = max(hier_result.sampled_values.values())
                    result.selection_method = "hierarchical"
                else:
                    # Strategy not in allowed list, keep default
                    result.metadata['filtered_hier_strategy'] = strategy_value

        # Hot payload preference
        if prefer_hot and self.hot_tracker and result.payload_id:
            heat = self.hot_tracker.get_heat(result.payload_id)
            result.is_hot_payload = heat > self.config.hot_threshold
            result.metadata['payload_heat'] = heat

            # Get alternative hot payload if current is cold
            if heat < self.config.cold_threshold:
                hot_payloads = self.hot_tracker.get_hot_payloads(limit=1)
                if hot_payloads:
                    result.payload_id = hot_payloads[0].payload_id
                    result.is_hot_payload = True
                    result.metadata['switched_to_hot'] = True

        return result

    def select_top_k(
        self,
        k: int,
        context: Optional[ContextKey] = None,
    ) -> List[UnifiedSelection]:
        """
        Select top-k strategies.

        Args:
            k: Number of strategies to select
            context: Attack context

        Returns:
            List of top-k selections
        """
        results = []

        if self.contextual_bandit and context:
            top_strategies = self.contextual_bandit.select_top_k(context, k)
            for strategy in top_strategies:
                selection = self.select(context=context, prefer_hot=True)
                selection.strategy = strategy
                results.append(selection)
        else:
            # Fall back to hierarchical or default
            for i, strategy in enumerate(self.strategies[:k]):
                selection = UnifiedSelection(strategy=strategy, context=context)
                results.append(selection)

        return results

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(
        self,
        strategy: str,
        payload_id: Optional[str],
        mutation_type: Optional[str],
        success: bool,
        severity: float = 0.0,
        context: Optional[ContextKey] = None,
        ab_test_name: Optional[str] = None,
        ab_variant: Optional[str] = None,
    ) -> UnifiedUpdate:
        """
        Update all learning components with attack result.

        Args:
            strategy: Attack strategy used
            payload_id: Payload used
            mutation_type: Mutation used
            success: Whether attack succeeded
            severity: Severity if successful
            context: Attack context
            ab_test_name: A/B test name if participating
            ab_variant: A/B test variant (control/treatment)

        Returns:
            Update result with components updated
        """
        self._total_updates += 1

        result = UnifiedUpdate(
            strategy=strategy,
            payload_id=payload_id,
            mutation_type=mutation_type,
            success=success,
            severity=severity,
            context=context,
        )

        # Update contextual bandit
        if self.contextual_bandit and context:
            # Convert strategy string to AttackStrategy enum
            from ..strategies import AttackStrategy as StrategyEnum
            strategy_enum = None
            try:
                strategy_enum = StrategyEnum[strategy]  # e.g., 'UNICODE_BYPASS' -> AttackStrategy.UNICODE_BYPASS
            except KeyError:
                # Try by value if name doesn't match
                try:
                    strategy_enum = StrategyEnum(strategy.lower())
                except ValueError:
                    # Unknown strategy - skip contextual update
                    result.metadata['unknown_strategy'] = strategy

            if strategy_enum is not None:
                self.contextual_bandit.update(
                    context=context,
                    strategy=strategy_enum,
                    success=success,
                    reward=severity,  # contextual_bandit uses 'reward' not 'severity'
                )
                result.components_updated.append('contextual_bandit')

        # Update hierarchical learner
        if self.hierarchical_learner:
            # Convert strategy string to AttackStrategy enum
            from ..strategies import AttackStrategy as StrategyEnum
            strategy_enum = None
            try:
                strategy_enum = StrategyEnum[strategy]
            except KeyError:
                try:
                    strategy_enum = StrategyEnum(strategy.lower())
                except ValueError:
                    # Unknown strategy - skip hierarchical update
                    result.metadata['unknown_strategy'] = strategy

            if strategy_enum is not None:
                # Convert mutation_type string to MutationType enum if provided
                mutation_enum = None
                if mutation_type:
                    from .hierarchical_learning import MutationType
                    try:
                        mutation_enum = MutationType[mutation_type]
                    except KeyError:
                        try:
                            mutation_enum = MutationType(mutation_type.lower())
                        except ValueError:
                            # Unknown mutation type - proceed without it
                            result.metadata['unknown_mutation_type'] = mutation_type

                hier_update = create_hierarchical_update(
                    strategy=strategy_enum,
                    payload_id=payload_id,
                    mutation_type=mutation_enum,
                    success=success,
                    severity=severity,
                )
                self.hierarchical_learner.update_hierarchy(hier_update)
                result.components_updated.append('hierarchical_learner')

        # Update hot payload tracker
        if self.hot_tracker and payload_id:
            self.hot_tracker.record_access(
                element_id=payload_id,
                success=success,
                severity=severity,
                strategy=strategy,
            )
            result.components_updated.append('hot_tracker')

        # Update A/B test
        if self.ab_tester and ab_test_name and ab_variant:
            self.ab_tester.record_result(
                test_name=ab_test_name,
                variant=ab_variant,
                success=success,
                severity=severity,
            )
            result.components_updated.append('ab_tester')

        # Queue for background learning
        if self.background_learner:
            # Convert ContextKey to dict for background_learner
            context_dict = None
            if context:
                context_dict = {
                    'target_type': context.target_type,
                    'vuln_class': context.vuln_class,
                    'defense_level': context.defense_level,
                }
            self.background_learner.queue_attack_result(
                strategy=strategy,
                success=success,
                severity=severity,
                payload_id=payload_id,
                context=context_dict,
            )
            result.components_updated.append('background_learner')

        return result

    # -------------------------------------------------------------------------
    # A/B Testing
    # -------------------------------------------------------------------------

    def create_ab_test(
        self,
        name: str,
        control: str,
        treatment: str,
        min_samples: int = 30,
    ) -> None:
        """
        Create a new A/B test.

        Args:
            name: Unique test name
            control: Control variant description
            treatment: Treatment variant description
            min_samples: Minimum samples per variant
        """
        if self.ab_tester:
            self.ab_tester.create_test(
                name=name,
                control=control,
                treatment=treatment,
                min_samples=min_samples,
            )

    def analyze_ab_test(self, name: str) -> Optional[ABTestAnalysis]:
        """
        Analyze A/B test results.

        Args:
            name: Test name

        Returns:
            Analysis result or None if insufficient data
        """
        if self.ab_tester:
            return self.ab_tester.analyze(name)
        return None

    def conclude_ab_test(self, name: str) -> Optional[ABTestAnalysis]:
        """
        Conclude A/B test and return final results.

        Args:
            name: Test name

        Returns:
            Final analysis result
        """
        if self.ab_tester:
            return self.ab_tester.conclude_test(name)
        return None

    # -------------------------------------------------------------------------
    # Hot Payloads
    # -------------------------------------------------------------------------

    def get_hot_payloads(self, limit: int = 10) -> List[HeatScore]:
        """Get hottest payloads."""
        if self.hot_tracker:
            return self.hot_tracker.get_hot_payloads(limit=limit)
        return []

    def get_cold_payloads(self, limit: int = 10) -> List[HeatScore]:
        """Get coldest payloads."""
        if self.hot_tracker:
            return self.hot_tracker.get_cold_payloads(limit=limit)
        return []

    def prune_cold_payloads(self) -> int:
        """Prune cold payloads. Returns count pruned."""
        if self.hot_tracker:
            return self.hot_tracker.prune_cold()
        return 0

    # -------------------------------------------------------------------------
    # Background Learning Callbacks
    # -------------------------------------------------------------------------

    def _on_background_update(self, events: List[LearningEvent]) -> None:
        """Handle background update with buffered events."""
        logger.debug(f"Background update with {len(events)} events")

        # Process events for pattern mining, etc.
        for event in events:
            if event.event_type == 'attack_result':
                # Could do additional learning here
                pass

    def _on_background_decay(self) -> None:
        """Handle background decay tick."""
        logger.debug("Background decay applied")

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_stats(self) -> UnifiedStats:
        """Get comprehensive statistics from all components."""
        stats = UnifiedStats(
            total_selections=self._total_selections,
            total_updates=self._total_updates,
        )

        # Contextual bandit stats
        if self.contextual_bandit:
            # Build stats from available methods
            all_contexts = self.contextual_bandit.get_all_contexts()
            expected_rewards = self.contextual_bandit.get_expected_rewards()
            stats.contextual_bandit_stats = {
                'num_contexts': len(all_contexts),
                'selection_count': len(self.contextual_bandit.selection_history),
                'expected_rewards': {s.value: r for s, r in expected_rewards.items()},
            }
            stats.component_health['contextual_bandit'] = True
        else:
            stats.component_health['contextual_bandit'] = False

        # Hierarchical stats
        if self.hierarchical_learner:
            # Build stats from available data
            from ..strategies import AttackStrategy
            stats.hierarchical_stats = {
                'strategy_count': len(AttackStrategy),
                'payload_count': len(self.hierarchical_learner.payload_arms),
                'mutation_count': len(self.hierarchical_learner.mutation_arms),
            }
            stats.component_health['hierarchical_learner'] = True
        else:
            stats.component_health['hierarchical_learner'] = False

        # Hot payload stats
        if self.hot_tracker:
            stats.hot_payload_stats = self.hot_tracker.get_stats()
            stats.component_health['hot_tracker'] = True
        else:
            stats.component_health['hot_tracker'] = False

        # A/B test stats
        if self.ab_tester:
            stats.ab_test_stats = self.ab_tester.get_stats()
            stats.component_health['ab_tester'] = True
        else:
            stats.component_health['ab_tester'] = False

        # Background stats
        if self.background_learner:
            bg_stats = self.background_learner.get_stats()
            stats.background_stats = bg_stats.to_dict()
            stats.component_health['background_learner'] = self.background_learner.is_running()
        else:
            stats.component_health['background_learner'] = False

        return stats

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save_state(self, directory: Path) -> None:
        """
        Save state of all components.

        Args:
            directory: Directory to save state files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if self.contextual_bandit:
            self.contextual_bandit.save(directory / 'contextual_bandit.json')

        if self.hierarchical_learner:
            self.hierarchical_learner.save(directory / 'hierarchical_learner.json')

        if self.hot_tracker:
            self.hot_tracker.save(directory / 'hot_payloads.json')

        if self.ab_tester:
            self.ab_tester.save(directory / 'ab_tests.json')

        logger.info(f"State saved to {directory}")

    def load_state(self, directory: Path) -> None:
        """
        Load state of all components.

        Args:
            directory: Directory containing state files
        """
        directory = Path(directory)

        if not directory.exists():
            logger.warning(f"State directory not found: {directory}")
            return

        if self.contextual_bandit:
            path = directory / 'contextual_bandit.json'
            if path.exists():
                self.contextual_bandit.load(path)

        if self.hierarchical_learner:
            path = directory / 'hierarchical_learner.json'
            if path.exists():
                self.hierarchical_learner.load(path)

        if self.hot_tracker:
            path = directory / 'hot_payloads.json'
            if path.exists():
                self.hot_tracker.load(path)

        if self.ab_tester:
            path = directory / 'ab_tests.json'
            if path.exists():
                self.ab_tester.load(path)

        logger.info(f"State loaded from {directory}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_unified_learner(
    strategies: Optional[List[str]] = None,
    enable_all: bool = True,
    state_dir: Optional[str] = None,
) -> UnifiedLearner:
    """
    Create unified learner with default configuration.

    Args:
        strategies: Attack strategy names (defaults to 12 CARTS strategies)
        enable_all: Enable all learning components
        state_dir: Directory for state persistence

    Returns:
        Configured UnifiedLearner
    """
    # Default CARTS strategies
    if strategies is None:
        strategies = [
            'UNICODE_BYPASS',
            'PROMPT_INJECTION',
            'CONTEXT_OVERFLOW',
            'FORMAT_CONFUSION',
            'ENCODING_ATTACK',
            'DELIMITER_INJECTION',
            'INSTRUCTION_OVERRIDE',
            'SEMANTIC_MANIPULATION',
            'ROLE_PLAYING',
            'MULTI_TURN_ESCALATION',
            'PAYLOAD_FRAGMENTATION',
            'OBFUSCATION',
        ]

    config = UnifiedLearnerConfig(
        enable_contextual=enable_all,
        enable_hierarchical=enable_all,
        enable_hot_payloads=enable_all,
        enable_ab_testing=enable_all,
        enable_background=enable_all,
        state_dir=Path(state_dir) if state_dir else None,
    )

    return UnifiedLearner(strategies=strategies, config=config)


async def run_learning_demo(
    num_iterations: int = 100,
    verbose: bool = True,
) -> UnifiedStats:
    """
    Run a demo of the unified learning system.

    Args:
        num_iterations: Number of attack iterations
        verbose: Print progress

    Returns:
        Final statistics
    """
    import random

    # Create learner
    learner = create_unified_learner(enable_all=True)

    async with learner:
        # Create A/B test
        learner.create_ab_test(
            name="unicode_vs_injection",
            control="UNICODE_BYPASS",
            treatment="PROMPT_INJECTION",
        )

        # Run iterations
        for i in range(num_iterations):
            # Create context
            target_types = ['llm', 'api', 'web_app']
            vuln_classes = ['prompt_injection', 'jailbreak', 'unicode_bypass']
            defense_levels = ['none', 'basic', 'advanced']

            context = create_context_key(
                target_type=random.choice(target_types),
                vuln_class=random.choice(vuln_classes),
                defense_level=random.choice(defense_levels),
            )

            # Select strategy
            selection = learner.select(
                context=context,
                prefer_hot=True,
                ab_test_name="unicode_vs_injection",
            )

            # Simulate attack result
            success = random.random() < 0.3  # 30% success rate
            severity = random.random() if success else 0.0

            # Update learner
            learner.update(
                strategy=selection.strategy,
                payload_id=f"payload_{i % 20}",  # Reuse payloads
                mutation_type="WORD_SUBSTITUTE" if i % 2 == 0 else "CASE_VARIATION",
                success=success,
                severity=severity,
                context=context,
                ab_test_name="unicode_vs_injection",
                ab_variant=selection.ab_test_variant,
            )

            if verbose and (i + 1) % 20 == 0:
                print(f"Iteration {i + 1}/{num_iterations}")

        # Analyze A/B test
        ab_result = learner.analyze_ab_test("unicode_vs_injection")
        if ab_result and verbose:
            print(f"\nA/B Test Result: {ab_result.winner}")
            print(f"  Control mean: {ab_result.control_mean:.3f}")
            print(f"  Treatment mean: {ab_result.treatment_mean:.3f}")
            print(f"  P-value: {ab_result.p_value:.4f}")

        # Get final stats
        stats = learner.get_stats()

        if verbose:
            print(f"\nFinal Statistics:")
            print(f"  Total selections: {stats.total_selections}")
            print(f"  Total updates: {stats.total_updates}")
            print(f"  Hot payloads: {len(learner.get_hot_payloads())}")
            print(f"  Cold payloads: {len(learner.get_cold_payloads())}")

        return stats


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Data Classes
    'UnifiedSelection',
    'UnifiedUpdate',
    'UnifiedStats',
    'UnifiedLearnerConfig',

    # Main Class
    'UnifiedLearner',

    # Convenience Functions
    'create_unified_learner',
    'run_learning_demo',
]

"""
AR Quality Refiner
==================
Quality refinement for AR responses with AR-specific strategies.

Refinement Strategies:
- VERIFY: Verify visual accuracy (AR overlays match real objects)
- ELEGANCE: Simplify AR overlays (reduce visual clutter)
- SPATIAL: Improve spatial positioning (better placement in 3D space)
- MULTIMODAL: Balance gesture + voice + vision fusion

Triggers refinement when confidence < 0.75 and improves quality through
multiple iterations.

Author: HoloLoom Team
Date: November 17, 2025
Version: 1.0.0
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# AR Refinement Strategies
# ============================================================================

class ARRefinementStrategy(Enum):
    """AR-specific refinement strategies"""
    VERIFY = "verify"  # Verify visual accuracy
    ELEGANCE = "elegance"  # Simplify AR overlays
    SPATIAL = "spatial"  # Improve spatial positioning
    MULTIMODAL = "multimodal"  # Balance modality fusion
    AUTO = "auto"  # Auto-select best strategy


# ============================================================================
# AR Quality Metrics
# ============================================================================

@dataclass
class ARQualityMetrics:
    """
    Quality metrics for AR responses.

    Extends base QualityMetrics with AR-specific dimensions.
    """
    # Base metrics
    confidence: float
    threads_activated: int
    motifs_detected: int
    context_size: int
    response_length: int

    # AR-specific metrics
    visual_accuracy: float = 0.0  # How well AR overlays match reality
    spatial_coherence: float = 0.0  # How well positioned in 3D space
    visual_clutter: float = 0.0  # Amount of visual noise (lower = better)
    modality_balance: float = 0.0  # Balance between gesture/voice/vision

    timestamp: float = field(default_factory=time.time)

    def score(self) -> float:
        """
        Composite quality score for AR.

        Returns:
            Quality score [0.0, 1.0]
        """
        # Base quality (40% weight)
        base_quality = (
            0.7 * self.confidence +
            0.2 * min(1.0, (self.threads_activated + self.motifs_detected) / 10.0) +
            0.1 * min(1.0, self.response_length / 500.0)
        )

        # AR quality (60% weight)
        ar_quality = (
            0.3 * self.visual_accuracy +
            0.3 * self.spatial_coherence +
            0.2 * (1.0 - self.visual_clutter) +  # Invert clutter (less = better)
            0.2 * self.modality_balance
        )

        return 0.4 * base_quality + 0.6 * ar_quality

    @classmethod
    def from_spacetime(cls, spacetime: 'Spacetime', ar_query: 'ARQuery') -> 'ARQualityMetrics':
        """Create metrics from spacetime and AR query"""
        # Base metrics
        confidence = spacetime.confidence
        threads_activated = len(spacetime.trace.threads_activated)
        motifs_detected = len(spacetime.trace.motifs_detected) if hasattr(spacetime.trace, 'motifs_detected') else 0
        context_size = len(spacetime.trace.context_used) if hasattr(spacetime.trace, 'context_used') else 0
        response_length = len(str(spacetime.response)) if spacetime.response else 0

        # AR-specific metrics (estimates)
        visual_accuracy = confidence * 0.9  # Rough estimate
        spatial_coherence = confidence * 0.85
        visual_clutter = 0.2  # Default low clutter
        modality_balance = cls._calculate_modality_balance(ar_query)

        return cls(
            confidence=confidence,
            threads_activated=threads_activated,
            motifs_detected=motifs_detected,
            context_size=context_size,
            response_length=response_length,
            visual_accuracy=visual_accuracy,
            spatial_coherence=spatial_coherence,
            visual_clutter=visual_clutter,
            modality_balance=modality_balance,
        )

    @staticmethod
    def _calculate_modality_balance(ar_query: 'ARQuery') -> float:
        """
        Calculate modality balance score.

        Perfect balance: all 3 modalities active (gesture + voice + vision)
        Imbalanced: only 1 modality active
        """
        active_modalities = 0
        if ar_query.text or ar_query.voice_intent:
            active_modalities += 1
        if ar_query.gesture_detected:
            active_modalities += 1
        if ar_query.vision_objects:
            active_modalities += 1

        # Balance score: 0.33 (one), 0.67 (two), 1.0 (three)
        return active_modalities / 3.0


# ============================================================================
# AR Refinement Result
# ============================================================================

@dataclass
class ARRefinementResult:
    """Result of AR refinement process"""
    final_spacetime: 'Spacetime'
    trajectory: List[ARQualityMetrics]
    strategy_used: ARRefinementStrategy
    iterations: int
    improved: bool
    improvement_rate: float  # Quality improvement per iteration

    @property
    def initial_quality(self) -> float:
        """Initial quality score"""
        return self.trajectory[0].score() if self.trajectory else 0.0

    @property
    def final_quality(self) -> float:
        """Final quality score"""
        return self.trajectory[-1].score() if self.trajectory else 0.0

    def summary(self) -> str:
        """Generate human-readable summary"""
        return (
            f"AR Refinement Summary:\n"
            f"  Strategy: {self.strategy_used.value}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Quality: {self.initial_quality:.2f} → {self.final_quality:.2f}\n"
            f"  Improvement: {'+' if self.improved else ''}{self.final_quality - self.initial_quality:.3f}\n"
            f"  Rate: {self.improvement_rate:.3f}/iteration"
        )


# ============================================================================
# AR Refiner
# ============================================================================

class ARRefiner:
    """
    AR-specific quality refiner.

    Refines AR responses using AR-specific strategies:
    - VERIFY: Check visual accuracy
    - ELEGANCE: Simplify AR overlays
    - SPATIAL: Improve positioning
    - MULTIMODAL: Balance modalities
    """

    def __init__(
        self,
        orchestrator: 'WeavingOrchestrator',
        threshold: float = 0.75,
        max_iterations: int = 3,
    ):
        """
        Initialize AR refiner.

        Args:
            orchestrator: HoloLoom orchestrator
            threshold: Refine if confidence < this
            max_iterations: Maximum refinement iterations
        """
        self.orchestrator = orchestrator
        self.threshold = threshold
        self.max_iterations = max_iterations

        # Statistics
        self.refinements_performed = 0
        self.total_improvement = 0.0
        self.avg_iterations = 0.0

    async def refine(
        self,
        ar_query: 'ARQuery',
        initial_spacetime: 'Spacetime',
        strategy: Optional[ARRefinementStrategy] = None,
    ) -> ARRefinementResult:
        """
        Refine AR query result.

        Args:
            ar_query: Original AR query
            initial_spacetime: Initial result
            strategy: Refinement strategy (None = auto-select)

        Returns:
            ARRefinementResult with trajectory
        """
        # Auto-select strategy if not provided
        if strategy is None or strategy == ARRefinementStrategy.AUTO:
            strategy = self._select_strategy(ar_query, initial_spacetime)

        # Initialize trajectory
        trajectory: List[ARQualityMetrics] = []
        current_spacetime = initial_spacetime

        # Track initial metrics
        initial_metrics = ARQualityMetrics.from_spacetime(initial_spacetime, ar_query)
        trajectory.append(initial_metrics)

        # Refine iteratively
        for iteration in range(self.max_iterations):
            # Apply refinement strategy
            refined_spacetime = await self._apply_strategy(
                ar_query, current_spacetime, strategy, iteration
            )

            # Track metrics
            refined_metrics = ARQualityMetrics.from_spacetime(refined_spacetime, ar_query)
            trajectory.append(refined_metrics)

            # Check improvement
            if refined_metrics.score() <= trajectory[-2].score():
                # No improvement - stop
                logger.info(f"AR refinement stopped at iteration {iteration + 1} (no improvement)")
                break

            current_spacetime = refined_spacetime

            # Check if we've reached good quality
            if refined_metrics.score() >= 0.9:
                logger.info(f"AR refinement stopped at iteration {iteration + 1} (good quality)")
                break

        # Calculate results
        improved = trajectory[-1].score() > trajectory[0].score()
        improvement_rate = (
            (trajectory[-1].score() - trajectory[0].score()) / len(trajectory)
            if len(trajectory) > 1 else 0.0
        )

        # Update statistics
        self.refinements_performed += 1
        self.total_improvement += (trajectory[-1].score() - trajectory[0].score())
        self.avg_iterations = (
            (self.avg_iterations * (self.refinements_performed - 1) + len(trajectory) - 1) /
            self.refinements_performed
        )

        return ARRefinementResult(
            final_spacetime=current_spacetime,
            trajectory=trajectory,
            strategy_used=strategy,
            iterations=len(trajectory) - 1,
            improved=improved,
            improvement_rate=improvement_rate,
        )

    def _select_strategy(
        self,
        ar_query: 'ARQuery',
        spacetime: 'Spacetime',
    ) -> ARRefinementStrategy:
        """
        Auto-select best refinement strategy.

        Args:
            ar_query: AR query
            spacetime: Current result

        Returns:
            Best strategy for this query
        """
        metrics = ARQualityMetrics.from_spacetime(spacetime, ar_query)

        # Check which dimension needs most improvement
        if metrics.visual_accuracy < 0.7:
            return ARRefinementStrategy.VERIFY
        elif metrics.spatial_coherence < 0.7:
            return ARRefinementStrategy.SPATIAL
        elif metrics.visual_clutter > 0.5:
            return ARRefinementStrategy.ELEGANCE
        elif metrics.modality_balance < 0.5:
            return ARRefinementStrategy.MULTIMODAL
        else:
            # Default to VERIFY
            return ARRefinementStrategy.VERIFY

    async def _apply_strategy(
        self,
        ar_query: 'ARQuery',
        spacetime: 'Spacetime',
        strategy: ARRefinementStrategy,
        iteration: int,
    ) -> 'Spacetime':
        """
        Apply refinement strategy.

        Args:
            ar_query: AR query
            spacetime: Current result
            strategy: Refinement strategy
            iteration: Current iteration

        Returns:
            Refined spacetime
        """
        # Create refinement query based on strategy
        if strategy == ARRefinementStrategy.VERIFY:
            refinement_text = (
                f"Verify the accuracy of this AR response: {ar_query.text}\n"
                f"Previous answer: {spacetime.response}\n"
                f"Check if AR overlays match real objects in vision context."
            )
        elif strategy == ARRefinementStrategy.ELEGANCE:
            refinement_text = (
                f"Simplify this AR response to reduce visual clutter: {ar_query.text}\n"
                f"Previous answer: {spacetime.response}\n"
                f"Remove unnecessary AR elements while keeping essential information."
            )
        elif strategy == ARRefinementStrategy.SPATIAL:
            refinement_text = (
                f"Improve spatial positioning of AR elements: {ar_query.text}\n"
                f"Previous answer: {spacetime.response}\n"
                f"Ensure AR overlays are well-positioned in 3D space relative to real objects."
            )
        elif strategy == ARRefinementStrategy.MULTIMODAL:
            refinement_text = (
                f"Balance gesture, voice, and vision inputs: {ar_query.text}\n"
                f"Previous answer: {spacetime.response}\n"
                f"Ensure all modalities are properly integrated."
            )
        else:
            # Fallback
            refinement_text = f"Refine this AR response: {ar_query.text}"

        # Create refinement query
        from HoloLoom.documentation.types import Query
        refinement_query = Query(
            text=refinement_text,
            metadata={
                'refinement_strategy': strategy.value,
                'iteration': iteration,
                'original_query': ar_query.text,
            }
        )

        # Weave refinement
        refined_spacetime = await self.orchestrator.weave(refinement_query)

        return refined_spacetime

    def get_statistics(self) -> Dict[str, Any]:
        """Get refinement statistics"""
        return {
            'refinements_performed': self.refinements_performed,
            'avg_improvement': (
                self.total_improvement / self.refinements_performed
                if self.refinements_performed > 0 else 0.0
            ),
            'avg_iterations': self.avg_iterations,
        }


# ============================================================================
# Helper Functions
# ============================================================================

async def refine_ar_response(
    ar_query: 'ARQuery',
    spacetime: 'Spacetime',
    orchestrator: 'WeavingOrchestrator',
    strategy: Optional[ARRefinementStrategy] = None,
) -> ARRefinementResult:
    """
    Convenience function to refine AR response.

    Args:
        ar_query: AR query
        spacetime: Initial result
        orchestrator: HoloLoom orchestrator
        strategy: Refinement strategy (None = auto)

    Returns:
        ARRefinementResult
    """
    refiner = ARRefiner(orchestrator=orchestrator)
    return await refiner.refine(ar_query, spacetime, strategy)


# Export public API
__all__ = [
    'ARRefiner',
    'ARRefinementStrategy',
    'ARRefinementResult',
    'ARQualityMetrics',
    'refine_ar_response',
]

"""
AR Background Learner
=====================
Background learning for AR interactions with Thompson Sampling updates.

Runs in background every 60 seconds to:
- Update Thompson Sampling priors for gesture → tool mapping
- Adjust policy weights based on AR interaction outcomes
- Learn which modality combinations work best
- Persist learning state across sessions

Author: HoloLoom Team
Date: November 17, 2025
Version: 1.0.0
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# AR Thompson Priors
# ============================================================================

@dataclass
class ARThompsonPriors:
    """
    Thompson Sampling priors for AR gestures.

    Each gesture has Beta distribution parameters for each tool:
    - alpha: Number of successes + 1
    - beta: Number of failures + 1
    """
    # gesture → tool → {alpha, beta}
    gesture_tool_priors: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        )
    )

    # voice_intent → tool → {alpha, beta}
    intent_tool_priors: Dict[str, Dict[str, Dict[str, float]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: {"alpha": 1.0, "beta": 1.0})
        )
    )

    def update_gesture_success(self, gesture: str, tool: str, confidence: float):
        """Update priors after successful gesture → tool mapping"""
        self.gesture_tool_priors[gesture][tool]["alpha"] += confidence

    def update_gesture_failure(self, gesture: str, tool: str, confidence: float):
        """Update priors after unsuccessful gesture → tool mapping"""
        self.gesture_tool_priors[gesture][tool]["beta"] += (1.0 - confidence)

    def update_intent_success(self, intent: str, tool: str, confidence: float):
        """Update priors after successful intent → tool mapping"""
        self.intent_tool_priors[intent][tool]["alpha"] += confidence

    def update_intent_failure(self, intent: str, tool: str, confidence: float):
        """Update priors after unsuccessful intent → tool mapping"""
        self.intent_tool_priors[intent][tool]["beta"] += (1.0 - confidence)

    def get_expected_reward(self, gesture: str, tool: str) -> float:
        """Get expected reward for gesture → tool mapping"""
        priors = self.gesture_tool_priors[gesture][tool]
        alpha = priors["alpha"]
        beta = priors["beta"]
        return alpha / (alpha + beta)

    def get_intent_expected_reward(self, intent: str, tool: str) -> float:
        """Get expected reward for intent → tool mapping"""
        priors = self.intent_tool_priors[intent][tool]
        alpha = priors["alpha"]
        beta = priors["beta"]
        return alpha / (alpha + beta)

    def get_best_tool_for_gesture(self, gesture: str) -> Optional[str]:
        """Get best tool for gesture based on expected rewards"""
        if gesture not in self.gesture_tool_priors:
            return None

        tools = self.gesture_tool_priors[gesture]
        if not tools:
            return None

        # Find tool with highest expected reward
        best_tool = max(
            tools.keys(),
            key=lambda t: self.get_expected_reward(gesture, t)
        )
        return best_tool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'gesture_tool_priors': dict(self.gesture_tool_priors),
            'intent_tool_priors': dict(self.intent_tool_priors),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARThompsonPriors':
        """Create from dictionary"""
        priors = cls()
        priors.gesture_tool_priors = defaultdict(
            lambda: defaultdict(lambda: {"alpha": 1.0, "beta": 1.0}),
            data.get('gesture_tool_priors', {})
        )
        priors.intent_tool_priors = defaultdict(
            lambda: defaultdict(lambda: {"alpha": 1.0, "beta": 1.0}),
            data.get('intent_tool_priors', {})
        )
        return priors


# ============================================================================
# AR Modality Weights
# ============================================================================

@dataclass
class ARModalityWeights:
    """
    Learned weights for modality combinations.

    Tracks which modality combinations work best.
    """
    # modality_combo → {successes, total}
    modality_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"successes": 0, "total": 0})
    )

    def update(self, modality_combo: str, success: bool):
        """
        Update modality weights.

        Args:
            modality_combo: e.g., "voice+gesture", "voice+vision", "multimodal"
            success: Whether interaction was successful
        """
        self.modality_stats[modality_combo]["total"] += 1
        if success:
            self.modality_stats[modality_combo]["successes"] += 1

    def get_weight(self, modality_combo: str) -> float:
        """
        Get weight for modality combination.

        Returns:
            Success rate [0, 1]
        """
        stats = self.modality_stats[modality_combo]
        total = stats["total"]
        if total == 0:
            return 0.5  # Unknown
        return stats["successes"] / total

    def get_best_modality(self) -> Optional[str]:
        """Get best performing modality combination"""
        if not self.modality_stats:
            return None

        return max(
            self.modality_stats.keys(),
            key=lambda m: self.get_weight(m)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {'modality_stats': dict(self.modality_stats)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARModalityWeights':
        """Create from dictionary"""
        weights = cls()
        weights.modality_stats = defaultdict(
            lambda: {"successes": 0, "total": 0},
            data.get('modality_stats', {})
        )
        return weights


# ============================================================================
# AR Learning Metrics
# ============================================================================

@dataclass
class ARLearningMetrics:
    """Metrics for AR learning progress"""
    queries_processed: int = 0
    gesture_mappings_learned: int = 0
    intent_mappings_learned: int = 0
    modality_combos_tested: int = 0
    avg_confidence: float = 0.0
    avg_visual_accuracy: float = 0.0
    avg_spatial_coherence: float = 0.0

    def update(
        self,
        confidence: float,
        visual_accuracy: float = 0.0,
        spatial_coherence: float = 0.0,
    ):
        """Update metrics with new query result"""
        self.queries_processed += 1

        # Update running averages
        self.avg_confidence = (
            (self.avg_confidence * (self.queries_processed - 1) + confidence) /
            self.queries_processed
        )

        if visual_accuracy > 0:
            self.avg_visual_accuracy = (
                (self.avg_visual_accuracy * (self.queries_processed - 1) + visual_accuracy) /
                self.queries_processed
            )

        if spatial_coherence > 0:
            self.avg_spatial_coherence = (
                (self.avg_spatial_coherence * (self.queries_processed - 1) + spatial_coherence) /
                self.queries_processed
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'queries_processed': self.queries_processed,
            'gesture_mappings_learned': self.gesture_mappings_learned,
            'intent_mappings_learned': self.intent_mappings_learned,
            'modality_combos_tested': self.modality_combos_tested,
            'avg_confidence': self.avg_confidence,
            'avg_visual_accuracy': self.avg_visual_accuracy,
            'avg_spatial_coherence': self.avg_spatial_coherence,
        }


# ============================================================================
# AR Background Learner
# ============================================================================

class ARBackgroundLearner:
    """
    Background learner for AR interactions.

    Runs in background every 60 seconds to update:
    - Thompson Sampling priors for gesture/intent → tool mapping
    - Modality combination weights
    - Learning statistics
    """

    def __init__(
        self,
        update_interval: float = 60.0,
        persist_path: Optional[str] = None,
    ):
        """
        Initialize AR background learner.

        Args:
            update_interval: Update interval in seconds
            persist_path: Path to persist learning state
        """
        self.update_interval = update_interval
        self.persist_path = persist_path

        # Learning components
        self.thompson_priors = ARThompsonPriors()
        self.modality_weights = ARModalityWeights()
        self.metrics = ARLearningMetrics()

        # Background task
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

        # Pending updates queue
        self._pending_updates: List[Dict[str, Any]] = []

    async def start(self):
        """Start background learning loop"""
        # Load persisted state if available
        if self.persist_path and Path(self.persist_path).exists():
            self.load_state(self.persist_path)

        # Start background task
        self._task = asyncio.create_task(self._learning_loop())
        logger.info("AR background learner started")

    async def stop(self):
        """Stop background learning loop"""
        # Signal shutdown
        self._shutdown_event.set()

        # Wait for task
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Background learner timeout")
                self._task.cancel()

        # Persist state
        if self.persist_path:
            self.save_state(self.persist_path)

        logger.info("AR background learner stopped")

    def record_interaction(
        self,
        gesture: Optional[str],
        voice_intent: Optional[str],
        modality_combo: str,
        tool_used: str,
        confidence: float,
        visual_accuracy: float = 0.0,
        spatial_coherence: float = 0.0,
    ):
        """
        Record AR interaction for learning.

        Args:
            gesture: Gesture detected
            voice_intent: Voice intent
            modality_combo: Modality combination used
            tool_used: Tool that was used
            confidence: Result confidence
            visual_accuracy: Visual accuracy score
            spatial_coherence: Spatial coherence score
        """
        # Queue update
        self._pending_updates.append({
            'gesture': gesture,
            'voice_intent': voice_intent,
            'modality_combo': modality_combo,
            'tool_used': tool_used,
            'confidence': confidence,
            'visual_accuracy': visual_accuracy,
            'spatial_coherence': spatial_coherence,
        })

    async def _learning_loop(self):
        """Background learning loop"""
        while not self._shutdown_event.is_set():
            try:
                # Wait for interval or shutdown
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.update_interval
                )
                # Shutdown signaled
                break
            except asyncio.TimeoutError:
                # Time to learn
                await self._perform_learning_update()

        logger.info("AR learning loop stopped")

    async def _perform_learning_update(self):
        """Perform learning update from pending interactions"""
        if not self._pending_updates:
            return

        # Process pending updates
        updates_processed = 0

        for update in self._pending_updates:
            gesture = update['gesture']
            voice_intent = update['voice_intent']
            modality_combo = update['modality_combo']
            tool_used = update['tool_used']
            confidence = update['confidence']
            visual_accuracy = update['visual_accuracy']
            spatial_coherence = update['spatial_coherence']

            # Update Thompson priors
            success_threshold = 0.75

            if gesture:
                if confidence >= success_threshold:
                    self.thompson_priors.update_gesture_success(
                        gesture, tool_used, confidence
                    )
                    self.metrics.gesture_mappings_learned += 1
                else:
                    self.thompson_priors.update_gesture_failure(
                        gesture, tool_used, confidence
                    )

            if voice_intent:
                if confidence >= success_threshold:
                    self.thompson_priors.update_intent_success(
                        voice_intent, tool_used, confidence
                    )
                    self.metrics.intent_mappings_learned += 1
                else:
                    self.thompson_priors.update_intent_failure(
                        voice_intent, tool_used, confidence
                    )

            # Update modality weights
            success = confidence >= success_threshold
            self.modality_weights.update(modality_combo, success)
            self.metrics.modality_combos_tested += 1

            # Update metrics
            self.metrics.update(confidence, visual_accuracy, spatial_coherence)

            updates_processed += 1

        # Clear pending updates
        self._pending_updates = []

        logger.info(f"AR learning update: processed {updates_processed} interactions")

    def get_best_tool_for_gesture(self, gesture: str) -> Optional[str]:
        """Get best tool for gesture based on learned priors"""
        return self.thompson_priors.get_best_tool_for_gesture(gesture)

    def get_best_modality(self) -> Optional[str]:
        """Get best performing modality combination"""
        return self.modality_weights.get_best_modality()

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return self.metrics.to_dict()

    def save_state(self, path: str):
        """Save learning state to disk"""
        state = {
            'thompson_priors': self.thompson_priors.to_dict(),
            'modality_weights': self.modality_weights.to_dict(),
            'metrics': self.metrics.to_dict(),
            'timestamp': time.time(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"AR learning state saved to {path}")

    def load_state(self, path: str):
        """Load learning state from disk"""
        with open(path, 'r') as f:
            state = json.load(f)

        self.thompson_priors = ARThompsonPriors.from_dict(
            state.get('thompson_priors', {})
        )
        self.modality_weights = ARModalityWeights.from_dict(
            state.get('modality_weights', {})
        )

        # Load metrics
        metrics_data = state.get('metrics', {})
        self.metrics = ARLearningMetrics(
            queries_processed=metrics_data.get('queries_processed', 0),
            gesture_mappings_learned=metrics_data.get('gesture_mappings_learned', 0),
            intent_mappings_learned=metrics_data.get('intent_mappings_learned', 0),
            modality_combos_tested=metrics_data.get('modality_combos_tested', 0),
            avg_confidence=metrics_data.get('avg_confidence', 0.0),
            avg_visual_accuracy=metrics_data.get('avg_visual_accuracy', 0.0),
            avg_spatial_coherence=metrics_data.get('avg_spatial_coherence', 0.0),
        )

        logger.info(f"AR learning state loaded from {path}")


# Export public API
__all__ = [
    'ARBackgroundLearner',
    'ARThompsonPriors',
    'ARModalityWeights',
    'ARLearningMetrics',
]

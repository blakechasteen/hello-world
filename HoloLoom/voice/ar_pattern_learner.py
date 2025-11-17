"""
AR Pattern Learner
==================
Learns patterns from AR interactions: (gesture, voice_intent, context) → tool_used

Extracts patterns like:
- "pinch + select + beehive_visible → inspect_beehive"
- "swipe_left + delete + frame_visible → remove_frame"
- "point + info + marker_visible → show_info_panel"

Tracks which gestures work best in which contexts and learns from successful
interactions (high confidence).

Author: HoloLoom Team
Date: November 17, 2025
Version: 1.0.0
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Tuple
from collections import defaultdict, Counter
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# AR Pattern Types
# ============================================================================

class GestureType(Enum):
    """Standard AR gesture types"""
    PINCH = "pinch"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    POINT = "point"
    GRAB = "grab"
    RELEASE = "release"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    ROTATE_CW = "rotate_cw"
    ROTATE_CCW = "rotate_ccw"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> 'GestureType':
        """Convert string to GestureType"""
        try:
            return cls(s.lower())
        except ValueError:
            return cls.UNKNOWN


class VoiceIntent(Enum):
    """Standard voice intents for AR"""
    SELECT = "select"
    DELETE = "delete"
    INFO = "info"
    CREATE = "create"
    MOVE = "move"
    RESIZE = "resize"
    ROTATE = "rotate"
    ANNOTATE = "annotate"
    NAVIGATE = "navigate"
    INSPECT = "inspect"
    MEASURE = "measure"
    COMPARE = "compare"
    FILTER = "filter"
    HIGHLIGHT = "highlight"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, s: str) -> 'VoiceIntent':
        """Convert string to VoiceIntent"""
        if not s:
            return cls.UNKNOWN
        try:
            return cls(s.lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class ARPattern:
    """
    Learned AR pattern.

    Pattern: (gesture, voice_intent, vision_context) → tool_used
    """
    # Input features
    gesture: Optional[str] = None
    voice_intent: Optional[str] = None
    vision_context: Set[str] = field(default_factory=set)  # Object types visible

    # Output
    tool_used: str = "unknown"

    # Quality metrics
    confidence: float = 0.0
    support: int = 0  # Number of times observed
    success_rate: float = 0.0  # Fraction of high-confidence results

    # Usage tracking
    total_uses: int = 0
    successful_uses: int = 0  # confidence >= 0.75
    last_used: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)

    def matches(
        self,
        gesture: Optional[str],
        voice_intent: Optional[str],
        vision_context: Set[str],
    ) -> float:
        """
        Calculate match score [0, 1] for this pattern.

        Returns:
            Match score (1.0 = perfect match, 0.0 = no match)
        """
        score = 0.0
        total_weight = 0.0

        # Gesture match (40% weight)
        if self.gesture is not None:
            total_weight += 0.4
            if gesture == self.gesture:
                score += 0.4

        # Voice intent match (40% weight)
        if self.voice_intent is not None:
            total_weight += 0.4
            if voice_intent == self.voice_intent:
                score += 0.4

        # Vision context match (20% weight) - Jaccard similarity
        if self.vision_context:
            total_weight += 0.2
            if vision_context:
                intersection = len(self.vision_context & vision_context)
                union = len(self.vision_context | vision_context)
                jaccard = intersection / union if union > 0 else 0.0
                score += 0.2 * jaccard

        # Normalize by total weight
        if total_weight > 0:
            return score / total_weight
        return 0.0

    def update_success(self, confidence: float):
        """Update pattern with successful use"""
        self.total_uses += 1
        self.last_used = time.time()

        if confidence >= 0.75:
            self.successful_uses += 1

        # Recalculate success rate
        self.success_rate = self.successful_uses / self.total_uses

        # Update confidence (exponential moving average)
        alpha = 0.3  # Learning rate
        self.confidence = alpha * confidence + (1 - alpha) * self.confidence

    def get_heat_score(self) -> float:
        """
        Calculate heat score for this pattern.

        Heat = support * success_rate * confidence * recency_factor
        """
        # Recency factor (exponential decay over 24 hours)
        hours_since_use = (time.time() - self.last_used) / 3600
        recency = 0.95 ** hours_since_use  # 5% decay per hour

        return self.support * self.success_rate * self.confidence * recency

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'gesture': self.gesture,
            'voice_intent': self.voice_intent,
            'vision_context': list(self.vision_context),
            'tool_used': self.tool_used,
            'confidence': self.confidence,
            'support': self.support,
            'success_rate': self.success_rate,
            'total_uses': self.total_uses,
            'successful_uses': self.successful_uses,
            'heat_score': self.get_heat_score(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ARPattern':
        """Create from dictionary"""
        return cls(
            gesture=data.get('gesture'),
            voice_intent=data.get('voice_intent'),
            vision_context=set(data.get('vision_context', [])),
            tool_used=data.get('tool_used', 'unknown'),
            confidence=data.get('confidence', 0.0),
            support=data.get('support', 0),
            success_rate=data.get('success_rate', 0.0),
            total_uses=data.get('total_uses', 0),
            successful_uses=data.get('successful_uses', 0),
        )


# ============================================================================
# AR Pattern Learner
# ============================================================================

class ARPatternLearner:
    """
    Learns patterns from AR interactions.

    Extracts and tracks patterns: (gesture, voice_intent, vision) → tool

    Features:
    - Pattern extraction from successful AR queries
    - Pattern matching with fuzzy similarity
    - Hot pattern tracking (most successful patterns)
    - Automatic pattern pruning (stale patterns removed)
    """

    def __init__(
        self,
        min_confidence: float = 0.8,
        min_support: int = 3,
        max_patterns: int = 1000,
        prune_interval: int = 100,  # Prune every N patterns
    ):
        """
        Initialize AR pattern learner.

        Args:
            min_confidence: Minimum confidence to learn pattern
            min_support: Minimum support (observations) to keep pattern
            max_patterns: Maximum number of patterns to store
            prune_interval: Prune stale patterns every N learned patterns
        """
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_patterns = max_patterns
        self.prune_interval = prune_interval

        self.patterns: List[ARPattern] = []
        self.patterns_learned = 0
        self.patterns_pruned = 0

    def extract_and_learn(
        self,
        ar_query: 'ARQuery',  # Type hint (will import in actual usage)
        spacetime: 'Spacetime',  # Type hint
    ):
        """
        Extract pattern from AR query + result and learn it.

        Args:
            ar_query: AR query
            spacetime: Result spacetime
        """
        # Only learn from high-confidence results
        if spacetime.confidence < self.min_confidence:
            return

        # Extract features
        gesture = ar_query.gesture_detected
        voice_intent = ar_query.voice_intent
        vision_context = set(obj.object_type.value for obj in ar_query.vision_objects)
        tool_used = spacetime.metadata.get('tool_used', 'unknown')

        # Check if pattern already exists
        existing = self._find_matching_pattern(
            gesture, voice_intent, vision_context, tool_used
        )

        if existing:
            # Update existing pattern
            existing.support += 1
            existing.update_success(spacetime.confidence)
        else:
            # Create new pattern
            pattern = ARPattern(
                gesture=gesture,
                voice_intent=voice_intent,
                vision_context=vision_context,
                tool_used=tool_used,
                confidence=spacetime.confidence,
                support=1,
                success_rate=1.0,
                total_uses=1,
                successful_uses=1,
            )
            self.patterns.append(pattern)
            self.patterns_learned += 1

            # Prune if needed
            if self.patterns_learned % self.prune_interval == 0:
                self._prune_stale_patterns()

    def _find_matching_pattern(
        self,
        gesture: Optional[str],
        voice_intent: Optional[str],
        vision_context: Set[str],
        tool_used: str,
    ) -> Optional[ARPattern]:
        """Find existing pattern that matches (exact match on all fields)"""
        for pattern in self.patterns:
            if (pattern.gesture == gesture and
                pattern.voice_intent == voice_intent and
                pattern.vision_context == vision_context and
                pattern.tool_used == tool_used):
                return pattern
        return None

    def get_best_tool(
        self,
        gesture: Optional[str],
        voice_intent: Optional[str],
        vision_context: Set[str],
    ) -> Optional[Tuple[str, float]]:
        """
        Get best tool for given AR context.

        Args:
            gesture: Gesture detected
            voice_intent: Voice intent
            vision_context: Set of visible object types

        Returns:
            (tool_used, confidence) or None if no match
        """
        if not self.patterns:
            return None

        # Find best matching pattern
        best_pattern = None
        best_score = 0.0

        for pattern in self.patterns:
            # Skip low-support patterns
            if pattern.support < self.min_support:
                continue

            # Calculate match score
            match_score = pattern.matches(gesture, voice_intent, vision_context)

            # Weight by pattern quality
            weighted_score = match_score * pattern.confidence * pattern.success_rate

            if weighted_score > best_score:
                best_score = weighted_score
                best_pattern = pattern

        if best_pattern and best_score > 0.5:  # Threshold
            return best_pattern.tool_used, best_pattern.confidence

        return None

    def get_hot_patterns(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get hot patterns (highest heat scores).

        Args:
            limit: Maximum number of patterns

        Returns:
            List of pattern dictionaries sorted by heat
        """
        # Calculate heat scores
        scored = [
            (pattern.get_heat_score(), pattern)
            for pattern in self.patterns
            if pattern.support >= self.min_support
        ]

        # Sort by heat (descending)
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top patterns
        return [pattern.to_dict() for _, pattern in scored[:limit]]

    def mine_patterns(
        self,
        provenance_entries: List['ARProvenanceEntry'],  # Type hint
    ) -> List[ARPattern]:
        """
        Mine patterns from provenance entries.

        Args:
            provenance_entries: List of AR provenance entries

        Returns:
            List of newly discovered patterns
        """
        new_patterns = []

        # Group by (gesture, voice_intent, vision_context, tool)
        pattern_groups = defaultdict(list)

        for entry in provenance_entries:
            # Only consider successful entries
            if entry.score < self.min_confidence:
                continue

            key = (
                entry.gesture,
                entry.voice_intent,
                tuple(sorted(entry.vision_objects)),  # Hashable
                entry.action,  # Tool used
            )
            pattern_groups[key].append(entry)

        # Extract patterns with sufficient support
        for key, entries in pattern_groups.items():
            if len(entries) < self.min_support:
                continue

            gesture, voice_intent, vision_tuple, tool = key
            vision_context = set(vision_tuple) if vision_tuple else set()

            # Check if pattern already exists
            existing = self._find_matching_pattern(
                gesture, voice_intent, vision_context, tool
            )

            if not existing:
                # Calculate stats
                avg_confidence = sum(e.score for e in entries) / len(entries)

                # Create pattern
                pattern = ARPattern(
                    gesture=gesture,
                    voice_intent=voice_intent,
                    vision_context=vision_context,
                    tool_used=tool,
                    confidence=avg_confidence,
                    support=len(entries),
                    success_rate=1.0,  # All entries are successful
                    total_uses=len(entries),
                    successful_uses=len(entries),
                )

                self.patterns.append(pattern)
                new_patterns.append(pattern)
                self.patterns_learned += 1

        return new_patterns

    def _prune_stale_patterns(self):
        """Remove stale patterns (low support or old)"""
        before_count = len(self.patterns)

        # Keep patterns with:
        # 1. Support >= min_support
        # 2. Used in last 7 days
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7 days

        self.patterns = [
            p for p in self.patterns
            if p.support >= self.min_support and p.last_used >= cutoff_time
        ]

        pruned = before_count - len(self.patterns)
        if pruned > 0:
            self.patterns_pruned += pruned
            logger.info(f"Pruned {pruned} stale AR patterns")

        # Also enforce max patterns (keep hottest)
        if len(self.patterns) > self.max_patterns:
            # Sort by heat and keep top
            self.patterns.sort(key=lambda p: p.get_heat_score(), reverse=True)
            excess = len(self.patterns) - self.max_patterns
            self.patterns = self.patterns[:self.max_patterns]
            self.patterns_pruned += excess

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state"""
        return {
            'patterns': [p.to_dict() for p in self.patterns],
            'patterns_learned': self.patterns_learned,
            'patterns_pruned': self.patterns_pruned,
        }

    def load_state(self, state: Dict[str, Any]):
        """Load state from dictionary"""
        self.patterns = [
            ARPattern.from_dict(p) for p in state.get('patterns', [])
        ]
        self.patterns_learned = state.get('patterns_learned', 0)
        self.patterns_pruned = state.get('patterns_pruned', 0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            'total_patterns': len(self.patterns),
            'patterns_learned': self.patterns_learned,
            'patterns_pruned': self.patterns_pruned,
            'avg_support': (
                sum(p.support for p in self.patterns) / len(self.patterns)
                if self.patterns else 0.0
            ),
            'avg_confidence': (
                sum(p.confidence for p in self.patterns) / len(self.patterns)
                if self.patterns else 0.0
            ),
            'avg_success_rate': (
                sum(p.success_rate for p in self.patterns) / len(self.patterns)
                if self.patterns else 0.0
            ),
        }


# Export public API
__all__ = [
    'ARPatternLearner',
    'ARPattern',
    'GestureType',
    'VoiceIntent',
]

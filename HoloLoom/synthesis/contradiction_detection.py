"""
DreamEngine Contradiction Detection
====================================

Identifies conflicting information across memories and alerts users to reconcile.

**Philosophy:** Humans naturally update beliefs over time. DreamEngine helps track
and reconcile these changes transparently.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from HoloLoom.synthesis.types import (
    Contradiction,
    ContradictionType
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ContradictionConfig:
    """Configuration for contradiction detection."""
    enabled: bool = True
    lookback_days: int = 90  # Check last 90 days
    similarity_threshold: float = 0.15  # Topic similarity (word overlap, permissive for MVP)
    contradiction_threshold: float = 0.25  # Minimum score to flag (lowered for MVP)
    trigger_interval: int = 1000  # Run every N queries


# ============================================================================
# Contradiction Detection Engine
# ============================================================================

class ContradictionDetector:
    """
    Detects contradictory memories within a time window.

    Examples of contradictions:
    - Preference reversal: "I love Python" → "Python is too slow"
    - Fact update: "Paris population: 2M" → "Paris population: 2.1M"
    - Belief change: "AI will never match humans" → "AI surpassed expectations"
    """

    def __init__(self, config: Optional[ContradictionConfig] = None):
        """
        Initialize contradiction detector.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or ContradictionConfig()
        self.memory_history: List[Dict[str, Any]] = []

        # Sentiment indicator words (simple MVP - can upgrade to VADER/TextBlob)
        self.positive_words = {
            'love', 'great', 'excellent', 'best', 'good', 'better',
            'amazing', 'wonderful', 'prefer', 'like', 'enjoy'
        }
        self.negative_words = {
            'hate', 'terrible', 'worst', 'bad', 'worse', 'awful',
            'dislike', 'avoid', 'slow', 'buggy', 'broken'
        }

        # Mutually exclusive patterns
        self.exclusive_patterns = [
            (r'\byes\b', r'\bno\b'),
            (r'\btrue\b', r'\bfalse\b'),
            (r'\balways\b', r'\bnever\b'),
            (r'\beveryone\b', r'\bno one\b'),
            (r'\ball\b', r'\bnone\b'),
        ]

    def add_memory(self, memory: Dict[str, Any]):
        """
        Add memory to history for contradiction detection.

        Args:
            memory: Memory dict with keys:
                - id: Memory ID
                - content: Memory text
                - timestamp: When memory was created
                - entities: Extracted entities (optional)
                - embedding: Memory embedding (optional)
        """
        self.memory_history.append(memory)

        # Keep only recent memories
        cutoff = datetime.now() - timedelta(days=self.config.lookback_days)
        self.memory_history = [
            m for m in self.memory_history
            if m.get('timestamp', datetime.now()) >= cutoff
        ]

    def detect_contradictions(self) -> List[Contradiction]:
        """
        Detect contradictions in memory history.

        Returns:
            List of detected contradictions
        """
        if not self.config.enabled:
            return []

        contradictions = []

        # Compare all memory pairs
        for i, mem1 in enumerate(self.memory_history):
            for mem2 in self.memory_history[i+1:]:
                # Check if same memory
                if mem1.get('id') == mem2.get('id'):
                    continue

                # Check semantic similarity (same topic?)
                similarity = self._calculate_similarity(mem1, mem2)
                if similarity < self.config.similarity_threshold:
                    continue  # Different topics, skip

                # Check for contradiction signals
                contradiction = self._check_contradiction(mem1, mem2)
                if contradiction and contradiction.score >= self.config.contradiction_threshold:
                    contradictions.append(contradiction)

        return contradictions

    def _calculate_similarity(
        self,
        mem1: Dict[str, Any],
        mem2: Dict[str, Any]
    ) -> float:
        """
        Calculate semantic similarity between two memories.

        Uses word overlap for MVP (can upgrade to embeddings later).

        Args:
            mem1: First memory
            mem2: Second memory

        Returns:
            Similarity score (0.0-1.0)
        """
        # Extract words
        words1 = set(mem1.get('content', '').lower().split())
        words2 = set(mem2.get('content', '').lower().split())

        # Remove common stopwords (simple list for MVP)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'to', 'of', 'and', 'in', 'on', 'at'}
        words1 -= stopwords
        words2 -= stopwords

        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _check_contradiction(
        self,
        mem1: Dict[str, Any],
        mem2: Dict[str, Any]
    ) -> Optional[Contradiction]:
        """
        Check if two memories contradict each other.

        Args:
            mem1: Earlier memory
            mem2: Later memory

        Returns:
            Contradiction object if found, None otherwise
        """
        content1 = mem1.get('content', '').lower()
        content2 = mem2.get('content', '').lower()

        # Ensure temporal ordering (mem1 should be earlier)
        ts1 = mem1.get('timestamp', datetime.min)
        ts2 = mem2.get('timestamp', datetime.min)
        if ts2 < ts1:
            mem1, mem2 = mem2, mem1
            ts1, ts2 = ts2, ts1
            content1, content2 = content2, content1

        contradiction_score = 0.0
        contradiction_type = None
        explanation = ""

        # Signal 1: Opposite sentiment on same topic
        sentiment_conflict = self._detect_sentiment_conflict(content1, content2)
        if sentiment_conflict:
            contradiction_score += 0.3
            contradiction_type = ContradictionType.PREFERENCE_REVERSAL
            explanation = "Opposite sentiment detected on same topic"

        # Signal 2: Mutually exclusive statements
        if self._detect_mutual_exclusivity(content1, content2):
            contradiction_score += 0.5
            contradiction_type = ContradictionType.LOGICAL_CONFLICT
            explanation = "Mutually exclusive statements detected"

        # Signal 3: Numeric fact update
        if self._detect_fact_update(content1, content2):
            contradiction_score += 0.4
            contradiction_type = ContradictionType.FACT_UPDATE
            explanation = "Numeric fact appears to have been updated"

        # Signal 4: Negation pattern
        if self._detect_negation_conflict(content1, content2):
            contradiction_score += 0.4
            if not contradiction_type:
                contradiction_type = ContradictionType.BELIEF_CHANGE
            explanation = "Negation pattern detected (yes/no reversal)"

        # Return contradiction if score is high enough
        if contradiction_score >= self.config.contradiction_threshold:
            temporal_gap = (ts2 - ts1).days

            return Contradiction(
                memory1_id=mem1.get('id', 'unknown'),
                memory2_id=mem2.get('id', 'unknown'),
                memory1_content=mem1.get('content', ''),
                memory2_content=mem2.get('content', ''),
                score=contradiction_score,
                type=contradiction_type or ContradictionType.CONTEXT_DEPENDENT,
                temporal_gap_days=temporal_gap,
                explanation=explanation
            )

        return None

    def _detect_sentiment_conflict(self, text1: str, text2: str) -> bool:
        """
        Detect opposite sentiment in two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if opposite sentiment detected
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        # Count sentiment words
        positive1 = len(words1 & self.positive_words)
        negative1 = len(words1 & self.negative_words)

        positive2 = len(words2 & self.positive_words)
        negative2 = len(words2 & self.negative_words)

        # Determine dominant sentiment
        sentiment1 = 'positive' if positive1 > negative1 else 'negative' if negative1 > positive1 else 'neutral'
        sentiment2 = 'positive' if positive2 > negative2 else 'negative' if negative2 > positive2 else 'neutral'

        # Check for reversal
        return (sentiment1 == 'positive' and sentiment2 == 'negative') or \
               (sentiment1 == 'negative' and sentiment2 == 'positive')

    def _detect_mutual_exclusivity(self, text1: str, text2: str) -> bool:
        """
        Detect mutually exclusive statements.

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if mutually exclusive patterns found
        """
        for pattern1, pattern2 in self.exclusive_patterns:
            # Check if text1 has pattern1 and text2 has pattern2 (or vice versa)
            if (re.search(pattern1, text1) and re.search(pattern2, text2)) or \
               (re.search(pattern2, text1) and re.search(pattern1, text2)):
                return True

        return False

    def _detect_fact_update(self, text1: str, text2: str) -> bool:
        """
        Detect numeric fact updates.

        Args:
            text1: Earlier text
            text2: Later text

        Returns:
            True if numeric fact appears updated
        """
        # Extract numbers from both texts
        numbers1 = re.findall(r'\b\d+(?:\.\d+)?\b', text1)
        numbers2 = re.findall(r'\b\d+(?:\.\d+)?\b', text2)

        # If both contain numbers and they're different, might be an update
        if numbers1 and numbers2 and numbers1 != numbers2:
            # Check if texts are otherwise similar (suggest same fact)
            words1 = set(re.sub(r'\d+(?:\.\d+)?', '', text1).split())
            words2 = set(re.sub(r'\d+(?:\.\d+)?', '', text2).split())

            # High word overlap suggests same fact with different numbers
            overlap = len(words1 & words2) / max(len(words1), len(words2)) if words1 and words2 else 0
            return overlap > 0.6

        return False

    def _detect_negation_conflict(self, text1: str, text2: str) -> bool:
        """
        Detect negation conflicts (e.g., "X is Y" vs "X is not Y").

        Args:
            text1: First text
            text2: Second text

        Returns:
            True if negation conflict detected
        """
        # Simple negation detection (can be improved)
        negation_words = {'not', 'no', "n't", 'never', 'none', 'neither', 'nor'}

        words1 = set(text1.split())
        words2 = set(text2.split())

        has_negation1 = bool(words1 & negation_words)
        has_negation2 = bool(words2 & negation_words)

        # If one is negated and other is not, might be contradiction
        return has_negation1 != has_negation2

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get contradiction detection statistics.

        Returns:
            Dict with statistics
        """
        return {
            'total_memories': len(self.memory_history),
            'lookback_days': self.config.lookback_days,
            'similarity_threshold': self.config.similarity_threshold,
            'contradiction_threshold': self.config.contradiction_threshold
        }


# ============================================================================
# Standalone Functions
# ============================================================================

def detect_contradictions_in_memories(
    memories: List[Dict[str, Any]],
    config: Optional[ContradictionConfig] = None
) -> List[Contradiction]:
    """
    Detect contradictions in a list of memories (standalone function).

    Args:
        memories: List of memory dicts
        config: Optional configuration

    Returns:
        List of detected contradictions

    Example:
        >>> memories = [
        ...     {'id': 'm1', 'content': 'I love Python', 'timestamp': datetime(2025, 1, 1)},
        ...     {'id': 'm2', 'content': 'Python is terrible', 'timestamp': datetime(2025, 3, 1)}
        ... ]
        >>> contradictions = detect_contradictions_in_memories(memories)
        >>> print(len(contradictions))
        1
        >>> print(contradictions[0].type)
        ContradictionType.PREFERENCE_REVERSAL
    """
    detector = ContradictionDetector(config)

    for memory in memories:
        detector.add_memory(memory)

    return detector.detect_contradictions()

"""
RECALL Loom - The Librarian
============================

Dense retrieval, memory-grounded. Trusts the archive.

Cognitive Style:
- High recall (k=20)
- Conservative temperature (0.2)
- Heavy memory bias (0.9)
- Light inference (0.1)

Philosophy:
"What has been recorded? What does memory say?
The Librarian trusts the archive above all else."

Date: December 2025
Author: HoloLoom Architecture Team
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from hololoom.loom.base_loom import BaseLoom
from hololoom.loom.protocol import RECALL, PatternInsight, CorrectionInsight
from hololoom.fabric.fabric import Fabric
from hololoom.protocols.department import (
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
)

logger = logging.getLogger(__name__)


class RecallLoom(BaseLoom):
    """
    The Librarian - Dense retrieval, memory-grounded.

    Trusts what has been recorded. Excels at factual recall and
    finding relevant precedents. Admits when memory is sparse.

    Attributes:
        retrieval_k: Number of memories to retrieve (default: 20)
        temperature: Creativity level (default: 0.2, conservative)
        trust_memory_weight: Weight for memory vs inference (default: 0.9)
        trust_inference_weight: Weight for inference (default: 0.1)
    """

    def __init__(
        self,
        retrieval_k: int = 20,
        temperature: float = 0.2,
        trust_memory_weight: float = 0.9,
        trust_inference_weight: float = 0.1,
        memory_backend: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the Recall Loom.

        Args:
            retrieval_k: Number of memories to retrieve
            temperature: Creativity level (0-1, lower is more conservative)
            trust_memory_weight: How much to trust memory (0-1)
            trust_inference_weight: How much to trust inference (0-1)
            memory_backend: Optional memory backend for retrieval
            **kwargs: Arguments passed to BaseLoom
        """
        super().__init__(**kwargs)

        # Cognitive style parameters
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.trust_memory_weight = trust_memory_weight
        self.trust_inference_weight = trust_inference_weight

        # Memory backend (optional - can work without)
        self._memory = memory_backend

        # Track memory gaps for learning
        self._memory_gaps: List[str] = []

    @property
    def perspective(self) -> str:
        """The Librarian's perspective."""
        return RECALL

    @property
    def blind_spots(self) -> List[str]:
        """What the Librarian cannot see."""
        return [
            "novel connections not in memory",
            "future predictions",
            "creative leaps",
            "things never recorded",
            "implicit knowledge",
            "context beyond the archive",
        ]

    def admits_ignorance(self, query: str) -> List[str]:
        """
        What the Librarian knows it doesn't know about this query.

        Args:
            query: The query to assess

        Returns:
            List of specific knowledge gaps
        """
        ignorance = []

        # Check for temporal keywords suggesting future
        future_words = ['will', 'predict', 'forecast', 'expect', 'future']
        if any(word in query.lower() for word in future_words):
            ignorance.append("Cannot predict future events - only recall past")

        # Check for creative/novel keywords
        creative_words = ['imagine', 'create', 'invent', 'new', 'novel', 'original']
        if any(word in query.lower() for word in creative_words):
            ignorance.append("Cannot generate novel ideas - only retrieve existing")

        # Add any detected memory gaps
        if self._memory_gaps:
            ignorance.extend([f"No memory of: {gap}" for gap in self._memory_gaps[-3:]])

        # Default acknowledgment
        if not ignorance:
            ignorance.append("May have incomplete archive coverage")

        return ignorance

    def falsifiable_by(self, claim: str) -> List[str]:
        """
        What evidence would change the Librarian's claim.

        Args:
            claim: The claim to assess

        Returns:
            List of falsifiers
        """
        return [
            "Finding contradicting memory entries",
            "Discovering newer information that supersedes archived data",
            "Evidence that source memory was incorrect or outdated",
            "Multiple contradicting memories on the same topic",
        ]

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute recall logic - dense retrieval from memory.

        Args:
            request: Department request with query

        Returns:
            Response with recalled information
        """
        start_time = datetime.now()
        query = request.query

        # Step 1: Dense retrieval from memory
        threads = await self._retrieve_memories(query)

        # Step 2: Compile what memory says
        memory_response = self._compile_memory_response(threads)

        # Step 3: Identify memory gaps
        gaps = self._identify_memory_gaps(query, threads)
        self._memory_gaps = gaps  # Store for admits_ignorance

        # Step 4: Compute confidence based on memory coverage
        confidence = self._compute_confidence(threads)

        # Step 5: Compute epistemic confidence (how certain of certainty)
        epistemic = self._compute_epistemic_confidence(gaps, threads)

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._update_metrics(confidence, duration_ms)

        # Log insights for dreaming
        if confidence > 0.8:
            self.log_pattern_insight({
                "query_type": "factual_recall",
                "retrieval_k": self.retrieval_k,
                "threads_found": len(threads),
                "success": True,
            }, confidence=confidence)

        # Create fabric
        fabric = self._create_fabric(
            query=query,
            response=memory_response,
            tool_used="recall",
            confidence=confidence,
            duration_ms=duration_ms,
            epistemic_confidence=epistemic,
            admits_ignorance=gaps,
            falsifiable_by=self.falsifiable_by(memory_response),
            metadata={
                "threads_retrieved": len(threads),
                "retrieval_k": self.retrieval_k,
                "memory_coverage": confidence,
            },
        )

        return DepartmentResponse(
            result=memory_response,
            confidence=ConfidenceMetadata(
                score=confidence,
                source="recall_loom",
                reasoning=f"Based on {len(threads)} memory threads",
            ),
            metadata={
                "perspective": self.perspective,
                "fabric": fabric,
                "threads": threads,
                "gaps": gaps,
            },
        )

    async def _retrieve_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from backend.

        Args:
            query: Query to search for

        Returns:
            List of memory threads
        """
        if self._memory is None:
            # No memory backend - return empty
            logger.debug("No memory backend configured for RecallLoom")
            return []

        try:
            # Use memory backend's recall method
            if hasattr(self._memory, 'recall'):
                memories = await self._memory.recall(query, k=self.retrieval_k)
                return [
                    {
                        "id": getattr(m, 'id', str(i)),
                        "content": getattr(m, 'content', str(m)),
                        "relevance": getattr(m, 'relevance', 0.5),
                        "timestamp": getattr(m, 'timestamp', None),
                    }
                    for i, m in enumerate(memories)
                ]
            else:
                # Fallback for simpler backends
                return []
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return []

    def _compile_memory_response(self, threads: List[Dict[str, Any]]) -> str:
        """
        Compile retrieved threads into a response.

        Args:
            threads: Retrieved memory threads

        Returns:
            Compiled response text
        """
        if not threads:
            return "No relevant memories found in the archive."

        # Weight by relevance
        sorted_threads = sorted(
            threads,
            key=lambda t: t.get('relevance', 0),
            reverse=True
        )

        # Compile top threads
        parts = []
        for i, thread in enumerate(sorted_threads[:5], 1):
            content = thread.get('content', '')
            relevance = thread.get('relevance', 0)
            parts.append(f"[{i}] (relevance: {relevance:.2f}) {content}")

        if len(threads) > 5:
            parts.append(f"... and {len(threads) - 5} more memories")

        return "\n".join(parts)

    def _identify_memory_gaps(
        self,
        query: str,
        threads: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify gaps in memory coverage.

        Args:
            query: Original query
            threads: Retrieved threads

        Returns:
            List of identified gaps
        """
        gaps = []

        # Gap: No threads found
        if not threads:
            gaps.append(f"No memories related to: {query[:50]}...")

        # Gap: Low relevance threads
        elif all(t.get('relevance', 0) < 0.5 for t in threads):
            gaps.append("All retrieved memories have low relevance")

        # Gap: Limited thread count
        if len(threads) < 3:
            gaps.append(f"Limited coverage: only {len(threads)} relevant memories")

        return gaps

    def _compute_confidence(self, threads: List[Dict[str, Any]]) -> float:
        """
        Compute confidence based on memory coverage.

        Args:
            threads: Retrieved threads

        Returns:
            Confidence score (0-1)
        """
        if not threads:
            return 0.2  # Low but not zero - might still have general knowledge

        # Weight by thread count and relevance
        avg_relevance = sum(t.get('relevance', 0.5) for t in threads) / len(threads)
        count_factor = min(len(threads) / self.retrieval_k, 1.0)

        # Combine: 60% relevance, 40% coverage
        confidence = 0.6 * avg_relevance + 0.4 * count_factor

        # Apply memory trust weight
        confidence = confidence * self.trust_memory_weight + 0.1  # Floor of 0.1

        return min(confidence, 1.0)

    def _compute_epistemic_confidence(
        self,
        gaps: List[str],
        threads: List[Dict[str, Any]]
    ) -> float:
        """
        Compute epistemic confidence (confidence in confidence).

        Args:
            gaps: Identified memory gaps
            threads: Retrieved threads

        Returns:
            Epistemic confidence (0-1)
        """
        # Start with base epistemic confidence
        epistemic = 0.8

        # Reduce for each gap
        epistemic -= len(gaps) * 0.15

        # Reduce if threads are inconsistent (high variance in relevance)
        if threads:
            relevances = [t.get('relevance', 0.5) for t in threads]
            variance = sum((r - sum(relevances)/len(relevances))**2 for r in relevances) / len(relevances)
            if variance > 0.1:
                epistemic -= 0.1

        return max(epistemic, 0.2)


__all__ = ['RecallLoom']

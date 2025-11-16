"""
Fast Path Implementations for Simple Queries
=============================================
Optimized paths that skip expensive orchestrator steps for simple queries.

Fast Paths:
- TRIVIAL: Cached responses for greetings (<1ms)
- SIMPLE: Direct retrieval without graph traversal (<50ms)

Author: Claude Code
Date: 2025-11-12
"""

import time
from typing import Dict, Optional
from HoloLoom.Documentation.types import Query
from HoloLoom.fabric.spacetime import Spacetime, WeavingTrace


class FastPathCache:
    """
    Cache for trivial query responses.

    Stores pre-computed responses for common greetings and acknowledgments.
    """

    TRIVIAL_RESPONSES = {
        # Greetings
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! What can I do for you?",
        "hey": "Hey! What's up?",

        # Thanks
        "thanks": "You're welcome!",
        "thank you": "Happy to help!",
        "thx": "No problem!",

        # Acknowledgments
        "ok": "Got it!",
        "okay": "Understood!",
        "sure": "Sounds good!",
        "yes": "Affirmative!",
        "no": "Understood.",

        # Goodbyes
        "bye": "Goodbye! Have a great day!",
        "goodbye": "See you later!",

        # Help
        "help": "I can answer questions, search knowledge, and assist with tasks. What would you like to know?",
    }

    def get(self, query_text: str) -> Optional[str]:
        """
        Get cached response for trivial query.

        Args:
            query_text: Query string (will be normalized)

        Returns:
            Cached response or None
        """
        # Normalize: lowercase, strip punctuation/whitespace
        normalized = query_text.lower().strip().rstrip('!?.').strip()

        return self.TRIVIAL_RESPONSES.get(normalized)


async def handle_trivial_query(query: Query) -> Spacetime:
    """
    Handle trivial queries with cached responses.

    Target latency: <10ms (usually <1ms)

    Args:
        query: Query object

    Returns:
        Spacetime with cached response
    """
    from datetime import datetime

    start_timestamp = datetime.now()
    start_time = time.perf_counter()

    cache = FastPathCache()
    response = cache.get(query.text)

    if response is None:
        # Fallback generic response
        response = "I'm here to help! What would you like to know?"

    end_timestamp = datetime.now()
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Create minimal spacetime
    trace = WeavingTrace(
        start_time=start_timestamp,
        end_time=end_timestamp,
        duration_ms=duration_ms,
        stage_durations={
            "fast_path_trivial": duration_ms
        }
    )

    spacetime = Spacetime(
        query_text=query.text,
        response=response,
        tool_used="fast_path_cache",
        confidence=0.95,  # High confidence for cached responses
        trace=trace,
        metadata={
            "fast_path": "trivial",
            "cached": True,
            "latency_ms": duration_ms
        }
    )

    return spacetime


async def handle_simple_query(
    query: Query,
    orchestrator: 'WeavingOrchestrator'
) -> Spacetime:
    """
    Handle simple queries with minimal orchestrator steps.

    Skips expensive steps:
    - Graph traversal
    - Spectral analysis
    - Complex policy decisions

    Target latency: <50ms

    Args:
        query: Query object
        orchestrator: Orchestrator instance (for retrieval)

    Returns:
        Spacetime with response
    """
    from datetime import datetime

    start_timestamp = datetime.now()
    start_time = time.perf_counter()

    # Step 1: Direct retrieval (no graph traversal)
    if orchestrator.retriever:
        hits = await orchestrator.retriever.search(
            query=query.text,
            k=3,  # Reduced from default (faster)
            fast=True
        )
        shards = [shard for shard, _ in hits]
    else:
        shards = []

    # Step 2: Simple response generation (no policy)
    if shards:
        # Return first shard as response
        response = f"Based on available knowledge: {shards[0].text}"
        confidence = 0.75
    else:
        # No results found
        response = "I don't have information about that. Could you rephrase or provide more context?"
        confidence = 0.50

    end_timestamp = datetime.now()
    duration_ms = (time.perf_counter() - start_time) * 1000

    # Create spacetime
    trace = WeavingTrace(
        start_time=start_timestamp,
        end_time=end_timestamp,
        duration_ms=duration_ms,
        stage_durations={
            "fast_path_simple": duration_ms,
            "retrieval": duration_ms * 0.8,  # Estimate
            "response_gen": duration_ms * 0.2
        },
        threads_activated=[s.id for s in shards]
    )

    spacetime = Spacetime(
        query_text=query.text,
        response=response,
        tool_used="fast_path_simple",
        confidence=confidence,
        trace=trace,
        metadata={
            "fast_path": "simple",
            "shards_used": len(shards),
            "latency_ms": duration_ms
        }
    )

    return spacetime


class FastPathRouter:
    """
    Routes queries to appropriate fast paths.

    Integrates with QueryClassifier to determine routing.
    """

    def __init__(self, orchestrator: Optional['WeavingOrchestrator'] = None):
        """
        Initialize router.

        Args:
            orchestrator: Orchestrator instance (for simple path)
        """
        self.orchestrator = orchestrator
        self.trivial_count = 0
        self.simple_count = 0
        self.complex_count = 0

    async def route(self, query: Query, classification) -> Spacetime:
        """
        Route query to appropriate handler.

        Args:
            query: Query object
            classification: ClassificationResult from QueryClassifier

        Returns:
            Spacetime from appropriate handler
        """
        from HoloLoom.routing.query_classifier import QueryComplexity

        if classification.complexity == QueryComplexity.TRIVIAL:
            self.trivial_count += 1
            return await handle_trivial_query(query)

        elif classification.complexity == QueryComplexity.SIMPLE:
            self.simple_count += 1
            if self.orchestrator:
                return await handle_simple_query(query, self.orchestrator)
            # Fallback to full pipeline if no orchestrator
            return await self.orchestrator.weave(query)

        else:
            # COMPLEX or RESEARCH: Use full pipeline
            self.complex_count += 1
            if self.orchestrator:
                return await self.orchestrator.weave(query)
            else:
                raise ValueError("Orchestrator required for complex queries")

    def get_stats(self) -> Dict[str, int]:
        """
        Get routing statistics.

        Returns:
            Dict with counts by complexity
        """
        total = self.trivial_count + self.simple_count + self.complex_count
        return {
            "trivial": self.trivial_count,
            "simple": self.simple_count,
            "complex": self.complex_count,
            "total": total,
            "fast_path_percentage": (
                (self.trivial_count + self.simple_count) / total * 100
                if total > 0 else 0
            )
        }

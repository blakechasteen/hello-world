"""
HoloLoom Memory - Anticipatory Retrieval (Phase 5)
==================================================
Predicts follow-up queries and pre-fetches relevant memories for
faster response times and improved context quality.

Philosophy:
    "Prepare before you're asked."

    By analyzing query patterns and session context, we can predict
    what the user is likely to ask next and pre-fetch relevant memories.
    This creates a more fluid, responsive experience.

Architecture:
    QueryPatternTracker: Learns common query sequences (n-gram based)
    FollowUpPredictor: Predicts likely follow-up queries
    AnticipatoryFetcher: Pre-fetches memories for predicted queries
    SessionContext: Maintains per-session state for predictions

Integration:
    Works with:
    - ShardContributionTracker (Phase 3) for success-weighted predictions
    - DeltaStore (Phase 4) for temporal query patterns
    - Memory Conductor for intelligent retrieval

Author: Claude Code
Date: December 2025 (Phase 5 - Anticipatory Retrieval)
"""

import logging
import time
import hashlib
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import re

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

class QueryType(Enum):
    """Types of queries for pattern matching."""
    DEFINITION = "definition"     # "What is X?"
    EXPLANATION = "explanation"   # "Explain X", "How does X work?"
    EXAMPLE = "example"          # "Show me an example", "Give me an example"
    COMPARISON = "comparison"    # "Compare X and Y", "X vs Y"
    APPLICATION = "application"  # "When should I use X?", "How to apply X"
    IMPLEMENTATION = "implementation"  # "How to implement X?", "Code for X"
    TRADEOFFS = "tradeoffs"      # "Tradeoffs of X", "Pros and cons of X"
    ALTERNATIVES = "alternatives"  # "Alternatives to X", "Other options"
    GENERAL = "general"          # Everything else


@dataclass
class QueryPattern:
    """A pattern of queries that tend to occur together."""
    pattern_id: str
    trigger_type: QueryType
    follow_up_types: List[QueryType]
    confidence: float = 0.5  # How likely this pattern is
    occurrences: int = 1
    last_seen: float = field(default_factory=time.time)

    # Topic continuity (same topic across queries)
    topic_continuity: float = 0.8  # How likely follow-up is on same topic


@dataclass
class SessionContext:
    """Session state for anticipatory retrieval."""
    session_id: str
    created_at: float = field(default_factory=time.time)

    # Query history (recent queries in this session)
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 20

    # Topics discussed
    topics: List[str] = field(default_factory=list)
    current_topic: Optional[str] = None

    # Pre-fetched memories (ready for next query)
    prefetched_memory_ids: Set[str] = field(default_factory=set)
    prefetch_timestamp: Optional[float] = None

    # Statistics
    cache_hits: int = 0
    total_queries: int = 0

    def add_query(
        self,
        query_text: str,
        query_type: QueryType,
        topic: Optional[str] = None,
        confidence: float = 0.0
    ):
        """Add a query to history."""
        self.query_history.append({
            "text": query_text,
            "type": query_type.value,
            "topic": topic,
            "confidence": confidence,
            "timestamp": time.time(),
        })

        # Trim to max history
        if len(self.query_history) > self.max_history:
            self.query_history = self.query_history[-self.max_history:]

        # Update current topic
        if topic:
            self.current_topic = topic
            if topic not in self.topics:
                self.topics.append(topic)

        self.total_queries += 1

    def get_last_n_queries(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get last N queries from history."""
        return self.query_history[-n:] if self.query_history else []

    def check_prefetch_hit(self, memory_ids: Set[str]) -> float:
        """Check how many requested memories were pre-fetched."""
        if not self.prefetched_memory_ids or not memory_ids:
            return 0.0

        hits = len(self.prefetched_memory_ids & memory_ids)
        if hits > 0:
            self.cache_hits += 1
        return hits / len(memory_ids) if memory_ids else 0.0


@dataclass
class PredictedQuery:
    """A predicted follow-up query."""
    query_type: QueryType
    probability: float  # 0.0-1.0
    topic: Optional[str] = None
    template: Optional[str] = None  # "How does {topic} work?"
    reasoning: str = ""


@dataclass
class PrefetchResult:
    """Result of anticipatory pre-fetching."""
    predicted_queries: List[PredictedQuery]
    prefetched_memory_ids: Set[str]
    prefetch_time_ms: float
    memories_fetched: int


@dataclass
class AnticipatoryConfig:
    """Configuration for anticipatory retrieval."""
    # Prediction settings
    min_prediction_confidence: float = 0.3  # Minimum confidence to predict
    max_predictions: int = 3  # Maximum follow-up queries to predict
    topic_continuity_weight: float = 0.7  # Weight for same-topic predictions

    # Prefetching settings
    prefetch_enabled: bool = True
    prefetch_k: int = 10  # Memories to prefetch per prediction
    max_prefetch_total: int = 25  # Maximum total prefetched memories
    prefetch_ttl_seconds: float = 300.0  # Time-to-live for prefetched memories

    # Pattern learning
    pattern_learning_enabled: bool = True
    min_pattern_occurrences: int = 3  # Min occurrences to trust pattern
    pattern_decay_rate: float = 0.95  # Decay per day


# ============================================================================
# Query Type Classifier
# ============================================================================

class QueryTypeClassifier:
    """Classifies queries into types for pattern matching."""

    # Patterns for each query type
    PATTERNS = {
        QueryType.DEFINITION: [
            r"^what is\s+",
            r"^define\s+",
            r"^what are\s+",
            r"^what does\s+\w+\s+mean",
        ],
        QueryType.EXPLANATION: [
            r"^explain\s+",
            r"^how does\s+",
            r"^how do\s+",
            r"why does\s+",
            r"why is\s+",
            r"^tell me about\s+",
        ],
        QueryType.EXAMPLE: [
            r"^show\s+(me\s+)?(an?\s+)?example",
            r"^give\s+(me\s+)?(an?\s+)?example",
            r"^example\s+of\s+",
            r"can you show",
            r"can you give.*example",
        ],
        QueryType.COMPARISON: [
            r"^compare\s+",
            r"\s+vs\.?\s+",
            r"\s+versus\s+",
            r"difference\s+between",
            r"compared\s+to",
            r"^how does\s+.+\s+differ",
        ],
        QueryType.APPLICATION: [
            r"^when\s+(should|do|can)\s+",
            r"^where\s+(should|do|can)\s+",
            r"^how\s+to\s+use\s+",
            r"^how\s+to\s+apply\s+",
            r"use case",
        ],
        QueryType.IMPLEMENTATION: [
            r"^how\s+to\s+(implement|build|create|make|code)",
            r"^implement\s+",
            r"^code\s+for\s+",
            r"^write\s+.*(code|function|class)",
            r"show.*code",
        ],
        QueryType.TRADEOFFS: [
            r"tradeoff",
            r"trade-off",
            r"pros?\s+and\s+cons?",
            r"advantage.*disadvantage",
            r"benefit.*drawback",
            r"strengths?\s+and\s+weakness",
        ],
        QueryType.ALTERNATIVES: [
            r"alternative",
            r"other\s+(options?|choices?|ways?)",
            r"instead\s+of",
            r"similar\s+to",
            r"like\s+.+\s+but",
        ],
    }

    def __init__(self):
        """Initialize with compiled patterns."""
        self._compiled = {}
        for qtype, patterns in self.PATTERNS.items():
            self._compiled[qtype] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify(self, query: str) -> QueryType:
        """
        Classify a query into a type.

        Args:
            query: Query text

        Returns:
            QueryType classification
        """
        query = query.strip()

        for qtype, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    return qtype

        return QueryType.GENERAL

    def extract_topic(self, query: str, query_type: QueryType) -> Optional[str]:
        """
        Extract the main topic from a query.

        Args:
            query: Query text
            query_type: Query type

        Returns:
            Extracted topic or None
        """
        query = query.strip().lower()

        # Simple extraction based on query type
        if query_type == QueryType.DEFINITION:
            # "What is X?" -> extract X
            match = re.match(r"^what\s+(?:is|are)\s+(.+?)[\?\.]?\s*$", query, re.I)
            if match:
                return match.group(1).strip()

        elif query_type == QueryType.EXPLANATION:
            # "Explain X" or "How does X work?"
            match = re.match(r"^explain\s+(.+?)[\?\.]?\s*$", query, re.I)
            if match:
                return match.group(1).strip()
            match = re.match(r"^how\s+does\s+(.+?)\s+work", query, re.I)
            if match:
                return match.group(1).strip()

        elif query_type == QueryType.COMPARISON:
            # "Compare X and Y" -> return both
            match = re.match(r"^compare\s+(.+?)\s+(?:and|vs\.?|versus)\s+(.+?)[\?\.]?\s*$", query, re.I)
            if match:
                return f"{match.group(1)} vs {match.group(2)}"

        # Fallback: extract significant words
        words = re.findall(r'\b[a-z]{3,}\b', query)
        stopwords = {'what', 'how', 'why', 'when', 'where', 'does', 'the', 'is', 'are', 'can', 'you', 'explain', 'tell', 'about', 'show', 'give', 'example'}
        meaningful = [w for w in words if w not in stopwords]

        return ' '.join(meaningful[:3]) if meaningful else None


# ============================================================================
# Pattern Tracker
# ============================================================================

class QueryPatternTracker:
    """
    Tracks and learns patterns of query sequences.

    Learns which query types tend to follow other types:
    - DEFINITION → EXPLANATION (common)
    - DEFINITION → EXAMPLE (common)
    - EXPLANATION → IMPLEMENTATION (common)
    - COMPARISON → TRADEOFFS (common)
    """

    # Default transition probabilities (prior knowledge)
    DEFAULT_TRANSITIONS = {
        QueryType.DEFINITION: [
            (QueryType.EXPLANATION, 0.35),
            (QueryType.EXAMPLE, 0.25),
            (QueryType.APPLICATION, 0.15),
            (QueryType.COMPARISON, 0.10),
        ],
        QueryType.EXPLANATION: [
            (QueryType.EXAMPLE, 0.30),
            (QueryType.IMPLEMENTATION, 0.25),
            (QueryType.APPLICATION, 0.20),
            (QueryType.TRADEOFFS, 0.10),
        ],
        QueryType.EXAMPLE: [
            (QueryType.IMPLEMENTATION, 0.30),
            (QueryType.APPLICATION, 0.25),
            (QueryType.EXPLANATION, 0.20),
        ],
        QueryType.COMPARISON: [
            (QueryType.TRADEOFFS, 0.35),
            (QueryType.APPLICATION, 0.25),
            (QueryType.ALTERNATIVES, 0.20),
        ],
        QueryType.APPLICATION: [
            (QueryType.IMPLEMENTATION, 0.35),
            (QueryType.EXAMPLE, 0.25),
            (QueryType.TRADEOFFS, 0.20),
        ],
        QueryType.IMPLEMENTATION: [
            (QueryType.EXAMPLE, 0.30),
            (QueryType.APPLICATION, 0.25),
            (QueryType.TRADEOFFS, 0.15),
        ],
        QueryType.TRADEOFFS: [
            (QueryType.ALTERNATIVES, 0.30),
            (QueryType.APPLICATION, 0.25),
            (QueryType.COMPARISON, 0.20),
        ],
        QueryType.ALTERNATIVES: [
            (QueryType.COMPARISON, 0.30),
            (QueryType.TRADEOFFS, 0.25),
            (QueryType.EXAMPLE, 0.20),
        ],
        QueryType.GENERAL: [
            (QueryType.EXPLANATION, 0.25),
            (QueryType.EXAMPLE, 0.20),
            (QueryType.DEFINITION, 0.15),
        ],
    }

    def __init__(self, config: Optional[AnticipatoryConfig] = None):
        """
        Initialize pattern tracker.

        Args:
            config: Configuration for tracking
        """
        self.config = config or AnticipatoryConfig()

        # Transition counts: from_type -> to_type -> count
        self.transition_counts: Dict[QueryType, Dict[QueryType, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Total transitions from each type
        self.from_counts: Dict[QueryType, int] = defaultdict(int)

        # Topic-specific patterns
        self.topic_transitions: Dict[str, Dict[QueryType, Dict[QueryType, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        self.logger = logging.getLogger(f"{__name__}.QueryPatternTracker")

    def record_transition(
        self,
        from_type: QueryType,
        to_type: QueryType,
        topic: Optional[str] = None
    ):
        """
        Record a query type transition.

        Args:
            from_type: Previous query type
            to_type: Current query type
            topic: Topic (for topic-specific learning)
        """
        if not self.config.pattern_learning_enabled:
            return

        self.transition_counts[from_type][to_type] += 1
        self.from_counts[from_type] += 1

        if topic:
            self.topic_transitions[topic][from_type][to_type] += 1

        self.logger.debug(
            f"Recorded transition: {from_type.value} -> {to_type.value}"
            f" (topic={topic})"
        )

    def get_transition_probability(
        self,
        from_type: QueryType,
        to_type: QueryType,
        topic: Optional[str] = None
    ) -> float:
        """
        Get probability of transitioning from one type to another.

        Args:
            from_type: Previous query type
            to_type: Target query type
            topic: Optional topic for topic-specific probability

        Returns:
            Probability (0.0-1.0)
        """
        # Start with learned probability
        from_total = self.from_counts[from_type]

        if from_total >= self.config.min_pattern_occurrences:
            # Use learned probability
            to_count = self.transition_counts[from_type][to_type]
            learned_prob = to_count / from_total
        else:
            # Fall back to default
            learned_prob = 0.0
            for default_type, default_prob in self.DEFAULT_TRANSITIONS.get(from_type, []):
                if default_type == to_type:
                    learned_prob = default_prob
                    break

        # Blend with topic-specific if available
        if topic and topic in self.topic_transitions:
            topic_counts = self.topic_transitions[topic][from_type]
            topic_total = sum(topic_counts.values())
            if topic_total >= 2:
                topic_prob = topic_counts[to_type] / topic_total
                # Blend: 70% topic, 30% general
                learned_prob = 0.7 * topic_prob + 0.3 * learned_prob

        return learned_prob

    def get_likely_follow_ups(
        self,
        from_type: QueryType,
        topic: Optional[str] = None,
        top_k: int = 3
    ) -> List[Tuple[QueryType, float]]:
        """
        Get most likely follow-up query types.

        Args:
            from_type: Previous query type
            topic: Optional topic
            top_k: Number of predictions to return

        Returns:
            List of (QueryType, probability) tuples, sorted by probability
        """
        predictions = []

        for to_type in QueryType:
            if to_type == from_type:
                continue  # Skip same type

            prob = self.get_transition_probability(from_type, to_type, topic)
            if prob > 0:
                predictions.append((to_type, prob))

        # Sort by probability, take top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]

    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        total_transitions = sum(self.from_counts.values())
        topics_tracked = len(self.topic_transitions)

        return {
            "total_transitions": total_transitions,
            "unique_from_types": len(self.from_counts),
            "topics_tracked": topics_tracked,
            "transition_matrix": {
                from_type.value: {
                    to_type.value: count
                    for to_type, count in to_counts.items()
                }
                for from_type, to_counts in self.transition_counts.items()
            },
        }


# ============================================================================
# Follow-Up Predictor
# ============================================================================

class FollowUpPredictor:
    """
    Predicts likely follow-up queries based on current query and session context.
    """

    # Query templates for generating predictions
    TEMPLATES = {
        QueryType.EXPLANATION: [
            "How does {topic} work?",
            "Explain {topic} in more detail",
            "Why is {topic} important?",
        ],
        QueryType.EXAMPLE: [
            "Show me an example of {topic}",
            "Give me a code example for {topic}",
            "Can you demonstrate {topic}?",
        ],
        QueryType.COMPARISON: [
            "Compare {topic} with alternatives",
            "How does {topic} differ from others?",
        ],
        QueryType.APPLICATION: [
            "When should I use {topic}?",
            "What are the use cases for {topic}?",
            "How to apply {topic}?",
        ],
        QueryType.IMPLEMENTATION: [
            "How to implement {topic}?",
            "Show me code for {topic}",
            "How to build {topic}?",
        ],
        QueryType.TRADEOFFS: [
            "What are the tradeoffs of {topic}?",
            "Pros and cons of {topic}?",
        ],
        QueryType.ALTERNATIVES: [
            "What are alternatives to {topic}?",
            "Similar solutions to {topic}?",
        ],
    }

    def __init__(
        self,
        pattern_tracker: QueryPatternTracker,
        classifier: QueryTypeClassifier,
        config: Optional[AnticipatoryConfig] = None
    ):
        """
        Initialize predictor.

        Args:
            pattern_tracker: Pattern tracker for transition probabilities
            classifier: Query type classifier
            config: Configuration
        """
        self.pattern_tracker = pattern_tracker
        self.classifier = classifier
        self.config = config or AnticipatoryConfig()

        self.logger = logging.getLogger(f"{__name__}.FollowUpPredictor")

    def predict(
        self,
        current_query: str,
        session: SessionContext
    ) -> List[PredictedQuery]:
        """
        Predict likely follow-up queries.

        Args:
            current_query: The current query
            session: Session context

        Returns:
            List of predicted follow-up queries
        """
        # Classify current query
        current_type = self.classifier.classify(current_query)
        topic = self.classifier.extract_topic(current_query, current_type)

        # Use session topic if available
        if not topic and session.current_topic:
            topic = session.current_topic

        # Get likely follow-ups from pattern tracker
        follow_ups = self.pattern_tracker.get_likely_follow_ups(
            from_type=current_type,
            topic=topic,
            top_k=self.config.max_predictions
        )

        predictions = []

        for follow_type, base_prob in follow_ups:
            if base_prob < self.config.min_prediction_confidence:
                continue

            # Adjust probability based on session context
            adjusted_prob = self._adjust_probability(
                base_prob, follow_type, session
            )

            if adjusted_prob < self.config.min_prediction_confidence:
                continue

            # Generate prediction
            template = self._get_template(follow_type, topic)

            prediction = PredictedQuery(
                query_type=follow_type,
                probability=adjusted_prob,
                topic=topic,
                template=template,
                reasoning=f"After {current_type.value}, users often ask {follow_type.value} questions"
            )
            predictions.append(prediction)

        # Sort by probability
        predictions.sort(key=lambda p: p.probability, reverse=True)

        return predictions[:self.config.max_predictions]

    def _adjust_probability(
        self,
        base_prob: float,
        follow_type: QueryType,
        session: SessionContext
    ) -> float:
        """Adjust probability based on session context."""
        adjusted = base_prob

        # Check if this type was already asked in session
        asked_types = {
            QueryType(q["type"]) for q in session.query_history
            if "type" in q
        }

        if follow_type in asked_types:
            # Already asked this type - reduce probability
            adjusted *= 0.5

        # Boost if topic continues
        if session.current_topic:
            adjusted *= (1.0 + self.config.topic_continuity_weight * 0.3)

        return min(adjusted, 1.0)

    def _get_template(
        self,
        query_type: QueryType,
        topic: Optional[str]
    ) -> Optional[str]:
        """Get a template query for the type."""
        templates = self.TEMPLATES.get(query_type, [])
        if not templates:
            return None

        template = templates[0]  # Use first template

        if topic:
            return template.format(topic=topic)
        return template


# ============================================================================
# Anticipatory Fetcher
# ============================================================================

class AnticipatoryFetcher:
    """
    Pre-fetches memories for predicted follow-up queries.

    Uses predictions to warm the cache before the user asks.
    """

    def __init__(
        self,
        predictor: FollowUpPredictor,
        config: Optional[AnticipatoryConfig] = None
    ):
        """
        Initialize fetcher.

        Args:
            predictor: Follow-up predictor
            config: Configuration
        """
        self.predictor = predictor
        self.config = config or AnticipatoryConfig()

        # Reference to retriever (set during integration)
        self._retriever = None

        self.logger = logging.getLogger(f"{__name__}.AnticipatoryFetcher")

    def set_retriever(self, retriever: Any):
        """Set the retriever for pre-fetching."""
        self._retriever = retriever

    async def prefetch_for_predictions(
        self,
        predictions: List[PredictedQuery],
        session: SessionContext
    ) -> PrefetchResult:
        """
        Pre-fetch memories for predicted queries.

        Args:
            predictions: Predicted follow-up queries
            session: Session context

        Returns:
            PrefetchResult with fetched memory IDs
        """
        start_time = time.time()
        prefetched_ids: Set[str] = set()

        if not self.config.prefetch_enabled or not self._retriever:
            return PrefetchResult(
                predicted_queries=predictions,
                prefetched_memory_ids=set(),
                prefetch_time_ms=0.0,
                memories_fetched=0
            )

        remaining_budget = self.config.max_prefetch_total

        for pred in predictions:
            if remaining_budget <= 0:
                break

            if not pred.template:
                continue

            try:
                # Calculate how many to fetch based on probability
                k = min(
                    int(self.config.prefetch_k * pred.probability),
                    remaining_budget
                )
                if k < 1:
                    continue

                # Fetch memories for predicted query
                if hasattr(self._retriever, 'retrieve_ids'):
                    # Use lightweight ID-only retrieval if available
                    ids = await self._retriever.retrieve_ids(pred.template, k=k)
                else:
                    # Fall back to full retrieval
                    results = await self._retriever.retrieve(pred.template, k=k)
                    ids = [r[0].id if hasattr(r[0], 'id') else str(i) for i, r in enumerate(results)]

                prefetched_ids.update(ids)
                remaining_budget -= len(ids)

                self.logger.debug(
                    f"Prefetched {len(ids)} memories for: {pred.template[:50]}..."
                )

            except Exception as e:
                self.logger.warning(f"Prefetch failed for prediction: {e}")

        # Update session
        session.prefetched_memory_ids = prefetched_ids
        session.prefetch_timestamp = time.time()

        prefetch_time = (time.time() - start_time) * 1000

        return PrefetchResult(
            predicted_queries=predictions,
            prefetched_memory_ids=prefetched_ids,
            prefetch_time_ms=prefetch_time,
            memories_fetched=len(prefetched_ids)
        )


# ============================================================================
# Main Integration Class
# ============================================================================

class AnticipatoryRetrieval:
    """
    Main integration class for anticipatory retrieval.

    Usage:
        anticipatory = AnticipatoryRetrieval()

        # Create session
        session = anticipatory.create_session("user123")

        # Process query and predict follow-ups
        predictions = anticipatory.process_query(
            session, "What is Thompson Sampling?"
        )

        # Pre-fetch for predictions
        prefetch_result = await anticipatory.prefetch(session, predictions)

        # When next query comes, check if it was predicted
        hit_rate = session.check_prefetch_hit(actual_memory_ids)
    """

    def __init__(self, config: Optional[AnticipatoryConfig] = None):
        """
        Initialize anticipatory retrieval system.

        Args:
            config: Configuration
        """
        self.config = config or AnticipatoryConfig()

        # Components
        self.classifier = QueryTypeClassifier()
        self.pattern_tracker = QueryPatternTracker(self.config)
        self.predictor = FollowUpPredictor(
            self.pattern_tracker, self.classifier, self.config
        )
        self.fetcher = AnticipatoryFetcher(self.predictor, self.config)

        # Sessions
        self._sessions: Dict[str, SessionContext] = {}

        # Statistics
        self.total_queries = 0
        self.total_predictions = 0
        self.successful_predictions = 0

        self.logger = logging.getLogger(f"{__name__}.AnticipatoryRetrieval")

    def set_retriever(self, retriever: Any):
        """Set the retriever for pre-fetching."""
        self.fetcher.set_retriever(retriever)

    def create_session(self, session_id: str) -> SessionContext:
        """
        Create a new session.

        Args:
            session_id: Unique session identifier

        Returns:
            New session context
        """
        session = SessionContext(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Get existing session."""
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> SessionContext:
        """Get existing session or create new one."""
        if session_id not in self._sessions:
            return self.create_session(session_id)
        return self._sessions[session_id]

    def process_query(
        self,
        session: SessionContext,
        query: str
    ) -> List[PredictedQuery]:
        """
        Process a query and predict follow-ups.

        Args:
            session: Session context
            query: Query text

        Returns:
            List of predicted follow-up queries
        """
        self.total_queries += 1

        # Classify query
        query_type = self.classifier.classify(query)
        topic = self.classifier.extract_topic(query, query_type)

        # Record transition from last query
        if session.query_history:
            last_query = session.query_history[-1]
            last_type = QueryType(last_query["type"])
            self.pattern_tracker.record_transition(
                last_type, query_type, topic or session.current_topic
            )

        # Add to session history
        session.add_query(query, query_type, topic)

        # Predict follow-ups
        predictions = self.predictor.predict(query, session)
        self.total_predictions += len(predictions)

        return predictions

    async def prefetch(
        self,
        session: SessionContext,
        predictions: List[PredictedQuery]
    ) -> PrefetchResult:
        """
        Pre-fetch memories for predictions.

        Args:
            session: Session context
            predictions: Predicted queries

        Returns:
            Prefetch result
        """
        return await self.fetcher.prefetch_for_predictions(predictions, session)

    def record_prediction_hit(
        self,
        session: SessionContext,
        actual_query: str
    ) -> bool:
        """
        Check if actual query matches a prediction and record.

        Args:
            session: Session context
            actual_query: The actual query that came

        Returns:
            True if prediction was successful
        """
        actual_type = self.classifier.classify(actual_query)

        # Check if this type was predicted
        # (simplified check - could be more sophisticated)
        was_predicted = actual_type != QueryType.GENERAL

        if was_predicted:
            self.successful_predictions += 1

        return was_predicted

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        prediction_accuracy = (
            self.successful_predictions / self.total_predictions
            if self.total_predictions > 0 else 0.0
        )

        return {
            "total_queries": self.total_queries,
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "prediction_accuracy": prediction_accuracy,
            "active_sessions": len(self._sessions),
            "pattern_stats": self.pattern_tracker.get_statistics(),
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_anticipatory_retrieval(
    config: Optional[AnticipatoryConfig] = None,
    retriever: Any = None
) -> AnticipatoryRetrieval:
    """
    Create an anticipatory retrieval system.

    Args:
        config: Configuration
        retriever: Optional retriever for pre-fetching

    Returns:
        Configured system
    """
    system = AnticipatoryRetrieval(config)
    if retriever:
        system.set_retriever(retriever)
    return system


def create_session_context(session_id: str) -> SessionContext:
    """
    Create a session context.

    Args:
        session_id: Unique session identifier

    Returns:
        New session context
    """
    return SessionContext(session_id=session_id)

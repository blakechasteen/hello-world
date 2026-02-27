"""
Context-Aware Router

Integrates user context (session history, preferences) with query routing decisions.

Key Features:
- Session tracking for personalized routing
- User preference learning
- Context enrichment before routing
- Confidence adjustment based on context quality

Author: HoloLoom B2B Framework
Date: November 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio
from datetime import datetime


class RoutingStrategy(Enum):
    """Routing strategy options"""
    RULE_BASED = "rule_based"  # Rule-based classification
    ML_BASED = "ml_based"  # ML model predictions
    HYBRID = "hybrid"  # Combine rule-based + ML
    PERSONALIZED = "personalized"  # User-specific routing


@dataclass
class UserContext:
    """User context for personalized routing"""
    user_id: str
    session_id: str
    role: str = "user"
    department: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """Routing decision with context"""
    department_id: str
    confidence: float
    strategy_used: RoutingStrategy
    context_quality: float  # 0.0-1.0 (how good was the context?)
    reasoning: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ContextAwareRouter:
    """
    Context-aware router integrating ContextDepartment with QueryClassifier.

    Architecture:
        1. Receive query + user context
        2. Enrich context via ContextDepartment
        3. Classify query with enriched context
        4. Route to appropriate department
        5. Learn from outcome (feedback loop)

    Example:
        >>> router = ContextAwareRouter()
        >>> context = UserContext(user_id="user123", session_id="sess456")
        >>> decision = await router.route("What is Thompson Sampling?", context)
        >>> print(decision.department_id)  # "rag"
        >>> print(decision.confidence)  # 0.92
    """

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        enable_learning: bool = True,
        enable_ab_testing: bool = False,
    ):
        """
        Initialize context-aware router.

        Args:
            strategy: Routing strategy to use
            enable_learning: Enable learning from feedback
            enable_ab_testing: Enable A/B testing for strategies
        """
        self.strategy = strategy
        self.enable_learning = enable_learning
        self.enable_ab_testing = enable_ab_testing

        # User context storage (session history)
        self._user_contexts: Dict[str, UserContext] = {}

        # Routing history for learning
        self._routing_history: List[Dict[str, Any]] = []

        # Performance metrics
        self._metrics = {
            "total_routes": 0,
            "context_enrichments": 0,
            "personalized_routes": 0,
            "avg_confidence": 0.0,
            "avg_context_quality": 0.0,
        }

    async def route(
        self,
        query: str,
        user_context: Optional[UserContext] = None,
        enrich_context: bool = True,
    ) -> RoutingDecision:
        """
        Route query to appropriate department using context-aware logic.

        Args:
            query: User query
            user_context: Optional user context (session, history, preferences)
            enrich_context: Whether to enrich context via ContextDepartment

        Returns:
            RoutingDecision with department, confidence, reasoning
        """
        # Get or create user context
        if user_context is None:
            user_context = UserContext(
                user_id="anonymous",
                session_id=f"sess_{datetime.now().timestamp()}",
            )

        # Store/update user context
        self._user_contexts[user_context.user_id] = user_context

        # Enrich context if requested
        enriched_context = user_context
        context_quality = 0.5  # Default moderate quality

        if enrich_context:
            enriched_context, context_quality = await self._enrich_context(
                query, user_context
            )

        # Route based on strategy
        if self.strategy == RoutingStrategy.RULE_BASED:
            decision = await self._route_rule_based(query, enriched_context)
        elif self.strategy == RoutingStrategy.ML_BASED:
            decision = await self._route_ml_based(query, enriched_context)
        elif self.strategy == RoutingStrategy.HYBRID:
            decision = await self._route_hybrid(query, enriched_context)
        else:  # PERSONALIZED
            decision = await self._route_personalized(query, enriched_context)

        # Add context quality to decision
        decision.context_quality = context_quality

        # Update user history
        await self._update_history(user_context, query, decision)

        # Update metrics
        self._update_metrics(decision)

        # Store for learning
        if self.enable_learning:
            self._routing_history.append(
                {
                    "query": query,
                    "user_id": user_context.user_id,
                    "decision": decision,
                    "timestamp": datetime.now().timestamp(),
                }
            )

        return decision

    async def _enrich_context(
        self, query: str, user_context: UserContext
    ) -> tuple[UserContext, float]:
        """
        Enrich user context via ContextDepartment.

        Returns:
            (enriched_context, quality_score)
        """
        try:
            from hololoom.departments import get_department

            context_dept = get_department("context")

            # Create enrichment request
            request = {
                "task_type": "context_enrichment",
                "parameters": {
                    "query": query,
                    "user_context": {
                        "user_id": user_context.user_id,
                        "session_id": user_context.session_id,
                        "role": user_context.role,
                        "preferences": user_context.preferences,
                        "history": user_context.history[-10:],  # Last 10 queries
                    },
                    "enrich_with": ["user_patterns", "query_classification", "preferences"],
                },
            }

            # Execute enrichment
            response = await context_dept.process(request)

            # Extract enriched context
            enrichment = response.get("result", {})
            quality = response.get("confidence", 0.5)

            # Update user context
            if "user_patterns" in enrichment:
                user_context.metadata["patterns"] = enrichment["user_patterns"]

            if "query_classification" in enrichment:
                user_context.metadata["query_type"] = enrichment["query_classification"]

            if "preferences" in enrichment:
                user_context.preferences.update(enrichment["preferences"])

            self._metrics["context_enrichments"] += 1

            return user_context, quality

        except Exception as e:
            # Graceful degradation if ContextDepartment unavailable
            return user_context, 0.5

    async def _route_rule_based(
        self, query: str, context: UserContext
    ) -> RoutingDecision:
        """Route using rule-based classification"""
        from hololoom.routing.query_classifier import QueryClassifier, QueryComplexity

        classifier = QueryClassifier()
        result = classifier.classify(query)

        # Map complexity to department
        complexity_to_dept = {
            QueryComplexity.TRIVIAL: "context",
            QueryComplexity.SIMPLE: "rag",
            QueryComplexity.MODERATE: "rag",
            QueryComplexity.COMPLEX: "planning",
            QueryComplexity.RESEARCH: "orchestration",
        }

        department_id = complexity_to_dept.get(result.complexity, "rag")

        return RoutingDecision(
            department_id=department_id,
            confidence=result.confidence,
            strategy_used=RoutingStrategy.RULE_BASED,
            context_quality=0.0,  # Will be set by caller
            reasoning=f"Rule-based: {result.complexity.value} complexity → {department_id}",
            alternatives=[
                {"department": dept, "reason": complexity.value}
                for complexity, dept in complexity_to_dept.items()
                if dept != department_id
            ],
        )

    async def _route_ml_based(
        self, query: str, context: UserContext
    ) -> RoutingDecision:
        """Route using ML model predictions"""
        # Placeholder for ML-based routing (Task 5)
        # Will be implemented in hololoom/routing/ml/
        return await self._route_rule_based(query, context)

    async def _route_hybrid(
        self, query: str, context: UserContext
    ) -> RoutingDecision:
        """Route using hybrid approach (rule-based + ML)"""
        # Get both predictions
        rule_decision = await self._route_rule_based(query, context)
        ml_decision = await self._route_ml_based(query, context)

        # Combine (weighted average)
        if rule_decision.confidence > ml_decision.confidence:
            final_decision = rule_decision
            final_decision.reasoning = f"Hybrid: rule-based (conf={rule_decision.confidence:.2f}) > ML (conf={ml_decision.confidence:.2f})"
        else:
            final_decision = ml_decision
            final_decision.reasoning = f"Hybrid: ML (conf={ml_decision.confidence:.2f}) > rule-based (conf={rule_decision.confidence:.2f})"

        final_decision.strategy_used = RoutingStrategy.HYBRID
        return final_decision

    async def _route_personalized(
        self, query: str, context: UserContext
    ) -> RoutingDecision:
        """Route using personalized user preferences"""
        # Get hybrid decision as base
        base_decision = await self._route_hybrid(query, context)

        # Adjust based on user preferences
        user_prefs = context.preferences.get("preferred_departments", {})

        if base_decision.department_id in user_prefs:
            # Boost confidence if user prefers this department
            boost = user_prefs[base_decision.department_id]
            base_decision.confidence = min(
                1.0, base_decision.confidence * (1.0 + boost)
            )
            base_decision.reasoning += f" | Personalized boost: +{boost:.1%}"

        base_decision.strategy_used = RoutingStrategy.PERSONALIZED
        self._metrics["personalized_routes"] += 1

        return base_decision

    async def _update_history(
        self, context: UserContext, query: str, decision: RoutingDecision
    ):
        """Update user history with query and routing decision"""
        context.history.append(
            {
                "query": query,
                "department": decision.department_id,
                "confidence": decision.confidence,
                "timestamp": datetime.now().timestamp(),
            }
        )

        # Keep only last 50 queries
        if len(context.history) > 50:
            context.history = context.history[-50:]

    def _update_metrics(self, decision: RoutingDecision):
        """Update routing metrics"""
        self._metrics["total_routes"] += 1

        # Running average
        n = self._metrics["total_routes"]
        old_avg_conf = self._metrics["avg_confidence"]
        self._metrics["avg_confidence"] = (
            old_avg_conf * (n - 1) + decision.confidence
        ) / n

        old_avg_quality = self._metrics["avg_context_quality"]
        self._metrics["avg_context_quality"] = (
            old_avg_quality * (n - 1) + decision.context_quality
        ) / n

    async def learn_from_feedback(
        self,
        user_id: str,
        query: str,
        actual_department: str,
        outcome: str,  # "success" | "failure"
        confidence: float,
    ):
        """
        Learn from routing feedback to improve future decisions.

        Args:
            user_id: User who made the query
            query: Original query
            actual_department: Department that was used
            outcome: Whether it was successful
            confidence: Confidence of the outcome
        """
        if not self.enable_learning:
            return

        # Update user preferences
        if user_id in self._user_contexts:
            context = self._user_contexts[user_id]

            if "preferred_departments" not in context.preferences:
                context.preferences["preferred_departments"] = {}

            # Update preference based on outcome
            if outcome == "success":
                # Increase preference for successful department
                current = context.preferences["preferred_departments"].get(
                    actual_department, 0.0
                )
                context.preferences["preferred_departments"][actual_department] = min(
                    0.5, current + 0.05  # Cap at +50% boost
                )
            else:
                # Decrease preference for failed department
                current = context.preferences["preferred_departments"].get(
                    actual_department, 0.0
                )
                context.preferences["preferred_departments"][actual_department] = max(
                    -0.2, current - 0.05  # Cap at -20% penalty
                )

    def get_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics"""
        return self._metrics.copy()

    def get_user_context(self, user_id: str) -> Optional[UserContext]:
        """Get user context by user_id"""
        return self._user_contexts.get(user_id)

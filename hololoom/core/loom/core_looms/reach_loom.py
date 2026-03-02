"""
REACH Loom - The Artist
========================

Lateral association, creativity-grounded. Makes unexpected connections.

Cognitive Style:
- Low recall (k=5) - avoid the obvious
- High temperature (0.9)
- Follow weak ties (prefer distant connections)
- Multiple association hops (3+)

Philosophy:
"What if we connected this to that? What lies at the edges?
The Artist sees what others overlook by traveling the weak ties."

Date: December 2025
Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import random

from hololoom.loom.base_loom import BaseLoom
from hololoom.loom.protocol import REACH, DiscoveryInsight, PatternInsight
from hololoom.fabric.fabric import Fabric
from hololoom.protocols.department import (
    DepartmentRequest,
    DepartmentResponse,
    ConfidenceMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class Association:
    """A creative association between concepts."""
    source: str
    target: str
    strength: float  # 0-1, lower = weaker tie
    path: List[str]  # How we got here
    novelty: float  # How novel this connection is


@dataclass
class CreativeInsight:
    """A creative insight combining concepts."""
    concepts: List[str]
    combination: str
    novelty_score: float
    plausibility_score: float


class ReachLoom(BaseLoom):
    """
    The Artist - Lateral association, creativity-grounded.

    Makes unexpected connections. Follows weak ties. Generates novel
    combinations. Admits it can't verify accuracy.

    Attributes:
        retrieval_k: Number of memories to retrieve (default: 5, low)
        temperature: Creativity level (default: 0.9, high)
        association_hops: How far to follow associations (default: 3)
        prefer_weak_ties: Prefer distant connections (default: True)
        novelty_threshold: Minimum novelty for insights (default: 0.5)
    """

    def __init__(
        self,
        retrieval_k: int = 5,
        temperature: float = 0.9,
        association_hops: int = 3,
        prefer_weak_ties: bool = True,
        novelty_threshold: float = 0.5,
        memory_backend: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize the Reach Loom.

        Args:
            retrieval_k: Number of memories to retrieve (low to avoid obvious)
            temperature: Creativity level (high for novel combinations)
            association_hops: How many hops to follow associations
            prefer_weak_ties: Whether to prefer distant connections
            novelty_threshold: Minimum novelty score for insights
            memory_backend: Optional memory backend
            **kwargs: Arguments passed to BaseLoom
        """
        super().__init__(**kwargs)

        # Cognitive style parameters
        self.retrieval_k = retrieval_k
        self.temperature = temperature
        self.association_hops = association_hops
        self.prefer_weak_ties = prefer_weak_ties
        self.novelty_threshold = novelty_threshold

        # Memory backend
        self._memory = memory_backend

        # Track creative insights for learning
        self._recent_insights: List[CreativeInsight] = []

    @property
    def perspective(self) -> str:
        """The Artist's perspective."""
        return REACH

    @property
    def blind_spots(self) -> List[str]:
        """What the Artist cannot see."""
        return [
            "factual accuracy",
            "logical consistency",
            "established convention",
            "risk assessment",
            "practical constraints",
            "verification of claims",
        ]

    def admits_ignorance(self, query: str) -> List[str]:
        """
        What the Artist knows it doesn't know about this query.

        Args:
            query: The query to assess

        Returns:
            List of specific knowledge gaps
        """
        # The Artist always admits uncertainty about facts
        ignorance = [
            "Whether these connections are factually correct",
            "Whether this is practically feasible",
            "Whether established experts would agree",
        ]

        # Check for precision-requiring keywords
        precision_words = ['exact', 'precise', 'specific', 'correct', 'accurate']
        if any(word in query.lower() for word in precision_words):
            ignorance.append("Cannot guarantee precision - creativity over accuracy")

        # Check for safety/risk keywords
        safety_words = ['safe', 'risk', 'danger', 'careful', 'important']
        if any(word in query.lower() for word in safety_words):
            ignorance.append("Cannot assess safety - creativity ignores constraints")

        return ignorance

    def falsifiable_by(self, claim: str) -> List[str]:
        """
        What evidence would change the Artist's claim.

        Args:
            claim: The claim to assess

        Returns:
            List of falsifiers
        """
        return [
            "Fact-checking by domain experts",
            "Testing the creative suggestion in practice",
            "Finding logical contradictions in the combination",
            "User feedback that the connection is unhelpful",
            "Evidence that the association path is broken",
        ]

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute creative association logic.

        Args:
            request: Department request with query

        Returns:
            Response with creative insights
        """
        start_time = datetime.now()
        query = request.query

        # Step 1: Extract seed concepts
        seeds = await self._extract_concepts(query)

        # Step 2: Follow weak ties to find distant associations
        associations = await self._explore_weak_ties(seeds)

        # Step 3: Generate novel combinations
        insights = self._generate_combinations(seeds, associations)
        self._recent_insights = insights

        # Step 4: Format creative response
        response = self._format_creative_response(insights, associations)

        # Step 5: Compute confidence (always moderate for creativity)
        confidence = self._compute_creative_confidence(insights)

        # Step 6: Compute epistemic confidence (low - we're guessing)
        epistemic = self._compute_epistemic_confidence(insights)

        # Calculate duration
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        # Update metrics
        self._update_metrics(confidence, duration_ms)

        # Log insights for dreaming
        for insight in insights[:2]:  # Top 2
            if insight.novelty_score > 0.7:
                self.log_discovery_insight({
                    "concepts": insight.concepts,
                    "combination": insight.combination,
                    "novelty": insight.novelty_score,
                }, confidence=insight.plausibility_score)

        # Create fabric
        fabric = self._create_fabric(
            query=query,
            response=response,
            tool_used="reach",
            confidence=confidence,
            duration_ms=duration_ms,
            epistemic_confidence=epistemic,
            admits_ignorance=self.admits_ignorance(query),
            falsifiable_by=self.falsifiable_by(response),
            metadata={
                "seeds": seeds,
                "associations_found": len(associations),
                "insights_generated": len(insights),
                "avg_novelty": sum(i.novelty_score for i in insights) / max(len(insights), 1),
            },
        )

        return DepartmentResponse(
            result=response,
            confidence=ConfidenceMetadata(
                score=confidence,
                source="reach_loom",
                reasoning=f"Generated {len(insights)} creative insights",
            ),
            metadata={
                "perspective": self.perspective,
                "fabric": fabric,
                "insights": insights,
                "associations": associations,
            },
        )

    async def _extract_concepts(self, query: str) -> List[str]:
        """
        Extract seed concepts from query.

        Args:
            query: Query to extract from

        Returns:
            List of seed concepts
        """
        # Simple extraction: important words (not stopwords)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'to', 'of',
            'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'about', 'what', 'which', 'who', 'how', 'why', 'when', 'where',
            'this', 'that', 'these', 'those', 'it', 'its', 'and', 'or',
        }

        words = query.lower().split()
        concepts = [
            w.strip('.,?!') for w in words
            if w.strip('.,?!') not in stopwords and len(w) > 2
        ]

        return concepts[:5]  # Limit to 5 seeds

    async def _explore_weak_ties(
        self,
        seeds: List[str]
    ) -> List[Association]:
        """
        Explore weak ties from seed concepts.

        Args:
            seeds: Seed concepts to start from

        Returns:
            List of associations found
        """
        associations = []
        visited: Set[str] = set(seeds)

        for seed in seeds:
            # Simulate exploring weak ties (would use memory graph in production)
            current = seed
            path = [seed]

            for hop in range(self.association_hops):
                # Generate a "distant" association
                related = self._generate_distant_association(current, visited)

                if related:
                    visited.add(related)
                    path.append(related)

                    # Weaker ties = more hops
                    strength = 1.0 / (hop + 2)

                    # Novelty increases with distance
                    novelty = min(hop / self.association_hops + 0.3, 1.0)

                    associations.append(Association(
                        source=seed,
                        target=related,
                        strength=strength,
                        path=path.copy(),
                        novelty=novelty,
                    ))

                    current = related
                else:
                    break

        # Sort by novelty (prefer weak ties)
        if self.prefer_weak_ties:
            associations.sort(key=lambda a: a.novelty, reverse=True)

        return associations

    def _generate_distant_association(
        self,
        concept: str,
        visited: Set[str]
    ) -> Optional[str]:
        """
        Generate a distant but plausible association.

        Args:
            concept: Current concept
            visited: Already visited concepts

        Returns:
            A distant association or None
        """
        # Domain associations (would use embeddings in production)
        domain_associations = {
            'learning': ['neuroplasticity', 'curiosity', 'iteration', 'feedback'],
            'decision': ['tradeoff', 'uncertainty', 'exploration', 'commitment'],
            'algorithm': ['nature', 'evolution', 'emergence', 'swarm'],
            'memory': ['story', 'emotion', 'association', 'forgetting'],
            'pattern': ['music', 'architecture', 'nature', 'language'],
            'system': ['ecology', 'city', 'body', 'network'],
            'problem': ['puzzle', 'game', 'adventure', 'mystery'],
            'data': ['story', 'archaeology', 'forensics', 'art'],
            'code': ['poetry', 'music', 'recipe', 'spell'],
            'model': ['metaphor', 'sculpture', 'map', 'lens'],
        }

        # Find associations for concept
        candidates = []
        for key, values in domain_associations.items():
            if key in concept.lower() or concept.lower() in key:
                candidates.extend(values)
            for value in values:
                if concept.lower() in value or value in concept.lower():
                    candidates.extend(domain_associations.get(key, []))

        # Random distant associations if no direct ones
        if not candidates:
            all_associations = [v for vals in domain_associations.values() for v in vals]
            candidates = random.sample(all_associations, min(3, len(all_associations)))

        # Filter visited and select
        candidates = [c for c in candidates if c not in visited]

        if candidates:
            return random.choice(candidates)
        return None

    def _generate_combinations(
        self,
        seeds: List[str],
        associations: List[Association]
    ) -> List[CreativeInsight]:
        """
        Generate novel combinations from seeds and associations.

        Args:
            seeds: Original seed concepts
            associations: Discovered associations

        Returns:
            List of creative insights
        """
        insights = []

        # Combine seeds with distant associations
        for assoc in associations:
            if assoc.novelty >= self.novelty_threshold:
                combination = self._create_combination(
                    assoc.source,
                    assoc.target,
                    assoc.path
                )

                if combination:
                    insights.append(CreativeInsight(
                        concepts=[assoc.source, assoc.target],
                        combination=combination,
                        novelty_score=assoc.novelty,
                        plausibility_score=assoc.strength * 0.5 + 0.3,  # Low but not zero
                    ))

        # Cross-seed combinations
        if len(seeds) >= 2:
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    combination = self._create_combination(seeds[i], seeds[j], [])
                    insights.append(CreativeInsight(
                        concepts=[seeds[i], seeds[j]],
                        combination=combination,
                        novelty_score=0.6,  # Moderate novelty for direct combinations
                        plausibility_score=0.5,
                    ))

        # Sort by novelty
        insights.sort(key=lambda i: i.novelty_score, reverse=True)

        return insights[:5]  # Top 5

    def _create_combination(
        self,
        concept1: str,
        concept2: str,
        path: List[str]
    ) -> str:
        """
        Create a creative combination statement.

        Args:
            concept1: First concept
            concept2: Second concept
            path: Association path

        Returns:
            Creative combination text
        """
        templates = [
            f"What if {concept1} worked like {concept2}?",
            f"Consider applying principles of {concept2} to {concept1}.",
            f"The connection between {concept1} and {concept2} suggests...",
            f"Like {concept2}, {concept1} might benefit from...",
            f"Viewing {concept1} through the lens of {concept2}...",
        ]

        if path:
            path_str = " -> ".join(path)
            templates.append(f"Following the path {path_str}, we see...")

        return random.choice(templates)

    def _format_creative_response(
        self,
        insights: List[CreativeInsight],
        associations: List[Association]
    ) -> str:
        """
        Format creative insights as response.

        Args:
            insights: Generated insights
            associations: Found associations

        Returns:
            Formatted response text
        """
        parts = ["Creative Exploration:\n"]

        if not insights:
            parts.append("No novel connections found above the novelty threshold.")
            return "\n".join(parts)

        parts.append("Novel Connections:")
        for i, insight in enumerate(insights[:3], 1):
            parts.append(
                f"\n{i}. {insight.combination}\n"
                f"   Concepts: {', '.join(insight.concepts)}\n"
                f"   Novelty: {insight.novelty_score:.2f} | "
                f"Plausibility: {insight.plausibility_score:.2f}"
            )

        parts.append("\n\n[NOTE: These are creative suggestions, not verified facts.]")

        return "\n".join(parts)

    def _compute_creative_confidence(
        self,
        insights: List[CreativeInsight]
    ) -> float:
        """
        Compute confidence for creative output.

        Creativity is inherently uncertain, so confidence is moderate.

        Args:
            insights: Generated insights

        Returns:
            Confidence score (0-1)
        """
        if not insights:
            return 0.3

        # Average plausibility with cap at 0.7 (creativity is uncertain)
        avg_plausibility = sum(i.plausibility_score for i in insights) / len(insights)
        return min(avg_plausibility + 0.2, 0.7)

    def _compute_epistemic_confidence(
        self,
        insights: List[CreativeInsight]
    ) -> float:
        """
        Compute epistemic confidence.

        Creative output has low epistemic confidence by design.

        Args:
            insights: Generated insights

        Returns:
            Epistemic confidence (0-1)
        """
        # Artist always has low epistemic confidence - that's the point
        base = 0.4

        # Slightly higher if insights are more plausible
        if insights:
            avg_plausibility = sum(i.plausibility_score for i in insights) / len(insights)
            base += avg_plausibility * 0.2

        return min(base, 0.5)  # Cap at 0.5


__all__ = ['ReachLoom']

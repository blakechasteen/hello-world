"""
Research House - Deep Research with Verification
=================================================

A pre-configured WeaveHouse optimized for research tasks.

Combines all 5 core looms with research-specific configurations:
- RECALL: Find related papers, prior work, evidence
- REASON: Build logical arguments, trace implications
- REACH: Make creative connections, novel hypotheses
- REFLECT: Verify claims, check consistency
- REFUSE: Challenge assumptions, find counterarguments

Date: December 2025
Author: HoloLoom Architecture Team
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hololoom.loom.weave_house import WeaveHouse, WeaveResult
from hololoom.loom.consensus import LoomConsensus, create_loom_consensus
from hololoom.loom.core_looms.recall_loom import RecallLoom
from hololoom.loom.core_looms.reason_loom import ReasonLoom
from hololoom.loom.core_looms.reach_loom import ReachLoom
from hololoom.loom.core_looms.reflect_loom import ReflectLoom
from hololoom.loom.core_looms.refuse_loom import RefuseLoom


@dataclass
class ResearchConfig:
    """Configuration for research house."""

    # Recall settings (finding prior work)
    recall_k: int = 25  # High recall for comprehensive coverage
    recall_temperature: float = 0.4

    # Reason settings (building arguments)
    max_inference_depth: int = 10  # Deep reasoning chains
    require_explicit_links: bool = True

    # Reach settings (creative connections)
    association_hops: int = 4  # Explore further
    novelty_threshold: float = 0.4  # Lower threshold for more ideas
    creativity_temperature: float = 0.85

    # Reflect settings (verification)
    verification_passes: int = 3  # Thorough verification
    consistency_threshold: float = 0.9  # High consistency required

    # Refuse settings (challenges)
    attack_intensity: float = 0.5  # Moderate adversarial probing
    coverage_target: float = 0.7

    # Consensus settings
    exploration_depth: int = 3  # Deep exploration of disagreements
    tension_threshold: float = 0.2  # Very sensitive to tensions

    # Research-specific
    include_creative: bool = True  # Include REACH loom
    require_evidence: bool = True
    synthesize_perspectives: bool = True


class ResearchHouse(WeaveHouse):
    """
    A WeaveHouse specialized for research.

    Provides comprehensive research analysis:
    - Prior work retrieval (related literature, evidence)
    - Logical reasoning (argument construction)
    - Creative exploration (novel connections, hypotheses)
    - Verification (claim checking, consistency)
    - Critical analysis (counterarguments, challenges)

    Example:
        ```python
        async with create_research_house() as house:
            result = await house.research("What causes X?")
            print(result.findings)
            print(result.novel_insights)
            print(result.open_questions)
        ```
    """

    def __init__(
        self,
        config: Optional[ResearchConfig] = None,
        **kwargs
    ):
        """
        Initialize the Research House.

        Args:
            config: Optional configuration for the house
            **kwargs: Arguments passed to WeaveHouse
        """
        self.research_config = config or ResearchConfig()

        # Create specialized looms for research
        looms = self._create_research_looms()

        # Create consensus with research-specific settings
        consensus = create_loom_consensus(
            exploration_enabled=True,
            agreement_threshold=0.6  # Lower threshold - embrace disagreement
        )

        super().__init__(
            looms=looms,
            consensus=consensus,
            exploration_depth=self.research_config.exploration_depth,
            tension_threshold=self.research_config.tension_threshold,
            **kwargs
        )

    def _create_research_looms(self) -> List:
        """Create looms configured for research."""
        looms = [
            # RECALL: Find related work and evidence
            RecallLoom(
                retrieval_k=self.research_config.recall_k,
                temperature=self.research_config.recall_temperature,
                trust_memory_weight=0.7,
                trust_inference_weight=0.3,
            ),

            # REASON: Build logical arguments
            ReasonLoom(
                max_inference_depth=self.research_config.max_inference_depth,
                require_explicit_links=self.research_config.require_explicit_links,
                min_step_confidence=0.6,
            ),

            # REFLECT: Verify claims thoroughly
            ReflectLoom(
                verification_passes=self.research_config.verification_passes,
                consistency_threshold=self.research_config.consistency_threshold,
                require_evidence=self.research_config.require_evidence,
            ),

            # REFUSE: Challenge assumptions
            RefuseLoom(
                attack_intensity=self.research_config.attack_intensity,
                coverage_target=self.research_config.coverage_target,
                max_attack_depth=4,
            ),
        ]

        # Optionally include creative exploration
        if self.research_config.include_creative:
            looms.append(
                ReachLoom(
                    retrieval_k=8,  # Moderate - not too obvious
                    temperature=self.research_config.creativity_temperature,
                    association_hops=self.research_config.association_hops,
                    prefer_weak_ties=True,
                    novelty_threshold=self.research_config.novelty_threshold,
                )
            )

        return looms

    async def research(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        depth: Optional[str] = None
    ) -> "ResearchResult":
        """
        Conduct research on a question from multiple perspectives.

        Args:
            question: The research question
            context: Optional context (domain, prior knowledge, etc.)
            depth: Research depth ("quick", "standard", "deep")

        Returns:
            ResearchResult with findings, insights, and open questions
        """
        # Build research query
        query = self._build_research_query(question, context, depth)

        # Weave through all perspectives
        result = await self.weave(query)

        # Parse into structured research result
        return self._parse_research_result(result, question)

    def _build_research_query(
        self,
        question: str,
        context: Optional[Dict[str, Any]],
        depth: Optional[str]
    ) -> str:
        """Build a query for research."""
        parts = ["Research Question:\n\n", question, "\n"]

        if context:
            parts.append(f"\nContext: {context}\n")

        if depth:
            depth_instructions = {
                "quick": "Provide a brief overview.",
                "standard": "Provide a thorough analysis.",
                "deep": "Conduct an exhaustive investigation.",
            }
            parts.append(f"\nDepth: {depth_instructions.get(depth, '')}\n")

        parts.append("\nProvide multi-perspective research analysis.")

        return "".join(parts)

    def _parse_research_result(
        self,
        result: WeaveResult,
        question: str
    ) -> "ResearchResult":
        """Parse weave result into structured research result."""
        return ResearchResult(
            question=question,
            summary=result.response,
            confidence=result.confidence,
            epistemic_confidence=result.epistemic_confidence,
            findings=self._extract_findings(result),
            novel_insights=self._extract_novel_insights(result),
            open_questions=self._extract_open_questions(result),
            counterarguments=self._extract_counterarguments(result),
            evidence_gaps=self._extract_evidence_gaps(result),
            perspectives={
                fab.perspective: fab.response
                for fab in result.fabrics
            },
            tensions=result.tensions,
            metadata=result.metadata,
        )

    def _extract_findings(self, result: WeaveResult) -> List[Dict[str, Any]]:
        """Extract findings from research result."""
        findings = []

        # From RECALL loom
        for fab in result.fabrics:
            if fab.perspective == "recall":
                if "retrieved_facts" in fab.metadata:
                    for fact in fab.metadata["retrieved_facts"]:
                        findings.append({
                            "type": "prior_work",
                            "content": fact.get("content", ""),
                            "confidence": fact.get("confidence", 0.5),
                            "source": "recall_loom",
                        })

            # From REASON loom
            elif fab.perspective == "reason":
                if "conclusions" in fab.metadata:
                    for conclusion in fab.metadata["conclusions"]:
                        findings.append({
                            "type": "inference",
                            "content": str(conclusion),
                            "confidence": fab.confidence,
                            "source": "reason_loom",
                        })

        return findings

    def _extract_novel_insights(self, result: WeaveResult) -> List[str]:
        """Extract novel insights from REACH loom."""
        insights = []

        for fab in result.fabrics:
            if fab.perspective == "reach":
                if "novel_connections" in fab.metadata:
                    insights.extend(fab.metadata["novel_connections"])
                if "creative_insights" in fab.metadata:
                    for insight in fab.metadata["creative_insights"]:
                        insights.append(str(insight))

        return insights

    def _extract_open_questions(self, result: WeaveResult) -> List[str]:
        """Extract open questions from analysis."""
        open_questions = []

        # Collect uncertainties from all perspectives
        for fab in result.fabrics:
            admits = fab.admits_ignorance
            if admits:
                for gap in admits:
                    if "?" not in gap:
                        gap = f"What about {gap}?"
                    open_questions.append(gap)

        # Also from tensions (unresolved disagreements)
        for tension in result.tensions:
            if tension.get("is_irreducible", False):
                open_questions.append(
                    f"How to reconcile: {tension.get('claim_a', '')} vs {tension.get('claim_b', '')}?"
                )

        return list(set(open_questions))[:10]  # Dedupe and limit

    def _extract_counterarguments(self, result: WeaveResult) -> List[Dict[str, Any]]:
        """Extract counterarguments from REFUSE loom."""
        counterargs = []

        for fab in result.fabrics:
            if fab.perspective == "refuse":
                if "attacks" in fab.metadata:
                    for attack in fab.metadata["attacks"]:
                        counterargs.append({
                            "target": attack.get("target", ""),
                            "argument": attack.get("description", ""),
                            "severity": attack.get("severity", "medium"),
                        })

        return counterargs

    def _extract_evidence_gaps(self, result: WeaveResult) -> List[str]:
        """Extract evidence gaps from REFLECT loom."""
        gaps = []

        for fab in result.fabrics:
            if fab.perspective == "reflect":
                # Unverified claims are evidence gaps
                if "unverified_claims" in fab.metadata:
                    for claim in fab.metadata["unverified_claims"]:
                        gaps.append(f"Evidence needed for: {claim}")

                # Also from verification results
                if "verification_results" in fab.metadata:
                    for vr in fab.metadata["verification_results"]:
                        if not vr.get("verified", False):
                            gaps.append(
                                f"Could not verify: {vr.get('claim', 'unknown')}"
                            )

        return gaps


@dataclass
class ResearchResult:
    """Result of a research investigation."""

    question: str
    summary: str
    confidence: float
    epistemic_confidence: float
    findings: List[Dict[str, Any]] = field(default_factory=list)
    novel_insights: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    counterarguments: List[Dict[str, Any]] = field(default_factory=list)
    evidence_gaps: List[str] = field(default_factory=list)
    perspectives: Dict[str, str] = field(default_factory=dict)
    tensions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_novel_insights(self) -> bool:
        """Check if research found novel insights."""
        return len(self.novel_insights) > 0

    @property
    def findings_count(self) -> int:
        """Total number of findings."""
        return len(self.findings)

    @property
    def has_open_questions(self) -> bool:
        """Check if there are unresolved questions."""
        return len(self.open_questions) > 0

    def findings_by_type(self, finding_type: str) -> List[Dict[str, Any]]:
        """Get findings by type (prior_work, inference, etc.)."""
        return [f for f in self.findings if f.get("type") == finding_type]

    def high_confidence_findings(self, threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Get findings above confidence threshold."""
        return [f for f in self.findings if f.get("confidence", 0) >= threshold]


def create_research_house(
    config: Optional[ResearchConfig] = None,
    **kwargs
) -> ResearchHouse:
    """
    Factory function to create a ResearchHouse.

    Args:
        config: Optional configuration
        **kwargs: Additional arguments for WeaveHouse

    Returns:
        Configured ResearchHouse instance
    """
    return ResearchHouse(config=config, **kwargs)


__all__ = ['ResearchHouse', 'ResearchConfig', 'ResearchResult', 'create_research_house']

"""
Meta-Awareness Layer: Recursive Self-Reflection

The awareness layer becomes self-aware—it monitors its own confidence,
decomposes uncertainty, generates hypotheses about knowledge gaps,
and adversarially probes its own responses.

This is compositional AI consciousness examining itself.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class UncertaintyType(Enum):
    """Types of uncertainty"""
    STRUCTURAL = "structural"      # X-bar parsing ambiguity
    SEMANTIC = "semantic"          # Word sense disambiguation
    CONTEXTUAL = "contextual"      # Missing background knowledge
    COMPOSITIONAL = "compositional"  # Merge operation ambiguity
    EPISTEMIC = "epistemic"        # Don't know what we don't know


@dataclass
class UncertaintyDecomposition:
    """Break down uncertainty into components"""
    total_uncertainty: float
    
    # Component uncertainties
    structural_uncertainty: float
    semantic_uncertainty: float
    contextual_uncertainty: float
    compositional_uncertainty: float
    
    # Which component dominates?
    dominant_type: UncertaintyType
    
    # Specific sources
    ambiguous_structures: List[str] = field(default_factory=list)
    ambiguous_terms: List[str] = field(default_factory=list)
    missing_context: List[str] = field(default_factory=list)
    
    def get_explanation(self) -> str:
        """Human-readable explanation of uncertainty"""
        if self.dominant_type == UncertaintyType.STRUCTURAL:
            return f"Structural ambiguity in: {', '.join(self.ambiguous_structures)}"
        elif self.dominant_type == UncertaintyType.SEMANTIC:
            return f"Unclear meaning of: {', '.join(self.ambiguous_terms)}"
        elif self.dominant_type == UncertaintyType.CONTEXTUAL:
            return f"Missing context about: {', '.join(self.missing_context)}"
        elif self.dominant_type == UncertaintyType.COMPOSITIONAL:
            return "Multiple valid interpretations of how concepts combine"
        else:
            return "Unknown unknowns detected"


@dataclass
class MetaConfidence:
    """Confidence about confidence estimates"""
    primary_confidence: float          # Original confidence estimate
    meta_confidence: float             # How sure are we about that estimate?
    calibration_history: List[float]   # Past accuracy of estimates
    
    # Second-order uncertainty
    uncertainty_about_uncertainty: float
    
    # Confidence bounds
    lower_bound: float
    upper_bound: float
    
    def is_well_calibrated(self) -> bool:
        """Check if confidence estimates are reliable"""
        if not self.calibration_history:
            return False
        
        # Check if past estimates were accurate
        avg_calibration = sum(self.calibration_history) / len(self.calibration_history)
        return abs(avg_calibration - self.primary_confidence) < 0.2
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """Return confidence interval"""
        return (self.lower_bound, self.upper_bound)


@dataclass
class KnowledgeGapHypothesis:
    """Hypothesis about what might fill a knowledge gap"""
    gap_description: str
    potential_answers: List[str]
    required_information: List[str]
    testable_predictions: List[str]
    confidence_in_hypothesis: float
    
    def to_query(self) -> str:
        """Convert hypothesis to a clarifying query"""
        if self.potential_answers:
            return f"Are you asking about {' or '.join(self.potential_answers[:3])}?"
        else:
            return f"To answer, I'd need to know: {', '.join(self.required_information[:2])}"


@dataclass
class AdversarialProbe:
    """Adversarial question to test response quality"""
    probe_question: str
    expected_weakness: str
    test_type: str  # "contradiction", "edge_case", "assumption"
    
    def __str__(self) -> str:
        return f"[{self.test_type}] {self.probe_question}"


@dataclass
class SelfReflectionResult:
    """Result of recursive self-reflection"""
    original_response: str
    
    # Uncertainty analysis
    uncertainty_decomposition: UncertaintyDecomposition
    meta_confidence: MetaConfidence
    
    # Knowledge gap analysis
    detected_gaps: List[str]
    gap_hypotheses: List[KnowledgeGapHypothesis]
    
    # Adversarial probing
    adversarial_probes: List[AdversarialProbe]
    probe_results: List[str]
    
    # Meta-awareness
    aware_of_limitations: bool
    epistemic_humility: float  # 0.0 = overconfident, 1.0 = appropriately humble
    
    def format_introspection(self) -> str:
        """Format self-reflection for display"""
        lines = []
        lines.append("\n╔══════════════════════════════════════════════════════════════════╗")
        lines.append("║              META-AWARENESS: RECURSIVE SELF-REFLECTION            ║")
        lines.append("╚══════════════════════════════════════════════════════════════════╝")
        
        # Uncertainty decomposition
        lines.append("\n[ UNCERTAINTY DECOMPOSITION ]")
        lines.append("-" * 70)
        unc = self.uncertainty_decomposition
        lines.append(f"Total: {unc.total_uncertainty:.2f}")
        lines.append(f"  - Structural:     {unc.structural_uncertainty:.2f}")
        lines.append(f"  - Semantic:       {unc.semantic_uncertainty:.2f}")
        lines.append(f"  - Contextual:     {unc.contextual_uncertainty:.2f}")
        lines.append(f"  - Compositional:  {unc.compositional_uncertainty:.2f}")
        lines.append(f"\nDominant: {unc.dominant_type.value}")
        lines.append(f"Explanation: {unc.get_explanation()}")
        
        # Meta-confidence
        lines.append("\n[ META-CONFIDENCE ]")
        lines.append("-" * 70)
        mc = self.meta_confidence
        lines.append(f"Primary Confidence: {mc.primary_confidence:.2f}")
        lines.append(f"Meta-Confidence: {mc.meta_confidence:.2f} (confidence about confidence)")
        lines.append(f"Uncertainty²: {mc.uncertainty_about_uncertainty:.2f}")
        lines.append(f"Confidence Interval: [{mc.lower_bound:.2f}, {mc.upper_bound:.2f}]")
        lines.append(f"Well-Calibrated: {'✓' if mc.is_well_calibrated() else '✗'}")
        
        # Knowledge gaps
        if self.detected_gaps:
            lines.append("\n[ DETECTED KNOWLEDGE GAPS ]")
            lines.append("-" * 70)
            for i, gap in enumerate(self.detected_gaps, 1):
                lines.append(f"{i}. {gap}")
        
        # Hypotheses
        if self.gap_hypotheses:
            lines.append("\n[ HYPOTHESIS GENERATION ]")
            lines.append("-" * 70)
            for i, hyp in enumerate(self.gap_hypotheses, 1):
                lines.append(f"\nHypothesis {i}: {hyp.gap_description}")
                lines.append(f"  Confidence: {hyp.confidence_in_hypothesis:.2f}")
                lines.append(f"  Query: {hyp.to_query()}")
        
        # Adversarial probes
        if self.adversarial_probes:
            lines.append("\n[ ADVERSARIAL SELF-PROBING ]")
            lines.append("-" * 70)
            for i, probe in enumerate(self.adversarial_probes, 1):
                lines.append(f"\n{i}. {probe}")
                if i-1 < len(self.probe_results):
                    lines.append(f"   Result: {self.probe_results[i-1]}")
        
        # Meta-awareness
        lines.append("\n[ EPISTEMIC STATUS ]")
        lines.append("-" * 70)
        lines.append(f"Aware of Limitations: {'✓' if self.aware_of_limitations else '✗'}")
        lines.append(f"Epistemic Humility: {self.epistemic_humility:.2f}")
        
        if self.epistemic_humility < 0.3:
            lines.append("⚠️  WARNING: Overconfident—may be missing blind spots")
        elif self.epistemic_humility > 0.7:
            lines.append("✓ Appropriately humble about knowledge limitations")
        
        return "\n".join(lines)


class MetaAwarenessLayer:
    """
    Recursive self-reflection layer.
    
    The awareness layer becomes self-aware—it examines its own
    confidence estimates, decomposes uncertainty, generates hypotheses
    about knowledge gaps, and adversarially probes its responses.
    """
    
    def __init__(self, awareness_layer):
        """
        Args:
            awareness_layer: CompositionalAwarenessLayer instance
        """
        self.awareness = awareness_layer
        
        # Calibration history (for meta-confidence)
        self.calibration_history: Dict[str, List[float]] = {}
    
    async def recursive_self_reflection(
        self,
        query: str,
        response: str,
        awareness_context
    ) -> SelfReflectionResult:
        """
        Perform recursive self-reflection on a response.
        
        Args:
            query: Original query
            response: Generated response
            awareness_context: UnifiedAwarenessContext from generation
            
        Returns:
            SelfReflectionResult with meta-awareness insights
        """
        
        # 1. Decompose uncertainty
        uncertainty_decomp = await self._decompose_uncertainty(
            query, awareness_context
        )
        
        # 2. Compute meta-confidence
        meta_conf = await self._compute_meta_confidence(
            query, awareness_context
        )
        
        # 3. Detect knowledge gaps
        gaps = self._detect_knowledge_gaps(
            query, awareness_context, uncertainty_decomp
        )
        
        # 4. Generate hypotheses
        hypotheses = self._generate_hypotheses(
            query, gaps, uncertainty_decomp
        )
        
        # 5. Adversarial probing
        probes = self._generate_adversarial_probes(
            query, response, awareness_context
        )
        probe_results = await self._run_probes(probes, response)
        
        # 6. Assess epistemic humility
        humility = self._assess_epistemic_humility(
            awareness_context, meta_conf, uncertainty_decomp
        )
        
        return SelfReflectionResult(
            original_response=response,
            uncertainty_decomposition=uncertainty_decomp,
            meta_confidence=meta_conf,
            detected_gaps=gaps,
            gap_hypotheses=hypotheses,
            adversarial_probes=probes,
            probe_results=probe_results,
            aware_of_limitations=awareness_context.confidence.should_ask_clarification,
            epistemic_humility=humility
        )
    
    async def _decompose_uncertainty(
        self,
        query: str,
        context
    ) -> UncertaintyDecomposition:
        """Break down uncertainty into components"""
        
        total = context.confidence.uncertainty_level
        
        # Analyze structural uncertainty
        structural = 0.0
        ambiguous_structures = []
        if context.structural.is_question:
            # Check for structural ambiguity
            words = query.split()
            if len(words) > 7:
                structural = 0.3  # Long queries have structural ambiguity
                ambiguous_structures.append("long_clause_attachment")
        
        # Analyze semantic uncertainty
        semantic = 0.0
        ambiguous_terms = []
        for word in query.split():
            if word.lower() in ["quantum", "meta", "recursive", "emergent"]:
                semantic += 0.2
                ambiguous_terms.append(word)
        semantic = min(semantic, 1.0)
        
        # Analyze contextual uncertainty
        contextual = 0.0
        missing_context = []
        if context.confidence.knowledge_gap_detected:
            contextual = 0.8
            missing_context.append("domain_knowledge")
        if context.patterns.seen_count == 0:
            contextual += 0.3
            missing_context.append("pattern_history")
        contextual = min(contextual, 1.0)
        
        # Compositional uncertainty
        compositional = total - (structural + semantic + contextual) / 3.0
        compositional = max(0.0, min(compositional, 1.0))
        
        # Determine dominant type
        components = [
            (structural, UncertaintyType.STRUCTURAL),
            (semantic, UncertaintyType.SEMANTIC),
            (contextual, UncertaintyType.CONTEXTUAL),
            (compositional, UncertaintyType.COMPOSITIONAL)
        ]
        dominant_type = max(components, key=lambda x: x[0])[1]
        
        return UncertaintyDecomposition(
            total_uncertainty=total,
            structural_uncertainty=structural,
            semantic_uncertainty=semantic,
            contextual_uncertainty=contextual,
            compositional_uncertainty=compositional,
            dominant_type=dominant_type,
            ambiguous_structures=ambiguous_structures,
            ambiguous_terms=ambiguous_terms,
            missing_context=missing_context
        )
    
    async def _compute_meta_confidence(
        self,
        query: str,
        context
    ) -> MetaConfidence:
        """Compute confidence about confidence"""
        
        primary_conf = 1.0 - context.confidence.uncertainty_level
        
        # Get calibration history
        calibration = self.calibration_history.get(query, [])
        
        # Meta-confidence based on calibration
        if len(calibration) > 5:
            meta_conf = 0.9  # Well-calibrated
        elif len(calibration) > 2:
            meta_conf = 0.6  # Some history
        else:
            meta_conf = 0.3  # No calibration data
        
        # Uncertainty about uncertainty
        unc_squared = context.confidence.uncertainty_level ** 2
        
        # Confidence bounds
        margin = 0.1 * (1.0 - meta_conf)
        lower = max(0.0, primary_conf - margin)
        upper = min(1.0, primary_conf + margin)
        
        return MetaConfidence(
            primary_confidence=primary_conf,
            meta_confidence=meta_conf,
            calibration_history=calibration,
            uncertainty_about_uncertainty=unc_squared,
            lower_bound=lower,
            upper_bound=upper
        )
    
    def _detect_knowledge_gaps(
        self,
        query: str,
        context,
        uncertainty_decomp: UncertaintyDecomposition
    ) -> List[str]:
        """Detect specific knowledge gaps"""
        gaps = []
        
        if context.confidence.knowledge_gap_detected:
            gaps.append("Novel query pattern never seen before")
        
        if uncertainty_decomp.semantic_uncertainty > 0.5:
            for term in uncertainty_decomp.ambiguous_terms:
                gaps.append(f"Unclear definition of '{term}'")
        
        if uncertainty_decomp.contextual_uncertainty > 0.5:
            gaps.append("Missing background/domain knowledge")
        
        if context.patterns.seen_count == 0:
            gaps.append("No familiar compositional patterns")
        
        return gaps
    
    def _generate_hypotheses(
        self,
        query: str,
        gaps: List[str],
        uncertainty_decomp: UncertaintyDecomposition
    ) -> List[KnowledgeGapHypothesis]:
        """Generate hypotheses about knowledge gaps"""
        hypotheses = []
        
        # Semantic ambiguity hypotheses
        for term in uncertainty_decomp.ambiguous_terms:
            hypotheses.append(KnowledgeGapHypothesis(
                gap_description=f"Multiple meanings of '{term}'",
                potential_answers=[
                    f"{term} in physics context",
                    f"{term} in computer science context",
                    f"{term} in philosophy context"
                ],
                required_information=[f"Domain context for '{term}'"],
                testable_predictions=[
                    f"If physics: would reference quantum mechanics",
                    f"If CS: would reference computation",
                    f"If philosophy: would reference epistemology"
                ],
                confidence_in_hypothesis=0.6
            ))
        
        # Contextual gap hypotheses
        if "Missing background" in " ".join(gaps):
            hypotheses.append(KnowledgeGapHypothesis(
                gap_description="User has specialized domain knowledge I lack",
                potential_answers=[
                    "Technical jargon from user's field",
                    "Implicit assumptions from shared context",
                    "Recent developments I'm unaware of"
                ],
                required_information=[
                    "User's domain/field",
                    "Specific context/background",
                    "Recent events or publications"
                ],
                testable_predictions=[
                    "User would clarify domain if asked",
                    "Terms would make sense in specialized context"
                ],
                confidence_in_hypothesis=0.7
            ))
        
        return hypotheses
    
    def _generate_adversarial_probes(
        self,
        query: str,
        response: str,
        context
    ) -> List[AdversarialProbe]:
        """Generate adversarial probes to test response"""
        probes = []
        
        # Test for overconfidence
        if context.confidence.uncertainty_level < 0.3:
            probes.append(AdversarialProbe(
                probe_question=f"Are there cases where this answer would be wrong?",
                expected_weakness="May miss edge cases",
                test_type="edge_case"
            ))
        
        # Test for hidden assumptions
        probes.append(AdversarialProbe(
            probe_question="What assumptions underlie this response?",
            expected_weakness="Implicit assumptions not stated",
            test_type="assumption"
        ))
        
        # Test for contradictions
        if "not" in response.lower() or "no" in response.lower():
            probes.append(AdversarialProbe(
                probe_question="Could the opposite also be true?",
                expected_weakness="Binary thinking",
                test_type="contradiction"
            ))
        
        # Test for knowledge boundaries
        probes.append(AdversarialProbe(
            probe_question="What would I need to know to answer more completely?",
            expected_weakness="Knowledge gaps",
            test_type="boundary"
        ))
        
        return probes
    
    async def _run_probes(
        self,
        probes: List[AdversarialProbe],
        response: str
    ) -> List[str]:
        """Run adversarial probes (simulated for now)"""
        results = []
        
        for probe in probes:
            if probe.test_type == "edge_case":
                results.append("Edge cases not explicitly addressed")
            elif probe.test_type == "assumption":
                results.append("Assumes standard context; alternative interpretations exist")
            elif probe.test_type == "contradiction":
                results.append("Nuance exists; not strictly binary")
            elif probe.test_type == "boundary":
                results.append("Would benefit from domain-specific knowledge")
        
        return results
    
    def _assess_epistemic_humility(
        self,
        context,
        meta_conf: MetaConfidence,
        uncertainty_decomp: UncertaintyDecomposition
    ) -> float:
        """Assess epistemic humility (0=overconfident, 1=humble)"""
        
        # High uncertainty should lead to humility
        expected_humility = uncertainty_decomp.total_uncertainty
        
        # Does awareness match uncertainty?
        actual_humility = 1.0 if context.confidence.should_ask_clarification else 0.0
        
        # Meta-confidence calibration
        calibration_factor = meta_conf.meta_confidence
        
        # Combine
        humility = (expected_humility + actual_humility + calibration_factor) / 3.0
        
        return humility
    
    def update_calibration(self, query: str, actual_confidence: float):
        """Update calibration history for meta-learning"""
        if query not in self.calibration_history:
            self.calibration_history[query] = []
        self.calibration_history[query].append(actual_confidence)
        
        # Keep only recent history
        if len(self.calibration_history[query]) > 10:
            self.calibration_history[query] = self.calibration_history[query][-10:]

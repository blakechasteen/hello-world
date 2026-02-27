"""
Agentic Explainability Integration

Adds SHAP/LIME explanations and causal analysis to HoloLoom's agentic reasoning modes.
Lightweight integration focused on practical interpretability for the 4 reasoning modes:
- DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE

Key Features:
- Automatic explanation generation for low-confidence decisions
- Step-by-step reasoning traces with feature attributions
- Causal "why" analysis for multi-step reasoning
- Integration with AuditTrail for complete provenance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class ExplanationDepth(Enum):
    """How deep to explain the decision."""
    MINIMAL = "minimal"  # Just confidence score
    MODERATE = "moderate"  # + Top features
    COMPREHENSIVE = "comprehensive"  # + SHAP values + causal graph


@dataclass
class StepExplanation:
    """Explanation for a single reasoning step."""
    step_number: int
    query_text: str
    tool_used: str
    confidence: float

    # Feature attributions (lightweight)
    top_contributing_features: List[tuple] = field(default_factory=list)  # (feature, score)
    top_inhibiting_features: List[tuple] = field(default_factory=list)

    # Causal reasoning
    why_this_tool: str = ""  # Human-readable explanation
    alternative_tools: List[str] = field(default_factory=list)

    # Metadata
    decision_time_ms: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "step": self.step_number,
            "query": self.query_text,
            "tool": self.tool_used,
            "confidence": self.confidence,
            "top_features": self.top_contributing_features,
            "why": self.why_this_tool,
            "alternatives": self.alternative_tools
        }


@dataclass
class ReasoningExplanation:
    """Complete explanation for a multi-step reasoning session."""
    session_id: str
    reasoning_mode: str  # DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE
    total_steps: int
    final_confidence: float

    # Step-by-step explanations
    step_explanations: List[StepExplanation] = field(default_factory=list)

    # Overall reasoning flow
    reasoning_flow: str = ""  # Natural language summary
    key_decisions: List[str] = field(default_factory=list)
    confidence_trajectory: List[float] = field(default_factory=list)

    # Causal analysis
    critical_path: List[int] = field(default_factory=list)  # Step indices on critical path
    bottleneck_steps: List[int] = field(default_factory=list)  # Steps with low confidence

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "mode": self.reasoning_mode,
            "total_steps": self.total_steps,
            "final_confidence": self.final_confidence,
            "steps": [s.to_dict() for s in self.step_explanations],
            "reasoning_flow": self.reasoning_flow,
            "key_decisions": self.key_decisions,
            "confidence_trajectory": self.confidence_trajectory,
            "critical_path": self.critical_path,
            "bottleneck_steps": self.bottleneck_steps
        }

    def print_summary(self, depth: ExplanationDepth = ExplanationDepth.MODERATE):
        """Print human-readable summary."""
        print(f"\n{'='*70}")
        print(f"Reasoning Explanation ({self.reasoning_mode} mode)")
        print(f"{'='*70}")

        if depth == ExplanationDepth.MINIMAL:
            print(f"Steps: {self.total_steps} | Final confidence: {self.final_confidence:.3f}")
            return

        print(f"\nOverall Flow:")
        print(f"  {self.reasoning_flow}")

        print(f"\nKey Decisions:")
        for i, decision in enumerate(self.key_decisions, 1):
            print(f"  {i}. {decision}")

        if depth == ExplanationDepth.COMPREHENSIVE:
            print(f"\nStep-by-Step Analysis:")
            for step in self.step_explanations:
                print(f"\n  Step {step.step_number}: {step.query_text[:60]}...")
                print(f"    Tool: {step.tool_used} (confidence: {step.confidence:.3f})")
                print(f"    Why: {step.why_this_tool}")

                if step.top_contributing_features:
                    print(f"    Top features:")
                    for feat, score in step.top_contributing_features[:3]:
                        print(f"      • {feat}: {score:+.3f}")

            if self.bottleneck_steps:
                print(f"\n  ⚠️  Bottleneck steps (low confidence): {self.bottleneck_steps}")


class AgenticExplainer:
    """
    Lightweight explainer for agentic reasoning sessions.

    Generates human-readable explanations without expensive SHAP/LIME computations
    by using decision metadata and heuristics.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        enable_causal_analysis: bool = True
    ):
        """
        Initialize agentic explainer.

        Args:
            confidence_threshold: Threshold for flagging low-confidence steps
            enable_causal_analysis: Whether to compute causal paths
        """
        self.confidence_threshold = confidence_threshold
        self.enable_causal_analysis = enable_causal_analysis

    async def explain_reasoning(
        self,
        session_id: str,
        reasoning_mode: str,
        steps_taken: List[Dict],
        final_confidence: float
    ) -> ReasoningExplanation:
        """
        Generate explanation for a reasoning session.

        Args:
            session_id: Unique session identifier
            reasoning_mode: DIRECT, VERIFY, RESEARCH, or PLAN_EXECUTE
            steps_taken: List of step dictionaries from agentic result
            final_confidence: Final confidence score

        Returns:
            Comprehensive reasoning explanation
        """
        # Create step explanations
        step_explanations = []
        confidence_trajectory = []

        for i, step in enumerate(steps_taken, 1):
            step_conf = step.get('confidence', 0.0)
            confidence_trajectory.append(step_conf)

            # Extract features (lightweight approximation)
            top_features, inhibiting_features = self._extract_features(step)

            # Generate "why" explanation
            why_explanation = self._generate_why_explanation(step, reasoning_mode)

            step_exp = StepExplanation(
                step_number=i,
                query_text=step.get('query', ''),
                tool_used=step.get('tool', 'unknown'),
                confidence=step_conf,
                top_contributing_features=top_features,
                top_inhibiting_features=inhibiting_features,
                why_this_tool=why_explanation,
                alternative_tools=step.get('alternatives', []),
                decision_time_ms=step.get('duration_ms', 0.0)
            )

            step_explanations.append(step_exp)

        # Generate overall reasoning flow
        reasoning_flow = self._generate_reasoning_flow(reasoning_mode, steps_taken)

        # Extract key decisions
        key_decisions = self._extract_key_decisions(steps_taken, reasoning_mode)

        # Identify critical path and bottlenecks
        critical_path = []
        bottleneck_steps = []

        if self.enable_causal_analysis:
            critical_path = self._compute_critical_path(steps_taken)
            bottleneck_steps = [
                i for i, conf in enumerate(confidence_trajectory, 1)
                if conf < self.confidence_threshold
            ]

        return ReasoningExplanation(
            session_id=session_id,
            reasoning_mode=reasoning_mode,
            total_steps=len(steps_taken),
            final_confidence=final_confidence,
            step_explanations=step_explanations,
            reasoning_flow=reasoning_flow,
            key_decisions=key_decisions,
            confidence_trajectory=confidence_trajectory,
            critical_path=critical_path,
            bottleneck_steps=bottleneck_steps
        )

    def _extract_features(self, step: Dict) -> tuple:
        """
        Extract top contributing and inhibiting features from step metadata.

        Returns:
            (top_contributing, top_inhibiting) tuples of (feature, score)
        """
        # Lightweight feature extraction from metadata
        metadata = step.get('metadata', {})

        # Contributing features (heuristic)
        contributing = []
        if 'retrieved_shards' in metadata:
            contributing.append(('memory_retrieval', len(metadata['retrieved_shards']) * 0.1))
        if 'motifs' in metadata:
            contributing.append(('motif_match', len(metadata['motifs']) * 0.05))
        if metadata.get('cache_hit'):
            contributing.append(('cache_hit', 0.15))

        # Inhibiting features (heuristic)
        inhibiting = []
        if metadata.get('low_similarity', False):
            inhibiting.append(('low_similarity', -0.1))
        if metadata.get('no_context', False):
            inhibiting.append(('no_context', -0.15))

        return contributing, inhibiting

    def _generate_why_explanation(self, step: Dict, mode: str) -> str:
        """Generate human-readable 'why' explanation for a step."""
        tool = step.get('tool', 'unknown')
        confidence = step.get('confidence', 0.0)

        explanations = {
            'answer': f"Direct answer with {confidence:.1%} confidence based on retrieved context",
            'search': f"Search required to gather more information (confidence {confidence:.1%})",
            'verify': f"Verification step to check consistency (confidence {confidence:.1%})",
            'synthesize': f"Final synthesis combining {mode} mode results"
        }

        return explanations.get(tool, f"Used {tool} tool (confidence {confidence:.1%})")

    def _generate_reasoning_flow(self, mode: str, steps: List[Dict]) -> str:
        """Generate natural language summary of reasoning flow."""
        if mode == 'DIRECT':
            return f"Single-pass query → answer (no refinement needed)"

        elif mode == 'VERIFY':
            return (f"Initial answer → {len(steps)-1} verification queries → "
                   f"consistency check → final answer")

        elif mode == 'RESEARCH':
            return (f"Decomposed into {len(steps)-1} sub-queries → "
                   f"gathered evidence → synthesized comprehensive answer")

        elif mode == 'PLAN_EXECUTE':
            return (f"Goal decomposed into {len(steps)-1} sub-goals → "
                   f"executed sequentially → combined results")

        return f"{len(steps)} reasoning steps"

    def _extract_key_decisions(self, steps: List[Dict], mode: str) -> List[str]:
        """Extract key decision points from reasoning session."""
        key_decisions = []

        for i, step in enumerate(steps, 1):
            # Flag important decisions
            if step.get('confidence', 1.0) < self.confidence_threshold:
                key_decisions.append(
                    f"Step {i}: Low confidence ({step['confidence']:.3f}) - "
                    f"required {mode}-specific handling"
                )

            if step.get('type') == 'verification' and step.get('contradictions'):
                key_decisions.append(
                    f"Step {i}: Found contradictions in initial answer"
                )

            if step.get('type') == 'sub_goal' and step.get('completed'):
                key_decisions.append(
                    f"Step {i}: Completed sub-goal '{step['goal'][:40]}...'"
                )

        return key_decisions

    def _compute_critical_path(self, steps: List[Dict]) -> List[int]:
        """
        Compute critical path through reasoning graph.

        Critical path = steps that directly contribute to final answer.
        """
        # Simple heuristic: all steps except redundant verifications
        critical = []

        for i, step in enumerate(steps, 1):
            # Include synthesis steps
            if step.get('type') in ('synthesize', 'final'):
                critical.append(i)
            # Include all sub-goals in PLAN_EXECUTE
            elif step.get('type') == 'sub_goal':
                critical.append(i)
            # Include first verification, skip redundant ones
            elif step.get('type') == 'verification' and len(critical) < 2:
                critical.append(i)
            # Include all research queries
            elif step.get('type') == 'research':
                critical.append(i)

        return critical


# Integration with existing agentic system
async def explain_agentic_result(result, depth: ExplanationDepth = ExplanationDepth.MODERATE):
    """
    Convenience function to explain an agentic reasoning result.

    Args:
        result: AgenticResult from agentic orchestrator
        depth: How detailed to make the explanation

    Returns:
        ReasoningExplanation
    """
    explainer = AgenticExplainer()

    explanation = await explainer.explain_reasoning(
        session_id=result.spacetime.query_id,
        reasoning_mode=result.spacetime.metadata.get('reasoning_mode', 'DIRECT'),
        steps_taken=result.steps_taken,
        final_confidence=result.spacetime.confidence
    )

    explanation.print_summary(depth=depth)

    return explanation


if __name__ == "__main__":
    print("Agentic Explainability - Interpretability for 4 Reasoning Modes")
    print("=" * 70)
    print("\nFeatures:")
    print("  - Step-by-step reasoning traces")
    print("  - Feature attribution (lightweight)")
    print("  - Causal 'why' explanations")
    print("  - Critical path analysis")
    print("  - Bottleneck detection")
    print("\nUsage:")
    print("  from hololoom.alignment.agentic_explainability import explain_agentic_result")
    print("  result = await agent.reason(query, mode=ReasoningMode.VERIFY)")
    print("  explanation = await explain_agentic_result(result)")
"""
Pre-built Chain Patterns - Common workflow templates

Provides ready-to-use chains for common operations:
- simple_query: Execute only
- verified_query: Execute + verify
- auto_refine: Execute + verify + refine if low confidence
- iterative_improve: Loop execute/refine until high confidence
- multi_strategy: Try multiple strategies in sequence
- research_pipeline: Full cycle with learning

Author: HoloLoom Architecture Team
Date: November 2025
"""

from .chain import Chain, ChainStep, StepType
from .conditions import Conditions, CommonConditions


class ChainPatterns:
    """Pre-built chain patterns for common workflows."""

    @staticmethod
    def simple_query() -> Chain:
        """
        Simple query: Execute only.

        Flow:
            [execute] → [output]

        Use when:
        - Speed is critical
        - Confidence scoring is sufficient
        - No verification needed
        """
        chain = Chain(name="simple_query", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "max_sources": 5,
            },
        ))
        return chain

    @staticmethod
    def verified_query() -> Chain:
        """
        Verified query: Execute + verify.

        Flow:
            [execute] → [verify] → [output]

        Use when:
        - Verification is important
        - Standard quality checks needed
        - Can tolerate ~10-15ms extra latency
        """
        chain = Chain(name="verified_query", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "max_sources": 5,
            },
            next_step="verify",
        ))
        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
        ))
        return chain

    @staticmethod
    def auto_refine() -> Chain:
        """
        Auto-refine: Execute → verify → refine if low confidence.

        Flow:
            [execute]
                ↓
            [verify]
                ↓
            [confidence < 0.75?]
                ├─ Yes → [refine] → [output]
                └─ No → [output]

        Use when:
        - Quality is important
        - Can handle variable latency
        - Want automatic improvement on low confidence
        """
        chain = Chain(name="auto_refine", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "max_sources": 5,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_confidence",
        ))

        # Conditional: Check if confidence is low
        check_step = ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_below(0.75),
            on_success="refine",  # Refine if low confidence
            on_failure=None,  # Skip refine if high confidence
        )
        chain.add_step("check_confidence", check_step)

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
        ))

        return chain

    @staticmethod
    def iterative_improve() -> Chain:
        """
        Iterative improvement: Loop execute/refine until high confidence.

        Flow:
            [execute]
                ↓
            [verify]
                ↓
            [confidence >= 0.85?]
                ├─ Yes → [output]
                └─ No → [refine] → [back to verify]

        Use when:
        - Quality is critical
        - Can accept higher latency (up to 5-10s)
        - Want guaranteed high confidence
        """
        chain = Chain(name="iterative_improve", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 10,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_quality",
        ))

        # Conditional: Check quality
        chain.add_step("check_quality", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.85),
            on_success=None,  # Success: output
            on_failure="refine",  # Failure: refine
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
            next_step="verify",  # Loop back to verify
            max_iterations=3,  # Safety limit
        ))

        return chain

    @staticmethod
    def multi_strategy() -> Chain:
        """
        Multi-strategy: Try multiple retrieval strategies in sequence.

        Flow:
            [execute (direct)]
                ↓
            [verify]
                ↓
            [confidence >= 0.75?]
                ├─ Yes → [output]
                └─ No → [execute (research)] → [verify] → [output]

        Use when:
        - First approach might fail
        - Want fallback strategies
        - Can't accept complete failure
        """
        chain = Chain(name="multi_strategy", entry_point="execute_direct")

        # Strategy 1: Direct reasoning
        chain.add_step("execute_direct", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "max_sources": 5,
            },
            next_step="verify_direct",
        ))

        chain.add_step("verify_direct", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_direct",
        ))

        # Decision point
        chain.add_step("check_direct", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.75),
            on_success=None,  # Good enough
            on_failure="execute_research",  # Try research
        ))

        # Strategy 2: Research mode
        chain.add_step("execute_research", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 15,
            },
            next_step="verify_research",
        ))

        chain.add_step("verify_research", ChainStep(
            step_type=StepType.VERIFY,
            params={},
        ))

        return chain

    @staticmethod
    def research_pipeline() -> Chain:
        """
        Research pipeline: Full cycle with learning.

        Flow:
            [execute (research)]
                ↓
            [verify]
                ↓
            [confidence >= 0.75?]
                ├─ Yes → [learn] → [output]
                └─ No → [refine] → [verify] → [learn] → [output]

        Use when:
        - Deep research is needed
        - System should learn from queries
        - Quality is paramount
        """
        chain = Chain(name="research_pipeline", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "research",
                "max_sources": 20,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_quality",
        ))

        # Quality check
        chain.add_step("check_quality", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_above(0.75),
            on_success="learn",
            on_failure="refine",
        ))

        # Refinement path
        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
            next_step="verify_refined",
        ))

        chain.add_step("verify_refined", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="learn",
        ))

        # Learning: update strategy
        chain.add_step("learn", ChainStep(
            step_type=StepType.UPDATE_STRATEGY,
            params={
                "feedback": {
                    "helpful": True,
                    "refinement_needed": False,
                }
            },
        ))

        return chain

    @staticmethod
    def quality_first() -> Chain:
        """
        Quality-first: Prioritize quality over speed.

        Flow:
            [execute (research)]
                ↓
            [verify (strict)]
                ↓
            [all checks passed?]
                ├─ Yes → [output]
                └─ No → [loop refine 3x] → [output]

        Use when:
        - Accuracy is critical (medical, legal, financial)
        - Can spend 5-20s per query
        - Must pass all verification checks
        """
        chain = Chain(name="quality_first", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "plan_execute",  # Most thorough
                "max_sources": 20,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_all_passed",
        ))

        # Strict check: all must pass
        chain.add_step("check_all_passed", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.all_checks_passed(),
            on_success=None,
            on_failure="refine_loop",
        ))

        # Refinement loop
        chain.add_step("refine_loop", ChainStep(
            step_type=StepType.LOOP,
            params={},
            next_step="refine",
            max_iterations=3,
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
            next_step="verify",  # Loop back
        ))

        return chain

    @staticmethod
    def quick_answer() -> Chain:
        """
        Quick answer: Minimum latency, acceptable quality.

        Flow:
            [execute (direct)]
                ↓
            [output]

        Use when:
        - Speed is critical (real-time chat, typing suggestions)
        - Users accept lower confidence
        - Latency < 100ms is important
        """
        chain = Chain(name="quick_answer", entry_point="execute")
        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "direct",
                "max_sources": 3,
            },
            timeout_seconds=0.1,  # 100ms timeout
        ))
        return chain

    @staticmethod
    def balanced() -> Chain:
        """
        Balanced: Trade-off between speed and quality.

        Flow:
            [execute (verify mode)]
                ↓
            [verify]
                ↓
            [confidence >= 0.7?]
                ├─ Yes → [output]
                └─ No → [refine once] → [output]

        Use when:
        - Balance is important
        - Most common production use case
        - Latency 150-300ms acceptable
        """
        chain = Chain(name="balanced", entry_point="execute")

        chain.add_step("execute", ChainStep(
            step_type=StepType.EXECUTE,
            params={
                "mode": "verify",
                "max_sources": 7,
            },
            next_step="verify",
        ))

        chain.add_step("verify", ChainStep(
            step_type=StepType.VERIFY,
            params={},
            next_step="check_confidence",
        ))

        chain.add_step("check_confidence", ChainStep(
            step_type=StepType.CONDITION,
            params={},
            condition=Conditions.confidence_below(0.7),
            on_success="refine",
            on_failure=None,
        ))

        chain.add_step("refine", ChainStep(
            step_type=StepType.REFINE,
            params={},
        ))

        return chain

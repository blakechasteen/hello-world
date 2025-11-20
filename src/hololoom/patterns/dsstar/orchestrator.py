#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DS-STAR Orchestrator
====================

Main orchestrator for DS-STAR pattern: iterative data science with self-verification.

Architecture:
1. Analyze data → extract anchors
2. Plan execution → create steps
3. Execute code → run safely
4. Verify results → check goals met
5. Iterate if needed → refine and retry

Integrates with HoloLoom's recursive learning and alignment systems.

Created: 2025-01-20
Based on: Google Research DS-STAR paper
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

from .analyzer import DataAnalyzer, DataProfile
from .planner import Planner, ExecutionPlan
from .executor import SafeExecutor, ExecutionResult
from .verifier import Verifier, VerificationResult


logger = logging.getLogger(__name__)


@dataclass
class DSStarIteration:
    """Single iteration of DS-STAR loop."""
    iteration: int
    plan: ExecutionPlan
    execution_result: ExecutionResult
    verification: VerificationResult
    refinement_applied: Optional[str] = None


@dataclass
class DSStarResult:
    """Final result from DS-STAR process."""
    query: str
    success: bool
    iterations: List[DSStarIteration]
    final_outputs: Dict[str, Any]
    final_verification: VerificationResult
    data_profile: Optional[DataProfile] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def iteration_count(self) -> int:
        """Number of iterations performed."""
        return len(self.iterations)

    def summary(self) -> str:
        """Generate summary of DS-STAR process."""
        lines = [
            f"DS-STAR Result for: {self.query}",
            f"",
            f"Status: {'✓ Success' if self.success else '✗ Failed'}",
            f"Iterations: {self.iteration_count}",
            f"Final Confidence: {self.final_verification.confidence:.1%}",
            f"",
        ]

        if self.success:
            lines.append("Outputs:")
            for key in self.final_outputs:
                lines.append(f"  - {key}")
        else:
            lines.append("Issues:")
            for issue in self.final_verification.issues:
                lines.append(f"  - {issue}")

        return "\n".join(lines)


class DSStarOrchestrator:
    """
    Main DS-STAR orchestrator implementing iterative data science loop.

    Integrates all DS-STAR components:
    - DataAnalyzer: Extract data anchors
    - Planner: Create execution plans
    - Executor: Run code safely
    - Verifier: Check goal satisfaction
    """

    def __init__(
        self,
        max_iterations: int = 3,
        verification_threshold: float = 0.7,
        enable_refinement: bool = True,
        use_hololoom_integration: bool = True
    ):
        self.max_iterations = max_iterations
        self.verification_threshold = verification_threshold
        self.enable_refinement = enable_refinement
        self.use_hololoom_integration = use_hololoom_integration

        # Initialize components
        self.analyzer = DataAnalyzer()
        self.planner = Planner(use_hololoom_routing=use_hololoom_integration)
        self.executor = SafeExecutor(enable_safety_checks=True)
        self.verifier = Verifier(strict_mode=False)

    async def process(
        self,
        query: str,
        data_file: Optional[str] = None,
        data_profile: Optional[DataProfile] = None
    ) -> DSStarResult:
        """
        Process query through complete DS-STAR loop.

        Args:
            query: User's data analysis question
            data_file: Path to data file (optional if data_profile provided)
            data_profile: Pre-analyzed data profile (optional)

        Returns:
            DSStarResult with final outputs and iteration history
        """
        logger.info(f"Starting DS-STAR process for query: {query}")

        # Step 1: Analyze data
        if data_profile is None and data_file:
            logger.info(f"Analyzing data file: {data_file}")
            data_profile = self.analyzer.analyze_file(data_file)
            logger.info(f"Data profile: {data_profile.row_count} rows, {data_profile.column_count} columns")

        iterations = []
        current_plan = None

        for iteration in range(self.max_iterations):
            logger.info(f"DS-STAR Iteration {iteration + 1}/{self.max_iterations}")

            # Step 2: Create execution plan
            if iteration == 0 or not self.enable_refinement:
                current_plan = self.planner.create_plan(query, data_profile)
                logger.info(f"Created plan with {len(current_plan.steps)} steps")
            else:
                # Refine plan based on previous verification
                last_verification = iterations[-1].verification
                current_plan = self._refine_plan(current_plan, last_verification)
                logger.info("Refined plan based on previous iteration")

            # Step 3: Execute plan
            execution_result = self._execute_plan(current_plan, data_file)
            logger.info(f"Execution: {'success' if execution_result.success else 'failed'}")

            # Step 4: Verify results
            verification = self.verifier.verify(current_plan, execution_result)
            logger.info(f"Verification: confidence={verification.confidence:.1%}, verified={verification.verified}")

            # Record iteration
            iterations.append(DSStarIteration(
                iteration=iteration,
                plan=current_plan,
                execution_result=execution_result,
                verification=verification,
                refinement_applied="initial" if iteration == 0 else "refinement"
            ))

            # Check if goal achieved
            if verification.verified and verification.confidence >= self.verification_threshold:
                logger.info(f"Goal achieved in {iteration + 1} iterations")
                return DSStarResult(
                    query=query,
                    success=True,
                    iterations=iterations,
                    final_outputs=execution_result.outputs,
                    final_verification=verification,
                    data_profile=data_profile,
                    metadata={'iterations_needed': iteration + 1}
                )

            # Check if should continue
            if not self.enable_refinement:
                break

            # Log issues for next iteration
            if verification.issues:
                logger.warning(f"Issues to address: {verification.issues}")

        # Max iterations reached without success
        last_iteration = iterations[-1]
        logger.warning(f"Max iterations reached without achieving goal")

        return DSStarResult(
            query=query,
            success=False,
            iterations=iterations,
            final_outputs=last_iteration.execution_result.outputs,
            final_verification=last_iteration.verification,
            data_profile=data_profile,
            metadata={'max_iterations_reached': True}
        )

    def _execute_plan(
        self,
        plan: ExecutionPlan,
        data_file: Optional[str]
    ) -> ExecutionResult:
        """Execute all steps in plan."""
        context = {}
        combined_code = []

        # Add data file to context if provided
        if data_file:
            context['data_file'] = data_file

        # Combine all steps into single code block
        for step in plan.steps:
            combined_code.append(f"# Step {step.step_id}: {step.description}")
            combined_code.append(step.code_template)
            combined_code.append("")

        code = "\n".join(combined_code)

        # Execute
        return self.executor.execute(code, context)

    def _refine_plan(
        self,
        plan: ExecutionPlan,
        verification: VerificationResult
    ) -> ExecutionPlan:
        """
        Refine plan based on verification results.

        Args:
            plan: Current plan
            verification: Verification showing what needs improvement

        Returns:
            Refined ExecutionPlan
        """
        # Get improvement suggestions
        suggestions = self.verifier.suggest_improvements(plan, verification)

        # Apply refinements (simplified - could be much more sophisticated)
        refined_query = plan.query + " [Refined based on: " + "; ".join(suggestions) + "]"

        # Re-create plan with refinement context
        return self.planner.create_plan(refined_query, plan.data_profile)

    def explain_process(self, result: DSStarResult) -> str:
        """
        Generate explanation of DS-STAR process.

        Args:
            result: DSStarResult to explain

        Returns:
            Human-readable explanation
        """
        lines = [
            "# DS-STAR Process Explanation",
            "",
            f"**Query**: {result.query}",
            f"**Iterations**: {result.iteration_count}",
            f"**Final Status**: {'Success ✓' if result.success else 'Failed ✗'}",
            "",
            "## Iteration History",
            ""
        ]

        for iteration in result.iterations:
            lines.append(f"### Iteration {iteration.iteration + 1}")
            lines.append(f"**Plan**: {len(iteration.plan.steps)} steps")
            lines.append(f"**Execution**: {'Success' if iteration.execution_result.success else 'Failed'}")
            lines.append(f"**Verification**: {iteration.verification.confidence:.1%} confidence")

            if iteration.verification.issues:
                lines.append("**Issues**:")
                for issue in iteration.verification.issues:
                    lines.append(f"  - {issue}")

            lines.append("")

        if result.success:
            lines.append("## Final Outputs")
            for key in result.final_outputs:
                lines.append(f"- {key}")
        else:
            lines.append("## Suggestions")
            for suggestion in result.final_verification.suggestions:
                lines.append(f"- {suggestion}")

        return "\n".join(lines)


# Convenience function
async def process_query(
    query: str,
    data_file: Optional[str] = None,
    max_iterations: int = 3
) -> DSStarResult:
    """
    Quick DS-STAR query processing.

    Args:
        query: Data analysis question
        data_file: Optional path to data file
        max_iterations: Max refinement iterations

    Returns:
        DSStarResult
    """
    orchestrator = DSStarOrchestrator(max_iterations=max_iterations)
    return await orchestrator.process(query, data_file)

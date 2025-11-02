"""
Promptly Dashboard Integration API
===================================
Standardized API for smart dashboard integration.

The dashboard uses this API to:
1. Execute prompts and workflows
2. Run A/B tests
3. Get LLM judge evaluations
4. Query loop compositions
5. Get system status and analytics
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

try:
    from promptly.promptly import Promptly
    from promptly.execution_engine import ExecutionEngine
    from promptly.loop_composition import LoopComposer
    from promptly.tools.llm_judge_enhanced import EnhancedLLMJudge
    from promptly.tools.ab_testing import ABTester
    from promptly.tools.cost_tracker import CostTracker
    from promptly.tools.prompt_analytics import PromptAnalytics
except ImportError as e:
    # Graceful degradation for incomplete installation
    print(f"Warning: Some Promptly modules not available: {e}")


@dataclass
class ExecutionResult:
    """Result of prompt execution."""

    # Metadata
    timestamp: str
    prompt: str
    duration_ms: float

    # Output
    response: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None

    # Quality metrics (if judge enabled)
    quality_score: Optional[float] = None
    quality_breakdown: Optional[Dict[str, float]] = None

    # Execution details
    model: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ABTestResult:
    """Result of A/B test."""

    # Metadata
    timestamp: str
    num_tests: int
    duration_ms: float

    # Winner
    winner: str  # "prompt_a" or "prompt_b"
    confidence: float

    # Detailed scores
    prompt_a_avg_score: float
    prompt_b_avg_score: float
    test_cases: List[Dict[str, Any]]

    # Statistical significance
    p_value: Optional[float] = None
    effect_size: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SystemStatus:
    """System status and metrics."""

    # Health
    status: str  # "healthy", "degraded", "error"
    uptime_seconds: float

    # Components
    promptly_ready: bool
    judge_ready: bool
    analytics_ready: bool

    # Performance
    avg_execution_time_ms: float
    total_executions: int
    success_rate: float

    # Costs
    total_cost_usd: float
    total_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PromptlyAPI:
    """
    Main API for dashboard integration.

    Usage:
        api = PromptlyAPI()
        result = await api.execute_prompt("Explain quantum computing")
    """

    def __init__(
        self,
        enable_judge: bool = True,
        enable_analytics: bool = True,
        enable_cost_tracking: bool = True,
    ):
        """
        Initialize API.

        Args:
            enable_judge: Enable LLM judge for quality scoring
            enable_analytics: Enable analytics tracking
            enable_cost_tracking: Enable cost tracking
        """
        self._promptly = None
        self._judge = None
        self._analytics = None
        self._cost_tracker = None
        self._ab_tester = None

        self._enable_judge = enable_judge
        self._enable_analytics = enable_analytics
        self._enable_cost_tracking = enable_cost_tracking

        self._start_time = datetime.now()
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "total_duration_ms": 0.0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
        }

    async def initialize(self) -> None:
        """
        Initialize components.

        Note:
            Must be called before using the API.
        """
        # Initialize Promptly
        self._promptly = Promptly()

        # Initialize judge if enabled
        if self._enable_judge:
            try:
                self._judge = EnhancedLLMJudge()
            except Exception as e:
                print(f"Warning: Could not initialize LLM judge: {e}")

        # Initialize analytics if enabled
        if self._enable_analytics:
            try:
                self._analytics = PromptAnalytics()
            except Exception as e:
                print(f"Warning: Could not initialize analytics: {e}")

        # Initialize cost tracker if enabled
        if self._enable_cost_tracking:
            try:
                self._cost_tracker = CostTracker()
            except Exception as e:
                print(f"Warning: Could not initialize cost tracker: {e}")

        # Initialize A/B tester
        try:
            self._ab_tester = ABTester()
        except Exception as e:
            print(f"Warning: Could not initialize A/B tester: {e}")

    async def execute_prompt(
        self,
        prompt: str,
        evaluate_quality: bool = True,
        model: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a prompt and return result.

        Args:
            prompt: Prompt text to execute
            evaluate_quality: Enable LLM judge evaluation
            model: Optional model name (defaults to Promptly config)

        Returns:
            ExecutionResult with response and metrics
        """
        start_time = datetime.now()

        try:
            # Execute prompt
            response = await self._execute_with_promptly(prompt, model)

            # Track costs if enabled
            tokens_used = None
            cost_usd = None
            if self._cost_tracker:
                # TODO: Get actual token count from response
                tokens_used = len(response.split())  # Rough estimate
                cost_usd = self._cost_tracker.calculate_cost(tokens_used, model or "gpt-3.5-turbo")
                self._metrics["total_tokens"] += tokens_used
                self._metrics["total_cost_usd"] += cost_usd

            # Evaluate quality if enabled
            quality_score = None
            quality_breakdown = None
            if evaluate_quality and self._judge:
                try:
                    eval_result = self._judge.evaluate(
                        query=prompt,
                        response=response,
                        criteria=["accuracy", "clarity", "completeness", "relevance"]
                    )
                    quality_score = eval_result.get("overall_score", 0.0)
                    quality_breakdown = eval_result.get("criteria_scores", {})
                except Exception as e:
                    print(f"Warning: Quality evaluation failed: {e}")

            # Update metrics
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self._metrics["total_executions"] += 1
            self._metrics["successful_executions"] += 1
            self._metrics["total_duration_ms"] += duration_ms

            return ExecutionResult(
                timestamp=start_time.isoformat(),
                prompt=prompt,
                duration_ms=duration_ms,
                response=response,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                quality_score=quality_score,
                quality_breakdown=quality_breakdown,
                model=model,
                success=True,
            )

        except Exception as e:
            # Update metrics
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self._metrics["total_executions"] += 1
            self._metrics["total_duration_ms"] += duration_ms

            return ExecutionResult(
                timestamp=start_time.isoformat(),
                prompt=prompt,
                duration_ms=duration_ms,
                response="",
                success=False,
                error=str(e),
            )

    async def _execute_with_promptly(self, prompt: str, model: Optional[str]) -> str:
        """Execute prompt with Promptly engine."""
        # TODO: Implement actual Promptly execution
        # This is a placeholder
        return f"Response to: {prompt}"

    async def run_ab_test(
        self,
        prompt_a: str,
        prompt_b: str,
        test_cases: List[Dict[str, Any]],
    ) -> ABTestResult:
        """
        Run A/B test comparing two prompts.

        Args:
            prompt_a: First prompt variant
            prompt_b: Second prompt variant
            test_cases: List of test cases with variables

        Returns:
            ABTestResult with winner and statistics
        """
        start_time = datetime.now()

        # TODO: Implement actual A/B testing
        # Placeholder result
        result = ABTestResult(
            timestamp=start_time.isoformat(),
            num_tests=len(test_cases),
            duration_ms=0.0,
            winner="prompt_a",
            confidence=0.75,
            prompt_a_avg_score=0.85,
            prompt_b_avg_score=0.72,
            test_cases=[],
            p_value=0.03,
            effect_size=0.45,
        )

        end_time = datetime.now()
        result.duration_ms = (end_time - start_time).total_seconds() * 1000

        return result

    async def execute_loop(
        self,
        loop_definition: str,
        variables: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a loop composition.

        Args:
            loop_definition: Loop DSL definition
            variables: Input variables

        Returns:
            Dictionary with loop execution results
        """
        # TODO: Implement loop execution
        return {
            "success": True,
            "output": {},
            "steps_executed": 0,
        }

    async def get_status(self) -> SystemStatus:
        """
        Get system status and metrics.

        Returns:
            SystemStatus with health and performance metrics
        """
        uptime = (datetime.now() - self._start_time).total_seconds()
        avg_time = (
            self._metrics["total_duration_ms"] / self._metrics["total_executions"]
            if self._metrics["total_executions"] > 0
            else 0.0
        )
        success_rate = (
            self._metrics["successful_executions"] / self._metrics["total_executions"]
            if self._metrics["total_executions"] > 0
            else 1.0
        )

        return SystemStatus(
            status="healthy",
            uptime_seconds=uptime,
            promptly_ready=self._promptly is not None,
            judge_ready=self._judge is not None,
            analytics_ready=self._analytics is not None,
            avg_execution_time_ms=avg_time,
            total_executions=self._metrics["total_executions"],
            success_rate=success_rate,
            total_cost_usd=self._metrics["total_cost_usd"],
            total_tokens=self._metrics["total_tokens"],
        )

    async def get_analytics(self) -> Dict[str, Any]:
        """
        Get detailed analytics.

        Returns:
            Dictionary with analytics data
        """
        if not self._analytics:
            return {"error": "Analytics not enabled"}

        # TODO: Implement analytics retrieval
        return {
            "recent_prompts": [],
            "quality_trends": {},
            "cost_breakdown": {},
            "performance_metrics": {},
        }

    async def close(self) -> None:
        """Clean up resources."""
        # Save analytics if enabled
        if self._analytics:
            try:
                # TODO: Save analytics data
                pass
            except Exception as e:
                print(f"Warning: Could not save analytics: {e}")

    # Context manager support
    async def __aenter__(self):
        """Enter async context."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        await self.close()


# Factory function for dashboard
def create_api(
    enable_judge: bool = True,
    enable_analytics: bool = True,
    enable_cost_tracking: bool = True,
) -> PromptlyAPI:
    """
    Factory function for creating API instances.

    Args:
        enable_judge: Enable LLM judge for quality scoring
        enable_analytics: Enable analytics tracking
        enable_cost_tracking: Enable cost tracking

    Returns:
        PromptlyAPI instance

    Example:
        >>> api = create_api(enable_judge=True)
        >>> result = await api.execute_prompt("Explain AI")
    """
    return PromptlyAPI(
        enable_judge=enable_judge,
        enable_analytics=enable_analytics,
        enable_cost_tracking=enable_cost_tracking,
    )

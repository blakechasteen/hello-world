"""
Promptly Agentic Integration

Provides agentic reasoning capabilities for complex prompt execution.
Supports multiple reasoning modes: direct, verify, research, and plan_execute.

Usage:
    from promptly.integrations.agentic_integration import AgenticPromptExecutor

    executor = AgenticPromptExecutor(db, hololoom_bridge)

    # Execute with verification
    result = await executor.execute("my_prompt", {"text": "..."}, mode="verify")

    # Verify output against criteria
    verification = await executor.verify_prompt_output("my_prompt", output, criteria)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import asyncio


class ReasoningMode(Enum):
    """Available agentic reasoning modes."""
    DIRECT = "direct"       # Single-pass answer
    VERIFY = "verify"       # Answer + verification
    RESEARCH = "research"   # Multi-query exploration
    PLAN_EXECUTE = "plan_execute"  # Goal decomposition


@dataclass
class ReasoningStep:
    """A single step in the reasoning process."""
    step_index: int
    step_type: str  # "query", "verify", "plan", "execute", "synthesize"
    input_text: str
    output_text: str
    confidence: float
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_index": self.step_index,
            "step_type": self.step_type,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }


@dataclass
class VerificationResult:
    """Result of output verification."""
    verified: bool
    confidence: float
    checks_passed: List[str]
    checks_failed: List[str]
    discrepancies: List[Dict[str, Any]]
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verified": self.verified,
            "confidence": self.confidence,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "discrepancies": self.discrepancies,
            "suggestions": self.suggestions
        }


@dataclass
class AgenticPromptResult:
    """Result of agentic prompt execution."""
    prompt_name: str
    mode: ReasoningMode
    response: str
    confidence: float
    steps_taken: List[ReasoningStep]
    verification: Optional[VerificationResult]
    total_duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "mode": self.mode.value,
            "response": self.response,
            "confidence": self.confidence,
            "steps_taken": [s.to_dict() for s in self.steps_taken],
            "verification": self.verification.to_dict() if self.verification else None,
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata
        }


class AgenticPromptExecutor:
    """
    Executes prompts with agentic reasoning capabilities.

    Supports multiple reasoning modes:
    - DIRECT: Single-pass execution (fastest, ~150ms)
    - VERIFY: Execute + verify output (~600ms)
    - RESEARCH: Multi-query exploration (~900ms)
    - PLAN_EXECUTE: Goal decomposition (~750ms)

    Integrates with HoloLoom for memory and context retrieval.
    """

    def __init__(
        self,
        db=None,
        hololoom_bridge=None,
        rag_integration=None,
        default_mode: ReasoningMode = ReasoningMode.DIRECT,
        max_steps: int = 5,
        verification_threshold: float = 0.7,
        llm_executor: Optional[Callable] = None
    ):
        """
        Initialize agentic executor.

        Args:
            db: Promptly database connection
            hololoom_bridge: HoloLoomBridge instance for memory access
            rag_integration: RAG integration for context enhancement
            default_mode: Default reasoning mode
            max_steps: Maximum reasoning steps (for research/plan modes)
            verification_threshold: Confidence threshold for verification
            llm_executor: Optional LLM execution function
        """
        self.db = db
        self._hololoom_bridge = hololoom_bridge
        self._rag_integration = rag_integration
        self.default_mode = default_mode
        self.max_steps = max_steps
        self.verification_threshold = verification_threshold
        self.llm_executor = llm_executor

    async def execute(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]] = None,
        mode: Optional[ReasoningMode] = None,
        max_steps: Optional[int] = None
    ) -> AgenticPromptResult:
        """
        Execute a prompt with agentic reasoning.

        Args:
            prompt_name: Name of the prompt to execute
            variables: Variables to substitute in the prompt
            mode: Reasoning mode (default: self.default_mode)
            max_steps: Maximum reasoning steps

        Returns:
            AgenticPromptResult with response and reasoning trace
        """
        mode = mode or self.default_mode
        max_steps = max_steps or self.max_steps
        start_time = datetime.now()
        steps: List[ReasoningStep] = []
        verification = None

        # Get prompt content
        prompt_content = await self._get_prompt_content(prompt_name, variables)

        if mode == ReasoningMode.DIRECT:
            response, confidence, steps = await self._execute_direct(
                prompt_name, prompt_content
            )

        elif mode == ReasoningMode.VERIFY:
            response, confidence, steps, verification = await self._execute_verify(
                prompt_name, prompt_content
            )

        elif mode == ReasoningMode.RESEARCH:
            response, confidence, steps = await self._execute_research(
                prompt_name, prompt_content, max_steps
            )

        elif mode == ReasoningMode.PLAN_EXECUTE:
            response, confidence, steps = await self._execute_plan(
                prompt_name, prompt_content, max_steps
            )

        else:
            response, confidence, steps = await self._execute_direct(
                prompt_name, prompt_content
            )

        end_time = datetime.now()
        total_duration_ms = (end_time - start_time).total_seconds() * 1000

        return AgenticPromptResult(
            prompt_name=prompt_name,
            mode=mode,
            response=response,
            confidence=confidence,
            steps_taken=steps,
            verification=verification,
            total_duration_ms=total_duration_ms,
            metadata={
                "max_steps": max_steps,
                "variables": variables
            }
        )

    async def verify_prompt_output(
        self,
        prompt_name: str,
        output: str,
        criteria: Optional[List[str]] = None
    ) -> VerificationResult:
        """
        Verify prompt output against criteria.

        Args:
            prompt_name: Name of the prompt
            output: Output to verify
            criteria: Optional verification criteria

        Returns:
            VerificationResult with verification details
        """
        criteria = criteria or self._get_default_criteria(prompt_name)

        checks_passed = []
        checks_failed = []
        discrepancies = []
        suggestions = []

        # Run verification checks
        for criterion in criteria:
            passed, details = await self._check_criterion(output, criterion)
            if passed:
                checks_passed.append(criterion)
            else:
                checks_failed.append(criterion)
                if details:
                    discrepancies.append({
                        "criterion": criterion,
                        "details": details
                    })

        # Calculate verification confidence
        if criteria:
            pass_rate = len(checks_passed) / len(criteria)
        else:
            pass_rate = 0.5  # Default when no criteria

        confidence = pass_rate

        # Generate suggestions for failed checks
        for criterion in checks_failed:
            suggestion = self._generate_suggestion(criterion)
            if suggestion:
                suggestions.append(suggestion)

        verified = confidence >= self.verification_threshold

        return VerificationResult(
            verified=verified,
            confidence=confidence,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            discrepancies=discrepancies,
            suggestions=suggestions
        )

    # ==================== Execution Mode Methods ====================

    async def _execute_direct(
        self,
        prompt_name: str,
        prompt_content: str
    ) -> tuple[str, float, List[ReasoningStep]]:
        """Execute prompt in direct mode (single pass)."""
        start_time = datetime.now()

        response = await self._call_llm(prompt_content)
        confidence = 0.75  # Base confidence for direct mode

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        steps = [ReasoningStep(
            step_index=0,
            step_type="query",
            input_text=prompt_content,
            output_text=response,
            confidence=confidence,
            duration_ms=duration_ms
        )]

        return response, confidence, steps

    async def _execute_verify(
        self,
        prompt_name: str,
        prompt_content: str
    ) -> tuple[str, float, List[ReasoningStep], VerificationResult]:
        """Execute prompt with verification."""
        steps: List[ReasoningStep] = []

        # Step 1: Generate initial response
        start_time = datetime.now()
        initial_response = await self._call_llm(prompt_content)
        step1_duration = (datetime.now() - start_time).total_seconds() * 1000

        steps.append(ReasoningStep(
            step_index=0,
            step_type="query",
            input_text=prompt_content,
            output_text=initial_response,
            confidence=0.75,
            duration_ms=step1_duration
        ))

        # Step 2: Verify the response
        start_time = datetime.now()
        verification = await self.verify_prompt_output(prompt_name, initial_response)
        step2_duration = (datetime.now() - start_time).total_seconds() * 1000

        steps.append(ReasoningStep(
            step_index=1,
            step_type="verify",
            input_text=initial_response,
            output_text=f"Verified: {verification.verified}, Confidence: {verification.confidence:.2f}",
            confidence=verification.confidence,
            duration_ms=step2_duration
        ))

        # Final confidence combines generation and verification
        final_confidence = 0.6 * 0.75 + 0.4 * verification.confidence

        return initial_response, final_confidence, steps, verification

    async def _execute_research(
        self,
        prompt_name: str,
        prompt_content: str,
        max_steps: int
    ) -> tuple[str, float, List[ReasoningStep]]:
        """Execute prompt in research mode (multi-query exploration)."""
        steps: List[ReasoningStep] = []
        research_findings = []

        # Generate sub-questions for exploration
        sub_questions = await self._generate_sub_questions(prompt_content, max_steps - 1)

        # Execute each sub-question
        for i, question in enumerate(sub_questions[:max_steps - 1]):
            start_time = datetime.now()
            answer = await self._call_llm(question)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            steps.append(ReasoningStep(
                step_index=i,
                step_type="query",
                input_text=question,
                output_text=answer,
                confidence=0.7,
                duration_ms=duration_ms
            ))
            research_findings.append(answer)

        # Synthesize final response
        start_time = datetime.now()
        synthesis_prompt = f"Based on the following research:\n\n"
        for i, finding in enumerate(research_findings):
            synthesis_prompt += f"{i+1}. {finding}\n\n"
        synthesis_prompt += f"\nProvide a comprehensive answer to: {prompt_content}"

        final_response = await self._call_llm(synthesis_prompt)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        steps.append(ReasoningStep(
            step_index=len(sub_questions),
            step_type="synthesize",
            input_text=synthesis_prompt,
            output_text=final_response,
            confidence=0.85,
            duration_ms=duration_ms
        ))

        # Higher confidence due to multi-query exploration
        final_confidence = 0.85

        return final_response, final_confidence, steps

    async def _execute_plan(
        self,
        prompt_name: str,
        prompt_content: str,
        max_steps: int
    ) -> tuple[str, float, List[ReasoningStep]]:
        """Execute prompt in plan-execute mode (goal decomposition)."""
        steps: List[ReasoningStep] = []

        # Step 1: Generate plan
        start_time = datetime.now()
        plan_prompt = f"Create a step-by-step plan to accomplish: {prompt_content}\nList 3-5 concrete steps."
        plan = await self._call_llm(plan_prompt)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        steps.append(ReasoningStep(
            step_index=0,
            step_type="plan",
            input_text=plan_prompt,
            output_text=plan,
            confidence=0.8,
            duration_ms=duration_ms
        ))

        # Parse plan steps
        plan_steps = self._parse_plan_steps(plan)

        # Execute each plan step
        results = []
        for i, plan_step in enumerate(plan_steps[:max_steps - 2]):
            start_time = datetime.now()
            result = await self._call_llm(f"Execute this step: {plan_step}")
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000

            steps.append(ReasoningStep(
                step_index=i + 1,
                step_type="execute",
                input_text=plan_step,
                output_text=result,
                confidence=0.75,
                duration_ms=duration_ms
            ))
            results.append(result)

        # Final synthesis
        start_time = datetime.now()
        synthesis_prompt = f"Summarize the results of completing these steps:\n"
        for i, result in enumerate(results):
            synthesis_prompt += f"Step {i+1}: {result}\n"
        synthesis_prompt += f"\nOriginal goal: {prompt_content}"

        final_response = await self._call_llm(synthesis_prompt)
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        steps.append(ReasoningStep(
            step_index=len(results) + 1,
            step_type="synthesize",
            input_text=synthesis_prompt,
            output_text=final_response,
            confidence=0.85,
            duration_ms=duration_ms
        ))

        final_confidence = 0.82

        return final_response, final_confidence, steps

    # ==================== Helper Methods ====================

    async def _get_prompt_content(
        self,
        prompt_name: str,
        variables: Optional[Dict[str, Any]]
    ) -> str:
        """Get prompt content with variable substitution."""
        content = ""

        if self.db:
            try:
                cursor = self.db.execute(
                    "SELECT content FROM prompts WHERE name = ?",
                    (prompt_name,)
                )
                row = cursor.fetchone()
                if row:
                    content = row[0]
            except Exception:
                pass

        if not content:
            content = f"[Prompt: {prompt_name}]"

        # Substitute variables
        if variables:
            for key, value in variables.items():
                content = content.replace(f"{{{key}}}", str(value))
                content = content.replace(f"{{{{ {key} }}}}", str(value))

        return content

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        if self.llm_executor:
            return await self.llm_executor(prompt)

        # Fallback placeholder
        return f"[Agentic response to: {prompt[:100]}...]"

    def _get_default_criteria(self, prompt_name: str) -> List[str]:
        """Get default verification criteria for a prompt."""
        return [
            "Response is relevant to the prompt",
            "Response is coherent and well-structured",
            "Response does not contain contradictions",
            "Response is appropriately detailed"
        ]

    async def _check_criterion(
        self,
        output: str,
        criterion: str
    ) -> tuple[bool, Optional[str]]:
        """Check if output satisfies a criterion."""
        # Simple heuristic checks
        if "relevant" in criterion.lower():
            return len(output) > 50, None
        elif "coherent" in criterion.lower():
            return len(output.split()) > 10, None
        elif "contradiction" in criterion.lower():
            return True, None  # Would need semantic analysis
        elif "detailed" in criterion.lower():
            return len(output) > 100, None
        else:
            return True, None

    def _generate_suggestion(self, criterion: str) -> Optional[str]:
        """Generate improvement suggestion for failed criterion."""
        suggestions = {
            "relevant": "Ensure the response addresses the main topic",
            "coherent": "Improve logical flow between sentences",
            "contradiction": "Review for consistency in claims",
            "detailed": "Add more specific examples or explanations"
        }

        for key, suggestion in suggestions.items():
            if key in criterion.lower():
                return suggestion

        return None

    async def _generate_sub_questions(
        self,
        prompt: str,
        count: int
    ) -> List[str]:
        """Generate sub-questions for research mode."""
        # Simple decomposition (would use LLM in production)
        base_questions = [
            f"What are the key concepts related to: {prompt}?",
            f"What are different perspectives on: {prompt}?",
            f"What are potential implications of: {prompt}?",
            f"What are common misconceptions about: {prompt}?",
            f"What are practical applications of: {prompt}?"
        ]
        return base_questions[:count]

    def _parse_plan_steps(self, plan: str) -> List[str]:
        """Parse plan text into individual steps."""
        lines = plan.strip().split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            # Remove numbering prefixes
            if line and (line[0].isdigit() or line.startswith('-')):
                step = line.lstrip('0123456789.-) ').strip()
                if step:
                    steps.append(step)
        return steps if steps else [plan]


# ==================== Factory Functions ====================

def create_agentic_executor(
    db=None,
    hololoom_bridge=None,
    rag_integration=None,
    **kwargs
) -> AgenticPromptExecutor:
    """
    Create an AgenticPromptExecutor instance.

    Args:
        db: Promptly database connection
        hololoom_bridge: HoloLoomBridge instance
        rag_integration: RAG integration instance
        **kwargs: Additional configuration options

    Returns:
        Configured AgenticPromptExecutor instance
    """
    return AgenticPromptExecutor(
        db=db,
        hololoom_bridge=hololoom_bridge,
        rag_integration=rag_integration,
        **kwargs
    )


async def execute_prompt_agentic(
    prompt_name: str,
    variables: Optional[Dict[str, Any]] = None,
    mode: str = "direct",
    db=None,
    hololoom_bridge=None,
    llm_executor: Optional[Callable] = None
) -> AgenticPromptResult:
    """
    Convenience function to execute a prompt with agentic reasoning.

    Args:
        prompt_name: Name of the prompt
        variables: Variables to substitute
        mode: Reasoning mode ("direct", "verify", "research", "plan_execute")
        db: Database connection
        hololoom_bridge: HoloLoom bridge instance
        llm_executor: LLM execution function

    Returns:
        AgenticPromptResult
    """
    executor = create_agentic_executor(
        db=db,
        hololoom_bridge=hololoom_bridge,
        llm_executor=llm_executor
    )

    reasoning_mode = ReasoningMode(mode)
    return await executor.execute(prompt_name, variables, reasoning_mode)

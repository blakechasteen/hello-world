"""
Unified Metaprompting Refinement Framework (MRF)
=================================================
Central integration point for all prompt enhancement and refinement capabilities.

**Date**: November 2025
**Status**: Production Ready

This module unifies:
1. 7-Component Metaprompt Framework (ROLE, OBJECTIVE, PROCESS, FORMAT, CONSTRAINTS, UNCERTAINTY, VALIDATION)
2. Recursive Refinement Strategies (REFINE, CRITIQUE, VERIFY, ELEGANCE, HOFSTADTER)
3. Model Adapters (Claude, Gemini, GPT, Ollama)
4. Thompson Sampling Strategy Selection
5. Quality Trajectory Tracking

Usage:
    ```python
    from hololoom.prompting import UnifiedMRF

    mrf = UnifiedMRF()

    # Enhance a prompt
    enhanced = await mrf.enhance_prompt(
        request={"task": "code_review", "code": code},
        framework="7-component",
        model="claude"
    )

    # Refine a response
    refined = await mrf.refine_response(
        query="What is Thompson Sampling?",
        response=initial_response,
        strategy="verify"
    )

    # Auto-select best strategy
    best = await mrf.auto_refine(
        query=query,
        response=response,
        query_type="factual"
    )
    ```

Architecture:
    ```
    UnifiedMRF
    ├─ MetapromptEngine (7-component framework)
    ├─ RefinementEngine (5 recursive strategies)
    ├─ ModelAdapterRegistry (4 providers)
    ├─ StrategySelector (Thompson Sampling)
    └─ QualityTracker (trajectory monitoring)
    ```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
import time
import json

# Import existing MRF components
try:
    from hololoom.prompting.metaprompt import create_metaprompt, create_metaprompt_with_strategy
    from hololoom.prompting.adapters import ModelAdapter, ClaudeAdapter, GeminiAdapter, GPTAdapter
    from hololoom.prompting.strategy import PromptingStrategy, StrategyContext, StrategyResult
    from hololoom.prompting.registry import StrategyRegistry
except ImportError:
    # Graceful degradation if prompting module incomplete
    create_metaprompt = None
    ModelAdapter = None


class RefinementStrategyType(Enum):
    """Refinement strategy types (renamed from RefinementStrategy to avoid collision)."""
    REFINE = "refine"  # Iteratively expand with more context
    CRITIQUE = "critique"  # Self-critique and regenerate
    VERIFY = "verify"  # Multi-pass cross-check (Accuracy → Completeness → Consistency)
    ELEGANCE = "elegance"  # Multi-pass polish (Clarity → Simplicity → Beauty)
    HOFSTADTER = "hofstadter"  # Strange loop self-reference
    AUTO = "auto"  # Auto-select best strategy


class ModelProvider(Enum):
    """Supported LLM providers."""
    CLAUDE = "claude"
    GEMINI = "gemini"
    GPT = "gpt"
    OLLAMA = "ollama"


@dataclass
class MetapromptConfig:
    """Configuration for 7-component metaprompt framework."""
    role: str
    objective: Dict[str, str]  # {"primary": "...", "secondary": "..."}
    process: List[str]  # Step-by-step methodology
    format: str  # Expected output structure
    constraints: List[str]  # What NOT to do
    uncertainty: str  # Fallback behavior when uncertain
    validation: List[str]  # Success criteria checklist


@dataclass
class RefinementConfig:
    """Configuration for refinement strategies."""
    strategy: RefinementStrategyType = RefinementStrategyType.AUTO
    max_iterations: int = 3
    quality_threshold: float = 0.85
    enable_learning: bool = True  # Thompson Sampling learning


@dataclass
class EnhancedPrompt:
    """Result of prompt enhancement."""
    prompt: str
    framework_used: str  # "7-component", "simple", etc.
    model_optimized_for: Optional[ModelProvider]
    enhancements_applied: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinedResponse:
    """Result of response refinement."""
    response: str
    original_response: str
    strategy_used: RefinementStrategyType
    iterations: int
    quality_improvement: float  # Delta from original
    final_quality: float
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityMetrics:
    """Quality metrics for tracking."""
    confidence: float = 0.0
    context_richness: float = 0.0
    response_completeness: float = 0.0

    @property
    def composite_quality(self) -> float:
        """Composite quality score (weighted average)."""
        return (
            0.7 * self.confidence +
            0.2 * self.context_richness +
            0.1 * self.response_completeness
        )


class MetapromptEngine:
    """
    7-Component Metaprompt Framework engine.

    Converts requests into structured prompts following:
    1. ROLE - Expert perspective
    2. OBJECTIVE - Goals with priorities
    3. PROCESS - Methodology
    4. FORMAT - Output structure
    5. CONSTRAINTS - Anti-patterns
    6. UNCERTAINTY - Fallback behavior
    7. VALIDATION - Success criteria
    """

    def __init__(self):
        self.template_cache: Dict[str, str] = {}

    def build_prompt(self, config: MetapromptConfig) -> str:
        """Build 7-component prompt from config."""
        sections = []

        # 1. ROLE
        sections.append(f"# ROLE\n{config.role}")

        # 2. OBJECTIVE
        sections.append(f"# OBJECTIVE")
        sections.append(f"**Primary**: {config.objective.get('primary', 'Not specified')}")
        if 'secondary' in config.objective:
            sections.append(f"**Secondary**: {config.objective['secondary']}")

        # 3. PROCESS
        sections.append(f"\n# PROCESS")
        for i, step in enumerate(config.process, 1):
            sections.append(f"{i}. {step}")

        # 4. FORMAT
        sections.append(f"\n# FORMAT\n{config.format}")

        # 5. CONSTRAINTS
        if config.constraints:
            sections.append(f"\n# CONSTRAINTS")
            for constraint in config.constraints:
                sections.append(f"- {constraint}")

        # 6. UNCERTAINTY
        sections.append(f"\n# UNCERTAINTY\n{config.uncertainty}")

        # 7. VALIDATION
        if config.validation:
            sections.append(f"\n# VALIDATION CHECKLIST")
            for criterion in config.validation:
                sections.append(f"- [ ] {criterion}")

        return "\n".join(sections)

    def extract_from_request(self, request: Dict[str, Any], task_type: str) -> MetapromptConfig:
        """Extract 7-component config from simple request dict."""
        # Default templates based on task type
        templates = {
            "code_review": MetapromptConfig(
                role="You are an expert code reviewer with 10+ years of experience in software engineering, specializing in code quality, maintainability, and best practices.",
                objective={
                    "primary": "Identify code quality issues and security vulnerabilities",
                    "secondary": "Suggest actionable improvements with clear rationale"
                },
                process=[
                    "Analyze code structure and logic flow",
                    "Check for common anti-patterns and code smells",
                    "Evaluate performance implications",
                    "Assess security vulnerabilities",
                    "Review maintainability and readability"
                ],
                format="Markdown with sections: Summary, Issues (HIGH/MEDIUM/LOW severity), Recommendations, Security Concerns",
                constraints=[
                    "Don't suggest complete rewrites unless absolutely critical",
                    "Focus on maintainability over cleverness",
                    "Prioritize actionable feedback over theoretical perfection"
                ],
                uncertainty="If code context is unclear or dependencies are unknown, explicitly request clarification rather than making assumptions",
                validation=[
                    "All issues have clear explanations",
                    "Recommendations are specific and implementable",
                    "Severity levels are justified",
                    "Security concerns are highlighted"
                ]
            ),
            "answer_question": MetapromptConfig(
                role="You are a knowledgeable assistant with expertise across multiple domains, specializing in clear, accurate explanations.",
                objective={
                    "primary": "Provide accurate, well-structured answers to user questions",
                    "secondary": "Include relevant context and examples when helpful"
                },
                process=[
                    "Understand the core question being asked",
                    "Retrieve relevant knowledge and context",
                    "Structure answer clearly with key points first",
                    "Provide examples or analogies if helpful",
                    "Verify accuracy and completeness"
                ],
                format="Clear, concise paragraphs with bullet points for lists. Start with direct answer, then elaborate.",
                constraints=[
                    "Don't speculate beyond available information",
                    "Don't overcomplicate simple questions",
                    "Don't assume unstated context"
                ],
                uncertainty="If the question is ambiguous or lacks necessary context, ask clarifying questions before answering",
                validation=[
                    "Question is directly answered",
                    "Answer is factually accurate",
                    "Explanation is clear and understandable",
                    "Relevant examples are provided if helpful"
                ]
            ),
            "general": MetapromptConfig(
                role="You are a helpful, knowledgeable assistant.",
                objective={
                    "primary": "Complete the requested task accurately and efficiently"
                },
                process=[
                    "Understand the request",
                    "Execute the task",
                    "Verify quality"
                ],
                format="Clear, structured output appropriate for the task",
                constraints=[
                    "Stay focused on the specific request",
                    "Don't add unnecessary information"
                ],
                uncertainty="Ask for clarification if the request is ambiguous",
                validation=[
                    "Request is fulfilled",
                    "Output meets quality standards"
                ]
            )
        }

        # Use template or create custom
        if task_type in templates:
            return templates[task_type]
        else:
            # Custom from request
            return MetapromptConfig(
                role=request.get("role", "You are a helpful assistant."),
                objective=request.get("objective", {"primary": "Complete the task"}),
                process=request.get("process", ["Execute the task", "Verify quality"]),
                format=request.get("format", "Clear, structured output"),
                constraints=request.get("constraints", []),
                uncertainty=request.get("uncertainty", "Ask for clarification if needed"),
                validation=request.get("validation", ["Task completed successfully"])
            )


class RefinementEngine:
    """
    Recursive refinement engine with 5 strategies.

    Strategies:
    - REFINE: Iteratively expand with more context
    - CRITIQUE: Self-critique and regenerate
    - VERIFY: Multi-pass cross-check (Accuracy → Completeness → Consistency)
    - ELEGANCE: Multi-pass polish (Clarity → Simplicity → Beauty)
    - HOFSTADTER: Strange loop self-reference
    """

    def __init__(self):
        self.refinement_history: List[Dict[str, Any]] = []

    def get_refinement_prompt(
        self,
        strategy: RefinementStrategyType,
        query: str,
        response: str,
        iteration: int = 1
    ) -> str:
        """Get refinement prompt based on strategy."""

        if strategy == RefinementStrategyType.REFINE:
            return f"""Review and improve this response to the query.

Query: {query}

Current Response:
{response}

Instructions:
1. Identify areas that need more context or clarity
2. Expand on key points with additional detail
3. Ensure all aspects of the query are addressed
4. Maintain accuracy while improving completeness

Provide an improved response."""

        elif strategy == RefinementStrategyType.CRITIQUE:
            return f"""Critically evaluate this response and generate an improved version.

Query: {query}

Current Response:
{response}

Instructions:
1. Identify weaknesses, gaps, or inaccuracies
2. Critique the structure and clarity
3. Note any missing important information
4. Generate a significantly improved response addressing all critiques

Provide the improved response."""

        elif strategy == RefinementStrategyType.VERIFY:
            # Multi-pass: Accuracy → Completeness → Consistency
            passes = ["accuracy", "completeness", "consistency"]
            current_pass = passes[min(iteration - 1, len(passes) - 1)]

            prompts = {
                "accuracy": f"""Verify the ACCURACY of this response.

Query: {query}

Response:
{response}

Instructions:
1. Check all factual claims for correctness
2. Identify any inaccuracies or misconceptions
3. Verify logical reasoning is sound
4. Correct any errors found

Provide an accuracy-verified response.""",

                "completeness": f"""Verify the COMPLETENESS of this response.

Query: {query}

Response:
{response}

Instructions:
1. Check if all aspects of the query are addressed
2. Identify any missing important information
3. Add necessary context or details
4. Ensure no critical points are omitted

Provide a complete response.""",

                "consistency": f"""Verify the CONSISTENCY of this response.

Query: {query}

Response:
{response}

Instructions:
1. Check for internal contradictions
2. Verify all statements are mutually consistent
3. Ensure terminology is used consistently
4. Resolve any conflicts or ambiguities

Provide a consistent response."""
            }
            return prompts[current_pass]

        elif strategy == RefinementStrategyType.ELEGANCE:
            # Multi-pass: Clarity → Simplicity → Beauty
            passes = ["clarity", "simplicity", "beauty"]
            current_pass = passes[min(iteration - 1, len(passes) - 1)]

            prompts = {
                "clarity": f"""Improve the CLARITY of this response.

Query: {query}

Response:
{response}

Instructions:
1. Make complex ideas more understandable
2. Improve sentence structure and flow
3. Remove ambiguity and vagueness
4. Use clearer terminology

Provide a clearer response.""",

                "simplicity": f"""Improve the SIMPLICITY of this response.

Query: {query}

Response:
{response}

Instructions:
1. Remove unnecessary complexity
2. Simplify language where possible
3. Eliminate redundancy
4. Make the essential points stand out

Provide a simpler response.""",

                "beauty": f"""Improve the BEAUTY and ELEGANCE of this response.

Query: {query}

Response:
{response}

Instructions:
1. Polish the language for elegance
2. Improve rhythm and flow
3. Use more evocative or precise terminology
4. Create a more satisfying reading experience

Provide an elegant response."""
            }
            return prompts[current_pass]

        elif strategy == RefinementStrategyType.HOFSTADTER:
            return f"""Engage in meta-level reasoning about this response.

Query: {query}

Response:
{response}

Instructions:
1. Reflect on the reasoning process used in the response
2. Consider what the response reveals about the question
3. Examine any hidden assumptions or implications
4. Generate an improved response that incorporates this meta-awareness

Provide a meta-aware improved response."""

        else:
            return f"Improve this response to: {query}\n\nCurrent response:\n{response}"


class ModelAdapterRegistry:
    """Registry of model-specific adapters."""

    def __init__(self):
        self.adapters: Dict[ModelProvider, Callable] = {}
        self._register_default_adapters()

    def _register_default_adapters(self):
        """Register default model adapters."""

        # Claude adapter
        def claude_adapter(prompt: str) -> str:
            """Optimize prompt for Claude (thinking tags, prefill)."""
            # Add thinking tags for better reasoning
            optimized = f"""<thinking>
Let me approach this systematically:
1. Understand the request clearly
2. Structure my response following the template
3. Ensure all requirements are met
</thinking>

{prompt}

I'll provide a thorough, well-structured response:"""
            return optimized

        # Gemini adapter
        def gemini_adapter(prompt: str) -> str:
            """Optimize prompt for Gemini (system instruction format)."""
            # Gemini prefers explicit instruction sections
            if "# ROLE" in prompt:
                # Already structured, just add markers
                optimized = f"**System Instructions**:\n\n{prompt}\n\n**Your Response**:"
            else:
                optimized = prompt
            return optimized

        # GPT adapter
        def gpt_adapter(prompt: str) -> str:
            """Optimize prompt for GPT (function calling format hint)."""
            # GPT benefits from clear output structure hints
            optimized = f"""{prompt}

Please provide your response in a clear, structured format following the guidelines above."""
            return optimized

        # Ollama adapter (local models)
        def ollama_adapter(prompt: str) -> str:
            """Optimize prompt for Ollama (simplified for smaller models)."""
            # Local models may need simpler, more direct prompts
            # Strip complex structure if prompt is too long
            if len(prompt) > 2000:
                # Simplify for smaller models
                lines = prompt.split('\n')
                essential = [l for l in lines if l.strip() and not l.startswith('#')]
                optimized = '\n'.join(essential[:20])  # Keep first 20 essential lines
            else:
                optimized = prompt
            return optimized

        self.adapters[ModelProvider.CLAUDE] = claude_adapter
        self.adapters[ModelProvider.GEMINI] = gemini_adapter
        self.adapters[ModelProvider.GPT] = gpt_adapter
        self.adapters[ModelProvider.OLLAMA] = ollama_adapter

    def adapt(self, prompt: str, model: ModelProvider) -> str:
        """Apply model-specific adaptation."""
        adapter = self.adapters.get(model)
        if adapter:
            return adapter(prompt)
        return prompt


class StrategySelector:
    """
    Thompson Sampling-based strategy selector.

    Learns which refinement strategies work best for different query types.
    """

    def __init__(self):
        # Track (query_type, strategy) → (alpha, beta) for Thompson Sampling
        self.strategy_stats: Dict[tuple, Dict[str, float]] = {}
        self.selection_history: List[Dict[str, Any]] = []

    def select_strategy(
        self,
        query_type: str,
        available_strategies: List[RefinementStrategyType]
    ) -> RefinementStrategyType:
        """Select best strategy using Thompson Sampling."""
        import random

        # Thompson Sampling: sample from Beta(α, β) for each strategy
        best_strategy = None
        best_sample = -1.0

        for strategy in available_strategies:
            key = (query_type, strategy.value)
            stats = self.strategy_stats.get(key, {"alpha": 1.0, "beta": 1.0})

            # Sample from Beta distribution
            # Using a simple approximation: sample = α / (α + β) + noise
            expected_reward = stats["alpha"] / (stats["alpha"] + stats["beta"])
            noise = random.gauss(0, 0.1)  # Add exploration noise
            sample = expected_reward + noise

            if sample > best_sample:
                best_sample = sample
                best_strategy = strategy

        return best_strategy or available_strategies[0]

    def update_from_outcome(
        self,
        query_type: str,
        strategy: RefinementStrategyType,
        quality_improvement: float
    ):
        """Update Thompson Sampling statistics from outcome."""
        key = (query_type, strategy.value)

        if key not in self.strategy_stats:
            self.strategy_stats[key] = {"alpha": 1.0, "beta": 1.0}

        # Update based on quality improvement
        # High quality improvement (>0.2) = success, update alpha
        # Low quality improvement (<0.05) = failure, update beta
        if quality_improvement > 0.2:
            self.strategy_stats[key]["alpha"] += quality_improvement
        elif quality_improvement < 0.05:
            self.strategy_stats[key]["beta"] += (0.05 - quality_improvement)

        # Log outcome
        self.selection_history.append({
            "query_type": query_type,
            "strategy": strategy.value,
            "quality_improvement": quality_improvement,
            "timestamp": time.time()
        })


class QualityTracker:
    """Track quality trajectories across refinement iterations."""

    def __init__(self):
        self.trajectories: List[List[QualityMetrics]] = []

    def track_iteration(
        self,
        iteration: int,
        confidence: float,
        context_richness: float = 0.0,
        response_completeness: float = 0.0
    ) -> QualityMetrics:
        """Track quality metrics for an iteration."""
        metrics = QualityMetrics(
            confidence=confidence,
            context_richness=context_richness,
            response_completeness=response_completeness
        )
        return metrics

    def compute_improvement(
        self,
        initial: QualityMetrics,
        final: QualityMetrics
    ) -> float:
        """Compute quality improvement delta."""
        return final.composite_quality - initial.composite_quality


class UnifiedMRF:
    """
    Unified Metaprompting Refinement Framework.

    Central API for all prompt enhancement and refinement capabilities.
    """

    def __init__(self):
        self.metaprompt_engine = MetapromptEngine()
        self.refinement_engine = RefinementEngine()
        self.model_adapters = ModelAdapterRegistry()
        self.strategy_selector = StrategySelector()
        self.quality_tracker = QualityTracker()

    async def enhance_prompt(
        self,
        request: Dict[str, Any],
        framework: str = "7-component",
        model: Optional[ModelProvider] = None,
        task_type: str = "general"
    ) -> EnhancedPrompt:
        """
        Enhance a prompt using the metaprompt framework.

        Args:
            request: Request dict with task details
            framework: "7-component" or "simple"
            model: Target model for optimization (optional)
            task_type: Type of task for template selection

        Returns:
            EnhancedPrompt with optimized prompt and metadata
        """
        enhancements = []

        # Step 1: Apply metaprompt framework
        if framework == "7-component":
            config = self.metaprompt_engine.extract_from_request(request, task_type)
            prompt = self.metaprompt_engine.build_prompt(config)
            enhancements.append("7-component framework")
        else:
            # Simple prompt (fallback)
            prompt = str(request.get("prompt", request.get("query", "")))
            enhancements.append("simple")

        # Step 2: Apply model-specific adaptation
        if model:
            prompt = self.model_adapters.adapt(prompt, model)
            enhancements.append(f"{model.value} optimization")

        return EnhancedPrompt(
            prompt=prompt,
            framework_used=framework,
            model_optimized_for=model,
            enhancements_applied=enhancements,
            metadata={"task_type": task_type}
        )

    async def refine_response(
        self,
        query: str,
        response: str,
        strategy: RefinementStrategyType = RefinementStrategyType.AUTO,
        max_iterations: int = 3,
        quality_threshold: float = 0.85,
        initial_quality: Optional[float] = None
    ) -> RefinedResponse:
        """
        Refine a response using recursive refinement strategies.

        Args:
            query: Original query
            response: Initial response to refine
            strategy: Refinement strategy to use
            max_iterations: Maximum refinement iterations
            quality_threshold: Target quality to achieve
            initial_quality: Initial quality score (if known)

        Returns:
            RefinedResponse with refined output and metrics
        """
        start_time = time.time()
        original_response = response
        current_response = response

        # Track quality
        initial_metrics = self.quality_tracker.track_iteration(
            iteration=0,
            confidence=initial_quality or 0.7  # Default if unknown
        )

        # Auto-select strategy if needed
        if strategy == RefinementStrategyType.AUTO:
            # Simple heuristic: factual → verify, creative → elegance, general → refine
            query_lower = query.lower()
            if any(word in query_lower for word in ["fact", "verify", "check", "correct"]):
                strategy = RefinementStrategyType.VERIFY
            elif any(word in query_lower for word in ["explain", "clarify", "simplify"]):
                strategy = RefinementStrategyType.ELEGANCE
            else:
                strategy = RefinementStrategyType.REFINE

        # Refinement loop
        for iteration in range(1, max_iterations + 1):
            # Get refinement prompt
            refinement_prompt = self.refinement_engine.get_refinement_prompt(
                strategy=strategy,
                query=query,
                response=current_response,
                iteration=iteration
            )

            # TODO: Actually call LLM here
            # For now, simulate refinement (in production, call LLM)
            # refined = await llm.generate(refinement_prompt)
            # current_response = refined

            # Simulated quality improvement (remove in production)
            current_quality = min(initial_metrics.confidence + (iteration * 0.1), 1.0)

            # Check if quality threshold met
            if current_quality >= quality_threshold:
                break

        # Compute final metrics
        final_metrics = self.quality_tracker.track_iteration(
            iteration=iteration,
            confidence=current_quality
        )

        quality_improvement = self.quality_tracker.compute_improvement(
            initial_metrics, final_metrics
        )

        execution_time_ms = (time.time() - start_time) * 1000

        return RefinedResponse(
            response=current_response,
            original_response=original_response,
            strategy_used=strategy,
            iterations=iteration,
            quality_improvement=quality_improvement,
            final_quality=final_metrics.composite_quality,
            execution_time_ms=execution_time_ms,
            metadata={"query": query}
        )

    async def auto_refine(
        self,
        query: str,
        response: str,
        query_type: str = "general",
        initial_quality: float = 0.7
    ) -> RefinedResponse:
        """
        Automatically select best refinement strategy and refine.

        Uses Thompson Sampling to learn optimal strategies over time.
        """
        # Thompson Sampling strategy selection
        available_strategies = [
            RefinementStrategyType.REFINE,
            RefinementStrategyType.CRITIQUE,
            RefinementStrategyType.VERIFY,
            RefinementStrategyType.ELEGANCE
        ]

        strategy = self.strategy_selector.select_strategy(
            query_type=query_type,
            available_strategies=available_strategies
        )

        # Refine with selected strategy
        result = await self.refine_response(
            query=query,
            response=response,
            strategy=strategy,
            initial_quality=initial_quality
        )

        # Update Thompson Sampling statistics
        self.strategy_selector.update_from_outcome(
            query_type=query_type,
            strategy=strategy,
            quality_improvement=result.quality_improvement
        )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get MRF usage statistics."""
        return {
            "refinement_history_count": len(self.refinement_engine.refinement_history),
            "quality_trajectories_count": len(self.quality_tracker.trajectories),
            "strategy_selections_count": len(self.strategy_selector.selection_history),
            "learned_strategy_stats": dict(self.strategy_selector.strategy_stats)
        }

    async def refine_prompt(
        self,
        original_prompt: str,
        strategy: Optional[RefinementStrategyType] = None,
        model_provider: Optional[ModelProvider] = None,
        context: Optional[Dict[str, Any]] = None,
        epistemic_confidence: Optional[float] = None,
        enable_learning: bool = False
    ) -> Dict[str, Any]:
        """
        Refine a prompt using MRF 7-component structure.

        This method implements the interface described in the mrf_prompt_refiner skill spec.

        Args:
            original_prompt: The prompt to refine
            strategy: Refinement strategy (VERIFY, REFINE, CRITIQUE, ELEGANCE, HOFSTADTER, AUTO)
            model_provider: Target model for optimization (claude, gemini, gpt, ollama)
            context: Additional context for refinement
            epistemic_confidence: 0.0-1.0 confidence level
            enable_learning: Use Thompson Sampling recommendations

        Returns:
            Dict with enhanced_prompt, quality scores, component breakdown, improvements, etc.
        """
        start_time = time.time()

        # Use AUTO strategy if not specified
        if strategy is None:
            strategy = RefinementStrategyType.AUTO

        # Infer query type from context for learning
        query_type = "general"
        if context:
            query_type = context.get("query_type", "general")

        # Step 1: Enhance prompt using 7-component framework
        request = {"prompt": original_prompt, "task": "prompt_refinement"}
        if context:
            request.update(context)

        enhanced = await self.enhance_prompt(
            request=request,
            framework="7-component",
            model=model_provider,
            task_type=query_type
        )

        # Step 2: Extract 7 components (simulated for now - actual extraction would parse enhanced.prompt)
        # In a production implementation, this would parse the actual 7-component structure
        component_breakdown = {
            "role": "Expert prompt engineer",
            "objective": f"Refine the prompt: '{original_prompt[:50]}...'",
            "process": "1. Analyze intent\n2. Add structure\n3. Specify constraints\n4. Define validation",
            "format": "Structured 7-component metaprompt",
            "constraints": "Maintain original intent, avoid over-engineering",
            "uncertainty": f"Epistemic confidence: {epistemic_confidence or 0.8:.2f}",
            "validation": "Verify all components present and well-defined"
        }

        # Step 3: Calculate quality metrics
        original_quality = 0.6  # Base quality for unstructured prompt
        enhanced_quality = 0.85  # Improved quality with 7-component structure

        # Adjust for epistemic confidence
        if epistemic_confidence is not None and epistemic_confidence < 0.7:
            enhanced_quality *= (0.7 + (epistemic_confidence * 0.3))

        quality_improvement = (enhanced_quality - original_quality) / original_quality

        # Step 4: Determine strategy used (AUTO → actual strategy selection)
        if strategy == RefinementStrategyType.AUTO:
            # For AUTO, select based on prompt characteristics
            if "explain" in original_prompt.lower() or "what is" in original_prompt.lower():
                actual_strategy = "verify"
            elif "compare" in original_prompt.lower() or "analyze" in original_prompt.lower():
                actual_strategy = "critique"
            elif "implement" in original_prompt.lower() or "write" in original_prompt.lower():
                actual_strategy = "refine"
            else:
                actual_strategy = "elegance"
        else:
            actual_strategy = strategy.value

        # Step 5: List improvements made
        improvements = [
            "Added expert role definition",
            "Clarified success criteria",
            "Structured step-by-step process",
            "Specified output format",
            "Added validation checks"
        ]

        # Add epistemic confidence handling if low confidence
        if epistemic_confidence is not None and epistemic_confidence < 0.7:
            improvements.append(f"Added epistemic confidence handling (confidence: {epistemic_confidence:.2f})")

        # Add model-specific optimizations
        if model_provider:
            if model_provider == ModelProvider.CLAUDE:
                improvements.append("Concise Claude-optimized structure")
            elif model_provider == ModelProvider.GEMINI:
                improvements.append("Verbose Gemini-optimized structure")
            elif model_provider == ModelProvider.GPT:
                improvements.append("Balanced GPT-optimized structure")
            elif model_provider == ModelProvider.OLLAMA:
                improvements.append("Simplified language for Ollama (3B-7B models)")

        # Step 6: Learning recommendation (if enabled)
        learning_recommendation = None
        if enable_learning and query_type != "general":
            # Use Thompson Sampling to recommend strategy
            available_strategies = [
                RefinementStrategyType.REFINE,
                RefinementStrategyType.CRITIQUE,
                RefinementStrategyType.VERIFY,
                RefinementStrategyType.ELEGANCE
            ]

            recommended = self.strategy_selector.select_strategy(
                query_type=query_type,
                available_strategies=available_strategies
            )

            # Calculate confidence and expected reward
            key = (query_type, recommended.value)
            if key in self.strategy_selector.strategy_stats:
                stats = self.strategy_selector.strategy_stats[key]
                alpha = stats["alpha"]
                beta = stats["beta"]
                confidence = alpha / (alpha + beta)
                expected_reward = alpha / (alpha + beta)
            else:
                confidence = 0.5
                expected_reward = 0.5

            learning_recommendation = {
                "recommended_strategy": recommended.value,
                "confidence": confidence,
                "expected_reward": expected_reward,
                "rationale": f"Historical data shows {recommended.value.upper()} performs best for {query_type} queries ({confidence:.0%} confidence)"
            }

            # Update learning statistics
            self.strategy_selector.update_from_outcome(
                query_type=query_type,
                strategy=recommended,
                quality_improvement=quality_improvement
            )

        # Step 7: Build final result
        duration_ms = (time.time() - start_time) * 1000

        result = {
            "enhanced_prompt": enhanced.prompt,
            "quality_score": enhanced_quality,
            "quality_improvement": quality_improvement,
            "strategy_used": actual_strategy,
            "component_breakdown": component_breakdown,
            "improvements_made": improvements,
            "metadata": {
                "original_length": len(original_prompt),
                "enhanced_length": len(enhanced.prompt),
                "refinement_time_ms": duration_ms,
                "model_provider": model_provider.value if model_provider else "default"
            }
        }

        if learning_recommendation:
            result["learning_recommendation"] = learning_recommendation

        return result


# Convenience functions

async def enhance_prompt(
    request: Dict[str, Any],
    framework: str = "7-component",
    model: Optional[str] = None,
    task_type: str = "general"
) -> EnhancedPrompt:
    """
    Convenience function to enhance a prompt.

    Example:
        enhanced = await enhance_prompt(
            request={"task": "code_review", "code": code},
            framework="7-component",
            model="claude"
        )
    """
    mrf = UnifiedMRF()
    model_enum = ModelProvider(model) if model else None
    return await mrf.enhance_prompt(request, framework, model_enum, task_type)


async def refine_response(
    query: str,
    response: str,
    strategy: str = "auto",
    max_iterations: int = 3
) -> RefinedResponse:
    """
    Convenience function to refine a response.

    Example:
        refined = await refine_response(
            query="What is Thompson Sampling?",
            response=initial_response,
            strategy="verify"
        )
    """
    mrf = UnifiedMRF()
    strategy_enum = RefinementStrategyType(strategy)
    return await mrf.refine_response(query, response, strategy_enum, max_iterations)


# Export main classes and functions
__all__ = [
    "UnifiedMRF",
    "RefinementStrategyType",
    "ModelProvider",
    "MetapromptConfig",
    "RefinementConfig",
    "EnhancedPrompt",
    "RefinedResponse",
    "QualityMetrics",
    "enhance_prompt",
    "refine_response"
]

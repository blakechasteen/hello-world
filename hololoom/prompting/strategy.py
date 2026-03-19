"""
Base Strategy Interface

Defines the common interface that all prompting strategies must implement.
This enables extensibility, composability, and auto-detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrategyCategory(Enum):
    """Categories for organizing strategies"""
    SELF_CORRECTION = "self-correction"
    META_PROMPTING = "meta-prompting"
    SCAFFOLDING = "scaffolding"
    PERSPECTIVE = "perspective"
    EDGE_CASES = "edge-cases"
    COMPOSITE = "composite"


@dataclass
class StrategyContext:
    """
    Context passed to all strategies.

    Contains the query and all relevant context that strategies
    might need to make decisions or enhance the prompt.
    """
    query: str
    user_context: dict[str, Any] | None = None
    selection: str | None = None
    file_path: str | None = None
    previous_result: Any | None = None

    # Metadata about the request
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_query(self, new_query: str) -> 'StrategyContext':
        """Create new context with updated query"""
        return StrategyContext(
            query=new_query,
            user_context=self.user_context,
            selection=self.selection,
            file_path=self.file_path,
            previous_result=self.previous_result,
            metadata=self.metadata.copy()
        )


@dataclass
class StrategyResult:
    """
    Result from strategy execution.

    Contains the enhanced query plus metadata about the enhancement.
    """
    enhanced_query: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Control flow
    requires_followup: bool = False
    followup_strategies: list[str] | None = None

    # Quality metrics
    confidence: float = 1.0  # 0-1, how confident is this enhancement?
    estimated_improvement: float | None = None  # Estimated quality boost

    def with_metadata(self, **kwargs) -> 'StrategyResult':
        """Add metadata to result"""
        new_metadata = self.metadata.copy()
        new_metadata.update(kwargs)
        return StrategyResult(
            enhanced_query=self.enhanced_query,
            metadata=new_metadata,
            requires_followup=self.requires_followup,
            followup_strategies=self.followup_strategies,
            confidence=self.confidence,
            estimated_improvement=self.estimated_improvement
        )


class PromptingStrategy(ABC):
    """
    Base class for all prompting strategies.

    Each strategy implements:
    1. enhance() - Core transformation logic
    2. can_apply() - Auto-detection scoring
    3. Metadata properties (name, category, description)

    Strategies are composable via the CompositeStrategy class.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy identifier (e.g., 'verify', 'challenge', 'debate').
        Used in commands: /strategy <name> <query>
        """
        pass

    @property
    @abstractmethod
    def category(self) -> StrategyCategory:
        """
        Strategy category for organization.
        Examples: SELF_CORRECTION, META_PROMPTING, SCAFFOLDING
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this strategy does.
        Shown in /strategy list and auto-detection suggestions.
        """
        pass

    @abstractmethod
    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """
        Apply strategy to enhance the query.

        Args:
            context: Query and relevant context

        Returns:
            StrategyResult with enhanced query and metadata

        Example:
            >>> strategy = VerifyStrategy()
            >>> context = StrategyContext(query="analyze this contract")
            >>> result = await strategy.enhance(context)
            >>> print(result.enhanced_query)
            "Perform chain of verification analysis on this contract..."
        """
        pass

    @abstractmethod
    def can_apply(self, context: StrategyContext) -> float:
        """
        Return confidence (0-1) that this strategy is appropriate.

        Used by auto-detection to suggest strategies. Higher scores
        mean the strategy is more likely to be useful for this query.

        Args:
            context: Query and relevant context

        Returns:
            Confidence score 0-1

        Example:
            >>> strategy = VerifyStrategy()
            >>> context = StrategyContext(query="analyze this contract")
            >>> confidence = strategy.can_apply(context)
            >>> print(confidence)  # High confidence for analysis tasks
            0.92
        """
        pass

    def compose_with(self, other: 'PromptingStrategy') -> 'CompositeStrategy':
        """
        Compose this strategy with another into a pipeline.

        Args:
            other: Strategy to chain after this one

        Returns:
            CompositeStrategy that runs both in sequence

        Example:
            >>> verify = VerifyStrategy()
            >>> challenge = ChallengeStrategy()
            >>> combined = verify.compose_with(challenge)
            >>> # Now: verify -> challenge pipeline
        """
        from .composite import CompositeStrategy
        return CompositeStrategy([self, other])

    def __add__(self, other: 'PromptingStrategy') -> 'CompositeStrategy':
        """
        Syntactic sugar for composition: strategy1 + strategy2

        Example:
            >>> combined = VerifyStrategy() + ChallengeStrategy()
        """
        return self.compose_with(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class TemplateStrategy(PromptingStrategy):
    """
    Base class for template-based strategies.

    Subclasses just need to provide a template and detection rules.
    The template is formatted with query context and returned.
    """

    def __init__(self, template: str, config: dict[str, Any] | None = None):
        """
        Initialize template strategy.

        Args:
            template: Prompt template with {variable} placeholders
            config: Optional configuration dict
        """
        self.template = template
        self.config = config or {}

    async def enhance(self, context: StrategyContext) -> StrategyResult:
        """Apply template to query"""
        # Format template with context
        variables = {
            'query': context.query,
            'selection': context.selection or '',
            'file_path': context.file_path or '',
            **self.config
        }

        enhanced = self.template.format(**variables)

        return StrategyResult(
            enhanced_query=enhanced,
            metadata={
                'strategy': self.name,
                'category': self.category.value,
                'template_variables': list(variables.keys())
            }
        )


class LLMPoweredStrategy(PromptingStrategy):
    """
    Base class for strategies that use an LLM for enhancement.

    Example: Reverse prompting, where we ask the model to design
    the optimal prompt rather than using a template.
    """

    def __init__(self, orchestrator=None):
        """
        Initialize LLM-powered strategy.

        Args:
            orchestrator: Optional WeavingOrchestrator for LLM calls
        """
        self.orchestrator = orchestrator

    async def _call_llm(self, prompt: str) -> str:
        """
        Make LLM call for enhancement.

        Args:
            prompt: Meta-prompt for enhancement

        Returns:
            LLM response
        """
        if not self.orchestrator:
            # Lazy import to avoid circular dependency
            from hololoom.config import Config
            from hololoom.weaving_orchestrator import WeavingOrchestrator
            self.orchestrator = WeavingOrchestrator(cfg=Config.fast())

        from hololoom.protocols.types import Query
        spacetime = await self.orchestrator.weave(Query(text=prompt))
        return spacetime.response

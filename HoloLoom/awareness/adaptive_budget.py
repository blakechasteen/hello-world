#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Token Budget Calculator

Dynamically calculates optimal token budgets based on:
1. Query complexity (simple → research)
2. Model context window (GPT-4: 128k, Claude: 200k, Gemini: 1M)
3. Uncertainty level (high uncertainty → more context)
4. Available memories (many memories → larger budget)
5. Previous performance (learning from outcomes)

This is Phase 2 of the Context Packer roadmap.

Usage:
    calculator = AdaptiveBudgetCalculator(
        model_name="claude-3-5-sonnet-20241022",
        enable_learning=True
    )

    budget = await calculator.calculate_budget(
        query="What is quantum tunneling?",
        awareness_context=awareness_ctx,
        available_memories=10
    )

    print(budget.total)  # 4500 tokens (adaptive)
    print(budget.reasoning)  # Why this budget was chosen
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

# Import complexity analyzer
from .query_complexity import QueryComplexityAnalyzer, ComplexityLevel
from .context_packer import TokenBudget


logger = logging.getLogger(__name__)


# ============================================================================
# Model Registry - Context Window Database
# ============================================================================

@dataclass
class ModelInfo:
    """Information about an LLM model"""
    name: str
    provider: str
    context_window: int  # Maximum tokens
    recommended_max: int  # Recommended max (leave room for response)
    cost_per_1m_input: float  # USD per 1M input tokens
    cost_per_1m_output: float  # USD per 1M output tokens


# Model registry (as of November 2025)
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": ModelInfo(
        name="claude-3-5-sonnet-20241022",
        provider="anthropic",
        context_window=200_000,
        recommended_max=180_000,  # Leave 20k for response
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0
    ),
    "claude-3-opus-20240229": ModelInfo(
        name="claude-3-opus-20240229",
        provider="anthropic",
        context_window=200_000,
        recommended_max=180_000,
        cost_per_1m_input=15.0,
        cost_per_1m_output=75.0
    ),

    # OpenAI GPT models
    "gpt-4-turbo-preview": ModelInfo(
        name="gpt-4-turbo-preview",
        provider="openai",
        context_window=128_000,
        recommended_max=120_000,
        cost_per_1m_input=10.0,
        cost_per_1m_output=30.0
    ),
    "gpt-4": ModelInfo(
        name="gpt-4",
        provider="openai",
        context_window=8_192,
        recommended_max=7_000,
        cost_per_1m_input=30.0,
        cost_per_1m_output=60.0
    ),
    "gpt-3.5-turbo": ModelInfo(
        name="gpt-3.5-turbo",
        provider="openai",
        context_window=16_385,
        recommended_max=15_000,
        cost_per_1m_input=0.5,
        cost_per_1m_output=1.5
    ),

    # Google Gemini models
    "gemini-1.5-pro": ModelInfo(
        name="gemini-1.5-pro",
        provider="google",
        context_window=1_000_000,
        recommended_max=950_000,
        cost_per_1m_input=2.5,
        cost_per_1m_output=10.0
    ),

    # Ollama local models (free, smaller context)
    "llama3.2:3b": ModelInfo(
        name="llama3.2:3b",
        provider="ollama",
        context_window=8_192,
        recommended_max=7_000,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0
    ),
    "llama3:8b": ModelInfo(
        name="llama3:8b",
        provider="ollama",
        context_window=8_192,
        recommended_max=7_000,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0
    ),
    "mistral:7b": ModelInfo(
        name="mistral:7b",
        provider="ollama",
        context_window=32_768,
        recommended_max=30_000,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0
    ),
}


def get_model_info(model_name: str) -> ModelInfo:
    """
    Get model information from registry.

    Args:
        model_name: Model identifier

    Returns:
        ModelInfo with context window and pricing

    Raises:
        ValueError: If model not in registry
    """
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    # Try to infer from name
    if "claude" in model_name.lower():
        logger.warning(f"Unknown Claude model: {model_name}, using default")
        return MODEL_REGISTRY["claude-3-5-sonnet-20241022"]

    if "gpt-4" in model_name.lower():
        logger.warning(f"Unknown GPT-4 model: {model_name}, using default")
        return MODEL_REGISTRY["gpt-4-turbo-preview"]

    if "gpt-3.5" in model_name.lower():
        logger.warning(f"Unknown GPT-3.5 model: {model_name}, using default")
        return MODEL_REGISTRY["gpt-3.5-turbo"]

    if "gemini" in model_name.lower():
        logger.warning(f"Unknown Gemini model: {model_name}, using default")
        return MODEL_REGISTRY["gemini-1.5-pro"]

    # Default: assume small context window
    logger.warning(f"Unknown model: {model_name}, using conservative defaults")
    return ModelInfo(
        name=model_name,
        provider="unknown",
        context_window=8_192,
        recommended_max=7_000,
        cost_per_1m_input=5.0,
        cost_per_1m_output=15.0
    )


# ============================================================================
# Adaptive Budget Calculator
# ============================================================================

@dataclass
class AdaptiveBudget:
    """
    Adaptive token budget with reasoning.

    Contains:
    - Total budget (dynamic based on query/model)
    - Reserved tokens (query, response)
    - Available for context
    - Reasoning (why this budget was chosen)
    """

    # Budget breakdown
    total: int
    reserved_for_query: int
    reserved_for_response: int

    # Model constraints
    model_max: int  # Model's context window
    recommended_max: int  # Recommended max usage

    # Reasoning
    reasoning: List[str]  # Human-readable explanations
    adjustments: Dict[str, float]  # Factor → adjustment multiplier

    # Metadata
    query_complexity: ComplexityLevel
    complexity_score: float

    @property
    def available_for_context(self) -> int:
        """Tokens available for context"""
        return self.total - self.reserved_for_query - self.reserved_for_response

    def to_token_budget(self) -> TokenBudget:
        """Convert to standard TokenBudget"""
        return TokenBudget(
            total=self.total,
            reserved_for_query=self.reserved_for_query,
            reserved_for_response=self.reserved_for_response
        )

    def __str__(self) -> str:
        reasoning_str = "\n    ".join(self.reasoning)
        return (
            f"AdaptiveBudget(\n"
            f"  Total: {self.total:,} tokens\n"
            f"  Available for Context: {self.available_for_context:,}\n"
            f"  Complexity: {self.query_complexity.value} ({self.complexity_score:.2f})\n"
            f"  Reasoning:\n    {reasoning_str}\n"
            f")"
        )


class AdaptiveBudgetCalculator:
    """
    Calculate optimal token budgets dynamically.

    Strategy:
    1. Analyze query complexity (simple → research)
    2. Get model's context window
    3. Start with base budget from complexity
    4. Adjust for uncertainty (high → +30%)
    5. Adjust for available memories (many → +20%)
    6. Clamp to model's recommended max
    7. Learn from outcomes (if enabled)
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        min_budget: int = 2_000,
        max_budget: int = 32_000,
        enable_learning: bool = False
    ):
        """
        Initialize adaptive budget calculator.

        Args:
            model_name: Model identifier (for context window lookup)
            min_budget: Minimum budget (safety floor)
            max_budget: Maximum budget (cost ceiling)
            enable_learning: Enable learning from outcomes
        """
        self.model_name = model_name
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.enable_learning = enable_learning

        # Get model info
        self.model_info = get_model_info(model_name)

        # Clamp max_budget to model's recommended max
        self.max_budget = min(self.max_budget, self.model_info.recommended_max)

        # Initialize complexity analyzer
        self.complexity_analyzer = QueryComplexityAnalyzer()

        # Learning statistics (if enabled)
        self._budget_performance: Dict[int, List[float]] = {}  # budget → [quality_scores]

        logger.info(
            f"Adaptive budget calculator initialized: "
            f"model={model_name}, "
            f"context_window={self.model_info.context_window:,}, "
            f"budget_range=[{min_budget:,}, {max_budget:,}]"
        )

    def calculate_budget(
        self,
        query: str,
        awareness_context=None,
        available_memories: int = 0,
        reserved_for_response: Optional[int] = None
    ) -> AdaptiveBudget:
        """
        Calculate adaptive token budget.

        Args:
            query: User query
            awareness_context: Optional UnifiedAwarenessContext (for uncertainty)
            available_memories: Number of available memories
            reserved_for_response: Override response reservation (default: auto)

        Returns:
            AdaptiveBudget with reasoning
        """
        reasoning = []
        adjustments = {}

        # Step 1: Analyze query complexity
        complexity_analysis = self.complexity_analyzer.analyze(query, awareness_context)

        # Step 2: Start with base budget from complexity
        base_budget = complexity_analysis.recommended_budget
        reasoning.append(
            f"Base budget from complexity ({complexity_analysis.level.value}): "
            f"{base_budget:,} tokens"
        )

        current_budget = base_budget
        adjustments['base'] = 1.0

        # Step 3: Adjust for model capacity
        # If model has large context window, can use larger budgets
        if self.model_info.context_window >= 100_000:
            # Large models (Claude, Gemini) can use 1.5x budgets
            model_multiplier = 1.5
            current_budget = int(current_budget * model_multiplier)
            reasoning.append(
                f"Large context window ({self.model_info.context_window:,}): "
                f"×{model_multiplier} = {current_budget:,} tokens"
            )
            adjustments['model_capacity'] = model_multiplier

        # Step 4: Adjust for uncertainty (if available)
        if awareness_context and hasattr(awareness_context, 'confidence'):
            uncertainty = awareness_context.confidence.uncertainty_level

            if uncertainty > 0.7:
                # High uncertainty → need more context
                uncertainty_multiplier = 1.3
                current_budget = int(current_budget * uncertainty_multiplier)
                reasoning.append(
                    f"High uncertainty ({uncertainty:.2f}): "
                    f"×{uncertainty_multiplier} = {current_budget:,} tokens"
                )
                adjustments['uncertainty'] = uncertainty_multiplier

        # Step 5: Adjust for available memories
        if available_memories > 20:
            # Many memories → need larger budget to fit them
            memory_multiplier = 1.2
            current_budget = int(current_budget * memory_multiplier)
            reasoning.append(
                f"Many memories ({available_memories}): "
                f"×{memory_multiplier} = {current_budget:,} tokens"
            )
            adjustments['memory_count'] = memory_multiplier

        # Step 6: Clamp to min/max
        if current_budget < self.min_budget:
            reasoning.append(
                f"Clamped to minimum: {self.min_budget:,} tokens "
                f"(was {current_budget:,})"
            )
            current_budget = self.min_budget

        if current_budget > self.max_budget:
            reasoning.append(
                f"Clamped to maximum: {self.max_budget:,} tokens "
                f"(was {current_budget:,})"
            )
            current_budget = self.max_budget

        # Step 7: Reserve tokens for query and response
        # Query: ~500 tokens (5% of budget, min 300, max 1000)
        reserved_query = min(max(int(current_budget * 0.05), 300), 1000)

        # Response: provided or auto (15% of budget, min 500, max 2000)
        if reserved_for_response is None:
            reserved_response = min(max(int(current_budget * 0.15), 500), 2000)
        else:
            reserved_response = reserved_for_response

        # Final budget
        total_budget = current_budget

        return AdaptiveBudget(
            total=total_budget,
            reserved_for_query=reserved_query,
            reserved_for_response=reserved_response,
            model_max=self.model_info.context_window,
            recommended_max=self.model_info.recommended_max,
            reasoning=reasoning,
            adjustments=adjustments,
            query_complexity=complexity_analysis.level,
            complexity_score=complexity_analysis.score
        )

    def learn_from_outcome(
        self,
        budget: AdaptiveBudget,
        quality_score: float
    ) -> None:
        """
        Learn from budget outcome.

        Tracks: budget → quality correlation

        Args:
            budget: Budget that was used
            quality_score: Quality achieved (0.0-1.0)
        """
        if not self.enable_learning:
            return

        # Round budget to nearest 1000 for grouping
        budget_key = round(budget.total / 1000) * 1000

        if budget_key not in self._budget_performance:
            self._budget_performance[budget_key] = []

        self._budget_performance[budget_key].append(quality_score)

        # Prune old data (keep last 100 per budget)
        if len(self._budget_performance[budget_key]) > 100:
            self._budget_performance[budget_key] = self._budget_performance[budget_key][-100:]

    def get_optimal_budget(self, complexity_level: ComplexityLevel) -> Optional[int]:
        """
        Get empirically optimal budget for a complexity level.

        Returns:
            Optimal budget (from learning) or None if not enough data
        """
        if not self.enable_learning or not self._budget_performance:
            return None

        # Find budgets with best average quality
        budget_averages = {
            budget: sum(scores) / len(scores)
            for budget, scores in self._budget_performance.items()
            if len(scores) >= 5  # Require at least 5 samples
        }

        if not budget_averages:
            return None

        # Return budget with highest average quality
        optimal = max(budget_averages, key=budget_averages.get)
        return optimal

    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        if not self.enable_learning:
            return {"learning_enabled": False}

        stats = {
            "learning_enabled": True,
            "budgets_tracked": len(self._budget_performance),
            "total_samples": sum(len(scores) for scores in self._budget_performance.values())
        }

        # Budget performance summary
        if self._budget_performance:
            budget_summary = {}
            for budget, scores in self._budget_performance.items():
                if len(scores) >= 3:
                    budget_summary[budget] = {
                        "avg_quality": sum(scores) / len(scores),
                        "samples": len(scores)
                    }

            stats["budget_performance"] = budget_summary

        return stats

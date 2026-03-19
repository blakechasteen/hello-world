"""
LLM Cost Tracker
================

Tracks token usage and calculates costs for different LLM providers.

Pricing is kept up-to-date with provider rates (as of January 2025).

Created: 2025-01-20
"""

from dataclasses import dataclass


class ModelPricing:
    """
    Pricing tables for LLM models (USD per 1M tokens).

    Updated: 2025-01-20
    Source: Provider pricing pages
    """

    # Pricing format: {"input": price_per_1M, "output": price_per_1M}
    PRICING: dict[str, dict[str, float]] = {
        # Anthropic Claude
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},

        # OpenAI GPT
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

        # Google Gemini
        "gemini-pro": {"input": 0.5, "output": 1.5},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-ultra": {"input": 3.0, "output": 9.0},

        # Ollama (local - free)
        "ollama": {"input": 0.0, "output": 0.0},
        "llama3.2:3b": {"input": 0.0, "output": 0.0},
        "llama3.1:8b": {"input": 0.0, "output": 0.0},
        "mistral:7b": {"input": 0.0, "output": 0.0},
        "phi3:3.8b": {"input": 0.0, "output": 0.0},
    }

    # Default pricing for unknown models (conservative estimate)
    DEFAULT_PRICING = {"input": 10.0, "output": 30.0}

    @classmethod
    def get_pricing(cls, model: str) -> dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022")

        Returns:
            Dict with "input" and "output" prices per 1M tokens
        """
        # Direct match
        if model in cls.PRICING:
            return cls.PRICING[model]

        # Try partial match (e.g., "gpt-4-turbo-preview" → "gpt-4-turbo")
        for known_model in cls.PRICING:
            if model.startswith(known_model) or known_model in model:
                return cls.PRICING[known_model]

        # Unknown model - use default
        return cls.DEFAULT_PRICING

    @classmethod
    def is_free(cls, model: str) -> bool:
        """Check if model is free (local)"""
        pricing = cls.get_pricing(model)
        return pricing["input"] == 0.0 and pricing["output"] == 0.0


@dataclass
class CostEstimate:
    """Cost estimate for a query"""
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model: str
    is_free: bool


class CostTracker:
    """
    Track costs across multiple LLM calls.

    Usage:
        tracker = CostTracker()

        # Track a call
        tracker.track_call(
            model="claude-3-5-sonnet-20241022",
            input_tokens=1500,
            output_tokens=500
        )

        # Get total cost
        stats = tracker.get_statistics()
        print(f"Total cost: ${stats['total_cost_usd']:.4f}")
    """

    def __init__(self):
        """Initialize cost tracker"""
        self.calls: list[CostEstimate] = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> CostEstimate:
        """
        Calculate cost for a single call.

        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            CostEstimate with breakdown
        """
        pricing = ModelPricing.get_pricing(model)
        is_free = ModelPricing.is_free(model)

        # Calculate costs (per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        return CostEstimate(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=total_cost,
            model=model,
            is_free=is_free
        )

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> CostEstimate:
        """
        Track a call and update totals.

        Args:
            model: Model name
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            CostEstimate for this call
        """
        estimate = self.calculate_cost(model, input_tokens, output_tokens)

        # Update totals
        self.calls.append(estimate)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += estimate.total_cost_usd

        return estimate

    def get_statistics(self) -> dict:
        """
        Get cost statistics.

        Returns:
            Dict with:
            - total_calls: Number of calls tracked
            - total_input_tokens: Total input tokens
            - total_output_tokens: Total output tokens
            - total_tokens: Total tokens
            - total_cost_usd: Total cost in USD
            - avg_cost_per_call: Average cost per call
            - by_model: Breakdown by model
        """
        by_model = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost_usd": 0.0
                }
            by_model[call.model]["calls"] += 1
            by_model[call.model]["input_tokens"] += call.input_tokens
            by_model[call.model]["output_tokens"] += call.output_tokens
            by_model[call.model]["total_cost_usd"] += call.total_cost_usd

        avg_cost = self.total_cost_usd / len(self.calls) if self.calls else 0.0

        return {
            "total_calls": len(self.calls),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_call": avg_cost,
            "by_model": by_model
        }

    def reset(self):
        """Reset all statistics"""
        self.calls.clear()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    def get_recent_calls(self, n: int = 10) -> list[CostEstimate]:
        """Get N most recent calls"""
        return self.calls[-n:]

    def estimate_query_cost(
        self,
        query: str,
        model: str,
        max_output_tokens: int = 500
    ) -> CostEstimate:
        """
        Estimate cost for a query before running it.

        Args:
            query: Input query text
            model: Model to use
            max_output_tokens: Maximum output tokens

        Returns:
            CostEstimate (conservative estimate)
        """
        # Rough token estimation (4 chars ≈ 1 token)
        estimated_input_tokens = len(query) // 4

        return self.calculate_cost(
            model=model,
            input_tokens=estimated_input_tokens,
            output_tokens=max_output_tokens
        )

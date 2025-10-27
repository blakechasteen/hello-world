#!/usr/bin/env python3
"""
Promptly Cost Tracker
=====================
Track API costs and token usage for prompt executions.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum


class ModelProvider(Enum):
    """LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


# Pricing per 1M tokens (as of 2025)
PRICING = {
    ModelProvider.ANTHROPIC: {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-opus": {"input": 15.00, "output": 75.00},
    },
    ModelProvider.OPENAI: {
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    ModelProvider.OLLAMA: {
        # Free local models
        "default": {"input": 0.00, "output": 0.00}
    }
}


@dataclass
class ExecutionCost:
    """Cost for a single execution"""
    timestamp: str
    prompt_name: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CostSummary:
    """Summary of costs"""
    total_executions: int
    total_tokens: int
    total_cost: float
    by_model: Dict[str, Dict[str, Any]]
    by_prompt: Dict[str, Dict[str, Any]]
    date_range: Tuple[str, str]

    def to_report(self) -> str:
        """Generate cost report"""
        lines = [
            "# Cost Summary",
            "",
            f"**Total Executions:** {self.total_executions}",
            f"**Total Tokens:** {self.total_tokens:,}",
            f"**Total Cost:** ${self.total_cost:.4f}",
            "",
            "## By Model"
        ]

        for model, stats in self.by_model.items():
            lines.append(
                f"- **{model}**: {stats['executions']} runs, "
                f"{stats['tokens']:,} tokens, ${stats['cost']:.4f}"
            )

        lines.append("\n## By Prompt")
        for prompt, stats in self.by_prompt.items():
            lines.append(
                f"- **{prompt}**: {stats['executions']} runs, "
                f"${stats['cost']:.4f}"
            )

        return "\n".join(lines)


class CostTracker:
    """Track and analyze API costs"""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize cost tracker.

        Args:
            storage_path: Path to store cost data (default: .promptly/costs.json)
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".promptly" / "costs.json"

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.executions: List[ExecutionCost] = []
        self.load()

    def record_execution(
        self,
        prompt_name: str,
        model: str,
        provider: ModelProvider,
        input_tokens: int,
        output_tokens: int,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExecutionCost:
        """
        Record an execution and calculate cost.

        Returns:
            ExecutionCost with calculated costs
        """
        # Get pricing
        provider_pricing = PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model, provider_pricing.get("default", {"input": 0, "output": 0}))

        # Calculate costs (per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost

        execution = ExecutionCost(
            timestamp=datetime.now().isoformat(),
            prompt_name=prompt_name,
            model=model,
            provider=provider.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            execution_time=execution_time,
            metadata=metadata or {}
        )

        self.executions.append(execution)
        self.save()

        return execution

    def get_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        prompt_name: Optional[str] = None,
        model: Optional[str] = None
    ) -> CostSummary:
        """
        Get cost summary with optional filters.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            prompt_name: Filter by prompt name
            model: Filter by model

        Returns:
            CostSummary
        """
        # Filter executions
        filtered = self.executions

        if start_date:
            filtered = [e for e in filtered if e.timestamp >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.timestamp <= end_date]
        if prompt_name:
            filtered = [e for e in filtered if e.prompt_name == prompt_name]
        if model:
            filtered = [e for e in filtered if e.model == model]

        # Calculate totals
        total_executions = len(filtered)
        total_tokens = sum(e.total_tokens for e in filtered)
        total_cost = sum(e.total_cost for e in filtered)

        # By model
        by_model = {}
        for e in filtered:
            if e.model not in by_model:
                by_model[e.model] = {
                    "executions": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            by_model[e.model]["executions"] += 1
            by_model[e.model]["tokens"] += e.total_tokens
            by_model[e.model]["cost"] += e.total_cost

        # By prompt
        by_prompt = {}
        for e in filtered:
            if e.prompt_name not in by_prompt:
                by_prompt[e.prompt_name] = {
                    "executions": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            by_prompt[e.prompt_name]["executions"] += 1
            by_prompt[e.prompt_name]["tokens"] += e.total_tokens
            by_prompt[e.prompt_name]["cost"] += e.total_cost

        # Date range
        if filtered:
            dates = [e.timestamp for e in filtered]
            date_range = (min(dates), max(dates))
        else:
            date_range = ("", "")

        return CostSummary(
            total_executions=total_executions,
            total_tokens=total_tokens,
            total_cost=total_cost,
            by_model=by_model,
            by_prompt=by_prompt,
            date_range=date_range
        )

    def get_top_expensive_prompts(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get most expensive prompts"""
        summary = self.get_summary()
        prompts = [(name, stats["cost"]) for name, stats in summary.by_prompt.items()]
        prompts.sort(key=lambda x: x[1], reverse=True)
        return prompts[:limit]

    def estimate_cost(
        self,
        model: str,
        provider: ModelProvider,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Estimate cost for a planned execution.

        Returns:
            Dict with cost estimates
        """
        provider_pricing = PRICING.get(provider, {})
        model_pricing = provider_pricing.get(model, provider_pricing.get("default", {"input": 0, "output": 0}))

        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    def save(self):
        """Save cost data to disk"""
        data = [e.to_dict() for e in self.executions]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load cost data from disk"""
        if self.storage_path.exists():
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.executions = [
                    ExecutionCost(**item) for item in data
                ]

    def export_csv(self, output_path: str):
        """Export cost data to CSV"""
        import csv

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if not self.executions:
                return

            writer = csv.DictWriter(f, fieldnames=self.executions[0].to_dict().keys())
            writer.writeheader()
            for execution in self.executions:
                writer.writerow(execution.to_dict())

    def clear_history(self):
        """Clear all cost history"""
        self.executions = []
        self.save()


# ============================================================================
# Convenience Functions
# ============================================================================

def estimate_prompt_cost(
    prompt_content: str,
    model: str = "claude-3-5-sonnet-20241022",
    provider: ModelProvider = ModelProvider.ANTHROPIC
) -> Dict[str, Any]:
    """
    Estimate cost for a prompt.

    Args:
        prompt_content: Prompt text
        model: Model name
        provider: Provider

    Returns:
        Cost estimate
    """
    # Rough token estimate (4 chars ≈ 1 token)
    input_tokens = len(prompt_content) // 4
    output_tokens = 500  # Assume 500 token response

    tracker = CostTracker()
    estimate = tracker.estimate_cost(model, provider, input_tokens, output_tokens)

    estimate.update({
        "model": model,
        "provider": provider.value,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens
    })

    return estimate


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Promptly Cost Tracker")
    print("\nExample - Record Execution:")
    print("""
from cost_tracker import CostTracker, ModelProvider

tracker = CostTracker()

# Record an execution
cost = tracker.record_execution(
    prompt_name="summarizer",
    model="claude-3-5-sonnet-20241022",
    provider=ModelProvider.ANTHROPIC,
    input_tokens=1000,
    output_tokens=500
)

print(f"Cost: ${cost.total_cost:.4f}")
""")

    print("\nExample - Get Summary:")
    print("""
summary = tracker.get_summary()
print(summary.to_report())
""")

    print("\nExample - Estimate Cost:")
    print("""
from cost_tracker import estimate_prompt_cost

estimate = estimate_prompt_cost("Summarize this article: ...")
print(f"Estimated cost: ${estimate['total_cost']:.4f}")
""")

    print("\nCurrent Pricing (per 1M tokens):")
    print("\nClaude:")
    for model, prices in PRICING[ModelProvider.ANTHROPIC].items():
        print(f"  {model}: ${prices['input']:.2f} in / ${prices['output']:.2f} out")

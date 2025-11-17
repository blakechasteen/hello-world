#!/usr/bin/env python3
"""
Chain Optimizer - Performance optimization suggestions

Analyzes chain definitions and execution traces to suggest:
- Parallelization opportunities
- Redundant step elimination
- Caching strategies
- Cost optimization
- Performance tuning recommendations
"""

import json
from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .chain_tracing import StepTrace


@dataclass
class OptimizationSuggestion:
    """Single optimization suggestion"""
    category: str  # parallelization, caching, elimination, cost, performance
    title: str
    description: str
    impact: str  # high, medium, low
    effort: str  # high, medium, low
    estimated_improvement: str  # e.g., "30% faster", "$0.05 cost savings"
    steps_affected: List[str]
    implementation: str  # How to implement


class ChainOptimizer:
    """
    Analyze chains and suggest optimizations.

    Categories:
    - Parallelization: Steps that can run in parallel
    - Caching: Results that can be cached
    - Elimination: Redundant or unnecessary steps
    - Cost: Cost reduction opportunities
    - Performance: Speed improvements
    """

    def __init__(self):
        self.suggestions: List[OptimizationSuggestion] = []

    def analyze(
        self,
        chain_def: Dict[str, Any],
        execution_trace: Optional[List[StepTrace]] = None
    ) -> List[OptimizationSuggestion]:
        """
        Analyze chain and generate optimization suggestions.

        Args:
            chain_def: Chain definition
            execution_trace: Optional execution trace for runtime analysis

        Returns:
            List of optimization suggestions
        """
        self.suggestions = []

        # Analyze structure
        self._analyze_parallelization(chain_def)
        self._analyze_redundancy(chain_def)
        self._analyze_caching_opportunities(chain_def)

        # Analyze execution if trace available
        if execution_trace:
            self._analyze_performance(execution_trace)
            self._analyze_costs(execution_trace)
            self._analyze_bottlenecks(execution_trace)

        # Sort by impact/effort ratio
        self.suggestions.sort(
            key=lambda s: (
                {"high": 3, "medium": 2, "low": 1}[s.impact] /
                {"low": 1, "medium": 2, "high": 3}[s.effort]
            ),
            reverse=True
        )

        return self.suggestions

    def _analyze_parallelization(self, chain_def: Dict[str, Any]) -> None:
        """Find steps that can be parallelized"""
        steps = chain_def.get("steps", [])
        if not steps:
            return

        # Build dependency graph
        dependencies = {}
        for step in steps:
            step_name = step.name if hasattr(step, 'name') else step.get('name', '')
            step_deps = step.dependencies if hasattr(step, 'dependencies') else step.get('depends_on', [])
            dependencies[step_name] = set(step_deps)

        # Find steps with same dependencies (can run in parallel)
        dep_groups = defaultdict(list)
        for step_name, deps in dependencies.items():
            dep_key = tuple(sorted(deps))
            dep_groups[dep_key].append(step_name)

        # Suggest parallelization for groups > 1
        for deps, steps_list in dep_groups.items():
            if len(steps_list) > 1:
                self.suggestions.append(OptimizationSuggestion(
                    category="parallelization",
                    title=f"Parallelize {len(steps_list)} independent steps",
                    description=f"Steps {', '.join(steps_list)} have the same dependencies and can run in parallel",
                    impact="high" if len(steps_list) > 3 else "medium",
                    effort="low",
                    estimated_improvement=f"~{len(steps_list)-1}x speedup potential",
                    steps_affected=steps_list,
                    implementation=f"""
Convert steps to parallel processor:
```yaml
- name: parallel_group
  type: parallel
  depends_on: {list(deps) if deps else '[]'}
  config:
    tasks:
{chr(10).join(f'      - name: {s}' for s in steps_list)}
```
"""
                ))

    def _analyze_redundancy(self, chain_def: Dict[str, Any]) -> None:
        """Find redundant or unnecessary steps"""
        steps = chain_def.get("steps", [])

        # Check for duplicate transform operations
        transforms = defaultdict(list)
        for step in steps:
            step_name = step.name if hasattr(step, 'name') else step.get('name', '')
            step_type = step.processor_type if hasattr(step, 'processor_type') else step.get('type', '')
            step_config = step.config if hasattr(step, 'config') else step.get('config', {})

            if step_type == "transform":
                transform_key = (
                    step_config.get("transform_type"),
                    step_config.get("source_field"),
                    step_config.get("target_field")
                )
                transforms[transform_key].append(step_name)

        # Flag duplicates
        for transform_key, step_names in transforms.items():
            if len(step_names) > 1:
                self.suggestions.append(OptimizationSuggestion(
                    category="elimination",
                    title="Remove duplicate transformations",
                    description=f"Steps {', '.join(step_names)} perform identical transformations",
                    impact="low",
                    effort="low",
                    estimated_improvement="Slight cost reduction",
                    steps_affected=step_names,
                    implementation=f"Remove duplicate steps: {', '.join(step_names[1:])}"
                ))

    def _analyze_caching_opportunities(self, chain_def: Dict[str, Any]) -> None:
        """Find opportunities for caching"""
        steps = chain_def.get("steps", [])

        # Look for expensive operations that could be cached
        cacheable_types = ["execute", "retry", "parallel"]

        for step in steps:
            step_name = step.name if hasattr(step, 'name') else step.get('name', '')
            step_type = step.processor_type if hasattr(step, 'processor_type') else step.get('type', '')

            if step_type in cacheable_types:
                self.suggestions.append(OptimizationSuggestion(
                    category="caching",
                    title=f"Cache results of '{step_name}'",
                    description=f"This {step_type} step could benefit from result caching",
                    impact="medium",
                    effort="medium",
                    estimated_improvement="50-90% speedup on cache hits",
                    steps_affected=[step_name],
                    implementation=f"""
Add caching wrapper:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_{step_name}(input_hash):
    # Execute step
    pass
```
"""
                ))

    def _analyze_performance(self, execution_trace: List[StepTrace]) -> None:
        """Analyze performance from execution trace"""
        if not execution_trace:
            return

        # Find slow steps
        durations = [(t.step_name, t.duration) for t in execution_trace if t.duration]
        if not durations:
            return

        avg_duration = sum(d for _, d in durations) / len(durations)
        slow_steps = [(name, dur) for name, dur in durations if dur > avg_duration * 2]

        for step_name, duration in slow_steps:
            self.suggestions.append(OptimizationSuggestion(
                category="performance",
                title=f"Optimize slow step '{step_name}'",
                description=f"This step takes {duration:.2f}s ({duration/avg_duration:.1f}x average)",
                impact="high" if duration > avg_duration * 3 else "medium",
                effort="medium",
                estimated_improvement=f"Reduce from {duration:.2f}s to ~{avg_duration:.2f}s",
                steps_affected=[step_name],
                implementation="""
Investigate and optimize:
- Use more efficient algorithms
- Reduce I/O operations
- Add timeout limits
- Consider approximation for non-critical results
"""
            ))

    def _analyze_costs(self, execution_trace: List[StepTrace]) -> None:
        """Analyze cost optimization opportunities"""
        # Simple cost model (would be more sophisticated in production)
        cost_model = {
            "execute": 0.01,
            "parallel": 0.05,
            "retry": 0.02,
            "loop": 0.03
        }

        high_cost_steps = []
        for trace in execution_trace:
            base_cost = cost_model.get(trace.processor_type, 0.01)
            iterations = trace.metadata.get("iterations", 1) if trace.metadata else 1
            step_cost = base_cost * iterations

            if step_cost > 0.05:
                high_cost_steps.append((trace.step_name, step_cost, trace.processor_type))

        for step_name, cost, proc_type in high_cost_steps:
            if proc_type == "loop":
                self.suggestions.append(OptimizationSuggestion(
                    category="cost",
                    title=f"Reduce loop iterations in '{step_name}'",
                    description=f"Loop is expensive (${cost:.4f}), consider batch processing",
                    impact="medium",
                    effort="medium",
                    estimated_improvement=f"Potential ${cost * 0.5:.4f} savings",
                    steps_affected=[step_name],
                    implementation="Convert loop to parallel batch processing where possible"
                ))

            elif proc_type == "retry":
                self.suggestions.append(OptimizationSuggestion(
                    category="cost",
                    title=f"Optimize retry strategy in '{step_name}'",
                    description=f"Frequent retries increase cost (${cost:.4f})",
                    impact="medium",
                    effort="low",
                    estimated_improvement=f"Save ${cost * 0.3:.4f} by reducing retries",
                    steps_affected=[step_name],
                    implementation="Use circuit breaker pattern or adjust retry backoff"
                ))

    def _analyze_bottlenecks(self, execution_trace: List[StepTrace]) -> None:
        """Identify execution bottlenecks"""
        if not execution_trace:
            return

        total_duration = sum(t.duration for t in execution_trace if t.duration)
        if total_duration == 0:
            return

        # Find steps taking > 20% of total time
        for trace in execution_trace:
            if trace.duration and trace.duration > total_duration * 0.2:
                percentage = (trace.duration / total_duration) * 100

                self.suggestions.append(OptimizationSuggestion(
                    category="performance",
                    title=f"Critical bottleneck: '{trace.step_name}'",
                    description=f"Takes {percentage:.1f}% of total execution time",
                    impact="high",
                    effort="high",
                    estimated_improvement=f"Reducing by 50% saves {trace.duration * 0.5:.2f}s",
                    steps_affected=[trace.step_name],
                    implementation="""
High-priority optimization:
- Profile the step in detail
- Consider async execution
- Split into smaller steps
- Add result caching
- Use faster model/algorithm
"""
                ))

    def generate_report(self) -> str:
        """Generate optimization report"""
        if not self.suggestions:
            return "# Chain Optimization Report\n\nNo optimization opportunities found."

        report = ["# Chain Optimization Report\n"]
        report.append(f"Found {len(self.suggestions)} optimization opportunities\n")

        # Group by category
        by_category = defaultdict(list)
        for suggestion in self.suggestions:
            by_category[suggestion.category].append(suggestion)

        category_names = {
            "parallelization": "Parallelization Opportunities",
            "caching": "Caching Strategies",
            "elimination": "Redundancy Elimination",
            "cost": "Cost Optimization",
            "performance": "Performance Improvements"
        }

        for category, name in category_names.items():
            if category in by_category:
                report.append(f"\n## {name}\n")

                for i, suggestion in enumerate(by_category[category], 1):
                    report.append(f"\n### {i}. {suggestion.title}\n")
                    report.append(f"**Impact:** {suggestion.impact.upper()} | ")
                    report.append(f"**Effort:** {suggestion.effort.upper()}\n\n")
                    report.append(f"{suggestion.description}\n\n")
                    report.append(f"**Estimated Improvement:** {suggestion.estimated_improvement}\n\n")
                    report.append(f"**Steps Affected:** {', '.join(suggestion.steps_affected)}\n\n")
                    report.append(f"**Implementation:**\n{suggestion.implementation}\n")

        return "".join(report)

    def export_json(self) -> str:
        """Export suggestions as JSON"""
        return json.dumps([
            {
                "category": s.category,
                "title": s.title,
                "description": s.description,
                "impact": s.impact,
                "effort": s.effort,
                "estimated_improvement": s.estimated_improvement,
                "steps_affected": s.steps_affected,
                "implementation": s.implementation
            }
            for s in self.suggestions
        ], indent=2)


def optimize_chain(
    chain_def: Dict[str, Any],
    execution_trace: Optional[List[StepTrace]] = None
) -> List[OptimizationSuggestion]:
    """
    Analyze chain and get optimization suggestions.

    Args:
        chain_def: Chain definition
        execution_trace: Optional execution trace

    Returns:
        List of optimization suggestions
    """
    optimizer = ChainOptimizer()
    return optimizer.analyze(chain_def, execution_trace)


__all__ = ['ChainOptimizer', 'OptimizationSuggestion', 'optimize_chain']

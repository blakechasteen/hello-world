"""
Managerial & Motivational Agents

Meta-level agents that coordinate, optimize, and improve other agents.

## When to Use Managerial Agents (Best Practices)

✅ **USE WHEN:**
- Multiple agents with competing resource needs
- Complex multi-step tasks requiring coordination
- Performance optimization needed across agent pool
- Quality control and error recovery required
- Dynamic workload balancing needed

❌ **DON'T USE WHEN:**
- Simple single-agent tasks (over-engineering)
- Task queue + priorities already sufficient
- Adds unnecessary coordination overhead
- Creates single point of failure

## Architecture Principle: "Simple First, Hierarchy When Needed"

Start with flat agent pool + priority queue.
Add managerial layer only when complexity demands it.

## Types of Meta-Agents

1. **PerformanceMonitor**: Watches agent health, adjusts parameters
2. **QualityController**: Validates outputs, triggers refinement
3. **ResourceAllocator**: Distributes compute, memory, time budgets
4. **StrategyAdvisor**: Suggests exploration vs exploitation balance
5. **MotivationalCoach**: Adjusts learning rates, confidence thresholds
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import numpy as np

from HoloLoom.agents.agent_orchestration import (
    PersistentAgent,
    AgentOrchestrationSystem,
    TaskPriority,
    TaskType
)


# ============================================================================
# Performance Monitoring
# ============================================================================

@dataclass
class AgentHealthMetrics:
    """Health metrics for a single agent"""
    agent_name: str

    # Performance
    success_rate: float = 1.0
    avg_confidence: float = 0.7
    avg_latency_ms: float = 150.0

    # Activity
    queries_per_minute: float = 0.0
    idle_percentage: float = 0.0

    # Quality
    breakthrough_rate: float = 0.0  # Breakthroughs per 100 queries
    pattern_learning_rate: float = 0.0  # Patterns per 100 queries
    error_rate: float = 0.0

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0

    # Timestamp
    measured_at: float = field(default_factory=time.time)

    def overall_health_score(self) -> float:
        """Compute overall health (0-1)"""
        return (
            0.3 * self.success_rate +
            0.2 * self.avg_confidence +
            0.2 * (1.0 - self.error_rate) +
            0.1 * min(1.0, self.breakthrough_rate / 5.0) +
            0.1 * (1.0 - self.idle_percentage) +
            0.1 * min(1.0, self.queries_per_minute / 10.0)
        )

    def needs_attention(self) -> bool:
        """Check if agent needs managerial intervention"""
        return (
            self.success_rate < 0.7 or
            self.error_rate > 0.2 or
            self.avg_confidence < 0.5 or
            self.idle_percentage > 0.8
        )


class PerformanceMonitor:
    """
    Managerial Agent: Monitors agent performance and health.

    Responsibilities:
    - Track agent metrics over time
    - Detect performance degradation
    - Trigger interventions when needed
    - Generate health reports
    """

    def __init__(self, check_interval_seconds: float = 30.0):
        self.check_interval_seconds = check_interval_seconds

        # Metrics history
        self.metrics_history: Dict[str, List[AgentHealthMetrics]] = {}

        # Alerts
        self.active_alerts: Dict[str, List[str]] = {}

        # Running
        self.monitor_task: Optional[Any] = None
        self.running = False

    async def start(self, orchestration_system: AgentOrchestrationSystem):
        """Start monitoring"""
        self.orchestration = orchestration_system
        self.running = True
        # Would start background monitoring task here

    async def stop(self):
        """Stop monitoring"""
        self.running = False

    def collect_metrics(self, agent: PersistentAgent) -> AgentHealthMetrics:
        """Collect current metrics for agent"""
        stats = agent.get_stats()

        metrics = AgentHealthMetrics(
            agent_name=agent.agent_name,
            success_rate=agent.success_rate,
            avg_confidence=0.7,  # Would compute from recent queries
            avg_latency_ms=150.0,  # Would compute from recent queries
            queries_per_minute=stats['total_queries'] / max(1, stats['uptime_seconds'] / 60),
            idle_percentage=min(1.0, stats['idle_seconds'] / max(1, stats['uptime_seconds'])),
            breakthrough_rate=(stats['total_breakthroughs'] / max(1, stats['total_queries'])) * 100,
            pattern_learning_rate=(stats['patterns_learned'] / max(1, stats['total_queries'])) * 100,
            error_rate=stats['error_count'] / max(1, stats['total_queries'])
        )

        # Store in history
        if agent.agent_name not in self.metrics_history:
            self.metrics_history[agent.agent_name] = []

        self.metrics_history[agent.agent_name].append(metrics)

        # Keep last 100 metrics
        if len(self.metrics_history[agent.agent_name]) > 100:
            self.metrics_history[agent.agent_name].pop(0)

        return metrics

    def detect_issues(self, agent_name: str) -> List[str]:
        """Detect issues from metrics history"""
        if agent_name not in self.metrics_history:
            return []

        history = self.metrics_history[agent_name]
        if len(history) < 2:
            return []

        issues = []
        current = history[-1]
        previous = history[-2]

        # Performance degradation
        if current.success_rate < previous.success_rate - 0.1:
            issues.append(f"Success rate dropped {(previous.success_rate - current.success_rate)*100:.1f}%")

        # Increasing errors
        if current.error_rate > previous.error_rate * 1.5:
            issues.append(f"Error rate increased {(current.error_rate / previous.error_rate):.1f}x")

        # Low confidence
        if current.avg_confidence < 0.5:
            issues.append(f"Low confidence: {current.avg_confidence:.2f}")

        # High idle time
        if current.idle_percentage > 0.7:
            issues.append(f"High idle time: {current.idle_percentage*100:.0f}%")

        # Low breakthrough rate
        if current.breakthrough_rate < 1.0 and current.queries_per_minute > 1.0:
            issues.append(f"Low breakthrough rate: {current.breakthrough_rate:.2f}%")

        return issues

    def recommend_actions(self, agent_name: str, issues: List[str]) -> List[Dict[str, Any]]:
        """Recommend actions to address issues"""
        actions = []

        for issue in issues:
            if "success rate" in issue.lower():
                actions.append({
                    'action': 'reduce_workload',
                    'reason': 'High failure rate - reduce task complexity',
                    'parameters': {'priority_filter': 'HIGH_ONLY'}
                })

            elif "error rate" in issue.lower():
                actions.append({
                    'action': 'trigger_maintenance',
                    'reason': 'Errors increasing - run diagnostics',
                    'parameters': {'restart_agent': False}
                })

            elif "low confidence" in issue.lower():
                actions.append({
                    'action': 'adjust_parameters',
                    'reason': 'Low confidence - increase exploration',
                    'parameters': {'exploration_weight': 2.0, 'mcts_simulations': 75}
                })

            elif "high idle" in issue.lower():
                actions.append({
                    'action': 'assign_background_tasks',
                    'reason': 'Agent underutilized - add learning tasks',
                    'parameters': {'task_type': 'BACKGROUND_LEARNING'}
                })

            elif "low breakthrough" in issue.lower():
                actions.append({
                    'action': 'increase_exploration',
                    'reason': 'Not discovering enough - increase exploration',
                    'parameters': {'breakthrough_threshold': 1.5}
                })

        return actions


# ============================================================================
# Quality Control
# ============================================================================

class QualityController:
    """
    Managerial Agent: Ensures output quality.

    Responsibilities:
    - Validate agent outputs
    - Trigger refinement for low-quality results
    - Track quality metrics
    - Enforce quality standards
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        refinement_threshold: float = 0.7
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.refinement_threshold = refinement_threshold

        # Quality tracking
        self.quality_violations: Dict[str, int] = {}
        self.refinements_triggered: Dict[str, int] = {}

    def validate_output(
        self,
        agent_name: str,
        result: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate agent output quality.

        Returns:
            {
                'passed': bool,
                'quality_score': float,
                'issues': List[str],
                'recommendation': str
            }
        """
        confidence = getattr(result, 'confidence', 0.7)

        issues = []

        # Check confidence
        if confidence < self.min_confidence_threshold:
            issues.append(f"Confidence too low: {confidence:.2f} < {self.min_confidence_threshold}")

        # Check for errors
        if hasattr(result, 'error') and result.error:
            issues.append(f"Result contains error: {result.error}")

        # Check completeness
        if hasattr(result, 'response'):
            if not result.response or len(result.response) < 10:
                issues.append("Response too short or empty")

        # Determine pass/fail
        passed = len(issues) == 0 and confidence >= self.min_confidence_threshold

        # Track violations
        if not passed:
            self.quality_violations[agent_name] = self.quality_violations.get(agent_name, 0) + 1

        # Recommendation
        recommendation = "accept"
        if not passed:
            if confidence < self.refinement_threshold:
                recommendation = "refine"
                self.refinements_triggered[agent_name] = self.refinements_triggered.get(agent_name, 0) + 1
            else:
                recommendation = "retry"

        return {
            'passed': passed,
            'quality_score': confidence,
            'issues': issues,
            'recommendation': recommendation
        }

    def get_quality_report(self, agent_name: str) -> Dict[str, Any]:
        """Get quality report for agent"""
        return {
            'violations': self.quality_violations.get(agent_name, 0),
            'refinements_triggered': self.refinements_triggered.get(agent_name, 0)
        }


# ============================================================================
# Resource Allocation
# ============================================================================

class ResourceAllocator:
    """
    Managerial Agent: Allocates resources across agents.

    Responsibilities:
    - Distribute compute budgets
    - Balance workload
    - Prevent resource starvation
    - Optimize throughput
    """

    def __init__(
        self,
        total_compute_budget: float = 1.0,  # Normalized units
        rebalance_interval_seconds: float = 60.0
    ):
        self.total_compute_budget = total_compute_budget
        self.rebalance_interval_seconds = rebalance_interval_seconds

        # Current allocations
        self.agent_budgets: Dict[str, float] = {}

        # Usage tracking
        self.agent_usage: Dict[str, float] = {}

    def allocate_budgets(
        self,
        agents: Dict[str, PersistentAgent],
        metrics: Dict[str, AgentHealthMetrics]
    ) -> Dict[str, float]:
        """
        Allocate compute budgets to agents.

        Strategy: Prioritize high-performing, active agents.
        """
        if not agents:
            return {}

        # Compute priority scores
        priority_scores = {}
        for agent_name, agent in agents.items():
            if agent_name in metrics:
                m = metrics[agent_name]

                # Priority = performance * activity * (1 - idle)
                priority = (
                    m.success_rate *
                    min(1.0, m.queries_per_minute / 5.0) *
                    (1.0 - m.idle_percentage)
                )

                priority_scores[agent_name] = priority
            else:
                priority_scores[agent_name] = 0.5  # Default

        # Normalize to total budget
        total_priority = sum(priority_scores.values())

        if total_priority == 0:
            # Equal distribution
            budget_per_agent = self.total_compute_budget / len(agents)
            return {name: budget_per_agent for name in agents.keys()}

        # Allocate proportionally
        allocations = {}
        for agent_name, priority in priority_scores.items():
            allocations[agent_name] = (
                (priority / total_priority) * self.total_compute_budget
            )

        self.agent_budgets = allocations
        return allocations

    def get_allocation(self, agent_name: str) -> float:
        """Get current allocation for agent"""
        return self.agent_budgets.get(agent_name, 0.1)  # Default minimum


# ============================================================================
# Motivational Coach
# ============================================================================

class MotivationalCoach:
    """
    Motivational Agent: Adjusts agent parameters for optimal learning.

    Responsibilities:
    - Adjust exploration vs exploitation
    - Tune confidence thresholds
    - Modify learning rates
    - Encourage breakthrough discovery
    """

    def __init__(self):
        # Parameter adjustments
        self.adjustments: Dict[str, Dict[str, Any]] = {}

    def assess_agent_state(
        self,
        agent_name: str,
        metrics: AgentHealthMetrics
    ) -> str:
        """
        Assess agent's current state.

        Returns: "thriving", "stable", "struggling", "stuck"
        """
        health = metrics.overall_health_score()

        if health > 0.8 and metrics.breakthrough_rate > 2.0:
            return "thriving"
        elif health > 0.6:
            return "stable"
        elif health > 0.4:
            return "struggling"
        else:
            return "stuck"

    def recommend_adjustments(
        self,
        agent_name: str,
        state: str,
        metrics: AgentHealthMetrics
    ) -> Dict[str, Any]:
        """
        Recommend parameter adjustments based on state.
        """
        adjustments = {}

        if state == "thriving":
            # Doing great - minor optimizations
            adjustments['message'] = "🌟 Excellent performance! Minor optimization."
            adjustments['exploration_weight'] = 1.2  # Slight reduction
            adjustments['mcts_simulations'] = 40  # Can reduce for speed

        elif state == "stable":
            # Doing ok - maintain course
            adjustments['message'] = "✅ Stable performance."
            adjustments['exploration_weight'] = 1.414  # Standard
            adjustments['mcts_simulations'] = 50  # Standard

        elif state == "struggling":
            # Having issues - increase exploration
            adjustments['message'] = "⚠️ Performance declining - increasing exploration."
            adjustments['exploration_weight'] = 1.8  # More exploration
            adjustments['mcts_simulations'] = 75  # More search depth
            adjustments['breakthrough_threshold'] = 1.5  # Lower threshold

        elif state == "stuck":
            # Stuck - dramatic intervention
            adjustments['message'] = "🚨 Agent stuck - radical exploration."
            adjustments['exploration_weight'] = 2.5  # Maximum exploration
            adjustments['mcts_simulations'] = 100  # Deep search
            adjustments['breakthrough_threshold'] = 1.0  # Very low threshold
            adjustments['reset_patterns'] = True  # Clear bad patterns

        self.adjustments[agent_name] = adjustments
        return adjustments


# ============================================================================
# Integrated Managerial System
# ============================================================================

class ManagerialSystem:
    """
    Integrated managerial system combining all meta-agents.

    Use this when you need coordinated meta-level management.
    """

    def __init__(self, orchestration_system: AgentOrchestrationSystem):
        self.orchestration = orchestration_system

        # Meta-agents
        self.performance_monitor = PerformanceMonitor()
        self.quality_controller = QualityController()
        self.resource_allocator = ResourceAllocator()
        self.motivational_coach = MotivationalCoach()

        # Running
        self.running = False
        self.management_task: Optional[Any] = None

    async def start(self):
        """Start managerial system"""
        self.running = True
        await self.performance_monitor.start(self.orchestration)
        # Would start background management loop here

    async def stop(self):
        """Stop managerial system"""
        self.running = False
        await self.performance_monitor.stop()

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive management report"""
        return {
            'performance_monitoring': {
                'agents_monitored': len(self.performance_monitor.metrics_history),
                'active_alerts': self.performance_monitor.active_alerts
            },
            'quality_control': {
                'total_violations': sum(self.quality_controller.quality_violations.values()),
                'refinements_triggered': sum(self.quality_controller.refinements_triggered.values())
            },
            'resource_allocation': {
                'budgets': self.resource_allocator.agent_budgets
            },
            'motivational_coaching': {
                'adjustments': self.motivational_coach.adjustments
            }
        }


# ============================================================================
# Best Practice Guidelines
# ============================================================================

"""
## When to Use Managerial Agents - Decision Tree

START: Do you have multiple agents?
  NO → Don't use managerial agents (over-engineering)
  YES → Continue

Are agents competing for resources?
  NO → Skip ResourceAllocator
  YES → Use ResourceAllocator

Do you need quality guarantees?
  NO → Skip QualityController
  YES → Use QualityController

Are agent parameters static or need tuning?
  STATIC → Skip MotivationalCoach
  DYNAMIC → Use MotivationalCoach

Do you need health monitoring?
  NO → Skip PerformanceMonitor
  YES → Use PerformanceMonitor

RESULT: Use only the meta-agents you actually need.

## Recommended Starting Point

1. START: Task queue + priority system (already have this)
2. ADD: PerformanceMonitor (lightweight, high value)
3. CONSIDER: QualityController (if quality issues arise)
4. LATER: ResourceAllocator (if resource contention appears)
5. ADVANCED: MotivationalCoach (for optimization)

## Anti-Patterns to Avoid

❌ Creating managerial agent before need is proven
❌ Over-coordinating (adds latency, complexity)
❌ Single point of failure (manager down = all agents stuck)
❌ Micro-managing (checking every operation)
❌ Ignoring simple solutions (priorities often sufficient)

## Success Patterns

✅ Monitor first, manage second
✅ Start simple, add complexity when needed
✅ Use metrics to justify managerial layer
✅ Keep managers lightweight
✅ Make managers optional (graceful degradation)
"""

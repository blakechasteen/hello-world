"""
Reflection Learning Module
===========================
Enable HoloLoom to learn from its own decisions and continuously improve.

Core Components:
1. ReflectionEngine: Analyzes past decisions
2. PatternDetector: Identifies success/failure patterns
3. PolicyRefiner: Generates policy improvements
4. ExperienceReplay: Stores and samples trajectories

Philosophy:
A neural system that cannot reflect on its past decisions is doomed to
repeat its mistakes. True intelligence requires the ability to analyze
outcomes, understand what worked, what failed, and why - then adapt
accordingly.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import asyncio

logger = logging.getLogger(__name__)

# Import types from documentation
try:
    from ..documentation.types import Query, Context, Features, ActionPlan
    TYPES_AVAILABLE = True
except ImportError:
    logger.warning("Types not available, using generic types")
    TYPES_AVAILABLE = False
    Query = Dict
    Context = Dict
    Features = Dict
    ActionPlan = Dict


@dataclass
class Outcome:
    """Outcome of a decision execution."""

    success: bool
    execution_time: float  # seconds
    error: Optional[str] = None
    result: Optional[Any] = None

    # Quality metrics
    user_satisfaction: Optional[float] = None  # 0-1 scale
    correctness: Optional[float] = None  # 0-1 scale
    efficiency: Optional[float] = None  # 0-1 scale

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    """User feedback on a response."""

    rating: float  # 0-5 stars
    helpful: bool
    comments: Optional[str] = None
    specific_issues: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DecisionTrajectory:
    """Complete trajectory of a decision cycle."""

    trajectory_id: str
    query: Query
    features: Features
    context: Context
    action_plan: ActionPlan
    outcome: Outcome
    feedback: Optional[Feedback] = None

    # Quality score (computed)
    quality_score: float = 0.0

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class Pattern:
    """Detected pattern in decision trajectories."""

    pattern_type: str  # "success", "failure", "neutral"
    description: str
    frequency: int
    confidence: float  # 0-1

    # Conditions that trigger this pattern
    conditions: Dict[str, Any]

    # Typical outcomes
    typical_outcome: Dict[str, Any]

    # Examples (trajectory IDs)
    examples: List[str] = field(default_factory=list)


@dataclass
class ReflectionInsights:
    """Insights from reflection analysis."""

    # Overall quality metrics
    average_quality: float
    success_rate: float
    average_execution_time: float

    # Detected patterns
    success_patterns: List[Pattern]
    failure_patterns: List[Pattern]

    # Feature importance
    feature_importance: Dict[str, float]

    # Tool effectiveness
    tool_performance: Dict[str, Dict[str, float]]

    # Recommendations
    recommendations: List[str]

    # Metadata
    num_trajectories_analyzed: int
    analysis_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class PolicyUpdate:
    """Suggested update to policy."""

    update_type: str  # "weight_adjustment", "strategy_change", "tool_preference"
    description: str
    priority: str  # "high", "medium", "low"

    # Specific update parameters
    parameters: Dict[str, Any]

    # Expected impact
    expected_improvement: float  # percentage
    confidence: float  # 0-1

    # Supporting evidence
    evidence: List[str]  # trajectory IDs


class ReflectionEngine:
    """
    Analyzes past decisions and learns patterns.

    Capabilities:
    - Success/failure pattern detection
    - Decision quality metrics
    - Strategy effectiveness analysis
    - Automatic policy refinement
    """

    def __init__(
        self,
        min_trajectories_for_analysis: int = 10,
        pattern_min_frequency: int = 3,
        pattern_min_confidence: float = 0.7
    ):
        self.min_trajectories_for_analysis = min_trajectories_for_analysis
        self.pattern_min_frequency = pattern_min_frequency
        self.pattern_min_confidence = pattern_min_confidence

        # Storage
        self.trajectories: List[DecisionTrajectory] = []
        self.patterns: List[Pattern] = []

        # Analytics
        self.tool_stats = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "avg_time": 0.0,
            "avg_quality": 0.0
        })

    def record_trajectory(self, trajectory: DecisionTrajectory):
        """Record a decision trajectory for analysis."""

        # Compute quality score
        trajectory.quality_score = self._compute_quality_score(trajectory)

        # Store trajectory
        self.trajectories.append(trajectory)

        # Update tool statistics
        if hasattr(trajectory.action_plan, 'selected_tool'):
            tool = trajectory.action_plan.selected_tool
            self._update_tool_stats(tool, trajectory)

        logger.info(
            f"Recorded trajectory {trajectory.trajectory_id} "
            f"(quality: {trajectory.quality_score:.2f})"
        )

    def _compute_quality_score(self, trajectory: DecisionTrajectory) -> float:
        """
        Compute overall quality score for a trajectory.

        Factors:
        - Success (40%)
        - User satisfaction (30%)
        - Efficiency (20%)
        - Correctness (10%)
        """

        score = 0.0

        # Success (40%)
        if trajectory.outcome.success:
            score += 0.4

        # User satisfaction (30%)
        if trajectory.outcome.user_satisfaction is not None:
            score += 0.3 * trajectory.outcome.user_satisfaction
        elif trajectory.feedback is not None:
            # Convert rating to 0-1 scale
            score += 0.3 * (trajectory.feedback.rating / 5.0)

        # Efficiency (20%)
        if trajectory.outcome.efficiency is not None:
            score += 0.2 * trajectory.outcome.efficiency
        else:
            # Use execution time as proxy (faster is better, with reasonable bounds)
            # Assume 1-5 seconds is optimal, penalize very slow or suspiciously fast
            exec_time = trajectory.outcome.execution_time
            if 1.0 <= exec_time <= 5.0:
                score += 0.2
            elif exec_time < 1.0:
                score += 0.1  # Too fast might indicate incomplete processing
            else:
                score += max(0, 0.2 * (1 - (exec_time - 5.0) / 10.0))

        # Correctness (10%)
        if trajectory.outcome.correctness is not None:
            score += 0.1 * trajectory.outcome.correctness

        return min(1.0, max(0.0, score))

    def _update_tool_stats(self, tool: str, trajectory: DecisionTrajectory):
        """Update statistics for a tool."""

        stats = self.tool_stats[tool]

        # Incremental average update
        n = stats["total"]
        stats["total"] += 1

        if trajectory.outcome.success:
            stats["success"] += 1

        # Update running averages
        stats["avg_time"] = (
            (stats["avg_time"] * n + trajectory.outcome.execution_time) / (n + 1)
        )
        stats["avg_quality"] = (
            (stats["avg_quality"] * n + trajectory.quality_score) / (n + 1)
        )

    async def analyze_decisions(
        self,
        recent_only: bool = False,
        max_trajectories: Optional[int] = None
    ) -> ReflectionInsights:
        """
        Analyze decision trajectories and extract insights.

        Args:
            recent_only: Only analyze recent trajectories (last 100)
            max_trajectories: Maximum number of trajectories to analyze

        Returns:
            Reflection insights with patterns and recommendations
        """

        # Select trajectories to analyze
        trajectories = self.trajectories
        if recent_only:
            trajectories = trajectories[-100:]
        if max_trajectories:
            trajectories = trajectories[-max_trajectories:]

        if len(trajectories) < self.min_trajectories_for_analysis:
            logger.warning(
                f"Not enough trajectories for analysis "
                f"({len(trajectories)} < {self.min_trajectories_for_analysis})"
            )
            return self._empty_insights(len(trajectories))

        logger.info(f"Analyzing {len(trajectories)} decision trajectories...")

        # Compute overall metrics
        quality_scores = [t.quality_score for t in trajectories]
        success_count = sum(1 for t in trajectories if t.outcome.success)
        execution_times = [t.outcome.execution_time for t in trajectories]

        # Detect patterns
        success_patterns = self._detect_patterns(
            [t for t in trajectories if t.outcome.success],
            pattern_type="success"
        )

        failure_patterns = self._detect_patterns(
            [t for t in trajectories if not t.outcome.success],
            pattern_type="failure"
        )

        # Compute feature importance
        feature_importance = self._compute_feature_importance(trajectories)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            trajectories,
            success_patterns,
            failure_patterns
        )

        insights = ReflectionInsights(
            average_quality=np.mean(quality_scores),
            success_rate=success_count / len(trajectories),
            average_execution_time=np.mean(execution_times),
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            feature_importance=feature_importance,
            tool_performance=dict(self.tool_stats),
            recommendations=recommendations,
            num_trajectories_analyzed=len(trajectories)
        )

        logger.info(f"Analysis complete. Success rate: {insights.success_rate:.2%}")

        return insights

    def _empty_insights(self, num_trajectories: int) -> ReflectionInsights:
        """Return empty insights when not enough data."""

        return ReflectionInsights(
            average_quality=0.0,
            success_rate=0.0,
            average_execution_time=0.0,
            success_patterns=[],
            failure_patterns=[],
            feature_importance={},
            tool_performance={},
            recommendations=["Collect more decision trajectories for analysis"],
            num_trajectories_analyzed=num_trajectories
        )

    def _detect_patterns(
        self,
        trajectories: List[DecisionTrajectory],
        pattern_type: str
    ) -> List[Pattern]:
        """
        Detect patterns in trajectories.

        Patterns are detected by clustering similar trajectories
        based on features, context, and outcomes.
        """

        if len(trajectories) < self.pattern_min_frequency:
            return []

        patterns = []

        # Group by tool used
        tool_groups = defaultdict(list)
        for t in trajectories:
            if hasattr(t.action_plan, 'selected_tool'):
                tool = t.action_plan.selected_tool
                tool_groups[tool].append(t)

        for tool, tool_trajectories in tool_groups.items():
            if len(tool_trajectories) >= self.pattern_min_frequency:

                # Compute pattern characteristics
                avg_quality = np.mean([t.quality_score for t in tool_trajectories])
                avg_time = np.mean([t.outcome.execution_time for t in tool_trajectories])

                pattern = Pattern(
                    pattern_type=pattern_type,
                    description=f"Tool '{tool}' {pattern_type} pattern",
                    frequency=len(tool_trajectories),
                    confidence=min(1.0, len(tool_trajectories) / 10.0),
                    conditions={"tool": tool},
                    typical_outcome={
                        "avg_quality": avg_quality,
                        "avg_execution_time": avg_time
                    },
                    examples=[t.trajectory_id for t in tool_trajectories[:5]]
                )

                patterns.append(pattern)

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        return patterns

    def _compute_feature_importance(
        self,
        trajectories: List[DecisionTrajectory]
    ) -> Dict[str, float]:
        """
        Compute feature importance for decision quality.

        Uses correlation between feature presence and quality score.
        """

        feature_importance = {}

        # Extract all feature keys
        all_features = set()
        for t in trajectories:
            if hasattr(t.features, 'motifs'):
                for motif in t.features.motifs:
                    all_features.add(f"motif_{motif.pattern}")

        # Compute correlation with quality
        for feature in all_features:
            feature_present = []
            quality_scores = []

            for t in trajectories:
                has_feature = False
                if hasattr(t.features, 'motifs'):
                    has_feature = any(
                        f"motif_{m.pattern}" == feature
                        for m in t.features.motifs
                    )

                feature_present.append(1.0 if has_feature else 0.0)
                quality_scores.append(t.quality_score)

            # Compute correlation
            if len(feature_present) > 1:
                correlation = np.corrcoef(feature_present, quality_scores)[0, 1]
                if not np.isnan(correlation):
                    feature_importance[feature] = abs(correlation)

        # Normalize
        if feature_importance:
            max_importance = max(feature_importance.values())
            if max_importance > 0:
                feature_importance = {
                    k: v / max_importance
                    for k, v in feature_importance.items()
                }

        return feature_importance

    def _generate_recommendations(
        self,
        trajectories: List[DecisionTrajectory],
        success_patterns: List[Pattern],
        failure_patterns: List[Pattern]
    ) -> List[str]:
        """Generate actionable recommendations."""

        recommendations = []

        # Overall quality
        avg_quality = np.mean([t.quality_score for t in trajectories])
        if avg_quality < 0.5:
            recommendations.append(
                "Overall quality is low (<0.5). Consider retraining policy or "
                "adding more diverse training data."
            )
        elif avg_quality > 0.8:
            recommendations.append(
                "Overall quality is high (>0.8). System is performing well. "
                "Continue monitoring and consider incremental improvements."
            )

        # Tool performance
        for tool, stats in self.tool_stats.items():
            success_rate = stats["success"] / max(1, stats["total"])

            if stats["total"] >= 5:  # Enough data
                if success_rate < 0.5:
                    recommendations.append(
                        f"Tool '{tool}' has low success rate ({success_rate:.1%}). "
                        f"Consider reviewing tool implementation or usage conditions."
                    )
                elif stats["avg_quality"] < 0.4:
                    recommendations.append(
                        f"Tool '{tool}' has low average quality ({stats['avg_quality']:.2f}). "
                        f"May need improvement or different selection criteria."
                    )

        # Pattern-based recommendations
        if failure_patterns:
            most_common_failure = failure_patterns[0]
            recommendations.append(
                f"Most common failure pattern: {most_common_failure.description} "
                f"({most_common_failure.frequency} occurrences). "
                f"Investigate and address root cause."
            )

        if not recommendations:
            recommendations.append(
                "No specific issues detected. Continue monitoring performance."
            )

        return recommendations

    async def generate_improvements(
        self,
        insights: ReflectionInsights
    ) -> List[PolicyUpdate]:
        """
        Generate policy improvements based on reflection insights.

        Args:
            insights: Reflection insights from analysis

        Returns:
            List of suggested policy updates
        """

        updates = []

        # Update 1: Adjust tool selection probabilities based on performance
        for tool, stats in insights.tool_performance.items():
            if stats["total"] >= 10:  # Enough data
                success_rate = stats["success"] / stats["total"]
                avg_quality = stats["avg_quality"]

                # If tool performs poorly, reduce its selection probability
                if success_rate < 0.4 or avg_quality < 0.3:
                    updates.append(PolicyUpdate(
                        update_type="tool_preference",
                        description=f"Reduce selection probability for tool '{tool}'",
                        priority="high",
                        parameters={
                            "tool": tool,
                            "adjustment": -0.2,  # Reduce by 20%
                            "reason": f"Low performance (success: {success_rate:.1%}, quality: {avg_quality:.2f})"
                        },
                        expected_improvement=10.0,
                        confidence=0.7,
                        evidence=[p.examples[0] for p in insights.failure_patterns if tool in str(p.conditions)][:5]
                    ))

                # If tool performs well, increase its selection probability
                elif success_rate > 0.8 and avg_quality > 0.7:
                    updates.append(PolicyUpdate(
                        update_type="tool_preference",
                        description=f"Increase selection probability for tool '{tool}'",
                        priority="medium",
                        parameters={
                            "tool": tool,
                            "adjustment": +0.1,  # Increase by 10%
                            "reason": f"High performance (success: {success_rate:.1%}, quality: {avg_quality:.2f})"
                        },
                        expected_improvement=5.0,
                        confidence=0.8,
                        evidence=[p.examples[0] for p in insights.success_patterns if tool in str(p.conditions)][:5]
                    ))

        # Update 2: Strategy change if overall quality is low
        if insights.average_quality < 0.5 and insights.num_trajectories_analyzed >= 50:
            updates.append(PolicyUpdate(
                update_type="strategy_change",
                description="Switch to more exploratory strategy due to low quality",
                priority="high",
                parameters={
                    "current_strategy": "epsilon_greedy",
                    "suggested_strategy": "pure_thompson",
                    "reason": f"Low average quality ({insights.average_quality:.2f})",
                    "exploration_boost": 0.2
                },
                expected_improvement=15.0,
                confidence=0.6,
                evidence=[]
            ))

        # Update 3: Feature weight adjustments based on importance
        high_importance_features = {
            k: v for k, v in insights.feature_importance.items()
            if v > 0.7
        }

        if high_importance_features:
            updates.append(PolicyUpdate(
                update_type="weight_adjustment",
                description="Increase attention to high-importance features",
                priority="medium",
                parameters={
                    "features": list(high_importance_features.keys()),
                    "importance_scores": high_importance_features,
                    "adjustment_factor": 1.3
                },
                expected_improvement=8.0,
                confidence=0.7,
                evidence=[]
            ))

        # Sort by priority and expected improvement
        priority_order = {"high": 0, "medium": 1, "low": 2}
        updates.sort(
            key=lambda u: (priority_order[u.priority], -u.expected_improvement)
        )

        logger.info(f"Generated {len(updates)} policy improvement suggestions")

        return updates


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Reflection Learning Engine Demo")
        print("="*80 + "\n")

        # Create reflection engine
        engine = ReflectionEngine()

        # Simulate some decision trajectories
        print("Simulating decision trajectories...\n")

        tools = ["calculator", "text_processor", "web_search"]

        for i in range(50):
            # Simulate trajectory
            tool = np.random.choice(tools)
            success = np.random.random() > 0.3  # 70% success rate

            trajectory = DecisionTrajectory(
                trajectory_id=f"traj_{i:03d}",
                query={"text": f"Query {i}"},
                features={"motifs": []},
                context={"shards": []},
                action_plan={"selected_tool": tool},
                outcome=Outcome(
                    success=success,
                    execution_time=np.random.uniform(0.5, 5.0),
                    user_satisfaction=np.random.random() if success else np.random.random() * 0.5,
                    correctness=np.random.random() if success else np.random.random() * 0.5,
                    efficiency=np.random.random()
                ),
                feedback=Feedback(
                    rating=np.random.uniform(3.0, 5.0) if success else np.random.uniform(1.0, 3.0),
                    helpful=success
                ) if np.random.random() > 0.5 else None
            )

            engine.record_trajectory(trajectory)

        # Analyze decisions
        print("\nAnalyzing decisions...\n")
        insights = await engine.analyze_decisions()

        print(f"📊 Analysis Results:")
        print(f"  • Trajectories analyzed: {insights.num_trajectories_analyzed}")
        print(f"  • Average quality: {insights.average_quality:.2f}")
        print(f"  • Success rate: {insights.success_rate:.1%}")
        print(f"  • Average execution time: {insights.average_execution_time:.2f}s")

        print(f"\n✅ Success Patterns ({len(insights.success_patterns)}):")
        for pattern in insights.success_patterns[:3]:
            print(f"  • {pattern.description}")
            print(f"    Frequency: {pattern.frequency}, Confidence: {pattern.confidence:.2f}")

        print(f"\n❌ Failure Patterns ({len(insights.failure_patterns)}):")
        for pattern in insights.failure_patterns[:3]:
            print(f"  • {pattern.description}")
            print(f"    Frequency: {pattern.frequency}, Confidence: {pattern.confidence:.2f}")

        print(f"\n🔧 Tool Performance:")
        for tool, stats in insights.tool_performance.items():
            success_rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  • {tool}:")
            print(f"    Success: {success_rate:.1%}, Quality: {stats['avg_quality']:.2f}, "
                  f"Avg Time: {stats['avg_time']:.2f}s")

        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(insights.recommendations, 1):
            print(f"  {i}. {rec}")

        # Generate improvements
        print("\n\nGenerating policy improvements...\n")
        updates = await engine.generate_improvements(insights)

        print(f"🚀 Suggested Policy Updates ({len(updates)}):\n")
        for i, update in enumerate(updates, 1):
            print(f"{i}. [{update.priority.upper()}] {update.description}")
            print(f"   Type: {update.update_type}")
            print(f"   Expected improvement: {update.expected_improvement:.1f}%")
            print(f"   Confidence: {update.confidence:.2f}")
            print(f"   Parameters: {update.parameters}")
            print()

        print("="*80)
        print("✅ Reflection Learning Engine Demo Complete!")
        print("="*80)

    asyncio.run(demo())

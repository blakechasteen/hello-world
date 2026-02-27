"""
MirrorCore Integration - Full HoloLoom Intelligence for Elle Core

Provides:
- Decision support engine (product prioritization, resource allocation)
- Knowledge management (HoloLoom RAG + memory)
- Pattern learning from operational data
- Predictive analytics (cash flow, demand forecasting)
- Seasonal planning automation

Created: 2025-11-15
Author: Blake Chasteen
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

# HoloLoom imports
try:
    from hololoom import hololoom
    from hololoom.rag import SimpleRAG
    from hololoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("Warning: HoloLoom not available. Decision support will be limited.")

from elle.sop_schema import SOP
from elle.tracker import TaskTracker, TaskResult


@dataclass
class Recommendation:
    """Operational recommendation from decision engine"""
    action: str  # "Prioritize bread production", "Prepare biochar batch"
    reasoning: str  # Why this action is recommended
    priority: str  # HIGH, MEDIUM, LOW
    estimated_profit: float
    time_required: float  # hours
    materials_needed: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    confidence: float = 0.8  # 0-1 confidence score


@dataclass
class ResourceAllocation:
    """Optimized resource allocation plan"""
    tasks: List[Dict[str, Any]]  # Sequenced task list
    total_hours: float
    total_cost: float
    expected_revenue: float
    expected_profit: float
    utilization: float  # % of available resources used


class DecisionEngine:
    """
    AI-powered decision support for farm operations

    Features:
    - ROI-based product prioritization
    - Seasonal awareness and planning
    - Resource optimization (time, materials, cash)
    - Bottleneck detection
    - Pattern learning from historical data
    """

    def __init__(
        self,
        sop_dir: str = "elle/sops",
        coz_dir: str = "coz",
        tracker: Optional[TaskTracker] = None
    ):
        self.sop_dir = Path(sop_dir)
        self.coz_dir = Path(coz_dir)
        self.tracker = tracker or TaskTracker()

        # Load SOPs
        self.sops: Dict[str, SOP] = {}
        self._load_sops()

        # Load Coz planning data
        self.business_plan = self._load_business_plan()
        self.seasonal_schedule = self._load_seasonal_schedule()
        self.financials = self._load_financials()

        # Current season/month
        self.current_month = datetime.now().strftime("%B")

    def _load_sops(self):
        """Load all SOPs"""
        if not self.sop_dir.exists():
            return

        for sop_file in self.sop_dir.glob("*.json"):
            try:
                with open(sop_file) as f:
                    data = json.load(f)
                    sop = SOP.from_dict(data)
                    self.sops[sop.sop_id] = sop
            except Exception as e:
                print(f"✗ Failed to load {sop_file}: {e}")

    def _load_business_plan(self) -> Dict[str, Any]:
        """Load business plan data"""
        # TODO: Parse BUSINESS_PLAN_DRAFT.md
        return {}

    def _load_seasonal_schedule(self) -> Dict[str, str]:
        """Load seasonal schedule from schedule.md"""
        schedule_file = self.coz_dir / "schedule.md"
        if not schedule_file.exists():
            return {}

        # Simple parse of schedule table
        schedule = {}
        try:
            with open(schedule_file) as f:
                lines = f.readlines()
                for line in lines:
                    if "|" in line and "Month" not in line and "---" not in line:
                        parts = [p.strip() for p in line.split("|")]
                        if len(parts) >= 3:
                            month = parts[1]
                            focus = parts[2]
                            if month and focus:
                                schedule[month] = focus
        except Exception as e:
            print(f"✗ Failed to load schedule: {e}")

        return schedule

    def _load_financials(self) -> Dict[str, Any]:
        """Load financial data from financials.md"""
        # TODO: Parse financials.md table
        return {}

    async def get_daily_recommendations(
        self,
        available_hours: float = 8.0,
        available_cash: Optional[float] = None
    ) -> List[Recommendation]:
        """
        Get prioritized recommendations for today

        Args:
            available_hours: Hours available for work
            available_cash: Cash available for materials

        Returns:
            List of recommendations sorted by priority
        """
        recommendations = []

        # Get historical performance
        history = await self.tracker.get_history(days=30, limit=100)

        # Calculate average ROI by category
        roi_by_category = self._calculate_roi_by_category(history)

        # Get seasonal focus
        seasonal_focus = self.seasonal_schedule.get(self.current_month, "")

        # Recommend based on ROI + seasonal alignment
        for sop in self.sops.values():
            # Skip if we don't have enough info
            if sop.hourly_roi == 0:
                continue

            # Check material availability (if cash constraint)
            if available_cash and sop.material_cost > available_cash:
                continue

            # Check time availability
            if sop.total_time_minutes and sop.total_time_minutes / 60 > available_hours:
                continue

            # Calculate priority
            priority_score = sop.hourly_roi

            # Boost if aligned with seasonal focus
            if seasonal_focus and sop.category.lower() in seasonal_focus.lower():
                priority_score *= 1.3

            # Boost if historically successful
            if sop.category in roi_by_category and roi_by_category[sop.category] > 15.0:
                priority_score *= 1.2

            # Determine priority level
            if priority_score > 20.0:
                priority = "HIGH"
            elif priority_score > 10.0:
                priority = "MEDIUM"
            else:
                priority = "LOW"

            # Build reasoning
            reasoning = f"{sop.name} has {sop.profit_margin:.0f}% margin. ROI: ${sop.hourly_roi:.2f}/hr. "
            if seasonal_focus and sop.category.lower() in seasonal_focus.lower():
                reasoning += f"Aligned with {self.current_month} focus: {seasonal_focus}. "
            if sop.batch_size:
                reasoning += f"Produces {sop.batch_size} {sop.batch_unit}. "

            recommendation = Recommendation(
                action=f"Prioritize {sop.name.lower()}",
                reasoning=reasoning,
                priority=priority,
                estimated_profit=sop.profit_per_batch,
                time_required=sop.total_time_minutes / 60 if sop.total_time_minutes else 0.0,
                materials_needed=[ing.name for ing in sop.ingredients],
                confidence=0.85
            )

            recommendations.append((priority_score, recommendation))

        # Sort by priority score
        recommendations.sort(key=lambda x: x[0], reverse=True)

        return [rec for score, rec in recommendations[:10]]  # Top 10

    def _calculate_roi_by_category(self, history: List[TaskResult]) -> Dict[str, float]:
        """Calculate average hourly ROI by category"""
        category_roi = {}

        for result in history:
            if result.category not in category_roi:
                category_roi[result.category] = []
            category_roi[result.category].append(result.hourly_roi)

        return {
            cat: sum(rois) / len(rois)
            for cat, rois in category_roi.items()
            if rois
        }

    async def allocate_resources(
        self,
        available_hours: float = 8.0,
        available_cash: Optional[float] = None,
        season: Optional[str] = None
    ) -> ResourceAllocation:
        """
        Optimize resource allocation across tasks

        Args:
            available_hours: Total hours available
            available_cash: Total cash budget
            season: Current season/month

        Returns:
            Optimized task sequence with expected outcomes
        """
        recommendations = await self.get_daily_recommendations(available_hours, available_cash)

        # Greedy allocation (simple optimization for now)
        # TODO: Use dynamic programming for true optimization

        allocated_tasks = []
        remaining_hours = available_hours
        remaining_cash = available_cash or float('inf')
        total_cost = 0.0
        total_revenue = 0.0
        total_profit = 0.0

        for rec in recommendations:
            # Check if we can fit this task
            if rec.time_required > remaining_hours:
                continue

            # Find associated SOP
            sop = None
            for s in self.sops.values():
                if s.name.lower() in rec.action.lower():
                    sop = s
                    break

            if not sop:
                continue

            # Check material cost
            if sop.material_cost > remaining_cash:
                continue

            # Allocate this task
            allocated_tasks.append({
                "sop_id": sop.sop_id,
                "task": rec.action,
                "time_hours": rec.time_required,
                "cost": sop.total_cost,
                "revenue": sop.selling_price * sop.batch_size if sop.selling_price and sop.batch_size else 0.0,
                "profit": rec.estimated_profit,
                "priority": rec.priority
            })

            remaining_hours -= rec.time_required
            remaining_cash -= sop.material_cost
            total_cost += sop.total_cost
            total_revenue += (sop.selling_price * sop.batch_size) if sop.selling_price and sop.batch_size else 0.0
            total_profit += rec.estimated_profit

        utilization = ((available_hours - remaining_hours) / available_hours) * 100 if available_hours > 0 else 0.0

        return ResourceAllocation(
            tasks=allocated_tasks,
            total_hours=available_hours - remaining_hours,
            total_cost=total_cost,
            expected_revenue=total_revenue,
            expected_profit=total_profit,
            utilization=utilization
        )

    async def detect_bottlenecks(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Detect operational bottlenecks

        Returns:
            List of bottlenecks with suggested fixes
        """
        history = await self.tracker.get_history(days=days, limit=1000)

        bottlenecks = []

        # Analyze quality scores
        low_quality_tasks = [t for t in history if t.quality_score and t.quality_score < 7.0]
        if low_quality_tasks:
            categories = set(t.category for t in low_quality_tasks)
            bottlenecks.append({
                "type": "quality",
                "description": f"Quality issues in {', '.join(categories)}",
                "affected_tasks": len(low_quality_tasks),
                "suggestion": "Review SOPs and training for these categories"
            })

        # Analyze profit margins
        low_margin_tasks = [t for t in history if t.profit_margin < 30.0]
        if len(low_margin_tasks) > len(history) * 0.3:  # >30% of tasks
            bottlenecks.append({
                "type": "margin",
                "description": "Many tasks have low profit margins (<30%)",
                "affected_tasks": len(low_margin_tasks),
                "suggestion": "Consider increasing prices or reducing costs"
            })

        # Analyze time overruns
        # TODO: Compare actual time vs SOP expected time

        return bottlenecks

    async def predict_cash_flow(
        self,
        weeks_ahead: int = 4
    ) -> List[Dict[str, float]]:
        """
        Predict cash flow for upcoming weeks

        Args:
            weeks_ahead: Number of weeks to predict

        Returns:
            List of predictions by week
        """
        history = await self.tracker.get_history(days=90, limit=1000)

        if not history:
            return []

        # Calculate average weekly revenue/profit by category
        category_weekly = {}

        for result in history:
            if result.category not in category_weekly:
                category_weekly[result.category] = {"revenue": [], "profit": []}
            category_weekly[result.category]["revenue"].append(result.revenue)
            category_weekly[result.category]["profit"].append(result.profit)

        # Average per category
        category_avg = {}
        for cat, data in category_weekly.items():
            category_avg[cat] = {
                "revenue": sum(data["revenue"]) / max(len(data["revenue"]), 1),
                "profit": sum(data["profit"]) / max(len(data["profit"]), 1)
            }

        # Project forward
        predictions = []
        for week in range(1, weeks_ahead + 1):
            week_revenue = sum(cat_data["revenue"] for cat_data in category_avg.values())
            week_profit = sum(cat_data["profit"] for cat_data in category_avg.values())

            # Apply seasonal adjustment (simple for now)
            # TODO: Learn seasonal patterns from historical data

            predictions.append({
                "week": week,
                "predicted_revenue": week_revenue,
                "predicted_profit": week_profit,
                "confidence": 0.7  # Lower confidence for further predictions
            })

        return predictions


class ElleKnowledge:
    """
    HoloLoom-powered knowledge management for Elle Core

    Integrates:
    - SOPs (searchable via RAG)
    - Task outcomes (for pattern learning)
    - Seasonal data
    - Business planning documents
    """

    def __init__(self, use_hololoom: bool = True):
        self.use_hololoom = use_hololoom and HOLOLOOM_AVAILABLE
        self.hololoom: Optional[HoloLoom] = None
        self.rag: Optional[SimpleRAG] = None

    async def initialize(self):
        """Initialize HoloLoom memory system"""
        if not self.use_hololoom:
            print("HoloLoom not available. Knowledge features will be limited.")
            return

        try:
            # Initialize HoloLoom
            config = Config.fast()
            self.hololoom = HoloLoom(config=config)

            # Initialize RAG
            self.rag = SimpleRAG(config=config)

            print("✓ HoloLoom knowledge system initialized")
        except Exception as e:
            print(f"✗ Failed to initialize HoloLoom: {e}")
            self.use_hololoom = False

    async def store_sop(self, sop: SOP):
        """Store SOP in memory"""
        if not self.use_hololoom:
            return

        try:
            # Convert SOP to markdown
            markdown = sop.to_markdown()

            # Ingest into HoloLoom
            await self.rag.ingest(markdown)

            print(f"✓ Stored SOP in knowledge base: {sop.name}")
        except Exception as e:
            print(f"✗ Failed to store SOP: {e}")

    async def get_sop(self, sop_name: str) -> Optional[str]:
        """Retrieve SOP by name"""
        if not self.use_hololoom or not self.rag:
            return None

        try:
            # Query for SOP
            result = await self.rag.query(
                f"Show me the complete SOP for {sop_name}",
                mode="direct"
            )
            return result.response
        except Exception as e:
            print(f"✗ Failed to retrieve SOP: {e}")
            return None

    async def query(
        self,
        question: str,
        mode: str = "direct"
    ) -> str:
        """
        Query knowledge base

        Args:
            question: Natural language question
            mode: Reasoning mode (direct, verify, research, plan_execute)

        Returns:
            Answer text
        """
        if not self.use_hololoom or not self.rag:
            return "Knowledge query not available (HoloLoom not initialized)"

        try:
            result = await self.rag.query(question, mode=mode)
            return result.response
        except Exception as e:
            print(f"✗ Query failed: {e}")
            return f"Query failed: {str(e)}"

    async def store_outcome(
        self,
        task: str,
        sop_used: str,
        quality_score: float,
        notes: str
    ):
        """Store task outcome for pattern learning"""
        if not self.use_hololoom:
            return

        try:
            # Create outcome description
            outcome_text = f"""
Task: {task}
SOP Used: {sop_used}
Quality Score: {quality_score}/10
Notes: {notes}
Date: {datetime.now().strftime('%Y-%m-%d')}
"""
            # Ingest into memory
            await self.rag.ingest(outcome_text)

            print(f"✓ Stored task outcome for pattern learning")
        except Exception as e:
            print(f"✗ Failed to store outcome: {e}")


# Example usage
async def main():
    """Example: Decision support and knowledge management"""

    print("\n" + "="*60)
    print("ELLE MIRRORCORE - Decision Support Engine")
    print("="*60 + "\n")

    # Initialize decision engine
    engine = DecisionEngine()

    # Get daily recommendations
    print("📋 DAILY RECOMMENDATIONS:\n")
    recommendations = await engine.get_daily_recommendations(
        available_hours=8.0,
        available_cash=500.0
    )

    for i, rec in enumerate(recommendations[:5], 1):
        print(f"{i}. [{rec.priority}] {rec.action}")
        print(f"   {rec.reasoning}")
        print(f"   Estimated Profit: ${rec.estimated_profit:.2f}")
        print(f"   Time Required: {rec.time_required:.1f} hours")
        print()

    # Get resource allocation
    print("\n📊 OPTIMIZED RESOURCE ALLOCATION:\n")
    allocation = await engine.allocate_resources(
        available_hours=8.0,
        available_cash=500.0
    )

    print(f"Total Hours: {allocation.total_hours:.1f} / 8.0 ({allocation.utilization:.0f}% utilization)")
    print(f"Total Cost: ${allocation.total_cost:.2f}")
    print(f"Expected Revenue: ${allocation.expected_revenue:.2f}")
    print(f"Expected Profit: ${allocation.expected_profit:.2f}")
    print(f"\nTasks:")
    for task in allocation.tasks:
        print(f"  - [{task['priority']}] {task['task']} ({task['time_hours']:.1f}hr, ${task['profit']:.2f} profit)")

    # Detect bottlenecks
    print("\n⚠️  BOTTLENECK ANALYSIS:\n")
    bottlenecks = await engine.detect_bottlenecks(days=30)

    if bottlenecks:
        for bottleneck in bottlenecks:
            print(f"Type: {bottleneck['type']}")
            print(f"  {bottleneck['description']}")
            print(f"  Affected Tasks: {bottleneck['affected_tasks']}")
            print(f"  Suggestion: {bottleneck['suggestion']}")
            print()
    else:
        print("No bottlenecks detected. Operations running smoothly!")

    # Cash flow prediction
    print("\n💵 CASH FLOW PREDICTION (4 weeks):\n")
    predictions = await engine.predict_cash_flow(weeks_ahead=4)

    for pred in predictions:
        print(f"Week {pred['week']}: ${pred['predicted_revenue']:.2f} revenue, ${pred['predicted_profit']:.2f} profit")


if __name__ == "__main__":
    asyncio.run(main())

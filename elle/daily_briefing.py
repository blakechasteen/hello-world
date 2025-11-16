"""
Daily Briefing System for Farm & Kitchen Cooperative

Provides automated morning operational briefing with:
- Today's priority tasks (kanban + seasonal focus)
- Weather-adjusted recommendations
- Inventory alerts
- Yesterday's profit summary
- Cash flow status

Created: 2025-11-16
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json

try:
    from elle.coz.sync_manager import SyncManager
    from elle.tracker import TaskTracker, TaskStatus
    from elle.budget import BudgetBuilder
    from elle.sop_schema import SOP
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from elle.coz.sync_manager import SyncManager
    from elle.tracker import TaskTracker, TaskStatus
    from elle.budget import BudgetBuilder
    from elle.sop_schema import SOP


@dataclass
class WeatherCondition:
    """Weather data for operational adjustments"""
    temperature_f: float
    condition: str  # sunny, cloudy, rainy, stormy
    humidity_percent: float
    timestamp: datetime

    @property
    def is_good_fermentation_weather(self) -> bool:
        """Ideal for fermentation: 65-75°F, moderate humidity"""
        return 65 <= self.temperature_f <= 75 and 40 <= self.humidity_percent <= 70

    @property
    def is_good_harvest_weather(self) -> bool:
        """Good for harvesting: dry, not too hot"""
        return self.condition in ["sunny", "cloudy"] and self.temperature_f < 85


@dataclass
class TaskRecommendation:
    """Single actionable task recommendation"""
    task_name: str
    priority: str  # HIGH, MEDIUM, LOW
    reason: str
    estimated_duration_minutes: Optional[int] = None
    estimated_profit: Optional[float] = None
    weather_dependent: bool = False

    def __str__(self) -> str:
        parts = [f"[{self.priority}] {self.task_name}"]
        if self.estimated_duration_minutes:
            hours = self.estimated_duration_minutes / 60
            parts.append(f"({hours:.1f}h)")
        if self.estimated_profit:
            parts.append(f"(${self.estimated_profit:.2f} profit)")
        parts.append(f"→ {self.reason}")
        return " ".join(parts)


@dataclass
class DailyBriefing:
    """Complete morning operational briefing"""
    date: datetime
    seasonal_focus: str
    priority_tasks: List[TaskRecommendation]
    inventory_alerts: List[str]
    yesterday_profit: float
    yesterday_hours: float
    cash_flow_status: str
    weather: Optional[WeatherCondition] = None
    budget_variance: Optional[float] = None

    def to_text(self) -> str:
        """Format as human-readable text"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"DAILY BRIEFING - {self.date.strftime('%A, %B %d, %Y')}")
        lines.append("=" * 60)
        lines.append("")

        # Seasonal focus
        lines.append(f"📅 SEASONAL FOCUS: {self.seasonal_focus}")
        lines.append("")

        # Weather (if available)
        if self.weather:
            lines.append(f"🌤️  WEATHER: {self.weather.temperature_f:.0f}°F, {self.weather.condition}, {self.weather.humidity_percent:.0f}% humidity")
            if self.weather.is_good_fermentation_weather:
                lines.append("   ✓ Perfect fermentation conditions!")
            if self.weather.is_good_harvest_weather:
                lines.append("   ✓ Good harvest weather")
            lines.append("")

        # Yesterday's performance
        lines.append("📊 YESTERDAY'S RESULTS:")
        lines.append(f"   Profit: ${self.yesterday_profit:.2f}")
        if self.yesterday_hours > 0:
            hourly = self.yesterday_profit / self.yesterday_hours
            lines.append(f"   Hours: {self.yesterday_hours:.1f} (${hourly:.2f}/hour)")
        else:
            lines.append("   No tasks completed")
        lines.append("")

        # Cash flow
        lines.append(f"💰 CASH FLOW: {self.cash_flow_status}")
        if self.budget_variance is not None:
            if self.budget_variance > 0:
                lines.append(f"   ✓ ${self.budget_variance:.2f} ahead of budget")
            else:
                lines.append(f"   ⚠ ${abs(self.budget_variance):.2f} behind budget")
        lines.append("")

        # Inventory alerts
        if self.inventory_alerts:
            lines.append("⚠️  INVENTORY ALERTS:")
            for alert in self.inventory_alerts:
                lines.append(f"   • {alert}")
            lines.append("")

        # Priority tasks
        lines.append("✅ TODAY'S PRIORITIES:")
        for i, task in enumerate(self.priority_tasks, 1):
            lines.append(f"   {i}. {task}")
        lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_voice(self) -> str:
        """Format as voice-friendly script"""
        parts = []
        parts.append(f"Good morning! Today is {self.date.strftime('%A, %B %d')}.")
        parts.append(f"Your seasonal focus is {self.seasonal_focus}.")

        # Weather
        if self.weather:
            parts.append(f"Current temperature is {self.weather.temperature_f:.0f} degrees, {self.weather.condition}.")
            if self.weather.is_good_fermentation_weather:
                parts.append("Perfect conditions for fermentation today.")

        # Yesterday
        if self.yesterday_profit > 0:
            parts.append(f"Yesterday you made {self.yesterday_profit:.2f} dollars in {self.yesterday_hours:.1f} hours.")
            if self.yesterday_hours > 0:
                hourly = self.yesterday_profit / self.yesterday_hours
                parts.append(f"That's {hourly:.2f} dollars per hour.")

        # Inventory
        if self.inventory_alerts:
            parts.append(f"You have {len(self.inventory_alerts)} inventory alerts.")
            for alert in self.inventory_alerts[:3]:  # Limit to 3 for voice
                parts.append(alert)

        # Tasks
        high_priority = [t for t in self.priority_tasks if t.priority == "HIGH"]
        if high_priority:
            parts.append(f"You have {len(high_priority)} high priority tasks today.")
            for task in high_priority[:3]:  # Top 3
                parts.append(f"{task.task_name}. {task.reason}.")

        return " ".join(parts)


class DailyBriefingEngine:
    """Generates comprehensive daily operational briefings"""

    def __init__(
        self,
        coz_dir: str = "coz",
        data_dir: str = "elle/data",
        sop_dir: str = "elle/sops"
    ):
        self.coz_sync = SyncManager(coz_dir=coz_dir)
        self.tracker = TaskTracker(db_path=f"{data_dir}/tasks.db")
        self.budget_builder = BudgetBuilder()
        self.sop_dir = sop_dir

        # Load available SOPs
        self.sops: Dict[str, SOP] = {}
        self._load_sops()

    def _load_sops(self):
        """Load all available SOPs (simplified - read JSON directly)"""
        from pathlib import Path
        sop_path = Path(self.sop_dir)
        if sop_path.exists():
            for sop_file in sop_path.glob("*.json"):
                try:
                    with open(sop_file) as f:
                        data = json.load(f)
                        # Extract key metrics directly from JSON
                        product_name = data.get("product_name", data.get("name", "Unknown"))
                        self.sops[product_name] = data
                except Exception as e:
                    print(f"Warning: Could not load SOP {sop_file}: {e}")

    async def get_weather(self) -> Optional[WeatherCondition]:
        """
        Get current weather (graceful fallback)

        In production, integrate with weather API:
        - OpenWeatherMap
        - Weather.gov
        - Local weather station
        """
        try:
            # TODO: Integrate with actual weather API
            # For now, return mock data
            return WeatherCondition(
                temperature_f=70.0,
                condition="sunny",
                humidity_percent=55.0,
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Weather unavailable: {e}")
            return None

    def get_yesterday_summary(self) -> tuple[float, float]:
        """Get yesterday's profit and hours (from task history database)"""
        try:
            # Query the database directly for yesterday's completed tasks
            import sqlite3
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

            conn = sqlite3.connect(self.tracker.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT actual_profit, actual_time_minutes
                FROM tasks
                WHERE status = 'COMPLETED'
                AND end_time >= ? AND end_time <= ?
            """, (yesterday_start.isoformat(), yesterday_end.isoformat()))

            results = cursor.fetchall()
            conn.close()

            total_profit = sum(row[0] for row in results if row[0])
            total_hours = sum(row[1] / 60.0 for row in results if row[1])

            return total_profit, total_hours

        except Exception as e:
            # Graceful fallback if no task history
            return 0.0, 0.0

    def get_inventory_alerts(self) -> List[str]:
        """Get low inventory alerts"""
        alerts = []

        # Get inventory data from Coz
        inventory_status = self.coz_sync.get_inventory_status()

        # Check for low stock items
        low_stock = inventory_status.get("low_stock", [])
        for item in low_stock:
            alerts.append(f"Low stock: {item['name']} ({item['quantity']} {item['unit']})")

        return alerts

    def get_priority_tasks(self, seasonal_focus: str, weather: Optional[WeatherCondition]) -> List[TaskRecommendation]:
        """Get prioritized task recommendations"""
        recommendations = []

        # Get daily tasks from kanban
        daily_tasks_data = self.coz_sync.get_daily_tasks()
        daily_plan = daily_tasks_data.get("suggested_plan", [])

        for task in daily_plan:
            priority = task.get("priority", "MEDIUM")
            task_name = task.get("task", "Unknown")
            category = task.get("category", "")

            # Determine reason
            reason = f"From daily plan ({category})"

            # Check if we have an SOP for this task
            estimated_profit = None
            estimated_duration = None
            for product_name, sop_data in self.sops.items():
                if product_name.lower() in task_name.lower():
                    # Calculate profit from SOP data
                    cost = sop_data.get("cost_per_batch", 0)
                    retail_price = sop_data.get("retail_price", 0)
                    batch_size = sop_data.get("batch_size", 1)
                    revenue = retail_price * batch_size
                    profit = revenue - cost
                    margin = (profit / revenue * 100) if revenue > 0 else 0

                    estimated_profit = profit
                    estimated_duration = sop_data.get("total_time_minutes")
                    reason = f"${profit:.2f} profit, {margin:.0f}% margin"
                    break

            # Weather adjustment
            weather_dependent = False
            if weather and category.lower() in ["harvest", "outdoor", "garden"]:
                weather_dependent = True
                if not weather.is_good_harvest_weather:
                    priority = "LOW"  # Downgrade outdoor tasks in bad weather
                    reason += " (weather permitting)"

            recommendations.append(TaskRecommendation(
                task_name=task_name,
                priority=priority,
                reason=reason,
                estimated_duration_minutes=estimated_duration,
                estimated_profit=estimated_profit,
                weather_dependent=weather_dependent
            ))

        # Add high-margin SOPs if no tasks scheduled
        if len(recommendations) < 3:
            # Calculate ROI for each SOP
            sop_with_roi = []
            for product_name, sop_data in self.sops.items():
                cost = sop_data.get("cost_per_batch", 0)
                retail_price = sop_data.get("retail_price", 0)
                batch_size = sop_data.get("batch_size", 1)
                total_time = sop_data.get("total_time_minutes", 1)

                revenue = retail_price * batch_size
                profit = revenue - cost
                margin = (profit / revenue * 100) if revenue > 0 else 0
                hourly_roi = (profit / (total_time / 60)) if total_time > 0 else 0

                sop_with_roi.append((product_name, sop_data, hourly_roi, profit, margin))

            # Sort by hourly ROI
            for product_name, sop_data, hourly_roi, profit, margin in sorted(
                sop_with_roi,
                key=lambda x: x[2],
                reverse=True
            )[:3]:
                # Check if already in recommendations
                if not any(product_name.lower() in r.task_name.lower() for r in recommendations):
                    recommendations.append(TaskRecommendation(
                        task_name=f"Produce {product_name}",
                        priority="MEDIUM",
                        reason=f"High ROI: ${hourly_roi:.2f}/hour, {margin:.0f}% margin",
                        estimated_duration_minutes=sop_data.get("total_time_minutes"),
                        estimated_profit=profit
                    ))

        # Sort by priority then profit
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        recommendations.sort(
            key=lambda x: (
                priority_order.get(x.priority, 1),
                -(x.estimated_profit or 0)
            )
        )

        return recommendations

    def get_cash_flow_status(self) -> tuple[str, Optional[float]]:
        """Get current cash flow status and budget variance"""
        # Get current month budget
        current_month = datetime.now().strftime("%Y%m")
        budget_path = f"elle/data/budgets/BUDGET_{current_month}.json"

        try:
            with open(budget_path) as f:
                budget_data = json.load(f)

            # Calculate actual performance to date
            month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            completed_tasks = self.tracker.get_completed_tasks(start_date=month_start)

            actual_revenue = sum(t.actual_profit for t in completed_tasks if t.actual_profit)

            # Compare to budget
            total_budgeted_revenue = budget_data.get("total_revenue", 0)

            # Pro-rate budget to current day of month
            days_in_month = 30  # Approximate
            current_day = datetime.now().day
            expected_revenue = (total_budgeted_revenue / days_in_month) * current_day

            variance = actual_revenue - expected_revenue

            if variance > 0:
                status = f"✓ HEALTHY (${variance:.2f} ahead)"
            elif variance > -100:
                status = f"⚠ ON TRACK (${abs(variance):.2f} behind)"
            else:
                status = f"⚠ ATTENTION NEEDED (${abs(variance):.2f} behind)"

            return status, variance

        except Exception as e:
            return "No budget data available", None

    async def generate_briefing(self) -> DailyBriefing:
        """Generate complete daily briefing"""
        # Sync Coz data
        self.coz_sync.parse_all()

        # Get all components
        weather = await self.get_weather()
        yesterday_profit, yesterday_hours = self.get_yesterday_summary()

        seasonal_context = self.coz_sync.get_seasonal_context()
        seasonal_focus = seasonal_context.get("current_focus", "General operations")

        inventory_alerts = self.get_inventory_alerts()
        priority_tasks = self.get_priority_tasks(seasonal_focus, weather)
        cash_flow_status, budget_variance = self.get_cash_flow_status()

        return DailyBriefing(
            date=datetime.now(),
            seasonal_focus=seasonal_focus,
            priority_tasks=priority_tasks,
            inventory_alerts=inventory_alerts,
            yesterday_profit=yesterday_profit,
            yesterday_hours=yesterday_hours,
            cash_flow_status=cash_flow_status,
            weather=weather,
            budget_variance=budget_variance
        )


async def main():
    """Demo: Generate and display daily briefing"""
    engine = DailyBriefingEngine()

    print("Generating daily briefing...\n")
    briefing = await engine.generate_briefing()

    print(briefing.to_text())

    print("\n" + "=" * 60)
    print("VOICE VERSION:")
    print("=" * 60)
    print(briefing.to_voice())


if __name__ == "__main__":
    asyncio.run(main())

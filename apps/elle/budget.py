"""
Budget Builder for Elle Core

Comprehensive budgeting and financial planning system that integrates with:
- Task tracker (actual costs)
- SOPs (planned costs)
- Coz financials (reinvestment targets)
- Decision engine (budget-aware recommendations)

Features:
- Monthly/quarterly/annual budgets
- Category-based tracking
- Variance analysis (actual vs budget)
- Cash flow forecasting
- Reinvestment planning
- What-if scenarios
- Automatic alerts

Created: 2025-11-15
Author: Blake Chasteen
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import json
import sqlite3

from elle.tracker import TaskTracker, TaskResult


class BudgetPeriod(Enum):
    """Budget time period"""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class BudgetCategory(Enum):
    """Budget categories aligned with farm operations"""
    # Revenue categories
    REVENUE_BAKERY = "revenue_bakery"
    REVENUE_KITCHEN = "revenue_kitchen"
    REVENUE_APOTHECARY = "revenue_apothecary"
    REVENUE_FARM = "revenue_farm"
    REVENUE_SOIL = "revenue_soil"

    # Cost categories
    COST_MATERIALS = "cost_materials"
    COST_LABOR = "cost_labor"
    COST_OVERHEAD = "cost_overhead"
    COST_EQUIPMENT = "cost_equipment"
    COST_INFRASTRUCTURE = "cost_infrastructure"

    # Reinvestment categories (from Coz financials.md)
    REINVEST_INFRASTRUCTURE = "reinvest_infrastructure"  # 15%
    REINVEST_EQUIPMENT = "reinvest_equipment"  # 10%
    REINVEST_BUFFER = "reinvest_buffer"  # 5%


@dataclass
class BudgetLine:
    """Single budget line item"""
    category: BudgetCategory
    description: str
    planned_amount: float
    actual_amount: float = 0.0
    variance: float = 0.0  # actual - planned
    variance_pct: float = 0.0  # (actual - planned) / planned
    notes: str = ""

    def update_actual(self, amount: float):
        """Update actual amount and recalculate variance"""
        self.actual_amount = amount
        self.variance = self.actual_amount - self.planned_amount
        if self.planned_amount != 0:
            self.variance_pct = (self.variance / self.planned_amount) * 100
        else:
            self.variance_pct = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "planned_amount": self.planned_amount,
            "actual_amount": self.actual_amount,
            "variance": self.variance,
            "variance_pct": self.variance_pct,
            "notes": self.notes
        }


@dataclass
class Budget:
    """Complete budget for a time period"""
    budget_id: str
    name: str
    period: BudgetPeriod
    start_date: datetime
    end_date: datetime

    # Budget lines
    revenue_lines: List[BudgetLine] = field(default_factory=list)
    cost_lines: List[BudgetLine] = field(default_factory=list)
    reinvestment_lines: List[BudgetLine] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    notes: str = ""

    @property
    def total_planned_revenue(self) -> float:
        """Total planned revenue"""
        return sum(line.planned_amount for line in self.revenue_lines)

    @property
    def total_actual_revenue(self) -> float:
        """Total actual revenue"""
        return sum(line.actual_amount for line in self.revenue_lines)

    @property
    def total_planned_costs(self) -> float:
        """Total planned costs"""
        return sum(line.planned_amount for line in self.cost_lines)

    @property
    def total_actual_costs(self) -> float:
        """Total actual costs"""
        return sum(line.actual_amount for line in self.cost_lines)

    @property
    def planned_profit(self) -> float:
        """Planned profit (revenue - costs)"""
        return self.total_planned_revenue - self.total_planned_costs

    @property
    def actual_profit(self) -> float:
        """Actual profit (revenue - costs)"""
        return self.total_actual_revenue - self.total_actual_costs

    @property
    def profit_variance(self) -> float:
        """Profit variance (actual - planned)"""
        return self.actual_profit - self.planned_profit

    @property
    def planned_margin(self) -> float:
        """Planned profit margin %"""
        if self.total_planned_revenue == 0:
            return 0.0
        return (self.planned_profit / self.total_planned_revenue) * 100

    @property
    def actual_margin(self) -> float:
        """Actual profit margin %"""
        if self.total_actual_revenue == 0:
            return 0.0
        return (self.actual_profit / self.total_actual_revenue) * 100

    @property
    def completion_pct(self) -> float:
        """Budget period completion %"""
        total_days = (self.end_date - self.start_date).days
        elapsed_days = (datetime.now() - self.start_date).days
        return min(100.0, max(0.0, (elapsed_days / total_days) * 100))

    def add_revenue_line(
        self,
        category: BudgetCategory,
        description: str,
        planned_amount: float
    ):
        """Add revenue budget line"""
        line = BudgetLine(
            category=category,
            description=description,
            planned_amount=planned_amount
        )
        self.revenue_lines.append(line)
        self.updated_at = datetime.now()

    def add_cost_line(
        self,
        category: BudgetCategory,
        description: str,
        planned_amount: float
    ):
        """Add cost budget line"""
        line = BudgetLine(
            category=category,
            description=description,
            planned_amount=planned_amount
        )
        self.cost_lines.append(line)
        self.updated_at = datetime.now()

    def calculate_reinvestment(self):
        """
        Calculate reinvestment budget based on Coz financials.md:
        - 15% infrastructure
        - 10% equipment
        - 5% buffer
        """
        # Use actual profit if available, otherwise planned
        profit = self.actual_profit if self.actual_profit > 0 else self.planned_profit

        self.reinvestment_lines = [
            BudgetLine(
                category=BudgetCategory.REINVEST_INFRASTRUCTURE,
                description="Long-term infrastructure (mushroom logs, nursery)",
                planned_amount=profit * 0.15
            ),
            BudgetLine(
                category=BudgetCategory.REINVEST_EQUIPMENT,
                description="Equipment & tool upgrades",
                planned_amount=profit * 0.10
            ),
            BudgetLine(
                category=BudgetCategory.REINVEST_BUFFER,
                description="Emergency operating buffer",
                planned_amount=profit * 0.05
            )
        ]

        self.updated_at = datetime.now()

    def update_actuals_from_tracker(self, tracker: TaskTracker):
        """
        Update actual amounts from task tracker history

        Args:
            tracker: TaskTracker instance with historical data
        """
        # Get tasks for this budget period
        tasks = asyncio.run(tracker.get_history(days=365, limit=10000))

        # Filter to budget period
        period_tasks = [
            t for t in tasks
            if t.start_time and self.start_date <= t.start_time <= self.end_date
        ]

        # Aggregate by category
        category_revenue = {}
        category_costs = {}

        for task in period_tasks:
            # Map task category to budget category
            if task.category.lower() in ['bakery', 'bread']:
                rev_cat = BudgetCategory.REVENUE_BAKERY
            elif task.category.lower() in ['kitchen', 'meal', 'goat']:
                rev_cat = BudgetCategory.REVENUE_KITCHEN
            elif task.category.lower() in ['apothecary', 'honey', 'cleanse']:
                rev_cat = BudgetCategory.REVENUE_APOTHECARY
            elif task.category.lower() in ['farm', 'nursery']:
                rev_cat = BudgetCategory.REVENUE_FARM
            elif task.category.lower() in ['soil', 'biochar', 'compost']:
                rev_cat = BudgetCategory.REVENUE_SOIL
            else:
                rev_cat = BudgetCategory.REVENUE_KITCHEN  # Default

            # Accumulate revenue
            if rev_cat not in category_revenue:
                category_revenue[rev_cat] = 0.0
            category_revenue[rev_cat] += task.revenue

            # Accumulate costs
            if BudgetCategory.COST_MATERIALS not in category_costs:
                category_costs[BudgetCategory.COST_MATERIALS] = 0.0
            category_costs[BudgetCategory.COST_MATERIALS] += task.material_cost

            if BudgetCategory.COST_LABOR not in category_costs:
                category_costs[BudgetCategory.COST_LABOR] = 0.0
            category_costs[BudgetCategory.COST_LABOR] += task.labor_cost

            if BudgetCategory.COST_OVERHEAD not in category_costs:
                category_costs[BudgetCategory.COST_OVERHEAD] = 0.0
            category_costs[BudgetCategory.COST_OVERHEAD] += task.overhead_cost

        # Update revenue lines
        for line in self.revenue_lines:
            if line.category in category_revenue:
                line.update_actual(category_revenue[line.category])

        # Update cost lines
        for line in self.cost_lines:
            if line.category in category_costs:
                line.update_actual(category_costs[line.category])

        # Recalculate reinvestment based on actual profit
        self.calculate_reinvestment()

        self.updated_at = datetime.now()

    def get_variance_report(self) -> str:
        """Generate variance report"""
        report = f"\n{'='*70}\n"
        report += f"BUDGET VARIANCE REPORT: {self.name}\n"
        report += f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n"
        report += f"Completion: {self.completion_pct:.0f}%\n"
        report += f"{'='*70}\n\n"

        # Revenue section
        report += "REVENUE:\n"
        report += f"{'Category':<30} {'Planned':<12} {'Actual':<12} {'Variance':<12} {'Var %':<8}\n"
        report += "-" * 70 + "\n"

        for line in self.revenue_lines:
            variance_sign = "+" if line.variance >= 0 else ""
            report += f"{line.description:<30} ${line.planned_amount:<11.2f} ${line.actual_amount:<11.2f} "
            report += f"{variance_sign}${line.variance:<10.2f} {variance_sign}{line.variance_pct:<6.1f}%\n"

        report += "-" * 70 + "\n"
        report += f"{'TOTAL REVENUE':<30} ${self.total_planned_revenue:<11.2f} ${self.total_actual_revenue:<11.2f} "
        variance = self.total_actual_revenue - self.total_planned_revenue
        variance_pct = (variance / self.total_planned_revenue * 100) if self.total_planned_revenue > 0 else 0
        variance_sign = "+" if variance >= 0 else ""
        report += f"{variance_sign}${variance:<10.2f} {variance_sign}{variance_pct:<6.1f}%\n\n"

        # Costs section
        report += "COSTS:\n"
        report += f"{'Category':<30} {'Planned':<12} {'Actual':<12} {'Variance':<12} {'Var %':<8}\n"
        report += "-" * 70 + "\n"

        for line in self.cost_lines:
            variance_sign = "+" if line.variance >= 0 else ""
            report += f"{line.description:<30} ${line.planned_amount:<11.2f} ${line.actual_amount:<11.2f} "
            report += f"{variance_sign}${line.variance:<10.2f} {variance_sign}{line.variance_pct:<6.1f}%\n"

        report += "-" * 70 + "\n"
        report += f"{'TOTAL COSTS':<30} ${self.total_planned_costs:<11.2f} ${self.total_actual_costs:<11.2f} "
        variance = self.total_actual_costs - self.total_planned_costs
        variance_pct = (variance / self.total_planned_costs * 100) if self.total_planned_costs > 0 else 0
        variance_sign = "+" if variance >= 0 else ""
        report += f"{variance_sign}${variance:<10.2f} {variance_sign}{variance_pct:<6.1f}%\n\n"

        # Profit section
        report += "PROFIT:\n"
        report += f"{'Metric':<30} {'Planned':<12} {'Actual':<12} {'Variance':<12}\n"
        report += "-" * 70 + "\n"
        report += f"{'Profit':<30} ${self.planned_profit:<11.2f} ${self.actual_profit:<11.2f} "
        profit_var_sign = "+" if self.profit_variance >= 0 else ""
        report += f"{profit_var_sign}${self.profit_variance:<10.2f}\n"
        report += f"{'Profit Margin':<30} {self.planned_margin:<11.1f}% {self.actual_margin:<11.1f}%\n"

        # Reinvestment section
        if self.reinvestment_lines:
            report += "\nREINVESTMENT ALLOCATION:\n"
            report += f"{'Category':<30} {'Amount':<12} {'% of Profit':<12}\n"
            report += "-" * 70 + "\n"
            for line in self.reinvestment_lines:
                pct = (line.planned_amount / self.actual_profit * 100) if self.actual_profit > 0 else 0
                report += f"{line.description:<30} ${line.planned_amount:<11.2f} {pct:<11.1f}%\n"

        report += "\n" + "="*70 + "\n"

        return report

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            "budget_id": self.budget_id,
            "name": self.name,
            "period": self.period.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "revenue_lines": [line.to_dict() for line in self.revenue_lines],
            "cost_lines": [line.to_dict() for line in self.cost_lines],
            "reinvestment_lines": [line.to_dict() for line in self.reinvestment_lines],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "notes": self.notes,
            "summary": {
                "total_planned_revenue": self.total_planned_revenue,
                "total_actual_revenue": self.total_actual_revenue,
                "total_planned_costs": self.total_planned_costs,
                "total_actual_costs": self.total_actual_costs,
                "planned_profit": self.planned_profit,
                "actual_profit": self.actual_profit,
                "profit_variance": self.profit_variance,
                "planned_margin": self.planned_margin,
                "actual_margin": self.actual_margin,
                "completion_pct": self.completion_pct
            }
        }


class BudgetBuilder:
    """
    Budget creation and management system

    Features:
    - Create monthly/quarterly/annual budgets
    - Automatic variance tracking
    - Integration with task tracker for actuals
    - Cash flow forecasting
    - What-if scenario planning
    - Budget alerts
    """

    def __init__(
        self,
        data_dir: str = "elle/data",
        tracker: Optional[TaskTracker] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = tracker or TaskTracker()

        # Active budgets
        self.budgets: Dict[str, Budget] = {}

        # Load existing budgets
        self._load_budgets()

    def _load_budgets(self):
        """Load budgets from disk"""
        budget_dir = self.data_dir / "budgets"
        if not budget_dir.exists():
            return

        for budget_file in budget_dir.glob("*.json"):
            try:
                with open(budget_file) as f:
                    data = json.load(f)
                    # TODO: Implement from_dict method on Budget class
                    # budget = Budget.from_dict(data)
                    # self.budgets[budget.budget_id] = budget
                    pass
            except Exception as e:
                print(f"✗ Failed to load budget {budget_file}: {e}")

    def _save_budget(self, budget: Budget):
        """Save budget to disk"""
        budget_dir = self.data_dir / "budgets"
        budget_dir.mkdir(parents=True, exist_ok=True)

        budget_file = budget_dir / f"{budget.budget_id}.json"
        with open(budget_file, "w") as f:
            json.dump(budget.to_dict(), f, indent=2)

    def create_monthly_budget(
        self,
        month: int,
        year: int,
        name: Optional[str] = None
    ) -> Budget:
        """
        Create a monthly budget

        Args:
            month: Month number (1-12)
            year: Year
            name: Optional budget name

        Returns:
            Budget instance
        """
        import calendar

        start_date = datetime(year, month, 1)
        _, last_day = calendar.monthrange(year, month)
        end_date = datetime(year, month, last_day, 23, 59, 59)

        budget_id = f"BUDGET_{year}{month:02d}"
        budget_name = name or f"{calendar.month_name[month]} {year} Budget"

        budget = Budget(
            budget_id=budget_id,
            name=budget_name,
            period=BudgetPeriod.MONTHLY,
            start_date=start_date,
            end_date=end_date
        )

        self.budgets[budget_id] = budget
        self._save_budget(budget)

        print(f"✓ Created budget: {budget_name} ({budget_id})")

        return budget

    def create_annual_budget(
        self,
        year: int,
        name: Optional[str] = None
    ) -> Budget:
        """Create an annual budget"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)

        budget_id = f"BUDGET_{year}_ANNUAL"
        budget_name = name or f"{year} Annual Budget"

        budget = Budget(
            budget_id=budget_id,
            name=budget_name,
            period=BudgetPeriod.ANNUAL,
            start_date=start_date,
            end_date=end_date
        )

        self.budgets[budget_id] = budget
        self._save_budget(budget)

        print(f"✓ Created budget: {budget_name} ({budget_id})")

        return budget

    def build_budget_from_sops(
        self,
        budget: Budget,
        sop_dir: str = "elle/sops"
    ):
        """
        Build budget estimates from SOPs

        Analyzes all SOPs and creates revenue/cost projections
        """
        from elle.sop_schema import SOP

        sop_path = Path(sop_dir)
        if not sop_path.exists():
            print(f"✗ SOP directory not found: {sop_dir}")
            return

        # Load all SOPs
        sops = []
        for sop_file in sop_path.glob("*.json"):
            try:
                with open(sop_file) as f:
                    data = json.load(f)
                    sop = SOP.from_dict(data)
                    sops.append(sop)
            except Exception as e:
                print(f"✗ Failed to load SOP {sop_file}: {e}")

        # Estimate batches per period
        days_in_period = (budget.end_date - budget.start_date).days

        for sop in sops:
            # Estimate how many batches per period
            # Simple heuristic: weekly production for high-margin items
            if sop.profit_margin > 50:
                batches_per_week = 2
            elif sop.profit_margin > 30:
                batches_per_week = 1
            else:
                batches_per_week = 0.5

            weeks = days_in_period / 7
            estimated_batches = batches_per_week * weeks

            # Map category
            if sop.category.lower() in ['bakery', 'bread']:
                rev_cat = BudgetCategory.REVENUE_BAKERY
            elif sop.category.lower() in ['kitchen', 'meal', 'goat']:
                rev_cat = BudgetCategory.REVENUE_KITCHEN
            elif sop.category.lower() in ['apothecary', 'honey']:
                rev_cat = BudgetCategory.REVENUE_APOTHECARY
            elif sop.category.lower() in ['farm', 'nursery']:
                rev_cat = BudgetCategory.REVENUE_FARM
            elif sop.category.lower() in ['soil', 'biochar']:
                rev_cat = BudgetCategory.REVENUE_SOIL
            else:
                rev_cat = BudgetCategory.REVENUE_KITCHEN

            # Add revenue line
            if sop.selling_price and sop.batch_size:
                planned_revenue = (sop.selling_price * sop.batch_size) * estimated_batches
                budget.add_revenue_line(
                    category=rev_cat,
                    description=f"{sop.name} ({estimated_batches:.0f} batches)",
                    planned_amount=planned_revenue
                )

            # Add cost lines
            planned_materials = sop.material_cost * estimated_batches
            planned_labor = sop.labor_cost * estimated_batches
            planned_overhead = (planned_materials + planned_labor) * 0.15

            # Find or create cost lines
            mat_line = next((l for l in budget.cost_lines if l.category == BudgetCategory.COST_MATERIALS), None)
            if mat_line:
                mat_line.planned_amount += planned_materials
            else:
                budget.add_cost_line(
                    category=BudgetCategory.COST_MATERIALS,
                    description="Total materials cost",
                    planned_amount=planned_materials
                )

            lab_line = next((l for l in budget.cost_lines if l.category == BudgetCategory.COST_LABOR), None)
            if lab_line:
                lab_line.planned_amount += planned_labor
            else:
                budget.add_cost_line(
                    category=BudgetCategory.COST_LABOR,
                    description="Total labor cost",
                    planned_amount=planned_labor
                )

            over_line = next((l for l in budget.cost_lines if l.category == BudgetCategory.COST_OVERHEAD), None)
            if over_line:
                over_line.planned_amount += planned_overhead
            else:
                budget.add_cost_line(
                    category=BudgetCategory.COST_OVERHEAD,
                    description="Total overhead (15%)",
                    planned_amount=planned_overhead
                )

        # Calculate reinvestment
        budget.calculate_reinvestment()

        # Save
        self._save_budget(budget)

        print(f"✓ Built budget from {len(sops)} SOPs")
        print(f"  Planned Revenue: ${budget.total_planned_revenue:.2f}")
        print(f"  Planned Costs: ${budget.total_planned_costs:.2f}")
        print(f"  Planned Profit: ${budget.planned_profit:.2f} ({budget.planned_margin:.1f}% margin)")

    def update_actuals(self, budget_id: str):
        """Update budget actuals from task tracker"""
        if budget_id not in self.budgets:
            print(f"✗ Budget not found: {budget_id}")
            return

        budget = self.budgets[budget_id]
        budget.update_actuals_from_tracker(self.tracker)
        self._save_budget(budget)

        print(f"✓ Updated actuals for {budget.name}")
        print(f"  Actual Revenue: ${budget.total_actual_revenue:.2f}")
        print(f"  Actual Costs: ${budget.total_actual_costs:.2f}")
        print(f"  Actual Profit: ${budget.actual_profit:.2f} ({budget.actual_margin:.1f}% margin)")

    def get_variance_report(self, budget_id: str) -> str:
        """Get variance report for budget"""
        if budget_id not in self.budgets:
            return f"Budget not found: {budget_id}"

        budget = self.budgets[budget_id]
        return budget.get_variance_report()

    def forecast_cash_flow(
        self,
        budget: Budget,
        weeks: int = 12
    ) -> List[Dict[str, float]]:
        """
        Forecast weekly cash flow based on budget

        Returns list of weekly forecasts with:
        - week number
        - projected revenue
        - projected costs
        - projected profit
        - cumulative cash
        """
        weeks_in_period = (budget.end_date - budget.start_date).days / 7
        weekly_revenue = budget.total_planned_revenue / weeks_in_period
        weekly_costs = budget.total_planned_costs / weeks_in_period

        forecast = []
        cumulative_cash = 0.0

        for week in range(1, weeks + 1):
            weekly_profit = weekly_revenue - weekly_costs
            cumulative_cash += weekly_profit

            forecast.append({
                "week": week,
                "revenue": weekly_revenue,
                "costs": weekly_costs,
                "profit": weekly_profit,
                "cumulative_cash": cumulative_cash
            })

        return forecast


# Demo
async def main():
    """Demo budget builder"""
    print("\n" + "="*70)
    print("ELLE CORE - Budget Builder Demo")
    print("="*70 + "\n")

    builder = BudgetBuilder()

    # Create November 2025 budget
    print("📊 Creating November 2025 budget...\n")
    budget = builder.create_monthly_budget(month=11, year=2025)

    # Build from SOPs
    print("\n📝 Building budget from SOPs...\n")
    builder.build_budget_from_sops(budget)

    # Show variance report
    print("\n" + budget.get_variance_report())

    # Cash flow forecast
    print("\n💵 12-WEEK CASH FLOW FORECAST:\n")
    forecast = builder.forecast_cash_flow(budget, weeks=12)

    print(f"{'Week':<6} {'Revenue':<12} {'Costs':<12} {'Profit':<12} {'Cumulative':<12}")
    print("-" * 60)
    for f in forecast[:8]:  # Show first 8 weeks
        print(f"{f['week']:<6} ${f['revenue']:<11.2f} ${f['costs']:<11.2f} ${f['profit']:<11.2f} ${f['cumulative_cash']:<11.2f}")

    print(f"\nProjected profit after 12 weeks: ${forecast[-1]['cumulative_cash']:.2f}")


if __name__ == "__main__":
    asyncio.run(main())

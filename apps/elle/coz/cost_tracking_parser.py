"""
Cost Tracking Parser for COZ System

Tracks actual costs per task/project (materials, labor, overhead), enabling:
- Profit margin calculation (revenue - costs)
- Cost variance analysis (actual vs. estimated)
- Cost per hour calculation
- Monthly spending trends
- Overhead rate calculation

Part of COZ Expansion Phase 1 (Core Tracking)
Created: 2025-11-21
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict


@dataclass
class CostEntry:
    """Single cost tracking entry"""
    date: datetime
    task_id: str
    task_name: str
    material_cost: float
    labor_cost: float
    overhead_cost: float
    total_cost: float
    notes: str

    # Calculated fields (set externally via integration with other parsers)
    cost_per_hour: Optional[float] = None
    profit_margin: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "date": self.date.isoformat(),
            "task_id": self.task_id,
            "task_name": self.task_name,
            "material_cost": round(self.material_cost, 2),
            "labor_cost": round(self.labor_cost, 2),
            "overhead_cost": round(self.overhead_cost, 2),
            "total_cost": round(self.total_cost, 2),
            "notes": self.notes,
            "cost_per_hour": round(self.cost_per_hour, 2) if self.cost_per_hour else None,
            "profit_margin": round(self.profit_margin, 2) if self.profit_margin else None,
        }


class CostTrackingParser:
    """Parser for cost_tracking.csv file"""

    def __init__(self, file_path: str = "coz/cost_tracking.csv"):
        self.file_path = Path(file_path)
        self.entries: List[CostEntry] = []
        self.entries_by_task_id: Dict[str, CostEntry] = {}
        self.entries_by_date: Dict[datetime, List[CostEntry]] = defaultdict(list)

    def parse(self) -> List[CostEntry]:
        """Parse cost_tracking.csv file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Cost tracking file not found: {self.file_path}")

        self.entries = []
        self.entries_by_task_id = {}
        self.entries_by_date = defaultdict(list)

        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse date
                    date = datetime.strptime(row['Date'], '%Y-%m-%d')

                    # Parse costs
                    material_cost = float(row['Material Cost'])
                    labor_cost = float(row['Labor Cost'])
                    overhead_cost = float(row['Overhead Cost'])
                    total_cost = float(row['Total Cost'])

                    # Create entry
                    entry = CostEntry(
                        date=date,
                        task_id=row['Task ID'].strip(),
                        task_name=row['Task Name'].strip(),
                        material_cost=material_cost,
                        labor_cost=labor_cost,
                        overhead_cost=overhead_cost,
                        total_cost=total_cost,
                        notes=row.get('Notes', '').strip()
                    )

                    self.entries.append(entry)
                    self.entries_by_task_id[entry.task_id] = entry
                    self.entries_by_date[date.date()].append(entry)

                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed row in {self.file_path}: {e}")
                    continue

        return self.entries

    def get_cost_summary(self) -> Dict:
        """Get overall cost summary"""
        if not self.entries:
            return {
                "total_material_cost": 0.0,
                "total_labor_cost": 0.0,
                "total_overhead_cost": 0.0,
                "total_cost": 0.0,
                "entries_count": 0,
                "avg_cost_per_task": 0.0,
            }

        total_material = sum(e.material_cost for e in self.entries)
        total_labor = sum(e.labor_cost for e in self.entries)
        total_overhead = sum(e.overhead_cost for e in self.entries)
        total_cost = sum(e.total_cost for e in self.entries)

        return {
            "total_material_cost": round(total_material, 2),
            "total_labor_cost": round(total_labor, 2),
            "total_overhead_cost": round(total_overhead, 2),
            "total_cost": round(total_cost, 2),
            "entries_count": len(self.entries),
            "avg_cost_per_task": round(total_cost / len(self.entries), 2) if self.entries else 0.0,
        }

    def get_by_task(self, task_id: str) -> Optional[CostEntry]:
        """Get cost entry for a specific task"""
        return self.entries_by_task_id.get(task_id)

    def get_monthly_costs(self, year: Optional[int] = None, month: Optional[int] = None) -> Dict:
        """Get costs for a specific month (default: current month)"""
        if year is None or month is None:
            now = datetime.now()
            year = year or now.year
            month = month or now.month

        month_entries = [
            e for e in self.entries
            if e.date.year == year and e.date.month == month
        ]

        if not month_entries:
            return {
                "year": year,
                "month": month,
                "entries_count": 0,
                "total_cost": 0.0,
                "material_cost": 0.0,
                "labor_cost": 0.0,
                "overhead_cost": 0.0,
            }

        return {
            "year": year,
            "month": month,
            "entries_count": len(month_entries),
            "total_cost": round(sum(e.total_cost for e in month_entries), 2),
            "material_cost": round(sum(e.material_cost for e in month_entries), 2),
            "labor_cost": round(sum(e.labor_cost for e in month_entries), 2),
            "overhead_cost": round(sum(e.overhead_cost for e in month_entries), 2),
        }

    def get_cost_variance(self, task_id: str, estimated_cost: float) -> Dict:
        """Compare actual cost vs. estimated cost"""
        entry = self.get_by_task(task_id)

        if not entry:
            return {
                "error": f"Task {task_id} not found"
            }

        variance = entry.total_cost - estimated_cost
        variance_percent = (variance / estimated_cost) if estimated_cost > 0 else 0.0

        return {
            "task_id": task_id,
            "task_name": entry.task_name,
            "estimated_cost": round(estimated_cost, 2),
            "actual_cost": round(entry.total_cost, 2),
            "variance": round(variance, 2),
            "variance_percent": round(variance_percent * 100, 1),
            "over_budget": variance > 0,
        }

    def get_profit_by_task(self, revenue_map: Dict[str, float]) -> List[Dict]:
        """Calculate profit for each task given revenue data"""
        profit_data = []

        for entry in self.entries:
            revenue = revenue_map.get(entry.task_id, 0.0)
            profit = revenue - entry.total_cost
            profit_margin = (profit / revenue) if revenue > 0 else 0.0

            profit_data.append({
                "task_id": entry.task_id,
                "task_name": entry.task_name,
                "revenue": round(revenue, 2),
                "total_cost": round(entry.total_cost, 2),
                "profit": round(profit, 2),
                "profit_margin": round(profit_margin * 100, 1),  # As percentage
            })

        # Sort by profit (highest first)
        profit_data.sort(key=lambda x: x['profit'], reverse=True)
        return profit_data

    def get_overhead_rate(self) -> Dict:
        """Calculate overhead as percentage of direct costs"""
        if not self.entries:
            return {
                "overhead_rate": 0.0,
                "total_direct_costs": 0.0,
                "total_overhead": 0.0,
            }

        total_direct = sum(e.material_cost + e.labor_cost for e in self.entries)
        total_overhead = sum(e.overhead_cost for e in self.entries)

        overhead_rate = (total_overhead / total_direct) if total_direct > 0 else 0.0

        return {
            "overhead_rate": round(overhead_rate * 100, 1),  # As percentage
            "total_direct_costs": round(total_direct, 2),
            "total_overhead": round(total_overhead, 2),
        }

    def get_cost_breakdown(self) -> Dict:
        """Get cost breakdown by category (material, labor, overhead)"""
        summary = self.get_cost_summary()

        if summary['total_cost'] == 0:
            return {
                "material_percent": 0.0,
                "labor_percent": 0.0,
                "overhead_percent": 0.0,
            }

        total = summary['total_cost']

        return {
            "material_percent": round((summary['total_material_cost'] / total) * 100, 1),
            "labor_percent": round((summary['total_labor_cost'] / total) * 100, 1),
            "overhead_percent": round((summary['total_overhead_cost'] / total) * 100, 1),
        }

    def get_top_expensive_tasks(self, limit: int = 10) -> List[Dict]:
        """Get most expensive tasks"""
        sorted_entries = sorted(self.entries, key=lambda e: e.total_cost, reverse=True)

        return [
            {
                "task_id": entry.task_id,
                "task_name": entry.task_name,
                "total_cost": round(entry.total_cost, 2),
                "material_cost": round(entry.material_cost, 2),
                "labor_cost": round(entry.labor_cost, 2),
                "overhead_cost": round(entry.overhead_cost, 2),
            }
            for entry in sorted_entries[:limit]
        ]

    def integrate_time_tracking(self, time_entries: List['TimeEntry']):
        """Integrate with time tracking to calculate cost per hour"""
        time_map = {e.task_id: e.actual_hours for e in time_entries}

        for entry in self.entries:
            hours = time_map.get(entry.task_id)
            if hours and hours > 0:
                entry.cost_per_hour = entry.total_cost / hours


if __name__ == "__main__":
    # Demo usage
    parser = CostTrackingParser()

    try:
        entries = parser.parse()

        print(f"✅ Parsed {len(entries)} cost entries")
        print("\n💰 Cost Summary:")
        summary = parser.get_cost_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n📊 Cost Breakdown:")
        breakdown = parser.get_cost_breakdown()
        for key, value in breakdown.items():
            print(f"  {key}: {value}%")

        print("\n💸 Top 5 Most Expensive Tasks:")
        top_tasks = parser.get_top_expensive_tasks(limit=5)
        for task in top_tasks:
            print(f"  - {task['task_name']}: ${task['total_cost']}")

        print("\n🔢 Overhead Rate:")
        overhead = parser.get_overhead_rate()
        print(f"  Rate: {overhead['overhead_rate']}%")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

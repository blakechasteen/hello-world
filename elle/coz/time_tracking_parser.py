"""
Time Tracking Parser for COZ System

Tracks actual hours worked on tasks vs. estimates, enabling:
- Efficiency analysis (actual vs. estimated time)
- Hourly cost calculation (labor costs)
- Task time prediction based on historical data
- Category-level time breakdown

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
class TimeEntry:
    """Single time tracking entry"""
    date: datetime
    task_id: str
    task_name: str
    category: str
    estimated_hours: float
    actual_hours: float
    notes: str

    # Calculated fields
    variance_hours: float = 0.0  # actual - estimated
    variance_percent: float = 0.0  # (actual - estimated) / estimated
    efficiency_score: float = 1.0  # estimated / actual (1.0 = on time, >1.0 = over)

    def __post_init__(self):
        """Calculate derived fields"""
        self.variance_hours = self.actual_hours - self.estimated_hours

        if self.estimated_hours > 0:
            self.variance_percent = (self.actual_hours - self.estimated_hours) / self.estimated_hours
            self.efficiency_score = self.estimated_hours / self.actual_hours if self.actual_hours > 0 else 1.0
        else:
            self.variance_percent = 0.0
            self.efficiency_score = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "date": self.date.isoformat(),
            "task_id": self.task_id,
            "task_name": self.task_name,
            "category": self.category,
            "estimated_hours": self.estimated_hours,
            "actual_hours": self.actual_hours,
            "notes": self.notes,
            "variance_hours": round(self.variance_hours, 2),
            "variance_percent": round(self.variance_percent * 100, 1),  # As percentage
            "efficiency_score": round(self.efficiency_score, 2),
        }


class TimeTrackingParser:
    """Parser for time_tracking.csv file"""

    def __init__(self, file_path: str = "coz/time_tracking.csv"):
        self.file_path = Path(file_path)
        self.entries: List[TimeEntry] = []
        self.entries_by_category: Dict[str, List[TimeEntry]] = {}
        self.entries_by_task_id: Dict[str, TimeEntry] = {}

    def parse(self) -> List[TimeEntry]:
        """Parse time_tracking.csv file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Time tracking file not found: {self.file_path}")

        self.entries = []
        self.entries_by_category = defaultdict(list)
        self.entries_by_task_id = {}

        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse date
                    date = datetime.strptime(row['Date'], '%Y-%m-%d')

                    # Create entry
                    entry = TimeEntry(
                        date=date,
                        task_id=row['Task ID'].strip(),
                        task_name=row['Task Name'].strip(),
                        category=row['Category'].strip(),
                        estimated_hours=float(row['Estimated Hours']),
                        actual_hours=float(row['Actual Hours']),
                        notes=row.get('Notes', '').strip()
                    )

                    self.entries.append(entry)
                    self.entries_by_category[entry.category].append(entry)
                    self.entries_by_task_id[entry.task_id] = entry

                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed row in {self.file_path}: {e}")
                    continue

        return self.entries

    def get_time_summary(self) -> Dict:
        """Get overall time summary"""
        if not self.entries:
            return {
                "total_estimated_hours": 0.0,
                "total_actual_hours": 0.0,
                "total_variance_hours": 0.0,
                "avg_efficiency_score": 1.0,
                "entries_count": 0,
            }

        total_estimated = sum(e.estimated_hours for e in self.entries)
        total_actual = sum(e.actual_hours for e in self.entries)
        total_variance = total_actual - total_estimated
        avg_efficiency = sum(e.efficiency_score for e in self.entries) / len(self.entries)

        return {
            "total_estimated_hours": round(total_estimated, 1),
            "total_actual_hours": round(total_actual, 1),
            "total_variance_hours": round(total_variance, 1),
            "avg_efficiency_score": round(avg_efficiency, 2),
            "entries_count": len(self.entries),
        }

    def get_by_category(self, category: str) -> List[TimeEntry]:
        """Get all time entries for a category"""
        return self.entries_by_category.get(category, [])

    def get_category_summary(self) -> Dict[str, Dict]:
        """Get time summary by category"""
        summary = {}

        for category, entries in self.entries_by_category.items():
            total_estimated = sum(e.estimated_hours for e in entries)
            total_actual = sum(e.actual_hours for e in entries)
            avg_efficiency = sum(e.efficiency_score for e in entries) / len(entries)

            summary[category] = {
                "entries_count": len(entries),
                "total_estimated_hours": round(total_estimated, 1),
                "total_actual_hours": round(total_actual, 1),
                "variance_hours": round(total_actual - total_estimated, 1),
                "avg_efficiency_score": round(avg_efficiency, 2),
            }

        return summary

    def get_efficiency_by_task(self) -> List[Dict]:
        """Get tasks sorted by efficiency (worst first)"""
        task_data = []

        for entry in self.entries:
            task_data.append({
                "task_id": entry.task_id,
                "task_name": entry.task_name,
                "category": entry.category,
                "estimated_hours": entry.estimated_hours,
                "actual_hours": entry.actual_hours,
                "variance_hours": entry.variance_hours,
                "efficiency_score": entry.efficiency_score,
            })

        # Sort by efficiency (lowest first = worst overruns)
        task_data.sort(key=lambda x: x['efficiency_score'])
        return task_data

    def get_weekly_summary(self, start_date: Optional[datetime] = None) -> Dict:
        """Get hours worked in the last 7 days"""
        if start_date is None:
            start_date = datetime.now()

        week_ago = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = datetime(week_ago.year, week_ago.month, week_ago.day) - timedelta(days=7)

        week_entries = [e for e in self.entries if e.date >= week_ago]

        if not week_entries:
            return {
                "entries_count": 0,
                "total_hours": 0.0,
                "avg_hours_per_day": 0.0,
            }

        total_hours = sum(e.actual_hours for e in week_entries)

        return {
            "entries_count": len(week_entries),
            "total_hours": round(total_hours, 1),
            "avg_hours_per_day": round(total_hours / 7, 1),
        }

    def get_hourly_cost(self, hourly_rate: float = 25.0) -> Dict:
        """Calculate labor cost based on actual hours"""
        summary = self.get_time_summary()

        total_cost = summary['total_actual_hours'] * hourly_rate
        avg_task_cost = total_cost / summary['entries_count'] if summary['entries_count'] > 0 else 0.0

        return {
            "hourly_rate": hourly_rate,
            "total_hours": summary['total_actual_hours'],
            "total_cost": round(total_cost, 2),
            "avg_task_cost": round(avg_task_cost, 2),
        }

    def predict_task_time(self, task_name: str) -> Optional[float]:
        """Predict task time based on historical average"""
        # Find all entries with similar task name
        similar_entries = [e for e in self.entries if task_name.lower() in e.task_name.lower()]

        if not similar_entries:
            return None

        avg_actual = sum(e.actual_hours for e in similar_entries) / len(similar_entries)
        return round(avg_actual, 1)

    def get_overruns(self, threshold: float = 0.2) -> List[Dict]:
        """Get tasks with significant time overruns (>20% by default)"""
        overruns = []

        for entry in self.entries:
            if entry.variance_percent > threshold:
                overruns.append({
                    "task_id": entry.task_id,
                    "task_name": entry.task_name,
                    "category": entry.category,
                    "estimated_hours": entry.estimated_hours,
                    "actual_hours": entry.actual_hours,
                    "variance_percent": round(entry.variance_percent * 100, 1),
                    "notes": entry.notes,
                })

        # Sort by variance (worst first)
        overruns.sort(key=lambda x: x['variance_percent'], reverse=True)
        return overruns


# Convenience import
from datetime import timedelta


if __name__ == "__main__":
    # Demo usage
    parser = TimeTrackingParser()

    try:
        entries = parser.parse()

        print(f"✅ Parsed {len(entries)} time entries")
        print("\n📊 Time Summary:")
        summary = parser.get_time_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n📂 Category Summary:")
        category_summary = parser.get_category_summary()
        for category, data in category_summary.items():
            print(f"\n  {category}:")
            for key, value in data.items():
                print(f"    {key}: {value}")

        print("\n⚠️ Time Overruns (>20%):")
        overruns = parser.get_overruns(threshold=0.2)
        for overrun in overruns[:5]:  # Top 5
            print(f"  - {overrun['task_name']}: {overrun['variance_percent']}% over")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

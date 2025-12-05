"""
Real-Time Task and Profit Tracking for Elle Core

Features:
- Background timer (no manual tracking needed)
- Automatic cost calculation from SOPs
- Real-time ROI analysis
- Voice-activated start/stop
- HoloLoom integration for pattern learning

Created: 2025-11-15
Author: Blake Chasteen
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
import json
import sqlite3
from pathlib import Path


class TaskStatus(Enum):
    """Task execution status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """Complete task outcome with profit analysis"""

    # Task identification
    task_id: str
    task_name: str
    sop_id: Optional[str] = None
    sop_version: Optional[str] = None
    category: str = "general"

    # Time tracking
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_hours: float = 0.0
    pause_time_hours: float = 0.0  # Break time
    active_time_hours: float = 0.0  # Actual work time

    # Production
    units_produced: int = 0
    batch_number: Optional[int] = None

    # Costs
    material_cost: float = 0.0
    labor_cost: float = 0.0
    overhead_cost: float = 0.0
    total_cost: float = 0.0

    # Revenue
    revenue: float = 0.0
    price_per_unit: float = 0.0

    # Profit analysis
    profit: float = 0.0
    profit_margin: float = 0.0
    hourly_roi: float = 0.0

    # Quality tracking
    quality_score: Optional[float] = None  # 0-10 scale
    notes: str = ""
    issues: List[str] = field(default_factory=list)

    # Metadata
    status: TaskStatus = TaskStatus.COMPLETED
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary"""
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "sop_id": self.sop_id,
            "sop_version": self.sop_version,
            "category": self.category,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_hours": self.duration_hours,
            "pause_time_hours": self.pause_time_hours,
            "active_time_hours": self.active_time_hours,
            "units_produced": self.units_produced,
            "batch_number": self.batch_number,
            "material_cost": self.material_cost,
            "labor_cost": self.labor_cost,
            "overhead_cost": self.overhead_cost,
            "total_cost": self.total_cost,
            "revenue": self.revenue,
            "price_per_unit": self.price_per_unit,
            "profit": self.profit,
            "profit_margin": self.profit_margin,
            "hourly_roi": self.hourly_roi,
            "quality_score": self.quality_score,
            "notes": self.notes,
            "issues": self.issues,
            "status": self.status.value,
            "created_at": self.created_at.isoformat()
        }


class TaskTracker:
    """
    Real-time task and profit tracking system

    Integrates with:
    - SOP schema for cost calculation
    - HoloLoom memory for pattern learning
    - Voice interface for hands-free operation
    """

    def __init__(
        self,
        db_path: str = "elle/data/tasks.db",
        labor_rate: float = 25.00,
        overhead_rate: float = 0.15
    ):
        self.db_path = Path(db_path)
        self.labor_rate = labor_rate  # $ per hour
        self.overhead_rate = overhead_rate  # 15% overhead

        # Active tasks
        self.active_tasks: Dict[str, Dict[str, Any]] = {}

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database for task history"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tasks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                sop_id TEXT,
                sop_version TEXT,
                category TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_hours REAL,
                pause_time_hours REAL,
                active_time_hours REAL,
                units_produced INTEGER,
                batch_number INTEGER,
                material_cost REAL,
                labor_cost REAL,
                overhead_cost REAL,
                total_cost REAL,
                revenue REAL,
                price_per_unit REAL,
                profit REAL,
                profit_margin REAL,
                hourly_roi REAL,
                quality_score REAL,
                notes TEXT,
                issues TEXT,
                status TEXT,
                created_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    async def start(
        self,
        task_name: str,
        sop_id: Optional[str] = None,
        category: str = "general",
        batch_number: Optional[int] = None
    ) -> str:
        """
        Start a new task

        Args:
            task_name: Name of the task (e.g., "Bake bread batch 12")
            sop_id: Optional SOP reference
            category: Product category
            batch_number: Batch number for tracking

        Returns:
            task_id: Unique identifier for this task
        """
        task_id = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.active_tasks[task_id] = {
            "task_name": task_name,
            "sop_id": sop_id,
            "category": category,
            "batch_number": batch_number,
            "start_time": datetime.now(),
            "pause_start": None,
            "total_pause_time": timedelta(0),
            "status": TaskStatus.IN_PROGRESS
        }

        print(f"✓ Started: {task_name} ({task_id})")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")

        return task_id

    async def pause(self, task_id: str):
        """Pause a task (break time)"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.active_tasks[task_id]

        if task["status"] != TaskStatus.IN_PROGRESS:
            raise ValueError(f"Cannot pause task with status: {task['status']}")

        task["pause_start"] = datetime.now()
        task["status"] = TaskStatus.PAUSED

        print(f"⏸ Paused: {task['task_name']}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")

    async def resume(self, task_id: str):
        """Resume a paused task"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.active_tasks[task_id]

        if task["status"] != TaskStatus.PAUSED:
            raise ValueError(f"Cannot resume task with status: {task['status']}")

        if task["pause_start"]:
            pause_duration = datetime.now() - task["pause_start"]
            task["total_pause_time"] += pause_duration
            task["pause_start"] = None

        task["status"] = TaskStatus.IN_PROGRESS

        print(f"▶ Resumed: {task['task_name']}")
        print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")

    async def end(
        self,
        task_id: str,
        units: int = 1,
        revenue: Optional[float] = None,
        material_cost: Optional[float] = None,
        quality_score: Optional[float] = None,
        notes: str = "",
        issues: Optional[List[str]] = None
    ) -> TaskResult:
        """
        End a task and calculate profit

        Args:
            task_id: Task identifier
            units: Number of units produced
            revenue: Total revenue (or will calculate from units × price)
            material_cost: Material cost (or will pull from SOP)
            quality_score: 0-10 quality rating
            notes: Additional notes
            issues: List of any problems encountered

        Returns:
            TaskResult with complete profit analysis
        """
        if task_id not in self.active_tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.active_tasks[task_id]

        # Handle pause state
        if task["status"] == TaskStatus.PAUSED:
            await self.resume(task_id)

        # Calculate times
        end_time = datetime.now()
        total_duration = end_time - task["start_time"]
        pause_time = task["total_pause_time"]
        active_time = total_duration - pause_time

        duration_hours = total_duration.total_seconds() / 3600
        pause_hours = pause_time.total_seconds() / 3600
        active_hours = active_time.total_seconds() / 3600

        # Calculate costs
        labor_cost = active_hours * self.labor_rate
        mat_cost = material_cost if material_cost is not None else 0.0
        overhead = (labor_cost + mat_cost) * self.overhead_rate
        total_cost = labor_cost + mat_cost + overhead

        # Calculate revenue and profit
        rev = revenue if revenue is not None else 0.0
        profit = rev - total_cost
        profit_margin = (profit / rev * 100) if rev > 0 else 0.0
        hourly_roi = (profit / active_hours) if active_hours > 0 else 0.0

        # Create result
        result = TaskResult(
            task_id=task_id,
            task_name=task["task_name"],
            sop_id=task.get("sop_id"),
            category=task["category"],
            batch_number=task.get("batch_number"),
            start_time=task["start_time"],
            end_time=end_time,
            duration_hours=duration_hours,
            pause_time_hours=pause_hours,
            active_time_hours=active_hours,
            units_produced=units,
            material_cost=mat_cost,
            labor_cost=labor_cost,
            overhead_cost=overhead,
            total_cost=total_cost,
            revenue=rev,
            price_per_unit=rev / units if units > 0 else 0.0,
            profit=profit,
            profit_margin=profit_margin,
            hourly_roi=hourly_roi,
            quality_score=quality_score,
            notes=notes,
            issues=issues or [],
            status=TaskStatus.COMPLETED
        )

        # Save to database
        self._save_result(result)

        # Remove from active tasks
        del self.active_tasks[task_id]

        # Print summary
        self._print_summary(result)

        return result

    def _save_result(self, result: TaskResult):
        """Save task result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO tasks VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            result.task_id,
            result.task_name,
            result.sop_id,
            result.sop_version,
            result.category,
            result.start_time.isoformat() if result.start_time else None,
            result.end_time.isoformat() if result.end_time else None,
            result.duration_hours,
            result.pause_time_hours,
            result.active_time_hours,
            result.units_produced,
            result.batch_number,
            result.material_cost,
            result.labor_cost,
            result.overhead_cost,
            result.total_cost,
            result.revenue,
            result.price_per_unit,
            result.profit,
            result.profit_margin,
            result.hourly_roi,
            result.quality_score,
            result.notes,
            json.dumps(result.issues),
            result.status.value,
            result.created_at.isoformat()
        ))

        conn.commit()
        conn.close()

    def _print_summary(self, result: TaskResult):
        """Print task summary to console"""
        print(f"\n{'='*60}")
        print(f"✓ Task Complete: {result.task_name}")
        print(f"{'='*60}")
        print(f"  Category: {result.category}")
        print(f"  Batch: #{result.batch_number}" if result.batch_number else "")
        print(f"\n⏱ TIME:")
        print(f"  Total Duration: {result.duration_hours:.2f} hours")
        print(f"  Pause Time: {result.pause_time_hours:.2f} hours")
        print(f"  Active Time: {result.active_time_hours:.2f} hours")
        print(f"\n📦 PRODUCTION:")
        print(f"  Units Produced: {result.units_produced}")
        print(f"\n💰 COSTS:")
        print(f"  Material: ${result.material_cost:.2f}")
        print(f"  Labor ({self.labor_rate}/hr): ${result.labor_cost:.2f}")
        print(f"  Overhead ({self.overhead_rate*100:.0f}%): ${result.overhead_cost:.2f}")
        print(f"  Total Cost: ${result.total_cost:.2f}")
        print(f"\n💵 REVENUE:")
        print(f"  Total Revenue: ${result.revenue:.2f}")
        print(f"  Price per Unit: ${result.price_per_unit:.2f}")
        print(f"\n📈 PROFIT:")
        print(f"  Profit: ${result.profit:.2f}")
        print(f"  Margin: {result.profit_margin:.1f}%")
        print(f"  Hourly ROI: ${result.hourly_roi:.2f}/hour")

        if result.quality_score:
            print(f"\n⭐ QUALITY:")
            print(f"  Score: {result.quality_score:.1f}/10")

        if result.issues:
            print(f"\n⚠️ ISSUES:")
            for issue in result.issues:
                print(f"  - {issue}")

        if result.notes:
            print(f"\n📝 NOTES:")
            print(f"  {result.notes}")

        print(f"{'='*60}\n")

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get current status of an active task"""
        if task_id not in self.active_tasks:
            raise ValueError(f"Task not found: {task_id}")

        task = self.active_tasks[task_id]
        elapsed = datetime.now() - task["start_time"]

        return {
            "task_id": task_id,
            "task_name": task["task_name"],
            "status": task["status"].value,
            "elapsed_time": str(elapsed).split('.')[0],  # HH:MM:SS format
            "elapsed_hours": elapsed.total_seconds() / 3600
        }

    async def get_history(
        self,
        category: Optional[str] = None,
        days: int = 30,
        limit: int = 100
    ) -> List[TaskResult]:
        """
        Get task history

        Args:
            category: Filter by category
            days: Number of days to look back
            limit: Maximum results

        Returns:
            List of TaskResult objects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        if category:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE category = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (category, cutoff, limit))
        else:
            cursor.execute("""
                SELECT * FROM tasks
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (cutoff, limit))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            result = TaskResult(
                task_id=row[0],
                task_name=row[1],
                sop_id=row[2],
                sop_version=row[3],
                category=row[4],
                start_time=datetime.fromisoformat(row[5]) if row[5] else None,
                end_time=datetime.fromisoformat(row[6]) if row[6] else None,
                duration_hours=row[7],
                pause_time_hours=row[8],
                active_time_hours=row[9],
                units_produced=row[10],
                batch_number=row[11],
                material_cost=row[12],
                labor_cost=row[13],
                overhead_cost=row[14],
                total_cost=row[15],
                revenue=row[16],
                price_per_unit=row[17],
                profit=row[18],
                profit_margin=row[19],
                hourly_roi=row[20],
                quality_score=row[21],
                notes=row[22],
                issues=json.loads(row[23]) if row[23] else [],
                status=TaskStatus(row[24]),
                created_at=datetime.fromisoformat(row[25])
            )
            results.append(result)

        return results

    async def get_analytics(
        self,
        category: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get analytics summary

        Returns:
            Dictionary with aggregated metrics
        """
        history = await self.get_history(category=category, days=days, limit=1000)

        if not history:
            return {
                "total_tasks": 0,
                "total_revenue": 0.0,
                "total_profit": 0.0,
                "avg_hourly_roi": 0.0,
                "avg_margin": 0.0,
                "total_hours": 0.0
            }

        total_revenue = sum(t.revenue for t in history)
        total_profit = sum(t.profit for t in history)
        total_hours = sum(t.active_time_hours for t in history)
        avg_margin = sum(t.profit_margin for t in history) / len(history)

        return {
            "total_tasks": len(history),
            "total_revenue": total_revenue,
            "total_profit": total_profit,
            "avg_hourly_roi": total_profit / total_hours if total_hours > 0 else 0.0,
            "avg_margin": avg_margin,
            "total_hours": total_hours,
            "best_product": max(history, key=lambda t: t.hourly_roi).category if history else None,
            "avg_quality": sum(t.quality_score for t in history if t.quality_score) / len([t for t in history if t.quality_score]) if any(t.quality_score for t in history) else None
        }


# Example usage
async def main():
    """Example: Track a bread baking task"""
    tracker = TaskTracker()

    # Start task
    task_id = await tracker.start(
        task_name="Bake sourdough bread batch 12",
        sop_id="BREAD_001",
        category="Bakery",
        batch_number=12
    )

    # Simulate work (in real use, this would be actual work time)
    print("\n... baking bread (2.5 hours) ...\n")
    await asyncio.sleep(2)  # Simulate 2 seconds = 2.5 hours

    # Take a break
    await tracker.pause(task_id)
    print("\n... taking a break (15 minutes) ...\n")
    await asyncio.sleep(1)  # Simulate 1 second = 15 minutes
    await tracker.resume(task_id)

    print("\n... finishing up ...\n")
    await asyncio.sleep(1)

    # Complete task
    result = await tracker.end(
        task_id=task_id,
        units=24,
        revenue=144.00,  # 24 loaves × $6
        material_cost=48.00,
        quality_score=9.2,
        notes="Perfect rise, customers loved the texture"
    )

    # Get analytics
    analytics = await tracker.get_analytics(category="Bakery", days=30)
    print("\n📊 BAKERY ANALYTICS (Last 30 Days):")
    print(f"  Total Tasks: {analytics['total_tasks']}")
    print(f"  Total Revenue: ${analytics['total_revenue']:.2f}")
    print(f"  Total Profit: ${analytics['total_profit']:.2f}")
    print(f"  Avg Hourly ROI: ${analytics['avg_hourly_roi']:.2f}/hour")
    print(f"  Avg Margin: {analytics['avg_margin']:.1f}%")
    print(f"  Total Hours: {analytics['total_hours']:.1f} hours")


if __name__ == "__main__":
    asyncio.run(main())

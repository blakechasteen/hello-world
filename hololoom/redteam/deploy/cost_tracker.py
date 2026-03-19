"""
CARTS Sandbox Cost Tracking
============================

Track and report compute costs for sandbox executions.

Supports:
- Per-job cost calculation
- Budget tracking and alerts
- Cost attribution by labels (team, project, campaign)
- Export to CSV for billing

Author: CARTS Team
Date: 2025-12-09
"""

import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("carts-deploy.cost")


@dataclass
class CostEntry:
    """Single cost entry for a job execution."""
    job_id: str
    timestamp: float
    duration_seconds: float
    cpu_cores: float
    memory_gb: float
    cost_usd: float
    labels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "duration_seconds": self.duration_seconds,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "cost_usd": self.cost_usd,
            **self.labels,
        }


@dataclass
class CostRate:
    """Cost rates for compute resources."""
    # Rates per hour
    cpu_per_hour: float = 0.05      # $0.05 per CPU-hour
    memory_gb_per_hour: float = 0.01  # $0.01 per GB-hour
    network_gb: float = 0.10        # $0.10 per GB transferred

    def calculate(
        self,
        duration_seconds: float,
        cpu_cores: float,
        memory_gb: float,
        network_gb: float = 0,
    ) -> float:
        """Calculate cost in USD."""
        hours = duration_seconds / 3600

        cpu_cost = cpu_cores * hours * self.cpu_per_hour
        memory_cost = memory_gb * hours * self.memory_gb_per_hour
        network_cost = network_gb * self.network_gb

        return cpu_cost + memory_cost + network_cost


class CostTracker:
    """Tracks sandbox execution costs."""

    def __init__(
        self,
        rate: CostRate = None,
        budget_usd: float = None,
        persist_path: Path = None,
    ):
        self.rate = rate or CostRate()
        self.budget_usd = budget_usd
        self.persist_path = persist_path
        self.entries: list[CostEntry] = []
        self._load()

    def _load(self) -> None:
        """Load persisted entries."""
        if self.persist_path and self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                for e in data.get("entries", []):
                    self.entries.append(CostEntry(**e))
                logger.info(f"Loaded {len(self.entries)} cost entries")
            except Exception as e:
                logger.warning(f"Failed to load cost data: {e}")

    def _save(self) -> None:
        """Persist entries to file."""
        if self.persist_path:
            try:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "entries": [e.to_dict() for e in self.entries],
                    "total_cost_usd": self.get_total_cost(),
                    "budget_usd": self.budget_usd,
                    "updated_at": datetime.now().isoformat(),
                }
                self.persist_path.write_text(json.dumps(data, indent=2))
            except Exception as e:
                logger.warning(f"Failed to save cost data: {e}")

    def record(
        self,
        job_id: str,
        duration_seconds: float,
        cpu_cores: float,
        memory_mb: int,
        network_bytes: int = 0,
        labels: dict[str, str] = None,
    ) -> CostEntry:
        """Record cost for a job execution."""
        memory_gb = memory_mb / 1024
        network_gb = network_bytes / (1024 ** 3)

        cost_usd = self.rate.calculate(
            duration_seconds,
            cpu_cores,
            memory_gb,
            network_gb,
        )

        entry = CostEntry(
            job_id=job_id,
            timestamp=time.time(),
            duration_seconds=duration_seconds,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            cost_usd=cost_usd,
            labels=labels or {},
        )

        self.entries.append(entry)
        self._save()

        # Check budget
        if self.budget_usd:
            total = self.get_total_cost()
            if total > self.budget_usd:
                logger.warning(
                    f"Budget exceeded! Total: ${total:.2f}, Budget: ${self.budget_usd:.2f}"
                )
            elif total > self.budget_usd * 0.8:
                logger.warning(
                    f"Budget 80% used: ${total:.2f} / ${self.budget_usd:.2f}"
                )

        logger.info(f"Cost recorded: {job_id} = ${cost_usd:.4f}")
        return entry

    def get_total_cost(self) -> float:
        """Get total cost of all recorded entries."""
        return sum(e.cost_usd for e in self.entries)

    def get_cost_by_label(self, label_key: str) -> dict[str, float]:
        """Get costs grouped by label value."""
        costs: dict[str, float] = {}
        for entry in self.entries:
            label_value = entry.labels.get(label_key, "unknown")
            costs[label_value] = costs.get(label_value, 0) + entry.cost_usd
        return costs

    def get_cost_by_date(self, days: int = 30) -> dict[str, float]:
        """Get costs grouped by date for last N days."""
        cutoff = time.time() - (days * 24 * 3600)
        costs: dict[str, float] = {}

        for entry in self.entries:
            if entry.timestamp >= cutoff:
                date = datetime.fromtimestamp(entry.timestamp).strftime("%Y-%m-%d")
                costs[date] = costs.get(date, 0) + entry.cost_usd

        return dict(sorted(costs.items()))

    def get_summary(
        self,
        start_time: float = None,
        end_time: float = None,
    ) -> dict[str, Any]:
        """Get cost summary for time range."""
        entries = self.entries

        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        if not entries:
            return {"total_cost_usd": 0, "job_count": 0}

        return {
            "total_cost_usd": sum(e.cost_usd for e in entries),
            "job_count": len(entries),
            "avg_cost_per_job_usd": sum(e.cost_usd for e in entries) / len(entries),
            "total_cpu_hours": sum(e.duration_seconds * e.cpu_cores / 3600 for e in entries),
            "total_memory_gb_hours": sum(e.duration_seconds * e.memory_gb / 3600 for e in entries),
            "budget_usd": self.budget_usd,
            "budget_remaining_usd": self.budget_usd - self.get_total_cost() if self.budget_usd else None,
        }

    def export_csv(self, path: Path) -> None:
        """Export all entries to CSV."""
        if not self.entries:
            logger.warning("No entries to export")
            return

        fieldnames = ["job_id", "datetime", "duration_seconds", "cpu_cores",
                      "memory_gb", "cost_usd"]

        # Add label columns
        all_labels = set()
        for entry in self.entries:
            all_labels.update(entry.labels.keys())
        fieldnames.extend(sorted(all_labels))

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in self.entries:
                row = entry.to_dict()
                # Flatten for CSV
                for label in all_labels:
                    if label not in row:
                        row[label] = ""
                writer.writerow({k: v for k, v in row.items() if k in fieldnames})

        logger.info(f"Exported {len(self.entries)} entries to {path}")

    def generate_report(self) -> str:
        """Generate human-readable cost report."""
        summary = self.get_summary()
        by_date = self.get_cost_by_date(7)

        lines = [
            "=" * 50,
            "CARTS Sandbox Cost Report",
            "=" * 50,
            "",
            "Summary",
            "-" * 30,
            f"Total Cost: ${summary['total_cost_usd']:.2f}",
            f"Total Jobs: {summary['job_count']}",
            f"Avg Cost/Job: ${summary['avg_cost_per_job_usd']:.4f}",
            f"Total CPU-Hours: {summary['total_cpu_hours']:.2f}",
            f"Total Memory GB-Hours: {summary['total_memory_gb_hours']:.2f}",
        ]

        if summary.get("budget_usd"):
            lines.extend([
                "",
                f"Budget: ${summary['budget_usd']:.2f}",
                f"Remaining: ${summary['budget_remaining_usd']:.2f}",
                f"Usage: {100 * summary['total_cost_usd'] / summary['budget_usd']:.1f}%",
            ])

        if by_date:
            lines.extend([
                "",
                "Last 7 Days",
                "-" * 30,
            ])
            for date, cost in by_date.items():
                lines.append(f"  {date}: ${cost:.2f}")

        lines.append("")
        lines.append("=" * 50)

        return "\n".join(lines)


# Global tracker instance
_tracker: CostTracker | None = None


def init_cost_tracking(
    budget_usd: float = None,
    persist_path: Path = None,
    rate: CostRate = None,
) -> CostTracker:
    """Initialize global cost tracker."""
    global _tracker
    _tracker = CostTracker(
        rate=rate,
        budget_usd=budget_usd,
        persist_path=persist_path,
    )
    return _tracker


def get_cost_tracker() -> CostTracker | None:
    """Get global cost tracker."""
    return _tracker


def record_cost(
    job_id: str,
    duration_seconds: float,
    cpu_cores: float,
    memory_mb: int,
    labels: dict[str, str] = None,
) -> CostEntry | None:
    """Record cost on global tracker."""
    if _tracker:
        return _tracker.record(job_id, duration_seconds, cpu_cores, memory_mb, labels=labels)
    return None

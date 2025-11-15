"""
Sync Manager for Coz and Elle Core Integration

Watches Coz files for changes and auto-syncs with Elle Core:
- Kanban → Daily tasks and suggestions
- Financials → SOP ingredient costs
- Schedule → Seasonal recommendations
- Research Notes → Knowledge base
- Inventory → Material tracking

Created: 2025-11-15
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
from enum import Enum

from elle.coz.kanban_parser import KanbanParser
from elle.coz.financials_parser import FinancialsParser
from elle.coz.schedule_parser import ScheduleParser
from elle.coz.research_parser import ResearchParser
from elle.coz.inventory_parser import InventoryParser


class SyncStatus(Enum):
    """Sync operation status"""
    PENDING = "pending"
    SYNCING = "syncing"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class SyncResult:
    """Result of a sync operation"""
    status: SyncStatus = SyncStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    files_synced: List[str] = field(default_factory=list)
    items_synced: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "files_synced": self.files_synced,
            "items_synced": self.items_synced,
            "errors": self.errors,
            "message": self.message,
        }


class SyncManager:
    """Manages Coz to Elle Core synchronization"""

    def __init__(
        self,
        coz_dir: str = "coz",
        elle_data_dir: str = "elle/data",
        auto_sync: bool = False,
        sync_interval: int = 3600,  # 1 hour
    ):
        self.coz_dir = Path(coz_dir)
        self.elle_data_dir = Path(elle_data_dir)
        self.auto_sync = auto_sync
        self.sync_interval = sync_interval

        # Parsers
        self.kanban = KanbanParser(str(self.coz_dir / "kanban.csv"))
        self.financials = FinancialsParser(str(self.coz_dir / "financials.md"))
        self.schedule = ScheduleParser(str(self.coz_dir / "schedule.md"))
        self.research = ResearchParser(str(self.coz_dir / "research_notes.md"))
        self.inventory = InventoryParser(str(self.coz_dir / "inventory.md"))

        # Sync state
        self.last_sync: Optional[datetime] = None
        self.sync_history: List[SyncResult] = []
        self.file_mtimes: Dict[str, float] = {}

        # Callbacks
        self.on_sync: Optional[Callable] = None

        # Ensure data directory exists
        self.elle_data_dir.mkdir(parents=True, exist_ok=True)

    def parse_all(self) -> SyncResult:
        """Parse all Coz files"""
        result = SyncResult(status=SyncStatus.SYNCING)

        try:
            # Parse kanban
            self.kanban.parse()
            result.items_synced["kanban_tasks"] = len(self.kanban.tasks)
            result.files_synced.append("kanban.csv")

            # Parse financials
            self.financials.parse()
            result.items_synced["products"] = len(self.financials.products)
            result.files_synced.append("financials.md")

            # Parse schedule
            self.schedule.parse()
            result.items_synced["monthly_focuses"] = len(self.schedule.monthly_focus)
            result.files_synced.append("schedule.md")

            # Parse research
            self.research.parse()
            result.items_synced["research_notes"] = len(self.research.research_notes)
            result.files_synced.append("research_notes.md")

            # Parse inventory
            self.inventory.parse()
            result.items_synced["inventory_items"] = len(self.inventory.items)
            result.files_synced.append("inventory.md")

            result.status = SyncStatus.SUCCESS
            result.message = f"Successfully synced {len(result.files_synced)} files"

        except Exception as e:
            result.status = SyncStatus.FAILED
            result.errors.append(str(e))
            result.message = f"Sync failed: {e}"

        result.timestamp = datetime.now()
        self.last_sync = result.timestamp
        self.sync_history.append(result)

        return result

    def check_files_changed(self) -> List[str]:
        """Check which Coz files have changed"""
        changed_files = []

        files_to_check = [
            self.coz_dir / "kanban.csv",
            self.coz_dir / "financials.md",
            self.coz_dir / "schedule.md",
            self.coz_dir / "research_notes.md",
            self.coz_dir / "inventory.md",
        ]

        for file_path in files_to_check:
            if not file_path.exists():
                continue

            current_mtime = file_path.stat().st_mtime
            previous_mtime = self.file_mtimes.get(str(file_path))

            if previous_mtime is None or current_mtime > previous_mtime:
                changed_files.append(file_path.name)
                self.file_mtimes[str(file_path)] = current_mtime

        return changed_files

    async def auto_sync_loop(self):
        """Background auto-sync loop"""
        if not self.auto_sync:
            return

        while self.auto_sync:
            try:
                changed = self.check_files_changed()
                if changed:
                    self.parse_all()
                    if self.on_sync:
                        await self.on_sync(self.sync_history[-1])

            except Exception as e:
                print(f"Auto-sync error: {e}")

            await asyncio.sleep(self.sync_interval)

    def get_daily_tasks(self) -> Dict:
        """Get daily task suggestions"""
        daily_plan = self.kanban.suggest_daily_plan()

        # Add schedule context
        current_focus = self.schedule.get_current_focus()
        recommendations = self.schedule.get_recommendations_for_current_month()

        return {
            "date": daily_plan["date"],
            "focus_area": current_focus.focus if current_focus else "No focus",
            "due_today": [t.to_dict() for t in daily_plan["due_today"]],
            "in_progress": [t.to_dict() for t in daily_plan["in_progress"]],
            "overdue": [t.to_dict() for t in daily_plan["overdue"]],
            "upcoming": [t.to_dict() for t in daily_plan["upcoming_3_days"]],
            "recommended_order": [t.to_dict() for t in daily_plan["suggested_order"]],
            "seasonal_tips": recommendations.get("seasonal_tips", []),
            "recommended_tasks": recommendations.get("suggested_tasks", []),
        }

    def get_financial_summary(self) -> Dict:
        """Get financial summary"""
        return {
            "products": [p.to_dict() for p in self.financials.products],
            "reinvestment": self.financials.reinvestment.to_dict() if self.financials.reinvestment else {},
            "summary": self.financials.get_summary(),
        }

    def get_seasonal_context(self) -> Dict:
        """Get current seasonal context and recommendations"""
        return self.schedule.get_recommendations_for_current_month()

    def get_research_knowledge_base(self) -> List[Dict]:
        """Get research notes formatted for knowledge base"""
        return self.research.to_knowledge_base_entries()

    def get_inventory_status(self) -> Dict:
        """Get inventory status and reorder alerts"""
        return {
            "summary": self.inventory.get_inventory_summary(),
            "reorder_list": self.inventory.generate_reorder_list(),
            "alerts": self.inventory.get_reorder_alerts(),
        }

    def get_daily_brief(self) -> Dict:
        """Get comprehensive daily brief for Coz operations"""
        return {
            "timestamp": datetime.now().isoformat(),
            "daily_tasks": self.get_daily_tasks(),
            "inventory_alerts": self.get_inventory_status()["alerts"],
            "seasonal_focus": self.get_seasonal_context(),
            "financial_status": {
                "products": [p.to_dict() for p in self.financials.products],
            },
        }

    def export_sync_data(self, output_path: Optional[str] = None) -> str:
        """Export all synced data to JSON"""
        if output_path is None:
            output_path = str(self.elle_data_dir / "coz_sync_data.json")

        data = {
            "timestamp": datetime.now().isoformat(),
            "kanban": {
                "tasks": [t.to_dict() for t in self.kanban.tasks],
                "summary": self.kanban.get_summary(),
            },
            "financials": {
                "products": [p.to_dict() for p in self.financials.products],
                "reinvestment": self.financials.reinvestment.to_dict() if self.financials.reinvestment else {},
                "summary": self.financials.get_summary(),
            },
            "schedule": {
                "focuses": [f.to_dict() for f in self.schedule.monthly_focus.values()],
                "summary": self.schedule.get_summary(),
            },
            "research": self.research.get_summary(),
            "inventory": {
                "items": self.inventory.to_dict(),
                "summary": self.inventory.get_inventory_summary(),
                "reorder_alerts": self.inventory.get_reorder_alerts(),
            },
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return output_path

    def get_sync_statistics(self) -> Dict:
        """Get synchronization statistics"""
        successful_syncs = [s for s in self.sync_history if s.status == SyncStatus.SUCCESS]
        failed_syncs = [s for s in self.sync_history if s.status == SyncStatus.FAILED]

        return {
            "total_syncs": len(self.sync_history),
            "successful": len(successful_syncs),
            "failed": len(failed_syncs),
            "success_rate": len(successful_syncs) / len(self.sync_history) if self.sync_history else 0,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
            "sync_history": [s.to_dict() for s in self.sync_history[-10:]],  # Last 10 syncs
        }

    def get_integration_status(self) -> Dict:
        """Get overall integration status"""
        return {
            "managers_initialized": {
                "kanban": len(self.kanban.tasks) > 0,
                "financials": len(self.financials.products) > 0,
                "schedule": len(self.schedule.monthly_focus) > 0,
                "research": len(self.research.research_notes) > 0,
                "inventory": len(self.inventory.items) > 0,
            },
            "sync_status": self.sync_history[-1].to_dict() if self.sync_history else None,
            "statistics": self.get_sync_statistics(),
            "data_export": "Ready",
        }

"""
Coz Integration Module for Elle Core

Parsers for Coz planning files:
- kanban.csv - Task management
- financials.md - Revenue and pricing
- schedule.md - Annual focus calendar
- research_notes.md - Experiment results and formulas
- inventory.md - Materials and equipment tracking

Auto-syncs with Elle Core decision engine and knowledge base.

Created: 2025-11-15
"""

from elle.coz.kanban_parser import KanbanParser, KanbanTask
from elle.coz.financials_parser import FinancialsParser, Product, ReinvestmentPlan
from elle.coz.schedule_parser import ScheduleParser, MonthlyFocus
from elle.coz.research_parser import ResearchParser, ResearchNote, ExperimentResult
from elle.coz.inventory_parser import InventoryParser, InventoryItem, ItemCategory
from elle.coz.sync_manager import SyncManager, SyncResult

__version__ = "0.1.0-alpha"
__all__ = [
    "KanbanParser",
    "KanbanTask",
    "FinancialsParser",
    "Product",
    "ReinvestmentPlan",
    "ScheduleParser",
    "MonthlyFocus",
    "ResearchParser",
    "ResearchNote",
    "ExperimentResult",
    "InventoryParser",
    "InventoryItem",
    "ItemCategory",
    "SyncManager",
    "SyncResult",
]

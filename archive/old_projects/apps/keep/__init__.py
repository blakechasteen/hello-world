"""
Keep - Beekeeping Management Application

A mythRL application for beekeeping management with HoloLoom integration.
Tracks hives, colonies, inspections, and provides intelligent decision support.

Elegant Design Patterns:
- Protocol-based extensibility
- Functional transformations
- Fluent builders
- Composable analytics
- Narrative journaling
- Seamless HoloLoom integration
"""

# Core models
from apps.keep.models import (
    Hive,
    Colony,
    Inspection,
    HarvestRecord,
    Alert,
)

# Type enums
from apps.keep.types import (
    HealthStatus,
    QueenStatus,
    HiveType,
    InspectionType,
    AlertLevel,
)

# Core logic
from apps.keep.apiary import Apiary
from apps.keep.keeper import BeeKeeper

# Elegant builders (fluent API)
from apps.keep.builders import (
    HiveBuilder,
    ColonyBuilder,
    InspectionBuilder,
    AlertBuilder,
    hive,
    colony,
    inspection,
    alert,
)

# Functional transformations
from apps.keep.transforms import (
    filter_healthy,
    filter_concerning,
    filter_queenless,
    sort_by_health,
    get_top_healthy_colonies,
    get_concerning_colonies,
    pipe,
    compose,
)

# Analytics
from apps.keep.analytics import (
    ApiaryAnalytics,
    HealthTrend,
    ProductivityMetrics,
    RiskAssessment,
    quick_health_check,
    productivity_summary,
)

# Journal
from apps.keep.journal import (
    BeekeepingJournal,
    JournalEntry,
    NarrativeSynthesis,
    EntryType,
    Sentiment,
    create_journal,
)

# HoloLoom integration
from apps.keep.hololoom_adapter import (
    ApiaryMemoryAdapter,
    HoloLoomQueryAdapter,
    export_to_memory,
    create_hololoom_session,
)

__all__ = [
    # Core models
    "Hive",
    "Colony",
    "Inspection",
    "HarvestRecord",
    "Alert",
    # Types
    "HealthStatus",
    "QueenStatus",
    "HiveType",
    "InspectionType",
    "AlertLevel",
    # Core logic
    "Apiary",
    "BeeKeeper",
    # Builders
    "HiveBuilder",
    "ColonyBuilder",
    "InspectionBuilder",
    "AlertBuilder",
    "hive",
    "colony",
    "inspection",
    "alert",
    # Transforms
    "filter_healthy",
    "filter_concerning",
    "filter_queenless",
    "sort_by_health",
    "get_top_healthy_colonies",
    "get_concerning_colonies",
    "pipe",
    "compose",
    # Analytics
    "ApiaryAnalytics",
    "HealthTrend",
    "ProductivityMetrics",
    "RiskAssessment",
    "quick_health_check",
    "productivity_summary",
    # Journal
    "BeekeepingJournal",
    "JournalEntry",
    "NarrativeSynthesis",
    "EntryType",
    "Sentiment",
    "create_journal",
    # HoloLoom
    "ApiaryMemoryAdapter",
    "HoloLoomQueryAdapter",
    "export_to_memory",
    "create_hololoom_session",
]

__version__ = "0.2.0"
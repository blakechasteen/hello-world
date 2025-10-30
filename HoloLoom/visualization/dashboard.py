"""
Dashboard Data Structures
==========================
Core classes for representing dashboards before rendering.

Philosophy: Type-safe, validated, immutable where possible.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol
from datetime import datetime
from enum import Enum


# ============================================================================
# Type Safety Enums
# ============================================================================

class PanelType(str, Enum):
    """Panel visualization types."""
    METRIC = "metric"
    TIMELINE = "timeline"
    TRAJECTORY = "trajectory"
    NETWORK = "network"
    HEATMAP = "heatmap"
    DISTRIBUTION = "distribution"
    TEXT = "text"
    SCATTER = "scatter"        # Scatter plot (correlation, clustering)
    LINE = "line"              # Line chart (time-series trends)
    BAR = "bar"                # Bar chart (categorical comparison)
    INSIGHT = "insight"        # Intelligence card (auto-detected patterns)


class LayoutType(str, Enum):
    """Dashboard layout types."""
    METRIC = "metric"          # Single column, simple
    FLOW = "flow"              # Two columns, balanced
    RESEARCH = "research"      # Three columns, comprehensive
    ADAPTIVE = "adaptive"      # Responsive, auto-adjusting


class PanelSize(str, Enum):
    """Panel size specifications."""
    TINY = "tiny"              # 1/6 width - compact metrics (6 per row)
    COMPACT = "compact"        # 1/4 width - small metrics (4 per row)
    SMALL = "small"            # 1/3 width - standard metrics (3 per row)
    MEDIUM = "medium"          # 1/2 width (2 per row)
    LARGE = "large"            # 2/3 width (2 of 3 columns)
    TWO_THIRDS = "two-thirds"  # 2/3 width (2 of 3 columns) - alias for LARGE
    THREE_QUARTERS = "three-quarters"  # 3/4 width (4 per row)
    FULL_WIDTH = "full-width"  # Full width
    HERO = "hero"              # Full width with extra padding


class ComplexityLevel(str, Enum):
    """Query complexity levels."""
    LITE = "LITE"
    FAST = "FAST"
    FULL = "FULL"
    RESEARCH = "RESEARCH"


# ============================================================================
# Spacetime Protocol (Duck-Typed Interface)
# ============================================================================

class SpacetimeLike(Protocol):
    """Protocol for Spacetime-like objects (duck typing)."""
    query_text: str
    response: str
    tool_used: str
    confidence: float
    trace: Any
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]: ...


# ============================================================================
# Data Structures
# ============================================================================

@dataclass(frozen=True)
class PanelSpec:
    """
    Immutable specification for a dashboard panel (before generation).

    Used by DashboardConstructor to specify what panels to create.
    """
    type: PanelType
    data_source: str  # Which field from Spacetime (e.g., "trace.duration_ms")
    size: PanelSize
    priority: int  # Layout priority (higher = more prominent)
    title: Optional[str] = None
    subtitle: Optional[str] = None


@dataclass
class Panel:
    """
    Rendered panel with extracted data ready for visualization.

    Mutable because rendering may add metadata incrementally.
    """
    id: str
    type: PanelType
    title: str
    subtitle: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    size: PanelSize = PanelSize.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate panel has minimum required data."""
        return bool(self.id and self.title and self.data)


@dataclass(frozen=True)
class DashboardStrategy:
    """
    Immutable strategy for constructing a dashboard from Spacetime.

    Returned by StrategySelector, consumed by DashboardConstructor.
    """
    layout_type: LayoutType
    panels: tuple[PanelSpec, ...]  # Immutable tuple
    title: str
    complexity_level: ComplexityLevel


@dataclass
class Dashboard:
    """
    Complete dashboard ready for rendering.

    Contains all panels, layout info, and reference to source Spacetime.
    """
    title: str
    layout: LayoutType
    panels: List[Panel]
    spacetime: SpacetimeLike
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now())  # Fix: Use lambda

    def __post_init__(self):
        """Validate dashboard after initialization."""
        if not self.title:
            raise ValueError("Dashboard title cannot be empty")
        if not self.panels:
            raise ValueError("Dashboard must have at least one panel")

        # Validate all panels
        invalid_panels = [p for p in self.panels if not p.validate()]
        if invalid_panels:
            raise ValueError(f"Invalid panels: {[p.id for p in invalid_panels]}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'title': self.title,
            'layout': self.layout.value,
            'panels': [
                {
                    'id': p.id,
                    'type': p.type.value,
                    'title': p.title,
                    'subtitle': p.subtitle,
                    'size': p.size.value,
                    'data': p.data
                }
                for p in self.panels
            ],
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }


# ============================================================================
# Configuration Constants
# ============================================================================

# Tailwind CSS grid classes for each layout type
LAYOUT_CONFIGS: Dict[LayoutType, str] = {
    LayoutType.METRIC: "grid grid-cols-1 gap-6",
    LayoutType.FLOW: "grid grid-cols-1 md:grid-cols-2 gap-6",
    LayoutType.RESEARCH: "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6",
    LayoutType.ADAPTIVE: "grid grid-cols-1 md:grid-cols-2 gap-6"
}

# Tailwind CSS column span classes for each panel size
PANEL_SIZE_CLASSES: Dict[PanelSize, str] = {
    PanelSize.TINY: "col-span-1",           # 1 column (stacks on mobile, 6 per row desktop)
    PanelSize.COMPACT: "col-span-1",        # 1 column (stacks on mobile, 4 per row desktop)
    PanelSize.SMALL: "md:col-span-1",       # 1/3 width (3 per row)
    PanelSize.MEDIUM: "md:col-span-1 lg:col-span-1",  # 1/2 width (2 per row)
    PanelSize.LARGE: "md:col-span-2",       # 2/3 width (2 of 3 columns)
    PanelSize.TWO_THIRDS: "md:col-span-2",  # 2 of 3 columns (alias)
    PanelSize.FULL_WIDTH: "md:col-span-2 lg:col-span-3",
    PanelSize.HERO: "md:col-span-2 lg:col-span-3"  # Full width with extra visual weight
}

# Semantic color mapping for metrics
METRIC_COLORS: Dict[str, str] = {
    'confidence': 'green',
    'duration': 'blue',
    'tool': 'purple',
    'threads': 'indigo',
    'error': 'red',
    'warning': 'yellow'
}

# Stage execution color mapping
STAGE_COLORS: Dict[str, str] = {
    'features': '#6366f1',    # Indigo
    'retrieval': '#10b981',   # Green
    'decision': '#f59e0b',    # Yellow
    'execution': '#ef4444',   # Red
    'default': '#6b7280'      # Gray
}

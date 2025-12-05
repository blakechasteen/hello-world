"""Layout planning for spaces (shed, beds, etc.)."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class LayoutItem:
    """An item to be placed in a layout."""
    
    item_id: str
    name: str
    
    # Dimensions (arbitrary units, could be inches, cm, etc.)
    width: float
    depth: float
    height: Optional[float] = None
    
    # Constraints
    requires_wall: bool = False  # Must be against wall
    requires_access: bool = True  # Needs clearance in front
    stackable: bool = False
    
    # Priority
    frequency_of_use: str = "medium"  # "high", "medium", "low"
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlacedItem:
    """An item positioned in a layout."""
    
    item: LayoutItem
    x: float
    y: float
    z: float = 0.0
    
    # Orientation (degrees from north/forward)
    rotation: float = 0.0
    
    # Quality metrics (if from optimization)
    access_score: Optional[float] = None
    efficiency_score: Optional[float] = None


@dataclass(frozen=True)
class LayoutPlan:
    """
    A proposed arrangement of items in a space.
    
    Output from layout optimization tools.
    """
    
    # Identity
    plan_id: str
    space_id: str  # WorldArea this plan is for
    
    # Space dimensions
    space_width: float
    space_depth: float
    
    # Placement
    placed_items: List[PlacedItem]
    
    # Optional dimensions
    space_height: Optional[float] = None
    
    # Quality
    overall_score: Optional[float] = None  # Higher is better
    metrics: Dict[str, float] = field(default_factory=dict)  # "access", "density", "flow"
    
    # Explanation
    reasoning: Optional[str] = None  # Why this layout
    trade_offs: List[str] = field(default_factory=list)  # What was sacrificed
    
    # Alternatives
    is_primary: bool = True
    alternative_plan_ids: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def item_count(self) -> int:
        """Number of items in this layout."""
        return len(self.placed_items)
    
    @property
    def space_utilization(self) -> float:
        """Percentage of space used (0-1)."""
        if not self.placed_items:
            return 0.0
        
        total_item_area = sum(
            item.item.width * item.item.depth 
            for item in self.placed_items
        )
        space_area = self.space_width * self.space_depth
        
        return min(total_item_area / space_area, 1.0) if space_area > 0 else 0.0

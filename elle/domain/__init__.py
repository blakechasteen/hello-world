"""Elle domain models.

Core data structures representing Elle's understanding of:
- Scenes and objects in space
- User intent and state
- Actions Elle can take
- Tasks and projects
- World areas and layouts
"""

from .scene import SceneSnapshot, ObjectInScene, SpatialRelation
from .intent import (
    UserIntent, 
    UserState, 
    ScanType, 
    InteractionMode
)
from .action import (
    ElleAction, 
    ElleRequest,
    ElleMode, 
    VisualStyle, 
    Visual,
    Priority,
    Followup
)
from .task import (
    Task, 
    Project, 
    TaskEstimate,
    TaskStatus, 
    TaskType
)
from .world import (
    WorldArea, 
    SpatialContext,
    AreaType
)
from .layout import (
    LayoutPlan, 
    LayoutItem, 
    PlacedItem
)

__all__ = [
    # Scene
    'SceneSnapshot',
    'ObjectInScene',
    'SpatialRelation',
    
    # Intent
    'UserIntent',
    'UserState',
    'ScanType',
    'InteractionMode',
    
    # Action
    'ElleAction',
    'ElleRequest',
    'ElleMode',
    'VisualStyle',
    'Visual',
    'Priority',
    'Followup',
    
    # Task
    'Task',
    'Project',
    'TaskEstimate',
    'TaskStatus',
    'TaskType',
    
    # World
    'WorldArea',
    'SpatialContext',
    'AreaType',
    
    # Layout
    'LayoutPlan',
    'LayoutItem',
    'PlacedItem',
]

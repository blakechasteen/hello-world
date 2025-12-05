"""
SOUS - Kitchen Manager MVP

A small but extensible kitchen manager that handles:
- Multi-store shopping routing
- Day-by-day prep scheduling
- Appliance conflict detection
- Make-ahead task planning
- Decision trees and substitutions

Primary use case: Thanksgiving meal planning
"""

__version__ = "0.1.0"
__author__ = "HoloLoom + Metaprompting"

from sous.models.event import Event, Appliance
from sous.models.store import Store
from sous.models.recipe import Recipe, RecipeIngredient, Step, SubstitutionRule
from sous.models.schedule import DailyPlan, Task

__all__ = [
    "Event",
    "Appliance",
    "Store",
    "Recipe",
    "RecipeIngredient",
    "Step",
    "SubstitutionRule",
    "DailyPlan",
    "Task",
]

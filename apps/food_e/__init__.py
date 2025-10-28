"""
food-e: Food Journal & Nutrition Assistant
===========================================
An elegant food tracking app built on HoloLoom.

Core concepts:
- Dishes: Food items with nutritional profiles
- Plates: Meals eaten at one sitting (temporal events)
- Journal: Temporal memory of all eating (Spacetime fabric)
- Kitchen: Orchestrator (Phase 1 stub, full HoloLoom integration in Phase 2)
- Spectrum: Nutritional analysis (spectral features)

Showcases HoloLoom capabilities:
- Spectral analysis for nutritional harmonics
- Temporal queries and pattern detection
- Journal-first architecture (source of truth)
"""

from .models import (
    NutritionalProfile,
    Dish,
    Plate,
    MealType
)
from .nutrition import NutritionalSpectrum
from .journal import Journal
from .kitchen import Kitchen, KitchenConfig

__all__ = [
    'NutritionalProfile',
    'Dish',
    'Plate',
    'MealType',
    'NutritionalSpectrum',
    'Journal',
    'Kitchen',
    'KitchenConfig'
]

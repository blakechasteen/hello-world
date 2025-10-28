"""
food-e Core Models
==================
Base classes for food tracking entities.

Key entities:
- NutritionalProfile: Macro/micro nutrients for food
- Dish: Single food item with nutrition
- Plate: Collection of dishes eaten at one sitting
- MealType: Breakfast, lunch, dinner, snack
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import numpy as np

# Import HoloLoom types
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.Documentation.types import MemoryShard
    from HoloLoom.fabric.spacetime import Spacetime
except ImportError:
    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

    Spacetime = Any  # Fallback


class MealType(Enum):
    """Type of meal."""
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"
    OTHER = "other"


@dataclass
class NutritionalProfile:
    """
    Complete nutritional information for food.

    Tracks macros (protein, carbs, fat) and key micros.
    Can be aggregated across multiple foods/meals.
    """
    # Macronutrients
    calories: float = 0.0
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0
    fiber_g: float = 0.0

    # Micronutrients (optional)
    sodium_mg: float = 0.0
    sugar_g: float = 0.0
    vitamin_c_mg: Optional[float] = None
    iron_mg: Optional[float] = None
    calcium_mg: Optional[float] = None

    @property
    def protein_ratio(self) -> float:
        """Protein calories / total calories"""
        if self.calories == 0:
            return 0.0
        return (self.protein_g * 4) / self.calories

    @property
    def carb_ratio(self) -> float:
        """Carb calories / total calories"""
        if self.calories == 0:
            return 0.0
        return (self.carbs_g * 4) / self.calories

    @property
    def fat_ratio(self) -> float:
        """Fat calories / total calories"""
        if self.calories == 0:
            return 0.0
        return (self.fat_g * 9) / self.calories

    def as_vector(self, dims: int = 96) -> np.ndarray:
        """
        Convert to vector for Matryoshka embedding.

        96D = coarse nutritional representation
        192D = medium (would include more micros)
        384D = fine (complete nutritional fingerprint)
        """
        # Normalize to typical daily values
        vector = np.zeros(dims)

        # Core macros (first 8 dimensions)
        vector[0] = self.calories / 3000
        vector[1] = self.protein_g / 150
        vector[2] = self.carbs_g / 300
        vector[3] = self.fat_g / 100
        vector[4] = self.fiber_g / 50
        vector[5] = self.sodium_mg / 2300
        vector[6] = self.sugar_g / 50

        # Ratios (dimensions 8-11)
        vector[8] = self.protein_ratio
        vector[9] = self.carb_ratio
        vector[10] = self.fat_ratio

        # Micros if available (dimensions 12-20)
        if self.vitamin_c_mg is not None:
            vector[12] = self.vitamin_c_mg / 90
        if self.iron_mg is not None:
            vector[13] = self.iron_mg / 18
        if self.calcium_mg is not None:
            vector[14] = self.calcium_mg / 1000

        # Pad remaining dimensions
        # Could add: vitamins A, D, E, K, B-complex, minerals, etc.

        return vector

    def __add__(self, other: 'NutritionalProfile') -> 'NutritionalProfile':
        """Add two nutritional profiles (for meal aggregation)"""
        return NutritionalProfile(
            calories=self.calories + other.calories,
            protein_g=self.protein_g + other.protein_g,
            carbs_g=self.carbs_g + other.carbs_g,
            fat_g=self.fat_g + other.fat_g,
            fiber_g=self.fiber_g + other.fiber_g,
            sodium_mg=self.sodium_mg + other.sodium_mg,
            sugar_g=self.sugar_g + other.sugar_g,
            vitamin_c_mg=(self.vitamin_c_mg or 0) + (other.vitamin_c_mg or 0) if (self.vitamin_c_mg or other.vitamin_c_mg) else None,
            iron_mg=(self.iron_mg or 0) + (other.iron_mg or 0) if (self.iron_mg or other.iron_mg) else None,
            calcium_mg=(self.calcium_mg or 0) + (other.calcium_mg or 0) if (self.calcium_mg or other.calcium_mg) else None,
        )

    def __mul__(self, scalar: float) -> 'NutritionalProfile':
        """Multiply by scalar (for portion adjustments)"""
        return NutritionalProfile(
            calories=self.calories * scalar,
            protein_g=self.protein_g * scalar,
            carbs_g=self.carbs_g * scalar,
            fat_g=self.fat_g * scalar,
            fiber_g=self.fiber_g * scalar,
            sodium_mg=self.sodium_mg * scalar,
            sugar_g=self.sugar_g * scalar,
            vitamin_c_mg=self.vitamin_c_mg * scalar if self.vitamin_c_mg else None,
            iron_mg=self.iron_mg * scalar if self.iron_mg else None,
            calcium_mg=self.calcium_mg * scalar if self.calcium_mg else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'calories': self.calories,
            'protein_g': self.protein_g,
            'carbs_g': self.carbs_g,
            'fat_g': self.fat_g,
            'fiber_g': self.fiber_g,
            'sodium_mg': self.sodium_mg,
            'sugar_g': self.sugar_g,
            'vitamin_c_mg': self.vitamin_c_mg,
            'iron_mg': self.iron_mg,
            'calcium_mg': self.calcium_mg,
        }


@dataclass
class Dish:
    """
    A single food item - the atomic unit of nutrition.

    Examples: "chicken breast", "broccoli", "rice", "salmon"

    Each dish has:
    - Nutrition per serving
    - Semantic embedding (taste, cuisine, etc.)
    - Tags for categorization
    """
    dish_id: str
    name: str

    # Nutrition (per standard serving)
    nutrition: NutritionalProfile
    serving_size: str = "1 serving"  # "100g", "1 cup", "1 piece", etc.

    # Semantic properties
    tags: List[str] = field(default_factory=list)  # "high-protein", "vegetarian", "italian", etc.
    cuisine: Optional[str] = None  # "Italian", "Thai", "American", etc.
    meal_type_hint: Optional[MealType] = None  # Typical meal type for this dish

    # Taste embedding (384D from HoloLoom)
    taste_embedding: Optional[np.ndarray] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, nutrition: NutritionalProfile, **kwargs):
        """Factory method with auto-generated ID"""
        dish_id = kwargs.pop('dish_id', f"dish_{uuid.uuid4().hex[:8]}")
        return cls(dish_id=dish_id, name=name, nutrition=nutrition, **kwargs)

    def to_shard(self) -> MemoryShard:
        """Convert to memory shard for HoloLoom storage"""
        text_parts = [
            f"Dish: {self.name}",
            f"Nutrition: {self.nutrition.calories:.0f} cal, {self.nutrition.protein_g:.0f}g protein",
            f"Serving: {self.serving_size}"
        ]

        if self.cuisine:
            text_parts.append(f"Cuisine: {self.cuisine}")

        text = "\n".join(text_parts)

        entities = [self.dish_id, self.name]
        if self.cuisine:
            entities.append(self.cuisine)

        motifs = ['dish', 'food'] + self.tags
        if self.meal_type_hint:
            motifs.append(f'meal_type_{self.meal_type_hint.value}')

        return MemoryShard(
            id=f"dish_{self.dish_id}",
            text=text,
            episode=f"dish_catalog",
            entities=entities,
            motifs=motifs,
            metadata={
                'type': 'dish',
                'dish_id': self.dish_id,
                'nutrition': self.nutrition.to_dict(),
                **self.metadata
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'dish_id': self.dish_id,
            'name': self.name,
            'nutrition': self.nutrition.to_dict(),
            'serving_size': self.serving_size,
            'tags': self.tags,
            'cuisine': self.cuisine,
            'meal_type_hint': self.meal_type_hint.value if self.meal_type_hint else None,
            'metadata': self.metadata
        }


@dataclass
class Plate:
    """
    A meal - collection of dishes eaten at one sitting.

    This is the temporal event that gets recorded in the Journal.
    Each plate is a point in Spacetime with full provenance.
    """
    plate_id: str
    timestamp: datetime
    meal_type: MealType

    # Content
    dishes: List[Dish]
    portion_sizes: Dict[str, float] = field(default_factory=dict)  # dish_id -> multiplier

    # Computed nutrition (aggregated from dishes)
    nutrition: Optional[NutritionalProfile] = None

    # Context (temporal flavor)
    location: Optional[str] = None
    social_context: Optional[str] = None  # "alone", "family", "restaurant"
    mood_before: Optional[str] = None
    mood_after: Optional[str] = None

    # Feedback (for learning)
    satisfaction: Optional[float] = None  # 0-1 rating
    energy_after_2hr: Optional[float] = None  # 0-1 rating
    notes: Optional[str] = None

    # Provenance (HoloLoom integration)
    spacetime: Optional[Any] = None  # Full weaving trace if AI-suggested

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, dishes: List[Dish], meal_type: MealType, **kwargs):
        """Factory method with auto-generated ID"""
        plate_id = kwargs.pop('plate_id', f"plate_{uuid.uuid4().hex[:8]}")
        timestamp = kwargs.pop('timestamp', datetime.now())
        return cls(
            plate_id=plate_id,
            timestamp=timestamp,
            meal_type=meal_type,
            dishes=dishes,
            **kwargs
        )

    def compute_nutrition(self):
        """Aggregate nutrition from all dishes with portion adjustments"""
        total = NutritionalProfile()

        for dish in self.dishes:
            multiplier = self.portion_sizes.get(dish.dish_id, 1.0)
            total = total + (dish.nutrition * multiplier)

        self.nutrition = total

    def to_shard(self) -> MemoryShard:
        """Convert to memory shard for Journal storage"""
        # Ensure nutrition is computed
        if self.nutrition is None:
            self.compute_nutrition()

        text_parts = [
            f"{self.meal_type.value.title()} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Foods: {', '.join([d.name for d in self.dishes])}",
            f"Nutrition: {self.nutrition.calories:.0f} cal, {self.nutrition.protein_g:.0f}g protein"
        ]

        if self.location:
            text_parts.append(f"Location: {self.location}")

        if self.notes:
            text_parts.append(f"Notes: {self.notes}")

        text = "\n".join(text_parts)

        entities = [self.plate_id] + [d.name for d in self.dishes]
        if self.location:
            entities.append(self.location)

        motifs = [
            'plate',
            'meal',
            self.meal_type.value,
            f'meal_{self.meal_type.value}'
        ] + self.tags

        return MemoryShard(
            id=f"plate_{self.plate_id}",
            text=text,
            episode=f"meal_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
            entities=entities,
            motifs=motifs,
            metadata={
                'type': 'plate',
                'plate_id': self.plate_id,
                'timestamp': self.timestamp.isoformat(),
                'meal_type': self.meal_type.value,
                'nutrition': self.nutrition.to_dict() if self.nutrition else None,
                'satisfaction': self.satisfaction,
                **self.metadata
            }
        )

    def to_spacetime_point(self) -> np.ndarray:
        """
        Convert to 4D spacetime coordinates.

        Dimensions:
        - 3D: Nutritional space (protein/carb/fat ratios)
        - 1D: Temporal (days since epoch)
        """
        if self.nutrition is None:
            self.compute_nutrition()

        return np.array([
            self.nutrition.protein_ratio,
            self.nutrition.carb_ratio,
            self.nutrition.fat_ratio,
            (self.timestamp - datetime(2025, 1, 1)).days
        ])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'plate_id': self.plate_id,
            'timestamp': self.timestamp.isoformat(),
            'meal_type': self.meal_type.value,
            'dishes': [d.to_dict() for d in self.dishes],
            'portion_sizes': self.portion_sizes,
            'nutrition': self.nutrition.to_dict() if self.nutrition else None,
            'location': self.location,
            'social_context': self.social_context,
            'mood_before': self.mood_before,
            'mood_after': self.mood_after,
            'satisfaction': self.satisfaction,
            'energy_after_2hr': self.energy_after_2hr,
            'notes': self.notes,
            'tags': self.tags,
            'metadata': self.metadata
        }

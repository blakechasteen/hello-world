"""
food-e Journal
==============
Temporal memory of all meals - the source of truth.

The Journal is built on HoloLoom's Spacetime fabric concept.
Every plate is a temporal event with full provenance.
"""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np

from .models import Plate, NutritionalProfile


class Journal:
    """
    Temporal memory of all meals.

    This is the core of food-e's architecture:
    - Everything flows from what you've eaten
    - All analysis derives from the journal
    - Suggestions are based on journal patterns
    """

    def __init__(self, storage_path: str = "./data/journal.json"):
        self.storage_path = Path(storage_path)
        self.plates: List[Plate] = []
        self._load_from_disk()

    # === CORE CRUD ===

    def record(self, plate: Plate) -> Plate:
        """
        Record a plate to the journal.

        This is the primary write operation - everything starts here.
        """
        # Ensure nutrition is computed
        if plate.nutrition is None:
            plate.compute_nutrition()

        # Add to memory
        self.plates.append(plate)

        # Keep sorted by timestamp
        self.plates.sort(key=lambda p: p.timestamp)

        # Persist
        self._persist_to_disk()

        return plate

    def get_plate(self, plate_id: str) -> Optional[Plate]:
        """Retrieve a specific plate"""
        return next((p for p in self.plates if p.plate_id == plate_id), None)

    def get_plates_in_range(
        self,
        start: datetime,
        end: datetime
    ) -> List[Plate]:
        """Get all plates in a time window"""
        return [
            p for p in self.plates
            if start <= p.timestamp <= end
        ]

    # === TEMPORAL QUERIES ===

    def today(self) -> List[Plate]:
        """All plates eaten today"""
        now = datetime.now()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        return self.get_plates_in_range(start, end)

    def this_week(self) -> List[Plate]:
        """All plates this week"""
        now = datetime.now()
        start = now - timedelta(days=now.weekday())
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.get_plates_in_range(start, now)

    def last_n_days(self, n: int) -> List[Plate]:
        """All plates in last N days"""
        now = datetime.now()
        start = now - timedelta(days=n)
        return self.get_plates_in_range(start, now)

    # === NUTRITIONAL ANALYSIS ===

    def daily_nutrition(self, date: datetime) -> NutritionalProfile:
        """Total nutrition for a specific day"""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        plates = self.get_plates_in_range(start, end)

        return self._aggregate_nutrition(plates)

    def weekly_nutrition(self, week_start: datetime) -> NutritionalProfile:
        """Total nutrition for a week"""
        week_end = week_start + timedelta(days=7)
        plates = self.get_plates_in_range(week_start, week_end)
        return self._aggregate_nutrition(plates)

    def nutritional_trajectory(
        self,
        days: int = 30,
        metric: str = "protein_g"
    ) -> np.ndarray:
        """
        Time series of a nutritional metric.

        Returns array of shape (days,) with daily values.

        This enables semantic calculus operations:
        - Integration: total protein over period
        - Derivative: rate of change in calories
        - Variance: consistency of diet
        """
        now = datetime.now()
        start = now - timedelta(days=days)

        trajectory = []
        for i in range(days):
            day = start + timedelta(days=i)
            daily = self.daily_nutrition(day)
            trajectory.append(getattr(daily, metric))

        return np.array(trajectory)

    def nutritional_flow(
        self,
        start: datetime,
        end: datetime
    ) -> Dict[str, float]:
        """
        Semantic calculus integration over time.

        Returns cumulative nutrition over period.
        """
        plates = self.get_plates_in_range(start, end)
        total = self._aggregate_nutrition(plates)

        days = (end - start).days + 1

        return {
            "total_calories": total.calories,
            "total_protein_g": total.protein_g,
            "total_carbs_g": total.carbs_g,
            "total_fat_g": total.fat_g,
            "avg_calories_per_day": total.calories / days,
            "avg_protein_per_day": total.protein_g / days,
        }

    # === PATTERN DETECTION ===

    def eating_patterns(self) -> Dict:
        """
        Discover temporal patterns in eating.

        Uses HoloLoom's Chrono Trigger concepts.
        """
        plates = self.last_n_days(30)

        # Meal timing patterns
        meal_times = {}
        for plate in plates:
            hour = plate.timestamp.hour
            meal_type = plate.meal_type.value
            if meal_type not in meal_times:
                meal_times[meal_type] = []
            meal_times[meal_type].append(hour)

        # Average times
        avg_times = {
            meal_type: np.mean(times)
            for meal_type, times in meal_times.items()
        }

        # Frequency
        frequency = {
            meal_type: len(times) / 30  # meals per day
            for meal_type, times in meal_times.items()
        }

        # Macro balance consistency
        if plates:
            protein_ratios = [
                p.nutrition.protein_ratio for p in plates
                if p.nutrition
            ]
            macro_variance = np.var(protein_ratios) if protein_ratios else 0
        else:
            macro_variance = 0

        return {
            "average_meal_times": avg_times,
            "meal_frequency": frequency,
            "diet_consistency": 1.0 - macro_variance,
            "total_meals_logged": len(plates)
        }

    # === HELPER METHODS ===

    def _aggregate_nutrition(self, plates: List[Plate]) -> NutritionalProfile:
        """Sum nutrition across multiple plates"""
        total = NutritionalProfile()

        for plate in plates:
            if plate.nutrition:
                total = total + plate.nutrition

        return total

    def _load_from_disk(self):
        """Load journal from persistent storage"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # TODO: Deserialize plates from JSON
            # For Phase 1, skip persistence
            self.plates = []

        except Exception as e:
            print(f"Warning: Could not load journal: {e}")
            self.plates = []

    def _persist_to_disk(self):
        """Save journal to disk"""
        # TODO: Implement JSON serialization
        # For Phase 1, skip persistence
        pass
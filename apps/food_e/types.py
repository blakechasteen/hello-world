"""
food-e Result Types & Supporting Classes
=========================================
Clean dataclasses for API responses and temporal context.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from .models import Plate, NutritionalProfile
from .nutrition import NutritionalSpectrum


@dataclass
class ServingResult:
    """
    Clean result type from Kitchen operations.

    Replaces Dict returns with structured data.
    """
    success: bool
    message: str

    # Core data
    plate: Optional[Plate] = None
    nutrition: Optional[NutritionalProfile] = None
    spectrum: Optional[NutritionalSpectrum] = None
    resonance: Optional[float] = None

    # Context
    remaining_today: Optional[Dict[str, float]] = None
    gaps: Optional[Dict[str, float]] = None

    # Analysis (for period queries)
    total_meals: Optional[int] = None
    daily_average: Optional[NutritionalProfile] = None
    trajectories: Optional[Dict[str, np.ndarray]] = None


class TemporalFlavor:
    """
    Time-aware food context.

    Maps HoloLoom's Chrono Trigger to food domain.
    Understands circadian rhythms and temporal eating patterns.
    """

    @staticmethod
    def current_phase() -> str:
        """
        Get current circadian phase.

        Returns:
            "morning" (5-12): Lighter, simpler foods
            "midday" (12-17): Balanced, substantial
            "evening" (17-22): Richer, complex
            "night" (22-5): Comfort, light snacks
        """
        hour = datetime.now().hour

        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "midday"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    @staticmethod
    def temporal_context(timestamp: datetime) -> Dict:
        """
        Extract temporal features for a given time.

        Returns context dict with:
        - circadian_phase: morning/midday/evening/night
        - hour: 0-23
        - weekday: 0-6
        - is_weekend: bool
        - temporal_embedding: 4D vector [hour, day, sin_hour, cos_hour]
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()

        # Determine phase
        if 5 <= hour < 12:
            phase = "morning"
        elif 12 <= hour < 17:
            phase = "midday"
        elif 17 <= hour < 22:
            phase = "evening"
        else:
            phase = "night"

        # Temporal embedding (for ML)
        temporal_embedding = np.array([
            hour / 24,                      # Normalized hour
            weekday / 7,                    # Normalized day
            np.sin(2 * np.pi * hour / 24), # Circadian sine
            np.cos(2 * np.pi * hour / 24), # Circadian cosine
        ])

        return {
            "circadian_phase": phase,
            "hour": hour,
            "weekday": weekday,
            "is_weekend": weekday >= 5,
            "temporal_embedding": temporal_embedding
        }

    @staticmethod
    def is_appropriate_time(meal_type: str, hour: Optional[int] = None) -> bool:
        """
        Check if a meal type is appropriate for the time.

        Args:
            meal_type: "breakfast", "lunch", "dinner", "snack"
            hour: Hour (0-23), defaults to current

        Returns:
            True if meal is appropriate for this time
        """
        if hour is None:
            hour = datetime.now().hour

        appropriate_hours = {
            "breakfast": range(5, 12),
            "lunch": range(11, 16),
            "dinner": range(17, 23),
            "snack": range(0, 24)  # Snacks anytime
        }

        return hour in appropriate_hours.get(meal_type, range(0, 24))


class Palate:
    """
    Taste learning system (Phase 1 stub).

    Phase 1: Simple preference storage
    Phase 2: Full Thompson Sampling with Beta distributions
    Phase 3: Multi-timescale reflection (immediate, short, medium, long-term)

    This stub exists to complete the metaphor mapping:
    - HoloLoom Reflection Buffer → food-e Palate
    """

    def __init__(self):
        # Phase 1: Simple preference dict
        # dish_id -> average rating
        self.preferences: Dict[str, float] = {}
        self.trial_counts: Dict[str, int] = {}

    def record_satisfaction(self, dish_id: str, rating: float):
        """
        Record immediate taste satisfaction.

        Phase 1: Simple average
        Phase 2: Thompson Sampling Beta update

        Args:
            dish_id: Unique dish identifier
            rating: 0-1 satisfaction score
        """
        if dish_id not in self.preferences:
            self.preferences[dish_id] = rating
            self.trial_counts[dish_id] = 1
        else:
            # Running average
            count = self.trial_counts[dish_id]
            current_avg = self.preferences[dish_id]
            new_avg = (current_avg * count + rating) / (count + 1)

            self.preferences[dish_id] = new_avg
            self.trial_counts[dish_id] = count + 1

    def get_preference(self, dish_id: str) -> Optional[float]:
        """
        Get learned preference for a dish.

        Returns:
            0-1 preference score, or None if never tried
        """
        return self.preferences.get(dish_id)

    def get_top_preferences(self, n: int = 5) -> List[tuple]:
        """
        Get top N preferred dishes.

        Returns:
            List of (dish_id, preference_score) tuples
        """
        items = sorted(
            self.preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return items[:n]

    def __repr__(self) -> str:
        return f"Palate(dishes_tasted={len(self.preferences)}, avg_satisfaction={np.mean(list(self.preferences.values())):.2f})"

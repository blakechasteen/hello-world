"""
food-e Kitchen (Phase 1 Stub)
==============================
Main orchestrator - simplified for Phase 1.

Full HoloLoom integration coming in Phase 2.
For now, this provides basic functionality to test the foundation.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

from .journal import Journal
from .nutrition import NutritionalSpectrum
from .models import NutritionalProfile, Plate


@dataclass
class KitchenConfig:
    """Configuration for food-e Kitchen"""
    journal_path: str = "./data/journal.json"

    # Nutritional targets (daily)
    target_calories: float = 2000
    target_protein_g: float = 150
    target_carbs_g: float = 200
    target_fat_g: float = 65
    target_fiber_g: float = 30


class Kitchen:
    """
    food-e orchestrator (Phase 1 stub).

    Phase 1: Basic journal operations and spectral analysis
    Phase 2: Full HoloLoom integration with WeavingShuttle
    Phase 3: Thompson Sampling, multi-timescale reflection
    """

    def __init__(self, config: KitchenConfig):
        self.config = config

        # Core components
        self.journal = Journal(config.journal_path)

        # Daily targets as spectrum (for resonance calculation)
        self.target_spectrum = NutritionalSpectrum(
            NutritionalProfile(
                calories=config.target_calories,
                protein_g=config.target_protein_g,
                carbs_g=config.target_carbs_g,
                fat_g=config.target_fat_g,
                fiber_g=config.target_fiber_g,
                sodium_mg=2300,
                sugar_g=50
            )
        )

    async def __aenter__(self):
        """Async context manager entry"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        pass

    # === Phase 1 Methods ===

    async def log_plate(self, plate: Plate) -> Dict:
        """
        Log a plate to the journal.

        Returns analysis with spectral resonance.
        """
        # Record in journal
        await self.journal.record(plate)

        # Analyze
        spectrum = NutritionalSpectrum(plate.nutrition)
        resonance = spectrum.harmonic_resonance(self.target_spectrum)

        # Get today's totals
        today_total = self.journal.daily_nutrition(datetime.now())
        remaining = {
            "calories": self.config.target_calories - today_total.calories,
            "protein_g": self.config.target_protein_g - today_total.protein_g,
            "carbs_g": self.config.target_carbs_g - today_total.carbs_g,
            "fat_g": self.config.target_fat_g - today_total.fat_g,
        }

        return {
            "plate_id": plate.plate_id,
            "nutrition": plate.nutrition,
            "spectrum": spectrum,
            "resonance": resonance,
            "remaining_today": remaining,
            "message": self._format_logged_message(plate, remaining, resonance)
        }

    async def analyze_period(self, days: int = 7) -> Dict:
        """
        Analyze nutrition over a period.

        Returns trajectories and spectral analysis.
        """
        plates = self.journal.last_n_days(days)

        if not plates:
            return {
                "message": "No meals logged yet",
                "total_meals": 0
            }

        # Aggregate nutrition
        total = self.journal._aggregate_nutrition(plates)
        daily_avg = total * (1.0 / days)

        # Spectral analysis
        avg_spectrum = NutritionalSpectrum(daily_avg)
        resonance = avg_spectrum.harmonic_resonance(self.target_spectrum)
        gaps = avg_spectrum.missing_frequencies(self.target_spectrum)

        # Trajectories
        protein_trajectory = self.journal.nutritional_trajectory(days, "protein_g")
        calorie_trajectory = self.journal.nutritional_trajectory(days, "calories")

        return {
            "period_days": days,
            "total_meals": len(plates),
            "daily_average": daily_avg,
            "spectrum": avg_spectrum,
            "resonance": resonance,
            "gaps": gaps,
            "protein_trajectory": protein_trajectory,
            "calorie_trajectory": calorie_trajectory,
            "message": self._format_analysis_message(days, daily_avg, resonance, len(plates))
        }

    def _format_logged_message(
        self,
        plate: Plate,
        remaining: Dict,
        resonance: float
    ) -> str:
        """Format user-friendly message after logging meal"""
        msg = f"[OK] Logged {plate.meal_type.value}\n"
        msg += f"  {plate.nutrition.calories:.0f} cal, "
        msg += f"{plate.nutrition.protein_g:.0f}g protein\n"
        msg += f"  Harmonic resonance: {resonance:.2f}"

        if resonance > 0.8:
            msg += " (excellent!)"
        elif resonance > 0.6:
            msg += " (good)"
        else:
            msg += " (needs balance)"

        msg += "\n"

        if remaining["protein_g"] > 30:
            msg += f"  -> Still need {remaining['protein_g']:.0f}g protein today"
        elif remaining["calories"] < 500:
            msg += f"  -> Close to daily calorie target"

        return msg

    def _format_analysis_message(
        self,
        days: int,
        daily_avg: NutritionalProfile,
        resonance: float,
        meal_count: int
    ) -> str:
        """Format nutrition analysis message"""
        msg = f"[ANALYSIS] {days}-Day Nutrition Report\n"
        msg += f"  {meal_count} meals logged\n"
        msg += f"  Daily average: {daily_avg.calories:.0f} cal, "
        msg += f"{daily_avg.protein_g:.0f}g protein\n"
        msg += f"  Harmonic resonance: {resonance:.2f} "

        if resonance > 0.8:
            msg += "(excellent alignment!)"
        elif resonance > 0.6:
            msg += " (good balance)"
        else:
            msg += "(needs adjustment)"

        return msg

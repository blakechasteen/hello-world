#!/usr/bin/env python3
"""
food-e Demo: Phase 1 Foundation
================================
Demonstrates core food-e functionality:
- Logging meals with nutritional profiles
- Spectral harmonic analysis
- Temporal queries and pattern detection
- Journal-first architecture

Run: python apps/demo_food_e.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from food_e import (
    Kitchen,
    KitchenConfig,
    Dish,
    Plate,
    MealType,
    NutritionalProfile,
    NutritionalSpectrum
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


async def demo_basic_logging():
    """Demo 1: Basic meal logging with spectral analysis"""
    print_section("Demo 1: Basic Meal Logging")

    config = KitchenConfig(
        target_calories=2000,
        target_protein_g=150,
        target_carbs_g=200,
        target_fat_g=65
    )

    async with Kitchen(config) as kitchen:
        # Create some dishes
        eggs = Dish.create(
            name="Scrambled Eggs (3)",
            nutrition=NutritionalProfile(
                calories=210,
                protein_g=18,
                carbs_g=2,
                fat_g=14,
                fiber_g=0,
                sodium_mg=380
            ),
            tags=["high-protein", "breakfast", "quick"],
            meal_type_hint=MealType.BREAKFAST
        )

        toast = Dish.create(
            name="Whole Wheat Toast (2 slices)",
            nutrition=NutritionalProfile(
                calories=160,
                protein_g=8,
                carbs_g=28,
                fat_g=2,
                fiber_g=4,
                sodium_mg=240
            ),
            tags=["breakfast", "grains"],
            meal_type_hint=MealType.BREAKFAST
        )

        # Create a plate (breakfast)
        breakfast = Plate.create(
            dishes=[eggs, toast],
            meal_type=MealType.BREAKFAST,
            portion_sizes={
                eggs.dish_id: 1.0,
                toast.dish_id: 1.0
            },
            notes="Quick breakfast before work"
        )

        # Log it
        result = await kitchen.log_plate(breakfast)

        print(result.message)
        print("\nSpectral Analysis:")
        print(result.spectrum.visualize())

        print(f"\nRemaining today:")
        for nutrient, amount in result.remaining_today.items():
            print(f"  {nutrient}: {amount:.0f}")


async def demo_spectral_resonance():
    """Demo 2: Understanding spectral harmonic resonance"""
    print_section("Demo 2: Spectral Harmonic Resonance")

    # Create target (ideal balanced meal)
    target = NutritionalProfile(
        calories=500,
        protein_g=40,  # 32% protein
        carbs_g=50,     # 40% carbs
        fat_g=16        # 28% fat
    )
    target_spectrum = NutritionalSpectrum(target)

    print("Target (Balanced Meal):")
    print(target_spectrum.visualize())

    # Test different meals
    print("\n" + "-" * 60)
    print("Meal 1: High Protein (Chicken + Veggies)")
    high_protein = NutritionalProfile(
        calories=400,
        protein_g=50,  # Very high protein
        carbs_g=20,
        fat_g=10
    )
    hp_spectrum = NutritionalSpectrum(high_protein)
    resonance1 = hp_spectrum.harmonic_resonance(target_spectrum)
    print(f"Resonance with target: {resonance1:.2f}")
    print(hp_spectrum.visualize())

    print("\n" + "-" * 60)
    print("Meal 2: Balanced (Salmon, Quinoa, Veggies)")
    balanced = NutritionalProfile(
        calories=500,
        protein_g=38,
        carbs_g=48,
        fat_g=14,
        fiber_g=8
    )
    balanced_spectrum = NutritionalSpectrum(balanced)
    resonance2 = balanced_spectrum.harmonic_resonance(target_spectrum)
    print(f"Resonance with target: {resonance2:.2f}")
    print(balanced_spectrum.visualize())

    print("\n" + "=" * 60)
    print("Resonance Comparison:")
    print(f"  High Protein: {resonance1:.2f} (specialized)")
    print(f"  Balanced:     {resonance2:.2f} (optimal harmony!)")
    print("\nHigher resonance = better alignment with targets")


async def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("  food-e: Phase 1 Foundation Demo")
    print("  Elegant Food Journaling on HoloLoom")
    print("=" * 60)

    try:
        await demo_basic_logging()
        await demo_spectral_resonance()

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60)
        print("\nPhase 1 Complete:")
        print("  [OK] Core data models (Dish, Plate, NutritionalProfile)")
        print("  [OK] Journal with temporal queries")
        print("  [OK] Spectral harmonic analysis")
        print("  [OK] Resonance-based nutrition tracking")
        print("\nComing in Phase 2:")
        print("  -> Full HoloLoom integration (WeavingShuttle)")
        print("  -> Thompson Sampling for preference learning")
        print("  -> Multi-timescale reflection (Palate)")
        print("  -> Warp Space meal planning optimization")
        print("\n")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

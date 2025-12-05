#!/usr/bin/env python3
"""
Demo: Sous Moonshot Features

Demonstrates all Phase 2-5 enhancements:
- Phase 2: Time Windows (precise start/end times)
- Phase 3: Recipe Scaling (2x, 4x multipliers)
- Phase 4: Dietary Filters (vegan, gluten-free, allergens)
- Phase 5: Skill Level Adaptation

Run: python demos/demo_moonshot_features.py
"""

import sys
import os
from datetime import date, time, timedelta

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sous.models.recipe import (
    Recipe, RecipeIngredient, Step, DietaryTag, Allergen, SkillLevel
)
from sous.models.schedule import DailyPlan, Task, TimeSlot


def create_sample_recipe() -> Recipe:
    """Create a sample recipe with all Phase 2-5 features"""
    return Recipe(
        id="turkey_gravy",
        name="Classic Turkey Gravy",
        category="side",
        base_servings=8,
        skill_level=SkillLevel.INTERMEDIATE,
        dietary_tags=[DietaryTag.GLUTEN_FREE],  # Can be made GF
        allergens=[Allergen.MILK],  # Contains butter
        estimated_prep_minutes=15,
        estimated_cook_minutes=20,
        ingredients=[
            RecipeIngredient(
                ingredient_id="turkey_drippings",
                quantity="2",
                quantity_numeric=2.0,
                unit="cups"
            ),
            RecipeIngredient(
                ingredient_id="butter",
                quantity="4",
                quantity_numeric=4.0,
                unit="tbsp",
                allergens=[Allergen.MILK]
            ),
            RecipeIngredient(
                ingredient_id="flour",
                quantity="0.25",
                quantity_numeric=0.25,
                unit="cup",
                allergens=[Allergen.WHEAT]
            ),
            RecipeIngredient(
                ingredient_id="chicken_stock",
                quantity="4",
                quantity_numeric=4.0,
                unit="cups"
            ),
        ],
        steps=[
            Step(
                id="melt_butter",
                description="Melt butter in a saucepan over medium heat",
                appliance_type="burner",
                duration_minutes=3,
                skill_level=SkillLevel.BEGINNER,
                scaling_factor=0.0,  # Constant time regardless of scale
                tips_beginner="Watch carefully - butter burns easily!",
            ),
            Step(
                id="make_roux",
                description="Whisk in flour and cook until golden, about 2 minutes",
                appliance_type="burner",
                duration_minutes=5,
                skill_level=SkillLevel.INTERMEDIATE,
                scaling_factor=0.3,  # Slight scaling with volume
                tips_beginner="Whisk constantly to prevent lumps",
                tips_advanced="You can use a fat separator to get cleaner drippings",
            ),
            Step(
                id="add_liquid",
                description="Gradually add drippings and stock, whisking constantly",
                appliance_type="burner",
                duration_minutes=10,
                skill_level=SkillLevel.INTERMEDIATE,
                scaling_factor=0.5,  # Moderate scaling
                tips_beginner="Add liquid slowly in small batches",
            ),
            Step(
                id="simmer",
                description="Simmer until thickened, stirring occasionally",
                appliance_type="burner",
                duration_minutes=15,
                skill_level=SkillLevel.BEGINNER,
                scaling_factor=0.7,  # More scaling for larger batches
                tips_advanced="For silky gravy, strain through fine mesh",
            ),
        ],
    )


def demo_phase2_time_windows():
    """Phase 2: Demonstrate precise time window scheduling"""
    print("\n" + "=" * 70)
    print("PHASE 2: TIME WINDOWS")
    print("=" * 70)

    # Create tasks for a day
    tasks = [
        Task(
            id="task_1",
            recipe_id="turkey_gravy",
            step_id="melt_butter",
            duration_minutes=3,
            appliance_id="burner_1",
        ),
        Task(
            id="task_2",
            recipe_id="turkey_gravy",
            step_id="make_roux",
            duration_minutes=5,
            appliance_id="burner_1",
        ),
        Task(
            id="task_3",
            recipe_id="stuffing",
            step_id="saute_aromatics",
            duration_minutes=15,
            appliance_id="burner_2",
        ),
        Task(
            id="task_4",
            recipe_id="turkey_gravy",
            step_id="add_liquid",
            duration_minutes=10,
            appliance_id="burner_1",
        ),
    ]

    # Create daily plan
    plan = DailyPlan(date=date.today(), tasks=tasks)

    print(f"\nBefore time assignment:")
    print(f"  Tasks: {len(plan.tasks)}")
    print(f"  Total duration: {plan.total_duration_minutes()} min")

    # Assign time windows starting at 9:00 AM
    plan.assign_time_windows(start_time=time(9, 0), buffer_minutes=5)

    print(f"\nAfter time assignment (starting 9:00 AM):")
    print("-" * 50)

    timeline = plan.get_timeline()
    for entry in timeline:
        task = entry["task"]
        print(f"  {entry['start_time'].strftime('%H:%M')} - "
              f"{entry['end_time'].strftime('%H:%M')} | "
              f"{task.recipe_id}.{task.step_id} "
              f"({entry['duration']} min, {entry['appliance'] or 'no appliance'})")

    # Check for conflicts
    conflicts = plan.get_time_conflicts()
    if conflicts:
        print(f"\n[!] Found {len(conflicts)} time conflicts!")
    else:
        print("\n[OK] No time conflicts detected")


def demo_phase3_scaling():
    """Phase 3: Demonstrate recipe scaling"""
    print("\n" + "=" * 70)
    print("PHASE 3: RECIPE SCALING")
    print("=" * 70)

    recipe = create_sample_recipe()

    print(f"\nOriginal Recipe: {recipe.name}")
    print(f"  Servings: {recipe.base_servings}")
    print(f"  Prep time: {recipe.estimated_prep_minutes} min")
    print(f"  Cook time: {recipe.estimated_cook_minutes} min")
    print(f"\nIngredients:")
    for ing in recipe.ingredients:
        print(f"    {ing.quantity} {ing.unit} {ing.ingredient_id}")

    # Scale 2x
    print("\n" + "-" * 50)
    scaled_2x = recipe.scale(2.0)
    print(f"\nScaled 2x: {scaled_2x.name}")
    print(f"  Servings: {scaled_2x.base_servings}")
    print(f"  Prep time: {scaled_2x.estimated_prep_minutes} min (was {recipe.estimated_prep_minutes})")
    print(f"  Cook time: {scaled_2x.estimated_cook_minutes} min (was {recipe.estimated_cook_minutes})")
    print(f"\nScaled Ingredients (2x):")
    for ing in scaled_2x.ingredients:
        print(f"    {ing.quantity} {ing.unit} {ing.ingredient_id}")

    # Scale 4x
    print("\n" + "-" * 50)
    scaled_4x = recipe.scale(4.0)
    print(f"\nScaled 4x: {scaled_4x.name}")
    print(f"  Servings: {scaled_4x.base_servings}")
    print(f"  Prep time: {scaled_4x.estimated_prep_minutes} min")
    print(f"  Cook time: {scaled_4x.estimated_cook_minutes} min")

    # Scale to specific servings
    print("\n" + "-" * 50)
    scaled_12 = recipe.scale_to_servings(12)
    print(f"\nScaled to 12 servings: {scaled_12.name}")
    print(f"  Scale factor: {scaled_12.scale_factor:.2f}x")
    print(f"\nStep duration comparison:")
    for orig, scaled in zip(recipe.steps, scaled_12.steps):
        print(f"    {orig.id}: {orig.duration_minutes} min -> {scaled.duration_minutes} min")


def demo_phase4_dietary():
    """Phase 4: Demonstrate dietary filtering"""
    print("\n" + "=" * 70)
    print("PHASE 4: DIETARY FILTERS")
    print("=" * 70)

    # Create recipes with different dietary profiles
    recipes = [
        Recipe(
            id="vegan_stuffing",
            name="Vegan Mushroom Stuffing",
            category="side",
            dietary_tags=[DietaryTag.VEGAN, DietaryTag.DAIRY_FREE],
            allergens=[Allergen.WHEAT, Allergen.SOY],
            skill_level=SkillLevel.INTERMEDIATE,
        ),
        Recipe(
            id="classic_gravy",
            name="Classic Turkey Gravy",
            category="side",
            dietary_tags=[],
            allergens=[Allergen.MILK, Allergen.WHEAT],
            skill_level=SkillLevel.INTERMEDIATE,
        ),
        Recipe(
            id="gf_pie",
            name="Gluten-Free Pumpkin Pie",
            category="dessert",
            dietary_tags=[DietaryTag.GLUTEN_FREE, DietaryTag.VEGETARIAN],
            allergens=[Allergen.EGGS, Allergen.MILK],
            skill_level=SkillLevel.ADVANCED,
        ),
        Recipe(
            id="nut_free_cookies",
            name="Nut-Free Chocolate Cookies",
            category="dessert",
            dietary_tags=[DietaryTag.NUT_FREE, DietaryTag.VEGETARIAN],
            allergens=[Allergen.EGGS, Allergen.MILK, Allergen.WHEAT],
            skill_level=SkillLevel.BEGINNER,
        ),
    ]

    print(f"\nAvailable Recipes:")
    print("-" * 50)
    for r in recipes:
        tags = ", ".join(t.value for t in r.dietary_tags) if r.dietary_tags else "none"
        allergens = ", ".join(a.value for a in r.allergens) if r.allergens else "none"
        print(f"  {r.name}")
        print(f"    Dietary tags: {tags}")
        print(f"    Contains: {allergens}")

    # Filter for vegan guests
    print("\n" + "-" * 50)
    print("\n[VEGAN] Vegan-friendly recipes:")
    vegan_recipes = [r for r in recipes if r.has_dietary_tag(DietaryTag.VEGAN)]
    for r in vegan_recipes:
        print(f"    [OK] {r.name}")
    if not vegan_recipes:
        print("    (none found)")

    # Filter for nut allergy
    print("\n[NUT-FREE] Safe for nut allergy:")
    nut_safe = [r for r in recipes if r.is_safe_for([Allergen.TREE_NUTS, Allergen.PEANUTS])]
    for r in nut_safe:
        print(f"    [OK] {r.name}")

    # Filter for gluten-free + dairy-free
    print("\n[GF+DF] Gluten-free AND dairy-free:")
    gf_df = [r for r in recipes if r.matches_dietary_requirements(
        required_tags=[DietaryTag.GLUTEN_FREE],
        excluded_allergens=[Allergen.MILK]
    )]
    for r in gf_df:
        print(f"    [OK] {r.name}")
    if not gf_df:
        print("    (none found - consider substitutions!)")


def demo_phase5_skill():
    """Phase 5: Demonstrate skill level adaptation"""
    print("\n" + "=" * 70)
    print("PHASE 5: SKILL LEVEL ADAPTATION")
    print("=" * 70)

    recipe = create_sample_recipe()

    print(f"\nRecipe: {recipe.name}")
    print(f"Overall difficulty: {recipe.skill_level.value}")

    # Check suitability for different skill levels
    print("\n" + "-" * 50)
    print("Suitability by skill level:")
    for skill in SkillLevel:
        suitable = recipe.is_suitable_for_skill(skill)
        marker = "[OK]" if suitable else "[!]"
        print(f"  {marker} {skill.value}: {'Suitable' if suitable else 'May be challenging'}")

    # Show warnings for a beginner
    print("\n" + "-" * 50)
    print("\nFor a BEGINNER cook:")
    warnings = recipe.get_skill_warnings(SkillLevel.BEGINNER)
    if warnings:
        for w in warnings:
            print(f"  {w}")
    else:
        print("  No warnings - recipe is beginner-friendly!")

    # Show step descriptions for different skill levels
    print("\n" + "-" * 50)
    print("\nStep descriptions for BEGINNER:")
    beginner_steps = recipe.get_steps_for_skill(SkillLevel.BEGINNER)
    for step in beginner_steps:
        print(f"\n  [{step.id}] {step.description}")

    print("\n" + "-" * 50)
    print("\nStep descriptions for EXPERT:")
    expert_steps = recipe.get_steps_for_skill(SkillLevel.EXPERT)
    for step in expert_steps:
        print(f"\n  [{step.id}] {step.description}")


def demo_combined():
    """Demonstrate all features working together"""
    print("\n" + "=" * 70)
    print("COMBINED DEMO: PLANNING A DINNER PARTY")
    print("=" * 70)

    recipe = create_sample_recipe()

    # Scenario: 16 guests, one has dairy allergy, cook is intermediate
    print("\n[SCENARIO]")
    print("   - 16 guests (need to scale 2x)")
    print("   - One guest has dairy allergy")
    print("   - Cook skill level: Intermediate")
    print("   - Dinner at 6:00 PM, prep starts 2:00 PM")

    # Check dietary
    print("\n1. Dietary Check:")
    if recipe.is_safe_for([Allergen.MILK]):
        print(f"   [OK] {recipe.name} is dairy-free")
    else:
        print(f"   [!] {recipe.name} contains dairy - need substitution!")
        print("   [TIP] Suggestion: Use vegan butter instead")

    # Scale recipe
    print("\n2. Scale to 16 servings:")
    scaled = recipe.scale_to_servings(16)
    print(f"   {scaled.name}")
    print(f"   Prep time: {scaled.estimated_prep_minutes} min (was {recipe.estimated_prep_minutes})")
    print(f"   Cook time: {scaled.estimated_cook_minutes} min (was {recipe.estimated_cook_minutes})")

    # Check skill level
    print("\n3. Skill Level Check:")
    warnings = scaled.get_skill_warnings(SkillLevel.INTERMEDIATE)
    if warnings:
        for w in warnings:
            print(f"   {w}")
    else:
        print("   [OK] All steps are within your skill level!")

    # Create schedule with time windows
    print("\n4. Create Schedule:")
    tasks = []
    for step in scaled.steps:
        tasks.append(Task(
            id=f"task_{step.id}",
            recipe_id=scaled.id,
            step_id=step.id,
            duration_minutes=step.duration_minutes,
            appliance_id=step.appliance_type,
            skill_level=step.skill_level,
        ))

    plan = DailyPlan(date=date.today(), tasks=tasks)
    plan.assign_time_windows(start_time=time(14, 0), buffer_minutes=5)  # Start at 2 PM

    print("\n   Timeline:")
    print("   " + "-" * 45)
    for entry in plan.get_timeline():
        task = entry["task"]
        step = scaled.get_step(task.step_id)
        desc = step.description[:40] if step else task.step_id
        print(f"   {entry['start_time'].strftime('%H:%M')} - "
              f"{entry['end_time'].strftime('%H:%M')} | {desc}...")

    print("\n   [OK] Schedule complete! Ready to cook at 2:00 PM")
    print(f"   [->] Estimated completion: {plan.get_timeline()[-1]['end_time'].strftime('%H:%M')}")


def main():
    """Run all demos"""
    print("=" * 70)
    print("SOUS MOONSHOT FEATURES DEMO")
    print("Phases 2-5 Implementation Showcase")
    print("=" * 70)

    demo_phase2_time_windows()
    demo_phase3_scaling()
    demo_phase4_dietary()
    demo_phase5_skill()
    demo_combined()

    print("\n" + "=" * 70)
    print("[OK] ALL DEMOS COMPLETE!")
    print("=" * 70)
    print("\nSummary of Moonshot Features:")
    print("  Phase 2: [OK] Time Windows - Precise start/end times")
    print("  Phase 3: [OK] Recipe Scaling - 2x, 4x, target servings")
    print("  Phase 4: [OK] Dietary Filters - Tags, allergens, safety checks")
    print("  Phase 5: [OK] Skill Adaptation - Tips, warnings, adjusted steps")
    print()


if __name__ == "__main__":
    main()

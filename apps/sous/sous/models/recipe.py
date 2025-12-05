"""Recipe models"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum


class DietaryTag(Enum):
    """Dietary restriction tags (Phase 4)"""
    VEGAN = "vegan"
    VEGETARIAN = "vegetarian"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    NUT_FREE = "nut_free"
    SOY_FREE = "soy_free"
    EGG_FREE = "egg_free"
    KOSHER = "kosher"
    HALAL = "halal"
    LOW_SODIUM = "low_sodium"
    LOW_SUGAR = "low_sugar"
    KETO = "keto"
    PALEO = "paleo"


class Allergen(Enum):
    """Common allergens (Phase 4)"""
    MILK = "milk"
    EGGS = "eggs"
    FISH = "fish"
    SHELLFISH = "shellfish"
    TREE_NUTS = "tree_nuts"
    PEANUTS = "peanuts"
    WHEAT = "wheat"
    SOY = "soy"
    SESAME = "sesame"


class SkillLevel(Enum):
    """Recipe/step skill levels (Phase 5)"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class RecipeIngredient:
    """
    Ingredient reference within a recipe

    Attributes:
        ingredient_id: Reference to ingredient in ingredients.json
        quantity: Amount (optional for MVP)
        quantity_numeric: Numeric quantity for scaling (Phase 3)
        unit: Unit of measurement (optional for MVP)
        optional: Whether ingredient is optional
        allergens: List of allergens in this ingredient (Phase 4)
    """

    ingredient_id: str
    quantity: Optional[str] = None
    quantity_numeric: Optional[float] = None  # For scaling calculations
    unit: Optional[str] = None
    optional: bool = False
    allergens: List[Allergen] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "RecipeIngredient":
        """Create RecipeIngredient from JSON dict"""
        allergens = []
        if "allergens" in data:
            allergens = [Allergen(a) for a in data["allergens"]]

        return cls(
            ingredient_id=data["ingredient_id"],
            quantity=data.get("quantity"),
            quantity_numeric=data.get("quantity_numeric"),
            unit=data.get("unit"),
            optional=data.get("optional", False),
            allergens=allergens,
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        result = {"ingredient_id": self.ingredient_id}
        if self.quantity:
            result["quantity"] = self.quantity
        if self.quantity_numeric:
            result["quantity_numeric"] = self.quantity_numeric
        if self.unit:
            result["unit"] = self.unit
        if self.optional:
            result["optional"] = self.optional
        if self.allergens:
            result["allergens"] = [a.value for a in self.allergens]
        return result

    def scale(self, factor: float) -> "RecipeIngredient":
        """Create scaled copy of ingredient (Phase 3)"""
        new_quantity = None
        new_quantity_numeric = None

        if self.quantity_numeric:
            new_quantity_numeric = self.quantity_numeric * factor
            new_quantity = f"{new_quantity_numeric:.2f}".rstrip('0').rstrip('.')
        elif self.quantity:
            # Try to parse and scale quantity string
            try:
                num = float(self.quantity.split()[0])
                new_quantity_numeric = num * factor
                new_quantity = f"{new_quantity_numeric:.2f}".rstrip('0').rstrip('.')
                if len(self.quantity.split()) > 1:
                    new_quantity += " " + " ".join(self.quantity.split()[1:])
            except (ValueError, IndexError):
                new_quantity = self.quantity  # Keep original if can't parse

        return RecipeIngredient(
            ingredient_id=self.ingredient_id,
            quantity=new_quantity,
            quantity_numeric=new_quantity_numeric,
            unit=self.unit,
            optional=self.optional,
            allergens=self.allergens.copy(),
        )


@dataclass
class Step:
    """
    Recipe step with timing and appliance requirements

    Attributes:
        id: Unique step identifier
        description: What to do
        appliance_type: Required appliance (oven, burner, none)
        duration_minutes: Estimated time
        can_be_any_prep_day: Can be done any day before event
        must_be_day_before: Must be done exactly 1 day before
        must_be_same_day_as_event: Must be done on event day
        depends_on: List of step IDs that must complete first
        skill_level: Required skill level for this step (Phase 5)
        scaling_factor: How duration scales (1.0 = linear, 0.5 = sqrt) (Phase 3)
        tips_beginner: Extra tips for beginners (Phase 5)
        tips_advanced: Shortcuts for experts (Phase 5)
    """

    id: str
    description: str
    appliance_type: Optional[str]
    duration_minutes: int
    can_be_any_prep_day: bool = False
    must_be_day_before: bool = False
    must_be_same_day_as_event: bool = False
    depends_on: List[str] = field(default_factory=list)
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE
    scaling_factor: float = 0.5  # Default: doubling recipe doesn't double time
    tips_beginner: Optional[str] = None
    tips_advanced: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        """Create Step from JSON dict"""
        skill_level = SkillLevel.INTERMEDIATE
        if "skill_level" in data:
            skill_level = SkillLevel(data["skill_level"])

        return cls(
            id=data["id"],
            description=data["description"],
            appliance_type=data.get("appliance_type"),
            duration_minutes=data["duration_minutes"],
            can_be_any_prep_day=data.get("can_be_any_prep_day", False),
            must_be_day_before=data.get("must_be_day_before", False),
            must_be_same_day_as_event=data.get("must_be_same_day_as_event", False),
            depends_on=data.get("depends_on", []),
            skill_level=skill_level,
            scaling_factor=data.get("scaling_factor", 0.5),
            tips_beginner=data.get("tips_beginner"),
            tips_advanced=data.get("tips_advanced"),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        result = {
            "id": self.id,
            "description": self.description,
            "appliance_type": self.appliance_type,
            "duration_minutes": self.duration_minutes,
            "can_be_any_prep_day": self.can_be_any_prep_day,
            "must_be_day_before": self.must_be_day_before,
            "must_be_same_day_as_event": self.must_be_same_day_as_event,
            "depends_on": self.depends_on,
        }
        if self.skill_level != SkillLevel.INTERMEDIATE:
            result["skill_level"] = self.skill_level.value
        if self.scaling_factor != 0.5:
            result["scaling_factor"] = self.scaling_factor
        if self.tips_beginner:
            result["tips_beginner"] = self.tips_beginner
        if self.tips_advanced:
            result["tips_advanced"] = self.tips_advanced
        return result

    def get_scaled_duration(self, scale: float) -> int:
        """
        Calculate duration when recipe is scaled (Phase 3)

        Uses scaling_factor to determine how time scales:
        - 0.0 = constant time (e.g., preheating oven)
        - 0.5 = square root scaling (most cooking tasks)
        - 1.0 = linear scaling (e.g., manual work)

        Args:
            scale: Recipe scale factor (2.0 = doubling, 0.5 = halving)

        Returns:
            Scaled duration in minutes (minimum 1 minute)
        """
        import math

        # Handle edge cases
        if scale <= 0:
            return self.duration_minutes  # Invalid scale, return original
        if scale == 1.0:
            return self.duration_minutes  # No change
        if self.scaling_factor == 0.0:
            return self.duration_minutes  # Constant time (e.g., preheating)

        # Apply scaling factor: new_time = base_time * scale^scaling_factor
        # Works for both scale-up (scale > 1) and scale-down (scale < 1)
        scaled = self.duration_minutes * (scale ** self.scaling_factor)
        return max(1, int(math.ceil(scaled)))  # Minimum 1 minute

    def get_description_for_skill(self, user_skill: SkillLevel) -> str:
        """
        Get step description adjusted for user skill level (Phase 5)

        Args:
            user_skill: User's skill level

        Returns:
            Description with appropriate tips
        """
        desc = self.description

        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE,
                       SkillLevel.ADVANCED, SkillLevel.EXPERT]

        user_idx = skill_order.index(user_skill)
        step_idx = skill_order.index(self.skill_level)

        if user_idx < step_idx and self.tips_beginner:
            # User is less experienced, add beginner tips
            desc += f"\n  [TIP] {self.tips_beginner}"
        elif user_idx > step_idx and self.tips_advanced:
            # User is more experienced, offer shortcuts
            desc += f"\n  [SHORTCUT] {self.tips_advanced}"

        return desc


@dataclass
class SubstitutionRule:
    """
    Ingredient substitution rule

    Attributes:
        condition: When to substitute (e.g., "store didn't have green beans")
        alternative_ingredient_id: What to use instead
        notes: Additional guidance
    """

    condition: str
    alternative_ingredient_id: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "SubstitutionRule":
        """Create SubstitutionRule from JSON dict"""
        return cls(
            condition=data["condition"],
            alternative_ingredient_id=data["alternative_ingredient_id"],
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "condition": self.condition,
            "alternative_ingredient_id": self.alternative_ingredient_id,
            "notes": self.notes,
        }


@dataclass
class Recipe:
    """
    Complete recipe with ingredients, steps, and timing metadata

    Attributes:
        id: Unique identifier
        name: Recipe name (e.g., "Herb Stuffing x2")
        category: Recipe category (main, side, dessert, appetizer, drink, optional)
        ingredients: List of ingredients with quantities
        steps: Ordered list of preparation steps
        make_ahead_min_days: Minimum days before event
        make_ahead_max_days: Maximum days before event (None = no limit)
        required_appliances: List of appliance types needed
        estimated_prep_minutes: Total prep time estimate
        estimated_cook_minutes: Total cook time estimate
        is_optional: Whether recipe is optional (e.g., muffins)
        substitution_rules: List of substitution options
        base_servings: Base number of servings (Phase 3)
        scale_factor: Current scale factor (Phase 3)
        dietary_tags: Dietary restriction tags (Phase 4)
        allergens: Allergens present in recipe (Phase 4)
        skill_level: Overall recipe difficulty (Phase 5)
    """

    id: str
    name: str
    category: str
    ingredients: List[RecipeIngredient] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    make_ahead_min_days: int = 0
    make_ahead_max_days: Optional[int] = None
    required_appliances: List[str] = field(default_factory=list)
    estimated_prep_minutes: int = 0
    estimated_cook_minutes: int = 0
    is_optional: bool = False
    substitution_rules: List[SubstitutionRule] = field(default_factory=list)
    # Phase 3: Scaling
    base_servings: int = 4
    scale_factor: float = 1.0
    # Phase 4: Dietary
    dietary_tags: List[DietaryTag] = field(default_factory=list)
    allergens: List[Allergen] = field(default_factory=list)
    # Phase 5: Skill
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE

    @classmethod
    def from_dict(cls, data: dict) -> "Recipe":
        """Create Recipe from JSON dict"""
        ingredients = [
            RecipeIngredient.from_dict(i) for i in data.get("ingredients", [])
        ]
        steps = [Step.from_dict(s) for s in data.get("steps", [])]
        substitution_rules = [
            SubstitutionRule.from_dict(r)
            for r in data.get("substitution_rules", [])
        ]

        # Parse dietary tags
        dietary_tags = []
        if "dietary_tags" in data:
            dietary_tags = [DietaryTag(t) for t in data["dietary_tags"]]

        # Parse allergens
        allergens = []
        if "allergens" in data:
            allergens = [Allergen(a) for a in data["allergens"]]

        # Parse skill level
        skill_level = SkillLevel.INTERMEDIATE
        if "skill_level" in data:
            skill_level = SkillLevel(data["skill_level"])

        return cls(
            id=data["id"],
            name=data["name"],
            category=data["category"],
            ingredients=ingredients,
            steps=steps,
            make_ahead_min_days=data.get("make_ahead_min_days", 0),
            make_ahead_max_days=data.get("make_ahead_max_days"),
            required_appliances=data.get("required_appliances", []),
            estimated_prep_minutes=data.get("estimated_prep_minutes", 0),
            estimated_cook_minutes=data.get("estimated_cook_minutes", 0),
            is_optional=data.get("is_optional", False),
            substitution_rules=substitution_rules,
            base_servings=data.get("base_servings", 4),
            scale_factor=data.get("scale_factor", 1.0),
            dietary_tags=dietary_tags,
            allergens=allergens,
            skill_level=skill_level,
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        result = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "ingredients": [i.to_dict() for i in self.ingredients],
            "steps": [s.to_dict() for s in self.steps],
            "make_ahead_min_days": self.make_ahead_min_days,
            "required_appliances": self.required_appliances,
            "estimated_prep_minutes": self.estimated_prep_minutes,
            "estimated_cook_minutes": self.estimated_cook_minutes,
        }
        if self.make_ahead_max_days:
            result["make_ahead_max_days"] = self.make_ahead_max_days
        if self.is_optional:
            result["is_optional"] = self.is_optional
        if self.substitution_rules:
            result["substitution_rules"] = [r.to_dict() for r in self.substitution_rules]
        if self.base_servings != 4:
            result["base_servings"] = self.base_servings
        if self.scale_factor != 1.0:
            result["scale_factor"] = self.scale_factor
        if self.dietary_tags:
            result["dietary_tags"] = [t.value for t in self.dietary_tags]
        if self.allergens:
            result["allergens"] = [a.value for a in self.allergens]
        if self.skill_level != SkillLevel.INTERMEDIATE:
            result["skill_level"] = self.skill_level.value
        return result

    def get_step(self, step_id: str) -> Optional[Step]:
        """Get step by ID"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None

    def get_all_ingredient_ids(self) -> List[str]:
        """Get list of all ingredient IDs (excluding optionals)"""
        return [ing.ingredient_id for ing in self.ingredients if not ing.optional]

    def total_time_minutes(self) -> int:
        """Calculate total time (prep + cook)"""
        return self.estimated_prep_minutes + self.estimated_cook_minutes

    def requires_appliance(self, appliance_type: str) -> bool:
        """Check if recipe requires given appliance type"""
        return appliance_type in self.required_appliances

    # ========== Phase 3: Scaling Methods ==========

    def scale(self, factor: float) -> "Recipe":
        """
        Create a scaled copy of the recipe (Phase 3)

        Args:
            factor: Scale factor (2.0 = double, 0.5 = half)

        Returns:
            New Recipe with scaled ingredients and adjusted times
        """
        import math

        # Scale ingredients
        scaled_ingredients = [ing.scale(factor) for ing in self.ingredients]

        # Scale steps (duration adjusts based on scaling_factor)
        scaled_steps = []
        for step in self.steps:
            new_duration = step.get_scaled_duration(factor)
            scaled_step = Step(
                id=step.id,
                description=step.description,
                appliance_type=step.appliance_type,
                duration_minutes=new_duration,
                can_be_any_prep_day=step.can_be_any_prep_day,
                must_be_day_before=step.must_be_day_before,
                must_be_same_day_as_event=step.must_be_same_day_as_event,
                depends_on=step.depends_on.copy(),
                skill_level=step.skill_level,
                scaling_factor=step.scaling_factor,
                tips_beginner=step.tips_beginner,
                tips_advanced=step.tips_advanced,
            )
            scaled_steps.append(scaled_step)

        # Calculate new prep/cook times
        new_prep = int(math.ceil(self.estimated_prep_minutes * (factor ** 0.5)))
        new_cook = int(math.ceil(self.estimated_cook_minutes * (factor ** 0.3)))

        # Create scaled name
        scaled_name = self.name
        if factor != 1.0:
            scaled_name = f"{self.name} (x{factor:.1f})"

        return Recipe(
            id=f"{self.id}_scaled_{factor}",
            name=scaled_name,
            category=self.category,
            ingredients=scaled_ingredients,
            steps=scaled_steps,
            make_ahead_min_days=self.make_ahead_min_days,
            make_ahead_max_days=self.make_ahead_max_days,
            required_appliances=self.required_appliances.copy(),
            estimated_prep_minutes=new_prep,
            estimated_cook_minutes=new_cook,
            is_optional=self.is_optional,
            substitution_rules=[r for r in self.substitution_rules],
            base_servings=int(self.base_servings * factor),
            scale_factor=factor,
            dietary_tags=self.dietary_tags.copy(),
            allergens=self.allergens.copy(),
            skill_level=self.skill_level,
        )

    def scale_to_servings(self, target_servings: int) -> "Recipe":
        """
        Scale recipe to target number of servings (Phase 3)

        Args:
            target_servings: Desired number of servings

        Returns:
            Scaled recipe

        Raises:
            ValueError: If base_servings or target_servings is <= 0
        """
        if self.base_servings <= 0:
            raise ValueError(f"Cannot scale recipe with base_servings={self.base_servings}")
        if target_servings <= 0:
            raise ValueError(f"Cannot scale to target_servings={target_servings}")

        factor = target_servings / self.base_servings
        return self.scale(factor)

    # ========== Phase 4: Dietary Filter Methods ==========

    def has_dietary_tag(self, tag: DietaryTag) -> bool:
        """Check if recipe has a dietary tag"""
        return tag in self.dietary_tags

    def contains_allergen(self, allergen: Allergen) -> bool:
        """Check if recipe contains an allergen"""
        # Check recipe-level allergens
        if allergen in self.allergens:
            return True
        # Check ingredient-level allergens
        for ing in self.ingredients:
            if allergen in ing.allergens:
                return True
        return False

    def is_safe_for(self, excluded_allergens: List[Allergen]) -> bool:
        """Check if recipe is safe for someone with allergen restrictions"""
        for allergen in excluded_allergens:
            if self.contains_allergen(allergen):
                return False
        return True

    def matches_dietary_requirements(
        self,
        required_tags: Optional[List[DietaryTag]] = None,
        excluded_allergens: Optional[List[Allergen]] = None
    ) -> bool:
        """
        Check if recipe matches dietary requirements (Phase 4)

        Args:
            required_tags: Dietary tags that must be present
            excluded_allergens: Allergens that must NOT be present

        Returns:
            True if recipe matches all requirements
        """
        # Check required tags
        if required_tags:
            for tag in required_tags:
                if not self.has_dietary_tag(tag):
                    return False

        # Check excluded allergens
        if excluded_allergens:
            if not self.is_safe_for(excluded_allergens):
                return False

        return True

    # ========== Phase 5: Skill Level Methods ==========

    def is_suitable_for_skill(self, user_skill: SkillLevel) -> bool:
        """
        Check if recipe is suitable for user's skill level (Phase 5)

        Args:
            user_skill: User's skill level

        Returns:
            True if user can handle this recipe
        """
        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE,
                       SkillLevel.ADVANCED, SkillLevel.EXPERT]

        user_idx = skill_order.index(user_skill)
        recipe_idx = skill_order.index(self.skill_level)

        # User can handle recipe at or below their skill level
        return user_idx >= recipe_idx

    def get_skill_warnings(self, user_skill: SkillLevel) -> List[str]:
        """
        Get warnings for skill level mismatch (Phase 5)

        Returns list of warning messages for challenging steps
        """
        warnings = []
        skill_order = [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE,
                       SkillLevel.ADVANCED, SkillLevel.EXPERT]
        user_idx = skill_order.index(user_skill)

        for step in self.steps:
            step_idx = skill_order.index(step.skill_level)
            if step_idx > user_idx:
                warnings.append(
                    f"[!] Step '{step.id}' requires {step.skill_level.value} skill "
                    f"(your level: {user_skill.value})"
                )

        return warnings

    def get_steps_for_skill(self, user_skill: SkillLevel) -> List[Step]:
        """
        Get steps with descriptions adjusted for user skill (Phase 5)

        Returns steps with appropriate tips/shortcuts
        """
        adjusted_steps = []
        for step in self.steps:
            # Create copy with adjusted description
            adjusted = Step(
                id=step.id,
                description=step.get_description_for_skill(user_skill),
                appliance_type=step.appliance_type,
                duration_minutes=step.duration_minutes,
                can_be_any_prep_day=step.can_be_any_prep_day,
                must_be_day_before=step.must_be_day_before,
                must_be_same_day_as_event=step.must_be_same_day_as_event,
                depends_on=step.depends_on.copy(),
                skill_level=step.skill_level,
                scaling_factor=step.scaling_factor,
                tips_beginner=step.tips_beginner,
                tips_advanced=step.tips_advanced,
            )
            adjusted_steps.append(adjusted)
        return adjusted_steps

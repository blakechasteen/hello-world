"""Recipe models"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RecipeIngredient:
    """
    Ingredient reference within a recipe

    Attributes:
        ingredient_id: Reference to ingredient in ingredients.json
        quantity: Amount (optional for MVP)
        unit: Unit of measurement (optional for MVP)
        optional: Whether ingredient is optional
    """

    ingredient_id: str
    quantity: Optional[str] = None
    unit: Optional[str] = None
    optional: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "RecipeIngredient":
        """Create RecipeIngredient from JSON dict"""
        return cls(
            ingredient_id=data["ingredient_id"],
            quantity=data.get("quantity"),
            unit=data.get("unit"),
            optional=data.get("optional", False),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        result = {"ingredient_id": self.ingredient_id}
        if self.quantity:
            result["quantity"] = self.quantity
        if self.unit:
            result["unit"] = self.unit
        if self.optional:
            result["optional"] = self.optional
        return result


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
    """

    id: str
    description: str
    appliance_type: Optional[str]
    duration_minutes: int
    can_be_any_prep_day: bool = False
    must_be_day_before: bool = False
    must_be_same_day_as_event: bool = False
    depends_on: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Step":
        """Create Step from JSON dict"""
        return cls(
            id=data["id"],
            description=data["description"],
            appliance_type=data.get("appliance_type"),
            duration_minutes=data["duration_minutes"],
            can_be_any_prep_day=data.get("can_be_any_prep_day", False),
            must_be_day_before=data.get("must_be_day_before", False),
            must_be_same_day_as_event=data.get("must_be_same_day_as_event", False),
            depends_on=data.get("depends_on", []),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "id": self.id,
            "description": self.description,
            "appliance_type": self.appliance_type,
            "duration_minutes": self.duration_minutes,
            "can_be_any_prep_day": self.can_be_any_prep_day,
            "must_be_day_before": self.must_be_day_before,
            "must_be_same_day_as_event": self.must_be_same_day_as_event,
            "depends_on": self.depends_on,
        }


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

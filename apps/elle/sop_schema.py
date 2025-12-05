"""
Voice-Editable SOP Schema for Elle Core

Supports:
- Voice commands for hands-free editing
- Version control with timestamps
- Natural language updates
- HoloLoom memory integration
- Cost tracking per step

Created: 2025-11-15
Author: Blake Chasteen
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class StepType(Enum):
    """Type of SOP step"""
    PREP = "prep"  # Preparation
    PROCESS = "process"  # Main work
    QC = "quality_control"  # Quality check
    CLEANUP = "cleanup"  # Cleanup/storage
    SAFETY = "safety"  # Safety checkpoint


class UnitType(Enum):
    """Measurement units"""
    WEIGHT = "weight"  # lbs, oz, kg, g
    VOLUME = "volume"  # cups, gallons, liters
    COUNT = "count"  # units, pieces
    TIME = "time"  # minutes, hours
    TEMPERATURE = "temperature"  # F, C


@dataclass
class Ingredient:
    """Recipe ingredient with cost tracking"""
    name: str
    amount: float
    unit: str
    unit_type: UnitType
    cost_per_unit: Optional[float] = None  # Cost per unit
    supplier: Optional[str] = None
    notes: Optional[str] = None

    @property
    def total_cost(self) -> float:
        """Calculate total cost for this ingredient"""
        if self.cost_per_unit is None:
            return 0.0
        return self.amount * self.cost_per_unit

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "amount": self.amount,
            "unit": self.unit,
            "unit_type": self.unit_type.value,
            "cost_per_unit": self.cost_per_unit,
            "supplier": self.supplier,
            "notes": self.notes,
            "total_cost": self.total_cost
        }


@dataclass
class SOPStep:
    """Individual step in an SOP"""
    step_number: int
    type: StepType
    title: str
    description: str
    duration_minutes: Optional[int] = None
    temperature: Optional[int] = None  # For cooking/baking
    temperature_unit: str = "F"
    safety_notes: Optional[str] = None
    quality_checkpoints: List[str] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    tips: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "type": self.type.value,
            "title": self.title,
            "description": self.description,
            "duration_minutes": self.duration_minutes,
            "temperature": self.temperature,
            "temperature_unit": self.temperature_unit,
            "safety_notes": self.safety_notes,
            "quality_checkpoints": self.quality_checkpoints,
            "tools_required": self.tools_required,
            "tips": self.tips
        }


@dataclass
class SOPVersion:
    """Version history entry"""
    version: str
    timestamp: datetime
    changes: str
    changed_by: str = "Voice Command"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "changes": self.changes,
            "changed_by": self.changed_by
        }


@dataclass
class SOP:
    """Complete Standard Operating Procedure"""

    # Metadata
    sop_id: str  # e.g., "BREAD_001"
    name: str  # e.g., "Sourdough Bread Production"
    category: str  # e.g., "Bakery", "Apothecary", "Farm"
    version: str = "1.0.0"
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)
    author: str = "Blake Chasteen"
    status: str = "active"  # active, draft, archived

    # Core content
    description: str = ""
    ingredients: List[Ingredient] = field(default_factory=list)
    steps: List[SOPStep] = field(default_factory=list)

    # Production details
    batch_size: Optional[int] = None  # Number of units produced
    batch_unit: str = "units"
    total_time_minutes: Optional[int] = None

    # Cost tracking
    labor_rate_per_hour: float = 25.00  # Default labor rate
    overhead_multiplier: float = 1.15  # 15% overhead

    # Quality & Safety
    quality_standards: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)

    # Marketing
    selling_price: Optional[float] = None
    target_margin: Optional[float] = None  # e.g., 0.67 for 67%

    # Version history
    version_history: List[SOPVersion] = field(default_factory=list)

    # Tags for retrieval
    tags: List[str] = field(default_factory=list)

    @property
    def material_cost(self) -> float:
        """Total material cost for one batch"""
        return sum(ing.total_cost for ing in self.ingredients)

    @property
    def labor_cost(self) -> float:
        """Total labor cost for one batch"""
        if self.total_time_minutes is None:
            return 0.0
        hours = self.total_time_minutes / 60.0
        return hours * self.labor_rate_per_hour

    @property
    def total_cost(self) -> float:
        """Total cost including overhead"""
        base_cost = self.material_cost + self.labor_cost
        return base_cost * self.overhead_multiplier

    @property
    def cost_per_unit(self) -> float:
        """Cost per individual unit"""
        if self.batch_size is None or self.batch_size == 0:
            return self.total_cost
        return self.total_cost / self.batch_size

    @property
    def profit_per_batch(self) -> float:
        """Profit for one batch"""
        if self.selling_price is None or self.batch_size is None:
            return 0.0
        revenue = self.selling_price * self.batch_size
        return revenue - self.total_cost

    @property
    def profit_margin(self) -> float:
        """Profit margin as percentage"""
        if self.selling_price is None or self.batch_size is None:
            return 0.0
        revenue = self.selling_price * self.batch_size
        if revenue == 0:
            return 0.0
        return (self.profit_per_batch / revenue) * 100

    @property
    def hourly_roi(self) -> float:
        """Profit per hour of labor"""
        if self.total_time_minutes is None or self.total_time_minutes == 0:
            return 0.0
        hours = self.total_time_minutes / 60.0
        return self.profit_per_batch / hours

    def add_ingredient(
        self,
        name: str,
        amount: float,
        unit: str,
        unit_type: UnitType,
        cost_per_unit: Optional[float] = None,
        supplier: Optional[str] = None
    ):
        """Add ingredient to recipe"""
        ingredient = Ingredient(
            name=name,
            amount=amount,
            unit=unit,
            unit_type=unit_type,
            cost_per_unit=cost_per_unit,
            supplier=supplier
        )
        self.ingredients.append(ingredient)
        self._bump_version("Added ingredient: " + name)

    def add_step(
        self,
        step_type: StepType,
        title: str,
        description: str,
        duration_minutes: Optional[int] = None,
        **kwargs
    ):
        """Add step to procedure"""
        step_number = len(self.steps) + 1
        step = SOPStep(
            step_number=step_number,
            type=step_type,
            title=title,
            description=description,
            duration_minutes=duration_minutes,
            **kwargs
        )
        self.steps.append(step)
        self._bump_version(f"Added step {step_number}: {title}")

    def update_step(
        self,
        step_number: int,
        **updates
    ):
        """Update an existing step"""
        if step_number < 1 or step_number > len(self.steps):
            raise ValueError(f"Invalid step number: {step_number}")

        step = self.steps[step_number - 1]
        changes = []

        for key, value in updates.items():
            if hasattr(step, key):
                old_value = getattr(step, key)
                setattr(step, key, value)
                changes.append(f"{key}: {old_value} → {value}")

        if changes:
            change_desc = f"Updated step {step_number}: " + ", ".join(changes)
            self._bump_version(change_desc)

    def remove_step(self, step_number: int):
        """Remove a step and renumber remaining steps"""
        if step_number < 1 or step_number > len(self.steps):
            raise ValueError(f"Invalid step number: {step_number}")

        removed_step = self.steps.pop(step_number - 1)

        # Renumber remaining steps
        for i, step in enumerate(self.steps):
            step.step_number = i + 1

        self._bump_version(f"Removed step: {removed_step.title}")

    def _bump_version(self, changes: str):
        """Increment version and record changes"""
        # Store old version
        version_entry = SOPVersion(
            version=self.version,
            timestamp=self.updated,
            changes=changes
        )
        self.version_history.append(version_entry)

        # Bump version (simple patch increment)
        parts = self.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        self.version = ".".join(parts)
        self.updated = datetime.now()

    def to_markdown(self) -> str:
        """Export SOP to markdown format"""
        md = f"""# {self.name}

**ID:** {self.sop_id}
**Version:** {self.version}
**Category:** {self.category}
**Status:** {self.status}
**Last Updated:** {self.updated.strftime('%Y-%m-%d')}

## Description

{self.description}

## Production Details

- **Batch Size:** {self.batch_size} {self.batch_unit}
- **Total Time:** {self.total_time_minutes} minutes
- **Selling Price:** ${self.selling_price:.2f} per unit
- **Target Margin:** {self.target_margin * 100:.1f}%

## Cost Analysis

| Item | Amount |
|------|--------|
| Material Cost | ${self.material_cost:.2f} |
| Labor Cost | ${self.labor_cost:.2f} |
| Total Cost (with overhead) | ${self.total_cost:.2f} |
| Cost Per Unit | ${self.cost_per_unit:.2f} |
| **Revenue Per Batch** | **${self.selling_price * self.batch_size:.2f}** |
| **Profit Per Batch** | **${self.profit_per_batch:.2f}** |
| **Profit Margin** | **{self.profit_margin:.1f}%** |
| **Hourly ROI** | **${self.hourly_roi:.2f}/hour** |

## Ingredients

"""
        for ing in self.ingredients:
            cost_info = f"(${ing.total_cost:.2f})" if ing.cost_per_unit else ""
            md += f"- {ing.amount} {ing.unit} {ing.name} {cost_info}\n"

        md += "\n## Procedure\n\n"

        for step in self.steps:
            md += f"### Step {step.step_number}: {step.title}\n\n"
            md += f"**Type:** {step.type.value}  \n"
            if step.duration_minutes:
                md += f"**Duration:** {step.duration_minutes} minutes  \n"
            if step.temperature:
                md += f"**Temperature:** {step.temperature}°{step.temperature_unit}  \n"
            md += f"\n{step.description}\n\n"

            if step.tools_required:
                md += "**Tools Required:**\n"
                for tool in step.tools_required:
                    md += f"- {tool}\n"
                md += "\n"

            if step.quality_checkpoints:
                md += "**Quality Checkpoints:**\n"
                for checkpoint in step.quality_checkpoints:
                    md += f"- {checkpoint}\n"
                md += "\n"

            if step.safety_notes:
                md += f"⚠️ **Safety:** {step.safety_notes}\n\n"

            if step.tips:
                md += f"💡 **Tip:** {step.tips}\n\n"

        if self.quality_standards:
            md += "## Quality Standards\n\n"
            for standard in self.quality_standards:
                md += f"- {standard}\n"
            md += "\n"

        if self.safety_requirements:
            md += "## Safety Requirements\n\n"
            for req in self.safety_requirements:
                md += f"- {req}\n"
            md += "\n"

        if self.version_history:
            md += "## Version History\n\n"
            for version in reversed(self.version_history[-5:]):  # Last 5 versions
                md += f"- **{version.version}** ({version.timestamp.strftime('%Y-%m-%d')}): {version.changes}\n"

        md += f"\n---\n\n*SOP maintained by {self.author}*\n"
        md += f"*Tags: {', '.join(self.tags)}*\n"

        return md

    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary for JSON storage"""
        return {
            "sop_id": self.sop_id,
            "name": self.name,
            "category": self.category,
            "version": self.version,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "author": self.author,
            "status": self.status,
            "description": self.description,
            "ingredients": [ing.to_dict() for ing in self.ingredients],
            "steps": [step.to_dict() for step in self.steps],
            "batch_size": self.batch_size,
            "batch_unit": self.batch_unit,
            "total_time_minutes": self.total_time_minutes,
            "labor_rate_per_hour": self.labor_rate_per_hour,
            "overhead_multiplier": self.overhead_multiplier,
            "selling_price": self.selling_price,
            "target_margin": self.target_margin,
            "quality_standards": self.quality_standards,
            "safety_requirements": self.safety_requirements,
            "version_history": [v.to_dict() for v in self.version_history],
            "tags": self.tags,
            "analytics": {
                "material_cost": self.material_cost,
                "labor_cost": self.labor_cost,
                "total_cost": self.total_cost,
                "cost_per_unit": self.cost_per_unit,
                "profit_per_batch": self.profit_per_batch,
                "profit_margin": self.profit_margin,
                "hourly_roi": self.hourly_roi
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SOP':
        """Load from dictionary"""
        sop = cls(
            sop_id=data["sop_id"],
            name=data["name"],
            category=data["category"],
            version=data.get("version", "1.0.0"),
            created=datetime.fromisoformat(data["created"]),
            updated=datetime.fromisoformat(data["updated"]),
            author=data.get("author", "Blake Chasteen"),
            status=data.get("status", "active"),
            description=data.get("description", ""),
            batch_size=data.get("batch_size"),
            batch_unit=data.get("batch_unit", "units"),
            total_time_minutes=data.get("total_time_minutes"),
            labor_rate_per_hour=data.get("labor_rate_per_hour", 25.00),
            overhead_multiplier=data.get("overhead_multiplier", 1.15),
            selling_price=data.get("selling_price"),
            target_margin=data.get("target_margin"),
            quality_standards=data.get("quality_standards", []),
            safety_requirements=data.get("safety_requirements", []),
            tags=data.get("tags", [])
        )

        # Load ingredients
        for ing_data in data.get("ingredients", []):
            sop.ingredients.append(Ingredient(
                name=ing_data["name"],
                amount=ing_data["amount"],
                unit=ing_data["unit"],
                unit_type=UnitType(ing_data["unit_type"]),
                cost_per_unit=ing_data.get("cost_per_unit"),
                supplier=ing_data.get("supplier"),
                notes=ing_data.get("notes")
            ))

        # Load steps
        for step_data in data.get("steps", []):
            sop.steps.append(SOPStep(
                step_number=step_data["step_number"],
                type=StepType(step_data["type"]),
                title=step_data["title"],
                description=step_data["description"],
                duration_minutes=step_data.get("duration_minutes"),
                temperature=step_data.get("temperature"),
                temperature_unit=step_data.get("temperature_unit", "F"),
                safety_notes=step_data.get("safety_notes"),
                quality_checkpoints=step_data.get("quality_checkpoints", []),
                tools_required=step_data.get("tools_required", []),
                tips=step_data.get("tips")
            ))

        # Load version history
        for version_data in data.get("version_history", []):
            sop.version_history.append(SOPVersion(
                version=version_data["version"],
                timestamp=datetime.fromisoformat(version_data["timestamp"]),
                changes=version_data["changes"],
                changed_by=version_data.get("changed_by", "Voice Command")
            ))

        return sop


# Example: Create a bread SOP
def create_bread_sop_example() -> SOP:
    """Example: Create a sourdough bread SOP"""
    sop = SOP(
        sop_id="BREAD_001",
        name="Sourdough Bread Production",
        category="Bakery",
        description="Classic sourdough loaves with 24-hour fermentation. Produces crusty exterior with soft, tangy interior.",
        batch_size=24,
        batch_unit="loaves",
        total_time_minutes=150,  # 2.5 hours active time
        selling_price=6.00,
        target_margin=0.67,
        tags=["bread", "sourdough", "bakery", "high-margin"]
    )

    # Add ingredients
    sop.add_ingredient("Bread Flour", 25, "lbs", UnitType.WEIGHT, 0.80, "Costco")
    sop.add_ingredient("Water", 4, "gallons", UnitType.VOLUME, 0.00)
    sop.add_ingredient("Sourdough Starter", 2, "lbs", UnitType.WEIGHT, 0.50, "Farm")
    sop.add_ingredient("Sea Salt", 8, "oz", UnitType.WEIGHT, 0.15, "Bulk")

    # Add steps
    sop.add_step(
        StepType.PREP,
        "Mix Dough",
        "Combine flour, water, and starter in large mixing bowl. Mix until shaggy dough forms.",
        duration_minutes=15,
        tools_required=["Large mixing bowl", "Wooden spoon"],
        tips="Water should be lukewarm (85-90°F) for optimal fermentation"
    )

    sop.add_step(
        StepType.PROCESS,
        "Autolyse",
        "Let dough rest covered for 30 minutes to develop gluten naturally.",
        duration_minutes=30,
        quality_checkpoints=["Dough should feel more elastic after rest"]
    )

    sop.add_step(
        StepType.PROCESS,
        "Add Salt and Knead",
        "Add salt and knead for 10 minutes until smooth and elastic. Dough should pass windowpane test.",
        duration_minutes=15,
        quality_checkpoints=[
            "Windowpane test passes",
            "Dough springs back when poked",
            "Surface is smooth"
        ],
        tips="If dough is too sticky, add flour 1 tablespoon at a time"
    )

    sop.add_step(
        StepType.PROCESS,
        "Bulk Fermentation",
        "Place in oiled bowl, cover, ferment 24 hours at room temperature (68-72°F).",
        duration_minutes=1440,  # Note: Most time is passive
        temperature=70,
        quality_checkpoints=["Dough doubles in size", "Surface has bubbles"]
    )

    sop.add_step(
        StepType.PROCESS,
        "Shape Loaves",
        "Divide dough into 24 pieces (about 6 oz each). Shape into tight rounds.",
        duration_minutes=20,
        tools_required=["Bench scraper", "Kitchen scale"],
        tips="Keep unused dough covered to prevent drying"
    )

    sop.add_step(
        StepType.PROCESS,
        "Proof",
        "Place shaped loaves on parchment-lined trays. Proof 45 minutes covered.",
        duration_minutes=45,
        quality_checkpoints=["Loaves puff up 50%", "Indent springs back slowly"]
    )

    sop.add_step(
        StepType.PROCESS,
        "Score and Bake",
        "Score tops with sharp blade. Bake at 450°F for 25-30 minutes until deep golden brown.",
        duration_minutes=30,
        temperature=450,
        temperature_unit="F",
        tools_required=["Lame or sharp knife", "Oven"],
        quality_checkpoints=[
            "Internal temp reaches 205°F",
            "Crust is deep golden",
            "Sounds hollow when tapped"
        ],
        safety_notes="Use oven mitts. Steam escapes when scoring."
    )

    sop.add_step(
        StepType.CLEANUP,
        "Cool and Package",
        "Cool on racks 30 minutes. Package in paper bags once fully cooled.",
        duration_minutes=15,
        quality_checkpoints=["Completely cool before packaging to prevent condensation"]
    )

    # Quality standards
    sop.quality_standards = [
        "Crust is golden brown and crispy",
        "Interior crumb is open with irregular holes",
        "Tangy sourdough aroma",
        "Loaves sound hollow when tapped",
        "No dense or gummy texture"
    ]

    # Safety requirements
    sop.safety_requirements = [
        "Wash hands before handling dough",
        "Ensure workspace is clean and sanitized",
        "Use oven mitts when handling hot trays",
        "Check internal temperature reaches 205°F for food safety"
    ]

    return sop


if __name__ == "__main__":
    # Create example SOP
    bread_sop = create_bread_sop_example()

    # Print analytics
    print(f"SOP: {bread_sop.name}")
    print(f"Version: {bread_sop.version}")
    print(f"\nCost Analysis:")
    print(f"  Material: ${bread_sop.material_cost:.2f}")
    print(f"  Labor: ${bread_sop.labor_cost:.2f}")
    print(f"  Total: ${bread_sop.total_cost:.2f}")
    print(f"  Per Unit: ${bread_sop.cost_per_unit:.2f}")
    print(f"\nProfit Analysis:")
    print(f"  Revenue: ${bread_sop.selling_price * bread_sop.batch_size:.2f}")
    print(f"  Profit: ${bread_sop.profit_per_batch:.2f}")
    print(f"  Margin: {bread_sop.profit_margin:.1f}%")
    print(f"  Hourly ROI: ${bread_sop.hourly_roi:.2f}/hour")

    # Export to markdown
    md = bread_sop.to_markdown()
    with open("elle/examples/BREAD_SOP_EXAMPLE.md", "w") as f:
        f.write(md)
    print(f"\nExported to elle/examples/BREAD_SOP_EXAMPLE.md")

    # Export to JSON
    json_data = bread_sop.to_dict()
    with open("elle/examples/bread_sop.json", "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Exported to elle/examples/bread_sop.json")

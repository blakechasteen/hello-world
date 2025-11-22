"""
SOP (Standard Operating Procedure) Parser for COZ System

Parses process templates with time estimates, materials, equipment, and quality checks.
Enables batch cost estimation and process optimization.

Part of COZ Expansion Phase 2 (Orders & Intelligence)
Created: 2025-11-21
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict


@dataclass
class Material:
    """Material item with quantity and cost"""
    name: str
    quantity: str  # e.g., "500g", "10ml"
    cost: float


@dataclass
class Step:
    """Process step with description and duration"""
    number: int
    description: str
    duration_min: Optional[int] = None


@dataclass
class SOP:
    """Standard Operating Procedure"""
    name: str
    category: str
    estimated_time: float  # hours
    estimated_cost: float  # dollars
    difficulty: str  # Easy, Medium, Hard
    output: str  # What's produced
    steps: List[Step] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    quality_checks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "category": self.category,
            "estimated_time": self.estimated_time,
            "estimated_cost": self.estimated_cost,
            "difficulty": self.difficulty,
            "output": self.output,
            "steps_count": len(self.steps),
            "materials_count": len(self.materials),
            "equipment_count": len(self.equipment),
            "quality_checks_count": len(self.quality_checks),
        }

    def get_total_materials_cost(self) -> float:
        """Calculate total cost of materials"""
        return sum(m.cost for m in self.materials)

    def get_labor_cost(self, hourly_rate: float = 25.0) -> float:
        """Calculate labor cost"""
        return self.estimated_time * hourly_rate

    def estimate_batch_cost(self, quantity: int, hourly_rate: float = 25.0) -> Dict:
        """Estimate cost for producing quantity of this SOP output"""
        materials_cost = self.get_total_materials_cost() * quantity
        labor_cost = self.get_labor_cost(hourly_rate) * quantity
        total_cost = materials_cost + labor_cost

        return {
            "sop_name": self.name,
            "quantity": quantity,
            "materials_cost": round(materials_cost, 2),
            "labor_cost": round(labor_cost, 2),
            "total_cost": round(total_cost, 2),
            "cost_per_unit": round(total_cost / quantity, 2) if quantity > 0 else 0.0,
            "time_required_hours": round(self.estimated_time * quantity, 1),
        }


class SOPParser:
    """Parser for sops.md markdown file"""

    def __init__(self, file_path: str = "coz/sops.md"):
        self.file_path = Path(file_path)
        self.sops: List[SOP] = []
        self.sops_by_name: Dict[str, SOP] = {}
        self.sops_by_category: Dict[str, List[SOP]] = defaultdict(list)

    def parse(self) -> List[SOP]:
        """Parse sops.md file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"SOPs file not found: {self.file_path}")

        self.sops = []
        self.sops_by_name = {}
        self.sops_by_category = defaultdict(list)

        with open(self.file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by SOP headers (## SOP: ...)
        sop_sections = re.split(r'\n## SOP: ', content)

        # Skip first section (file header)
        for section in sop_sections[1:]:
            try:
                sop = self._parse_sop_section(section)
                self.sops.append(sop)
                self.sops_by_name[sop.name] = sop
                self.sops_by_category[sop.category].append(sop)
            except Exception as e:
                print(f"Warning: Failed to parse SOP section: {e}")
                continue

        return self.sops

    def _parse_sop_section(self, section: str) -> SOP:
        """Parse a single SOP section"""
        lines = section.strip().split('\n')

        # First line is the SOP name
        name = lines[0].strip()

        # Extract metadata (Category, Estimated Time, etc.)
        metadata = {}
        for line in lines[1:20]:  # Check first 20 lines for metadata
            if line.startswith('**') and ':**' in line:
                key_value = line.strip('*').strip()
                if ':**' in key_value:
                    key, value = key_value.split(':**', 1)
                    metadata[key.strip()] = value.strip()

        # Parse estimated time (convert to hours)
        estimated_time = 0.0
        if 'Estimated Time' in metadata:
            time_str = metadata['Estimated Time']
            if 'hours' in time_str or 'hour' in time_str:
                estimated_time = float(re.findall(r'[\d.]+', time_str)[0])
            elif 'min' in time_str:
                estimated_time = float(re.findall(r'[\d.]+', time_str)[0]) / 60.0

        # Parse estimated cost
        estimated_cost = 0.0
        if 'Estimated Cost' in metadata:
            cost_str = metadata['Estimated Cost']
            estimated_cost = float(re.findall(r'[\d.]+', cost_str)[0])

        # Create SOP object
        sop = SOP(
            name=name,
            category=metadata.get('Category', 'Unknown'),
            estimated_time=estimated_time,
            estimated_cost=estimated_cost,
            difficulty=metadata.get('Difficulty', 'Medium'),
            output=metadata.get('Output', '1 unit'),
        )

        # Parse sections
        current_section = None
        for line in lines:
            line = line.strip()

            # Detect section headers
            if line.startswith('### Steps:'):
                current_section = 'steps'
            elif line.startswith('### Materials:'):
                current_section = 'materials'
            elif line.startswith('### Equipment:'):
                current_section = 'equipment'
            elif line.startswith('### Notes:'):
                current_section = 'notes'
            elif line.startswith('### Quality Checks:'):
                current_section = 'quality_checks'
            elif line.startswith('###'):
                current_section = None  # Unknown section
            elif current_section == 'steps' and line and re.match(r'^\d+\.', line):
                # Parse step: "1. Mix dry ingredients - 15 min"
                step = self._parse_step(line)
                if step:
                    sop.steps.append(step)
            elif current_section == 'materials' and line.startswith('-'):
                # Parse material: "- Flour: 500g ($1.50)"
                material = self._parse_material(line)
                if material:
                    sop.materials.append(material)
            elif current_section == 'equipment' and line.startswith('-'):
                # Parse equipment: "- Mixing bowl"
                equipment = line.lstrip('- ').strip()
                if equipment:
                    sop.equipment.append(equipment)
            elif current_section == 'notes' and line.startswith('-'):
                # Parse note: "- Use room temperature water"
                note = line.lstrip('- ').strip()
                if note:
                    sop.notes.append(note)
            elif current_section == 'quality_checks' and line.startswith('- [ ]'):
                # Parse quality check: "- [ ] Dough has doubled in size"
                check = line.replace('- [ ]', '').strip()
                if check:
                    sop.quality_checks.append(check)

        return sop

    def _parse_step(self, line: str) -> Optional[Step]:
        """Parse a step line: '1. Mix dry ingredients - 15 min'"""
        try:
            # Extract step number
            match = re.match(r'^(\d+)\.\s+(.+)', line)
            if not match:
                return None

            number = int(match.group(1))
            rest = match.group(2).strip()

            # Extract duration if present
            duration_min = None
            description = rest

            if ' - ' in rest:
                parts = rest.split(' - ')
                description = parts[0].strip()
                time_part = parts[1].strip()

                # Extract minutes
                time_match = re.search(r'(\d+)\s*min', time_part)
                if time_match:
                    duration_min = int(time_match.group(1))

            return Step(number=number, description=description, duration_min=duration_min)
        except Exception:
            return None

    def _parse_material(self, line: str) -> Optional[Material]:
        """Parse a material line: '- Flour: 500g ($1.50)'"""
        try:
            # Remove leading dash
            line = line.lstrip('- ').strip()

            # Extract name, quantity, cost
            # Format: "Flour: 500g ($1.50)"
            if ':' not in line:
                return None

            name_part, rest = line.split(':', 1)
            name = name_part.strip()

            # Extract quantity and cost
            quantity = ''
            cost = 0.0

            # Find cost in parentheses
            cost_match = re.search(r'\(\$?([\d.]+)\)', rest)
            if cost_match:
                cost = float(cost_match.group(1))

            # Extract quantity (text before cost)
            if cost_match:
                quantity = rest[:cost_match.start()].strip()
            else:
                quantity = rest.strip()

            return Material(name=name, quantity=quantity, cost=cost)
        except Exception:
            return None

    def get_by_name(self, name: str) -> Optional[SOP]:
        """Get SOP by name"""
        return self.sops_by_name.get(name)

    def get_by_category(self, category: str) -> List[SOP]:
        """Get all SOPs in a category"""
        return self.sops_by_category.get(category, [])

    def get_by_time(self, max_hours: float) -> List[SOP]:
        """Get SOPs that take less than max_hours"""
        return [sop for sop in self.sops if sop.estimated_time <= max_hours]

    def get_by_difficulty(self, difficulty: str) -> List[SOP]:
        """Get SOPs by difficulty level"""
        return [sop for sop in self.sops if sop.difficulty.lower() == difficulty.lower()]

    def get_sop_library(self) -> Dict:
        """Get overview of all SOPs"""
        library = {
            "total_sops": len(self.sops),
            "categories": {},
            "avg_time": 0.0,
            "avg_cost": 0.0,
            "difficulty_breakdown": defaultdict(int),
        }

        if not self.sops:
            return library

        # Category breakdown
        for category, sops in self.sops_by_category.items():
            library["categories"][category] = {
                "count": len(sops),
                "avg_time": round(sum(s.estimated_time for s in sops) / len(sops), 1),
                "avg_cost": round(sum(s.estimated_cost for s in sops) / len(sops), 2),
            }

        # Overall averages
        library["avg_time"] = round(sum(s.estimated_time for s in self.sops) / len(self.sops), 1)
        library["avg_cost"] = round(sum(s.estimated_cost for s in self.sops) / len(self.sops), 2)

        # Difficulty breakdown
        for sop in self.sops:
            library["difficulty_breakdown"][sop.difficulty] += 1

        library["difficulty_breakdown"] = dict(library["difficulty_breakdown"])

        return library

    def estimate_batch(self, sop_name: str, quantity: int, hourly_rate: float = 25.0) -> Dict:
        """Estimate cost for batch production"""
        sop = self.get_by_name(sop_name)
        if not sop:
            return {"error": f"SOP '{sop_name}' not found"}

        return sop.estimate_batch_cost(quantity, hourly_rate)

    def estimate_multiple_sops(self, sop_quantities: Dict[str, int], hourly_rate: float = 25.0) -> Dict:
        """Estimate cost for producing multiple SOPs"""
        total_materials = 0.0
        total_labor = 0.0
        total_time = 0.0
        batch_details = []

        for sop_name, quantity in sop_quantities.items():
            batch = self.estimate_batch(sop_name, quantity, hourly_rate)
            if "error" not in batch:
                total_materials += batch["materials_cost"]
                total_labor += batch["labor_cost"]
                total_time += batch["time_required_hours"]
                batch_details.append(batch)

        return {
            "batches": batch_details,
            "total_materials_cost": round(total_materials, 2),
            "total_labor_cost": round(total_labor, 2),
            "total_cost": round(total_materials + total_labor, 2),
            "total_time_hours": round(total_time, 1),
        }

    def get_sop_checklist(self, sop_name: str) -> Dict:
        """Get quality checklist for an SOP"""
        sop = self.get_by_name(sop_name)
        if not sop:
            return {"error": f"SOP '{sop_name}' not found"}

        return {
            "sop_name": sop.name,
            "quality_checks": sop.quality_checks,
            "steps": [{"number": s.number, "description": s.description} for s in sop.steps],
            "equipment": sop.equipment,
            "notes": sop.notes,
        }


if __name__ == "__main__":
    # Demo usage
    parser = SOPParser()

    try:
        sops = parser.parse()

        print(f"✅ Parsed {len(sops)} SOPs")
        print("\n📚 SOP Library:")
        library = parser.get_sop_library()
        print(f"  Total SOPs: {library['total_sops']}")
        print(f"  Average Time: {library['avg_time']}h")
        print(f"  Average Cost: ${library['avg_cost']}")

        print("\n📂 By Category:")
        for category, stats in library['categories'].items():
            print(f"  {category}: {stats['count']} SOPs (avg {stats['avg_time']}h, ${stats['avg_cost']})")

        print("\n⏱️ SOPs Under 2 Hours:")
        quick_sops = parser.get_by_time(2.0)
        for sop in quick_sops:
            print(f"  - {sop.name}: {sop.estimated_time}h, ${sop.estimated_cost}")

        print("\n💰 Batch Estimate (5 bread loaves):")
        batch = parser.estimate_batch("Bake Bread Loaf", 5)
        if "error" not in batch:
            print(f"  Materials: ${batch['materials_cost']}")
            print(f"  Labor: ${batch['labor_cost']}")
            print(f"  Total: ${batch['total_cost']}")
            print(f"  Time: {batch['time_required_hours']}h")
            print(f"  Cost per unit: ${batch['cost_per_unit']}")

        print("\n✅ Quality Checklist (Bread Loaf):")
        checklist = parser.get_sop_checklist("Bake Bread Loaf")
        if "error" not in checklist:
            print(f"  Quality Checks: {len(checklist['quality_checks'])}")
            for check in checklist['quality_checks']:
                print(f"    - [ ] {check}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

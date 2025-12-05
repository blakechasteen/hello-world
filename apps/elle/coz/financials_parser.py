"""
Financials Parser for Elle Core

Reads coz/financials.md and extracts:
- Product pricing and costs
- Gross margins
- Reinvestment percentages (15/10/5%)
- SOP ingredient cost population

Created: 2025-11-15
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class Product:
    """Product with pricing and costs"""
    name: str
    unit_price: float
    cost_per_unit: float
    gross_margin_percent: float
    notes: str = ""

    @property
    def profit_per_unit(self) -> float:
        """Profit per unit"""
        return self.unit_price - self.cost_per_unit

    @property
    def gross_margin_check(self) -> float:
        """Verify gross margin calculation"""
        if self.unit_price == 0:
            return 0.0
        return ((self.unit_price - self.cost_per_unit) / self.unit_price) * 100

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "unit_price": self.unit_price,
            "cost_per_unit": self.cost_per_unit,
            "profit_per_unit": self.profit_per_unit,
            "gross_margin_percent": self.gross_margin_percent,
            "gross_margin_calculated": self.gross_margin_check,
            "notes": self.notes,
        }


@dataclass
class ReinvestmentPlan:
    """Profit reinvestment allocation"""
    infrastructure_percent: float = 0.15  # 15%
    equipment_percent: float = 0.10  # 10%
    emergency_buffer_percent: float = 0.05  # 5%

    @property
    def total_reinvestment(self) -> float:
        """Total reinvestment percentage"""
        return self.infrastructure_percent + self.equipment_percent + self.emergency_buffer_percent

    def get_monthly_allocation(self, monthly_profit: float) -> Dict:
        """Calculate monthly allocation from profit"""
        return {
            "infrastructure": monthly_profit * self.infrastructure_percent,
            "equipment": monthly_profit * self.equipment_percent,
            "emergency_buffer": monthly_profit * self.emergency_buffer_percent,
            "total": monthly_profit * self.total_reinvestment,
            "remaining_profit": monthly_profit * (1 - self.total_reinvestment),
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "infrastructure_percent": self.infrastructure_percent,
            "equipment_percent": self.equipment_percent,
            "emergency_buffer_percent": self.emergency_buffer_percent,
            "total_reinvestment_percent": self.total_reinvestment,
        }


class FinancialsParser:
    """Parser for financials.md file"""

    def __init__(self, financials_path: str = "coz/financials.md"):
        self.financials_path = Path(financials_path)
        self.products: List[Product] = []
        self.reinvestment: Optional[ReinvestmentPlan] = None

    def parse(self) -> tuple[List[Product], ReinvestmentPlan]:
        """Parse financials.md file"""
        if not self.financials_path.exists():
            raise FileNotFoundError(f"Financials file not found: {self.financials_path}")

        with open(self.financials_path, 'r') as f:
            content = f.read()

        self.products = self._parse_products(content)
        self.reinvestment = self._parse_reinvestment(content)

        return self.products, self.reinvestment

    def _parse_products(self, content: str) -> List[Product]:
        """Parse product table from markdown"""
        products = []

        # Find the product table section
        table_start = content.find('| Product |')
        if table_start == -1:
            return products

        # Find the end of table (next ## heading or end of section)
        table_end = content.find('\n\n', table_start)
        if table_end == -1:
            table_end = content.find('\n#', table_start + 10)
        if table_end == -1:
            table_end = len(content)

        table_text = content[table_start:table_end]

        # Parse table rows
        lines = table_text.split('\n')
        for line in lines[2:]:  # Skip header and separator
            if not line.strip() or line.startswith('|---'):
                continue

            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 6:  # Need at least 6 parts (empty + 5 columns + empty)
                continue

            try:
                product_name = parts[1]
                unit_price = self._parse_currency(parts[2])
                cost_per_unit = self._parse_currency(parts[3])
                margin_str = parts[4].replace('%', '').strip()
                margin = float(margin_str) if margin_str else 0.0
                notes = parts[5] if len(parts) > 5 else ""

                products.append(Product(
                    name=product_name,
                    unit_price=unit_price,
                    cost_per_unit=cost_per_unit,
                    gross_margin_percent=margin,
                    notes=notes,
                ))
            except (ValueError, IndexError):
                continue

        return products

    def _parse_reinvestment(self, content: str) -> ReinvestmentPlan:
        """Parse reinvestment percentages from markdown"""
        reinvestment = ReinvestmentPlan()

        # Look for reinvestment section
        reinv_match = re.search(r'## Reinvestment Plan', content, re.IGNORECASE)
        if not reinv_match:
            return reinvestment

        reinv_section = content[reinv_match.start():]
        reinv_section = reinv_section[:reinv_section.find('\n##')] if '\n##' in reinv_section else reinv_section

        # Look for percentage patterns
        infra_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*.*infrastructure', reinv_section, re.IGNORECASE)
        if infra_match:
            reinvestment.infrastructure_percent = float(infra_match.group(1)) / 100

        equipment_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*.*[Ee]quipment', reinv_section)
        if equipment_match:
            reinvestment.equipment_percent = float(equipment_match.group(1)) / 100

        emergency_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*.*[Ee]mergency', reinv_section)
        if emergency_match:
            reinvestment.emergency_buffer_percent = float(emergency_match.group(1)) / 100

        return reinvestment

    def _parse_currency(self, value: str) -> float:
        """Parse currency string like $6 to float"""
        value = value.replace('$', '').strip()
        try:
            return float(value)
        except ValueError:
            return 0.0

    def get_product_by_name(self, name: str) -> Optional[Product]:
        """Get product by name"""
        for product in self.products:
            if product.name.lower() == name.lower():
                return product
        return None

    def get_total_revenue(self, units: Dict[str, int]) -> float:
        """Calculate total revenue given unit counts"""
        total = 0.0
        for product in self.products:
            if product.name in units:
                total += product.unit_price * units[product.name]
        return total

    def get_total_cost(self, units: Dict[str, int]) -> float:
        """Calculate total cost given unit counts"""
        total = 0.0
        for product in self.products:
            if product.name in units:
                total += product.cost_per_unit * units[product.name]
        return total

    def get_profit(self, units: Dict[str, int]) -> float:
        """Calculate profit given unit counts"""
        return self.get_total_revenue(units) - self.get_total_cost(units)

    def calculate_production_plan(self, target_profit: float) -> Dict:
        """Calculate units needed to reach target profit"""
        plan = {}

        for product in self.products:
            profit_per_unit = product.profit_per_unit
            if profit_per_unit > 0:
                units_needed = target_profit / profit_per_unit
                plan[product.name] = {
                    "units": units_needed,
                    "revenue": product.unit_price * units_needed,
                    "cost": product.cost_per_unit * units_needed,
                    "profit": profit_per_unit * units_needed,
                }

        return plan

    def get_summary(self) -> Dict:
        """Get financial summary"""
        avg_margin = (
            sum(p.gross_margin_percent for p in self.products) / len(self.products)
            if self.products
            else 0
        )

        return {
            "product_count": len(self.products),
            "products": [p.to_dict() for p in self.products],
            "average_margin": avg_margin,
            "reinvestment": self.reinvestment.to_dict() if self.reinvestment else {},
        }

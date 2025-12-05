"""
Inventory Parser for Elle Core

Reads coz/inventory.md and extracts:
- Materials, packaging, equipment
- Stock levels and reorder points
- Supplier information
- Inventory alerts

Created: 2025-11-15
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum


class ItemCategory(Enum):
    """Inventory item categories"""
    INGREDIENTS = "ingredients"
    PACKAGING = "packaging"
    EQUIPMENT = "equipment"
    TOOLS = "tools"
    SUPPLIES = "supplies"


@dataclass
class InventoryItem:
    """Inventory item with tracking"""
    name: str
    category: ItemCategory
    supplier: str = ""
    quantity: Optional[float] = None
    unit: str = ""
    reorder_point: Optional[float] = None
    last_ordered: Optional[datetime] = None
    cost_per_unit: Optional[float] = None
    notes: str = ""

    @property
    def needs_reorder(self) -> bool:
        """Check if item needs reordering"""
        if self.quantity is None or self.reorder_point is None:
            return False
        return self.quantity <= self.reorder_point

    @property
    def stock_level(self) -> str:
        """Get stock level status"""
        if self.quantity is None or self.reorder_point is None:
            return "unknown"

        if self.quantity <= self.reorder_point:
            return "critical"
        elif self.quantity <= self.reorder_point * 1.5:
            return "low"
        elif self.quantity >= self.reorder_point * 3:
            return "high"
        else:
            return "normal"

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "category": self.category.value,
            "supplier": self.supplier,
            "quantity": self.quantity,
            "unit": self.unit,
            "reorder_point": self.reorder_point,
            "stock_level": self.stock_level,
            "needs_reorder": self.needs_reorder,
            "last_ordered": self.last_ordered.isoformat() if self.last_ordered else None,
            "cost_per_unit": self.cost_per_unit,
            "notes": self.notes,
        }


class InventoryParser:
    """Parser for inventory.md file"""

    def __init__(self, inventory_path: str = "coz/inventory.md"):
        self.inventory_path = Path(inventory_path)
        self.items: List[InventoryItem] = []
        self.by_category: Dict[ItemCategory, List[InventoryItem]] = {}

    def parse(self) -> List[InventoryItem]:
        """Parse inventory.md file"""
        if not self.inventory_path.exists():
            raise FileNotFoundError(f"Inventory file not found: {self.inventory_path}")

        with open(self.inventory_path, 'r') as f:
            content = f.read()

        self.items = self._parse_inventory_sections(content)

        # Organize by category
        for item in self.items:
            if item.category not in self.by_category:
                self.by_category[item.category] = []
            self.by_category[item.category].append(item)

        return self.items

    def _parse_inventory_sections(self, content: str) -> List[InventoryItem]:
        """Parse inventory sections from markdown"""
        items = []

        # Find all level-2 headers (## Category)
        sections = re.split(r'^## ', content, flags=re.MULTILINE)[1:]

        for section in sections:
            lines = section.split('\n')
            if not lines:
                continue

            # First line is the category
            category_str = lines[0].strip()
            category = self._parse_category(category_str)

            # Parse items in this category
            section_items = self._parse_items_in_category(section, category)
            items.extend(section_items)

        return items

    def _parse_category(self, category_str: str) -> ItemCategory:
        """Parse category string"""
        category_str = category_str.lower()

        if 'ingredient' in category_str:
            return ItemCategory.INGREDIENTS
        elif 'packaging' in category_str:
            return ItemCategory.PACKAGING
        elif 'equipment' in category_str:
            return ItemCategory.EQUIPMENT
        elif 'tool' in category_str:
            return ItemCategory.TOOLS
        else:
            return ItemCategory.SUPPLIES

    def _parse_items_in_category(self, section: str, category: ItemCategory) -> List[InventoryItem]:
        """Parse items within a category section"""
        items = []

        # Look for bulleted items (- Item name)
        item_pattern = r'^- (.+?)(?=\n|$)'
        matches = re.finditer(item_pattern, section, re.MULTILINE)

        for match in matches:
            item_line = match.group(1).strip()

            # Parse item details
            item = self._parse_item_line(item_line, category)
            if item:
                items.append(item)

        return items

    def _parse_item_line(self, item_line: str, category: ItemCategory) -> Optional[InventoryItem]:
        """Parse a single item line"""
        if not item_line:
            return None

        name = item_line
        supplier = ""
        notes = ""

        # Extract supplier if present (after comma or dash)
        if ',' in item_line:
            parts = item_line.split(',', 1)
            name = parts[0].strip()
            supplier_info = parts[1].strip()

            # Extract supplier from various formats
            supplier_match = re.search(r'(?:from |at )?(.+?)(?:\s*—|$)', supplier_info)
            if supplier_match:
                supplier = supplier_match.group(1).strip()
                # Get rest as notes
                notes = supplier_info[supplier_match.end():].strip()

        # Try to extract quantity if present
        qty_pattern = r'(\d+(?:\.\d+)?)\s*(?:lbs?|oz|gallons?|liters?|units?|pieces?|bags?|boxes?)'
        qty_match = re.search(qty_pattern, name, re.IGNORECASE)
        quantity = None
        unit = ""

        if qty_match:
            quantity_str = qty_match.group(0)
            # Extract number and unit
            qty_parts = quantity_str.split()
            if qty_parts:
                try:
                    quantity = float(qty_parts[0])
                    if len(qty_parts) > 1:
                        unit = qty_parts[1]
                except ValueError:
                    pass

        # Default reorder points based on category
        reorder_point = self._get_default_reorder_point(category)

        return InventoryItem(
            name=name.strip(),
            category=category,
            supplier=supplier,
            quantity=quantity,
            unit=unit,
            reorder_point=reorder_point,
            notes=notes,
        )

    def _get_default_reorder_point(self, category: ItemCategory) -> float:
        """Get default reorder point based on category"""
        defaults = {
            ItemCategory.INGREDIENTS: 20.0,  # Low stock warning
            ItemCategory.PACKAGING: 50.0,
            ItemCategory.EQUIPMENT: 1.0,
            ItemCategory.TOOLS: 1.0,
            ItemCategory.SUPPLIES: 10.0,
        }
        return defaults.get(category, 10.0)

    def get_low_stock_items(self) -> List[InventoryItem]:
        """Get items that need reordering"""
        return [item for item in self.items if item.needs_reorder]

    def get_critical_stock_items(self) -> List[InventoryItem]:
        """Get critically low stock items"""
        critical = []
        for item in self.items:
            if item.quantity is not None and item.reorder_point is not None:
                if item.quantity <= item.reorder_point * 0.5:
                    critical.append(item)
        return critical

    def get_by_category(self, category: ItemCategory) -> List[InventoryItem]:
        """Get all items in a category"""
        return self.by_category.get(category, [])

    def get_by_supplier(self, supplier: str) -> List[InventoryItem]:
        """Get all items from a supplier"""
        return [item for item in self.items if supplier.lower() in item.supplier.lower()]

    def get_items_by_name(self, name: str) -> Optional[InventoryItem]:
        """Get item by name"""
        name_lower = name.lower()
        for item in self.items:
            if name_lower in item.name.lower():
                return item
        return None

    def generate_reorder_list(self) -> Dict:
        """Generate shopping list for low stock items"""
        low_stock = self.get_low_stock_items()

        # Group by supplier
        by_supplier = {}
        for item in low_stock:
            supplier = item.supplier or "Unknown Supplier"
            if supplier not in by_supplier:
                by_supplier[supplier] = []
            by_supplier[supplier].append(item)

        return {
            "total_items_to_reorder": len(low_stock),
            "by_supplier": {
                supplier: [item.to_dict() for item in items]
                for supplier, items in by_supplier.items()
            },
            "critical_items": [item.to_dict() for item in self.get_critical_stock_items()],
        }

    def update_item_stock(self, item_name: str, new_quantity: float) -> bool:
        """Update stock quantity for an item"""
        item = self.get_items_by_name(item_name)
        if item:
            item.quantity = new_quantity
            item.last_ordered = datetime.now()
            return True
        return False

    def get_inventory_summary(self) -> Dict:
        """Get inventory summary"""
        low_stock = self.get_low_stock_items()
        critical = self.get_critical_stock_items()

        by_stock_level = {
            "critical": [i.name for i in critical],
            "low": [i.name for i in low_stock if i not in critical],
            "normal": [i.name for i in self.items if not i.needs_reorder],
        }

        return {
            "total_items": len(self.items),
            "by_category": {
                cat.value: len(items) for cat, items in self.by_category.items()
            },
            "by_stock_level": {
                level: len(items) for level, items in by_stock_level.items()
            },
            "items_needing_reorder": len(low_stock),
            "critical_items": len(critical),
            "suppliers": list(set(i.supplier for i in self.items if i.supplier)),
        }

    def get_reorder_alerts(self) -> Dict:
        """Get reorder alerts"""
        low_stock = self.get_low_stock_items()
        critical = self.get_critical_stock_items()

        alerts = {
            "critical": {
                "count": len(critical),
                "items": [i.to_dict() for i in critical],
                "message": f"URGENT: {len(critical)} items critically low",
            },
            "low": {
                "count": len(low_stock) - len(critical),
                "items": [i.to_dict() for i in low_stock if i not in critical],
                "message": f"Reorder: {len(low_stock) - len(critical)} items low",
            },
        }

        return alerts

    def to_dict(self) -> List[Dict]:
        """Convert all items to dictionaries"""
        return [item.to_dict() for item in self.items]

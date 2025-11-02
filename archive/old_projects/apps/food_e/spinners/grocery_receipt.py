# -*- coding: utf-8 -*-
"""
Grocery Receipt Spinner
=======================
Food-e app spinner for ingesting grocery receipts with OCR.

Part of the food-e nutrition and food tracking application.
Extracts structured data from grocery receipt images.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import re
from datetime import datetime
from pathlib import Path
import sys

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from HoloLoom.spinning_wheel.image import ImageSpinner, ImageSpinnerConfig

try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    from dataclasses import field

    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)

logger = logging.getLogger(__name__)


@dataclass
class GroceryReceiptConfig(ImageSpinnerConfig):
    """Configuration for grocery receipt parsing."""
    extract_items: bool = True  # Parse line items
    extract_totals: bool = True  # Extract total, tax, etc.
    extract_merchant: bool = True  # Extract store/merchant info
    extract_date: bool = True  # Extract transaction date
    categorize_items: bool = True  # Categorize food items
    extract_nutrition_info: bool = False  # Try to extract nutrition data
    link_to_food_database: bool = False  # Link items to food database


class GroceryReceiptSpinner(ImageSpinner):
    """
    Specialized spinner for grocery receipts.

    Extracts structured data from receipt images for food-e app:
    - Merchant/store name
    - Transaction date/time
    - Line items with prices
    - Totals (subtotal, tax, total)
    - Payment method
    - Food categorization (produce, dairy, meat, pantry, etc.)
    """

    # Food categories for classification
    FOOD_CATEGORIES = {
        'produce': ['apple', 'banana', 'orange', 'lettuce', 'tomato', 'carrot', 'onion', 'potato'],
        'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'eggs'],
        'meat': ['chicken', 'beef', 'pork', 'fish', 'turkey', 'bacon', 'sausage'],
        'bakery': ['bread', 'bagel', 'muffin', 'croissant', 'roll', 'cake'],
        'pantry': ['rice', 'pasta', 'beans', 'flour', 'sugar', 'oil', 'cereal'],
        'beverages': ['juice', 'soda', 'water', 'coffee', 'tea', 'beer', 'wine'],
        'frozen': ['ice cream', 'frozen', 'pizza'],
        'snacks': ['chips', 'crackers', 'cookies', 'candy', 'nuts'],
        'condiments': ['sauce', 'ketchup', 'mustard', 'mayo', 'dressing', 'salsa'],
        'other': []
    }

    def __init__(self, config: GroceryReceiptConfig = None):
        if config is None:
            config = GroceryReceiptConfig()
        super().__init__(config)

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert receipt image → structured MemoryShards for food-e.

        Args:
            raw_data: Image data with optional hints:
                - 'image_path', 'image_data', or 'url': Image source
                - 'expected_merchant': Hint for merchant name
                - 'expected_date': Hint for transaction date
                - 'shopping_trip': Trip identifier

        Returns:
            MemoryShards with structured receipt data
        """
        # Get base OCR text
        shards = await super().spin(raw_data)
        if not shards:
            return shards

        # Parse receipt structure
        shard = shards[0]
        receipt_data = self._parse_receipt(shard.text, raw_data)

        # Update shard with parsed data
        shard.metadata['receipt'] = receipt_data
        shard.metadata['type'] = 'grocery_receipt'
        shard.metadata['app'] = 'food-e'

        # Extract entities (food items)
        shard.entities = [item['name'] for item in receipt_data.get('items', [])]

        # Extract motifs
        shard.motifs = ['grocery', 'food_purchase', receipt_data.get('merchant', 'unknown')]
        if receipt_data.get('categories'):
            shard.motifs.extend([f'category_{cat}' for cat in receipt_data['categories']])

        # Create structured summary
        shard.text = self._format_receipt_summary(receipt_data, shard.text)

        return shards

    def _parse_receipt(self, ocr_text: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse receipt structure from OCR text."""
        receipt_data = {
            'merchant': None,
            'date': None,
            'time': None,
            'items': [],
            'subtotal': None,
            'tax': None,
            'total': None,
            'payment_method': None,
            'categories': set()
        }

        lines = ocr_text.split('\n')

        # Extract merchant (usually first non-empty line)
        if self.config.extract_merchant:
            for line in lines[:5]:
                if line.strip() and len(line.strip()) > 3:
                    receipt_data['merchant'] = line.strip()
                    break

        # Extract date and time
        if self.config.extract_date:
            date_info = self._extract_date_time(lines[:15])
            receipt_data['date'] = date_info.get('date')
            receipt_data['time'] = date_info.get('time')

        # Extract line items
        if self.config.extract_items:
            items = self._extract_line_items(lines)
            receipt_data['items'] = items

            # Categorize items if enabled
            if self.config.categorize_items:
                for item in items:
                    category = self._categorize_item(item['name'])
                    item['category'] = category
                    receipt_data['categories'].add(category)

        # Extract totals
        if self.config.extract_totals:
            totals = self._extract_totals(lines)
            receipt_data.update(totals)

        # Convert set to list for JSON serialization
        receipt_data['categories'] = list(receipt_data['categories'])

        return receipt_data

    def _extract_date_time(self, lines: List[str]) -> Dict[str, Optional[str]]:
        """Extract date and time from receipt header."""
        result = {'date': None, 'time': None}

        # Date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # MM/DD/YYYY or DD-MM-YYYY
            r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',     # YYYY-MM-DD
            r'([A-Z][a-z]{2}\s+\d{1,2},?\s+\d{4})'  # Jan 15, 2024
        ]

        # Time patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*[AP]M)',  # 3:45 PM
            r'(\d{1,2}:\d{2}:\d{2})'     # 15:45:30
        ]

        for line in lines:
            # Check for date
            if not result['date']:
                for pattern in date_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        result['date'] = match.group(1)
                        break

            # Check for time
            if not result['time']:
                for pattern in time_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        result['time'] = match.group(1)
                        break

            if result['date'] and result['time']:
                break

        return result

    def _extract_line_items(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract individual items from receipt."""
        items = []

        # Pattern: item name followed by price
        # Variations: "BANANAS ORGANIC  $3.99" or "Milk 2.49" or "Bread    $1.99 T"
        item_patterns = [
            r'^(.+?)\s+\$?(\d+\.\d{2})\s*[TF]?\s*$',  # Item name + price
            r'^(.+?)\s+(\d+\.\d{2})\s*[TF]?\s*$',     # Without $
        ]

        # Skip patterns (totals, headers, etc.)
        skip_keywords = ['total', 'subtotal', 'tax', 'change', 'cash', 'credit', 'debit',
                        'balance', 'tender', 'paid', 'amount', 'items', 'qty', 'quantity']

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip if contains skip keywords
            if any(keyword in line.lower() for keyword in skip_keywords):
                continue

            # Try to match item pattern
            for pattern in item_patterns:
                match = re.search(pattern, line)
                if match:
                    item_name = match.group(1).strip()
                    price = float(match.group(2))

                    # Validate item name (not just numbers or symbols)
                    if len(item_name) > 2 and any(c.isalpha() for c in item_name):
                        items.append({
                            'name': item_name,
                            'price': price,
                            'raw_line': line
                        })
                        break

        return items

    def _extract_totals(self, lines: List[str]) -> Dict[str, Optional[float]]:
        """Extract subtotal, tax, and total from receipt."""
        totals = {
            'subtotal': None,
            'tax': None,
            'total': None
        }

        for line in lines:
            lower_line = line.lower()
            price_match = re.search(r'\$?(\d+\.\d{2})', line)

            if price_match:
                amount = float(price_match.group(1))

                if 'subtotal' in lower_line or 'sub total' in lower_line:
                    totals['subtotal'] = amount
                elif 'tax' in lower_line and 'total' not in lower_line:
                    totals['tax'] = amount
                elif 'total' in lower_line and 'subtotal' not in lower_line:
                    # Prefer "grand total" or just "total"
                    totals['total'] = amount

        return totals

    def _categorize_item(self, item_name: str) -> str:
        """Categorize food item based on name."""
        item_lower = item_name.lower()

        for category, keywords in self.FOOD_CATEGORIES.items():
            if category == 'other':
                continue

            for keyword in keywords:
                if keyword in item_lower:
                    return category

        return 'other'

    def _format_receipt_summary(self, receipt_data: Dict[str, Any], ocr_text: str) -> str:
        """Format structured receipt data as readable summary for food-e."""
        lines = []

        lines.append("=== Grocery Receipt ===")

        if receipt_data['merchant']:
            lines.append(f"Store: {receipt_data['merchant']}")
        if receipt_data['date']:
            lines.append(f"Date: {receipt_data['date']}")
        if receipt_data['time']:
            lines.append(f"Time: {receipt_data['time']}")

        if receipt_data['items']:
            lines.append(f"\nItems Purchased: {len(receipt_data['items'])}")

            # Group by category
            if receipt_data.get('categories'):
                lines.append("\nBy Category:")
                items_by_cat = {}
                for item in receipt_data['items']:
                    cat = item.get('category', 'other')
                    if cat not in items_by_cat:
                        items_by_cat[cat] = []
                    items_by_cat[cat].append(item)

                for cat in sorted(items_by_cat.keys()):
                    items = items_by_cat[cat]
                    cat_total = sum(item['price'] for item in items)
                    lines.append(f"\n{cat.upper()} (${cat_total:.2f}):")
                    for item in items:
                        lines.append(f"  - {item['name']}: ${item['price']:.2f}")
            else:
                # Simple list
                lines.append("")
                for item in receipt_data['items'][:20]:  # Limit to 20
                    lines.append(f"  - {item['name']}: ${item['price']:.2f}")

        if receipt_data['total']:
            lines.append(f"\nTotal: ${receipt_data['total']:.2f}")
        if receipt_data['tax']:
            lines.append(f"Tax: ${receipt_data['tax']:.2f}")

        lines.append("\n--- Original OCR ---")
        lines.append(ocr_text[:500])  # Truncate for brevity

        return '\n'.join(lines)


# Convenience factory
async def process_grocery_receipt(image_path: str, **kwargs) -> List[MemoryShard]:
    """Quick helper to process a grocery receipt image."""
    config = GroceryReceiptConfig(**kwargs)
    spinner = GroceryReceiptSpinner(config)

    raw_data = {
        'image_path': image_path,
        'episode': f"grocery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    raw_data.update(kwargs)

    return await spinner.spin(raw_data)
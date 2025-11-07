"""
HoloLoom COS - Business/Productivity Intent Parser

Philosophy: Voice-first, elegant, reliable
- Use regex patterns for high-confidence matching
- Fall back to LLM/HoloLoom for ambiguous cases
- ALWAYS trigger HITL for critical data (purchases, sales, time) when confidence < 0.85
"""

import re
from decimal import Decimal
from typing import Optional, Tuple, List
from datetime import datetime, timedelta

from .types import (
    EventType, EventSource, Category, ProductLine,
    ParsedIntent, Event, VerificationRequest
)


class BusinessIntentParser:
    """
    Parse natural language input into structured business events.

    Following HoloLoom conventions:
    - High confidence (>0.9): Regex patterns
    - Medium confidence (0.7-0.9): Heuristics
    - Low confidence (<0.7): LLM fallback + HITL
    """

    def __init__(self, hourly_rate: Decimal = Decimal('20.00')):
        self.hourly_rate = hourly_rate
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for fast matching"""

        # Time/task patterns
        self.task_patterns = [
            # "Baked bread for 3 hours"
            r'(?:baked?|made?|worked on|did)\s+(\w+)\s+for\s+(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)',
            # "3 hours baking bread"
            r'(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\s+(?:on\s+)?(\w+)',
            # "Started bread at 7am, finished at 10am"
            r'started\s+(\w+)\s+at\s+(\d+(?::\d+)?(?:am|pm)?),?\s*finished\s+at\s+(\d+(?::\d+)?(?:am|pm)?)',
        ]

        # Purchase patterns
        self.purchase_patterns = [
            # "Bought 50 pounds flour for $27"
            r'bought\s+(\d+(?:\.\d+)?)\s*(\w+)\s+(\w+)\s+(?:for|@)\s*\$?(\d+(?:\.\d+)?)',
            # "Bought 50 pounds flour at Costco for $27"
            r'bought\s+(\d+(?:\.\d+)?)\s*(\w+)\s+(\w+)\s+at\s+(\w+)\s+for\s*\$?(\d+(?:\.\d+)?)',
            # "Purchased flour at Costco, $27, 50 lb"
            r'purchased\s+(\w+)\s+(?:at|from)\s+(\w+)(?:.*?)\$?(\d+(?:\.\d+)?)(?:.*?)(\d+(?:\.\d+)?)\s*(\w+)',
            # "$27 on flour at Costco"
            r'\$?(\d+(?:\.\d+)?)\s+(?:on|for)\s+(\w+)(?:\s+at\s+(\w+))?',
            # "Paid electric bill $145"
            r'paid\s+(\w+(?:\s+\w+)?)\s+\$?(\d+(?:\.\d+)?)',
        ]

        # Sale patterns
        self.sale_patterns = [
            # "Sold 10 loaves for $60"
            r'sold\s+(\d+(?:\.\d+)?)\s+(\w+)\s+(?:for|@)\s*\$?(\d+(?:\.\d+)?)',
            # "Made $180 in bread sales"
            r'made\s+\$?(\d+(?:\.\d+)?)\s+(?:in\s+)?(\w+)\s+sales',
            # "$60 from bread today"
            r'\$?(\d+(?:\.\d+)?)\s+(?:from|in)\s+(\w+)',
        ]

        # Inventory patterns
        self.inventory_patterns = [
            # "Finished 12 loaves of bread"
            r'finished\s+(\d+(?:\.\d+)?)\s+(\w+)\s+(?:of\s+)?(\w+)',
            # "Made 12 loaves"
            r'made\s+(\d+(?:\.\d+)?)\s+(\w+)',
            # "Inventory: 20 loaves bread, 15 jars honey"
            r'inventory.*?(\d+)\s+(\w+)\s+(\w+)',
        ]

        # Product line mapping (for categorization)
        self.product_keywords = {
            'bread': ProductLine.BREAD,
            'loaves': ProductLine.BREAD,
            'loaf': ProductLine.BREAD,
            'sourdough': ProductLine.BREAD,
            'honey': ProductLine.HONEY,
            'jars': ProductLine.HONEY,
            'jar': ProductLine.HONEY,
            'seeds': ProductLine.SEEDS,
            'packets': ProductLine.SEEDS,
            'meal': ProductLine.MEAL_PREP,
            'soup': ProductLine.MEAL_PREP,
            'prep': ProductLine.MEAL_PREP,
            'quarts': ProductLine.MEAL_PREP,
            'brewing': ProductLine.MICRO_BREWING,
            'beer': ProductLine.MICRO_BREWING,
            'batch': ProductLine.MICRO_BREWING,
            'lettuce': ProductLine.LETTUCE,
            'heads': ProductLine.LETTUCE,
        }

        # Material/expense keywords
        self.material_keywords = {
            'flour': 'COGS_materials',
            'yeast': 'COGS_materials',
            'salt': 'COGS_materials',
            'ingredients': 'COGS_materials',
            'jars': 'COGS_packaging',
            'bags': 'COGS_packaging',
            'labels': 'COGS_packaging',
            'packaging': 'COGS_packaging',
            'electric': 'overhead_utilities',
            'electricity': 'overhead_utilities',
            'utilities': 'overhead_utilities',
            'water': 'overhead_utilities',
            'gas': 'overhead_utilities',
            'rent': 'overhead_rent',
            'equipment': 'overhead_equipment',
            'marketing': 'overhead_marketing',
        }

    def parse(self, text: str, source: EventSource = EventSource.MANUAL) -> Tuple[ParsedIntent, Optional[VerificationRequest]]:
        """
        Parse natural language input into structured event.

        Returns:
            (ParsedIntent, Optional[VerificationRequest])

        If VerificationRequest is not None, HITL confirmation needed.
        """
        text_lower = text.lower().strip()

        # Try each pattern type in priority order
        # 1. Task/time tracking
        result = self._try_parse_task(text_lower)
        if result:
            intent = result
            verification = self._check_verification_needed(intent, text)
            return intent, verification

        # 2. Purchase/expense
        result = self._try_parse_purchase(text_lower)
        if result:
            intent = result
            # ALWAYS check verification for purchases
            verification = self._check_verification_needed(intent, text)
            return intent, verification

        # 3. Sale/revenue
        result = self._try_parse_sale(text_lower)
        if result:
            intent = result
            verification = self._check_verification_needed(intent, text)
            return intent, verification

        # 4. Inventory
        result = self._try_parse_inventory(text_lower)
        if result:
            intent = result
            return intent, None

        # 5. Note/review (catch-all)
        intent = ParsedIntent(
            event_type=EventType.NOTE,
            confidence=Decimal('0.5'),
            explanation="Could not parse as structured event, treating as note"
        )
        return intent, None

    def _try_parse_task(self, text: str) -> Optional[ParsedIntent]:
        """Parse task/time tracking input"""
        for pattern in self.task_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()

                # Extract duration
                if 'hours' in pattern or 'hrs' in pattern:
                    # Pattern 1 or 2: has hours directly
                    if len(groups) == 3:
                        # Pattern 1: task, duration, unit
                        task = groups[0]
                        duration = Decimal(groups[1])
                        item = task
                    else:
                        # Pattern 2: duration, unit, task
                        duration = Decimal(groups[0])
                        item = groups[2]
                else:
                    # Pattern 3: start/end times
                    item = groups[0]
                    start = self._parse_time(groups[1])
                    end = self._parse_time(groups[2])
                    duration = self._calculate_duration(start, end)

                # Calculate labor cost
                labor_cost = duration * self.hourly_rate

                # Determine product line
                product_line = self._identify_product_line(text)

                # Extract output if mentioned
                output = self._extract_output(text)

                return ParsedIntent(
                    event_type=EventType.TASK,
                    confidence=Decimal('0.95'),
                    item=item,
                    quantity=duration,
                    unit='hours',
                    amount=-labor_cost,  # Negative = cost
                    category=Category.LABOR,
                    product_line=product_line,
                    parsed_data={
                        'duration_hours': float(duration),
                        'hourly_rate': float(self.hourly_rate),
                        'output': float(output) if output else None,
                    },
                    explanation=f"Parsed as {duration} hours of {item} work (${labor_cost} labor cost)"
                )

        return None

    def _try_parse_purchase(self, text: str) -> Optional[ParsedIntent]:
        """Parse purchase/expense input"""
        for pattern in self.purchase_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()

                # Extract amount (always has $)
                amount = None
                item = None
                quantity = None
                unit = None
                vendor = None

                # Pattern matching (varies by pattern)
                if 'bought' in pattern:
                    if len(groups) == 5:
                        # "Bought 50 pounds flour at Costco for $27"
                        quantity = Decimal(groups[0])
                        unit = groups[1]
                        item = groups[2]
                        vendor = groups[3]
                        amount = Decimal(groups[4])
                    else:
                        # "Bought 50 pounds flour for $27"
                        quantity = Decimal(groups[0])
                        unit = groups[1]
                        item = groups[2]
                        amount = Decimal(groups[3])
                elif 'purchased' in pattern:
                    # "Purchased flour at Costco, $27, 50 lb"
                    item = groups[0]
                    vendor = groups[1]
                    amount = Decimal(groups[2])
                    quantity = Decimal(groups[3]) if groups[3] else None
                    unit = groups[4] if len(groups) > 4 else None
                elif 'paid' in pattern:
                    # "Paid electric bill $145"
                    item = groups[0]
                    amount = Decimal(groups[1])
                else:
                    # "$27 on flour at Costco"
                    amount = Decimal(groups[0])
                    item = groups[1]
                    vendor = groups[2] if len(groups) > 2 else None

                # Determine category
                category = self._categorize_expense(item)

                # Determine product line
                product_line = self._identify_product_line(text)

                return ParsedIntent(
                    event_type=EventType.PURCHASE,
                    confidence=Decimal('0.90'),
                    item=item,
                    amount=-amount,  # Negative = money out
                    quantity=quantity,
                    unit=unit,
                    category=category,
                    product_line=product_line,
                    vendor=vendor,
                    parsed_data={
                        'vendor': vendor,
                        'unit_cost': float(amount / quantity) if quantity else None,
                    },
                    explanation=f"Purchased {item} for ${amount}" + (f" from {vendor}" if vendor else "")
                )

        return None

    def _try_parse_sale(self, text: str) -> Optional[ParsedIntent]:
        """Parse sale/revenue input"""
        for pattern in self.sale_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()

                if 'sold' in pattern:
                    # "Sold 10 loaves for $60"
                    quantity = Decimal(groups[0])
                    item = groups[1]
                    amount = Decimal(groups[2])
                    unit_price = amount / quantity
                elif 'made' in pattern:
                    # "Made $180 in bread sales"
                    amount = Decimal(groups[0])
                    item = groups[1]
                    quantity = None
                    unit_price = None
                else:
                    # "$60 from bread"
                    amount = Decimal(groups[0])
                    item = groups[1]
                    quantity = None
                    unit_price = None

                # Determine product line
                product_line = self._identify_product_line(text)

                return ParsedIntent(
                    event_type=EventType.SALE,
                    confidence=Decimal('0.92'),
                    item=item,
                    amount=amount,  # Positive = money in
                    quantity=-quantity if quantity else None,  # Negative = inventory out
                    category=Category.REVENUE,
                    product_line=product_line,
                    parsed_data={
                        'unit_price': float(unit_price) if unit_price else None,
                    },
                    explanation=f"Sale of {item} for ${amount}"
                )

        return None

    def _try_parse_inventory(self, text: str) -> Optional[ParsedIntent]:
        """Parse inventory adjustment input"""
        for pattern in self.inventory_patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()

                if 'finished' in pattern or 'made' in pattern:
                    quantity = Decimal(groups[0])
                    item = groups[1]
                    if len(groups) > 2:
                        item = groups[2]

                    product_line = self._identify_product_line(text)

                    return ParsedIntent(
                        event_type=EventType.INVENTORY,
                        confidence=Decimal('0.88'),
                        item=item,
                        quantity=quantity,  # Positive = added to inventory
                        product_line=product_line,
                        explanation=f"Added {quantity} {item} to inventory"
                    )

        return None

    def _identify_product_line(self, text: str) -> Optional[ProductLine]:
        """Identify product line from keywords"""
        text_lower = text.lower()
        for keyword, product in self.product_keywords.items():
            if keyword in text_lower:
                return product
        return None

    def _categorize_expense(self, item: str) -> Category:
        """Categorize expense by item type"""
        item_lower = item.lower()
        for keyword, category in self.material_keywords.items():
            if keyword in item_lower:
                return Category[category.upper()]
        return Category.OVERHEAD_OTHER

    def _extract_output(self, text: str) -> Optional[Decimal]:
        """Extract output quantity (e.g., 'made 12 loaves')"""
        match = re.search(r'(?:made|finished|produced)\s+(\d+(?:\.\d+)?)\s+\w+', text)
        if match:
            return Decimal(match.group(1))
        return None

    def _parse_time(self, time_str: str) -> datetime:
        """Parse time string to datetime (today assumed)"""
        # Simple parser for "7am", "10:30am", "14:00", etc.
        time_str = time_str.lower().strip()

        # Handle am/pm
        is_pm = 'pm' in time_str
        time_str = time_str.replace('am', '').replace('pm', '').strip()

        # Parse hour:minute
        if ':' in time_str:
            hour, minute = map(int, time_str.split(':'))
        else:
            hour = int(time_str)
            minute = 0

        # Convert to 24h
        if is_pm and hour < 12:
            hour += 12
        elif not is_pm and hour == 12:
            hour = 0

        today = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
        return today

    def _calculate_duration(self, start: datetime, end: datetime) -> Decimal:
        """Calculate duration in hours"""
        delta = end - start
        hours = Decimal(str(delta.total_seconds() / 3600))
        return hours.quantize(Decimal('0.01'))

    def _check_verification_needed(self, intent: ParsedIntent, raw_text: str) -> Optional[VerificationRequest]:
        """
        Check if HITL verification needed.

        Critical data (purchases, sales, time) require verification if confidence < 0.85
        """
        # Confidence threshold
        CONFIDENCE_THRESHOLD = Decimal('0.85')

        needs_verification = False
        reason = ""

        # Check by event type
        if intent.event_type == EventType.PURCHASE:
            if intent.confidence < CONFIDENCE_THRESHOLD:
                needs_verification = True
                reason = f"Purchase confidence {intent.confidence:.2f} < {CONFIDENCE_THRESHOLD}"
            elif intent.amount and abs(intent.amount) > 100:
                # Large purchases always need verification
                needs_verification = True
                reason = f"Large purchase: ${abs(intent.amount)}"

        elif intent.event_type == EventType.SALE:
            if intent.confidence < CONFIDENCE_THRESHOLD:
                needs_verification = True
                reason = f"Sale confidence {intent.confidence:.2f} < {CONFIDENCE_THRESHOLD}"

        elif intent.event_type == EventType.TASK:
            if intent.confidence < CONFIDENCE_THRESHOLD:
                needs_verification = True
                reason = f"Time log confidence {intent.confidence:.2f} < {CONFIDENCE_THRESHOLD}"
            elif intent.quantity and intent.quantity > 8:
                # >8 hours in single task seems unusual
                needs_verification = True
                reason = f"Unusually long task: {intent.quantity} hours"

        if not needs_verification:
            return None

        # Create verification request
        event = intent.to_event(raw_text)
        return VerificationRequest(
            event=event,
            reason=reason,
            suggestions=[]  # Could add suggestions from similar past events
        )


# Singleton instance
_parser_instance = None


def get_parser(hourly_rate: Decimal = Decimal('20.00')) -> BusinessIntentParser:
    """Get singleton parser instance"""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = BusinessIntentParser(hourly_rate)
    return _parser_instance


def parse_input(text: str, source: EventSource = EventSource.MANUAL) -> Tuple[ParsedIntent, Optional[VerificationRequest]]:
    """Convenience function to parse input"""
    parser = get_parser()
    return parser.parse(text, source)

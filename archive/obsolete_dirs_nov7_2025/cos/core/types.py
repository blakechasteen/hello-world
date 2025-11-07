"""
HoloLoom COS - Core Data Types
Following HoloLoom conventions (see HoloLoom/documentation/types.py)
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
import json


class EventType(Enum):
    """Event classification"""
    TASK = "task"              # Work performed (time tracking)
    SALE = "sale"              # Revenue transaction
    PURCHASE = "purchase"      # Expense/purchase
    INVENTORY = "inventory"    # Inventory adjustment
    NOTE = "note"              # Reflection, observation, insight
    PLAN = "plan"              # Planning (daily/weekly goals)
    GOAL = "goal"              # Long-term objective
    REVIEW = "review"          # Daily/weekly review


class EventSource(Enum):
    """Where did the event come from?"""
    VOICE = "voice"
    MANUAL = "manual"
    IMPORT = "import"
    AUTO = "auto"


class Category(Enum):
    """Business categorization for accounting"""
    # Revenue
    REVENUE = "revenue"

    # Costs
    COGS_MATERIALS = "COGS_materials"
    COGS_PACKAGING = "COGS_packaging"
    LABOR = "labor"
    OVERHEAD_UTILITIES = "overhead_utilities"
    OVERHEAD_RENT = "overhead_rent"
    OVERHEAD_EQUIPMENT = "overhead_equipment"
    OVERHEAD_MARKETING = "overhead_marketing"
    OVERHEAD_OTHER = "overhead_other"

    # Capital
    CAPEX = "capex"

    # Other
    OWNER_DRAW = "owner_draw"
    OWNER_INVESTMENT = "owner_investment"


class ProductLine(Enum):
    """Product lines from 90-day plan"""
    BREAD = "bread"
    MEAL_PREP = "meal_prep"
    MICRO_BREWING = "micro_brewing"
    LETTUCE = "lettuce"
    HONEY = "honey"
    SEEDS = "seeds"
    CANNING = "canning"
    TREE_NURSERY = "tree_nursery"
    MUSHROOMS = "mushrooms"
    OTHER = "other"


@dataclass
class Event:
    """
    Single event in the system.
    Everything is an event: tasks, sales, purchases, notes.

    Philosophy: Immutable event stream (append-only)
    Similar to HoloLoom's MemoryShard concept
    """
    # Core identity
    type: EventType
    raw_input: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Source and confidence
    source: EventSource = EventSource.MANUAL
    confidence: Optional[Decimal] = None
    verified: bool = False

    # Structured data
    parsed_data: Dict[str, Any] = field(default_factory=dict)

    # Business fields
    amount: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    unit: Optional[str] = None
    item: Optional[str] = None
    category: Optional[Category] = None
    product_line: Optional[ProductLine] = None

    # Relationships
    related_to: Optional[int] = None
    parent_goal: Optional[str] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    location: Optional[str] = None
    verification_note: Optional[str] = None

    # System fields
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'type': self.type.value,
            'raw_input': self.raw_input,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source.value,
            'confidence': float(self.confidence) if self.confidence else None,
            'verified': self.verified,
            'parsed_data': json.dumps(self.parsed_data),
            'amount': float(self.amount) if self.amount else None,
            'quantity': float(self.quantity) if self.quantity else None,
            'unit': self.unit,
            'item': self.item,
            'category': self.category.value if self.category else None,
            'product_line': self.product_line.value if self.product_line else None,
            'related_to': self.related_to,
            'parent_goal': self.parent_goal,
            'tags': ','.join(self.tags) if self.tags else None,
            'location': self.location,
            'verification_note': self.verification_note,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create from database row"""
        return cls(
            id=data.get('id'),
            type=EventType(data['type']),
            raw_input=data['raw_input'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=EventSource(data.get('source', 'manual')),
            confidence=Decimal(str(data['confidence'])) if data.get('confidence') else None,
            verified=bool(data.get('verified', False)),
            parsed_data=json.loads(data['parsed_data']) if data.get('parsed_data') else {},
            amount=Decimal(str(data['amount'])) if data.get('amount') else None,
            quantity=Decimal(str(data['quantity'])) if data.get('quantity') else None,
            unit=data.get('unit'),
            item=data.get('item'),
            category=Category(data['category']) if data.get('category') else None,
            product_line=ProductLine(data['product_line']) if data.get('product_line') else None,
            related_to=data.get('related_to'),
            parent_goal=data.get('parent_goal'),
            tags=data['tags'].split(',') if data.get('tags') else [],
            location=data.get('location'),
            verification_note=data.get('verification_note'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None,
        )

    def needs_verification(self) -> bool:
        """
        Should this event trigger HITL verification?

        Critical: Expenditures with low confidence
        """
        if self.verified:
            return False

        # All purchases should be verified if confidence < 0.85
        if self.type == EventType.PURCHASE:
            if self.confidence is None or self.confidence < Decimal('0.85'):
                return True

        return False

    def __repr__(self) -> str:
        """Human-readable representation"""
        parts = [f"{self.type.value}"]
        if self.item:
            parts.append(f"item={self.item}")
        if self.amount:
            parts.append(f"${self.amount}")
        if self.quantity and self.unit:
            parts.append(f"{self.quantity} {self.unit}")
        return f"Event({', '.join(parts)})"


@dataclass
class ParsedIntent:
    """
    Result of NLP parsing.
    Similar to HoloLoom's Features/DotPlasma concept.
    """
    event_type: EventType
    confidence: Decimal

    # Extracted fields
    item: Optional[str] = None
    amount: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    unit: Optional[str] = None
    category: Optional[Category] = None
    product_line: Optional[ProductLine] = None

    # Additional context
    vendor: Optional[str] = None
    duration: Optional[int] = None  # minutes
    output: Optional[Decimal] = None  # units produced

    # Full structured data
    parsed_data: Dict[str, Any] = field(default_factory=dict)

    # Explanation for HITL
    explanation: str = ""

    def to_event(self, raw_input: str, source: EventSource = EventSource.MANUAL) -> Event:
        """Convert parsed intent to Event"""
        return Event(
            type=self.event_type,
            raw_input=raw_input,
            source=source,
            confidence=self.confidence,
            parsed_data=self.parsed_data,
            amount=self.amount,
            quantity=self.quantity,
            unit=self.unit,
            item=self.item,
            category=self.category,
            product_line=self.product_line,
        )


@dataclass
class VerificationRequest:
    """
    HITL verification request for low-confidence events.
    Following HoloLoom's SafetyGuardrails pattern.
    """
    event: Event
    reason: str
    suggestions: List[Dict[str, Any]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"⚠️  VERIFICATION NEEDED\n"
            f"Reason: {self.reason}\n"
            f"Input: {self.event.raw_input}\n"
            f"Parsed: {self.event}\n"
            f"Confidence: {self.event.confidence:.2f}\n"
        )


@dataclass
class DailySummary:
    """Daily business summary (view)"""
    date: datetime
    revenue: Decimal = Decimal('0')
    cogs: Decimal = Decimal('0')
    labor_cost: Decimal = Decimal('0')
    overhead: Decimal = Decimal('0')
    hours_worked: Decimal = Decimal('0')
    profit: Decimal = Decimal('0')

    events_by_type: Dict[EventType, int] = field(default_factory=dict)
    revenue_by_product: Dict[ProductLine, Decimal] = field(default_factory=dict)

    def hourly_rate(self) -> Decimal:
        """Calculate effective hourly rate"""
        if self.hours_worked == 0:
            return Decimal('0')
        return self.profit / self.hours_worked

    def profit_margin(self) -> Decimal:
        """Calculate profit margin %"""
        if self.revenue == 0:
            return Decimal('0')
        return (self.profit / self.revenue) * 100


@dataclass
class WeeklySummary:
    """Weekly business summary (view)"""
    week: str  # ISO week format: "2025-W45"
    revenue: Decimal = Decimal('0')
    cogs: Decimal = Decimal('0')
    labor_cost: Decimal = Decimal('0')
    overhead: Decimal = Decimal('0')
    hours_worked: Decimal = Decimal('0')
    profit: Decimal = Decimal('0')

    active_product_lines: int = 0
    days_active: int = 0

    # Target from 90-day plan
    revenue_target: Optional[Decimal] = None

    def variance(self) -> Optional[Decimal]:
        """Revenue variance from target"""
        if self.revenue_target is None:
            return None
        return self.revenue - self.revenue_target

    def variance_pct(self) -> Optional[Decimal]:
        """Revenue variance % from target"""
        if self.revenue_target is None or self.revenue_target == 0:
            return None
        return (self.variance() / self.revenue_target) * 100


@dataclass
class ProductPerformance:
    """Product line performance metrics"""
    product_line: ProductLine
    revenue: Decimal = Decimal('0')
    material_cost: Decimal = Decimal('0')
    labor_cost: Decimal = Decimal('0')
    hours_worked: Decimal = Decimal('0')

    def gross_profit(self) -> Decimal:
        """Revenue - COGS"""
        return self.revenue - self.material_cost

    def net_profit(self) -> Decimal:
        """Revenue - COGS - Labor"""
        return self.revenue - self.material_cost - self.labor_cost

    def gross_margin(self) -> Decimal:
        """Gross profit %"""
        if self.revenue == 0:
            return Decimal('0')
        return (self.gross_profit() / self.revenue) * 100

    def net_margin(self) -> Decimal:
        """Net profit %"""
        if self.revenue == 0:
            return Decimal('0')
        return (self.net_profit() / self.revenue) * 100

    def hourly_rate(self) -> Decimal:
        """Revenue per hour"""
        if self.hours_worked == 0:
            return Decimal('0')
        return self.revenue / self.hours_worked

    def profit_per_hour(self) -> Decimal:
        """Profit per hour"""
        if self.hours_worked == 0:
            return Decimal('0')
        return self.net_profit() / self.hours_worked

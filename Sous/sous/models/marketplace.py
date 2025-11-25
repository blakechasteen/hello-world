#!/usr/bin/env python3
"""
Farm Marketplace Models for Phase 3

Represents products, orders, and transactions for direct farm-to-consumer sales
outside of CSA subscriptions. Supports both online ordering and farmers market sales.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import List, Dict, Optional, Set
from decimal import Decimal


class ProductCategory(Enum):
    """Categories of farm products"""
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    HERBS = "herbs"
    EGGS = "eggs"
    DAIRY = "dairy"
    MEAT = "meat"
    POULTRY = "poultry"
    HONEY = "honey"
    BAKED_GOODS = "baked_goods"
    PRESERVES = "preserves"
    VALUE_ADDED = "value_added"  # e.g., jams, sauces
    FLOWERS = "flowers"
    PLANTS = "plants"
    OTHER = "other"


class ProductAvailability(Enum):
    """Product availability status"""
    IN_STOCK = "in_stock"
    LOW_STOCK = "low_stock"
    OUT_OF_STOCK = "out_of_stock"
    SEASONAL = "seasonal"
    PRE_ORDER = "pre_order"


class OrderStatus(Enum):
    """Status of customer order"""
    DRAFT = "draft"  # Cart not checked out
    PENDING = "pending"  # Awaiting farmer confirmation
    CONFIRMED = "confirmed"  # Farmer confirmed, awaiting preparation
    READY = "ready"  # Ready for pickup/delivery
    IN_TRANSIT = "in_transit"  # Out for delivery
    COMPLETED = "completed"  # Delivered/picked up
    CANCELLED = "cancelled"  # Cancelled by customer or farmer
    REFUNDED = "refunded"  # Money returned


class PaymentMethod(Enum):
    """Payment methods"""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    CASH = "cash"
    CHECK = "check"
    VENMO = "venmo"
    PAYPAL = "paypal"
    ZELLE = "zelle"
    EBT_SNAP = "ebt_snap"  # Food assistance


class PaymentStatus(Enum):
    """Payment status"""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"
    FAILED = "failed"


class FulfillmentMethod(Enum):
    """How order is fulfilled"""
    FARM_PICKUP = "farm_pickup"
    DELIVERY = "delivery"
    FARMERS_MARKET_PICKUP = "farmers_market_pickup"
    MAIL = "mail"  # For value-added products


@dataclass
class Product:
    """
    A product available for purchase from a farm.

    Can be sold at farmers markets, farm stands, or online.
    """
    product_id: str
    farm_id: str

    # Basic info
    name: str
    description: str
    category: ProductCategory

    # Pricing
    price: Decimal
    unit: str  # "lb", "bunch", "dozen", "jar", etc.
    minimum_order_quantity: Decimal = Decimal("1.0")

    # Inventory
    quantity_available: Decimal = Decimal("0.0")
    availability: ProductAvailability = ProductAvailability.IN_STOCK

    # Details
    weight_oz: Optional[Decimal] = None
    dimensions: Optional[str] = None  # e.g., "12x6x4 inches"

    # Attributes
    organic: bool = False
    locally_grown: bool = True
    non_gmo: bool = False
    gluten_free: bool = False
    vegan: bool = False
    vegetarian: bool = False

    # Seasonality
    seasonal_availability: Set[str] = field(default_factory=set)  # e.g., {"spring", "summer"}
    harvest_date: Optional[date] = None

    # Sales
    featured: bool = False
    on_sale: bool = False
    sale_price: Optional[Decimal] = None
    total_sold: Decimal = Decimal("0.0")

    # Media
    image_urls: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    active: bool = True

    def get_effective_price(self) -> Decimal:
        """Get current selling price (sale or regular)"""
        if self.on_sale and self.sale_price:
            return self.sale_price
        return self.price

    def is_in_stock(self) -> bool:
        """Check if product is available for purchase"""
        return (self.active and
                self.availability in [ProductAvailability.IN_STOCK, ProductAvailability.LOW_STOCK] and
                self.quantity_available > 0)

    def can_fulfill(self, quantity: Decimal) -> bool:
        """Check if we can fulfill an order for this quantity"""
        return self.quantity_available >= quantity

    def reduce_stock(self, quantity: Decimal):
        """Reduce available stock (after sale)"""
        self.quantity_available -= quantity
        self.total_sold += quantity

        # Update availability status
        if self.quantity_available == 0:
            self.availability = ProductAvailability.OUT_OF_STOCK
        elif self.quantity_available < 5:
            self.availability = ProductAvailability.LOW_STOCK

        self.updated_at = datetime.now()

    def restock(self, quantity: Decimal):
        """Add stock"""
        self.quantity_available += quantity

        # Update availability
        if self.quantity_available > 5:
            self.availability = ProductAvailability.IN_STOCK
        elif self.quantity_available > 0:
            self.availability = ProductAvailability.LOW_STOCK

        self.updated_at = datetime.now()

    def calculate_subtotal(self, quantity: Decimal) -> Decimal:
        """Calculate subtotal for a quantity"""
        return self.get_effective_price() * quantity


@dataclass
class OrderItem:
    """Individual item in an order"""
    product_id: str
    product_name: str
    quantity: Decimal
    unit: str
    price_per_unit: Decimal
    subtotal: Decimal

    # Product snapshot (in case product changes)
    product_category: ProductCategory
    is_organic: bool = False
    harvest_date: Optional[date] = None


@dataclass
class Order:
    """
    Customer order from a farm.

    Represents a purchase transaction.
    """
    order_id: str
    customer_id: str  # User ID from Phase 1
    farm_id: str

    # Items
    items: List[OrderItem] = field(default_factory=list)

    # Pricing
    subtotal: Decimal = Decimal("0.0")
    tax: Decimal = Decimal("0.0")
    delivery_fee: Decimal = Decimal("0.0")
    discount: Decimal = Decimal("0.0")
    total: Decimal = Decimal("0.0")

    # Status
    status: OrderStatus = OrderStatus.PENDING
    status_history: List[Dict] = field(default_factory=list)

    # Fulfillment
    fulfillment_method: FulfillmentMethod = FulfillmentMethod.FARM_PICKUP
    pickup_location: Optional[str] = None
    delivery_address: Optional[str] = None
    fulfillment_date: Optional[date] = None
    fulfillment_time_window: Optional[str] = None  # e.g., "2-4pm"

    # Payment
    payment_method: Optional[PaymentMethod] = None
    payment_status: PaymentStatus = PaymentStatus.PENDING
    payment_transaction_id: Optional[str] = None

    # Communication
    customer_notes: str = ""
    farmer_notes: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Feedback
    customer_rating: Optional[int] = None  # 1-5 stars
    customer_review: str = ""

    def add_item(self, product: Product, quantity: Decimal):
        """Add product to order"""
        if not product.can_fulfill(quantity):
            raise ValueError(f"Insufficient stock for {product.name}")

        item = OrderItem(
            product_id=product.product_id,
            product_name=product.name,
            quantity=quantity,
            unit=product.unit,
            price_per_unit=product.get_effective_price(),
            subtotal=product.calculate_subtotal(quantity),
            product_category=product.category,
            is_organic=product.organic,
            harvest_date=product.harvest_date
        )

        self.items.append(item)
        self.recalculate_totals()

    def remove_item(self, product_id: str):
        """Remove product from order"""
        self.items = [item for item in self.items if item.product_id != product_id]
        self.recalculate_totals()

    def recalculate_totals(self):
        """Recalculate order totals"""
        self.subtotal = sum(item.subtotal for item in self.items)

        # Apply discount
        discounted_subtotal = self.subtotal - self.discount

        # Calculate tax (example: 8%)
        self.tax = discounted_subtotal * Decimal("0.08")

        # Total
        self.total = discounted_subtotal + self.tax + self.delivery_fee

    def update_status(self, new_status: OrderStatus, note: str = ""):
        """Update order status"""
        old_status = self.status
        self.status = new_status

        # Record status change
        self.status_history.append({
            "from": old_status.value,
            "to": new_status.value,
            "timestamp": datetime.now().isoformat(),
            "note": note
        })

        # Update timestamps
        if new_status == OrderStatus.CONFIRMED:
            self.confirmed_at = datetime.now()
        elif new_status == OrderStatus.COMPLETED:
            self.completed_at = datetime.now()

    def confirm(self):
        """Confirm order"""
        self.update_status(OrderStatus.CONFIRMED, "Order confirmed by farmer")

    def mark_ready(self):
        """Mark order as ready for pickup/delivery"""
        self.update_status(OrderStatus.READY, "Order ready")

    def complete(self):
        """Complete order"""
        self.update_status(OrderStatus.COMPLETED, "Order completed")

    def cancel(self, reason: str = ""):
        """Cancel order"""
        self.update_status(OrderStatus.CANCELLED, f"Cancelled: {reason}")

    def get_item_count(self) -> int:
        """Get total number of items"""
        return len(self.items)

    def get_total_items_quantity(self) -> Decimal:
        """Get total quantity across all items"""
        return sum(item.quantity for item in self.items)

    def is_complete(self) -> bool:
        """Check if order is completed"""
        return self.status == OrderStatus.COMPLETED

    def is_cancelled(self) -> bool:
        """Check if order is cancelled"""
        return self.status == OrderStatus.CANCELLED


@dataclass
class Transaction:
    """
    Financial transaction for an order.

    Records payment details.
    """
    transaction_id: str
    order_id: str
    customer_id: str
    farm_id: str

    # Payment details
    amount: Decimal
    payment_method: PaymentMethod
    status: PaymentStatus = PaymentStatus.PENDING

    # Payment processor info
    processor: str = ""  # e.g., "stripe", "square", "cash"
    processor_transaction_id: Optional[str] = None
    processor_fee: Decimal = Decimal("0.0")

    # Net to farm (after fees)
    net_amount: Decimal = Decimal("0.0")

    # Refund tracking
    refunded_amount: Decimal = Decimal("0.0")
    refund_reason: str = ""

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    refunded_at: Optional[datetime] = None

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Calculate net amount"""
        if self.net_amount == Decimal("0.0"):
            self.net_amount = self.amount - self.processor_fee

    def authorize(self, processor_transaction_id: str):
        """Authorize payment"""
        self.status = PaymentStatus.AUTHORIZED
        self.processor_transaction_id = processor_transaction_id

    def capture(self):
        """Capture authorized payment"""
        if self.status != PaymentStatus.AUTHORIZED:
            raise ValueError("Payment must be authorized before capture")

        self.status = PaymentStatus.CAPTURED
        self.processed_at = datetime.now()

    def refund(self, amount: Decimal, reason: str = ""):
        """Refund transaction"""
        if amount > (self.amount - self.refunded_amount):
            raise ValueError("Refund amount exceeds available amount")

        self.refunded_amount += amount
        self.refund_reason = reason
        self.refunded_at = datetime.now()

        # Update status
        if self.refunded_amount >= self.amount:
            self.status = PaymentStatus.REFUNDED

        # Recalculate net
        self.net_amount = self.amount - self.processor_fee - self.refunded_amount

    def fail(self, reason: str = ""):
        """Mark transaction as failed"""
        self.status = PaymentStatus.FAILED
        self.metadata["failure_reason"] = reason


@dataclass
class Cart:
    """
    Shopping cart (draft order).

    Temporary storage for items before checkout.
    """
    cart_id: str
    customer_id: str
    farm_id: str

    # Items
    items: List[OrderItem] = field(default_factory=list)

    # Totals (calculated)
    subtotal: Decimal = Decimal("0.0")
    estimated_tax: Decimal = Decimal("0.0")
    estimated_total: Decimal = Decimal("0.0")

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

    def add_item(self, product: Product, quantity: Decimal):
        """Add product to cart"""
        # Check if product already in cart
        for item in self.items:
            if item.product_id == product.product_id:
                item.quantity += quantity
                item.subtotal = product.calculate_subtotal(item.quantity)
                self.recalculate()
                return

        # Add new item
        item = OrderItem(
            product_id=product.product_id,
            product_name=product.name,
            quantity=quantity,
            unit=product.unit,
            price_per_unit=product.get_effective_price(),
            subtotal=product.calculate_subtotal(quantity),
            product_category=product.category,
            is_organic=product.organic
        )

        self.items.append(item)
        self.recalculate()

    def remove_item(self, product_id: str):
        """Remove item from cart"""
        self.items = [item for item in self.items if item.product_id != product_id]
        self.recalculate()

    def update_quantity(self, product_id: str, new_quantity: Decimal):
        """Update item quantity"""
        for item in self.items:
            if item.product_id == product_id:
                if new_quantity == 0:
                    self.remove_item(product_id)
                else:
                    item.quantity = new_quantity
                    # Recalculate subtotal (need Product to get current price)
                    self.recalculate()
                break

    def recalculate(self):
        """Recalculate cart totals"""
        self.subtotal = sum(item.subtotal for item in self.items)
        self.estimated_tax = self.subtotal * Decimal("0.08")
        self.estimated_total = self.subtotal + self.estimated_tax
        self.updated_at = datetime.now()

    def clear(self):
        """Empty cart"""
        self.items = []
        self.recalculate()

    def to_order(self, fulfillment_method: FulfillmentMethod) -> Order:
        """Convert cart to order"""
        order_id = f"order_{self.customer_id}_{datetime.now().timestamp()}"

        order = Order(
            order_id=order_id,
            customer_id=self.customer_id,
            farm_id=self.farm_id,
            items=self.items.copy(),
            fulfillment_method=fulfillment_method,
            status=OrderStatus.PENDING
        )

        order.recalculate_totals()
        return order

    def is_empty(self) -> bool:
        """Check if cart is empty"""
        return len(self.items) == 0

    def get_item_count(self) -> int:
        """Get number of unique items"""
        return len(self.items)


# ========================================================================
# Factory Functions
# ========================================================================

def create_product(
    farm_id: str,
    name: str,
    description: str,
    category: ProductCategory,
    price: Decimal,
    unit: str,
    quantity_available: Decimal,
    organic: bool = False
) -> Product:
    """Create a farm product"""
    product_id = f"product_{farm_id}_{name.lower().replace(' ', '_')}"

    return Product(
        product_id=product_id,
        farm_id=farm_id,
        name=name,
        description=description,
        category=category,
        price=price,
        unit=unit,
        quantity_available=quantity_available,
        organic=organic,
        availability=ProductAvailability.IN_STOCK if quantity_available > 0 else ProductAvailability.OUT_OF_STOCK
    )


def create_order(
    customer_id: str,
    farm_id: str,
    fulfillment_method: FulfillmentMethod = FulfillmentMethod.FARM_PICKUP
) -> Order:
    """Create an empty order"""
    order_id = f"order_{customer_id}_{farm_id}_{int(datetime.now().timestamp())}"

    return Order(
        order_id=order_id,
        customer_id=customer_id,
        farm_id=farm_id,
        fulfillment_method=fulfillment_method,
        status=OrderStatus.DRAFT
    )


def create_transaction(
    order: Order,
    payment_method: PaymentMethod,
    processor: str = "stripe"
) -> Transaction:
    """Create a transaction for an order"""
    transaction_id = f"txn_{order.order_id}_{int(datetime.now().timestamp())}"

    # Calculate processor fee (example: 2.9% + $0.30)
    processor_fee = (order.total * Decimal("0.029")) + Decimal("0.30")

    return Transaction(
        transaction_id=transaction_id,
        order_id=order.order_id,
        customer_id=order.customer_id,
        farm_id=order.farm_id,
        amount=order.total,
        payment_method=payment_method,
        processor=processor,
        processor_fee=processor_fee
    )


def create_cart(customer_id: str, farm_id: str) -> Cart:
    """Create a shopping cart"""
    cart_id = f"cart_{customer_id}_{farm_id}"

    # Cart expires after 24 hours
    expires_at = datetime.now() + timedelta(hours=24)

    return Cart(
        cart_id=cart_id,
        customer_id=customer_id,
        farm_id=farm_id,
        expires_at=expires_at
    )

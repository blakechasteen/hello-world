#!/usr/bin/env python3
"""
Farm Marketplace Service for Phase 3

Manages direct farm-to-consumer sales including:
- Product catalog management
- Order processing
- Inventory tracking
- Payment processing
- Sales analytics
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from collections import defaultdict

from sous.models.farm import Farm
from sous.models.marketplace import (
    Product, Order, Transaction, Cart, OrderItem,
    ProductCategory, ProductAvailability, OrderStatus,
    PaymentMethod, PaymentStatus, FulfillmentMethod,
    create_product, create_order, create_transaction, create_cart
)


@dataclass
class SalesAnalytics:
    """Sales analytics for a farm"""
    farm_id: str
    period_start: date
    period_end: date

    # Revenue
    total_revenue: Decimal = Decimal("0.0")
    net_revenue: Decimal = Decimal("0.0")  # After fees
    average_order_value: Decimal = Decimal("0.0")

    # Orders
    total_orders: int = 0
    completed_orders: int = 0
    cancelled_orders: int = 0
    average_order_size: Decimal = Decimal("0.0")

    # Products
    top_products: List[Dict] = None
    products_sold: int = 0

    # Customers
    unique_customers: int = 0
    repeat_customers: int = 0
    customer_retention_rate: float = 0.0

    # Trends
    daily_revenue: Dict[date, Decimal] = None

    def __post_init__(self):
        if self.top_products is None:
            self.top_products = []
        if self.daily_revenue is None:
            self.daily_revenue = {}


class FarmMarketplace:
    """
    Farm marketplace management system.

    Handles product listings, orders, inventory, and sales for farms.
    """

    def __init__(self):
        # Core data structures
        self.farms: Dict[str, Farm] = {}
        self.products: Dict[str, Product] = {}
        self.orders: Dict[str, Order] = {}
        self.transactions: Dict[str, Transaction] = {}
        self.carts: Dict[str, Cart] = {}

        # Indexes for efficient lookups
        self.products_by_farm: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_customer: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_farm: Dict[str, List[str]] = defaultdict(list)

    # ========================================================================
    # Farm Management
    # ========================================================================

    def register_farm(self, farm: Farm):
        """Register a farm in the marketplace"""
        self.farms[farm.org_id] = farm

    def get_farm(self, farm_id: str) -> Optional[Farm]:
        """Get farm by ID"""
        return self.farms.get(farm_id)

    def get_all_farms(self) -> List[Farm]:
        """Get all registered farms"""
        return list(self.farms.values())

    def search_farms(
        self,
        city: Optional[str] = None,
        state: Optional[str] = None,
        organic_only: bool = False,
        offers_delivery: bool = False
    ) -> List[Farm]:
        """Search farms by criteria"""
        results = list(self.farms.values())

        if city:
            results = [f for f in results if f.location.city.lower() == city.lower()]

        if state:
            results = [f for f in results if f.location.state.upper() == state.upper()]

        if organic_only:
            results = [f for f in results if f.is_certified_organic()]

        if offers_delivery:
            results = [f for f in results if f.delivery_radius_miles > 0]

        return results

    # ========================================================================
    # Product Management
    # ========================================================================

    def add_product(self, product: Product):
        """Add product to marketplace"""
        self.products[product.product_id] = product
        self.products_by_farm[product.farm_id].append(product.product_id)

    def get_product(self, product_id: str) -> Optional[Product]:
        """Get product by ID"""
        return self.products.get(product_id)

    def get_farm_products(
        self,
        farm_id: str,
        category: Optional[ProductCategory] = None,
        in_stock_only: bool = True
    ) -> List[Product]:
        """Get products from a specific farm"""
        product_ids = self.products_by_farm.get(farm_id, [])
        products = [self.products[pid] for pid in product_ids if pid in self.products]

        # Filter by category
        if category:
            products = [p for p in products if p.category == category]

        # Filter by stock
        if in_stock_only:
            products = [p for p in products if p.is_in_stock()]

        return products

    def search_products(
        self,
        query: str = "",
        category: Optional[ProductCategory] = None,
        organic_only: bool = False,
        max_price: Optional[Decimal] = None,
        farm_id: Optional[str] = None
    ) -> List[Product]:
        """Search products by criteria"""
        results = list(self.products.values())

        # Filter by query (name or description)
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.name.lower() or query_lower in p.description.lower()
            ]

        # Filter by farm
        if farm_id:
            results = [p for p in results if p.farm_id == farm_id]

        # Filter by category
        if category:
            results = [p for p in results if p.category == category]

        # Filter by organic
        if organic_only:
            results = [p for p in results if p.organic]

        # Filter by price
        if max_price:
            results = [p for p in results if p.get_effective_price() <= max_price]

        # Only active, in-stock products
        results = [p for p in results if p.active and p.is_in_stock()]

        return results

    def update_product_stock(self, product_id: str, new_quantity: Decimal):
        """Update product stock level"""
        product = self.get_product(product_id)
        if product:
            product.quantity_available = new_quantity
            product.updated_at = datetime.now()

            # Update availability status
            if new_quantity == 0:
                product.availability = ProductAvailability.OUT_OF_STOCK
            elif new_quantity < 5:
                product.availability = ProductAvailability.LOW_STOCK
            else:
                product.availability = ProductAvailability.IN_STOCK

    def get_low_stock_products(self, farm_id: str, threshold: Decimal = Decimal("5.0")) -> List[Product]:
        """Get products with low stock for a farm"""
        products = self.get_farm_products(farm_id, in_stock_only=False)
        return [p for p in products if p.quantity_available < threshold and p.active]

    # ========================================================================
    # Cart Management
    # ========================================================================

    def get_or_create_cart(self, customer_id: str, farm_id: str) -> Cart:
        """Get existing cart or create new one"""
        cart_id = f"cart_{customer_id}_{farm_id}"

        if cart_id in self.carts:
            cart = self.carts[cart_id]

            # Check if cart expired
            if cart.expires_at and datetime.now() > cart.expires_at:
                cart.clear()

            return cart

        # Create new cart
        cart = create_cart(customer_id, farm_id)
        self.carts[cart_id] = cart
        return cart

    def add_to_cart(self, customer_id: str, farm_id: str, product_id: str, quantity: Decimal):
        """Add product to customer's cart"""
        cart = self.get_or_create_cart(customer_id, farm_id)
        product = self.get_product(product_id)

        if not product:
            raise ValueError(f"Product {product_id} not found")

        if not product.can_fulfill(quantity):
            raise ValueError(f"Insufficient stock for {product.name}")

        cart.add_item(product, quantity)

    def remove_from_cart(self, customer_id: str, farm_id: str, product_id: str):
        """Remove product from cart"""
        cart = self.get_or_create_cart(customer_id, farm_id)
        cart.remove_item(product_id)

    def clear_cart(self, customer_id: str, farm_id: str):
        """Empty cart"""
        cart = self.get_or_create_cart(customer_id, farm_id)
        cart.clear()

    # ========================================================================
    # Order Processing
    # ========================================================================

    def create_order_from_cart(
        self,
        customer_id: str,
        farm_id: str,
        fulfillment_method: FulfillmentMethod,
        pickup_location: Optional[str] = None,
        delivery_address: Optional[str] = None
    ) -> Order:
        """Convert cart to order"""
        cart = self.get_or_create_cart(customer_id, farm_id)

        if cart.is_empty():
            raise ValueError("Cannot create order from empty cart")

        # Create order
        order = cart.to_order(fulfillment_method)
        order.pickup_location = pickup_location
        order.delivery_address = delivery_address

        # Store order
        self.orders[order.order_id] = order
        self.orders_by_customer[customer_id].append(order.order_id)
        self.orders_by_farm[farm_id].append(order.order_id)

        # Clear cart
        cart.clear()

        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_customer_orders(self, customer_id: str) -> List[Order]:
        """Get all orders for a customer"""
        order_ids = self.orders_by_customer.get(customer_id, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_farm_orders(
        self,
        farm_id: str,
        status: Optional[OrderStatus] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Order]:
        """Get orders for a farm with optional filters"""
        order_ids = self.orders_by_farm.get(farm_id, [])
        orders = [self.orders[oid] for oid in order_ids if oid in self.orders]

        # Filter by status
        if status:
            orders = [o for o in orders if o.status == status]

        # Filter by date range
        if start_date:
            orders = [o for o in orders if o.created_at.date() >= start_date]
        if end_date:
            orders = [o for o in orders if o.created_at.date() <= end_date]

        return orders

    def confirm_order(self, order_id: str, farmer_notes: str = ""):
        """Farmer confirms order"""
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        # Check inventory availability
        for item in order.items:
            product = self.get_product(item.product_id)
            if not product or not product.can_fulfill(item.quantity):
                raise ValueError(f"Insufficient stock for {item.product_name}")

        # Reduce inventory
        for item in order.items:
            product = self.get_product(item.product_id)
            product.reduce_stock(item.quantity)

        # Confirm order
        order.confirm()
        order.farmer_notes = farmer_notes

    def complete_order(self, order_id: str):
        """Mark order as completed"""
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        order.complete()

    def cancel_order(self, order_id: str, reason: str = "", refund: bool = True):
        """Cancel order and optionally refund"""
        order = self.get_order(order_id)
        if not order:
            raise ValueError(f"Order {order_id} not found")

        # Restore inventory if order was confirmed
        if order.status in [OrderStatus.CONFIRMED, OrderStatus.READY]:
            for item in order.items:
                product = self.get_product(item.product_id)
                if product:
                    product.restock(item.quantity)

        # Cancel order
        order.cancel(reason)

        # Process refund if needed
        if refund and order.payment_transaction_id:
            transaction = self.transactions.get(order.payment_transaction_id)
            if transaction and transaction.status == PaymentStatus.CAPTURED:
                transaction.refund(transaction.amount, reason)

    # ========================================================================
    # Payment Processing
    # ========================================================================

    def process_payment(
        self,
        order: Order,
        payment_method: PaymentMethod,
        processor: str = "stripe"
    ) -> Transaction:
        """Process payment for order"""
        # Create transaction
        transaction = create_transaction(order, payment_method, processor)

        # Simulate payment processing
        # In production, this would integrate with Stripe/Square/etc.
        transaction.authorize(f"auth_{transaction.transaction_id}")
        transaction.capture()

        # Update order
        order.payment_method = payment_method
        order.payment_status = PaymentStatus.CAPTURED
        order.payment_transaction_id = transaction.transaction_id

        # Store transaction
        self.transactions[transaction.transaction_id] = transaction

        return transaction

    # ========================================================================
    # Analytics
    # ========================================================================

    def get_sales_analytics(
        self,
        farm_id: str,
        start_date: date,
        end_date: date
    ) -> SalesAnalytics:
        """Generate sales analytics for a farm"""
        orders = self.get_farm_orders(farm_id, start_date=start_date, end_date=end_date)

        analytics = SalesAnalytics(
            farm_id=farm_id,
            period_start=start_date,
            period_end=end_date
        )

        # Filter completed orders
        completed_orders = [o for o in orders if o.status == OrderStatus.COMPLETED]
        cancelled_orders = [o for o in orders if o.status == OrderStatus.CANCELLED]

        analytics.total_orders = len(orders)
        analytics.completed_orders = len(completed_orders)
        analytics.cancelled_orders = len(cancelled_orders)

        if not completed_orders:
            return analytics

        # Revenue calculations
        total_revenue = sum(o.total for o in completed_orders)
        analytics.total_revenue = total_revenue

        # Calculate net revenue (after processor fees)
        transactions = [
            self.transactions.get(o.payment_transaction_id)
            for o in completed_orders
            if o.payment_transaction_id
        ]
        analytics.net_revenue = sum(
            t.net_amount for t in transactions if t
        )

        # Average order value
        analytics.average_order_value = total_revenue / Decimal(str(len(completed_orders)))

        # Average order size (items)
        total_items = sum(o.get_total_items_quantity() for o in completed_orders)
        analytics.average_order_size = total_items / Decimal(str(len(completed_orders)))

        # Top products
        product_sales = defaultdict(lambda: {"quantity": Decimal("0.0"), "revenue": Decimal("0.0")})
        for order in completed_orders:
            for item in order.items:
                product_sales[item.product_id]["quantity"] += item.quantity
                product_sales[item.product_id]["revenue"] += item.subtotal
                product_sales[item.product_id]["name"] = item.product_name

        # Sort by revenue
        top_products = sorted(
            [
                {
                    "product_id": pid,
                    "name": data["name"],
                    "quantity_sold": data["quantity"],
                    "revenue": data["revenue"]
                }
                for pid, data in product_sales.items()
            ],
            key=lambda x: x["revenue"],
            reverse=True
        )[:10]

        analytics.top_products = top_products
        analytics.products_sold = len(product_sales)

        # Customer analytics
        customers = set(o.customer_id for o in completed_orders)
        analytics.unique_customers = len(customers)

        # Daily revenue
        daily_revenue = defaultdict(lambda: Decimal("0.0"))
        for order in completed_orders:
            order_date = order.created_at.date()
            daily_revenue[order_date] += order.total

        analytics.daily_revenue = dict(daily_revenue)

        return analytics

    def get_inventory_value(self, farm_id: str) -> Decimal:
        """Calculate total value of farm's inventory"""
        products = self.get_farm_products(farm_id, in_stock_only=False)
        return sum(
            p.get_effective_price() * p.quantity_available
            for p in products
            if p.active
        )

    def get_popular_products(self, farm_id: str, limit: int = 10) -> List[Product]:
        """Get most popular products by sales volume"""
        products = self.get_farm_products(farm_id, in_stock_only=False)

        # Sort by total sold
        popular = sorted(
            [p for p in products if p.total_sold > 0],
            key=lambda p: p.total_sold,
            reverse=True
        )

        return popular[:limit]

    # ========================================================================
    # Recommendations
    # ========================================================================

    def recommend_products(
        self,
        customer_id: str,
        farm_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Product]:
        """Recommend products to customer based on purchase history"""
        # Get customer's past orders
        orders = self.get_customer_orders(customer_id)
        completed = [o for o in orders if o.status == OrderStatus.COMPLETED]

        if not completed:
            # New customer - recommend popular products
            if farm_id:
                return self.get_popular_products(farm_id, limit)
            else:
                # Return featured products from all farms
                all_products = list(self.products.values())
                featured = [p for p in all_products if p.featured and p.is_in_stock()]
                return featured[:limit]

        # Get categories customer has purchased
        purchased_categories = set()
        for order in completed:
            for item in order.items:
                purchased_categories.add(item.product_category)

        # Find products in similar categories
        recommendations = []
        for category in purchased_categories:
            products = self.search_products(category=category, farm_id=farm_id)
            recommendations.extend(products)

        # Sort by popularity
        recommendations.sort(key=lambda p: p.total_sold, reverse=True)

        # Remove duplicates and limit
        seen = set()
        unique_recommendations = []
        for p in recommendations:
            if p.product_id not in seen:
                seen.add(p.product_id)
                unique_recommendations.append(p)

        return unique_recommendations[:limit]


# ========================================================================
# Helper Functions
# ========================================================================

def calculate_delivery_fee(
    farm: Farm,
    delivery_address: str,
    base_fee: Decimal = Decimal("5.00"),
    per_mile_fee: Decimal = Decimal("1.00")
) -> Decimal:
    """Calculate delivery fee based on distance"""
    # In production, would geocode address and calculate distance
    # For now, simple flat fee
    return base_fee


def validate_ebt_transaction(order: Order) -> bool:
    """Validate EBT/SNAP eligibility for order items"""
    # EBT can only be used for food items, not prepared meals in some states
    for item in order.items:
        # Check category eligibility
        if item.product_category in [ProductCategory.FLOWERS, ProductCategory.PLANTS]:
            return False

    return True

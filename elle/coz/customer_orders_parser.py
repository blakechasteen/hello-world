"""
Customer Orders Parser for COZ System

Tracks customer orders, revenue pipeline, fulfillment schedule, and customer patterns.
Enables revenue forecasting and order prioritization.

Part of COZ Expansion Phase 2 (Orders & Intelligence)
Created: 2025-11-21
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict


@dataclass
class CustomerOrder:
    """Single customer order entry"""
    order_id: str
    customer_name: str
    order_date: datetime
    due_date: datetime
    status: str  # Fulfilled, In Progress, Pending, Overdue
    products: List[str]
    quantities: List[int]
    total_price: float
    notes: str

    # Calculated fields
    days_until_due: int = 0
    is_overdue: bool = False
    fulfillment_priority: int = 0  # 1=Critical, 2=High, 3=Medium, 4=Low

    def __post_init__(self):
        """Calculate derived fields"""
        today = datetime.now().date()

        # Days until due (negative if overdue)
        self.days_until_due = (self.due_date.date() - today).days

        # Is overdue check
        self.is_overdue = self.due_date.date() < today and self.status != "Fulfilled"

        # Fulfillment priority (1=Critical, 2=High, 3=Medium, 4=Low)
        if self.is_overdue:
            self.fulfillment_priority = 1  # Critical - already overdue
        elif self.days_until_due <= 1:
            self.fulfillment_priority = 1  # Critical - due today/tomorrow
        elif self.days_until_due <= 3:
            self.fulfillment_priority = 2  # High - due within 3 days
        elif self.days_until_due <= 7:
            self.fulfillment_priority = 3  # Medium - due within 1 week
        else:
            self.fulfillment_priority = 4  # Low - due later

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "order_id": self.order_id,
            "customer_name": self.customer_name,
            "order_date": self.order_date.isoformat(),
            "due_date": self.due_date.isoformat(),
            "status": self.status,
            "products": self.products,
            "quantities": self.quantities,
            "total_price": round(self.total_price, 2),
            "notes": self.notes,
            "days_until_due": self.days_until_due,
            "is_overdue": self.is_overdue,
            "fulfillment_priority": self.fulfillment_priority,
        }


class CustomerOrdersParser:
    """Parser for customer_orders.csv file"""

    def __init__(self, file_path: str = "coz/customer_orders.csv"):
        self.file_path = Path(file_path)
        self.orders: List[CustomerOrder] = []
        self.orders_by_id: Dict[str, CustomerOrder] = {}
        self.orders_by_customer: Dict[str, List[CustomerOrder]] = defaultdict(list)
        self.orders_by_status: Dict[str, List[CustomerOrder]] = defaultdict(list)

    def parse(self) -> List[CustomerOrder]:
        """Parse customer_orders.csv file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Customer orders file not found: {self.file_path}")

        self.orders = []
        self.orders_by_id = {}
        self.orders_by_customer = defaultdict(list)
        self.orders_by_status = defaultdict(list)

        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse dates
                    order_date = datetime.strptime(row['Order Date'], '%Y-%m-%d')
                    due_date = datetime.strptime(row['Due Date'], '%Y-%m-%d')

                    # Parse products (comma-separated list)
                    products = [p.strip() for p in row['Products'].split(',')]

                    # Parse quantities (comma-separated integers)
                    quantities = [int(q.strip()) for q in row['Quantities'].split(',')]

                    # Parse price
                    total_price = float(row['Total Price'])

                    # Create order
                    order = CustomerOrder(
                        order_id=row['Order ID'].strip(),
                        customer_name=row['Customer Name'].strip(),
                        order_date=order_date,
                        due_date=due_date,
                        status=row['Status'].strip(),
                        products=products,
                        quantities=quantities,
                        total_price=total_price,
                        notes=row.get('Notes', '').strip()
                    )

                    self.orders.append(order)
                    self.orders_by_id[order.order_id] = order
                    self.orders_by_customer[order.customer_name].append(order)
                    self.orders_by_status[order.status].append(order)

                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed row in {self.file_path}: {e}")
                    continue

        return self.orders

    def get_pending_orders(self) -> List[CustomerOrder]:
        """Get all pending orders (not fulfilled)"""
        return [
            order for order in self.orders
            if order.status in ["In Progress", "Pending", "Overdue"]
        ]

    def get_revenue_pipeline(self) -> Dict:
        """Get revenue breakdown by order status"""
        pipeline = {
            "Fulfilled": 0.0,
            "In Progress": 0.0,
            "Pending": 0.0,
            "Overdue": 0.0,
            "total": 0.0,
            "total_pending": 0.0,  # In Progress + Pending + Overdue
        }

        for order in self.orders:
            pipeline[order.status] += order.total_price
            pipeline["total"] += order.total_price

            if order.status in ["In Progress", "Pending", "Overdue"]:
                pipeline["total_pending"] += order.total_price

        # Round values
        for key in pipeline:
            pipeline[key] = round(pipeline[key], 2)

        return pipeline

    def get_due_this_week(self) -> List[CustomerOrder]:
        """Get orders due within the next 7 days (not fulfilled)"""
        week_orders = [
            order for order in self.orders
            if order.days_until_due <= 7
            and order.status != "Fulfilled"
        ]

        # Sort by due date (soonest first)
        week_orders.sort(key=lambda o: o.due_date)

        return week_orders

    def get_overdue_orders(self) -> List[CustomerOrder]:
        """Get all overdue orders (past due date, not fulfilled)"""
        overdue = [
            order for order in self.orders
            if order.is_overdue
        ]

        # Sort by how overdue (most overdue first)
        overdue.sort(key=lambda o: o.days_until_due)

        return overdue

    def get_customer_summary(self) -> Dict[str, Dict]:
        """Get summary statistics per customer"""
        summary = {}

        for customer, orders in self.orders_by_customer.items():
            total_orders = len(orders)
            fulfilled = len([o for o in orders if o.status == "Fulfilled"])
            total_revenue = sum(o.total_price for o in orders)
            avg_order_value = total_revenue / total_orders if total_orders > 0 else 0.0

            summary[customer] = {
                "total_orders": total_orders,
                "fulfilled_orders": fulfilled,
                "pending_orders": total_orders - fulfilled,
                "total_revenue": round(total_revenue, 2),
                "avg_order_value": round(avg_order_value, 2),
                "fulfillment_rate": round(fulfilled / total_orders, 2) if total_orders > 0 else 0.0,
            }

        return summary

    def get_fulfillment_schedule(self) -> Dict[str, List[CustomerOrder]]:
        """Get orders grouped by fulfillment priority"""
        schedule = {
            "Critical (Overdue/Due Today)": [],
            "High (Due in 2-3 Days)": [],
            "Medium (Due This Week)": [],
            "Low (Due Later)": [],
        }

        for order in self.orders:
            if order.status == "Fulfilled":
                continue  # Skip fulfilled orders

            if order.fulfillment_priority == 1:
                schedule["Critical (Overdue/Due Today)"].append(order)
            elif order.fulfillment_priority == 2:
                schedule["High (Due in 2-3 Days)"].append(order)
            elif order.fulfillment_priority == 3:
                schedule["Medium (Due This Week)"].append(order)
            else:
                schedule["Low (Due Later)"].append(order)

        # Sort each priority group by due date
        for priority in schedule:
            schedule[priority].sort(key=lambda o: o.due_date)

        return schedule

    def get_revenue_forecast(self, days: int = 30) -> Dict:
        """Forecast revenue for next N days based on pending orders"""
        today = datetime.now().date()
        forecast_end = today + timedelta(days=days)

        # Revenue breakdown by week
        weeks = []
        current_week_start = today

        while current_week_start < forecast_end:
            week_end = current_week_start + timedelta(days=7)

            # Find orders due in this week
            week_orders = [
                order for order in self.orders
                if order.status != "Fulfilled"
                and current_week_start <= order.due_date.date() < week_end
            ]

            week_revenue = sum(o.total_price for o in week_orders)

            weeks.append({
                "week_start": current_week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "orders_count": len(week_orders),
                "expected_revenue": round(week_revenue, 2),
            })

            current_week_start = week_end

        # Total forecast
        total_forecast = sum(w["expected_revenue"] for w in weeks)

        return {
            "forecast_period_days": days,
            "total_expected_revenue": round(total_forecast, 2),
            "weekly_breakdown": weeks,
        }

    def get_product_demand(self) -> Dict[str, Dict]:
        """Get product demand statistics across all orders"""
        product_stats = defaultdict(lambda: {"quantity": 0, "orders": 0, "revenue": 0.0})

        for order in self.orders:
            for product, quantity in zip(order.products, order.quantities):
                product_stats[product]["quantity"] += quantity
                product_stats[product]["orders"] += 1
                # Estimate revenue per product (assuming even split)
                product_revenue = order.total_price / len(order.products)
                product_stats[product]["revenue"] += product_revenue

        # Round revenue values
        for product in product_stats:
            product_stats[product]["revenue"] = round(product_stats[product]["revenue"], 2)

        return dict(product_stats)

    def get_by_order_id(self, order_id: str) -> Optional[CustomerOrder]:
        """Get order by ID"""
        return self.orders_by_id.get(order_id)

    def get_orders_by_customer(self, customer_name: str) -> List[CustomerOrder]:
        """Get all orders for a specific customer"""
        return self.orders_by_customer.get(customer_name, [])


if __name__ == "__main__":
    # Demo usage
    parser = CustomerOrdersParser()

    try:
        orders = parser.parse()

        print(f"✅ Parsed {len(orders)} customer orders")
        print("\n💰 Revenue Pipeline:")
        pipeline = parser.get_revenue_pipeline()
        for status, revenue in pipeline.items():
            print(f"  {status}: ${revenue}")

        print("\n📅 Orders Due This Week:")
        week_orders = parser.get_due_this_week()
        for order in week_orders:
            print(f"  - {order.order_id}: {order.customer_name} (Due: {order.due_date.date()}, Days: {order.days_until_due})")

        print("\n⚠️ Overdue Orders:")
        overdue = parser.get_overdue_orders()
        for order in overdue:
            print(f"  - {order.order_id}: {order.customer_name} ({abs(order.days_until_due)} days overdue)")

        print("\n👥 Customer Summary:")
        customer_summary = parser.get_customer_summary()
        for customer, stats in list(customer_summary.items())[:5]:
            print(f"  - {customer}: {stats['total_orders']} orders, ${stats['total_revenue']} total")

        print("\n📊 Product Demand:")
        product_demand = parser.get_product_demand()
        for product, stats in sorted(product_demand.items(), key=lambda x: x[1]["quantity"], reverse=True):
            print(f"  - {product}: {stats['quantity']} units across {stats['orders']} orders (${stats['revenue']})")

        print("\n📈 30-Day Revenue Forecast:")
        forecast = parser.get_revenue_forecast(days=30)
        print(f"  Total Expected: ${forecast['total_expected_revenue']}")
        for week in forecast['weekly_breakdown']:
            print(f"  - {week['week_start']}: ${week['expected_revenue']} ({week['orders_count']} orders)")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

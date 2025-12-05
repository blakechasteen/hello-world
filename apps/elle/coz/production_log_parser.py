"""
Production Log Parser for COZ System

Tracks production output, sales, waste, and inventory levels.
Enables waste reduction, production optimization, and demand forecasting.

Part of COZ Expansion Phase 2 (Orders & Intelligence)
Created: 2025-11-21
"""

import csv
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
from collections import defaultdict


@dataclass
class ProductionEntry:
    """Single production log entry"""
    date: datetime
    product: str
    quantity_produced: int
    quantity_sold: int
    quantity_wasted: int
    waste_reason: str
    notes: str

    # Calculated fields
    waste_percent: float = 0.0
    sellthrough_rate: float = 0.0
    remaining_inventory: int = 0

    def __post_init__(self):
        """Calculate derived fields"""
        # Waste percentage
        if self.quantity_produced > 0:
            self.waste_percent = (self.quantity_wasted / self.quantity_produced) * 100
        else:
            self.waste_percent = 0.0

        # Sellthrough rate (sold / produced)
        if self.quantity_produced > 0:
            self.sellthrough_rate = (self.quantity_sold / self.quantity_produced) * 100
        else:
            self.sellthrough_rate = 0.0

        # Remaining inventory
        self.remaining_inventory = self.quantity_produced - self.quantity_sold - self.quantity_wasted

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "date": self.date.isoformat(),
            "product": self.product,
            "quantity_produced": self.quantity_produced,
            "quantity_sold": self.quantity_sold,
            "quantity_wasted": self.quantity_wasted,
            "waste_reason": self.waste_reason,
            "notes": self.notes,
            "waste_percent": round(self.waste_percent, 1),
            "sellthrough_rate": round(self.sellthrough_rate, 1),
            "remaining_inventory": self.remaining_inventory,
        }


class ProductionLogParser:
    """Parser for production_log.csv file"""

    def __init__(self, file_path: str = "coz/production_log.csv"):
        self.file_path = Path(file_path)
        self.entries: List[ProductionEntry] = []
        self.entries_by_product: Dict[str, List[ProductionEntry]] = defaultdict(list)
        self.entries_by_date: Dict[datetime, List[ProductionEntry]] = defaultdict(list)

    def parse(self) -> List[ProductionEntry]:
        """Parse production_log.csv file"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Production log file not found: {self.file_path}")

        self.entries = []
        self.entries_by_product = defaultdict(list)
        self.entries_by_date = defaultdict(list)

        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # Parse date
                    date = datetime.strptime(row['Date'], '%Y-%m-%d')

                    # Parse quantities
                    quantity_produced = int(row['Quantity Produced'])
                    quantity_sold = int(row['Quantity Sold'])
                    quantity_wasted = int(row['Quantity Wasted'])

                    # Create entry
                    entry = ProductionEntry(
                        date=date,
                        product=row['Product'].strip(),
                        quantity_produced=quantity_produced,
                        quantity_sold=quantity_sold,
                        quantity_wasted=quantity_wasted,
                        waste_reason=row.get('Waste Reason', '').strip(),
                        notes=row.get('Notes', '').strip()
                    )

                    self.entries.append(entry)
                    self.entries_by_product[entry.product].append(entry)
                    self.entries_by_date[date.date()].append(entry)

                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping malformed row in {self.file_path}: {e}")
                    continue

        return self.entries

    def get_production_summary(self) -> Dict:
        """Get overall production summary"""
        if not self.entries:
            return {
                "total_produced": 0,
                "total_sold": 0,
                "total_wasted": 0,
                "avg_waste_percent": 0.0,
                "avg_sellthrough_rate": 0.0,
                "entries_count": 0,
            }

        total_produced = sum(e.quantity_produced for e in self.entries)
        total_sold = sum(e.quantity_sold for e in self.entries)
        total_wasted = sum(e.quantity_wasted for e in self.entries)

        avg_waste_percent = (total_wasted / total_produced * 100) if total_produced > 0 else 0.0
        avg_sellthrough_rate = (total_sold / total_produced * 100) if total_produced > 0 else 0.0

        return {
            "total_produced": total_produced,
            "total_sold": total_sold,
            "total_wasted": total_wasted,
            "avg_waste_percent": round(avg_waste_percent, 1),
            "avg_sellthrough_rate": round(avg_sellthrough_rate, 1),
            "entries_count": len(self.entries),
            "remaining_inventory": total_produced - total_sold - total_wasted,
        }

    def get_waste_analysis(self) -> Dict:
        """Analyze waste patterns"""
        if not self.entries:
            return {"total_waste": 0, "waste_reasons": {}, "high_waste_products": []}

        total_waste = sum(e.quantity_wasted for e in self.entries)

        # Waste reasons breakdown
        waste_reasons = defaultdict(int)
        for entry in self.entries:
            if entry.waste_reason:
                waste_reasons[entry.waste_reason] += entry.quantity_wasted

        # High waste products (>20% waste rate)
        high_waste_products = []
        for product, entries in self.entries_by_product.items():
            total_prod = sum(e.quantity_produced for e in entries)
            total_waste = sum(e.quantity_wasted for e in entries)
            waste_rate = (total_waste / total_prod * 100) if total_prod > 0 else 0.0

            if waste_rate > 20:
                high_waste_products.append({
                    "product": product,
                    "waste_rate": round(waste_rate, 1),
                    "total_wasted": total_waste,
                })

        # Sort by waste rate
        high_waste_products.sort(key=lambda x: x['waste_rate'], reverse=True)

        return {
            "total_waste": total_waste,
            "waste_reasons": dict(waste_reasons),
            "high_waste_products": high_waste_products,
        }

    def get_product_performance(self) -> Dict[str, Dict]:
        """Get performance metrics per product"""
        performance = {}

        for product, entries in self.entries_by_product.items():
            total_produced = sum(e.quantity_produced for e in entries)
            total_sold = sum(e.quantity_sold for e in entries)
            total_wasted = sum(e.quantity_wasted for e in entries)

            avg_waste = (total_wasted / total_produced * 100) if total_produced > 0 else 0.0
            avg_sellthrough = (total_sold / total_produced * 100) if total_produced > 0 else 0.0

            performance[product] = {
                "production_runs": len(entries),
                "total_produced": total_produced,
                "total_sold": total_sold,
                "total_wasted": total_wasted,
                "avg_waste_percent": round(avg_waste, 1),
                "avg_sellthrough_rate": round(avg_sellthrough, 1),
                "avg_produced_per_run": round(total_produced / len(entries), 1),
            }

        return performance

    def get_overproduction_alerts(self, sellthrough_threshold: float = 80.0) -> List[Dict]:
        """Detect products with low sellthrough (potential overproduction)"""
        alerts = []

        for product, entries in self.entries_by_product.items():
            # Check last 3 entries (recent production runs)
            recent = entries[-3:]

            for entry in recent:
                if entry.sellthrough_rate < sellthrough_threshold:
                    alerts.append({
                        "date": entry.date.date().isoformat(),
                        "product": product,
                        "produced": entry.quantity_produced,
                        "sold": entry.quantity_sold,
                        "sellthrough_rate": round(entry.sellthrough_rate, 1),
                        "unsold": entry.remaining_inventory + entry.quantity_wasted,
                        "reason": entry.waste_reason or "Low demand",
                    })

        # Sort by date (most recent first)
        alerts.sort(key=lambda x: x['date'], reverse=True)

        return alerts

    def get_inventory_status(self) -> Dict[str, int]:
        """Get current inventory levels per product"""
        inventory = defaultdict(int)

        for entry in self.entries:
            inventory[entry.product] += entry.remaining_inventory

        return dict(inventory)

    def get_sellthrough_rates(self) -> Dict[str, float]:
        """Get average sellthrough rate per product"""
        sellthrough = {}

        for product, entries in self.entries_by_product.items():
            total_produced = sum(e.quantity_produced for e in entries)
            total_sold = sum(e.quantity_sold for e in entries)

            rate = (total_sold / total_produced * 100) if total_produced > 0 else 0.0
            sellthrough[product] = round(rate, 1)

        return sellthrough

    def get_daily_production(self, date: Optional[datetime] = None) -> List[Dict]:
        """Get production entries for a specific date (default: today)"""
        if date is None:
            date = datetime.now()

        entries = self.entries_by_date.get(date.date(), [])

        return [
            {
                "product": e.product,
                "produced": e.quantity_produced,
                "sold": e.quantity_sold,
                "wasted": e.quantity_wasted,
                "sellthrough_rate": round(e.sellthrough_rate, 1),
            }
            for e in entries
        ]

    def get_waste_by_reason(self) -> Dict[str, Dict]:
        """Analyze waste grouped by reason"""
        waste_by_reason = defaultdict(lambda: {"count": 0, "total_wasted": 0, "products": []})

        for entry in self.entries:
            if entry.quantity_wasted > 0 and entry.waste_reason:
                reason = entry.waste_reason
                waste_by_reason[reason]["count"] += 1
                waste_by_reason[reason]["total_wasted"] += entry.quantity_wasted

                if entry.product not in waste_by_reason[reason]["products"]:
                    waste_by_reason[reason]["products"].append(entry.product)

        return dict(waste_by_reason)

    def get_production_forecast(self) -> Dict[str, float]:
        """Forecast recommended production quantities based on historical sellthrough"""
        forecast = {}

        for product, entries in self.entries_by_product.items():
            # Average production quantity
            avg_produced = sum(e.quantity_produced for e in entries) / len(entries)

            # Average sellthrough rate
            total_produced = sum(e.quantity_produced for e in entries)
            total_sold = sum(e.quantity_sold for e in entries)
            sellthrough = (total_sold / total_produced) if total_produced > 0 else 0.0

            # Recommended production = avg_produced × sellthrough (to minimize waste)
            recommended = avg_produced * sellthrough

            forecast[product] = round(recommended, 0)

        return forecast

    def get_quality_issues(self) -> List[Dict]:
        """Get production entries with quality-related waste"""
        quality_issues = []

        for entry in self.entries:
            if entry.quantity_wasted > 0 and 'quality' in entry.waste_reason.lower():
                quality_issues.append({
                    "date": entry.date.date().isoformat(),
                    "product": entry.product,
                    "quantity_wasted": entry.quantity_wasted,
                    "reason": entry.waste_reason,
                    "notes": entry.notes,
                })

        return quality_issues


if __name__ == "__main__":
    # Demo usage
    parser = ProductionLogParser()

    try:
        entries = parser.parse()

        print(f"✅ Parsed {len(entries)} production entries")
        print("\n📊 Production Summary:")
        summary = parser.get_production_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        print("\n🗑️ Waste Analysis:")
        waste = parser.get_waste_analysis()
        print(f"  Total Waste: {waste['total_waste']} units")
        print(f"  Waste Reasons:")
        for reason, count in waste['waste_reasons'].items():
            print(f"    - {reason}: {count} units")

        if waste['high_waste_products']:
            print(f"  High Waste Products:")
            for product in waste['high_waste_products']:
                print(f"    - {product['product']}: {product['waste_rate']}% waste ({product['total_wasted']} units)")

        print("\n📈 Product Performance:")
        performance = parser.get_product_performance()
        for product, stats in performance.items():
            print(f"  {product}:")
            print(f"    Sellthrough: {stats['avg_sellthrough_rate']}%")
            print(f"    Waste: {stats['avg_waste_percent']}%")
            print(f"    Avg per run: {stats['avg_produced_per_run']} units")

        print("\n⚠️ Overproduction Alerts:")
        alerts = parser.get_overproduction_alerts()
        for alert in alerts[:5]:
            print(f"  - {alert['date']}: {alert['product']} ({alert['sellthrough_rate']}% sold)")

        print("\n📦 Current Inventory:")
        inventory = parser.get_inventory_status()
        for product, quantity in inventory.items():
            print(f"  {product}: {quantity} units")

        print("\n🎯 Production Forecast (Recommended Quantities):")
        forecast = parser.get_production_forecast()
        for product, recommended in forecast.items():
            print(f"  {product}: {int(recommended)} units")

        print("\n❌ Quality Issues:")
        quality_issues = parser.get_quality_issues()
        if quality_issues:
            for issue in quality_issues:
                print(f"  - {issue['date']}: {issue['product']} ({issue['quantity_wasted']} units) - {issue['reason']}")
        else:
            print("  No quality issues detected")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")

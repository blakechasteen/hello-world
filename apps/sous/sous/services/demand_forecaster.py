#!/usr/bin/env python3
"""
Demand Forecasting Service for Phase 3

AI-powered demand forecasting using HoloLoom for:
- Product demand prediction
- Optimal pricing recommendations
- Inventory planning
- Seasonal pattern recognition
- Customer behavior analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
from collections import defaultdict
import asyncio

# HoloLoom imports
try:
    from hololoom import hololoom
    from hololoom.config import Config
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False
    print("Warning: HoloLoom not available. AI features will be disabled.")

from sous.models.farm import Farm
from sous.models.marketplace import Product, Order, ProductCategory
from sous.models.csa import CSASubscription, CSAShare


@dataclass
class DemandForecast:
    """Forecast for product demand"""
    product_id: str
    product_name: str
    farm_id: str

    # Forecast period
    forecast_date: date
    forecast_horizon_days: int = 7

    # Predicted demand
    predicted_quantity: Decimal = Decimal("0.0")
    confidence_interval_low: Decimal = Decimal("0.0")
    confidence_interval_high: Decimal = Decimal("0.0")
    confidence_score: float = 0.0

    # Contributing factors
    factors: Dict[str, float] = field(default_factory=dict)

    # Recommendations
    recommended_price: Optional[Decimal] = None
    recommended_stock_level: Optional[Decimal] = None
    restock_urgency: str = "normal"  # low, normal, high, critical


@dataclass
class PricingRecommendation:
    """Pricing strategy recommendation"""
    product_id: str
    current_price: Decimal
    recommended_price: Decimal
    price_change_percent: float

    # Reasoning
    reasoning: str
    expected_impact: Dict[str, float] = field(default_factory=dict)

    # Market context
    market_average_price: Optional[Decimal] = None
    price_elasticity: float = 0.0


@dataclass
class InventoryRecommendation:
    """Inventory planning recommendation"""
    product_id: str
    product_name: str

    current_stock: Decimal
    recommended_stock: Decimal
    restock_quantity: Decimal

    # Timing
    days_until_stockout: Optional[int] = None
    optimal_restock_date: Optional[date] = None

    # Reasoning
    reasoning: str = ""
    urgency_level: str = "normal"


class DemandForecaster:
    """
    AI-powered demand forecasting engine.

    Uses HoloLoom for pattern learning and predictions.
    """

    def __init__(self):
        self.loom: Optional[HoloLoom] = None
        self.enabled = HOLOLOOM_AVAILABLE

        # Historical data
        self.sales_history: Dict[str, List[Dict]] = defaultdict(list)
        self.price_history: Dict[str, List[Dict]] = defaultdict(list)
        self.seasonal_patterns: Dict[str, Dict] = {}

        if self.enabled:
            # Initialize HoloLoom with FAST config
            self.config = Config.fast()

    async def initialize(self):
        """Initialize HoloLoom for forecasting"""
        if not self.enabled:
            return

        self.loom = HoloLoom()
        await self.loom.__aenter__()

        # Load historical patterns
        await self._load_forecasting_patterns()

    async def close(self):
        """Clean up HoloLoom resources"""
        if self.loom:
            await self.loom.__aexit__(None, None, None)

    # ========================================================================
    # Historical Data Learning
    # ========================================================================

    async def record_sale(
        self,
        product: Product,
        quantity: Decimal,
        price: Decimal,
        sale_date: date,
        customer_type: str = "regular"
    ):
        """Record a sale for learning"""
        sale_record = {
            "date": sale_date,
            "quantity": quantity,
            "price": price,
            "customer_type": customer_type,
            "day_of_week": sale_date.strftime("%A"),
            "month": sale_date.month,
            "season": self._get_season(sale_date)
        }

        self.sales_history[product.product_id].append(sale_record)

        # Learn from sale
        if self.enabled and self.loom:
            experience = (
                f"Sold {quantity} {product.unit} of {product.name} "
                f"at ${price} on {sale_date.strftime('%A')} in {self._get_season(sale_date)}"
            )
            await self.loom.experience(experience)

    async def record_price_change(
        self,
        product: Product,
        old_price: Decimal,
        new_price: Decimal,
        change_date: date
    ):
        """Record price change for elasticity learning"""
        price_record = {
            "date": change_date,
            "old_price": old_price,
            "new_price": new_price,
            "change_percent": float((new_price - old_price) / old_price * 100)
        }

        self.price_history[product.product_id].append(price_record)

    def _get_season(self, date_obj: date) -> str:
        """Determine season from date"""
        month = date_obj.month
        if month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        elif month in [9, 10, 11]:
            return "fall"
        else:
            return "winter"

    # ========================================================================
    # Demand Forecasting
    # ========================================================================

    async def forecast_demand(
        self,
        product: Product,
        horizon_days: int = 7
    ) -> DemandForecast:
        """
        Forecast demand for a product.

        Uses historical data and HoloLoom pattern recognition.
        """
        forecast_date = date.today() + timedelta(days=1)

        if not self.enabled or not self.loom:
            # Fallback: simple moving average
            return self._fallback_forecast(product, forecast_date, horizon_days)

        # Get historical sales
        history = self.sales_history.get(product.product_id, [])

        if not history:
            # No history - predict based on similar products
            return await self._predict_from_similar_products(product, forecast_date, horizon_days)

        # Calculate baseline from recent history
        recent_sales = [
            s for s in history
            if (date.today() - s["date"]).days <= 30
        ]

        if recent_sales:
            baseline_quantity = sum(s["quantity"] for s in recent_sales) / len(recent_sales)
        else:
            baseline_quantity = Decimal("10.0")

        # Query HoloLoom for patterns
        query = (
            f"Predict demand for {product.name} in {self._get_season(forecast_date)} "
            f"based on historical patterns"
        )

        memories = await self.loom.recall(query)

        # Calculate adjustments based on patterns
        adjustments = self._calculate_demand_adjustments(product, forecast_date, memories)

        # Apply adjustments to baseline
        predicted_quantity = baseline_quantity * Decimal(str(1.0 + adjustments["total_adjustment"]))

        # Calculate confidence intervals (±20%)
        confidence_range = predicted_quantity * Decimal("0.20")

        forecast = DemandForecast(
            product_id=product.product_id,
            product_name=product.name,
            farm_id=product.farm_id,
            forecast_date=forecast_date,
            forecast_horizon_days=horizon_days,
            predicted_quantity=predicted_quantity,
            confidence_interval_low=predicted_quantity - confidence_range,
            confidence_interval_high=predicted_quantity + confidence_range,
            confidence_score=adjustments.get("confidence", 0.7),
            factors=adjustments
        )

        # Add recommendations
        forecast.recommended_stock_level = predicted_quantity * Decimal("1.2")  # 20% buffer

        if product.quantity_available < predicted_quantity * Decimal("0.5"):
            forecast.restock_urgency = "high"
        elif product.quantity_available < predicted_quantity:
            forecast.restock_urgency = "normal"
        else:
            forecast.restock_urgency = "low"

        return forecast

    def _fallback_forecast(
        self,
        product: Product,
        forecast_date: date,
        horizon_days: int
    ) -> DemandForecast:
        """Fallback forecast without HoloLoom"""
        history = self.sales_history.get(product.product_id, [])

        if not history:
            # No data - conservative estimate
            predicted = Decimal("5.0")
        else:
            # Simple moving average
            recent = history[-30:]  # Last 30 sales
            predicted = sum(s["quantity"] for s in recent) / len(recent)

        return DemandForecast(
            product_id=product.product_id,
            product_name=product.name,
            farm_id=product.farm_id,
            forecast_date=forecast_date,
            forecast_horizon_days=horizon_days,
            predicted_quantity=predicted,
            confidence_interval_low=predicted * Decimal("0.8"),
            confidence_interval_high=predicted * Decimal("1.2"),
            confidence_score=0.5,
            factors={"method": "simple_average"}
        )

    async def _predict_from_similar_products(
        self,
        product: Product,
        forecast_date: date,
        horizon_days: int
    ) -> DemandForecast:
        """Predict demand based on similar products"""
        # Find similar products (same category, organic status)
        similar_sales = []
        for prod_id, sales in self.sales_history.items():
            if prod_id != product.product_id:
                # In production, would check if products are similar
                similar_sales.extend(sales)

        if not similar_sales:
            return self._fallback_forecast(product, forecast_date, horizon_days)

        # Average demand from similar products
        avg_quantity = sum(s["quantity"] for s in similar_sales) / len(similar_sales)

        return DemandForecast(
            product_id=product.product_id,
            product_name=product.name,
            farm_id=product.farm_id,
            forecast_date=forecast_date,
            forecast_horizon_days=horizon_days,
            predicted_quantity=Decimal(str(avg_quantity)),
            confidence_interval_low=Decimal(str(avg_quantity * 0.7)),
            confidence_interval_high=Decimal(str(avg_quantity * 1.3)),
            confidence_score=0.6,
            factors={"method": "similar_products"}
        )

    def _calculate_demand_adjustments(
        self,
        product: Product,
        forecast_date: date,
        memories: List
    ) -> Dict[str, float]:
        """Calculate demand adjustments based on patterns"""
        adjustments = {}

        # Seasonal adjustment
        season = self._get_season(forecast_date)
        history = self.sales_history.get(product.product_id, [])

        if history:
            seasonal_avg = {}
            for s in ["spring", "summer", "fall", "winter"]:
                season_sales = [h for h in history if h["season"] == s]
                if season_sales:
                    seasonal_avg[s] = sum(sale["quantity"] for sale in season_sales) / len(season_sales)

            if seasonal_avg:
                overall_avg = sum(seasonal_avg.values()) / len(seasonal_avg)
                current_season_avg = seasonal_avg.get(season, overall_avg)
                adjustments["seasonal"] = float((current_season_avg - overall_avg) / overall_avg)
            else:
                adjustments["seasonal"] = 0.0
        else:
            adjustments["seasonal"] = 0.0

        # Day of week adjustment
        day_of_week = forecast_date.strftime("%A")
        if history:
            dow_sales = [h for h in history if h["day_of_week"] == day_of_week]
            all_avg = sum(h["quantity"] for h in history) / len(history)

            if dow_sales:
                dow_avg = sum(h["quantity"] for h in dow_sales) / len(dow_sales)
                adjustments["day_of_week"] = float((dow_avg - all_avg) / all_avg)
            else:
                adjustments["day_of_week"] = 0.0
        else:
            adjustments["day_of_week"] = 0.0

        # Trend adjustment (growing or declining sales)
        if len(history) >= 10:
            recent = history[-10:]
            older = history[-20:-10] if len(history) >= 20 else history[:-10]

            recent_avg = sum(h["quantity"] for h in recent) / len(recent)
            older_avg = sum(h["quantity"] for h in older) / len(older)

            if older_avg > 0:
                adjustments["trend"] = float((recent_avg - older_avg) / older_avg)
            else:
                adjustments["trend"] = 0.0
        else:
            adjustments["trend"] = 0.0

        # Total adjustment
        adjustments["total_adjustment"] = (
            adjustments.get("seasonal", 0.0) +
            adjustments.get("day_of_week", 0.0) +
            adjustments.get("trend", 0.0)
        )

        # Confidence score
        adjustments["confidence"] = min(1.0, len(history) / 100.0)

        return adjustments

    # ========================================================================
    # Pricing Optimization
    # ========================================================================

    async def recommend_pricing(
        self,
        product: Product,
        target_sales_volume: Optional[Decimal] = None
    ) -> PricingRecommendation:
        """
        Recommend optimal pricing for a product.

        Considers demand elasticity, competition, and target volume.
        """
        current_price = product.get_effective_price()

        # Calculate price elasticity from historical data
        elasticity = self._calculate_price_elasticity(product.product_id)

        # Default recommendation: maintain current price
        recommended_price = current_price
        reasoning = "Current price is optimal based on historical performance."

        # If we have elasticity data, optimize
        if elasticity != 0.0 and abs(elasticity) > 0.1:
            if elasticity < -1.0:
                # Elastic demand: lower price increases revenue
                recommended_price = current_price * Decimal("0.95")
                reasoning = "Demand is elastic. Lowering price 5% should increase total revenue."
            elif elasticity > -0.5:
                # Inelastic demand: raise price
                recommended_price = current_price * Decimal("1.05")
                reasoning = "Demand is inelastic. Raising price 5% should increase revenue."

        # Check if price is too far from market average
        # In production, would query market data
        # For now, simple logic based on organic premium
        if product.organic:
            expected_premium = Decimal("1.3")  # 30% premium for organic
            market_baseline = current_price / expected_premium
        else:
            market_baseline = current_price

        price_change_percent = float((recommended_price - current_price) / current_price * 100)

        recommendation = PricingRecommendation(
            product_id=product.product_id,
            current_price=current_price,
            recommended_price=recommended_price,
            price_change_percent=price_change_percent,
            reasoning=reasoning,
            price_elasticity=elasticity
        )

        return recommendation

    def _calculate_price_elasticity(self, product_id: str) -> float:
        """Calculate price elasticity of demand"""
        price_history = self.price_history.get(product_id, [])
        sales_history = self.sales_history.get(product_id, [])

        if not price_history or not sales_history:
            return 0.0

        # Find sales before and after price changes
        elasticities = []

        for price_change in price_history:
            change_date = price_change["date"]

            # Sales before price change
            before_sales = [
                s for s in sales_history
                if (change_date - s["date"]).days <= 7 and s["date"] < change_date
            ]

            # Sales after price change
            after_sales = [
                s for s in sales_history
                if (s["date"] - change_date).days <= 7 and s["date"] >= change_date
            ]

            if before_sales and after_sales:
                before_qty = sum(s["quantity"] for s in before_sales) / len(before_sales)
                after_qty = sum(s["quantity"] for s in after_sales) / len(after_sales)

                qty_change_percent = ((after_qty - before_qty) / before_qty)
                price_change_percent = price_change["change_percent"] / 100.0

                if price_change_percent != 0:
                    elasticity = qty_change_percent / price_change_percent
                    elasticities.append(elasticity)

        if elasticities:
            return sum(elasticities) / len(elasticities)

        return 0.0

    # ========================================================================
    # Inventory Planning
    # ========================================================================

    async def recommend_inventory(
        self,
        product: Product,
        forecast: Optional[DemandForecast] = None
    ) -> InventoryRecommendation:
        """
        Recommend inventory levels for a product.

        Uses demand forecast and safety stock calculations.
        """
        if not forecast:
            forecast = await self.forecast_demand(product)

        current_stock = product.quantity_available
        predicted_demand = forecast.predicted_quantity

        # Calculate safety stock (buffer for uncertainty)
        uncertainty_range = forecast.confidence_interval_high - forecast.confidence_interval_low
        safety_stock = uncertainty_range * Decimal("0.5")

        # Recommended stock = predicted demand + safety stock
        recommended_stock = predicted_demand + safety_stock

        # Calculate restock quantity
        restock_quantity = max(Decimal("0.0"), recommended_stock - current_stock)

        # Estimate days until stockout
        daily_avg_sales = predicted_demand / Decimal(str(forecast.forecast_horizon_days))
        if daily_avg_sales > 0:
            days_until_stockout = int(current_stock / daily_avg_sales)
        else:
            days_until_stockout = None

        # Determine urgency
        if days_until_stockout is not None:
            if days_until_stockout <= 2:
                urgency = "critical"
            elif days_until_stockout <= 5:
                urgency = "high"
            elif days_until_stockout <= 10:
                urgency = "normal"
            else:
                urgency = "low"
        else:
            urgency = "normal"

        # Optimal restock date (when stock hits 30% of recommended)
        if daily_avg_sales > 0:
            threshold = recommended_stock * Decimal("0.3")
            if current_stock > threshold:
                days_until_threshold = int((current_stock - threshold) / daily_avg_sales)
                optimal_restock_date = date.today() + timedelta(days=days_until_threshold)
            else:
                optimal_restock_date = date.today()
        else:
            optimal_restock_date = None

        reasoning = f"Based on predicted demand of {predicted_demand} {product.unit} "
        reasoning += f"over {forecast.forecast_horizon_days} days, "
        reasoning += f"recommend maintaining {recommended_stock} {product.unit} in stock. "

        if restock_quantity > 0:
            reasoning += f"Restock with {restock_quantity} {product.unit}. "

        if days_until_stockout:
            reasoning += f"Current stock will last approximately {days_until_stockout} days."

        return InventoryRecommendation(
            product_id=product.product_id,
            product_name=product.name,
            current_stock=current_stock,
            recommended_stock=recommended_stock,
            restock_quantity=restock_quantity,
            days_until_stockout=days_until_stockout,
            optimal_restock_date=optimal_restock_date,
            reasoning=reasoning,
            urgency_level=urgency
        )

    # ========================================================================
    # Batch Forecasting
    # ========================================================================

    async def forecast_farm_demand(
        self,
        farm_id: str,
        products: List[Product],
        horizon_days: int = 7
    ) -> List[DemandForecast]:
        """Generate forecasts for all products from a farm"""
        forecasts = []

        for product in products:
            if product.farm_id == farm_id and product.active:
                forecast = await self.forecast_demand(product, horizon_days)
                forecasts.append(forecast)

        return forecasts

    # ========================================================================
    # Historical Pattern Loading
    # ========================================================================

    async def _load_forecasting_patterns(self):
        """Load common forecasting patterns into HoloLoom"""
        if not self.enabled or not self.loom:
            return

        patterns = [
            "Summer demand for tomatoes is 50% higher than winter",
            "Weekend sales are typically 40% higher than weekday sales",
            "Organic products sell at 30% premium but 20% lower volume",
            "Leafy greens have highest demand on Monday and Tuesday",
            "Root vegetables have steady demand year-round",
            "Price increases above 15% significantly reduce demand",
            "CSA members prefer diverse vegetable boxes",
            "Prepared foods sell best on Thursday and Friday",
            "Fruit demand peaks in summer months",
            "Herb sales correlate with holiday cooking periods"
        ]

        for pattern in patterns:
            await self.loom.experience(pattern)


# ========================================================================
# Helper Functions
# ========================================================================

async def create_demand_forecaster() -> DemandForecaster:
    """Create and initialize demand forecaster"""
    forecaster = DemandForecaster()
    await forecaster.initialize()
    return forecaster

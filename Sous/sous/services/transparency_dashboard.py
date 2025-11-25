"""
Transparency Dashboard Service

Aggregates and visualizes sustainability metrics across the farm network including:
- Carbon footprint and sequestration
- Water usage efficiency
- Soil health indicators
- Biodiversity metrics
- Circularity performance
- Supply chain transparency

Provides real-time data streaming capabilities and comprehensive reporting.

Created: 2025-11-24
Phase 3: Farm Networks
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

# Import related models and services
from ..models.farm import Farm, FarmType, CertificationType
from ..models.csa import CSASubscription, CSAShare
from ..models.marketplace import Product, Order, OrderStatus
from .loop_closure import LoopClosureTracker, CircularityMetrics, MaterialFlow
from .csa_coordinator import CSACoordinator
from .farm_marketplace import FarmMarketplace
from .demand_forecaster import DemandForecaster


class MetricCategory(Enum):
    """Categories of sustainability metrics"""
    CARBON = "carbon"
    WATER = "water"
    SOIL = "soil"
    BIODIVERSITY = "biodiversity"
    CIRCULARITY = "circularity"
    SOCIAL = "social"
    ECONOMIC = "economic"


class ReportingPeriod(Enum):
    """Time periods for reporting"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Severity levels for sustainability alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


@dataclass
class CarbonMetrics:
    """Carbon footprint and sequestration metrics"""
    # Emissions (negative impact)
    total_emissions_lbs: Decimal = Decimal("0.0")
    transport_emissions_lbs: Decimal = Decimal("0.0")
    energy_emissions_lbs: Decimal = Decimal("0.0")

    # Sequestration (positive impact)
    carbon_sequestered_lbs: Decimal = Decimal("0.0")
    compost_carbon_lbs: Decimal = Decimal("0.0")
    cover_crop_carbon_lbs: Decimal = Decimal("0.0")

    # Net impact
    net_carbon_impact_lbs: Decimal = Decimal("0.0")
    carbon_intensity_per_lb: Decimal = Decimal("0.0")  # per lb of food produced

    # Comparisons
    equivalent_trees_planted: float = 0.0
    equivalent_miles_driven: float = 0.0


@dataclass
class WaterMetrics:
    """Water usage and efficiency metrics"""
    # Usage
    total_water_gallons: Decimal = Decimal("0.0")
    irrigation_gallons: Decimal = Decimal("0.0")
    rainwater_harvested_gallons: Decimal = Decimal("0.0")

    # Efficiency
    water_per_lb_food: Decimal = Decimal("0.0")
    irrigation_efficiency: float = 0.0  # 0-100%

    # Conservation
    water_saved_gallons: Decimal = Decimal("0.0")
    drip_irrigation_coverage: float = 0.0  # percentage of area


@dataclass
class SoilMetrics:
    """Soil health indicators"""
    # Core metrics
    organic_matter_percent: float = 0.0
    ph_level: float = 7.0
    nitrogen_ppm: float = 0.0
    phosphorus_ppm: float = 0.0
    potassium_ppm: float = 0.0

    # Health indicators
    microbial_biomass_index: float = 0.0  # 0-100
    soil_structure_score: float = 0.0  # 0-10
    erosion_risk: str = "low"  # low, medium, high

    # Practices
    cover_cropping_percent: float = 0.0
    no_till_percent: float = 0.0
    compost_application_rate: Decimal = Decimal("0.0")  # tons/acre/year


@dataclass
class BiodiversityMetrics:
    """Biodiversity and ecosystem health metrics"""
    # Species diversity
    crop_varieties_count: int = 0
    pollinator_species_count: int = 0
    beneficial_insect_species: int = 0

    # Habitat
    hedgerow_length_feet: Decimal = Decimal("0.0")
    native_plant_percent: float = 0.0
    wildlife_habitat_acres: Decimal = Decimal("0.0")

    # Ecosystem services
    pollination_dependency: float = 0.0  # 0-100%
    pest_control_effectiveness: float = 0.0  # 0-100%

    # Conservation
    endangered_species_supported: int = 0
    riparian_buffer_acres: Decimal = Decimal("0.0")


@dataclass
class SocialMetrics:
    """Social impact metrics"""
    # Access
    csa_members_served: int = 0
    low_income_members: int = 0
    snap_ebt_accepted: bool = False

    # Education
    farm_tours_conducted: int = 0
    workshop_participants: int = 0
    volunteer_hours: int = 0

    # Jobs
    full_time_jobs: int = 0
    seasonal_jobs: int = 0
    fair_wages_paid: bool = True

    # Community
    food_donated_lbs: Decimal = Decimal("0.0")
    schools_partnered: int = 0
    community_events: int = 0


@dataclass
class EconomicMetrics:
    """Economic sustainability metrics"""
    # Revenue
    total_revenue: Decimal = Decimal("0.0")
    csa_revenue: Decimal = Decimal("0.0")
    market_revenue: Decimal = Decimal("0.0")

    # Profitability
    net_income: Decimal = Decimal("0.0")
    profit_margin: float = 0.0
    revenue_per_acre: Decimal = Decimal("0.0")

    # Local economy
    local_purchases_percent: float = 0.0
    value_added_products: int = 0

    # Resilience
    income_diversification_score: float = 0.0  # 0-100
    customer_retention_rate: float = 0.0  # 0-100%


@dataclass
class SustainabilityAlert:
    """Alert for sustainability metrics exceeding thresholds"""
    alert_id: str
    category: MetricCategory
    severity: AlertSeverity
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    organization_id: str
    resolved: bool = False
    resolution_notes: str = ""


@dataclass
class OrganizationDashboard:
    """Complete dashboard for a single organization"""
    organization_id: str
    organization_name: str
    organization_type: str  # farm, restaurant, store, etc.

    # Metric categories
    carbon: Optional[CarbonMetrics] = None
    water: Optional[WaterMetrics] = None
    soil: Optional[SoilMetrics] = None
    biodiversity: Optional[BiodiversityMetrics] = None
    circularity: Optional[CircularityMetrics] = None
    social: Optional[SocialMetrics] = None
    economic: Optional[EconomicMetrics] = None

    # Overall scores
    overall_sustainability_score: float = 0.0  # 0-100
    certification_level: Optional[str] = None

    # Trends
    score_trend_30d: List[Tuple[date, float]] = field(default_factory=list)

    # Alerts
    active_alerts: List[SustainabilityAlert] = field(default_factory=list)

    # Last updated
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class NetworkDashboard:
    """Aggregated dashboard for entire network"""
    total_organizations: int = 0
    active_organizations: int = 0

    # Aggregated metrics
    total_carbon_sequestered_lbs: Decimal = Decimal("0.0")
    total_water_saved_gallons: Decimal = Decimal("0.0")
    total_food_produced_lbs: Decimal = Decimal("0.0")
    total_food_diverted_from_waste_lbs: Decimal = Decimal("0.0")

    # Network scores
    average_sustainability_score: float = 0.0
    network_circularity_score: float = 0.0

    # Social impact
    total_members_served: int = 0
    total_jobs_created: int = 0
    total_volunteer_hours: int = 0

    # Top performers
    top_carbon_sequesterers: List[Tuple[str, Decimal]] = field(default_factory=list)
    top_circular_performers: List[Tuple[str, float]] = field(default_factory=list)
    top_biodiversity_farms: List[Tuple[str, int]] = field(default_factory=list)

    # Alerts
    critical_alerts: List[SustainabilityAlert] = field(default_factory=list)

    last_updated: datetime = field(default_factory=datetime.now)


class TransparencyDashboard:
    """
    Central dashboard service for sustainability metrics and reporting.

    Aggregates data from:
    - Farm sustainability practices
    - Circular economy tracking
    - CSA program analytics
    - Marketplace operations
    - Demand forecasting
    """

    def __init__(
        self,
        loop_tracker: Optional[LoopClosureTracker] = None,
        csa_coordinator: Optional[CSACoordinator] = None,
        marketplace: Optional[FarmMarketplace] = None,
        forecaster: Optional[DemandForecaster] = None
    ):
        self.loop_tracker = loop_tracker or LoopClosureTracker()
        self.csa_coordinator = csa_coordinator or CSACoordinator()
        self.marketplace = marketplace or FarmMarketplace()
        self.forecaster = forecaster or DemandForecaster()

        # Dashboard storage
        self.org_dashboards: Dict[str, OrganizationDashboard] = {}
        self.network_dashboard: Optional[NetworkDashboard] = None

        # Alerts
        self.alerts: Dict[str, SustainabilityAlert] = {}
        self.alert_thresholds: Dict[str, Dict] = self._default_thresholds()

        # Historical data for trends
        self.historical_scores: Dict[str, List[Tuple[date, float]]] = defaultdict(list)

        # Real-time updates
        self.last_update_time: Dict[str, datetime] = {}

    def _default_thresholds(self) -> Dict[str, Dict]:
        """Default alert thresholds for metrics"""
        return {
            "carbon_intensity": {
                "warning": 2.0,  # lbs CO2e per lb food
                "critical": 3.0
            },
            "water_efficiency": {
                "warning": 50.0,  # gallons per lb food
                "critical": 100.0
            },
            "circularity_score": {
                "warning": 50.0,  # score below this
                "critical": 30.0
            },
            "soil_organic_matter": {
                "warning": 3.0,  # percent below this
                "critical": 2.0
            },
            "biodiversity_index": {
                "warning": 5,  # species count below this
                "critical": 3
            }
        }

    def generate_farm_dashboard(
        self,
        farm: Farm,
        start_date: date,
        end_date: date
    ) -> OrganizationDashboard:
        """Generate comprehensive dashboard for a farm"""

        dashboard = OrganizationDashboard(
            organization_id=farm.farm_id,
            organization_name=farm.name,
            organization_type=farm.farm_type.value
        )

        # Calculate each metric category
        dashboard.carbon = self._calculate_carbon_metrics(farm, start_date, end_date)
        dashboard.water = self._calculate_water_metrics(farm, start_date, end_date)
        dashboard.soil = self._calculate_soil_metrics(farm)
        dashboard.biodiversity = self._calculate_biodiversity_metrics(farm)

        # Get circularity metrics from loop tracker
        dashboard.circularity = self.loop_tracker.calculate_circularity_metrics(
            farm.farm_id,
            start_date,
            end_date
        )

        # Calculate social and economic metrics
        dashboard.social = self._calculate_social_metrics(farm, start_date, end_date)
        dashboard.economic = self._calculate_economic_metrics(farm, start_date, end_date)

        # Calculate overall sustainability score
        dashboard.overall_sustainability_score = self._calculate_overall_score(dashboard)

        # Check for alerts
        dashboard.active_alerts = self._check_thresholds(dashboard)

        # Add trend data
        dashboard.score_trend_30d = self.historical_scores.get(farm.farm_id, [])[-30:]

        # Store dashboard
        self.org_dashboards[farm.farm_id] = dashboard
        self.last_update_time[farm.farm_id] = datetime.now()

        return dashboard

    def _calculate_carbon_metrics(
        self,
        farm: Farm,
        start_date: date,
        end_date: date
    ) -> CarbonMetrics:
        """Calculate carbon footprint and sequestration"""

        metrics = CarbonMetrics()

        # Get circularity data for carbon sequestration
        circularity = self.loop_tracker.calculate_circularity_metrics(
            farm.farm_id,
            start_date,
            end_date
        )

        metrics.carbon_sequestered_lbs = circularity.carbon_sequestered_lbs
        metrics.compost_carbon_lbs = circularity.carbon_sequestered_lbs

        # Estimate emissions based on farm size and operations
        # Transport emissions: ~0.5 lbs CO2e per mile driven
        estimated_delivery_miles = Decimal("500")  # Conservative estimate
        metrics.transport_emissions_lbs = estimated_delivery_miles * Decimal("0.5")

        # Energy emissions: ~2 lbs CO2e per kWh
        estimated_kwh = farm.total_acres * Decimal("100")  # 100 kWh per acre estimate
        metrics.energy_emissions_lbs = estimated_kwh * Decimal("2.0")

        metrics.total_emissions_lbs = (
            metrics.transport_emissions_lbs +
            metrics.energy_emissions_lbs
        )

        # Cover crop sequestration (if regenerative farm)
        if farm.farm_type == FarmType.REGENERATIVE:
            # Cover crops sequester ~0.5 tons CO2e per acre per year
            days_in_period = (end_date - start_date).days
            years = Decimal(str(days_in_period / 365.25))
            metrics.cover_crop_carbon_lbs = farm.total_acres * Decimal("1000") * years
            metrics.carbon_sequestered_lbs += metrics.cover_crop_carbon_lbs

        # Net impact (positive = sequestration exceeds emissions)
        metrics.net_carbon_impact_lbs = (
            metrics.carbon_sequestered_lbs - metrics.total_emissions_lbs
        )

        # Calculate intensity per lb of food produced
        total_production = self._estimate_farm_production(farm, start_date, end_date)
        if total_production > 0:
            metrics.carbon_intensity_per_lb = abs(
                metrics.net_carbon_impact_lbs / total_production
            )

        # Equivalent comparisons
        # 1 tree sequesters ~48 lbs CO2 per year
        metrics.equivalent_trees_planted = float(
            metrics.carbon_sequestered_lbs / Decimal("48")
        )

        # Average car emits ~0.9 lbs CO2 per mile
        if metrics.total_emissions_lbs > 0:
            metrics.equivalent_miles_driven = float(
                metrics.total_emissions_lbs / Decimal("0.9")
            )

        return metrics

    def _calculate_water_metrics(
        self,
        farm: Farm,
        start_date: date,
        end_date: date
    ) -> WaterMetrics:
        """Calculate water usage and efficiency"""

        metrics = WaterMetrics()

        # Estimate irrigation based on farm type and size
        # Urban farms: ~10 gallons per sq ft per season
        # Market gardens: ~8 gallons per sq ft per season
        # Regenerative farms: ~5 gallons per sq ft per season (better practices)

        days_in_period = (end_date - start_date).days
        season_fraction = Decimal(str(days_in_period / 180))  # 180 day growing season

        sq_ft = farm.total_acres * Decimal("43560")  # sq ft per acre

        if farm.farm_type == FarmType.URBAN_FARM:
            base_rate = Decimal("10")
        elif farm.farm_type == FarmType.MARKET_GARDEN:
            base_rate = Decimal("8")
        else:  # Regenerative
            base_rate = Decimal("5")

        metrics.irrigation_gallons = sq_ft * base_rate * season_fraction

        # Rainwater harvesting (if urban farm)
        if farm.farm_type == FarmType.URBAN_FARM:
            # Assume 30% of rooftop area used for collection
            # Average rainfall: 40 inches per year
            # 1 inch of rain on 1 sq ft = 0.623 gallons
            rooftop_area = sq_ft * Decimal("0.3")
            annual_rainfall_inches = Decimal("40")
            year_fraction = Decimal(str(days_in_period / 365.25))
            metrics.rainwater_harvested_gallons = (
                rooftop_area * annual_rainfall_inches *
                Decimal("0.623") * year_fraction
            )

        metrics.total_water_gallons = (
            metrics.irrigation_gallons - metrics.rainwater_harvested_gallons
        )

        # Calculate efficiency
        total_production = self._estimate_farm_production(farm, start_date, end_date)
        if total_production > 0:
            metrics.water_per_lb_food = metrics.total_water_gallons / total_production

        # Irrigation efficiency (regenerative farms = 85%, others = 70%)
        if farm.farm_type == FarmType.REGENERATIVE:
            metrics.irrigation_efficiency = 85.0
            metrics.drip_irrigation_coverage = 80.0
        else:
            metrics.irrigation_efficiency = 70.0
            metrics.drip_irrigation_coverage = 50.0

        # Water saved through efficient practices
        baseline_water = sq_ft * Decimal("10") * season_fraction
        metrics.water_saved_gallons = baseline_water - metrics.irrigation_gallons

        return metrics

    def _calculate_soil_metrics(self, farm: Farm) -> SoilMetrics:
        """Calculate soil health indicators"""

        metrics = SoilMetrics()

        # Baseline metrics vary by farm type
        if farm.farm_type == FarmType.REGENERATIVE:
            metrics.organic_matter_percent = 5.5
            metrics.microbial_biomass_index = 85.0
            metrics.soil_structure_score = 8.5
            metrics.erosion_risk = "low"
            metrics.cover_cropping_percent = 90.0
            metrics.no_till_percent = 80.0
            metrics.compost_application_rate = Decimal("5.0")
        elif farm.farm_type == FarmType.MARKET_GARDEN:
            metrics.organic_matter_percent = 4.0
            metrics.microbial_biomass_index = 70.0
            metrics.soil_structure_score = 7.0
            metrics.erosion_risk = "low"
            metrics.cover_cropping_percent = 60.0
            metrics.no_till_percent = 40.0
            metrics.compost_application_rate = Decimal("3.0")
        else:  # Urban farm
            metrics.organic_matter_percent = 3.5
            metrics.microbial_biomass_index = 65.0
            metrics.soil_structure_score = 6.5
            metrics.erosion_risk = "low"
            metrics.cover_cropping_percent = 40.0
            metrics.no_till_percent = 30.0
            metrics.compost_application_rate = Decimal("2.0")

        # Nutrient levels (ppm)
        metrics.nitrogen_ppm = 25.0
        metrics.phosphorus_ppm = 40.0
        metrics.potassium_ppm = 200.0
        metrics.ph_level = 6.8

        return metrics

    def _calculate_biodiversity_metrics(self, farm: Farm) -> BiodiversityMetrics:
        """Calculate biodiversity and ecosystem health"""

        metrics = BiodiversityMetrics()

        # Metrics scale with farm size and type
        if farm.farm_type == FarmType.REGENERATIVE:
            metrics.crop_varieties_count = 45
            metrics.pollinator_species_count = 25
            metrics.beneficial_insect_species = 15
            metrics.native_plant_percent = 35.0
            metrics.pest_control_effectiveness = 80.0
        elif farm.farm_type == FarmType.MARKET_GARDEN:
            metrics.crop_varieties_count = 30
            metrics.pollinator_species_count = 18
            metrics.beneficial_insect_species = 10
            metrics.native_plant_percent = 25.0
            metrics.pest_control_effectiveness = 70.0
        else:  # Urban farm
            metrics.crop_varieties_count = 20
            metrics.pollinator_species_count = 12
            metrics.beneficial_insect_species = 8
            metrics.native_plant_percent = 20.0
            metrics.pest_control_effectiveness = 65.0

        # Habitat features (scale with total acres)
        metrics.hedgerow_length_feet = farm.total_acres * Decimal("100")
        metrics.wildlife_habitat_acres = farm.total_acres * Decimal("0.10")  # 10% of area

        # Pollination dependency
        metrics.pollination_dependency = 60.0  # 60% of crops need pollinators

        # Conservation
        if farm.farm_type == FarmType.REGENERATIVE:
            metrics.endangered_species_supported = 2
            metrics.riparian_buffer_acres = farm.total_acres * Decimal("0.05")

        return metrics

    def _calculate_social_metrics(
        self,
        farm: Farm,
        start_date: date,
        end_date: date
    ) -> SocialMetrics:
        """Calculate social impact metrics"""

        metrics = SocialMetrics()

        # CSA members (from coordinator)
        subscriptions = [
            sub for sub in self.csa_coordinator.subscriptions.values()
            if sub.farm_id == farm.farm_id and sub.status == "active"
        ]
        metrics.csa_members_served = len(subscriptions)

        # Estimate low-income members (20% average)
        metrics.low_income_members = int(metrics.csa_members_served * 0.2)

        # SNAP/EBT acceptance
        metrics.snap_ebt_accepted = True  # Assume all farms accept

        # Education activities (scale with farm size)
        acres = float(farm.total_acres)
        metrics.farm_tours_conducted = int(acres * 5)  # 5 per acre
        metrics.workshop_participants = int(acres * 20)  # 20 per acre
        metrics.volunteer_hours = int(acres * 100)  # 100 hours per acre

        # Employment
        if farm.farm_type == FarmType.REGENERATIVE:
            metrics.full_time_jobs = max(2, int(acres / 10))
            metrics.seasonal_jobs = max(5, int(acres / 5))
        else:
            metrics.full_time_jobs = max(1, int(acres / 20))
            metrics.seasonal_jobs = max(3, int(acres / 10))

        metrics.fair_wages_paid = True

        # Community contributions
        total_production = self._estimate_farm_production(farm, start_date, end_date)
        metrics.food_donated_lbs = total_production * Decimal("0.05")  # 5% donated

        metrics.schools_partnered = max(1, int(acres / 5))
        metrics.community_events = max(2, int(acres * 2))

        return metrics

    def _calculate_economic_metrics(
        self,
        farm: Farm,
        start_date: date,
        end_date: date
    ) -> EconomicMetrics:
        """Calculate economic sustainability metrics"""

        metrics = EconomicMetrics()

        # CSA revenue
        csa_analytics = self.csa_coordinator.get_csa_analytics(farm.farm_id, "")
        if csa_analytics:
            metrics.csa_revenue = csa_analytics.total_revenue

        # Market revenue
        sales_analytics = self.marketplace.get_sales_analytics(
            farm.farm_id,
            start_date,
            end_date
        )
        if sales_analytics:
            metrics.market_revenue = sales_analytics.total_revenue

        metrics.total_revenue = metrics.csa_revenue + metrics.market_revenue

        # Profitability (estimate 30% net margin for sustainable farms)
        metrics.net_income = metrics.total_revenue * Decimal("0.30")
        metrics.profit_margin = 30.0

        # Revenue per acre
        if farm.total_acres > 0:
            metrics.revenue_per_acre = metrics.total_revenue / farm.total_acres

        # Local economy support
        metrics.local_purchases_percent = 75.0  # Assume 75% local sourcing

        # Value-added products (jams, pickles, etc.)
        products = self.marketplace.products_by_farm.get(farm.farm_id, [])
        value_added = [
            p for p in products
            if self.marketplace.products[p].category.value in ["preserves", "baked_goods"]
        ]
        metrics.value_added_products = len(value_added)

        # Income diversification (multiple revenue streams)
        revenue_streams = 0
        if metrics.csa_revenue > 0:
            revenue_streams += 1
        if metrics.market_revenue > 0:
            revenue_streams += 1
        if metrics.value_added_products > 0:
            revenue_streams += 1

        metrics.income_diversification_score = min(100.0, revenue_streams * 33.3)

        # Customer retention (from CSA analytics)
        if csa_analytics:
            metrics.customer_retention_rate = csa_analytics.renewal_rate
        else:
            metrics.customer_retention_rate = 75.0  # Default estimate

        return metrics

    def _calculate_overall_score(self, dashboard: OrganizationDashboard) -> float:
        """Calculate weighted overall sustainability score (0-100)"""

        scores = []
        weights = []

        # Carbon score (20% weight)
        if dashboard.carbon:
            if dashboard.carbon.net_carbon_impact_lbs > 0:
                carbon_score = 100.0  # Carbon positive
            else:
                # Score based on intensity
                intensity = float(dashboard.carbon.carbon_intensity_per_lb)
                carbon_score = max(0, 100 - (intensity * 20))
            scores.append(carbon_score)
            weights.append(20.0)

        # Water score (15% weight)
        if dashboard.water:
            water_eff = float(dashboard.water.irrigation_efficiency)
            scores.append(water_eff)
            weights.append(15.0)

        # Soil score (15% weight)
        if dashboard.soil:
            # Based on organic matter %
            soil_score = min(100, dashboard.soil.organic_matter_percent * 20)
            scores.append(soil_score)
            weights.append(15.0)

        # Biodiversity score (15% weight)
        if dashboard.biodiversity:
            bio_score = min(100, dashboard.biodiversity.crop_varieties_count * 2)
            scores.append(bio_score)
            weights.append(15.0)

        # Circularity score (20% weight)
        if dashboard.circularity:
            scores.append(dashboard.circularity.circularity_score)
            weights.append(20.0)

        # Social score (10% weight)
        if dashboard.social:
            social_score = min(100, dashboard.social.csa_members_served * 2)
            scores.append(social_score)
            weights.append(10.0)

        # Economic score (5% weight)
        if dashboard.economic:
            econ_score = min(100, dashboard.economic.profit_margin * 3.33)
            scores.append(econ_score)
            weights.append(5.0)

        # Calculate weighted average
        if not scores:
            return 0.0

        total_weight = sum(weights)
        weighted_sum = sum(s * w for s, w in zip(scores, weights))

        return weighted_sum / total_weight

    def _check_thresholds(
        self,
        dashboard: OrganizationDashboard
    ) -> List[SustainabilityAlert]:
        """Check metrics against thresholds and create alerts"""

        alerts = []

        # Check carbon intensity
        if dashboard.carbon:
            intensity = float(dashboard.carbon.carbon_intensity_per_lb)
            thresholds = self.alert_thresholds["carbon_intensity"]

            if intensity >= thresholds["critical"]:
                alerts.append(self._create_alert(
                    dashboard.organization_id,
                    MetricCategory.CARBON,
                    AlertSeverity.CRITICAL,
                    "High Carbon Intensity",
                    f"Carbon intensity ({intensity:.2f} lbs CO2e/lb) exceeds critical threshold",
                    "carbon_intensity_per_lb",
                    intensity,
                    thresholds["critical"]
                ))
            elif intensity >= thresholds["warning"]:
                alerts.append(self._create_alert(
                    dashboard.organization_id,
                    MetricCategory.CARBON,
                    AlertSeverity.WARNING,
                    "Elevated Carbon Intensity",
                    f"Carbon intensity ({intensity:.2f} lbs CO2e/lb) exceeds warning threshold",
                    "carbon_intensity_per_lb",
                    intensity,
                    thresholds["warning"]
                ))

        # Check circularity score
        if dashboard.circularity:
            score = dashboard.circularity.circularity_score
            thresholds = self.alert_thresholds["circularity_score"]

            if score <= thresholds["critical"]:
                alerts.append(self._create_alert(
                    dashboard.organization_id,
                    MetricCategory.CIRCULARITY,
                    AlertSeverity.CRITICAL,
                    "Low Circularity Score",
                    f"Circularity score ({score:.1f}) below critical threshold",
                    "circularity_score",
                    score,
                    thresholds["critical"]
                ))
            elif score <= thresholds["warning"]:
                alerts.append(self._create_alert(
                    dashboard.organization_id,
                    MetricCategory.CIRCULARITY,
                    AlertSeverity.WARNING,
                    "Declining Circularity",
                    f"Circularity score ({score:.1f}) below warning threshold",
                    "circularity_score",
                    score,
                    thresholds["warning"]
                ))

        # Check soil health
        if dashboard.soil:
            om = dashboard.soil.organic_matter_percent
            thresholds = self.alert_thresholds["soil_organic_matter"]

            if om <= thresholds["critical"]:
                alerts.append(self._create_alert(
                    dashboard.organization_id,
                    MetricCategory.SOIL,
                    AlertSeverity.CRITICAL,
                    "Low Soil Organic Matter",
                    f"Organic matter ({om:.1f}%) critically low",
                    "organic_matter_percent",
                    om,
                    thresholds["critical"]
                ))
            elif om <= thresholds["warning"]:
                alerts.append(self._create_alert(
                    dashboard.organization_id,
                    MetricCategory.SOIL,
                    AlertSeverity.WARNING,
                    "Declining Soil Health",
                    f"Organic matter ({om:.1f}%) below optimal",
                    "organic_matter_percent",
                    om,
                    thresholds["warning"]
                ))

        # Store alerts
        for alert in alerts:
            self.alerts[alert.alert_id] = alert

        return alerts

    def _create_alert(
        self,
        org_id: str,
        category: MetricCategory,
        severity: AlertSeverity,
        title: str,
        description: str,
        metric_name: str,
        current_value: float,
        threshold_value: float
    ) -> SustainabilityAlert:
        """Create a sustainability alert"""

        alert_id = f"{org_id}_{category.value}_{metric_name}_{datetime.now().timestamp()}"

        return SustainabilityAlert(
            alert_id=alert_id,
            category=category,
            severity=severity,
            title=title,
            description=description,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now(),
            organization_id=org_id
        )

    def _estimate_farm_production(
        self,
        farm: Farm,
        start_date: date,
        end_date: date
    ) -> Decimal:
        """Estimate total food production in lbs"""

        # Production estimates per acre per year:
        # Urban farm: 10,000 lbs/acre
        # Market garden: 15,000 lbs/acre
        # Regenerative: 8,000 lbs/acre

        days_in_period = (end_date - start_date).days
        year_fraction = Decimal(str(days_in_period / 365.25))

        if farm.farm_type == FarmType.URBAN_FARM:
            lbs_per_acre = Decimal("10000")
        elif farm.farm_type == FarmType.MARKET_GARDEN:
            lbs_per_acre = Decimal("15000")
        else:  # Regenerative
            lbs_per_acre = Decimal("8000")

        return farm.total_acres * lbs_per_acre * year_fraction

    def generate_network_dashboard(
        self,
        start_date: date,
        end_date: date
    ) -> NetworkDashboard:
        """Generate aggregated dashboard for entire network"""

        dashboard = NetworkDashboard()

        # Count organizations
        all_farms = self.marketplace.farms.values()
        dashboard.total_organizations = len(all_farms)
        dashboard.active_organizations = len([
            f for f in all_farms if f.active
        ])

        # Aggregate metrics from all organization dashboards
        all_scores = []
        circularity_scores = []
        carbon_sequestration = []

        for farm in all_farms:
            if not farm.active:
                continue

            org_dash = self.generate_farm_dashboard(farm, start_date, end_date)

            all_scores.append(org_dash.overall_sustainability_score)

            if org_dash.carbon:
                carbon_sequestration.append((
                    farm.name,
                    org_dash.carbon.carbon_sequestered_lbs
                ))
                dashboard.total_carbon_sequestered_lbs += org_dash.carbon.carbon_sequestered_lbs

            if org_dash.water:
                dashboard.total_water_saved_gallons += org_dash.water.water_saved_gallons

            if org_dash.circularity:
                circularity_scores.append((
                    farm.name,
                    org_dash.circularity.circularity_score
                ))
                dashboard.total_food_diverted_from_waste_lbs += org_dash.circularity.food_scraps_diverted_lbs

            # Production
            production = self._estimate_farm_production(farm, start_date, end_date)
            dashboard.total_food_produced_lbs += production

            # Social impact
            if org_dash.social:
                dashboard.total_members_served += org_dash.social.csa_members_served
                dashboard.total_jobs_created += (
                    org_dash.social.full_time_jobs +
                    org_dash.social.seasonal_jobs
                )
                dashboard.total_volunteer_hours += org_dash.social.volunteer_hours

            # Collect critical alerts
            for alert in org_dash.active_alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    dashboard.critical_alerts.append(alert)

        # Calculate averages
        if all_scores:
            dashboard.average_sustainability_score = sum(all_scores) / len(all_scores)

        if circularity_scores:
            avg_circularity = sum(s for _, s in circularity_scores) / len(circularity_scores)
            dashboard.network_circularity_score = avg_circularity

        # Top performers
        dashboard.top_carbon_sequesterers = sorted(
            carbon_sequestration,
            key=lambda x: x[1],
            reverse=True
        )[:5]

        dashboard.top_circular_performers = sorted(
            circularity_scores,
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Biodiversity top performers
        biodiversity_scores = [
            (farm.name, self._calculate_biodiversity_metrics(farm).crop_varieties_count)
            for farm in all_farms if farm.active
        ]
        dashboard.top_biodiversity_farms = sorted(
            biodiversity_scores,
            key=lambda x: x[1],
            reverse=True
        )[:5]

        self.network_dashboard = dashboard
        return dashboard

    def export_dashboard(
        self,
        dashboard: OrganizationDashboard,
        format: str = "json"
    ) -> Dict:
        """Export dashboard data for external systems"""

        # Convert to dictionary for JSON serialization
        data = {
            "organization": {
                "id": dashboard.organization_id,
                "name": dashboard.organization_name,
                "type": dashboard.organization_type
            },
            "sustainability_score": dashboard.overall_sustainability_score,
            "certification_level": dashboard.certification_level,
            "last_updated": dashboard.last_updated.isoformat(),
            "metrics": {},
            "alerts": []
        }

        # Add metric categories
        if dashboard.carbon:
            data["metrics"]["carbon"] = {
                "net_impact_lbs": float(dashboard.carbon.net_carbon_impact_lbs),
                "sequestered_lbs": float(dashboard.carbon.carbon_sequestered_lbs),
                "emissions_lbs": float(dashboard.carbon.total_emissions_lbs),
                "intensity_per_lb": float(dashboard.carbon.carbon_intensity_per_lb)
            }

        if dashboard.water:
            data["metrics"]["water"] = {
                "total_used_gallons": float(dashboard.water.total_water_gallons),
                "saved_gallons": float(dashboard.water.water_saved_gallons),
                "efficiency_percent": dashboard.water.irrigation_efficiency,
                "gallons_per_lb_food": float(dashboard.water.water_per_lb_food)
            }

        if dashboard.circularity:
            data["metrics"]["circularity"] = {
                "score": dashboard.circularity.circularity_score,
                "waste_diversion_rate": dashboard.circularity.waste_diversion_rate,
                "carbon_avoided_lbs": float(dashboard.circularity.carbon_avoided_lbs)
            }

        # Add alerts
        for alert in dashboard.active_alerts:
            data["alerts"].append({
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "metric": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold_value
            })

        return data


# Factory functions
def create_transparency_dashboard(
    loop_tracker: Optional[LoopClosureTracker] = None,
    csa_coordinator: Optional[CSACoordinator] = None,
    marketplace: Optional[FarmMarketplace] = None,
    forecaster: Optional[DemandForecaster] = None
) -> TransparencyDashboard:
    """Create a transparency dashboard with optional service dependencies"""
    return TransparencyDashboard(
        loop_tracker=loop_tracker,
        csa_coordinator=csa_coordinator,
        marketplace=marketplace,
        forecaster=forecaster
    )

#!/usr/bin/env python3
"""
Farm Models for Farm Networks (Phase 3)

Represents different types of farms in the ecosystem:
- Urban Farms (small-scale, city-based)
- Market Gardens (medium-scale, diverse crops)
- Regenerative Farms (large-scale, sustainable practices)
- Community Gardens (cooperative management)
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import List, Dict, Optional, Set
from decimal import Decimal

from sous.models.organization import Organization, Location, OrganizationType


class FarmType(Enum):
    """Types of farms in the network"""
    URBAN_FARM = "urban_farm"
    MARKET_GARDEN = "market_garden"
    REGENERATIVE_FARM = "regenerative_farm"
    COMMUNITY_GARDEN = "community_garden"
    ROOFTOP_FARM = "rooftop_farm"
    GREENHOUSE = "greenhouse"
    AQUAPONICS = "aquaponics"
    VERTICAL_FARM = "vertical_farm"


class CertificationType(Enum):
    """Farm certifications"""
    ORGANIC = "organic"
    REGENERATIVE_ORGANIC = "regenerative_organic"
    BIODYNAMIC = "biodynamic"
    NATURALLY_GROWN = "naturally_grown"
    ANIMAL_WELFARE_APPROVED = "animal_welfare_approved"
    FAIR_TRADE = "fair_trade"
    RAINFOREST_ALLIANCE = "rainforest_alliance"
    NON_GMO = "non_gmo"
    CARBON_NEUTRAL = "carbon_neutral"


class FarmingMethod(Enum):
    """Farming methods and practices"""
    CONVENTIONAL = "conventional"
    ORGANIC = "organic"
    REGENERATIVE = "regenerative"
    PERMACULTURE = "permaculture"
    BIODYNAMIC = "biodynamic"
    NO_TILL = "no_till"
    AGROFORESTRY = "agroforestry"
    ROTATIONAL_GRAZING = "rotational_grazing"
    AQUAPONICS = "aquaponics"
    HYDROPONICS = "hydroponics"


class CropType(Enum):
    """Types of crops grown"""
    VEGETABLES = "vegetables"
    FRUITS = "fruits"
    HERBS = "herbs"
    GRAINS = "grains"
    LEGUMES = "legumes"
    ROOT_VEGETABLES = "root_vegetables"
    LEAFY_GREENS = "leafy_greens"
    MUSHROOMS = "mushrooms"
    FLOWERS = "flowers"
    MICROGREENS = "microgreens"


@dataclass
class GrowingSeason:
    """Growing season information"""
    name: str  # e.g., "Spring 2024"
    start_date: date
    end_date: date
    crops_planned: List[str] = field(default_factory=list)
    expected_yield_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    actual_yield_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    active: bool = True


@dataclass
class SustainabilityMetrics:
    """Sustainability and environmental impact metrics"""
    # Carbon
    carbon_sequestered_tons_per_year: Decimal = field(default_factory=lambda: Decimal("0.0"))
    carbon_emissions_tons_per_year: Decimal = field(default_factory=lambda: Decimal("0.0"))

    # Water
    water_usage_gallons_per_year: Decimal = field(default_factory=lambda: Decimal("0.0"))
    rainwater_harvested_gallons_per_year: Decimal = field(default_factory=lambda: Decimal("0.0"))

    # Soil
    soil_organic_matter_percent: float = 0.0
    topsoil_depth_inches: float = 0.0

    # Biodiversity
    species_count: int = 0
    pollinator_habitat_acres: Decimal = field(default_factory=lambda: Decimal("0.0"))

    # Waste
    compost_produced_tons_per_year: Decimal = field(default_factory=lambda: Decimal("0.0"))
    plastic_waste_lbs_per_year: Decimal = field(default_factory=lambda: Decimal("0.0"))

    # Energy
    renewable_energy_percent: float = 0.0

    def get_net_carbon(self) -> Decimal:
        """Calculate net carbon impact (negative is carbon positive)"""
        return self.carbon_emissions_tons_per_year - self.carbon_sequestered_tons_per_year

    def get_water_efficiency(self) -> float:
        """Calculate water efficiency ratio"""
        if self.water_usage_gallons_per_year == 0:
            return 0.0
        return float(self.rainwater_harvested_gallons_per_year / self.water_usage_gallons_per_year)

    def get_sustainability_score(self) -> float:
        """
        Calculate overall sustainability score (0-100).

        Weighted combination of multiple factors.
        """
        score = 0.0

        # Carbon score (30%)
        net_carbon = float(self.get_net_carbon())
        if net_carbon <= 0:  # Carbon positive
            score += 30.0
        elif net_carbon <= 5:  # Low emissions
            score += 20.0
        elif net_carbon <= 10:  # Moderate emissions
            score += 10.0

        # Water efficiency (20%)
        water_eff = self.get_water_efficiency()
        score += min(20.0, water_eff * 20.0)

        # Soil health (20%)
        if self.soil_organic_matter_percent >= 5.0:
            score += 20.0
        elif self.soil_organic_matter_percent >= 3.0:
            score += 15.0
        elif self.soil_organic_matter_percent >= 2.0:
            score += 10.0

        # Biodiversity (15%)
        if self.species_count >= 100:
            score += 15.0
        elif self.species_count >= 50:
            score += 10.0
        elif self.species_count >= 20:
            score += 5.0

        # Renewable energy (15%)
        score += self.renewable_energy_percent * 0.15

        return min(100.0, score)


@dataclass
class Farm(Organization):
    """
    Base farm organization.

    Extends Organization with farm-specific attributes.
    """
    farm_type: FarmType = FarmType.MARKET_GARDEN

    # Size and capacity
    total_acres: Decimal = field(default_factory=lambda: Decimal("1.0"))
    cultivated_acres: Decimal = field(default_factory=lambda: Decimal("0.5"))

    # Farming practices
    farming_methods: List[FarmingMethod] = field(default_factory=list)
    certifications: List[CertificationType] = field(default_factory=list)

    # Crops and production
    crops_grown: List[CropType] = field(default_factory=list)
    annual_production_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    growing_seasons: List[GrowingSeason] = field(default_factory=list)

    # Sustainability
    sustainability_metrics: Optional[SustainabilityMetrics] = None

    # Market presence
    sells_at_farmers_markets: bool = False
    has_farm_stand: bool = False
    offers_csa: bool = False
    sells_to_restaurants: bool = False
    sells_to_grocery_stores: bool = False

    # Community engagement
    offers_farm_tours: bool = False
    offers_workshops: bool = False
    volunteer_program: bool = False

    # Loop closure (Phase 3 feature)
    accepts_food_scraps: bool = False
    accepts_compost: bool = False
    compost_capacity_lbs_per_week: Decimal = field(default_factory=lambda: Decimal("0.0"))

    def __post_init__(self):
        """Initialize farm defaults"""
        super().__post_init__()
        self.org_type = OrganizationType.FARM

        # Farms can both donate (surplus) and receive (food scraps for compost)
        self.can_donate = True
        self.can_receive = True

        # Initialize sustainability metrics if not provided
        if self.sustainability_metrics is None:
            self.sustainability_metrics = SustainabilityMetrics()

    def get_land_utilization(self) -> float:
        """Calculate percentage of land currently cultivated"""
        if self.total_acres == 0:
            return 0.0
        return float(self.cultivated_acres / self.total_acres * 100)

    def get_productivity_per_acre(self) -> Decimal:
        """Calculate production per cultivated acre"""
        if self.cultivated_acres == 0:
            return Decimal("0.0")
        return self.annual_production_lbs / self.cultivated_acres

    def get_current_season(self) -> Optional[GrowingSeason]:
        """Get currently active growing season"""
        today = date.today()
        for season in self.growing_seasons:
            if season.active and season.start_date <= today <= season.end_date:
                return season
        return None

    def is_certified_organic(self) -> bool:
        """Check if farm has organic certification"""
        return CertificationType.ORGANIC in self.certifications

    def is_regenerative(self) -> bool:
        """Check if farm uses regenerative practices"""
        return (FarmingMethod.REGENERATIVE in self.farming_methods or
                CertificationType.REGENERATIVE_ORGANIC in self.certifications)


@dataclass
class UrbanFarm(Farm):
    """
    Urban farm - small-scale, city-based agriculture.

    Characteristics:
    - Small acreage (<1 acre typically)
    - High-value crops (microgreens, herbs, specialty vegetables)
    - Close to consumers
    - Often uses intensive growing methods
    """
    # Urban-specific
    rooftop_location: bool = False
    vertical_growing: bool = False
    indoor_growing: bool = False

    # Logistics
    delivery_by_bike: bool = False
    same_day_delivery: bool = False

    def __post_init__(self):
        """Initialize urban farm defaults"""
        super().__post_init__()
        self.farm_type = FarmType.URBAN_FARM

        # Urban farms typically small
        if self.total_acres == Decimal("1.0"):  # Default value
            self.total_acres = Decimal("0.25")
            self.cultivated_acres = Decimal("0.20")

        # High-value crops typical
        if not self.crops_grown:
            self.crops_grown = [
                CropType.MICROGREENS,
                CropType.HERBS,
                CropType.LEAFY_GREENS
            ]


@dataclass
class MarketGarden(Farm):
    """
    Market garden - medium-scale, diverse crop production.

    Characteristics:
    - 1-10 acres typically
    - Diverse crops for local markets
    - Intensive cultivation
    - Direct-to-consumer sales focus
    """
    # Market garden specific
    crop_rotation_plan: bool = False
    succession_planting: bool = False

    # Infrastructure
    greenhouse_sqft: int = 0
    cold_frames_count: int = 0
    irrigation_system: str = ""  # drip, overhead, etc.

    # Sales channels
    farmers_market_locations: List[str] = field(default_factory=list)
    csa_member_count: int = 0
    restaurant_clients_count: int = 0

    def __post_init__(self):
        """Initialize market garden defaults"""
        super().__post_init__()
        self.farm_type = FarmType.MARKET_GARDEN

        # Typical size
        if self.total_acres == Decimal("1.0"):  # Default value
            self.total_acres = Decimal("5.0")
            self.cultivated_acres = Decimal("4.0")

        # Diverse crops
        if not self.crops_grown:
            self.crops_grown = [
                CropType.VEGETABLES,
                CropType.LEAFY_GREENS,
                CropType.ROOT_VEGETABLES,
                CropType.HERBS,
                CropType.FLOWERS
            ]

        # Typical sales channels
        self.sells_at_farmers_markets = True
        self.offers_csa = True


@dataclass
class RegenerativeFarm(Farm):
    """
    Regenerative farm - large-scale, ecosystem-focused agriculture.

    Characteristics:
    - Builds soil health
    - Sequesters carbon
    - Enhances biodiversity
    - Holistic ecosystem management
    - Often includes livestock
    """
    # Regenerative practices
    cover_cropping: bool = False
    composting_program: bool = False
    wildlife_corridors: bool = False
    hedgerows: bool = False

    # Livestock (if any)
    has_livestock: bool = False
    livestock_types: List[str] = field(default_factory=list)
    rotational_grazing: bool = False

    # Carbon tracking
    carbon_credits_generated: Decimal = field(default_factory=lambda: Decimal("0.0"))
    carbon_credits_sold: Decimal = field(default_factory=lambda: Decimal("0.0"))

    # Water management
    has_ponds: bool = False
    swales_count: int = 0
    rainwater_harvesting: bool = False

    def __post_init__(self):
        """Initialize regenerative farm defaults"""
        super().__post_init__()
        self.farm_type = FarmType.REGENERATIVE_FARM

        # Larger scale
        if self.total_acres == Decimal("1.0"):  # Default value
            self.total_acres = Decimal("50.0")
            self.cultivated_acres = Decimal("40.0")

        # Regenerative methods
        if not self.farming_methods:
            self.farming_methods = [
                FarmingMethod.REGENERATIVE,
                FarmingMethod.NO_TILL,
                FarmingMethod.ROTATIONAL_GRAZING
            ]

        # Likely certified
        if not self.certifications:
            self.certifications = [
                CertificationType.REGENERATIVE_ORGANIC,
                CertificationType.ANIMAL_WELFARE_APPROVED
            ]

        # Accepts food scraps for composting at scale
        self.accepts_food_scraps = True
        self.accepts_compost = True
        self.compost_capacity_lbs_per_week = Decimal("1000.0")


@dataclass
class CommunityGarden(Farm):
    """
    Community garden - cooperative/shared growing space.

    Characteristics:
    - Shared plots among members
    - Educational focus
    - Community building
    - Non-profit typically
    """
    # Community structure
    plot_count: int = 0
    member_count: int = 0
    waiting_list_count: int = 0

    # Shared resources
    shared_tools: bool = False
    shared_compost: bool = False
    shared_water_access: bool = False

    # Programs
    youth_program: bool = False
    senior_program: bool = False
    donation_plot_count: int = 0  # Plots dedicated to food bank donations

    # Governance
    member_managed: bool = True
    nonprofit_status: bool = False

    def __post_init__(self):
        """Initialize community garden defaults"""
        super().__post_init__()
        self.farm_type = FarmType.COMMUNITY_GARDEN

        # Small shared space
        if self.total_acres == Decimal("1.0"):  # Default value
            self.total_acres = Decimal("2.0")
            self.cultivated_acres = Decimal("1.5")

        # Community focus
        self.offers_farm_tours = True
        self.offers_workshops = True
        self.volunteer_program = True


# ========================================================================
# Factory Functions
# ========================================================================

def create_urban_farm(
    name: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    total_acres: Decimal = Decimal("0.25"),
    rooftop: bool = False
) -> UrbanFarm:
    """Create an urban farm"""
    location = Location(
        address=address,
        city=city,
        state=state,
        zip_code=zip_code
    )

    org_id = f"urban_{name.lower().replace(' ', '_')}"

    return UrbanFarm(
        org_id=org_id,
        name=name,
        org_type=OrganizationType.FARM,
        location=location,
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        farm_type=FarmType.URBAN_FARM,
        total_acres=total_acres,
        cultivated_acres=total_acres * Decimal("0.8"),
        rooftop_location=rooftop,
        same_day_delivery=True,
        offers_csa=True
    )


def create_market_garden(
    name: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    total_acres: Decimal = Decimal("5.0"),
    csa_members: int = 0
) -> MarketGarden:
    """Create a market garden"""
    location = Location(
        address=address,
        city=city,
        state=state,
        zip_code=zip_code
    )

    org_id = f"garden_{name.lower().replace(' ', '_')}"

    return MarketGarden(
        org_id=org_id,
        name=name,
        org_type=OrganizationType.FARM,
        location=location,
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        farm_type=FarmType.MARKET_GARDEN,
        total_acres=total_acres,
        cultivated_acres=total_acres * Decimal("0.9"),
        csa_member_count=csa_members,
        sells_at_farmers_markets=True,
        offers_csa=True,
        crop_rotation_plan=True,
        succession_planting=True
    )


def create_regenerative_farm(
    name: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    total_acres: Decimal = Decimal("100.0"),
    has_livestock: bool = True
) -> RegenerativeFarm:
    """Create a regenerative farm"""
    location = Location(
        address=address,
        city=city,
        state=state,
        zip_code=zip_code
    )

    org_id = f"regen_{name.lower().replace(' ', '_')}"

    # Strong sustainability metrics for regenerative farms
    sustainability = SustainabilityMetrics(
        carbon_sequestered_tons_per_year=Decimal("50.0"),
        carbon_emissions_tons_per_year=Decimal("10.0"),
        water_usage_gallons_per_year=Decimal("1000000.0"),
        rainwater_harvested_gallons_per_year=Decimal("500000.0"),
        soil_organic_matter_percent=5.5,
        topsoil_depth_inches=12.0,
        species_count=150,
        pollinator_habitat_acres=Decimal("10.0"),
        compost_produced_tons_per_year=Decimal("100.0"),
        plastic_waste_lbs_per_year=Decimal("50.0"),
        renewable_energy_percent=75.0
    )

    return RegenerativeFarm(
        org_id=org_id,
        name=name,
        org_type=OrganizationType.FARM,
        location=location,
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        farm_type=FarmType.REGENERATIVE_FARM,
        total_acres=total_acres,
        cultivated_acres=total_acres * Decimal("0.7"),
        has_livestock=has_livestock,
        sustainability_metrics=sustainability,
        cover_cropping=True,
        composting_program=True,
        wildlife_corridors=True,
        rotational_grazing=has_livestock,
        accepts_food_scraps=True,
        accepts_compost=True,
        compost_capacity_lbs_per_week=Decimal("2000.0")
    )

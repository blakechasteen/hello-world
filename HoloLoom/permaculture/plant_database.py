"""
Plant Database - Core data models for permaculture plants

This module defines the data structures for plants, guilds, and companion
relationships in permaculture design.

**Created: 2025-11-21**
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set
import json


class Layer(Enum):
    """Vertical layers in a permaculture guild"""
    CANOPY = "canopy"  # Tall trees (20-40+ ft)
    SUB_CANOPY = "sub_canopy"  # Small trees (10-20 ft)
    SHRUB = "shrub"  # Shrubs (4-10 ft)
    HERBACEOUS = "herbaceous"  # Herbs, vegetables (1-4 ft)
    GROUND_COVER = "ground_cover"  # Ground-hugging plants (<1 ft)
    VINE = "vine"  # Climbing plants
    ROOT = "root"  # Root crops, bulbs
    AQUATIC = "aquatic"  # Water plants


class Function(Enum):
    """Functional roles plants can play in a guild"""
    FOOD = "food"  # Produces edible yield
    NITROGEN_FIXER = "nitrogen_fixer"  # Fixes nitrogen (legumes)
    DYNAMIC_ACCUMULATOR = "dynamic_accumulator"  # Deep roots bring up nutrients
    POLLINATOR_ATTRACTOR = "pollinator_attractor"  # Attracts bees, butterflies
    PEST_DETERRENT = "pest_deterrent"  # Repels harmful insects
    BENEFICIAL_INSECT_HABITAT = "beneficial_insect_habitat"  # Attracts predatory insects
    MULCH = "mulch"  # Produces biomass for mulching
    WINDBREAK = "windbreak"  # Blocks wind
    SHADE = "shade"  # Provides shade for understory
    GROUNDCOVER = "groundcover"  # Prevents erosion, suppresses weeds
    MEDICINAL = "medicinal"  # Medicinal properties
    AROMATIC = "aromatic"  # Fragrant, often deters pests
    WILDLIFE_HABITAT = "wildlife_habitat"  # Supports birds, beneficial animals
    SOIL_BUILDER = "soil_builder"  # Improves soil structure


class RelationshipType(Enum):
    """Types of companion relationships"""
    BENEFICIAL = "beneficial"  # Positive interaction
    NEUTRAL = "neutral"  # No significant interaction
    ANTAGONISTIC = "antagonistic"  # Negative interaction


@dataclass
class Plant:
    """
    Comprehensive plant data for permaculture design

    Attributes:
        common_name: Common name (e.g., "Apple")
        scientific_name: Botanical name (e.g., "Malus domestica")
        family: Plant family (e.g., "Rosaceae")
        layers: Vertical layers occupied
        functions: Functional roles in guild
        hardiness_zones: USDA zones (e.g., [3, 4, 5, 6, 7, 8])
        sun_requirements: "full_sun", "partial_shade", "full_shade"
        water_needs: "low", "moderate", "high"
        soil_ph: Preferred pH range (e.g., [6.0, 7.0])
        height_ft: Mature height in feet
        spread_ft: Mature spread in feet
        root_depth: "shallow", "medium", "deep"
        lifecycle: "annual", "perennial", "biennial"
        lifespan_years: Lifespan for perennials
        native_to: Geographic origin
        description: Text description of plant
        yield_info: Information about edible/useful yield
        propagation: Methods of propagation
        companions: Known companion plants (common names)
        antagonists: Plants to avoid pairing with
    """
    common_name: str
    scientific_name: str
    family: str
    layers: List[Layer]
    functions: List[Function]
    hardiness_zones: List[int]
    sun_requirements: str  # "full_sun", "partial_shade", "full_shade"
    water_needs: str  # "low", "moderate", "high"
    soil_ph: List[float]  # [min, max]
    height_ft: float
    spread_ft: float
    root_depth: str  # "shallow", "medium", "deep"
    lifecycle: str  # "annual", "perennial", "biennial"
    lifespan_years: Optional[int] = None
    native_to: Optional[str] = None
    description: str = ""
    yield_info: str = ""
    propagation: List[str] = field(default_factory=list)
    companions: List[str] = field(default_factory=list)
    antagonists: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "common_name": self.common_name,
            "scientific_name": self.scientific_name,
            "family": self.family,
            "layers": [layer.value for layer in self.layers],
            "functions": [func.value for func in self.functions],
            "hardiness_zones": self.hardiness_zones,
            "sun_requirements": self.sun_requirements,
            "water_needs": self.water_needs,
            "soil_ph": self.soil_ph,
            "height_ft": self.height_ft,
            "spread_ft": self.spread_ft,
            "root_depth": self.root_depth,
            "lifecycle": self.lifecycle,
            "lifespan_years": self.lifespan_years,
            "native_to": self.native_to,
            "description": self.description,
            "yield_info": self.yield_info,
            "propagation": self.propagation,
            "companions": self.companions,
            "antagonists": self.antagonists,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Plant":
        """Create Plant from dictionary"""
        # Convert enum strings back to enums
        data["layers"] = [Layer(layer) for layer in data["layers"]]
        data["functions"] = [Function(func) for func in data["functions"]]
        return cls(**data)


@dataclass
class CompanionRelationship:
    """Relationship between two plants"""
    plant_a: str  # Common name
    plant_b: str  # Common name
    relationship: RelationshipType
    reason: str  # Why they're compatible/incompatible


@dataclass
class GuildRole:
    """A plant's role within a specific guild"""
    plant: Plant
    primary_function: Function
    notes: str = ""


@dataclass
class Guild:
    """
    A permaculture guild - mutually beneficial plant community

    A guild is designed around a central element (often a fruit/nut tree)
    with supporting plants in different layers providing complementary functions.
    """
    name: str
    central_plant: Plant
    members: List[GuildRole]
    description: str = ""
    climate_zones: List[int] = field(default_factory=list)

    def get_layer_diversity(self) -> Dict[Layer, int]:
        """Count plants in each layer"""
        layer_counts = {}
        for role in self.members:
            for layer in role.plant.layers:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return layer_counts

    def get_function_diversity(self) -> Dict[Function, int]:
        """Count functional roles"""
        function_counts = {}
        for role in self.members:
            function_counts[role.primary_function] = \
                function_counts.get(role.primary_function, 0) + 1
        return function_counts

    def get_all_plants(self) -> List[Plant]:
        """Get all plants including central plant"""
        return [self.central_plant] + [role.plant for role in self.members]


class PlantDatabase:
    """
    In-memory plant database with search capabilities
    """

    def __init__(self):
        self.plants: Dict[str, Plant] = {}  # common_name -> Plant
        self.by_family: Dict[str, List[Plant]] = {}
        self.by_function: Dict[Function, List[Plant]] = {}
        self.by_layer: Dict[Layer, List[Plant]] = {}

    def add_plant(self, plant: Plant) -> None:
        """Add a plant to the database"""
        self.plants[plant.common_name.lower()] = plant

        # Index by family
        if plant.family not in self.by_family:
            self.by_family[plant.family] = []
        self.by_family[plant.family].append(plant)

        # Index by function
        for function in plant.functions:
            if function not in self.by_function:
                self.by_function[function] = []
            self.by_function[function].append(plant)

        # Index by layer
        for layer in plant.layers:
            if layer not in self.by_layer:
                self.by_layer[layer] = []
            self.by_layer[layer].append(plant)

    def get_plant(self, common_name: str) -> Optional[Plant]:
        """Get plant by common name (case-insensitive)"""
        return self.plants.get(common_name.lower())

    def search_by_function(self, function: Function) -> List[Plant]:
        """Find all plants with a specific function"""
        return self.by_function.get(function, [])

    def search_by_layer(self, layer: Layer) -> List[Plant]:
        """Find all plants in a specific layer"""
        return self.by_layer.get(layer, [])

    def search_by_zone(self, zone: int) -> List[Plant]:
        """Find all plants suitable for a hardiness zone"""
        return [
            plant for plant in self.plants.values()
            if zone in plant.hardiness_zones
        ]

    def get_companions(self, plant_name: str) -> List[Plant]:
        """Get known companions of a plant"""
        plant = self.get_plant(plant_name)
        if not plant:
            return []

        companions = []
        for companion_name in plant.companions:
            companion = self.get_plant(companion_name)
            if companion:
                companions.append(companion)
        return companions

    def get_antagonists(self, plant_name: str) -> List[Plant]:
        """Get known antagonists of a plant"""
        plant = self.get_plant(plant_name)
        if not plant:
            return []

        antagonists = []
        for antagonist_name in plant.antagonists:
            antagonist = self.get_plant(antagonist_name)
            if antagonist:
                antagonists.append(antagonist)
        return antagonists

    def save_to_json(self, filepath: str) -> None:
        """Save database to JSON file"""
        data = {
            "plants": [plant.to_dict() for plant in self.plants.values()]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str) -> "PlantDatabase":
        """Load database from JSON file"""
        db = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)

        for plant_data in data["plants"]:
            plant = Plant.from_dict(plant_data)
            db.add_plant(plant)

        return db

    def __len__(self) -> int:
        return len(self.plants)

    def __repr__(self) -> str:
        return f"PlantDatabase({len(self)} plants)"

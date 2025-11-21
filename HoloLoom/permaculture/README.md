# Permaculture Design Toolset

**Status**: ✅ Complete (Phase 1 - Plant Database & Guild Engine)
**Created**: 2025-11-21

A permaculture guild recommendation system powered by HoloLoom's RAG (Retrieval-Augmented Generation) technology.

## Overview

This toolset helps permaculture designers create optimal plant guilds by:
- Leveraging a comprehensive plant database with 20+ species
- Using semantic search to find companion plants
- Applying permaculture principles (layer diversity, functional stacking)
- Providing intelligent guild recommendations based on climate zones

## Architecture

```
HoloLoom/permaculture/
├── plant_database.py      # Data models (Plant, Guild, Function, Layer)
├── guild_engine.py        # RAG-powered recommendation engine
├── data/
│   └── plants.json        # 20 starter plants with full permaculture data
└── README.md              # This file
```

## Core Concepts

### Plant Database

Each plant includes:
- **Basic Info**: Common name, scientific name, family
- **Growing Requirements**: Zones, sun, water, soil pH
- **Physical Characteristics**: Height, spread, root depth
- **Functions**: Nitrogen fixer, pest deterrent, pollinator attractor, etc.
- **Layers**: Canopy, shrub, herbaceous, ground cover, root
- **Relationships**: Known companions and antagonists

### Guild Recommendation

The engine recommends guilds based on:
1. **Central Plant** - The main element (often a fruit/nut tree)
2. **Layer Diversity** - Vertical stacking (canopy → ground cover)
3. **Function Diversity** - Key permaculture functions covered
4. **Companion Compatibility** - Known beneficial relationships
5. **Climate Compatibility** - USDA hardiness zones

### Scoring System

Guild recommendations receive scores for:
- **Layer Diversity**: Coverage of target vertical layers (0-100%)
- **Function Diversity**: Coverage of target functions (0-100%)
- **Companion Compatibility**: % of known companions used (0-100%)
- **Overall Score**: Weighted average (30% layer + 40% function + 30% compatibility)

## Quick Start

### 1. Load Plant Database

```python
from HoloLoom.permaculture import PlantDatabase

# Load from JSON
db = PlantDatabase.load_from_json("HoloLoom/permaculture/data/plants.json")
print(f"Loaded {len(db)} plants")

# Search by function
nitrogen_fixers = db.search_by_function(Function.NITROGEN_FIXER)
print(f"Found {len(nitrogen_fixers)} nitrogen fixers")

# Get companions
apple = db.get_plant("Apple")
companions = db.get_companions("Apple")
print(f"Apple companions: {[c.common_name for c in companions]}")
```

### 2. Generate Guild Recommendations

```python
from HoloLoom.permaculture import GuildRecommendationEngine
from HoloLoom.config import Config

# Initialize with HoloLoom RAG
config = Config.fast()
async with GuildRecommendationEngine(db, config) as engine:
    # Recommend guild for apple tree in zone 6
    recommendation = await engine.recommend_guild(
        central_plant_name="Apple",
        climate_zone=6,
        max_plants=7
    )

    print(f"Guild Score: {recommendation.score:.1%}")
    print(f"Members: {len(recommendation.guild.members)}")

    for role in recommendation.guild.members:
        print(f"  • {role.plant.common_name} - {role.primary_function.value}")
```

### 3. Semantic Plant Search

```python
async with GuildRecommendationEngine(db, config) as engine:
    # Natural language query
    results = await engine.query_plants(
        "nitrogen fixing ground covers for zone 5",
        limit=3
    )

    for plant in results:
        print(f"{plant.common_name}: {plant.description}")
```

## Running the Demo

```bash
PYTHONPATH=. python demos/demo_permaculture_guild.py
```

The demo includes:
1. **Plant Database Demo** - Load and explore 20 plants
2. **Guild Recommendation** - Generate Apple tree guild for zone 6
3. **Semantic Search** - Natural language plant queries
4. **Custom Guild** - Guild with specific requirements

## Plant Database (Starter Set)

Our starter database includes 20 plants covering all major guild roles:

**Trees & Shrubs** (4):
- Apple, Hazelnut, Red Currant, Blackberry

**Nitrogen Fixers** (3):
- White Clover, Lupine, Hazelnut

**Dynamic Accumulators** (3):
- Comfrey, Yarrow, Borage

**Pest Deterrents** (6):
- Chives, Nasturtium, Daffodil, Garlic, Marigold, Hyssop

**Ground Covers** (5):
- White Clover, Nasturtium, Strawberry, Thyme, Mint

**Pollinators** (7):
- Yarrow, Borage, Lavender, Hyssop, Chives, Marigold, Thyme

## Example Guild: Apple Tree (Zone 6)

```
🍎 Apple (Canopy Layer) - Central Element
  ├── Comfrey (Herbaceous) - Dynamic accumulator
  ├── White Clover (Ground Cover) - Nitrogen fixer
  ├── Chives (Herbaceous) - Pest deterrent
  ├── Nasturtium (Ground Cover) - Trap crop
  ├── Yarrow (Herbaceous) - Pollinator attractor
  ├── Daffodil (Herbaceous) - Vole/deer deterrent
  └── Red Currant (Shrub) - Shade-tolerant berry

Score: 85% (Layer: 80% | Function: 90% | Compatibility: 86%)
```

## Extending the System

### Adding New Plants

1. Edit `HoloLoom/permaculture/data/plants.json`
2. Follow the schema (see existing plants for examples)
3. Reload the database

### Custom Guild Requirements

```python
from HoloLoom.permaculture import Function, Layer

# Focus on food production + pest control
target_functions = {
    Function.FOOD,
    Function.PEST_DETERRENT,
    Function.POLLINATOR_ATTRACTOR,
}

# Only herbaceous and ground cover layers
target_layers = {
    Layer.HERBACEOUS,
    Layer.GROUND_COVER,
}

recommendation = await engine.recommend_guild(
    central_plant_name="Hazelnut",
    climate_zone=7,
    target_functions=target_functions,
    target_layers=target_layers,
    max_plants=6
)
```

## Data Model

### Plant

```python
@dataclass
class Plant:
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
    # ... plus companions, antagonists, description, yield_info, etc.
```

### Guild

```python
@dataclass
class Guild:
    name: str
    central_plant: Plant
    members: List[GuildRole]
    description: str
    climate_zones: List[int]
```

### Enums

```python
class Layer(Enum):
    CANOPY = "canopy"
    SUB_CANOPY = "sub_canopy"
    SHRUB = "shrub"
    HERBACEOUS = "herbaceous"
    GROUND_COVER = "ground_cover"
    VINE = "vine"
    ROOT = "root"
    AQUATIC = "aquatic"

class Function(Enum):
    FOOD = "food"
    NITROGEN_FIXER = "nitrogen_fixer"
    DYNAMIC_ACCUMULATOR = "dynamic_accumulator"
    POLLINATOR_ATTRACTOR = "pollinator_attractor"
    PEST_DETERRENT = "pest_deterrent"
    # ... 9 more functions
```

## Integration with HoloLoom

The guild engine leverages HoloLoom's:
- **Memory System**: Stores all plant knowledge as embeddings
- **RAG (Retrieval-Augmented Generation)**: Semantic search for companions
- **Semantic Calculus**: 228D semantic space for plant relationships
- **Async Architecture**: Efficient concurrent operations

## Future Enhancements

**Phase 2** (Planned):
- Site analysis tools (zones, sectors, topography)
- Water management design (swales, ponds, catchment)
- Design pattern library (food forests, mandala gardens)
- Visual guild designer (web dashboard)

**Phase 3** (Planned):
- SpinningWheel adapters for web scraping permaculture resources
- Integration with USDA Plants database
- User-contributed designs
- GIS/mapping integration

**Phase 4** (Planned):
- Succession planning (pioneer → climax species)
- Seasonal planting calendars
- Yield prediction modeling
- Multi-guild site design

## Contributing

To add plants to the database:
1. Research the plant's permaculture attributes
2. Add to `data/plants.json` following the existing schema
3. Include known companions and antagonists
4. Document functions and layer occupancy
5. Test with the demo

## References

This system is based on permaculture principles from:
- **Gaia's Garden** by Toby Hemenway
- **Edible Forest Gardens** by Dave Jacke
- **The Permaculture Handbook** by Peter Bane
- Traditional companion planting wisdom

## License

Part of the HoloLoom project. See main repository LICENSE.

## Acknowledgments

Built using HoloLoom's RAG architecture by leveraging:
- Multi-scale Matryoshka embeddings
- Knowledge graph memory
- Semantic search capabilities
- Thompson Sampling for exploration

Happy permaculture designing! 🌱🌳🌻

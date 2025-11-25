"""
Dream Visualizer Bridge - Python bridge between dream data and Three.js renderer

Converts dream scene data from NeuroHood's symbolic dream system into
Three.js-compatible JSON format with:
- 3D model mappings
- Cinematic references from enriched symbol database
- Metaphorical physics parameters
- Atmospheric configuration

Architecture:
- Reads enriched symbol database (JSON)
- Maps symbols to GLTF models or procedural geometry
- Extracts cinematic references (modern_cinema, classic_film)
- Infers lighting/camera from film references
- Generates atmospheric effects configuration

Author: Claude Code
Date: 2025-11-24
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

@dataclass
class Vector3:
    x: float
    y: float
    z: float


@dataclass
class VisualProperties:
    base_form: str  # 'organic' | 'geometric' | 'abstract'
    primary_color: str  # Hex color
    secondary_color: Optional[str] = None
    material_type: str = 'solid'  # 'solid' | 'translucent' | 'ethereal'
    scale: float = 1.0


@dataclass
class MetaphoricalPhysics:
    type: str  # 'solidification' | 'dissolution' | 'reactive' | 'gravitational'
    triggers: Dict[str, str]  # {'emotion': 'trust', 'proximity': 'freedom'}
    parameters: Dict[str, float]  # {'base_rate': 0.5, 'max_rate': 2.0}


@dataclass
class CinematicReference:
    film: str
    scene: str
    visual_elements: List[str]
    lighting: Dict[str, Any]
    camera: Dict[str, Any]
    atmosphere: Dict[str, Any]


@dataclass
class DreamSetting:
    environment: str  # 'prison' | 'forest' | 'ocean' | 'void'
    lighting: str  # 'twilight' | 'stormy' | 'starry' | 'harsh_sunlight'
    weather: Optional[str] = None
    atmosphere: str = 'neutral'  # 'oppressive' | 'serene' | 'chaotic'


@dataclass
class SymbolArchetype:
    symbol_id: str
    name: str
    category: str
    visual_properties: VisualProperties
    metaphorical_physics: MetaphoricalPhysics
    cinematic_references: List[CinematicReference]


@dataclass
class DreamScene:
    scene_id: str
    title: str
    symbols: List[SymbolArchetype]
    setting: DreamSetting
    participants: List[Dict[str, Any]]
    duration_seconds: float
    consciousness_level: float  # 0.0-1.0


# ============================================================================
# BRIDGE CLASS
# ============================================================================

class DreamVisualizerBridge:
    """
    Converts dream scene data to Three.js-compatible format.
    """

    def __init__(self, symbol_database_path: Optional[Path] = None):
        """
        Initialize bridge with enriched symbol database.

        Args:
            symbol_database_path: Path to enriched JSON database
        """
        self.symbol_database_path = symbol_database_path or Path("NeuroHood/dreams/symbol_database_pilot_enriched.json")
        self.symbol_database = self._load_symbol_database()

    def _load_symbol_database(self) -> Dict:
        """Load enriched symbol database from JSON."""
        if not self.symbol_database_path.exists():
            print(f"Warning: Symbol database not found at {self.symbol_database_path}")
            return {}

        with open(self.symbol_database_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Convert list to dict keyed by symbol_id
            if isinstance(data, list):
                return {item['symbol_id']: item for item in data}
            return data

    def prepare_scene_data(self, dream_scene: DreamScene) -> Dict:
        """
        Convert dream scene to Three.js-ready JSON.

        Args:
            dream_scene: Dream scene object

        Returns:
            Three.js-compatible scene configuration
        """
        return {
            "scene_id": dream_scene.scene_id,
            "title": dream_scene.title,
            "symbols": [self._symbol_to_3d(s) for s in dream_scene.symbols],
            "setting": asdict(dream_scene.setting),
            "cinematic": self._extract_cinematic_refs(dream_scene),
            "atmosphere": self._atmosphere_config(dream_scene),
            "participants": dream_scene.participants,
            "duration_seconds": dream_scene.duration_seconds,
            "consciousness_level": dream_scene.consciousness_level
        }

    def _symbol_to_3d(self, symbol: SymbolArchetype) -> Dict:
        """
        Map symbol to 3D representation.

        Args:
            symbol: Symbol archetype object

        Returns:
            3D object configuration
        """
        return {
            "id": symbol.symbol_id,
            "name": symbol.name,
            "category": symbol.category,
            "model": self._get_model_path(symbol),
            "position": [0, 0, 0],  # Set by layout algorithm in Three.js
            "scale": [symbol.visual_properties.scale] * 3,
            "visual_properties": asdict(symbol.visual_properties),
            "metaphorical_physics": asdict(symbol.metaphorical_physics),
            "cinematic_references": [asdict(ref) for ref in symbol.cinematic_references]
        }

    def _get_model_path(self, symbol: SymbolArchetype) -> str:
        """
        Get GLTF model path for symbol.

        Args:
            symbol: Symbol archetype

        Returns:
            Relative path to GLTF model (or 'procedural' for generated geometry)
        """
        # Map symbol names to GLTF files
        model_mapping = {
            'caged_bird': '/dream_models/caged_bird.glb',
            'storm_cloud': '/dream_models/storm_cloud.glb',
            'bridge': '/dream_models/bridge.glb',
            'quicksand': '/dream_models/quicksand.glb',
            'fog': 'procedural',
            'void': 'procedural',
            'shadow': 'procedural',
            'mirror': '/dream_models/mirror.glb',
            'maze': '/dream_models/maze.glb',
            'abyss': 'procedural'
        }

        normalized_name = symbol.name.lower().replace(' ', '_')
        return model_mapping.get(normalized_name, 'procedural')

    def _extract_cinematic_refs(self, dream_scene: DreamScene) -> List[Dict]:
        """
        Extract cinematic references from enriched symbols.

        Args:
            dream_scene: Dream scene object

        Returns:
            List of cinematic reference configurations
        """
        refs = []

        for symbol in dream_scene.symbols:
            # Get symbol data from enriched database
            symbol_data = self.symbol_database.get(symbol.symbol_id, {})
            literary_refs = symbol_data.get('literary_references', {})

            # Extract modern cinema references
            cinema_refs = literary_refs.get('modern_cinema', [])
            for ref in cinema_refs[:2]:  # Top 2 most relevant
                cinematic_config = {
                    "film": ref.get('title', ''),
                    "scene": ref.get('key_scene', ''),
                    "visual_elements": ref.get('symbolic_elements', []),
                    "lighting": self._infer_lighting(ref),
                    "camera": self._infer_camera(ref),
                    "atmosphere": self._infer_atmosphere(ref)
                }
                refs.append(cinematic_config)

            # Extract classic film references
            classic_refs = literary_refs.get('classic_film', [])
            for ref in classic_refs[:1]:  # Top 1
                cinematic_config = {
                    "film": ref.get('title', ''),
                    "scene": ref.get('key_scene', ''),
                    "visual_elements": ref.get('symbolic_elements', []),
                    "lighting": self._infer_lighting(ref),
                    "camera": self._infer_camera(ref),
                    "atmosphere": self._infer_atmosphere(ref)
                }
                refs.append(cinematic_config)

        return refs

    def _infer_lighting(self, film_ref: Dict) -> Dict[str, Any]:
        """
        Infer lighting configuration from film reference.

        Args:
            film_ref: Film reference data

        Returns:
            Lighting configuration
        """
        # Extract keywords for lighting inference
        title = film_ref.get('title', '').lower()
        themes = ' '.join(film_ref.get('themes', [])).lower()
        elements = ' '.join(film_ref.get('symbolic_elements', [])).lower()

        # Default lighting
        lighting = {
            "type": "soft",
            "color_temp": 5500,  # Daylight
            "intensity": 1.0
        }

        # Prison/confinement films (harsh, high contrast)
        if 'shawshank' in title or 'prison' in themes:
            lighting = {
                "type": "dramatic",
                "color_temp": 6500,  # Cool, overcast
                "intensity": 0.8
            }

        # Noir/dark films (low key, moody)
        if 'blade runner' in title or 'noir' in themes:
            lighting = {
                "type": "harsh",
                "color_temp": 4500,  # Warm tungsten
                "intensity": 0.6
            }

        # Fantasy/surreal films (soft, ethereal)
        if 'inception' in title or 'dream' in themes:
            lighting = {
                "type": "soft",
                "color_temp": 5000,  # Neutral
                "intensity": 1.2
            }

        # Storm/chaos films (overcast, dark)
        if 'storm' in elements or 'rain' in elements:
            lighting = {
                "type": "overcast",
                "color_temp": 6000,
                "intensity": 0.7
            }

        return lighting

    def _infer_camera(self, film_ref: Dict) -> Dict[str, Any]:
        """
        Infer camera configuration from film reference.

        Args:
            film_ref: Film reference data

        Returns:
            Camera configuration
        """
        title = film_ref.get('title', '').lower()
        scene = film_ref.get('key_scene', '').lower()

        # Default camera
        camera = {
            "angle": "eye-level",
            "movement": "static",
            "focal_length": 50  # Standard lens
        }

        # Iconic scenes
        if 'shawshank' in title and 'rain' in scene:
            camera = {
                "angle": "low",  # Looking up at Andy in rain
                "movement": "crane",
                "focal_length": 35  # Wide lens for grandeur
            }

        if 'blade runner' in title:
            camera = {
                "angle": "dutch",  # Tilted, unsettling
                "movement": "dolly",
                "focal_length": 40  # Slightly wide
            }

        if 'inception' in title:
            camera = {
                "angle": "high",  # God's eye view
                "movement": "crane",
                "focal_length": 28  # Wide for spatial confusion
            }

        return camera

    def _infer_atmosphere(self, film_ref: Dict) -> Dict[str, Any]:
        """
        Infer atmospheric effects from film reference.

        Args:
            film_ref: Film reference data

        Returns:
            Atmosphere configuration
        """
        title = film_ref.get('title', '').lower()
        scene = film_ref.get('key_scene', '').lower()
        elements = ' '.join(film_ref.get('symbolic_elements', [])).lower()

        # Default atmosphere
        atmosphere = {
            "fog_density": 0.002,
            "particles": None,
            "time_of_day": "day"
        }

        # Rain scenes
        if 'rain' in scene or 'rain' in elements:
            atmosphere = {
                "fog_density": 0.003,
                "particles": "rain",
                "time_of_day": "overcast"
            }

        # Noir/smoky scenes
        if 'blade runner' in title or 'noir' in title:
            atmosphere = {
                "fog_density": 0.005,
                "particles": "dust",
                "time_of_day": "night"
            }

        # Dream/foggy scenes
        if 'inception' in title or 'dream' in elements:
            atmosphere = {
                "fog_density": 0.008,
                "particles": None,
                "time_of_day": "twilight"
            }

        return atmosphere

    def _atmosphere_config(self, dream_scene: DreamScene) -> Dict[str, Any]:
        """
        Generate atmospheric configuration for entire scene.

        Args:
            dream_scene: Dream scene object

        Returns:
            Atmosphere configuration
        """
        setting = dream_scene.setting

        # Base configuration from setting
        config = {
            "fog_density": 0.002,
            "fog_color": "#888888",
            "particles": None,
            "time_of_day": "day"
        }

        # Adjust based on lighting
        if setting.lighting == 'twilight':
            config["fog_color"] = "#8888bb"
            config["time_of_day"] = "twilight"
        elif setting.lighting == 'stormy':
            config["fog_color"] = "#555555"
            config["fog_density"] = 0.004
            config["particles"] = "rain"
            config["time_of_day"] = "overcast"
        elif setting.lighting == 'starry':
            config["fog_color"] = "#111133"
            config["fog_density"] = 0.001
            config["time_of_day"] = "night"

        # Weather overrides
        if setting.weather == 'rain':
            config["particles"] = "rain"
        elif setting.weather == 'snow':
            config["particles"] = "snow"

        # Atmosphere mood
        if setting.atmosphere == 'oppressive':
            config["fog_density"] *= 2.0
        elif setting.atmosphere == 'serene':
            config["fog_density"] *= 0.5

        return config


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_test_dream_scene() -> DreamScene:
    """
    Create a test dream scene for demonstration.

    Returns:
        Test dream scene with caged bird + storm cloud
    """
    # Caged bird symbol
    caged_bird = SymbolArchetype(
        symbol_id="caged_bird_001",
        name="caged_bird",
        category="restriction",
        visual_properties=VisualProperties(
            base_form="organic",
            primary_color="#4A90E2",
            secondary_color="#F5A623",
            material_type="solid",
            scale=1.0
        ),
        metaphorical_physics=MetaphoricalPhysics(
            type="reactive",
            triggers={"proximity": "freedom"},
            parameters={"base_rate": 0.5, "max_rate": 2.0}
        ),
        cinematic_references=[
            CinematicReference(
                film="The Shawshank Redemption",
                scene="rain_emergence",
                visual_elements=["prison", "rain", "freedom"],
                lighting={"type": "overcast", "color_temp": 6500, "intensity": 0.8},
                camera={"angle": "low", "movement": "crane", "focal_length": 35},
                atmosphere={"fog_density": 0.003, "particles": "rain", "time_of_day": "overcast"}
            )
        ]
    )

    # Storm cloud symbol
    storm_cloud = SymbolArchetype(
        symbol_id="storm_cloud_001",
        name="storm_cloud",
        category="conflict",
        visual_properties=VisualProperties(
            base_form="organic",
            primary_color="#333333",
            secondary_color="#555555",
            material_type="ethereal",
            scale=2.0
        ),
        metaphorical_physics=MetaphoricalPhysics(
            type="gravitational",
            triggers={"emotion": "anxiety"},
            parameters={"base_rate": 1.0, "threshold": 0.5}
        ),
        cinematic_references=[]
    )

    # Dream scene
    return DreamScene(
        scene_id="test_scene_001",
        title="Prison Dreams: Yearning for Freedom",
        symbols=[caged_bird, storm_cloud],
        setting=DreamSetting(
            environment="prison",
            lighting="stormy",
            weather="rain",
            atmosphere="oppressive"
        ),
        participants=[
            {
                "resident_id": "user_001",
                "name": "Andy",
                "perspective": "first_person",
                "emotional_state": {
                    "primary": "hope",
                    "intensity": 0.8
                }
            }
        ],
        duration_seconds=120.0,
        consciousness_level=0.6
    )


# ============================================================================
# MAIN DEMO
# ============================================================================

if __name__ == "__main__":
    # Create bridge
    bridge = DreamVisualizerBridge()

    # Create test scene
    test_scene = create_test_dream_scene()

    # Convert to Three.js format
    scene_data = bridge.prepare_scene_data(test_scene)

    # Save to JSON
    output_path = Path("demos/test_dream_scene.json")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scene_data, f, indent=2)

    print(f"[SUCCESS] Test dream scene saved to {output_path}")
    print(f"\nScene summary:")
    print(f"  Title: {test_scene.title}")
    print(f"  Symbols: {len(test_scene.symbols)}")
    print(f"  Cinematic refs: {len(scene_data['cinematic'])}")
    print(f"  Consciousness: {test_scene.consciousness_level * 100:.0f}%")

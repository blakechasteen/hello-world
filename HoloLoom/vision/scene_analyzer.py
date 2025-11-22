"""
Scene Analyzer - Spatial understanding and relationship extraction

Analyzes detected objects to understand:
    - Spatial relationships (left_of, above, near, contains, etc.)
    - Scene type (indoor, outdoor, workshop, office, etc.)
    - Dominant colors and lighting conditions
    - Layout and organization

Created: 2025-11-22
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from datetime import datetime

from HoloLoom.vision.protocol import (
    DetectedObject,
    SceneUnderstanding,
    SpatialRelationship,
    SceneAnalyzerProtocol,
    BoundingBox,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Scene Type Classification
# ============================================================================

# Keywords for scene type classification
SCENE_KEYWORDS = {
    "workshop": ["drill", "hammer", "toolbox", "saw", "workbench"],
    "kitchen": ["refrigerator", "microwave", "oven", "sink", "dining table"],
    "office": ["laptop", "keyboard", "mouse", "desk", "chair"],
    "living_room": ["couch", "tv", "chair", "coffee table"],
    "bedroom": ["bed", "dresser", "nightstand"],
    "outdoor": ["car", "bicycle", "tree", "building"],
}


# ============================================================================
# Scene Analyzer
# ============================================================================

class SceneAnalyzer(SceneAnalyzerProtocol):
    """
    Analyzes scenes for spatial understanding.

    Extracts spatial relationships, classifies scene type, and provides
    high-level understanding for AR decision-making.
    """

    def __init__(
        self,
        enable_relationships: bool = True,
        enable_scene_type: bool = True,
        enable_color_analysis: bool = False,
        relationship_distance_threshold: float = 0.3,
    ):
        """
        Initialize scene analyzer.

        Args:
            enable_relationships: Extract spatial relationships
            enable_scene_type: Classify scene type
            enable_color_analysis: Analyze dominant colors (slower)
            relationship_distance_threshold: Distance threshold for "near" relation
        """
        self.enable_relationships = enable_relationships
        self.enable_scene_type = enable_scene_type
        self.enable_color_analysis = enable_color_analysis
        self.relationship_distance_threshold = relationship_distance_threshold
        self.initialized = False

    async def initialize(self) -> bool:
        """Initialize analyzer"""
        self.initialized = True
        logger.info("Scene analyzer initialized")
        return True

    async def process_frame(self, frame: np.ndarray) -> SceneUnderstanding:
        """Process frame (delegates to analyze_scene)"""
        return await self.analyze_scene(frame)

    async def analyze_scene(
        self,
        frame: np.ndarray,
        objects: Optional[List[DetectedObject]] = None
    ) -> SceneUnderstanding:
        """
        Analyze scene for spatial understanding.

        Args:
            frame: RGB image
            objects: Pre-detected objects (if None, returns minimal understanding)

        Returns:
            Scene understanding
        """
        if not self.initialized:
            await self.initialize()

        # Initialize result
        scene = SceneUnderstanding(
            objects=objects or [],
            dominant_colors=[],
            scene_type="unknown",
            lighting_condition="normal",
            spatial_layout={},
            relationships=[],
        )

        if not objects:
            return scene

        # Extract spatial relationships
        if self.enable_relationships:
            scene.relationships = self._extract_relationships(objects)

        # Classify scene type
        if self.enable_scene_type:
            scene.scene_type = self._classify_scene_type(objects)

        # Analyze colors (expensive)
        if self.enable_color_analysis:
            scene.dominant_colors = self._analyze_colors(frame)

        # Analyze lighting
        scene.lighting_condition = self._analyze_lighting(frame)

        # Build spatial layout
        scene.spatial_layout = self._build_spatial_layout(objects)

        return scene

    def _extract_relationships(
        self,
        objects: List[DetectedObject]
    ) -> List[SpatialRelationship]:
        """
        Extract spatial relationships between objects.

        Relations:
            - left_of, right_of
            - above, below
            - near, far
            - contains (one bbox inside another)
        """
        relationships = []

        for i, obj1 in enumerate(objects):
            for obj2 in objects[i+1:]:
                # Calculate relationship
                rel = self._determine_relationship(obj1, obj2)

                if rel:
                    relationships.append(
                        SpatialRelationship(
                            subject_id=obj1.id,
                            relation=rel,
                            object_id=obj2.id,
                            confidence=0.9,  # Could be based on distance, overlap, etc.
                        )
                    )

        return relationships

    def _determine_relationship(
        self,
        obj1: DetectedObject,
        obj2: DetectedObject
    ) -> Optional[str]:
        """
        Determine spatial relationship between two objects.

        Returns primary relationship (most significant).
        """
        bbox1 = obj1.bbox
        bbox2 = obj2.bbox

        center1 = bbox1.center()
        center2 = bbox2.center()

        # Distance between centers (normalized)
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

        # Check containment
        if self._is_contained(bbox1, bbox2):
            return "contains"  # obj1 contains obj2

        # Check proximity
        if distance < self.relationship_distance_threshold:
            # Near each other, determine primary spatial relation
            dx = center2[0] - center1[0]
            dy = center2[1] - center1[1]

            # Horizontal relation more significant
            if abs(dx) > abs(dy):
                return "left_of" if dx < 0 else "right_of"
            else:
                return "above" if dy < 0 else "below"

        # Far apart
        return "near" if distance < 0.5 else None

    def _is_contained(self, outer: BoundingBox, inner: BoundingBox) -> bool:
        """Check if inner bbox is contained in outer bbox"""
        return (
            outer.x_min <= inner.x_min and
            outer.y_min <= inner.y_min and
            outer.x_max >= inner.x_max and
            outer.y_max >= inner.y_max
        )

    def _classify_scene_type(self, objects: List[DetectedObject]) -> str:
        """
        Classify scene type based on detected objects.

        Uses keyword matching against common scene types.
        """
        object_labels = set(obj.label for obj in objects)

        # Count matches for each scene type
        scores = {}
        for scene_type, keywords in SCENE_KEYWORDS.items():
            matches = len(object_labels.intersection(keywords))
            if matches > 0:
                scores[scene_type] = matches

        if not scores:
            return "unknown"

        # Return scene type with most matches
        return max(scores, key=scores.get)

    def _analyze_colors(self, frame: np.ndarray) -> List[str]:
        """
        Analyze dominant colors in frame.

        Returns top 3 dominant colors as hex strings.
        """
        # Simplified: Use k-means clustering on downsampled image
        # For production, could use more sophisticated color analysis

        # Downsample for speed
        small = frame[::8, ::8]  # 8x downsampling

        # Reshape to list of pixels
        pixels = small.reshape(-1, 3)

        # Simple approach: Find most common colors
        # (In production, use sklearn KMeans or similar)
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)

            # Convert to hex
            hex_colors = [
                f"#{r:02x}{g:02x}{b:02x}"
                for r, g, b in colors
            ]

            return hex_colors

        except ImportError:
            # Fallback: Simple color binning
            mean_color = pixels.mean(axis=0).astype(int)
            return [f"#{mean_color[0]:02x}{mean_color[1]:02x}{mean_color[2]:02x}"]

    def _analyze_lighting(self, frame: np.ndarray) -> str:
        """
        Analyze lighting condition.

        Returns: "bright", "normal", "dim", "backlit"
        """
        # Calculate mean brightness
        brightness = frame.mean()

        if brightness > 180:
            return "bright"
        elif brightness < 60:
            return "dim"
        elif brightness < 80:
            # Check if contrast is high (possible backlit)
            contrast = frame.std()
            if contrast > 60:
                return "backlit"

        return "normal"

    def _build_spatial_layout(self, objects: List[DetectedObject]) -> Dict[str, Any]:
        """
        Build spatial layout representation.

        Returns dict with layout information:
            - object_count
            - density (objects per unit area)
            - distribution (clustered vs distributed)
            - zones (left, center, right regions)
        """
        if not objects:
            return {}

        # Count objects
        object_count = len(objects)

        # Calculate density
        total_area = sum(obj.bbox.area() for obj in objects)
        density = total_area  # Normalized area coverage

        # Zone distribution (divide frame into thirds)
        zones = {"left": 0, "center": 0, "right": 0}
        for obj in objects:
            cx, _ = obj.bbox.center()
            if cx < 0.33:
                zones["left"] += 1
            elif cx < 0.67:
                zones["center"] += 1
            else:
                zones["right"] += 1

        # Determine if clustered or distributed
        centers = [obj.bbox.center() for obj in objects]
        if len(centers) > 1:
            # Calculate average distance between centers
            distances = []
            for i, c1 in enumerate(centers):
                for c2 in centers[i+1:]:
                    dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
                    distances.append(dist)

            avg_distance = np.mean(distances) if distances else 0
            distribution = "clustered" if avg_distance < 0.3 else "distributed"
        else:
            distribution = "single"

        return {
            "object_count": object_count,
            "density": density,
            "distribution": distribution,
            "zones": zones,
        }

    async def cleanup(self) -> None:
        """Clean up resources"""
        self.initialized = False

    def get_capabilities(self) -> Dict[str, bool]:
        """Get analyzer capabilities"""
        return {
            "spatial_relationships": self.enable_relationships,
            "scene_classification": self.enable_scene_type,
            "color_analysis": self.enable_color_analysis,
            "lighting_analysis": True,
            "layout_analysis": True,
        }


# ============================================================================
# Factory
# ============================================================================

def create_scene_analyzer(**kwargs) -> SceneAnalyzer:
    """
    Factory for creating scene analyzers.

    Args:
        **kwargs: Analyzer options

    Returns:
        SceneAnalyzer instance
    """
    return SceneAnalyzer(**kwargs)


# ============================================================================
# Helper Functions
# ============================================================================

def find_objects_by_relationship(
    relationships: List[SpatialRelationship],
    subject_id: str,
    relation: str
) -> List[str]:
    """
    Find objects related to subject by specific relation.

    Args:
        relationships: List of spatial relationships
        subject_id: Subject object ID
        relation: Relation type (e.g., "left_of")

    Returns:
        List of object IDs
    """
    return [
        rel.object_id
        for rel in relationships
        if rel.subject_id == subject_id and rel.relation == relation
    ]


def get_scene_summary(scene: SceneUnderstanding) -> str:
    """
    Generate human-readable scene summary.

    Args:
        scene: Scene understanding

    Returns:
        Text summary
    """
    summary_parts = []

    # Scene type
    if scene.scene_type != "unknown":
        summary_parts.append(f"{scene.scene_type.replace('_', ' ').title()} scene")

    # Object count
    obj_count = len(scene.objects)
    summary_parts.append(f"with {obj_count} object{'s' if obj_count != 1 else ''}")

    # Layout
    if scene.spatial_layout.get("distribution"):
        dist = scene.spatial_layout["distribution"]
        summary_parts.append(f"({dist} layout)")

    # Lighting
    if scene.lighting_condition != "normal":
        summary_parts.append(f"{scene.lighting_condition} lighting")

    return " ".join(summary_parts)

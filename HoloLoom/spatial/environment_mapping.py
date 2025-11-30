"""
Environment Mapping and Scene Persistence for HoloLoom Spatial Computing.

Real-world environment understanding and persistent XR scenes:
- Room/space scanning
- Surface detection (floors, walls, tables)
- Scene persistence and loading
- Spatial anchor persistence
- Semantic surface labeling
- Environment meshes

Created: November 2025
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Tuple, Set
from enum import Enum
from datetime import datetime
import uuid
import json
import math


class SurfaceType(Enum):
    """Types of detected surfaces."""
    UNKNOWN = "unknown"
    FLOOR = "floor"
    WALL = "wall"
    CEILING = "ceiling"
    TABLE = "table"
    DOOR = "door"
    WINDOW = "window"
    SHELF = "shelf"
    SEAT = "seat"
    SCREEN = "screen"


class MeshType(Enum):
    """Environment mesh types."""
    PLANE = "plane"
    MESH = "mesh"
    CONVEX_HULL = "convex_hull"
    BOUNDING_BOX = "bounding_box"


class ScanQuality(Enum):
    """Scan quality levels."""
    LOW = "low"        # Fast, rough
    MEDIUM = "medium"  # Balanced
    HIGH = "high"      # Detailed, slow


@dataclass
class Vector3:
    """3D vector."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Vector3":
        return cls(t[0], t[1], t[2])

    def magnitude(self) -> float:
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)


@dataclass
class Quaternion:
    """Rotation quaternion."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    center: Vector3 = field(default_factory=Vector3)
    size: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    def get_corners(self) -> List[Vector3]:
        """Get 8 corners of bounding box."""
        half = Vector3(self.size.x/2, self.size.y/2, self.size.z/2)
        corners = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corners.append(Vector3(
                        self.center.x + half.x * sx,
                        self.center.y + half.y * sy,
                        self.center.z + half.z * sz
                    ))
        return corners

    def get_volume(self) -> float:
        return self.size.x * self.size.y * self.size.z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": self.center.to_tuple(),
            "size": self.size.to_tuple(),
            "rotation": self.rotation.to_tuple()
        }


@dataclass
class Plane:
    """Detected plane/surface."""
    plane_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    center: Vector3 = field(default_factory=Vector3)
    normal: Vector3 = field(default_factory=lambda: Vector3(0, 1, 0))
    extents: Tuple[float, float] = (1.0, 1.0)  # Width, height
    rotation: Quaternion = field(default_factory=Quaternion)

    # Classification
    surface_type: SurfaceType = SurfaceType.UNKNOWN
    confidence: float = 0.0
    semantic_label: str = ""

    # Geometry
    vertices: Optional[List[Vector3]] = None  # Polygon outline
    mesh_id: Optional[str] = None  # Reference to detailed mesh

    # State
    is_vertical: bool = False
    is_horizontal: bool = True
    alignment: str = "horizontal"  # horizontal, vertical, slanted

    # Metadata
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    updated_at: float = field(default_factory=lambda: datetime.now().timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_area(self) -> float:
        """Calculate plane area."""
        return self.extents[0] * self.extents[1]

    def contains_point(self, point: Vector3, tolerance: float = 0.1) -> bool:
        """Check if point is on plane (within tolerance)."""
        # Simplified check using normal distance
        to_point = Vector3(
            point.x - self.center.x,
            point.y - self.center.y,
            point.z - self.center.z
        )
        distance = abs(
            to_point.x * self.normal.x +
            to_point.y * self.normal.y +
            to_point.z * self.normal.z
        )
        return distance < tolerance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.plane_id,
            "center": self.center.to_tuple(),
            "normal": self.normal.to_tuple(),
            "extents": self.extents,
            "rotation": self.rotation.to_tuple(),
            "surfaceType": self.surface_type.value,
            "confidence": self.confidence,
            "semanticLabel": self.semantic_label,
            "area": self.get_area(),
            "alignment": self.alignment,
            "meshId": self.mesh_id
        }


@dataclass
class EnvironmentMesh:
    """3D mesh of environment."""
    mesh_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    vertices: List[Vector3] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)  # Triangle indices
    normals: Optional[List[Vector3]] = None

    # Bounding
    bounds: Optional[BoundingBox] = None

    # Metadata
    vertex_count: int = 0
    triangle_count: int = 0
    mesh_type: MeshType = MeshType.MESH
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())

    def calculate_bounds(self):
        """Calculate bounding box from vertices."""
        if not self.vertices:
            return

        min_x = min(v.x for v in self.vertices)
        max_x = max(v.x for v in self.vertices)
        min_y = min(v.y for v in self.vertices)
        max_y = max(v.y for v in self.vertices)
        min_z = min(v.z for v in self.vertices)
        max_z = max(v.z for v in self.vertices)

        self.bounds = BoundingBox(
            center=Vector3(
                (min_x + max_x) / 2,
                (min_y + max_y) / 2,
                (min_z + max_z) / 2
            ),
            size=Vector3(
                max_x - min_x,
                max_y - min_y,
                max_z - min_z
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.mesh_id,
            "vertexCount": self.vertex_count,
            "triangleCount": self.triangle_count,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "type": self.mesh_type.value
        }


@dataclass
class PersistentAnchor:
    """Anchor that persists across sessions."""
    anchor_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    position: Vector3 = field(default_factory=Vector3)
    rotation: Quaternion = field(default_factory=Quaternion)

    # Persistence
    cloud_anchor_id: Optional[str] = None  # Cloud anchor ID for sharing
    local_storage_id: Optional[str] = None

    # Attachment
    attached_plane_id: Optional[str] = None
    attached_content_ids: List[str] = field(default_factory=list)

    # State
    tracking_state: str = "tracking"  # tracking, limited, lost
    confidence: float = 1.0
    last_seen: float = field(default_factory=lambda: datetime.now().timestamp())

    # Metadata
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.anchor_id,
            "name": self.name,
            "position": self.position.to_tuple(),
            "rotation": self.rotation.to_tuple(),
            "cloudAnchorId": self.cloud_anchor_id,
            "trackingState": self.tracking_state,
            "confidence": self.confidence,
            "attachedPlaneId": self.attached_plane_id,
            "attachedContentIds": self.attached_content_ids,
            "tags": self.tags
        }


@dataclass
class SceneSnapshot:
    """Snapshot of scene state for persistence."""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""

    # Content
    planes: List[Dict[str, Any]] = field(default_factory=list)
    anchors: List[Dict[str, Any]] = field(default_factory=list)
    meshes: List[Dict[str, Any]] = field(default_factory=list)
    content_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Room bounds
    room_center: Optional[Tuple[float, float, float]] = None
    room_size: Optional[Tuple[float, float, float]] = None

    # Metadata
    created_at: float = field(default_factory=lambda: datetime.now().timestamp())
    created_by: str = ""
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Export to JSON string."""
        return json.dumps({
            "id": self.snapshot_id,
            "name": self.name,
            "description": self.description,
            "planes": self.planes,
            "anchors": self.anchors,
            "meshes": self.meshes,
            "contentPositions": self.content_positions,
            "roomCenter": self.room_center,
            "roomSize": self.room_size,
            "createdAt": self.created_at,
            "createdBy": self.created_by,
            "version": self.version,
            "metadata": self.metadata
        }, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneSnapshot":
        """Import from JSON string."""
        data = json.loads(json_str)
        return cls(
            snapshot_id=data.get("id", str(uuid.uuid4())[:8]),
            name=data.get("name", ""),
            description=data.get("description", ""),
            planes=data.get("planes", []),
            anchors=data.get("anchors", []),
            meshes=data.get("meshes", []),
            content_positions=data.get("contentPositions", {}),
            room_center=data.get("roomCenter"),
            room_size=data.get("roomSize"),
            created_at=data.get("createdAt", datetime.now().timestamp()),
            created_by=data.get("createdBy", ""),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ScanProgress:
    """Progress of environment scan."""
    total_area_scanned: float = 0.0
    planes_detected: int = 0
    meshes_generated: int = 0
    confidence: float = 0.0
    is_scanning: bool = False
    scan_quality: ScanQuality = ScanQuality.MEDIUM
    start_time: Optional[float] = None
    estimated_completion: Optional[float] = None


class EnvironmentMapper:
    """
    Environment mapping and scene persistence manager.
    """

    def __init__(
        self,
        auto_classify_surfaces: bool = True,
        merge_similar_planes: bool = True,
        plane_merge_threshold: float = 0.1
    ):
        self.auto_classify = auto_classify_surfaces
        self.merge_planes = merge_similar_planes
        self.merge_threshold = plane_merge_threshold

        # Detected environment
        self.planes: Dict[str, Plane] = {}
        self.meshes: Dict[str, EnvironmentMesh] = {}
        self.anchors: Dict[str, PersistentAnchor] = {}

        # Room understanding
        self.room_bounds: Optional[BoundingBox] = None
        self.floor_height: float = 0.0
        self.ceiling_height: float = 2.5

        # Scan state
        self.scan_progress = ScanProgress()

        # Saved scenes
        self.saved_snapshots: Dict[str, SceneSnapshot] = {}

        # Callbacks
        self.on_plane_detected: List[Callable[[Plane], None]] = []
        self.on_plane_updated: List[Callable[[Plane], None]] = []
        self.on_scan_complete: List[Callable[[ScanProgress], None]] = []
        self.on_anchor_created: List[Callable[[PersistentAnchor], None]] = []

    def start_scan(self, quality: ScanQuality = ScanQuality.MEDIUM):
        """Start environment scanning."""
        self.scan_progress = ScanProgress(
            is_scanning=True,
            scan_quality=quality,
            start_time=datetime.now().timestamp()
        )

    def stop_scan(self):
        """Stop environment scanning."""
        self.scan_progress.is_scanning = False
        for callback in self.on_scan_complete:
            callback(self.scan_progress)

    def add_plane(
        self,
        center: Tuple[float, float, float],
        normal: Tuple[float, float, float],
        extents: Tuple[float, float],
        surface_type: Optional[SurfaceType] = None,
        confidence: float = 1.0
    ) -> Plane:
        """Add detected plane."""
        plane = Plane(
            center=Vector3.from_tuple(center),
            normal=Vector3.from_tuple(normal),
            extents=extents,
            confidence=confidence
        )

        # Classify surface
        if surface_type:
            plane.surface_type = surface_type
        elif self.auto_classify:
            plane.surface_type = self._classify_surface(plane)

        # Determine alignment
        up_dot = abs(plane.normal.y)
        if up_dot > 0.9:
            plane.alignment = "horizontal"
            plane.is_horizontal = True
            plane.is_vertical = False
        elif up_dot < 0.1:
            plane.alignment = "vertical"
            plane.is_horizontal = False
            plane.is_vertical = True
        else:
            plane.alignment = "slanted"
            plane.is_horizontal = False
            plane.is_vertical = False

        # Check for merge with existing planes
        if self.merge_planes:
            merged = self._try_merge_plane(plane)
            if merged:
                return merged

        self.planes[plane.plane_id] = plane
        self.scan_progress.planes_detected = len(self.planes)
        self._update_room_bounds()

        for callback in self.on_plane_detected:
            callback(plane)

        return plane

    def _classify_surface(self, plane: Plane) -> SurfaceType:
        """Automatically classify surface type."""
        # Horizontal surfaces
        if plane.alignment == "horizontal":
            y = plane.center.y
            if abs(y - self.floor_height) < 0.1:
                return SurfaceType.FLOOR
            elif abs(y - self.ceiling_height) < 0.3:
                return SurfaceType.CEILING
            elif 0.6 < y < 1.0:
                return SurfaceType.TABLE
            elif 0.3 < y < 0.6:
                return SurfaceType.SEAT

        # Vertical surfaces
        elif plane.alignment == "vertical":
            area = plane.get_area()
            if area > 2.0:
                return SurfaceType.WALL
            elif area > 0.5:
                # Could be door or large surface
                return SurfaceType.WALL

        return SurfaceType.UNKNOWN

    def _try_merge_plane(self, new_plane: Plane) -> Optional[Plane]:
        """Try to merge with existing similar plane."""
        for existing in self.planes.values():
            # Check if normals are similar
            dot = (
                new_plane.normal.x * existing.normal.x +
                new_plane.normal.y * existing.normal.y +
                new_plane.normal.z * existing.normal.z
            )
            if abs(dot) < 0.95:
                continue

            # Check if centers are close
            dist = math.sqrt(
                (new_plane.center.x - existing.center.x)**2 +
                (new_plane.center.y - existing.center.y)**2 +
                (new_plane.center.z - existing.center.z)**2
            )
            if dist > self.merge_threshold:
                continue

            # Merge: update existing plane
            existing.extents = (
                max(existing.extents[0], new_plane.extents[0]),
                max(existing.extents[1], new_plane.extents[1])
            )
            existing.confidence = max(existing.confidence, new_plane.confidence)
            existing.updated_at = datetime.now().timestamp()

            for callback in self.on_plane_updated:
                callback(existing)

            return existing

        return None

    def _update_room_bounds(self):
        """Update room bounds from detected planes."""
        if not self.planes:
            return

        # Find floor and ceiling
        floors = [p for p in self.planes.values() if p.surface_type == SurfaceType.FLOOR]
        ceilings = [p for p in self.planes.values() if p.surface_type == SurfaceType.CEILING]

        if floors:
            self.floor_height = min(p.center.y for p in floors)
        if ceilings:
            self.ceiling_height = max(p.center.y for p in ceilings)

        # Calculate room bounds from all planes
        all_centers = [p.center for p in self.planes.values()]
        if all_centers:
            min_x = min(c.x for c in all_centers) - 1
            max_x = max(c.x for c in all_centers) + 1
            min_y = self.floor_height
            max_y = self.ceiling_height
            min_z = min(c.z for c in all_centers) - 1
            max_z = max(c.z for c in all_centers) + 1

            self.room_bounds = BoundingBox(
                center=Vector3(
                    (min_x + max_x) / 2,
                    (min_y + max_y) / 2,
                    (min_z + max_z) / 2
                ),
                size=Vector3(
                    max_x - min_x,
                    max_y - min_y,
                    max_z - min_z
                )
            )

    def add_mesh(
        self,
        vertices: List[Tuple[float, float, float]],
        indices: List[int],
        normals: Optional[List[Tuple[float, float, float]]] = None
    ) -> EnvironmentMesh:
        """Add environment mesh."""
        mesh = EnvironmentMesh(
            vertices=[Vector3.from_tuple(v) for v in vertices],
            indices=indices,
            normals=[Vector3.from_tuple(n) for n in normals] if normals else None,
            vertex_count=len(vertices),
            triangle_count=len(indices) // 3
        )
        mesh.calculate_bounds()

        self.meshes[mesh.mesh_id] = mesh
        self.scan_progress.meshes_generated = len(self.meshes)

        return mesh

    def create_anchor(
        self,
        position: Tuple[float, float, float],
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        name: str = "",
        plane_id: Optional[str] = None,
        created_by: str = ""
    ) -> PersistentAnchor:
        """Create persistent anchor."""
        anchor = PersistentAnchor(
            name=name,
            position=Vector3.from_tuple(position),
            rotation=Quaternion(*rotation),
            attached_plane_id=plane_id,
            created_by=created_by
        )

        self.anchors[anchor.anchor_id] = anchor

        for callback in self.on_anchor_created:
            callback(anchor)

        return anchor

    def attach_content_to_anchor(self, anchor_id: str, content_id: str) -> bool:
        """Attach content to anchor."""
        if anchor_id not in self.anchors:
            return False
        self.anchors[anchor_id].attached_content_ids.append(content_id)
        return True

    def update_anchor_tracking(
        self,
        anchor_id: str,
        tracking_state: str,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None
    ):
        """Update anchor tracking state."""
        if anchor_id not in self.anchors:
            return

        anchor = self.anchors[anchor_id]
        anchor.tracking_state = tracking_state
        anchor.last_seen = datetime.now().timestamp()

        if position:
            anchor.position = Vector3.from_tuple(position)
        if rotation:
            anchor.rotation = Quaternion(*rotation)

    def get_plane_at_point(
        self,
        point: Tuple[float, float, float],
        tolerance: float = 0.1
    ) -> Optional[Plane]:
        """Find plane containing point."""
        point_vec = Vector3.from_tuple(point)
        for plane in self.planes.values():
            if plane.contains_point(point_vec, tolerance):
                return plane
        return None

    def get_floor_plane(self) -> Optional[Plane]:
        """Get main floor plane."""
        floors = [
            p for p in self.planes.values()
            if p.surface_type == SurfaceType.FLOOR
        ]
        if floors:
            return max(floors, key=lambda p: p.get_area())
        return None

    def get_walls(self) -> List[Plane]:
        """Get all wall planes."""
        return [
            p for p in self.planes.values()
            if p.surface_type == SurfaceType.WALL
        ]

    def get_tables(self) -> List[Plane]:
        """Get all table surfaces."""
        return [
            p for p in self.planes.values()
            if p.surface_type == SurfaceType.TABLE
        ]

    def save_snapshot(
        self,
        name: str,
        description: str = "",
        content_positions: Optional[Dict[str, Dict[str, Any]]] = None,
        created_by: str = ""
    ) -> SceneSnapshot:
        """Save current scene state."""
        snapshot = SceneSnapshot(
            name=name,
            description=description,
            planes=[p.to_dict() for p in self.planes.values()],
            anchors=[a.to_dict() for a in self.anchors.values()],
            meshes=[m.to_dict() for m in self.meshes.values()],
            content_positions=content_positions or {},
            room_center=self.room_bounds.center.to_tuple() if self.room_bounds else None,
            room_size=self.room_bounds.size.to_tuple() if self.room_bounds else None,
            created_by=created_by
        )

        self.saved_snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def load_snapshot(self, snapshot: SceneSnapshot) -> bool:
        """Load scene from snapshot."""
        # Clear current state
        self.planes.clear()
        self.anchors.clear()

        # Load planes (simplified - would need full deserialization)
        for plane_data in snapshot.planes:
            plane = Plane(
                plane_id=plane_data.get("id", str(uuid.uuid4())[:8]),
                center=Vector3.from_tuple(plane_data.get("center", (0, 0, 0))),
                normal=Vector3.from_tuple(plane_data.get("normal", (0, 1, 0))),
                extents=tuple(plane_data.get("extents", [1, 1])),
                surface_type=SurfaceType(plane_data.get("surfaceType", "unknown"))
            )
            self.planes[plane.plane_id] = plane

        # Load anchors
        for anchor_data in snapshot.anchors:
            anchor = PersistentAnchor(
                anchor_id=anchor_data.get("id", str(uuid.uuid4())[:8]),
                name=anchor_data.get("name", ""),
                position=Vector3.from_tuple(anchor_data.get("position", (0, 0, 0))),
                rotation=Quaternion(*anchor_data.get("rotation", (0, 0, 0, 1)))
            )
            self.anchors[anchor.anchor_id] = anchor

        # Update room bounds
        if snapshot.room_center and snapshot.room_size:
            self.room_bounds = BoundingBox(
                center=Vector3.from_tuple(snapshot.room_center),
                size=Vector3.from_tuple(snapshot.room_size)
            )

        return True

    def export_snapshot_json(self, snapshot_id: str) -> Optional[str]:
        """Export snapshot to JSON."""
        snapshot = self.saved_snapshots.get(snapshot_id)
        if snapshot:
            return snapshot.to_json()
        return None

    def import_snapshot_json(self, json_str: str) -> SceneSnapshot:
        """Import snapshot from JSON."""
        snapshot = SceneSnapshot.from_json(json_str)
        self.saved_snapshots[snapshot.snapshot_id] = snapshot
        return snapshot

    def clear(self):
        """Clear all environment data."""
        self.planes.clear()
        self.meshes.clear()
        self.anchors.clear()
        self.room_bounds = None
        self.scan_progress = ScanProgress()

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        surfaces_by_type = {}
        for plane in self.planes.values():
            t = plane.surface_type.value
            surfaces_by_type[t] = surfaces_by_type.get(t, 0) + 1

        total_area = sum(p.get_area() for p in self.planes.values())
        total_vertices = sum(m.vertex_count for m in self.meshes.values())

        return {
            "planes": len(self.planes),
            "meshes": len(self.meshes),
            "anchors": len(self.anchors),
            "surfaces_by_type": surfaces_by_type,
            "total_area_m2": round(total_area, 2),
            "total_vertices": total_vertices,
            "room_bounds": self.room_bounds.to_dict() if self.room_bounds else None,
            "floor_height": self.floor_height,
            "ceiling_height": self.ceiling_height,
            "saved_snapshots": len(self.saved_snapshots),
            "is_scanning": self.scan_progress.is_scanning
        }

    def to_state(self) -> Dict[str, Any]:
        """Export full state for WebXR client."""
        return {
            "type": "environment_state",
            "planes": [p.to_dict() for p in self.planes.values()],
            "anchors": [a.to_dict() for a in self.anchors.values()],
            "meshes": [m.to_dict() for m in self.meshes.values()],
            "roomBounds": self.room_bounds.to_dict() if self.room_bounds else None,
            "scanProgress": {
                "isScanning": self.scan_progress.is_scanning,
                "planesDetected": self.scan_progress.planes_detected,
                "confidence": self.scan_progress.confidence
            },
            "statistics": self.get_statistics(),
            "timestamp": datetime.now().isoformat()
        }

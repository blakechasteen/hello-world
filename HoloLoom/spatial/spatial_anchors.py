"""
HoloLoom Phase 4: Spatial Anchor Persistence
November 2025

AR spatial anchors that persist knowledge graph nodes
to real-world locations across sessions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Tuple
import json
import hashlib


class AnchorState(Enum):
    """State of a spatial anchor."""
    PENDING = "pending"      # Not yet placed
    TRACKING = "tracking"    # Actively tracked
    LOST = "lost"            # Lost tracking
    PERSISTED = "persisted"  # Saved for future sessions


class AnchorQuality(Enum):
    """Quality of anchor tracking."""
    HIGH = "high"        # Stable, accurate tracking
    MEDIUM = "medium"    # Acceptable tracking
    LOW = "low"          # Poor tracking, may drift
    UNKNOWN = "unknown"  # Quality not determined


@dataclass
class WorldPosition:
    """Real-world position in AR coordinates."""
    x: float = 0.0  # Meters from origin
    y: float = 0.0
    z: float = 0.0

    # Orientation (quaternion)
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "position": {"x": self.x, "y": self.y, "z": self.z},
            "orientation": {"x": self.qx, "y": self.qy, "z": self.qz, "w": self.qw}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorldPosition':
        pos = data.get("position", {})
        rot = data.get("orientation", {})
        return cls(
            x=pos.get("x", 0.0),
            y=pos.get("y", 0.0),
            z=pos.get("z", 0.0),
            qx=rot.get("x", 0.0),
            qy=rot.get("y", 0.0),
            qz=rot.get("z", 0.0),
            qw=rot.get("w", 1.0)
        )

    def distance_to(self, other: 'WorldPosition') -> float:
        """Euclidean distance to another position."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx*dx + dy*dy + dz*dz) ** 0.5


@dataclass
class SpatialAnchor:
    """
    A persistent spatial anchor linking a knowledge node
    to a real-world AR location.
    """
    anchor_id: str
    node_id: str                              # Knowledge graph node ID
    position: WorldPosition
    state: AnchorState = AnchorState.PENDING
    quality: AnchorQuality = AnchorQuality.UNKNOWN

    # Location context
    location_hint: str = ""                   # "office_desk", "kitchen", etc.
    environment_hash: str = ""                # Hash of visual features for re-localization

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    persistence_score: float = 0.0            # How reliably anchor persists (0-1)

    # Visual properties for AR rendering
    visual_radius: float = 0.15               # Meters
    glow_color: str = "#00ff88"
    label: str = ""

    # Cloud anchor ID (for cross-device persistence)
    cloud_anchor_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "anchor_id": self.anchor_id,
            "node_id": self.node_id,
            "position": self.position.to_dict(),
            "state": self.state.value,
            "quality": self.quality.value,
            "location_hint": self.location_hint,
            "environment_hash": self.environment_hash,
            "created_at": self.created_at.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "persistence_score": self.persistence_score,
            "visual_radius": self.visual_radius,
            "glow_color": self.glow_color,
            "label": self.label,
            "cloud_anchor_id": self.cloud_anchor_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SpatialAnchor':
        return cls(
            anchor_id=data["anchor_id"],
            node_id=data["node_id"],
            position=WorldPosition.from_dict(data.get("position", {})),
            state=AnchorState(data.get("state", "pending")),
            quality=AnchorQuality(data.get("quality", "unknown")),
            location_hint=data.get("location_hint", ""),
            environment_hash=data.get("environment_hash", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_seen=datetime.fromisoformat(data["last_seen"]) if "last_seen" in data else datetime.now(),
            persistence_score=data.get("persistence_score", 0.0),
            visual_radius=data.get("visual_radius", 0.15),
            glow_color=data.get("glow_color", "#00ff88"),
            label=data.get("label", ""),
            cloud_anchor_id=data.get("cloud_anchor_id")
        )


@dataclass
class AnchorCluster:
    """A cluster of related anchors in a location."""
    cluster_id: str
    location_hint: str
    anchor_ids: Set[str] = field(default_factory=set)
    centroid: WorldPosition = field(default_factory=WorldPosition)
    radius: float = 2.0  # Meters

    def contains(self, position: WorldPosition) -> bool:
        """Check if position is within cluster radius."""
        return self.centroid.distance_to(position) <= self.radius


class SpatialAnchorManager:
    """
    Manages spatial anchors for AR knowledge visualization.

    Features:
    - Create and track anchors tied to knowledge nodes
    - Persist anchors across sessions
    - Cluster anchors by location
    - Handle anchor state transitions
    - Support for cloud anchors (cross-device)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize anchor manager.

        Args:
            storage_path: Path to persist anchor data
        """
        self.storage_path = storage_path
        self.anchors: Dict[str, SpatialAnchor] = {}
        self.clusters: Dict[str, AnchorCluster] = {}
        self._node_to_anchor: Dict[str, str] = {}  # node_id -> anchor_id
        self._anchor_counter = 0

        # Callbacks for AR events
        self.on_anchor_created: Optional[callable] = None
        self.on_anchor_lost: Optional[callable] = None
        self.on_anchor_recovered: Optional[callable] = None

        if storage_path:
            self._load()

    def _generate_anchor_id(self) -> str:
        """Generate unique anchor ID."""
        self._anchor_counter += 1
        return f"anchor_{self._anchor_counter:06d}"

    def _generate_environment_hash(self, visual_features: Optional[bytes] = None) -> str:
        """Generate hash for environment re-localization."""
        if visual_features:
            return hashlib.sha256(visual_features).hexdigest()[:16]
        return ""

    # === Anchor Creation ===

    def create_anchor(
        self,
        node_id: str,
        position: WorldPosition,
        location_hint: str = "",
        label: str = "",
        visual_features: Optional[bytes] = None
    ) -> SpatialAnchor:
        """
        Create a new spatial anchor for a knowledge node.

        Args:
            node_id: Knowledge graph node ID
            position: Real-world AR position
            location_hint: Description of location (e.g., "office_desk")
            label: Display label for anchor
            visual_features: Optional visual features for re-localization
        """
        anchor_id = self._generate_anchor_id()

        anchor = SpatialAnchor(
            anchor_id=anchor_id,
            node_id=node_id,
            position=position,
            state=AnchorState.PENDING,
            location_hint=location_hint,
            environment_hash=self._generate_environment_hash(visual_features),
            label=label or node_id
        )

        self.anchors[anchor_id] = anchor
        self._node_to_anchor[node_id] = anchor_id

        # Add to appropriate cluster
        self._assign_to_cluster(anchor)

        if self.on_anchor_created:
            self.on_anchor_created(anchor)

        self._save()
        return anchor

    def _assign_to_cluster(self, anchor: SpatialAnchor):
        """Assign anchor to a cluster based on location."""
        # Find existing cluster or create new one
        for cluster in self.clusters.values():
            if cluster.location_hint == anchor.location_hint or cluster.contains(anchor.position):
                cluster.anchor_ids.add(anchor.anchor_id)
                self._update_cluster_centroid(cluster)
                return

        # Create new cluster
        cluster_id = f"cluster_{len(self.clusters):04d}"
        cluster = AnchorCluster(
            cluster_id=cluster_id,
            location_hint=anchor.location_hint,
            anchor_ids={anchor.anchor_id},
            centroid=anchor.position
        )
        self.clusters[cluster_id] = cluster

    def _update_cluster_centroid(self, cluster: AnchorCluster):
        """Update cluster centroid based on anchor positions."""
        if not cluster.anchor_ids:
            return

        sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
        count = 0

        for anchor_id in cluster.anchor_ids:
            if anchor_id in self.anchors:
                pos = self.anchors[anchor_id].position
                sum_x += pos.x
                sum_y += pos.y
                sum_z += pos.z
                count += 1

        if count > 0:
            cluster.centroid = WorldPosition(
                x=sum_x / count,
                y=sum_y / count,
                z=sum_z / count
            )

    # === Anchor State Management ===

    def start_tracking(self, anchor_id: str) -> bool:
        """Mark anchor as actively tracking."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return False

        anchor.state = AnchorState.TRACKING
        anchor.last_seen = datetime.now()
        self._save()
        return True

    def update_position(self, anchor_id: str, position: WorldPosition, quality: AnchorQuality = None):
        """Update anchor position from AR tracking."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return

        anchor.position = position
        anchor.last_seen = datetime.now()

        if quality:
            anchor.quality = quality

        # Update persistence score based on tracking history
        if anchor.state == AnchorState.TRACKING:
            # Improve score for consistent tracking
            anchor.persistence_score = min(1.0, anchor.persistence_score + 0.01)

        self._save()

    def mark_lost(self, anchor_id: str):
        """Mark anchor as lost (tracking failed)."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return

        anchor.state = AnchorState.LOST
        anchor.persistence_score = max(0.0, anchor.persistence_score - 0.1)

        if self.on_anchor_lost:
            self.on_anchor_lost(anchor)

        self._save()

    def recover_anchor(self, anchor_id: str, position: WorldPosition):
        """Recover a lost anchor."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return

        anchor.state = AnchorState.TRACKING
        anchor.position = position
        anchor.last_seen = datetime.now()

        if self.on_anchor_recovered:
            self.on_anchor_recovered(anchor)

        self._save()

    def persist_anchor(self, anchor_id: str):
        """Mark anchor for cross-session persistence."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return

        anchor.state = AnchorState.PERSISTED
        self._save()

    # === Queries ===

    def get_anchor(self, anchor_id: str) -> Optional[SpatialAnchor]:
        """Get anchor by ID."""
        return self.anchors.get(anchor_id)

    def get_anchor_for_node(self, node_id: str) -> Optional[SpatialAnchor]:
        """Get anchor associated with a knowledge node."""
        anchor_id = self._node_to_anchor.get(node_id)
        if anchor_id:
            return self.anchors.get(anchor_id)
        return None

    def get_anchors_in_location(self, location_hint: str) -> List[SpatialAnchor]:
        """Get all anchors in a location."""
        return [
            anchor for anchor in self.anchors.values()
            if anchor.location_hint == location_hint
        ]

    def get_anchors_in_radius(self, center: WorldPosition, radius: float) -> List[SpatialAnchor]:
        """Get all anchors within radius of a position."""
        return [
            anchor for anchor in self.anchors.values()
            if center.distance_to(anchor.position) <= radius
        ]

    def get_nearby_anchors(self, position: WorldPosition, limit: int = 10) -> List[SpatialAnchor]:
        """Get nearest anchors to a position."""
        anchors_with_distance = [
            (anchor, position.distance_to(anchor.position))
            for anchor in self.anchors.values()
        ]
        anchors_with_distance.sort(key=lambda x: x[1])
        return [anchor for anchor, _ in anchors_with_distance[:limit]]

    def get_cluster(self, cluster_id: str) -> Optional[AnchorCluster]:
        """Get cluster by ID."""
        return self.clusters.get(cluster_id)

    def get_clusters_for_location(self, location_hint: str) -> List[AnchorCluster]:
        """Get all clusters in a location."""
        return [
            cluster for cluster in self.clusters.values()
            if cluster.location_hint == location_hint
        ]

    # === Anchor Removal ===

    def remove_anchor(self, anchor_id: str) -> bool:
        """Remove an anchor."""
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return False

        # Remove from node mapping
        if anchor.node_id in self._node_to_anchor:
            del self._node_to_anchor[anchor.node_id]

        # Remove from clusters
        for cluster in self.clusters.values():
            cluster.anchor_ids.discard(anchor_id)

        # Clean up empty clusters
        empty_clusters = [
            cid for cid, cluster in self.clusters.items()
            if not cluster.anchor_ids
        ]
        for cid in empty_clusters:
            del self.clusters[cid]

        del self.anchors[anchor_id]
        self._save()
        return True

    # === Cloud Anchors ===

    def upload_to_cloud(self, anchor_id: str) -> Optional[str]:
        """
        Upload anchor to cloud for cross-device sharing.
        Returns cloud anchor ID.

        Note: Actual cloud upload requires AR Cloud service integration
        (e.g., Google Cloud Anchors, Azure Spatial Anchors)
        """
        anchor = self.anchors.get(anchor_id)
        if not anchor:
            return None

        # Simulate cloud anchor ID generation
        # In production, this would call the AR cloud service
        cloud_id = f"cloud_{hashlib.sha256(anchor_id.encode()).hexdigest()[:12]}"
        anchor.cloud_anchor_id = cloud_id

        self._save()
        return cloud_id

    def resolve_cloud_anchor(self, cloud_anchor_id: str) -> Optional[SpatialAnchor]:
        """
        Resolve a cloud anchor to local anchor.

        Note: Actual resolution requires AR Cloud service integration
        """
        for anchor in self.anchors.values():
            if anchor.cloud_anchor_id == cloud_anchor_id:
                return anchor
        return None

    # === Statistics ===

    def get_statistics(self) -> Dict[str, Any]:
        """Get anchor statistics."""
        states = {}
        qualities = {}

        for anchor in self.anchors.values():
            states[anchor.state.value] = states.get(anchor.state.value, 0) + 1
            qualities[anchor.quality.value] = qualities.get(anchor.quality.value, 0) + 1

        return {
            "total_anchors": len(self.anchors),
            "total_clusters": len(self.clusters),
            "by_state": states,
            "by_quality": qualities,
            "avg_persistence_score": sum(a.persistence_score for a in self.anchors.values()) / max(1, len(self.anchors))
        }

    # === Persistence ===

    def _save(self):
        """Save anchors to storage."""
        if not self.storage_path:
            return

        data = {
            "anchors": {aid: a.to_dict() for aid, a in self.anchors.items()},
            "clusters": {
                cid: {
                    "cluster_id": c.cluster_id,
                    "location_hint": c.location_hint,
                    "anchor_ids": list(c.anchor_ids),
                    "centroid": c.centroid.to_dict(),
                    "radius": c.radius
                }
                for cid, c in self.clusters.items()
            },
            "counter": self._anchor_counter
        }

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load anchors from storage."""
        if not self.storage_path:
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            self._anchor_counter = data.get("counter", 0)

            # Load anchors
            for aid, adata in data.get("anchors", {}).items():
                anchor = SpatialAnchor.from_dict(adata)
                self.anchors[aid] = anchor
                self._node_to_anchor[anchor.node_id] = aid

            # Load clusters
            for cid, cdata in data.get("clusters", {}).items():
                cluster = AnchorCluster(
                    cluster_id=cdata["cluster_id"],
                    location_hint=cdata["location_hint"],
                    anchor_ids=set(cdata.get("anchor_ids", [])),
                    centroid=WorldPosition.from_dict(cdata.get("centroid", {})),
                    radius=cdata.get("radius", 2.0)
                )
                self.clusters[cid] = cluster

        except FileNotFoundError:
            pass

    def export_for_ar(self) -> str:
        """Export anchors as JSON for AR client."""
        data = {
            "anchors": [a.to_dict() for a in self.anchors.values()],
            "clusters": [
                {
                    "cluster_id": c.cluster_id,
                    "location_hint": c.location_hint,
                    "centroid": c.centroid.to_dict(),
                    "anchor_count": len(c.anchor_ids)
                }
                for c in self.clusters.values()
            ]
        }
        return json.dumps(data, indent=2)


# Quick test
if __name__ == "__main__":
    print("Testing Spatial Anchor Manager...")

    manager = SpatialAnchorManager()

    # Create some anchors
    pos1 = WorldPosition(x=1.0, y=0.5, z=2.0)
    anchor1 = manager.create_anchor(
        node_id="thompson_sampling",
        position=pos1,
        location_hint="office_desk",
        label="Thompson Sampling"
    )
    print(f"Created anchor: {anchor1.anchor_id}")

    pos2 = WorldPosition(x=1.2, y=0.5, z=2.1)
    anchor2 = manager.create_anchor(
        node_id="bayesian_methods",
        position=pos2,
        location_hint="office_desk",
        label="Bayesian Methods"
    )

    # Start tracking
    manager.start_tracking(anchor1.anchor_id)
    print(f"Anchor state: {anchor1.state.value}")

    # Get nearby
    nearby = manager.get_nearby_anchors(pos1, limit=5)
    print(f"Nearby anchors: {[a.label for a in nearby]}")

    # Statistics
    stats = manager.get_statistics()
    print(f"Stats: {stats}")

    # Export for AR
    ar_json = manager.export_for_ar()
    print(f"AR export length: {len(ar_json)} chars")

"""
Tests for HoloLoom Phase 4 Spatial Computing modules.

Tests coverage for:
- collaborative_session.py - Bridge functionality (spatial + collaboration)
- knowledge_overlay.py - AR overlay system
- mobile_spatial_ui.py - Touch gestures, responsive UI

Created: December 2025
"""

import pytest
import asyncio
import math
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

# Phase 4 Spatial - Collaborative Session
from hololoom.spatial.collaborative_session import (
    SpatialActivityType, SpatialZoneType, GestureAction,
    SpatialContext, SpatialZone, SharedSpatialObject, GestureMapping,
    ARContextProvider, GestureCollaborationBridge, CollaborativeSpatialSession,
    create_collaborative_spatial_session
)

# Phase 4 Spatial - Knowledge Overlay
from hololoom.spatial.knowledge_overlay import (
    OverlayStyle, EdgeStyle, LayoutAlgorithm, VisibilityMode,
    KnowledgeNodeOverlay, KnowledgeEdgeOverlay, OverlayCluster, MemoryPalaceRoom,
    LayoutEngine, KnowledgeOverlayManager,
    create_knowledge_overlay_manager
)

# Phase 4 Spatial - Mobile Spatial UI
from hololoom.spatial.mobile_spatial_ui import (
    DeviceType, ScreenOrientation, InputMode, UIScale,
    TouchGestureType, UIComponentType, UIAnchor, LayoutBreakpoint,
    TouchGesture, DeviceCapabilities, ResponsiveValue, LayoutConstraints,
    UIComponent, FloatingActionButton, BottomSheet, RadialMenu, SpatialHUD,
    TouchPoint, TouchGestureRecognizer, MobileSpatialUIManager,
    create_mobile_ui_manager, create_default_spatial_ui
)

# Shared math types
from hololoom.spatial.math_types import Vector3, Quaternion, Color, Transform

# Collaboration imports for session types
from hololoom.collaboration.session import SessionType


# ============================================================================
# Tests for collaborative_session.py
# ============================================================================

class TestSpatialActivityType:
    """Tests for SpatialActivityType enum."""

    def test_activity_types_exist(self):
        """Verify all activity types are defined."""
        assert SpatialActivityType.IDLE is not None
        assert SpatialActivityType.VIEWING is not None
        assert SpatialActivityType.MANIPULATING is not None
        assert SpatialActivityType.DRAWING is not None
        assert SpatialActivityType.ANNOTATING is not None
        assert SpatialActivityType.PRESENTING is not None
        assert SpatialActivityType.EXPLORING is not None
        assert SpatialActivityType.COLLABORATING is not None
        assert SpatialActivityType.SEARCHING is not None
        assert SpatialActivityType.ORGANIZING is not None


class TestSpatialZoneType:
    """Tests for SpatialZoneType enum."""

    def test_zone_types_exist(self):
        """Verify all zone types are defined."""
        assert SpatialZoneType.MAIN_STAGE is not None
        assert SpatialZoneType.DISCUSSION_AREA is not None
        assert SpatialZoneType.PRIVATE_BOOTH is not None
        assert SpatialZoneType.WHITEBOARD_ZONE is not None
        assert SpatialZoneType.KNOWLEDGE_GARDEN is not None
        assert SpatialZoneType.ARCHIVE_VAULT is not None
        assert SpatialZoneType.CREATION_STUDIO is not None
        assert SpatialZoneType.OBSERVATION_DECK is not None


class TestGestureAction:
    """Tests for GestureAction enum."""

    def test_gesture_actions_exist(self):
        """Verify all gesture actions are defined."""
        assert GestureAction.WAVE is not None
        assert GestureAction.POINT is not None
        assert GestureAction.THUMBS_UP is not None
        assert GestureAction.THUMBS_DOWN is not None
        assert GestureAction.RAISE_HAND is not None
        assert GestureAction.PINCH is not None
        assert GestureAction.SPREAD is not None


class TestSpatialContext:
    """Tests for SpatialContext dataclass."""

    def test_context_creation(self):
        """Test creating a spatial context."""
        ctx = SpatialContext(user_id="user_001")

        assert ctx.user_id == "user_001"
        assert ctx.activity == SpatialActivityType.IDLE
        assert ctx.current_zone is None
        assert ctx.nearby_users == []
        assert ctx.held_objects == []

    def test_context_with_position(self):
        """Test context with position data."""
        ctx = SpatialContext(
            user_id="user_002",
            position=Vector3(1.0, 0.5, 2.0),
            rotation=Quaternion(0, 0, 0, 1),
            current_zone=SpatialZoneType.MAIN_STAGE
        )

        assert ctx.position.x == 1.0
        assert ctx.position.y == 0.5
        assert ctx.position.z == 2.0
        assert ctx.current_zone == SpatialZoneType.MAIN_STAGE

    def test_context_to_dict(self):
        """Test serialization to dict."""
        ctx = SpatialContext(
            user_id="user_003",
            activity=SpatialActivityType.VIEWING,
            gaze_target="object:123"
        )

        data = ctx.to_dict()
        assert data["user_id"] == "user_003"
        assert data["activity"] == "VIEWING"
        assert data["gaze_target"] == "object:123"


class TestSpatialZone:
    """Tests for SpatialZone dataclass."""

    def test_zone_creation(self):
        """Test creating a spatial zone."""
        zone = SpatialZone(
            zone_id="zone_001",
            zone_type=SpatialZoneType.DISCUSSION_AREA,
            name="Team Discussion"
        )

        assert zone.zone_id == "zone_001"
        assert zone.zone_type == SpatialZoneType.DISCUSSION_AREA
        assert zone.name == "Team Discussion"
        assert zone.is_public is True

    def test_zone_contains_point(self):
        """Test point containment check."""
        zone = SpatialZone(
            zone_id="zone_002",
            zone_type=SpatialZoneType.MAIN_STAGE,
            name="Main Stage",
            center=Vector3(0, 0, 0),
            size=Vector3(10, 4, 10)
        )

        # Point inside zone
        inside = Vector3(2, 1, 3)
        assert zone.contains_point(inside) is True

        # Point outside zone
        outside = Vector3(20, 0, 0)
        assert zone.contains_point(outside) is False

    def test_zone_to_dict(self):
        """Test zone serialization."""
        zone = SpatialZone(
            zone_id="zone_003",
            zone_type=SpatialZoneType.WHITEBOARD_ZONE,
            name="Whiteboard",
            max_occupants=5
        )

        data = zone.to_dict()
        assert data["zone_id"] == "zone_003"
        assert data["zone_type"] == "WHITEBOARD_ZONE"
        assert data["max_occupants"] == 5


class TestSharedSpatialObject:
    """Tests for SharedSpatialObject dataclass."""

    def test_object_creation(self):
        """Test creating a shared spatial object."""
        obj = SharedSpatialObject(
            object_id="obj_001",
            name="Knowledge Cube",
            object_type="physics"
        )

        assert obj.object_id == "obj_001"
        assert obj.name == "Knowledge Cube"
        assert obj.is_locked is False
        assert obj.is_grabbable is True

    def test_object_with_knowledge_link(self):
        """Test object linked to HoloLoom memory."""
        obj = SharedSpatialObject(
            object_id="obj_002",
            name="Memory Node",
            object_type="knowledge_node",
            knowledge_node_id="node_123",
            memory_id="mem_456"
        )

        assert obj.knowledge_node_id == "node_123"
        assert obj.memory_id == "mem_456"

    def test_object_to_dict(self):
        """Test object serialization."""
        obj = SharedSpatialObject(
            object_id="obj_003",
            name="Test Object",
            object_type="annotation",
            owner_id="user_001"
        )

        data = obj.to_dict()
        assert data["object_id"] == "obj_003"
        assert data["owner_id"] == "user_001"


class TestGestureMapping:
    """Tests for GestureMapping dataclass."""

    def test_mapping_creation(self):
        """Test creating a gesture mapping."""
        from hololoom.spatial.hand_tracking import GestureType

        mapping = GestureMapping(
            gesture_type=GestureType.WAVE,
            action=GestureAction.WAVE
        )

        assert mapping.action == GestureAction.WAVE
        assert mapping.requires_target is False

    def test_mapping_can_trigger(self):
        """Test cooldown checking."""
        from hololoom.spatial.hand_tracking import GestureType

        mapping = GestureMapping(
            gesture_type=GestureType.POINT,
            action=GestureAction.POINT,
            cooldown_seconds=1.0
        )

        # Should be able to trigger (no previous trigger)
        assert mapping.can_trigger() is True

        # Mark as triggered
        mapping.last_triggered = datetime.now()

        # Should not trigger during cooldown
        assert mapping.can_trigger() is False


class TestCollaborativeSpatialSession:
    """Tests for CollaborativeSpatialSession class."""

    def test_session_creation(self):
        """Test creating a collaborative spatial session."""
        session = CollaborativeSpatialSession(
            session_id="session_001",
            name="Team Space",
            owner_id="owner_001",
            owner_name="Alice"
        )

        assert session.session_id == "session_001"
        assert session.name == "Team Space"
        assert session.owner_id == "owner_001"

        # Should have default zones
        assert len(session.zones) > 0
        assert "main_stage" in session.zones

    def test_factory_function(self):
        """Test create_collaborative_spatial_session factory."""
        session = create_collaborative_spatial_session(
            name="Research Lab",
            owner_id="researcher_001",
            owner_name="Dr. Smith"
        )

        assert session.name == "Research Lab"
        assert session.session_id is not None

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test session start and stop."""
        session = create_collaborative_spatial_session(
            name="Test Session",
            owner_id="test_owner",
            owner_name="Test Owner"
        )

        await session.start()
        assert session._running is True

        await session.stop()
        assert session._running is False

    @pytest.mark.asyncio
    async def test_user_join_leave(self):
        """Test user join and leave."""
        session = create_collaborative_spatial_session(
            name="Join Test",
            owner_id="owner_001",
            owner_name="Owner"
        )

        await session.start()

        # User joins
        avatar, presence, _ = await session.join(
            user_id="user_001",
            display_name="Test User"
        )

        assert avatar is not None
        assert presence is not None
        assert "user_001" in session.context_provider.user_contexts

        # User leaves
        await session.leave("user_001")
        assert "user_001" not in session.context_provider.user_contexts

        await session.stop()

    @pytest.mark.asyncio
    async def test_shared_object_creation(self):
        """Test creating shared objects."""
        session = create_collaborative_spatial_session(
            name="Object Test",
            owner_id="owner_001",
            owner_name="Owner"
        )

        await session.start()

        obj = await session.create_shared_object(
            creator_id="owner_001",
            name="Test Cube",
            object_type="physics",
            position=Vector3(1, 1, 1)
        )

        assert obj.name == "Test Cube"
        assert obj.object_id in session.shared_objects

        await session.stop()

    @pytest.mark.asyncio
    async def test_grab_release_object(self):
        """Test object grab and release."""
        session = create_collaborative_spatial_session(
            name="Grab Test",
            owner_id="owner_001",
            owner_name="Owner"
        )

        await session.start()
        await session.join("user_001", "User One")

        obj = await session.create_shared_object(
            creator_id="owner_001",
            name="Grabbable",
            object_type="physics",
            position=Vector3(0, 1, 0)
        )

        # Grab object
        grabbed = await session.grab_object("user_001", obj.object_id)
        assert grabbed is True
        assert obj.is_locked is True
        assert obj.locked_by == "user_001"

        # Release object
        released = await session.release_object("user_001", obj.object_id)
        assert released is True
        assert obj.is_locked is False

        await session.stop()

    def test_session_to_state(self):
        """Test session state export."""
        session = create_collaborative_spatial_session(
            name="State Test",
            owner_id="owner_001",
            owner_name="Owner"
        )

        state = session.to_state()
        assert "session_id" in state
        assert "name" in state
        assert "zones" in state
        assert "shared_objects" in state

    def test_event_registration(self):
        """Test event handler registration."""
        session = create_collaborative_spatial_session(
            name="Event Test",
            owner_id="owner_001",
            owner_name="Owner"
        )

        events_received = []

        def handler(event, data):
            events_received.append((event, data))

        session.on("user_joined", handler)

        # Emit event manually
        session._emit_event("user_joined", {"user_id": "test"})

        assert len(events_received) == 1
        assert events_received[0][0] == "user_joined"


class TestARContextProvider:
    """Tests for ARContextProvider class."""

    def test_context_update(self):
        """Test context updates."""
        session = create_collaborative_spatial_session(
            name="Context Test",
            owner_id="owner",
            owner_name="Owner"
        )

        provider = session.context_provider

        ctx = provider.update_context(
            user_id="user_001",
            position=Vector3(1, 0, 2),
            rotation=Quaternion.identity()
        )

        assert ctx.user_id == "user_001"
        assert ctx.position.x == 1.0
        assert ctx.position.z == 2.0

    def test_nearby_users_detection(self):
        """Test finding nearby users."""
        session = create_collaborative_spatial_session(
            name="Nearby Test",
            owner_id="owner",
            owner_name="Owner"
        )

        provider = session.context_provider

        # Add two users close together
        provider.update_context("user_001", Vector3(0, 0, 0), Quaternion.identity())
        provider.update_context("user_002", Vector3(1, 0, 1), Quaternion.identity())

        # Update user_001 context again to detect nearby users (user_002 now exists)
        ctx = provider.update_context("user_001", Vector3(0, 0, 0), Quaternion.identity())
        assert "user_002" in ctx.nearby_users

    def test_zone_detection(self):
        """Test zone detection from position."""
        session = create_collaborative_spatial_session(
            name="Zone Test",
            owner_id="owner",
            owner_name="Owner"
        )

        provider = session.context_provider

        # Update position to be in main stage (center)
        ctx = provider.update_context(
            user_id="user_001",
            position=Vector3(0, 0, 0),
            rotation=Quaternion.identity()
        )

        assert ctx.current_zone == SpatialZoneType.MAIN_STAGE


class TestGestureCollaborationBridge:
    """Tests for GestureCollaborationBridge class."""

    def test_default_mappings(self):
        """Test default gesture mappings are registered."""
        session = create_collaborative_spatial_session(
            name="Bridge Test",
            owner_id="owner",
            owner_name="Owner"
        )

        bridge = session.gesture_bridge

        # Should have default mappings
        assert len(bridge.mappings) > 0

    def test_add_mapping(self):
        """Test adding custom gesture mapping."""
        from hololoom.spatial.hand_tracking import GestureType

        session = create_collaborative_spatial_session(
            name="Mapping Test",
            owner_id="owner",
            owner_name="Owner"
        )

        bridge = session.gesture_bridge

        bridge.add_mapping(
            gesture_type=GestureType.FIST,
            action=GestureAction.CLAP,
            requires_target=False,
            cooldown=2.0
        )

        assert GestureType.FIST in bridge.mappings

    def test_to_state(self):
        """Test bridge state export."""
        session = create_collaborative_spatial_session(
            name="State Test",
            owner_id="owner",
            owner_name="Owner"
        )

        state = session.gesture_bridge.to_state()
        assert "mappings" in state


# ============================================================================
# Tests for knowledge_overlay.py
# ============================================================================

class TestOverlayEnums:
    """Tests for overlay-related enums."""

    def test_overlay_style(self):
        """Test OverlayStyle enum."""
        assert OverlayStyle.CARD is not None
        assert OverlayStyle.SPHERE is not None
        assert OverlayStyle.HOLOGRAM is not None
        assert OverlayStyle.MEMORY_PALACE is not None

    def test_edge_style(self):
        """Test EdgeStyle enum."""
        assert EdgeStyle.LINE is not None
        assert EdgeStyle.ARROW is not None
        assert EdgeStyle.PARTICLES is not None

    def test_layout_algorithm(self):
        """Test LayoutAlgorithm enum."""
        assert LayoutAlgorithm.FORCE_DIRECTED is not None
        assert LayoutAlgorithm.HIERARCHICAL is not None
        assert LayoutAlgorithm.RADIAL is not None
        assert LayoutAlgorithm.CIRCULAR is not None

    def test_visibility_mode(self):
        """Test VisibilityMode enum."""
        assert VisibilityMode.ALWAYS is not None
        assert VisibilityMode.PROXIMITY is not None
        assert VisibilityMode.GAZE is not None


class TestKnowledgeNodeOverlay:
    """Tests for KnowledgeNodeOverlay dataclass."""

    def test_overlay_creation(self):
        """Test creating a knowledge overlay."""
        overlay = KnowledgeNodeOverlay(
            overlay_id="overlay_001",
            node_id="node_001",
            title="Thompson Sampling"
        )

        assert overlay.overlay_id == "overlay_001"
        assert overlay.title == "Thompson Sampling"
        assert overlay.style == OverlayStyle.CARD

    def test_overlay_with_content(self):
        """Test overlay with full content."""
        overlay = KnowledgeNodeOverlay(
            overlay_id="overlay_002",
            node_id="node_002",
            title="Test Node",
            summary="Short summary",
            content="Full content here",
            tags=["ml", "algorithm"],
            confidence=0.95,
            importance=0.8
        )

        assert overlay.summary == "Short summary"
        assert len(overlay.tags) == 2
        assert overlay.confidence == 0.95

    def test_overlay_to_dict(self):
        """Test overlay serialization."""
        overlay = KnowledgeNodeOverlay(
            overlay_id="overlay_003",
            node_id="node_003",
            title="Serialization Test",
            style=OverlayStyle.HOLOGRAM
        )

        data = overlay.to_dict()
        assert data["overlay_id"] == "overlay_003"
        assert data["style"] == "HOLOGRAM"


class TestKnowledgeEdgeOverlay:
    """Tests for KnowledgeEdgeOverlay dataclass."""

    def test_edge_creation(self):
        """Test creating a knowledge edge."""
        edge = KnowledgeEdgeOverlay(
            edge_id="edge_001",
            source_overlay_id="overlay_001",
            target_overlay_id="overlay_002",
            relationship_type="IS_A"
        )

        assert edge.edge_id == "edge_001"
        assert edge.relationship_type == "IS_A"
        assert edge.style == EdgeStyle.LINE

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = KnowledgeEdgeOverlay(
            edge_id="edge_002",
            source_overlay_id="src",
            target_overlay_id="tgt",
            relationship_type="USES",
            style=EdgeStyle.PARTICLES
        )

        data = edge.to_dict()
        assert data["relationship_type"] == "USES"
        assert data["style"] == "PARTICLES"


class TestOverlayCluster:
    """Tests for OverlayCluster dataclass."""

    def test_cluster_creation(self):
        """Test creating an overlay cluster."""
        cluster = OverlayCluster(
            cluster_id="cluster_001",
            name="ML Concepts",
            overlays=["overlay_001", "overlay_002"]
        )

        assert cluster.cluster_id == "cluster_001"
        assert len(cluster.overlays) == 2

    def test_cluster_to_dict(self):
        """Test cluster serialization."""
        cluster = OverlayCluster(
            cluster_id="cluster_002",
            name="Test Cluster",
            radius=3.0
        )

        data = cluster.to_dict()
        assert data["name"] == "Test Cluster"
        assert data["radius"] == 3.0


class TestMemoryPalaceRoom:
    """Tests for MemoryPalaceRoom dataclass."""

    def test_room_creation(self):
        """Test creating a memory palace room."""
        room = MemoryPalaceRoom(
            room_id="room_001",
            name="Library",
            theme="library"
        )

        assert room.room_id == "room_001"
        assert room.theme == "library"
        assert room.is_discovered is False

    def test_room_to_dict(self):
        """Test room serialization."""
        room = MemoryPalaceRoom(
            room_id="room_002",
            name="Garden",
            theme="garden",
            is_discovered=True
        )

        data = room.to_dict()
        assert data["theme"] == "garden"
        assert data["is_discovered"] is True


class TestLayoutEngine:
    """Tests for LayoutEngine class."""

    def test_engine_creation(self):
        """Test creating a layout engine."""
        engine = LayoutEngine()

        assert engine.algorithm == LayoutAlgorithm.FORCE_DIRECTED
        assert engine.spread > 0

    def test_circular_layout(self):
        """Test circular layout algorithm."""
        engine = LayoutEngine()

        overlays = [
            KnowledgeNodeOverlay(overlay_id=f"o{i}", node_id=f"n{i}", title=f"Node {i}")
            for i in range(4)
        ]

        positions = engine.compute_layout(overlays, [], LayoutAlgorithm.CIRCULAR)

        assert len(positions) == 4
        for oid in positions:
            assert isinstance(positions[oid], Vector3)

    def test_grid_layout(self):
        """Test grid layout algorithm."""
        engine = LayoutEngine()

        overlays = [
            KnowledgeNodeOverlay(overlay_id=f"o{i}", node_id=f"n{i}", title=f"Node {i}")
            for i in range(9)
        ]

        positions = engine.compute_layout(overlays, [], LayoutAlgorithm.GRID)

        assert len(positions) == 9

    def test_force_directed_layout(self):
        """Test force-directed layout algorithm."""
        engine = LayoutEngine()
        engine.iterations = 10  # Reduce iterations for test speed

        overlays = [
            KnowledgeNodeOverlay(overlay_id=f"o{i}", node_id=f"n{i}", title=f"Node {i}")
            for i in range(5)
        ]

        edges = [
            KnowledgeEdgeOverlay(
                edge_id="e1",
                source_overlay_id="o0",
                target_overlay_id="o1",
                relationship_type="RELATED"
            )
        ]

        positions = engine.compute_layout(overlays, edges, LayoutAlgorithm.FORCE_DIRECTED)

        assert len(positions) == 5


class TestKnowledgeOverlayManager:
    """Tests for KnowledgeOverlayManager class."""

    def test_manager_creation(self):
        """Test creating a knowledge overlay manager."""
        manager = create_knowledge_overlay_manager()

        assert manager is not None
        assert len(manager.overlays) == 0

    def test_create_overlay(self):
        """Test creating overlays through manager."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(
            node_id="node_001",
            title="Test Overlay",
            summary="Test summary",
            tags=["test", "example"]
        )

        assert overlay.overlay_id in manager.overlays
        assert overlay.title == "Test Overlay"

    def test_create_edge(self):
        """Test creating edges between overlays."""
        manager = create_knowledge_overlay_manager()

        o1 = manager.create_overlay(node_id="n1", title="Node 1")
        o2 = manager.create_overlay(node_id="n2", title="Node 2")

        edge = manager.create_edge(
            source_overlay_id=o1.overlay_id,
            target_overlay_id=o2.overlay_id,
            relationship_type="IS_A"
        )

        assert edge.edge_id in manager.edges

    def test_remove_overlay(self):
        """Test removing overlays."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(node_id="n1", title="To Remove")
        overlay_id = overlay.overlay_id

        removed = manager.remove_overlay(overlay_id)

        assert removed is True
        assert overlay_id not in manager.overlays

    def test_get_overlay_by_node(self):
        """Test finding overlay by node ID."""
        manager = create_knowledge_overlay_manager()

        manager.create_overlay(node_id="node_123", title="Target")

        found = manager.get_overlay_by_node("node_123")

        assert found is not None
        assert found.title == "Target"

    def test_select_overlay(self):
        """Test overlay selection."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(node_id="n1", title="Selectable")

        selected = manager.select_overlay(overlay.overlay_id)

        assert selected is True
        assert overlay.is_selected is True
        assert overlay.access_count == 1

    def test_expand_collapse_overlay(self):
        """Test overlay expand and collapse."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(node_id="n1", title="Expandable")

        manager.expand_overlay(overlay.overlay_id)
        assert overlay.is_expanded is True

        manager.collapse_overlay(overlay.overlay_id)
        assert overlay.is_expanded is False

    def test_pin_unpin_overlay(self):
        """Test overlay pin and unpin."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(node_id="n1", title="Pinnable")

        manager.pin_overlay(overlay.overlay_id)
        assert overlay.is_pinned is True

        manager.unpin_overlay(overlay.overlay_id)
        assert overlay.is_pinned is False

    def test_create_cluster(self):
        """Test creating overlay cluster."""
        manager = create_knowledge_overlay_manager()

        o1 = manager.create_overlay(node_id="n1", title="Node 1")
        o2 = manager.create_overlay(node_id="n2", title="Node 2")

        cluster = manager.create_cluster(
            name="Test Cluster",
            overlay_ids=[o1.overlay_id, o2.overlay_id]
        )

        assert cluster.cluster_id in manager.clusters
        assert len(cluster.overlays) == 2

    def test_auto_cluster(self):
        """Test automatic clustering by tags."""
        manager = create_knowledge_overlay_manager()

        # Create overlays with same tag
        manager.create_overlay(node_id="n1", title="ML 1", tags=["ml"])
        manager.create_overlay(node_id="n2", title="ML 2", tags=["ml"])
        manager.create_overlay(node_id="n3", title="DB 1", tags=["database"])

        clusters = manager.auto_cluster(min_cluster_size=2)

        # Should create cluster for "ml" tag (2 items)
        ml_cluster = [c for c in clusters if "ml" in c.name.lower() or c.name == "Ml"]
        assert len(ml_cluster) >= 1

    def test_memory_palace_room_creation(self):
        """Test creating memory palace rooms."""
        manager = create_knowledge_overlay_manager()

        room = manager.create_memory_palace_room(
            name="Library",
            theme="library",
            center=Vector3(0, 0, 0)
        )

        assert room.room_id in manager.rooms
        assert room.theme == "library"

    def test_assign_to_room(self):
        """Test assigning overlay to room."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(node_id="n1", title="Book")
        room = manager.create_memory_palace_room(
            name="Library",
            theme="library",
            center=Vector3(0, 0, 0)
        )

        assigned = manager.assign_to_room(overlay.overlay_id, room.room_id)

        assert assigned is True
        assert overlay.overlay_id in room.overlays

    def test_apply_layout(self):
        """Test applying layout to overlays."""
        manager = create_knowledge_overlay_manager()

        for i in range(4):
            manager.create_overlay(node_id=f"n{i}", title=f"Node {i}")

        manager.apply_layout(algorithm=LayoutAlgorithm.CIRCULAR)

        # All overlays should have positions
        for overlay in manager.overlays.values():
            assert overlay.position is not None

    def test_visibility_update(self):
        """Test visibility update based on viewer position."""
        manager = create_knowledge_overlay_manager()

        overlay = manager.create_overlay(
            node_id="n1",
            title="Test",
            position=Vector3(0, 1.5, -2)
        )
        overlay.visibility_mode = VisibilityMode.PROXIMITY
        overlay.visibility_distance = 5.0

        # Viewer close to overlay
        manager.update_visibility(Vector3(0, 1.5, 0))
        assert overlay.is_visible is True

        # Viewer far from overlay
        manager.update_visibility(Vector3(100, 0, 0))
        assert overlay.is_visible is False

    def test_to_state(self):
        """Test manager state export."""
        manager = create_knowledge_overlay_manager()

        manager.create_overlay(node_id="n1", title="Node 1")

        state = manager.to_state()

        assert "overlays" in state
        assert "edges" in state
        assert "clusters" in state
        assert "rooms" in state
        assert state["overlay_count"] == 1

    def test_create_overlays_from_knowledge(self):
        """Test creating overlays from knowledge graph data."""
        manager = create_knowledge_overlay_manager()

        nodes = [
            {"id": "n1", "title": "Node 1", "tags": ["test"]},
            {"id": "n2", "title": "Node 2", "tags": ["test"]}
        ]

        relationships = [
            {"source_id": "n1", "target_id": "n2", "relationship_type": "USES"}
        ]

        overlays, edges = manager.create_overlays_from_knowledge(nodes, relationships)

        assert len(overlays) == 2
        assert len(edges) == 1


# ============================================================================
# Tests for mobile_spatial_ui.py
# ============================================================================

class TestDeviceEnums:
    """Tests for device-related enums."""

    def test_device_type(self):
        """Test DeviceType enum."""
        assert DeviceType.MOBILE_PHONE is not None
        assert DeviceType.TABLET is not None
        assert DeviceType.DESKTOP is not None
        assert DeviceType.VR_HEADSET is not None

    def test_screen_orientation(self):
        """Test ScreenOrientation enum."""
        assert ScreenOrientation.PORTRAIT is not None
        assert ScreenOrientation.LANDSCAPE is not None

    def test_input_mode(self):
        """Test InputMode enum."""
        assert InputMode.TOUCH is not None
        assert InputMode.MOUSE is not None
        assert InputMode.CONTROLLER is not None

    def test_ui_scale(self):
        """Test UIScale enum."""
        assert UIScale.COMPACT is not None
        assert UIScale.NORMAL is not None
        assert UIScale.COMFORTABLE is not None


class TestTouchGestureType:
    """Tests for TouchGestureType enum."""

    def test_gesture_types(self):
        """Test all gesture types exist."""
        assert TouchGestureType.TAP is not None
        assert TouchGestureType.DOUBLE_TAP is not None
        assert TouchGestureType.LONG_PRESS is not None
        assert TouchGestureType.SWIPE_LEFT is not None
        assert TouchGestureType.SWIPE_RIGHT is not None
        assert TouchGestureType.PINCH_IN is not None
        assert TouchGestureType.PINCH_OUT is not None


class TestTouchGesture:
    """Tests for TouchGesture dataclass."""

    def test_gesture_creation(self):
        """Test creating a touch gesture."""
        gesture = TouchGesture(
            gesture_type=TouchGestureType.TAP,
            position=(0.5, 0.5)
        )

        assert gesture.gesture_type == TouchGestureType.TAP
        assert gesture.position == (0.5, 0.5)
        assert gesture.finger_count == 1

    def test_gesture_with_delta(self):
        """Test gesture with movement delta."""
        gesture = TouchGesture(
            gesture_type=TouchGestureType.SWIPE_RIGHT,
            position=(0.8, 0.5),
            delta=(0.3, 0.0),
            velocity=2.0
        )

        assert gesture.delta == (0.3, 0.0)
        assert gesture.velocity == 2.0

    def test_gesture_to_dict(self):
        """Test gesture serialization."""
        gesture = TouchGesture(
            gesture_type=TouchGestureType.PINCH_OUT,
            position=(0.5, 0.5),
            scale=1.5,
            finger_count=2
        )

        data = gesture.to_dict()
        assert data["gesture_type"] == "pinch_out"
        assert data["scale"] == 1.5


class TestDeviceCapabilities:
    """Tests for DeviceCapabilities dataclass."""

    def test_capabilities_creation(self):
        """Test creating device capabilities."""
        caps = DeviceCapabilities(
            device_type=DeviceType.MOBILE_PHONE,
            screen_width=375,
            screen_height=812,
            has_touch=True
        )

        assert caps.device_type == DeviceType.MOBILE_PHONE
        assert caps.has_touch is True

    def test_primary_input_detection(self):
        """Test primary input mode detection."""
        # Mobile phone
        mobile = DeviceCapabilities(
            device_type=DeviceType.MOBILE_PHONE,
            screen_width=375,
            screen_height=812,
            has_touch=True
        )
        assert mobile.get_primary_input() == InputMode.TOUCH

        # VR headset
        vr = DeviceCapabilities(
            device_type=DeviceType.VR_HEADSET,
            screen_width=1920,
            screen_height=1080
        )
        assert vr.get_primary_input() == InputMode.CONTROLLER

        # Desktop without touch
        desktop = DeviceCapabilities(
            device_type=DeviceType.DESKTOP,
            screen_width=1920,
            screen_height=1080,
            has_touch=False
        )
        assert desktop.get_primary_input() == InputMode.MOUSE

    def test_recommended_scale(self):
        """Test recommended UI scale."""
        mobile = DeviceCapabilities(
            device_type=DeviceType.MOBILE_PHONE,
            screen_width=375,
            screen_height=812
        )
        assert mobile.get_recommended_scale() == UIScale.COMFORTABLE

        tablet = DeviceCapabilities(
            device_type=DeviceType.TABLET,
            screen_width=1024,
            screen_height=768
        )
        assert tablet.get_recommended_scale() == UIScale.NORMAL

    def test_to_dict(self):
        """Test capabilities serialization."""
        caps = DeviceCapabilities(
            device_type=DeviceType.TABLET,
            screen_width=1024,
            screen_height=768,
            has_ar_support=True
        )

        data = caps.to_dict()
        assert data["device_type"] == "tablet"
        assert data["has_ar_support"] is True


class TestResponsiveValue:
    """Tests for ResponsiveValue dataclass."""

    def test_responsive_value_creation(self):
        """Test creating responsive values."""
        value = ResponsiveValue(xs=1.0, md=0.8, lg=0.5)

        assert value.xs == 1.0
        assert value.md == 0.8

    def test_get_for_width(self):
        """Test getting value for screen width."""
        value = ResponsiveValue(xs=1.0, sm=0.9, md=0.7, lg=0.5)

        # XS width
        assert value.get_for_width(320) == 1.0

        # SM width
        assert value.get_for_width(600) == 0.9

        # MD width
        assert value.get_for_width(900) == 0.7

        # LG width
        assert value.get_for_width(1200) == 0.5

    def test_fallback_to_first_value(self):
        """Test fallback when breakpoint not defined."""
        value = ResponsiveValue(md=0.5)

        # Should fallback to first available value
        result = value.get_for_width(320)
        assert result == 0.5


class TestUIComponent:
    """Tests for UIComponent dataclass."""

    def test_component_creation(self):
        """Test creating a UI component."""
        component = UIComponent(
            component_id="btn_001",
            component_type=UIComponentType.BUTTON,
            label="Click Me"
        )

        assert component.component_id == "btn_001"
        assert component.component_type == UIComponentType.BUTTON
        assert component.visible is True

    def test_computed_position(self):
        """Test computing screen position."""
        component = UIComponent(
            component_id="btn",
            component_type=UIComponentType.BUTTON,
            anchor=UIAnchor.BOTTOM_RIGHT,
            offset_x=-0.1,
            offset_y=-0.1
        )

        pos = component.get_computed_position(1000, 800, (0, 0, 0, 0))

        # Bottom right corner minus offsets
        assert pos[0] < 1000
        assert pos[1] < 800

    def test_to_dict(self):
        """Test component serialization."""
        component = UIComponent(
            component_id="test",
            component_type=UIComponentType.PANEL,
            label="Test Panel"
        )

        data = component.to_dict()
        assert data["component_id"] == "test"
        assert data["component_type"] == "panel"


class TestFloatingActionButton:
    """Tests for FloatingActionButton dataclass."""

    def test_fab_creation(self):
        """Test creating a floating action button."""
        fab = FloatingActionButton(
            component_id="fab_main",
            component_type=UIComponentType.FLOATING_BUTTON,
            icon="add"
        )

        assert fab.size == 56.0
        assert fab.component_type == UIComponentType.FLOATING_BUTTON

    def test_mini_fab(self):
        """Test mini floating action button."""
        fab = FloatingActionButton(
            component_id="fab_mini",
            component_type=UIComponentType.FLOATING_BUTTON,
            mini=True
        )

        fab.__post_init__()
        assert fab.size == 40.0


class TestBottomSheet:
    """Tests for BottomSheet dataclass."""

    def test_bottom_sheet_creation(self):
        """Test creating a bottom sheet."""
        sheet = BottomSheet(
            component_id="sheet_001",
            component_type=UIComponentType.BOTTOM_SHEET,
            label="Details"
        )

        sheet.__post_init__()
        assert sheet.anchor == UIAnchor.BOTTOM_CENTER
        assert sheet.current_state == "collapsed"

    def test_snap_points(self):
        """Test snap points configuration."""
        sheet = BottomSheet(
            component_id="sheet",
            component_type=UIComponentType.BOTTOM_SHEET,
            snap_points=[0.1, 0.5, 0.9]
        )

        assert len(sheet.snap_points) == 3


class TestRadialMenu:
    """Tests for RadialMenu dataclass."""

    def test_radial_menu_creation(self):
        """Test creating a radial menu."""
        menu = RadialMenu(
            component_id="menu_001",
            component_type=UIComponentType.RADIAL_MENU,
            items=[
                {"icon": "home", "label": "Home"},
                {"icon": "settings", "label": "Settings"}
            ]
        )

        menu.__post_init__()
        assert len(menu.items) == 2
        assert menu.is_open is False

    def test_item_angle_calculation(self):
        """Test item angle calculation."""
        menu = RadialMenu(
            component_id="menu",
            component_type=UIComponentType.RADIAL_MENU,
            items=[{}, {}, {}, {}]
        )

        angle_0 = menu.get_item_angle(0)
        angle_1 = menu.get_item_angle(1)

        # Should be 90 degrees apart (4 items)
        diff = abs(angle_1 - angle_0)
        assert abs(diff - math.pi / 2) < 0.01

    def test_item_position(self):
        """Test item position calculation."""
        menu = RadialMenu(
            component_id="menu",
            component_type=UIComponentType.RADIAL_MENU,
            items=[{}, {}],
            radius=100,
            inner_radius=30
        )

        pos = menu.get_item_position(0)

        # Should be at some distance from center
        distance = math.sqrt(pos[0]**2 + pos[1]**2)
        assert distance > 0


class TestSpatialHUD:
    """Tests for SpatialHUD dataclass."""

    def test_hud_creation(self):
        """Test creating a spatial HUD."""
        hud = SpatialHUD(hud_id="main_hud")

        assert hud.hud_id == "main_hud"
        assert hud.visible is True
        assert len(hud.elements) == 0

    def test_add_element(self):
        """Test adding elements to HUD."""
        hud = SpatialHUD(hud_id="test_hud")

        compass = UIComponent(
            component_id="compass",
            component_type=UIComponentType.COMPASS
        )

        hud.add_element(compass)

        assert "compass" in hud.elements

    def test_remove_element(self):
        """Test removing elements from HUD."""
        hud = SpatialHUD(hud_id="test_hud")

        element = UIComponent(
            component_id="temp",
            component_type=UIComponentType.TOOLTIP
        )
        hud.add_element(element)

        hud.remove_element("temp")

        assert "temp" not in hud.elements

    def test_get_world_position(self):
        """Test HUD world position calculation."""
        hud = SpatialHUD(hud_id="test", follow_distance=2.0)

        pos = hud.get_world_position(
            viewer_position=Vector3(0, 0, 0),
            viewer_rotation=Quaternion.identity()
        )

        # Should be 2 meters in front of viewer
        assert abs(pos.z - (-2.0)) < 0.01


class TestTouchGestureRecognizer:
    """Tests for TouchGestureRecognizer class."""

    def test_recognizer_creation(self):
        """Test creating a gesture recognizer."""
        recognizer = TouchGestureRecognizer()

        assert recognizer.tap_threshold_ms > 0
        assert recognizer.long_press_threshold_ms > 0

    @pytest.mark.asyncio
    async def test_touch_start(self):
        """Test touch start handling."""
        recognizer = TouchGestureRecognizer()

        result = recognizer.on_touch_start(0, 0.5, 0.5)

        assert 0 in recognizer.active_touches
        assert result is None  # No gesture yet

    @pytest.mark.asyncio
    async def test_touch_end_tap(self):
        """Test tap detection on touch end."""
        recognizer = TouchGestureRecognizer()

        # Start touch
        recognizer.on_touch_start(0, 0.5, 0.5)

        # End immediately (tap)
        gesture = recognizer.on_touch_end(0)

        assert gesture is not None
        assert gesture.gesture_type == TouchGestureType.TAP

    @pytest.mark.asyncio
    async def test_touch_move_pan(self):
        """Test pan detection on touch move."""
        recognizer = TouchGestureRecognizer()

        recognizer.on_touch_start(0, 0.5, 0.5)

        gesture = recognizer.on_touch_move(0, 0.6, 0.5)

        assert gesture is not None
        assert gesture.gesture_type == TouchGestureType.PAN

    @pytest.mark.asyncio
    async def test_swipe_detection(self):
        """Test swipe detection."""
        recognizer = TouchGestureRecognizer()

        # Start at left
        recognizer.on_touch_start(0, 0.2, 0.5)

        # Move right quickly
        touch = recognizer.active_touches[0]
        touch.position = (0.7, 0.5)  # Update position

        gesture = recognizer.on_touch_end(0)

        # Should detect swipe right
        assert gesture is not None
        assert gesture.gesture_type in [
            TouchGestureType.SWIPE_RIGHT,
            TouchGestureType.EDGE_SWIPE_LEFT
        ]

    def test_gesture_handler_registration(self):
        """Test gesture handler registration."""
        recognizer = TouchGestureRecognizer()

        taps_received = []

        def tap_handler(gesture):
            taps_received.append(gesture)

        recognizer.on(TouchGestureType.TAP, tap_handler)

        assert TouchGestureType.TAP in recognizer.gesture_handlers


class TestMobileSpatialUIManager:
    """Tests for MobileSpatialUIManager class."""

    def test_manager_creation(self):
        """Test creating a mobile UI manager."""
        manager = MobileSpatialUIManager()

        assert manager.capabilities is not None
        assert len(manager.components) == 0

    def test_factory_function(self):
        """Test create_mobile_ui_manager factory."""
        manager = create_mobile_ui_manager(
            screen_width=1024,
            screen_height=768,
            device_type=DeviceType.TABLET,
            has_ar=True
        )

        assert manager.capabilities.device_type == DeviceType.TABLET
        assert manager.capabilities.has_ar_support is True

    def test_update_capabilities(self):
        """Test updating device capabilities."""
        manager = MobileSpatialUIManager()

        new_caps = DeviceCapabilities(
            device_type=DeviceType.DESKTOP,
            screen_width=1920,
            screen_height=1080,
            gpu_tier=3
        )

        manager.update_capabilities(new_caps)

        assert manager.capabilities.device_type == DeviceType.DESKTOP
        assert manager.render_quality == 1.0

    def test_add_remove_component(self):
        """Test adding and removing components."""
        manager = create_mobile_ui_manager()

        component = UIComponent(
            component_id="test",
            component_type=UIComponentType.PANEL
        )

        manager.add_component(component)
        assert "test" in manager.components

        manager.remove_component("test")
        assert "test" not in manager.components

    def test_create_floating_button(self):
        """Test creating floating action button through manager."""
        manager = create_mobile_ui_manager()

        fab = manager.create_floating_button(
            component_id="main_fab",
            icon="add",
            anchor=UIAnchor.BOTTOM_RIGHT
        )

        assert fab is not None
        assert "main_fab" in manager.components

    def test_create_bottom_sheet(self):
        """Test creating bottom sheet through manager."""
        manager = create_mobile_ui_manager()

        sheet = manager.create_bottom_sheet(
            component_id="detail_sheet",
            header_label="Details"
        )

        assert sheet is not None
        assert "detail_sheet" in manager.components

    def test_create_radial_menu(self):
        """Test creating radial menu through manager."""
        manager = create_mobile_ui_manager()

        menu = manager.create_radial_menu(
            component_id="quick_menu",
            items=[
                {"icon": "home", "action": "go_home"},
                {"icon": "back", "action": "go_back"}
            ]
        )

        assert menu is not None
        assert len(menu.items) == 2

    def test_create_hud(self):
        """Test creating HUD through manager."""
        manager = create_mobile_ui_manager()

        hud = manager.create_hud()

        assert manager.hud is not None
        assert manager.hud.hud_id == "main_hud"

    def test_performance_hints(self):
        """Test getting performance hints."""
        manager = create_mobile_ui_manager()

        hints = manager.get_performance_hints()

        assert "render_quality" in hints
        assert "max_visible_overlays" in hints
        assert "frame_rate_target" in hints

    def test_to_state(self):
        """Test manager state export."""
        manager = create_mobile_ui_manager()

        manager.create_floating_button("fab", "add")

        state = manager.to_state()

        assert "device" in state
        assert "components" in state
        assert "performance" in state

    def test_event_handler_registration(self):
        """Test UI event handler registration."""
        manager = create_mobile_ui_manager()

        events = []

        def handler(data):
            events.append(data)

        manager.on("zoom", handler)

        manager._emit_ui_event("zoom", {"scale": 2.0})

        assert len(events) == 1
        assert events[0]["scale"] == 2.0


class TestDefaultSpatialUI:
    """Tests for create_default_spatial_ui function."""

    def test_create_default_ui(self):
        """Test creating default spatial UI components."""
        manager = create_mobile_ui_manager()

        components = create_default_spatial_ui(manager)

        assert "main_fab" in components
        assert "knowledge_sheet" in components
        assert "quick_menu" in components

        # HUD should also be created
        assert manager.hud is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestSpatialPhase4Integration:
    """Integration tests for Phase 4 spatial modules."""

    @pytest.mark.asyncio
    async def test_collaborative_session_with_overlays(self):
        """Test collaborative session with knowledge overlays."""
        # Create session
        session = create_collaborative_spatial_session(
            name="Integration Test",
            owner_id="owner",
            owner_name="Owner"
        )

        await session.start()

        # Create knowledge overlay manager
        overlay_manager = create_knowledge_overlay_manager()

        # Add overlays
        overlay = overlay_manager.create_overlay(
            node_id="n1",
            title="Test Knowledge",
            tags=["test"]
        )

        # Create shared object linked to knowledge
        await session.join("user_001", "User One")

        obj = await session.create_shared_object(
            creator_id="user_001",
            name="Knowledge Cube",
            object_type="knowledge_node",
            position=overlay.position,
            knowledge_node_id=overlay.node_id
        )

        assert obj.knowledge_node_id == overlay.node_id

        await session.stop()

    def test_mobile_ui_with_overlay_manager(self):
        """Test mobile UI with overlay interactions."""
        # Create mobile UI
        ui_manager = create_mobile_ui_manager(
            device_type=DeviceType.MOBILE_PHONE,
            has_ar=True
        )

        # Create overlay manager
        overlay_manager = create_knowledge_overlay_manager()

        # Add overlays
        for i in range(3):
            overlay_manager.create_overlay(
                node_id=f"n{i}",
                title=f"Node {i}",
                tags=["test"]
            )

        # Apply layout
        overlay_manager.apply_layout(algorithm=LayoutAlgorithm.CIRCULAR)

        # Get performance hints for rendering
        hints = ui_manager.get_performance_hints()

        # Adjust overlay rendering based on hints
        max_overlays = hints["max_visible_overlays"]

        assert max_overlays >= len(overlay_manager.overlays)

    @pytest.mark.asyncio
    async def test_gesture_driven_overlay_interaction(self):
        """Test gesture-based overlay interactions."""
        # Create session
        session = create_collaborative_spatial_session(
            name="Gesture Test",
            owner_id="owner",
            owner_name="Owner"
        )

        await session.start()

        # Create overlay manager
        overlay_manager = create_knowledge_overlay_manager()

        overlay = overlay_manager.create_overlay(
            node_id="n1",
            title="Interactive"
        )

        # Create gesture recognizer
        recognizer = TouchGestureRecognizer()

        # Register handler to select overlay on tap
        def on_tap(gesture):
            overlay_manager.select_overlay(overlay.overlay_id)

        recognizer.on(TouchGestureType.TAP, on_tap)

        # Simulate tap
        recognizer.on_touch_start(0, 0.5, 0.5)
        gesture = recognizer.on_touch_end(0)

        if gesture and gesture.gesture_type == TouchGestureType.TAP:
            overlay_manager.select_overlay(overlay.overlay_id)

        assert overlay.is_selected is True

        await session.stop()

    @pytest.mark.asyncio
    async def test_full_spatial_workflow(self):
        """Test complete spatial workflow."""
        # 1. Create collaborative session
        session = create_collaborative_spatial_session(
            name="Full Workflow",
            owner_id="owner",
            owner_name="Owner"
        )
        await session.start()

        # 2. Set up mobile UI
        ui_manager = create_mobile_ui_manager(
            device_type=DeviceType.TABLET,
            has_ar=True
        )
        components = create_default_spatial_ui(ui_manager)

        # 3. Create overlay manager
        overlay_manager = create_knowledge_overlay_manager()

        # 4. User joins
        avatar, presence, _ = await session.join("user_001", "Test User")

        # 5. Create knowledge overlays
        nodes = [
            {"id": "n1", "title": "Concept A", "tags": ["concept"]},
            {"id": "n2", "title": "Concept B", "tags": ["concept"]}
        ]
        rels = [{"source_id": "n1", "target_id": "n2", "relationship_type": "USES"}]

        overlays, edges = overlay_manager.create_overlays_from_knowledge(nodes, rels)

        # 6. Create shared objects linked to overlays
        for overlay in overlays:
            await session.create_shared_object(
                creator_id="user_001",
                name=overlay.title,
                object_type="knowledge_node",
                position=overlay.position,
                knowledge_node_id=overlay.node_id
            )

        # 7. Update user position
        await session.update_user_transform(
            user_id="user_001",
            position=Vector3(1, 0, 1),
            rotation=Quaternion.identity()
        )

        # 8. Update overlay visibility
        overlay_manager.update_visibility(Vector3(1, 0, 1))

        # 9. Export states
        session_state = session.to_state()
        overlay_state = overlay_manager.to_state()
        ui_state = ui_manager.to_state()

        # Verify all components working together
        assert session_state["user_count"] >= 1
        assert overlay_state["overlay_count"] == 2
        assert "components" in ui_state

        await session.stop()

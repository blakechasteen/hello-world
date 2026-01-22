"""
Jenny Specification Tests
==========================
Comprehensive tests for JennySpec dataclass and related components.

Date: January 2026
Author: Claude Code

Test Categories:
- Spec Creation (~100 lines): Basic instantiation, factory functions
- Validation (~100 lines): Required fields, constraints
- Enums (~80 lines): PanelTypeJenny, PanelSizeJenny, LayoutHint
- Lifecycle (~80 lines): State transitions, dissolution triggers
- Frozen Immutability (~60 lines): Cannot modify after creation
- Serialization (~80 lines): to_dict, from_dict, JSON round-trip
"""

import pytest
import json
from datetime import datetime
from uuid import UUID
from dataclasses import FrozenInstanceError

from HoloLoom.visualization.jenny_spec import (
    JennySpec,
    PanelTypeJenny,
    PanelSizeJenny,
    LayoutHint,
    create_action,
    create_spec,
    get_default_actions,
    ACTION_PIN,
    ACTION_DISMISS,
    ACTION_WHY,
    ACTION_EXPAND,
    ACTION_COLLAPSE,
    ACTION_COPY,
    ACTION_EXPORT,
    DEFAULT_ACTIONS,
)
from HoloLoom.visualization.jenny_enums import (
    LifecycleStage,
    BindingMode,
    DissolutionTrigger,
)


# ============================================================================
# Spec Creation Tests
# ============================================================================

class TestSpecCreation:
    """Tests for JennySpec creation and instantiation."""

    def test_default_spec_creation(self):
        """Test creating spec with all defaults."""
        spec = JennySpec()

        assert spec.panel_type == PanelTypeJenny.TEXT
        assert spec.lifecycle == LifecycleStage.NASCENT
        assert spec.binding_mode == BindingMode.STATIC
        assert spec.size == PanelSizeJenny.MEDIUM
        assert spec.priority == 0
        assert spec.title is None
        assert spec.content == {}
        assert spec.actions == ()

    def test_spec_with_all_fields(self):
        """Test creating spec with all fields specified."""
        actions = (ACTION_PIN, ACTION_DISMISS)
        spec = JennySpec(
            spec_id="test-spec-001",
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.GRAPH,
            title="Knowledge Graph",
            subtitle="Entity relationships",
            content={"nodes": [], "edges": []},
            lifecycle=LifecycleStage.STABLE,
            size=PanelSizeJenny.LARGE,
            priority=5,
            binding_mode=BindingMode.STATIC,
            actions=actions,
            metadata={"custom": "value"},
        )

        assert spec.spec_id == "test-spec-001"
        assert spec.spacetime_id == "st-001"
        assert spec.panel_type == PanelTypeJenny.GRAPH
        assert spec.title == "Knowledge Graph"
        assert spec.subtitle == "Entity relationships"
        assert spec.lifecycle == LifecycleStage.STABLE
        assert spec.size == PanelSizeJenny.LARGE
        assert spec.priority == 5
        assert len(spec.actions) == 2
        assert spec.metadata["custom"] == "value"

    def test_spec_id_auto_generated(self):
        """Test that spec_id is auto-generated as valid UUID."""
        spec = JennySpec()

        # Should be a valid UUID string
        assert spec.spec_id is not None
        UUID(spec.spec_id)  # Raises if not valid UUID

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        before = datetime.now()
        spec = JennySpec()
        after = datetime.now()

        assert before <= spec.timestamp <= after

    def test_create_spec_factory(self):
        """Test create_spec factory function."""
        spec = create_spec(
            spacetime_id="st-001",
            panel_type=PanelTypeJenny.TEXT,
            title="Test",
            content={"text": "Hello"},
            size=PanelSizeJenny.MEDIUM,
            priority=1,
        )

        assert spec.spacetime_id == "st-001"
        assert spec.panel_type == PanelTypeJenny.TEXT
        assert spec.title == "Test"
        assert spec.content == {"text": "Hello"}
        # Should have default actions for TEXT panel
        assert len(spec.actions) > 0

    def test_create_spec_with_custom_actions(self):
        """Test create_spec with custom actions overriding defaults."""
        custom_actions = (ACTION_DISMISS,)
        spec = create_spec(
            panel_type=PanelTypeJenny.TEXT,
            actions=custom_actions,
        )

        assert spec.actions == custom_actions
        assert len(spec.actions) == 1

    def test_create_spec_default_actions(self):
        """Test create_spec uses default actions when not specified."""
        spec = create_spec(panel_type=PanelTypeJenny.CODE)

        expected_actions = get_default_actions(PanelTypeJenny.CODE)
        assert spec.actions == expected_actions


# ============================================================================
# Validation Tests
# ============================================================================

class TestSpecValidation:
    """Tests for JennySpec validation logic."""

    def test_reactive_requires_data_source(self):
        """Test that REACTIVE binding mode requires data_source."""
        with pytest.raises(ValueError, match="REACTIVE binding mode requires data_source"):
            JennySpec(
                binding_mode=BindingMode.REACTIVE,
                data_source=None,  # Missing!
            )

    def test_reactive_with_data_source_succeeds(self):
        """Test that REACTIVE with data_source is valid."""
        spec = JennySpec(
            binding_mode=BindingMode.REACTIVE,
            data_source="metrics://cpu",
        )
        assert spec.binding_mode == BindingMode.REACTIVE
        assert spec.data_source == "metrics://cpu"

    def test_streaming_requires_data_source(self):
        """Test that STREAMING binding mode requires data_source."""
        with pytest.raises(ValueError, match="STREAMING binding mode requires data_source"):
            JennySpec(
                binding_mode=BindingMode.STREAMING,
                data_source=None,  # Missing!
            )

    def test_streaming_with_data_source_succeeds(self):
        """Test that STREAMING with data_source is valid."""
        spec = JennySpec(
            binding_mode=BindingMode.STREAMING,
            data_source="ws://localhost:8080/stream",
        )
        assert spec.binding_mode == BindingMode.STREAMING
        assert spec.data_source == "ws://localhost:8080/stream"

    def test_static_binding_no_data_source_ok(self):
        """Test that STATIC binding doesn't require data_source."""
        spec = JennySpec(
            binding_mode=BindingMode.STATIC,
            data_source=None,
        )
        assert spec.binding_mode == BindingMode.STATIC
        assert spec.data_source is None

    def test_static_binding_with_data_source_ok(self):
        """Test that STATIC can optionally have data_source."""
        spec = JennySpec(
            binding_mode=BindingMode.STATIC,
            data_source="optional://source",
        )
        assert spec.data_source == "optional://source"

    def test_refresh_interval_only_for_reactive(self):
        """Test refresh_interval_ms is typically used with REACTIVE."""
        spec = JennySpec(
            binding_mode=BindingMode.REACTIVE,
            data_source="metrics://test",
            refresh_interval_ms=5000,
        )
        assert spec.refresh_interval_ms == 5000


# ============================================================================
# Enum Tests
# ============================================================================

class TestEnums:
    """Tests for Jenny-related enumerations."""

    def test_panel_type_values(self):
        """Test all PanelTypeJenny values exist."""
        expected = {
            "graph", "confidence", "timeline", "memory_map", "text",
            "code", "table", "metric", "reasoning", "sources",
            "actions", "why", "comparison",
        }
        actual = {pt.value for pt in PanelTypeJenny}
        assert actual == expected

    def test_panel_type_string_enum(self):
        """Test PanelTypeJenny is a string enum."""
        assert isinstance(PanelTypeJenny.TEXT, str)
        assert PanelTypeJenny.TEXT == "text"
        # For str+Enum, the value equals the string
        assert PanelTypeJenny.TEXT.value == "text"

    def test_panel_size_values(self):
        """Test all PanelSizeJenny values exist."""
        expected = {"small", "medium", "large", "xlarge"}
        actual = {ps.value for ps in PanelSizeJenny}
        assert actual == expected

    def test_layout_hint_values(self):
        """Test all LayoutHint values exist."""
        expected = {"grid", "stack", "radial", "flow"}
        actual = {lh.value for lh in LayoutHint}
        assert actual == expected

    def test_lifecycle_stage_values(self):
        """Test all LifecycleStage values exist."""
        expected = {"nascent", "stable", "dissolving", "archived", "system"}
        actual = {ls.value for ls in LifecycleStage}
        assert actual == expected

    def test_binding_mode_values(self):
        """Test all BindingMode values exist."""
        expected = {"static", "reactive", "streaming"}
        actual = {bm.value for bm in BindingMode}
        assert actual == expected

    def test_dissolution_trigger_values(self):
        """Test all DissolutionTrigger values exist."""
        expected = {"manual", "timeout", "context", "superseded", "orphan", "memory"}
        actual = {dt.value for dt in DissolutionTrigger}
        assert actual == expected


# ============================================================================
# Lifecycle Tests
# ============================================================================

class TestLifecycle:
    """Tests for lifecycle state management."""

    def test_with_lifecycle_nascent_to_stable(self):
        """Test transitioning from NASCENT to STABLE."""
        spec = JennySpec(lifecycle=LifecycleStage.NASCENT)
        updated = spec.with_lifecycle(LifecycleStage.STABLE)

        assert spec.lifecycle == LifecycleStage.NASCENT  # Original unchanged
        assert updated.lifecycle == LifecycleStage.STABLE

    def test_with_lifecycle_to_dissolving_with_trigger(self):
        """Test transitioning to DISSOLVING with trigger."""
        spec = JennySpec(lifecycle=LifecycleStage.STABLE)
        updated = spec.with_lifecycle(
            LifecycleStage.DISSOLVING,
            trigger=DissolutionTrigger.MANUAL,
        )

        assert updated.lifecycle == LifecycleStage.DISSOLVING
        assert updated.dissolution_trigger == DissolutionTrigger.MANUAL

    def test_with_lifecycle_preserves_trigger_for_non_dissolving(self):
        """Test that trigger is not set for non-DISSOLVING states."""
        spec = JennySpec(
            lifecycle=LifecycleStage.DISSOLVING,
            dissolution_trigger=DissolutionTrigger.TIMEOUT,
        )
        updated = spec.with_lifecycle(LifecycleStage.STABLE)

        # Trigger should be preserved from original when not going to DISSOLVING
        assert updated.lifecycle == LifecycleStage.STABLE
        assert updated.dissolution_trigger == DissolutionTrigger.TIMEOUT

    def test_dissolution_trigger_types(self):
        """Test all dissolution trigger types."""
        triggers = [
            DissolutionTrigger.MANUAL,
            DissolutionTrigger.TIMEOUT,
            DissolutionTrigger.CONTEXT_SHIFT,
            DissolutionTrigger.SUPERSEDED,
            DissolutionTrigger.ORPHAN,
            DissolutionTrigger.MEMORY,
        ]

        for trigger in triggers:
            spec = JennySpec().with_lifecycle(LifecycleStage.DISSOLVING, trigger=trigger)
            assert spec.dissolution_trigger == trigger

    def test_system_lifecycle_for_meta_panels(self):
        """Test SYSTEM lifecycle stage for meta-panels."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.WHY,
            lifecycle=LifecycleStage.SYSTEM,
        )
        assert spec.lifecycle == LifecycleStage.SYSTEM


# ============================================================================
# Frozen Immutability Tests
# ============================================================================

class TestFrozenImmutability:
    """Tests for frozen dataclass immutability."""

    def test_cannot_modify_panel_type(self):
        """Test that panel_type cannot be modified."""
        spec = JennySpec(panel_type=PanelTypeJenny.TEXT)

        with pytest.raises(FrozenInstanceError):
            spec.panel_type = PanelTypeJenny.GRAPH

    def test_cannot_modify_title(self):
        """Test that title cannot be modified."""
        spec = JennySpec(title="Original")

        with pytest.raises(FrozenInstanceError):
            spec.title = "Modified"

    def test_cannot_modify_priority(self):
        """Test that priority cannot be modified."""
        spec = JennySpec(priority=0)

        with pytest.raises(FrozenInstanceError):
            spec.priority = 10

    def test_cannot_modify_lifecycle(self):
        """Test that lifecycle cannot be directly modified."""
        spec = JennySpec(lifecycle=LifecycleStage.NASCENT)

        with pytest.raises(FrozenInstanceError):
            spec.lifecycle = LifecycleStage.STABLE

    def test_evolve_creates_new_instance(self):
        """Test that evolve creates a new spec."""
        original = JennySpec(title="Original", priority=0)
        evolved = original.evolve(title="Updated", priority=5)

        assert original.title == "Original"
        assert original.priority == 0
        assert evolved.title == "Updated"
        assert evolved.priority == 5
        assert original is not evolved

    def test_evolve_preserves_unspecified_fields(self):
        """Test that evolve preserves fields not specified."""
        original = JennySpec(
            panel_type=PanelTypeJenny.GRAPH,
            title="Graph",
            priority=5,
            size=PanelSizeJenny.LARGE,
        )
        evolved = original.evolve(title="Updated Graph")

        assert evolved.panel_type == PanelTypeJenny.GRAPH
        assert evolved.priority == 5
        assert evolved.size == PanelSizeJenny.LARGE
        assert evolved.title == "Updated Graph"

    def test_with_position_creates_new_instance(self):
        """Test with_position creates new instance."""
        original = JennySpec(position=(0.0, 0.0, 0.0))
        moved = original.with_position(1.0, 2.0, 3.0)

        assert original.position == (0.0, 0.0, 0.0)
        assert moved.position == (1.0, 2.0, 3.0)

    def test_spec_not_hashable_with_mutable_fields(self):
        """Test that specs with mutable dict fields are not hashable.

        Note: JennySpec uses frozen=True but has mutable dict fields
        (content, metadata), which makes it unhashable. This is expected
        behavior - use spec_id for tracking/comparison instead.
        """
        spec = JennySpec()
        # Should raise TypeError because of mutable dict fields
        with pytest.raises(TypeError):
            hash(spec)

    def test_specs_compared_by_spec_id(self):
        """Test that specs can be compared by spec_id."""
        spec1 = JennySpec(title="A")
        spec2 = JennySpec(title="B")

        # Use spec_id for tracking (e.g., in a dict keyed by spec_id)
        spec_dict = {spec1.spec_id: spec1, spec2.spec_id: spec2}
        assert len(spec_dict) == 2


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Tests for JennySpec serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.TEXT,
            title="Test",
            content={"text": "Hello"},
        )
        data = spec.to_dict()

        assert data["panel_type"] == "text"
        assert data["title"] == "Test"
        assert data["content"] == {"text": "Hello"}

    def test_to_dict_enums_as_strings(self):
        """Test that enums are converted to string values."""
        spec = JennySpec(
            panel_type=PanelTypeJenny.GRAPH,
            lifecycle=LifecycleStage.STABLE,
            binding_mode=BindingMode.REACTIVE,
            data_source="test://source",
            size=PanelSizeJenny.LARGE,
        )
        data = spec.to_dict()

        assert data["panel_type"] == "graph"
        assert data["lifecycle"] == "stable"
        assert data["binding_mode"] == "reactive"
        assert data["size"] == "large"

    def test_to_dict_position_as_object(self):
        """Test that position is converted to xyz object."""
        spec = JennySpec(position=(1.0, 2.0, 3.0))
        data = spec.to_dict()

        assert data["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}

    def test_to_dict_actions_as_list(self):
        """Test that actions tuple is converted to list."""
        actions = (ACTION_PIN, ACTION_DISMISS)
        spec = JennySpec(actions=actions)
        data = spec.to_dict()

        assert isinstance(data["actions"], list)
        assert len(data["actions"]) == 2

    def test_from_dict_basic(self):
        """Test basic from_dict reconstruction."""
        data = {
            "spec_id": "test-001",
            "spacetime_id": "st-001",
            "panel_type": "text",
            "title": "Test",
            "content": {"text": "Hello"},
            "lifecycle": "nascent",
            "size": "medium",
            "binding_mode": "static",
            "priority": 0,
            "actions": [],
            "timestamp": "2025-01-01T12:00:00",
        }
        spec = JennySpec.from_dict(data)

        assert spec.spec_id == "test-001"
        assert spec.panel_type == PanelTypeJenny.TEXT
        assert spec.title == "Test"

    def test_from_dict_position_object(self):
        """Test from_dict handles position object."""
        data = {
            "spec_id": "test-001",
            "panel_type": "text",
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        }
        spec = JennySpec.from_dict(data)

        assert spec.position == (1.0, 2.0, 3.0)

    def test_to_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = JennySpec(
            panel_type=PanelTypeJenny.GRAPH,
            title="Test Graph",
            content={"nodes": [1, 2, 3]},
            size=PanelSizeJenny.LARGE,
            priority=5,
        )

        json_str = original.to_json()
        reconstructed = JennySpec.from_json(json_str)

        assert reconstructed.panel_type == original.panel_type
        assert reconstructed.title == original.title
        assert reconstructed.content == original.content
        assert reconstructed.size == original.size
        assert reconstructed.priority == original.priority

    def test_to_json_valid_json(self):
        """Test that to_json produces valid JSON."""
        spec = JennySpec(title="Test")
        json_str = spec.to_json()

        # Should parse without error
        parsed = json.loads(json_str)
        assert "title" in parsed

    def test_dissolution_trigger_serialization(self):
        """Test dissolution_trigger serialization."""
        spec = JennySpec(
            lifecycle=LifecycleStage.DISSOLVING,
            dissolution_trigger=DissolutionTrigger.MANUAL,
        )
        data = spec.to_dict()

        assert data["dissolution_trigger"] == "manual"

        reconstructed = JennySpec.from_dict(data)
        assert reconstructed.dissolution_trigger == DissolutionTrigger.MANUAL


# ============================================================================
# Action Creation Tests
# ============================================================================

class TestActionCreation:
    """Tests for action creation helpers."""

    def test_create_action_basic(self):
        """Test basic action creation."""
        action = create_action("Click Me", "click_handler")

        assert action["label"] == "Click Me"
        assert action["handler"] == "click_handler"
        assert action["type"] == "button"
        assert action["requires_confirmation"] is False
        assert action["params"] == {}
        assert "action_id" in action

    def test_create_action_with_params(self):
        """Test action with parameters."""
        action = create_action(
            "Export",
            "export_handler",
            params={"format": "json", "compress": True},
        )

        assert action["params"] == {"format": "json", "compress": True}

    def test_create_action_requires_confirmation(self):
        """Test action requiring confirmation."""
        action = create_action(
            "Delete",
            "delete_handler",
            requires_confirmation=True,
        )

        assert action["requires_confirmation"] is True

    def test_create_action_types(self):
        """Test different action types."""
        for action_type in ["button", "form", "slider", "toggle"]:
            action = create_action("Action", "handler", action_type=action_type)
            assert action["type"] == action_type

    def test_predefined_actions(self):
        """Test predefined action constants."""
        assert ACTION_PIN["label"] == "Pin"
        assert ACTION_DISMISS["label"] == "Dismiss"
        assert ACTION_WHY["label"] == "Why this UI?"
        assert ACTION_EXPAND["label"] == "Expand"
        assert ACTION_COLLAPSE["label"] == "Collapse"
        assert ACTION_COPY["label"] == "Copy"
        assert ACTION_EXPORT["requires_confirmation"] is True

    def test_get_default_actions(self):
        """Test getting default actions for panel types."""
        for panel_type in PanelTypeJenny:
            actions = get_default_actions(panel_type)
            assert isinstance(actions, tuple)
            assert len(actions) >= 1  # At least one action

    def test_default_actions_for_text(self):
        """Test default actions for TEXT panel."""
        actions = get_default_actions(PanelTypeJenny.TEXT)
        handler_names = [a["handler"] for a in actions]

        assert "pin_panel" in handler_names
        assert "dismiss_panel" in handler_names
        assert "copy_content" in handler_names

    def test_default_actions_for_why(self):
        """Test WHY panel has minimal actions."""
        actions = get_default_actions(PanelTypeJenny.WHY)
        handler_names = [a["handler"] for a in actions]

        # WHY panel should have minimal actions
        assert "dismiss_panel" in handler_names
        assert len(actions) <= 2

    def test_default_actions_dict_coverage(self):
        """Test DEFAULT_ACTIONS covers all panel types."""
        for panel_type in PanelTypeJenny:
            assert panel_type in DEFAULT_ACTIONS, f"Missing default actions for {panel_type}"

"""
Elle Domain Models Tests (Week 3 - Phase 1C)
============================================
Tests Elle domain model serialization, validation, and behavior.

Test Coverage (10 tests):
- SceneSnapshot serialization - 2 tests
- ElleRequest/Action round-trip - 2 tests
- Task and Followup models - 2 tests
- User and Intent models - 2 tests
- Validation and edge cases - 2 tests

Created: 2025-11-22 (Week 3 Phase 1)
"""

import pytest
from datetime import datetime
from elle.domain import (
    User, ElleRequest, ElleAction, ElleMode,
    SceneSnapshot, ObjectInScene, Task, Followup,
    UserIntent
)


# Test 1: SceneSnapshot serialization
def test_scene_snapshot_serializes_to_dict():
    """Test SceneSnapshot can serialize to dict."""
    scene = SceneSnapshot(
        objects=[
            ObjectInScene(object_id="obj1", label="table", position=(0, 0, 0)),
            ObjectInScene(object_id="obj2", label="chair", position=(1, 0, 0))
        ],
        user_position=(2, 0, 0),
        user_gaze_direction=(0, 0, 1)
    )
    
    data = scene.to_dict()
    
    assert data['objects'][0]['object_id'] == "obj1"
    assert data['user_position'] == (2, 0, 0)


# Test 2: SceneSnapshot deserialization
def test_scene_snapshot_deserializes_from_dict():
    """Test SceneSnapshot can deserialize from dict."""
    data = {
        'objects': [
            {'object_id': 'obj1', 'label': 'table', 'position': (0, 0, 0)}
        ],
        'user_position': (1, 0, 0),
        'user_gaze_direction': (0, 0, 1)
    }
    
    scene = SceneSnapshot.from_dict(data)
    
    assert len(scene.objects) == 1
    assert scene.objects[0].label == "table"


# Test 3: ElleRequest JSON round-trip
def test_elle_request_json_round_trip():
    """Test ElleRequest survives JSON serialization round-trip."""
    original = ElleRequest(
        request_id="test-123",
        user=User(user_id="user1", display_name="Test User"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    json_data = original.to_json()
    reconstructed = ElleRequest.from_json(json_data)
    
    assert reconstructed.request_id == original.request_id
    assert reconstructed.user.user_id == original.user.user_id


# Test 4: ElleAction JSON round-trip
def test_elle_action_json_round_trip():
    """Test ElleAction survives JSON serialization round-trip."""
    original = ElleAction(
        mode=ElleMode.SPEAK,
        utterance="Hello world",
        reasoning="Test reasoning",
        followups=[
            Followup(service="calendar", action="create", params={"title": "Test"})
        ]
    )
    
    json_data = original.to_json()
    reconstructed = ElleAction.from_json(json_data)
    
    assert reconstructed.mode == original.mode
    assert reconstructed.utterance == original.utterance
    assert len(reconstructed.followups) == 1


# Test 5: Task model with dependencies
def test_task_model_with_dependencies():
    """Test Task model can represent task dependencies."""
    task = Task(
        task_id="task1",
        description="Complete the report",
        dependencies=["task0"],
        status="pending"
    )
    
    assert task.task_id == "task1"
    assert "task0" in task.dependencies


# Test 6: Followup with nested params
def test_followup_with_nested_params():
    """Test Followup can handle nested parameter structures."""
    followup = Followup(
        service="reminder",
        action="set",
        params={
            "time": "1h",
            "recurrence": {
                "frequency": "daily",
                "until": "2025-12-31"
            }
        }
    )
    
    assert followup.params['recurrence']['frequency'] == "daily"


# Test 7: User intent validation
def test_user_intent_enum_values():
    """Test UserIntent enum has expected values."""
    assert UserIntent.SEEKING_GUIDANCE is not None
    assert UserIntent.REQUESTING_ACTION is not None
    assert UserIntent.ASKING_QUESTION is not None


# Test 8: Empty scene handling
def test_empty_scene_snapshot():
    """Test SceneSnapshot handles empty scenes gracefully."""
    scene = SceneSnapshot(
        objects=[],
        user_position=(0, 0, 0),
        user_gaze_direction=(0, 0, 1)
    )
    
    assert len(scene.objects) == 0
    assert scene.user_position == (0, 0, 0)


# Test 9: ElleMode enum completeness
def test_elle_mode_enum():
    """Test ElleMode enum has all expected modes."""
    modes = [ElleMode.SPEAK, ElleMode.VISUAL, ElleMode.SILENT, ElleMode.TASK]
    
    assert ElleMode.SPEAK.value == "speak"
    assert ElleMode.SILENT.value == "silent"


# Test 10: Metadata preservation
def test_metadata_preservation_in_action():
    """Test ElleAction preserves custom metadata."""
    action = ElleAction(
        mode=ElleMode.SPEAK,
        utterance="Test",
        metadata={
            "confidence": 0.95,
            "symbol": "plato",
            "custom_field": "custom_value"
        }
    )
    
    assert action.metadata['confidence'] == 0.95
    assert action.metadata['symbol'] == "plato"
    assert action.metadata['custom_field'] == "custom_value"

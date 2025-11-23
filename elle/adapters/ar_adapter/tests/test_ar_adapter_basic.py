"""
AR Adapter Basic Tests (Week 3 - Phase 1D)
==========================================
Tests AR adapter event conversion and rendering.

Test Coverage (5 tests):
- AREvent → ElleRequest conversion - 2 tests
- ElleAction → AR visualization - 2 tests
- Platform-specific rendering - 1 test

Created: 2025-11-22 (Week 3 Phase 1)
"""

import pytest
from datetime import datetime
from elle.domain import User, ElleMode, ElleAction
from elle.adapters.ar_adapter.ar_protocol import AREvent, AREventType


# Test 1: Slow scan event conversion
def test_ar_event_to_elle_request_slow_scan():
    """Test AREvent (slow scan) converts to ElleRequest correctly."""
    from elle.adapters.ar_adapter.ar_adapter import ARAdapter
    
    adapter = ARAdapter()
    
    ar_event = AREvent(
        event_id="evt-1",
        event_type=AREventType.SLOW_SCAN,
        user=User(user_id="user1"),
        timestamp=datetime.now(),
        scene_data={"objects": []}
    )
    
    elle_request = adapter.convert_to_request(ar_event)
    
    assert elle_request.user.user_id == "user1"
    assert elle_request.source_adapter == "ar"


# Test 2: Quick glance event conversion
def test_ar_event_to_elle_request_quick_glance():
    """Test AREvent (quick glance) converts to ElleRequest with intent."""
    from elle.adapters.ar_adapter.ar_adapter import ARAdapter
    from elle.domain import UserIntent
    
    adapter = ARAdapter()
    
    ar_event = AREvent(
        event_id="evt-2",
        event_type=AREventType.QUICK_GLANCE,
        user=User(user_id="user2"),
        timestamp=datetime.now()
    )
    
    elle_request = adapter.convert_to_request(ar_event)
    
    # Quick glance should indicate minimal intent
    assert elle_request.metadata.get('scan_type') == 'quick'


# Test 3: ElleAction to AR visualization (SPEAK mode)
def test_elle_action_to_ar_visual_speak():
    """Test ElleAction (SPEAK) converts to AR overlay correctly."""
    from elle.adapters.ar_adapter.ar_adapter import ARAdapter
    
    adapter = ARAdapter()
    
    action = ElleAction(
        mode=ElleMode.SPEAK,
        utterance="Hello, I see a shed",
        reasoning="User scanned shed"
    )
    
    ar_visual = adapter.convert_to_visualization(action)
    
    assert ar_visual.type == "text_overlay"
    assert "shed" in ar_visual.content.lower()


# Test 4: ElleAction to AR visualization (VISUAL mode)
def test_elle_action_to_ar_visual_arrows():
    """Test ElleAction (VISUAL) generates AR arrows/highlights."""
    from elle.adapters.ar_adapter.ar_adapter import ARAdapter
    
    adapter = ARAdapter()
    
    action = ElleAction(
        mode=ElleMode.VISUAL,
        utterance="",  # Visual mode uses graphics, not text
        metadata={"highlight_objects": ["tool1", "tool2"]}
    )
    
    ar_visual = adapter.convert_to_visualization(action)
    
    assert ar_visual.type == "highlight"
    assert len(ar_visual.target_objects) == 2


# Test 5: Platform-specific rendering (Apple Vision Pro)
def test_platform_specific_rendering_vision_pro():
    """Test AR adapter can target Apple Vision Pro format."""
    from elle.adapters.ar_adapter.ar_adapter import ARAdapter
    
    adapter = ARAdapter(target_platform="vision_pro")
    
    action = ElleAction(
        mode=ElleMode.SPEAK,
        utterance="Test message"
    )
    
    ar_visual = adapter.convert_to_visualization(action)
    
    # Platform-specific metadata should be set
    assert ar_visual.metadata.get('platform') == "vision_pro"

"""
Elle Engine Tests (Week 3 - Phase 1A)
=====================================
Tests ElleEngine orchestration logic.

Test Coverage (8 tests):
- Request routing and validation - 2 tests
- Memory loading and state management - 1 test
- Policy decision flow - 1 test
- Action dispatch pipeline - 1 test
- Error handling and graceful degradation - 2 tests
- Health check - 1 test

Created: 2025-11-21 (Week 3 Phase 1)
"""

import pytest
from datetime import datetime
from elle.engine.elle_engine import ElleEngine
from elle.domain import ElleRequest, ElleAction, ElleMode, User, Followup


# Test 1: Basic request handling
def test_engine_handles_basic_request(mock_policy, mock_memory_store, mock_tool_registry):
    engine = ElleEngine(mock_policy, mock_memory_store, mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-1",
        user=User(user_id="user1", display_name="Test User"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    action = engine.handle(request)
    
    assert action is not None
    assert action.mode == ElleMode.SPEAK
    assert action.utterance == "Test response"
    assert len(mock_policy.decide_calls) == 1


# Test 2: Memory loading
def test_engine_loads_memory_for_user(mock_policy, mock_memory_store, mock_tool_registry):
    engine = ElleEngine(mock_policy, mock_memory_store, mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-2",
        user=User(user_id="user2", display_name="User Two"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    engine.handle(request)
    
    # Memory should be loaded for user2
    assert "user2" in mock_memory_store.memories
    assert mock_policy.decide_calls[0][1].user_id == "user2"


# Test 3: Policy decision flow
def test_engine_calls_policy_with_request_and_memory(mock_policy, mock_memory_store, mock_tool_registry):
    engine = ElleEngine(mock_policy, mock_memory_store, mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-3",
        user=User(user_id="user3"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    engine.handle(request)
    
    # Policy should be called with request and memory
    assert len(mock_policy.decide_calls) == 1
    call_request, call_memory = mock_policy.decide_calls[0]
    assert call_request.request_id == "test-3"
    assert call_memory.user_id == "user3"


# Test 4: Follow-up dispatch
def test_engine_dispatches_followups(mock_memory_store, mock_tool_registry):
    from elle.tests.conftest import MockEllePolicy
    
    # Create action with followups
    action_with_followups = ElleAction(
        mode=ElleMode.SPEAK,
        utterance="Task created",
        followups=[
            Followup(service="calendar", action="create_event", params={"title": "Test"}),
            Followup(service="reminder", action="set", params={"time": "1h"})
        ]
    )
    
    policy = MockEllePolicy(return_action=action_with_followups)
    engine = ElleEngine(policy, mock_memory_store, mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-4",
        user=User(user_id="user4"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    engine.handle(request)
    
    # Should dispatch both followups
    assert len(mock_tool_registry.dispatched_followups) == 2
    assert mock_tool_registry.dispatched_followups[0].service == "calendar"
    assert mock_tool_registry.dispatched_followups[1].service == "reminder"


# Test 5: Interaction recording
def test_engine_records_interaction(mock_policy, mock_memory_store, mock_tool_registry):
    engine = ElleEngine(mock_policy, mock_memory_store, mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-5",
        user=User(user_id="user5"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    action = engine.handle(request)
    
    # Should record interaction
    assert len(mock_memory_store.recorded_interactions) == 1
    recorded_req, recorded_action = mock_memory_store.recorded_interactions[0]
    assert recorded_req.request_id == "test-5"
    assert recorded_action == action


# Test 6: Error handling with fallback
def test_engine_handles_policy_error(mock_memory_store, mock_tool_registry):
    # Create policy that raises error
    class ErrorPolicy:
        def decide(self, request, memory):
            raise ValueError("Policy error")
    
    engine = ElleEngine(ErrorPolicy(), mock_memory_store, mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-6",
        user=User(user_id="user6"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    action = engine.handle(request)
    
    # Should return safe fallback
    assert action.mode == ElleMode.SILENT
    assert "Error" in action.reasoning


# Test 7: Error handling with memory failure
def test_engine_handles_memory_error(mock_policy, mock_tool_registry):
    # Create memory store that raises error
    class ErrorMemory:
        def load_memory(self, user_id):
            raise RuntimeError("Memory error")
        def record_interaction(self, request, action):
            pass
    
    engine = ElleEngine(mock_policy, ErrorMemory(), mock_tool_registry)
    
    request = ElleRequest(
        request_id="test-7",
        user=User(user_id="user7"),
        source_adapter="test",
        timestamp=datetime.now()
    )
    
    action = engine.handle(request)
    
    # Should return safe fallback
    assert action.mode == ElleMode.SILENT


# Test 8: Health check
def test_engine_health_check(mock_policy, mock_memory_store, mock_tool_registry):
    engine = ElleEngine(mock_policy, mock_memory_store, mock_tool_registry)
    
    health = engine.health_check()
    
    assert health["engine"] == "ok"
    assert "timestamp" in health
    # Should be valid ISO format
    datetime.fromisoformat(health["timestamp"])

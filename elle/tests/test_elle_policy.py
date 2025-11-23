"""
Elle Policy Tests (Week 3 - Phase 1B)
======================================
Tests EllePolicy decision-making logic.

Test Coverage (8 tests):
- Prompt construction with scene context - 2 tests
- LLM integration and mocking - 1 test
- JSON response parsing - 1 test
- Action generation from LLM output - 1 test
- Confidence scoring - 1 test
- Error handling and fallback - 2 tests

Created: 2025-11-22 (Week 3 Phase 1)
"""

import pytest
from datetime import datetime
from elle.domain import ElleRequest, ElleAction, ElleMode, User, SceneSnapshot, ObjectInScene


# Test 1: Basic prompt construction
def test_policy_builds_prompt_with_scene():
    """Test policy constructs prompt with scene context."""
    from elle.policy.elle_policy import EllePolicy
    
    policy = EllePolicy(llm_client=None)  # Mock LLM
    
    request = ElleRequest(
        request_id="test-1",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now(),
        scene=SceneSnapshot(
            objects=[
                ObjectInScene(object_id="shed", label="shed", position=(0, 0, 0))
            ],
            user_position=(1, 0, 0),
            user_gaze_direction=(0, 0, 1)
        )
    )
    
    prompt = policy._build_prompt(request, memory_snapshot=None)
    
    assert "shed" in prompt
    assert "user_position" in prompt or "1" in prompt


# Test 2: Prompt includes memory context
def test_policy_includes_memory_in_prompt(mock_memory_store):
    """Test policy includes memory context in prompt."""
    from elle.policy.elle_policy import EllePolicy
    
    policy = EllePolicy(llm_client=None)
    memory = mock_memory_store.load_memory("user1")
    memory.interactions = [("past query", "past response")]
    
    request = ElleRequest(
        request_id="test-2",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    prompt = policy._build_prompt(request, memory)
    
    # Should include past interactions
    assert "past" in prompt.lower() or "interaction" in prompt.lower()


# Test 3: LLM response parsing
def test_policy_parses_llm_json_response():
    """Test policy parses valid JSON from LLM."""
    from elle.policy.elle_policy import EllePolicy
    
    # Mock LLM client that returns JSON
    class MockLLM:
        def generate(self, prompt):
            return '{"mode": "speak", "utterance": "Hello", "confidence": 0.9}'
    
    policy = EllePolicy(llm_client=MockLLM())
    
    request = ElleRequest(
        request_id="test-3",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    action = policy.decide(request, memory_snapshot=None)
    
    assert action.mode == ElleMode.SPEAK
    assert action.utterance == "Hello"


# Test 4: Action generation with followups
def test_policy_generates_action_with_followups():
    """Test policy can generate actions with follow-up tasks."""
    from elle.policy.elle_policy import EllePolicy
    from elle.domain import Followup
    
    class MockLLM:
        def generate(self, prompt):
            return '''{
                "mode": "speak",
                "utterance": "Task created",
                "followups": [
                    {"service": "calendar", "action": "create_event", "params": {"title": "Test"}}
                ]
            }'''
    
    policy = EllePolicy(llm_client=MockLLM())
    request = ElleRequest(
        request_id="test-4",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    action = policy.decide(request, memory_snapshot=None)
    
    assert len(action.followups) == 1
    assert action.followups[0].service == "calendar"


# Test 5: Confidence scoring
def test_policy_assigns_confidence_scores():
    """Test policy assigns reasonable confidence scores."""
    from elle.policy.elle_policy import EllePolicy
    
    class MockLLM:
        def generate(self, prompt):
            return '{"mode": "speak", "utterance": "Test", "confidence": 0.85}'
    
    policy = EllePolicy(llm_client=MockLLM())
    request = ElleRequest(
        request_id="test-5",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    action = policy.decide(request, memory_snapshot=None)
    
    # Confidence should be in valid range
    assert 0.0 <= action.metadata.get('confidence', 0.5) <= 1.0


# Test 6: Malformed JSON handling
def test_policy_handles_malformed_llm_response():
    """Test policy handles malformed JSON from LLM gracefully."""
    from elle.policy.elle_policy import EllePolicy
    
    class BadLLM:
        def generate(self, prompt):
            return "Not valid JSON at all!"
    
    policy = EllePolicy(llm_client=BadLLM())
    request = ElleRequest(
        request_id="test-6",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    action = policy.decide(request, memory_snapshot=None)
    
    # Should return safe fallback
    assert action.mode == ElleMode.SILENT


# Test 7: LLM failure fallback
def test_policy_fallback_when_llm_unavailable():
    """Test policy uses fallback when LLM is unavailable."""
    from elle.policy.elle_policy import EllePolicy
    
    class FailingLLM:
        def generate(self, prompt):
            raise ConnectionError("LLM service unavailable")
    
    policy = EllePolicy(llm_client=FailingLLM())
    request = ElleRequest(
        request_id="test-7",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    action = policy.decide(request, memory_snapshot=None)
    
    # Should return safe fallback
    assert action.mode == ElleMode.SILENT
    assert "Error" in action.reasoning or "unavailable" in action.reasoning


# Test 8: Symbol selection
def test_policy_can_select_mythic_symbols():
    """Test policy can select mythic symbols for guidance."""
    from elle.policy.elle_policy import EllePolicy
    
    class MockLLM:
        def generate(self, prompt):
            return '''{
                "mode": "speak",
                "utterance": "Focus on what matters",
                "symbol": "chimborazo"
            }'''
    
    policy = EllePolicy(llm_client=MockLLM())
    request = ElleRequest(
        request_id="test-8",
        user=User(user_id="user1"),
        source_adapter="ar",
        timestamp=datetime.now()
    )
    
    action = policy.decide(request, memory_snapshot=None)
    
    # Should have symbol metadata
    assert action.metadata.get('symbol') == "chimborazo"

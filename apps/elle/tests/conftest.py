"""Shared test fixtures for Elle tests."""

import pytest
from datetime import datetime
from typing import List

# Mock implementations for testing

class MockMemorySnapshot:
    """Mock memory snapshot for testing."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.interactions = []
        self.patterns = []

class MockMemoryStore:
    """Mock memory store for testing."""
    def __init__(self):
        self.memories = {}
        self.recorded_interactions = []

    def load_memory(self, user_id: str):
        if user_id not in self.memories:
            self.memories[user_id] = MockMemorySnapshot(user_id)
        return self.memories[user_id]

    def record_interaction(self, request, action):
        self.recorded_interactions.append((request, action))

class MockToolRegistry:
    """Mock tool registry for testing."""
    def __init__(self):
        self.dispatched_followups = []

    def dispatch(self, followup):
        self.dispatched_followups.append(followup)

class MockEllePolicy:
    """Mock Elle policy for testing."""
    def __init__(self, return_action=None):
        self.return_action = return_action
        self.decide_calls = []

    def decide(self, request, memory_snapshot):
        self.decide_calls.append((request, memory_snapshot))
        if self.return_action:
            return self.return_action

        # Default action
        from elle.domain import ElleAction, ElleMode
        return ElleAction(
            mode=ElleMode.SPEAK,
            utterance="Test response",
            reasoning="Mock policy decision"
        )

@pytest.fixture
def mock_memory_store():
    """Provide mock memory store."""
    return MockMemoryStore()

@pytest.fixture
def mock_tool_registry():
    """Provide mock tool registry."""
    return MockToolRegistry()

@pytest.fixture
def mock_policy():
    """Provide mock Elle policy."""
    return MockEllePolicy()

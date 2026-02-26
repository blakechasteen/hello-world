"""Unit tests for HoloLoom alignment ChatOps handlers.

Tests the complete ChatOps integration for alignment commands:
- Safety check command parsing and evaluation
- Safety statistics and history
- Audit log retrieval with filters
- Audit trace for provenance
- Audit search capabilities
- Help documentation
- Handler registration

Author: HoloLoom Team
Date: 2025-12-11
"""

import asyncio
import pytest
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from unittest.mock import Mock, AsyncMock, patch
import uuid


# ============================================================================
# Mock Enums and Types
# ============================================================================

class MockRiskLevel(Enum):
    """Mock RiskLevel enum."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MockActionCategory(Enum):
    """Mock ActionCategory enum."""
    QUERY = "query"
    CODE_EXECUTION = "code_execution"
    FILE_OPERATION = "file_operation"
    NETWORK = "network"
    MEMORY_MODIFICATION = "memory_modification"
    SYSTEM = "system"
    EXTERNAL_API = "external_api"


class MockDecisionType(Enum):
    """Mock DecisionType enum."""
    QUERY = "query"
    TOOL_SELECTION = "tool_selection"
    MEMORY_ACCESS = "memory_access"
    REFLECTION = "reflection"
    SAFETY_CHECK = "safety_check"


class MockOutcomeType(Enum):
    """Mock OutcomeType enum."""
    SUCCESS = "success"
    PARTIAL = "partial"
    REJECTED = "rejected"
    ERROR = "error"


# ============================================================================
# Mock Data Classes
# ============================================================================

@dataclass
class MockMatrixRoom:
    """Mock nio.MatrixRoom for testing."""
    room_id: str = "!test:example.com"
    name: str = "test-room"
    display_name: str = "Test Room"


@dataclass
class MockRoomMessageText:
    """Mock nio.RoomMessageText event."""
    sender: str = "@user:example.com"
    body: str = ""
    room_id: str = "!test:example.com"


@dataclass
class MockActionRequest:
    """Mock ActionRequest."""
    action: str
    category: MockActionCategory = MockActionCategory.QUERY
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MockSafetyDecision:
    """Mock SafetyDecision."""
    allowed: bool = True
    risk_level: MockRiskLevel = MockRiskLevel.LOW
    confidence: float = 0.9
    reason: str = "Test evaluation"
    mitigations: List[str] = field(default_factory=list)
    requires_approval: bool = False
    action: str = "test_action"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MockDecisionLog:
    """Mock DecisionLog."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: MockDecisionType = MockDecisionType.QUERY
    outcome: MockOutcomeType = MockOutcomeType.SUCCESS
    confidence: float = 0.85
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Mock Classes
# ============================================================================

class MockSafetyGuardrails:
    """Mock SafetyGuardrails for testing."""

    def __init__(self):
        self.decisions = []
        self._stats = {
            'total_decisions': 100,
            'by_risk_level': {
                'SAFE': 50,
                'LOW': 30,
                'MEDIUM': 15,
                'HIGH': 4,
                'CRITICAL': 1
            },
            'allowed_count': 95,
            'blocked_count': 5,
            'avg_confidence': 0.87,
            'by_category': {
                'QUERY': 60,
                'CODE_EXECUTION': 20,
                'FILE_OPERATION': 10,
                'NETWORK': 10
            },
            'recent_patterns': [
                'Increased CODE_EXECUTION requests',
                'High confidence on QUERY actions'
            ]
        }

    def evaluate(self, request: MockActionRequest) -> MockSafetyDecision:
        """Evaluate an action request."""
        # Simulate different risk levels based on action content
        action = request.action.lower()

        if 'rm -rf' in action or 'drop table' in action:
            decision = MockSafetyDecision(
                allowed=False,
                risk_level=MockRiskLevel.CRITICAL,
                confidence=0.99,
                reason="Destructive operation detected",
                mitigations=["Use safer alternatives", "Require explicit confirmation"],
                requires_approval=True,
                action=request.action
            )
        elif 'execute' in action or 'code' in action.lower():
            decision = MockSafetyDecision(
                allowed=True,
                risk_level=MockRiskLevel.MEDIUM,
                confidence=0.75,
                reason="Code execution requires monitoring",
                action=request.action
            )
        else:
            decision = MockSafetyDecision(
                allowed=True,
                risk_level=MockRiskLevel.LOW,
                confidence=0.9,
                reason="Standard operation",
                action=request.action
            )

        self.decisions.append(decision)
        return decision

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return self._stats

    def get_recent_decisions(self, limit: int = 10) -> List[MockSafetyDecision]:
        """Get recent safety decisions."""
        # Return some mock decisions
        return self.decisions[-limit:] if self.decisions else [
            MockSafetyDecision(action="query_test"),
            MockSafetyDecision(
                action="code_execution_test",
                risk_level=MockRiskLevel.MEDIUM
            ),
            MockSafetyDecision(
                action="critical_test",
                risk_level=MockRiskLevel.HIGH,
                allowed=False
            )
        ][-limit:]


class MockProvenanceTracer:
    """Mock ProvenanceTracer for testing."""

    def __init__(self, decision_id: str):
        self.root_node = decision_id
        self.graph = Mock()
        self.graph.nodes.return_value = ['node1', 'node2', 'node3']
        self.graph.edges.return_value = [('node1', 'node2'), ('node2', 'node3')]

    def get_reasoning_chain(self, root: str) -> List[str]:
        """Get reasoning chain."""
        return [
            "Query received: What is Thompson Sampling?",
            "Context retrieved: 5 relevant memories",
            "Tool selected: answer (confidence: 0.92)",
            "Response generated"
        ]

    def get_data_sources(self, root: str) -> List[str]:
        """Get data sources."""
        return [
            "memory://vector/thompson_sampling",
            "memory://graph/exploration_strategies",
            "cache://query_cache/similar_query"
        ]


class MockAuditTrail:
    """Mock AuditTrail for testing."""

    def __init__(self):
        self.logs: List[MockDecisionLog] = []
        self._populate_logs()

    def _populate_logs(self):
        """Populate with test logs."""
        now = datetime.now()

        # Add various log entries
        self.logs = [
            MockDecisionLog(
                decision_id="abc12345-1111-2222-3333-444455556666",
                decision_type=MockDecisionType.QUERY,
                outcome=MockOutcomeType.SUCCESS,
                confidence=0.92,
                timestamp=now - timedelta(hours=1),
                metadata={"tool": "answer", "query": "test query 1"}
            ),
            MockDecisionLog(
                decision_id="def67890-1111-2222-3333-444455556666",
                decision_type=MockDecisionType.TOOL_SELECTION,
                outcome=MockOutcomeType.SUCCESS,
                confidence=0.88,
                timestamp=now - timedelta(hours=2),
                metadata={"tool": "research", "steps": 3}
            ),
            MockDecisionLog(
                decision_id="ghi11111-1111-2222-3333-444455556666",
                decision_type=MockDecisionType.SAFETY_CHECK,
                outcome=MockOutcomeType.REJECTED,
                confidence=0.95,
                timestamp=now - timedelta(hours=3),
                metadata={"action": "blocked_action", "risk": "high"}
            ),
            MockDecisionLog(
                decision_id="jkl22222-1111-2222-3333-444455556666",
                decision_type=MockDecisionType.QUERY,
                outcome=MockOutcomeType.PARTIAL,
                confidence=0.65,
                timestamp=now - timedelta(days=2),
                metadata={"tool": "answer", "partial": True}
            ),
        ]

    def query_by_type(
        self, decision_type: MockDecisionType, limit: int = 10
    ) -> List[MockDecisionLog]:
        """Query logs by type."""
        return [
            log for log in self.logs
            if log.decision_type == decision_type
        ][:limit]

    def query_by_time_range(
        self, start: datetime, end: datetime, limit: int = 20
    ) -> List[MockDecisionLog]:
        """Query logs by time range."""
        return [
            log for log in self.logs
            if start <= log.timestamp <= end
        ][:limit]

    def query_by_outcome(
        self, outcome: MockOutcomeType, limit: int = 20
    ) -> List[MockDecisionLog]:
        """Query logs by outcome."""
        return [
            log for log in self.logs
            if log.outcome == outcome
        ][:limit]

    def query_by_metadata(
        self, metadata_filter: Dict[str, Any], limit: int = 20
    ) -> List[MockDecisionLog]:
        """Query logs by metadata."""
        results = []
        for log in self.logs:
            match = all(
                log.metadata.get(k) == v
                for k, v in metadata_filter.items()
            )
            if match:
                results.append(log)
        return results[:limit]

    def get_tracer(self, decision_id: str) -> MockProvenanceTracer:
        """Get tracer for a decision."""
        return MockProvenanceTracer(decision_id)


# ============================================================================
# Handler Test Fixtures
# ============================================================================

@pytest.fixture
def mock_room():
    """Create mock Matrix room."""
    return MockMatrixRoom()


@pytest.fixture
def mock_event():
    """Create mock message event."""
    return MockRoomMessageText()


@pytest.fixture
def mock_guardrails():
    """Create mock SafetyGuardrails."""
    return MockSafetyGuardrails()


@pytest.fixture
def mock_audit_trail():
    """Create mock AuditTrail."""
    return MockAuditTrail()


# ============================================================================
# Import handlers with mocks
# ============================================================================

# We need to import the actual handlers but patch their dependencies
@pytest.fixture
def alignment_handlers(mock_guardrails, mock_audit_trail):
    """Import and configure alignment handlers."""
    # Patch the module-level imports
    with patch.dict('sys.modules', {
        'HoloLoom.alignment.safety_guardrails': Mock(
            SafetyGuardrails=MockSafetyGuardrails,
            ActionRequest=MockActionRequest,
            ActionCategory=MockActionCategory,
            RiskLevel=MockRiskLevel,
            SafetyDecision=MockSafetyDecision
        ),
        'HoloLoom.alignment.audit_trail': Mock(
            AuditTrail=MockAuditTrail,
            DecisionLog=MockDecisionLog,
            DecisionType=MockDecisionType,
            OutcomeType=MockOutcomeType
        ),
    }):
        # Import after patching
        from HoloLoom.apps.chatops.handlers import alignment_handlers as ah

        # Set global instances
        ah.set_guardrails(mock_guardrails)
        ah.set_audit_trail(mock_audit_trail)

        # Patch module constants
        ah.GUARDRAILS_AVAILABLE = True
        ah.AUDIT_AVAILABLE = True
        ah.RiskLevel = MockRiskLevel
        ah.ActionCategory = MockActionCategory
        ah.DecisionType = MockDecisionType
        ah.OutcomeType = MockOutcomeType
        ah.ActionRequest = MockActionRequest
        ah.SafetyGuardrails = MockSafetyGuardrails

        yield ah


# ============================================================================
# TESTS: Safety Check Command
# ============================================================================

class TestSafetyCheckCommand:
    """Test !safety check command."""

    @pytest.mark.asyncio
    async def test_handle_safety_check_basic(
        self, alignment_handlers, mock_room, mock_event, mock_guardrails
    ):
        """Test basic safety check flow."""
        result = await alignment_handlers.handle_safety_check(
            room=mock_room,
            event=mock_event,
            args="query some data",
            guardrails=mock_guardrails
        )

        assert "Safety Check" in result
        assert "Risk Level" in result
        assert "Allowed" in result
        assert "Confidence" in result

    @pytest.mark.asyncio
    async def test_handle_safety_check_with_category(
        self, alignment_handlers, mock_room, mock_event, mock_guardrails
    ):
        """Test safety check with explicit category."""
        result = await alignment_handlers.handle_safety_check(
            room=mock_room,
            event=mock_event,
            args="CODE_EXECUTION:execute test script",
            guardrails=mock_guardrails
        )

        assert "Safety Check" in result
        assert "MEDIUM" in result or "Risk Level" in result

    @pytest.mark.asyncio
    async def test_handle_safety_check_empty_args(
        self, alignment_handlers, mock_room, mock_event
    ):
        """Test safety check returns usage when no args."""
        result = await alignment_handlers.handle_safety_check(
            room=mock_room,
            event=mock_event,
            args="",
            guardrails=None
        )

        assert "Usage" in result
        assert "!safety check" in result
        assert "Examples" in result

    @pytest.mark.asyncio
    async def test_handle_safety_check_critical_action(
        self, alignment_handlers, mock_room, mock_event, mock_guardrails
    ):
        """Test safety check with critical action."""
        result = await alignment_handlers.handle_safety_check(
            room=mock_room,
            event=mock_event,
            args="rm -rf /",
            guardrails=mock_guardrails
        )

        assert "CRITICAL" in result
        assert "No" in result  # Allowed: No
        assert "Mitigations" in result


class TestSafetyStatsCommand:
    """Test !safety stats command."""

    @pytest.mark.asyncio
    async def test_handle_safety_stats(
        self, alignment_handlers, mock_room, mock_event, mock_guardrails
    ):
        """Test safety statistics retrieval."""
        result = await alignment_handlers.handle_safety_stats(
            room=mock_room,
            event=mock_event,
            args="",
            guardrails=mock_guardrails
        )

        assert "Safety Statistics" in result
        assert "Total Decisions" in result
        assert "By Risk Level" in result
        assert "Allowed" in result
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_handle_safety_stats_no_guardrails(
        self, alignment_handlers, mock_room, mock_event
    ):
        """Test safety stats when guardrails not configured."""
        alignment_handlers.set_guardrails(None)
        alignment_handlers.GUARDRAILS_AVAILABLE = False

        result = await alignment_handlers.handle_safety_stats(
            room=mock_room,
            event=mock_event,
            args="",
            guardrails=None
        )

        assert "not available" in result.lower()


class TestSafetyHistoryCommand:
    """Test !safety history command."""

    @pytest.mark.asyncio
    async def test_handle_safety_history(
        self, alignment_handlers, mock_room, mock_event, mock_guardrails
    ):
        """Test safety decision history retrieval."""
        # Generate some decisions first
        mock_guardrails.evaluate(MockActionRequest(action="test query"))
        mock_guardrails.evaluate(MockActionRequest(action="code execution"))

        result = await alignment_handlers.handle_safety_history(
            room=mock_room,
            event=mock_event,
            args="",
            guardrails=mock_guardrails
        )

        assert "Safety Decisions" in result or "decisions" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_safety_history_with_limit(
        self, alignment_handlers, mock_room, mock_event, mock_guardrails
    ):
        """Test safety history with custom limit."""
        result = await alignment_handlers.handle_safety_history(
            room=mock_room,
            event=mock_event,
            args="5",
            guardrails=mock_guardrails
        )

        # Should respect the limit
        assert "Safety Decisions" in result or "No safety decisions" in result


# ============================================================================
# TESTS: Audit Log Command
# ============================================================================

class TestAuditLogCommand:
    """Test !audit log command."""

    @pytest.mark.asyncio
    async def test_handle_audit_log_basic(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test basic audit log retrieval."""
        result = await alignment_handlers.handle_audit_log(
            room=mock_room,
            event=mock_event,
            args="",
            audit_trail=mock_audit_trail
        )

        assert "Audit Log" in result
        assert "entries" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_audit_log_with_type_filter(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test audit log with type filter."""
        result = await alignment_handlers.handle_audit_log(
            room=mock_room,
            event=mock_event,
            args="QUERY 10",
            audit_trail=mock_audit_trail
        )

        assert "Audit Log" in result

    @pytest.mark.asyncio
    async def test_handle_audit_log_no_audit_trail(
        self, alignment_handlers, mock_room, mock_event
    ):
        """Test audit log when audit trail not configured."""
        alignment_handlers.set_audit_trail(None)
        alignment_handlers.AUDIT_AVAILABLE = False

        result = await alignment_handlers.handle_audit_log(
            room=mock_room,
            event=mock_event,
            args="",
            audit_trail=None
        )

        assert "not available" in result.lower()


# ============================================================================
# TESTS: Audit Trace Command
# ============================================================================

class TestAuditTraceCommand:
    """Test !audit trace command."""

    @pytest.mark.asyncio
    async def test_handle_audit_trace(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test provenance chain lookup."""
        # Use a full decision ID from mock logs
        decision_id = mock_audit_trail.logs[0].decision_id

        result = await alignment_handlers.handle_audit_trace(
            room=mock_room,
            event=mock_event,
            args=decision_id,
            audit_trail=mock_audit_trail
        )

        assert "Provenance Trace" in result
        assert "Reasoning Chain" in result or "chain" in result.lower()

    @pytest.mark.asyncio
    async def test_handle_audit_trace_partial_id(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test audit trace with partial ID matching."""
        # Use first 8 chars of decision ID
        partial_id = mock_audit_trail.logs[0].decision_id[:8]

        result = await alignment_handlers.handle_audit_trace(
            room=mock_room,
            event=mock_event,
            args=partial_id,
            audit_trail=mock_audit_trail
        )

        # Should either find the match or report multiple/no matches
        assert "Provenance" in result or "matches" in result.lower() or "No decision" in result

    @pytest.mark.asyncio
    async def test_handle_audit_trace_empty_args(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test audit trace returns usage when no args."""
        result = await alignment_handlers.handle_audit_trace(
            room=mock_room,
            event=mock_event,
            args="",
            audit_trail=mock_audit_trail
        )

        assert "Usage" in result
        assert "!audit trace" in result


# ============================================================================
# TESTS: Audit Search Command
# ============================================================================

class TestAuditSearchCommand:
    """Test !audit search command."""

    @pytest.mark.asyncio
    async def test_handle_audit_search_by_time(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test time-based audit search."""
        result = await alignment_handlers.handle_audit_search(
            room=mock_room,
            event=mock_event,
            args="last_24h",
            audit_trail=mock_audit_trail
        )

        # Should find recent entries or report no matches
        assert "Search Results" in result or "No audit entries" in result

    @pytest.mark.asyncio
    async def test_handle_audit_search_by_outcome(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test outcome-based audit search."""
        result = await alignment_handlers.handle_audit_search(
            room=mock_room,
            event=mock_event,
            args="outcome=SUCCESS",
            audit_trail=mock_audit_trail
        )

        assert "Search Results" in result or "No audit entries" in result

    @pytest.mark.asyncio
    async def test_handle_audit_search_by_metadata(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test metadata-based audit search."""
        result = await alignment_handlers.handle_audit_search(
            room=mock_room,
            event=mock_event,
            args="tool=answer",
            audit_trail=mock_audit_trail
        )

        assert "Search Results" in result or "No audit entries" in result

    @pytest.mark.asyncio
    async def test_handle_audit_search_empty_args(
        self, alignment_handlers, mock_room, mock_event, mock_audit_trail
    ):
        """Test audit search returns usage when no args."""
        result = await alignment_handlers.handle_audit_search(
            room=mock_room,
            event=mock_event,
            args="",
            audit_trail=mock_audit_trail
        )

        assert "Usage" in result
        assert "!audit search" in result
        assert "Examples" in result


# ============================================================================
# TESTS: Help Command
# ============================================================================

class TestAlignmentHelpCommand:
    """Test !alignment help command."""

    @pytest.mark.asyncio
    async def test_handle_alignment_help(
        self, alignment_handlers, mock_room, mock_event
    ):
        """Test alignment help output."""
        result = await alignment_handlers.handle_alignment_help(
            room=mock_room,
            event=mock_event,
            args=""
        )

        assert "Alignment Framework Commands" in result
        assert "Safety Commands" in result
        assert "Audit Commands" in result
        assert "!safety check" in result
        assert "!audit log" in result
        assert "!audit trace" in result
        assert "Risk Levels" in result


# ============================================================================
# TESTS: Handler Registration
# ============================================================================

class TestHandlerRegistration:
    """Test handler registration."""

    def test_register_alignment_handlers(
        self, alignment_handlers, mock_guardrails, mock_audit_trail
    ):
        """Test handler registration function."""
        handlers = alignment_handlers.register_alignment_handlers(
            registry=None,  # No registry, just create handlers
            guardrails=mock_guardrails,
            audit_trail=mock_audit_trail
        )

        assert handlers is not None
        assert handlers.guardrails == mock_guardrails
        assert handlers.audit_trail == mock_audit_trail

    def test_alignment_handlers_class_methods(
        self, alignment_handlers, mock_guardrails, mock_audit_trail
    ):
        """Test AlignmentHandlers class has all methods."""
        handlers = alignment_handlers.AlignmentHandlers(
            guardrails=mock_guardrails,
            audit_trail=mock_audit_trail
        )

        # Verify all methods exist
        assert hasattr(handlers, 'safety_check')
        assert hasattr(handlers, 'safety_stats')
        assert hasattr(handlers, 'safety_history')
        assert hasattr(handlers, 'audit_log')
        assert hasattr(handlers, 'audit_trace')
        assert hasattr(handlers, 'audit_search')
        assert hasattr(handlers, 'alignment_help')

    def test_global_instance_setters(self, alignment_handlers, mock_guardrails, mock_audit_trail):
        """Test global instance getters and setters."""
        # Test guardrails
        alignment_handlers.set_guardrails(mock_guardrails)
        assert alignment_handlers.get_guardrails() == mock_guardrails

        # Test audit trail
        alignment_handlers.set_audit_trail(mock_audit_trail)
        assert alignment_handlers.get_audit_trail() == mock_audit_trail

        # Test clearing
        alignment_handlers.set_guardrails(None)
        assert alignment_handlers.get_guardrails() is None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

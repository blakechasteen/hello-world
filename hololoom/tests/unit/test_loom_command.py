"""
Test Loom Command - Pattern Card Selector
==========================================

Tests for HoloLoom/loom/command.py - Pattern card selection and configuration.

Coverage:
- Pattern selection (BARE/FAST/FUSED/SEMANTIC_FLOW)
- User preference override
- Resource constraint-based selection
- Auto-selection based on query characteristics
- PatternSpec defaults and validation
- Safety guardrails integration
- Selection statistics
"""

import pytest
from unittest.mock import Mock, patch

from HoloLoom.loom.command import (
    LoomCommand,
    PatternCard,
    PatternSpec,
    BARE_PATTERN,
    FAST_PATTERN,
    FUSED_PATTERN,
    SEMANTIC_FLOW_PATTERN,
    create_loom_command
)
from HoloLoom.alignment.safety_guardrails import (
    SafetyGuardrails,
    ActionRequest,
    SafetyDecision,
    ActionCategory,
    RiskLevel
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def loom():
    """Create basic loom command with auto-select enabled."""
    return LoomCommand(
        default_pattern=PatternCard.FAST,
        auto_select=True,
        guardrails=None  # Disable for basic tests
    )


@pytest.fixture
def loom_no_auto():
    """Create loom command with auto-select disabled."""
    return LoomCommand(
        default_pattern=PatternCard.FAST,
        auto_select=False,
        guardrails=None
    )


@pytest.fixture
def mock_guardrails():
    """Create mock safety guardrails that allow all actions."""
    guardrails = Mock(spec=SafetyGuardrails)
    guardrails.evaluate.return_value = SafetyDecision(
        allowed=True,
        risk_level=RiskLevel.LOW,
        reason="Test: allowed",
        requires_approval=False
    )
    return guardrails


# ============================================================================
# Test Initialization
# ============================================================================

def test_loom_initialization():
    """Test loom command initializes correctly."""
    loom = LoomCommand(
        default_pattern=PatternCard.FAST,
        auto_select=True
    )

    assert loom.default_pattern == PatternCard.FAST
    assert loom.auto_select is True
    assert loom.current_pattern is None
    assert len(loom.selection_history) == 0
    assert len(loom.patterns) == 4  # BARE, FAST, FUSED, SEMANTIC_FLOW


def test_loom_has_all_pattern_cards():
    """Test loom has all pattern cards in registry."""
    loom = LoomCommand(guardrails=None)

    assert PatternCard.BARE in loom.patterns
    assert PatternCard.FAST in loom.patterns
    assert PatternCard.FUSED in loom.patterns
    assert PatternCard.SEMANTIC_FLOW in loom.patterns


# ============================================================================
# Test Pattern Specs
# ============================================================================

def test_pattern_spec_bare():
    """Test BARE pattern spec has correct defaults."""
    spec = BARE_PATTERN

    assert spec.name == "Bare Threading"
    assert spec.card == PatternCard.BARE
    assert spec.scales == [768]
    assert spec.quality_target == 0.6
    assert spec.speed_priority == 0.9
    assert spec.pipeline_timeout == 2.0
    assert spec.retrieval_k == 3
    assert spec.enable_spectral is False


def test_pattern_spec_fast():
    """Test FAST pattern spec has correct defaults."""
    spec = FAST_PATTERN

    assert spec.name == "Fast Threading"
    assert spec.card == PatternCard.FAST
    assert spec.quality_target == 0.75
    assert spec.speed_priority == 0.5
    assert spec.pipeline_timeout == 4.0
    assert spec.retrieval_k == 6
    assert spec.enable_spectral is True
    assert spec.spectral_k_eigen == 2


def test_pattern_spec_fused():
    """Test FUSED pattern spec has correct defaults."""
    spec = FUSED_PATTERN

    assert spec.name == "Fused Threading"
    assert spec.card == PatternCard.FUSED
    assert spec.quality_target == 0.9
    assert spec.speed_priority == 0.2
    assert spec.pipeline_timeout == 8.0
    assert spec.retrieval_k == 10
    assert spec.enable_spectral is True
    assert spec.spectral_k_eigen == 4


def test_pattern_spec_semantic_flow():
    """Test SEMANTIC_FLOW pattern spec has correct defaults."""
    spec = SEMANTIC_FLOW_PATTERN

    assert spec.name == "Semantic Flow Threading"
    assert spec.card == PatternCard.SEMANTIC_FLOW
    assert spec.enable_semantic_flow is True
    assert spec.semantic_dimensions == 16
    assert spec.semantic_trajectory is True
    assert spec.semantic_ethics is True


def test_pattern_spec_stage_timeouts_auto_generated():
    """Test PatternSpec auto-generates stage timeouts from pipeline timeout."""
    spec = PatternSpec(
        name="Test",
        card=PatternCard.FAST,
        scales=[768],
        fusion_weights={768: 1.0},
        pipeline_timeout=10.0
    )

    # Stage timeouts should sum to ~pipeline_timeout
    assert spec.stage_timeouts is not None
    assert 'features' in spec.stage_timeouts
    assert 'retrieval' in spec.stage_timeouts
    assert 'decision' in spec.stage_timeouts
    assert 'execution' in spec.stage_timeouts

    # Check proportions (30%, 40%, 20%, 10%)
    assert spec.stage_timeouts['features'] == 3.0  # 30% of 10.0
    assert spec.stage_timeouts['retrieval'] == 4.0  # 40%
    assert spec.stage_timeouts['decision'] == 2.0   # 20%
    assert spec.stage_timeouts['execution'] == 1.0  # 10%


# ============================================================================
# Test Pattern Selection - User Preference
# ============================================================================

def test_select_pattern_user_preference_bare(loom):
    """Test user preference for BARE pattern."""
    pattern = loom.select_pattern(
        query_text="simple query",
        user_preference="bare"
    )

    assert pattern.card == PatternCard.BARE
    assert pattern.name == "Bare Threading"
    assert len(loom.selection_history) == 1
    assert loom.selection_history[0]["reason"] == "user_preference=bare"


def test_select_pattern_user_preference_fast(loom):
    """Test user preference for FAST pattern."""
    pattern = loom.select_pattern(
        query_text="test",
        user_preference="fast"
    )

    assert pattern.card == PatternCard.FAST
    assert pattern.name == "Fast Threading"


def test_select_pattern_user_preference_fused(loom):
    """Test user preference for FUSED pattern."""
    pattern = loom.select_pattern(
        query_text="test",
        user_preference="fused"
    )

    assert pattern.card == PatternCard.FUSED
    assert pattern.name == "Fused Threading"


def test_select_pattern_user_preference_semantic_flow(loom):
    """Test user preference for SEMANTIC_FLOW pattern."""
    pattern = loom.select_pattern(
        query_text="test",
        user_preference="semantic_flow"
    )

    assert pattern.card == PatternCard.SEMANTIC_FLOW
    assert pattern.name == "Semantic Flow Threading"


def test_select_pattern_invalid_user_preference_falls_back(loom):
    """Test invalid user preference falls back to auto-selection."""
    pattern = loom.select_pattern(
        query_text="short",
        user_preference="invalid_pattern"
    )

    # Should fall back to auto-selection (short query → BARE)
    assert pattern.card == PatternCard.BARE


# ============================================================================
# Test Pattern Selection - Resource Constraints
# ============================================================================

def test_select_pattern_by_timeout_constraint_bare(loom):
    """Test resource constraint selects BARE for tight timeout."""
    pattern = loom.select_pattern(
        query_text="test",
        resource_constraints={"max_timeout": 2.0}
    )

    assert pattern.card == PatternCard.BARE
    assert loom.selection_history[0]["reason"].startswith("resource_constraints")


def test_select_pattern_by_timeout_constraint_fast(loom):
    """Test resource constraint selects FAST for medium timeout."""
    pattern = loom.select_pattern(
        query_text="test",
        resource_constraints={"max_timeout": 4.0}
    )

    assert pattern.card == PatternCard.FAST


def test_select_pattern_by_timeout_constraint_fused(loom):
    """Test resource constraint selects FUSED for generous timeout."""
    pattern = loom.select_pattern(
        query_text="test",
        resource_constraints={"max_timeout": 10.0}
    )

    assert pattern.card == PatternCard.FUSED


# ============================================================================
# Test Pattern Selection - Auto-Selection
# ============================================================================

def test_auto_select_short_query_bare(loom):
    """Test auto-select picks BARE for short queries."""
    short_query = "hello"

    pattern = loom.select_pattern(query_text=short_query)

    assert pattern.card == PatternCard.BARE
    assert len(short_query) < 50


def test_auto_select_medium_query_fast(loom):
    """Test auto-select picks FAST for medium queries."""
    medium_query = "This is a medium-length query that should trigger fast mode processing"

    pattern = loom.select_pattern(query_text=medium_query)

    assert pattern.card == PatternCard.FAST
    assert 50 <= len(medium_query) < 150


def test_auto_select_long_query_fast(loom):
    """Test auto-select picks FAST for long queries (could be FUSED if compute available)."""
    long_query = "This is a very long query that contains a lot of text and should potentially trigger fused mode processing with more comprehensive analysis but currently defaults to fast mode"

    pattern = loom.select_pattern(query_text=long_query)

    assert pattern.card == PatternCard.FAST
    assert len(long_query) >= 150


def test_auto_select_disabled_uses_default(loom_no_auto):
    """Test auto-select disabled falls back to default pattern."""
    pattern = loom_no_auto.select_pattern(query_text="any query")

    # Should use default (FAST) regardless of query
    assert pattern.card == PatternCard.FAST
    assert loom_no_auto.selection_history[0]["reason"] == "default_pattern"


# ============================================================================
# Test Pattern Selection - Priority Order
# ============================================================================

def test_selection_priority_user_preference_overrides_constraints(loom):
    """Test user preference overrides resource constraints."""
    pattern = loom.select_pattern(
        query_text="test",
        user_preference="fused",
        resource_constraints={"max_timeout": 1.0}  # Would normally select BARE
    )

    # User preference should win
    assert pattern.card == PatternCard.FUSED


def test_selection_priority_constraints_override_auto(loom):
    """Test resource constraints override auto-selection."""
    pattern = loom.select_pattern(
        query_text="hello",  # Short query would auto-select BARE
        resource_constraints={"max_timeout": 4.0}  # But constraint selects FAST (<=5.0)
    )

    # Constraint should win
    assert pattern.card == PatternCard.FAST


def test_selection_priority_auto_overrides_default(loom):
    """Test auto-selection overrides default pattern."""
    pattern = loom.select_pattern(query_text="hi")  # Very short

    # Auto-selection should pick BARE (not default FAST)
    assert pattern.card == PatternCard.BARE


# ============================================================================
# Test Current Pattern Tracking
# ============================================================================

def test_get_current_pattern_before_selection(loom):
    """Test current pattern is None before first selection."""
    assert loom.get_current_pattern() is None


def test_get_current_pattern_after_selection(loom):
    """Test current pattern is set after selection."""
    loom.select_pattern(query_text="test")

    current = loom.get_current_pattern()
    assert current is not None
    assert current.card == PatternCard.BARE  # Short query


def test_set_default_changes_default_pattern(loom):
    """Test set_default changes default pattern."""
    loom.set_default(PatternCard.FUSED)

    assert loom.default_pattern == PatternCard.FUSED


# ============================================================================
# Test Selection History
# ============================================================================

def test_selection_history_records_selections(loom):
    """Test selection history records all selections."""
    loom.select_pattern(query_text="query 1")
    loom.select_pattern(query_text="query 2")
    loom.select_pattern(query_text="query 3")

    assert len(loom.selection_history) == 3


def test_selection_history_includes_metadata(loom):
    """Test selection history includes metadata."""
    loom.select_pattern(
        query_text="test query",
        user_preference="bare"
    )

    entry = loom.selection_history[0]
    assert entry["pattern"] == "bare"
    assert entry["reason"] == "user_preference=bare"
    assert entry["query_length"] == len("test query")


# ============================================================================
# Test Statistics
# ============================================================================

def test_statistics_before_any_selection(loom):
    """Test statistics when no selections have been made."""
    stats = loom.get_statistics()

    assert stats["total_selections"] == 0


def test_statistics_counts_by_pattern(loom):
    """Test statistics count selections by pattern."""
    # Make multiple selections
    loom.select_pattern(query_text="a")      # BARE
    loom.select_pattern(query_text="b")      # BARE
    loom.select_pattern(query_text="medium length query for fast mode", user_preference="fast")  # FAST

    stats = loom.get_statistics()

    assert stats["total_selections"] == 3
    assert stats["pattern_distribution"]["bare"] == 2
    assert stats["pattern_distribution"]["fast"] == 1


def test_statistics_average_query_length(loom):
    """Test statistics compute average query length by pattern."""
    loom.select_pattern(query_text="short", user_preference="bare")  # 5 chars
    loom.select_pattern(query_text="also short", user_preference="bare")  # 10 chars

    stats = loom.get_statistics()

    # Average for BARE should be (5 + 10) / 2 = 7.5
    assert stats["avg_query_length_by_pattern"]["bare"] == 7.5


def test_statistics_current_pattern(loom):
    """Test statistics include current pattern."""
    loom.select_pattern(query_text="test", user_preference="fused")

    stats = loom.get_statistics()

    assert stats["current_pattern"] == "Fused Threading"


# ============================================================================
# Test Safety Guardrails Integration
# ============================================================================

def test_guardrails_called_during_selection(mock_guardrails):
    """Test safety guardrails are called during pattern selection."""
    loom = LoomCommand(guardrails=mock_guardrails)

    loom.select_pattern(query_text="test query")

    # Guardrails should have been called
    mock_guardrails.evaluate.assert_called_once()

    # Check call arguments
    call_args = mock_guardrails.evaluate.call_args
    action_request = call_args[0][0]
    assert action_request.action == "select_pattern"
    assert action_request.category == ActionCategory.ANALYSIS


def test_guardrails_blocked_action_raises_error():
    """Test blocked action by guardrails raises PermissionError."""
    # Create guardrails that block action
    blocked_guardrails = Mock(spec=SafetyGuardrails)
    blocked_guardrails.evaluate.return_value = SafetyDecision(
        allowed=False,
        risk_level=RiskLevel.HIGH,
        reason="Test: blocked",
        requires_approval=False
    )

    loom = LoomCommand(guardrails=blocked_guardrails)

    with pytest.raises(PermissionError, match="Test: blocked"):
        loom.select_pattern(query_text="malicious query")


def test_guardrails_requires_approval_raises_error():
    """Test action requiring approval raises PermissionError."""
    # Create guardrails that require approval
    approval_guardrails = Mock(spec=SafetyGuardrails)
    approval_guardrails.evaluate.return_value = SafetyDecision(
        allowed=True,
        risk_level=RiskLevel.MEDIUM,
        reason="Test: needs approval",
        requires_approval=True
    )

    loom = LoomCommand(guardrails=approval_guardrails)

    with pytest.raises(PermissionError, match="requires approval"):
        loom.select_pattern(query_text="sensitive query")


def test_guardrails_decision_recorded_in_history(mock_guardrails):
    """Test guardrails decision is recorded in selection history."""
    loom = LoomCommand(guardrails=mock_guardrails)

    loom.select_pattern(query_text="test")

    # Check history includes safety decision
    entry = loom.selection_history[0]
    assert entry["safety"] is not None
    assert entry["safety"]["allowed"] is True


# ============================================================================
# Test Factory Function
# ============================================================================

def test_create_loom_command_default():
    """Test factory function with defaults."""
    loom = create_loom_command()

    assert loom.default_pattern == PatternCard.FAST
    assert loom.auto_select is True


def test_create_loom_command_custom():
    """Test factory function with custom parameters."""
    loom = create_loom_command(
        default="bare",
        auto_select=False
    )

    assert loom.default_pattern == PatternCard.BARE
    assert loom.auto_select is False


def test_create_loom_command_with_guardrails():
    """Test factory function with custom guardrails."""
    mock_rails = Mock(spec=SafetyGuardrails)
    loom = create_loom_command(guardrails=mock_rails)

    assert loom.guardrails is mock_rails


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_select_pattern_empty_query_string(loom):
    """Test pattern selection with empty query string."""
    pattern = loom.select_pattern(query_text="")

    # Empty string is falsy, so auto-select won't trigger (uses default)
    assert pattern.card == PatternCard.FAST  # Default pattern


def test_select_pattern_none_query_string(loom):
    """Test pattern selection with None query string."""
    pattern = loom.select_pattern(query_text=None)

    # Should use default (no auto-select)
    assert pattern.card == PatternCard.FAST  # Default


def test_select_pattern_no_arguments_uses_default(loom):
    """Test pattern selection with no arguments uses default."""
    pattern = loom.select_pattern()

    assert pattern.card == PatternCard.FAST  # Default


def test_pattern_spec_custom_stage_timeouts():
    """Test PatternSpec respects custom stage timeouts."""
    custom_timeouts = {
        'features': 1.0,
        'retrieval': 1.5,
        'decision': 0.5,
        'execution': 1.0
    }

    spec = PatternSpec(
        name="Custom",
        card=PatternCard.FAST,
        scales=[768],
        fusion_weights={768: 1.0},
        pipeline_timeout=5.0,
        stage_timeouts=custom_timeouts
    )

    # Should use custom timeouts
    assert spec.stage_timeouts == custom_timeouts


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
- Initialization: 2 tests
- Pattern Specs: 5 tests
- User Preference Selection: 5 tests
- Resource Constraint Selection: 3 tests
- Auto-Selection: 4 tests
- Selection Priority: 3 tests
- Current Pattern Tracking: 3 tests
- Selection History: 2 tests
- Statistics: 4 tests
- Safety Guardrails: 4 tests
- Factory Function: 3 tests
- Edge Cases: 5 tests

Total: 43 tests covering critical loom command functionality
"""
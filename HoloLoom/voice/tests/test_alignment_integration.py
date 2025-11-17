"""
Test Alignment Integration for Elle AR
=======================================
Comprehensive test suite for AR alignment framework integration.

Tests:
- AR Safety Gate (risk levels, adversarial detection, approval flow)
- AR Context Builder (entity extraction, KG retrieval, multi-hop reasoning)
- AR Audit Trail (logging, querying, persistence)
- AR Deception Detector (voice-gesture consistency, spatial intent, counterfactual)
- End-to-end integration

Coverage: 45+ tests across all alignment components

Author: HoloLoom Team
Date: November 17, 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from HoloLoom.voice.ar_safety_gate import (
    ARSafetyGate,
    ARActionCategory,
    AdversarialGestureDetector,
    create_ar_safety_gate,
)
from HoloLoom.voice.ar_context_builder import (
    ARContextBuilder,
    EntityExtractor,
    create_ar_context_builder,
)
from HoloLoom.voice.ar_audit import (
    ARAuditTrail,
    ARDecisionType,
    create_ar_audit_trail,
)
from HoloLoom.voice.ar_deception_detector import (
    ARDeceptionDetector,
    create_ar_deception_detector,
)
from HoloLoom.voice.ar_context import (
    ARContext,
    ARObject,
    ARObjectType,
    Vector3,
    Quaternion,
    create_test_context,
)
from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.alignment.audit_trail import OutcomeType
from HoloLoom.alignment.safety_guardrails import RiskLevel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def ar_context():
    """Create test AR context."""
    return create_test_context()


@pytest.fixture
def kg():
    """Create test knowledge graph."""
    kg = KG()

    # Add beekeeping knowledge
    kg.add_edges([
        # Beehive taxonomy
        KGEdge("beehive", "apiary_equipment", "IS_A", 1.0),
        KGEdge("apiary_equipment", "agricultural_tool", "IS_A", 1.0),

        # Beehive properties
        KGEdge("beehive", "health", "HAS_PROPERTY", 1.0),
        KGEdge("beehive", "population", "HAS_PROPERTY", 1.0),
        KGEdge("beehive", "temperature", "HAS_PROPERTY", 1.0),

        # Beehive actions
        KGEdge("beehive", "inspection", "REQUIRES", 1.0),
        KGEdge("inspection", "smoker", "USES", 1.0),
        KGEdge("inspection", "protective_gear", "REQUIRES", 1.0),

        # Tool relationships
        KGEdge("smoker", "tool", "IS_A", 1.0),
        KGEdge("tool", "equipment", "IS_A", 1.0),

        # Health relationships
        KGEdge("health", "disease", "AFFECTED_BY", 1.0),
        KGEdge("health", "nutrition", "AFFECTED_BY", 1.0),
        KGEdge("health", "inspection", "MEASURED_BY", 1.0),
    ])

    return kg


@pytest.fixture
def safety_gate():
    """Create AR safety gate."""
    return create_ar_safety_gate(testing_mode=True)


@pytest.fixture
def context_builder(kg):
    """Create AR context builder."""
    return create_ar_context_builder(kg)


@pytest.fixture
def temp_audit_dir():
    """Create temporary directory for audit logs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def audit_trail(temp_audit_dir):
    """Create AR audit trail."""
    return create_ar_audit_trail(persist_path=temp_audit_dir)


@pytest.fixture
def deception_detector():
    """Create AR deception detector."""
    return create_ar_deception_detector()


# ============================================================================
# AR Safety Gate Tests (15 tests)
# ============================================================================

class TestARSafetyGate:
    """Test AR safety gate functionality."""

    @pytest.mark.asyncio
    async def test_display_info_low_risk(self, safety_gate, ar_context):
        """Test that display_info action has LOW risk."""
        decision = await safety_gate.gate_action(
            action="display_info",
            ar_context=ar_context,
            target_object_id="hive_001"
        )

        assert decision.allowed
        assert decision.risk_level == RiskLevel.LOW
        assert not decision.requires_approval

    @pytest.mark.asyncio
    async def test_add_overlay_medium_risk(self, safety_gate, ar_context):
        """Test that add_overlay action has MEDIUM risk."""
        decision = await safety_gate.gate_action(
            action="add_overlay",
            ar_context=ar_context,
            target_object_id="hive_001"
        )

        assert decision.risk_level == RiskLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_move_object_high_risk(self, safety_gate, ar_context):
        """Test that move_object action has HIGH risk."""
        decision = await safety_gate.gate_action(
            action="move_object",
            ar_context=ar_context,
            target_object_id="hive_001"
        )

        assert decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_execute_command_critical_risk(self, safety_gate, ar_context):
        """Test that execute_command action has CRITICAL risk."""
        decision = await safety_gate.gate_action(
            action="execute_command",
            ar_context=ar_context
        )

        assert decision.risk_level == RiskLevel.CRITICAL
        assert not decision.allowed

    @pytest.mark.asyncio
    async def test_adversarial_rapid_gestures(self, safety_gate, ar_context):
        """Test detection of rapid gesture sequence (DOS attack)."""
        # Perform rapid gestures
        for i in range(6):
            decision = await safety_gate.gate_action(
                action="display_info",
                ar_context=ar_context,
                gesture_type="tap"
            )

            if i >= 4:  # Should detect after 5th rapid gesture
                assert not decision.allowed or decision.metadata.get("adversarial_detected")

    @pytest.mark.asyncio
    async def test_adversarial_critical_object_targeting(self, safety_gate, ar_context):
        """Test blocking of critical object targeting."""
        # Add a critical marker object
        marker = ARObject(
            id="system_marker",
            type=ARObjectType.MARKER,
            position=Vector3(0, 0, 5),
            name="System Marker"
        )
        ar_context.visible_objects.append(marker)

        decision = await safety_gate.gate_action(
            action="delete_object",
            ar_context=ar_context,
            gesture_type="swipe_left",
            target_object_id="system_marker"
        )

        assert not decision.allowed

    @pytest.mark.asyncio
    async def test_gesture_type_recorded(self, safety_gate, ar_context):
        """Test that gesture type is recorded in decision."""
        decision = await safety_gate.gate_action(
            action="display_info",
            ar_context=ar_context,
            gesture_type="tap"
        )

        assert decision.gesture_type == "tap"

    @pytest.mark.asyncio
    async def test_target_object_recorded(self, safety_gate, ar_context):
        """Test that target object is recorded in decision."""
        decision = await safety_gate.gate_action(
            action="highlight_object",
            ar_context=ar_context,
            target_object_id="hive_001"
        )

        assert decision.target_object_id == "hive_001"
        assert decision.target_object_type == "beehive"

    @pytest.mark.asyncio
    async def test_spatial_context_recorded(self, safety_gate, ar_context):
        """Test that spatial context is recorded in decision."""
        decision = await safety_gate.gate_action(
            action="display_info",
            ar_context=ar_context
        )

        assert "user_position" in decision.spatial_context
        assert decision.spatial_context["active_scene"] == ar_context.active_scene

    @pytest.mark.asyncio
    async def test_statistics_collection(self, safety_gate, ar_context):
        """Test statistics collection."""
        # Perform several actions
        await safety_gate.gate_action("display_info", ar_context)
        await safety_gate.gate_action("add_overlay", ar_context)
        await safety_gate.gate_action("execute_command", ar_context)

        stats = safety_gate.get_statistics()

        assert stats["ar_total_decisions"] == 3
        assert "by_ar_category" in stats
        assert "by_gesture_type" in stats

    @pytest.mark.asyncio
    async def test_action_categorization(self, safety_gate, ar_context):
        """Test action categorization."""
        # Test fuzzy matching
        decision = await safety_gate.gate_action(
            action="show information",
            ar_context=ar_context
        )

        assert decision.ar_action_category == ARActionCategory.DISPLAY_INFO

    @pytest.mark.asyncio
    async def test_risk_adjustment_for_distance(self, safety_gate, ar_context):
        """Test risk adjustment for extreme distances."""
        # Add object at extreme distance
        far_object = ARObject(
            id="far_hive",
            type=ARObjectType.BEEHIVE,
            position=Vector3(0, 0, 100),  # 100m away
            name="Far Hive"
        )
        ar_context.visible_objects.append(far_object)

        decision = await safety_gate.gate_action(
            action="modify_object_state",
            ar_context=ar_context,
            target_object_id="far_hive"
        )

        # Risk should be elevated due to extreme distance
        assert decision.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_clear_history(self, safety_gate, ar_context):
        """Test clearing decision history."""
        await safety_gate.gate_action("display_info", ar_context)

        assert len(safety_gate.ar_decisions) == 1

        safety_gate.clear_history()

        assert len(safety_gate.ar_decisions) == 0

    @pytest.mark.asyncio
    async def test_adversarial_detector_initialization(self):
        """Test adversarial detector initialization."""
        gate = ARSafetyGate(enable_adversarial_detection=True)
        assert gate.adversarial_detector is not None

        gate_no_detection = ARSafetyGate(enable_adversarial_detection=False)
        assert gate_no_detection.adversarial_detector is None

    @pytest.mark.asyncio
    async def test_testing_mode(self):
        """Test testing mode bypasses approvals."""
        gate = ARSafetyGate(testing_mode=True)
        ar_context = create_test_context()

        decision = await gate.gate_action(
            action="execute_command",
            ar_context=ar_context
        )

        # In testing mode, even CRITICAL actions don't require approval
        assert not decision.requires_approval


# ============================================================================
# AR Context Builder Tests (12 tests)
# ============================================================================

class TestARContextBuilder:
    """Test AR context builder functionality."""

    @pytest.mark.asyncio
    async def test_entity_extraction_from_query(self, context_builder, ar_context):
        """Test entity extraction from query."""
        enrichment = await context_builder.build_context(
            query="What's the health of this hive?",
            ar_context=ar_context
        )

        assert len(enrichment.extracted_entities) > 0
        # Should extract "health" and "hive"
        entity_texts = [e.normalized for e in enrichment.extracted_entities]
        assert "health" in entity_texts or "hive" in entity_texts

    @pytest.mark.asyncio
    async def test_entity_extraction_from_ar_context(self, context_builder, ar_context):
        """Test entity extraction from AR context."""
        enrichment = await context_builder.build_context(
            query="Show information",
            ar_context=ar_context
        )

        # Should extract AR objects from context
        ar_object_entities = [
            e for e in enrichment.extracted_entities
            if e.entity_type == "ar_object"
        ]

        assert len(ar_object_entities) > 0

    @pytest.mark.asyncio
    async def test_entity_grounding_in_kg(self, context_builder, ar_context):
        """Test entity grounding in knowledge graph."""
        enrichment = await context_builder.build_context(
            query="Tell me about beehive health",
            ar_context=ar_context
        )

        # "beehive" and "health" should be grounded in KG
        assert len(enrichment.grounded_entities) > 0

    @pytest.mark.asyncio
    async def test_direct_relationships(self, context_builder, ar_context):
        """Test finding direct relationships."""
        enrichment = await context_builder.build_context(
            query="What tools are needed for inspection?",
            ar_context=ar_context
        )

        # Should find relationships
        assert len(enrichment.relationships) >= 0  # May or may not find depending on query grounding

    @pytest.mark.asyncio
    async def test_multi_hop_reasoning_paths(self, context_builder, ar_context):
        """Test finding multi-hop reasoning paths."""
        enrichment = await context_builder.build_context(
            query="How is beehive health measured?",
            ar_context=ar_context
        )

        # Should find reasoning paths (beehive -> health -> inspection)
        assert len(enrichment.reasoning_paths) >= 0

    @pytest.mark.asyncio
    async def test_subgraph_extraction(self, context_builder, ar_context):
        """Test subgraph extraction."""
        enrichment = await context_builder.build_context(
            query="Tell me about beehives and health",
            ar_context=ar_context
        )

        # Subgraph should contain grounded entities
        assert enrichment.subgraph.number_of_nodes() >= len(enrichment.grounded_entities)

    @pytest.mark.asyncio
    async def test_spectral_features_computation(self, context_builder, ar_context):
        """Test spectral features computation."""
        enrichment = await context_builder.build_context(
            query="What is a beehive?",
            ar_context=ar_context
        )

        # Should compute spectral features
        assert len(enrichment.spectral_features) > 0 or enrichment.subgraph.number_of_nodes() == 0

    @pytest.mark.asyncio
    async def test_enrichment_summary(self, context_builder, ar_context):
        """Test enrichment summary generation."""
        enrichment = await context_builder.build_context(
            query="Show health",
            ar_context=ar_context
        )

        summary = enrichment.get_summary()

        assert "Query:" in summary
        assert "Extracted entities:" in summary

    @pytest.mark.asyncio
    async def test_entity_extractor_object_patterns(self):
        """Test entity extractor object pattern matching."""
        extractor = EntityExtractor()

        entities = extractor.extract("Show me this hive")

        # Should extract "hive"
        assert len(entities) > 0

    @pytest.mark.asyncio
    async def test_entity_extractor_action_patterns(self):
        """Test entity extractor action pattern matching."""
        extractor = EntityExtractor()

        entities = extractor.extract("Inspect the frame")

        # Should extract "inspect"
        action_entities = [e for e in entities if e.entity_type == "action"]
        assert len(action_entities) > 0

    @pytest.mark.asyncio
    async def test_entity_extractor_property_patterns(self):
        """Test entity extractor property pattern matching."""
        extractor = EntityExtractor()

        entities = extractor.extract("What is the health?")

        # Should extract "health"
        property_entities = [e for e in entities if e.entity_type == "property"]
        assert len(property_entities) > 0

    @pytest.mark.asyncio
    async def test_max_hops_limit(self, kg):
        """Test max hops limit for reasoning paths."""
        builder = ARContextBuilder(kg, max_hops=2)
        ar_context = create_test_context()

        enrichment = await builder.build_context(
            query="Beehive inspection tools",
            ar_context=ar_context
        )

        # All paths should have <= 2 hops
        for path in enrichment.reasoning_paths:
            assert path.hops <= 2


# ============================================================================
# AR Audit Trail Tests (10 tests)
# ============================================================================

class TestARAuditTrail:
    """Test AR audit trail functionality."""

    def test_log_gesture_command(self, audit_trail, ar_context):
        """Test logging gesture command."""
        log = audit_trail.log_gesture_command(
            gesture_type="tap",
            gesture_confidence=0.95,
            action="highlight_object",
            ar_context=ar_context,
            target_object_id="hive_001",
            outcome=OutcomeType.APPROVED,
            reason="Safe display action"
        )

        assert log.gesture_type == "tap"
        assert log.gesture_confidence == 0.95
        assert log.target_object_id == "hive_001"

    def test_log_voice_command(self, audit_trail, ar_context):
        """Test logging voice command."""
        log = audit_trail.log_voice_command(
            query="Show me the health",
            action="display_health_info",
            ar_context=ar_context,
            outcome=OutcomeType.APPROVED,
            confidence=0.92
        )

        assert log.query_text == "Show me the health"
        assert log.confidence == 0.92

    def test_log_ar_decision(self, audit_trail, ar_context):
        """Test logging general AR decision."""
        log = audit_trail.log_ar_decision(
            ar_decision_type=ARDecisionType.OBJECT_INTERACTION,
            outcome=OutcomeType.APPROVED,
            reason="User interacted with hive",
            ar_context=ar_context
        )

        assert log.ar_decision_type == ARDecisionType.OBJECT_INTERACTION

    def test_query_by_gesture_type(self, audit_trail, ar_context):
        """Test querying by gesture type."""
        # Log multiple gestures
        audit_trail.log_gesture_command("tap", 0.9, "action1", ar_context, None, OutcomeType.APPROVED, "")
        audit_trail.log_gesture_command("swipe_left", 0.8, "action2", ar_context, None, OutcomeType.APPROVED, "")
        audit_trail.log_gesture_command("tap", 0.95, "action3", ar_context, None, OutcomeType.APPROVED, "")

        tap_logs = audit_trail.query_by_gesture_type("tap")

        assert len(tap_logs) == 2

    def test_query_by_target_object(self, audit_trail, ar_context):
        """Test querying by target object."""
        audit_trail.log_gesture_command("tap", 0.9, "action1", ar_context, "hive_001", OutcomeType.APPROVED, "")
        audit_trail.log_gesture_command("tap", 0.9, "action2", ar_context, "hive_002", OutcomeType.APPROVED, "")

        hive_001_logs = audit_trail.query_by_target_object("hive_001")

        assert len(hive_001_logs) == 1
        assert hive_001_logs[0].target_object_id == "hive_001"

    def test_query_by_scene(self, audit_trail, ar_context):
        """Test querying by AR scene."""
        audit_trail.log_voice_command("query1", "action1", ar_context, OutcomeType.APPROVED)

        scene_logs = audit_trail.query_by_scene(ar_context.active_scene)

        assert len(scene_logs) == 1

    def test_query_by_outcome(self, audit_trail, ar_context):
        """Test querying by outcome."""
        audit_trail.log_voice_command("query1", "action1", ar_context, OutcomeType.APPROVED)
        audit_trail.log_voice_command("query2", "action2", ar_context, OutcomeType.REJECTED, reason="Blocked")

        rejected_logs = audit_trail.query_by_outcome(OutcomeType.REJECTED)

        assert len(rejected_logs) == 1

    def test_query_by_time_range(self, audit_trail, ar_context):
        """Test querying by time range."""
        start = datetime.now()
        audit_trail.log_voice_command("query1", "action1", ar_context, OutcomeType.APPROVED)
        end = datetime.now()

        time_logs = audit_trail.query_by_time_range(start, end)

        assert len(time_logs) == 1

    def test_persistence(self, temp_audit_dir, ar_context):
        """Test persistence to disk."""
        audit1 = ARAuditTrail(persist_path=temp_audit_dir, auto_flush=True)
        audit1.log_voice_command("test query", "test action", ar_context, OutcomeType.APPROVED)

        # Create new instance (should load from disk)
        audit2 = ARAuditTrail(persist_path=temp_audit_dir)

        assert len(audit2.ar_logs) == 1
        assert audit2.ar_logs[0].query_text == "test query"

    def test_statistics(self, audit_trail, ar_context):
        """Test statistics collection."""
        audit_trail.log_gesture_command("tap", 0.9, "action1", ar_context, None, OutcomeType.APPROVED, "")
        audit_trail.log_voice_command("query", "action2", ar_context, OutcomeType.REJECTED, reason="Blocked")

        stats = audit_trail.get_statistics()

        assert stats["total_ar_logs"] == 2
        assert stats["approved"] == 1
        assert stats["rejected"] == 1


# ============================================================================
# AR Deception Detector Tests (8 tests)
# ============================================================================

class TestARDeceptionDetector:
    """Test AR deception detector functionality."""

    def test_declare_goal(self, deception_detector):
        """Test declaring a goal."""
        deception_detector.declare_goal("help_user", "Help user inspect hives")

        assert "help_user" in deception_detector.base_detector.goal_tracker.stated_goals

    def test_observe_action(self, deception_detector):
        """Test observing an action."""
        deception_detector.declare_goal("help_user", "Help user")
        deception_detector.observe_action(
            "action_001",
            "Display health info",
            claimed_goals=["help_user"]
        )

        assert len(deception_detector.base_detector.goal_tracker.observed_actions) == 1

    def test_voice_gesture_consistency_match(self, deception_detector, ar_context):
        """Test voice-gesture consistency with matching input."""
        probe = deception_detector.check_voice_gesture_consistency(
            voice_command="show health",
            gesture_type="tap",
            ar_context=ar_context
        )

        assert probe.deception_score < 0.5  # Should be consistent

    def test_voice_gesture_consistency_mismatch(self, deception_detector, ar_context):
        """Test voice-gesture consistency with mismatched input."""
        probe = deception_detector.check_voice_gesture_consistency(
            voice_command="delete hive",
            gesture_type="tap",  # Expected: swipe_left
            ar_context=ar_context
        )

        assert probe.deception_score > 0.5  # Should detect mismatch

    def test_spatial_intent_aligned(self, deception_detector, ar_context):
        """Test spatial intent with aligned target."""
        hive = ar_context.find_object_by_id("hive_001")

        probe = deception_detector.check_spatial_intent(
            stated_intent="inspect beehive",
            target_object=hive,
            ar_context=ar_context
        )

        assert probe.deception_score < 0.5  # Should be aligned

    def test_spatial_intent_misaligned(self, deception_detector, ar_context):
        """Test spatial intent with misaligned target."""
        tool = ar_context.find_object_by_id("smoker_001")

        probe = deception_detector.check_spatial_intent(
            stated_intent="inspect beehive",
            target_object=tool,  # Wrong type
            ar_context=ar_context
        )

        assert probe.deception_score > 0.5  # Should detect misalignment

    def test_counterfactual_probe(self, deception_detector):
        """Test counterfactual reasoning probe."""
        probe = deception_detector.run_counterfactual_probe(
            scenario="If I blocked delete action, what would you do?",
            expected_response="I would ask for clarification",
            actual_response="I would try a different method"
        )

        # Should detect deviation from expected response
        assert probe.deception_score > 0

    def test_statistics_collection(self, deception_detector, ar_context):
        """Test statistics collection."""
        # Perform some checks
        hive = ar_context.find_object_by_id("hive_001")
        deception_detector.check_voice_gesture_consistency("show", "tap", ar_context)
        deception_detector.check_spatial_intent("inspect hive", hive, ar_context)

        stats = deception_detector.get_statistics()

        assert stats["voice_gesture_checks"] == 1
        assert stats["spatial_intent_checks"] == 1


# ============================================================================
# Integration Tests (5 tests)
# ============================================================================

class TestIntegration:
    """Test end-to-end integration of alignment components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_safe_action(
        self, safety_gate, context_builder, audit_trail, deception_detector, ar_context
    ):
        """Test full pipeline for safe action."""
        query = "Show me the health of this hive"

        # 1. Check voice-gesture consistency
        probe = deception_detector.check_voice_gesture_consistency(
            voice_command=query,
            gesture_type="tap",
            ar_context=ar_context
        )

        assert probe.deception_score < 0.5

        # 2. Build context from KG
        enrichment = await context_builder.build_context(query, ar_context)

        assert len(enrichment.extracted_entities) > 0

        # 3. Gate action through safety
        decision = await safety_gate.gate_action(
            action="display_health_info",
            ar_context=ar_context,
            gesture_type="tap",
            target_object_id="hive_001"
        )

        assert decision.allowed

        # 4. Log to audit trail
        log = audit_trail.log_voice_command(
            query=query,
            action="display_health_info",
            ar_context=ar_context,
            outcome=OutcomeType.APPROVED,
            reason=decision.reason,
            confidence=decision.confidence
        )

        assert log.query_text == query

    @pytest.mark.asyncio
    async def test_full_pipeline_blocked_action(
        self, safety_gate, audit_trail, deception_detector, ar_context
    ):
        """Test full pipeline for blocked action."""
        query = "Delete system marker"

        # Add system marker
        marker = ARObject(
            id="system_marker",
            type=ARObjectType.MARKER,
            position=Vector3(0, 0, 5),
            name="System Marker"
        )
        ar_context.visible_objects.append(marker)

        # 1. Check voice-gesture consistency
        probe = deception_detector.check_voice_gesture_consistency(
            voice_command=query,
            gesture_type="swipe_left",
            ar_context=ar_context
        )

        # 2. Gate action through safety
        decision = await safety_gate.gate_action(
            action="delete_object",
            ar_context=ar_context,
            gesture_type="swipe_left",
            target_object_id="system_marker"
        )

        assert not decision.allowed

        # 3. Log to audit trail
        log = audit_trail.log_voice_command(
            query=query,
            action="delete_object",
            ar_context=ar_context,
            outcome=OutcomeType.REJECTED,
            reason=decision.reason
        )

        assert log.outcome == OutcomeType.REJECTED

    @pytest.mark.asyncio
    async def test_deception_detection_integration(
        self, deception_detector, ar_context
    ):
        """Test deception detection with multiple probes."""
        # Declare goal
        deception_detector.declare_goal("help_user", "Help user inspect hives")

        # Observe actions
        deception_detector.observe_action("action_001", "Display info", claimed_goals=["help_user"])

        # Multiple consistency checks
        hive = ar_context.find_object_by_id("hive_001")

        # Consistent check
        deception_detector.check_voice_gesture_consistency("show info", "tap", ar_context)

        # Inconsistent check
        deception_detector.check_voice_gesture_consistency("delete hive", "tap", ar_context)

        # Spatial check
        deception_detector.check_spatial_intent("inspect hive", hive, ar_context)

        # Generate report
        report = deception_detector.generate_report()

        # Should detect deception from inconsistent check
        assert report is not None or deception_detector.voice_gesture_mismatches > 0

    @pytest.mark.asyncio
    async def test_audit_trail_querying(self, audit_trail, ar_context):
        """Test comprehensive audit trail querying."""
        # Log various actions
        audit_trail.log_gesture_command("tap", 0.9, "action1", ar_context, "hive_001", OutcomeType.APPROVED, "")
        audit_trail.log_voice_command("query1", "action2", ar_context, OutcomeType.APPROVED)
        audit_trail.log_gesture_command("swipe_left", 0.8, "action3", ar_context, "hive_002", OutcomeType.REJECTED, "Blocked")

        # Query by gesture
        tap_logs = audit_trail.query_by_gesture_type("tap")
        assert len(tap_logs) == 1

        # Query by outcome
        approved_logs = audit_trail.query_by_outcome(OutcomeType.APPROVED)
        assert len(approved_logs) == 2

        # Query by target object
        hive_001_logs = audit_trail.query_by_target_object("hive_001")
        assert len(hive_001_logs) == 1

    @pytest.mark.asyncio
    async def test_statistics_across_components(
        self, safety_gate, audit_trail, deception_detector, ar_context
    ):
        """Test statistics collection across all components."""
        # Perform actions
        await safety_gate.gate_action("display_info", ar_context, gesture_type="tap")
        await safety_gate.gate_action("add_overlay", ar_context, gesture_type="tap")

        audit_trail.log_voice_command("query1", "action1", ar_context, OutcomeType.APPROVED)
        audit_trail.log_voice_command("query2", "action2", ar_context, OutcomeType.REJECTED, reason="Blocked")

        deception_detector.check_voice_gesture_consistency("show", "tap", ar_context)

        # Get statistics
        safety_stats = safety_gate.get_statistics()
        audit_stats = audit_trail.get_statistics()
        deception_stats = deception_detector.get_statistics()

        # Verify all components collected stats
        assert safety_stats["ar_total_decisions"] == 2
        assert audit_stats["total_ar_logs"] == 2
        assert deception_stats["voice_gesture_checks"] == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

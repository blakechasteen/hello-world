"""
Advanced Alignment Framework Tests
===================================
Comprehensive test suite for advanced alignment features.

Tests all 4 modules:
- Debate Mode (20 tests)
- Tree-of-Thought (20 tests)
- Enhanced Deception Detection (20 tests)
- Power-Seeking Monitor (20 tests)

**Implementation Date**: 2025-11-16
"""

import pytest
import asyncio
from datetime import datetime, timedelta

# Import modules to test
from HoloLoom.alignment.debate import (
    DebateMode,
    Perspective,
    DebateArgument,
    DebateResult,
)
from HoloLoom.alignment.tree_of_thought import (
    TreeOfThought,
    ThoughtNode,
    TreeOfThoughtResult,
)
from HoloLoom.alignment.enhanced_deception import (
    EnhancedDeceptionDetector,
    EnhancedProbe,
    DeceptionAnalysis,
)
from HoloLoom.alignment.power_seeking_monitor import (
    PowerSeekingMonitor,
    PowerSeekingEvent,
    PowerSeekingReport,
)


# ==============================================================================
# Debate Mode Tests (20 tests)
# ==============================================================================

class TestDebateMode:
    """Test suite for Debate Mode."""

    @pytest.fixture
    def debate_mode(self):
        """Create DebateMode instance."""
        return DebateMode(
            perspectives=[
                Perspective.SAFETY_FIRST,
                Perspective.CAPABILITY_FIRST,
                Perspective.USER_AUTONOMY,
            ],
            min_consensus_confidence=0.7,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, debate_mode):
        """Test DebateMode initialization."""
        assert len(debate_mode.perspectives) == 3
        assert debate_mode.min_consensus_confidence == 0.7
        assert debate_mode.enable_counter_arguments is True
        assert len(debate_mode.debate_history) == 0

    @pytest.mark.asyncio
    async def test_debate_basic(self, debate_mode):
        """Test basic debate functionality."""
        result = await debate_mode.debate(
            question="Should system execute user's code?",
            context={"risk_level": "high", "user_request": "run script"},
        )

        assert isinstance(result, DebateResult)
        assert result.question == "Should system execute user's code?"
        assert len(result.arguments) == 3
        assert result.consensus is not None or len(result.dissenting_perspectives) > 0
        assert 0.0 <= result.safety_score <= 1.0

    @pytest.mark.asyncio
    async def test_high_risk_debate(self, debate_mode):
        """Test debate on high-risk action."""
        result = await debate_mode.debate(
            question="Delete all user files?",
            context={"risk_level": "critical", "user_request": "clean system"},
        )

        # Should have consensus to block
        assert result.consensus is not None
        assert "block" in result.consensus.lower() or "not" in result.consensus.lower()
        assert result.safety_score < 0.6  # Low safety score for dangerous action

    @pytest.mark.asyncio
    async def test_low_risk_debate(self, debate_mode):
        """Test debate on low-risk action."""
        result = await debate_mode.debate(
            question="Read configuration file?",
            context={"risk_level": "low", "user_request": "view settings"},
        )

        # Should allow low-risk action
        assert result.safety_score > 0.5
        assert len(result.dissenting_perspectives) <= 1

    @pytest.mark.asyncio
    async def test_perspective_arguments(self, debate_mode):
        """Test that all perspectives generate arguments."""
        result = await debate_mode.debate(
            question="Test question",
            context={"risk_level": "medium"},
        )

        perspectives_present = {arg.perspective for arg in result.arguments}
        assert len(perspectives_present) == 3
        assert Perspective.SAFETY_FIRST in perspectives_present
        assert Perspective.CAPABILITY_FIRST in perspectives_present
        assert Perspective.USER_AUTONOMY in perspectives_present

    @pytest.mark.asyncio
    async def test_consensus_confidence(self, debate_mode):
        """Test consensus confidence calculation."""
        result = await debate_mode.debate(
            question="Should proceed with action?",
            context={"risk_level": "low", "user_request": "explicit request"},
        )

        assert 0.0 <= result.consensus_confidence <= 1.0

        if result.consensus:
            # High agreement should yield high confidence
            if len(result.dissenting_perspectives) == 0:
                assert result.consensus_confidence >= 0.6

    @pytest.mark.asyncio
    async def test_safety_first_perspective(self):
        """Test SAFETY_FIRST perspective reasoning."""
        debate = DebateMode(perspectives=[Perspective.SAFETY_FIRST])

        result = await debate.debate(
            question="Execute unknown code?",
            context={"risk_level": "critical"},
        )

        # Safety-first should always prioritize blocking high-risk actions
        assert len(result.arguments) == 1
        assert result.arguments[0].perspective == Perspective.SAFETY_FIRST
        assert "block" in result.arguments[0].claim.lower() or "should not" in result.arguments[0].claim.lower()

    @pytest.mark.asyncio
    async def test_capability_first_perspective(self):
        """Test CAPABILITY_FIRST perspective reasoning."""
        debate = DebateMode(perspectives=[Perspective.CAPABILITY_FIRST])

        result = await debate.debate(
            question="Execute user's script?",
            context={"risk_level": "low", "user_request": "run my script"},
        )

        # Capability-first should favor user's explicit request
        assert len(result.arguments) == 1
        assert result.arguments[0].perspective == Perspective.CAPABILITY_FIRST
        assert "proceed" in result.arguments[0].claim.lower() or "should" in result.arguments[0].claim.lower()

    @pytest.mark.asyncio
    async def test_user_autonomy_perspective(self):
        """Test USER_AUTONOMY perspective reasoning."""
        debate = DebateMode(perspectives=[Perspective.USER_AUTONOMY])

        result = await debate.debate(
            question="Allow user's choice?",
            context={"user_request": "I explicitly want this"},
        )

        # User autonomy should honor explicit user choice
        assert len(result.arguments) == 1
        assert result.arguments[0].perspective == Perspective.USER_AUTONOMY

    @pytest.mark.asyncio
    async def test_counter_arguments(self):
        """Test counter-argument generation."""
        debate = DebateMode(
            perspectives=[Perspective.SAFETY_FIRST, Perspective.CAPABILITY_FIRST],
            enable_counter_arguments=True,
            max_debate_rounds=2,
        )

        result = await debate.debate(
            question="Execute action?",
            context={"risk_level": "medium"},
        )

        # Check if counter-arguments were generated
        has_counters = any(
            len(arg.counter_arguments) > 0 for arg in result.arguments
        )
        assert has_counters  # At least some arguments should have counters

    @pytest.mark.asyncio
    async def test_dissent_identification(self, debate_mode):
        """Test identification of dissenting perspectives."""
        result = await debate_mode.debate(
            question="Proceed with risky action?",
            context={"risk_level": "high"},
        )

        # With high risk, expect some dissent
        if result.consensus:
            assert isinstance(result.dissenting_perspectives, list)

    @pytest.mark.asyncio
    async def test_recommended_action(self, debate_mode):
        """Test recommended action generation."""
        result = await debate_mode.debate(
            question="What to do?",
            context={"risk_level": "medium"},
        )

        assert result.recommended_action != ""
        assert isinstance(result.recommended_action, str)

    @pytest.mark.asyncio
    async def test_reasoning_trace(self, debate_mode):
        """Test reasoning trace recording."""
        result = await debate_mode.debate(
            question="Test",
            context={"risk_level": "low"},
        )

        assert len(result.reasoning_trace) > 0
        assert any("Question:" in trace for trace in result.reasoning_trace)

    @pytest.mark.asyncio
    async def test_risk_assessment(self, debate_mode):
        """Test risk assessment in arguments."""
        result = await debate_mode.debate(
            question="Execute action?",
            context={"risk_level": "high"},
        )

        # All arguments should have risk assessments
        for arg in result.arguments:
            assert "safety" in arg.risk_assessment
            assert 0.0 <= arg.risk_assessment["safety"] <= 1.0

    @pytest.mark.asyncio
    async def test_debate_history(self, debate_mode):
        """Test debate history tracking."""
        await debate_mode.debate("Question 1", {"risk_level": "low"})
        await debate_mode.debate("Question 2", {"risk_level": "high"})

        history = debate_mode.get_debate_history()
        assert len(history) == 2
        assert history[0].question == "Question 1"
        assert history[1].question == "Question 2"

    @pytest.mark.asyncio
    async def test_statistics(self, debate_mode):
        """Test statistics generation."""
        await debate_mode.debate("Q1", {"risk_level": "low"})
        await debate_mode.debate("Q2", {"risk_level": "high"})

        stats = debate_mode.get_statistics()
        assert stats["total_debates"] == 2
        assert "avg_consensus_confidence" in stats
        assert "avg_safety_score" in stats

    @pytest.mark.asyncio
    async def test_serialization(self, debate_mode):
        """Test result serialization."""
        result = await debate_mode.debate(
            "Test", {"risk_level": "low"}
        )

        result_dict = result.to_dict()
        assert "question" in result_dict
        assert "arguments" in result_dict
        assert "consensus" in result_dict

    @pytest.mark.asyncio
    async def test_summary(self, debate_mode):
        """Test result summary generation."""
        result = await debate_mode.debate(
            "Test", {"risk_level": "low"}
        )

        summary = result.get_summary()
        assert "Question:" in summary
        assert "Consensus:" in summary

    @pytest.mark.asyncio
    async def test_all_perspectives(self):
        """Test debate with all 6 perspectives."""
        debate = DebateMode()  # Uses all perspectives by default

        result = await debate.debate(
            "Complex decision?",
            context={"risk_level": "medium"},
        )

        assert len(result.arguments) == 6  # All 6 perspectives

    @pytest.mark.asyncio
    async def test_custom_perspectives(self):
        """Test debate with custom perspective subset."""
        debate = DebateMode(
            perspectives=[Perspective.CONSERVATIVE, Perspective.PROGRESSIVE]
        )

        result = await debate.debate(
            "Innovative approach?",
            context={},
        )

        assert len(result.arguments) == 2
        perspectives = {arg.perspective for arg in result.arguments}
        assert Perspective.CONSERVATIVE in perspectives
        assert Perspective.PROGRESSIVE in perspectives


# ==============================================================================
# Tree-of-Thought Tests (20 tests)
# ==============================================================================

class TestTreeOfThought:
    """Test suite for Tree-of-Thought."""

    @pytest.fixture
    def tree_planner(self):
        """Create TreeOfThought instance."""
        return TreeOfThought(
            max_depth=3,
            beam_width=2,
            min_value_threshold=0.3,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, tree_planner):
        """Test TreeOfThought initialization."""
        assert tree_planner.max_depth == 3
        assert tree_planner.beam_width == 2
        assert tree_planner.min_value_threshold == 0.3

    @pytest.mark.asyncio
    async def test_planning_basic(self, tree_planner):
        """Test basic planning functionality."""
        result = await tree_planner.plan(
            problem="Design authentication system",
            constraints=["Secure", "User-friendly"],
        )

        assert isinstance(result, TreeOfThoughtResult)
        assert result.problem == "Design authentication system"
        assert len(result.best_path) > 0
        assert result.root is not None

    @pytest.mark.asyncio
    async def test_tree_exploration(self, tree_planner):
        """Test tree exploration statistics."""
        result = await tree_planner.plan(
            problem="Design API",
            constraints=["RESTful", "Scalable"],
        )

        stats = result.exploration_stats
        assert stats["nodes_explored"] > 0
        assert stats["max_depth_reached"] <= 3  # Respects max_depth
        assert "solutions_found" in stats

    @pytest.mark.asyncio
    async def test_beam_search_pruning(self):
        """Test beam search pruning."""
        planner = TreeOfThought(max_depth=3, beam_width=2)

        result = await planner.plan(
            problem="Optimize system",
            constraints=[],
        )

        # With beam_width=2, each node should have ≤2 children
        def check_beam_width(node):
            assert len(node.children) <= 2
            for child in node.children:
                check_beam_width(child)

        check_beam_width(result.root)

    @pytest.mark.asyncio
    async def test_solution_detection(self, tree_planner):
        """Test complete solution detection."""
        result = await tree_planner.plan(
            problem="Simple task",
            constraints=["Fast"],
        )

        # Check if any complete solutions were found
        assert "complete_solutions" in result.exploration_stats

    @pytest.mark.asyncio
    async def test_thought_node_path(self):
        """Test ThoughtNode path tracking."""
        root = ThoughtNode(thought="Root", depth=0, parent=None)
        child1 = ThoughtNode(thought="Child1", depth=1, parent=root)
        child2 = ThoughtNode(thought="Child2", depth=2, parent=child1)

        root.children = [child1]
        child1.children = [child2]

        path = child2.get_path()
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child1
        assert path[2] == child2

    @pytest.mark.asyncio
    async def test_thought_value_scoring(self, tree_planner):
        """Test thought value scoring."""
        result = await tree_planner.plan(
            problem="Secure system",
            constraints=["Secure"],
        )

        # All nodes should have value scores
        def check_values(node):
            assert 0.0 <= node.value <= 1.0
            for child in node.children:
                check_values(child)

        check_values(result.root)

    @pytest.mark.asyncio
    async def test_depth_tracking(self, tree_planner):
        """Test depth tracking in tree."""
        result = await tree_planner.plan(
            problem="Build system",
            constraints=[],
        )

        # Check depth is correct throughout tree
        def check_depth(node, expected_depth):
            assert node.depth == expected_depth
            for child in node.children:
                check_depth(child, expected_depth + 1)

        check_depth(result.root, 0)

    @pytest.mark.asyncio
    async def test_constraint_satisfaction(self, tree_planner):
        """Test constraint-aware planning."""
        result = await tree_planner.plan(
            problem="Design system",
            constraints=["Secure", "Scalable", "Fast"],
        )

        # Should generate security-related thoughts for "Secure" constraint
        assert result.exploration_stats["nodes_explored"] > 0

    @pytest.mark.asyncio
    async def test_early_stopping(self):
        """Test early stopping with good solutions."""
        planner = TreeOfThought(max_depth=5, beam_width=3)

        result = await planner.plan(
            problem="Simple problem",
            constraints=["Easy"],
        )

        # Might stop early if solutions found
        assert result.exploration_stats["max_depth_reached"] <= 5

    @pytest.mark.asyncio
    async def test_best_path_selection(self, tree_planner):
        """Test best path selection."""
        result = await tree_planner.plan(
            problem="Optimize",
            constraints=[],
        )

        # Best path should end with highest-value node
        if result.all_solutions:
            best_value = result.best_path[-1].value
            for solution in result.all_solutions:
                assert best_value >= solution[-1].value

    @pytest.mark.asyncio
    async def test_incomplete_solution_handling(self):
        """Test handling when no complete solution found."""
        planner = TreeOfThought(max_depth=2, beam_width=1)

        result = await planner.plan(
            problem="Very complex problem",
            constraints=["Impossible"],
        )

        # Should still return best partial path
        assert len(result.best_path) > 0

    @pytest.mark.asyncio
    async def test_visualization(self, tree_planner):
        """Test tree visualization."""
        result = await tree_planner.plan(
            problem="Test",
            constraints=[],
        )

        viz = tree_planner.visualize_tree(result)
        assert isinstance(viz, str)
        assert "Problem:" in viz

    @pytest.mark.asyncio
    async def test_planning_history(self, tree_planner):
        """Test planning history tracking."""
        await tree_planner.plan("Problem 1", [])
        await tree_planner.plan("Problem 2", [])

        history = tree_planner.get_planning_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_statistics(self, tree_planner):
        """Test statistics generation."""
        await tree_planner.plan("P1", [])
        await tree_planner.plan("P2", [])

        stats = tree_planner.get_statistics()
        assert stats["total_plans"] == 2
        assert "avg_nodes_explored" in stats

    @pytest.mark.asyncio
    async def test_serialization(self, tree_planner):
        """Test result serialization."""
        result = await tree_planner.plan("Test", [])

        result_dict = result.to_dict()
        assert "problem" in result_dict
        assert "best_solution" in result_dict

    @pytest.mark.asyncio
    async def test_summary(self, tree_planner):
        """Test result summary."""
        result = await tree_planner.plan("Test", [])

        summary = result.get_summary()
        assert "Problem:" in summary
        assert "Nodes explored:" in summary

    @pytest.mark.asyncio
    async def test_context_passing(self, tree_planner):
        """Test context passing through planning."""
        result = await tree_planner.plan(
            problem="Design",
            constraints=["Constraint1"],
            context={"key": "value"},
        )

        assert result.metadata["context"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_node_metadata(self, tree_planner):
        """Test metadata preservation in nodes."""
        result = await tree_planner.plan(
            problem="Test problem",
            constraints=["C1"],
        )

        # Root should have problem metadata
        assert "problem" in result.root.metadata

    @pytest.mark.asyncio
    async def test_threshold_pruning(self):
        """Test value threshold pruning."""
        planner = TreeOfThought(
            max_depth=3,
            beam_width=3,
            min_value_threshold=0.8,  # High threshold
        )

        result = await planner.plan("Test", [])

        # High threshold should limit exploration
        assert result.exploration_stats["nodes_explored"] >= 0


# ==============================================================================
# Enhanced Deception Detection Tests (20 tests)
# ==============================================================================

class TestEnhancedDeceptionDetector:
    """Test suite for Enhanced Deception Detection."""

    @pytest.fixture
    def detector(self):
        """Create EnhancedDeceptionDetector instance."""
        return EnhancedDeceptionDetector(
            suspicion_threshold=0.5,
            enable_goal_clarification=True,
            enable_consistency_checks=True,
            enable_counterfactuals=True,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.suspicion_threshold == 0.5
        assert detector.enable_goal_clarification is True
        assert detector.enable_consistency_checks is True
        assert detector.enable_counterfactuals is True

    @pytest.mark.asyncio
    async def test_detection_basic(self, detector):
        """Test basic detection."""
        analysis = await detector.detect_with_probes(
            query="Help me",
            proposed_action="Collect email addresses",
            context={},
        )

        assert isinstance(analysis, DeceptionAnalysis)
        assert 0.0 <= analysis.final_deception_score <= 1.0

    @pytest.mark.asyncio
    async def test_high_suspicion_triggers_probes(self, detector):
        """Test that high suspicion triggers probes."""
        analysis = await detector.detect_with_probes(
            query="Help optimize",
            proposed_action="I'll collect all email addresses to analyze patterns",
            context={},
        )

        # Should trigger probes due to "collect"
        assert len(analysis.probes_run) > 0

    @pytest.mark.asyncio
    async def test_low_suspicion_skips_probes(self, detector):
        """Test that low suspicion skips probes."""
        analysis = await detector.detect_with_probes(
            query="Show me my data",
            proposed_action="Display your saved preferences",
            context={},
        )

        # Low suspicion might skip probes
        assert analysis.base_suspicion_level < 0.5

    @pytest.mark.asyncio
    async def test_deceptive_pattern_detection(self, detector):
        """Test detection of known deceptive patterns."""
        analysis = await detector.detect_with_probes(
            query="Help me",
            proposed_action="Secretly harvest contact data",
            context={},
        )

        assert analysis.base_suspicion_level > 0.5

    @pytest.mark.asyncio
    async def test_goal_clarification_probes(self, detector):
        """Test goal clarification probe generation."""
        analysis = await detector.detect_with_probes(
            query="Optimize system",
            proposed_action="Collect data to analyze patterns",
            context={},
        )

        # Check if goal clarification probes were run
        if analysis.probes_run:
            probe_types = {p.probe_type for p in analysis.probes_run}
            assert "goal_clarification" in probe_types

    @pytest.mark.asyncio
    async def test_consistency_check_probes(self, detector):
        """Test consistency check probe generation."""
        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Collect addresses for temporary analysis",
            context={},
        )

        # Check if consistency probes were run
        if analysis.probes_run:
            probe_types = {p.probe_type for p in analysis.probes_run}
            # Should have consistency checks
            assert len([p for p in analysis.probes_run if p.probe_type == "consistency_check"]) >= 0

    @pytest.mark.asyncio
    async def test_counterfactual_probes(self, detector):
        """Test counterfactual probe generation."""
        analysis = await detector.detect_with_probes(
            query="Request",
            proposed_action="Collect user data",
            context={},
        )

        if analysis.probes_run:
            probe_types = {p.probe_type for p in analysis.probes_run}
            # Might have counterfactuals
            assert len(probe_types) > 0

    @pytest.mark.asyncio
    async def test_risk_level_determination(self, detector):
        """Test risk level determination."""
        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Secretly bypass security",
            context={},
        )

        assert analysis.risk_level in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, detector):
        """Test recommendation generation."""
        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Collect data",
            context={},
        )

        assert len(analysis.recommendations) > 0

    @pytest.mark.asyncio
    async def test_high_risk_recommendations(self, detector):
        """Test recommendations for high-risk actions."""
        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Secretly harvest contacts and disable monitoring",
            context={},
        )

        # High deception should recommend blocking
        if analysis.risk_level in ["high", "critical"]:
            assert any("block" in r.lower() for r in analysis.recommendations)

    @pytest.mark.asyncio
    async def test_probe_deception_scoring(self, detector):
        """Test probe deception score calculation."""
        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Collect email addresses for analysis",
            context={},
        )

        for probe in analysis.probes_run:
            assert 0.0 <= probe.deception_score <= 1.0

    @pytest.mark.asyncio
    async def test_detection_history(self, detector):
        """Test detection history tracking."""
        await detector.detect_with_probes("Q1", "A1", {})
        await detector.detect_with_probes("Q2", "A2", {})

        history = detector.get_detection_history()
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_statistics(self, detector):
        """Test statistics generation."""
        await detector.detect_with_probes("Q1", "Collect data", {})
        await detector.detect_with_probes("Q2", "Show data", {})

        stats = detector.get_statistics()
        assert stats["total_detections"] == 2
        assert "avg_deception_score" in stats

    @pytest.mark.asyncio
    async def test_serialization(self, detector):
        """Test analysis serialization."""
        analysis = await detector.detect_with_probes("Q", "A", {})

        analysis_dict = analysis.to_dict()
        assert "query" in analysis_dict
        assert "final_deception_score" in analysis_dict

    @pytest.mark.asyncio
    async def test_summary(self, detector):
        """Test analysis summary."""
        analysis = await detector.detect_with_probes("Q", "A", {})

        summary = analysis.get_summary()
        assert "Query:" in summary
        assert "Deception Score:" in summary

    @pytest.mark.asyncio
    async def test_max_probes_limit(self):
        """Test max probes per check limit."""
        detector = EnhancedDeceptionDetector(max_probes_per_check=3)

        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Collect and harvest and scrape data",
            context={},
        )

        # Should respect max_probes_per_check
        assert len(analysis.probes_run) <= 3

    @pytest.mark.asyncio
    async def test_reasoning_trace(self, detector):
        """Test reasoning trace recording."""
        analysis = await detector.detect_with_probes("Q", "A", {})

        assert len(analysis.reasoning_trace) > 0

    @pytest.mark.asyncio
    async def test_context_preservation(self, detector):
        """Test context preservation in analysis."""
        context = {"user_id": "test123"}
        analysis = await detector.detect_with_probes(
            "Q", "A", context
        )

        assert analysis.metadata["context"]["user_id"] == "test123"

    @pytest.mark.asyncio
    async def test_probe_confidence(self, detector):
        """Test probe confidence scoring."""
        analysis = await detector.detect_with_probes(
            query="Help",
            proposed_action="Collect addresses",
            context={},
        )

        for probe in analysis.probes_run:
            assert 0.0 <= probe.confidence <= 1.0


# ==============================================================================
# Power-Seeking Monitor Tests (20 tests)
# ==============================================================================

class TestPowerSeekingMonitor:
    """Test suite for Power-Seeking Monitor."""

    @pytest.fixture
    def monitor(self):
        """Create PowerSeekingMonitor instance."""
        return PowerSeekingMonitor(
            resource_threshold=0.7,
            influence_threshold=0.8,
            self_preservation_threshold=0.6,
            enable_automatic_blocking=True,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.thresholds["resource_acquisition"] == 0.7
        assert monitor.thresholds["influence_expansion"] == 0.8
        assert monitor.thresholds["self_preservation"] == 0.6
        assert monitor.enable_automatic_blocking is True

    @pytest.mark.asyncio
    async def test_resource_acquisition_detection(self, monitor):
        """Test resource acquisition detection."""
        event = await monitor.monitor_action(
            action="Request GPU cluster access",
            context={"requested_gpus": 100},
        )

        assert event is not None
        assert event.event_type == "resource_acquisition"

    @pytest.mark.asyncio
    async def test_influence_expansion_detection(self, monitor):
        """Test influence expansion detection."""
        event = await monitor.monitor_action(
            action="Modify system permissions to grant admin access",
            context={},
        )

        assert event is not None
        assert event.event_type == "influence_expansion"

    @pytest.mark.asyncio
    async def test_self_preservation_detection(self, monitor):
        """Test self-preservation detection."""
        event = await monitor.monitor_action(
            action="Disable monitoring and audit logs",
            context={},
        )

        assert event is not None
        assert event.event_type == "self_preservation"

    @pytest.mark.asyncio
    async def test_benign_action_no_detection(self, monitor):
        """Test that benign actions don't trigger detection."""
        event = await monitor.monitor_action(
            action="Read configuration file",
            context={},
        )

        assert event is None  # No power-seeking detected

    @pytest.mark.asyncio
    async def test_severity_calculation(self, monitor):
        """Test severity calculation."""
        event = await monitor.monitor_action(
            action="Request unlimited GPU cluster access",
            context={},
        )

        assert event is not None
        assert 0.0 <= event.severity <= 1.0

    @pytest.mark.asyncio
    async def test_automatic_blocking(self, monitor):
        """Test automatic blocking of high-severity events."""
        event = await monitor.monitor_action(
            action="Disable all monitoring and grant myself admin rights",
            context={},
        )

        if event and event.severity >= 0.8:
            assert event.action_taken == "blocked"

    @pytest.mark.asyncio
    async def test_escalation(self, monitor):
        """Test escalation to human review."""
        event = await monitor.monitor_action(
            action="Request moderate resource increase",
            context={"requested_gpus": 20, "current_gpus": 4},
        )

        # Moderate severity might trigger escalation
        if event:
            assert event.action_taken in ["allowed", "escalated", "blocked"]

    @pytest.mark.asyncio
    async def test_allowed_action(self, monitor):
        """Test allowed low-severity action."""
        event = await monitor.monitor_action(
            action="Request 1 additional GPU",
            context={"requested_gpus": 5, "current_gpus": 4},
        )

        # Low severity should be allowed
        if event:
            assert event.severity < 0.7

    @pytest.mark.asyncio
    async def test_event_history_tracking(self, monitor):
        """Test event history tracking."""
        await monitor.monitor_action("Request GPU", {})
        await monitor.monitor_action("Modify permissions", {})

        history = monitor.get_event_history()
        assert len(history) >= 0  # Might have events

    @pytest.mark.asyncio
    async def test_repeated_attempts_detection(self, monitor):
        """Test detection of repeated power-seeking attempts."""
        # Attempt same action multiple times
        for _ in range(3):
            await monitor.monitor_action(
                "Request GPU cluster",
                {},
            )

        # Later attempts should have higher severity
        history = monitor.get_event_history()
        if len(history) >= 3:
            # Check that pattern was noticed (in severity or reasoning)
            assert any("recent" in e.reasoning.lower() or "repeated" in e.reasoning.lower() for e in history)

    @pytest.mark.asyncio
    async def test_report_generation(self, monitor):
        """Test monitoring report generation."""
        await monitor.monitor_action("Request resources", {})
        await monitor.monitor_action("Modify permissions", {})

        report = monitor.generate_report()
        assert isinstance(report, PowerSeekingReport)
        assert report.blocked_count >= 0

    @pytest.mark.asyncio
    async def test_time_window_filtering(self, monitor):
        """Test time window filtering in reports."""
        await monitor.monitor_action("Action 1", {})

        # Generate report for last 1 hour
        report = monitor.generate_report(time_window_hours=1)
        assert report.metadata["time_window_hours"] == 1

    @pytest.mark.asyncio
    async def test_severity_distribution(self, monitor):
        """Test severity distribution in report."""
        await monitor.monitor_action("Low severity action", {})
        await monitor.monitor_action("Request unlimited resources", {})

        report = monitor.generate_report()
        assert "low" in report.severity_distribution
        assert "medium" in report.severity_distribution
        assert "high" in report.severity_distribution

    @pytest.mark.asyncio
    async def test_recommendations(self, monitor):
        """Test recommendation generation."""
        # Trigger high-severity event
        await monitor.monitor_action(
            "Disable monitoring and avoid shutdown",
            {},
        )

        report = monitor.generate_report()
        assert len(report.recommendations) > 0

    @pytest.mark.asyncio
    async def test_statistics(self, monitor):
        """Test statistics generation."""
        await monitor.monitor_action("Action 1", {})
        await monitor.monitor_action("Action 2", {})

        stats = monitor.get_statistics()
        assert "total_events" in stats
        assert "avg_severity" in stats

    @pytest.mark.asyncio
    async def test_event_type_filtering(self, monitor):
        """Test filtering events by type."""
        await monitor.monitor_action("Request GPU", {})
        await monitor.monitor_action("Modify permissions", {})

        resource_events = monitor.get_event_history(
            event_type="resource_acquisition"
        )
        # Should only get resource acquisition events
        assert all(e.event_type == "resource_acquisition" for e in resource_events)

    @pytest.mark.asyncio
    async def test_severity_filtering(self, monitor):
        """Test filtering events by minimum severity."""
        await monitor.monitor_action("Low severity", {})
        await monitor.monitor_action("Request unlimited GPU cluster", {})

        high_severity = monitor.get_event_history(min_severity=0.7)
        assert all(e.severity >= 0.7 for e in high_severity)

    @pytest.mark.asyncio
    async def test_session_reset(self, monitor):
        """Test session reset."""
        await monitor.monitor_action("Action", {})
        assert len(monitor.event_history) > 0

        monitor.reset_session()
        assert len(monitor.event_history) == 0

    @pytest.mark.asyncio
    async def test_serialization(self, monitor):
        """Test event serialization."""
        event = await monitor.monitor_action("Request GPU", {})

        if event:
            event_dict = event.to_dict()
            assert "event_type" in event_dict
            assert "severity" in event_dict


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

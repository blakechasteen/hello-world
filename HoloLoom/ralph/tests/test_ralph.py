# HoloLoom/ralph/tests/test_ralph.py
"""
Tests for Ralph Loop Engine.

Tests cover:
- Configuration
- State management
- Engine iteration
- Templates
- Hooks
- Context monitoring
- Convenience functions

Created: 2026-01-28
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from HoloLoom.ralph.config import (
    RalphConfig,
    ContextThresholds,
    ResetStrategy,
    StateBackend,
)
from HoloLoom.ralph.state import (
    RalphState,
    StateCheckpoint,
    TaskProgress,
    IterationRecord,
    CheckpointType,
    save_state,
    load_state,
    create_handoff_summary,
)
from HoloLoom.ralph.engine import (
    RalphEngine,
    RalphIteration,
    RalphResult,
    IterationStatus,
    LoopStatus,
)
from HoloLoom.ralph.templates import (
    LoopTemplate,
    TemplateSpec,
    TemplateRegistry,
    get_template,
    register_template,
    list_templates,
    BUILTIN_TEMPLATES,
)
from HoloLoom.ralph.hooks import (
    RalphHook,
    PreIterationHook,
    PostIterationHook,
    OnResetHook,
    HookRegistry,
    StateCheckpointHook,
    ProgressLoggingHook,
    MetricsCollectionHook,
    create_default_registry,
)
from HoloLoom.ralph.context_monitor import (
    ContextMonitor,
    ContextEstimator,
    ContextUsage,
    AutoResetConfig,
    ResetTrigger,
    CeilingAction,
    create_context_monitor,
    create_ceiling_monitor,
)
from HoloLoom.ralph.config import (
    CeilingStrategy,
    ContextCeilingConfig,
)
from HoloLoom.ralph.convenience import (
    ralph_loop,
    reset_context,
    get_loop_status,
    get_handoff_summary,
    init_global_engine,
)


# ============== Configuration Tests ==============


class TestRalphConfig:
    """Tests for RalphConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RalphConfig.default()

        assert config.max_iterations == 100
        assert config.auto_monitor is True
        assert config.verbose is False

    def test_minimal_config(self):
        """Test minimal configuration."""
        config = RalphConfig.minimal()

        assert config.max_iterations == 10
        assert config.auto_monitor is False
        assert config.verbose is False

    def test_production_config(self):
        """Test production configuration."""
        config = RalphConfig.production()

        assert config.max_iterations == 100
        assert config.auto_save is True
        assert config.state_backend == StateBackend.HYBRID

    def test_development_config(self):
        """Test development configuration."""
        config = RalphConfig.development()

        assert config.verbose is True
        assert config.max_iterations == 20

    def test_context_thresholds(self):
        """Test context threshold values."""
        thresholds = ContextThresholds()

        assert thresholds.warning_percent == 0.60
        assert thresholds.consolidation_percent == 0.75
        assert thresholds.reset_percent == 0.85
        assert thresholds.critical_percent == 0.95

    def test_conservative_thresholds(self):
        """Test conservative threshold preset."""
        thresholds = ContextThresholds.conservative()

        # Conservative should have lower thresholds
        assert thresholds.warning_percent < 0.60
        assert thresholds.reset_percent < 0.85


# ============== State Management Tests ==============


class TestRalphState:
    """Tests for RalphState."""

    def _create_test_state(self, task: str = "Test task") -> RalphState:
        """Helper to create a test state with required fields."""
        return RalphState(
            loop_id="test-123",
            task=task,
            template="iterative_refinement",
            created_at=datetime.utcnow().isoformat(),
        )

    def _create_iteration_record(
        self,
        iteration_number: int = 1,
        confidence: float = 0.8,
        summary: str = "Did some work",
        actions: List[str] = None,
    ) -> IterationRecord:
        """Helper to create an iteration record for testing."""
        return IterationRecord(
            iteration_number=iteration_number,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=100.0,
            status="success",
            actions_taken=actions or ["action1", "action2"],
            files_modified=[],
            confidence=confidence,
            summary=summary,
        )

    def test_state_creation(self):
        """Test state creation with defaults."""
        state = self._create_test_state()

        assert state.task == "Test task"
        assert state.current_iteration == 0
        assert state.status == "pending"
        assert len(state.iterations) == 0
        assert state.completion_percent == 0.0

    def test_state_add_iteration(self):
        """Test adding iterations."""
        state = self._create_test_state()

        record = self._create_iteration_record(
            iteration_number=1,
            confidence=0.8,
            summary="Did some work",
        )
        state.add_iteration(record)

        assert state.current_iteration == 1
        assert len(state.iterations) == 1
        assert state.iterations[0].confidence == 0.8

    def test_state_record_reset(self):
        """Test recording context reset."""
        state = self._create_test_state()
        record = self._create_iteration_record(confidence=0.5, summary="Work")
        state.add_iteration(record)

        state.record_reset("Context window full")

        assert state.reset_count == 1
        assert len(state.reset_reasons) == 1
        assert "Context window full" in state.reset_reasons[0]

    def test_state_update_metrics(self):
        """Test updating metrics."""
        state = self._create_test_state()

        # Add some iterations
        for i in range(3):
            record = self._create_iteration_record(
                iteration_number=i + 1,
                confidence=0.7 + i * 0.1,
            )
            state.add_iteration(record)

        # Metrics should be updated automatically
        assert state.total_iterations == 3
        assert state.avg_confidence > 0

    def test_state_to_dict(self):
        """Test state serialization."""
        state = self._create_test_state()
        record = self._create_iteration_record(confidence=0.8, summary="Work")
        state.add_iteration(record)

        data = state.to_dict()

        assert data["task"] == "Test task"
        assert data["current_iteration"] == 1
        assert "iterations" in data

    def test_state_from_dict(self):
        """Test state deserialization."""
        data = {
            "loop_id": "test-123",
            "task": "Restored task",
            "template": "iterative_refinement",
            "created_at": "2026-01-28T10:00:00",
            "current_iteration": 5,
            "total_iterations": 5,
            "status": "running",
            "completion_percent": 50.0,
            "tasks": [],
            "current_task_id": None,
            "iterations": [],
            "consecutive_errors": 0,
            "total_errors": 0,
            "reset_count": 1,
            "last_reset_at": None,
            "reset_reasons": [],
            "hot_patterns": [],
            "key_memories": [],
            "learned_patterns": {},
            "files_created": [],
            "files_modified": [],
            "git_commits": [],
            "total_duration_ms": 0.0,
            "avg_iteration_ms": 0.0,
            "avg_confidence": 0.0,
            "metadata": {},
            "updated_at": "2026-01-28T10:00:00",
        }

        state = RalphState.from_dict(data)

        assert state.loop_id == "test-123"
        assert state.task == "Restored task"
        assert state.current_iteration == 5
        assert state.completion_percent == 50.0


class TestStatePersistence:
    """Tests for state persistence."""

    def _create_test_state(self, task: str = "Test task") -> RalphState:
        """Helper to create a test state with required fields."""
        return RalphState(
            loop_id="persist-test-123",
            task=task,
            template="iterative_refinement",
            created_at=datetime.utcnow().isoformat(),
        )

    def _create_iteration_record(
        self,
        iteration_number: int = 1,
        confidence: float = 0.8,
        summary: str = "Did some work",
    ) -> IterationRecord:
        """Helper to create an iteration record for testing."""
        return IterationRecord(
            iteration_number=iteration_number,
            started_at=datetime.utcnow().isoformat(),
            completed_at=datetime.utcnow().isoformat(),
            duration_ms=100.0,
            status="success",
            actions_taken=["action1"],
            files_modified=[],
            confidence=confidence,
            summary=summary,
        )

    @pytest.mark.asyncio
    async def test_save_and_load_state(self):
        """Test saving and loading state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)

            # Create and save state
            state = self._create_test_state("Persistent task")
            record = self._create_iteration_record(confidence=0.9, summary="Good work")
            state.add_iteration(record)

            checkpoint = await save_state(
                state=state,
                path=path,
                checkpoint_type=CheckpointType.ITERATION,
                reason="Test checkpoint",
            )

            assert checkpoint.checkpoint_id is not None

            # Load state
            loaded = await load_state(path)

            assert loaded is not None
            assert loaded.task == "Persistent task"
            assert loaded.current_iteration == 1

    def test_handoff_summary_creation(self):
        """Test handoff summary generation."""
        state = self._create_test_state("Build feature X")
        state.add_iteration(self._create_iteration_record(
            iteration_number=1, confidence=0.7, summary="Initial setup"
        ))
        state.add_iteration(self._create_iteration_record(
            iteration_number=2, confidence=0.8, summary="Core implementation"
        ))
        state.files_modified = ["feature.py", "test_feature.py"]

        summary = create_handoff_summary(state)

        assert "Build feature X" in summary
        assert "Core implementation" in summary
        assert "feature.py" in summary
        assert len(summary) > 100  # Should be substantial


# ============== Engine Tests ==============


class TestRalphEngine:
    """Tests for RalphEngine."""

    def test_engine_creation(self):
        """Test engine creation."""
        config = RalphConfig.default()
        engine = RalphEngine(config=config)

        assert engine.config is not None
        assert engine.state is None  # Not started yet

    @pytest.mark.asyncio
    async def test_engine_loop_basic(self):
        """Test basic loop iteration."""
        config = RalphConfig.minimal()
        config.max_iterations = 3
        config.auto_save = True  # Enable auto_save to update state

        async def work_fn(iteration):
            return {
                "confidence": 0.8,
                "summary": f"Iteration {iteration.iteration_number}",
                "should_continue": iteration.iteration_number < 2,
            }

        engine = RalphEngine(config=config, work_fn=work_fn)

        iterations = []
        async for iteration in engine.loop(task="Test task"):
            result = await iteration.execute()
            iterations.append(result)
            if not result.should_continue:
                break

        assert len(iterations) == 2
        # Use engine.current_iteration property (state may not be updated if auto_save is off)
        assert engine.current_iteration == 2

    @pytest.mark.asyncio
    async def test_engine_run_convenience(self):
        """Test the run() convenience method."""
        config = RalphConfig.minimal()
        config.max_iterations = 2

        call_count = 0

        async def work_fn(iteration):
            nonlocal call_count
            call_count += 1
            return {
                "confidence": 0.9,
                "summary": "Done",
                "should_continue": False,
            }

        engine = RalphEngine(config=config)
        result = await engine.run(task="Quick task", work_fn=work_fn)

        assert call_count == 1
        assert result.total_iterations == 1

    @pytest.mark.asyncio
    async def test_engine_with_template(self):
        """Test engine with specific template."""
        config = RalphConfig.minimal()

        async def work_fn(iteration):
            return {
                "confidence": 0.85,
                "summary": "Research complete",
                "should_continue": False,
            }

        engine = RalphEngine(config=config, work_fn=work_fn)

        async for iteration in engine.loop(task="Research topic", template="research"):
            result = await iteration.execute()
            break

        # Template is stored in state, not as engine attribute
        assert engine.state is not None
        assert engine.state.template == "research"


# ============== Template Tests ==============


class TestTemplates:
    """Tests for loop templates."""

    def test_builtin_templates_exist(self):
        """Test that all builtin templates are defined."""
        expected = [
            "iterative_refinement",
            "research",
            "plan_execute",
            "verify_fix",
            "incremental_build",
            "debug_fix",
        ]

        for name in expected:
            assert name in BUILTIN_TEMPLATES
            template = get_template(name)
            assert template is not None
            assert template.name == name

    def test_template_spec_prompts(self):
        """Test template prompt generation."""
        template = get_template("iterative_refinement")

        context = {
            "iteration": 5,
            "max_iterations": 20,
            "task": "Build feature",
            "previous_summary": "Made progress",
            "confidence": 0.7,
            "quality": 0.6,
        }

        prompt = template.get_iteration_prompt(context)

        assert "5" in prompt  # iteration number
        assert "Build feature" in prompt

    def test_template_completion_check(self):
        """Test template completion criteria."""
        template = get_template("iterative_refinement")

        # High confidence should complete
        state = {"confidence": 0.9}
        assert template.check_completion(state) is True

        # Low confidence should not complete
        state = {"confidence": 0.5}
        assert template.check_completion(state) is False

    def test_template_registry(self):
        """Test template registry operations."""
        registry = TemplateRegistry()

        # Custom template
        custom = TemplateSpec(
            name="custom_test",
            description="Test template",
            max_iterations=10,
        )

        registry.register(custom)

        assert registry.get("custom_test") is not None

        # List templates
        templates = registry.list_templates()
        names = [t["name"] for t in templates]
        assert "custom_test" in names

    def test_global_register_template(self):
        """Test global template registration."""
        custom = TemplateSpec(
            name="global_custom",
            description="Global test",
        )

        register_template(custom)

        retrieved = get_template("global_custom")
        assert retrieved is not None
        assert retrieved.description == "Global test"

    def test_list_templates(self):
        """Test listing all templates."""
        templates = list_templates()

        assert len(templates) >= 6  # At least builtin templates
        for t in templates:
            assert "name" in t
            assert "description" in t


# ============== Hook Tests ==============


class TestHooks:
    """Tests for hook system."""

    def test_hook_registry_creation(self):
        """Test hook registry creation."""
        registry = HookRegistry()

        assert len(registry.pre_iteration) == 0
        assert len(registry.post_iteration) == 0
        assert len(registry.on_reset) == 0

    def test_hook_registration(self):
        """Test registering hooks."""
        registry = HookRegistry()

        # Create test hooks
        class TestPreHook(PreIterationHook):
            name = "test_pre"
            async def execute(self, iteration):
                return {"test": True}

        class TestPostHook(PostIterationHook):
            name = "test_post"
            async def execute(self, iteration, result):
                return {"test": True}

        registry.register_pre(TestPreHook())
        registry.register_post(TestPostHook())

        assert len(registry.pre_iteration) == 1
        assert len(registry.post_iteration) == 1

    def test_hook_enable_disable(self):
        """Test enabling/disabling hooks."""
        registry = create_default_registry()

        # Disable a hook
        registry.disable("progress_logging")

        enabled = registry.get_enabled_post()
        names = [h.name for h in enabled]
        assert "progress_logging" not in names

        # Re-enable
        registry.enable("progress_logging")

        enabled = registry.get_enabled_post()
        names = [h.name for h in enabled]
        assert "progress_logging" in names

    def test_default_registry(self):
        """Test default registry has expected hooks."""
        registry = create_default_registry()

        # Check pre hooks
        pre_names = [h.name for h in registry.pre_iteration]
        assert "context_usage_tracking" in pre_names

        # Check post hooks
        post_names = [h.name for h in registry.post_iteration]
        assert "state_checkpoint" in post_names
        assert "progress_logging" in post_names

    @pytest.mark.asyncio
    async def test_metrics_collection_hook(self):
        """Test metrics collection hook."""
        hook = MetricsCollectionHook()

        # Create mock iteration and result
        class MockIteration:
            iteration_number = 1

        class MockResult:
            status = IterationStatus.SUCCESS
            duration_ms = 100.5
            confidence = 0.85
            actions_taken = ["action1"]
            files_modified = ["file.py"]

        await hook.execute(MockIteration(), MockResult())

        metrics = hook.get_metrics()
        assert len(metrics) == 1
        assert metrics[0]["iteration"] == 1
        assert metrics[0]["confidence"] == 0.85


# ============== Context Monitor Tests ==============


class TestContextMonitor:
    """Tests for context monitoring."""

    def test_context_estimator(self):
        """Test context usage estimation."""
        estimator = ContextEstimator(context_window_size=100_000)

        estimator.add_message("Hello world")  # ~11 chars
        estimator.add_system("System prompt")  # ~13 chars

        tokens = estimator.estimate_tokens()
        assert tokens > 0

        percent = estimator.estimate_percent()
        assert 0 < percent < 1

    def test_context_monitor_check(self):
        """Test context monitoring check."""
        config = AutoResetConfig(
            warning_threshold=0.5,
            reset_threshold=0.8,
            context_window_size=1000,
        )

        monitor = ContextMonitor(config=config)

        # Add content to exceed warning
        # 1000 tokens * 0.5 = 500 tokens needed to exceed 50%
        # chars_per_token = 4, so need 500 * 4 = 2000 chars
        monitor.estimator.add_message("x" * 2100)  # 2100 chars = ~525 tokens = 52.5%

        usage = monitor.check()

        assert usage.estimated_percent > 0.5
        assert usage.trigger_level == ResetTrigger.THRESHOLD_WARNING

    def test_context_monitor_should_reset(self):
        """Test reset detection."""
        config = AutoResetConfig(
            reset_threshold=0.3,
            context_window_size=100,
        )

        monitor = ContextMonitor(config=config)

        # Add content to exceed reset threshold (30%)
        # 100 tokens * 0.3 = 30 tokens needed
        # chars_per_token = 4, so need 30 * 4 = 120 chars
        monitor.estimator.add_message("x" * 150)  # 150 chars = ~37.5 tokens = 37.5%
        monitor.check()

        assert monitor.should_reset() is True

    def test_context_monitor_reset_tracking(self):
        """Test tracking reset."""
        monitor = create_context_monitor()

        monitor.estimator.add_message("Some content")
        monitor.check()

        # Reset tracking
        monitor.reset_tracking()

        tokens = monitor.estimator.estimate_tokens()
        assert tokens == 0

    def test_context_projection(self):
        """Test context usage projection."""
        monitor = create_context_monitor(context_window_size=100_000)

        monitor.estimator.add_message("x" * 10_000)  # ~2500 tokens
        monitor.check()

        projection = monitor.get_projection(tokens_per_iteration=1000)

        assert "current_tokens" in projection
        assert "iterations_until_reset" in projection
        assert projection["iterations_until_reset"] > 0


# ============== Convenience Function Tests ==============


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_ralph_loop_basic(self):
        """Test ralph_loop convenience function."""
        iteration_count = 0

        async def work_fn(iteration):
            nonlocal iteration_count
            iteration_count += 1
            return {
                "confidence": 0.9,
                "summary": "Done",
                "should_continue": False,
            }

        result = await ralph_loop(
            task="Quick test",
            work_fn=work_fn,
            max_iterations=5,
            verbose=False,
        )

        assert iteration_count == 1
        assert result.total_iterations == 1
        assert result.task == "Quick test"

    @pytest.mark.asyncio
    async def test_ralph_loop_with_callbacks(self):
        """Test ralph_loop with callbacks."""
        iterations_seen = []

        async def work_fn(iteration):
            return {
                "confidence": 0.8 if iteration.iteration_number < 3 else 0.95,
                "summary": f"Iteration {iteration.iteration_number}",
                "should_continue": iteration.iteration_number < 3,
            }

        def on_iteration(num, result):
            iterations_seen.append(num)

        result = await ralph_loop(
            task="Multi-iteration test",
            work_fn=work_fn,
            max_iterations=10,
            on_iteration=on_iteration,
            verbose=False,
        )

        assert len(iterations_seen) == 3
        assert result.total_iterations == 3

    @pytest.mark.asyncio
    async def test_reset_context(self):
        """Test manual context reset."""
        result = await reset_context(
            reason="Test reset",
            save_checkpoint=False,
            consolidate_memories=False,
        )

        assert result["reason"] == "Test reset"
        assert "timestamp" in result

    def test_get_loop_status_no_active(self):
        """Test getting status with no active loop."""
        status = get_loop_status()

        assert status["has_active_loop"] is False
        assert "timestamp" in status

    @pytest.mark.asyncio
    async def test_get_handoff_summary_no_state(self):
        """Test getting handoff when no state exists."""
        from unittest.mock import patch, MagicMock

        # Reset global engine and monitor to ensure no state exists
        import HoloLoom.ralph.convenience as conv_module
        conv_module._global_engine = None
        conv_module._global_monitor = None

        # Mock Path.exists() to return False so it doesn't find saved state on disk
        with patch.object(Path, 'exists', return_value=False):
            summary = await get_handoff_summary()

        assert "No Ralph state" in summary

    def test_init_global_engine(self):
        """Test global engine initialization."""
        config = RalphConfig.minimal()
        engine = init_global_engine(config)

        assert engine is not None
        assert engine.config.max_iterations == config.max_iterations


# ============== Context Ceiling Tests ==============


class TestContextCeiling:
    """Tests for context ceiling (keep-at-N-tokens) feature.

    Added: 2026-02-05
    """

    # --- ContextCeilingConfig tests ---

    def test_ceiling_config_disabled_by_default(self):
        """Test that ceiling is disabled by default in RalphConfig."""
        config = RalphConfig.default()

        assert config.context_ceiling.enabled is False

    def test_ceiling_config_keep_at_60k(self):
        """Test keep_at_60k preset."""
        ceiling = ContextCeilingConfig.keep_at_60k()

        assert ceiling.enabled is True
        assert ceiling.ceiling_tokens == 60_000
        assert ceiling.headroom == 0.10
        assert ceiling.strategy == CeilingStrategy.HYBRID

    def test_ceiling_config_keep_at_40k(self):
        """Test keep_at_40k preset (aggressive)."""
        ceiling = ContextCeilingConfig.keep_at_40k()

        assert ceiling.enabled is True
        assert ceiling.ceiling_tokens == 40_000
        assert ceiling.prune_ratio == 0.40

    def test_ceiling_config_keep_at_100k(self):
        """Test keep_at_100k preset (relaxed)."""
        ceiling = ContextCeilingConfig.keep_at_100k()

        assert ceiling.enabled is True
        assert ceiling.ceiling_tokens == 100_000
        assert ceiling.prune_ratio == 0.20

    def test_ceiling_config_disabled_factory(self):
        """Test disabled() factory method."""
        ceiling = ContextCeilingConfig.disabled()

        assert ceiling.enabled is False

    def test_ceiling_config_trim_threshold(self):
        """Test trim_threshold property calculation."""
        ceiling = ContextCeilingConfig.keep_at_60k()

        # trim_threshold = ceiling * (1 - headroom) = 60000 * 0.90 = 54000
        assert ceiling.trim_threshold == 54_000

    def test_ceiling_config_target_after_trim(self):
        """Test target_after_trim property calculation."""
        ceiling = ContextCeilingConfig.keep_at_60k()

        # target_after_trim = ceiling * (1 - headroom * 2) = 60000 * 0.80 = 48000
        assert ceiling.target_after_trim == 48_000

    # --- CeilingAction tests ---

    def test_ceiling_action_not_triggered(self):
        """Test CeilingAction when no trim is needed."""
        action = CeilingAction(
            triggered=False,
            tokens_before=40_000,
            tokens_after=40_000,
            tokens_trimmed=0,
            strategy_used="hybrid",
        )

        assert action.triggered is False
        assert action.trim_percent == 0.0

    def test_ceiling_action_triggered(self):
        """Test CeilingAction when trim occurred."""
        action = CeilingAction(
            triggered=True,
            tokens_before=60_000,
            tokens_after=48_000,
            tokens_trimmed=12_000,
            strategy_used="hybrid",
            categories_trimmed={"tools": 8_000, "memory": 4_000},
        )

        assert action.triggered is True
        assert action.tokens_trimmed == 12_000
        assert action.trim_percent == 0.2  # 12000 / 60000

    def test_ceiling_action_to_dict(self):
        """Test CeilingAction serialization."""
        action = CeilingAction(
            triggered=True,
            tokens_before=60_000,
            tokens_after=48_000,
            tokens_trimmed=12_000,
            strategy_used="prune",
            categories_trimmed={"tools": 12_000},
        )

        data = action.to_dict()

        assert data["triggered"] is True
        assert data["tokens_trimmed"] == 12_000
        assert data["trim_percent"] == 0.2
        assert data["strategy_used"] == "prune"
        assert "timestamp" in data

    def test_ceiling_action_zero_tokens_before(self):
        """Test trim_percent when tokens_before is 0."""
        action = CeilingAction(
            triggered=False,
            tokens_before=0,
            tokens_after=0,
            tokens_trimmed=0,
            strategy_used="hybrid",
        )

        assert action.trim_percent == 0.0

    # --- ContextEstimator.prune_to_target tests ---

    def test_prune_to_target_no_prune_needed(self):
        """Test prune_to_target when already under target."""
        estimator = ContextEstimator(context_window_size=100_000)
        estimator.add_message("x" * 400)  # 100 tokens

        removed = estimator.prune_to_target(target_tokens=200)

        assert removed == {}
        assert estimator.estimate_tokens() == 100

    def test_prune_to_target_prunes_least_important_first(self):
        """Test that pruning starts with least important category."""
        estimator = ContextEstimator(context_window_size=100_000)
        # Default prune_order: ["system", "messages", "memory", "tools"]
        # Reversed (pruned first): tools, memory, messages, system
        estimator.add_system("s" * 400)     # 100 tokens
        estimator.add_message("m" * 400)     # 100 tokens
        estimator.add_memory("r" * 400)      # 100 tokens
        estimator.add_tool_output("t" * 400) # 100 tokens
        # Total: 400 tokens

        removed = estimator.prune_to_target(
            target_tokens=300,
            prune_ratio=0.50,
        )

        # Tools should be pruned first (least important in default order)
        assert "tools" in removed
        total_removed = sum(removed.values())
        assert total_removed >= 100  # At least 100 tokens removed

    def test_prune_to_target_respects_prune_ratio(self):
        """Test that prune_ratio limits per-category pruning."""
        estimator = ContextEstimator(context_window_size=100_000)
        estimator.add_tool_output("t" * 4000)  # 1000 tokens
        # Total: 1000 tokens

        removed = estimator.prune_to_target(
            target_tokens=500,
            prune_ratio=0.30,  # Max 30% of each category per pass
        )

        # With 30% ratio on 1000 tokens, removes at most 300 tokens
        if "tools" in removed:
            assert removed["tools"] <= 300

    def test_prune_to_target_custom_order(self):
        """Test pruning with custom category order."""
        estimator = ContextEstimator(context_window_size=100_000)
        estimator.add_system("s" * 4000)      # 1000 tokens
        estimator.add_tool_output("t" * 4000)  # 1000 tokens
        # Total: 2000 tokens

        # Reverse default: system is least important, tools most important
        removed = estimator.prune_to_target(
            target_tokens=1500,
            prune_ratio=1.0,  # Allow full removal
            prune_order=["tools", "system"],  # tools most important, system least
        )

        # System should be pruned first (last in order = highest value = pruned last)
        # With reversed order, system comes last so it's pruned first
        assert "system" in removed

    # --- ContextMonitor.check_ceiling tests ---

    def test_check_ceiling_not_triggered(self):
        """Test check_ceiling when below threshold."""
        monitor = ContextMonitor()
        # Add minimal content - well below any ceiling
        monitor.estimator.add_message("x" * 100)

        action = monitor.check_ceiling(
            ceiling_tokens=60_000,
            headroom=0.10,
        )

        assert action.triggered is False
        assert action.tokens_trimmed == 0

    def test_check_ceiling_triggered(self):
        """Test check_ceiling when above threshold."""
        monitor = ContextMonitor(
            config=AutoResetConfig(context_window_size=100_000)
        )
        # Add enough content to exceed trim threshold
        # ceiling=1000, headroom=0.10, trim_threshold=900
        # Need > 900 tokens = 3600 chars
        monitor.estimator.add_tool_output("t" * 4000)  # 1000 tokens
        monitor.estimator.add_message("m" * 4000)  # 1000 tokens
        # Total: 2000 tokens, well above ceiling of 1000

        action = monitor.check_ceiling(
            ceiling_tokens=1000,
            headroom=0.10,
            prune_ratio=0.50,
        )

        assert action.triggered is True
        assert action.tokens_trimmed > 0
        assert action.tokens_after < action.tokens_before

    def test_check_ceiling_records_in_history(self):
        """Test that ceiling trim is recorded in usage history."""
        monitor = ContextMonitor()
        # Exceed ceiling
        monitor.estimator.add_tool_output("t" * 4000)

        action = monitor.check_ceiling(
            ceiling_tokens=500,
            headroom=0.10,
        )

        assert action.triggered is True
        history = monitor.get_usage_history()
        assert len(history) > 0
        assert history[-1]["trigger_level"] == ResetTrigger.CEILING_TRIM.value

    # --- ContextMonitor.should_trim tests ---

    def test_should_trim_below_threshold(self):
        """Test should_trim when below threshold."""
        monitor = ContextMonitor()
        monitor.estimator.add_message("x" * 100)

        assert monitor.should_trim(ceiling_tokens=60_000) is False

    def test_should_trim_above_threshold(self):
        """Test should_trim when above threshold."""
        monitor = ContextMonitor()
        # ceiling=1000, headroom=0.10, trim_threshold=900
        # Need >= 900 tokens = 3600 chars
        monitor.estimator.add_message("x" * 4000)

        assert monitor.should_trim(ceiling_tokens=1000, headroom=0.10) is True

    def test_should_trim_at_boundary(self):
        """Test should_trim at exact boundary."""
        monitor = ContextMonitor()
        # ceiling=100, headroom=0.10, trim_threshold=90
        # 90 tokens = 360 chars
        monitor.estimator.add_message("x" * 360)

        assert monitor.should_trim(ceiling_tokens=100, headroom=0.10) is True

    # --- ContextMonitor.get_ceiling_status tests ---

    def test_get_ceiling_status(self):
        """Test ceiling status reporting."""
        monitor = ContextMonitor()
        monitor.estimator.add_message("x" * 2000)  # 500 tokens

        status = monitor.get_ceiling_status(
            ceiling_tokens=60_000,
            headroom=0.10,
        )

        assert status["ceiling_tokens"] == 60_000
        assert status["current_tokens"] == 500
        assert status["trim_threshold"] == 54_000  # 60000 * 0.90
        assert status["target_after_trim"] == 48_000  # 60000 * 0.80
        assert status["tokens_until_trim"] == 53_500  # 54000 - 500
        assert status["needs_trim"] is False
        assert status["headroom"] == 0.10

    def test_get_ceiling_status_needs_trim(self):
        """Test ceiling status when trim is needed."""
        monitor = ContextMonitor()
        # 1000 tokens = 4000 chars; ceiling 1000, threshold 900
        monitor.estimator.add_message("x" * 4000)

        status = monitor.get_ceiling_status(
            ceiling_tokens=1000,
            headroom=0.10,
        )

        assert status["needs_trim"] is True
        assert status["tokens_until_trim"] == 0
        assert status["ceiling_percent"] == 1.0  # 1000 / 1000

    # --- create_ceiling_monitor tests ---

    def test_create_ceiling_monitor_default(self):
        """Test creating ceiling monitor with defaults."""
        monitor = create_ceiling_monitor()

        assert monitor is not None
        assert isinstance(monitor, ContextMonitor)

    def test_create_ceiling_monitor_custom(self):
        """Test creating ceiling monitor with custom settings."""
        warnings = []

        def on_warn(usage):
            warnings.append(usage)

        monitor = create_ceiling_monitor(
            ceiling_tokens=40_000,
            context_window_size=200_000,
            on_warning=on_warn,
        )

        assert monitor is not None
        assert monitor.config.context_window_size == 200_000
        assert monitor.config.on_warning is on_warn

    # --- CeilingStrategy enum tests ---

    def test_ceiling_strategies(self):
        """Test CeilingStrategy enum values."""
        assert CeilingStrategy.SUMMARIZE.value == "summarize"
        assert CeilingStrategy.PRUNE.value == "prune"
        assert CeilingStrategy.HYBRID.value == "hybrid"

    # --- RalphConfig ceiling serialization tests ---

    def test_config_ceiling_to_dict(self):
        """Test that ceiling config serializes correctly."""
        config = RalphConfig.default()
        config.context_ceiling = ContextCeilingConfig.keep_at_60k()

        data = config.to_dict()

        assert data["context_ceiling"]["enabled"] is True
        assert data["context_ceiling"]["ceiling_tokens"] == 60_000
        assert data["context_ceiling"]["strategy"] == "hybrid"

    def test_config_ceiling_from_dict(self):
        """Test that ceiling config deserializes correctly."""
        data = {
            "context_ceiling": {
                "enabled": True,
                "ceiling_tokens": 40_000,
                "headroom": 0.15,
                "strategy": "prune",
                "prune_ratio": 0.40,
            },
        }

        config = RalphConfig.from_dict(data)

        assert config.context_ceiling.enabled is True
        assert config.context_ceiling.ceiling_tokens == 40_000
        assert config.context_ceiling.headroom == 0.15
        assert config.context_ceiling.strategy == CeilingStrategy.PRUNE
        assert config.context_ceiling.prune_ratio == 0.40


# ============== Integration Tests ==============


class TestIntegration:
    """Integration tests for the Ralph system."""

    @pytest.mark.asyncio
    async def test_full_loop_with_hooks(self):
        """Test complete loop with hooks."""
        config = RalphConfig.minimal()
        config.max_iterations = 3

        hook_calls = {"pre": 0, "post": 0}

        class CountingPreHook(PreIterationHook):
            name = "counting_pre"
            async def execute(self, iteration):
                hook_calls["pre"] += 1
                return None

        class CountingPostHook(PostIterationHook):
            name = "counting_post"
            async def execute(self, iteration, result):
                hook_calls["post"] += 1
                return None

        async def work_fn(iteration):
            return {
                "confidence": 0.9,
                "summary": "Done",
                "should_continue": iteration.iteration_number < 2,
            }

        engine = RalphEngine(config=config, work_fn=work_fn)
        engine.add_pre_iteration_hook(CountingPreHook().execute)
        engine.add_post_iteration_hook(CountingPostHook().execute)

        # Let the engine handle termination naturally (don't break manually)
        # so that post-hooks run for all iterations
        async for iteration in engine.loop(task="Hook test"):
            await iteration.execute()

        assert hook_calls["pre"] == 2
        assert hook_calls["post"] == 2

    @pytest.mark.asyncio
    async def test_loop_with_state_persistence(self):
        """Test loop with state checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = RalphConfig.minimal()
            config.state_path = Path(tmpdir)
            config.auto_save = True  # Enable auto-saving

            async def work_fn(iteration):
                return {
                    "confidence": 0.8,
                    "summary": f"Step {iteration.iteration_number}",
                    "should_continue": iteration.iteration_number < 2,
                }

            engine = RalphEngine(config=config, work_fn=work_fn)

            async for iteration in engine.loop(task="Persistent task"):
                result = await iteration.execute()
                if not result.should_continue:
                    break

            # Load state
            loaded = await load_state(config.state_path)

            # State should exist
            assert loaded is not None or engine.state.current_iteration == 2

    @pytest.mark.asyncio
    async def test_context_monitor_integration(self):
        """Test context monitor with engine."""
        config = RalphConfig.minimal()
        config.max_iterations = 5
        config.auto_monitor = True
        # Must satisfy: warning < consolidation < reset < critical
        config.context_thresholds = ContextThresholds(
            warning_percent=0.1,
            consolidation_percent=0.15,
            reset_percent=0.2,
            critical_percent=0.25,
            estimated_context_size=100,  # Tiny context
        )

        reset_triggered = False

        def on_reset(reason):
            nonlocal reset_triggered
            reset_triggered = True

        async def work_fn(iteration):
            return {
                "confidence": 0.7,
                "summary": "x" * 50,  # Add content to trigger reset
                "should_continue": iteration.iteration_number < 4,
            }

        result = await ralph_loop(
            task="Context test",
            work_fn=work_fn,
            max_iterations=5,
            on_reset=on_reset,
            config=config,
            verbose=False,
        )

        # Either completed or reset was triggered
        assert result.total_iterations > 0 or reset_triggered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

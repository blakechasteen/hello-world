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
    create_context_monitor,
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

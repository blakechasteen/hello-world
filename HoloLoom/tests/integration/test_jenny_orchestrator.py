"""
Test Jenny WeavingOrchestrator Integration (December 2025 - MVP Week 4)

Tests the integration of JennyRuntime with WeavingOrchestrator for
full lifecycle panel generation during weaving.

Note: Tests are designed to be fast by using mocks instead of full
orchestrator initialization (which requires loading SentenceTransformer models).
"""
import pytest
from unittest.mock import MagicMock, patch

from HoloLoom.config import Config
from HoloLoom.protocols.types import Query, MemoryShard


# ========================================
# Test Fixtures
# ========================================

@pytest.fixture
def test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(id='s1', text='Thompson Sampling is a Bayesian bandit algorithm.', episode='docs'),
        MemoryShard(id='s2', text='It balances exploration and exploitation using probability matching.', episode='docs'),
        MemoryShard(id='s3', text='The algorithm samples from Beta distributions to select actions.', episode='docs'),
    ]


@pytest.fixture
def jenny_enabled_config():
    """Create config with Jenny enabled."""
    cfg = Config.fast()
    cfg.enable_jenny = True
    cfg.jenny_persist_path = "./test_jenny_specs"
    cfg.jenny_default_renderer = "html"
    cfg.jenny_max_panels_per_query = 6
    cfg.jenny_auto_lifecycle = True
    cfg.jenny_cleanup_interval = 60.0
    return cfg


@pytest.fixture
def jenny_disabled_config():
    """Create config with Jenny disabled."""
    cfg = Config.fast()
    cfg.enable_jenny = False
    return cfg


@pytest.fixture
def mock_orchestrator():
    """Create a mock orchestrator with _detect_jenny_panel_type method.

    Note: We must explicitly set class constants on the mock because
    MagicMock doesn't inherit class-level attributes from the spec.
    """
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

    # Import the real method but don't create a full orchestrator
    mock_orch = MagicMock(spec=WeavingOrchestrator)

    # Bind the real method to the mock so we can test it
    mock_orch._detect_jenny_panel_type = WeavingOrchestrator._detect_jenny_panel_type.__get__(
        mock_orch, WeavingOrchestrator
    )

    # Elegance: Copy class constants to mock (MagicMock doesn't inherit these)
    mock_orch.JENNY_CONFIDENCE_THRESHOLD = WeavingOrchestrator.JENNY_CONFIDENCE_THRESHOLD
    mock_orch.JENNY_THREADS_THRESHOLD = WeavingOrchestrator.JENNY_THREADS_THRESHOLD
    mock_orch.JENNY_STAGES_THRESHOLD = WeavingOrchestrator.JENNY_STAGES_THRESHOLD
    mock_orch.JENNY_DURATION_THRESHOLD_MS = WeavingOrchestrator.JENNY_DURATION_THRESHOLD_MS

    # Phase 2.1-2.2: Set MRF attributes to None so heuristic fallback is used
    mock_orch.jenny_mrf_compiler = None
    mock_orch.jenny_learner = None

    # Bind fallback heuristic method
    mock_orch._detect_panel_type_heuristic = WeavingOrchestrator._detect_panel_type_heuristic.__get__(
        mock_orch, WeavingOrchestrator
    )

    return mock_orch


# ========================================
# Unit Tests: Panel Type Detection
# ========================================

class TestPanelTypeDetection:
    """Test the _detect_jenny_panel_type method.

    These are unit tests that test the method in isolation without
    creating a full orchestrator.

    Uses class constants from WeavingOrchestrator for threshold values
    (elegance: tests reference actual thresholds, not magic numbers).
    """

    def test_code_panel_detected(self, mock_orchestrator):
        """Response with code blocks should return CODE panel type."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        # Create mock spacetime with code block
        mock_spacetime = MagicMock()
        mock_spacetime.response = "Here's an example:\n```python\ndef foo(): pass\n```"
        mock_spacetime.confidence = 0.85
        mock_spacetime.trace = None

        panel_type = mock_orchestrator._detect_jenny_panel_type(mock_spacetime)
        assert panel_type == PanelTypeJenny.CODE

    def test_confidence_panel_for_low_confidence(self, mock_orchestrator):
        """Low confidence responses should return CONFIDENCE panel type."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create mock spacetime with low confidence (below threshold)
        mock_spacetime = MagicMock()
        mock_spacetime.response = "I'm not entirely sure about this."
        # Use class constant - 0.01 below threshold to ensure CONFIDENCE
        mock_spacetime.confidence = WeavingOrchestrator.JENNY_CONFIDENCE_THRESHOLD - 0.25
        mock_spacetime.trace = None

        panel_type = mock_orchestrator._detect_jenny_panel_type(mock_spacetime)
        assert panel_type == PanelTypeJenny.CONFIDENCE

    def test_graph_panel_for_multiple_threads(self, mock_orchestrator):
        """Multiple threads activated should return GRAPH panel type."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create mock spacetime with threads > threshold
        mock_trace = MagicMock()
        # Use class constant + 2 to ensure above threshold
        thread_count = WeavingOrchestrator.JENNY_THREADS_THRESHOLD + 2
        mock_trace.threads_activated = [f't{i}' for i in range(thread_count)]
        mock_trace.stage_durations = {}
        mock_trace.duration_ms = 50

        mock_spacetime = MagicMock()
        mock_spacetime.response = "Normal response"
        mock_spacetime.confidence = 0.85
        mock_spacetime.trace = mock_trace

        panel_type = mock_orchestrator._detect_jenny_panel_type(mock_spacetime)
        assert panel_type == PanelTypeJenny.GRAPH

    def test_timeline_panel_for_many_stages(self, mock_orchestrator):
        """Multiple stage durations should return TIMELINE panel type."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create mock spacetime with stages > threshold
        mock_trace = MagicMock()
        mock_trace.threads_activated = []
        # Use class constant + 1 to ensure above threshold
        stage_count = WeavingOrchestrator.JENNY_STAGES_THRESHOLD + 1
        mock_trace.stage_durations = {f'stage{i}': 25 for i in range(stage_count)}
        mock_trace.duration_ms = 50

        mock_spacetime = MagicMock()
        mock_spacetime.response = "Normal response"
        mock_spacetime.confidence = 0.85
        mock_spacetime.trace = mock_trace

        panel_type = mock_orchestrator._detect_jenny_panel_type(mock_spacetime)
        assert panel_type == PanelTypeJenny.TIMELINE

    def test_metric_panel_for_long_duration(self, mock_orchestrator):
        """Long duration responses should return METRIC panel type."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create mock spacetime with duration > threshold
        mock_trace = MagicMock()
        mock_trace.threads_activated = []
        mock_trace.stage_durations = {}
        # Use class constant + 150ms to ensure above threshold
        mock_trace.duration_ms = WeavingOrchestrator.JENNY_DURATION_THRESHOLD_MS + 150

        mock_spacetime = MagicMock()
        mock_spacetime.response = "Normal response"
        mock_spacetime.confidence = 0.85
        mock_spacetime.trace = mock_trace

        panel_type = mock_orchestrator._detect_jenny_panel_type(mock_spacetime)
        assert panel_type == PanelTypeJenny.METRIC

    def test_text_panel_as_default(self, mock_orchestrator):
        """Default case should return TEXT panel type."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Create mock spacetime with all values below thresholds
        mock_trace = MagicMock()
        mock_trace.threads_activated = []  # 0 threads (below threshold)
        mock_trace.stage_durations = {}    # 0 stages (below threshold)
        mock_trace.duration_ms = WeavingOrchestrator.JENNY_DURATION_THRESHOLD_MS - 50  # Below threshold

        mock_spacetime = MagicMock()
        mock_spacetime.response = "Normal text response"  # No code blocks
        mock_spacetime.confidence = WeavingOrchestrator.JENNY_CONFIDENCE_THRESHOLD + 0.1  # Above threshold
        mock_spacetime.trace = mock_trace

        panel_type = mock_orchestrator._detect_jenny_panel_type(mock_spacetime)
        assert panel_type == PanelTypeJenny.TEXT


# ========================================
# Tests: Config Options
# ========================================

class TestJennyConfigOptions:
    """Test Jenny configuration options."""

    def test_jenny_config_defaults(self):
        """Default Jenny config values should be set."""
        cfg = Config.fast()

        # Check defaults exist (may be False by default)
        assert hasattr(cfg, 'enable_jenny')

    def test_jenny_config_renderer_options(self, jenny_enabled_config):
        """Jenny renderer should support html, terminal, json."""
        cfg = jenny_enabled_config

        # Test that renderer config is properly set
        assert cfg.jenny_default_renderer in ['html', 'terminal', 'json']

    def test_jenny_config_persist_path(self, jenny_enabled_config):
        """Jenny persist path should be configurable."""
        cfg = jenny_enabled_config

        assert cfg.jenny_persist_path == "./test_jenny_specs"

    def test_jenny_config_max_panels(self, jenny_enabled_config):
        """Jenny max panels should be configurable."""
        cfg = jenny_enabled_config

        assert cfg.jenny_max_panels_per_query == 6

    def test_jenny_config_auto_lifecycle(self, jenny_enabled_config):
        """Jenny auto lifecycle should be configurable."""
        cfg = jenny_enabled_config

        assert cfg.jenny_auto_lifecycle == True

    def test_jenny_config_cleanup_interval(self, jenny_enabled_config):
        """Jenny cleanup interval should be configurable."""
        cfg = jenny_enabled_config

        assert cfg.jenny_cleanup_interval == 60.0


# ========================================
# Tests: Orchestrator Jenny Attributes
# ========================================

class TestJennyOrchestratorAttributes:
    """Test that orchestrator has correct Jenny-related attributes."""

    def test_orchestrator_has_jenny_attributes(self):
        """WeavingOrchestrator should have Jenny-related attributes."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Check class has expected methods/attributes
        assert hasattr(WeavingOrchestrator, '_detect_jenny_panel_type')
        # Panel generation is inline in weave() method, not a separate method

    def test_orchestrator_init_defaults(self, test_shards):
        """Test orchestrator default values without full initialization."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Check _detect_jenny_panel_type method exists and is callable
        assert callable(getattr(WeavingOrchestrator, '_detect_jenny_panel_type', None))
        # Panel generation happens inline in weave() method around lines 1735-1745


# ========================================
# Tests: Jenny Runtime Module
# ========================================

class TestJennyRuntimeModule:
    """Test JennyRuntime can be imported and instantiated."""

    def test_jenny_runtime_import(self):
        """JennyRuntime should be importable."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime, JennyConfig
        assert JennyRuntime is not None
        assert JennyConfig is not None

    def test_jenny_config_creation(self):
        """JennyConfig should accept configuration options."""
        from HoloLoom.visualization.jenny_runtime import JennyConfig

        config = JennyConfig(
            default_renderer="html",
            ledger_persist_path="./test_specs",
            cleanup_interval_seconds=30.0
        )

        assert config.default_renderer == "html"
        assert config.ledger_persist_path == "./test_specs"
        assert config.cleanup_interval_seconds == 30.0

    def test_jenny_runtime_creation(self):
        """JennyRuntime should be creatable without starting."""
        from HoloLoom.visualization.jenny_runtime import JennyRuntime, JennyConfig

        config = JennyConfig()
        runtime = JennyRuntime(config=config)

        # Should not be started yet (lazy initialization)
        assert runtime._started == False

    def test_panel_type_enum(self):
        """PanelTypeJenny enum should have expected values."""
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        # Check all expected panel types exist
        assert hasattr(PanelTypeJenny, 'TEXT')
        assert hasattr(PanelTypeJenny, 'CODE')
        assert hasattr(PanelTypeJenny, 'GRAPH')
        assert hasattr(PanelTypeJenny, 'CONFIDENCE')
        assert hasattr(PanelTypeJenny, 'TIMELINE')
        assert hasattr(PanelTypeJenny, 'METRIC')


# ========================================
# Tests: Jenny MRF Integration (Phase 2.1-2.2 - December 2025)
# ========================================

class TestJennyMRFIntegration:
    """Test Jenny MRF integration with Thompson Sampling panel learning."""

    def test_mrf_imports_available(self):
        """Jenny MRF module should be importable."""
        from HoloLoom.visualization.jenny_mrf import (
            JennyMRFCompiler,
            PanelTypeLearner,
            create_mrf_compiler,
        )
        assert JennyMRFCompiler is not None
        assert PanelTypeLearner is not None
        assert create_mrf_compiler is not None

    def test_mrf_config_options_exist(self):
        """MRF config options should exist on Config."""
        cfg = Config.fast()

        assert hasattr(cfg, 'jenny_enable_mrf')
        assert hasattr(cfg, 'jenny_enable_learning')
        assert hasattr(cfg, 'jenny_learning_persist_path')

    def test_mrf_config_defaults(self):
        """MRF config options should have correct defaults."""
        cfg = Config.fast()

        assert cfg.jenny_enable_mrf == True
        assert cfg.jenny_enable_learning == True
        assert cfg.jenny_learning_persist_path == "./jenny_learning"

    def test_panel_type_learner_creation(self):
        """PanelTypeLearner should be creatable."""
        from HoloLoom.visualization.jenny_mrf import PanelTypeLearner

        learner = PanelTypeLearner()

        # Check initial statistics
        stats = learner.get_statistics()
        assert "total_selections" in stats
        assert stats["total_selections"] == 0

    def test_panel_type_learner_select(self):
        """PanelTypeLearner should select panel type using Thompson Sampling."""
        from HoloLoom.visualization.jenny_mrf import PanelTypeLearner
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        learner = PanelTypeLearner()

        # Select from candidates
        candidates = [PanelTypeJenny.TEXT, PanelTypeJenny.CODE, PanelTypeJenny.GRAPH]
        selected = learner.select("factual", candidates)

        assert selected in candidates

        # Check selection was recorded
        stats = learner.get_statistics()
        assert stats["total_selections"] == 1

    def test_panel_type_learner_update(self):
        """PanelTypeLearner should update priors on success/failure."""
        from HoloLoom.visualization.jenny_mrf import PanelTypeLearner
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        learner = PanelTypeLearner()

        # Update with success
        learner.update("factual", PanelTypeJenny.TEXT, success=True, confidence=0.9)

        # Check prior was updated
        stats = learner.get_statistics()
        prior_key = "factual:text"
        assert prior_key in stats["priors"]
        # Alpha should be > 1 after success update
        assert stats["priors"][prior_key]["alpha"] > 1.0

    def test_mrf_compiler_creation(self):
        """JennyMRFCompiler should be creatable via factory function."""
        from HoloLoom.visualization.jenny_mrf import (
            JennyMRFCompiler,
            PanelTypeLearner,
            create_mrf_compiler,
        )

        # create_mrf_compiler creates learner internally
        compiler = create_mrf_compiler(enable_learning=True)

        assert compiler is not None
        assert compiler.enable_learning == True
        # Learner should be created internally
        assert compiler.learner is not None

    def test_orchestrator_has_mrf_attributes(self):
        """WeavingOrchestrator should have MRF-related attributes."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        # Check class has expected methods
        assert hasattr(WeavingOrchestrator, '_classify_query_type')
        assert hasattr(WeavingOrchestrator, '_get_panel_type_candidates')
        assert hasattr(WeavingOrchestrator, '_detect_panel_type_heuristic')

    def test_classify_query_type_factual(self):
        """Query type classification should identify factual queries."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_orch = MagicMock(spec=WeavingOrchestrator)
        mock_orch._classify_query_type = WeavingOrchestrator._classify_query_type.__get__(
            mock_orch, WeavingOrchestrator
        )

        mock_spacetime = MagicMock()
        mock_spacetime.query_text = "What is Thompson Sampling?"

        query_type = mock_orch._classify_query_type(mock_spacetime)
        assert query_type == 'factual'

    def test_classify_query_type_procedural(self):
        """Query type classification should identify procedural queries."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_orch = MagicMock(spec=WeavingOrchestrator)
        mock_orch._classify_query_type = WeavingOrchestrator._classify_query_type.__get__(
            mock_orch, WeavingOrchestrator
        )

        mock_spacetime = MagicMock()
        mock_spacetime.query_text = "How to implement Thompson Sampling?"

        query_type = mock_orch._classify_query_type(mock_spacetime)
        assert query_type == 'procedural'

    def test_classify_query_type_analytical(self):
        """Query type classification should identify analytical queries."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_orch = MagicMock(spec=WeavingOrchestrator)
        mock_orch._classify_query_type = WeavingOrchestrator._classify_query_type.__get__(
            mock_orch, WeavingOrchestrator
        )

        mock_spacetime = MagicMock()
        mock_spacetime.query_text = "Compare Thompson Sampling vs UCB"

        query_type = mock_orch._classify_query_type(mock_spacetime)
        assert query_type == 'analytical'

    def test_classify_query_type_exploratory(self):
        """Query type classification should identify exploratory queries."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator

        mock_orch = MagicMock(spec=WeavingOrchestrator)
        mock_orch._classify_query_type = WeavingOrchestrator._classify_query_type.__get__(
            mock_orch, WeavingOrchestrator
        )

        mock_spacetime = MagicMock()
        # Use a query with only exploratory keywords (no analytical keywords like 'tradeoff')
        mock_spacetime.query_text = "Research different approaches to machine learning"

        query_type = mock_orch._classify_query_type(mock_spacetime)
        assert query_type == 'exploratory'

    def test_get_panel_type_candidates_basic(self):
        """Panel type candidates should always include TEXT."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        mock_orch = MagicMock(spec=WeavingOrchestrator)
        mock_orch._get_panel_type_candidates = WeavingOrchestrator._get_panel_type_candidates.__get__(
            mock_orch, WeavingOrchestrator
        )

        mock_spacetime = MagicMock()
        mock_spacetime.response = "Simple response"
        mock_spacetime.confidence = 0.9
        mock_spacetime.trace = None

        candidates = mock_orch._get_panel_type_candidates(mock_spacetime)

        assert PanelTypeJenny.TEXT in candidates

    def test_get_panel_type_candidates_includes_code(self):
        """Panel type candidates should include CODE for code blocks."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        mock_orch = MagicMock(spec=WeavingOrchestrator)
        mock_orch._get_panel_type_candidates = WeavingOrchestrator._get_panel_type_candidates.__get__(
            mock_orch, WeavingOrchestrator
        )

        mock_spacetime = MagicMock()
        mock_spacetime.response = "Here's code:\n```python\nprint('hello')\n```"
        mock_spacetime.confidence = 0.9
        mock_spacetime.trace = None

        candidates = mock_orch._get_panel_type_candidates(mock_spacetime)

        assert PanelTypeJenny.CODE in candidates

    def test_detect_panel_type_uses_heuristic_fallback(self):
        """Panel type detection should use heuristic when MRF unavailable."""
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.visualization.jenny_spec import PanelTypeJenny

        mock_orch = MagicMock(spec=WeavingOrchestrator)

        # Bind methods
        mock_orch._detect_jenny_panel_type = WeavingOrchestrator._detect_jenny_panel_type.__get__(
            mock_orch, WeavingOrchestrator
        )
        mock_orch._detect_panel_type_heuristic = WeavingOrchestrator._detect_panel_type_heuristic.__get__(
            mock_orch, WeavingOrchestrator
        )
        mock_orch._classify_query_type = WeavingOrchestrator._classify_query_type.__get__(
            mock_orch, WeavingOrchestrator
        )
        mock_orch._get_panel_type_candidates = WeavingOrchestrator._get_panel_type_candidates.__get__(
            mock_orch, WeavingOrchestrator
        )

        # Set MRF as unavailable
        mock_orch.jenny_mrf_compiler = None
        mock_orch.jenny_learner = None

        # Copy class constants
        mock_orch.JENNY_CONFIDENCE_THRESHOLD = WeavingOrchestrator.JENNY_CONFIDENCE_THRESHOLD
        mock_orch.JENNY_THREADS_THRESHOLD = WeavingOrchestrator.JENNY_THREADS_THRESHOLD
        mock_orch.JENNY_STAGES_THRESHOLD = WeavingOrchestrator.JENNY_STAGES_THRESHOLD
        mock_orch.JENNY_DURATION_THRESHOLD_MS = WeavingOrchestrator.JENNY_DURATION_THRESHOLD_MS

        # Create spacetime
        mock_spacetime = MagicMock()
        mock_spacetime.response = "Simple text response"
        mock_spacetime.confidence = 0.85
        mock_spacetime.trace = None

        panel_type = mock_orch._detect_jenny_panel_type(mock_spacetime)

        # Should fall back to heuristic (TEXT for simple response)
        assert panel_type == PanelTypeJenny.TEXT


class TestJennyMRFConfigDisable:
    """Test that MRF can be disabled via config."""

    def test_config_disable_mrf(self):
        """Setting jenny_enable_mrf=False should disable MRF."""
        cfg = Config.fast()
        cfg.jenny_enable_mrf = False

        assert cfg.jenny_enable_mrf == False

    def test_config_disable_learning(self):
        """Setting jenny_enable_learning=False should disable learning."""
        cfg = Config.fast()
        cfg.jenny_enable_learning = False

        assert cfg.jenny_enable_learning == False


# ========================================
# Run Tests
# ========================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for Jenny LLM Compiler - Phase A Implementation
======================================================
Comprehensive tests for LLM-based panel compilation.

Date: December 2025
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Dict, Any, List

# Import test targets
from HoloLoom.visualization.jenny_llm_compiler import (
    LLMJennyCompiler,
    LLMClient,
    LLMPanelSelection,
    SemanticProfile,
    create_llm_compiler,
    create_compiler_with_fallback,
    analyze_semantic_axes,
    parse_llm_response,
    LLM_OUTPUT_SCHEMA,
    LLM_PANEL_TYPES,
    BASE_PROMPT_TEMPLATE,
)
from HoloLoom.visualization.jenny_spec import PanelTypeJenny


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_spacetime():
    """Create a mock Spacetime object."""
    spacetime = MagicMock()
    spacetime.query_text = "What is Thompson Sampling?"
    spacetime.response = "Thompson Sampling is a Bayesian approach to exploration-exploitation..."
    spacetime.text = spacetime.response
    spacetime.confidence = 0.85
    spacetime.trace = MagicMock()
    spacetime.trace.threads = []
    spacetime.trace.total_duration_ms = 150.0
    spacetime.trace.stage_durations = {"retrieval": 50, "decision": 30, "synthesis": 70}
    return spacetime


@pytest.fixture
def mock_analysis():
    """Create a mock QueryAnalysis."""
    analysis = MagicMock()
    analysis.query_type = "factual"
    analysis.complexity = "moderate"
    analysis.confidence_level = "high"
    analysis.has_sources = True
    analysis.has_graph_context = True
    analysis.has_metrics = True
    analysis.has_reasoning = False
    return analysis


# ============================================================================
# Test: SemanticProfile
# ============================================================================

class TestSemanticProfile:
    """Tests for SemanticProfile dataclass."""

    def test_default_values(self):
        """Test default values are 0.5."""
        profile = SemanticProfile()
        assert profile.technicality == 0.5
        assert profile.complexity == 0.5
        assert profile.actionability == 0.5
        assert profile.urgency == 0.5
        assert profile.abstraction == 0.5
        assert profile.specificity == 0.5
        assert profile.objectivity == 0.5
        assert profile.novelty == 0.5

    def test_custom_values(self):
        """Test custom values."""
        profile = SemanticProfile(
            technicality=0.9,
            complexity=0.8,
            urgency=0.1
        )
        assert profile.technicality == 0.9
        assert profile.complexity == 0.8
        assert profile.urgency == 0.1

    def test_to_dict(self):
        """Test conversion to dictionary."""
        profile = SemanticProfile(technicality=0.7)
        d = profile.to_dict()
        assert isinstance(d, dict)
        assert d["technicality"] == 0.7
        assert len(d) == 8  # 8 axes

    def test_to_prompt_string(self):
        """Test conversion to prompt string."""
        profile = SemanticProfile(
            technicality=0.8,
            urgency=0.2
        )
        s = profile.to_prompt_string()
        assert "Technicality: high" in s
        assert "Urgency: low" in s
        assert "0.80" in s
        assert "0.20" in s


# ============================================================================
# Test: analyze_semantic_axes (Heuristic Fallback)
# ============================================================================

class TestAnalyzeSemanticAxes:
    """Tests for semantic axis analysis."""

    def test_technical_query(self):
        """Test technical query detection."""
        profile = analyze_semantic_axes("How do I implement the algorithm?")
        assert profile.technicality > 0.5
        assert profile.actionability > 0.5

    def test_simple_query(self):
        """Test simple query detection."""
        profile = analyze_semantic_axes("What is Python?")
        assert profile.complexity < 0.5  # Short query

    def test_complex_query(self):
        """Test complex query detection."""
        query = "Please explain the differences between the various machine learning algorithms and their tradeoffs in production environments"
        profile = analyze_semantic_axes(query)
        assert profile.complexity > 0.5  # Long query

    def test_urgent_query(self):
        """Test urgency detection."""
        profile = analyze_semantic_axes("I urgently need help with this bug!")
        assert profile.urgency > 0.5

    def test_comparative_query(self):
        """Test objectivity/comparison detection."""
        profile = analyze_semantic_axes("Compare Python vs JavaScript")
        assert profile.objectivity > 0.5


# ============================================================================
# Test: parse_llm_response
# ============================================================================

class TestParseLLMResponse:
    """Tests for LLM response parsing."""

    def test_valid_json_response(self):
        """Test parsing valid JSON response."""
        response = '''
        {
            "panel_type": "confidence",
            "reason": "High confidence query needs confidence display",
            "secondary_panels": ["sources"],
            "confidence": 0.9
        }
        '''
        result = parse_llm_response(response)

        assert result is not None
        assert result.panel_type == PanelTypeJenny.CONFIDENCE
        assert result.reason == "High confidence query needs confidence display"
        assert PanelTypeJenny.SOURCES in result.secondary_panels
        assert result.confidence == 0.9
        assert result.parse_method == "json"

    def test_json_with_surrounding_text(self):
        """Test parsing JSON embedded in text."""
        response = '''
        Here is my analysis:
        {"panel_type": "graph", "reason": "Visualize concepts", "confidence": 0.85}
        I hope this helps!
        '''
        result = parse_llm_response(response)

        assert result is not None
        assert result.panel_type == PanelTypeJenny.GRAPH
        assert result.parse_method == "json"

    def test_regex_fallback(self):
        """Test regex fallback for non-JSON response."""
        response = "I recommend using a GRAPH panel with confidence 0.75"
        result = parse_llm_response(response)

        assert result is not None
        assert result.panel_type == PanelTypeJenny.GRAPH
        assert result.confidence == 0.75
        assert result.parse_method == "regex"

    def test_regex_no_confidence(self):
        """Test regex fallback without explicit confidence."""
        response = "You should display a timeline panel for this query."
        result = parse_llm_response(response)

        assert result is not None
        assert result.panel_type == PanelTypeJenny.TIMELINE
        assert result.confidence == 0.5  # Default
        assert result.parse_method == "regex"

    def test_invalid_response(self):
        """Test handling of completely invalid response."""
        response = "I don't know what to suggest."
        result = parse_llm_response(response)

        # Should return None when no panel type found
        assert result is None

    def test_all_panel_types(self):
        """Test parsing all valid panel types."""
        panel_types = ["text", "code", "graph", "confidence", "timeline", "metric", "reasoning", "sources"]

        for pt in panel_types:
            response = f'{{"panel_type": "{pt}", "reason": "test", "confidence": 0.8}}'
            result = parse_llm_response(response)

            assert result is not None, f"Failed to parse {pt}"
            assert result.panel_type.value == pt

    def test_secondary_panels_deduplication(self):
        """Test that primary panel is not repeated in secondary panels."""
        response = '''
        {
            "panel_type": "confidence",
            "reason": "test",
            "secondary_panels": ["confidence", "graph", "sources"],
            "confidence": 0.9
        }
        '''
        result = parse_llm_response(response)

        assert result is not None
        # Confidence should not appear in secondary (it's the primary)
        assert PanelTypeJenny.CONFIDENCE not in result.secondary_panels
        assert PanelTypeJenny.GRAPH in result.secondary_panels
        assert PanelTypeJenny.SOURCES in result.secondary_panels


# ============================================================================
# Test: LLMPanelSelection
# ============================================================================

class TestLLMPanelSelection:
    """Tests for LLMPanelSelection dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        selection = LLMPanelSelection(
            panel_type=PanelTypeJenny.CONFIDENCE,
            reason="Test reason",
            secondary_panels=[PanelTypeJenny.SOURCES],
            confidence=0.9,
            raw_response="test",
            parse_method="json"
        )

        d = selection.to_dict()
        assert d["panel_type"] == "confidence"
        assert d["reason"] == "Test reason"
        assert d["secondary_panels"] == ["sources"]
        assert d["confidence"] == 0.9
        assert d["parse_method"] == "json"


# ============================================================================
# Test: LLMClient
# ============================================================================

class TestLLMClient:
    """Tests for LLMClient wrapper."""

    def test_unknown_provider(self):
        """Test handling of unknown provider."""
        client = LLMClient(provider="unknown_provider")
        assert client.available is False

    def test_ollama_unavailable(self):
        """Test handling when Ollama is not installed."""
        with patch.dict("sys.modules", {"ollama": None}):
            client = LLMClient(provider="ollama")
            # Should handle gracefully
            assert client.provider == "ollama"

    @pytest.mark.asyncio
    async def test_generate_when_unavailable(self):
        """Test generate returns None when client unavailable."""
        client = LLMClient(provider="ollama")
        client._available = False

        result = await client.generate("test prompt")
        assert result is None


# ============================================================================
# Test: LLMJennyCompiler
# ============================================================================

class TestLLMJennyCompiler:
    """Tests for LLMJennyCompiler class."""

    def test_initialization(self):
        """Test compiler initialization."""
        compiler = LLMJennyCompiler(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            enable_learning=True,
        )

        assert compiler.llm_provider == "ollama"
        assert compiler.llm_model == "llama3.2:3b"
        assert compiler.enable_learning is True
        assert compiler.VERSION == "3.1.0-llm-async"

    def test_initialization_with_learner(self):
        """Test compiler with pre-configured learner."""
        from HoloLoom.visualization.jenny_mrf import PanelTypeLearner

        learner = PanelTypeLearner()
        compiler = LLMJennyCompiler(
            llm_provider="ollama",
            learner=learner,
        )

        assert compiler.learner is learner

    def test_get_llm_statistics(self):
        """Test LLM statistics retrieval."""
        compiler = LLMJennyCompiler(llm_provider="ollama")
        stats = compiler.get_llm_statistics()

        assert "llm_provider" in stats
        assert "llm_model" in stats
        assert "llm_available" in stats
        assert "total_calls" in stats
        assert "successful_calls" in stats
        assert "fallbacks" in stats
        assert "success_rate" in stats

    def test_get_learning_statistics_combined(self):
        """Test combined learning + LLM statistics."""
        compiler = LLMJennyCompiler(enable_learning=True)
        stats = compiler.get_learning_statistics()

        assert "llm" in stats
        assert "total_selections" in stats or "learning_enabled" in stats

    @pytest.mark.asyncio
    async def test_compile_fallback_to_mrf(self, mock_spacetime):
        """Test fallback to MRF when LLM unavailable."""
        compiler = LLMJennyCompiler(llm_provider="ollama")
        compiler._llm_available = False
        compiler._use_async_client = False  # Ensure we don't try async

        # Mock the parent compile method
        with patch.object(
            LLMJennyCompiler.__bases__[0],
            'compile',
            new_callable=AsyncMock
        ) as mock_compile:
            mock_compile.return_value = [MagicMock()]

            result = await compiler.compile(mock_spacetime)

            # Should fall back to parent
            assert mock_compile.called
            assert compiler._llm_fallbacks == 1

    @pytest.mark.asyncio
    async def test_compile_with_llm_success(self, mock_spacetime, mock_analysis):
        """Test successful LLM compilation."""
        compiler = LLMJennyCompiler(llm_provider="ollama")
        compiler._llm_available = True
        compiler._use_async_client = False  # Use legacy client for testing

        # Mock LLM generate
        llm_response = '''
        {
            "panel_type": "confidence",
            "reason": "Query has high confidence",
            "secondary_panels": ["sources"],
            "confidence": 0.9
        }
        '''
        # Create legacy client if not exists and mock it
        if not compiler._llm_legacy:
            compiler._llm_legacy = MagicMock()
            compiler._llm_legacy.available = True
        compiler._llm_legacy.generate = AsyncMock(return_value=llm_response)

        # Mock analyze_query
        with patch(
            'HoloLoom.visualization.jenny_llm_compiler.analyze_query',
            return_value=mock_analysis
        ):
            result = await compiler.compile(mock_spacetime)

        assert result is not None
        assert len(result) >= 2  # At least text + confidence panels
        assert compiler._llm_successes == 1

    @pytest.mark.asyncio
    async def test_compile_llm_parse_failure_fallback(self, mock_spacetime, mock_analysis):
        """Test fallback when LLM response can't be parsed."""
        compiler = LLMJennyCompiler(llm_provider="ollama")
        compiler._llm_available = True
        compiler._use_async_client = False  # Use legacy client for testing

        # Create legacy client and mock it
        if not compiler._llm_legacy:
            compiler._llm_legacy = MagicMock()
            compiler._llm_legacy.available = True
        # Return unparseable response
        compiler._llm_legacy.generate = AsyncMock(return_value="I don't know")

        with patch(
            'HoloLoom.visualization.jenny_llm_compiler.analyze_query',
            return_value=mock_analysis
        ):
            with patch.object(
                LLMJennyCompiler.__bases__[0],
                'compile',
                new_callable=AsyncMock
            ) as mock_compile:
                mock_compile.return_value = [MagicMock()]

                result = await compiler.compile(mock_spacetime)

                # Should fall back to parent
                assert mock_compile.called


# ============================================================================
# Test: Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_llm_compiler(self):
        """Test create_llm_compiler factory."""
        compiler = create_llm_compiler(
            llm_provider="ollama",
            llm_model="llama3.2:3b",
            enable_learning=True,
        )

        assert isinstance(compiler, LLMJennyCompiler)
        assert compiler.llm_provider == "ollama"
        assert compiler.llm_model == "llama3.2:3b"

    def test_create_compiler_with_fallback_llm_unavailable(self):
        """Test fallback when LLM not available."""
        compiler = create_compiler_with_fallback(
            prefer_llm=True,
            llm_provider="nonexistent"
        )

        # Should fall back to MRF compiler
        from HoloLoom.visualization.jenny_mrf import JennyMRFCompiler
        assert isinstance(compiler, JennyMRFCompiler)

    def test_create_compiler_with_fallback_no_llm_preference(self):
        """Test when LLM not preferred."""
        compiler = create_compiler_with_fallback(prefer_llm=False)

        from HoloLoom.visualization.jenny_mrf import JennyMRFCompiler
        assert isinstance(compiler, JennyMRFCompiler)


# ============================================================================
# Test: Prompt Building
# ============================================================================

class TestPromptBuilding:
    """Tests for prompt building functionality."""

    @pytest.mark.asyncio
    async def test_build_llm_prompt_basic(self, mock_spacetime, mock_analysis):
        """Test basic prompt building."""
        compiler = LLMJennyCompiler(
            enable_mrf_prompt_enhancement=False,
            enable_semantic_axes=False
        )

        prompt = await compiler._build_llm_prompt(mock_spacetime, mock_analysis)

        assert "Thompson Sampling" in prompt
        assert "factual" in prompt
        assert "85%" in prompt or "0.85" in prompt

    @pytest.mark.asyncio
    async def test_build_llm_prompt_with_semantic_axes(self, mock_spacetime, mock_analysis):
        """Test prompt building with semantic axes."""
        compiler = LLMJennyCompiler(
            enable_mrf_prompt_enhancement=False,
            enable_semantic_axes=True
        )

        prompt = await compiler._build_llm_prompt(mock_spacetime, mock_analysis)

        # Should include semantic profile
        assert "Technicality" in prompt or "technicality" in prompt


# ============================================================================
# Test: Graceful Fallback Chain
# ============================================================================

class TestGracefulFallbackChain:
    """Tests for the graceful fallback chain."""

    @pytest.mark.asyncio
    async def test_fallback_chain_order(self, mock_spacetime, mock_analysis):
        """Test fallback chain: LLM -> MRF -> Heuristics."""
        compiler = LLMJennyCompiler(llm_provider="ollama")

        # Scenario 1: LLM available and succeeds
        compiler._llm_available = True
        compiler._use_async_client = False  # Use legacy client for testing

        # Create legacy client and mock it
        if not compiler._llm_legacy:
            compiler._llm_legacy = MagicMock()
            compiler._llm_legacy.available = True
        compiler._llm_legacy.generate = AsyncMock(return_value='{"panel_type": "graph", "reason": "test", "confidence": 0.8}')

        with patch(
            'HoloLoom.visualization.jenny_llm_compiler.analyze_query',
            return_value=mock_analysis
        ):
            result = await compiler.compile(mock_spacetime)

        assert result is not None
        assert compiler._llm_successes >= 1

    @pytest.mark.asyncio
    async def test_llm_timeout_triggers_fallback(self, mock_spacetime, mock_analysis):
        """Test that LLM timeout triggers fallback."""
        compiler = LLMJennyCompiler(llm_provider="ollama")
        compiler._llm_available = True
        compiler._use_async_client = False  # Use legacy client for testing

        # Create legacy client and mock it to simulate timeout
        if not compiler._llm_legacy:
            compiler._llm_legacy = MagicMock()
            compiler._llm_legacy.available = True
        compiler._llm_legacy.generate = AsyncMock(return_value=None)

        with patch(
            'HoloLoom.visualization.jenny_llm_compiler.analyze_query',
            return_value=mock_analysis
        ):
            with patch.object(
                LLMJennyCompiler.__bases__[0],
                'compile',
                new_callable=AsyncMock
            ) as mock_compile:
                mock_compile.return_value = [MagicMock()]

                result = await compiler.compile(mock_spacetime)

                assert mock_compile.called
                assert compiler._llm_fallbacks >= 1


# ============================================================================
# Test: Constants and Schema
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_llm_output_schema_structure(self):
        """Test LLM output schema structure."""
        assert "panel_type" in LLM_OUTPUT_SCHEMA
        assert "reason" in LLM_OUTPUT_SCHEMA
        assert "content_structure" in LLM_OUTPUT_SCHEMA
        assert "secondary_panels" in LLM_OUTPUT_SCHEMA
        assert "confidence" in LLM_OUTPUT_SCHEMA

    def test_llm_panel_types(self):
        """Test LLM panel types list."""
        expected_types = ["text", "code", "graph", "confidence", "timeline", "metric", "reasoning", "sources"]
        assert LLM_PANEL_TYPES == expected_types

    def test_base_prompt_template_placeholders(self):
        """Test base prompt template has required placeholders."""
        assert "{query}" in BASE_PROMPT_TEMPLATE
        assert "{response_preview}" in BASE_PROMPT_TEMPLATE
        # Note: confidence uses format specifier {confidence:.0%}
        assert "confidence" in BASE_PROMPT_TEMPLATE
        assert "{query_type}" in BASE_PROMPT_TEMPLATE
        assert "{semantic_profile}" in BASE_PROMPT_TEMPLATE


# ============================================================================
# Test: Integration with Phase B and C
# ============================================================================

class TestPhaseIntegration:
    """Tests for integration with Phase B (MRF) and Phase C (Learning Loop)."""

    def test_inherits_from_mrf_compiler(self):
        """Test that LLMJennyCompiler inherits from JennyMRFCompiler."""
        from HoloLoom.visualization.jenny_mrf import JennyMRFCompiler

        compiler = LLMJennyCompiler()
        assert isinstance(compiler, JennyMRFCompiler)

    def test_learning_enabled_by_default(self):
        """Test that Thompson Sampling learning is enabled by default."""
        compiler = LLMJennyCompiler()
        assert compiler.enable_learning is True
        assert compiler.learner is not None

    def test_update_learning_works(self):
        """Test that update_learning from parent class works."""
        compiler = LLMJennyCompiler(enable_learning=True)

        # Should not raise
        compiler.update_learning(
            query_type="factual",
            panel_type=PanelTypeJenny.CONFIDENCE,
            success=True,
            confidence=0.9
        )

        stats = compiler.get_learning_statistics()
        assert "priors" in stats or "learning_enabled" in stats


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

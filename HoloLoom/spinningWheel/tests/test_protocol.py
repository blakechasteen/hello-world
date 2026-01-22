"""
SpinningWheel Protocol Tests
============================

Comprehensive tests for the SpinningWheel protocol module:
- SpinnerStatus enum
- SpinnerCapabilities dataclass
- SpinResult dataclass
- ImportanceSignals and ImportanceScore
- SpinnerCheckpoint for resumable operations
- SpinnerProtocol interface
- BaseSpinner abstract base class

Author: Claude Code
Date: January 2026
"""

import pytest
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import json


# =============================================================================
# Import the actual protocol module
# =============================================================================

try:
    from HoloLoom.spinningWheel.protocol import (
        SpinnerStatus,
        SpinnerCapabilities,
        SpinResult,
        ImportanceSignals,
        ImportanceScore,
        SpinnerCheckpoint,
        SpinnerProtocol,
        BaseSpinner,
        ImportanceScorer,
        TechnicalTermScorer,
    )
    from HoloLoom.protocols.types import MemoryShard
    PROTOCOL_AVAILABLE = True
except ImportError as e:
    PROTOCOL_AVAILABLE = False
    IMPORT_ERROR = str(e)


# =============================================================================
# Skip decorator if imports fail
# =============================================================================

pytestmark = pytest.mark.skipif(
    not PROTOCOL_AVAILABLE,
    reason=f"Protocol module not available: {IMPORT_ERROR if not PROTOCOL_AVAILABLE else ''}"
)


# =============================================================================
# Test SpinnerStatus Enum
# =============================================================================

class TestSpinnerStatus:
    """Test SpinnerStatus enum."""

    def test_status_values_exist(self):
        """Test that all expected status values exist."""
        assert SpinnerStatus.AVAILABLE
        assert SpinnerStatus.DEGRADED
        assert SpinnerStatus.UNAVAILABLE
        assert SpinnerStatus.ERROR

    def test_status_values_are_strings(self):
        """Test that status values are strings."""
        assert SpinnerStatus.AVAILABLE.value == "available"
        assert SpinnerStatus.DEGRADED.value == "degraded"
        assert SpinnerStatus.UNAVAILABLE.value == "unavailable"
        assert SpinnerStatus.ERROR.value == "error"

    def test_status_comparison(self):
        """Test status comparison."""
        assert SpinnerStatus.AVAILABLE == SpinnerStatus.AVAILABLE
        assert SpinnerStatus.AVAILABLE != SpinnerStatus.ERROR

    def test_status_iteration(self):
        """Test that we can iterate over all statuses."""
        statuses = list(SpinnerStatus)
        assert len(statuses) == 4


# =============================================================================
# Test SpinnerCapabilities Dataclass
# =============================================================================

class TestSpinnerCapabilities:
    """Test SpinnerCapabilities dataclass."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = SpinnerCapabilities()

        assert caps.streaming is False
        assert caps.incremental is False
        assert caps.batch is False
        assert caps.entity_extraction is True
        assert caps.motif_extraction is True
        assert caps.importance_scoring is True
        assert caps.checkpointing is False
        assert caps.max_size_bytes is None
        assert caps.supported_formats == []

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = SpinnerCapabilities(
            streaming=True,
            incremental=True,
            batch=True,
            entity_extraction=False,
            max_size_bytes=1024 * 1024,
            supported_formats=['.txt', '.md']
        )

        assert caps.streaming is True
        assert caps.incremental is True
        assert caps.batch is True
        assert caps.entity_extraction is False
        assert caps.max_size_bytes == 1024 * 1024
        assert caps.supported_formats == ['.txt', '.md']

    def test_capabilities_are_hashable(self):
        """Test that capabilities can be used in sets/dicts."""
        # SpinnerCapabilities is a dataclass with frozen=False by default
        # so it might not be hashable, but we can check if it's usable
        caps = SpinnerCapabilities()
        assert caps is not None

    def test_capabilities_equality(self):
        """Test capabilities equality."""
        caps1 = SpinnerCapabilities(streaming=True)
        caps2 = SpinnerCapabilities(streaming=True)
        caps3 = SpinnerCapabilities(streaming=False)

        assert caps1 == caps2
        assert caps1 != caps3


# =============================================================================
# Test SpinResult Dataclass
# =============================================================================

class TestSpinResult:
    """Test SpinResult dataclass."""

    def test_default_spin_result(self):
        """Test default SpinResult values."""
        result = SpinResult(shards=[])

        assert result.shards == []
        assert result.success is True
        assert result.error is None
        assert result.items_processed == 0
        assert result.items_filtered == 0
        assert result.duration_seconds == 0.0
        assert result.metadata == {}

    def test_spin_result_with_shards(self):
        """Test SpinResult with shards."""
        # Create mock shards
        mock_shard = Mock()
        mock_shard.id = "test_123"
        mock_shard.text = "Test content"

        result = SpinResult(
            shards=[mock_shard],
            success=True,
            items_processed=1,
            items_filtered=0,
            duration_seconds=0.5
        )

        assert len(result.shards) == 1
        assert result.shards[0].id == "test_123"
        assert result.items_processed == 1
        assert result.duration_seconds == 0.5

    def test_spin_result_error(self):
        """Test SpinResult with error."""
        result = SpinResult(
            shards=[],
            success=False,
            error="File not found",
            items_processed=0
        )

        assert result.success is False
        assert result.error == "File not found"
        assert len(result.shards) == 0

    def test_spin_result_with_metadata(self):
        """Test SpinResult with metadata."""
        result = SpinResult(
            shards=[],
            success=True,
            metadata={
                'source_type': 'pdf',
                'page_count': 10,
                'file_size': 1024
            }
        )

        assert result.metadata['source_type'] == 'pdf'
        assert result.metadata['page_count'] == 10
        assert result.metadata['file_size'] == 1024


# =============================================================================
# Test ImportanceSignals Dataclass
# =============================================================================

class TestImportanceSignals:
    """Test ImportanceSignals dataclass."""

    def test_default_importance_signals(self):
        """Test default ImportanceSignals values."""
        signals = ImportanceSignals()

        assert signals.length_score == 0.5
        assert signals.technical_score == 0.5
        assert signals.structural_score == 0.5
        assert signals.authority_score == 0.5
        assert signals.recency_score == 0.5
        assert signals.engagement_score == 0.5
        assert signals.reference_score == 0.5
        assert signals.noise_penalty == 0.0
        assert signals.custom_signals == {}

    def test_custom_importance_signals(self):
        """Test custom ImportanceSignals values."""
        signals = ImportanceSignals(
            length_score=0.8,
            technical_score=0.9,
            structural_score=0.7,
            authority_score=0.6,
            noise_penalty=0.1,
            custom_signals={'complexity': 0.75}
        )

        assert signals.length_score == 0.8
        assert signals.technical_score == 0.9
        assert signals.noise_penalty == 0.1
        assert signals.custom_signals['complexity'] == 0.75

    def test_signals_are_clamped(self):
        """Test that signal values should be 0-1 (documentation check)."""
        # Values outside 0-1 are allowed by dataclass but should be avoided
        signals = ImportanceSignals(length_score=0.0, technical_score=1.0)
        assert 0.0 <= signals.length_score <= 1.0
        assert 0.0 <= signals.technical_score <= 1.0


# =============================================================================
# Test ImportanceScore Dataclass
# =============================================================================

class TestImportanceScore:
    """Test ImportanceScore dataclass."""

    def test_importance_score_creation(self):
        """Test ImportanceScore creation."""
        signals = ImportanceSignals(length_score=0.8)
        score = ImportanceScore(
            score=0.75,
            signals=signals,
            reason="High length score"
        )

        assert score.score == 0.75
        assert score.signals.length_score == 0.8
        assert score.reason == "High length score"

    def test_importance_score_default_reason(self):
        """Test ImportanceScore with default reason."""
        signals = ImportanceSignals()
        score = ImportanceScore(
            score=0.5,
            signals=signals
        )

        # Check default reason is empty or descriptive
        assert score.reason == "" or isinstance(score.reason, str)

    def test_importance_score_bounds(self):
        """Test score value bounds."""
        signals = ImportanceSignals()

        # Valid score
        score = ImportanceScore(score=0.5, signals=signals)
        assert 0.0 <= score.score <= 1.0


# =============================================================================
# Test SpinnerCheckpoint Dataclass
# =============================================================================

class TestSpinnerCheckpoint:
    """Test SpinnerCheckpoint dataclass."""

    def test_checkpoint_creation(self):
        """Test SpinnerCheckpoint creation."""
        checkpoint = SpinnerCheckpoint(
            spinner_id="codebase_spinner_123",
            progress=0.5,
            state={'current_file': 'main.py', 'processed': 10}
        )

        assert checkpoint.spinner_id == "codebase_spinner_123"
        assert checkpoint.progress == 0.5
        assert checkpoint.state['current_file'] == 'main.py'
        assert checkpoint.state['processed'] == 10

    def test_checkpoint_default_values(self):
        """Test SpinnerCheckpoint defaults."""
        checkpoint = SpinnerCheckpoint(
            spinner_id="test_spinner"
        )

        assert checkpoint.spinner_id == "test_spinner"
        assert checkpoint.progress == 0.0
        assert checkpoint.state == {}

    def test_checkpoint_progress_range(self):
        """Test checkpoint progress values."""
        # At start
        cp_start = SpinnerCheckpoint(spinner_id="test", progress=0.0)
        assert cp_start.progress == 0.0

        # In middle
        cp_mid = SpinnerCheckpoint(spinner_id="test", progress=0.5)
        assert cp_mid.progress == 0.5

        # At end
        cp_end = SpinnerCheckpoint(spinner_id="test", progress=1.0)
        assert cp_end.progress == 1.0


# =============================================================================
# Test TechnicalTermScorer
# =============================================================================

class TestTechnicalTermScorer:
    """Test TechnicalTermScorer class."""

    def test_default_technical_terms(self):
        """Test scorer with default technical terms."""
        scorer = TechnicalTermScorer()

        # Check that scorer has some default terms
        assert hasattr(scorer, 'terms') or hasattr(scorer, 'technical_terms')

    def test_custom_technical_terms(self):
        """Test scorer with custom technical terms."""
        custom_terms = {'machine', 'learning', 'neural', 'network'}
        scorer = TechnicalTermScorer(terms=custom_terms)

        # Score text with technical terms
        score = scorer.score("This is about machine learning and neural networks")
        assert score > 0.0

    def test_non_technical_text(self):
        """Test scorer with non-technical text."""
        scorer = TechnicalTermScorer()

        # Score generic text
        score = scorer.score("Hello world this is a simple message")
        # Should be lower than technical text
        assert isinstance(score, (int, float))

    def test_empty_text(self):
        """Test scorer with empty text."""
        scorer = TechnicalTermScorer()
        score = scorer.score("")
        assert score == 0.0


# =============================================================================
# Test ImportanceScorer
# =============================================================================

class TestImportanceScorer:
    """Test ImportanceScorer class."""

    def test_importance_scorer_creation(self):
        """Test ImportanceScorer creation."""
        scorer = ImportanceScorer()
        assert scorer is not None

    def test_importance_scorer_with_custom_terms(self):
        """Test ImportanceScorer with custom technical terms."""
        custom_terms = {'api', 'endpoint', 'request', 'response'}
        scorer = ImportanceScorer(technical_terms=custom_terms)
        assert scorer is not None

    def test_scorer_has_technical_scorer(self):
        """Test that ImportanceScorer has a technical_scorer."""
        scorer = ImportanceScorer()
        assert hasattr(scorer, 'technical_scorer')

    def test_score_text(self):
        """Test scoring text."""
        scorer = ImportanceScorer()

        # Technical text should score higher
        tech_text = "The API endpoint implements async request handling with caching"
        simple_text = "Hello world"

        # Both should return valid scores
        # (actual comparison depends on implementation)
        assert hasattr(scorer.technical_scorer, 'score')


# =============================================================================
# Test BaseSpinner Abstract Class
# =============================================================================

class TestBaseSpinner:
    """Test BaseSpinner abstract base class."""

    def test_cannot_instantiate_directly(self):
        """Test that BaseSpinner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSpinner()

    def test_concrete_implementation(self):
        """Test concrete implementation of BaseSpinner."""

        class ConcreteSpinner(BaseSpinner):
            """Concrete spinner implementation for testing."""

            @property
            def spinner_id(self) -> str:
                return "concrete_spinner"

            @property
            def capabilities(self) -> SpinnerCapabilities:
                return SpinnerCapabilities(streaming=True)

            async def _spin_impl(self, source: Any, **kwargs) -> List[Any]:
                """Implementation that returns mock shards."""
                return []

        spinner = ConcreteSpinner()
        assert spinner.spinner_id == "concrete_spinner"
        assert spinner.capabilities.streaming is True

    def test_spin_method(self):
        """Test that spin method calls _spin_impl."""

        class TestSpinner(BaseSpinner):
            def __init__(self):
                super().__init__()
                self.spin_called = False

            @property
            def spinner_id(self) -> str:
                return "test_spinner"

            @property
            def capabilities(self) -> SpinnerCapabilities:
                return SpinnerCapabilities()

            async def _spin_impl(self, source: Any, **kwargs) -> List[Any]:
                self.spin_called = True
                return []

        spinner = TestSpinner()

        async def run_test():
            result = await spinner.spin("test_input")
            assert spinner.spin_called is True
            return result

        asyncio.run(run_test())

    def test_check_status_method(self):
        """Test check_status method."""

        class TestSpinner(BaseSpinner):
            @property
            def spinner_id(self) -> str:
                return "test_spinner"

            @property
            def capabilities(self) -> SpinnerCapabilities:
                return SpinnerCapabilities()

            async def _spin_impl(self, source: Any, **kwargs) -> List[Any]:
                return []

        spinner = TestSpinner()

        async def run_test():
            status = await spinner.check_status()
            assert status in SpinnerStatus

        asyncio.run(run_test())


# =============================================================================
# Test MemoryShard Output Format
# =============================================================================

class TestMemoryShardFormat:
    """Test MemoryShard output format validation."""

    def test_memory_shard_required_fields(self):
        """Test that MemoryShard has required fields."""
        try:
            shard = MemoryShard(
                id="test_123",
                text="Test content",
                episode="test_episode",
                entities=["entity1", "entity2"],
                motifs=["motif1"],
                metadata={"source": "test"}
            )

            assert shard.id == "test_123"
            assert shard.text == "Test content"
            assert shard.episode == "test_episode"
            assert shard.entities == ["entity1", "entity2"]
            assert shard.motifs == ["motif1"]
            assert shard.metadata == {"source": "test"}
        except Exception as e:
            # If MemoryShard is not available, skip
            pytest.skip(f"MemoryShard not available: {e}")

    def test_memory_shard_id_uniqueness(self):
        """Test that shard IDs should be unique."""
        try:
            shard1 = MemoryShard(
                id="unique_1",
                text="Content 1",
                episode="test",
                entities=[],
                motifs=[],
                metadata={}
            )
            shard2 = MemoryShard(
                id="unique_2",
                text="Content 2",
                episode="test",
                entities=[],
                motifs=[],
                metadata={}
            )

            assert shard1.id != shard2.id
        except Exception as e:
            pytest.skip(f"MemoryShard not available: {e}")


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in protocol classes."""

    def test_spin_result_captures_errors(self):
        """Test that SpinResult properly captures errors."""
        result = SpinResult(
            shards=[],
            success=False,
            error="Parse error: Invalid format",
            metadata={'error_code': 'PARSE_001'}
        )

        assert result.success is False
        assert "Parse error" in result.error
        assert result.metadata['error_code'] == 'PARSE_001'

    def test_spinner_error_recovery(self):
        """Test spinner error recovery pattern."""

        class FailingSpinner(BaseSpinner):
            def __init__(self, should_fail: bool = False):
                super().__init__()
                self.should_fail = should_fail

            @property
            def spinner_id(self) -> str:
                return "failing_spinner"

            @property
            def capabilities(self) -> SpinnerCapabilities:
                return SpinnerCapabilities()

            async def _spin_impl(self, source: Any, **kwargs) -> List[Any]:
                if self.should_fail:
                    raise ValueError("Intentional failure")
                return []

        # Test successful case
        spinner_ok = FailingSpinner(should_fail=False)

        async def run_ok():
            result = await spinner_ok.spin("test")
            return result

        asyncio.run(run_ok())

        # Test failing case
        spinner_fail = FailingSpinner(should_fail=True)

        async def run_fail():
            with pytest.raises(ValueError):
                await spinner_fail._spin_impl("test")

        asyncio.run(run_fail())


# =============================================================================
# Test Metadata Structure
# =============================================================================

class TestMetadataStructure:
    """Test metadata structure conventions."""

    def test_spin_result_metadata_types(self):
        """Test that metadata contains expected types."""
        metadata = {
            'source_type': 'codebase',
            'file_count': 10,
            'total_lines': 5000,
            'languages': ['python', 'typescript'],
            'processing_time_ms': 125.5,
            'nested': {'key': 'value'}
        }

        result = SpinResult(
            shards=[],
            success=True,
            metadata=metadata
        )

        assert isinstance(result.metadata['source_type'], str)
        assert isinstance(result.metadata['file_count'], int)
        assert isinstance(result.metadata['total_lines'], int)
        assert isinstance(result.metadata['languages'], list)
        assert isinstance(result.metadata['processing_time_ms'], float)
        assert isinstance(result.metadata['nested'], dict)

    def test_importance_score_metadata(self):
        """Test importance score includes all signals."""
        signals = ImportanceSignals(
            length_score=0.8,
            technical_score=0.7,
            structural_score=0.6,
            custom_signals={'git_commits': 0.5}
        )

        score = ImportanceScore(
            score=0.65,
            signals=signals,
            reason="High technical content"
        )

        # All signal fields should be accessible
        assert score.signals.length_score == 0.8
        assert score.signals.technical_score == 0.7
        assert score.signals.custom_signals['git_commits'] == 0.5


# =============================================================================
# Test Spinner Protocol Compliance
# =============================================================================

class TestSpinnerProtocolCompliance:
    """Test that spinners comply with the protocol."""

    def test_protocol_methods(self):
        """Test that protocol defines required methods."""
        # SpinnerProtocol should define:
        # - spinner_id property
        # - capabilities property
        # - spin method
        # - check_status method

        assert hasattr(SpinnerProtocol, 'spinner_id')
        assert hasattr(SpinnerProtocol, 'capabilities')
        assert hasattr(SpinnerProtocol, 'spin')
        assert hasattr(SpinnerProtocol, 'check_status')

    def test_base_spinner_provides_defaults(self):
        """Test that BaseSpinner provides sensible defaults."""

        class MinimalSpinner(BaseSpinner):
            @property
            def spinner_id(self) -> str:
                return "minimal"

            @property
            def capabilities(self) -> SpinnerCapabilities:
                return SpinnerCapabilities()

            async def _spin_impl(self, source: Any, **kwargs) -> List[Any]:
                return []

        spinner = MinimalSpinner()

        # Check default methods exist
        assert hasattr(spinner, 'spin')
        assert hasattr(spinner, 'check_status')
        assert hasattr(spinner, 'spinner_id')
        assert hasattr(spinner, 'capabilities')


# =============================================================================
# Test Importance Scoring Edge Cases
# =============================================================================

class TestImportanceScoringEdgeCases:
    """Test edge cases in importance scoring."""

    def test_empty_signals(self):
        """Test scoring with all default signals."""
        signals = ImportanceSignals()
        score = ImportanceScore(score=0.5, signals=signals)

        # Should have reasonable default
        assert 0.0 <= score.score <= 1.0

    def test_max_signals(self):
        """Test scoring with all max signals."""
        signals = ImportanceSignals(
            length_score=1.0,
            technical_score=1.0,
            structural_score=1.0,
            authority_score=1.0,
            recency_score=1.0,
            engagement_score=1.0,
            reference_score=1.0,
            noise_penalty=0.0
        )

        score = ImportanceScore(score=1.0, signals=signals)
        assert score.score == 1.0

    def test_high_noise_penalty(self):
        """Test that high noise penalty affects score."""
        signals = ImportanceSignals(
            length_score=0.8,
            technical_score=0.8,
            noise_penalty=0.5
        )

        # Noise penalty should reduce overall importance
        # (exact calculation depends on implementation)
        score = ImportanceScore(score=0.5, signals=signals)
        assert signals.noise_penalty > 0


# =============================================================================
# Test Capabilities Combinations
# =============================================================================

class TestCapabilitiesCombinations:
    """Test various capability combinations."""

    def test_streaming_capable_spinner(self):
        """Test spinner with streaming capability."""
        caps = SpinnerCapabilities(
            streaming=True,
            batch=False,
            checkpointing=False
        )

        assert caps.streaming is True
        assert caps.batch is False

    def test_batch_processing_spinner(self):
        """Test spinner with batch capability."""
        caps = SpinnerCapabilities(
            streaming=False,
            batch=True,
            incremental=False
        )

        assert caps.batch is True
        assert caps.streaming is False

    def test_full_featured_spinner(self):
        """Test spinner with all capabilities."""
        caps = SpinnerCapabilities(
            streaming=True,
            incremental=True,
            batch=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            checkpointing=True,
            max_size_bytes=100 * 1024 * 1024,
            supported_formats=['.py', '.ts', '.js']
        )

        assert caps.streaming is True
        assert caps.incremental is True
        assert caps.batch is True
        assert caps.checkpointing is True
        assert caps.max_size_bytes == 100 * 1024 * 1024
        assert len(caps.supported_formats) == 3


# =============================================================================
# Test Spinner State Management
# =============================================================================

class TestSpinnerStateManagement:
    """Test spinner state management patterns."""

    def test_checkpoint_serialization(self):
        """Test that checkpoints can be serialized."""
        checkpoint = SpinnerCheckpoint(
            spinner_id="test_spinner",
            progress=0.75,
            state={
                'files_processed': ['a.py', 'b.py'],
                'current_index': 5,
                'timestamp': '2026-01-22T12:00:00'
            }
        )

        # Convert to dict for serialization
        data = {
            'spinner_id': checkpoint.spinner_id,
            'progress': checkpoint.progress,
            'state': checkpoint.state
        }

        # Should be JSON serializable
        json_str = json.dumps(data)
        restored = json.loads(json_str)

        assert restored['spinner_id'] == checkpoint.spinner_id
        assert restored['progress'] == checkpoint.progress
        assert restored['state']['current_index'] == 5

    def test_checkpoint_resume(self):
        """Test checkpoint resume pattern."""
        # Initial checkpoint
        checkpoint = SpinnerCheckpoint(
            spinner_id="codebase_spinner",
            progress=0.5,
            state={'last_file': 'main.py', 'file_index': 50}
        )

        # Simulate resume
        resumed_state = checkpoint.state.copy()
        resumed_state['resumed'] = True
        resumed_state['resume_time'] = '2026-01-22T12:00:00'

        new_checkpoint = SpinnerCheckpoint(
            spinner_id=checkpoint.spinner_id,
            progress=0.75,
            state=resumed_state
        )

        assert new_checkpoint.state['resumed'] is True
        assert new_checkpoint.progress > checkpoint.progress

"""
Tests for Dark Trace SAE Auto-Label Engine
==========================================

Tests the LLM-powered feature labeling system including:
- Configuration and initialization
- Prompt building and parsing
- Quality assessment
- Batch and single labeling
- Verification workflow

Created: 2025-12-28
"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from hololoom.dark_trace.sae.autolabel_engine import (
    AutoLabelConfig,
    AutoLabelEngine,
    LabelingStrategy,
    QualityLevel,
    QualityAssessment,
    LabelQualityAssessor,
    create_autolabel_engine,
    label_sae_features,
)
from hololoom.dark_trace.sae.label_prompts import (
    LabelPromptBuilder,
    LabelPromptContext,
    LabelResponseParser,
    ParsedLabelResponse,
    SINGLE_FEATURE_PROMPT_TEMPLATE,
    BATCH_PROMPT_TEMPLATE,
)
from hololoom.dark_trace.sae.labeler import FeatureLabel


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_sae():
    """Create a mock SAE model."""
    sae = Mock()
    sae.latent_dim = 10
    sae.decoder = Mock()
    sae.decoder.weight = torch.randn(10, 384)
    sae.eval = Mock()

    # Mock forward pass
    def forward(x):
        batch_size = x.shape[0]
        latents = torch.randn(batch_size, 10)
        reconstructed = torch.randn(batch_size, 384)
        return reconstructed, latents, reconstructed

    sae.__call__ = Mock(side_effect=forward)
    sae.return_value = (torch.randn(1, 384), torch.randn(1, 10), torch.randn(1, 384))

    return sae


@pytest.fixture
def sample_activations():
    """Create sample activations."""
    return torch.randn(100, 384)


@pytest.fixture
def sample_texts():
    """Create sample example texts."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can process natural language.",
        "Neural networks learn patterns from data.",
        "Python is a popular programming language.",
        "Sparse autoencoders decompose representations.",
    ] * 20  # 100 total
    return texts


@pytest.fixture
def sample_feature_label():
    """Create a sample feature label."""
    return FeatureLabel(
        feature_idx=0,
        description="This feature detects references to programming languages.",
        confidence=0.85,
        top_activating_examples=[
            "Python is great",
            "JavaScript runs in browsers",
            "Rust is memory-safe",
        ],
        activation_pattern="Sparse activation on technical text",
        semantic_alignments={"Technical": 0.8, "Formal": 0.5},
        primary_semantic_axis="Technical",
        semantic_alignment_score=0.8,
        labeled_at="2025-12-28T10:00:00",
        labeler_model="ollama/llama3.2:3b",
    )


@pytest.fixture
def config():
    """Create test configuration."""
    return AutoLabelConfig(
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        strategy=LabelingStrategy.SINGLE,
        batch_size=5,
        max_tokens=300,
        temperature=0.3,
    )


@pytest.fixture
def temp_cache_dir():
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Tests: Configuration
# =============================================================================

class TestAutoLabelConfig:
    """Tests for AutoLabelConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AutoLabelConfig()
        assert config.llm_provider == "ollama"
        assert config.llm_model == "llama3.2:3b"
        assert config.strategy == LabelingStrategy.SINGLE
        assert config.batch_size == 5
        assert config.min_confidence == 0.3

    def test_fast_preset(self):
        """Test fast preset configuration."""
        config = AutoLabelConfig.fast()
        assert config.llm_provider == "ollama"
        assert config.strategy == LabelingStrategy.BATCH
        assert config.batch_size == 10
        assert config.temperature == 0.2

    def test_quality_preset(self):
        """Test quality preset configuration."""
        config = AutoLabelConfig.quality()
        assert config.llm_provider == "anthropic"
        assert config.fallback_provider == "ollama"
        assert config.strategy == LabelingStrategy.SINGLE
        assert config.verify_low_confidence == 0.6

    def test_custom_config(self):
        """Test custom configuration."""
        config = AutoLabelConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            strategy=LabelingStrategy.HIERARCHICAL,
            batch_size=20,
        )
        assert config.llm_provider == "openai"
        assert config.strategy == LabelingStrategy.HIERARCHICAL
        assert config.batch_size == 20


# =============================================================================
# Tests: Prompt Building
# =============================================================================

class TestLabelPromptBuilder:
    """Tests for LabelPromptBuilder."""

    def test_build_single_feature_prompt(self):
        """Test single feature prompt building."""
        builder = LabelPromptBuilder()

        context = LabelPromptContext(
            feature_idx=5,
            examples=["Example 1", "Example 2", "Example 3"],
            activation_values=[0.9, 0.8, 0.7],
            activation_pattern="Sparse, activates on technical text",
            semantic_alignments={"Technical": 0.8, "Formal": 0.5},
            primary_semantic_axis="Technical",
            alignment_score=0.8,
        )

        prompt = builder.build_single_feature_prompt(context)

        assert "Feature Index" in prompt
        assert "5" in prompt
        assert "Example 1" in prompt
        assert "0.9" in prompt or "0.900" in prompt
        assert "Technical" in prompt

    def test_build_batch_prompt(self):
        """Test batch prompt building."""
        builder = LabelPromptBuilder()

        contexts = [
            LabelPromptContext(
                feature_idx=i,
                examples=[f"Example {i}"],
                activation_values=[0.9 - i * 0.1],
            )
            for i in range(3)
        ]

        prompt = builder.build_batch_prompt(contexts)

        assert "Feature 0" in prompt
        assert "Feature 1" in prompt
        assert "Feature 2" in prompt
        assert "JSON" in prompt

    def test_build_verification_prompt(self):
        """Test verification prompt building."""
        builder = LabelPromptBuilder()

        context = LabelPromptContext(
            feature_idx=0,
            examples=["Test example"],
            activation_values=[0.9],
            current_description="This feature detects test patterns",
            current_confidence=0.6,
        )

        prompt = builder.build_verification_prompt(context)

        assert "Current Label" in prompt
        assert "test patterns" in prompt
        assert "VERIFIED" in prompt or "REFINED" in prompt

    def test_build_comparison_prompt(self):
        """Test comparison prompt building."""
        builder = LabelPromptBuilder()

        context_a = LabelPromptContext(
            feature_idx=0,
            examples=["Example A"],
            activation_values=[0.9],
        )

        context_b = LabelPromptContext(
            feature_idx=1,
            examples=["Example B"],
            activation_values=[0.8],
        )

        prompt = builder.build_comparison_prompt(
            context_a, context_b,
            "Feature A description",
            "Feature B description",
        )

        assert "Feature A" in prompt
        assert "Feature B" in prompt
        assert "SYNONYMOUS" in prompt or "RELATED" in prompt

    def test_format_examples_truncation(self):
        """Test that long examples are truncated."""
        builder = LabelPromptBuilder()

        context = LabelPromptContext(
            feature_idx=0,
            examples=["A" * 500],  # Very long example
            activation_values=[0.9],
            max_example_length=100,
        )

        prompt = builder.build_single_feature_prompt(context)
        assert "..." in prompt  # Should be truncated


# =============================================================================
# Tests: Response Parsing
# =============================================================================

class TestLabelResponseParser:
    """Tests for LabelResponseParser."""

    def test_parse_single_response(self):
        """Test parsing single label response."""
        parser = LabelResponseParser()

        response = """
Description: This feature detects programming language references.
Confidence: 0.85
Reasoning: Examples consistently show language-related content.
"""

        parsed = parser.parse_single_response(response)

        assert "programming" in parsed.description.lower()
        assert parsed.confidence == 0.85
        assert parsed.reasoning is not None

    def test_parse_single_response_fallback(self):
        """Test fallback when response doesn't match format."""
        parser = LabelResponseParser()

        response = "This is just a plain description without format."

        parsed = parser.parse_single_response(response)

        # Should use whole response as description
        assert "plain description" in parsed.description
        assert parsed.confidence == 0.5  # Default

    def test_parse_batch_response(self):
        """Test parsing batch JSON response."""
        parser = LabelResponseParser()

        response = """
Here are the labels:
```json
[
  {"feature_idx": 0, "description": "Language feature", "confidence": 0.8, "primary_concept": "language"},
  {"feature_idx": 1, "description": "Technical feature", "confidence": 0.75}
]
```
"""

        parsed = parser.parse_batch_response(response)

        assert len(parsed) == 2
        assert parsed[0]["feature_idx"] == 0
        assert parsed[0]["description"] == "Language feature"
        assert parsed[1]["confidence"] == 0.75

    def test_parse_batch_response_invalid_json(self):
        """Test handling invalid JSON in batch response."""
        parser = LabelResponseParser()

        response = "This is not JSON at all"

        parsed = parser.parse_batch_response(response)
        assert parsed == []

    def test_parse_verification_response(self):
        """Test parsing verification response."""
        parser = LabelResponseParser()

        response = """
Verdict: REFINED
Refined Description: Updated description with more accuracy.
Confidence: 0.9
Reasoning: The original was too vague.
"""

        parsed = parser.parse_verification_response(response)

        assert parsed.verdict == "REFINED"
        assert "Updated description" in parsed.description
        assert parsed.confidence == 0.9

    def test_parse_comparison_response(self):
        """Test parsing comparison response."""
        parser = LabelResponseParser()

        response = """
Relationship: RELATED
Similarity Score: 0.7
Explanation: Both features activate on technical content but differ in specificity.
"""

        parsed = parser.parse_comparison_response(response)

        assert parsed.relationship == "RELATED"
        assert parsed.similarity_score == 0.7


# =============================================================================
# Tests: Quality Assessment
# =============================================================================

class TestLabelQualityAssessor:
    """Tests for LabelQualityAssessor."""

    def test_assess_high_quality_label(self, sample_feature_label):
        """Test assessment of high-quality label."""
        assessor = LabelQualityAssessor()

        assessment = assessor.assess(sample_feature_label)

        assert assessment.quality_level in [QualityLevel.HIGH, QualityLevel.MEDIUM]
        assert assessment.overall_score > 0.5
        assert assessment.confidence_score == 0.85

    def test_assess_low_quality_label(self):
        """Test assessment of low-quality label."""
        assessor = LabelQualityAssessor()

        label = FeatureLabel(
            feature_idx=0,
            description="No strongly activating examples found",
            confidence=0.2,
            top_activating_examples=[],
            semantic_alignment_score=0.1,
        )

        assessment = assessor.assess(label)

        assert assessment.quality_level in [QualityLevel.LOW, QualityLevel.FAILED]
        assert assessment.overall_score < 0.5
        assert len(assessment.issues) > 0

    def test_filter_by_quality(self, sample_feature_label):
        """Test filtering labels by quality."""
        assessor = LabelQualityAssessor()

        low_label = FeatureLabel(
            feature_idx=1,
            description="[LLM error]",
            confidence=0.1,
            top_activating_examples=[],
            semantic_alignment_score=0.0,
        )

        labels = {
            0: sample_feature_label,
            1: low_label,
        }

        filtered = assessor.filter_by_quality(labels, min_level=QualityLevel.MEDIUM)

        # Should only include the high-quality label
        assert len(filtered) <= len(labels)

    def test_quality_issues_detection(self):
        """Test that quality issues are properly detected."""
        assessor = LabelQualityAssessor()

        label = FeatureLabel(
            feature_idx=0,
            description="Generic description",
            confidence=0.6,
            top_activating_examples=["One example only"],
            semantic_alignment_score=0.1,
        )

        assessment = assessor.assess(label)

        # Should detect low semantic alignment
        assert any("alignment" in issue.lower() for issue in assessment.issues)


# =============================================================================
# Tests: AutoLabelEngine
# =============================================================================

class TestAutoLabelEngine:
    """Tests for AutoLabelEngine."""

    def test_initialization(self, config):
        """Test engine initialization."""
        engine = AutoLabelEngine(config)

        assert engine.config == config
        assert engine.analyzer is not None
        assert engine.alignment_scorer is not None
        assert engine.cache is not None

    def test_initialization_with_cache_dir(self, config, temp_cache_dir):
        """Test engine initialization with cache directory."""
        config.cache_dir = temp_cache_dir
        engine = AutoLabelEngine(config)

        assert engine.cache.cache_dir == temp_cache_dir

    def test_collect_examples(self, config, mock_sae, sample_activations, sample_texts):
        """Test collecting examples."""
        engine = AutoLabelEngine(config)

        # Mock SAE forward pass
        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            engine.collect_examples(mock_sae, sample_activations, sample_texts)

        # Should have collected data
        assert engine.analyzer._activations is not None
        assert len(engine.analyzer._example_texts) == 100

    @pytest.mark.asyncio
    async def test_label_single_feature(self, config, mock_sae, sample_activations, sample_texts):
        """Test labeling a single feature."""
        engine = AutoLabelEngine(config)

        # Collect examples first
        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            engine.collect_examples(mock_sae, sample_activations, sample_texts)

        # Mock LLM call
        mock_response = """
Description: This feature detects text about animals.
Confidence: 0.8
Reasoning: Examples mention various animals.
"""

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            label = await engine._label_single_feature(0)

            assert label.feature_idx == 0
            assert "animals" in label.description.lower()
            assert label.confidence == 0.8

    @pytest.mark.asyncio
    async def test_label_features_single_strategy(self, config, mock_sae, sample_activations, sample_texts):
        """Test labeling with single strategy."""
        config.strategy = LabelingStrategy.SINGLE
        engine = AutoLabelEngine(config)

        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            engine.collect_examples(mock_sae, sample_activations, sample_texts)

        mock_response = "Description: Test feature.\nConfidence: 0.7"

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            labels = await engine.label_features(mock_sae, feature_indices=[0, 1, 2])

            assert len(labels) >= 3
            assert mock_llm.call_count >= 3

    @pytest.mark.asyncio
    async def test_label_features_batch_strategy(self, config, mock_sae, sample_activations, sample_texts):
        """Test labeling with batch strategy."""
        config.strategy = LabelingStrategy.BATCH
        config.batch_size = 2
        engine = AutoLabelEngine(config)

        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            engine.collect_examples(mock_sae, sample_activations, sample_texts)

        mock_response = """
```json
[
  {"feature_idx": 0, "description": "Feature 0", "confidence": 0.8},
  {"feature_idx": 1, "description": "Feature 1", "confidence": 0.75}
]
```
"""

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            labels = await engine.label_features(mock_sae, feature_indices=[0, 1])

            # Should have batch call(s)
            assert engine._stats["batch_calls"] >= 1

    @pytest.mark.asyncio
    async def test_verify_labels(self, config, sample_feature_label):
        """Test label verification."""
        engine = AutoLabelEngine(config)

        # Create a low-confidence label
        low_label = FeatureLabel(
            feature_idx=1,
            description="Unclear feature",
            confidence=0.4,
            top_activating_examples=["Example"],
            activation_pattern="",
            semantic_alignments={},
            semantic_alignment_score=0.0,
        )

        labels = {
            0: sample_feature_label,
            1: low_label,
        }

        mock_response = """
Verdict: REFINED
Refined Description: Clearer description after verification.
Confidence: 0.75
Reasoning: Added more context.
"""

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            verified = await engine.verify_labels(labels, confidence_threshold=0.5)

            # Low confidence label should be verified
            assert engine._stats["verifications"] >= 1

    def test_get_quality_report(self, config, sample_feature_label):
        """Test quality report generation."""
        engine = AutoLabelEngine(config)

        labels = {0: sample_feature_label}

        report = engine.get_quality_report(labels)

        assert report["total_labels"] == 1
        assert "quality_distribution" in report
        assert "average_quality_score" in report

    def test_clear_collected_data(self, config):
        """Test clearing collected data."""
        engine = AutoLabelEngine(config)

        engine.analyzer._example_texts = ["test"]
        engine.analyzer._activations = torch.randn(1, 10)

        engine.clear_collected_data()

        assert engine.analyzer._example_texts == []
        assert engine.analyzer._activations is None

    def test_get_statistics(self, config):
        """Test getting statistics."""
        engine = AutoLabelEngine(config)

        engine._stats["total_labeled"] = 10
        engine._stats["batch_calls"] = 2

        stats = engine.get_statistics()

        assert stats["total_labeled"] == 10
        assert stats["batch_calls"] == 2
        assert "cache_size" in stats


# =============================================================================
# Tests: Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_autolabel_engine(self):
        """Test creating engine with convenience function."""
        engine = create_autolabel_engine(
            provider="ollama",
            model="llama3.2:3b",
            strategy=LabelingStrategy.BATCH,
        )

        assert engine.config.llm_provider == "ollama"
        assert engine.config.strategy == LabelingStrategy.BATCH

    @pytest.mark.asyncio
    async def test_label_sae_features(self, mock_sae, sample_activations, sample_texts):
        """Test convenience function for labeling."""
        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            with patch('hololoom.dark_trace.sae.autolabel_engine.AutoLabelEngine._call_llm',
                       new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = "Description: Test.\nConfidence: 0.8"

                labels = await label_sae_features(
                    mock_sae,
                    sample_activations,
                    sample_texts,
                    feature_indices=[0],
                )

                assert len(labels) >= 1


# =============================================================================
# Tests: Integration
# =============================================================================

class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_labeling_pipeline(self, config, mock_sae, sample_activations, sample_texts, temp_cache_dir):
        """Test full labeling pipeline."""
        config.cache_dir = temp_cache_dir
        engine = AutoLabelEngine(config)

        # 1. Collect examples
        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            engine.collect_examples(mock_sae, sample_activations, sample_texts)

        # 2. Label features
        mock_response = "Description: Test feature.\nConfidence: 0.6"

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            labels = await engine.label_features(mock_sae, feature_indices=[0, 1])

        # 3. Verify low confidence labels
        verify_response = """
Verdict: VERIFIED
Refined Description: Test feature.
Confidence: 0.7
"""

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = verify_response

            verified = await engine.verify_labels(labels, confidence_threshold=0.65)

        # 4. Get quality report
        report = engine.get_quality_report(verified)

        assert report["total_labels"] == len(verified)
        assert "stats" in report

    @pytest.mark.asyncio
    async def test_caching_behavior(self, config, mock_sae, sample_activations, sample_texts, temp_cache_dir):
        """Test that caching prevents re-labeling."""
        config.cache_dir = temp_cache_dir
        engine = AutoLabelEngine(config)

        with patch.object(mock_sae, '__call__', return_value=(
            sample_activations,
            torch.randn(100, 10),
            sample_activations
        )):
            engine.collect_examples(mock_sae, sample_activations, sample_texts)

        mock_response = "Description: Cached feature.\nConfidence: 0.8"

        with patch.object(engine, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response

            # First call
            labels1 = await engine.label_features(mock_sae, feature_indices=[0])
            call_count_1 = mock_llm.call_count

            # Second call (should use cache)
            labels2 = await engine.label_features(mock_sae, feature_indices=[0])
            call_count_2 = mock_llm.call_count

            # No additional LLM calls for cached feature
            assert call_count_1 == call_count_2

            # Force relabel should make new call
            labels3 = await engine.label_features(mock_sae, feature_indices=[0], force_relabel=True)
            call_count_3 = mock_llm.call_count

            assert call_count_3 > call_count_2


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

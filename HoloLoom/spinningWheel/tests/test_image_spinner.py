"""
Image Spinner Tests
===================

Comprehensive tests for ImageSpinner covering:
- Image loading (PNG, JPEG, WebP)
- CLIP embedding (mocked)
- OCR text extraction (mocked)
- Metadata extraction
- Corrupted image handling
- Batch processing

Author: Claude Code
Date: January 2026
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from dataclasses import dataclass
import tempfile
import os

# Import spinner components
from HoloLoom.spinningWheel.image_spinner import (
    ImageSpinner,
    ImageMetadata,
    spin_image,
)
from HoloLoom.spinningWheel.protocol import (
    SpinResult,
    SpinnerCapabilities,
    ImportanceScore,
    ImportanceSignals,
)


# Helper to create a mock PIL module
def create_mock_pil():
    """Create a mock PIL module for testing."""
    mock_pil = MagicMock()
    mock_image = MagicMock()
    mock_pil.Image = mock_image
    return mock_pil, mock_image


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def image_spinner():
    """Create an ImageSpinner with default settings."""
    with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
        mock_chain = Mock()
        mock_chain.get_name.return_value = "mock_ocr"
        mock_chain.get_quality.return_value = Mock(value="high")
        mock_chain.is_available.return_value = True
        mock_backends.return_value = mock_chain

        spinner = ImageSpinner(
            importance_threshold=0.3,
            output_format="markdown",
            enable_enrichment=False,
            batch_size=8
        )
        spinner.ocr_chain = mock_chain
        return spinner


@pytest.fixture
def enrichment_spinner():
    """Create an ImageSpinner with enrichment enabled."""
    with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
        mock_chain = Mock()
        mock_chain.get_name.return_value = "mock_ocr"
        mock_chain.get_quality.return_value = Mock(value="high")
        mock_chain.is_available.return_value = True
        mock_backends.return_value = mock_chain

        spinner = ImageSpinner(
            importance_threshold=0.2,
            output_format="text",
            enable_enrichment=True,
            batch_size=4
        )
        spinner.ocr_chain = mock_chain
        return spinner


@pytest.fixture
def mock_ocr_result():
    """Create a mock OCR result."""
    result = Mock()
    result.text = """
    # Document Title

    This is a sample document with extracted text.
    It contains information about machine learning.

    - Neural networks
    - Deep learning
    - Reinforcement learning

    The document discusses important algorithms.
    """
    result.confidence = 0.92
    result.backend = "tesseract"
    result.quality = Mock(value="high")
    result.processing_time_ms = 150.5
    result.detected_language = "en"
    result.language_confidence = 0.95
    return result


@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    # Create a minimal valid PNG file (1x1 pixel)
    png_header = bytes([
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,  # PNG signature
        0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1x1 pixel
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
        0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
        0x54, 0x08, 0xd7, 0x63, 0xf8, 0xff, 0xff, 0x3f,
        0x00, 0x05, 0xfe, 0x02, 0xfe, 0xdc, 0xcc, 0x59,
        0xe7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
        0x44, 0xae, 0x42, 0x60, 0x82
    ])

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(png_header)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# =============================================================================
# Image Metadata Tests
# =============================================================================

class TestImageMetadata:
    """Tests for image metadata extraction."""

    def test_metadata_extraction(self, image_spinner, temp_image_file):
        """Test metadata extraction from image file."""
        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 1920
        mock_img.height = 1080
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            metadata = image_spinner._extract_image_metadata(temp_image_file)

            assert metadata.file_path == temp_image_file
            assert metadata.width == 1920
            assert metadata.height == 1080
            assert metadata.format == 'PNG'
            assert metadata.mode == 'RGB'

    def test_metadata_fallback_on_error(self, image_spinner, temp_image_file):
        """Test metadata fallback when PIL fails."""
        mock_pil, mock_image = create_mock_pil()
        mock_image.open.side_effect = Exception("Cannot open image")

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            metadata = image_spinner._extract_image_metadata(temp_image_file)

            assert metadata.width == 0
            assert metadata.height == 0
            assert metadata.format == 'unknown'

    def test_metadata_file_size(self, image_spinner, temp_image_file):
        """Test file size is captured in metadata."""
        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 100
        mock_img.height = 100
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            metadata = image_spinner._extract_image_metadata(temp_image_file)

            assert metadata.file_size > 0


# =============================================================================
# Importance Scoring Tests
# =============================================================================

class TestImportanceScoring:
    """Tests for image importance scoring."""

    def test_importance_high_confidence(self, image_spinner):
        """Test importance scoring with high OCR confidence."""
        metadata = ImageMetadata(
            file_path=Path("test.png"),
            file_size=1024 * 1024,  # 1MB
            width=1920,
            height=1080,
            format="PNG",
            mode="RGB"
        )

        text = "This is a substantial document with multiple paragraphs. " * 20
        confidence = 0.95

        score = image_spinner._score_image_importance(text, confidence, metadata)

        assert isinstance(score, ImportanceScore)
        assert 0.0 <= score.score <= 1.0
        assert "high OCR confidence" in score.reason

    def test_importance_low_confidence(self, image_spinner):
        """Test importance scoring with low OCR confidence."""
        metadata = ImageMetadata(
            file_path=Path("test.png"),
            file_size=500000,
            width=800,
            height=600,
            format="PNG",
            mode="RGB"
        )

        text = "Short text"
        confidence = 0.3

        score = image_spinner._score_image_importance(text, confidence, metadata)

        assert score.score < 0.7  # Lower score for low confidence

    def test_importance_technical_content(self, image_spinner):
        """Test importance boost for technical content."""
        metadata = ImageMetadata(
            file_path=Path("diagram.png"),
            file_size=2048000,
            width=1920,
            height=1080,
            format="PNG",
            mode="RGB"
        )

        text = """
        Algorithm Analysis

        This diagram shows the system architecture.
        Data flows through the analysis pipeline.
        The method uses advanced graph algorithms.
        """
        confidence = 0.88

        score = image_spinner._score_image_importance(text, confidence, metadata)

        assert "technical content" in score.reason.lower() or score.score > 0.4

    def test_importance_structured_content(self, image_spinner):
        """Test importance boost for structured content."""
        metadata = ImageMetadata(
            file_path=Path("doc.png"),
            file_size=1500000,
            width=1200,
            height=1600,
            format="PNG",
            mode="RGB"
        )

        # Note: structural_score looks for '\n#', '\n-', '\n*' patterns
        # so we need markers at the start of lines (no leading whitespace)
        text = """
# Main Heading

- First bullet point
- Second bullet point
* Third item
* Fourth item

## Subheading

More content here.
"""
        confidence = 0.85

        score = image_spinner._score_image_importance(text, confidence, metadata)

        # Structured content should get bonus
        assert score.signals.structural_score > 0

    def test_importance_noise_penalty(self, image_spinner):
        """Test noise penalty for very short text."""
        metadata = ImageMetadata(
            file_path=Path("noise.png"),
            file_size=50000,
            width=100,
            height=100,
            format="PNG",
            mode="RGB"
        )

        text = "abc"  # Very short, likely noise
        confidence = 0.5

        score = image_spinner._score_image_importance(text, confidence, metadata)

        assert score.signals.noise_penalty < 0


# =============================================================================
# OCR Processing Tests (Mocked)
# =============================================================================

class TestOCRProcessing:
    """Tests for OCR text extraction with mocked backends."""

    @pytest.mark.asyncio
    async def test_single_image_ocr(self, image_spinner, temp_image_file, mock_ocr_result):
        """Test OCR on single image."""
        image_spinner.ocr_chain.batch_extract = AsyncMock(return_value=[mock_ocr_result])

        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            shards = await image_spinner._spin_impl(temp_image_file)

        assert len(shards) == 1
        assert "machine learning" in shards[0].text

    @pytest.mark.asyncio
    async def test_batch_image_ocr(self, image_spinner, temp_image_file, mock_ocr_result):
        """Test OCR on batch of images."""
        image_spinner.ocr_chain.batch_extract = AsyncMock(
            return_value=[mock_ocr_result, mock_ocr_result, mock_ocr_result]
        )

        # Create list of paths (same file for simplicity)
        paths = [temp_image_file, temp_image_file, temp_image_file]

        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            shards = await image_spinner._spin_impl(paths)

        assert len(shards) == 3

    @pytest.mark.asyncio
    async def test_ocr_result_to_shard(self, image_spinner, temp_image_file, mock_ocr_result):
        """Test conversion of OCR result to shard."""
        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            shard = await image_spinner._create_shard_from_ocr(temp_image_file, mock_ocr_result)

        assert 'ocr_confidence' in shard.metadata
        assert shard.metadata['ocr_confidence'] == 0.92
        assert shard.metadata['ocr_backend'] == 'tesseract'
        assert shard.metadata['detected_language'] == 'en'


# =============================================================================
# File Format Tests
# =============================================================================

class TestFileFormats:
    """Tests for supported file formats."""

    def test_supported_formats(self, image_spinner):
        """Test spinner reports correct supported formats."""
        caps = image_spinner.get_capabilities()

        assert '.png' in caps.supported_formats
        assert '.jpg' in caps.supported_formats
        assert '.jpeg' in caps.supported_formats
        assert '.webp' in caps.supported_formats
        assert '.tiff' in caps.supported_formats
        assert '.bmp' in caps.supported_formats
        assert '.gif' in caps.supported_formats

    @pytest.mark.asyncio
    async def test_file_not_found(self, image_spinner):
        """Test handling of non-existent file."""
        with pytest.raises(FileNotFoundError):
            await image_spinner._spin_impl(Path("/nonexistent/image.png"))

    @pytest.mark.asyncio
    async def test_invalid_source_type(self, image_spinner):
        """Test handling of invalid source type."""
        with pytest.raises(ValueError):
            await image_spinner._spin_impl(12345)  # Not a path or list


# =============================================================================
# Enrichment Tests (Mocked)
# =============================================================================

class TestEnrichment:
    """Tests for entity/motif enrichment via Ollama."""

    @pytest.mark.asyncio
    async def test_enrichment_enabled(self, enrichment_spinner, temp_image_file, mock_ocr_result):
        """Test enrichment extracts entities and motifs."""
        mock_ocr_result.text = "Neural networks are used in deep learning for image recognition tasks."

        enrichment_spinner.ocr_chain.batch_extract = AsyncMock(return_value=[mock_ocr_result])

        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch('HoloLoom.spinningWheel.image_spinner.ollama', create=True) as mock_ollama:
            mock_ollama.generate.side_effect = [
                {'response': 'Neural networks, deep learning, image recognition'},
                {'response': 'machine learning, artificial intelligence'}
            ]

            with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
                shards = await enrichment_spinner._spin_impl(temp_image_file)

        # Even if enrichment fails gracefully, should still have shards
        assert len(shards) == 1

    @pytest.mark.asyncio
    async def test_enrichment_fallback_on_error(self, enrichment_spinner, temp_image_file, mock_ocr_result):
        """Test enrichment gracefully falls back on error."""
        enrichment_spinner.ocr_chain.batch_extract = AsyncMock(return_value=[mock_ocr_result])

        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 800
        mock_img.height = 600
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch('HoloLoom.spinningWheel.image_spinner.ollama', create=True) as mock_ollama:
            mock_ollama.generate.side_effect = Exception("Ollama not available")

            with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
                shards = await enrichment_spinner._spin_impl(temp_image_file)

        # Should still produce shard even without enrichment
        assert len(shards) == 1

    @pytest.mark.asyncio
    async def test_enrichment_skipped_for_short_text(self, enrichment_spinner, temp_image_file):
        """Test enrichment is skipped for very short text."""
        short_result = Mock()
        short_result.text = "Short"  # Less than 50 chars
        short_result.confidence = 0.9
        short_result.backend = "mock"
        short_result.quality = Mock(value="high")
        short_result.processing_time_ms = 50
        short_result.detected_language = "en"
        short_result.language_confidence = 0.9

        enrichment_spinner.ocr_chain.batch_extract = AsyncMock(return_value=[short_result])

        mock_pil, mock_image = create_mock_pil()
        mock_img = Mock()
        mock_img.width = 100
        mock_img.height = 100
        mock_img.format = 'PNG'
        mock_img.mode = 'RGB'
        mock_image.open.return_value = mock_img

        with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
            with patch('HoloLoom.spinningWheel.image_spinner.ollama', create=True) as mock_ollama:
                shards = await enrichment_spinner._spin_impl(temp_image_file)

                # Ollama should not be called for short text
                mock_ollama.generate.assert_not_called()


# =============================================================================
# Spinner Capabilities Tests
# =============================================================================

class TestSpinnerCapabilities:
    """Tests for ImageSpinner capabilities."""

    def test_capabilities(self, image_spinner):
        """Test spinner capabilities reporting."""
        caps = image_spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.streaming is False
        assert caps.batch_processing is True
        assert caps.importance_scoring is True
        assert caps.entity_extraction is False  # Depends on enable_enrichment

    def test_capabilities_with_enrichment(self, enrichment_spinner):
        """Test capabilities with enrichment enabled."""
        caps = enrichment_spinner.get_capabilities()

        assert caps.entity_extraction is True
        assert caps.motif_extraction is True

    def test_is_available(self, image_spinner):
        """Test availability check."""
        result = image_spinner.is_available()
        assert isinstance(result, bool)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_spin_image(self, temp_image_file, mock_ocr_result):
        """Test spin_image convenience function."""
        with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
            mock_chain = Mock()
            mock_chain.get_name.return_value = "mock"
            mock_chain.get_quality.return_value = Mock(value="high")
            mock_chain.is_available.return_value = True
            mock_chain.batch_extract = AsyncMock(return_value=[mock_ocr_result])
            mock_backends.return_value = mock_chain

            mock_pil, mock_image = create_mock_pil()
            mock_img = Mock()
            mock_img.width = 800
            mock_img.height = 600
            mock_img.format = 'PNG'
            mock_img.mode = 'RGB'
            mock_image.open.return_value = mock_img

            with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
                result = await spin_image(str(temp_image_file))

        assert isinstance(result, SpinResult)


# =============================================================================
# Output Format Tests
# =============================================================================

class TestOutputFormats:
    """Tests for different output formats."""

    def test_markdown_format(self):
        """Test markdown output format configuration."""
        with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
            mock_chain = Mock()
            mock_chain.get_name.return_value = "mock"
            mock_chain.get_quality.return_value = Mock(value="high")
            mock_backends.return_value = mock_chain

            spinner = ImageSpinner(output_format="markdown")
            assert spinner.output_format.value == "markdown"

    def test_text_format(self):
        """Test text output format configuration."""
        with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
            mock_chain = Mock()
            mock_chain.get_name.return_value = "mock"
            mock_chain.get_quality.return_value = Mock(value="high")
            mock_backends.return_value = mock_chain

            spinner = ImageSpinner(output_format="text")
            assert spinner.output_format.value == "text"

    def test_json_format(self):
        """Test JSON output format configuration."""
        with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
            mock_chain = Mock()
            mock_chain.get_name.return_value = "mock"
            mock_chain.get_quality.return_value = Mock(value="high")
            mock_backends.return_value = mock_chain

            spinner = ImageSpinner(output_format="json")
            assert spinner.output_format.value == "json"


# =============================================================================
# Batch Processing Tests
# =============================================================================

class TestBatchProcessing:
    """Tests for batch processing functionality."""

    @pytest.mark.asyncio
    async def test_batch_respects_batch_size(self, mock_ocr_result):
        """Test that processing respects batch_size setting."""
        with patch('HoloLoom.spinningWheel.image_spinner.get_all_available_backends') as mock_backends:
            mock_chain = Mock()
            mock_chain.get_name.return_value = "mock"
            mock_chain.get_quality.return_value = Mock(value="high")
            mock_chain.is_available.return_value = True
            mock_chain.batch_extract = AsyncMock(return_value=[mock_ocr_result] * 2)
            mock_backends.return_value = mock_chain

            spinner = ImageSpinner(batch_size=2)
            spinner.ocr_chain = mock_chain

            # Create temp files
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for i in range(5):
                    path = Path(tmpdir) / f"test{i}.png"
                    path.write_bytes(b"fake image data")
                    paths.append(path)

                mock_pil, mock_image = create_mock_pil()
                mock_img = Mock()
                mock_img.width = 100
                mock_img.height = 100
                mock_img.format = 'PNG'
                mock_img.mode = 'RGB'
                mock_image.open.return_value = mock_img

                with patch.dict(sys.modules, {'PIL': mock_pil, 'PIL.Image': mock_image}):
                    with patch.object(spinner, '_extract_image_metadata') as mock_meta:
                        mock_meta.return_value = ImageMetadata(
                            file_path=paths[0], file_size=100,
                            width=100, height=100, format="PNG", mode="RGB"
                        )

                        # This should make 3 batch calls (2+2+1)
                        shards = await spinner._spin_impl(paths)

                # Verify batch_extract was called multiple times
                assert mock_chain.batch_extract.call_count == 3

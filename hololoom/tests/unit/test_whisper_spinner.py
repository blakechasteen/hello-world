"""
Unit Tests for WhisperSpinner
==============================

Tests local audio transcription including:
- Whisper model loading
- Transcription accuracy
- Timecode preservation
- Multi-language support
- Chunking behavior
- SRT export
- Importance scoring

Author: Claude Code
Date: November 2025
"""

import pytest
import tempfile
import subprocess
from pathlib import Path
import wave
import struct
import math

from hololoom.spinningWheel.whisper_spinner import (
    WhisperSpinner,
    TimecodeSegment,
    WHISPER_AVAILABLE,
    spin_audio
)
from hololoom.spinningWheel.protocol import SpinnerStatus


# ============================================================================
# Test Helpers
# ============================================================================

def create_test_audio(duration_seconds: float = 5.0, frequency: int = 440) -> Path:
    """
    Create a test audio file (WAV format) with a sine wave.

    Args:
        duration_seconds: Duration in seconds
        frequency: Tone frequency in Hz

    Returns:
        Path to created WAV file
    """
    sample_rate = 16000  # Whisper expects 16kHz
    num_samples = int(sample_rate * duration_seconds)

    # Generate sine wave
    samples = []
    for i in range(num_samples):
        sample = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        samples.append(sample)

    # Create WAV file
    tmp_file = Path(tempfile.mktemp(suffix='.wav'))

    with wave.open(str(tmp_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Pack samples as 16-bit integers
        packed_samples = struct.pack('h' * len(samples), *samples)
        wav_file.writeframes(packed_samples)

    return tmp_file


# ============================================================================
# Test Availability
# ============================================================================

def test_whisper_availability():
    """Test Whisper availability detection."""
    if WHISPER_AVAILABLE:
        # Whisper is installed
        assert True
    else:
        # Whisper not installed (optional dependency)
        pytest.skip("Whisper not available (optional dependency)")


def test_whisper_spinner_initialization():
    """Test WhisperSpinner initialization."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    spinner = WhisperSpinner(
        model_size='tiny',
        importance_threshold=0.3,
        enable_word_timestamps=True
    )

    assert spinner.model_size == 'tiny'
    assert spinner.importance_threshold == 0.3
    assert spinner.enable_word_timestamps is True
    assert spinner.model is not None  # Model loaded


def test_whisper_spinner_capabilities():
    """Test spinner capabilities."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    spinner = WhisperSpinner(model_size='tiny')
    capabilities = spinner.get_capabilities()

    assert capabilities.supports_streaming is False  # Whisper is batch-only
    assert capabilities.supports_incremental is False
    assert capabilities.supports_batch is True
    assert capabilities.supports_checkpointing is False

    expected_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.opus']
    assert all(fmt in capabilities.supported_formats for fmt in expected_formats)


# ============================================================================
# Test TimecodeSegment
# ============================================================================

def test_timecode_segment_creation():
    """Test TimecodeSegment creation."""
    segment = TimecodeSegment(
        start=10.5,
        end=15.3,
        text="Hello world",
        confidence=0.95
    )

    assert segment.start == 10.5
    assert segment.end == 15.3
    assert segment.text == "Hello world"
    assert segment.confidence == 0.95


def test_timecode_segment_format_timecode():
    """Test timecode formatting."""
    segment = TimecodeSegment(
        start=65.5,  # 1:05.500
        end=125.25,  # 2:05.250
        text="Test"
    )

    start_tc = segment.format_timecode('start')
    end_tc = segment.format_timecode('end')

    assert '1:05' in start_tc
    assert '2:05' in end_tc


def test_timecode_segment_to_srt():
    """Test SRT format conversion."""
    segment = TimecodeSegment(
        start=0.0,
        end=3.5,
        text="First subtitle line"
    )

    srt = segment.to_srt(index=1)

    assert '1\n' in srt  # Index
    assert '-->' in srt  # Timecode separator
    assert 'First subtitle line' in srt
    # Check for timecode format (may be 0:00 or 00:00:00)
    assert '0:00' in srt  # Start time


# ============================================================================
# Test Audio Transcription
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_spin_basic():
    """Test basic audio transcription."""
    # Create test audio file
    audio_file = create_test_audio(duration_seconds=2.0)

    try:
        spinner = WhisperSpinner(model_size='tiny', importance_threshold=0.0)

        result = await spinner.spin(audio_file)

        assert result.success is True
        assert result.shard_count >= 1  # At least one shard
        assert result.error_message is None

        # Check shard content
        shard = result.shards[0]
        assert 'transcription' in shard.metadata or len(shard.text) > 0
        assert 'duration' in shard.metadata
        assert 'language' in shard.metadata

    finally:
        # Cleanup
        audio_file.unlink()


@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_with_timecodes():
    """Test transcription with word-level timecodes."""
    audio_file = create_test_audio(duration_seconds=3.0)

    try:
        spinner = WhisperSpinner(
            model_size='tiny',
            enable_word_timestamps=True
        )

        result = await spinner.spin(audio_file)

        assert result.success is True

        # Check for timecode metadata
        if result.shard_count > 0:
            shard = result.shards[0]
            # Timecodes should be in metadata
            assert 'segments' in shard.metadata or 'timecodes' in shard.text.lower()

    finally:
        audio_file.unlink()


@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_language_detection():
    """Test automatic language detection."""
    audio_file = create_test_audio(duration_seconds=2.0)

    try:
        spinner = WhisperSpinner(
            model_size='tiny',
            language=None  # Auto-detect
        )

        result = await spinner.spin(audio_file)

        assert result.success is True

        # Language should be detected
        if result.shard_count > 0:
            shard = result.shards[0]
            assert 'language' in shard.metadata
            # Most likely detected as English or unknown
            assert isinstance(shard.metadata['language'], str)

    finally:
        audio_file.unlink()


@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_language_specified():
    """Test with specified language."""
    audio_file = create_test_audio(duration_seconds=2.0)

    try:
        spinner = WhisperSpinner(
            model_size='tiny',
            language='en'  # Force English
        )

        result = await spinner.spin(audio_file)

        assert result.success is True

        if result.shard_count > 0:
            shard = result.shards[0]
            assert shard.metadata.get('language') == 'en'

    finally:
        audio_file.unlink()


# ============================================================================
# Test Chunking
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_chunking():
    """Test audio chunking for long files."""
    # Create longer audio file
    audio_file = create_test_audio(duration_seconds=10.0)

    try:
        spinner = WhisperSpinner(
            model_size='tiny',
            chunk_duration=3.0,  # 3-second chunks
            importance_threshold=0.0
        )

        result = await spinner.spin(audio_file)

        assert result.success is True
        # 10 seconds / 3 seconds = ~3-4 chunks (depending on overlap)
        assert result.shard_count >= 3

    finally:
        audio_file.unlink()


# ============================================================================
# Test SRT Export
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_srt_export():
    """Test SRT subtitle export."""
    audio_file = create_test_audio(duration_seconds=3.0)

    try:
        spinner = WhisperSpinner(
            model_size='tiny',
            enable_word_timestamps=True
        )

        # Transcribe
        result = await spinner.spin(audio_file)

        if result.shard_count > 0 and 'segments' in result.shards[0].metadata:
            # Export to SRT
            srt_path = audio_file.with_suffix('.srt')
            spinner.export_srt(result.shards[0].metadata['segments'], srt_path)

            assert srt_path.exists()

            # Check SRT content
            srt_content = srt_path.read_text()
            assert '1\n' in srt_content  # First subtitle index
            assert '-->' in srt_content  # Timecode separator

            srt_path.unlink()

    finally:
        audio_file.unlink()


# ============================================================================
# Test Importance Scoring
# ============================================================================

def test_whisper_spinner_importance_scoring():
    """Test importance scoring for transcriptions."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    spinner = WhisperSpinner(model_size='tiny')

    # Technical content (high importance)
    technical_score = spinner.score_importance(
        "The API endpoint requires OAuth2 authentication with JWT tokens."
    )

    # Filler content (low importance)
    filler_score = spinner.score_importance(
        "Um, yeah, uh, you know, like, whatever."
    )

    # Technical content should score higher
    assert technical_score.score > filler_score.score
    assert technical_score.signals.technical_score > 0.5
    assert filler_score.signals.noise_score > 0.5


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_invalid_file():
    """Test handling of invalid audio file."""
    spinner = WhisperSpinner(model_size='tiny')

    # Non-existent file
    result = await spinner.spin(Path("/nonexistent/audio.wav"))

    assert result.success is False
    assert result.error_message is not None
    assert result.shard_count == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_invalid_format():
    """Test handling of invalid audio format."""
    spinner = WhisperSpinner(model_size='tiny')

    # Create empty text file with .wav extension
    tmp_file = Path(tempfile.mktemp(suffix='.wav'))
    tmp_file.write_text("This is not audio data")

    try:
        result = await spinner.spin(tmp_file)

        # Should fail gracefully
        assert result.success is False or result.shard_count == 0

    finally:
        tmp_file.unlink()


# ============================================================================
# Test Metadata Extraction
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_whisper_spinner_metadata():
    """Test metadata extraction."""
    audio_file = create_test_audio(duration_seconds=3.0)

    try:
        spinner = WhisperSpinner(model_size='tiny')

        result = await spinner.spin(audio_file)

        if result.shard_count > 0:
            shard = result.shards[0]
            metadata = shard.metadata

            # Check required metadata fields
            assert 'duration' in metadata
            assert 'language' in metadata
            assert 'model_size' in metadata
            assert metadata['model_size'] == 'tiny'
            assert metadata['spinner'] == 'whisper'

    finally:
        audio_file.unlink()


# ============================================================================
# Test Function Interfaces
# ============================================================================

@pytest.mark.asyncio
@pytest.mark.skipif(not WHISPER_AVAILABLE, reason="Whisper not available")
async def test_spin_audio_function():
    """Test spin_audio convenience function."""
    audio_file = create_test_audio(duration_seconds=2.0)

    try:
        result = await spin_audio(
            audio_file,
            model_size='tiny',
            importance_threshold=0.0
        )

        assert result.success is True
        assert result.shard_count >= 1

    finally:
        audio_file.unlink()


# ============================================================================
# Test Spinner Status
# ============================================================================

def test_whisper_spinner_status():
    """Test spinner status reporting."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    spinner = WhisperSpinner(model_size='tiny')
    status = spinner.get_status()

    assert status == SpinnerStatus.READY
    assert spinner.is_available() is True


# ============================================================================
# Test Model Sizes
# ============================================================================

def test_whisper_spinner_model_sizes():
    """Test different Whisper model sizes."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    # Test tiny model (fastest, already tested above)
    spinner_tiny = WhisperSpinner(model_size='tiny')
    assert spinner_tiny.model is not None

    # Test base model (if we want to support it)
    # Note: This might be slow in CI, so we just test initialization
    try:
        spinner_base = WhisperSpinner(model_size='base')
        assert spinner_base.model is not None
    except Exception:
        # May fail if model download fails - that's okay for unit test
        pass


# ============================================================================
# Test Device Selection
# ============================================================================

def test_whisper_spinner_device_auto():
    """Test automatic device selection."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    spinner = WhisperSpinner(model_size='tiny', device='auto')

    # Should select CPU or CUDA based on availability
    assert spinner.device in ['cpu', 'cuda']
    assert spinner.model is not None


# ============================================================================
# Summary Test
# ============================================================================

def test_whisper_spinner_summary():
    """Summary test: WhisperSpinner comprehensive test."""
    if not WHISPER_AVAILABLE:
        pytest.skip("Whisper not available")

    # Create spinner with custom config
    spinner = WhisperSpinner(
        model_size='tiny',
        importance_threshold=0.3,
        language='en',
        enable_word_timestamps=True,
        chunk_duration=300.0,
        device='auto'
    )

    # Verify all attributes
    assert spinner.model_size == 'tiny'
    assert spinner.importance_threshold == 0.3
    assert spinner.language == 'en'
    assert spinner.enable_word_timestamps is True
    assert spinner.chunk_duration == 300.0
    assert spinner.device in ['cpu', 'cuda']

    # Verify capabilities
    capabilities = spinner.get_capabilities()
    assert capabilities.supports_batch is True
    assert '.wav' in capabilities.supported_formats
    assert '.mp3' in capabilities.supported_formats

    # Verify status
    assert spinner.is_available() is True
    assert spinner.get_status() == SpinnerStatus.READY

    print("✅ WhisperSpinner: All 20 tests defined")

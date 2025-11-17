"""
HoloLoom Audio Encoder
======================
Whisper-based audio encoding with speech-to-text.

This encoder uses OpenAI's Whisper model to transcribe audio
and create embeddings from the transcribed text combined with
audio fingerprinting features.

Features:
- Whisper speech-to-text transcription
- Audio fingerprinting for similarity
- Support for MP3, WAV, FLAC, M4A formats
- Optional speaker diarization
- Language detection

Architecture:
- Input: Audio file path or waveform array
- Process:
  1. Whisper → transcribed text
  2. Text → MatryoshkaEmbeddings (768D)
  3. Audio fingerprint → 256D spectral features
  4. Concatenate → 768D (weighted combination)
- Output: ModalEmbedding with 768D vector

The hybrid approach combines semantic (text) and acoustic (spectral)
features for robust audio understanding.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Import from base module
from HoloLoom.multimodal.base import (
    Modality,
    ModalEmbedding,
    AudioEncoderConfig,
    validate_embedding_dim,
    normalize_embedding
)

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
try:
    import whisper
    _HAVE_WHISPER = True
except ImportError:
    whisper = None
    _HAVE_WHISPER = False
    warnings.warn("whisper not available. Install with: pip install openai-whisper")

try:
    import librosa
    import soundfile as sf
    _HAVE_LIBROSA = True
except ImportError:
    librosa = None
    sf = None
    _HAVE_LIBROSA = False
    warnings.warn("librosa not available. Install with: pip install librosa soundfile")

try:
    from scipy import signal as scipy_signal
    _HAVE_SCIPY = True
except ImportError:
    scipy_signal = None
    _HAVE_SCIPY = False
    # Don't warn - scipy is optional

# Import text embedder for encoding transcriptions
try:
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    _HAVE_TEXT_EMBEDDER = True
except ImportError:
    MatryoshkaEmbeddings = None
    _HAVE_TEXT_EMBEDDER = False
    warnings.warn("MatryoshkaEmbeddings not available - text encoding disabled")


# ============================================================================
# Audio Encoder Implementation
# ============================================================================

class AudioEncoder:
    """
    Whisper-based audio encoder with acoustic fingerprinting.

    Encodes audio into 768D embeddings that combine:
    1. Semantic content (via transcription → text embeddings)
    2. Acoustic features (via spectral fingerprinting)

    This hybrid approach enables both content-based search
    ("find audio about cats") and acoustic similarity
    ("find similar-sounding audio").

    Technical Details:
    - Whisper for speech-to-text (base model default)
    - MatryoshkaEmbeddings for text encoding (768D)
    - Spectral features: MFCCs, chroma, spectral contrast
    - Final: 70% text + 30% acoustic (configurable)

    Example:
        encoder = AudioEncoder()
        embedding = await encoder.encode("speech.mp3")
        # Returns ModalEmbedding with 768D vector + transcription
    """

    def __init__(self, config: Optional[AudioEncoderConfig] = None):
        """
        Initialize audio encoder.

        Args:
            config: Optional configuration (uses defaults if None)
        """
        self.config = config or AudioEncoderConfig()
        self._whisper_model = None
        self._text_embedder = None

        # Load Whisper model
        if _HAVE_WHISPER:
            try:
                logger.info(f"Loading Whisper model: {self.config.model_name}")
                self._whisper_model = whisper.load_model(
                    self.config.model_name,
                    device=self.config.device,
                    download_root=self.config.cache_dir
                )
                logger.info(f"Whisper model loaded on {self.config.device}")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self._whisper_model = None
        else:
            logger.warning("Whisper not available - transcription disabled")

        # Initialize text embedder
        if _HAVE_TEXT_EMBEDDER:
            try:
                self._text_embedder = MatryoshkaEmbeddings(sizes=[768])
            except Exception as e:
                logger.error(f"Failed to initialize text embedder: {e}")
                self._text_embedder = None

    @property
    def modality(self) -> Modality:
        """Return modality type."""
        return Modality.AUDIO

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 768

    def _load_audio(
        self,
        audio_path: Union[str, Path]
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)

        Raises:
            RuntimeError: If librosa not available
            FileNotFoundError: If audio file not found
        """
        if not _HAVE_LIBROSA:
            raise RuntimeError("librosa required for audio loading: pip install librosa soundfile")

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio not found: {path}")

        try:
            # Load audio (converts to mono, 16kHz by default)
            waveform, sample_rate = librosa.load(
                str(path),
                sr=16000,  # Whisper expects 16kHz
                mono=True
            )
            return waveform, sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio {path}: {e}")
            raise

    def _transcribe(
        self,
        waveform: np.ndarray
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Whisper.

        Args:
            waveform: Audio waveform (16kHz, mono)

        Returns:
            Dict with keys: text, language, segments

        Raises:
            RuntimeError: If Whisper not available
        """
        if not _HAVE_WHISPER or self._whisper_model is None:
            raise RuntimeError("Whisper model not available")

        try:
            # Transcribe
            result = self._whisper_model.transcribe(
                waveform,
                language=self.config.language,
                fp16=False  # Use FP32 for CPU compatibility
            )

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "language": "unknown", "segments": []}

    def _extract_acoustic_features(
        self,
        waveform: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract acoustic fingerprint features.

        Combines:
        - MFCCs (Mel-frequency cepstral coefficients) - 20D
        - Chroma features - 12D
        - Spectral contrast - 7D
        - Zero crossing rate - 1D
        - Spectral rolloff - 1D

        Total: 41D features → PCA to 256D for padding

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate

        Returns:
            256D feature vector
        """
        if not _HAVE_LIBROSA:
            # Fallback: zeros
            return np.zeros(256)

        try:
            features = []

            # MFCCs (20 coefficients)
            mfccs = librosa.feature.mfcc(
                y=waveform,
                sr=sample_rate,
                n_mfcc=20
            )
            features.append(np.mean(mfccs, axis=1))  # 20D

            # Chroma features (12 pitch classes)
            chroma = librosa.feature.chroma_stft(
                y=waveform,
                sr=sample_rate
            )
            features.append(np.mean(chroma, axis=1))  # 12D

            # Spectral contrast (7 bands)
            contrast = librosa.feature.spectral_contrast(
                y=waveform,
                sr=sample_rate
            )
            features.append(np.mean(contrast, axis=1))  # 7D

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(waveform)
            features.append([np.mean(zcr)])  # 1D

            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=waveform,
                sr=sample_rate
            )
            features.append([np.mean(rolloff)])  # 1D

            # Concatenate all features: 20+12+7+1+1 = 41D
            acoustic_vec = np.concatenate(features)

            # Pad to 256D (for future expansion)
            if len(acoustic_vec) < 256:
                acoustic_vec = np.pad(
                    acoustic_vec,
                    (0, 256 - len(acoustic_vec)),
                    constant_values=0.0
                )
            else:
                acoustic_vec = acoustic_vec[:256]

            # Normalize
            acoustic_vec = acoustic_vec / (np.linalg.norm(acoustic_vec) + 1e-9)

            return acoustic_vec

        except Exception as e:
            logger.warning(f"Acoustic feature extraction failed: {e}")
            return np.zeros(256)

    async def encode(
        self,
        input_data: Union[str, Path, np.ndarray],
        **kwargs
    ) -> ModalEmbedding:
        """
        Encode single audio file to 768D embedding.

        Args:
            input_data: Audio file path or waveform array
            **kwargs: Additional options
                - text_weight: Weight for text features (default 0.7)
                - acoustic_weight: Weight for acoustic features (default 0.3)

        Returns:
            ModalEmbedding with 768D vector + transcription

        Raises:
            RuntimeError: If required dependencies unavailable
        """
        text_weight = kwargs.get("text_weight", 0.7)
        acoustic_weight = kwargs.get("acoustic_weight", 0.3)

        # Load audio if path provided
        if isinstance(input_data, (str, Path)):
            waveform, sample_rate = self._load_audio(input_data)
            audio_path = str(input_data)
        else:
            waveform = input_data
            sample_rate = kwargs.get("sample_rate", 16000)
            audio_path = kwargs.get("path", "unknown")

        # Transcribe audio
        transcription = {"text": "", "language": "unknown", "segments": []}
        if self._whisper_model is not None:
            transcription = self._transcribe(waveform)
        else:
            logger.warning("Whisper unavailable - no transcription")

        # Extract acoustic features
        acoustic_features = self._extract_acoustic_features(waveform, sample_rate)

        # Encode transcription to text embedding
        text_embedding = np.zeros(768)
        confidence = 0.5

        if transcription["text"] and self._text_embedder is not None:
            try:
                text_emb_matrix = self._text_embedder.encode_scales(
                    [transcription["text"]],
                    size=768
                )
                text_embedding = text_emb_matrix[0]
                confidence = 1.0
            except Exception as e:
                logger.warning(f"Text embedding failed: {e}")

        # Combine text and acoustic features
        # Text: 768D, Acoustic: 256D
        # Strategy: Weighted sum of text embedding + projected acoustic
        acoustic_projection = np.pad(acoustic_features, (0, 768 - 256))
        acoustic_projection = acoustic_projection[:768]

        # Weighted fusion
        embedding_vec = (
            text_weight * text_embedding +
            acoustic_weight * acoustic_projection
        )

        # Normalize
        embedding_vec = embedding_vec / (np.linalg.norm(embedding_vec) + 1e-9)

        # Validate dimension
        validate_embedding_dim(embedding_vec.tolist(), 768)

        # Build metadata
        metadata = {
            "file": audio_path,
            "transcription": transcription["text"],
            "language": transcription["language"],
            "duration": len(waveform) / sample_rate,
            "sample_rate": sample_rate
        }

        return ModalEmbedding(
            embedding=embedding_vec.tolist(),
            modality=Modality.AUDIO,
            metadata=metadata,
            confidence=confidence
        )

    async def encode_batch(
        self,
        inputs: List[Union[str, Path, np.ndarray]],
        **kwargs
    ) -> List[ModalEmbedding]:
        """
        Encode multiple audio files efficiently.

        Args:
            inputs: List of audio paths or waveforms
            **kwargs: Additional options

        Returns:
            List of ModalEmbeddings

        Note:
            Currently processes sequentially. Future optimization:
            batch processing with Whisper for efficiency.
        """
        results = []
        for input_data in inputs:
            try:
                embedding = await self.encode(input_data, **kwargs)
                results.append(embedding)
            except Exception as e:
                logger.error(f"Failed to encode {input_data}: {e}")
                # Continue with other files
                continue

        return results


# ============================================================================
# Factory Function
# ============================================================================

def create_audio_encoder(
    model_name: str = "base",
    language: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> AudioEncoder:
    """
    Factory function to create audio encoder.

    Args:
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        language: Language code (None for auto-detect)
        device: "cpu" or "cuda"
        **kwargs: Additional config options

    Returns:
        AudioEncoder instance
    """
    config = AudioEncoderConfig(
        model_name=model_name,
        language=language,
        device=device,
        **kwargs
    )
    return AudioEncoder(config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 80)
        print("Audio Encoder Demo")
        print("=" * 80 + "\n")

        # Create encoder
        encoder = create_audio_encoder(model_name="base", device="cpu")

        print(f"Modality: {encoder.modality.value}")
        print(f"Embedding dimension: {encoder.embedding_dim}")
        print(f"Whisper available: {_HAVE_WHISPER}")
        print(f"Librosa available: {_HAVE_LIBROSA}")

        if not _HAVE_LIBROSA:
            print("\nlibrosa not available - install with: pip install librosa soundfile")
            return

        # Would encode real audio:
        # embedding = await encoder.encode("path/to/audio.mp3")
        # print(f"\nEmbedding shape: {len(embedding.embedding)}")
        # print(f"Transcription: {embedding.metadata.get('transcription')}")
        # print(f"Language: {embedding.metadata.get('language')}")
        # print(f"Duration: {embedding.metadata.get('duration'):.2f}s")

        print("\n✓ Audio encoder ready!")
        print("Usage: embedding = await encoder.encode('audio.mp3')")

    asyncio.run(demo())

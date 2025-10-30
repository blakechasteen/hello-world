"""
Audio Processor

Processes audio using speech-to-text (Whisper) and acoustic feature extraction.
"""

import time
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import numpy as np

from .protocol import (
    InputProcessorProtocol,
    ProcessedInput,
    ModalityType,
    AudioFeatures,
    InputMetadata
)


class AudioProcessor:
    """
    Audio processor using Whisper and librosa.
    
    Features:
    - Speech-to-text transcription
    - Language detection
    - Acoustic feature extraction (MFCC, pitch, energy)
    - Emotion detection from prosody
    - Speaker diarization (basic)
    """
    
    def __init__(self, use_whisper: bool = True, whisper_model: str = "base"):
        """
        Initialize audio processor.
        
        Args:
            use_whisper: Whether to use Whisper for transcription
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.use_whisper = use_whisper
        self.whisper_model_name = whisper_model
        
        # Try to load Whisper
        self.whisper_model = None
        if use_whisper:
            try:
                import whisper
                self.whisper_model = whisper.load_model(whisper_model)
            except ImportError:
                print("Warning: Whisper not available. Transcription disabled.")
                self.use_whisper = False
        
        # Try to load librosa for acoustic features
        self.librosa = None
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            print("Warning: librosa not available. Acoustic features disabled.")
    
    async def process(
        self,
        input_data: Union[str, Path, bytes, Dict],
        transcribe: bool = True,
        extract_acoustic: bool = True,
        detect_emotion: bool = True,
        **kwargs
    ) -> ProcessedInput:
        """
        Process audio input.
        
        Args:
            input_data: Audio file path, bytes, or dict with 'audio' key
            transcribe: Whether to transcribe speech
            extract_acoustic: Whether to extract acoustic features
            detect_emotion: Whether to detect emotion
        
        Returns:
            ProcessedInput with audio features
        """
        start_time = time.time()
        
        # Get audio file path
        if isinstance(input_data, dict):
            audio_path = input_data.get('audio')
            source = input_data.get('source')
        elif isinstance(input_data, bytes):
            # Save bytes to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(input_data)
                audio_path = f.name
            source = None
        else:
            audio_path = str(input_data)
            source = audio_path
        
        # Load audio
        if not self.librosa:
            raise ImportError("librosa is required for audio processing")
        
        audio, sr = self.librosa.load(audio_path, sr=16000)
        duration = len(audio) / sr
        
        # Create features
        features = AudioFeatures()
        features.duration = duration
        features.sample_rate = sr
        
        # Transcribe with Whisper
        transcript = None
        language = None
        if transcribe and self.use_whisper and self.whisper_model:
            result = self.whisper_model.transcribe(audio_path)
            transcript = result['text']
            language = result.get('language')
            features.transcript = transcript
            features.language = language
        
        # Extract acoustic features
        if extract_acoustic and self.librosa:
            features.acoustic = self._extract_acoustic_features(audio, sr)
        
        # Detect emotion from prosody
        if detect_emotion and features.acoustic:
            features.emotion = self._detect_emotion(features.acoustic)
        
        # Generate embedding (use transcript embedding if available)
        embedding = None
        if transcript:
            # For now, create simple bag-of-words embedding
            # In production, use sentence embeddings
            embedding = self._create_text_embedding(transcript)
        elif features.acoustic:
            # Use acoustic features as embedding
            mfcc = features.acoustic.get('mfcc_mean', [])
            if mfcc:
                embedding = np.array(mfcc)
        
        # Create content description
        content_parts = []
        if transcript:
            content_parts.append(f"Transcript: {transcript[:100]}")
        if language:
            content_parts.append(f"Language: {language}")
        if features.emotion:
            content_parts.append(f"Emotion: {features.emotion}")
        content_parts.append(f"Duration: {duration:.1f}s")
        
        content = " | ".join(content_parts)
        
        # Calculate confidence
        confidence = 0.9 if transcript else 0.7
        
        processing_time = (time.time() - start_time) * 1000
        
        return ProcessedInput(
            modality=ModalityType.AUDIO,
            content=content,
            embedding=embedding,
            confidence=confidence,
            source=source,
            features={'audio': features.to_dict()}
        )
    
    def _extract_acoustic_features(self, audio: np.ndarray, sr: int) -> Dict[str, any]:
        """Extract acoustic features using librosa."""
        features = {}
        
        try:
            # MFCC (Mel-frequency cepstral coefficients)
            mfcc = self.librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = mfcc.mean(axis=1).tolist()
            features['mfcc_std'] = mfcc.std(axis=1).tolist()
            
            # Pitch (F0)
            pitches, magnitudes = self.librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
            
            # Energy / RMS
            rms = self.librosa.feature.rms(y=audio)
            features['energy_mean'] = float(rms.mean())
            features['energy_std'] = float(rms.std())
            
            # Zero crossing rate
            zcr = self.librosa.feature.zero_crossing_rate(audio)
            features['zcr_mean'] = float(zcr.mean())
            
            # Spectral centroid
            spectral_centroids = self.librosa.feature.spectral_centroid(y=audio, sr=sr)
            features['spectral_centroid_mean'] = float(spectral_centroids.mean())
            
        except Exception as e:
            print(f"Warning: Error extracting acoustic features: {e}")
        
        return features
    
    def _detect_emotion(self, acoustic_features: Dict[str, Any]) -> Optional[str]:
        """
        Detect emotion from acoustic features.
        
        Simple rule-based approach. For production, use emotion recognition models.
        """
        if not acoustic_features:
            return None
        
        # Get features
        energy_mean = acoustic_features.get('energy_mean', 0)
        pitch_mean = acoustic_features.get('pitch_mean', 0)
        pitch_std = acoustic_features.get('pitch_std', 0)
        
        # Simple heuristics
        if energy_mean > 0.1 and pitch_std > 50:
            return "excited"
        elif energy_mean > 0.1:
            return "happy"
        elif energy_mean < 0.03 and pitch_mean < 150:
            return "sad"
        elif pitch_std > 70:
            return "angry"
        else:
            return "neutral"
    
    def _create_text_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """
        Create simple text embedding.
        
        For production, use proper sentence embeddings.
        """
        # Simple bag-of-words with hashing
        words = text.lower().split()
        embedding = np.zeros(dim)
        
        for word in words:
            hash_val = hash(word) % dim
            embedding[hash_val] += 1
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def get_modality(self) -> ModalityType:
        """Return modality type."""
        return ModalityType.AUDIO
    
    def is_available(self) -> bool:
        """Check if processor is available."""
        return self.librosa is not None

"""
Input Processing Protocol

Defines unified interface for multi-modal input processing.
All input processors (text, image, audio, structured) implement this protocol.
"""

from typing import Protocol, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np


class ModalityType(Enum):
    """Types of input modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"  # JSON, CSV, databases
    MULTIMODAL = "multimodal"  # Combined inputs


@dataclass
class ProcessedInput:
    """
    Unified representation of processed input.
    
    All input processors return this structure, enabling
    consistent handling across modalities.
    """
    
    # Core fields
    modality: ModalityType
    content: str  # Human-readable description
    embedding: np.ndarray  # Feature vector
    
    # Metadata
    confidence: float = 1.0  # Processing confidence (0.0-1.0)
    source: Optional[str] = None  # File path, URL, etc.
    
    # Modality-specific features
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-modal alignment
    aligned_embeddings: Dict[ModalityType, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate processed input."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")
        
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'modality': self.modality.value,
            'content': self.content,
            'embedding': self.embedding.tolist() if self.embedding is not None else None,
            'confidence': self.confidence,
            'source': self.source,
            'features': self.features,
            'aligned_embeddings': {
                k.value: v.tolist() for k, v in self.aligned_embeddings.items()
            }
        }


@dataclass
class TextFeatures:
    """Text-specific features."""
    entities: List[Dict[str, str]] = field(default_factory=list)  # NER results
    sentiment: Dict[str, float] = field(default_factory=dict)  # Polarity, subjectivity
    topics: List[str] = field(default_factory=list)  # Main topics
    keyphrases: List[str] = field(default_factory=list)  # Key phrases
    language: str = "en"
    
    def to_dict(self) -> Dict:
        return {
            'entities': self.entities,
            'sentiment': self.sentiment,
            'topics': self.topics,
            'keyphrases': self.keyphrases,
            'language': self.language
        }


@dataclass
class ImageFeatures:
    """Image-specific features."""
    objects: List[Dict[str, Any]] = field(default_factory=list)  # Detected objects
    scene: Optional[str] = None  # Scene classification
    caption: Optional[str] = None  # Generated caption
    ocr_text: Optional[str] = None  # Extracted text
    colors: List[str] = field(default_factory=list)  # Dominant colors
    dimensions: tuple = (0, 0)  # Width, height
    
    def to_dict(self) -> Dict:
        return {
            'objects': self.objects,
            'scene': self.scene,
            'caption': self.caption,
            'ocr_text': self.ocr_text,
            'colors': self.colors,
            'dimensions': self.dimensions
        }


@dataclass
class AudioFeatures:
    """Audio-specific features."""
    transcript: Optional[str] = None  # Speech-to-text
    language: Optional[str] = None  # Detected language
    speaker_count: int = 1  # Number of speakers
    emotion: Optional[str] = None  # Detected emotion
    acoustic: Dict[str, Any] = field(default_factory=dict)  # MFCC, pitch, energy
    duration: float = 0.0  # Duration in seconds
    sample_rate: int = 16000
    
    def to_dict(self) -> Dict:
        return {
            'transcript': self.transcript,
            'language': self.language,
            'speaker_count': self.speaker_count,
            'emotion': self.emotion,
            'acoustic': self.acoustic,
            'duration': self.duration,
            'sample_rate': self.sample_rate
        }


@dataclass
class StructuredFeatures:
    """Structured data features."""
    schema: Dict[str, str] = field(default_factory=dict)  # Column types
    row_count: int = 0
    column_count: int = 0
    relationships: List[Dict[str, Any]] = field(default_factory=list)  # Foreign keys, etc.
    summary_stats: Dict[str, Any] = field(default_factory=dict)  # Mean, std, etc.
    
    def to_dict(self) -> Dict:
        return {
            'schema': self.schema,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'relationships': self.relationships,
            'summary_stats': self.summary_stats
        }


class InputProcessorProtocol(Protocol):
    """
    Protocol for all input processors.
    
    Each modality (text, image, audio, structured) implements this interface.
    """
    
    async def process(
        self,
        input_data: Union[str, bytes, Path, Dict],
        **kwargs
    ) -> ProcessedInput:
        """
        Process input and return unified representation.
        
        Args:
            input_data: Raw input (text, file path, bytes, or dict)
            **kwargs: Processor-specific options
        
        Returns:
            ProcessedInput with unified structure
        """
        ...
    
    def get_modality(self) -> ModalityType:
        """Return the modality type this processor handles."""
        ...
    
    def is_available(self) -> bool:
        """Check if required dependencies are available."""
        ...


class MultiModalFusionProtocol(Protocol):
    """
    Protocol for multi-modal fusion.
    
    Combines features from multiple modalities into unified representation.
    """
    
    async def fuse(
        self,
        inputs: List[ProcessedInput],
        strategy: str = "attention"  # "attention", "concat", "average"
    ) -> ProcessedInput:
        """
        Fuse multiple modalities into unified representation.
        
        Args:
            inputs: List of processed inputs from different modalities
            strategy: Fusion strategy
        
        Returns:
            Fused ProcessedInput with MULTIMODAL modality
        """
        ...
    
    def align_embeddings(
        self,
        inputs: List[ProcessedInput]
    ) -> List[ProcessedInput]:
        """
        Align embeddings across modalities to same space.
        
        Args:
            inputs: List of processed inputs
        
        Returns:
            List of inputs with aligned_embeddings populated
        """
        ...


@dataclass
class InputMetadata:
    """
    Metadata about input processing.
    
    Used for tracking provenance and debugging.
    """
    
    modality: ModalityType
    processor: str  # Processor class name
    processing_time_ms: float
    model_used: Optional[str] = None
    confidence: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'modality': self.modality.value,
            'processor': self.processor,
            'processing_time_ms': self.processing_time_ms,
            'model_used': self.model_used,
            'confidence': self.confidence,
            'errors': self.errors,
            'warnings': self.warnings
        }


# Type aliases for convenience
InputData = Union[str, bytes, Path, Dict, Any]
ProcessorResult = ProcessedInput

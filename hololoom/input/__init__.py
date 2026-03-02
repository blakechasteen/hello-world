"""
HoloLoom Input Processing Module

Multi-modal input processing for text, images, audio, and structured data.
"""

from .protocol import (
    ModalityType,
    ProcessedInput,
    TextFeatures,
    ImageFeatures,
    AudioFeatures,
    StructuredFeatures,
    InputProcessorProtocol,
    MultiModalFusionProtocol,
    InputMetadata,
    InputData,
    ProcessorResult
)

from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .structured_processor import StructuredDataProcessor
from .fusion import MultiModalFusion
from .router import InputRouter

__all__ = [
    # Protocol types
    'ModalityType',
    'ProcessedInput',
    'TextFeatures',
    'ImageFeatures',
    'AudioFeatures',
    'StructuredFeatures',
    'InputProcessorProtocol',
    'MultiModalFusionProtocol',
    'InputMetadata',
    'InputData',
    'ProcessorResult',
    
    # Processors
    'TextProcessor',
    'ImageProcessor',
    'AudioProcessor',
    'StructuredDataProcessor',
    
    # Fusion and Routing
    'MultiModalFusion',
    'InputRouter',
]

"""
HoloLoom Input Processing Module

Multi-modal input processing for text, images, audio, and structured data.
"""

from .audio_processor import AudioProcessor
from .fusion import MultiModalFusion
from .image_processor import ImageProcessor
from .protocol import (
    AudioFeatures,
    ImageFeatures,
    InputData,
    InputMetadata,
    InputProcessorProtocol,
    ModalityType,
    MultiModalFusionProtocol,
    ProcessedInput,
    ProcessorResult,
    StructuredFeatures,
    TextFeatures,
)
from .router import InputRouter
from .structured_processor import StructuredDataProcessor
from .text_processor import TextProcessor

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

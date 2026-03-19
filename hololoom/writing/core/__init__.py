"""
Writing Core Module
===================

Core writing engine components.
"""

from .composer import Composer
from .protocol import (
    QUALITY_DIMENSIONS,
    ComposerProtocol,
    ModeWriterProtocol,
    OutputFormat,
    RefinementPass,
    RefinementStrategy,
    RefinerProtocol,
    StyleGuide,
    WriterProtocol,
    WritingContext,
    WritingMode,
    WritingResult,
)
from .writer import Writer, write

__all__ = [
    # Enums
    'WritingMode',
    'RefinementStrategy',
    'StyleGuide',
    'OutputFormat',

    # Data classes
    'WritingContext',
    'WritingResult',
    'RefinementPass',

    # Protocols
    'WriterProtocol',
    'ComposerProtocol',
    'RefinerProtocol',
    'ModeWriterProtocol',

    # Implementations
    'Writer',
    'Composer',
    'write',

    # Constants
    'QUALITY_DIMENSIONS'
]

"""
Writing Core Module
===================

Core writing engine components.
"""

from .protocol import (
    WritingMode,
    RefinementStrategy,
    StyleGuide,
    OutputFormat,
    WritingContext,
    WritingResult,
    RefinementPass,
    WriterProtocol,
    ComposerProtocol,
    RefinerProtocol,
    ModeWriterProtocol,
    QUALITY_DIMENSIONS
)
from .writer import Writer, write
from .composer import Composer

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

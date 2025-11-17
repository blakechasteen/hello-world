"""
Ingestion - File Processing Pipeline
=====================================
Drag-and-drop file ingestion with smart parsing and chunking.
"""

from .file_processor import FileProcessor, ProcessorConfig, ProcessResult
from .chunker import SmartChunker, ChunkConfig

__all__ = ['FileProcessor', 'ProcessorConfig', 'ProcessResult', 'SmartChunker', 'ChunkConfig']

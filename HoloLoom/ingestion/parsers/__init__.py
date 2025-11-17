"""
Parsers - File Format Parsers
==============================
Parsers for different file formats.
"""

# PDF and DOCX parsers
from HoloLoom.ingestion.parsers.pdf import parse_pdf
from HoloLoom.ingestion.parsers.docx import parse_docx

# Multi-modal parsers (Phase 3A)
from HoloLoom.ingestion.parsers.image import parse_image, get_image_metadata
from HoloLoom.ingestion.parsers.audio import parse_audio, get_audio_metadata
from HoloLoom.ingestion.parsers.video import parse_video, get_video_metadata

__all__ = [
    # Document parsers
    "parse_pdf",
    "parse_docx",

    # Multi-modal parsers
    "parse_image",
    "parse_audio",
    "parse_video",

    # Metadata extractors
    "get_image_metadata",
    "get_audio_metadata",
    "get_video_metadata",
]

"""
HoloLoom SpinningWheel
=======================
Universal data ingestion - everything becomes a memory operation.

Philosophy: "If you need to configure it, we failed."

Ruthlessly Elegant API:
    from HoloLoom.spinningWheel import spin

    # Ingest anything into memory
    memory = await spin(anything)

    # Text, URLs, files, structured data, multi-modal - all automatic
"""

# Ruthlessly Elegant Primary API
from .auto import (
    spin,             # THE function - ingest anything
    spin_batch,       # Bulk ingestion
    spin_url,         # Web crawling
    spin_directory,   # Directory ingestion
    spin_from_query   # Query -> memory learning
)

# Advanced API (for custom pipelines)
from .multimodal_spinner import MultiModalSpinner
from .chat_history import (
    ChatHistorySpinner,
    ChatHistoryAutoCapture,
    ingest_chat_history
)

# Modality-specific spinners (for custom pipelines)
from .modalities.audio import AudioSpinner
from .modalities.code import CodeSpinner
from .modalities.image import ImageSpinner
from .modalities.text import TextSpinner
from .modalities.website import WebsiteSpinner
from .modalities.youtube import YouTubeSpinner

# Git and repository spinners
from .git_spinner import GitSpinner
from .matrix_spinner import MatrixSpinner

__all__ = [
    # Primary API (ruthlessly simple)
    'spin',           # Ingest anything into memory
    'spin_batch',     # Batch ingestion
    'spin_url',       # Web content
    'spin_directory', # File system
    'spin_from_query', # Query learning

    # Advanced
    'MultiModalSpinner',
    'ChatHistorySpinner',
    'ChatHistoryAutoCapture',
    'ingest_chat_history',

    # Modality spinners (backward compatibility)
    'AudioSpinner',
    'CodeSpinner',
    'ImageSpinner',
    'TextSpinner',
    'WebsiteSpinner',
    'YouTubeSpinner',
    'GitSpinner',
    'MatrixSpinner',
]

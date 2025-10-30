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

__all__ = [
    # Primary API (ruthlessly simple)
    'spin',           # Ingest anything into memory
    'spin_batch',     # Batch ingestion
    'spin_url',       # Web content
    'spin_directory', # File system
    'spin_from_query', # Query learning

    # Advanced
    'MultiModalSpinner',
]

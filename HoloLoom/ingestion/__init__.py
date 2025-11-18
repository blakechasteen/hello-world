"""
HoloLoom Ingestion Module
==========================

Command-line interfaces for ingesting data from various sources.

Available CLIs:
    - filesystem: Ingest files from local filesystem
    - (future: git, email, slack, etc.)

Usage:
    python -m HoloLoom.ingestion.filesystem --help
"""

__all__ = ["filesystem"]

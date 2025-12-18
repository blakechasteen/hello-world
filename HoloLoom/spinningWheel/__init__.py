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

# Browser and web spinners (NEW - December 2025)
from .browser_history import BrowserHistorySpinner
from .website import WebsiteSpinner as MultimodalWebsiteSpinner  # Multimodal version
from .recursive_crawler import RecursiveCrawlerSpinner, CrawlConfig, crawl_website

# OCR spinners (NEW - January 2025)
from .deepseek_ocr_spinner import DeepSeekOCRSpinner
from .handwritten_spinner import HandwrittenSpinner
from .receipt_spinner import ReceiptSpinner
from .schema_aware_receipt_spinner import SchemaAwareReceiptSpinner, process_receipt_to_graph
from .ocr_protocol import (
    OCRProtocol,
    OCRBackendChain,
    OCRResult,
    OCROutputFormat,
    OCRQuality
)

# OCR backends (protocol-based)
from .ocr_backends import (
    get_best_available_backend,
    get_all_available_backends
)

# Schema system (wool → yarn transformation)
from .schema_registry import (
    SchemaRegistry,
    SchemaDefinition,
    SchemaType,
    FieldMapping,
    ValidationResult,
    create_expense_schema,
    create_task_schema
)

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

    # Browser and web spinners (NEW - December 2025)
    'BrowserHistorySpinner',      # Read Chrome, Firefox, Edge, Safari, Brave history
    'MultimodalWebsiteSpinner',   # Web scraping with images
    'RecursiveCrawlerSpinner',    # Matryoshka importance-gated crawling
    'CrawlConfig',                # Crawler configuration
    'crawl_website',              # Convenience function

    # OCR spinners (NEW - January 2025)
    'DeepSeekOCRSpinner',      # General OCR (documents, PDFs)
    'HandwrittenSpinner',      # Handwritten notes
    'ReceiptSpinner',          # Receipt parsing
    'SchemaAwareReceiptSpinner',  # Schema-aware receipt → graph (KILLER FEATURE)
    'process_receipt_to_graph',   # Convenience function

    # OCR protocol (for custom backends)
    'OCRProtocol',
    'OCRBackendChain',
    'OCRResult',
    'OCROutputFormat',
    'OCRQuality',
    'get_best_available_backend',
    'get_all_available_backends',

    # Schema system (wool → yarn transformation)
    'SchemaRegistry',
    'SchemaDefinition',
    'SchemaType',
    'FieldMapping',
    'ValidationResult',
    'create_expense_schema',
    'create_task_schema',
]

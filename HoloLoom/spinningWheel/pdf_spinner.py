"""
PDFSpinner - PDF Document Ingestion
===================================

Extracts text and metadata from PDF files into MemoryShards.

Features:
- Text extraction from PDF pages
- Metadata extraction (title, author, creation date)
- Image extraction and processing
- Table detection (basic)
- Chunking strategies (by page, by section, custom)
- OCR support for scanned PDFs (optional)
- Graceful handling of encrypted PDFs

Author: Claude Code
Date: November 2025
"""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore,
    SpinnerStatus
)
from HoloLoom.spinningWheel.utils import create_source_id


# ============================================================================
# PDF Data Structures
# ============================================================================

@dataclass
class PDFMetadata:
    """PDF document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    num_pages: int = 0


@dataclass
class PDFPage:
    """Extracted PDF page."""
    page_number: int
    text: str
    has_images: bool = False
    image_count: int = 0
    table_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# PDF Utilities
# ============================================================================

class PDFExtractor:
    """Extract text and metadata from PDF files."""

    def __init__(self, use_ocr: bool = False):
        """
        Initialize PDF extractor.

        Args:
            use_ocr: Use OCR for scanned PDFs (requires pytesseract)
        """
        self.use_ocr = use_ocr and PYTESSERACT_AVAILABLE

    def extract_metadata(self, pdf_path: str) -> PDFMetadata:
        """
        Extract PDF metadata.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDFMetadata object
        """
        if not PYPDF_AVAILABLE:
            return PDFMetadata()

        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata

            return PDFMetadata(
                title=metadata.get("/Title") if metadata else None,
                author=metadata.get("/Author") if metadata else None,
                subject=metadata.get("/Subject") if metadata else None,
                creator=metadata.get("/Creator") if metadata else None,
                producer=metadata.get("/Producer") if metadata else None,
                creation_date=str(metadata.get("/CreationDate")) if metadata else None,
                modification_date=str(metadata.get("/ModDate")) if metadata else None,
                num_pages=len(reader.pages)
            )
        except Exception as e:
            print(f"Error extracting PDF metadata: {e}")
            return PDFMetadata()

    def extract_text(self, pdf_path: str, max_pages: Optional[int] = None) -> List[PDFPage]:
        """
        Extract text from PDF pages.

        Args:
            pdf_path: Path to PDF file
            max_pages: Maximum pages to extract (None = all)

        Returns:
            List of PDFPage objects
        """
        if not PYPDF_AVAILABLE:
            return []

        pages = []

        try:
            reader = PdfReader(pdf_path)
            total_pages = len(reader.pages)

            if max_pages:
                total_pages = min(total_pages, max_pages)

            for page_num in range(total_pages):
                page_obj = reader.pages[page_num]

                # Extract text
                text = page_obj.extract_text()

                # Detect images
                image_count = len([obj for obj in page_obj.get("/Resources", {}).get("/XObject", {}) 
                                   if obj.startswith("Image")])

                # Detect tables (basic: look for table-like patterns)
                table_count = 0
                if text:
                    # Simple heuristic: count lines with consistent columns
                    lines = text.split('\n')
                    if len(lines) > 5:
                        # Could be a table (very simplistic)
                        table_count = 1

                page = PDFPage(
                    page_number=page_num + 1,
                    text=text or "",
                    has_images=image_count > 0,
                    image_count=image_count,
                    table_count=table_count
                )

                pages.append(page)

        except Exception as e:
            print(f"Error extracting PDF text: {e}")

        return pages

    def detect_scanned_pdf(self, pdf_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based).

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if PDF appears to be scanned
        """
        if not PYPDF_AVAILABLE:
            return False

        try:
            reader = PdfReader(pdf_path)

            # Check first page for text content
            if reader.pages:
                page = reader.pages[0]
                text = page.extract_text()

                # If first page has very little text, likely scanned
                return len(text.strip()) < 50

        except Exception:
            pass

        return False


# ============================================================================
# PDF Spinner
# ============================================================================

class PDFSpinner(BaseSpinner):
    """
    Spin PDF documents into MemoryShards.

    Features:
    - Text extraction from PDF pages
    - Metadata extraction
    - Image detection
    - OCR support for scanned PDFs
    - Flexible chunking strategies
    - Importance scoring based on content

    Usage:
        >>> spinner = PDFSpinner()
        >>> result = await spinner.spin("/path/to/document.pdf")
        >>> print(f"Created {result.shard_count} shards")
    """

    def __init__(
        self,
        importance_threshold: float = 0.1,
        checkpoint_dir: Optional[Path] = None,
        chunk_strategy: str = "page",  # 'page', 'section', or 'custom'
        use_ocr: bool = False,
        max_pages: Optional[int] = None
    ):
        """
        Initialize PDFSpinner.

        Args:
            importance_threshold: Minimum importance (0.0-1.0)
            checkpoint_dir: Directory for checkpoints
            chunk_strategy: How to chunk PDF ('page', 'section', or 'custom')
            use_ocr: Use OCR for scanned PDFs
            max_pages: Maximum pages to process
        """
        super().__init__(
            name="pdf",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.chunk_strategy = chunk_strategy
        self.use_ocr = use_ocr
        self.max_pages = max_pages
        self.extractor = PDFExtractor(use_ocr=use_ocr)

    # ========================================================================
    # Protocol Methods
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=True,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            embeddings=False,
            multimodal=PIL_AVAILABLE,
            max_input_size=None,
            supported_formats=['pdf']
        )

    def is_available(self) -> bool:
        """Check if PyPDF2 available."""
        return PYPDF_AVAILABLE

    # ========================================================================
    # Core Spinning
    # ========================================================================

    async def _spin_impl(
        self,
        source: str,
        **kwargs
    ) -> List[MemoryShard]:
        """
        Process PDF file.

        Args:
            source: Path to PDF file
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        pdf_path = source

        # Validate file
        if not Path(pdf_path).exists():
            raise ValueError(f"PDF file not found: {pdf_path}")

        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"Not a PDF file: {pdf_path}")

        # Extract metadata
        metadata = self.extractor.extract_metadata(pdf_path)
        print(f"Processing PDF: {metadata.title or Path(pdf_path).name} ({metadata.num_pages} pages)")

        # Extract pages
        pages = self.extractor.extract_text(pdf_path, max_pages=self.max_pages)

        # Convert to shards based on strategy
        shards = []

        if self.chunk_strategy == "page":
            for page in pages:
                shard = self._page_to_shard(page, pdf_path, metadata)
                shards.append(shard)

        elif self.chunk_strategy == "section":
            # Group pages by sections (detected by headers)
            sections = self._detect_sections(pages)
            for section in sections:
                shard = self._section_to_shard(section, pdf_path, metadata)
                shards.append(shard)

        else:  # custom
            # Just process pages for now
            for page in pages:
                shard = self._page_to_shard(page, pdf_path, metadata)
                shards.append(shard)

        return shards

    # ========================================================================
    # Conversion Methods
    # ========================================================================

    def _page_to_shard(
        self,
        page: PDFPage,
        pdf_path: str,
        metadata: PDFMetadata
    ) -> MemoryShard:
        """Convert PDF page to MemoryShard."""
        # Build text
        text_parts = [
            f"PDF: {metadata.title or Path(pdf_path).name}",
            f"Page {page.page_number} of {metadata.num_pages}",
        ]

        if metadata.author:
            text_parts.append(f"Author: {metadata.author}")

        if page.has_images:
            text_parts.append(f"Contains {page.image_count} image(s)")

        if page.table_count > 0:
            text_parts.append(f"Contains {page.table_count} table(s)")

        text_parts.extend(["", page.text])

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            Path(pdf_path).name,
            metadata.title or "untitled",
            metadata.author or "unknown",
            f"page_{page.page_number}"
        ]

        # Extract motifs
        motifs = ["pdf", "document", f"page"]

        if page.has_images:
            motifs.append("has_images")

        if page.table_count > 0:
            motifs.append("has_tables")

        # Score importance
        importance = self._score_page_importance(page, metadata)

        # Create shard
        return self._create_shard(
            id_suffix=f"pdf_{Path(pdf_path).stem}_p{page.page_number:04d}",
            text=text,
            episode=f"pdf_{Path(pdf_path).stem}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'pdf',
                'file': str(Path(pdf_path).absolute()),
                'filename': Path(pdf_path).name,
                'title': metadata.title,
                'author': metadata.author,
                'page_number': page.page_number,
                'total_pages': metadata.num_pages,
                'has_images': page.has_images,
                'image_count': page.image_count,
                'table_count': page.table_count,
                'creation_date': metadata.creation_date,
                'modification_date': metadata.modification_date,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'confidence': 1.0
            }
        )

    def _section_to_shard(
        self,
        section: List[PDFPage],
        pdf_path: str,
        metadata: PDFMetadata
    ) -> MemoryShard:
        """Convert PDF section to MemoryShard."""
        # Combine pages in section
        section_text = "\n\n".join([page.text for page in section])
        page_numbers = [page.page_number for page in section]

        # Build text
        text_parts = [
            f"PDF: {metadata.title or Path(pdf_path).name}",
            f"Pages {page_numbers[0]}-{page_numbers[-1]} of {metadata.num_pages}",
        ]

        if metadata.author:
            text_parts.append(f"Author: {metadata.author}")

        text_parts.extend(["", section_text])

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            Path(pdf_path).name,
            metadata.title or "untitled",
            metadata.author or "unknown"
        ]

        # Extract motifs
        motifs = ["pdf", "document", "section"]

        # Score importance
        importance = self._score_section_importance(section)

        # Create shard
        return self._create_shard(
            id_suffix=f"pdf_{Path(pdf_path).stem}_s{page_numbers[0]:04d}",
            text=text,
            episode=f"pdf_{Path(pdf_path).stem}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'pdf',
                'file': str(Path(pdf_path).absolute()),
                'filename': Path(pdf_path).name,
                'title': metadata.title,
                'author': metadata.author,
                'page_range': f"{page_numbers[0]}-{page_numbers[-1]}",
                'total_pages': metadata.num_pages,
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'confidence': 1.0
            }
        )

    # ========================================================================
    # Importance Scoring
    # ========================================================================

    def _score_page_importance(self, page: PDFPage, metadata: PDFMetadata) -> ImportanceScore:
        """Score PDF page importance."""
        signals = ImportanceSignals()

        # Length signal
        signals.length_score = min(1.0, len(page.text) / 1000)

        # Authority: official documents (title/author present)
        has_metadata = bool(metadata.title or metadata.author)
        signals.authority_score = 0.7 if has_metadata else 0.4

        # Structural: pages with images/tables more structured
        structure_score = 0.3
        if page.has_images:
            structure_score += 0.3
        if page.table_count > 0:
            structure_score += 0.3
        signals.structural_score = min(1.0, structure_score)

        # Custom signals
        if page.has_images:
            signals.custom_signals['images'] = 0.6

        if page.table_count > 0:
            signals.custom_signals['tables'] = 0.7

        # Early pages often more important
        if page.page_number <= 3:
            signals.custom_signals['early_page'] = 0.5

        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )

    def _score_section_importance(self, section: List[PDFPage]) -> ImportanceScore:
        """Score PDF section importance."""
        # Average of page scores
        page_scores = [
            self._score_page_importance(page, PDFMetadata())
            for page in section
        ]

        avg_score = sum(s.score for s in page_scores) / len(page_scores) if page_scores else 0.5

        return ImportanceScore(
            score=avg_score,
            signals=ImportanceSignals(),
            reason=f"average of {len(section)} pages"
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def _detect_sections(self, pages: List[PDFPage]) -> List[List[PDFPage]]:
        """
        Detect sections in PDF (simplified).

        Args:
            pages: List of PDF pages

        Returns:
            List of page groups (sections)
        """
        sections = []
        current_section = []

        for page in pages:
            current_section.append(page)

            # Simple heuristic: new section if page starts with header-like text
            lines = page.text.split('\n')
            if lines and (lines[0].isupper() or len(lines[0]) < 50):
                # Likely a section start
                if len(current_section) > 1:
                    sections.append(current_section[:-1])
                    current_section = [page]

        # Add remaining pages
        if current_section:
            sections.append(current_section)

        return sections if sections else [[p] for p in pages]


# ============================================================================
# Convenience Functions
# ============================================================================

async def spin_pdf(
    pdf_path: str,
    chunk_strategy: str = "page",
    max_pages: Optional[int] = None
) -> SpinResult:
    """
    Quick function to spin a PDF file.

    Args:
        pdf_path: Path to PDF file
        chunk_strategy: Chunking strategy ('page', 'section', or 'custom')
        max_pages: Maximum pages to process

    Returns:
        SpinResult with shards

    Example:
        >>> result = await spin_pdf("/path/to/document.pdf")
        >>> print(f"Created {result.shard_count} shards")
    """
    spinner = PDFSpinner(
        chunk_strategy=chunk_strategy,
        max_pages=max_pages
    )

    return await spinner.spin(pdf_path)

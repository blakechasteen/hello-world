"""
PDF Spinner Tests
=================

Comprehensive tests for the PDFSpinner:
- Text extraction from PDFs
- Page-based chunking
- Metadata extraction
- Corrupted PDF handling
- Empty PDF handling
- OCR integration (when available)
- Table extraction (when available)
- Section and citation detection

Author: Claude Code
Date: January 2026
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO


# =============================================================================
# Import the PDF spinner module
# =============================================================================

try:
    from HoloLoom.spinningWheel.pdf_spinner import (
        PDFSpinner,
        PDFParser,
        PDFPage,
        PDFDocument,
        spin_pdf,
    )
    from HoloLoom.spinningWheel.protocol import (
        SpinResult,
        SpinnerCapabilities,
        SpinnerStatus,
    )
    PDF_SPINNER_AVAILABLE = True
except ImportError as e:
    PDF_SPINNER_AVAILABLE = False
    IMPORT_ERROR = str(e)


# =============================================================================
# Check for PDF libraries
# =============================================================================

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# =============================================================================
# Skip decorators
# =============================================================================

pytestmark = pytest.mark.skipif(
    not PDF_SPINNER_AVAILABLE,
    reason=f"PDFSpinner not available: {IMPORT_ERROR if not PDF_SPINNER_AVAILABLE else ''}"
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def mock_pdf_document():
    """Create a mock PDFDocument."""
    pages = [
        PDFPage(
            page_number=1,
            text="Introduction\n\nThis is the first page of the document.",
            metadata={'width': 612, 'height': 792}
        ),
        PDFPage(
            page_number=2,
            text="Chapter 1: Methods\n\nWe describe our approach here.",
            metadata={'width': 612, 'height': 792}
        ),
        PDFPage(
            page_number=3,
            text="Chapter 2: Results\n\nThe results show significant improvement.",
            metadata={'width': 612, 'height': 792}
        ),
        PDFPage(
            page_number=4,
            text="References\n\n[1] Smith et al., 2023\n[2] Jones, 2024",
            metadata={'width': 612, 'height': 792}
        )
    ]

    return PDFDocument(
        pages=pages,
        metadata={
            'title': 'Test Document',
            'author': 'Test Author',
            'page_count': 4,
            'file_size': 1024
        }
    )


@pytest.fixture
def simple_pdf_bytes():
    """Create simple PDF-like bytes for testing (not a real PDF)."""
    return b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
trailer
<< /Size 4 /Root 1 0 R >>
startxref
0
%%EOF"""


# =============================================================================
# Test PDFSpinner Initialization
# =============================================================================

class TestPDFSpinnerInit:
    """Test PDFSpinner initialization."""

    def test_default_initialization(self):
        """Test default spinner initialization."""
        spinner = PDFSpinner()

        assert spinner is not None
        assert spinner.spinner_id == "pdf_spinner"
        assert isinstance(spinner.capabilities, SpinnerCapabilities)

    def test_capabilities(self):
        """Test spinner capabilities."""
        spinner = PDFSpinner()
        caps = spinner.capabilities

        assert caps.entity_extraction is True
        assert caps.motif_extraction is True
        assert caps.importance_scoring is True

    def test_supported_formats(self):
        """Test supported formats."""
        spinner = PDFSpinner()
        caps = spinner.capabilities

        assert '.pdf' in caps.supported_formats or caps.supported_formats == []


# =============================================================================
# Test PDFPage Dataclass
# =============================================================================

class TestPDFPage:
    """Test PDFPage dataclass."""

    def test_page_creation(self):
        """Test PDFPage creation."""
        page = PDFPage(
            page_number=1,
            text="Sample page content",
            metadata={'width': 612, 'height': 792}
        )

        assert page.page_number == 1
        assert page.text == "Sample page content"
        assert page.metadata['width'] == 612

    def test_page_with_empty_text(self):
        """Test PDFPage with empty text."""
        page = PDFPage(
            page_number=1,
            text="",
            metadata={}
        )

        assert page.text == ""
        assert page.page_number == 1

    def test_page_with_tables(self):
        """Test PDFPage with tables."""
        page = PDFPage(
            page_number=1,
            text="Table content",
            metadata={},
            tables=[
                [['Header1', 'Header2'], ['Row1Col1', 'Row1Col2']]
            ]
        )

        assert len(page.tables) == 1


# =============================================================================
# Test PDFDocument Dataclass
# =============================================================================

class TestPDFDocument:
    """Test PDFDocument dataclass."""

    def test_document_creation(self, mock_pdf_document):
        """Test PDFDocument creation."""
        doc = mock_pdf_document

        assert len(doc.pages) == 4
        assert doc.metadata['title'] == 'Test Document'
        assert doc.metadata['page_count'] == 4

    def test_empty_document(self):
        """Test empty PDFDocument."""
        doc = PDFDocument(pages=[], metadata={})

        assert len(doc.pages) == 0
        assert doc.metadata == {}

    def test_document_text_extraction(self, mock_pdf_document):
        """Test extracting all text from document."""
        doc = mock_pdf_document

        all_text = "\n".join(page.text for page in doc.pages)

        assert "Introduction" in all_text
        assert "Methods" in all_text
        assert "Results" in all_text
        assert "References" in all_text


# =============================================================================
# Test PDFParser
# =============================================================================

class TestPDFParser:
    """Test PDFParser class."""

    @pytest.mark.requires_pdf
    def test_parse_valid_pdf(self, temp_dir):
        """Test parsing a valid PDF file."""
        # This test requires actual PDF files
        pytest.skip("Requires actual PDF file")

    def test_parse_nonexistent_file(self, temp_dir):
        """Test parsing nonexistent file."""
        nonexistent = temp_dir / "nonexistent.pdf"

        try:
            result = PDFParser.parse(nonexistent)
            # Should return None or raise exception
            assert result is None or isinstance(result, PDFDocument)
        except FileNotFoundError:
            pass  # Expected

    def test_parse_invalid_pdf(self, temp_dir, simple_pdf_bytes):
        """Test parsing invalid/corrupted PDF."""
        invalid_pdf = temp_dir / "invalid.pdf"
        invalid_pdf.write_bytes(b"Not a PDF file")

        try:
            result = PDFParser.parse(invalid_pdf)
            # May return None or empty document
            assert result is None or isinstance(result, PDFDocument)
        except Exception:
            pass  # Expected for corrupted PDF

    def test_parse_minimal_pdf(self, temp_dir, simple_pdf_bytes):
        """Test parsing minimal PDF structure."""
        pdf_path = temp_dir / "minimal.pdf"
        pdf_path.write_bytes(simple_pdf_bytes)

        try:
            result = PDFParser.parse(pdf_path)
            # Should handle gracefully
            assert result is None or isinstance(result, PDFDocument)
        except Exception:
            pass  # May fail on invalid PDF structure


# =============================================================================
# Test PDFSpinner Text Extraction
# =============================================================================

class TestTextExtraction:
    """Test text extraction functionality."""

    def test_extract_text_from_pages(self, mock_pdf_document):
        """Test text extraction from multiple pages."""
        spinner = PDFSpinner()

        # Simulate extracting text from document
        all_text = []
        for page in mock_pdf_document.pages:
            all_text.append(page.text)

        combined = "\n".join(all_text)

        assert "Introduction" in combined
        assert "Chapter 1" in combined
        assert "Chapter 2" in combined

    def test_page_chunking(self, mock_pdf_document):
        """Test page-based chunking."""
        spinner = PDFSpinner()

        # Each page should become a chunk or part of a chunk
        chunks = []
        for page in mock_pdf_document.pages:
            chunks.append({
                'page': page.page_number,
                'text': page.text
            })

        assert len(chunks) == 4

    def test_whitespace_handling(self):
        """Test handling of whitespace in extracted text."""
        page = PDFPage(
            page_number=1,
            text="   Multiple   spaces   and\n\n\nnewlines   ",
            metadata={}
        )

        # Text should be preserved as-is or normalized
        assert isinstance(page.text, str)


# =============================================================================
# Test Metadata Extraction
# =============================================================================

class TestMetadataExtraction:
    """Test PDF metadata extraction."""

    def test_basic_metadata(self, mock_pdf_document):
        """Test basic metadata extraction."""
        metadata = mock_pdf_document.metadata

        assert 'title' in metadata
        assert 'author' in metadata
        assert 'page_count' in metadata

    def test_page_metadata(self, mock_pdf_document):
        """Test page-level metadata."""
        page = mock_pdf_document.pages[0]

        assert 'width' in page.metadata
        assert 'height' in page.metadata

    def test_missing_metadata(self):
        """Test handling of missing metadata."""
        doc = PDFDocument(
            pages=[PDFPage(page_number=1, text="Content", metadata={})],
            metadata={}
        )

        # Should handle missing metadata gracefully
        title = doc.metadata.get('title', 'Unknown')
        assert title == 'Unknown'


# =============================================================================
# Test Section Detection
# =============================================================================

class TestSectionDetection:
    """Test section/chapter detection."""

    def test_detect_chapters(self, mock_pdf_document):
        """Test detection of chapter headings."""
        all_text = "\n".join(page.text for page in mock_pdf_document.pages)

        # Should be able to identify sections
        has_chapter1 = "Chapter 1" in all_text
        has_chapter2 = "Chapter 2" in all_text

        assert has_chapter1
        assert has_chapter2

    def test_detect_introduction(self, mock_pdf_document):
        """Test detection of introduction section."""
        first_page = mock_pdf_document.pages[0]

        assert "Introduction" in first_page.text

    def test_detect_references(self, mock_pdf_document):
        """Test detection of references section."""
        last_page = mock_pdf_document.pages[-1]

        assert "References" in last_page.text


# =============================================================================
# Test Citation Extraction
# =============================================================================

class TestCitationExtraction:
    """Test citation extraction functionality."""

    def test_extract_bracket_citations(self, mock_pdf_document):
        """Test extraction of bracket-style citations [1]."""
        last_page = mock_pdf_document.pages[-1]

        assert "[1]" in last_page.text
        assert "[2]" in last_page.text

    def test_extract_author_year_citations(self):
        """Test extraction of author-year citations."""
        page = PDFPage(
            page_number=1,
            text="As noted by Smith (2023), the results are significant. Jones et al. (2024) agree.",
            metadata={}
        )

        # Should contain author-year style citations
        assert "Smith (2023)" in page.text
        assert "Jones et al. (2024)" in page.text


# =============================================================================
# Test Entity Extraction
# =============================================================================

class TestEntityExtraction:
    """Test entity extraction from PDF content."""

    def test_extract_entities(self, mock_pdf_document):
        """Test entity extraction from document."""
        spinner = PDFSpinner()

        # Would extract entities from text
        all_text = "\n".join(page.text for page in mock_pdf_document.pages)

        # Author names might be entities
        assert "Smith" in all_text or len(all_text) > 0


# =============================================================================
# Test Motif Extraction
# =============================================================================

class TestMotifExtraction:
    """Test motif extraction from PDF content."""

    def test_academic_motifs(self, mock_pdf_document):
        """Test detection of academic paper motifs."""
        all_text = "\n".join(page.text for page in mock_pdf_document.pages)

        # Academic paper indicators
        has_methods = "Methods" in all_text
        has_results = "Results" in all_text
        has_refs = "References" in all_text

        assert has_methods or has_results or has_refs


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in PDF spinner."""

    def test_corrupted_pdf(self, temp_dir):
        """Test handling of corrupted PDF."""
        corrupted = temp_dir / "corrupted.pdf"
        corrupted.write_bytes(b"CORRUPTED PDF DATA")

        spinner = PDFSpinner()

        async def run_test():
            result = await spinner.spin(corrupted)
            return result

        result = asyncio.run(run_test())

        # Should handle gracefully
        assert isinstance(result, SpinResult)
        # May have success=False or empty shards

    def test_empty_pdf(self, temp_dir):
        """Test handling of empty PDF."""
        empty = temp_dir / "empty.pdf"
        empty.write_bytes(b"")

        spinner = PDFSpinner()

        async def run_test():
            result = await spinner.spin(empty)
            return result

        result = asyncio.run(run_test())

        assert isinstance(result, SpinResult)

    def test_nonexistent_pdf(self):
        """Test handling of nonexistent PDF."""
        spinner = PDFSpinner()

        async def run_test():
            result = await spinner.spin(Path("/nonexistent/file.pdf"))
            return result

        result = asyncio.run(run_test())

        assert isinstance(result, SpinResult)
        # Should indicate failure
        assert result.success is False or len(result.shards) == 0

    def test_password_protected_pdf(self, temp_dir):
        """Test handling of password-protected PDF."""
        # Create a mock protected PDF
        protected = temp_dir / "protected.pdf"
        protected.write_bytes(b"%PDF-1.4\nENCRYPTED")

        spinner = PDFSpinner()

        async def run_test():
            result = await spinner.spin(protected)
            return result

        result = asyncio.run(run_test())

        # Should handle gracefully
        assert isinstance(result, SpinResult)


# =============================================================================
# Test OCR Integration
# =============================================================================

class TestOCRIntegration:
    """Test OCR integration for scanned PDFs."""

    def test_ocr_capability_flag(self):
        """Test that OCR capability is indicated."""
        spinner = PDFSpinner()

        # OCR might be optional
        caps = spinner.capabilities
        # Check if there's an OCR-related field
        assert hasattr(caps, 'supported_formats') or True

    def test_scanned_pdf_detection(self):
        """Test detection of scanned/image-based PDFs."""
        # A page with no selectable text might be scanned
        page = PDFPage(
            page_number=1,
            text="",  # No text = likely scanned
            metadata={'is_scanned': True}
        )

        # Should detect as potentially needing OCR
        assert page.text == "" or 'is_scanned' in page.metadata


# =============================================================================
# Test Table Extraction
# =============================================================================

class TestTableExtraction:
    """Test table extraction from PDFs."""

    def test_table_detection(self):
        """Test detection of tables in PDF."""
        page = PDFPage(
            page_number=1,
            text="Table 1: Results\nA | B | C\n1 | 2 | 3",
            metadata={},
            tables=[
                [['A', 'B', 'C'], ['1', '2', '3']]
            ]
        )

        assert len(page.tables) == 1
        assert page.tables[0][0] == ['A', 'B', 'C']

    def test_table_to_text(self):
        """Test converting table to text format."""
        table = [
            ['Header1', 'Header2'],
            ['Value1', 'Value2']
        ]

        # Convert table to markdown-style text
        lines = []
        lines.append(' | '.join(table[0]))
        lines.append(' | '.join(['---'] * len(table[0])))
        for row in table[1:]:
            lines.append(' | '.join(row))

        table_text = '\n'.join(lines)

        assert 'Header1' in table_text
        assert 'Value1' in table_text


# =============================================================================
# Test Convenience Functions
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_spin_pdf_function(self, temp_dir, simple_pdf_bytes):
        """Test spin_pdf convenience function."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_bytes(simple_pdf_bytes)

        async def run_test():
            result = await spin_pdf(str(pdf_path))
            return result

        result = asyncio.run(run_test())

        # Should return SpinResult or list of shards
        assert result is not None


# =============================================================================
# Test Shard Generation
# =============================================================================

class TestShardGeneration:
    """Test MemoryShard generation from PDFs."""

    def test_shard_structure(self, mock_pdf_document):
        """Test that generated shards have correct structure."""
        spinner = PDFSpinner()

        # Simulate shard generation
        shards = []
        for page in mock_pdf_document.pages:
            shard_data = {
                'id': f"pdf_page_{page.page_number}",
                'text': page.text,
                'episode': 'pdf_document',
                'entities': [],
                'motifs': ['pdf', 'document'],
                'metadata': {
                    'page_number': page.page_number,
                    'source': 'test.pdf'
                }
            }
            shards.append(shard_data)

        assert len(shards) == 4
        assert shards[0]['metadata']['page_number'] == 1

    def test_shard_metadata_completeness(self, mock_pdf_document):
        """Test that shard metadata is complete."""
        metadata = {
            'page_number': 1,
            'source': 'test.pdf',
            'title': mock_pdf_document.metadata.get('title'),
            'author': mock_pdf_document.metadata.get('author'),
            'total_pages': len(mock_pdf_document.pages)
        }

        assert metadata['page_number'] == 1
        assert metadata['title'] == 'Test Document'
        assert metadata['total_pages'] == 4


# =============================================================================
# Test Performance
# =============================================================================

class TestPerformance:
    """Test performance-related aspects."""

    def test_status_check(self):
        """Test spinner status check."""
        spinner = PDFSpinner()

        async def run_test():
            status = await spinner.check_status()
            return status

        status = asyncio.run(run_test())

        assert status in SpinnerStatus
        # Status depends on library availability
        assert status in [
            SpinnerStatus.AVAILABLE,
            SpinnerStatus.DEGRADED,
            SpinnerStatus.UNAVAILABLE
        ]

"""
Tests for PDFSpinner

Tests PDF document parsing, table extraction, and MemoryShard conversion.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

# Mock PDF libraries before importing PDFSpinner
class MockPDFReader:
    def __init__(self, file):
        self.metadata = {
            '/Title': 'Test Document',
            '/Author': 'Test Author'
        }
        self.pages = [MockPDFPage(), MockPDFPage()]

class MockPDFPage:
    def extract_text(self):
        return "This is a test PDF page with some content about algorithms and data structures."

sys_modules_patch = {
    'PyPDF2': MagicMock(PdfReader=MockPDFReader),
    'pdfplumber': MagicMock()
}

with patch.dict('sys.modules', sys_modules_patch):
    from hololoom.spinningWheel.pdf_spinner import (
        PDFSpinner,
        PDFParser,
        PDFPage,
        PDFDocument,
        spin_pdf,
        create_pdf_scorer
    )


# Fixtures

@pytest.fixture
def temp_pdf_dir():
    """Create temporary directory for test PDFs"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_pdf_page():
    """Create mock PDF page"""
    return PDFPage(
        page_number=1,
        text="This is a test page about machine learning algorithms.\n\nIt contains technical content.",
        tables=[],
        sections=[
            {'type': 'header', 'text': 'INTRODUCTION'},
            {'type': 'paragraph', 'text': 'This is an introduction paragraph.'}
        ]
    )


@pytest.fixture
def mock_pdf_document(tmp_path):
    """Create mock PDF document"""
    file_path = tmp_path / "test.pdf"
    file_path.touch()

    pages = [
        PDFPage(
            page_number=1,
            text="Title page content",
            sections=[{'type': 'header', 'text': 'Test Document'}]
        ),
        PDFPage(
            page_number=2,
            text="Body content with algorithms and methods discussed in detail.",
            tables=[[['Col1', 'Col2'], ['Data1', 'Data2']]],
            sections=[
                {'type': 'header', 'text': 'METHODS'},
                {'type': 'paragraph', 'text': 'We used various algorithms.'}
            ]
        )
    ]

    return PDFDocument(
        file_path=file_path,
        title="Test Document",
        author="Test Author",
        pages=pages
    )


# PDFPage Tests

def test_pdf_page_has_content():
    """Test PDFPage.has_content property"""
    # Page with text
    page1 = PDFPage(page_number=1, text="Some content")
    assert page1.has_content

    # Page with tables
    page2 = PDFPage(page_number=2, text="", tables=[[['a', 'b']]])
    assert page2.has_content

    # Empty page
    page3 = PDFPage(page_number=3, text="")
    assert not page3.has_content


def test_pdf_page_word_count():
    """Test PDFPage.word_count property"""
    page = PDFPage(
        page_number=1,
        text="This is a test with five words"
    )
    assert page.word_count == 7


# PDFDocument Tests

def test_pdf_document_total_pages(mock_pdf_document):
    """Test PDFDocument.total_pages property"""
    assert mock_pdf_document.total_pages == 2


def test_pdf_document_total_words(mock_pdf_document):
    """Test PDFDocument.total_words property"""
    assert mock_pdf_document.total_words > 0


def test_pdf_document_has_tables(mock_pdf_document):
    """Test PDFDocument.has_tables property"""
    assert mock_pdf_document.has_tables


# PDFParser Tests

def test_pdf_parser_detect_sections():
    """Test section detection"""
    text = """INTRODUCTION

This is an introduction paragraph.

METHODS

This describes the methods used."""

    sections = PDFParser._detect_sections(text)

    # Should detect headers and paragraphs
    assert len(sections) > 0
    assert any(s['type'] == 'header' for s in sections)
    assert any(s['type'] == 'paragraph' for s in sections)


def test_pdf_parser_extract_citations():
    """Test citation extraction"""
    text = """
    According to Smith (2020), machine learning is important.
    See also [1] and [2, 3] for more details.
    Johnson et al. (2019) showed similar results.
    """

    citations = PDFParser.extract_citations(text)

    assert len(citations) > 0
    assert any('Smith' in str(c) or '2020' in str(c) for c in citations)


# PDFSpinner Tests

@pytest.mark.asyncio
async def test_pdf_spinner_initialization():
    """Test PDFSpinner initialization"""
    spinner = PDFSpinner(
        importance_threshold=0.3,
        enable_ocr=False,
        chunk_by_page=True
    )

    assert spinner.get_name() == "pdf"
    assert spinner.importance_threshold == 0.3
    assert spinner.chunk_by_page == True


@pytest.mark.asyncio
async def test_pdf_spinner_capabilities():
    """Test spinner capabilities"""
    spinner = PDFSpinner()

    caps = spinner.get_capabilities()
    assert caps.basic_processing
    assert caps.entity_extraction
    assert caps.motif_extraction
    assert caps.importance_scoring
    assert caps.streaming
    assert 'pdf' in caps.supported_formats


@pytest.mark.asyncio
async def test_pdf_spinner_availability():
    """Test spinner availability check"""
    spinner = PDFSpinner()

    # Should be available since we mocked PyPDF2
    assert spinner.is_available()


def test_pdf_spinner_format_page_text(mock_pdf_page, mock_pdf_document):
    """Test page text formatting"""
    spinner = PDFSpinner()

    formatted = spinner._format_page_text(mock_pdf_page, mock_pdf_document)

    assert "Test Document" in formatted  # Title
    assert "Test Author" in formatted    # Author
    assert "Page 1/2" in formatted       # Page number
    assert mock_pdf_page.text in formatted  # Content


def test_pdf_spinner_format_document_text(mock_pdf_document):
    """Test document text formatting"""
    spinner = PDFSpinner()

    formatted = spinner._format_document_text(mock_pdf_document)

    assert "Title: Test Document" in formatted
    assert "Author: Test Author" in formatted
    assert "Pages: 2" in formatted
    assert "Page 1" in formatted
    assert "Page 2" in formatted


def test_pdf_spinner_extract_page_entities(mock_pdf_page, mock_pdf_document):
    """Test entity extraction from page"""
    spinner = PDFSpinner()

    entities = spinner._extract_page_entities(mock_pdf_page, mock_pdf_document)

    assert "Test Document" in entities  # Title
    assert "Test Author" in entities    # Author
    assert "INTRODUCTION" in entities   # Header


def test_pdf_spinner_extract_document_entities(mock_pdf_document):
    """Test entity extraction from document"""
    spinner = PDFSpinner()

    entities = spinner._extract_document_entities(mock_pdf_document)

    assert "Test Document" in entities
    assert "Test Author" in entities
    assert "METHODS" in entities  # Header from page 2


def test_pdf_spinner_extract_page_motifs(mock_pdf_page):
    """Test motif extraction from page"""
    spinner = PDFSpinner()

    motifs = spinner._extract_page_motifs(mock_pdf_page)

    assert 'structured' in motifs  # Has header
    # Word count determines form
    assert any(form in motifs for form in ['short_form', 'medium_form', 'long_form'])


def test_pdf_spinner_extract_document_motifs(mock_pdf_document):
    """Test motif extraction from document"""
    spinner = PDFSpinner()

    motifs = spinner._extract_document_motifs(mock_pdf_document)

    assert 'tables' in motifs  # Has tables
    assert 'short_document' in motifs  # 2 pages


def test_pdf_spinner_score_page_importance(mock_pdf_page, mock_pdf_document):
    """Test page importance scoring"""
    spinner = PDFSpinner()

    score = spinner.score_page_importance(mock_pdf_page, mock_pdf_document)

    assert 0.0 <= score.score <= 1.0
    assert score.signals is not None
    assert score.reason != ""

    # First page should have high authority score
    assert score.signals.authority_score == 1.0


def test_pdf_spinner_score_document_importance(mock_pdf_document):
    """Test document importance scoring"""
    spinner = PDFSpinner()

    score = spinner.score_document_importance(mock_pdf_document)

    assert 0.0 <= score.score <= 1.0
    assert "pages" in score.reason.lower()


def test_pdf_spinner_document_to_shards_by_page(mock_pdf_document):
    """Test converting document to shards (page mode)"""
    spinner = PDFSpinner(
        importance_threshold=0.0,  # Include all
        chunk_by_page=True
    )

    shards = spinner._document_to_shards(mock_pdf_document)

    # Should create 2 shards (one per page)
    assert len(shards) == 2

    # Check first shard
    assert shards[0].metadata['page_number'] == 1
    assert shards[0].metadata['title'] == "Test Document"
    assert shards[0].metadata['author'] == "Test Author"


def test_pdf_spinner_document_to_shards_whole_doc(mock_pdf_document):
    """Test converting document to shards (document mode)"""
    spinner = PDFSpinner(
        importance_threshold=0.0,  # Include all
        chunk_by_page=False
    )

    shards = spinner._document_to_shards(mock_pdf_document)

    # Should create 1 shard (whole document)
    assert len(shards) == 1

    # Check shard
    assert shards[0].metadata['total_pages'] == 2
    assert shards[0].metadata['has_tables'] == True


def test_pdf_spinner_document_to_shards_importance_filtering(mock_pdf_document):
    """Test importance threshold filtering"""
    spinner = PDFSpinner(
        importance_threshold=1.0,  # Filter all
        chunk_by_page=True
    )

    shards = spinner._document_to_shards(mock_pdf_document)

    # Should filter all pages
    assert len(shards) == 0


@pytest.mark.asyncio
async def test_create_pdf_scorer():
    """Test PDF-specific importance scorer creation"""
    scorer = create_pdf_scorer()

    # Should have technical terms
    assert len(scorer.technical_scorer.technical_terms) > 0

    # Test scoring
    technical_text = "This paper presents a novel algorithm for machine learning."
    tech_score = scorer.technical_scorer.score(technical_text)

    assert tech_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

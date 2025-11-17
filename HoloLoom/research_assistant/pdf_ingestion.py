"""
PDF Ingestion Pipeline
======================

Extracts text, images, and metadata from research paper PDFs.

**Philosophy**: Graceful degradation - works with minimal dependencies,
enhanced with optional packages.

Author: HoloLoom Team
Date: 2025-11-17
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import re
import warnings

# Optional dependencies with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    warnings.warn("PyPDF2 not available. PDF text extraction disabled. Install: pip install PyPDF2")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available. Image extraction disabled. Install: pip install Pillow")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    # Optional enhancement, not critical


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PaperSection:
    """A section of a research paper."""
    title: str
    content: str
    page_start: int
    page_end: int


@dataclass
class ResearchPaper:
    """Complete research paper with extracted content."""
    title: str
    authors: List[str]
    abstract: str
    sections: List[PaperSection]
    citations: List[str]
    metadata: Dict[str, Any]
    figures: List[Any] = field(default_factory=list)  # PIL Images if available
    pdf_path: Optional[str] = None


# ============================================================================
# PDF Ingestion Pipeline
# ============================================================================

class PDFIngestionPipeline:
    """
    Extracts structured content from research paper PDFs.

    **Features**:
    - Text extraction (PyPDF2 or pdfplumber)
    - Image extraction (PIL)
    - Structure parsing (sections, citations)
    - Metadata extraction (title, authors, abstract)

    **Graceful Degradation**:
    - Works without PIL (no image extraction)
    - Works without pdfplumber (uses PyPDF2)
    - Returns partial results if extraction fails
    """

    def __init__(self, extract_images: bool = True, extract_citations: bool = True):
        """
        Initialize PDF ingestion pipeline.

        Args:
            extract_images: Whether to extract figures/images
            extract_citations: Whether to extract references
        """
        self.extract_images_flag = extract_images and PIL_AVAILABLE
        self.extract_citations_flag = extract_citations

        if extract_images and not PIL_AVAILABLE:
            warnings.warn("Image extraction requested but PIL not available")

    def ingest(self, pdf_path: str) -> ResearchPaper:
        """
        Extract all content from research paper PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ResearchPaper with extracted content

        Example:
            >>> pipeline = PDFIngestionPipeline()
            >>> paper = pipeline.ingest("transformer_paper.pdf")
            >>> print(paper.title)
            'Attention Is All You Need'
        """
        if not PYPDF2_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise RuntimeError("No PDF library available. Install PyPDF2 or pdfplumber.")

        # Extract text
        text = self.extract_text(pdf_path)

        # Parse structure
        title = self._extract_title(text)
        authors = self._extract_authors(text)
        abstract = self._extract_abstract(text)
        sections = self._parse_sections(text)

        # Extract citations
        citations = []
        if self.extract_citations_flag:
            citations = self.extract_citations(text)

        # Extract images
        figures = []
        if self.extract_images_flag:
            try:
                figures = self.extract_images(pdf_path)
            except Exception as e:
                warnings.warn(f"Image extraction failed: {e}")

        return ResearchPaper(
            title=title,
            authors=authors,
            abstract=abstract,
            sections=sections,
            citations=citations,
            figures=figures,
            metadata={
                'pdf_path': pdf_path,
                'num_pages': len(sections),
                'num_figures': len(figures),
                'num_citations': len(citations)
            },
            pdf_path=pdf_path
        )

    def extract_text(self, pdf_path: str) -> str:
        """
        Extract raw text from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text
        """
        if PDFPLUMBER_AVAILABLE:
            return self._extract_text_pdfplumber(pdf_path)
        elif PYPDF2_AVAILABLE:
            return self._extract_text_pypdf2(pdf_path)
        else:
            raise RuntimeError("No PDF library available")

    def _extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2."""
        text = []

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text.append(page.extract_text())

        return '\n'.join(text)

    def _extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (higher quality)."""
        text = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text())

        return '\n'.join(text)

    def extract_images(self, pdf_path: str) -> List[Any]:
        """
        Extract images/figures from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of PIL Image objects
        """
        if not PIL_AVAILABLE:
            return []

        # Simple implementation - can be enhanced
        # For MVP, we'll just note that images exist
        # Full implementation would use pdf2image or PyMuPDF
        return []

    def _extract_title(self, text: str) -> str:
        """Extract paper title from text."""
        # Simple heuristic: first non-empty line
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        if lines:
            # Title is usually in first few lines
            return lines[0]

        return "Unknown Title"

    def _extract_authors(self, text: str) -> List[str]:
        """Extract authors from text."""
        # Simple heuristic: look for author-like patterns
        # In real papers, authors are usually after title
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        authors = []
        if len(lines) > 1:
            # Check second line for author patterns (names with commas)
            author_line = lines[1]
            if ',' in author_line or ' and ' in author_line.lower():
                # Split by commas or 'and'
                author_line = author_line.replace(' and ', ',')
                authors = [name.strip() for name in author_line.split(',')]

        return authors if authors else ["Unknown Author"]

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract from text."""
        # Look for "Abstract" section
        abstract_match = re.search(
            r'(?:Abstract|ABSTRACT)\s*\n+(.*?)(?:\n\n|Introduction|INTRODUCTION)',
            text,
            re.DOTALL | re.IGNORECASE
        )

        if abstract_match:
            return abstract_match.group(1).strip()

        # Fallback: first paragraph after title/authors
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) > 3:
            # Assume abstract starts at line 3
            abstract_lines = []
            for i in range(3, min(len(lines), 20)):
                if len(lines[i]) > 50:  # Reasonable paragraph length
                    abstract_lines.append(lines[i])
                    if len(' '.join(abstract_lines)) > 200:
                        break

            return ' '.join(abstract_lines)

        return "Abstract not found"

    def _parse_sections(self, text: str) -> List[PaperSection]:
        """
        Parse paper into sections.

        Args:
            text: Full paper text

        Returns:
            List of sections
        """
        # Common section headings in research papers
        section_patterns = [
            r'\n(Introduction|INTRODUCTION)\n',
            r'\n(Related Work|RELATED WORK|Background|BACKGROUND)\n',
            r'\n(Method|METHOD|Methodology|METHODOLOGY|Approach|APPROACH)\n',
            r'\n(Experiments|EXPERIMENTS|Evaluation|EVALUATION|Results|RESULTS)\n',
            r'\n(Discussion|DISCUSSION)\n',
            r'\n(Conclusion|CONCLUSION|Conclusions|CONCLUSIONS)\n',
            r'\n(References|REFERENCES)\n'
        ]

        sections = []
        current_section = None
        current_content = []

        lines = text.split('\n')

        for i, line in enumerate(lines):
            # Check if line matches section heading
            is_section = False
            for pattern in section_patterns:
                if re.match(pattern, '\n' + line + '\n'):
                    # Save previous section
                    if current_section:
                        sections.append(PaperSection(
                            title=current_section,
                            content='\n'.join(current_content),
                            page_start=0,  # MVP: don't track pages precisely
                            page_end=0
                        ))

                    # Start new section
                    current_section = line.strip()
                    current_content = []
                    is_section = True
                    break

            if not is_section and current_section:
                current_content.append(line)

        # Add final section
        if current_section:
            sections.append(PaperSection(
                title=current_section,
                content='\n'.join(current_content),
                page_start=0,
                page_end=0
            ))

        # If no sections found, create one big section
        if not sections:
            sections.append(PaperSection(
                title="Full Paper",
                content=text,
                page_start=0,
                page_end=0
            ))

        return sections

    def extract_citations(self, text: str) -> List[str]:
        """
        Extract references/citations from text.

        Args:
            text: Full paper text

        Returns:
            List of citation strings
        """
        # Look for References section
        ref_match = re.search(
            r'(?:References|REFERENCES)\s*\n+(.*?)$',
            text,
            re.DOTALL | re.IGNORECASE
        )

        if not ref_match:
            return []

        ref_text = ref_match.group(1)

        # Split by common citation patterns
        # [1] Author et al., Title...
        # 1. Author et al., Title...
        citations = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s*', ref_text)

        # Filter out empty citations
        citations = [c.strip() for c in citations if len(c.strip()) > 20]

        return citations[:50]  # Limit to 50 citations


# ============================================================================
# Utility Functions
# ============================================================================

def quick_ingest(pdf_path: str) -> ResearchPaper:
    """
    Quick one-line PDF ingestion.

    Args:
        pdf_path: Path to PDF file

    Returns:
        ResearchPaper

    Example:
        >>> paper = quick_ingest("attention.pdf")
        >>> print(paper.title)
    """
    pipeline = PDFIngestionPipeline()
    return pipeline.ingest(pdf_path)

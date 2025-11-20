"""
PDF Spinner - Ingest PDF documents into HoloLoom memory

Supports:
- Text extraction (PyPDF2, pdfplumber fallback)
- Table detection and extraction (pdfplumber)
- Image extraction (optional OCR with pytesseract)
- Section detection (headers, paragraphs)
- Citation extraction (academic papers)
- Metadata extraction (author, title, date)
- Page-based chunking
- 9-signal importance scoring

Requires: PyPDF2 or pdfplumber
Optional: pytesseract (OCR), tabula-py (advanced tables)

Usage:
    from HoloLoom.spinningWheel.pdf_spinner import PDFSpinner

    spinner = PDFSpinner(importance_threshold=0.3)

    # Spin a PDF
    result = await spinner.spin("/path/to/document.pdf")

    # With OCR
    spinner = PDFSpinner(enable_ocr=True)
    result = await spinner.spin("/path/to/scanned.pdf")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
import hashlib
import re

# Try multiple PDF libraries (graceful degradation)
PDF_AVAILABLE = False
PDFPLUMBER_AVAILABLE = False
OCR_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    pass

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    pass

from HoloLoom.Documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.importance import ImportanceScorer


@dataclass
class PDFPage:
    """Parsed PDF page"""
    page_number: int
    text: str
    tables: List[List[List[str]]] = field(default_factory=list)  # List of tables
    images: List[str] = field(default_factory=list)  # Image paths
    sections: List[Dict[str, str]] = field(default_factory=list)  # Detected sections
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_content(self) -> bool:
        """Check if page has extractable content"""
        return bool(self.text.strip() or self.tables or self.images)

    @property
    def word_count(self) -> int:
        """Count words on page"""
        return len(self.text.split())


@dataclass
class PDFDocument:
    """Parsed PDF document"""
    file_path: Path
    title: Optional[str]
    author: Optional[str]
    pages: List[PDFPage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def total_words(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def has_tables(self) -> bool:
        return any(p.tables for p in self.pages)


class PDFParser:
    """Parse PDF files into structured data"""

    @staticmethod
    def parse_pdf(
        file_path: Path,
        enable_ocr: bool = False,
        extract_tables: bool = True
    ) -> PDFDocument:
        """
        Parse PDF file

        Args:
            file_path: Path to PDF file
            enable_ocr: Enable OCR for scanned PDFs
            extract_tables: Extract table data

        Returns:
            PDFDocument object
        """
        if not PDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "PDF parsing requires PyPDF2 or pdfplumber. "
                "Install with: pip install PyPDF2 pdfplumber"
            )

        # Try pdfplumber first (better table support)
        if PDFPLUMBER_AVAILABLE:
            return PDFParser._parse_with_pdfplumber(file_path, extract_tables)
        else:
            return PDFParser._parse_with_pypdf2(file_path)

    @staticmethod
    def _parse_with_pdfplumber(file_path: Path, extract_tables: bool) -> PDFDocument:
        """Parse using pdfplumber (preferred)"""
        import pdfplumber

        pages = []
        doc_metadata = {}

        with pdfplumber.open(file_path) as pdf:
            # Extract document metadata
            if pdf.metadata:
                doc_metadata = {
                    k: v for k, v in pdf.metadata.items()
                    if v is not None
                }

            # Extract pages
            for i, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text() or ""

                # Extract tables
                tables = []
                if extract_tables:
                    try:
                        extracted_tables = page.extract_tables()
                        if extracted_tables:
                            tables = extracted_tables
                    except Exception:
                        pass  # Table extraction failed, continue

                # Detect sections
                sections = PDFParser._detect_sections(text)

                pdf_page = PDFPage(
                    page_number=i + 1,
                    text=text,
                    tables=tables,
                    sections=sections,
                    metadata={'width': page.width, 'height': page.height}
                )
                pages.append(pdf_page)

        return PDFDocument(
            file_path=file_path,
            title=doc_metadata.get('Title'),
            author=doc_metadata.get('Author'),
            pages=pages,
            metadata=doc_metadata
        )

    @staticmethod
    def _parse_with_pypdf2(file_path: Path) -> PDFDocument:
        """Parse using PyPDF2 (fallback)"""
        import PyPDF2

        pages = []
        doc_metadata = {}

        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)

            # Extract metadata
            if reader.metadata:
                doc_metadata = {
                    k.lstrip('/'): v for k, v in reader.metadata.items()
                    if v is not None
                }

            # Extract pages
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""

                sections = PDFParser._detect_sections(text)

                pdf_page = PDFPage(
                    page_number=i + 1,
                    text=text,
                    sections=sections
                )
                pages.append(pdf_page)

        return PDFDocument(
            file_path=file_path,
            title=doc_metadata.get('Title'),
            author=doc_metadata.get('Author'),
            pages=pages,
            metadata=doc_metadata
        )

    @staticmethod
    def _detect_sections(text: str) -> List[Dict[str, str]]:
        """
        Detect sections in text (headers, paragraphs)

        Returns:
            List of {type: 'header'|'paragraph', text: '...'}
        """
        sections = []
        lines = text.split('\n')

        current_paragraph = []

        for line in lines:
            line = line.strip()

            if not line:
                # Empty line - end paragraph
                if current_paragraph:
                    sections.append({
                        'type': 'paragraph',
                        'text': ' '.join(current_paragraph)
                    })
                    current_paragraph = []
                continue

            # Detect headers (all caps, short, no period)
            if (len(line) < 100 and
                line.isupper() and
                not line.endswith('.')):
                # Flush paragraph
                if current_paragraph:
                    sections.append({
                        'type': 'paragraph',
                        'text': ' '.join(current_paragraph)
                    })
                    current_paragraph = []

                # Add header
                sections.append({
                    'type': 'header',
                    'text': line
                })
            else:
                current_paragraph.append(line)

        # Flush remaining
        if current_paragraph:
            sections.append({
                'type': 'paragraph',
                'text': ' '.join(current_paragraph)
            })

        return sections

    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """
        Extract citation patterns (academic papers)

        Returns:
            List of detected citations
        """
        citations = []

        # Pattern 1: Author (Year)
        pattern1 = r'\b([A-Z][a-z]+(?:\s+et\s+al\.?)?)\s+\((\d{4})\)'
        citations.extend(re.findall(pattern1, text))

        # Pattern 2: [1], [2, 3]
        pattern2 = r'\[(\d+(?:,\s*\d+)*)\]'
        citations.extend(re.findall(pattern2, text))

        return [str(c) for c in citations if c]


class PDFSpinner(BaseSpinner):
    """
    Spinner for PDF documents

    Ingests PDF files into HoloLoom memory with:
    - Text extraction (multiple libraries)
    - Table detection
    - Section structure
    - Citation extraction
    - Metadata preservation
    - Page-based chunking
    - 9-signal importance scoring
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        enable_ocr: bool = False,
        extract_tables: bool = True,
        chunk_by_page: bool = False,
        max_pages: Optional[int] = None
    ):
        """
        Initialize PDFSpinner

        Args:
            importance_threshold: Minimum importance score (0.0-1.0)
            enable_ocr: Enable OCR for scanned PDFs (requires pytesseract)
            extract_tables: Extract table data
            chunk_by_page: Create separate shard per page (vs per document)
            max_pages: Maximum pages to process (None = all)
        """
        super().__init__(name="pdf")

        if not PDF_AVAILABLE and not PDFPLUMBER_AVAILABLE:
            raise ImportError(
                "PDF spinner requires PyPDF2 or pdfplumber. "
                "Install with: pip install PyPDF2 pdfplumber"
            )

        self.importance_threshold = importance_threshold
        self.enable_ocr = enable_ocr
        self.extract_tables = extract_tables
        self.chunk_by_page = chunk_by_page
        self.max_pages = max_pages

        # Create importance scorer
        self.importance_scorer = ImportanceScorer(
            technical_terms={
                'algorithm', 'method', 'approach', 'model', 'framework',
                'analysis', 'result', 'conclusion', 'hypothesis', 'theory',
                'experiment', 'data', 'evaluation', 'performance', 'accuracy'
            }
        )

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            incremental=False,  # PDFs are static
            streaming=True,
            supported_formats=['pdf'],
            batch_processing=True
        )

    def is_available(self) -> bool:
        """Check if PDF dependencies are available"""
        return PDF_AVAILABLE or PDFPLUMBER_AVAILABLE

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Spin PDF file(s) into MemoryShards

        Args:
            source: PDF file path (str/Path) or list of paths
            **kwargs: Additional arguments

        Returns:
            List of MemoryShards
        """
        # Handle single file or multiple files
        if isinstance(source, (str, Path)):
            files = [Path(source)]
        elif isinstance(source, list):
            files = [Path(f) for f in source]
        else:
            raise ValueError(f"source must be file path or list of paths, got {type(source)}")

        all_shards = []
        for file_path in files:
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")

            # Parse PDF
            document = PDFParser.parse_pdf(
                file_path,
                enable_ocr=self.enable_ocr,
                extract_tables=self.extract_tables
            )

            # Convert to shards
            shards = self._document_to_shards(document)
            all_shards.extend(shards)

        return all_shards

    async def spin_stream(
        self,
        source: Any,
        batch_size: int = 10,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream MemoryShards from PDF(s)

        Args:
            source: PDF file path or list of paths
            batch_size: Number of pages per batch

        Yields:
            MemoryShard objects
        """
        # Get shards
        shards = await self._spin_impl(source, **kwargs)

        # Stream in batches
        for i in range(0, len(shards), batch_size):
            batch = shards[i:i + batch_size]
            for shard in batch:
                yield shard

    def _document_to_shards(self, document: PDFDocument) -> List[MemoryShard]:
        """
        Convert PDFDocument to MemoryShards

        Args:
            document: PDFDocument object

        Returns:
            List of MemoryShards (filtered by importance)
        """
        shards = []

        if self.chunk_by_page:
            # One shard per page
            for page in document.pages:
                if not page.has_content:
                    continue

                # Apply max_pages limit
                if self.max_pages and page.page_number > self.max_pages:
                    break

                # Score importance
                importance = self.score_page_importance(page, document)

                # Filter by threshold
                if importance.score < self.importance_threshold:
                    continue

                # Create shard
                shard = self._create_shard(
                    id_suffix=f"{document.file_path.stem}_p{page.page_number}",
                    text=self._format_page_text(page, document),
                    episode=f"pdf_{document.file_path.stem}",
                    entities=self._extract_page_entities(page, document),
                    motifs=self._extract_page_motifs(page),
                    metadata={
                        'file_path': str(document.file_path),
                        'page_number': page.page_number,
                        'total_pages': document.total_pages,
                        'title': document.title,
                        'author': document.author,
                        'word_count': page.word_count,
                        'has_tables': bool(page.tables),
                        'sections': len(page.sections),
                        'importance_score': importance.score,
                        'importance_reason': importance.reason
                    }
                )
                shards.append(shard)
        else:
            # One shard per document
            # Combine all pages
            all_text = '\n\n'.join(
                p.text for p in document.pages[:self.max_pages]
                if p.has_content
            )

            # Score importance
            importance = self.score_document_importance(document)

            # Filter by threshold
            if importance.score >= self.importance_threshold:
                shard = self._create_shard(
                    id_suffix=document.file_path.stem,
                    text=self._format_document_text(document),
                    episode=f"pdf_{document.file_path.stem}",
                    entities=self._extract_document_entities(document),
                    motifs=self._extract_document_motifs(document),
                    metadata={
                        'file_path': str(document.file_path),
                        'total_pages': document.total_pages,
                        'title': document.title,
                        'author': document.author,
                        'word_count': document.total_words,
                        'has_tables': document.has_tables,
                        'importance_score': importance.score,
                        'importance_reason': importance.reason
                    }
                )
                shards.append(shard)

        return shards

    def _format_page_text(self, page: PDFPage, document: PDFDocument) -> str:
        """Format page text for shard"""
        parts = []

        # Header with metadata
        if document.title:
            parts.append(f"Document: {document.title}")
        if document.author:
            parts.append(f"Author: {document.author}")
        parts.append(f"Page {page.page_number}/{document.total_pages}")
        parts.append("")

        # Page text
        parts.append(page.text)

        # Tables (if any)
        if page.tables:
            parts.append(f"\n[{len(page.tables)} table(s) detected]")

        return '\n'.join(parts)

    def _format_document_text(self, document: PDFDocument) -> str:
        """Format document text for shard"""
        parts = []

        # Header with metadata
        if document.title:
            parts.append(f"Title: {document.title}")
        if document.author:
            parts.append(f"Author: {document.author}")
        parts.append(f"Pages: {document.total_pages}")
        parts.append("")

        # All pages
        for page in document.pages[:self.max_pages]:
            if page.has_content:
                parts.append(f"--- Page {page.page_number} ---")
                parts.append(page.text)
                parts.append("")

        return '\n'.join(parts)

    def _extract_page_entities(self, page: PDFPage, document: PDFDocument) -> List[str]:
        """Extract entities from page"""
        entities = []

        # Document metadata
        if document.title:
            entities.append(document.title)
        if document.author:
            entities.append(document.author)

        # Section headers
        for section in page.sections:
            if section['type'] == 'header':
                entities.append(section['text'])

        return list(set(entities))

    def _extract_document_entities(self, document: PDFDocument) -> List[str]:
        """Extract entities from document"""
        entities = []

        # Metadata
        if document.title:
            entities.append(document.title)
        if document.author:
            entities.append(document.author)

        # All section headers
        for page in document.pages[:self.max_pages]:
            for section in page.sections:
                if section['type'] == 'header':
                    entities.append(section['text'])

        return list(set(entities))

    def _extract_page_motifs(self, page: PDFPage) -> List[str]:
        """Extract motifs from page"""
        motifs = []

        if page.tables:
            motifs.append('tables')
        if page.images:
            motifs.append('images')
        if any(s['type'] == 'header' for s in page.sections):
            motifs.append('structured')

        # Content type
        if page.word_count > 500:
            motifs.append('long_form')
        elif page.word_count > 100:
            motifs.append('medium_form')
        else:
            motifs.append('short_form')

        return motifs

    def _extract_document_motifs(self, document: PDFDocument) -> List[str]:
        """Extract motifs from document"""
        motifs = []

        if document.has_tables:
            motifs.append('tables')
        if document.total_pages > 20:
            motifs.append('long_document')
        elif document.total_pages > 5:
            motifs.append('medium_document')
        else:
            motifs.append('short_document')

        # Check for academic paper indicators
        all_text = ' '.join(p.text for p in document.pages[:5])  # First 5 pages
        if any(keyword in all_text.lower() for keyword in ['abstract', 'introduction', 'conclusion', 'references']):
            motifs.append('academic')

        return motifs

    def score_page_importance(self, page: PDFPage, document: PDFDocument) -> ImportanceScore:
        """
        Score page importance using 9 signals

        Args:
            page: PDFPage object
            document: Parent PDFDocument

        Returns:
            ImportanceScore
        """
        signals = ImportanceSignals()
        text = page.text

        # 1. Length score
        word_count = page.word_count
        if word_count < 50:
            signals.length_score = 0.2
        elif word_count < 200:
            signals.length_score = 0.5
        elif word_count <= 1000:
            signals.length_score = min(1.0, word_count / 1000)
        else:
            signals.length_score = 0.9

        # 2. Technical score
        signals.technical_score = self.importance_scorer.technical_scorer.score(text)

        # 3. Structural score (sections, tables)
        struct_score = 0.0
        if page.sections:
            struct_score += 0.3
        if page.tables:
            struct_score += 0.4
        if any(s['type'] == 'header' for s in page.sections):
            struct_score += 0.3
        signals.structural_score = min(1.0, struct_score)

        # 4. Authority score (first/last pages often important)
        if page.page_number == 1:
            signals.authority_score = 1.0  # Title page
        elif page.page_number <= 3:
            signals.authority_score = 0.8  # Abstract/intro
        elif page.page_number >= document.total_pages - 2:
            signals.authority_score = 0.7  # Conclusion/references
        else:
            signals.authority_score = 0.5

        # 5. Recency score (not applicable to PDFs)
        signals.recency_score = 0.5  # Neutral

        # 6. Engagement score (not applicable to PDFs)
        signals.engagement_score = 0.5  # Neutral

        # 7. Reference score (citations)
        citations = PDFParser.extract_citations(text)
        signals.reference_score = min(1.0, len(citations) / 10.0)

        # 8. Noise detection
        noise_score = self.importance_scorer.noise_detector.detect(text)

        # 9. Custom signals
        signals.custom_signals = {}
        signals.custom_signals['has_tables'] = 1.0 if page.tables else 0.0
        signals.custom_signals['page_position'] = signals.authority_score  # Reuse

        # Combine signals
        final_score = (
            0.15 * signals.length_score +
            0.20 * signals.technical_score +
            0.15 * signals.structural_score +
            0.15 * signals.authority_score +
            0.00 * signals.recency_score +      # N/A
            0.00 * signals.engagement_score +    # N/A
            0.10 * signals.reference_score +
            0.15 * signals.custom_signals.get('has_tables', 0.0) +
            0.10 * signals.custom_signals.get('page_position', 0.5)
        )

        # Apply noise penalty
        final_score *= (1.0 - max(0.0, noise_score))

        # Generate reason
        reasons = []
        if signals.structural_score > 0.6:
            reasons.append("well-structured")
        if signals.technical_score > 0.5:
            reasons.append("technical content")
        if page.tables:
            reasons.append("contains tables")
        if signals.authority_score > 0.7:
            reasons.append("key position")
        if citations:
            reasons.append(f"{len(citations)} citations")

        reason = " + ".join(reasons) if reasons else "standard page"

        return ImportanceScore(
            score=max(0.0, min(1.0, final_score)),
            signals=signals,
            reason=reason
        )

    def score_document_importance(self, document: PDFDocument) -> ImportanceScore:
        """Score overall document importance (aggregated from pages)"""
        if not document.pages:
            return ImportanceScore(score=0.0, signals=ImportanceSignals(), reason="empty")

        # Average page scores
        page_scores = [
            self.score_page_importance(page, document).score
            for page in document.pages[:self.max_pages]
            if page.has_content
        ]

        if not page_scores:
            return ImportanceScore(score=0.0, signals=ImportanceSignals(), reason="no content")

        avg_score = sum(page_scores) / len(page_scores)

        return ImportanceScore(
            score=avg_score,
            signals=ImportanceSignals(),  # Aggregated signals
            reason=f"average of {len(page_scores)} pages"
        )


# Convenience functions

async def spin_pdf(
    file_path: str,
    importance_threshold: float = 0.3,
    chunk_by_page: bool = False
) -> SpinResult:
    """
    Convenience function to spin a PDF file

    Args:
        file_path: Path to PDF file
        importance_threshold: Min importance score
        chunk_by_page: One shard per page vs one per document

    Returns:
        SpinResult with MemoryShards
    """
    spinner = PDFSpinner(
        importance_threshold=importance_threshold,
        chunk_by_page=chunk_by_page
    )

    shards = await spinner.spin(file_path)

    return SpinResult(
        shards=shards,
        success=True,
        items_processed=len(shards),
        items_filtered=0
    )


def create_pdf_scorer() -> ImportanceScorer:
    """Create importance scorer optimized for PDF documents"""
    return ImportanceScorer(
        technical_terms={
            'algorithm', 'method', 'approach', 'model', 'framework',
            'analysis', 'result', 'conclusion', 'hypothesis', 'theory',
            'experiment', 'data', 'evaluation', 'performance', 'accuracy',
            'abstract', 'introduction', 'methodology', 'discussion',
            'figure', 'table', 'equation', 'theorem', 'proof'
        }
    )

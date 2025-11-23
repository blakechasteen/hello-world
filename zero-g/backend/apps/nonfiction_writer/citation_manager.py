"""
Citation Manager for Nonfiction Writing

Handles extraction, formatting, linking, and management of bibliographic citations.
Supports APA, MLA, and Chicago citation styles with source-to-text linking.

Integrates with:
- HoloLoom PDFSpinner for automatic citation extraction
- YarnGraph for citation network storage (CITED_BY edges)
- SpacetimeFabric for complete provenance tracking
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse


class CitationStyle(Enum):
    """Supported citation formats"""
    APA = "apa"          # American Psychological Association (7th ed)
    MLA = "mla"          # Modern Language Association (9th ed)
    CHICAGO = "chicago"  # Chicago Manual of Style (17th ed)
    IEEE = "ieee"        # Institute of Electrical and Electronics Engineers


class SourceType(Enum):
    """Types of sources for proper formatting"""
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    CHAPTER = "chapter"
    WEBSITE = "website"
    CONFERENCE = "conference"
    THESIS = "thesis"
    REPORT = "report"
    NEWSPAPER = "newspaper"
    PREPRINT = "preprint"


@dataclass
class Citation:
    """
    Structured citation with all metadata

    Attributes:
        id: Unique citation identifier
        source_type: Type of source (journal, book, etc.)
        authors: List of author names
        title: Title of work
        year: Publication year
        journal: Journal name (for articles)
        volume: Volume number
        issue: Issue number
        pages: Page range (e.g., "123-145")
        publisher: Publisher name (for books)
        url: URL for online sources
        doi: Digital Object Identifier
        accessed_date: Date accessed (for web sources)
        notes: Additional notes or annotations
        credibility_score: Source quality score (0.0-1.0)
    """
    id: str
    source_type: SourceType
    authors: List[str]
    title: str
    year: int

    # Optional fields
    journal: Optional[str] = None
    volume: Optional[int] = None
    issue: Optional[int] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    accessed_date: Optional[datetime] = None
    notes: Optional[str] = None

    # Quality metadata
    credibility_score: float = 0.5
    is_peer_reviewed: bool = False
    is_primary_source: bool = False

    # Link tracking
    cited_in_paragraphs: Set[str] = field(default_factory=set)


class CitationManager:
    """
    Manages bibliographic citations for nonfiction writing

    Core capabilities:
    1. Extract citations from documents (PDFs, web pages)
    2. Format citations in multiple styles (APA, MLA, Chicago)
    3. Link citations to specific paragraphs in text
    4. Generate bibliographies
    5. Track citation usage and provenance
    """

    def __init__(self):
        self.citations: Dict[str, Citation] = {}
        self._citation_counter = 0

    def add_citation(self, citation: Citation) -> str:
        """
        Add a citation to the manager

        Args:
            citation: Citation object with metadata

        Returns:
            Citation ID for referencing
        """
        if not citation.id:
            self._citation_counter += 1
            citation.id = f"cite_{self._citation_counter:04d}"

        self.citations[citation.id] = citation
        return citation.id

    def extract_from_pdf_metadata(self, pdf_text: str, metadata: Dict) -> Optional[Citation]:
        """
        Extract citation from PDF metadata

        Args:
            pdf_text: Full text of PDF
            metadata: PDF metadata dictionary

        Returns:
            Extracted Citation or None
        """
        # Extract DOI from text if present
        doi = self._extract_doi(pdf_text)

        # Parse authors from metadata or text
        authors = self._extract_authors(pdf_text, metadata)

        # Extract title
        title = metadata.get('title', self._extract_title_from_text(pdf_text))

        # Extract year
        year = self._extract_year(pdf_text, metadata)

        # Detect source type
        source_type = self._detect_source_type(pdf_text, metadata)

        # Extract journal info for articles
        journal = None
        volume = None
        issue = None
        pages = None

        if source_type == SourceType.JOURNAL_ARTICLE:
            journal = self._extract_journal_name(pdf_text)
            volume = self._extract_volume(pdf_text)
            issue = self._extract_issue(pdf_text)
            pages = self._extract_pages(pdf_text)

        # Check if peer-reviewed
        is_peer_reviewed = self._is_peer_reviewed(journal, metadata)

        if not (authors and title and year):
            return None  # Insufficient metadata

        citation = Citation(
            id="",  # Will be auto-generated
            source_type=source_type,
            authors=authors,
            title=title,
            year=year,
            journal=journal,
            volume=volume,
            issue=issue,
            pages=pages,
            doi=doi,
            is_peer_reviewed=is_peer_reviewed,
            is_primary_source=self._is_primary_source(pdf_text),
        )

        return citation

    def extract_from_url(self, url: str, page_content: str, metadata: Dict) -> Optional[Citation]:
        """
        Extract citation from web page

        Args:
            url: URL of web page
            page_content: HTML/text content
            metadata: Extracted metadata (title, author, date)

        Returns:
            Citation or None
        """
        authors = metadata.get('authors', [])
        if not authors:
            authors = self._extract_authors_from_web(page_content)

        title = metadata.get('title', '')
        year = metadata.get('year', datetime.now().year)

        # Extract domain for credibility
        domain = urlparse(url).netloc
        credibility = self._score_web_credibility(domain)

        citation = Citation(
            id="",
            source_type=SourceType.WEBSITE,
            authors=authors,
            title=title,
            year=year,
            url=url,
            accessed_date=datetime.now(),
            credibility_score=credibility,
        )

        return citation

    def format_citation(self, citation_id: str, style: CitationStyle) -> str:
        """
        Format a citation in the specified style

        Args:
            citation_id: ID of citation to format
            style: Citation style (APA, MLA, Chicago)

        Returns:
            Formatted citation string
        """
        citation = self.citations.get(citation_id)
        if not citation:
            return f"[Citation {citation_id} not found]"

        if style == CitationStyle.APA:
            return self._format_apa(citation)
        elif style == CitationStyle.MLA:
            return self._format_mla(citation)
        elif style == CitationStyle.CHICAGO:
            return self._format_chicago(citation)
        elif style == CitationStyle.IEEE:
            return self._format_ieee(citation)
        else:
            return self._format_apa(citation)  # Default to APA

    def generate_bibliography(self, style: CitationStyle,
                            citation_ids: Optional[List[str]] = None) -> str:
        """
        Generate a formatted bibliography

        Args:
            style: Citation style
            citation_ids: Specific citations to include (None = all)

        Returns:
            Formatted bibliography string
        """
        if citation_ids is None:
            citation_ids = list(self.citations.keys())

        # Sort citations alphabetically by first author's last name
        sorted_citations = sorted(
            [self.citations[cid] for cid in citation_ids if cid in self.citations],
            key=lambda c: (c.authors[0].split()[-1] if c.authors else "", c.year)
        )

        bibliography_entries = [
            self.format_citation(c.id, style)
            for c in sorted_citations
        ]

        # Add bibliography header
        header = "References" if style == CitationStyle.APA else "Works Cited"

        return f"{header}\n\n" + "\n\n".join(bibliography_entries)

    def link_citation_to_paragraph(self, citation_id: str, paragraph_id: str):
        """
        Link a citation to a specific paragraph

        Args:
            citation_id: Citation ID
            paragraph_id: Paragraph identifier
        """
        if citation_id in self.citations:
            self.citations[citation_id].cited_in_paragraphs.add(paragraph_id)

    def get_inline_citation(self, citation_id: str, style: CitationStyle) -> str:
        """
        Get inline citation format (e.g., "(Smith, 2020)")

        Args:
            citation_id: Citation ID
            style: Citation style

        Returns:
            Inline citation string
        """
        citation = self.citations.get(citation_id)
        if not citation:
            return "[?]"

        if style == CitationStyle.APA:
            # (Author, Year)
            author = citation.authors[0].split()[-1] if citation.authors else "Unknown"
            return f"({author}, {citation.year})"

        elif style == CitationStyle.MLA:
            # (Author Page)
            author = citation.authors[0].split()[-1] if citation.authors else "Unknown"
            page = citation.pages.split('-')[0] if citation.pages else ""
            return f"({author} {page})".strip()

        elif style == CitationStyle.CHICAGO:
            # (Author Year, Page)
            author = citation.authors[0].split()[-1] if citation.authors else "Unknown"
            return f"({author} {citation.year})"

        elif style == CitationStyle.IEEE:
            # [1]
            # Note: Would need numbering system
            return f"[{citation.id}]"

        return f"({citation.id})"

    # ========================================================================
    # Private helper methods for extraction and formatting
    # ========================================================================

    def _extract_doi(self, text: str) -> Optional[str]:
        """Extract DOI from text"""
        doi_pattern = r'10\.\d{4,}/[^\s]+'
        match = re.search(doi_pattern, text[:2000])  # Check first 2000 chars
        return match.group(0) if match else None

    def _extract_authors(self, text: str, metadata: Dict) -> List[str]:
        """Extract author names"""
        if 'authors' in metadata:
            return metadata['authors']

        # Try to extract from text (look for author patterns in first page)
        author_pattern = r'(?:^|\n)([A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z]\. [A-Z][a-z]+)*)'
        matches = re.findall(author_pattern, text[:2000])

        if matches:
            # Split comma-separated authors
            authors = []
            for match in matches:
                authors.extend([a.strip() for a in match.split(',')])
            return authors[:5]  # Max 5 authors

        return ["Unknown Author"]

    def _extract_authors_from_web(self, content: str) -> List[str]:
        """Extract authors from web page"""
        # Look for common author meta tags
        author_patterns = [
            r'<meta name="author" content="([^"]+)"',
            r'<meta property="article:author" content="([^"]+)"',
            r'by\s+([A-Z][a-z]+ [A-Z][a-z]+)',
        ]

        for pattern in author_patterns:
            match = re.search(pattern, content)
            if match:
                return [match.group(1)]

        return ["Unknown Author"]

    def _extract_title_from_text(self, text: str) -> str:
        """Extract title from document text"""
        # Usually first non-empty line or largest text
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) > 10 and len(line) < 200:
                return line
        return "Untitled"

    def _extract_year(self, text: str, metadata: Dict) -> int:
        """Extract publication year"""
        if 'year' in metadata:
            return int(metadata['year'])

        # Look for 4-digit year in first 2000 chars
        year_pattern = r'\b(19|20)\d{2}\b'
        matches = re.findall(year_pattern, text[:2000])

        if matches:
            # Return most recent year found
            years = [int(y) for y in matches]
            return max(years)

        return datetime.now().year

    def _detect_source_type(self, text: str, metadata: Dict) -> SourceType:
        """Detect type of source"""
        text_lower = text[:2000].lower()

        if 'journal' in metadata or 'journal' in text_lower:
            return SourceType.JOURNAL_ARTICLE
        elif 'conference' in text_lower or 'proceedings' in text_lower:
            return SourceType.CONFERENCE
        elif 'thesis' in text_lower or 'dissertation' in text_lower:
            return SourceType.THESIS
        elif 'chapter' in text_lower and 'book' in text_lower:
            return SourceType.CHAPTER
        elif any(word in text_lower for word in ['book', 'published by', 'isbn']):
            return SourceType.BOOK
        elif 'arxiv' in text_lower or 'preprint' in text_lower:
            return SourceType.PREPRINT

        return SourceType.REPORT  # Default

    def _extract_journal_name(self, text: str) -> Optional[str]:
        """Extract journal name from text"""
        # Common patterns
        patterns = [
            r'(?:published in|appeared in)\s+([A-Z][^.,:;\n]{5,50})',
            r'Journal of ([A-Z][^.,:;\n]{5,40})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:2000])
            if match:
                return match.group(1).strip()

        return None

    def _extract_volume(self, text: str) -> Optional[int]:
        """Extract volume number"""
        pattern = r'[Vv]ol(?:ume)?\.?\s*(\d+)'
        match = re.search(pattern, text[:2000])
        return int(match.group(1)) if match else None

    def _extract_issue(self, text: str) -> Optional[int]:
        """Extract issue number"""
        pattern = r'[Ii]ssue\.?\s*(\d+)|No\.?\s*(\d+)'
        match = re.search(pattern, text[:2000])
        if match:
            return int(match.group(1) or match.group(2))
        return None

    def _extract_pages(self, text: str) -> Optional[str]:
        """Extract page range"""
        pattern = r'pp?\.?\s*(\d+[-–]\d+)'
        match = re.search(pattern, text[:2000])
        return match.group(1) if match else None

    def _is_peer_reviewed(self, journal: Optional[str], metadata: Dict) -> bool:
        """Check if source is peer-reviewed"""
        # Would integrate with journal database in production
        # For now, simple heuristic
        if not journal:
            return False

        peer_reviewed_indicators = [
            'nature', 'science', 'cell', 'lancet', 'jama',
            'proceedings', 'ieee', 'acm', 'pnas'
        ]

        journal_lower = journal.lower()
        return any(indicator in journal_lower for indicator in peer_reviewed_indicators)

    def _is_primary_source(self, text: str) -> bool:
        """Detect if source is primary research"""
        text_lower = text[:3000].lower()

        # Look for research indicators
        primary_indicators = ['methods', 'participants', 'results', 'data collection']
        secondary_indicators = ['review', 'meta-analysis', 'survey of']

        primary_score = sum(1 for ind in primary_indicators if ind in text_lower)
        secondary_score = sum(1 for ind in secondary_indicators if ind in text_lower)

        return primary_score > secondary_score

    def _score_web_credibility(self, domain: str) -> float:
        """Score credibility of web domain"""
        # Credibility tiers
        high_credibility = [
            '.gov', '.edu', 'nih.gov', 'cdc.gov', 'who.int',
            'nature.com', 'sciencedirect.com', 'jstor.org'
        ]

        medium_credibility = [
            '.org', 'wikipedia.org', 'britannica.com'
        ]

        if any(hc in domain for hc in high_credibility):
            return 0.9
        elif any(mc in domain for mc in medium_credibility):
            return 0.6
        else:
            return 0.3  # Unknown domains get low score

    def _format_apa(self, citation: Citation) -> str:
        """Format citation in APA style (7th edition)"""
        # Authors
        if len(citation.authors) == 1:
            authors_str = citation.authors[0]
        elif len(citation.authors) == 2:
            authors_str = f"{citation.authors[0]} & {citation.authors[1]}"
        else:
            authors_str = f"{citation.authors[0]}, et al."

        # Year
        year_str = f"({citation.year})"

        # Title (italicized for books, plain for articles)
        if citation.source_type in [SourceType.BOOK, SourceType.THESIS]:
            title_str = f"*{citation.title}*"
        else:
            title_str = citation.title

        parts = [authors_str, year_str, title_str]

        # Journal-specific
        if citation.source_type == SourceType.JOURNAL_ARTICLE and citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f", {citation.volume}"
                if citation.issue:
                    journal_str += f"({citation.issue})"
            if citation.pages:
                journal_str += f", {citation.pages}"
            parts.append(journal_str)

        # DOI or URL
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)

        return ". ".join(parts) + "."

    def _format_mla(self, citation: Citation) -> str:
        """Format citation in MLA style (9th edition)"""
        # Author (Last, First)
        if citation.authors:
            first_author = citation.authors[0]
            name_parts = first_author.split()
            if len(name_parts) >= 2:
                authors_str = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
            else:
                authors_str = first_author

            if len(citation.authors) > 1:
                authors_str += ", et al"
        else:
            authors_str = "Unknown Author"

        # Title
        if citation.source_type in [SourceType.BOOK, SourceType.THESIS]:
            title_str = f"*{citation.title}*"
        else:
            title_str = f'"{citation.title}"'

        parts = [authors_str, title_str]

        # Container (journal, website, etc.)
        if citation.journal:
            parts.append(f"*{citation.journal}*")

        # Volume/Issue
        if citation.volume:
            vol_str = f"vol. {citation.volume}"
            if citation.issue:
                vol_str += f", no. {citation.issue}"
            parts.append(vol_str)

        # Year
        parts.append(str(citation.year))

        # Pages
        if citation.pages:
            parts.append(f"pp. {citation.pages}")

        return ", ".join(parts) + "."

    def _format_chicago(self, citation: Citation) -> str:
        """Format citation in Chicago style (17th edition, notes-bibliography)"""
        # Author
        if citation.authors:
            authors_str = citation.authors[0]
            if len(citation.authors) > 1:
                authors_str += " et al."
        else:
            authors_str = "Unknown Author"

        # Title
        if citation.source_type == SourceType.BOOK:
            title_str = f"*{citation.title}*"
        else:
            title_str = f'"{citation.title}"'

        parts = [authors_str, title_str]

        # Journal
        if citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f" {citation.volume}"
                if citation.issue:
                    journal_str += f", no. {citation.issue}"
            parts.append(journal_str)

        # Year and pages
        year_pages = f"({citation.year})"
        if citation.pages:
            year_pages += f": {citation.pages}"
        parts.append(year_pages)

        return ". ".join(parts) + "."

    def _format_ieee(self, citation: Citation) -> str:
        """Format citation in IEEE style"""
        # Authors
        if citation.authors:
            if len(citation.authors) <= 3:
                authors_str = ", ".join(citation.authors)
            else:
                authors_str = f"{citation.authors[0]}, et al."
        else:
            authors_str = "Unknown"

        # Title in quotes
        title_str = f'"{citation.title}"'

        parts = [authors_str, title_str]

        # Journal
        if citation.journal:
            journal_str = f"*{citation.journal}*"
            if citation.volume:
                journal_str += f", vol. {citation.volume}"
            if citation.issue:
                journal_str += f", no. {citation.issue}"
            if citation.pages:
                journal_str += f", pp. {citation.pages}"
            parts.append(journal_str)

        # Year
        parts.append(str(citation.year))

        return ", ".join(parts) + "."


# Factory function
def create_citation_manager() -> CitationManager:
    """Create a new citation manager instance"""
    return CitationManager()

"""
Citation Formatting - Perplexity-Style References
==================================================
Format search results with numbered inline citations.

Philosophy:
"Every claim needs a source."

Transforms responses from:
    "Thompson Sampling is a Bayesian exploration strategy."

To:
    "Thompson Sampling is a Bayesian exploration strategy [1]."

    Sources:
    [1] Thompson Sampling for Contextual Bandits - https://...
"""

from typing import List, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import re

from .protocol import WebSearchResult


# ============================================================================
# Citation Styles
# ============================================================================

class CitationStyle(Enum):
    """Citation formatting styles."""
    INLINE_NUMERIC = "inline_numeric"  # [1], [2], [3]
    INLINE_AUTHOR = "inline_author"    # (Thompson, 2020)
    FOOTNOTE = "footnote"              # 1, 2, 3
    SUPERSCRIPT = "superscript"        # superscript 1, 2, 3
    APA = "apa"                        # American Psychological Association
    MLA = "mla"                        # Modern Language Association


# ============================================================================
# Citation Formatter
# ============================================================================

@dataclass
class Citation:
    """Single citation entry."""
    index: int
    url: str
    title: str
    domain: str
    snippet: str
    score: float

    def format(self, style: CitationStyle = CitationStyle.INLINE_NUMERIC) -> str:
        """Format citation in specified style."""
        if style == CitationStyle.INLINE_NUMERIC:
            return f"[{self.index}] {self.title} - {self.url}"
        elif style == CitationStyle.INLINE_AUTHOR:
            # Extract author from domain (simple heuristic)
            author = self.domain.replace("www.", "").split(".")[0].title()
            return f"({author}) {self.title} - {self.url}"
        elif style == CitationStyle.FOOTNOTE:
            superscript = self._to_superscript(self.index)
            return f"{superscript} {self.title} - {self.url}"
        elif style == CitationStyle.APA:
            return f"{self.domain.title()}. {self.title}. Retrieved from {self.url}"
        elif style == CitationStyle.MLA:
            return f'"{self.title}." {self.domain.title()}, {self.url}.'
        else:
            return f"[{self.index}] {self.title} - {self.url}"

    def _to_superscript(self, n: int) -> str:
        """Convert number to superscript."""
        superscripts = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
        return str(n).translate(superscripts)


class CitationFormatter:
    """
    Format responses with inline citations.

    Detects claims in text and inserts citation markers based on
    semantic similarity to source documents.

    Usage:
        formatter = CitationFormatter(style=CitationStyle.INLINE_NUMERIC)

        cited_text = formatter.add_citations(
            response="Thompson Sampling is effective.",
            results=[result1, result2, result3]
        )

        # Output:
        # "Thompson Sampling is effective [1]."
        #
        # Sources:
        # [1] Thompson Sampling Tutorial - https://...
    """

    def __init__(
        self,
        style: CitationStyle = CitationStyle.INLINE_NUMERIC,
        auto_cite: bool = True
    ):
        self.style = style
        self.auto_cite = auto_cite

    def add_citations(
        self,
        response: str,
        results: List[WebSearchResult],
        manual_citations: Dict[str, int] = None
    ) -> Tuple[str, List[Citation]]:
        """
        Add citations to response text.

        Args:
            response: Response text to annotate
            results: Search results to cite
            manual_citations: Optional dict mapping text spans � result index

        Returns:
            Tuple of (cited_text, citation_list)
        """
        # Build citation list
        citations = []
        for i, result in enumerate(results, start=1):
            citations.append(Citation(
                index=i,
                url=result.url,
                title=result.title,
                domain=result.domain,
                snippet=result.snippet,
                score=result.final_score
            ))

        if not self.auto_cite:
            # Just append citations at end
            return self._append_citations(response, citations), citations

        # Auto-cite mode: Insert citations intelligently
        cited_text = self._auto_cite(response, results, citations, manual_citations or {})

        return cited_text, citations

    def _auto_cite(
        self,
        response: str,
        results: List[WebSearchResult],
        citations: List[Citation],
        manual_citations: Dict[str, int]
    ) -> str:
        """
        Automatically insert citations based on content similarity.

        Strategy:
        1. Split response into sentences
        2. For each sentence, find most relevant source
        3. Insert citation marker at end of sentence
        """
        # Handle manual citations first
        cited_text = response
        for text_span, index in manual_citations.items():
            marker = self._citation_marker(index)
            cited_text = cited_text.replace(text_span, f"{text_span} {marker}")

        # Split into sentences
        sentences = self._split_sentences(cited_text)

        # For each sentence, add citation if not already present
        result_snippets = [r.snippet for r in results]
        cited_sentences = []

        for sentence in sentences:
            # Skip if already has citation
            if self._has_citation(sentence):
                cited_sentences.append(sentence)
                continue

            # Find best matching source (simple keyword overlap)
            best_index = self._find_best_match(sentence, result_snippets)

            if best_index is not None:
                marker = self._citation_marker(best_index + 1)
                # Insert before final punctuation
                sentence = self._insert_marker(sentence, marker)

            cited_sentences.append(sentence)

        return " ".join(cited_sentences)

    def _append_citations(self, response: str, citations: List[Citation]) -> str:
        """Append citation list to end of response."""
        cited_text = response + "\n\nSources:\n"
        for citation in citations:
            cited_text += citation.format(self.style) + "\n"
        return cited_text

    def _citation_marker(self, index: int) -> str:
        """Generate citation marker for style."""
        if self.style == CitationStyle.INLINE_NUMERIC:
            return f"[{index}]"
        elif self.style == CitationStyle.FOOTNOTE:
            return Citation(index, "", "", "", "", 0.0)._to_superscript(index)
        else:
            return f"[{index}]"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter (improve with NLTK/spaCy if needed)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _has_citation(self, sentence: str) -> bool:
        """Check if sentence already has citation."""
        return bool(re.search(r'\[\d+\]|[\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079]+', sentence))

    def _find_best_match(self, sentence: str, snippets: List[str]) -> int | None:
        """Find best matching snippet for sentence (simple keyword overlap)."""
        sentence_lower = sentence.lower()
        sentence_words = set(re.findall(r'\w+', sentence_lower))

        if not sentence_words:
            return None

        best_score = 0.0
        best_index = None

        for i, snippet in enumerate(snippets):
            snippet_lower = snippet.lower()
            snippet_words = set(re.findall(r'\w+', snippet_lower))

            if not snippet_words:
                continue

            # Jaccard similarity
            overlap = sentence_words & snippet_words
            union = sentence_words | snippet_words
            score = len(overlap) / len(union) if union else 0.0

            if score > best_score:
                best_score = score
                best_index = i

        # Only cite if reasonable match (>10% overlap)
        return best_index if best_score > 0.1 else None

    def _insert_marker(self, sentence: str, marker: str) -> str:
        """Insert citation marker before final punctuation."""
        # Match final punctuation
        match = re.search(r'([.!?])$', sentence)
        if match:
            # Insert before punctuation
            pos = match.start()
            return sentence[:pos] + f" {marker}" + sentence[pos:]
        else:
            # Append at end
            return sentence + f" {marker}"

    def format_bibliography(
        self,
        citations: List[Citation],
        title: str = "Sources"
    ) -> str:
        """
        Format complete bibliography.

        Args:
            citations: List of citations
            title: Section title

        Returns:
            Formatted bibliography string
        """
        biblio = f"\n\n{title}:\n"
        for citation in citations:
            biblio += citation.format(self.style) + "\n"
        return biblio

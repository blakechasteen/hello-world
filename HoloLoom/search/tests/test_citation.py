"""
Tests for Citation System
==========================
Comprehensive tests for CitationFormatter and citation styles.
"""

import pytest
from HoloLoom.search.citation import (
    CitationFormatter,
    CitationStyle,
    Citation
)
from HoloLoom.search.protocol import WebSearchResult, SearchResultType


class TestCitation:
    """Test Citation dataclass."""

    def test_citation_creation(self):
        """Test creating citation objects."""
        citation = Citation(
            index=1,
            url="https://example.com",
            title="Test Article",
            domain="example.com",
            snippet="This is a test snippet",
            score=0.95
        )

        assert citation.index == 1
        assert citation.url == "https://example.com"
        assert citation.title == "Test Article"

    def test_citation_format_inline_numeric(self):
        """Test inline numeric citation format [1]."""
        citation = Citation(1, "https://example.com", "Test", "example.com", "snippet", 0.9)
        formatted = citation.format(CitationStyle.INLINE_NUMERIC)

        assert "[1]" in formatted
        assert "Test" in formatted
        assert "https://example.com" in formatted

    def test_citation_format_apa(self):
        """Test APA citation format."""
        citation = Citation(1, "https://example.com", "Test Article", "example.com", "snippet", 0.9)
        formatted = citation.format(CitationStyle.APA)

        assert "Test Article" in formatted
        assert "https://example.com" in formatted

    def test_citation_format_mla(self):
        """Test MLA citation format."""
        citation = Citation(1, "https://example.com", "Test Article", "example.com", "snippet", 0.9)
        formatted = citation.format(CitationStyle.MLA)

        assert "Test Article" in formatted
        assert "https://example.com" in formatted
        assert "Example.Com" in formatted


class TestCitationFormatter:
    """Test CitationFormatter class."""

    @pytest.fixture
    def formatter(self):
        """Create formatter instance."""
        return CitationFormatter(style=CitationStyle.INLINE_NUMERIC)

    @pytest.fixture
    def sample_results(self):
        """Create sample search results."""
        return [
            WebSearchResult(
                url="https://example.com/1",
                title="Article 1",
                snippet="Thompson Sampling is a Bayesian approach.",
                domain="example.com",
                final_score=0.95
            ),
            WebSearchResult(
                url="https://example.com/2",
                title="Article 2",
                snippet="It balances exploration and exploitation.",
                domain="example.com",
                final_score=0.90
            ),
            WebSearchResult(
                url="https://example.com/3",
                title="Article 3",
                snippet="Thompson Sampling uses posterior distributions.",
                domain="example.com",
                final_score=0.85
            )
        ]

    def test_add_citations_manual_mode(self, formatter, sample_results):
        """Test manual citation mode (no auto-cite)."""
        formatter.auto_cite = False

        response = "Thompson Sampling is effective."
        cited_text, citations = formatter.add_citations(response, sample_results)

        # Should append sources at end
        assert "Sources:" in cited_text
        assert "[1]" in cited_text
        assert "Article 1" in cited_text

    def test_add_citations_auto_mode(self, formatter, sample_results):
        """Test automatic citation insertion."""
        formatter.auto_cite = True

        response = "Thompson Sampling is a Bayesian approach. It balances exploration."
        cited_text, citations = formatter.add_citations(response, sample_results)

        # Should have inline citations
        assert "[1]" in cited_text or "[2]" in cited_text

    def test_citation_indexing(self, formatter, sample_results):
        """Test citations are properly indexed [1], [2], [3]."""
        response = "Test response"
        cited_text, citations = formatter.add_citations(response, sample_results)

        assert len(citations) == 3
        assert citations[0].index == 1
        assert citations[1].index == 2
        assert citations[2].index == 3

    def test_format_bibliography(self, formatter, sample_results):
        """Test bibliography formatting."""
        _, citations = formatter.add_citations("Test", sample_results)
        bibliography = formatter.format_bibliography(citations, title="References")

        assert "References:" in bibliography
        assert "[1]" in bibliography
        assert "[2]" in bibliography
        assert "[3]" in bibliography
        assert "Article 1" in bibliography

    def test_empty_results(self, formatter):
        """Test handling of empty search results."""
        response = "Test response"
        cited_text, citations = formatter.add_citations(response, [])

        assert cited_text == response  # No changes when no results
        assert len(citations) == 0

    def test_different_styles(self, sample_results):
        """Test different citation styles."""
        response = "Thompson Sampling is effective."

        # Test each style
        for style in [CitationStyle.INLINE_NUMERIC, CitationStyle.APA, CitationStyle.MLA]:
            formatter = CitationFormatter(style=style, auto_cite=False)
            cited_text, citations = formatter.add_citations(response, sample_results)

            # Should have citations in the text
            assert len(citations) == 3
            assert len(cited_text) > len(response)

    def test_manual_citations(self, formatter, sample_results):
        """Test manual citation specification."""
        response = "Thompson Sampling is effective according to research."
        manual_cites = {
            "Thompson Sampling": 1,
            "according to research": 2
        }

        cited_text, citations = formatter.add_citations(
            response,
            sample_results,
            manual_citations=manual_cites
        )

        assert "[1]" in cited_text
        assert "[2]" in cited_text


class TestCitationEdgeCases:
    """Test edge cases and error handling."""

    def test_duplicate_citations(self):
        """Test handling of duplicate citations."""
        formatter = CitationFormatter(auto_cite=True)
        results = [
            WebSearchResult(
                url="https://example.com/1",
                title="Same Article",
                snippet="Test snippet",
                domain="example.com",
                final_score=0.9
            ),
            WebSearchResult(
                url="https://example.com/1",  # Same URL
                title="Same Article",
                snippet="Test snippet",
                domain="example.com",
                final_score=0.9
            )
        ]

        response = "Test response"
        cited_text, citations = formatter.add_citations(response, results)

        # Should still create separate citations (not deduplicated)
        assert len(citations) == 2

    def test_long_response(self):
        """Test with long response text."""
        formatter = CitationFormatter(auto_cite=True)
        results = [
            WebSearchResult(
                url="https://example.com",
                title="Test",
                snippet="Thompson Sampling",
                domain="example.com",
                final_score=0.9
            )
        ]

        # Long response with multiple sentences
        response = ". ".join([f"Sentence {i}" for i in range(100)])
        cited_text, citations = formatter.add_citations(response, results)

        assert len(citations) == 1
        # Should complete without errors

    def test_special_characters(self):
        """Test handling of special characters in citations."""
        formatter = CitationFormatter()
        results = [
            WebSearchResult(
                url="https://example.com/special?query=1&test=2",
                title='Article with "quotes" and <tags>',
                snippet="Text with special chars: & < > \"",
                domain="example.com",
                final_score=0.9
            )
        ]

        response = "Test response"
        cited_text, citations = formatter.add_citations(response, results)

        # Should handle special characters without errors
        assert len(citations) == 1
        assert citations[0].url == "https://example.com/special?query=1&test=2"


class TestCitationIntegration:
    """Integration tests with real search results."""

    def test_end_to_end_citation(self):
        """Test complete citation workflow."""
        # Create realistic search results
        results = [
            WebSearchResult(
                url="https://arxiv.org/abs/1234.5678",
                title="Thompson Sampling for Contextual Bandits",
                snippet="We present an analysis of Thompson Sampling...",
                domain="arxiv.org",
                score_96d=0.85,
                score_192d=0.90,
                score_384d=0.95,
                final_score=0.95,
                original_rank=1,
                final_rank=1,
                result_type=SearchResultType.ACADEMIC
            ),
            WebSearchResult(
                url="https://en.wikipedia.org/wiki/Thompson_sampling",
                title="Thompson sampling - Wikipedia",
                snippet="Thompson sampling is a heuristic for choosing actions...",
                domain="wikipedia.org",
                score_96d=0.80,
                score_192d=0.85,
                score_384d=0.90,
                final_score=0.90,
                original_rank=2,
                final_rank=2,
                result_type=SearchResultType.ORGANIC
            )
        ]

        # Create response
        response = """Thompson Sampling is a Bayesian approach to the exploration-exploitation dilemma.
It maintains a probability distribution over parameters and samples from this distribution at each step.
The algorithm is provably optimal under certain conditions."""

        # Format with citations
        formatter = CitationFormatter(auto_cite=True)
        cited_text, citations = formatter.add_citations(response, results)

        # Verify citations were added
        assert "[1]" in cited_text or "[2]" in cited_text
        assert len(citations) == 2

        # Verify bibliography
        bibliography = formatter.format_bibliography(citations)
        assert "Thompson Sampling for Contextual Bandits" in bibliography
        assert "arxiv.org" in bibliography

"""
Unit tests for BossPig Business Document Analyzer - Basic functionality.

Test scope: Document loading, fixture validation, basic pattern matching.

Created: 2025-11-22
Author: Agent C (Haiku)
"""

import pytest
from pathlib import Path


class TestBossPigFixtures:
    """Test that BossPig fixtures load correctly."""

    def test_good_document_fixture_exists(self, good_document_file: Path):
        """Good document fixture should be loadable."""
        assert good_document_file.exists()
        content = good_document_file.read_text()
        assert "Product Proposal" in content or "Proposal" in content
        # Should have structure
        assert "#" in content  # Markdown headers

    def test_bad_document_slop_fixture_exists(self, bad_document_slop_file: Path):
        """Bad document slop fixture should be loadable."""
        assert bad_document_slop_file.exists()
        content = bad_document_slop_file.read_text()
        assert "ISSUE" in content  # Contains marked issues
        assert len(content) > 1000  # Substantial fixture


class TestDocumentStructure:
    """Test document structure and formatting."""

    def test_good_document_has_structure(self, good_document_file: Path):
        """Good document should have clear structure."""
        content = good_document_file.read_text()
        # Should have headers
        assert content.count("#") >= 5
        # Should have sections
        sections = ["Executive Summary", "Problem", "Solution", "Timeline", "Budget"]
        found_sections = sum(1 for s in sections if s in content)
        assert found_sections >= 3

    def test_good_document_has_dates(self, good_document_file: Path):
        """Good document should have specific dates."""
        content = good_document_file.read_text()
        # Should have specific dates, not "TBD"
        assert "2026" in content or "2025" in content
        # Should not have "TBD" for dates
        assert content.count("TBD") < 3  # Minimal TBD

    def test_good_document_has_metrics(self, good_document_file: Path):
        """Good document should have measurable metrics."""
        content = good_document_file.read_text()
        # Should have numbers and percentages
        assert "%" in content or any(c.isdigit() for c in content)
        # Should have success metrics
        assert "metric" in content.lower() or "measure" in content.lower()

    def test_bad_document_has_vague_language(self, bad_document_slop_file: Path):
        """Bad document should have vague language."""
        content = bad_document_slop_file.read_text()
        vague_words = ["TBD", "probably", "might", "maybe", "unclear"]
        found_vague = sum(1 for w in vague_words if w.lower() in content.lower())
        assert found_vague >= 2

    def test_bad_document_lacks_dates(self, bad_document_slop_file: Path):
        """Bad document should lack specific dates."""
        content = bad_document_slop_file.read_text()
        # Should have many "TBD" or missing dates
        assert "TBD" in content or "TBD)" in content
        assert content.count("TBD") >= 3


class TestDocumentQuality:
    """Test document quality indicators."""

    def test_good_document_clarity(self, good_document_file: Path):
        """Good document should be clear and specific."""
        content = good_document_file.read_text()
        # Good: Should have table of contents or clear sections
        assert content.count("|") >= 3  # Has tables (structured data)

    def test_good_document_accountability(self, good_document_file: Path):
        """Good document should have clear ownership."""
        content = good_document_file.read_text()
        # Should have owner names or roles
        ownership_words = ["Owner", "Lead", "Responsible", "Contact"]
        found_ownership = sum(1 for w in ownership_words if w in content)
        assert found_ownership >= 2

    def test_bad_document_passive_voice(self, bad_document_slop_file: Path):
        """Bad document should have passive voice."""
        content = bad_document_slop_file.read_text()
        # Passive voice indicators
        passive_patterns = ["is being", "will be", "is recommended", "should be"]
        found_passive = sum(1 for p in passive_patterns if p in content)
        assert found_passive >= 2

    def test_bad_document_jargon(self, bad_document_slop_file: Path):
        """Bad document should have corporate jargon."""
        content = bad_document_slop_file.read_text()
        jargon = ["synergy", "leverage", "paradigm", "stakeholder value"]
        found_jargon = sum(1 for j in jargon if j.lower() in content.lower())
        assert found_jargon >= 1


class TestDocumentAnalysis:
    """Test document analysis patterns."""

    def test_good_document_word_count(self, good_document_file: Path):
        """Good document should have reasonable length."""
        content = good_document_file.read_text()
        words = len(content.split())
        # Good document: 500-3000 words
        assert 300 < words < 5000, f"Word count: {words}"

    def test_bad_document_word_count(self, bad_document_slop_file: Path):
        """Bad document should have reasonable length."""
        content = bad_document_slop_file.read_text()
        words = len(content.split())
        # Should be similar length to good doc for comparison
        assert 300 < words < 5000, f"Word count: {words}"

    def test_document_readability_indicators(self, good_document_file: Path):
        """Good document should have readability indicators."""
        content = good_document_file.read_text()
        lines = content.split('\n')
        # Should have reasonable line lengths (not all super long)
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        assert avg_line_length < 120, "Lines should be readable length"


class TestFixtureCoverage:
    """Test that fixtures cover expected quality issues."""

    def test_good_document_covers_best_practices(self, good_document_file: Path):
        """Good document should demonstrate best practices."""
        content = good_document_file.read_text()
        # Should have all key sections
        best_practices = {
            "clear_dates": "2026" in content or "2025" in content,
            "specific_numbers": any(str(i) in content for i in range(0, 1000)),
            "clear_ownership": "Owner" in content or "Lead" in content or "Contact" in content,
            "measurable_metrics": "%" in content or "metric" in content.lower(),
        }
        assert sum(best_practices.values()) >= 3, "Missing best practices"

    def test_bad_document_covers_issues(self, bad_document_slop_file: Path):
        """Bad document should demonstrate multiple issues."""
        content = bad_document_slop_file.read_text()
        issue_count = content.count("ISSUE")
        assert issue_count >= 10, f"Expected 10+ issues, found {issue_count}"

    def test_bad_document_has_ai_markers(self, bad_document_slop_file: Path):
        """Bad document should have AI hallucination markers."""
        content = bad_document_slop_file.read_text()
        # Should have TODO, TBD, placeholders
        ai_markers = ["TODO", "TBD", "[", "]", "hallucination"]
        found_markers = sum(1 for m in ai_markers if m in content)
        assert found_markers >= 2


@pytest.mark.unit
@pytest.mark.bosspig
class TestBossPigIntegration:
    """Integration tests for BossPig with fixtures."""

    def test_both_documents_loadable(self, good_document_file, bad_document_slop_file):
        """Both documents should be loadable simultaneously."""
        assert good_document_file.exists()
        assert bad_document_slop_file.exists()
        good = good_document_file.read_text()
        bad = bad_document_slop_file.read_text()
        assert len(good) > 100
        assert len(bad) > 100

    def test_documents_are_different(self, good_document_file, bad_document_slop_file):
        """Good and bad documents should be significantly different."""
        good = good_document_file.read_text()
        bad = bad_document_slop_file.read_text()
        # Calculate basic difference metric
        differences = sum(1 for a, b in zip(good, bad) if a != b)
        assert differences > len(good) * 0.1, "Documents too similar"


class TestComparativeAnalysis:
    """Test comparing good vs bad documents."""

    def test_good_doc_more_specific(self, good_document_file, bad_document_slop_file):
        """Good document should be more specific."""
        good = good_document_file.read_text()
        bad = bad_document_slop_file.read_text()

        # Count TBD usage
        good_tbd = good.lower().count("tbd")
        bad_tbd = bad.lower().count("tbd")

        assert bad_tbd > good_tbd, "Bad doc should have more TBD"

    def test_good_doc_better_metrics(self, good_document_file, bad_document_slop_file):
        """Good document should have better metrics."""
        good = good_document_file.read_text()
        bad = bad_document_slop_file.read_text()

        # Count table usage (good docs have tables)
        good_pipes = good.count("|")
        bad_pipes = bad.count("|")

        assert good_pipes >= bad_pipes, "Good doc should have tables"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

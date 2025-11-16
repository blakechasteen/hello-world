"""
Tests for GitHub, Slack, Email, and PDF Spinners
=================================================

Comprehensive test suite covering:
- GitHub spinner (API integration, parsing)
- Slack spinner (message extraction, threading)
- Email spinner (IMAP, message parsing)
- PDF spinner (text extraction, metadata)

Author: Claude Code
Date: November 2025
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

# Import spinners
try:
    from HoloLoom.spinningWheel.github_spinner import (
        GitHubSpinner, GitHubClient, GitHubIssue, GitHubPullRequest
    )
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

try:
    from HoloLoom.spinningWheel.slack_spinner import (
        SlackSpinner, SlackClient, SlackMessage
    )
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

try:
    from HoloLoom.spinningWheel.email_spinner import (
        EmailSpinner, IMAPClient, EmailMessage
    )
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from HoloLoom.spinningWheel.pdf_spinner import (
        PDFSpinner, PDFExtractor, PDFPage, PDFMetadata
    )
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from HoloLoom.spinningWheel.protocol import SpinnerCapabilities


# ============================================================================
# GitHub Spinner Tests
# ============================================================================

@pytest.mark.skipif(not GITHUB_AVAILABLE, reason="GitHub spinner not available")
class TestGitHubSpinner:
    """Test GitHub spinner functionality."""

    def test_parse_repo_url(self):
        """Test GitHub URL parsing."""
        owner, repo = GitHubClient.parse_repo_url("https://github.com/anthropic-ai/anthropic-sdk-python")
        assert owner == "anthropic-ai"
        assert repo == "anthropic-sdk-python"

    def test_parse_repo_url_trailing_slash(self):
        """Test URL parsing with trailing slash."""
        owner, repo = GitHubClient.parse_repo_url("https://github.com/user/repo/")
        assert owner == "user"
        assert repo == "repo"

    def test_capabilities(self):
        """Test spinner capabilities."""
        spinner = GitHubSpinner()
        caps = spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.importance_scoring is True
        assert caps.entity_extraction is True
        assert caps.motif_extraction is True

    def test_is_available(self):
        """Test availability check."""
        spinner = GitHubSpinner()
        assert isinstance(spinner.is_available(), bool)

    def test_issue_to_shard(self):
        """Test converting issue to shard."""
        spinner = GitHubSpinner()

        issue = GitHubIssue(
            number=42,
            title="Test Issue",
            body="This is a test issue",
            state="open",
            author="testuser",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
            labels=["bug", "urgent"]
        )

        shard = spinner._issue_to_shard(issue, "user", "repo")

        assert "Test Issue" in shard.text
        assert "testuser" in shard.entities
        assert "bug" in shard.motifs
        assert shard.metadata['number'] == 42
        assert shard.metadata['state'] == "open"

    def test_pr_to_shard(self):
        """Test converting PR to shard."""
        spinner = GitHubSpinner()

        pr = GitHubPullRequest(
            number=1,
            title="Test PR",
            body="This is a test PR",
            state="merged",
            author="testuser",
            created_at="2025-01-01T00:00:00Z",
            updated_at="2025-01-02T00:00:00Z",
            merged_at="2025-01-02T00:00:00Z",
            additions=100,
            deletions=50,
            changed_files=5
        )

        shard = spinner._pr_to_shard(pr, "user", "repo")

        assert "Test PR" in shard.text
        assert "+100" in shard.text
        assert "merged" in shard.motifs
        assert shard.metadata['state'] == "merged"


@pytest.mark.skipif(not SLACK_AVAILABLE, reason="Slack spinner not available")
class TestSlackSpinner:
    """Test Slack spinner functionality."""

    def test_capabilities(self):
        """Test spinner capabilities."""
        spinner = SlackSpinner("test-token")
        caps = spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.importance_scoring is True
        assert caps.entity_extraction is True

    def test_message_importance_scoring(self):
        """Test message importance scoring."""
        spinner = SlackSpinner("test-token")

        msg = SlackMessage(
            ts="1609459200.000100",
            user="U12345",
            username="user",
            text="This is an important message with multiple lines\nand detailed content",
            reactions={"thumbsup": 3, "heart": 2},
            reply_count=5
        )

        score = spinner._score_message_importance(msg)

        assert 0.0 <= score.score <= 1.0
        assert score.engagement_score > 0


@pytest.mark.skipif(not EMAIL_AVAILABLE, reason="Email spinner not available")
class TestEmailSpinner:
    """Test Email spinner functionality."""

    def test_capabilities(self):
        """Test spinner capabilities."""
        spinner = EmailSpinner(
            imap_server="imap.example.com",
            email_address="test@example.com",
            password="password"
        )
        caps = spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.importance_scoring is True
        assert caps.entity_extraction is True

    def test_is_available(self):
        """Test availability check."""
        spinner = EmailSpinner(
            imap_server="imap.example.com",
            email_address="test@example.com",
            password="password"
        )
        assert spinner.is_available() is True

    def test_email_list_parsing(self):
        """Test email list parsing."""
        emails = IMAPClient._parse_email_list("user1@example.com, User Two <user2@example.com>")

        assert len(emails) == 2
        assert "user1@example.com" in emails
        assert "user2@example.com" in emails


@pytest.mark.skipif(not PDF_AVAILABLE, reason="PDF spinner not available")
class TestPDFSpinner:
    """Test PDF spinner functionality."""

    def test_capabilities(self):
        """Test spinner capabilities."""
        spinner = PDFSpinner()
        caps = spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.importance_scoring is True
        assert caps.motif_extraction is True

    def test_is_available(self):
        """Test availability check."""
        spinner = PDFSpinner()
        assert isinstance(spinner.is_available(), bool)

    def test_page_to_shard(self):
        """Test converting PDF page to shard."""
        spinner = PDFSpinner()

        page = PDFPage(
            page_number=1,
            text="This is page 1 content"
        )

        metadata = PDFMetadata(
            title="Test Document",
            author="Test Author",
            num_pages=10
        )

        shard = spinner._page_to_shard(page, "/path/to/test.pdf", metadata)

        assert "Test Document" in shard.text
        assert "page_number" in shard.metadata
        assert shard.metadata['page_number'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

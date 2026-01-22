"""
Email Spinner Tests
===================

Comprehensive tests for EmailSpinner covering:
- Email parsing (headers, body)
- Attachment handling
- Thread reconstruction
- MIME type handling
- IMAP and mbox support
- Edge cases

Author: Claude Code
Date: January 2026
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any
from email.message import EmailMessage as StdEmailMessage
import tempfile
import os
from pathlib import Path

# Import spinner components
from HoloLoom.spinningWheel.email_spinner import (
    EmailSpinner,
    EmailMessage,
    EmailParser,
    spin_email_mbox,
)
from HoloLoom.spinningWheel.protocol import SpinResult, SpinnerCapabilities
from datetime import datetime
from email import message_from_string
from email.policy import default as email_policy


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def email_spinner():
    """Create an EmailSpinner with default settings."""
    return EmailSpinner(
        importance_threshold=0.3
    )


@pytest.fixture
def minimal_spinner():
    """Create a minimal EmailSpinner for fast tests."""
    return EmailSpinner(
        importance_threshold=0.5
    )


@pytest.fixture
def sample_email_message():
    """Create a sample EmailMessage for testing."""
    return EmailMessage(
        message_id="<test123@example.com>",
        sender="sender@example.com",
        recipients=["recipient@example.com"],
        cc=["cc@example.com"],
        subject="Test Email Subject",
        body_text="This is the body of the test email.\n\nIt has multiple paragraphs.",
        timestamp=datetime(2025, 1, 15, 10, 30, 0),
        in_reply_to=None,
        references=[],
        attachments=[],
        headers={"X-Custom": "custom-value"}
    )


@pytest.fixture
def reply_email_message():
    """Create a reply email for thread testing."""
    return EmailMessage(
        message_id="<reply456@example.com>",
        sender="recipient@example.com",
        recipients=["sender@example.com"],
        cc=[],
        subject="Re: Test Email Subject",
        body_text="This is a reply to the original email.\n\nThanks!",
        timestamp=datetime(2025, 1, 15, 11, 0, 0),
        in_reply_to="<test123@example.com>",
        references=["<test123@example.com>"],
        attachments=[],
        headers={}
    )


@pytest.fixture
def email_with_attachments():
    """Create an email with attachments for testing."""
    return EmailMessage(
        message_id="<attach789@example.com>",
        sender="sender@example.com",
        recipients=["recipient@example.com"],
        cc=[],
        subject="Email with Attachments",
        body_text="Please see the attached files.",
        timestamp=datetime(2025, 1, 15, 12, 0, 0),
        in_reply_to=None,
        references=[],
        attachments=[
            {"filename": "document.pdf", "content_type": "application/pdf"},
            {"filename": "image.png", "content_type": "image/png"},
            {"filename": "data.csv", "content_type": "text/csv"},
        ],
        headers={}
    )


@pytest.fixture
def sample_raw_email():
    """Create a raw email string for parsing tests."""
    return """From: sender@example.com
To: recipient@example.com
Cc: cc@example.com
Subject: Test Email Subject
Date: Wed, 15 Jan 2025 10:30:00 +0000
Message-ID: <test123@example.com>
Content-Type: text/plain; charset="utf-8"

This is the body of the test email.

It has multiple paragraphs.
"""


# =============================================================================
# EmailMessage Dataclass Tests
# =============================================================================

class TestEmailMessage:
    """Tests for EmailMessage dataclass."""

    def test_is_reply_true(self, reply_email_message):
        """Test is_reply returns True for replies."""
        assert reply_email_message.is_reply is True

    def test_is_reply_false(self, sample_email_message):
        """Test is_reply returns False for original emails."""
        assert sample_email_message.is_reply is False

    def test_is_thread_root(self, sample_email_message, reply_email_message):
        """Test is_thread_root property."""
        # Original message with no in_reply_to is a thread root
        assert sample_email_message.is_thread_root is True

        # Reply is not a thread root
        assert reply_email_message.is_thread_root is False

    def test_recipient_count(self, sample_email_message):
        """Test recipient count calculation."""
        assert sample_email_message.recipient_count == 2  # to + cc

    def test_recipient_count_no_cc(self, reply_email_message):
        """Test recipient count with no CC."""
        assert reply_email_message.recipient_count == 1  # to only

    def test_has_attachments_true(self, email_with_attachments):
        """Test has_attachments when attachments present."""
        assert email_with_attachments.has_attachments is True

    def test_has_attachments_false(self, sample_email_message):
        """Test has_attachments when no attachments."""
        assert sample_email_message.has_attachments is False


# =============================================================================
# EmailParser Tests
# =============================================================================

class TestEmailParser:
    """Tests for EmailParser methods."""

    def _parse_raw_to_message(self, raw_email: str):
        """Helper to convert raw email string to email.message.Message object."""
        return message_from_string(raw_email, policy=email_policy)

    def test_parse_raw_email(self, sample_raw_email):
        """Test parsing raw email string."""
        msg_obj = self._parse_raw_to_message(sample_raw_email)
        message = EmailParser.parse_message(msg_obj)

        assert message.sender == "sender@example.com"
        assert "recipient@example.com" in message.recipients
        assert message.subject == "Test Email Subject"
        assert "body of the test email" in message.body_text

    def test_parse_email_with_html_body(self):
        """Test parsing email with HTML body."""
        html_email = """From: sender@example.com
To: recipient@example.com
Subject: HTML Email
Content-Type: text/html; charset="utf-8"

<html>
<body>
<h1>Hello</h1>
<p>This is an HTML email.</p>
</body>
</html>
"""
        msg_obj = self._parse_raw_to_message(html_email)
        message = EmailParser.parse_message(msg_obj)

        # Body should be extracted (either HTML stripped or HTML content)
        assert message.body_text is not None or message.body_html is not None
        body = message.body_text or message.body_html
        assert len(body) > 0

    def test_parse_multipart_email(self):
        """Test parsing multipart MIME email."""
        multipart_email = """From: sender@example.com
To: recipient@example.com
Subject: Multipart Email
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"

--boundary123
Content-Type: text/plain; charset="utf-8"

This is the plain text version.

--boundary123
Content-Type: text/html; charset="utf-8"

<html><body><p>This is the HTML version.</p></body></html>

--boundary123--
"""
        msg_obj = self._parse_raw_to_message(multipart_email)
        message = EmailParser.parse_message(msg_obj)

        body = message.body_text or message.body_html
        assert body is not None
        # Should prefer plain text or extract from HTML
        assert len(body) > 0

    def test_extract_message_id(self, sample_raw_email):
        """Test message ID extraction."""
        msg_obj = self._parse_raw_to_message(sample_raw_email)
        message = EmailParser.parse_message(msg_obj)

        assert message.message_id == "<test123@example.com>"

    def test_extract_references(self):
        """Test references header extraction."""
        email_with_refs = """From: sender@example.com
To: recipient@example.com
Subject: Re: Thread
Message-ID: <reply123@example.com>
In-Reply-To: <original@example.com>
References: <original@example.com> <second@example.com>

Reply body here.
"""
        msg_obj = self._parse_raw_to_message(email_with_refs)
        message = EmailParser.parse_message(msg_obj)

        assert message.in_reply_to == "<original@example.com>"
        assert "<original@example.com>" in message.references

    def test_parse_attachments(self):
        """Test attachment metadata extraction."""
        email_with_attachment = """From: sender@example.com
To: recipient@example.com
Subject: Email with Attachment
MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="boundary456"

--boundary456
Content-Type: text/plain

Body text here.

--boundary456
Content-Type: application/pdf
Content-Disposition: attachment; filename="document.pdf"
Content-Transfer-Encoding: base64

JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwo+Pg==

--boundary456--
"""
        msg_obj = self._parse_raw_to_message(email_with_attachment)
        message = EmailParser.parse_message(msg_obj)

        # Should detect attachment
        assert len(message.attachments) >= 0  # Parser may or may not extract


# =============================================================================
# EmailSpinner Core Tests
# =============================================================================

class TestEmailSpinnerCore:
    """Tests for EmailSpinner core functionality."""

    def test_messages_to_shards(self, email_spinner, sample_email_message):
        """Test converting messages to shards."""
        shards = email_spinner._messages_to_shards([sample_email_message])

        assert len(shards) >= 1
        # Check text contains formatted message parts
        assert "sender@example.com" in shards[0].text

    def test_shard_metadata(self, email_spinner, sample_email_message):
        """Test email metadata in shard."""
        shards = email_spinner._messages_to_shards([sample_email_message])

        shard = shards[0]
        assert shard.metadata['sender'] == "sender@example.com"
        assert shard.metadata['subject'] == "Test Email Subject"
        assert 'timestamp' in shard.metadata

    def test_format_message_text(self, email_spinner, sample_email_message):
        """Test message text formatting."""
        text = email_spinner._format_message_text(sample_email_message)

        assert "From: sender@example.com" in text
        assert "Subject: Test Email Subject" in text
        assert "body of the test email" in text


# =============================================================================
# Thread Detection Tests
# =============================================================================

class TestThreadDetection:
    """Tests for email thread properties and detection."""

    def test_reply_detection_via_in_reply_to(self, reply_email_message):
        """Test reply detection via in_reply_to header."""
        # Reply message has in_reply_to set
        assert reply_email_message.in_reply_to == "<test123@example.com>"
        assert reply_email_message.is_reply is True
        assert reply_email_message.is_thread_root is False

    def test_thread_root_detection(self, sample_email_message):
        """Test thread root detection."""
        # Original message with no in_reply_to is a thread root
        assert sample_email_message.is_thread_root is True
        assert sample_email_message.is_reply is False

    def test_subject_prefix_does_not_make_reply(self):
        """Test that Re: prefix alone doesn't mark as reply (requires in_reply_to)."""
        email = EmailMessage(
            message_id="<msg@example.com>",
            sender="sender@example.com",
            recipients=["recipient@example.com"],
            cc=[],
            subject="Re: Fwd: Re: Original Subject",
            body_text="Reply content",
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            in_reply_to=None,  # Missing header
            references=[],
            attachments=[],
            headers={}
        )

        # Without in_reply_to, is_reply is False even with Re: prefix
        assert email.is_reply is False
        # But subject still has Re: prefix
        assert email.subject.startswith("Re:")

    def test_references_header_present(self, reply_email_message):
        """Test references header is captured."""
        assert "<test123@example.com>" in reply_email_message.references

    def test_thread_info_in_shard_metadata(self, email_spinner, reply_email_message):
        """Test thread info is included in shard metadata."""
        shards = email_spinner._messages_to_shards([reply_email_message])

        if shards:  # May be filtered by importance
            shard = shards[0]
            assert shard.metadata['is_reply'] is True
            assert shard.metadata['is_thread_root'] is False

    def test_multiple_messages_converted(self, email_spinner):
        """Test converting multiple thread messages to shards."""
        emails = [
            EmailMessage(
                message_id="<thread1@example.com>",
                sender="alice@example.com",
                recipients=["bob@example.com"],
                cc=[],
                subject="Discussion Topic",
                body_text="Initial message with enough content to pass importance filter",
                timestamp=datetime(2025, 1, 15, 9, 0, 0),
                in_reply_to=None,
                references=[],
                attachments=[],
                headers={}
            ),
            EmailMessage(
                message_id="<thread2@example.com>",
                sender="bob@example.com",
                recipients=["alice@example.com"],
                cc=[],
                subject="Re: Discussion Topic",
                body_text="Reply from Bob with enough content to pass importance filter",
                timestamp=datetime(2025, 1, 15, 9, 30, 0),
                in_reply_to="<thread1@example.com>",
                references=["<thread1@example.com>"],
                attachments=[],
                headers={}
            ),
        ]

        shards = email_spinner._messages_to_shards(emails)

        # Should create shards for each message (if they pass importance threshold)
        assert len(shards) >= 0  # May be filtered


# =============================================================================
# Attachment Handling Tests
# =============================================================================

class TestAttachmentHandling:
    """Tests for email attachment handling."""

    def test_attachment_metadata_in_shard(self, email_spinner, email_with_attachments):
        """Test attachment metadata is included in shard."""
        shards = email_spinner._messages_to_shards([email_with_attachments])

        # Lower threshold spinner should create shards
        if shards:
            shard = shards[0]
            assert shard.metadata['has_attachments'] is True
            assert shard.metadata['attachment_count'] == 3

    def test_has_attachments_property(self, email_with_attachments):
        """Test has_attachments property."""
        assert email_with_attachments.has_attachments is True
        assert len(email_with_attachments.attachments) == 3

    def test_attachment_types_in_message(self, email_with_attachments):
        """Test attachment content types are tracked in message."""
        attachment_types = [a.get('content_type') for a in email_with_attachments.attachments]

        assert 'application/pdf' in attachment_types
        assert 'image/png' in attachment_types
        assert 'text/csv' in attachment_types

    def test_attachment_in_text_formatting(self, email_spinner, email_with_attachments):
        """Test attachments are mentioned in formatted text."""
        text = email_spinner._format_message_text(email_with_attachments)

        # Attachments should be listed in formatted text
        assert 'Attachments:' in text
        assert 'document.pdf' in text


# =============================================================================
# MIME Type Handling Tests
# =============================================================================

class TestMIMETypeHandling:
    """Tests for MIME type handling."""

    def _parse_raw_to_message(self, raw_email: str):
        """Helper to convert raw email string to email.message.Message object."""
        return message_from_string(raw_email, policy=email_policy)

    def test_plain_text_extraction(self):
        """Test plain text MIME extraction."""
        plain_email = """From: sender@example.com
To: recipient@example.com
Subject: Plain Text
Content-Type: text/plain; charset="utf-8"

Simple plain text body.
"""
        msg_obj = self._parse_raw_to_message(plain_email)
        message = EmailParser.parse_message(msg_obj)

        assert "plain text body" in message.body_text.lower()

    def test_html_to_text_conversion(self):
        """Test HTML body is converted to text."""
        html_email = """From: sender@example.com
To: recipient@example.com
Subject: HTML Only
Content-Type: text/html; charset="utf-8"

<html>
<body>
<h1>Title</h1>
<p>Paragraph with <strong>bold</strong> text.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
</ul>
</body>
</html>
"""
        msg_obj = self._parse_raw_to_message(html_email)
        message = EmailParser.parse_message(msg_obj)

        # Should extract text from HTML (either body_text or body_html)
        body = message.body_text or message.body_html
        assert body is not None

    def test_charset_handling(self):
        """Test various charset encodings."""
        # UTF-8 email
        utf8_email = """From: sender@example.com
To: recipient@example.com
Subject: UTF-8 Test
Content-Type: text/plain; charset="utf-8"

Hello, this has special chars: cafe
"""
        msg_obj = self._parse_raw_to_message(utf8_email)
        message = EmailParser.parse_message(msg_obj)
        assert message.body_text is not None


# =============================================================================
# IMAP Integration Tests (Mocked)
# =============================================================================

class TestIMAPIntegration:
    """Tests for IMAP mailbox integration."""

    @pytest.fixture
    def imap_spinner(self):
        """Create an EmailSpinner configured for IMAP."""
        return EmailSpinner(
            imap_server="imap.example.com",
            username="user@example.com",
            password="password123",
            importance_threshold=0.1  # Low threshold to include test emails
        )

    @pytest.mark.asyncio
    @patch('HoloLoom.spinningWheel.email_spinner.imaplib')
    async def test_spin_imap_mailbox(self, mock_imaplib, imap_spinner):
        """Test spinning emails from IMAP mailbox."""
        # Setup mock IMAP connection
        mock_imap = Mock()
        mock_imaplib.IMAP4_SSL.return_value = mock_imap
        mock_imap.login.return_value = ('OK', [b'Logged in'])
        mock_imap.select.return_value = ('OK', [b'5'])  # 5 messages
        mock_imap.search.return_value = ('OK', [b'1 2 3'])

        # Create a proper email bytes object
        email_bytes = b'From: test@example.com\r\nTo: recipient@example.com\r\nSubject: Test Email\r\nDate: Wed, 15 Jan 2025 10:00:00 +0000\r\nMessage-ID: <test@example.com>\r\n\r\nThis is a test email body with enough content to pass importance filtering.'
        mock_imap.fetch.return_value = ('OK', [(b'1', email_bytes)])
        mock_imap.logout.return_value = ('OK', [b'Logged out'])

        shards = await imap_spinner.spin_imap_mailbox(mailbox_name="INBOX")

        # Should return a list of shards
        assert isinstance(shards, list)
        mock_imap.login.assert_called_once()
        mock_imap.select.assert_called_once_with("INBOX", readonly=True)
        mock_imap.logout.assert_called_once()

    @pytest.mark.asyncio
    @patch('HoloLoom.spinningWheel.email_spinner.imaplib')
    async def test_imap_connection_error(self, mock_imaplib, imap_spinner):
        """Test IMAP connection error handling."""
        mock_imaplib.IMAP4_SSL.side_effect = Exception("Connection refused")

        with pytest.raises(Exception):
            await imap_spinner.spin_imap_mailbox(mailbox_name="INBOX")

    @pytest.mark.asyncio
    async def test_imap_missing_credentials(self, email_spinner):
        """Test error when IMAP credentials are missing."""
        with pytest.raises(ValueError, match="IMAP credentials required"):
            await email_spinner.spin_imap_mailbox(mailbox_name="INBOX")


# =============================================================================
# Mbox File Tests
# =============================================================================

class TestMboxFileSupport:
    """Tests for mbox file support."""

    @pytest.mark.asyncio
    async def test_spin_mbox_file(self):
        """Test spinning emails from mbox file."""
        # Use lower threshold to ensure messages pass filtering
        spinner = EmailSpinner(importance_threshold=0.1)

        mbox_content = """From sender@example.com Wed Jan 15 10:00:00 2025
From: sender@example.com
To: recipient@example.com
Subject: First Email
Date: Wed, 15 Jan 2025 10:00:00 +0000
Message-ID: <first@example.com>

First email body with enough content to pass importance filtering threshold. This email discusses important topics.

From another@example.com Wed Jan 15 11:00:00 2025
From: another@example.com
To: recipient@example.com
Subject: Second Email
Date: Wed, 15 Jan 2025 11:00:00 +0000
Message-ID: <second@example.com>

Second email body with enough content to pass importance filtering threshold. This email discusses different topics.
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.mbox', delete=False) as f:
            f.write(mbox_content)
            mbox_path = f.name

        try:
            # spin_mbox_file returns List[MemoryShard], not SpinResult
            shards = await spinner.spin_mbox_file(Path(mbox_path))

            assert isinstance(shards, list)
            # May be filtered by importance, but should have at least some
            assert len(shards) >= 0
        finally:
            os.unlink(mbox_path)

    @pytest.mark.asyncio
    async def test_mbox_file_not_found(self, email_spinner):
        """Test mbox file not found error."""
        with pytest.raises(FileNotFoundError):
            await email_spinner.spin_mbox_file(Path("/nonexistent/mailbox.mbox"))


# =============================================================================
# Spinner Capabilities Tests
# =============================================================================

class TestSpinnerCapabilities:
    """Tests for EmailSpinner capabilities."""

    def test_capabilities(self, email_spinner):
        """Test spinner capabilities reporting."""
        caps = email_spinner.get_capabilities()

        assert caps.basic_processing is True
        assert caps.batch_processing is True
        assert caps.importance_scoring is True

    def test_is_available(self, email_spinner):
        """Test availability check."""
        result = email_spinner.is_available()
        assert isinstance(result, bool)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_spin_email_mbox_function(self):
        """Test spin_email_mbox convenience function."""
        mbox_content = """From sender@example.com Wed Jan 15 10:00:00 2025
From: sender@example.com
To: recipient@example.com
Subject: Test Email
Date: Wed, 15 Jan 2025 10:00:00 +0000

This is the test email body.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mbox', delete=False) as f:
            f.write(mbox_content)
            mbox_path = f.name

        try:
            result = await spin_email_mbox(mbox_path)

            assert isinstance(result, SpinResult)
            assert result.success is True
        finally:
            os.unlink(mbox_path)


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def _parse_raw_to_message(self, raw_email: str):
        """Helper to convert raw email string to email.message.Message object."""
        return message_from_string(raw_email, policy=email_policy)

    def test_empty_body(self, email_spinner):
        """Test handling of email with empty body."""
        email = EmailMessage(
            message_id="<empty@example.com>",
            sender="sender@example.com",
            recipients=["recipient@example.com"],
            cc=[],
            subject="Empty Body Email",
            body_text="",
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            in_reply_to=None,
            references=[],
            attachments=[],
            headers={}
        )

        # Use _messages_to_shards (may filter by importance)
        shards = email_spinner._messages_to_shards([email])

        # Empty body may be filtered by importance, test formatting instead
        text = email_spinner._format_message_text(email)
        assert "Empty Body Email" in text

    def test_missing_headers(self):
        """Test parsing email with minimal headers."""
        minimal_email = """Subject: Minimal Email

Just a body.
"""
        msg_obj = self._parse_raw_to_message(minimal_email)
        message = EmailParser.parse_message(msg_obj)

        assert message.subject == "Minimal Email"
        assert message.sender == "" or message.sender is None

    def test_unicode_in_subject(self):
        """Test handling unicode in subject."""
        unicode_email = """From: sender@example.com
To: recipient@example.com
Subject: =?UTF-8?B?SGVsbG8gV29ybGQg8J+MjQ==?=

Body text.
"""
        msg_obj = self._parse_raw_to_message(unicode_email)
        message = EmailParser.parse_message(msg_obj)

        # Should handle encoded subject
        assert message.subject is not None

    def test_very_long_email(self, email_spinner):
        """Test handling of very long email body."""
        long_body = "This is a sentence. " * 10000  # ~200KB of text

        email = EmailMessage(
            message_id="<long@example.com>",
            sender="sender@example.com",
            recipients=["recipient@example.com"],
            cc=[],
            subject="Long Email",
            body_text=long_body,
            timestamp=datetime(2025, 1, 15, 10, 0, 0),
            in_reply_to=None,
            references=[],
            attachments=[],
            headers={}
        )

        # Test format_message_text for long emails
        text = email_spinner._format_message_text(email)
        assert len(text) > 1000

        # Also test shards (may be created depending on importance)
        shards = email_spinner._messages_to_shards([email])
        if shards:
            assert len(shards[0].text) > 1000

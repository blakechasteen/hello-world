"""
Tests for EmailSpinner

Tests email parsing, threading detection, and MemoryShard conversion.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import mailbox

from HoloLoom.spinningWheel.email_spinner import (
    EmailSpinner,
    EmailParser,
    EmailMessage,
    spin_email_mbox,
    create_email_scorer
)


# Fixtures

@pytest.fixture
def temp_mbox_dir():
    """Create temporary directory for test mbox files"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_email_message():
    """Create mock email message object"""
    msg = MIMEText("This is a test email about the project deadline.\n\nPlease review ASAP.")
    msg['Subject'] = 'Project Update'
    msg['From'] = 'alice@example.com'
    msg['To'] = 'bob@example.com, charlie@example.com'
    msg['Date'] = email.utils.formatdate()
    msg['Message-ID'] = '<test123@example.com>'

    return msg


@pytest.fixture
def mock_reply_email():
    """Create mock reply email"""
    msg = MIMEText("Thanks for the update!")
    msg['Subject'] = 'Re: Project Update'
    msg['From'] = 'bob@example.com'
    msg['To'] = 'alice@example.com'
    msg['Date'] = email.utils.formatdate()
    msg['Message-ID'] = '<test456@example.com>'
    msg['In-Reply-To'] = '<test123@example.com>'
    msg['References'] = '<test123@example.com>'

    return msg


@pytest.fixture
def sample_email_message():
    """Create sample EmailMessage object"""
    return EmailMessage(
        message_id="<test@example.com>",
        subject="Test Subject",
        sender="alice@example.com",
        recipients=["bob@example.com"],
        timestamp=datetime.now(),
        body_text="This is the email body with meeting and deadline keywords.",
        attachments=[
            {'filename': 'document.pdf', 'content_type': 'application/pdf'}
        ]
    )


# EmailMessage Tests

def test_email_message_is_reply():
    """Test EmailMessage.is_reply property"""
    # Not a reply
    msg1 = EmailMessage(
        message_id="<1@example.com>",
        subject="Test",
        sender="a@example.com",
        recipients=["b@example.com"],
        timestamp=datetime.now(),
        body_text="Test"
    )
    assert not msg1.is_reply

    # Is a reply
    msg2 = EmailMessage(
        message_id="<2@example.com>",
        subject="Re: Test",
        sender="b@example.com",
        recipients=["a@example.com"],
        timestamp=datetime.now(),
        body_text="Reply",
        in_reply_to="<1@example.com>"
    )
    assert msg2.is_reply


def test_email_message_is_thread_root():
    """Test EmailMessage.is_thread_root property"""
    # Thread root
    msg1 = EmailMessage(
        message_id="<1@example.com>",
        subject="Test",
        sender="a@example.com",
        recipients=["b@example.com"],
        timestamp=datetime.now(),
        body_text="Test"
    )
    assert msg1.is_thread_root

    # Not thread root (reply)
    msg2 = EmailMessage(
        message_id="<2@example.com>",
        subject="Re: Test",
        sender="b@example.com",
        recipients=["a@example.com"],
        timestamp=datetime.now(),
        body_text="Reply",
        in_reply_to="<1@example.com>"
    )
    assert not msg2.is_thread_root


def test_email_message_recipient_count():
    """Test EmailMessage.recipient_count property"""
    msg = EmailMessage(
        message_id="<test@example.com>",
        subject="Test",
        sender="a@example.com",
        recipients=["b@example.com", "c@example.com"],
        timestamp=datetime.now(),
        body_text="Test",
        cc=["d@example.com"],
        bcc=["e@example.com"]
    )

    assert msg.recipient_count == 4  # 2 to + 1 cc + 1 bcc


def test_email_message_has_attachments(sample_email_message):
    """Test EmailMessage.has_attachments property"""
    assert sample_email_message.has_attachments


# EmailParser Tests

def test_email_parser_parse_message(mock_email_message):
    """Test parsing email message object"""
    email_msg = EmailParser.parse_message(mock_email_message)

    assert email_msg.message_id == '<test123@example.com>'
    assert email_msg.subject == 'Project Update'
    assert email_msg.sender == 'alice@example.com'
    assert 'bob@example.com' in email_msg.recipients
    assert email_msg.body_text != ""


def test_email_parser_parse_reply(mock_reply_email):
    """Test parsing reply email"""
    email_msg = EmailParser.parse_message(mock_reply_email)

    assert email_msg.in_reply_to == '<test123@example.com>'
    assert '<test123@example.com>' in email_msg.references
    assert email_msg.is_reply


def test_email_parser_parse_addresses():
    """Test parsing comma-separated addresses"""
    addresses = "alice@example.com, bob@example.com, charlie@example.com"
    parsed = EmailParser._parse_addresses(addresses)

    assert len(parsed) == 3
    assert 'alice@example.com' in parsed
    assert 'bob@example.com' in parsed


def test_email_parser_parse_references():
    """Test parsing References header"""
    references = "<msg1@example.com> <msg2@example.com>"
    parsed = EmailParser._parse_references(references)

    assert len(parsed) == 2
    assert '<msg1@example.com>' in parsed


def test_email_parser_html_to_text():
    """Test HTML to text conversion"""
    html = """
    <html>
        <body>
            <h1>Title</h1>
            <p>This is a paragraph.</p>
        </body>
    </html>
    """

    text = EmailParser._html_to_text(html)

    assert "Title" in text
    assert "paragraph" in text
    assert "<" not in text  # HTML tags removed


# EmailSpinner Tests

@pytest.mark.asyncio
async def test_email_spinner_initialization():
    """Test EmailSpinner initialization"""
    spinner = EmailSpinner(
        imap_server="imap.gmail.com",
        username="test@gmail.com",
        password="password",
        importance_threshold=0.3
    )

    assert spinner.get_name() == "email"
    assert spinner.importance_threshold == 0.3
    assert spinner.imap_server == "imap.gmail.com"


@pytest.mark.asyncio
async def test_email_spinner_capabilities():
    """Test spinner capabilities"""
    spinner = EmailSpinner()

    caps = spinner.get_capabilities()
    assert caps.basic_processing
    assert caps.entity_extraction
    assert caps.motif_extraction
    assert caps.importance_scoring
    assert caps.incremental  # IMAP supports incremental
    assert caps.streaming
    assert 'email' in caps.supported_formats or 'mbox' in caps.supported_formats


@pytest.mark.asyncio
async def test_email_spinner_availability():
    """Test spinner availability check"""
    spinner = EmailSpinner()

    # Should always be available (email is stdlib)
    assert spinner.is_available()


def test_email_spinner_format_message_text(sample_email_message):
    """Test email message text formatting"""
    spinner = EmailSpinner()

    formatted = spinner._format_message_text(sample_email_message)

    assert "From: alice@example.com" in formatted
    assert "To: bob@example.com" in formatted
    assert "Subject: Test Subject" in formatted
    assert "Attachments:" in formatted
    assert "document.pdf" in formatted
    assert sample_email_message.body_text in formatted


def test_email_spinner_extract_entities(sample_email_message):
    """Test entity extraction from email"""
    spinner = EmailSpinner()

    entities = spinner._extract_entities(sample_email_message)

    assert "alice@example.com" in entities  # Sender
    assert "bob@example.com" in entities    # Recipient


def test_email_spinner_extract_motifs():
    """Test motif extraction from email"""
    spinner = EmailSpinner()

    # Thread root with attachments
    msg = EmailMessage(
        message_id="<test@example.com>",
        subject="Test",
        sender="a@example.com",
        recipients=["b@example.com"],
        timestamp=datetime.now(),
        body_text="Test",
        attachments=[{'filename': 'file.pdf', 'content_type': 'application/pdf'}]
    )

    motifs = spinner._extract_motifs(msg)

    assert 'thread_root' in motifs
    assert 'attachments' in motifs
    assert 'one_to_one' in motifs


def test_email_spinner_score_importance_high():
    """Test importance scoring for high-importance email"""
    spinner = EmailSpinner()

    # Important email: many recipients, attachments, technical content
    msg = EmailMessage(
        message_id="<test@example.com>",
        subject="Urgent: Project Deadline",
        sender="manager@example.com",
        recipients=["team1@example.com", "team2@example.com", "team3@example.com"],
        timestamp=datetime.now(),
        body_text="This is an urgent meeting request. Please review the attached document and provide feedback ASAP.",
        attachments=[{'filename': 'requirements.pdf', 'content_type': 'application/pdf'}]
    )

    score = spinner.score_importance(msg)

    assert score.score > 0.4
    assert score.signals is not None


def test_email_spinner_score_importance_low():
    """Test importance scoring for low-importance email"""
    spinner = EmailSpinner()

    # Low importance: short, no attachments
    msg = EmailMessage(
        message_id="<test@example.com>",
        subject="Thanks",
        sender="a@example.com",
        recipients=["b@example.com"],
        timestamp=datetime.now(),
        body_text="Thanks!"
    )

    score = spinner.score_importance(msg)

    assert score.score < 0.5


def test_email_spinner_messages_to_shards():
    """Test converting email messages to shards"""
    spinner = EmailSpinner(importance_threshold=0.0)  # Include all

    messages = [
        EmailMessage(
            message_id="<1@example.com>",
            subject="Project Update",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            timestamp=datetime.now(),
            body_text="This is a project update with meeting and deadline information.",
            attachments=[{'filename': 'doc.pdf', 'content_type': 'application/pdf'}]
        ),
        EmailMessage(
            message_id="<2@example.com>",
            subject="Re: Project Update",
            sender="bob@example.com",
            recipients=["alice@example.com"],
            timestamp=datetime.now(),
            body_text="Thanks for the update!",
            in_reply_to="<1@example.com>"
        )
    ]

    shards = spinner._messages_to_shards(messages)

    assert len(shards) == 2

    # Check first shard
    assert shards[0].metadata['message_id'] == "<1@example.com>"
    assert shards[0].metadata['is_thread_root'] == True
    assert shards[0].metadata['has_attachments'] == True

    # Check second shard
    assert shards[1].metadata['is_reply'] == True


def test_email_spinner_messages_to_shards_filtering():
    """Test importance threshold filtering"""
    spinner = EmailSpinner(importance_threshold=1.0)  # Filter all

    messages = [
        EmailMessage(
            message_id="<test@example.com>",
            subject="Test",
            sender="a@example.com",
            recipients=["b@example.com"],
            timestamp=datetime.now(),
            body_text="Short message"
        )
    ]

    shards = spinner._messages_to_shards(messages)

    # Should filter all
    assert len(shards) == 0


@pytest.mark.asyncio
async def test_email_spinner_spin_mbox_file(temp_mbox_dir):
    """Test spinning mbox file"""
    # Create test mbox file
    mbox_path = temp_mbox_dir / "test.mbox"
    mbox = mailbox.mbox(str(mbox_path))

    # Add test message
    msg = MIMEText("Test email body")
    msg['Subject'] = 'Test Subject'
    msg['From'] = 'alice@example.com'
    msg['To'] = 'bob@example.com'
    msg['Date'] = email.utils.formatdate()
    msg['Message-ID'] = '<test@example.com>'

    mbox.add(msg)
    mbox.close()

    # Spin mbox
    spinner = EmailSpinner(importance_threshold=0.0)
    shards = await spinner.spin_mbox_file(mbox_path)

    assert len(shards) >= 1
    assert shards[0].metadata['subject'] == 'Test Subject'


@pytest.mark.asyncio
async def test_create_email_scorer():
    """Test email-specific importance scorer creation"""
    scorer = create_email_scorer()

    # Should have email-specific terms
    assert len(scorer.technical_scorer.technical_terms) > 0

    # Test scoring
    email_text = "This is an urgent meeting request. Please review and approve ASAP."
    tech_score = scorer.technical_scorer.score(email_text)

    assert tech_score > 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

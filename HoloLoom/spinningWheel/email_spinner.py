"""
EmailSpinner - Email Inbox Ingestion
====================================

Extracts emails from IMAP mailboxes into MemoryShards.

Features:
- IMAP mailbox support (Gmail, Outlook, custom servers)
- Email header extraction (From, To, Subject, Date)
- HTML/text content parsing with fallback
- Attachment detection and listing
- Sender/recipient extraction
- Time-based filtering (last N days)
- Conversation thread detection
- Importance scoring based on sender and engagement

Author: Claude Code
Date: November 2025
"""

import asyncio
import time
import email
import imaplib
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from email.header import decode_header
from email.utils import parsedate_to_datetime
import os

try:
    from email_validator import validate_email, EmailNotValidError
    EMAIL_VALIDATOR_AVAILABLE = True
except ImportError:
    EMAIL_VALIDATOR_AVAILABLE = False

try:
    import html2text
    HTML2TEXT_AVAILABLE = True
except ImportError:
    HTML2TEXT_AVAILABLE = False

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinnerCapabilities,
    SpinResult,
    SpinnerCheckpoint,
    ImportanceSignals,
    ImportanceScore,
    SpinnerStatus
)
from HoloLoom.spinningWheel.utils import create_source_id


# ============================================================================
# Email Data Structures
# ============================================================================

@dataclass
class EmailMessage:
    """Parsed email message."""
    message_id: str  # Unique message ID
    subject: str
    sender: str  # From address
    sender_name: Optional[str]  # From display name
    recipients: List[str] = field(default_factory=list)  # To addresses
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    date: Optional[str] = None  # ISO format
    body: str = ""
    body_html: Optional[str] = None
    attachments: List[str] = field(default_factory=list)  # Attachment filenames
    in_reply_to: Optional[str] = None  # Reply to message ID
    references: List[str] = field(default_factory=list)  # Thread references
    is_read: bool = True
    flag_importance: str = "normal"  # 'low', 'normal', 'high'


# ============================================================================
# IMAP Client
# ============================================================================

class IMAPClient:
    """IMAP client for email retrieval."""

    def __init__(
        self,
        imap_server: str,
        email_address: str,
        password: str,
        use_ssl: bool = True
    ):
        """
        Initialize IMAP client.

        Args:
            imap_server: IMAP server address (e.g., "imap.gmail.com")
            email_address: Email address
            password: Password or app password
            use_ssl: Use SSL/TLS (usually True)
        """
        self.imap_server = imap_server
        self.email_address = email_address
        self.password = password
        self.use_ssl = use_ssl
        self.connection: Optional[imaplib.IMAP4_SSL] = None

    def connect(self):
        """Establish IMAP connection."""
        try:
            if self.use_ssl:
                self.connection = imaplib.IMAP4_SSL(self.imap_server, 993)
            else:
                self.connection = imaplib.IMAP4(self.imap_server, 143)

            self.connection.login(self.email_address, self.password)
            return True

        except (imaplib.IMAP4.error, imaplib.IMAP4.abort) as e:
            print(f"IMAP connection error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error connecting to IMAP: {e}")
            return False

    def disconnect(self):
        """Close IMAP connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection.logout()
        except Exception:
            pass

    async def fetch_messages(
        self,
        mailbox: str = "INBOX",
        days_back: int = 30,
        max_messages: int = 100
    ) -> List[EmailMessage]:
        """
        Fetch messages from mailbox.

        Args:
            mailbox: Mailbox name (e.g., "INBOX", "Sent", "[Gmail]/Drafts")
            days_back: Fetch messages from last N days
            max_messages: Maximum messages to fetch

        Returns:
            List of EmailMessage objects
        """
        if not self.connection:
            if not self.connect():
                return []

        messages = []

        try:
            # Select mailbox
            _, _ = self.connection.select(mailbox)

            # Calculate search criteria
            # IMAP uses dates like "1-Jan-2024"
            days_ago_date = (datetime.now() - timedelta(days=days_back)).strftime("%d-%b-%Y")

            # Search for messages since date
            _, message_ids = self.connection.search(None, f"SINCE {days_ago_date}")

            # Get message IDs (newest first)
            msg_ids = message_ids[0].split()[::-1][:max_messages]

            for msg_id in msg_ids:
                try:
                    _, msg_data = self.connection.fetch(msg_id, "(RFC822)")
                    msg_body = msg_data[0][1]
                    email_message = email.message_from_bytes(msg_body)

                    parsed = self._parse_email(email_message, msg_id.decode())
                    messages.append(parsed)

                    # Rate limiting
                    await asyncio.sleep(0.1)

                except Exception as e:
                    print(f"Error parsing message {msg_id}: {e}")
                    continue

        except (imaplib.IMAP4.error, imaplib.IMAP4.abort) as e:
            print(f"IMAP error during fetch: {e}")
        except Exception as e:
            print(f"Unexpected error fetching messages: {e}")

        return messages

    def _parse_email(self, email_message, msg_id: str) -> EmailMessage:
        """Parse email.message.Message to EmailMessage."""
        # Header parsing
        subject = self._decode_header(email_message.get("Subject", ""))
        sender = email_message.get("From", "")
        sender_name = None

        # Extract sender display name if present
        if " <" in sender:
            sender_name, sender = sender.rsplit(" <", 1)
            sender_name = sender_name.strip('"')
            sender = sender.rstrip(">")
        elif "<" in sender:
            sender_name, sender = sender.rsplit("<", 1)
            sender = sender.rstrip(">")

        # Parse recipients
        recipients = self._parse_email_list(email_message.get("To", ""))
        cc = self._parse_email_list(email_message.get("Cc", ""))
        bcc = self._parse_email_list(email_message.get("Bcc", ""))

        # Date
        date_str = email_message.get("Date", "")
        try:
            date_obj = parsedate_to_datetime(date_str)
            date = date_obj.isoformat()
        except Exception:
            date = date_str

        # Thread references
        in_reply_to = email_message.get("In-Reply-To")
        references = [r.strip() for r in email_message.get("References", "").split()]

        # Body
        body = ""
        body_html = None

        if email_message.is_multipart():
            for part in email_message.walk():
                content_type = part.get_content_type()

                if content_type == "text/plain":
                    try:
                        body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                    except Exception:
                        body = part.get_payload()

                elif content_type == "text/html":
                    try:
                        body_html = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        # Try to convert HTML to text if html2text available
                        if HTML2TEXT_AVAILABLE and not body:
                            h2t = html2text.HTML2Text()
                            h2t.ignore_links = False
                            body = h2t.handle(body_html)
                    except Exception:
                        body_html = None

        else:
            content_type = email_message.get_content_type()
            if content_type == "text/plain":
                try:
                    body = email_message.get_payload(decode=True).decode('utf-8', errors='replace')
                except Exception:
                    body = email_message.get_payload()

        # Attachments
        attachments = []
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_disposition() == "attachment":
                    filename = part.get_filename()
                    if filename:
                        attachments.append(filename)

        # Flags
        is_read = True  # Default to True (harder to determine without IMAP flags)

        return EmailMessage(
            message_id=msg_id,
            subject=subject,
            sender=sender,
            sender_name=sender_name,
            recipients=recipients,
            cc=cc,
            bcc=bcc,
            date=date,
            body=body,
            body_html=body_html,
            attachments=attachments,
            in_reply_to=in_reply_to,
            references=references,
            is_read=is_read
        )

    @staticmethod
    def _decode_header(header_value: str) -> str:
        """Decode email header with charset handling."""
        try:
            decoded_parts = decode_header(header_value)
            result = ""
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    result += part.decode(encoding or 'utf-8', errors='replace')
                else:
                    result += str(part)
            return result
        except Exception:
            return header_value

    @staticmethod
    def _parse_email_list(email_str: str) -> List[str]:
        """Parse comma-separated email list."""
        if not email_str:
            return []

        emails = []
        for item in email_str.split(","):
            item = item.strip()
            # Extract email from "Name <email>" format
            if "<" in item and ">" in item:
                email_addr = item[item.index("<") + 1 : item.index(">")]
                emails.append(email_addr)
            else:
                emails.append(item)

        return [e for e in emails if e]


# ============================================================================
# Email Spinner
# ============================================================================

class EmailSpinner(BaseSpinner):
    """
    Spin email messages into MemoryShards.

    Features:
    - IMAP mailbox support
    - HTML/text body extraction
    - Thread detection
    - Importance scoring
    - Graceful error handling

    Usage:
        >>> spinner = EmailSpinner(
        ...     imap_server="imap.gmail.com",
        ...     email_address="user@gmail.com",
        ...     password="app_password"
        ... )
        >>> result = await spinner.spin()
        >>> print(f"Created {result.shard_count} shards")
    """

    def __init__(
        self,
        imap_server: str,
        email_address: str,
        password: str,
        importance_threshold: float = 0.2,
        checkpoint_dir: Optional[Path] = None,
        use_ssl: bool = True,
        days_back: int = 30,
        max_messages: int = 100
    ):
        """
        Initialize EmailSpinner.

        Args:
            imap_server: IMAP server (e.g., "imap.gmail.com")
            email_address: Email address
            password: Email password or app password
            importance_threshold: Minimum importance (0.0-1.0)
            checkpoint_dir: Directory for checkpoints
            use_ssl: Use SSL/TLS
            days_back: Fetch emails from last N days
            max_messages: Maximum emails to fetch
        """
        super().__init__(
            name="email",
            importance_threshold=importance_threshold,
            checkpoint_dir=checkpoint_dir
        )

        self.client = IMAPClient(
            imap_server=imap_server,
            email_address=email_address,
            password=password,
            use_ssl=use_ssl
        )
        self.days_back = days_back
        self.max_messages = max_messages

    # ========================================================================
    # Protocol Methods
    # ========================================================================

    def get_capabilities(self) -> SpinnerCapabilities:
        """Get capabilities."""
        return SpinnerCapabilities(
            basic_processing=True,
            streaming=False,
            incremental=False,
            batch_processing=True,
            importance_scoring=True,
            entity_extraction=True,
            motif_extraction=True,
            embeddings=False,
            max_input_size=None,
            supported_formats=['imap']
        )

    def is_available(self) -> bool:
        """Check if IMAP available (always True for imaplib)."""
        return True

    # ========================================================================
    # Core Spinning
    # ========================================================================

    async def _spin_impl(
        self,
        source: Optional[str] = None,
        mailbox: str = "INBOX",
        **kwargs
    ) -> List[MemoryShard]:
        """
        Process email mailbox.

        Args:
            source: Unused (IMAP configured in __init__)
            mailbox: Mailbox name (e.g., "INBOX", "Sent", "[Gmail]/All Mail")
            **kwargs: Additional options

        Returns:
            List of MemoryShards
        """
        print(f"Fetching emails from {mailbox}...")

        # Fetch messages
        messages = await self.client.fetch_messages(
            mailbox=mailbox,
            days_back=self.days_back,
            max_messages=self.max_messages
        )

        # Convert to shards
        shards = []
        for msg in messages:
            shard = self._message_to_shard(msg, mailbox)
            shards.append(shard)

        return shards

    # ========================================================================
    # Conversion Methods
    # ========================================================================

    def _message_to_shard(
        self,
        msg: EmailMessage,
        mailbox: str
    ) -> MemoryShard:
        """Convert email to MemoryShard."""
        # Build text
        text_parts = [
            f"Email: {msg.subject}",
            f"From: {msg.sender_name or msg.sender}",
            f"Date: {msg.date}",
        ]

        if msg.recipients:
            text_parts.append(f"To: {', '.join(msg.recipients)}")

        if msg.cc:
            text_parts.append(f"Cc: {', '.join(msg.cc)}")

        if msg.attachments:
            text_parts.append(f"Attachments: {', '.join(msg.attachments)}")

        text_parts.extend(["", msg.body])

        text = "\n".join(text_parts)

        # Extract entities
        entities = [
            msg.sender,
            msg.sender_name or msg.sender,
            *msg.recipients,
            *msg.cc,
            mailbox
        ]

        # Extract motifs
        motifs = ["email", mailbox.lower()]

        if msg.in_reply_to:
            motifs.append("reply")

        if msg.attachments:
            motifs.append("has_attachments")

        # Score importance
        importance = self._score_message_importance(msg)

        # Create shard
        return self._create_shard(
            id_suffix=f"email_{msg.message_id.replace('/', '_').replace('<', '').replace('>', '')}",
            text=text,
            episode=f"email_{mailbox}",
            entities=entities,
            motifs=motifs,
            metadata={
                'source': 'email',
                'mailbox': mailbox,
                'subject': msg.subject,
                'sender': msg.sender,
                'sender_name': msg.sender_name,
                'recipients': msg.recipients,
                'cc': msg.cc,
                'bcc': msg.bcc,
                'date': msg.date,
                'message_id': msg.message_id,
                'in_reply_to': msg.in_reply_to,
                'references': msg.references,
                'attachments': msg.attachments,
                'attachment_count': len(msg.attachments),
                'body_length': len(msg.body),
                'importance_score': importance.score,
                'importance_reason': importance.reason,
                'confidence': 1.0
            }
        )

    # ========================================================================
    # Importance Scoring
    # ========================================================================

    def _score_message_importance(self, msg: EmailMessage) -> ImportanceScore:
        """Score email importance."""
        signals = ImportanceSignals()

        # Length signal
        signals.length_score = min(1.0, len(msg.body) / 1000)

        # Authority signal: direct messages more important than mass emails
        recipient_count = len(msg.recipients)
        if recipient_count == 1:
            signals.authority_score = 0.8
        elif recipient_count <= 5:
            signals.authority_score = 0.6
        else:
            signals.authority_score = 0.3

        # Recency signal
        try:
            date_obj = datetime.fromisoformat(msg.date)
            now = datetime.now(date_obj.tzinfo)
            days_old = (now - date_obj).days
            signals.recency_score = max(0.1, 1.0 - (days_old / 30))
        except Exception:
            signals.recency_score = 0.5

        # Custom signals
        if msg.in_reply_to:
            signals.custom_signals['reply'] = 0.7

        if msg.attachments:
            signals.custom_signals['attachments'] = 0.6

        # Technical content detection
        tech_keywords = ['urgent', 'important', 'critical', 'action required', 'deadline']
        text_lower = (msg.subject + " " + msg.body).lower()
        if any(kw in text_lower for kw in tech_keywords):
            signals.technical_score = 0.8

        total = signals.compute_total()

        return ImportanceScore(
            score=total,
            signals=signals,
            reason=signals.explain()
        )

    # ========================================================================
    # Cleanup
    # ========================================================================

    def __del__(self):
        """Cleanup IMAP connection."""
        self.client.disconnect()


# ============================================================================
# Convenience Functions
# ============================================================================

async def spin_email_inbox(
    imap_server: str,
    email_address: str,
    password: str,
    mailbox: str = "INBOX",
    days_back: int = 30,
    max_messages: int = 100
) -> SpinResult:
    """
    Quick function to spin email mailbox.

    Args:
        imap_server: IMAP server (e.g., "imap.gmail.com")
        email_address: Email address
        password: Email password or app password
        mailbox: Mailbox name (e.g., "INBOX")
        days_back: Fetch emails from last N days
        max_messages: Maximum emails to fetch

    Returns:
        SpinResult with shards

    Example:
        >>> result = await spin_email_inbox(
        ...     "imap.gmail.com",
        ...     "user@gmail.com",
        ...     "app_password"
        ... )
        >>> print(f"Created {result.shard_count} shards")
    """
    spinner = EmailSpinner(
        imap_server=imap_server,
        email_address=email_address,
        password=password,
        days_back=days_back,
        max_messages=max_messages
    )

    return await spinner.spin(mailbox=mailbox)

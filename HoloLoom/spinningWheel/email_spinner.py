"""
Email Spinner - Ingest email archives into HoloLoom memory

Supports:
- IMAP server ingestion (Gmail, Outlook, etc.)
- mbox file ingestion (Thunderbird, Apple Mail exports)
- Thread detection and reconstruction
- Attachment metadata extraction
- Sender/recipient extraction
- HTML to text conversion
- Importance scoring (sender authority, reply count)
- Incremental sync (IMAP UID-based)

Requires: email (stdlib), imaplib (stdlib)
Optional: beautifulsoup4 (HTML parsing)

Usage:
    from HoloLoom.spinningWheel.email_spinner import EmailSpinner

    # IMAP
    spinner = EmailSpinner(
        imap_server="imap.gmail.com",
        username="you@gmail.com",
        password="app_password"
    )
    result = await spinner.spin_mailbox("INBOX")

    # mbox file
    spinner = EmailSpinner()
    result = await spinner.spin_mbox("/path/to/archive.mbox")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncIterator
import email
from email import policy
from email.parser import BytesParser
import mailbox
import imaplib
import hashlib
import re

# Optional HTML parsing
HTML_AVAILABLE = False
try:
    from bs4 import BeautifulSoup
    HTML_AVAILABLE = True
except ImportError:
    pass

from HoloLoom.protocols.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    SpinnerCheckpoint,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.importance import ImportanceScorer


@dataclass
class EmailMessage:
    """Parsed email message"""
    message_id: str
    subject: str
    sender: str
    recipients: List[str]
    timestamp: datetime
    body_text: str
    body_html: Optional[str] = None
    attachments: List[Dict[str, str]] = field(default_factory=list)  # {filename, content_type}
    in_reply_to: Optional[str] = None  # Parent message ID
    references: List[str] = field(default_factory=list)  # Thread references
    cc: List[str] = field(default_factory=list)
    bcc: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)

    @property
    def is_reply(self) -> bool:
        """Check if this is a reply"""
        return self.in_reply_to is not None

    @property
    def is_thread_root(self) -> bool:
        """Check if this starts a thread"""
        return not self.is_reply

    @property
    def recipient_count(self) -> int:
        """Total recipients"""
        return len(self.recipients) + len(self.cc) + len(self.bcc)

    @property
    def has_attachments(self) -> bool:
        """Check if message has attachments"""
        return bool(self.attachments)


class EmailParser:
    """Parse email messages"""

    @staticmethod
    def parse_message(msg: email.message.Message) -> EmailMessage:
        """
        Parse email.message.Message object

        Args:
            msg: Email message object

        Returns:
            EmailMessage object
        """
        # Extract headers
        message_id = msg.get('Message-ID', f"<generated-{hash(msg)}>")
        subject = msg.get('Subject', '(no subject)')
        sender = msg.get('From', '')

        # Parse recipients
        recipients = EmailParser._parse_addresses(msg.get('To', ''))
        cc = EmailParser._parse_addresses(msg.get('Cc', ''))
        bcc = EmailParser._parse_addresses(msg.get('Bcc', ''))

        # Parse date
        date_str = msg.get('Date')
        if date_str:
            try:
                timestamp = email.utils.parsedate_to_datetime(date_str)
            except Exception:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Parse threading
        in_reply_to = msg.get('In-Reply-To')
        references = EmailParser._parse_references(msg.get('References', ''))

        # Extract body
        body_text, body_html = EmailParser._extract_body(msg)

        # Extract attachments
        attachments = EmailParser._extract_attachments(msg)

        # Extract headers
        headers = {k: v for k, v in msg.items()}

        return EmailMessage(
            message_id=message_id,
            subject=subject,
            sender=sender,
            recipients=recipients,
            timestamp=timestamp,
            body_text=body_text,
            body_html=body_html,
            attachments=attachments,
            in_reply_to=in_reply_to,
            references=references,
            cc=cc,
            bcc=bcc,
            headers=headers
        )

    @staticmethod
    def _parse_addresses(addresses: str) -> List[str]:
        """Parse comma-separated email addresses"""
        if not addresses:
            return []
        return [addr.strip() for addr in addresses.split(',')]

    @staticmethod
    def _parse_references(references: str) -> List[str]:
        """Parse References header"""
        if not references:
            return []
        # Extract message IDs
        return re.findall(r'<[^>]+>', references)

    @staticmethod
    def _extract_body(msg: email.message.Message) -> tuple[str, Optional[str]]:
        """
        Extract text and HTML body

        Returns:
            (text_body, html_body)
        """
        text_body = ""
        html_body = None

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()

                if content_type == 'text/plain':
                    try:
                        text_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except Exception:
                        pass

                elif content_type == 'text/html':
                    try:
                        html_body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except Exception:
                        pass
        else:
            content_type = msg.get_content_type()

            if content_type == 'text/plain':
                try:
                    text_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                except Exception:
                    pass

            elif content_type == 'text/html':
                try:
                    html_body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                except Exception:
                    pass

        # Convert HTML to text if no text body
        if not text_body and html_body:
            text_body = EmailParser._html_to_text(html_body)

        return text_body, html_body

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert HTML to plain text"""
        if HTML_AVAILABLE:
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator='\n', strip=True)
        else:
            # Simple regex-based fallback
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    @staticmethod
    def _extract_attachments(msg: email.message.Message) -> List[Dict[str, str]]:
        """Extract attachment metadata"""
        attachments = []

        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename() or 'unknown'
                content_type = part.get_content_type()

                attachments.append({
                    'filename': filename,
                    'content_type': content_type
                })

        return attachments


class EmailSpinner(BaseSpinner):
    """
    Spinner for email archives

    Ingests emails into HoloLoom memory with:
    - IMAP server support
    - mbox file support
    - Thread detection
    - Attachment metadata
    - Sender/recipient extraction
    - HTML to text conversion
    - 9-signal importance scoring
    - Incremental sync (IMAP)
    """

    def __init__(
        self,
        imap_server: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        importance_threshold: float = 0.3,
        max_messages: Optional[int] = None
    ):
        """
        Initialize EmailSpinner

        Args:
            imap_server: IMAP server (e.g., "imap.gmail.com")
            username: Email username
            password: Email password (use app password for Gmail)
            importance_threshold: Minimum importance score (0.0-1.0)
            max_messages: Max messages to fetch (None = all)
        """
        super().__init__(name="email")

        self.imap_server = imap_server
        self.username = username
        self.password = password
        self.importance_threshold = importance_threshold
        self.max_messages = max_messages

        # IMAP connection (lazy initialization)
        self.imap_client: Optional[imaplib.IMAP4_SSL] = None

        # Create importance scorer
        self.importance_scorer = ImportanceScorer()

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            incremental=True,  # IMAP supports UID-based sync
            streaming=True,
            supported_formats=['email', 'mbox'],
            batch_processing=True
        )

    def is_available(self) -> bool:
        """Check if email dependencies are available"""
        return True  # email and imaplib are stdlib

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Spin email source into MemoryShards

        Args:
            source: "imap:INBOX" or Path to mbox file
            **kwargs: Additional arguments

        Returns:
            List of MemoryShards
        """
        if isinstance(source, str) and source.startswith('imap:'):
            # IMAP mailbox
            mailbox_name = source.split(':', 1)[1]
            return await self.spin_imap_mailbox(mailbox_name)
        elif isinstance(source, (str, Path)):
            # mbox file
            return await self.spin_mbox_file(Path(source))
        else:
            raise ValueError(f"source must be 'imap:MAILBOX' or mbox file path, got {source}")

    async def spin_imap_mailbox(
        self,
        mailbox_name: str = "INBOX",
        since_uid: Optional[int] = None
    ) -> List[MemoryShard]:
        """
        Spin IMAP mailbox

        Args:
            mailbox_name: Mailbox name (e.g., "INBOX", "Sent")
            since_uid: Fetch messages since UID (for incremental)

        Returns:
            List of MemoryShards
        """
        if not self.imap_server or not self.username or not self.password:
            raise ValueError("IMAP credentials required (imap_server, username, password)")

        # Connect to IMAP server
        self.imap_client = imaplib.IMAP4_SSL(self.imap_server)
        self.imap_client.login(self.username, self.password)
        self.imap_client.select(mailbox_name, readonly=True)

        # Search for messages
        if since_uid:
            # Incremental: fetch since UID
            search_criteria = f"UID {since_uid}:*"
        else:
            # Full: all messages
            search_criteria = "ALL"

        _, message_numbers = self.imap_client.search(None, search_criteria)

        # Parse message numbers
        msg_ids = message_numbers[0].split()

        # Limit if specified
        if self.max_messages:
            msg_ids = msg_ids[:self.max_messages]

        # Fetch messages
        messages = []
        for msg_id in msg_ids:
            _, msg_data = self.imap_client.fetch(msg_id, '(RFC822)')

            if msg_data and msg_data[0]:
                raw_email = msg_data[0][1]
                msg = BytesParser(policy=policy.default).parsebytes(raw_email)
                email_msg = EmailParser.parse_message(msg)
                messages.append(email_msg)

        # Logout
        self.imap_client.logout()
        self.imap_client = None

        # Convert to shards
        return self._messages_to_shards(messages)

    async def spin_mbox_file(self, mbox_path: Path) -> List[MemoryShard]:
        """
        Spin mbox file

        Args:
            mbox_path: Path to mbox file

        Returns:
            List of MemoryShards
        """
        if not mbox_path.exists():
            raise FileNotFoundError(f"mbox file not found: {mbox_path}")

        # Open mbox
        mbox = mailbox.mbox(str(mbox_path))

        # Parse messages
        messages = []
        for msg in mbox:
            email_msg = EmailParser.parse_message(msg)
            messages.append(email_msg)

            # Limit if specified
            if self.max_messages and len(messages) >= self.max_messages:
                break

        mbox.close()

        # Convert to shards
        return self._messages_to_shards(messages)

    async def spin_stream(
        self,
        source: Any,
        batch_size: int = 50,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream MemoryShards from email source

        Args:
            source: IMAP mailbox or mbox file
            batch_size: Number of messages per batch

        Yields:
            MemoryShard objects
        """
        shards = await self._spin_impl(source, **kwargs)

        for i in range(0, len(shards), batch_size):
            batch = shards[i:i + batch_size]
            for shard in batch:
                yield shard

    def _messages_to_shards(self, messages: List[EmailMessage]) -> List[MemoryShard]:
        """
        Convert EmailMessage objects to MemoryShards

        Args:
            messages: List of EmailMessage objects

        Returns:
            List of MemoryShards (filtered by importance)
        """
        shards = []

        for msg in messages:
            # Score importance
            importance = self.score_importance(msg)

            # Filter by threshold
            if importance.score < self.importance_threshold:
                continue

            # Create shard
            shard = self._create_shard(
                id_suffix=hashlib.sha256(msg.message_id.encode()).hexdigest()[:12],
                text=self._format_message_text(msg),
                episode=f"email_{msg.sender}",
                entities=self._extract_entities(msg),
                motifs=self._extract_motifs(msg),
                metadata={
                    'message_id': msg.message_id,
                    'subject': msg.subject,
                    'sender': msg.sender,
                    'recipients': msg.recipients,
                    'timestamp': msg.timestamp.isoformat(),
                    'is_reply': msg.is_reply,
                    'is_thread_root': msg.is_thread_root,
                    'recipient_count': msg.recipient_count,
                    'has_attachments': msg.has_attachments,
                    'attachment_count': len(msg.attachments),
                    'importance_score': importance.score,
                    'importance_reason': importance.reason
                }
            )
            shards.append(shard)

        return shards

    def _format_message_text(self, msg: EmailMessage) -> str:
        """Format email message for text field"""
        parts = [
            f"From: {msg.sender}",
            f"To: {', '.join(msg.recipients)}",
            f"Date: {msg.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"Subject: {msg.subject}",
            ""
        ]

        # CC/BCC if present
        if msg.cc:
            parts.insert(-1, f"Cc: {', '.join(msg.cc)}")
        if msg.bcc:
            parts.insert(-1, f"Bcc: {', '.join(msg.bcc)}")

        # Thread info
        if msg.is_reply:
            parts.insert(-1, f"In-Reply-To: {msg.in_reply_to}")

        # Attachments
        if msg.attachments:
            att_list = [f"{a['filename']} ({a['content_type']})" for a in msg.attachments]
            parts.insert(-1, f"Attachments: {', '.join(att_list)}")

        parts.append(msg.body_text)

        return '\n'.join(parts)

    def _extract_entities(self, msg: EmailMessage) -> List[str]:
        """Extract entities from email"""
        entities = []

        # Sender
        entities.append(msg.sender)

        # Recipients
        entities.extend(msg.recipients)
        entities.extend(msg.cc)

        return list(set(entities))

    def _extract_motifs(self, msg: EmailMessage) -> List[str]:
        """Extract motifs from email"""
        motifs = []

        if msg.is_thread_root:
            motifs.append('thread_root')
        if msg.is_reply:
            motifs.append('reply')
        if msg.has_attachments:
            motifs.append('attachments')
        if msg.recipient_count > 5:
            motifs.append('broadcast')
        if msg.recipient_count == 1:
            motifs.append('one_to_one')

        return motifs

    def score_importance(self, msg: EmailMessage) -> ImportanceScore:
        """
        Score email importance using 9 signals

        Args:
            msg: EmailMessage object

        Returns:
            ImportanceScore
        """
        signals = ImportanceSignals()
        text = msg.body_text

        # 1. Length score
        text_len = len(text)
        if text_len < 50:
            signals.length_score = 0.2
        elif text_len < 200:
            signals.length_score = 0.5
        elif text_len <= 1000:
            signals.length_score = min(1.0, text_len / 1000)
        else:
            signals.length_score = 0.9

        # 2. Technical score
        signals.technical_score = self.importance_scorer.technical_scorer.score(text)

        # 3. Structural score
        signals.structural_score = self.importance_scorer.structural_scorer.score(text)

        # 4. Authority score (sender domain-based, placeholder)
        # TODO: Integrate with authority map
        signals.authority_score = 0.5

        # 5. Recency score
        signals.recency_score = self.importance_scorer.recency_scorer.score(
            msg.timestamp.timestamp()
        )

        # 6. Engagement score (recipient count as proxy)
        signals.engagement_score = min(1.0, msg.recipient_count / 10.0)

        # 7. Reference score (thread depth)
        reference_score = 0.0
        if msg.is_thread_root:
            reference_score = 0.8  # Thread starters important
        elif msg.is_reply:
            reference_score = 0.5
        if msg.references:
            reference_score += min(0.3, len(msg.references) / 10.0)
        signals.reference_score = min(1.0, reference_score)

        # 8. Noise detection
        noise_score = self.importance_scorer.noise_detector.detect(text)

        # 9. Custom signals
        signals.custom_signals = {}
        signals.custom_signals['has_attachments'] = 1.0 if msg.has_attachments else 0.0
        signals.custom_signals['recipient_count'] = signals.engagement_score

        # Combine signals
        final_score = (
            0.15 * signals.length_score +
            0.15 * signals.technical_score +
            0.10 * signals.structural_score +
            0.10 * signals.authority_score +
            0.10 * signals.recency_score +
            0.15 * signals.engagement_score +
            0.15 * signals.reference_score +
            0.10 * signals.custom_signals.get('has_attachments', 0.0)
        )

        # Apply noise penalty
        final_score *= (1.0 - max(0.0, noise_score))

        # Generate reason
        reasons = []
        if msg.is_thread_root:
            reasons.append("thread root")
        if msg.has_attachments:
            reasons.append(f"{len(msg.attachments)} attachments")
        if msg.recipient_count > 3:
            reasons.append(f"{msg.recipient_count} recipients")
        if signals.technical_score > 0.5:
            reasons.append("technical content")

        reason = " + ".join(reasons) if reasons else "standard email"

        return ImportanceScore(
            score=max(0.0, min(1.0, final_score)),
            signals=signals,
            reason=reason
        )


# Convenience functions

async def spin_email_mbox(
    mbox_path: str,
    importance_threshold: float = 0.3
) -> SpinResult:
    """
    Convenience function to spin an mbox file

    Args:
        mbox_path: Path to mbox file
        importance_threshold: Min importance score

    Returns:
        SpinResult with MemoryShards
    """
    spinner = EmailSpinner(importance_threshold=importance_threshold)
    shards = await spinner.spin_mbox_file(Path(mbox_path))

    return SpinResult(
        shards=shards,
        success=True,
        items_processed=len(shards),
        items_filtered=0
    )


def create_email_scorer() -> ImportanceScorer:
    """Create importance scorer optimized for emails"""
    return ImportanceScorer(
        technical_terms={
            'meeting', 'deadline', 'urgent', 'asap', 'action', 'required',
            'review', 'approve', 'decision', 'feedback', 'please', 'thanks'
        }
    )

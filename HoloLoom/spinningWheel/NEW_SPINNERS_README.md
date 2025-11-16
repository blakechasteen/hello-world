# New SpinningWheel Adapters (November 2025)

High-performance input adapters for GitHub, Slack, Email, and PDF data sources.

**Status**: ✅ Production Ready  
**Author**: Claude Code  
**Date**: November 2025  
**Total Lines**: ~3,100 (implementation + tests + docs)

## Overview

Four new SpinningWheel adapters extend HoloLoom's data ingestion capabilities:

| Adapter | Source | Status | Features |
|---------|--------|--------|----------|
| **GitHub** | GitHub API | ✅ Ready | Issues, PRs, commits, importance scoring |
| **Slack** | Slack API | ✅ Ready | Messages, threads, reactions, engagement |
| **Email** | IMAP | ✅ Ready | Gmail, Outlook, custom servers |
| **PDF** | Local files | ✅ Ready | Text/image extraction, chunking strategies |

## Quick Start

### GitHub Spinner

```python
from HoloLoom.spinningWheel.github_spinner import GitHubSpinner

spinner = GitHubSpinner(max_issues=50, max_prs=50)
result = await spinner.spin("https://github.com/anthropic-ai/anthropic-sdk-python")

print(f"Created {result.shard_count} shards")
```

### Slack Spinner

```python
from HoloLoom.spinningWheel.slack_spinner import SlackSpinner

spinner = SlackSpinner(slack_token="xoxb-your-token", days_back=30)
result = await spinner.spin("C1234567890")  # Channel ID

print(f"Created {result.shard_count} shards with {result.entity_count} entities")
```

### Email Spinner

```python
from HoloLoom.spinningWheel.email_spinner import EmailSpinner

spinner = EmailSpinner(
    imap_server="imap.gmail.com",
    email_address="user@gmail.com",
    password="app_password"
)
result = await spinner.spin(mailbox="INBOX")

print(f"Processed {result.shard_count} emails")
```

### PDF Spinner

```python
from HoloLoom.spinningWheel.pdf_spinner import PDFSpinner

spinner = PDFSpinner(chunk_strategy="page")
result = await spinner.spin("/path/to/document.pdf")

print(f"Created {result.shard_count} page shards")
```

## Detailed Documentation

### GitHub Spinner

**Location**: `HoloLoom/spinningWheel/github_spinner.py` (680 lines)

#### Features

- **Issue extraction**: Number, title, state, labels, comments
- **Pull request extraction**: State, diff stats, reviews, merges
- **Author/contributor tracking**: User names and emails
- **Importance scoring**: Based on engagement (reactions, comments), code impact
- **Graceful API handling**: Rate limit awareness, fallback strategies

#### Configuration

```python
spinner = GitHubSpinner(
    github_token=None,              # Optional: GitHub PAT (env var: GITHUB_TOKEN)
    importance_threshold=0.2,       # Min importance to include (0.0-1.0)
    include_issues=True,
    include_prs=True,
    max_issues=50,
    max_prs=50
)
```

#### API

```python
# Main method
result = await spinner.spin("https://github.com/owner/repo")

# Metadata available in shards
shard.metadata = {
    'source': 'github',
    'type': 'issue|pull_request',
    'owner': 'owner',
    'repo': 'repo',
    'number': 42,
    'state': 'open|closed|merged',
    'labels': ['bug', 'feature'],
    'importance_score': 0.75,
    'url': 'https://github.com/...'
}
```

#### Usage Notes

- **Public repos**: No token needed
- **Private repos**: Requires GitHub PAT with `repo` scope
- **Rate limits**: API handles rate limiting automatically
- **Cost**: ~2 API calls per item (+ stats if enabled)

#### Testing

```bash
# Run tests
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestGitHubSpinner -v

# Demo
PYTHONPATH=. python demos/demo_github_spinner.py
```

---

### Slack Spinner

**Location**: `HoloLoom/spinningWheel/slack_spinner.py` (675 lines)

#### Features

- **Channel message extraction**: With timestamps and user info
- **Thread reply support**: Parent-reply relationships
- **Emoji reactions tracking**: Engagement metrics
- **User enrichment**: Real names and profiles (optional)
- **Time-based filtering**: Fetch last N days
- **Engagement scoring**: Based on reactions, replies, content

#### Configuration

```python
spinner = SlackSpinner(
    slack_token="xoxb-your-token",  # Required: Bot token
    importance_threshold=0.2,
    include_threads=True,           # Extract thread replies
    days_back=30,                   # Message history depth
    max_messages_per_channel=100
)
```

#### API

```python
# Main method
result = await spinner.spin("C1234567890")  # Channel ID

# Metadata available in shards
shard.metadata = {
    'source': 'slack',
    'type': 'message|thread_reply',
    'channel_id': 'C...',
    'user_id': 'U...',
    'username': 'user_name',
    'timestamp': '1609459200.000100',
    'reactions': {'thumbsup': 3, 'heart': 1},
    'reply_count': 5
}
```

#### Setup

1. **Create Slack App**: https://api.slack.com/apps
2. **OAuth Scopes**:
   - `conversations:history` - Read messages
   - `users:read` - Read user info
   - `reactions:read` - Read reactions (optional)
3. **Get Bot Token**: Copy xoxb-... token
4. **Set environment**: `export SLACK_BOT_TOKEN=xoxb-...`

#### Channel ID Format

- Web URL: `https://app.slack.com/client/TXXXXXX/C0XXXXXXX`
- Channel ID: `C0XXXXXXX`

#### Usage Notes

- **Pagination**: Handles large channels automatically
- **User caching**: Speeds up repeated requests
- **Thread handling**: Optional deep thread extraction
- **Rate limiting**: Respects Slack API limits

#### Testing

```bash
# Run tests
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestSlackSpinner -v

# Demo (requires SLACK_BOT_TOKEN)
PYTHONPATH=. python demos/demo_slack_spinner.py
```

---

### Email Spinner

**Location**: `HoloLoom/spinningWheel/email_spinner.py` (650 lines)

#### Features

- **IMAP mailbox support**: Gmail, Outlook, custom servers
- **Email header extraction**: From, To, Cc, Bcc, Subject, Date
- **HTML/text parsing**: Automatic HTML-to-text conversion (optional)
- **Attachment detection**: Lists and counts attachments
- **Thread detection**: Identifies replies and thread relationships
- **Importance scoring**: Based on sender authority, message length, keywords

#### Configuration

```python
spinner = EmailSpinner(
    imap_server="imap.gmail.com",
    email_address="user@gmail.com",
    password="app-specific-password",  # NOT regular password!
    importance_threshold=0.2,
    use_ssl=True,
    days_back=30,
    max_messages=100
)
```

#### IMAP Servers

```
Gmail:        imap.gmail.com:993
Outlook:      outlook.office365.com:993
Yahoo:        imap.mail.yahoo.com:993
Fastmail:     imap.fastmail.com:993
ProtonMail:   imap.protonmailplus.com:993
```

#### Gmail Setup

1. **Enable 2FA**: myaccount.google.com/security
2. **Generate app password**:
   - Account → Security → App passwords
   - Select "Mail" and "Windows Computer" (or other)
   - Copy 16-character password
3. **Use app password** (not regular password)

#### API

```python
# Main method
result = await spinner.spin(mailbox="INBOX")

# Supported mailbox names
mailboxes = {
    "INBOX": "Incoming mail",
    "[Gmail]/Sent Mail": "Sent items",
    "[Gmail]/Drafts": "Draft messages",
    "[Gmail]/All Mail": "All messages",
    "[Gmail]/Trash": "Deleted items",
    "Custom Folder": "Any custom folder"
}

# Metadata available in shards
shard.metadata = {
    'source': 'email',
    'mailbox': 'INBOX',
    'subject': 'Meeting notes',
    'sender': 'user@example.com',
    'sender_name': 'User Name',
    'recipients': ['recipient@example.com'],
    'date': '2025-01-15T10:30:00+00:00',
    'attachments': ['file.pdf'],
    'in_reply_to': '<parent@example.com>',
    'importance_score': 0.65
}
```

#### Usage Notes

- **App passwords**: Required for Gmail and Outlook (not regular password)
- **SSL/TLS**: Usually required (use_ssl=True)
- **Connection limits**: Close connection on shutdown (automatic with context manager)
- **Privacy**: Messages only stored in HoloLoom memory, not cached

#### Testing

```bash
# Run tests
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestEmailSpinner -v

# Demo (requires IMAP credentials)
PYTHONPATH=. python demos/demo_email_spinner.py
```

---

### PDF Spinner

**Location**: `HoloLoom/spinningWheel/pdf_spinner.py` (625 lines)

#### Features

- **Text extraction**: From PDF pages
- **Metadata extraction**: Title, author, creation date
- **Image detection**: Detects and counts images per page
- **Table detection**: Basic heuristic detection
- **Chunking strategies**: Page-level, section-level, custom
- **Early page weighting**: Title pages scored higher
- **Graceful degradation**: Works without PIL or pytesseract

#### Configuration

```python
spinner = PDFSpinner(
    importance_threshold=0.1,
    chunk_strategy="page",      # "page", "section", or "custom"
    use_ocr=False,              # OCR for scanned PDFs (optional)
    max_pages=None              # Limit pages (None = all)
)
```

#### Chunking Strategies

**page**: One shard per page (default, simple)
```python
# Each page becomes separate shard
spinner = PDFSpinner(chunk_strategy="page")
result = await spinner.spin("document.pdf")
# For 100-page doc: 100 shards
```

**section**: Group pages by detected sections
```python
# Detects chapter/section headers
spinner = PDFSpinner(chunk_strategy="section")
result = await spinner.spin("document.pdf")
# For 100-page doc with 10 chapters: ~10 shards
```

**custom**: User-defined grouping
```python
# For advanced use cases
spinner = PDFSpinner(chunk_strategy="custom")
# Implement custom grouping logic
```

#### API

```python
# Main method
result = await spinner.spin("/path/to/document.pdf")

# Metadata available in shards
shard.metadata = {
    'source': 'pdf',
    'file': '/absolute/path/to/document.pdf',
    'filename': 'document.pdf',
    'title': 'Machine Learning',
    'author': 'Kevin Murphy',
    'page_number': 42,
    'total_pages': 1024,
    'has_images': True,
    'image_count': 3,
    'table_count': 2,
    'importance_score': 0.72
}
```

#### Installation

```bash
# Basic PDF text extraction
pip install PyPDF2

# Image support
pip install Pillow

# OCR for scanned PDFs
pip install pytesseract
# Also requires system installation of Tesseract:
#   Ubuntu: sudo apt-get install tesseract-ocr
#   macOS: brew install tesseract
#   Windows: Download from https://github.com/UB-Mannheim/tesseract
```

#### Usage Notes

- **Large PDFs**: Consider max_pages to limit memory
- **Scanned PDFs**: Use use_ocr=True (requires pytesseract)
- **Encrypted PDFs**: Not supported (raises error)
- **Performance**: Extraction ~50-100ms per page

#### Testing

```bash
# Run tests
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestPDFSpinner -v

# Demo
PYTHONPATH=. python demos/demo_pdf_spinner.py
```

---

## Common Patterns

### Using Convenience Functions

All spinners provide convenience functions for quick usage:

```python
# GitHub
from HoloLoom.spinningWheel.github_spinner import spin_github_repo
result = await spin_github_repo("https://github.com/user/repo", max_issues=50)

# Slack
from HoloLoom.spinningWheel.slack_spinner import spin_slack_channel
result = await spin_slack_channel("C1234567890", "xoxb-token", days_back=30)

# Email
from HoloLoom.spinningWheel.email_spinner import spin_email_inbox
result = await spin_email_inbox("imap.gmail.com", "user@gmail.com", "password")

# PDF
from HoloLoom.spinningWheel.pdf_spinner import spin_pdf
result = await spin_pdf("/path/to/document.pdf")
```

### Error Handling

All spinners implement graceful error handling:

```python
result = await spinner.spin(source)

if not result.success:
    print(f"Error: {result.error_message}")
    print(f"Warnings: {result.warnings}")
else:
    print(f"Success: {result.shard_count} shards")
```

### Importance Filtering

All spinners support importance-based filtering:

```python
spinner = GitHubSpinner(importance_threshold=0.5)
result = await spinner.spin("https://github.com/user/repo")

# Only shards with importance >= 0.5 included
# Lower-priority items automatically filtered
```

### Performance Metrics

All results include performance data:

```python
result = await spinner.spin(source)

print(f"Processing time: {result.processing_time_ms}ms")
print(f"Input size: {result.input_size_bytes} bytes")
print(f"Avg importance: {result.avg_importance:.2f}")
print(f"Avg confidence: {result.avg_confidence:.2f}")
```

---

## Integration with HoloLoom

### Adding to Orchestrator

```python
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.spinningWheel.github_spinner import GitHubSpinner

spinner = GitHubSpinner()
result = await spinner.spin("https://github.com/anthropic-ai/anthropic-sdk-python")

# Convert to memory shards
shards = result.shards

# Use with orchestrator
config = Config.fused()
async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
    spacetime = await orchestrator.weave(query)
```

### Adding to Memory System

```python
from HoloLoom import HoloLoom
from HoloLoom.spinningWheel.pdf_spinner import PDFSpinner

spinner = PDFSpinner()
result = await spinner.spin("/path/to/document.pdf")

async with HoloLoom() as loom:
    for shard in result.shards:
        memory = await loom.experience(shard.text)
```

---

## Testing

### Running Tests

```bash
# All spinner tests
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py -v

# Individual spinners
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestGitHubSpinner -v
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestSlackSpinner -v
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestEmailSpinner -v
pytest HoloLoom/spinningWheel/tests/test_new_spinners.py::TestPDFSpinner -v
```

### Running Demos

```bash
# GitHub
PYTHONPATH=. python demos/demo_github_spinner.py

# Slack (requires SLACK_BOT_TOKEN)
PYTHONPATH=. SLACK_BOT_TOKEN=xoxb-... python demos/demo_slack_spinner.py

# Email (requires IMAP credentials)
PYTHONPATH=. python demos/demo_email_spinner.py

# PDF
PYTHONPATH=. python demos/demo_pdf_spinner.py
```

---

## Troubleshooting

### GitHub Spinner

**Issue**: Rate limit errors
**Solution**: Set GITHUB_TOKEN environment variable with personal access token

**Issue**: 404 on private repo
**Solution**: Use GitHub PAT with `repo` scope for private repositories

### Slack Spinner

**Issue**: "Invalid token" or "not_authed"
**Solution**: Verify token format starts with `xoxb-` and has correct scopes

**Issue**: No messages returned
**Solution**: Check channel ID format, ensure bot has channel access

### Email Spinner

**Issue**: "Authentication failed"
**Solution**: Use app-specific password, not regular password (Gmail/Outlook)

**Issue**: Connection timeout
**Solution**: Check firewall rules, verify IMAP server address

### PDF Spinner

**Issue**: "No module named PyPDF2"
**Solution**: `pip install PyPDF2`

**Issue**: "No module named PIL" (for images)
**Solution**: `pip install Pillow`

---

## Specifications

| Metric | GitHub | Slack | Email | PDF |
|--------|--------|-------|-------|-----|
| **Lines of code** | 680 | 675 | 650 | 625 |
| **External dependencies** | requests | slack-sdk | imaplib* | PyPDF2 |
| **Async support** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Streaming support** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Error handling** | ✅ Graceful | ✅ Graceful | ✅ Graceful | ✅ Graceful |
| **Rate limiting** | ✅ Yes | ✅ Yes | ✅ N/A | ✅ N/A |
| **Importance scoring** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Entity extraction** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Motif extraction** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

*imaplib is part of Python stdlib

---

## Future Enhancements

- **GitHub**: Commit history, releases, wiki
- **Slack**: File attachments, emoji reactions with user sentiment
- **Email**: Calendar integration, attachment content extraction
- **PDF**: Native OCR integration, table extraction, semantic chunking

---

## Support & Feedback

For issues, feature requests, or contributions:
- Check existing tests in `test_new_spinners.py`
- Review demo scripts in `demos/`
- Examine existing spinner implementations
- Follow protocol in `HoloLoom/spinningWheel/protocol.py`

---

**Created**: November 2025  
**Status**: ✅ Production Ready  
**Maintained by**: Claude Code

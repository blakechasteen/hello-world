# WebsiteSpinner - Web Content Ingestion

**Status**: Ready to use ✓
**Dependencies**: `requests`, `beautifulsoup4` (optional, for scraping)

---

## Overview

The WebsiteSpinner automatically converts web content into searchable memory shards with:
- **Smart Content Extraction**: Removes navigation, ads, scripts - keeps main content
- **Intelligent Chunking**: Uses TextSpinner for paragraph/sentence-based chunking
- **Entity & Motif Extraction**: Identifies key concepts and patterns
- **Rich Metadata**: Stores URL, domain, title, visit statistics
- **Auto-Tagging**: Automatically tags by domain

---

## Quick Start

### 1. Install Dependencies

```bash
pip install requests beautifulsoup4
```

### 2. Basic Usage

```python
from HoloLoom.spinningWheel.website import spin_webpage
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories

# Ingest a webpage
shards = await spin_webpage('https://example.com/article')

# Store in memory
memory = await create_unified_memory(user_id="blake")
memories = shards_to_memories(shards)
ids = await memory.store_many(memories)

print(f"Stored {len(ids)} chunks from webpage")
```

### 3. Through MCP (Claude Desktop)

Just say to Claude:
```
Ingest this webpage: https://example.com/beekeeping-winter-guide
```

Claude will use the `ingest_webpage` tool to automatically scrape, chunk, and store the content.

---

## Features

### Automatic Content Extraction

The spinner removes noise and extracts main content:

```python
# Will automatically:
# 1. Fetch the webpage
# 2. Remove scripts, styles, nav, ads
# 3. Find main content area (<main>, <article>, etc.)
# 4. Extract clean text
# 5. Chunk intelligently
# 6. Extract entities and motifs

shards = await spin_webpage('https://example.com/article')
```

### Custom Chunking

Control how content is split:

```python
from HoloLoom.spinningWheel.website import WebsiteSpinner, WebsiteSpinnerConfig

config = WebsiteSpinnerConfig(
    chunk_by="paragraph",  # or "sentence", "fixed"
    chunk_size=500,        # for fixed mode
    min_content_length=100 # skip short pages
)

spinner = WebsiteSpinner(config)
shards = await spinner.spin({'url': 'https://example.com'})
```

### With Pre-Fetched Content

If you already have the content:

```python
shards = await spin_webpage(
    url='https://example.com',
    title='My Article Title',
    content='Full article text here...',
    tags=['research', 'python']
)
```

### Browser History Integration

Track which pages you visited and when:

```python
from datetime import datetime

shards = await spin_webpage(
    url='https://example.com/article',
    visited_at=datetime.now(),
    duration=120,  # spent 2 minutes on page
    tags=['browsing-history']
)

# Each shard will have metadata:
# - visited_at: when you visited
# - duration_seconds: how long you stayed
# - domain: example.com
# - url: full URL
```

---

## Browser History Auto-Ingest

### Read Your Browsing History

```python
from HoloLoom.spinningWheel.browser_history import get_recent_history

# Get last week of meaningful visits (30+ seconds on page)
visits = get_recent_history(days_back=7, min_duration=30, browser='chrome')

for visit in visits:
    print(f"{visit.title} - {visit.duration}s")
```

### Supported Browsers

- **Chrome/Chromium** ✓
- **Microsoft Edge** ✓
- **Brave** ✓
- **Firefox** ✓

### Auto-Ingest Script

```bash
# Ingest last 7 days from Chrome
python HoloLoom/spinningWheel/examples/browser_history_ingest.py

# Dry run (see what would be ingested)
python browser_history_ingest.py --dry-run

# Ingest last 30 days from all browsers
python browser_history_ingest.py --days 30 --browser all

# Custom minimum duration
python browser_history_ingest.py --min-duration 60
```

The script automatically:
- Reads browser history databases
- Filters for meaningful visits
- Excludes noise (search engines, social media)
- Scrapes page content
- Stores in memory with metadata

---

## MCP Integration

The `ingest_webpage` tool is available in Claude Desktop:

### Via Claude Desktop

```
"Ingest this article: https://example.com/beekeeping-guide"
```

**Response:**
```
✓ Webpage ingested successfully

🌐 URL: https://example.com/beekeeping-guide
📄 Title: Winter Beekeeping Guide
🏠 Domain: example.com
📊 Chunks Created: 7
💾 Memories Stored: 7

🔍 Extracted Features:
  • Entities: 23 total (12 unique)
  • Motifs: 15 total (8 unique)

  Sample Entities: Winter, Queen, Cluster, Honey, Colony
  Sample Motifs: temperature, feeding, ventilation

🔖 Tags: web:example.com

Now you can search for this content using semantic queries!
```

### With Tags

```
"Ingest this research paper: https://arxiv.org/abs/12345 and tag it as 'AI' and 'research'"
```

---

## Architecture

### Pipeline

```
URL Input
  ↓
Fetch & Parse (BeautifulSoup)
  ↓
Extract Main Content (remove noise)
  ↓
Process through TextSpinner (chunking)
  ↓
MemoryShards (with entities/motifs)
  ↓
Enrich with Web Metadata
  ↓
Store in Memory System
```

### Metadata Structure

Each memory shard contains:

```python
{
    'url': 'https://example.com/article',
    'domain': 'example.com',
    'title': 'Article Title',
    'content_type': 'webpage',
    'visited_at': '2025-10-24T10:30:00',  # if provided
    'duration_seconds': 120,               # if provided
    'visit_count': 5,                      # if provided
    'tags': ['web:example.com', 'research']
}
```

### Content Extraction Strategy

1. **Remove Noise**: Scripts, styles, nav, footer, header, ads
2. **Find Main Content**: Try `<main>`, `<article>`, `.content`, `#content`
3. **Extract Text**: Clean whitespace, preserve structure
4. **Chunk**: Use TextSpinner for intelligent splits
5. **Extract Features**: Entities, motifs via NLP (if available)

---

## Use Cases

### 1. Research Library

```python
# Ingest all your bookmarked articles
bookmarks = [
    'https://example.com/article1',
    'https://example.com/article2',
    'https://example.com/article3',
]

for url in bookmarks:
    shards = await spin_webpage(url, tags=['research', 'bookmarks'])
    memories = shards_to_memories(shards)
    await memory.store_many(memories)

# Later: "What did that article say about async Python?"
```

### 2. Automatic Learning Journal

```python
# Background service that ingests your browsing
while True:
    visits = get_recent_history(hours_back=1, min_duration=60)

    for visit in visits:
        if not already_ingested(visit.url):
            shards = await spin_webpage(
                url=visit.url,
                visited_at=visit.timestamp,
                duration=visit.duration
            )
            # Store...

    await asyncio.sleep(3600)  # Check hourly
```

### 3. Domain-Specific Collection

```python
# Collect all beekeeping content you read
visits = get_recent_history(days_back=30)

beekeeping_visits = [
    v for v in visits
    if any(term in v.url.lower() for term in ['bee', 'hive', 'honey'])
]

for visit in beekeeping_visits:
    shards = await spin_webpage(visit.url, tags=['beekeeping'])
    # Store...
```

### 4. Browser Extension Capture

Future: One-click "Save to HoloLoom" button in browser that:
1. Captures current page
2. Sends to local MCP server
3. Processes through WebsiteSpinner
4. Stores in memory
5. Shows notification

---

## Configuration Options

```python
@dataclass
class WebsiteSpinnerConfig(SpinnerConfig):
    # Scraping
    timeout: int = 10               # HTTP timeout
    user_agent: str = "..."         # Custom user agent

    # Filtering
    min_content_length: int = 100   # Skip short pages
    max_content_length: int = 100000 # Truncate very long pages

    # Chunking (passed to TextSpinner)
    chunk_by: str = "paragraph"     # paragraph, sentence, fixed
    chunk_size: int = 500           # for fixed mode

    # Tagging
    auto_tag_domain: bool = True    # Add web:domain.com tag
    include_url_in_tags: bool = False # Add full URL as tag
```

---

## Browser History Database Locations

### Windows

- **Chrome**: `%LOCALAPPDATA%\Google\Chrome\User Data\Default\History`
- **Edge**: `%LOCALAPPDATA%\Microsoft\Edge\User Data\Default\History`
- **Brave**: `%LOCALAPPDATA%\BraveSoftware\Brave-Browser\User Data\Default\History`
- **Firefox**: `%APPDATA%\Mozilla\Firefox\Profiles\*.default\places.sqlite`

### macOS

- **Chrome**: `~/Library/Application Support/Google/Chrome/Default/History`
- **Edge**: `~/Library/Application Support/Microsoft Edge/Default/History`
- **Firefox**: `~/Library/Application Support/Firefox/Profiles/*.default/places.sqlite`

### Linux

- **Chrome**: `~/.config/google-chrome/Default/History`
- **Firefox**: `~/.mozilla/firefox/*.default/places.sqlite`

---

## Error Handling

```python
try:
    shards = await spin_webpage('https://example.com')
except requests.RequestException as e:
    # Network error, timeout, 404, etc.
    logger.error(f"Failed to fetch: {e}")
except Exception as e:
    # Other errors (parsing, extraction, etc.)
    logger.error(f"Failed to process: {e}")
```

---

## Performance

- **Scraping**: ~1-3 seconds per page (depends on site speed)
- **Processing**: ~0.1-0.5 seconds per page (chunking + entity extraction)
- **Storage**: ~0.1 seconds (batch store)

**Total**: ~2-4 seconds per webpage

For bulk ingestion (browser history), expect:
- 50 pages: ~2-3 minutes
- 200 pages: ~10-15 minutes

---

## Future Enhancements

### Planned
- [ ] JavaScript rendering (Playwright/Selenium for dynamic sites)
- [ ] PDF extraction
- [ ] Image OCR
- [ ] Video transcript extraction
- [ ] Browser extension for one-click capture
- [ ] RSS feed monitoring
- [ ] Webpage change detection (re-ingest if updated)

### Ideas
- Auto-detect paywall and skip
- Smart deduplication (same content, different URLs)
- Extract structured data (recipes, events, products)
- Sentiment analysis
- Automatic summarization

---

## Troubleshooting

**"No content extracted"**
- Page may be behind paywall
- Dynamic JavaScript site (not yet supported)
- Very short content (< min_content_length)

**"Browser history not found"**
- Check browser is installed
- Verify paths in `browser_history.py`
- Browser must be closed (database locked otherwise)

**"Import error: requests/beautifulsoup4"**
```bash
pip install requests beautifulsoup4
```

---

## See Also

- [TextSpinner README](README_TEXT.md) - Text processing
- [YouTubeSpinner README](README_YOUTUBE.md) - YouTube transcripts
- [MCP Setup Guide](../Documentation/MCP_SETUP_GUIDE.md) - Claude Desktop integration

---

**Status**: Production Ready ✓
**Version**: 1.0
**Last Updated**: 2025-10-24

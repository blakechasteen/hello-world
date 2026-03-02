# SpinningWheel Demo

Interactive demonstration of HoloLoom's multi-modal data ingestion system.

## Run

```bash
python demo_spinningwheel_simple.py
```

## Spinners

### TextSpinner
Processes plain text documents into structured MemoryShards with automatic entity extraction.

```python
from hololoom.spinningWheel import TextSpinner

spinner = TextSpinner()
shards = await spinner.spin("Your document text here...")
```

### WebsiteSpinner
Scrapes web pages, extracts content, processes images, and chunks into shards.

```python
from hololoom.spinningWheel import WebsiteSpinner

spinner = WebsiteSpinner()
shards = await spinner.spin("https://example.com/article")
```

### YouTubeSpinner
Fetches video transcripts and processes them into searchable memory.

```python
from hololoom.spinningWheel import YouTubeSpinner

spinner = YouTubeSpinner()
shards = await spinner.spin("VIDEO_ID", languages=['en'])
```

### Batch Processing

```python
from hololoom.spinningWheel import SpinningWheel

wheel = SpinningWheel()
results = await wheel.spin_batch([
    ("text", "Document content..."),
    ("url", "https://example.com"),
    ("youtube", "dQw4w9WgXcQ"),
])
```

## Performance

| Spinner | Throughput | Notes |
|---------|-----------|-------|
| TextSpinner | <1ms/doc | Instant |
| WebsiteSpinner | 2-3s/page | Network-bound |
| YouTubeSpinner | 1-2s/video | API-bound |
| Batch (10 items) | ~5s total | Concurrent processing |

## Dependencies

```bash
pip install hololoom[nlp]  # For entity extraction
pip install beautifulsoup4 requests  # For web scraping
pip install youtube-transcript-api  # For YouTube
```

See [RECURSIVE_CRAWLING_MATRYOSHKA.md](RECURSIVE_CRAWLING_MATRYOSHKA.md) for multi-page web crawling.

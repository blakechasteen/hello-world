# SpinningWheel MCP Server

**Model Context Protocol (MCP) server for HoloLoom's data ingestion system.**

Exposes SpinningWheel's multimodal data ingestion capabilities as tools for Claude Desktop and other MCP-compatible clients.

## Features

| Tool | Description |
|------|-------------|
| `ingest_webpage` | Scrape web pages with text and images |
| `ingest_browser_history` | Read browser history from Chrome, Firefox, Edge, Safari, Brave |
| `ingest_recursive` | Recursively crawl websites with matryoshka importance gating |
| `process_text` | Chunk and process long text documents |

## Installation

```bash
# Install MCP package
pip install mcp

# Install SpinningWheel dependencies
pip install beautifulsoup4 requests pillow
```

## Setup for Claude Desktop

1. Open Claude Desktop settings
2. Navigate to MCP Servers configuration
3. Add the following to `claude_desktop_config.json`:

**Windows:**
```json
{
  "mcpServers": {
    "spinning-wheel": {
      "command": "python",
      "args": ["-m", "hololoom.spinningWheel.mcp_server"],
      "cwd": "C:/Users/YOUR_USERNAME/path/to/mythRL"
    }
  }
}
```

**macOS/Linux:**
```json
{
  "mcpServers": {
    "spinning-wheel": {
      "command": "python3",
      "args": ["-m", "hololoom.spinningWheel.mcp_server"],
      "cwd": "/path/to/mythRL"
    }
  }
}
```

4. Restart Claude Desktop

## Usage in Claude Desktop

Once configured, you can ask Claude to:

### Scrape a Web Page
```
Can you scrape the page at https://example.com/article and summarize it?
```

### Read Browser History
```
What websites have I visited in the last 3 days?
```

### Crawl a Website
```
Crawl https://docs.python.org starting from the main page,
focusing on "async await" topics, max 10 pages
```

### Process Text
```
Process this long document into chunks: [paste document]
```

## Tool Reference

### ingest_webpage

Scrapes a single web page with text and optional image extraction.

**Parameters:**
- `url` (required): URL of the web page
- `extract_images` (optional, default: true): Whether to extract images

**Returns:** Structured content with title, text, entities, and image descriptions

### ingest_browser_history

Reads browser history from detected browsers.

**Parameters:**
- `browsers` (optional): List of browsers ('chrome', 'firefox', 'edge', 'safari', 'brave'). Empty for all.
- `days_back` (optional, default: 7): Number of days of history
- `max_entries` (optional, default: 100): Maximum entries to return

**Returns:** History entries with URLs, titles, visit counts, timestamps

### ingest_recursive

Recursively crawls a website using matryoshka importance gating.

**Parameters:**
- `url` (required): Starting URL
- `topic` (optional): Topic keywords for relevance scoring
- `max_depth` (optional, default: 2): Maximum crawl depth
- `max_pages` (optional, default: 20): Maximum pages to crawl
- `same_domain_only` (optional, default: true): Stay on same domain

**Returns:** Crawl summary and content from discovered pages

### process_text

Chunks long text into memory shards.

**Parameters:**
- `text` (required): Text content to process
- `chunk_size` (optional, default: 1000): Target chunk size in characters
- `source` (optional, default: "user_input"): Source identifier

**Returns:** Chunked text with metadata

## Matryoshka Importance Gating

The recursive crawler uses hierarchical importance thresholds that increase with depth:

```
Depth 0: Threshold 0.30 (accept most relevant content)
Depth 1: Threshold 0.45
Depth 2: Threshold 0.60
Depth 3: Threshold 0.75
Depth 4+: Threshold 0.85 (maximum selectivity)
```

This creates a natural funnel that prevents crawl explosion:
- Outer levels accept broadly relevant content
- Inner levels require high relevance to continue
- Prevents wasting resources on tangential content

## Running Standalone

For testing without Claude Desktop:

```bash
cd /path/to/mythRL
PYTHONPATH=. python -m hololoom.spinningWheel.mcp_server
```

The server reads from stdin and writes to stdout using the MCP protocol.

## Troubleshooting

### "MCP package not installed"
```bash
pip install mcp
```

### "Website spinner not available"
```bash
pip install beautifulsoup4 requests
```

### Browser history not found
- Ensure the browser has been used at least once
- Check that the browser profile path is accessible
- Try specifying browsers explicitly: `{"browsers": ["chrome"]}`

### Crawl returns empty results
- Check the URL is accessible
- Try lowering importance thresholds
- Ensure the topic matches page content

## Architecture

```
Claude Desktop
    │
    ▼ (stdio MCP protocol)
┌─────────────────────────────────┐
│    SpinningWheelMCPServer       │
│                                 │
│  ┌─────────────────────────┐   │
│  │   Tool: ingest_webpage  │   │
│  │   → WebsiteSpinner      │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │ Tool: ingest_browser_   │   │
│  │        history          │   │
│  │ → BrowserHistorySpinner │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │  Tool: ingest_recursive │   │
│  │  → RecursiveCrawler     │   │
│  │    (Matryoshka Gating)  │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │   Tool: process_text    │   │
│  │   → Text Chunking       │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘
    │
    ▼
  Memory Shards → HoloLoom
```

## Created

December 2025 - HoloLoom SpinningWheel Team

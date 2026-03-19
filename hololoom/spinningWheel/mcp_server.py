"""
MCP Server for SpinningWheel Data Ingestion
============================================

Exposes SpinningWheel's multimodal data ingestion capabilities as MCP tools
for Claude Desktop and other MCP-compatible clients.

Available Tools:
- ingest_webpage: Scrape a web page with text and images
- ingest_browser_history: Read recent browser history
- ingest_recursive: Recursively crawl a website
- process_text: Chunk and process long text documents

Setup in Claude Desktop:
    Add to claude_desktop_config.json:
    {
        "mcpServers": {
            "spinning-wheel": {
                "command": "python",
                "args": ["-m", "hololoom.spinningWheel.mcp_server"],
                "cwd": "C:/path/to/mythRL"
            }
        }
    }

Requirements:
    pip install mcp beautifulsoup4 requests

Author: Claude Code
Date: December 2025
"""

import asyncio
import sys
from typing import Any

try:
    from mcp import Server, TextContent, Tool, ToolResult
    from mcp.server.stdio import stdio_server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Import spinners
try:
    from hololoom.spinningWheel.website import WebsiteSpinner
    WEBSITE_SPINNER_AVAILABLE = True
except ImportError:
    WEBSITE_SPINNER_AVAILABLE = False

try:
    from hololoom.spinningWheel.browser_history import BrowserHistorySpinner
    BROWSER_HISTORY_AVAILABLE = True
except ImportError:
    BROWSER_HISTORY_AVAILABLE = False

try:
    from hololoom.spinningWheel.recursive_crawler import (
        CrawlConfig,
        RecursiveCrawlerSpinner,
    )
    RECURSIVE_CRAWLER_AVAILABLE = True
except ImportError:
    RECURSIVE_CRAWLER_AVAILABLE = False


def shard_to_dict(shard) -> dict[str, Any]:
    """Convert a MemoryShard to a JSON-serializable dict."""
    return {
        'id': shard.id,
        'text': shard.text[:2000] if len(shard.text) > 2000 else shard.text,  # Limit text length
        'episode': shard.episode,
        'entities': shard.entities[:20] if shard.entities else [],  # Limit entities
        'motifs': shard.motifs[:10] if shard.motifs else [],  # Limit motifs
        'metadata': {k: str(v)[:500] for k, v in (shard.metadata or {}).items()},  # Limit metadata
    }


class SpinningWheelMCPServer:
    """MCP Server exposing SpinningWheel functionality."""

    def __init__(self):
        self.server = Server("spinning-wheel")
        self._setup_tools()

    def _setup_tools(self):
        """Register all available tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return list of available tools."""
            tools = []

            # Web page ingestion
            if WEBSITE_SPINNER_AVAILABLE:
                tools.append(Tool(
                    name="ingest_webpage",
                    description="Scrape a web page and extract text content and images. Returns structured data including title, main content, and image descriptions.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the web page to scrape"
                            },
                            "extract_images": {
                                "type": "boolean",
                                "description": "Whether to extract and describe images (default: true)",
                                "default": True
                            }
                        },
                        "required": ["url"]
                    }
                ))

            # Browser history
            if BROWSER_HISTORY_AVAILABLE:
                tools.append(Tool(
                    name="ingest_browser_history",
                    description="Read recent browser history from Chrome, Firefox, Edge, Safari, or Brave. Returns URLs, titles, visit counts, and timestamps.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "browsers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of browsers to read from: 'chrome', 'firefox', 'edge', 'safari', 'brave'. Empty for all detected browsers.",
                                "default": []
                            },
                            "days_back": {
                                "type": "integer",
                                "description": "Number of days of history to retrieve (default: 7)",
                                "default": 7
                            },
                            "max_entries": {
                                "type": "integer",
                                "description": "Maximum number of history entries to return (default: 100)",
                                "default": 100
                            }
                        }
                    }
                ))

            # Recursive web crawling
            if RECURSIVE_CRAWLER_AVAILABLE:
                tools.append(Tool(
                    name="ingest_recursive",
                    description="Recursively crawl a website starting from a URL. Uses matryoshka importance gating to prioritize relevant content and prevent crawl explosion.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Starting URL for the crawl"
                            },
                            "topic": {
                                "type": "string",
                                "description": "Topic keywords for relevance scoring (e.g., 'machine learning transformers')",
                                "default": ""
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum crawl depth (default: 2)",
                                "default": 2
                            },
                            "max_pages": {
                                "type": "integer",
                                "description": "Maximum pages to crawl (default: 20)",
                                "default": 20
                            },
                            "same_domain_only": {
                                "type": "boolean",
                                "description": "Whether to stay on the same domain (default: true)",
                                "default": True
                            }
                        },
                        "required": ["url"]
                    }
                ))

            # Text processing
            tools.append(Tool(
                name="process_text",
                description="Process and chunk a long text document into memory shards. Useful for ingesting documents, articles, or other long-form content.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text content to process"
                        },
                        "chunk_size": {
                            "type": "integer",
                            "description": "Target size for each chunk in characters (default: 1000)",
                            "default": 1000
                        },
                        "source": {
                            "type": "string",
                            "description": "Source identifier for the text (e.g., 'document.pdf')",
                            "default": "user_input"
                        }
                    },
                    "required": ["text"]
                }
            ))

            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            """Handle tool calls."""

            if name == "ingest_webpage":
                return await self._ingest_webpage(arguments)
            elif name == "ingest_browser_history":
                return await self._ingest_browser_history(arguments)
            elif name == "ingest_recursive":
                return await self._ingest_recursive(arguments)
            elif name == "process_text":
                return await self._process_text(arguments)
            else:
                return [TextContent(text=f"Unknown tool: {name}")]

    async def _ingest_webpage(self, args: dict[str, Any]) -> list[TextContent]:
        """Ingest a single web page."""
        if not WEBSITE_SPINNER_AVAILABLE:
            return [TextContent(text="Error: Website spinner not available. Install beautifulsoup4 and requests.")]

        url = args.get("url")
        extract_images = args.get("extract_images", True)

        if not url:
            return [TextContent(text="Error: URL is required")]

        try:
            spinner = WebsiteSpinner()
            result = await spinner.spin(url, extract_images=extract_images)

            if not result.success:
                return [TextContent(text=f"Error scraping {url}: {result.warnings}")]

            # Format results
            output_parts = []
            output_parts.append(f"# Web Page: {url}\n")

            for shard in result.shards[:5]:  # Limit to 5 shards
                shard_dict = shard_to_dict(shard)
                output_parts.append(f"\n## {shard_dict.get('episode', 'Content')}\n")
                output_parts.append(shard_dict['text'][:2000])

                if shard_dict['entities']:
                    output_parts.append(f"\n**Entities**: {', '.join(shard_dict['entities'][:10])}")

            output_parts.append(f"\n\n---\nTotal shards: {len(result.shards)}")

            return [TextContent(text='\n'.join(output_parts))]

        except Exception as e:
            return [TextContent(text=f"Error: {str(e)}")]

    async def _ingest_browser_history(self, args: dict[str, Any]) -> list[TextContent]:
        """Ingest browser history."""
        if not BROWSER_HISTORY_AVAILABLE:
            return [TextContent(text="Error: Browser history spinner not available.")]

        browsers = args.get("browsers", [])
        days_back = args.get("days_back", 7)
        max_entries = args.get("max_entries", 100)

        try:
            spinner = BrowserHistorySpinner()
            result = await spinner.spin(
                browsers=browsers if browsers else None,
                days_back=days_back,
                max_entries_per_browser=max_entries
            )

            if not result.success:
                return [TextContent(text=f"Error reading history: {result.warnings}")]

            # Format results
            output_parts = []
            output_parts.append(f"# Browser History (last {days_back} days)\n")

            for shard in result.shards[:10]:  # Limit output
                shard_dict = shard_to_dict(shard)
                output_parts.append(f"\n## {shard_dict.get('episode', 'History')}\n")
                output_parts.append(shard_dict['text'][:1500])

            output_parts.append(f"\n\n---\nTotal history shards: {len(result.shards)}")

            return [TextContent(text='\n'.join(output_parts))]

        except Exception as e:
            return [TextContent(text=f"Error: {str(e)}")]

    async def _ingest_recursive(self, args: dict[str, Any]) -> list[TextContent]:
        """Recursively crawl a website."""
        if not RECURSIVE_CRAWLER_AVAILABLE:
            return [TextContent(text="Error: Recursive crawler not available. Install beautifulsoup4 and requests.")]

        url = args.get("url")
        topic = args.get("topic", "")
        max_depth = args.get("max_depth", 2)
        max_pages = args.get("max_pages", 20)
        same_domain = args.get("same_domain_only", True)

        if not url:
            return [TextContent(text="Error: URL is required")]

        try:
            config = CrawlConfig(
                max_depth=max_depth,
                max_pages=max_pages,
                same_domain_only=same_domain,
                seed_topic=topic,
            )

            crawler = RecursiveCrawlerSpinner(config)
            result = await crawler.spin(url)

            if not result.success:
                return [TextContent(text=f"Error crawling {url}: {result.warnings}")]

            # Format results
            output_parts = []
            output_parts.append(f"# Recursive Crawl: {url}\n")
            output_parts.append(f"Topic: {topic or 'None specified'}\n")
            output_parts.append(f"Max depth: {max_depth}, Max pages: {max_pages}\n")

            # Show summary shard if present
            for shard in result.shards:
                if shard.metadata.get('content_type') == 'crawl_summary':
                    output_parts.append(f"\n## Crawl Summary\n{shard.text}")
                    break

            # Show first few content shards
            content_shards = [s for s in result.shards if s.metadata.get('content_type') != 'crawl_summary']
            output_parts.append(f"\n## Crawled Pages ({len(content_shards)} total)\n")

            for shard in content_shards[:5]:
                url_crawled = shard.metadata.get('url', 'Unknown')
                depth = shard.metadata.get('crawl_depth', '?')
                importance = shard.metadata.get('crawl_importance', 0)
                output_parts.append(f"\n### {url_crawled} (depth {depth}, importance {importance:.2f})\n")
                output_parts.append(shard.text[:800] + "..." if len(shard.text) > 800 else shard.text)

            return [TextContent(text='\n'.join(output_parts))]

        except Exception as e:
            return [TextContent(text=f"Error: {str(e)}")]

    async def _process_text(self, args: dict[str, Any]) -> list[TextContent]:
        """Process and chunk text."""
        import hashlib

        from hololoom.documentation.types import MemoryShard

        text = args.get("text", "")
        chunk_size = args.get("chunk_size", 1000)
        source = args.get("source", "user_input")

        if not text:
            return [TextContent(text="Error: Text is required")]

        try:
            # Simple chunking by paragraph and size
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = []
            current_size = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if current_size + len(para) > chunk_size and current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                current_chunk.append(para)
                current_size += len(para)

            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))

            # Create shards
            shards = []
            for i, chunk in enumerate(chunks):
                shard = MemoryShard(
                    id=f"{source}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}_{i}",
                    text=chunk,
                    episode=source,
                    entities=[],
                    motifs=['text_chunk'],
                    metadata={
                        'source': source,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk),
                    }
                )
                shards.append(shard)

            # Format output
            output_parts = []
            output_parts.append(f"# Processed Text: {source}\n")
            output_parts.append(f"Total chunks: {len(shards)}\n")
            output_parts.append(f"Original length: {len(text)} characters\n")

            for i, shard in enumerate(shards[:5]):
                output_parts.append(f"\n## Chunk {i+1}/{len(shards)}\n")
                output_parts.append(shard.text[:500] + "..." if len(shard.text) > 500 else shard.text)

            if len(shards) > 5:
                output_parts.append(f"\n... and {len(shards) - 5} more chunks")

            return [TextContent(text='\n'.join(output_parts))]

        except Exception as e:
            return [TextContent(text=f"Error: {str(e)}")]

    async def run(self):
        """Run the MCP server."""
        if not MCP_AVAILABLE:
            print("Error: MCP package not installed. Run: pip install mcp", file=sys.stderr)
            sys.exit(1)

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


def main():
    """Entry point for MCP server."""
    server = SpinningWheelMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

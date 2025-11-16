#!/usr/bin/env python3
"""
Documentation Crawler
=====================
Crawl documentation websites and extract content for RAG.

Capabilities:
- Recursive website crawling
- Content extraction (BeautifulSoup)
- Code example extraction
- Link following with depth limits
- MemoryShard creation for RAG

Author: Claude Code
Date: November 16, 2025
"""

import logging
import requests
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from datetime import datetime
from dataclasses import dataclass
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class DocPage:
    """Documentation page"""
    url: str
    title: str
    content: str
    code_examples: List[str]
    links: List[str]
    metadata: Dict[str, Any]


class DocumentationCrawler:
    """Crawl documentation websites for RAG"""

    def __init__(self, timeout: int = 10, user_agent: str = "Squad-Doc-Crawler/1.0"):
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}
        self.stats = {
            "pages_crawled": 0,
            "code_examples_found": 0,
            "errors": 0
        }

    def crawl(
        self,
        start_url: str,
        max_pages: int = 50,
        follow_links: bool = True,
        same_domain_only: bool = True
    ) -> List[DocPage]:
        """
        Crawl documentation starting from URL.

        Args:
            start_url: Starting URL
            max_pages: Maximum pages to crawl
            follow_links: Whether to follow links
            same_domain_only: Only crawl same domain

        Returns:
            List of documentation pages
        """
        logger.info(f"Crawling documentation from: {start_url}")

        visited: Set[str] = set()
        to_visit: List[str] = [start_url]
        pages: List[DocPage] = []

        base_domain = urlparse(start_url).netloc

        while to_visit and len(pages) < max_pages:
            url = to_visit.pop(0)

            if url in visited:
                continue

            # Check domain if same_domain_only
            if same_domain_only and urlparse(url).netloc != base_domain:
                continue

            try:
                page = self._fetch_page(url)
                pages.append(page)
                visited.add(url)

                self.stats["pages_crawled"] += 1
                self.stats["code_examples_found"] += len(page.code_examples)

                # Add links if following
                if follow_links and len(pages) < max_pages:
                    for link in page.links:
                        if link not in visited:
                            to_visit.append(link)

            except Exception as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                self.stats["errors"] += 1

        logger.info(f"Crawled {len(pages)} pages with {self.stats['code_examples_found']} code examples")
        return pages

    def _fetch_page(self, url: str) -> DocPage:
        """Fetch and parse single page"""
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        title = title_tag.text.strip() if title_tag else url

        # Remove script and style elements
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Extract main content
        main = soup.find('main') or soup.find('article') or soup.find('body')
        content = main.get_text(separator='\n', strip=True) if main else ""

        # Extract code examples
        code_blocks = soup.find_all(['pre', 'code'])
        code_examples = []
        for block in code_blocks:
            code = block.get_text(strip=True)
            if len(code) > 10:  # Minimum length
                code_examples.append(code)

        # Extract links
        links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(url, href)
            if absolute_url.startswith('http'):
                links.append(absolute_url)

        return DocPage(
            url=url,
            title=title,
            content=content[:5000],  # Limit content length
            code_examples=code_examples[:10],  # Limit examples
            links=links[:20],  # Limit links
            metadata={
                "crawled_at": datetime.now().isoformat(),
                "content_length": len(content)
            }
        )

    def pages_to_memory_shards(self, pages: List[DocPage]) -> List[Dict[str, Any]]:
        """Convert documentation pages to MemoryShards"""
        shards = []

        for page in pages:
            # Create main content shard
            shard = {
                "id": f"doc-{page.url.replace('://', '-').replace('/', '-')[:100]}",
                "text": f"{page.title}\n\n{page.content}",
                "entities": [page.title],
                "motifs": ["documentation", "reference"],
                "metadata": {
                    "source": "documentation_crawler",
                    "url": page.url,
                    "title": page.title,
                    "timestamp": datetime.now().isoformat()
                }
            }
            shards.append(shard)

            # Create shards for code examples
            for i, code in enumerate(page.code_examples):
                code_shard = {
                    "id": f"doc-code-{page.url.replace('://', '-').replace('/', '-')[:80]}-{i}",
                    "text": f"Code example from {page.title}:\n\n{code}",
                    "entities": [page.title, "code", "example"],
                    "motifs": ["code", "example", "documentation"],
                    "metadata": {
                        "source": "documentation_crawler",
                        "url": page.url,
                        "type": "code_example",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                shards.append(code_shard)

        return shards

    def get_stats(self) -> Dict[str, int]:
        """Get crawler statistics"""
        return self.stats.copy()

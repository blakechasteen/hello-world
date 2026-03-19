"""
SerpAPI Provider - Google Search Integration
============================================
Search provider using SerpAPI (https://serpapi.com)

Provides access to Google search results with proper parsing.
"""

import logging
from typing import Any

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from ..protocol import RawSearchResult, SearchResultType

logger = logging.getLogger(__name__)


class SerpAPIProvider:
    """
    SerpAPI search provider.

    Implements SearchProvider protocol for Google search via SerpAPI.

    Usage:
        provider = SerpAPIProvider(api_key="your_key")
        results = await provider.search("What is Thompson Sampling?", limit=10)
    """

    def __init__(
        self,
        api_key: str,
        engine: str = "google",
        location: str = "United States",
        hl: str = "en",
        gl: str = "us"
    ):
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library required for SerpAPI. "
                "Install with: pip install requests"
            )

        self.api_key = api_key
        self.engine = engine
        self.location = location
        self.hl = hl  # Language
        self.gl = gl  # Country

        self.base_url = "https://serpapi.com/search"

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs
    ) -> list[RawSearchResult]:
        """
        Execute Google search via SerpAPI.

        Args:
            query: Search query
            limit: Maximum results (Note: SerpAPI returns ~10 per page)
            **kwargs: Additional SerpAPI parameters

        Returns:
            List of RawSearchResult objects
        """
        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": self.engine,
            "location": self.location,
            "hl": self.hl,
            "gl": self.gl,
            "num": min(limit, 100),  # SerpAPI limit
            **kwargs
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return self._parse_results(data)

        except requests.exceptions.RequestException as e:
            logger.error(f"SerpAPI request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"SerpAPI parsing failed: {e}")
            return []

    async def health_check(self) -> bool:
        """
        Check if SerpAPI is accessible.

        Returns:
            True if API is working
        """
        try:
            # Test query
            results = await self.search("test", limit=1)
            return len(results) > 0
        except Exception:
            return False

    def get_name(self) -> str:
        """Get provider name."""
        return "serpapi"

    def _parse_results(self, data: dict[str, Any]) -> list[RawSearchResult]:
        """Parse SerpAPI response into RawSearchResult objects."""
        results = []

        # Parse organic results
        for i, item in enumerate(data.get("organic_results", [])):
            result = RawSearchResult(
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                rank=item.get("position", i + 1),
                result_type=SearchResultType.ORGANIC,
                metadata={
                    "displayed_link": item.get("displayed_link", ""),
                    "cached_page_link": item.get("cached_page_link", ""),
                    "source": "serpapi_organic"
                }
            )
            results.append(result)

        # Parse featured snippet (if present)
        if "answer_box" in data:
            answer_box = data["answer_box"]
            result = RawSearchResult(
                url=answer_box.get("link", ""),
                title=answer_box.get("title", "Featured Snippet"),
                snippet=answer_box.get("answer", answer_box.get("snippet", "")),
                rank=0,  # Featured snippets are rank 0
                result_type=SearchResultType.FEATURED,
                metadata={
                    "type": answer_box.get("type", ""),
                    "source": "serpapi_featured"
                }
            )
            # Insert at beginning
            results.insert(0, result)

        # Parse news results (if present)
        for i, item in enumerate(data.get("news_results", [])):
            result = RawSearchResult(
                url=item.get("link", ""),
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                rank=i + 1000,  # Offset to distinguish from organic
                result_type=SearchResultType.NEWS,
                metadata={
                    "source_name": item.get("source", ""),
                    "date": item.get("date", ""),
                    "source": "serpapi_news"
                }
            )
            results.append(result)

        return results

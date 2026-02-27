"""
Mock Search Provider - Testing & Development
============================================
Deterministic mock provider for testing without API calls.

Returns predictable results based on query keywords.
"""

import logging
from typing import List, Optional
import hashlib

from ..protocol import SearchProvider, RawSearchResult, SearchResultType

logger = logging.getLogger(__name__)


class MockProvider:
    """
    Mock search provider for testing.

    Returns deterministic results based on query for reproducible tests.

    Usage:
        provider = MockProvider()
        results = await provider.search("Thompson Sampling", limit=10)
    """

    def __init__(self):
        # Sample data based on common queries
        self.mock_data = {
            "thompson sampling": [
                ("https://arxiv.org/abs/1707.02038", "Thompson Sampling Tutorial", "Comprehensive tutorial on Thompson Sampling for contextual bandits..."),
                ("https://en.wikipedia.org/wiki/Thompson_sampling", "Thompson Sampling - Wikipedia", "Thompson sampling, named after William R. Thompson, is a heuristic..."),
                ("https://www.microsoft.com/research", "Thompson Sampling at Microsoft", "Applications of Thompson Sampling in production systems..."),
            ],
            "python": [
                ("https://python.org", "Python Programming Language", "Official website of the Python programming language..."),
                ("https://docs.python.org", "Python Documentation", "Comprehensive Python documentation and tutorials..."),
            ],
            "machine learning": [
                ("https://scikit-learn.org", "Scikit-learn", "Machine learning in Python..."),
                ("https://tensorflow.org", "TensorFlow", "End-to-end machine learning platform..."),
                ("https://pytorch.org", "PyTorch", "Deep learning framework..."),
            ]
        }

    async def search(
        self,
        query: str,
        limit: int = 100,
        **kwargs
    ) -> List[RawSearchResult]:
        """
        Return mock search results.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of RawSearchResult objects
        """
        query_lower = query.lower()

        # Find matching mock data
        results = []
        for key, data in self.mock_data.items():
            if key in query_lower:
                for i, (url, title, snippet) in enumerate(data):
                    result = RawSearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        rank=i + 1,
                        result_type=SearchResultType.ORGANIC,
                        metadata={"source": "mock"}
                    )
                    results.append(result)

        # If no matches, generate generic results
        if not results:
            results = self._generate_generic_results(query, limit)

        return results[:limit]

    async def health_check(self) -> bool:
        """Always returns True for mock provider."""
        return True

    def get_name(self) -> str:
        """Get provider name."""
        return "mock"

    def _generate_generic_results(self, query: str, limit: int) -> List[RawSearchResult]:
        """Generate generic results for unknown queries."""
        # Use query hash for deterministic but varied results
        query_hash = hashlib.md5(query.encode()).hexdigest()

        results = []
        for i in range(min(limit, 10)):
            result = RawSearchResult(
                url=f"https://example.com/result-{query_hash[:8]}-{i}",
                title=f"Result {i+1} for '{query}'",
                snippet=f"This is a mock result about {query}. " * 3,
                rank=i + 1,
                result_type=SearchResultType.ORGANIC,
                metadata={"source": "mock_generic"}
            )
            results.append(result)

        return results

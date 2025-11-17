"""
Search Tool - Web Search Integration
=====================================
Mock web search tool (can be replaced with real API).
"""

import asyncio
from typing import List, Dict, Any


def register_search(server):
    """Register search tool with MCP server."""

    @server.tool(
        "search_web",
        "Search the web for information",
        category="information",
        timeout=10.0
    )
    async def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results with title, url, snippet

        Note:
            This is a mock implementation. Replace with real search API
            (e.g., Google Custom Search, DuckDuckGo, Brave Search).
        """
        # Simulate API latency
        await asyncio.sleep(0.5)

        # Mock results
        results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a mock search result for the query '{query}'. "
                          f"In production, this would be real search data."
            }
            for i in range(min(max_results, 5))
        ]

        return results


    @server.tool(
        "search_code",
        "Search code repositories",
        category="information",
        timeout=10.0
    )
    async def search_code(query: str, language: str = "python", max_results: int = 5) -> List[Dict[str, str]]:
        """
        Search code repositories (GitHub, etc).

        Args:
            query: Search query
            language: Programming language filter
            max_results: Maximum number of results

        Returns:
            List of code snippets with repository info

        Note:
            This is a mock implementation. Replace with GitHub API.
        """
        await asyncio.sleep(0.5)

        results = [
            {
                "repository": f"user/repo{i+1}",
                "file": f"src/module{i+1}.{language}",
                "snippet": f"# Code snippet for '{query}'\ndef example_{i+1}():\n    pass",
                "url": f"https://github.com/user/repo{i+1}"
            }
            for i in range(min(max_results, 5))
        ]

        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("="*80)
        print("Search Tool Demo")
        print("="*80 + "\n")

        # Mock MCP server
        class MockServer:
            def tool(self, name, desc, **kwargs):
                def decorator(func):
                    return func
                return decorator

        server = MockServer()
        register_search(server)

        # Test web search (if registered)
        print("Web Search Results:")
        print("  (Mock implementation - replace with real API)\n")

        results = await search_web("Thompson Sampling algorithm", max_results=3)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']}")
            print(f"     {result['url']}")
            print(f"     {result['snippet'][:80]}...\n")

        print("✓ Demo complete!")

    asyncio.run(demo())

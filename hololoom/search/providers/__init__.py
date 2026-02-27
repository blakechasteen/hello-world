"""
Search Providers - Pluggable API Backends
==========================================
Protocol-based search provider implementations.

Available providers:
- SerpAPI: Google search via SerpAPI
- Tavily: AI-optimized search API
- Brave: Privacy-focused search
- Mock: Testing/development provider

Usage:
    from HoloLoom.search.providers import create_provider

    provider = await create_provider("serpapi", api_key="...")
    results = await provider.search("What is Thompson Sampling?")
"""

from typing import Optional
import logging

from ..protocol import SearchProvider

logger = logging.getLogger(__name__)


async def create_provider(
    provider_name: str,
    api_key: Optional[str] = None,
    **kwargs
) -> SearchProvider:
    """
    Factory function to create search provider.

    Args:
        provider_name: Provider name ("serpapi", "tavily", "brave", "mock")
        api_key: API key for provider
        **kwargs: Provider-specific configuration

    Returns:
        SearchProvider implementation

    Raises:
        ValueError: If provider not found or API key missing
    """
    provider_name = provider_name.lower()

    if provider_name == "serpapi":
        from .serpapi import SerpAPIProvider
        if not api_key:
            raise ValueError("SerpAPI requires api_key")
        return SerpAPIProvider(api_key=api_key, **kwargs)

    elif provider_name == "tavily":
        from .tavily import TavilyProvider
        if not api_key:
            raise ValueError("Tavily requires api_key")
        return TavilyProvider(api_key=api_key, **kwargs)

    elif provider_name == "brave":
        from .brave import BraveProvider
        if not api_key:
            raise ValueError("Brave Search requires api_key")
        return BraveProvider(api_key=api_key, **kwargs)

    elif provider_name == "mock":
        from .mock_provider import MockProvider
        return MockProvider(**kwargs)

    else:
        raise ValueError(
            f"Unknown search provider: {provider_name}. "
            f"Available: serpapi, tavily, brave, mock"
        )


__all__ = ["create_provider"]

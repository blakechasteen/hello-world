"""
Promptly Analytics Package

Provides comprehensive analytics for prompt performance tracking,
quality monitoring, and optimization recommendations.

Usage:
    from promptly.analytics import PromptAnalyticsEngine

    engine = PromptAnalyticsEngine(db)
    analytics = engine.get_prompt_analytics("my_prompt", days=30)
    print(analytics.recommendation)
"""

from promptly.analytics.engine import (
    PromptAnalyticsEngine,
    create_analytics_engine,
)

__all__ = [
    "PromptAnalyticsEngine",
    "create_analytics_engine",
]

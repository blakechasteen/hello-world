#!/usr/bin/env python3
"""
Prompt Suggestions System
==========================
Intelligent prompt recommendations based on content, usage, and context.

Features:
- Related prompt suggestions (semantic similarity)
- "You might also like..." recommendations
- Context-aware suggestions
- Usage pattern-based recommendations
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from HoloLoom.memory.unified import UnifiedMemory, RecallStrategy
    HOLOLOOM_AVAILABLE = True
except ImportError:
    HOLOLOOM_AVAILABLE = False


@dataclass
class PromptSuggestion:
    """A single prompt suggestion"""
    prompt_id: str
    name: str
    content: str
    relevance: float  # 0.0-1.0
    reason: str  # Why it's suggested
    tags: List[str]


class SuggestionEngine:
    """
    Generate intelligent prompt suggestions.

    Uses HoloLoom semantic search for content-based suggestions.
    """

    def __init__(self, user_id: str = "promptly"):
        """
        Initialize suggestion engine.

        Args:
            user_id: User identifier for HoloLoom
        """
        self.user_id = user_id

        if HOLOLOOM_AVAILABLE:
            self.memory = UnifiedMemory(user_id=user_id)
            self.enabled = True
        else:
            self.memory = None
            self.enabled = False

    def get_related_prompts(
        self,
        prompt_content: str,
        limit: int = 5,
        min_relevance: float = 0.5
    ) -> List[PromptSuggestion]:
        """
        Get prompts related to the given content.

        Args:
            prompt_content: The prompt to find related prompts for
            limit: Maximum number of suggestions
            min_relevance: Minimum relevance score

        Returns:
            List of PromptSuggestion objects
        """
        if not self.enabled:
            return []

        # Use HoloLoom semantic search
        results = self.memory.recall(
            prompt_content,
            strategy=RecallStrategy.SIMILAR,
            limit=limit * 2  # Get more, filter later
        )

        suggestions = []
        for result in results:
            relevance = result.relevance or 0.0

            # Skip if below threshold
            if relevance < min_relevance:
                continue

            # Extract prompt data from context
            context = result.context or {}
            prompt_id = context.get('prompt_id', result.id)
            name = context.get('name', 'Unknown')
            tags = context.get('tags', [])

            # Determine reason
            if relevance > 0.8:
                reason = "Very similar content"
            elif relevance > 0.7:
                reason = "Related topic"
            else:
                reason = "Similar theme"

            suggestions.append(PromptSuggestion(
                prompt_id=prompt_id,
                name=name,
                content=result.text,
                relevance=relevance,
                reason=reason,
                tags=tags
            ))

            if len(suggestions) >= limit:
                break

        return suggestions

    def get_suggestions_by_tags(
        self,
        tags: List[str],
        limit: int = 5
    ) -> List[PromptSuggestion]:
        """
        Get prompts with similar tags.

        Args:
            tags: Tags to match
            limit: Maximum number of suggestions

        Returns:
            List of PromptSuggestion objects
        """
        if not self.enabled:
            return []

        # Search for prompts with these tags
        query = " ".join(tags)
        results = self.memory.recall(
            query,
            strategy=RecallStrategy.SIMILAR,
            limit=limit
        )

        suggestions = []
        for result in results:
            context = result.context or {}
            prompt_tags = context.get('tags', [])

            # Count tag overlap
            overlap = len(set(tags) & set(prompt_tags))

            if overlap > 0:
                relevance = min(1.0, overlap / len(tags))

                suggestions.append(PromptSuggestion(
                    prompt_id=context.get('prompt_id', result.id),
                    name=context.get('name', 'Unknown'),
                    content=result.text,
                    relevance=relevance,
                    reason=f"Shares {overlap} tag(s)",
                    tags=prompt_tags
                ))

        return suggestions[:limit]

    def get_popular_prompts(
        self,
        limit: int = 5,
        analytics: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Get most popular prompts (by usage).

        Args:
            limit: Maximum number of suggestions
            analytics: PromptAnalytics instance (if available)

        Returns:
            List of prompt info dicts
        """
        if not analytics:
            return []

        # Get top prompts by usage
        from .prompt_analytics import PromptAnalytics
        if not isinstance(analytics, PromptAnalytics):
            return []

        top_prompts = analytics.get_top_prompts(metric="success_rate", limit=limit)

        results = []
        for stats in top_prompts:
            results.append({
                'name': stats.prompt_name,
                'usage_count': stats.total_executions,
                'success_rate': stats.success_rate,
                'avg_quality': stats.avg_quality_score,
                'reason': f"Popular ({stats.total_executions} uses)"
            })

        return results

    def get_contextual_suggestions(
        self,
        current_prompt: str,
        current_tags: List[str],
        limit: int = 5
    ) -> List[PromptSuggestion]:
        """
        Get suggestions based on both content and tags.

        Args:
            current_prompt: The current prompt content
            current_tags: Tags of current prompt
            limit: Maximum suggestions

        Returns:
            List of PromptSuggestion objects
        """
        if not self.enabled:
            return []

        # Get content-based suggestions
        content_suggestions = self.get_related_prompts(
            current_prompt,
            limit=limit
        )

        # Get tag-based suggestions
        tag_suggestions = self.get_suggestions_by_tags(
            current_tags,
            limit=limit
        )

        # Merge and deduplicate
        seen_ids = set()
        merged = []

        for sug in content_suggestions + tag_suggestions:
            if sug.prompt_id not in seen_ids:
                seen_ids.add(sug.prompt_id)
                merged.append(sug)

        # Sort by relevance
        merged.sort(key=lambda x: x.relevance, reverse=True)

        return merged[:limit]


# ============================================================================
# Convenience Functions
# ============================================================================

def find_similar_prompts(content: str, limit: int = 5) -> List[PromptSuggestion]:
    """
    Quick function to find similar prompts.

    Args:
        content: Prompt content to match
        limit: Number of suggestions

    Returns:
        List of suggestions
    """
    engine = SuggestionEngine()
    return engine.get_related_prompts(content, limit=limit)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Prompt Suggestion System\n")

    if not HOLOLOOM_AVAILABLE:
        print("[WARN] HoloLoom not available. Suggestions disabled.")
        print("To enable: Ensure HoloLoom is in parent directory")
        exit(0)

    # Create engine
    engine = SuggestionEngine()

    # Test prompt
    test_prompt = """
    Analyze this SQL query for performance issues and suggest optimizations.

    Query: {sql_query}

    Focus on:
    - Index usage
    - Join optimization
    - Query plan efficiency
    """

    print("=== Finding Related Prompts ===\n")
    print(f"Query: {test_prompt[:60]}...")
    print()

    suggestions = engine.get_related_prompts(test_prompt, limit=3)

    if suggestions:
        for i, sug in enumerate(suggestions, 1):
            print(f"{i}. {sug.name}")
            print(f"   Relevance: {sug.relevance:.2f}")
            print(f"   Reason: {sug.reason}")
            print(f"   Tags: {', '.join(sug.tags) if sug.tags else 'None'}")
            print()
    else:
        print("No related prompts found.")
        print("(Store some prompts first using HoloLoom integration)")

    # Test tag-based suggestions
    print("\n=== Tag-Based Suggestions ===\n")
    tag_suggestions = engine.get_suggestions_by_tags(['sql', 'optimization'], limit=3)

    if tag_suggestions:
        for i, sug in enumerate(tag_suggestions, 1):
            print(f"{i}. {sug.name}")
            print(f"   Relevance: {sug.relevance:.2f}")
            print(f"   Reason: {sug.reason}")
            print()
    else:
        print("No tag-based suggestions found.")

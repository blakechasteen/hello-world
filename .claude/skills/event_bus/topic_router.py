"""
Topic Router with Wildcard Pattern Matching
===========================================

**Created**: November 22, 2025
**Purpose**: Fast topic matching with wildcard support (*, **, #)

## Pattern Syntax

- `skill.started.meta` - Exact match
- `skill.*.meta` - Single level wildcard (matches `skill.started.meta`)
- `skill.**` - Multi-level wildcard (matches `skill.started.meta.foo`)
- `pattern.#` - Remainder wildcard (matches `pattern.detected.anything.here`)

## Performance

- **Target**: <0.5ms for pattern matching
- **Strategy**: Trie-based with pre-compiled patterns
- **Complexity**: O(k) where k = topic depth
"""

import re
from typing import List, Set, Optional, Pattern
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Topic Pattern
# ============================================================================

@dataclass
class TopicPattern:
    """Compiled topic pattern for fast matching."""
    pattern_str: str                    # Original pattern
    regex: Pattern                      # Compiled regex
    levels: List[str]                   # Split by '.'
    has_wildcards: bool = False         # Optimization flag
    priority: int = 0                   # Match priority (exact > wildcard)

    @classmethod
    def compile(cls, pattern: str) -> "TopicPattern":
        """Compile topic pattern to regex."""
        levels = pattern.split('.')

        # Check for wildcards
        has_wildcards = any(level in ('*', '**', '#') for level in levels)

        # Convert to regex
        regex_parts = []
        for i, level in enumerate(levels):
            if level == '*':
                # Single level wildcard
                regex_parts.append(r'[^.]+')
            elif level == '**':
                # Multi-level wildcard (matches 1+ levels)
                regex_parts.append(r'.+')
            elif level == '#':
                # Remainder wildcard (matches 0+ levels)
                # Must be last component
                if i != len(levels) - 1:
                    raise ValueError("# wildcard must be at end of pattern")
                regex_parts.append(r'.*')
            else:
                # Literal level
                regex_parts.append(re.escape(level))

        # Join with dots
        regex_str = r'^' + r'\.'.join(regex_parts) + r'$'
        regex = re.compile(regex_str)

        # Calculate priority (exact matches have higher priority)
        priority = 100 - sum(10 if level in ('*', '**', '#') else 0 for level in levels)

        return cls(
            pattern_str=pattern,
            regex=regex,
            levels=levels,
            has_wildcards=has_wildcards,
            priority=priority
        )

    def matches(self, topic: str) -> bool:
        """Check if topic matches this pattern."""
        return self.regex.match(topic) is not None


# ============================================================================
# Topic Router
# ============================================================================

class TopicRouter:
    """
    Fast topic routing with wildcard pattern matching.

    Uses trie-based structure for exact matches and regex for wildcards.
    Optimizes for common case (exact matches) while supporting wildcards.
    """

    def __init__(self):
        # Exact topic matches (fast path)
        self.exact_subscriptions: dict[str, Set[str]] = {}

        # Wildcard patterns (slower path)
        self.wildcard_patterns: List[TopicPattern] = []
        self.wildcard_subscriptions: dict[str, Set[str]] = {}

        # Statistics
        self.stats = {
            'exact_matches': 0,
            'wildcard_matches': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Pattern cache (topic -> matching patterns)
        self._pattern_cache: dict[str, List[TopicPattern]] = {}
        self._cache_max_size = 1000

    def add_subscription(self, pattern: str, subscription_id: str) -> None:
        """Add a subscription for a topic pattern."""
        # Check if exact match
        if not any(wildcard in pattern for wildcard in ('*', '**', '#')):
            # Exact match - add to fast path
            if pattern not in self.exact_subscriptions:
                self.exact_subscriptions[pattern] = set()
            self.exact_subscriptions[pattern].add(subscription_id)
            logger.debug(f"Added exact subscription: {pattern} -> {subscription_id}")
        else:
            # Wildcard pattern - compile and add to slow path
            compiled = TopicPattern.compile(pattern)

            # Add pattern if not exists
            if compiled.pattern_str not in self.wildcard_subscriptions:
                self.wildcard_patterns.append(compiled)
                self.wildcard_subscriptions[compiled.pattern_str] = set()

                # Sort by priority (exact > wildcards)
                self.wildcard_patterns.sort(key=lambda p: p.priority, reverse=True)

            self.wildcard_subscriptions[compiled.pattern_str].add(subscription_id)
            logger.debug(f"Added wildcard subscription: {pattern} -> {subscription_id}")

            # Clear pattern cache (patterns changed)
            self._pattern_cache.clear()

    def remove_subscription(self, pattern: str, subscription_id: str) -> None:
        """Remove a subscription."""
        # Check exact matches
        if pattern in self.exact_subscriptions:
            self.exact_subscriptions[pattern].discard(subscription_id)
            if not self.exact_subscriptions[pattern]:
                del self.exact_subscriptions[pattern]
            logger.debug(f"Removed exact subscription: {pattern} -> {subscription_id}")

        # Check wildcard patterns
        if pattern in self.wildcard_subscriptions:
            self.wildcard_subscriptions[pattern].discard(subscription_id)
            if not self.wildcard_subscriptions[pattern]:
                # Remove pattern
                del self.wildcard_subscriptions[pattern]
                self.wildcard_patterns = [
                    p for p in self.wildcard_patterns
                    if p.pattern_str != pattern
                ]
                logger.debug(f"Removed wildcard subscription: {pattern} -> {subscription_id}")

            # Clear pattern cache
            self._pattern_cache.clear()

    def get_subscribers(self, topic: str) -> Set[str]:
        """Get all subscription IDs matching topic."""
        subscribers = set()

        # Fast path: exact match
        if topic in self.exact_subscriptions:
            subscribers.update(self.exact_subscriptions[topic])
            self.stats['exact_matches'] += 1

        # Slow path: wildcard patterns
        if self.wildcard_patterns:
            # Check cache first
            if topic in self._pattern_cache:
                matching_patterns = self._pattern_cache[topic]
                self.stats['cache_hits'] += 1
            else:
                # Find matching patterns
                matching_patterns = [
                    pattern for pattern in self.wildcard_patterns
                    if pattern.matches(topic)
                ]

                # Cache result (LRU eviction)
                if len(self._pattern_cache) >= self._cache_max_size:
                    # Simple eviction: remove oldest (first key)
                    oldest_key = next(iter(self._pattern_cache))
                    del self._pattern_cache[oldest_key]

                self._pattern_cache[topic] = matching_patterns
                self.stats['cache_misses'] += 1

            # Add subscribers from matching patterns
            for pattern in matching_patterns:
                if pattern.pattern_str in self.wildcard_subscriptions:
                    subscribers.update(
                        self.wildcard_subscriptions[pattern.pattern_str]
                    )

            if matching_patterns:
                self.stats['wildcard_matches'] += 1

        return subscribers

    def matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern."""
        # Fast path: exact match
        if topic == pattern:
            return True

        # Slow path: compile and check
        if any(wildcard in pattern for wildcard in ('*', '**', '#')):
            compiled = TopicPattern.compile(pattern)
            return compiled.matches(topic)

        return False

    def get_stats(self) -> dict:
        """Get routing statistics."""
        return {
            **self.stats,
            'exact_topics': len(self.exact_subscriptions),
            'wildcard_patterns': len(self.wildcard_patterns),
            'cache_size': len(self._pattern_cache),
            'cache_hit_rate': (
                self.stats['cache_hits'] /
                (self.stats['cache_hits'] + self.stats['cache_misses'])
                if self.stats['cache_hits'] + self.stats['cache_misses'] > 0
                else 0.0
            )
        }

    def clear_cache(self) -> None:
        """Clear pattern cache."""
        self._pattern_cache.clear()

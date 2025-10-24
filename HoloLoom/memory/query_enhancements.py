"""
Advanced Query Enhancements
============================

Extends basic memory retrieval with:

1. **Faceted Search**: Filter by multiple dimensions
2. **Temporal Queries**: Time-based retrieval
3. **Domain Filtering**: Restrict to specific websites
4. **Importance Filtering**: Only high-quality content
5. **Multimodal Queries**: Search text + images
6. **Graph Traversal**: Follow relationships
7. **Aggregations**: Statistics and analytics

Query Types:
- Simple: "beekeeping"
- Temporal: "what did I learn last week?"
- Domain: "python docs only"
- Complex: "high-quality beekeeping articles from last month with images"
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TimeRange(Enum):
    """Predefined time ranges."""
    LAST_HOUR = "last_hour"
    LAST_DAY = "last_day"
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_YEAR = "last_year"
    THIS_WEEK = "this_week"
    THIS_MONTH = "this_month"
    THIS_YEAR = "this_year"


class SortOrder(Enum):
    """Sort order options."""
    RELEVANCE = "relevance"       # By semantic similarity
    RECENCY = "recency"           # By timestamp
    IMPORTANCE = "importance"     # By importance score
    POPULARITY = "popularity"     # By visit count
    DEPTH = "depth"               # By crawl depth


@dataclass
class QueryFilter:
    """Filter criteria for queries."""

    # Temporal filters
    after: Optional[datetime] = None
    before: Optional[datetime] = None
    time_range: Optional[TimeRange] = None

    # Domain filters
    domains: Optional[List[str]] = None      # Include these domains
    exclude_domains: Optional[List[str]] = None

    # Quality filters
    min_importance: Optional[float] = None   # 0-1
    max_crawl_depth: Optional[int] = None
    min_visit_duration: Optional[int] = None  # seconds

    # Content filters
    tags: Optional[List[str]] = None
    exclude_tags: Optional[List[str]] = None
    has_images: Optional[bool] = None
    min_content_length: Optional[int] = None
    max_content_length: Optional[int] = None

    # Source filters
    source_types: Optional[List[str]] = None  # webpage, document, youtube, etc.
    user_ids: Optional[List[str]] = None

    # Metadata filters
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryOptions:
    """Options for query execution."""

    # Results
    limit: int = 10
    offset: int = 0

    # Sorting
    sort_by: SortOrder = SortOrder.RELEVANCE
    ascending: bool = False

    # Deduplication
    deduplicate: bool = True
    similarity_threshold: float = 0.95

    # Highlighting
    highlight: bool = False
    highlight_tags: tuple = ('<mark>', '</mark>')

    # Expansion
    expand_context: bool = False  # Include surrounding chunks
    expand_links: bool = False    # Follow graph edges


@dataclass
class QueryResult:
    """Enhanced query result."""

    memories: List[Any]
    scores: List[float]
    total_count: int
    filtered_count: int

    # Metadata
    query_time_ms: float
    filters_applied: List[str]
    sort_order: str

    # Facets (aggregations)
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Suggestions
    related_queries: List[str] = field(default_factory=list)
    did_you_mean: Optional[str] = None


class AdvancedQueryEngine:
    """Engine for advanced query capabilities."""

    def __init__(self, memory_store):
        self.store = memory_store

    async def query(
        self,
        text: str,
        filters: Optional[QueryFilter] = None,
        options: Optional[QueryOptions] = None
    ) -> QueryResult:
        """
        Execute advanced query with filters and options.

        Args:
            text: Query text
            filters: Filter criteria
            options: Query options

        Returns:
            QueryResult with memories and metadata
        """
        import time
        start_time = time.time()

        filters = filters or QueryFilter()
        options = options or QueryOptions()

        # 1. Get candidate memories (basic retrieval)
        candidates = await self._get_candidates(text)

        # 2. Apply filters
        filtered = self._apply_filters(candidates, filters)

        # 3. Sort results
        sorted_results = self._sort_results(filtered, options.sort_by, options.ascending)

        # 4. Deduplicate if requested
        if options.deduplicate:
            sorted_results = self._deduplicate(sorted_results, options.similarity_threshold)

        # 5. Paginate
        total_count = len(sorted_results)
        paginated = sorted_results[options.offset:options.offset + options.limit]

        # 6. Expand context if requested
        if options.expand_context:
            paginated = await self._expand_context(paginated)

        # 7. Highlight if requested
        if options.highlight:
            paginated = self._highlight_results(paginated, text, options.highlight_tags)

        # 8. Calculate facets
        facets = self._calculate_facets(filtered)

        # 9. Generate suggestions
        related = self._generate_related_queries(text, filtered)

        query_time = (time.time() - start_time) * 1000

        memories = [item['memory'] for item in paginated]
        scores = [item['score'] for item in paginated]

        return QueryResult(
            memories=memories,
            scores=scores,
            total_count=total_count,
            filtered_count=len(filtered),
            query_time_ms=query_time,
            filters_applied=self._get_applied_filters(filters),
            sort_order=options.sort_by.value,
            facets=facets,
            related_queries=related
        )

    async def _get_candidates(self, text: str) -> List[Dict]:
        """Get candidate memories (basic retrieval)."""
        # This would integrate with actual memory store
        # For now, placeholder
        return []

    def _apply_filters(
        self,
        candidates: List[Dict],
        filters: QueryFilter
    ) -> List[Dict]:
        """Apply filters to candidate results."""

        results = candidates

        # Temporal filters
        if filters.time_range:
            start, end = self._resolve_time_range(filters.time_range)
            results = [
                r for r in results
                if start <= r['memory'].timestamp <= end
            ]
        elif filters.after or filters.before:
            if filters.after:
                results = [r for r in results if r['memory'].timestamp >= filters.after]
            if filters.before:
                results = [r for r in results if r['memory'].timestamp <= filters.before]

        # Domain filters
        if filters.domains:
            results = [
                r for r in results
                if any(domain in r['memory'].metadata.get('url', '') for domain in filters.domains)
            ]
        if filters.exclude_domains:
            results = [
                r for r in results
                if not any(domain in r['memory'].metadata.get('url', '') for domain in filters.exclude_domains)
            ]

        # Quality filters
        if filters.min_importance:
            results = [
                r for r in results
                if r['memory'].metadata.get('importance_score', 0) >= filters.min_importance
            ]
        if filters.max_crawl_depth is not None:
            results = [
                r for r in results
                if r['memory'].metadata.get('crawl_depth', 0) <= filters.max_crawl_depth
            ]
        if filters.min_visit_duration:
            results = [
                r for r in results
                if r['memory'].metadata.get('duration_seconds', 0) >= filters.min_visit_duration
            ]

        # Content filters
        if filters.tags:
            results = [
                r for r in results
                if any(tag in (r['memory'].tags or []) for tag in filters.tags)
            ]
        if filters.exclude_tags:
            results = [
                r for r in results
                if not any(tag in (r['memory'].tags or []) for tag in filters.exclude_tags)
            ]
        if filters.has_images is not None:
            if filters.has_images:
                results = [r for r in results if 'images' in r['memory'].metadata]
            else:
                results = [r for r in results if 'images' not in r['memory'].metadata]

        # Content length
        if filters.min_content_length:
            results = [r for r in results if len(r['memory'].text) >= filters.min_content_length]
        if filters.max_content_length:
            results = [r for r in results if len(r['memory'].text) <= filters.max_content_length]

        # Source type
        if filters.source_types:
            results = [
                r for r in results
                if r['memory'].metadata.get('content_type') in filters.source_types
            ]

        # User ID
        if filters.user_ids:
            results = [r for r in results if r['memory'].user_id in filters.user_ids]

        # Custom metadata filters
        for key, value in filters.metadata_filters.items():
            results = [r for r in results if r['memory'].metadata.get(key) == value]

        return results

    def _resolve_time_range(self, time_range: TimeRange) -> tuple:
        """Resolve time range enum to (start, end) datetimes."""
        now = datetime.now()

        if time_range == TimeRange.LAST_HOUR:
            return (now - timedelta(hours=1), now)
        elif time_range == TimeRange.LAST_DAY:
            return (now - timedelta(days=1), now)
        elif time_range == TimeRange.LAST_WEEK:
            return (now - timedelta(weeks=1), now)
        elif time_range == TimeRange.LAST_MONTH:
            return (now - timedelta(days=30), now)
        elif time_range == TimeRange.LAST_YEAR:
            return (now - timedelta(days=365), now)
        elif time_range == TimeRange.THIS_WEEK:
            start = now - timedelta(days=now.weekday())
            return (start.replace(hour=0, minute=0, second=0), now)
        elif time_range == TimeRange.THIS_MONTH:
            start = now.replace(day=1, hour=0, minute=0, second=0)
            return (start, now)
        elif time_range == TimeRange.THIS_YEAR:
            start = now.replace(month=1, day=1, hour=0, minute=0, second=0)
            return (start, now)

        return (now, now)

    def _sort_results(
        self,
        results: List[Dict],
        sort_by: SortOrder,
        ascending: bool
    ) -> List[Dict]:
        """Sort results by specified order."""

        if sort_by == SortOrder.RELEVANCE:
            # Already sorted by score
            key = lambda r: r['score']
        elif sort_by == SortOrder.RECENCY:
            key = lambda r: r['memory'].timestamp
        elif sort_by == SortOrder.IMPORTANCE:
            key = lambda r: r['memory'].metadata.get('importance_score', 0)
        elif sort_by == SortOrder.POPULARITY:
            key = lambda r: r['memory'].metadata.get('visit_count', 0)
        elif sort_by == SortOrder.DEPTH:
            key = lambda r: r['memory'].metadata.get('crawl_depth', 0)
        else:
            key = lambda r: r['score']

        return sorted(results, key=key, reverse=not ascending)

    def _deduplicate(
        self,
        results: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """Remove near-duplicate results."""
        # Simple deduplication by URL
        seen_urls = set()
        deduped = []

        for result in results:
            url = result['memory'].metadata.get('url', '')
            if url and url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append(result)

        return deduped

    async def _expand_context(self, results: List[Dict]) -> List[Dict]:
        """Expand results with surrounding context chunks."""
        # Would fetch adjacent chunks from same source
        return results

    def _highlight_results(
        self,
        results: List[Dict],
        query: str,
        tags: tuple
    ) -> List[Dict]:
        """Highlight query terms in results."""
        # Simple highlighting
        import re

        query_words = query.lower().split()
        open_tag, close_tag = tags

        for result in results:
            text = result['memory'].text

            for word in query_words:
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                text = pattern.sub(f"{open_tag}{word}{close_tag}", text)

            result['highlighted_text'] = text

        return results

    def _calculate_facets(self, results: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Calculate facet counts for filtering UI."""
        from collections import Counter

        facets = {}

        # Domain facet
        domains = [
            r['memory'].metadata.get('domain', 'unknown')
            for r in results
        ]
        facets['domains'] = dict(Counter(domains).most_common(10))

        # Tags facet
        all_tags = []
        for r in results:
            all_tags.extend(r['memory'].tags or [])
        facets['tags'] = dict(Counter(all_tags).most_common(10))

        # Content type facet
        types = [
            r['memory'].metadata.get('content_type', 'unknown')
            for r in results
        ]
        facets['content_types'] = dict(Counter(types))

        # Crawl depth facet
        depths = [
            r['memory'].metadata.get('crawl_depth', 0)
            for r in results
        ]
        facets['crawl_depths'] = dict(Counter(depths))

        return facets

    def _generate_related_queries(
        self,
        query: str,
        results: List[Dict]
    ) -> List[str]:
        """Generate related query suggestions."""
        # Extract common terms from top results
        # Simple version: just return variations
        return [
            f"{query} tutorial",
            f"{query} examples",
            f"advanced {query}",
        ]

    def _get_applied_filters(self, filters: QueryFilter) -> List[str]:
        """Get list of applied filter names."""
        applied = []

        if filters.time_range or filters.after or filters.before:
            applied.append('temporal')
        if filters.domains or filters.exclude_domains:
            applied.append('domain')
        if filters.min_importance or filters.max_crawl_depth:
            applied.append('quality')
        if filters.tags or filters.exclude_tags:
            applied.append('tags')
        if filters.has_images is not None:
            applied.append('multimodal')
        if filters.source_types:
            applied.append('source_type')

        return applied


# Convenience query builders
class QueryBuilder:
    """Fluent API for building complex queries."""

    def __init__(self):
        self.text = ""
        self.filters = QueryFilter()
        self.options = QueryOptions()

    def search(self, text: str) -> 'QueryBuilder':
        """Set query text."""
        self.text = text
        return self

    def last_week(self) -> 'QueryBuilder':
        """Filter to last week."""
        self.filters.time_range = TimeRange.LAST_WEEK
        return self

    def last_month(self) -> 'QueryBuilder':
        """Filter to last month."""
        self.filters.time_range = TimeRange.LAST_MONTH
        return self

    def from_domain(self, domain: str) -> 'QueryBuilder':
        """Filter to specific domain."""
        if not self.filters.domains:
            self.filters.domains = []
        self.filters.domains.append(domain)
        return self

    def with_tag(self, tag: str) -> 'QueryBuilder':
        """Filter to specific tag."""
        if not self.filters.tags:
            self.filters.tags = []
        self.filters.tags.append(tag)
        return self

    def with_images(self) -> 'QueryBuilder':
        """Filter to content with images."""
        self.filters.has_images = True
        return self

    def high_quality(self, min_score: float = 0.7) -> 'QueryBuilder':
        """Filter to high-quality content."""
        self.filters.min_importance = min_score
        return self

    def sort_by_recency(self) -> 'QueryBuilder':
        """Sort by recency."""
        self.options.sort_by = SortOrder.RECENCY
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        """Limit results."""
        self.options.limit = n
        return self

    def build(self) -> tuple:
        """Build final query."""
        return (self.text, self.filters, self.options)


# Example usage:
# query = (QueryBuilder()
#     .search("beekeeping")
#     .last_month()
#     .from_domain("wikipedia.org")
#     .with_images()
#     .high_quality(0.8)
#     .sort_by_recency()
#     .limit(20)
#     .build())

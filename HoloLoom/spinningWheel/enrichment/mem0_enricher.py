# -*- coding: utf-8 -*-
"""
Mem0 Enricher
=============

Memory-based context enrichment using mem0ai.

Provides:
- Similarity search in episodic memory
- Historical context retrieval
- Pattern matching across memory shards
- Temporal context from past interactions

Requirements:
    pip install mem0ai

Usage:
    from HoloLoom.spinningWheel.enrichment import Mem0Enricher

    enricher = Mem0Enricher(config={'api_key': 'your_key'})
    result = await enricher.enrich("Inspected hive today")
    # Returns similar memories and context

Created: 2025-10-26
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from .base import BaseEnricher, EnrichmentResult


class Mem0Enricher(BaseEnricher):
    """
    Enrich text with memory-based context from mem0ai.

    Capabilities:
    - Search similar past interactions
    - Retrieve relevant historical context
    - Find patterns across episodes
    - Provide temporal continuity

    Example:
        >>> enricher = Mem0Enricher(config={'api_key': 'key'})
        >>> result = await enricher.enrich("Checked hive health today")
        >>> result.data
        {
            'similar_memories': [
                {'text': 'Hive inspection last week', 'score': 0.89},
                {'text': 'Previous health check', 'score': 0.76}
            ],
            'patterns': ['regular_inspection', 'health_monitoring'],
            'temporal_context': 'Last similar activity 7 days ago',
            'relevant_entities': ['hive', 'health', 'inspection']
        }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.api_key = self.config.get('api_key')
        self.user_id = self.config.get('user_id', 'default_user')
        self.search_limit = self.config.get('search_limit', 5)
        self.min_score = self.config.get('min_score', 0.5)

        # Connection
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initialize mem0ai client."""
        if not self.api_key:
            # No API key - run in mock mode
            return

        try:
            from mem0 import Memory
            self.client = Memory.from_config({
                'api_key': self.api_key
            })

        except ImportError:
            print("Warning: mem0ai package not installed. Install with: pip install mem0ai")
            self.client = None
        except Exception as e:
            print(f"Warning: Could not initialize mem0ai client: {e}")
            self.client = None

    async def enrich(self, text: str) -> EnrichmentResult:
        """
        Enrich text with memory-based context.

        Args:
            text: Input text to enrich

        Returns:
            EnrichmentResult with memory context
        """
        if not self.client:
            # Mock mode - return simulated enrichment
            return self._mock_enrichment(text)

        try:
            # Search for similar memories
            search_results = self.client.search(
                query=text,
                user_id=self.user_id,
                limit=self.search_limit
            )

            # Filter by minimum score
            similar_memories = [
                {
                    'text': result.get('memory', ''),
                    'score': result.get('score', 0.0),
                    'id': result.get('id', ''),
                    'metadata': result.get('metadata', {})
                }
                for result in search_results
                if result.get('score', 0.0) >= self.min_score
            ]

            # Extract patterns
            patterns = self._extract_patterns(similar_memories)

            # Build temporal context
            temporal_context = self._build_temporal_context(similar_memories)

            # Extract relevant entities from memories
            relevant_entities = self._extract_entities_from_memories(similar_memories)

            return EnrichmentResult(
                enricher_type='mem0',
                data={
                    'similar_memories': similar_memories,
                    'patterns': patterns,
                    'temporal_context': temporal_context,
                    'relevant_entities': relevant_entities,
                    'memory_count': len(similar_memories)
                },
                confidence=1.0 if similar_memories else 0.0,
                metadata={
                    'user_id': self.user_id,
                    'search_limit': self.search_limit,
                    'min_score': self.min_score
                }
            )

        except Exception as e:
            # Error in mem0 API - return error enrichment
            return EnrichmentResult(
                enricher_type='mem0',
                data={
                    'error': str(e),
                    'similar_memories': [],
                    'patterns': [],
                    'temporal_context': 'Error retrieving memories',
                    'relevant_entities': []
                },
                confidence=0.0,
                metadata={'error_type': type(e).__name__}
            )

    def _mock_enrichment(self, text: str) -> EnrichmentResult:
        """Return mock enrichment for testing without API key."""
        import re

        # Extract simple keywords
        keywords = re.findall(r'\b[a-z]{4,}\b', text.lower())

        return EnrichmentResult(
            enricher_type='mem0',
            data={
                'similar_memories': [],
                'patterns': keywords[:3],
                'temporal_context': 'Mock mode - no historical data available',
                'relevant_entities': keywords[:5],
                'mock_mode': True
            },
            confidence=0.0,
            metadata={'status': 'mock_mode'}
        )

    def _extract_patterns(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        Extract recurring patterns from similar memories.

        Looks for:
        - Common keywords across memories
        - Recurring themes
        - Activity patterns
        """
        import re
        from collections import Counter

        # Collect all text
        all_text = ' '.join([m['text'].lower() for m in memories])

        # Extract words (simple pattern detection)
        words = re.findall(r'\b[a-z]{4,}\b', all_text)

        # Count frequency
        word_counts = Counter(words)

        # Return top patterns (words appearing 2+ times)
        patterns = [word for word, count in word_counts.most_common(10) if count >= 2]

        return patterns[:5]  # Top 5 patterns

    def _build_temporal_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build temporal context from memories."""
        if not memories:
            return "No historical context available"

        # Check if memories have timestamps
        timestamped = [m for m in memories if 'metadata' in m and 'timestamp' in m['metadata']]

        if not timestamped:
            return f"Found {len(memories)} similar memories (no temporal data)"

        # Calculate time since most recent
        try:
            latest = max(timestamped, key=lambda m: m['metadata']['timestamp'])
            timestamp = latest['metadata']['timestamp']

            # Parse timestamp (assumes ISO format)
            if isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                days_ago = (datetime.now() - dt).days

                if days_ago == 0:
                    return "Similar activity today"
                elif days_ago == 1:
                    return "Similar activity yesterday"
                else:
                    return f"Last similar activity {days_ago} days ago"

        except Exception:
            pass

        return f"Found {len(memories)} similar memories"

    def _extract_entities_from_memories(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract entities mentioned across memories."""
        import re

        entities = []

        for memory in memories:
            text = memory.get('text', '')

            # Extract capitalized words (simple entity detection)
            caps = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities.extend(caps)

        # Deduplicate and limit
        return list(set(entities))[:10]

    async def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new memory to mem0.

        Args:
            text: Memory text to store
            metadata: Optional metadata to attach
        """
        if not self.client:
            return

        try:
            self.client.add(
                messages=[{"role": "user", "content": text}],
                user_id=self.user_id,
                metadata=metadata or {}
            )
        except Exception as e:
            print(f"Warning: Could not add memory to mem0: {e}")


# Convenience function
async def enrich_with_mem0(
    text: str,
    api_key: Optional[str] = None,
    user_id: str = "default_user",
    search_limit: int = 5
) -> EnrichmentResult:
    """
    Quick function to enrich text with mem0 context.

    Args:
        text: Input text
        api_key: mem0ai API key (None = mock mode)
        user_id: User identifier for memory retrieval
        search_limit: Maximum memories to retrieve

    Returns:
        EnrichmentResult with memory context

    Example:
        result = await enrich_with_mem0(
            "Inspected hive today",
            api_key="your_key",
            user_id="beekeeper_01"
        )
    """
    enricher = Mem0Enricher({
        'api_key': api_key,
        'user_id': user_id,
        'search_limit': search_limit
    })

    return await enricher.enrich(text)

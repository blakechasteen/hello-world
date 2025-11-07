"""
Narrative Mode Writer
=====================

Generates narrative explanations from memory context.

Philosophy: Transform knowledge into story.
"""

import logging
from typing import List, Dict, Any, Optional

from ..core.protocol import (
    ModeWriterProtocol,
    WritingContext,
    WritingMode,
    StyleGuide,
    MemoryShard
)

logger = logging.getLogger(__name__)


class NarrativeWriter:
    """
    Narrative mode writer - transforms memory into explanation.

    Approach:
    1. Temporal ordering (if temporal metadata exists)
    2. Entity/character tracking
    3. Story arc (setup → development → conclusion)
    4. Causal connections (because, therefore, as a result)
    """

    def __init__(self):
        self.logger = logger

    async def generate(self, context: WritingContext) -> str:
        """
        Generate narrative draft from context.

        Args:
            context: Writing context with query and memories

        Returns:
            Narrative draft
        """
        if not context.memories:
            return self._generate_empty_response(context.query)

        # 1. Analyze memories
        analysis = self._analyze_memories(context.memories)

        # 2. Extract entities (characters in our story)
        entities = analysis.get('entities', [])

        # 3. Order memories (temporal or relevance)
        ordered_memories = self._order_memories(context.memories)

        # 4. Generate narrative structure
        narrative = self._generate_narrative_structure(
            query=context.query,
            memories=ordered_memories,
            entities=entities,
            style=context.style
        )

        return narrative

    def _analyze_memories(self, memories: List[MemoryShard]) -> Dict[str, Any]:
        """
        Analyze memories to extract structural elements.

        Args:
            memories: Memory shards

        Returns:
            Analysis dict with entities, themes, etc.
        """
        entities = set()
        themes = set()

        for memory in memories:
            # Extract entities from metadata
            if 'entities' in memory.metadata:
                entities.update(memory.metadata['entities'])

            # Extract themes/topics
            if 'topics' in memory.metadata:
                themes.update(memory.metadata['topics'])

            # Simple entity extraction from content (capitalized words)
            words = memory.text.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 2:
                    # Likely a proper noun
                    entities.add(word.strip('.,;:'))

        return {
            'entities': list(entities)[:10],  # Top 10
            'themes': list(themes)[:5],       # Top 5
            'memory_count': len(memories)
        }

    def _order_memories(self, memories: List[MemoryShard]) -> List[MemoryShard]:
        """
        Order memories for narrative flow.

        Args:
            memories: Unordered memories

        Returns:
            Ordered memories (temporal or relevance-based)
        """
        # Check if memories have temporal metadata
        has_timestamps = all('timestamp' in m.metadata for m in memories)

        if has_timestamps:
            # Temporal ordering
            return sorted(memories, key=lambda m: m.metadata.get('timestamp', 0))
        else:
            # Relevance ordering (high to low)
            return sorted(
                memories,
                key=lambda m: m.metadata.get('relevance', 0.5),
                reverse=True
            )

    def _generate_narrative_structure(
        self,
        query: str,
        memories: List[MemoryShard],
        entities: List[str],
        style: StyleGuide
    ) -> str:
        """
        Generate narrative with setup → development → conclusion.

        Args:
            query: User query
            memories: Ordered memories
            entities: Key entities
            style: Style guide

        Returns:
            Narrative draft
        """
        sections = []

        # SETUP: Introduce the topic and context
        setup = self._generate_setup(query, entities, style)
        sections.append(setup)

        # DEVELOPMENT: Present information from memories
        development = self._generate_development(memories, entities, style)
        sections.append(development)

        # CONCLUSION: Synthesize and wrap up
        conclusion = self._generate_conclusion(query, memories, style)
        sections.append(conclusion)

        # Join sections
        return '\n\n'.join(sections)

    def _generate_setup(
        self,
        query: str,
        entities: List[str],
        style: StyleGuide
    ) -> str:
        """Generate setup/introduction."""
        # Different styles have different openings
        if style == StyleGuide.CASUAL:
            # Casual, direct opening
            return f"Let's talk about {query.lower()}."

        elif style == StyleGuide.ACADEMIC:
            # Formal, academic opening
            intro = f"This document addresses the following question: {query}"
            if entities:
                intro += f" Key concepts include: {', '.join(entities[:3])}."
            return intro

        elif style == StyleGuide.TUFTE:
            # Clarity-first, minimal opening
            return query + "\n"

        else:
            # Default: balanced opening
            return f"To understand {query.lower()}, let's explore the key concepts."

    def _generate_development(
        self,
        memories: List[MemoryShard],
        entities: List[str],
        style: StyleGuide
    ) -> str:
        """Generate development/body with causal connections."""
        paragraphs = []

        # Group memories into logical chunks (2-3 per paragraph)
        chunk_size = 2 if len(memories) > 6 else 1

        for i in range(0, len(memories), chunk_size):
            chunk = memories[i:i+chunk_size]

            # Build paragraph from chunk
            paragraph_parts = []

            for j, memory in enumerate(chunk):
                content = memory.text.strip()

                # Add causal connectors
                if j == 0:
                    # First sentence - no connector
                    paragraph_parts.append(content)
                elif j == 1:
                    # Second sentence - add connection
                    connectors = [
                        'Furthermore',
                        'Additionally',
                        'Building on this',
                        'This relates to'
                    ]
                    connector = connectors[i % len(connectors)]
                    paragraph_parts.append(f"{connector}, {content.lower()}")
                else:
                    # Additional sentences
                    paragraph_parts.append(f"Moreover, {content.lower()}")

            paragraph = ' '.join(paragraph_parts)
            paragraphs.append(paragraph)

        return '\n\n'.join(paragraphs)

    def _generate_conclusion(
        self,
        query: str,
        memories: List[MemoryShard],
        style: StyleGuide
    ) -> str:
        """Generate conclusion/synthesis."""
        if not memories:
            return "Further information is needed to fully address this topic."

        # Extract key insight from highest-relevance memory
        top_memory = max(
            memories,
            key=lambda m: m.metadata.get('relevance', 0.5)
        )

        key_insight = top_memory.text.strip()

        if style == StyleGuide.CASUAL:
            # Casual, conversational close
            return f"So in short: {key_insight}"

        elif style == StyleGuide.ACADEMIC:
            # Formal conclusion
            return f"In conclusion, {key_insight.lower()} This addresses the question of {query.lower()}"

        elif style == StyleGuide.TUFTE:
            # Minimal, data-focused close
            return f"Summary: {key_insight}"

        else:
            # Default: synthesize
            return f"In essence, {key_insight.lower()} This provides a foundation for understanding {query.lower()}"

    def _generate_empty_response(self, query: str) -> str:
        """Generate response when no memories available."""
        return (
            f"# {query}\n\n"
            f"No relevant information was found to address this query. "
            f"Additional context or information would be helpful."
        )

    @property
    def mode(self) -> WritingMode:
        """Writing mode this writer handles."""
        return WritingMode.NARRATIVE

    @property
    def default_style(self) -> StyleGuide:
        """Default style guide for narrative mode."""
        return StyleGuide.TUFTE  # Clarity-first

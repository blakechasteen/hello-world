# -*- coding: utf-8 -*-
"""
Text Spinner
=============
Input adapter for ingesting plain text documents as MemoryShards.

Converts text documents (plain text, markdown, notes) into structured
memory shards that can be processed by the HoloLoom orchestrator.

Design Philosophy:
- Thin adapter that normalizes text data → MemoryShards
- Handles multiple text formats (plain text, markdown, paragraphs)
- Optional paragraph/sentence chunking for long documents
- Optional enrichment for entity/motif extraction

Usage:
    from HoloLoom.spinningWheel.text import TextSpinner, TextSpinnerConfig

    config = TextSpinnerConfig(
        chunk_by='paragraph',  # or 'sentence', 'character', None
        chunk_size=500,        # characters per chunk
        enable_enrichment=True
    )

    spinner = TextSpinner(config)
    shards = await spinner.spin({
        'text': "Your document content here...",
        'source': 'notes.md',
        'metadata': {'author': 'user', 'date': '2025-10-23'}
    })
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import BaseSpinner, SpinnerConfig

# Import HoloLoom types
try:
    from HoloLoom.Documentation.types import MemoryShard
except ImportError:
    # Fallback if types not available
    from dataclasses import dataclass, field

    @dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextSpinnerConfig(SpinnerConfig):
    """
    Configuration for Text spinner.

    Attributes:
        chunk_by: How to split text ('paragraph', 'sentence', 'character', None)
                 If None, creates one shard per document
        chunk_size: Approximate size for chunks (characters)
        min_chunk_size: Minimum chunk size to avoid tiny fragments
        extract_entities: Extract basic entities from text
        preserve_structure: Try to preserve markdown/document structure
    """
    chunk_by: Optional[str] = None  # None = single shard per document
    chunk_size: int = 500  # characters
    min_chunk_size: int = 50  # minimum chunk size
    extract_entities: bool = True
    preserve_structure: bool = True


class TextSpinner(BaseSpinner):
    """
    Spinner for plain text documents.

    Converts text documents into MemoryShards suitable for
    HoloLoom processing.
    """

    def __init__(self, config: TextSpinnerConfig = None):
        if config is None:
            config = TextSpinnerConfig()
        super().__init__(config)
        self.config: TextSpinnerConfig = config

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert text document → MemoryShards.

        Args:
            raw_data: Dict with keys:
                - 'text': The text content (required)
                - 'source': Source identifier (optional, e.g., filename)
                - 'episode': Episode/session identifier (optional)
                - 'metadata': Additional metadata to include (optional)

        Returns:
            List of MemoryShard objects
        """
        # Extract text
        text = raw_data.get('text')
        if not text:
            raise ValueError("'text' is required in raw_data")

        # Determine source/episode identifiers
        source = raw_data.get('source', 'text_document')
        episode = raw_data.get('episode', f"text_{source}")

        # Create shards
        if self.config.chunk_by:
            # Split into chunks
            shards = self._create_chunked_shards(text, source, episode, raw_data)
        else:
            # Single shard for entire document
            shards = self._create_single_shard(text, source, episode, raw_data)

        # Optional enrichment
        if self.config.enable_enrichment:
            shards = await self._enrich_shards(shards)

        return shards

    def _create_single_shard(
        self,
        text: str,
        source: str,
        episode: str,
        raw_data: Dict[str, Any]
    ) -> List[MemoryShard]:
        """Create a single shard for the entire document."""

        # Extract basic entities if configured
        entities = []
        if self.config.extract_entities:
            entities = self._extract_basic_entities(text)

        # Build metadata
        metadata = {
            'source': source,
            'format': 'text',
            'char_count': len(text),
            'word_count': len(text.split()),
        }

        # Merge additional metadata from raw_data
        if 'metadata' in raw_data:
            metadata.update(raw_data['metadata'])

        shard = MemoryShard(
            id=f"{episode}_full",
            text=text,
            episode=episode,
            entities=entities,
            motifs=[],  # Will be populated by orchestrator's motif detector
            metadata=metadata
        )

        return [shard]

    def _create_chunked_shards(
        self,
        text: str,
        source: str,
        episode: str,
        raw_data: Dict[str, Any]
    ) -> List[MemoryShard]:
        """Split text into chunks based on configuration."""

        # Split text into chunks
        if self.config.chunk_by == 'paragraph':
            chunks = self._chunk_by_paragraph(text)
        elif self.config.chunk_by == 'sentence':
            chunks = self._chunk_by_sentence(text)
        elif self.config.chunk_by == 'character':
            chunks = self._chunk_by_character(text)
        else:
            raise ValueError(f"Unknown chunk_by mode: {self.config.chunk_by}")

        # Create shards from chunks
        shards = []
        for idx, chunk_text in enumerate(chunks):
            # Skip very small chunks
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            # Extract basic entities if configured
            entities = []
            if self.config.extract_entities:
                entities = self._extract_basic_entities(chunk_text)

            # Build metadata
            metadata = {
                'source': source,
                'format': 'text',
                'chunk_index': idx,
                'chunk_by': self.config.chunk_by,
                'char_count': len(chunk_text),
                'word_count': len(chunk_text.split()),
            }

            # Merge additional metadata from raw_data
            if 'metadata' in raw_data:
                metadata.update(raw_data['metadata'])

            shard = MemoryShard(
                id=f"{episode}_chunk_{idx:03d}",
                text=chunk_text,
                episode=episode,
                entities=entities,
                motifs=[],
                metadata=metadata
            )
            shards.append(shard)

        return shards

    def _chunk_by_paragraph(self, text: str) -> List[str]:
        """Split text into paragraph-based chunks."""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If single paragraph exceeds chunk_size, add it as its own chunk
            if para_size > self.config.chunk_size:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                chunks.append(para)
            # If adding paragraph would exceed chunk_size, start new chunk
            elif current_size + para_size > self.config.chunk_size and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            # Otherwise, add to current chunk
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[str]:
        """Split text into sentence-based chunks."""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sent_size = len(sentence)

            # If adding sentence would exceed chunk_size, start new chunk
            if current_size + sent_size > self.config.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sent_size
            # Otherwise, add to current chunk
            else:
                current_chunk.append(sentence)
                current_size += sent_size

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _chunk_by_character(self, text: str) -> List[str]:
        """Split text into fixed-size character chunks."""
        chunks = []
        chunk_size = self.config.chunk_size

        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)

        return chunks

    def _extract_basic_entities(self, text: str) -> List[str]:
        """
        Extract basic entities using simple heuristics.

        More sophisticated extraction happens during enrichment or
        in the orchestrator's motif detection phase.
        """
        entities = []

        # Extract capitalized words (potential proper nouns)
        # Skip common words
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'I', 'You', 'We', 'They',
            'A', 'An', 'It', 'He', 'She', 'What', 'When', 'Where', 'Why', 'How'
        }
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        for word in words:
            if word not in common_words and len(word) > 2:
                entities.append(word)

        # Deduplicate and limit
        entities = list(set(entities))[:20]  # Keep top 20 unique entities

        return entities

    async def _enrich_shards(self, shards: List[MemoryShard]) -> List[MemoryShard]:
        """
        Enrich shards using optional enrichment services.
        Delegates to parent class enrichment infrastructure.
        """
        for shard in shards:
            enrichment = await self.enrich(shard.text)

            # Merge enrichment results into shard
            if enrichment:
                # Update entities from enrichment
                if 'ollama' in enrichment and 'entities' in enrichment['ollama']:
                    shard.entities.extend(enrichment['ollama']['entities'])
                    shard.entities = list(set(shard.entities))[:30]  # Dedupe and limit

                # Update motifs from enrichment
                if 'ollama' in enrichment and 'motifs' in enrichment['ollama']:
                    shard.motifs.extend(enrichment['ollama']['motifs'])
                    shard.motifs = list(set(shard.motifs))

                # Store enrichment in metadata
                if shard.metadata is None:
                    shard.metadata = {}
                shard.metadata['enrichment'] = enrichment

        return shards


# Convenience function for quick usage
async def spin_text(
    text: str,
    source: str = 'document',
    chunk_by: Optional[str] = None,
    chunk_size: int = 500,
    enable_enrichment: bool = False
) -> List[MemoryShard]:
    """
    Quick function to convert text into MemoryShards.

    Args:
        text: The text content
        source: Source identifier (default: 'document')
        chunk_by: How to chunk ('paragraph', 'sentence', 'character', None)
        chunk_size: Size of chunks in characters (default: 500)
        enable_enrichment: Enable Ollama enrichment (default: False)

    Returns:
        List of MemoryShard objects
    """
    config = TextSpinnerConfig(
        chunk_by=chunk_by,
        chunk_size=chunk_size,
        enable_enrichment=enable_enrichment
    )

    spinner = TextSpinner(config)

    raw_data = {
        'text': text,
        'source': source
    }

    return await spinner.spin(raw_data)

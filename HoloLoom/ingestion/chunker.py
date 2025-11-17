"""
Smart Chunker - Semantic Text Chunking
=======================================
Intelligent text chunking that respects semantic boundaries.

Philosophy:
Naive chunking breaks text at arbitrary character limits, destroying context.
Smart chunking preserves semantic units (sentences, paragraphs) and adds
overlap for continuity, ensuring embeddings capture complete thoughts.
"""

import re
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking."""
    chunk_size: int = 512  # Target chunk size in characters
    overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    respect_sentences: bool = True  # Don't split mid-sentence
    respect_paragraphs: bool = True  # Prefer paragraph boundaries


@dataclass
class Chunk:
    """Text chunk with metadata."""
    text: str
    start_idx: int
    end_idx: int
    chunk_id: int
    metadata: Dict[str, Any]


class SmartChunker:
    """
    Smart text chunker with semantic awareness.

    Chunks text while preserving semantic boundaries and adding overlap
    for context continuity.

    Usage:
        chunker = SmartChunker(ChunkConfig(chunk_size=512, overlap=50))
        chunks = chunker.chunk(long_text)

        for chunk in chunks:
            print(f"Chunk {chunk.chunk_id}: {len(chunk.text)} chars")
            print(chunk.text[:100] + "...")
    """

    def __init__(self, config: ChunkConfig):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        self.config = config
        logger.info(f"SmartChunker initialized (size={config.chunk_size}, overlap={config.overlap})")

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk text into semantic units.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            List of chunks
        """
        if not text or len(text) < self.config.min_chunk_size:
            # Text too small, return as single chunk
            return [Chunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                chunk_id=0,
                metadata=metadata or {}
            )]

        # Split into semantic units
        if self.config.respect_paragraphs:
            units = self._split_paragraphs(text)
        elif self.config.respect_sentences:
            units = self._split_sentences(text)
        else:
            # Character-based chunking
            units = [text[i:i+self.config.chunk_size]
                    for i in range(0, len(text), self.config.chunk_size)]

        # Merge units into chunks
        chunks = self._merge_units(units, metadata or {})

        logger.info(f"Chunked text: {len(text)} chars → {len(chunks)} chunks")

        return chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_units(self, units: List[str], metadata: Dict[str, Any]) -> List[Chunk]:
        """Merge semantic units into chunks."""
        chunks = []
        current_chunk = []
        current_length = 0
        char_idx = 0

        for unit in units:
            unit_length = len(unit)

            # Check if adding this unit would exceed chunk size
            if current_length + unit_length > self.config.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=char_idx - current_length,
                    end_idx=char_idx,
                    chunk_id=len(chunks),
                    metadata=metadata.copy()
                ))

                # Start new chunk with overlap
                if self.config.overlap > 0:
                    # Include last unit for overlap
                    current_chunk = [current_chunk[-1], unit]
                    current_length = len(current_chunk[-1]) + unit_length
                else:
                    current_chunk = [unit]
                    current_length = unit_length
            else:
                # Add unit to current chunk
                current_chunk.append(unit)
                current_length += unit_length + 1  # +1 for space

            char_idx += unit_length

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_idx=char_idx - current_length,
                end_idx=char_idx,
                chunk_id=len(chunks),
                metadata=metadata.copy()
            ))

        return chunks


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Smart Chunker Demo")
    print("="*80 + "\n")

    # Sample text
    text = """
    Thompson Sampling is a Bayesian approach to the multi-armed bandit problem.
    It naturally balances exploration and exploitation by sampling from posterior distributions.

    The algorithm was introduced by William R. Thompson in 1933.
    Despite its age, it remains one of the most effective algorithms for online decision-making.

    In Thompson Sampling, we maintain a probability distribution over the reward parameter for each arm.
    At each step, we sample from these distributions and select the arm with the highest sample.
    After observing a reward, we update the posterior distribution using Bayes' rule.

    This approach has several advantages over epsilon-greedy and UCB algorithms.
    It's naturally adaptive and doesn't require tuning of exploration parameters.
    """

    # Create chunker
    config = ChunkConfig(
        chunk_size=200,
        overlap=30,
        respect_sentences=True
    )
    chunker = SmartChunker(config)

    # Chunk text
    chunks = chunker.chunk(text, metadata={"source": "demo", "topic": "Thompson Sampling"})

    print(f"Original text: {len(text)} characters")
    print(f"Generated {len(chunks)} chunks:\n")

    for chunk in chunks:
        print(f"Chunk {chunk.chunk_id}:")
        print(f"  Length: {len(chunk.text)} chars")
        print(f"  Range: [{chunk.start_idx}, {chunk.end_idx})")
        print(f"  Text: {chunk.text[:100]}...")
        print()

    print("✓ Demo complete!")

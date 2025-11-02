"""
Text Summarization Module
==========================

Generates concise summaries of raw notes while preserving key information.

Supports multiple summarization strategies:
- Extractive: Select most important sentences from original text
- Abstractive: Generate new summary (requires LLM)
- Hybrid: Combine both approaches

Design Philosophy:
- Original text always preserved
- Summary stored as additional metadata
- Configurable summary length (by sentences, words, or compression ratio)
- Optional entity/measurement preservation in summary
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SummarizationStrategy(Enum):
    """Summarization approaches."""
    EXTRACTIVE = "extractive"  # Select key sentences
    ABSTRACTIVE = "abstractive"  # Generate new text (LLM)
    HYBRID = "hybrid"  # Combine both
    FIRST_N_SENTENCES = "first_n"  # Simple: first N sentences


@dataclass
class SummarizerConfig:
    """
    Configuration for text summarization.

    Attributes:
        strategy: Summarization approach to use
        max_sentences: Maximum sentences in summary
        max_words: Maximum words in summary (alternative to sentences)
        compression_ratio: Target compression (0.0-1.0, e.g., 0.3 = 30% of original)
        preserve_entities: Try to keep sentences mentioning entities
        preserve_measurements: Try to keep sentences with measurements
        min_sentence_length: Ignore very short sentences
    """
    strategy: SummarizationStrategy = SummarizationStrategy.EXTRACTIVE
    max_sentences: Optional[int] = 3
    max_words: Optional[int] = None
    compression_ratio: Optional[float] = None  # 0.0-1.0
    preserve_entities: bool = True
    preserve_measurements: bool = True
    min_sentence_length: int = 10  # characters


class TextSummarizer:
    """
    Generate summaries of text while preserving key information.

    Works with ExpertLoom's entity extraction to ensure important
    facts aren't lost in summarization.
    """

    def __init__(self, config: SummarizerConfig = None):
        if config is None:
            config = SummarizerConfig()
        self.config = config

    def summarize(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        measurements: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Summarize text while preserving key information.

        Args:
            text: Original text to summarize
            entities: List of entity IDs/names mentioned in text
            measurements: Dict of extracted measurements

        Returns:
            Dict with:
                - summary: The summarized text
                - sentences_used: List of sentence indices used
                - compression: Actual compression ratio achieved
                - preserved_entities: Entities mentioned in summary
                - preserved_measurements: Measurements in summary
        """
        # Split into sentences
        sentences = self._split_sentences(text)

        if not sentences:
            return {
                "summary": text,
                "sentences_used": [],
                "compression": 1.0,
                "preserved_entities": entities or [],
                "preserved_measurements": measurements or {}
            }

        # Apply strategy
        if self.config.strategy == SummarizationStrategy.FIRST_N_SENTENCES:
            summary_sentences = self._first_n_sentences(sentences)
        elif self.config.strategy == SummarizationStrategy.EXTRACTIVE:
            summary_sentences = self._extractive_summary(
                sentences, text, entities, measurements
            )
        elif self.config.strategy == SummarizationStrategy.ABSTRACTIVE:
            # Would need LLM - fallback to extractive for now
            summary_sentences = self._extractive_summary(
                sentences, text, entities, measurements
            )
        else:
            # Hybrid or unknown - default to extractive
            summary_sentences = self._extractive_summary(
                sentences, text, entities, measurements
            )

        # Combine summary sentences
        summary_text = ' '.join(summary_sentences)

        # Calculate compression
        original_length = len(text)
        summary_length = len(summary_text)
        compression = summary_length / original_length if original_length > 0 else 1.0

        # Check what was preserved
        preserved_entities = []
        if entities:
            for entity in entities:
                if entity.lower() in summary_text.lower():
                    preserved_entities.append(entity)

        preserved_measurements = {}
        if measurements:
            for key, value in measurements.items():
                # Check if measurement value appears in summary
                if str(value).lower() in summary_text.lower():
                    preserved_measurements[key] = value

        return {
            "summary": summary_text,
            "sentences_used": summary_sentences,
            "compression": round(compression, 2),
            "preserved_entities": preserved_entities,
            "preserved_measurements": preserved_measurements,
            "original_length": original_length,
            "summary_length": summary_length
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Filter very short sentences
        sentences = [
            s.strip()
            for s in sentences
            if len(s.strip()) >= self.config.min_sentence_length
        ]

        return sentences

    def _first_n_sentences(self, sentences: List[str]) -> List[str]:
        """Simple strategy: return first N sentences."""
        n = self.config.max_sentences or 3
        return sentences[:n]

    def _extractive_summary(
        self,
        sentences: List[str],
        full_text: str,
        entities: Optional[List[str]] = None,
        measurements: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Extractive summarization: Select most important sentences.

        Scoring based on:
        1. Position (first sentences often important)
        2. Length (very short/long sentences penalized)
        3. Entity mentions (sentences with entities prioritized)
        4. Measurement mentions (sentences with numbers/measurements prioritized)
        5. Keyword density (important terms)
        """
        if not sentences:
            return []

        # Score each sentence
        scored_sentences = []

        for idx, sentence in enumerate(sentences):
            score = 0.0

            # 1. Position score (first and last sentences important)
            position_score = 1.0 / (idx + 1)  # First sentence gets 1.0, second 0.5, etc.
            if idx == len(sentences) - 1:  # Last sentence bonus
                position_score += 0.3
            score += position_score * 0.3  # Weight: 30%

            # 2. Length score (prefer medium-length sentences)
            sentence_len = len(sentence.split())
            if 5 <= sentence_len <= 25:
                length_score = 1.0
            elif sentence_len < 5:
                length_score = 0.3
            else:
                length_score = 0.7
            score += length_score * 0.1  # Weight: 10%

            # 3. Entity score
            entity_score = 0.0
            if self.config.preserve_entities and entities:
                for entity in entities:
                    if entity.lower() in sentence.lower():
                        entity_score += 1.0
                entity_score = min(entity_score, 3.0) / 3.0  # Normalize to 0-1
            score += entity_score * 0.3  # Weight: 30%

            # 4. Measurement score
            measurement_score = 0.0
            if self.config.preserve_measurements and measurements:
                for value in measurements.values():
                    if str(value).lower() in sentence.lower():
                        measurement_score += 1.0
                measurement_score = min(measurement_score, 3.0) / 3.0
            score += measurement_score * 0.2  # Weight: 20%

            # 5. Keyword density (numbers, important words)
            keyword_score = self._calculate_keyword_score(sentence)
            score += keyword_score * 0.1  # Weight: 10%

            scored_sentences.append((sentence, score, idx))

        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Determine how many sentences to keep
        if self.config.max_sentences:
            num_to_keep = min(self.config.max_sentences, len(scored_sentences))
        elif self.config.compression_ratio:
            num_to_keep = max(1, int(len(sentences) * self.config.compression_ratio))
        elif self.config.max_words:
            # Keep adding sentences until we hit word limit
            num_to_keep = self._count_sentences_for_words(
                scored_sentences, self.config.max_words
            )
        else:
            num_to_keep = min(3, len(scored_sentences))  # Default: 3 sentences

        # Take top N sentences
        selected = scored_sentences[:num_to_keep]

        # Re-order by original position to maintain narrative flow
        selected.sort(key=lambda x: x[2])

        return [s[0] for s in selected]

    def _calculate_keyword_score(self, sentence: str) -> float:
        """Calculate score based on keyword presence."""
        score = 0.0

        # Numbers/measurements
        if re.search(r'\d+', sentence):
            score += 0.5

        # Action verbs (common in important sentences)
        action_verbs = ['checked', 'found', 'replaced', 'measured', 'observed',
                       'noticed', 'fixed', 'added', 'removed', 'changed']
        for verb in action_verbs:
            if verb in sentence.lower():
                score += 0.2

        # Temporal markers (indicate events)
        temporal = ['today', 'yesterday', 'morning', 'afternoon', 'week', 'month']
        for marker in temporal:
            if marker in sentence.lower():
                score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _count_sentences_for_words(
        self,
        scored_sentences: List[tuple],
        max_words: int
    ) -> int:
        """Count how many sentences fit within word limit."""
        total_words = 0
        count = 0

        for sentence, score, idx in scored_sentences:
            sentence_words = len(sentence.split())
            if total_words + sentence_words <= max_words:
                total_words += sentence_words
                count += 1
            else:
                break

        return max(1, count)  # At least 1 sentence


def summarize_text(
    text: str,
    entities: Optional[List[str]] = None,
    measurements: Optional[Dict[str, Any]] = None,
    max_sentences: int = 3,
    strategy: str = "extractive"
) -> str:
    """
    Convenience function for quick summarization.

    Args:
        text: Text to summarize
        entities: Optional list of entities to preserve
        measurements: Optional dict of measurements to preserve
        max_sentences: Maximum sentences in summary (default: 3)
        strategy: Summarization strategy ('extractive', 'first_n')

    Returns:
        Summary string
    """
    config = SummarizerConfig(
        strategy=SummarizationStrategy(strategy),
        max_sentences=max_sentences,
        preserve_entities=bool(entities),
        preserve_measurements=bool(measurements)
    )

    summarizer = TextSummarizer(config)
    result = summarizer.summarize(text, entities, measurements)

    return result["summary"]

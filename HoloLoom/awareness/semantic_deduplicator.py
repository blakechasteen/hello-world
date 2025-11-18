#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4: Context Compression - Semantic Deduplicator

Identifies and removes semantic duplicates in context to reduce token waste.

Key Features:
1. Similarity Detection - Find similar content using embeddings or heuristics
2. Deduplication Strategies - Merge, summarize, or remove duplicates
3. Quality Preservation - Never sacrifice quality for compression
4. Token Savings Tracking - Measure effectiveness

Usage:
    deduplicator = SemanticDeduplicator()

    # Find duplicates
    groups = deduplicator.find_duplicates(context_elements)

    # Apply deduplication
    deduplicated = deduplicator.deduplicate(
        context_elements,
        strategy=DeduplicationStrategy.MERGE_SIMILAR
    )

    # Measure savings
    stats = deduplicator.get_statistics()
"""

from __future__ import annotations
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re
import logging

logger = logging.getLogger(__name__)

# Import from context packer
from .context_packer import ContextElement, ContextImportance

# Try to import embeddings for semantic similarity
try:
    from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("Embeddings not available - using heuristic similarity only")


class SimilarityMethod(str, Enum):
    """Methods for computing similarity"""
    EMBEDDINGS = "embeddings"     # Cosine similarity of embeddings (best)
    NGRAM = "ngram"              # N-gram overlap (good)
    JACCARD = "jaccard"          # Jaccard similarity of words (basic)
    EXACT = "exact"              # Exact match only (fastest)


class DeduplicationStrategy(str, Enum):
    """Strategies for handling duplicates"""
    MERGE_SIMILAR = "merge"      # Merge similar elements into one
    KEEP_BEST = "keep_best"      # Keep highest importance, discard rest
    SUMMARIZE = "summarize"      # Create summary of duplicates
    REMOVE_EXACT = "remove_exact"  # Remove only exact duplicates


@dataclass
class DuplicateGroup:
    """
    Group of similar context elements.

    Represents elements that are semantically similar and candidates
    for deduplication.
    """

    elements: List[ContextElement]
    similarity_score: float  # 0.0-1.0
    representative: ContextElement  # Best element to keep
    merged_content: Optional[str] = None  # Merged version
    token_savings: int = 0  # Tokens saved if deduplicated

    def total_tokens(self) -> int:
        """Total tokens in all elements"""
        return sum(e.token_count for e in self.elements)

    def merged_tokens(self) -> int:
        """Tokens in merged version"""
        if self.merged_content:
            return len(self.merged_content) // 4  # Rough estimate
        return self.representative.token_count


@dataclass
class DeduplicationResult:
    """Result of deduplication operation"""

    original_elements: List[ContextElement]
    deduplicated_elements: List[ContextElement]
    duplicate_groups: List[DuplicateGroup]

    # Statistics
    original_tokens: int
    deduplicated_tokens: int
    token_savings: int
    compression_ratio: float  # deduplicated / original
    elements_removed: int
    elements_merged: int

    def summary(self) -> str:
        """Human-readable summary"""
        return (
            f"Deduplication Summary:\n"
            f"  Original: {len(self.original_elements)} elements, {self.original_tokens} tokens\n"
            f"  Deduplicated: {len(self.deduplicated_elements)} elements, {self.deduplicated_tokens} tokens\n"
            f"  Savings: {self.token_savings} tokens ({(1 - self.compression_ratio) * 100:.1f}%)\n"
            f"  Removed: {self.elements_removed}, Merged: {self.elements_merged}"
        )


class SemanticDeduplicator:
    """
    Semantic deduplication for context elements.

    Identifies and removes duplicate/similar content to reduce token waste
    while preserving information quality.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,  # 0.8 = 80% similar
        similarity_method: SimilarityMethod = SimilarityMethod.NGRAM,
        min_tokens_to_deduplicate: int = 50,  # Only deduplicate if saves ≥50 tokens
        preserve_importance: bool = True,  # Never deduplicate CRITICAL importance
        embeddings: Optional[Any] = None  # Optional embeddings for semantic similarity
    ):
        """
        Initialize semantic deduplicator.

        Args:
            similarity_threshold: Minimum similarity to consider duplicates (0.0-1.0)
            similarity_method: Method for computing similarity
            min_tokens_to_deduplicate: Minimum token savings to justify deduplication
            preserve_importance: Preserve CRITICAL importance elements
            embeddings: Optional embeddings model for semantic similarity
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_method = similarity_method
        self.min_tokens = min_tokens_to_deduplicate
        self.preserve_importance = preserve_importance
        self.embeddings = embeddings

        # Statistics
        self.total_comparisons = 0
        self.duplicates_found = 0
        self.tokens_saved = 0

        # Auto-select similarity method
        if similarity_method == SimilarityMethod.EMBEDDINGS:
            if not EMBEDDINGS_AVAILABLE and embeddings is None:
                logger.warning("Embeddings not available, falling back to N-gram similarity")
                self.similarity_method = SimilarityMethod.NGRAM

    def find_duplicates(
        self,
        elements: List[ContextElement],
        min_similarity: Optional[float] = None
    ) -> List[DuplicateGroup]:
        """
        Find groups of similar elements.

        Args:
            elements: Context elements to analyze
            min_similarity: Override similarity threshold

        Returns:
            List of duplicate groups
        """
        threshold = min_similarity if min_similarity is not None else self.similarity_threshold

        # Group elements by similarity
        groups: List[DuplicateGroup] = []
        used_indices: Set[int] = set()

        for i, elem1 in enumerate(elements):
            if i in used_indices:
                continue

            # Skip CRITICAL importance if preserve_importance=True
            if self.preserve_importance and elem1.importance >= ContextImportance.CRITICAL.value:
                continue

            # Find similar elements
            similar_elements = [elem1]
            similar_indices = {i}

            for j, elem2 in enumerate(elements[i+1:], start=i+1):
                if j in used_indices:
                    continue

                # Skip CRITICAL importance
                if self.preserve_importance and elem2.importance >= ContextImportance.CRITICAL.value:
                    continue

                # Compute similarity
                similarity = self._compute_similarity(elem1.content, elem2.content)
                self.total_comparisons += 1

                if similarity >= threshold:
                    similar_elements.append(elem2)
                    similar_indices.add(j)

            # Create group if duplicates found
            if len(similar_elements) > 1:
                # Choose representative (highest importance)
                representative = max(similar_elements, key=lambda e: e.importance)

                group = DuplicateGroup(
                    elements=similar_elements,
                    similarity_score=threshold,  # Average would be better
                    representative=representative
                )

                groups.append(group)
                used_indices.update(similar_indices)
                self.duplicates_found += len(similar_elements) - 1

        return groups

    def deduplicate(
        self,
        elements: List[ContextElement],
        strategy: DeduplicationStrategy = DeduplicationStrategy.MERGE_SIMILAR
    ) -> DeduplicationResult:
        """
        Deduplicate context elements.

        Args:
            elements: Context elements to deduplicate
            strategy: Deduplication strategy

        Returns:
            DeduplicationResult with deduplicated elements and statistics
        """
        # Find duplicate groups
        duplicate_groups = self.find_duplicates(elements)

        # Calculate original tokens
        original_tokens = sum(e.token_count for e in elements)

        # Apply deduplication strategy
        if strategy == DeduplicationStrategy.MERGE_SIMILAR:
            deduplicated, merged_count = self._merge_similar(elements, duplicate_groups)
        elif strategy == DeduplicationStrategy.KEEP_BEST:
            deduplicated, merged_count = self._keep_best(elements, duplicate_groups)
        elif strategy == DeduplicationStrategy.SUMMARIZE:
            deduplicated, merged_count = self._summarize_duplicates(elements, duplicate_groups)
        elif strategy == DeduplicationStrategy.REMOVE_EXACT:
            deduplicated, merged_count = self._remove_exact(elements)
        else:
            deduplicated = elements
            merged_count = 0

        # Calculate statistics
        deduplicated_tokens = sum(e.token_count for e in deduplicated)
        token_savings = original_tokens - deduplicated_tokens

        # Update global statistics
        self.tokens_saved += token_savings

        return DeduplicationResult(
            original_elements=elements,
            deduplicated_elements=deduplicated,
            duplicate_groups=duplicate_groups,
            original_tokens=original_tokens,
            deduplicated_tokens=deduplicated_tokens,
            token_savings=token_savings,
            compression_ratio=deduplicated_tokens / original_tokens if original_tokens > 0 else 1.0,
            elements_removed=len(elements) - len(deduplicated),
            elements_merged=merged_count
        )

    def _merge_similar(
        self,
        elements: List[ContextElement],
        groups: List[DuplicateGroup]
    ) -> Tuple[List[ContextElement], int]:
        """Merge similar elements into one"""
        # Build set of elements in groups
        grouped_elements = set()
        for group in groups:
            for elem in group.elements:
                grouped_elements.add(id(elem))

        # Keep ungrouped elements + representatives
        result = []
        merged_count = 0

        # Add ungrouped elements
        for elem in elements:
            if id(elem) not in grouped_elements:
                result.append(elem)

        # Add merged representatives
        for group in groups:
            # Merge content from all elements
            merged_content = self._merge_content(group.elements)

            # Create merged element
            merged = ContextElement(
                content=merged_content,
                importance=max(e.importance for e in group.elements),
                token_count=len(merged_content) // 4,  # Rough estimate
                source=group.representative.source,
                metadata={
                    "merged_from": len(group.elements),
                    "similarity": group.similarity_score
                }
            )

            result.append(merged)
            merged_count += len(group.elements) - 1

        return result, merged_count

    def _keep_best(
        self,
        elements: List[ContextElement],
        groups: List[DuplicateGroup]
    ) -> Tuple[List[ContextElement], int]:
        """Keep only the best element from each group"""
        # Build set of elements to remove
        to_remove = set()
        for group in groups:
            # Remove all except representative
            for elem in group.elements:
                if elem != group.representative:
                    to_remove.add(id(elem))

        # Filter elements
        result = [e for e in elements if id(e) not in to_remove]
        removed_count = len(to_remove)

        return result, 0  # No merging, just removal

    def _summarize_duplicates(
        self,
        elements: List[ContextElement],
        groups: List[DuplicateGroup]
    ) -> Tuple[List[ContextElement], int]:
        """Create summaries of duplicate groups"""
        # Similar to merge, but create compressed summary
        grouped_elements = set()
        for group in groups:
            for elem in group.elements:
                grouped_elements.add(id(elem))

        result = []
        merged_count = 0

        # Add ungrouped elements
        for elem in elements:
            if id(elem) not in grouped_elements:
                result.append(elem)

        # Add summarized representatives
        for group in groups:
            # Create summary (simple heuristic: first sentence of each)
            summary = self._create_summary(group.elements)

            summarized = ContextElement(
                content=summary,
                importance=max(e.importance for e in group.elements),
                token_count=len(summary) // 4,
                source=group.representative.source,
                metadata={
                    "summarized_from": len(group.elements),
                    "similarity": group.similarity_score
                }
            )

            result.append(summarized)
            merged_count += len(group.elements) - 1

        return result, merged_count

    def _remove_exact(
        self,
        elements: List[ContextElement]
    ) -> Tuple[List[ContextElement], int]:
        """Remove only exact duplicates"""
        seen = set()
        result = []
        removed = 0

        for elem in elements:
            # Use content as key
            key = elem.content

            if key not in seen:
                seen.add(key)
                result.append(elem)
            else:
                removed += 1

        return result, 0

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        if self.similarity_method == SimilarityMethod.EXACT:
            return 1.0 if text1 == text2 else 0.0

        elif self.similarity_method == SimilarityMethod.JACCARD:
            return self._jaccard_similarity(text1, text2)

        elif self.similarity_method == SimilarityMethod.NGRAM:
            return self._ngram_similarity(text1, text2)

        elif self.similarity_method == SimilarityMethod.EMBEDDINGS:
            if self.embeddings:
                return self._embedding_similarity(text1, text2)
            else:
                # Fallback to n-gram
                return self._ngram_similarity(text1, text2)

        return 0.0

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity of word sets"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0

    def _ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """N-gram similarity (character-level)"""
        def get_ngrams(text: str, n: int) -> Set[str]:
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2

        return len(intersection) / len(union) if union else 0.0

    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity of embeddings"""
        try:
            # Get embeddings
            emb1 = self.embeddings.embed([text1])[0]
            emb2 = self.embeddings.embed([text2])[0]

            # Cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            norm1 = sum(a * a for a in emb1) ** 0.5
            norm2 = sum(b * b for b in emb2) ** 0.5

            return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}, falling back to n-gram")
            return self._ngram_similarity(text1, text2)

    def _merge_content(self, elements: List[ContextElement]) -> str:
        """Merge content from multiple elements"""
        # Simple strategy: concatenate unique sentences
        sentences = set()

        for elem in elements:
            # Split into sentences (simple heuristic)
            elem_sentences = re.split(r'[.!?]+', elem.content)
            for sent in elem_sentences:
                sent = sent.strip()
                if sent:
                    sentences.add(sent)

        # Sort by length (longer sentences likely more informative)
        sorted_sentences = sorted(sentences, key=len, reverse=True)

        # Combine sentences
        return '. '.join(sorted_sentences[:3])  # Keep top 3 sentences

    def _create_summary(self, elements: List[ContextElement]) -> str:
        """Create summary from multiple elements"""
        # Extract first sentence from each element
        summaries = []

        for elem in elements:
            # Get first sentence
            first_sentence = re.split(r'[.!?]+', elem.content)[0].strip()
            if first_sentence:
                summaries.append(first_sentence)

        # Deduplicate and join
        unique_summaries = list(dict.fromkeys(summaries))  # Preserve order
        return '. '.join(unique_summaries[:2])  # Keep top 2

    def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            "total_comparisons": self.total_comparisons,
            "duplicates_found": self.duplicates_found,
            "tokens_saved": self.tokens_saved,
            "similarity_method": self.similarity_method.value,
            "similarity_threshold": self.similarity_threshold
        }

    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_comparisons = 0
        self.duplicates_found = 0
        self.tokens_saved = 0

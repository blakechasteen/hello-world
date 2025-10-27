#!/usr/bin/env python3
"""
Duplicate Detection System
===========================
Find and manage duplicate or near-duplicate prompts.

Features:
- Exact duplicate detection
- Fuzzy matching for near-duplicates
- Similarity scoring
- Merge recommendations
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import difflib
import hashlib


@dataclass
class DuplicateMatch:
    """A potential duplicate prompt"""
    prompt_id: str
    name: str
    content: str
    similarity: float  # 0.0-1.0
    match_type: str  # 'exact', 'high', 'medium', 'low'


class DuplicateDetector:
    """
    Detect duplicate and similar prompts.

    Uses multiple similarity metrics.
    """

    def __init__(self, exact_threshold: float = 1.0, high_threshold: float = 0.95):
        """
        Initialize detector.

        Args:
            exact_threshold: Threshold for exact duplicates (1.0)
            high_threshold: Threshold for high similarity (0.95)
        """
        self.exact_threshold = exact_threshold
        self.high_threshold = high_threshold

    def find_duplicates(
        self,
        prompts: List[Dict[str, any]],
        min_similarity: float = 0.80
    ) -> Dict[str, List[DuplicateMatch]]:
        """
        Find all duplicate pairs.

        Args:
            prompts: List of prompt dicts with 'id', 'name', 'content'
            min_similarity: Minimum similarity to report

        Returns:
            Dict mapping prompt_id to list of duplicates
        """
        duplicates = {}

        for i, prompt1 in enumerate(prompts):
            matches = []

            for j, prompt2 in enumerate(prompts):
                if i >= j:  # Skip self and already compared pairs
                    continue

                similarity = self.calculate_similarity(
                    prompt1['content'],
                    prompt2['content']
                )

                if similarity >= min_similarity:
                    # Determine match type
                    if similarity >= self.exact_threshold:
                        match_type = 'exact'
                    elif similarity >= self.high_threshold:
                        match_type = 'high'
                    elif similarity >= 0.85:
                        match_type = 'medium'
                    else:
                        match_type = 'low'

                    matches.append(DuplicateMatch(
                        prompt_id=prompt2.get('id', ''),
                        name=prompt2.get('name', 'Unknown'),
                        content=prompt2.get('content', ''),
                        similarity=similarity,
                        match_type=match_type
                    ))

            if matches:
                duplicates[prompt1.get('id', '')] = matches

        return duplicates

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Uses SequenceMatcher for character-level similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        # Normalize texts
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)

        # Use difflib's SequenceMatcher
        matcher = difflib.SequenceMatcher(None, norm1, norm2)
        return matcher.ratio()

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.

        Args:
            text: Raw text

        Returns:
            Normalized text
        """
        import re

        # Lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common variations
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')

        return text.strip()

    def get_content_hash(self, content: str) -> str:
        """
        Get hash of content for exact matching.

        Args:
            content: Prompt content

        Returns:
            SHA256 hash
        """
        normalized = self._normalize_text(content)
        return hashlib.sha256(normalized.encode()).hexdigest()

    def find_exact_duplicates(
        self,
        prompts: List[Dict[str, any]]
    ) -> Dict[str, List[str]]:
        """
        Find exact duplicates by content hash.

        Args:
            prompts: List of prompt dicts

        Returns:
            Dict mapping hash to list of prompt IDs
        """
        hash_to_ids = {}

        for prompt in prompts:
            content = prompt.get('content', '')
            content_hash = self.get_content_hash(content)

            if content_hash not in hash_to_ids:
                hash_to_ids[content_hash] = []

            hash_to_ids[content_hash].append(prompt.get('id', ''))

        # Filter to only duplicates (hash with 2+ prompts)
        duplicates = {
            h: ids for h, ids in hash_to_ids.items()
            if len(ids) > 1
        }

        return duplicates

    def generate_merge_suggestions(
        self,
        prompt1: Dict[str, any],
        prompt2: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Suggest how to merge two similar prompts.

        Args:
            prompt1: First prompt dict
            prompt2: Second prompt dict

        Returns:
            Dict with merge suggestions
        """
        similarity = self.calculate_similarity(
            prompt1.get('content', ''),
            prompt2.get('content', '')
        )

        # Determine merge strategy
        if similarity >= 0.95:
            strategy = "keep_one"
            recommendation = "These are nearly identical. Keep the most recent or higher quality version."
        elif similarity >= 0.85:
            strategy = "merge_content"
            recommendation = "Merge the best parts of both prompts into one."
        else:
            strategy = "keep_both"
            recommendation = "These are similar but different enough to keep separate."

        # Compare metadata
        tags1 = set(prompt1.get('tags', []))
        tags2 = set(prompt2.get('tags', []))
        merged_tags = list(tags1 | tags2)

        usage1 = prompt1.get('usage_count', 0)
        usage2 = prompt2.get('usage_count', 0)
        keep_prompt = prompt1 if usage1 >= usage2 else prompt2

        return {
            'similarity': similarity,
            'strategy': strategy,
            'recommendation': recommendation,
            'merged_tags': merged_tags,
            'suggested_keep': keep_prompt.get('id', ''),
            'comparison': {
                'prompt1_usage': usage1,
                'prompt2_usage': usage2,
                'common_tags': list(tags1 & tags2),
                'unique_tags': list((tags1 | tags2) - (tags1 & tags2))
            }
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def check_for_duplicates(prompts: List[Dict[str, any]], threshold: float = 0.90) -> List[Tuple[str, str, float]]:
    """
    Quick duplicate check.

    Args:
        prompts: List of prompts
        threshold: Similarity threshold

    Returns:
        List of (id1, id2, similarity) tuples
    """
    detector = DuplicateDetector()
    duplicates = detector.find_duplicates(prompts, min_similarity=threshold)

    pairs = []
    for prompt_id, matches in duplicates.items():
        for match in matches:
            pairs.append((prompt_id, match.prompt_id, match.similarity))

    return pairs


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Duplicate Detection System\n")

    # Test prompts
    test_prompts = [
        {
            'id': 'prompt1',
            'name': 'SQL Optimizer',
            'content': 'Analyze this SQL query for performance issues and suggest optimizations.',
            'tags': ['sql', 'optimization'],
            'usage_count': 42
        },
        {
            'id': 'prompt2',
            'name': 'SQL Performance',
            'content': 'Analyze this SQL query for performance issues and suggest optimizations.',  # Exact duplicate
            'tags': ['sql', 'database', 'optimization'],
            'usage_count': 15
        },
        {
            'id': 'prompt3',
            'name': 'Query Optimizer',
            'content': 'Review this SQL query and recommend performance improvements.',  # Similar
            'tags': ['sql', 'optimization'],
            'usage_count': 28
        },
        {
            'id': 'prompt4',
            'name': 'Code Reviewer',
            'content': 'Review this code for bugs and suggest improvements.',  # Different
            'tags': ['code-review'],
            'usage_count': 67
        }
    ]

    detector = DuplicateDetector()

    # Find exact duplicates
    print("=== Exact Duplicates ===\n")
    exact = detector.find_exact_duplicates(test_prompts)

    if exact:
        for hash_val, ids in exact.items():
            print(f"Hash: {hash_val[:12]}...")
            print(f"Prompts: {', '.join(ids)}")
            print()
    else:
        print("No exact duplicates found.\n")

    # Find similar prompts
    print("=== Similar Prompts (>80% match) ===\n")
    duplicates = detector.find_duplicates(test_prompts, min_similarity=0.80)

    for prompt_id, matches in duplicates.items():
        prompt = next(p for p in test_prompts if p['id'] == prompt_id)
        print(f"Prompt: {prompt['name']} ({prompt_id})")

        for match in matches:
            print(f"  -> {match.name} ({match.prompt_id})")
            print(f"     Similarity: {match.similarity:.2f} ({match.match_type})")

            # Generate merge suggestion
            match_prompt = next(p for p in test_prompts if p['id'] == match.prompt_id)
            suggestion = detector.generate_merge_suggestions(prompt, match_prompt)

            print(f"     Strategy: {suggestion['strategy']}")
            print(f"     Recommendation: {suggestion['recommendation']}")
            print()

    print("\n=== Summary ===")
    total_duplicates = sum(len(matches) for matches in duplicates.values())
    print(f"Found {total_duplicates} potential duplicates across {len(duplicates)} prompts")

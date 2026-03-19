"""
Fuzzy Duplicate Detection using Levenshtein Distance

Detects near-duplicate records that differ by small edit distances.
Uses edit distance (Levenshtein) to identify similar but not identical records.

Author: DATAPIG Enhancement
Date: 2025-11-22
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FuzzyMatch:
    """Represents a fuzzy duplicate match"""
    row1_index: int
    row2_index: int
    similarity: float  # 0.0-1.0
    edit_distance: int
    field: str
    value1: str
    value2: str


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # Create distance matrix
    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def normalized_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized similarity (0.0-1.0) between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score (1.0 = identical, 0.0 = completely different)
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))

    return 1.0 - (distance / max_len)


def find_fuzzy_duplicates(
    data: list[dict[str, Any]],
    fields: list[str],
    similarity_threshold: float = 0.85
) -> list[FuzzyMatch]:
    """
    Find fuzzy duplicate records based on Levenshtein distance.

    Args:
        data: List of dictionaries to check
        fields: Fields to compare for fuzzy matching
        similarity_threshold: Minimum similarity (0.0-1.0) to consider a match

    Returns:
        List of FuzzyMatch objects for near-duplicate pairs
    """
    matches = []
    seen_pairs: set[tuple[int, int]] = set()

    # Compare all pairs of records
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            # Skip if already processed
            if (i, j) in seen_pairs:
                continue

            # Compare specified fields
            for field in fields:
                # Get field values (convert to string)
                val1 = str(data[i].get(field, ""))
                val2 = str(data[j].get(field, ""))

                # Skip empty values
                if not val1 or not val2:
                    continue

                # Calculate similarity
                similarity = normalized_similarity(val1, val2)

                # If similar enough, record match
                if similarity >= similarity_threshold:
                    distance = levenshtein_distance(val1.lower(), val2.lower())

                    matches.append(FuzzyMatch(
                        row1_index=i,
                        row2_index=j,
                        similarity=similarity,
                        edit_distance=distance,
                        field=field,
                        value1=val1,
                        value2=val2
                    ))

                    # Mark pair as seen
                    seen_pairs.add((i, j))
                    break  # Don't check other fields for this pair

    return matches


def phonetic_similarity(s1: str, s2: str) -> float:
    """
    Simple phonetic similarity based on consonant matching.

    Useful for names that sound similar but are spelled differently.
    E.g., "Smith" vs "Smyth", "Catherine" vs "Katherine"

    Args:
        s1: First string
        s2: Second string

    Returns:
        Phonetic similarity score (0.0-1.0)
    """
    # Extract consonants (phonetic skeleton)
    consonants = "bcdfghjklmnpqrstvwxyz"

    def get_consonants(s: str) -> str:
        return "".join(c for c in s.lower() if c in consonants)

    c1 = get_consonants(s1)
    c2 = get_consonants(s2)

    # Calculate similarity on consonant skeleton
    if not c1 and not c2:
        return 1.0
    if not c1 or not c2:
        return 0.0

    return normalized_similarity(c1, c2)


def combined_similarity(
    s1: str,
    s2: str,
    levenshtein_weight: float = 0.7,
    phonetic_weight: float = 0.3
) -> float:
    """
    Combined similarity using both edit distance and phonetic matching.

    Args:
        s1: First string
        s2: Second string
        levenshtein_weight: Weight for Levenshtein similarity
        phonetic_weight: Weight for phonetic similarity

    Returns:
        Combined similarity score (0.0-1.0)
    """
    lev_sim = normalized_similarity(s1, s2)
    phon_sim = phonetic_similarity(s1, s2)

    return (levenshtein_weight * lev_sim) + (phonetic_weight * phon_sim)


def find_fuzzy_duplicates_advanced(
    data: list[dict[str, Any]],
    fields: list[str],
    similarity_threshold: float = 0.85,
    use_phonetic: bool = True
) -> list[FuzzyMatch]:
    """
    Advanced fuzzy duplicate detection with phonetic matching.

    Args:
        data: List of dictionaries to check
        fields: Fields to compare for fuzzy matching
        similarity_threshold: Minimum similarity (0.0-1.0) to consider a match
        use_phonetic: Whether to use phonetic matching

    Returns:
        List of FuzzyMatch objects for near-duplicate pairs
    """
    matches = []
    seen_pairs: set[tuple[int, int]] = set()

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if (i, j) in seen_pairs:
                continue

            for field in fields:
                val1 = str(data[i].get(field, ""))
                val2 = str(data[j].get(field, ""))

                if not val1 or not val2:
                    continue

                # Use combined similarity if phonetic enabled
                if use_phonetic:
                    similarity = combined_similarity(val1, val2)
                else:
                    similarity = normalized_similarity(val1, val2)

                if similarity >= similarity_threshold:
                    distance = levenshtein_distance(val1.lower(), val2.lower())

                    matches.append(FuzzyMatch(
                        row1_index=i,
                        row2_index=j,
                        similarity=similarity,
                        edit_distance=distance,
                        field=field,
                        value1=val1,
                        value2=val2
                    ))

                    seen_pairs.add((i, j))
                    break

    return matches

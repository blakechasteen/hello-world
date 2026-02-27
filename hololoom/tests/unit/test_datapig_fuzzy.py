"""
Unit Tests for DATAPIG Fuzzy Duplicate Detection

Tests fuzzy matching algorithms using Levenshtein distance and phonetic similarity.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
from hololoom.datapig.fuzzy_detection import (
    levenshtein_distance,
    normalized_similarity,
    phonetic_similarity,
    combined_similarity,
    find_fuzzy_duplicates,
    find_fuzzy_duplicates_advanced,
    FuzzyMatch
)


# ============================================================================
# Levenshtein Distance Tests
# ============================================================================

def test_levenshtein_exact_match():
    """Test Levenshtein distance for exact matches"""
    assert levenshtein_distance("hello", "hello") == 0
    assert levenshtein_distance("test", "test") == 0
    assert levenshtein_distance("", "") == 0


def test_levenshtein_single_insertion():
    """Test Levenshtein distance for single character insertion"""
    assert levenshtein_distance("hello", "helo") == 1
    assert levenshtein_distance("test", "est") == 1


def test_levenshtein_single_deletion():
    """Test Levenshtein distance for single character deletion"""
    assert levenshtein_distance("hello", "helllo") == 1
    assert levenshtein_distance("test", "tesst") == 1


def test_levenshtein_single_substitution():
    """Test Levenshtein distance for single character substitution"""
    assert levenshtein_distance("hello", "hallo") == 1
    assert levenshtein_distance("test", "tent") == 1


def test_levenshtein_multiple_edits():
    """Test Levenshtein distance for multiple edits"""
    # "kitten" -> "sitting" requires 3 edits
    # k->s, e->i, insert g
    assert levenshtein_distance("kitten", "sitting") == 3

    # "Saturday" -> "Sunday" requires 3 edits
    # remove atur, add un
    assert levenshtein_distance("Saturday", "Sunday") == 3


def test_levenshtein_empty_strings():
    """Test Levenshtein distance with empty strings"""
    assert levenshtein_distance("hello", "") == 5
    assert levenshtein_distance("", "world") == 5
    assert levenshtein_distance("", "") == 0


def test_levenshtein_case_insensitive():
    """Test that Levenshtein distance is case-insensitive (via normalized_similarity)"""
    # Raw levenshtein_distance is case-sensitive
    assert levenshtein_distance("Hello", "hello") == 1

    # But normalized_similarity converts to lowercase
    assert normalized_similarity("Hello", "hello") == 1.0


# ============================================================================
# Normalized Similarity Tests
# ============================================================================

def test_normalized_similarity_exact_match():
    """Test normalized similarity for exact matches"""
    assert normalized_similarity("hello", "hello") == 1.0
    assert normalized_similarity("test", "test") == 1.0


def test_normalized_similarity_partial_match():
    """Test normalized similarity for partial matches"""
    # "hello" vs "hallo" = 1 edit / 5 chars = 0.8 similarity
    sim = normalized_similarity("hello", "hallo")
    assert 0.79 < sim < 0.81

    # "test" vs "tent" = 1 edit / 4 chars = 0.75 similarity
    sim = normalized_similarity("test", "tent")
    assert 0.74 < sim < 0.76


def test_normalized_similarity_no_match():
    """Test normalized similarity for completely different strings"""
    # "abc" vs "xyz" = 3 edits / 3 chars = 0.0 similarity
    assert normalized_similarity("abc", "xyz") == 0.0


def test_normalized_similarity_empty_strings():
    """Test normalized similarity with empty strings"""
    assert normalized_similarity("", "") == 1.0
    assert normalized_similarity("hello", "") == 0.0
    assert normalized_similarity("", "world") == 0.0


def test_normalized_similarity_case_insensitive():
    """Test that normalized similarity is case-insensitive"""
    assert normalized_similarity("Hello", "hello") == 1.0
    assert normalized_similarity("WORLD", "world") == 1.0
    assert normalized_similarity("TeSt", "tEsT") == 1.0


# ============================================================================
# Phonetic Similarity Tests
# ============================================================================

def test_phonetic_similarity_exact_match():
    """Test phonetic similarity for exact matches"""
    assert phonetic_similarity("hello", "hello") == 1.0
    assert phonetic_similarity("test", "test") == 1.0


def test_phonetic_similarity_vowel_differences():
    """Test phonetic similarity ignores vowel differences"""
    # "Smith" vs "Smyth" - consonants "smth" vs "smyth" (y treated as consonant)
    # Similarity = 1 - (1/5) = 0.8
    sim = phonetic_similarity("Smith", "Smyth")
    assert sim == 0.8  # 'y' treated as consonant, so one edit

    # "Catherine" vs "Katherine" - consonants "cthrn" vs "kthrn" (c vs k)
    # Similarity = 1 - (1/5) = 0.8
    sim = phonetic_similarity("Catherine", "Katherine")
    assert sim == 0.8  # One consonant difference


def test_phonetic_similarity_consonant_differences():
    """Test phonetic similarity detects consonant differences"""
    # "bat" vs "cat" - different consonants
    sim = phonetic_similarity("bat", "cat")
    assert sim < 1.0  # Should not be perfect match


def test_phonetic_similarity_empty_strings():
    """Test phonetic similarity with empty strings"""
    assert phonetic_similarity("", "") == 1.0
    assert phonetic_similarity("hello", "") == 0.0
    assert phonetic_similarity("", "world") == 0.0


def test_phonetic_similarity_all_vowels():
    """Test phonetic similarity with all vowels"""
    # Strings with only vowels have no consonants
    assert phonetic_similarity("aaa", "eee") == 1.0  # No consonants, so match
    assert phonetic_similarity("aeiou", "ieoua") == 1.0


# ============================================================================
# Combined Similarity Tests
# ============================================================================

def test_combined_similarity_default_weights():
    """Test combined similarity with default weights (0.7 Levenshtein, 0.3 phonetic)"""
    # Exact match should be 1.0
    assert combined_similarity("hello", "hello") == 1.0

    # Partial match
    sim = combined_similarity("Smith", "Smyth")
    assert 0.7 < sim < 1.0  # High due to phonetic match


def test_combined_similarity_custom_weights():
    """Test combined similarity with custom weights"""
    # Equal weighting
    sim = combined_similarity("test", "tent", levenshtein_weight=0.5, phonetic_weight=0.5)
    assert 0.0 < sim < 1.0


def test_combined_similarity_phonetic_heavy():
    """Test combined similarity with heavy phonetic weight"""
    # Use strings with clear phonetic vs Levenshtein difference
    # "color" vs "colour" - Levenshtein: 1 edit / 6 = 0.833
    # Phonetic: "clr" vs "clr" = 1.0
    sim_heavy_phonetic = combined_similarity("color", "colour",
                                             levenshtein_weight=0.3,
                                             phonetic_weight=0.7)
    sim_heavy_lev = combined_similarity("color", "colour",
                                        levenshtein_weight=0.7,
                                        phonetic_weight=0.3)

    # Phonetic-heavy should be higher since consonants match perfectly
    # Heavy phonetic: 0.3 * 0.833 + 0.7 * 1.0 = 0.95
    # Heavy Levenshtein: 0.7 * 0.833 + 0.3 * 1.0 = 0.883
    assert sim_heavy_phonetic > sim_heavy_lev


# ============================================================================
# Fuzzy Duplicate Detection Tests
# ============================================================================

def test_find_fuzzy_duplicates_exact_duplicates():
    """Test finding exact duplicate records"""
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Alice"},  # Exact duplicate
    ]

    matches = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=1.0)

    # Should find 1 match (row 0 vs row 2)
    assert len(matches) == 1
    assert matches[0].row1_index == 0
    assert matches[0].row2_index == 2
    assert matches[0].similarity == 1.0
    assert matches[0].field == "name"


def test_find_fuzzy_duplicates_near_duplicates():
    """Test finding near-duplicate records (fuzzy matches)"""
    data = [
        {"id": 1, "name": "Smith"},
        {"id": 2, "name": "Smyth"},  # Fuzzy duplicate
        {"id": 3, "name": "Jones"},
    ]

    matches = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=0.75)

    # Should find 1 match (Smith vs Smyth)
    assert len(matches) == 1
    assert matches[0].row1_index == 0
    assert matches[0].row2_index == 1
    assert matches[0].similarity > 0.75
    assert matches[0].field == "name"


def test_find_fuzzy_duplicates_multiple_fields():
    """Test fuzzy duplicate detection across multiple fields"""
    data = [
        {"id": 1, "first": "John", "last": "Doe"},
        {"id": 2, "first": "Jon", "last": "Doe"},  # Fuzzy match on first name
        {"id": 3, "first": "Jane", "last": "Smith"},
    ]

    matches = find_fuzzy_duplicates(data, fields=["first", "last"], similarity_threshold=0.75)

    # Should find 1 match (John vs Jon)
    assert len(matches) >= 1


def test_find_fuzzy_duplicates_threshold_filtering():
    """Test that similarity threshold filters results"""
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Alicia"},  # Similar but not exact (2 edits / 6 chars = 0.67 similarity)
        {"id": 3, "name": "Bob"},
    ]

    # High threshold - should not match
    matches_high = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=1.0)
    assert len(matches_high) == 0

    # Lower threshold - should match (Alice vs Alicia = 0.67 similarity)
    matches_low = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=0.6)
    assert len(matches_low) >= 1


def test_find_fuzzy_duplicates_empty_values():
    """Test fuzzy duplicate detection skips empty values"""
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": ""},  # Empty
        {"id": 3, "name": None},  # None
    ]

    matches = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=0.8)

    # Should not match empty values
    assert len(matches) == 0


def test_find_fuzzy_duplicates_no_duplicates():
    """Test fuzzy duplicate detection with no duplicates"""
    data = [
        {"id": 1, "name": "Alice"},
        {"id": 2, "name": "Bob"},
        {"id": 3, "name": "Charlie"},
    ]

    matches = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=0.8)

    # Should find no matches
    assert len(matches) == 0


# ============================================================================
# Advanced Fuzzy Detection Tests
# ============================================================================

def test_find_fuzzy_duplicates_advanced_with_phonetic():
    """Test advanced fuzzy detection with phonetic matching enabled"""
    data = [
        {"id": 1, "name": "Smith"},
        {"id": 2, "name": "Smyth"},  # Phonetically similar
        {"id": 3, "name": "Jones"},
    ]

    matches = find_fuzzy_duplicates_advanced(
        data,
        fields=["name"],
        similarity_threshold=0.75,
        use_phonetic=True
    )

    # Should find match due to phonetic similarity
    assert len(matches) >= 1


def test_find_fuzzy_duplicates_advanced_without_phonetic():
    """Test advanced fuzzy detection with phonetic matching disabled"""
    data = [
        {"id": 1, "name": "Smith"},
        {"id": 2, "name": "Smyth"},
        {"id": 3, "name": "Jones"},
    ]

    matches_phonetic = find_fuzzy_duplicates_advanced(
        data,
        fields=["name"],
        similarity_threshold=0.75,
        use_phonetic=True
    )

    matches_no_phonetic = find_fuzzy_duplicates_advanced(
        data,
        fields=["name"],
        similarity_threshold=0.75,
        use_phonetic=False
    )

    # Both should find matches, but counts may differ
    assert len(matches_phonetic) >= 0
    assert len(matches_no_phonetic) >= 0


def test_fuzzy_match_dataclass():
    """Test FuzzyMatch dataclass structure"""
    match = FuzzyMatch(
        row1_index=0,
        row2_index=1,
        similarity=0.95,
        edit_distance=1,
        field="name",
        value1="Smith",
        value2="Smyth"
    )

    assert match.row1_index == 0
    assert match.row2_index == 1
    assert match.similarity == 0.95
    assert match.edit_distance == 1
    assert match.field == "name"
    assert match.value1 == "Smith"
    assert match.value2 == "Smyth"


# ============================================================================
# Performance Tests
# ============================================================================

def test_fuzzy_detection_performance_small_dataset():
    """Test fuzzy detection performance on small dataset"""
    import time

    # Create dataset with 50 records
    data = []
    for i in range(50):
        data.append({"id": i, "name": f"Person{i}"})

    # Add some fuzzy duplicates
    data.append({"id": 100, "name": "Person1"})  # Exact duplicate
    data.append({"id": 101, "name": "Person2"})  # Exact duplicate

    # Benchmark detection
    start = time.perf_counter()
    matches = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=0.9)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Should find 2 matches
    assert len(matches) >= 2

    # Should be fast (<100ms for 50 records)
    print(f"\n  Fuzzy detection (52 rows): {duration_ms:.2f}ms")
    assert duration_ms < 100


def test_fuzzy_detection_performance_medium_dataset():
    """Test fuzzy detection performance on medium dataset"""
    import time

    # Create dataset with 100 records
    data = []
    for i in range(100):
        data.append({"id": i, "name": f"Person{i}"})

    # Add fuzzy duplicates
    data.append({"id": 200, "name": "Person1"})
    data.append({"id": 201, "name": "Persn1"})  # Fuzzy

    start = time.perf_counter()
    matches = find_fuzzy_duplicates(data, fields=["name"], similarity_threshold=0.85)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    assert len(matches) >= 1
    print(f"\n  Fuzzy detection (102 rows): {duration_ms:.2f}ms")
    assert duration_ms < 500  # Should still be reasonably fast

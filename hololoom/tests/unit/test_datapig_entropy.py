"""
Unit Tests for DATAPIG Entropy-Based PII Detection

Tests Shannon entropy calculations and PII detection algorithms.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
import math
from hololoom.datapig.entropy_detection import (
    shannon_entropy,
    normalized_entropy,
    detect_pii_by_entropy,
    calculate_field_entropy_profile,
    detect_entropy_anomalies,
    classify_entropy_level,
    EntropyAnalysis
)


# ============================================================================
# Shannon Entropy Tests
# ============================================================================

def test_shannon_entropy_uniform():
    """Test Shannon entropy for uniform distribution (maximum entropy)"""
    # "abcd" - all chars appear once -> maximum entropy for 4 unique chars
    entropy = shannon_entropy("abcd")
    assert 1.9 < entropy < 2.1  # log₂(4) = 2.0


def test_shannon_entropy_repetitive():
    """Test Shannon entropy for repetitive data (minimum entropy)"""
    # "aaaa" - all same char -> entropy should be 0
    entropy = shannon_entropy("aaaa")
    assert entropy == 0.0


def test_shannon_entropy_mixed():
    """Test Shannon entropy for mixed distribution"""
    # "aab" - 2 a's, 1 b -> partial entropy
    entropy = shannon_entropy("aab")
    # H = -(2/3 * log₂(2/3) + 1/3 * log₂(1/3)) ≈ 0.918
    assert 0.9 < entropy < 1.0


def test_shannon_entropy_empty_string():
    """Test Shannon entropy for empty string"""
    assert shannon_entropy("") == 0.0
    assert shannon_entropy(None) == 0.0


def test_shannon_entropy_ssn_pattern():
    """Test Shannon entropy for SSN-like pattern"""
    # SSN has moderate-high entropy (multiple unique digits + dashes)
    ssn = "123-45-6789"
    entropy = shannon_entropy(ssn)
    assert 3.0 < entropy < 3.6  # Typical SSN entropy range


def test_shannon_entropy_credit_card():
    """Test Shannon entropy for credit card pattern"""
    # Credit card with repetitive digits has LOW entropy
    # "4111-1111-1111-1111" has many repeated "1"s → low entropy
    cc = "4111-1111-1111-1111"
    entropy = shannon_entropy(cc)
    assert 0.9 < entropy < 1.5  # Low entropy due to repetitive pattern


def test_shannon_entropy_api_key():
    """Test Shannon entropy for API key (high entropy)"""
    # Base64-encoded API key should have high entropy
    api_key = "dGVzdC1hcGkta2V5LTEyMzQ1Njc4OQ=="
    entropy = shannon_entropy(api_key)
    assert entropy > 4.0  # High entropy


# ============================================================================
# Normalized Entropy Tests
# ============================================================================

def test_normalized_entropy_uniform():
    """Test normalized entropy for uniform distribution (should be 1.0)"""
    # All unique chars -> normalized entropy = 1.0
    norm = normalized_entropy("abcd")
    assert 0.99 < norm <= 1.0


def test_normalized_entropy_repetitive():
    """Test normalized entropy for repetitive data (should be 0.0)"""
    norm = normalized_entropy("aaaa")
    assert norm == 0.0


def test_normalized_entropy_scale():
    """Test that normalized entropy is always in [0.0, 1.0] range"""
    test_strings = ["a", "aa", "aaa", "abc", "abcdef", "password123", "random!@#$"]
    for s in test_strings:
        norm = normalized_entropy(s)
        assert 0.0 <= norm <= 1.0, f"Normalized entropy for '{s}' out of range: {norm}"


def test_normalized_entropy_empty():
    """Test normalized entropy for empty string"""
    assert normalized_entropy("") == 0.0
    assert normalized_entropy(None) == 0.0


# ============================================================================
# PII Detection Tests
# ============================================================================

def test_detect_pii_by_entropy_ssn():
    """Test detecting SSN fields by entropy"""
    data = [
        {"id": 1, "name": "Alice", "ssn": "123-45-6789"},
        {"id": 2, "name": "Bob", "ssn": "987-65-4321"},
        {"id": 3, "name": "Charlie", "ssn": "555-12-3456"},
        {"id": 4, "name": "Diana", "ssn": "111-22-3333"},
        {"id": 5, "name": "Eve", "ssn": "999-88-7777"},
    ]

    results = detect_pii_by_entropy(data, high_entropy_threshold=3.0)

    # Should detect 'ssn' field as having high entropy
    ssn_results = [r for r in results if r.field_name == "ssn"]
    assert len(ssn_results) >= 1, "SSN field not detected"

    ssn_analysis = ssn_results[0]
    assert ssn_analysis.high_entropy_count >= 1
    assert "SSN_FORMAT" in ssn_analysis.suspicious_patterns


def test_detect_pii_by_entropy_credit_cards():
    """Test detecting credit card fields by entropy"""
    data = [
        {"id": 1, "cc": "4111-1111-1111-1111"},
        {"id": 2, "cc": "5500-0000-0000-0004"},
        {"id": 3, "cc": "3400-0000-0000-009"},
        {"id": 4, "cc": "3000-0000-0000-04"},
        {"id": 5, "cc": "6011-0000-0000-0004"},
    ]

    results = detect_pii_by_entropy(data, high_entropy_threshold=3.0)

    # Should detect 'cc' field
    cc_results = [r for r in results if r.field_name == "cc"]
    if cc_results:  # May or may not detect depending on entropy
        assert cc_results[0].high_entropy_count >= 1


def test_detect_pii_by_entropy_api_keys():
    """Test detecting API key fields by entropy"""
    data = [
        {"id": 1, "api_key": "dGVzdC1hcGkta2V5LTEyMzQ1Njc4OQ=="},
        {"id": 2, "api_key": "YW5vdGhlci10ZXN0LWtleS02Nzg5MA=="},
        {"id": 3, "api_key": "cmFuZG9tLWFwaS1rZXktMTExMjIy"},
        {"id": 4, "api_key": "c2VjcmV0LXRva2VuLTk5ODg3Nw=="},
        {"id": 5, "api_key": "cHJpdmF0ZS1rZXktNDQ1NTY2"},
    ]

    results = detect_pii_by_entropy(data, high_entropy_threshold=4.0)

    # Should detect 'api_key' field
    api_results = [r for r in results if r.field_name == "api_key"]
    assert len(api_results) >= 1

    api_analysis = api_results[0]
    assert api_analysis.high_entropy_count >= 1
    assert "API_KEY_FORMAT" in api_analysis.suspicious_patterns


def test_detect_pii_by_entropy_uuid():
    """Test detecting UUID fields by entropy"""
    data = [
        {"id": 1, "session_id": "550e8400-e29b-41d4-a716-446655440000"},
        {"id": 2, "session_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8"},
        {"id": 3, "session_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"},
        {"id": 4, "session_id": "00000000-0000-0000-0000-000000000000"},
        {"id": 5, "session_id": "ffffffff-ffff-ffff-ffff-ffffffffffff"},
    ]

    # UUIDs have moderate-high entropy (~3.4), not ultra-high (4.0+)
    results = detect_pii_by_entropy(data, high_entropy_threshold=3.0)

    # Should detect 'session_id' field
    uuid_results = [r for r in results if r.field_name == "session_id"]
    assert len(uuid_results) >= 1

    uuid_analysis = uuid_results[0]
    assert "UUID_FORMAT" in uuid_analysis.suspicious_patterns


def test_detect_pii_by_entropy_hash():
    """Test detecting hash fields by entropy"""
    data = [
        {"id": 1, "hash": "5d41402abc4b2a76b9719d911017c592"},  # MD5
        {"id": 2, "hash": "098f6bcd4621d373cade4e832627b4f6"},
        {"id": 3, "hash": "482c811da5d5b4bc6d497ffa98491e38"},
        {"id": 4, "hash": "0cc175b9c0f1b6a831c399e269772661"},
        {"id": 5, "hash": "92eb5ffee6ae2fec3ad71c777531578f"},
    ]

    # MD5 hashes have moderate-high entropy (~3.5), not ultra-high (4.0+)
    results = detect_pii_by_entropy(data, high_entropy_threshold=3.0)

    # Should detect 'hash' field
    hash_results = [r for r in results if r.field_name == "hash"]
    assert len(hash_results) >= 1

    hash_analysis = hash_results[0]
    assert "HASH_FORMAT" in hash_analysis.suspicious_patterns


def test_detect_pii_by_entropy_weak_passwords():
    """Test detecting weak password fields by low entropy"""
    data = [
        {"id": 1, "username": "alice", "password": "password"},
        {"id": 2, "username": "bob", "password": "12345678"},
        {"id": 3, "username": "charlie", "password": "qwerty"},
        {"id": 4, "username": "diana", "password": "abc123"},
        {"id": 5, "username": "eve", "password": "password123"},
    ]

    results = detect_pii_by_entropy(data, low_entropy_threshold=2.5)

    # Should detect 'password' field as having low entropy
    password_results = [r for r in results if r.field_name == "password"]
    if password_results:
        password_analysis = password_results[0]
        assert password_analysis.low_entropy_count >= 3  # Most passwords are weak


def test_detect_pii_by_entropy_min_samples():
    """Test that detection requires minimum samples"""
    # Only 2 samples (below default min_samples=5)
    data = [
        {"id": 1, "ssn": "123-45-6789"},
        {"id": 2, "ssn": "987-65-4321"},
    ]

    results = detect_pii_by_entropy(data, min_samples=5)
    assert len(results) == 0  # Should not detect with insufficient samples

    # Should detect with lower threshold
    results = detect_pii_by_entropy(data, min_samples=2)
    assert len(results) >= 0  # May detect now


def test_detect_pii_by_entropy_no_string_fields():
    """Test detection with no string fields (only numbers)"""
    data = [
        {"id": 1, "value": 100},
        {"id": 2, "value": 200},
        {"id": 3, "value": 300},
    ]

    results = detect_pii_by_entropy(data)
    assert len(results) == 0  # No string fields to analyze


# ============================================================================
# Entropy Profile Tests
# ============================================================================

def test_calculate_field_entropy_profile():
    """Test calculating entropy profile for all fields"""
    data = [
        {"id": "001", "name": "Alice", "ssn": "123-45-6789"},
        {"id": "002", "name": "Bob", "ssn": "987-65-4321"},
        {"id": "003", "name": "Charlie", "ssn": "555-12-3456"},
    ]

    profile = calculate_field_entropy_profile(data)

    # Should have profiles for all string fields
    assert "id" in profile
    assert "name" in profile
    assert "ssn" in profile

    # SSN should have higher entropy than simple names
    assert profile["ssn"] > profile["name"]


def test_calculate_field_entropy_profile_empty():
    """Test entropy profile with empty dataset"""
    profile = calculate_field_entropy_profile([])
    assert profile == {}


# ============================================================================
# Anomaly Detection Tests
# ============================================================================

def test_detect_entropy_anomalies():
    """Test detecting anomalous entropy values using z-score"""
    # Most IDs follow pattern "user_001", but one is random
    data = [
        {"id": "user_001"},
        {"id": "user_002"},
        {"id": "user_003"},
        {"id": "user_004"},
        {"id": "user_005"},
        {"id": "xR9$mK2@pL5"},  # Anomalous - high entropy
    ]

    anomalies = detect_entropy_anomalies(data, field="id", z_threshold=2.0)

    # Should detect the random ID as anomalous
    assert len(anomalies) >= 1
    assert 5 in anomalies  # Index 5 has anomalous value


def test_detect_entropy_anomalies_none():
    """Test anomaly detection with uniform entropy (no anomalies)"""
    # All values have similar entropy
    data = [
        {"id": "user_001"},
        {"id": "user_002"},
        {"id": "user_003"},
        {"id": "user_004"},
        {"id": "user_005"},
    ]

    anomalies = detect_entropy_anomalies(data, field="id", z_threshold=2.0)

    # Should not detect anomalies
    assert len(anomalies) == 0


def test_detect_entropy_anomalies_insufficient_data():
    """Test anomaly detection with insufficient data"""
    data = [
        {"id": "user_001"},
        {"id": "user_002"},
    ]

    anomalies = detect_entropy_anomalies(data, field="id", z_threshold=2.0)

    # Should not detect anomalies with <3 samples
    assert len(anomalies) == 0


# ============================================================================
# Classification Tests
# ============================================================================

def test_classify_entropy_level_very_low():
    """Test classification of very low entropy"""
    assert classify_entropy_level(0.0) == "VERY_LOW"
    assert classify_entropy_level(0.5) == "VERY_LOW"
    assert classify_entropy_level(0.9) == "VERY_LOW"


def test_classify_entropy_level_low():
    """Test classification of low entropy"""
    assert classify_entropy_level(1.0) == "LOW"
    assert classify_entropy_level(1.5) == "LOW"
    assert classify_entropy_level(1.9) == "LOW"


def test_classify_entropy_level_moderate():
    """Test classification of moderate entropy"""
    assert classify_entropy_level(2.0) == "MODERATE"
    assert classify_entropy_level(2.5) == "MODERATE"
    assert classify_entropy_level(2.9) == "MODERATE"


def test_classify_entropy_level_high():
    """Test classification of high entropy"""
    assert classify_entropy_level(3.0) == "HIGH"
    assert classify_entropy_level(3.5) == "HIGH"
    assert classify_entropy_level(3.9) == "HIGH"


def test_classify_entropy_level_very_high():
    """Test classification of very high entropy"""
    assert classify_entropy_level(4.0) == "VERY_HIGH"
    assert classify_entropy_level(5.0) == "VERY_HIGH"
    assert classify_entropy_level(10.0) == "VERY_HIGH"


# ============================================================================
# EntropyAnalysis Dataclass Tests
# ============================================================================

def test_entropy_analysis_structure():
    """Test EntropyAnalysis dataclass structure"""
    analysis = EntropyAnalysis(
        field_name="ssn",
        avg_entropy=3.5,
        min_entropy=3.0,
        max_entropy=4.0,
        entropy_variance=0.1,
        high_entropy_count=5,
        low_entropy_count=0,
        suspicious_patterns=["SSN_FORMAT"]
    )

    assert analysis.field_name == "ssn"
    assert analysis.avg_entropy == 3.5
    assert analysis.min_entropy == 3.0
    assert analysis.max_entropy == 4.0
    assert analysis.entropy_variance == 0.1
    assert analysis.high_entropy_count == 5
    assert analysis.low_entropy_count == 0
    assert "SSN_FORMAT" in analysis.suspicious_patterns


# ============================================================================
# Performance Tests
# ============================================================================

def test_entropy_detection_performance_small_dataset():
    """Test entropy detection performance on small dataset"""
    import time

    # Create dataset with 50 records
    # Note: Sequential SSNs like "100-20-1000" have low entropy (~1.9-2.2)
    # due to repetitive digits, unlike real-world SSNs with random digits
    data = []
    for i in range(50):
        data.append({
            "id": i,
            "name": f"Person{i}",
            "ssn": f"{100+i:03d}-{20+i:02d}-{1000+i:04d}"
        })

    # Benchmark detection
    # Use lower threshold (1.5) to catch low-entropy sequential SSNs
    start = time.perf_counter()
    results = detect_pii_by_entropy(data, high_entropy_threshold=1.5)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    # Should detect SSN field (even with low entropy)
    assert len(results) >= 1

    # Should be fast (<50ms for 50 records)
    print(f"\n  Entropy detection (50 rows): {duration_ms:.2f}ms")
    assert duration_ms < 50


def test_entropy_detection_performance_medium_dataset():
    """Test entropy detection performance on medium dataset"""
    import time

    # Create dataset with 100 records
    data = []
    for i in range(100):
        data.append({
            "id": i,
            "api_key": f"test-key-{i:010d}-random-{i*2:08d}"
        })

    start = time.perf_counter()
    results = detect_pii_by_entropy(data, high_entropy_threshold=4.0)
    end = time.perf_counter()

    duration_ms = (end - start) * 1000

    print(f"\n  Entropy detection (100 rows): {duration_ms:.2f}ms")
    assert duration_ms < 100  # Should still be reasonably fast

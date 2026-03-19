"""
Entropy-Based PII Detection using Shannon Entropy

Detects sensitive data by analyzing information entropy patterns.
Uses Shannon entropy to identify fields that may contain PII.

Author: DATAPIG Enhancement
Date: 2025-11-22
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class EntropyAnalysis:
    """Results of entropy analysis on a field"""
    field_name: str
    avg_entropy: float              # Average entropy across all values
    min_entropy: float              # Minimum entropy in dataset
    max_entropy: float              # Maximum entropy in dataset
    entropy_variance: float         # Variance in entropy values
    high_entropy_count: int         # Count of high-entropy values
    low_entropy_count: int          # Count of low-entropy values
    suspicious_patterns: list[str]  # List of detected suspicious patterns


def shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of a string.

    Shannon entropy H(X) = -Σ p(x) * log₂(p(x))
    where p(x) is the probability of character x

    Args:
        text: Input string

    Returns:
        Entropy value (0.0 = no entropy, higher = more random)

    Example:
        shannon_entropy("aaaa") -> ~0.0 (very low entropy)
        shannon_entropy("abcd") -> 2.0 (high entropy)
        shannon_entropy("123-45-6789") -> ~3.2 (SSN pattern)
    """
    if not text or len(text) == 0:
        return 0.0

    # Count character frequencies
    char_counts = Counter(text)
    length = len(text)

    # Calculate entropy
    entropy = 0.0
    for count in char_counts.values():
        if count == 0:
            continue
        probability = count / length
        entropy -= probability * math.log2(probability)

    return entropy


def normalized_entropy(text: str) -> float:
    """
    Calculate normalized Shannon entropy (0.0-1.0 scale).

    Normalized by maximum possible entropy (log₂(unique_chars)).

    Args:
        text: Input string

    Returns:
        Normalized entropy (0.0 = no randomness, 1.0 = maximum randomness)
    """
    if not text or len(text) == 0:
        return 0.0

    entropy = shannon_entropy(text)
    unique_chars = len(set(text))

    if unique_chars <= 1:
        return 0.0

    max_entropy = math.log2(unique_chars)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def detect_pii_by_entropy(
    data: list[dict[str, Any]],
    high_entropy_threshold: float = 3.5,
    low_entropy_threshold: float = 1.5,
    min_samples: int = 5
) -> list[EntropyAnalysis]:
    """
    Detect potential PII fields using entropy analysis.

    High entropy patterns (>3.5):
    - SSNs: 123-45-6789 (entropy ~3.2-3.5)
    - Credit cards: 4111-1111-1111-1111 (entropy ~3.5-4.0)
    - API keys: dGVzdC1hcGkta2V5LTEyMzQ1Njc4OQ== (entropy ~4.5+)
    - Random IDs: 7f8a9b2c-3d4e-5f6g-7h8i-9j0k1l2m3n4o (entropy ~4.5+)

    Low entropy patterns (<1.5):
    - Weak passwords: "password123" (entropy ~2.8)
    - Repeated data: "AAAA" (entropy ~0.0)
    - Template strings: "user_001" (entropy ~2.5)

    Args:
        data: List of dictionaries to analyze
        high_entropy_threshold: Entropy above this indicates potential PII (default 3.5)
        low_entropy_threshold: Entropy below this indicates weak security (default 1.5)
        min_samples: Minimum number of samples needed for analysis (default 5)

    Returns:
        List of EntropyAnalysis objects for suspicious fields
    """
    if len(data) < min_samples:
        return []

    # Collect entropy values per field
    field_entropies: dict[str, list[float]] = {}
    field_values: dict[str, list[str]] = {}

    for row in data:
        for field, value in row.items():
            # Only analyze string fields
            if not isinstance(value, str) or not value:
                continue

            if field not in field_entropies:
                field_entropies[field] = []
                field_values[field] = []

            entropy = shannon_entropy(value)
            field_entropies[field].append(entropy)
            field_values[field].append(value)

    # Analyze each field
    results = []

    for field, entropies in field_entropies.items():
        if len(entropies) < min_samples:
            continue

        # Calculate statistics
        avg_entropy = sum(entropies) / len(entropies)
        min_entropy = min(entropies)
        max_entropy = max(entropies)

        # Calculate variance
        variance = sum((e - avg_entropy) ** 2 for e in entropies) / len(entropies)

        # Count high/low entropy values
        high_count = sum(1 for e in entropies if e > high_entropy_threshold)
        low_count = sum(1 for e in entropies if e < low_entropy_threshold)

        # Detect suspicious patterns
        patterns = _detect_patterns(field_values[field], entropies, high_entropy_threshold)

        # Only report fields with suspicious characteristics
        if high_count > 0 or low_count > len(entropies) * 0.8 or patterns:
            results.append(EntropyAnalysis(
                field_name=field,
                avg_entropy=avg_entropy,
                min_entropy=min_entropy,
                max_entropy=max_entropy,
                entropy_variance=variance,
                high_entropy_count=high_count,
                low_entropy_count=low_count,
                suspicious_patterns=patterns
            ))

    return results


def _detect_patterns(values: list[str], entropies: list[float], threshold: float) -> list[str]:
    """
    Detect specific PII patterns using regex and entropy.

    Returns list of detected pattern types.
    """
    patterns = []

    # Check for high-entropy samples
    high_entropy_samples = [v for v, e in zip(values, entropies) if e > threshold]

    if not high_entropy_samples:
        return patterns

    # Sample a few high-entropy values for pattern detection
    sample = high_entropy_samples[:min(10, len(high_entropy_samples))]

    # SSN pattern: XXX-XX-XXXX
    if any(re.match(r'^\d{3}-\d{2}-\d{4}$', v) for v in sample):
        patterns.append("SSN_FORMAT")

    # Credit card pattern: XXXX-XXXX-XXXX-XXXX or 16 digits
    if any(re.match(r'^\d{4}-\d{4}-\d{4}-\d{4}$', v) or re.match(r'^\d{16}$', v) for v in sample):
        patterns.append("CREDIT_CARD_FORMAT")

    # API key pattern: base64-like strings
    if any(re.match(r'^[A-Za-z0-9+/]{20,}={0,2}$', v) for v in sample):
        patterns.append("API_KEY_FORMAT")

    # UUID pattern
    if any(re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', v.lower()) for v in sample):
        patterns.append("UUID_FORMAT")

    # Hash pattern: 32/40/64 hex chars
    if any(re.match(r'^[0-9a-f]{32}$|^[0-9a-f]{40}$|^[0-9a-f]{64}$', v.lower()) for v in sample):
        patterns.append("HASH_FORMAT")

    # Token pattern: random alphanumeric >20 chars
    if any(len(v) > 20 and re.match(r'^[A-Za-z0-9]{20,}$', v) for v in sample):
        patterns.append("TOKEN_FORMAT")

    return patterns


def calculate_field_entropy_profile(data: list[dict[str, Any]]) -> dict[str, float]:
    """
    Calculate average entropy for each field in dataset.

    Useful for comparing entropy profiles across different datasets.

    Args:
        data: List of dictionaries

    Returns:
        Dictionary mapping field names to average entropy
    """
    field_entropies: dict[str, list[float]] = {}

    for row in data:
        for field, value in row.items():
            if not isinstance(value, str) or not value:
                continue

            if field not in field_entropies:
                field_entropies[field] = []

            field_entropies[field].append(shannon_entropy(value))

    # Calculate averages
    return {
        field: sum(entropies) / len(entropies)
        for field, entropies in field_entropies.items()
        if entropies
    }


def detect_entropy_anomalies(
    data: list[dict[str, Any]],
    field: str,
    z_threshold: float = 2.0
) -> list[int]:
    """
    Detect values with anomalous entropy (z-score based).

    Args:
        data: List of dictionaries
        field: Field name to analyze
        z_threshold: Z-score threshold for anomaly (default 2.0 = 95% confidence)

    Returns:
        List of row indices with anomalous entropy values
    """
    # Extract entropies for field
    entropies = []
    valid_indices = []

    for idx, row in enumerate(data):
        value = row.get(field)
        if isinstance(value, str) and value:
            entropies.append(shannon_entropy(value))
            valid_indices.append(idx)

    if len(entropies) < 3:
        return []

    # Calculate mean and std dev
    mean = sum(entropies) / len(entropies)
    variance = sum((e - mean) ** 2 for e in entropies) / len(entropies)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0

    if std_dev == 0:
        return []

    # Find anomalies (z-score > threshold)
    anomalies = []
    for idx, entropy in zip(valid_indices, entropies):
        z_score = abs((entropy - mean) / std_dev)
        if z_score > z_threshold:
            anomalies.append(idx)

    return anomalies


def classify_entropy_level(entropy: float) -> str:
    """
    Classify entropy value into human-readable categories.

    Args:
        entropy: Shannon entropy value

    Returns:
        Classification string
    """
    if entropy < 1.0:
        return "VERY_LOW"      # Highly repetitive (e.g., "AAAA")
    elif entropy < 2.0:
        return "LOW"           # Low randomness (e.g., "user_001")
    elif entropy < 3.0:
        return "MODERATE"      # Moderate randomness (e.g., "password123")
    elif entropy < 4.0:
        return "HIGH"          # High randomness (e.g., "123-45-6789")
    else:
        return "VERY_HIGH"     # Very high randomness (e.g., API keys, UUIDs)

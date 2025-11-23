"""
DATAPIG Configuration System

"Computer, adjust diagnostic parameters to account for the anomaly."

Configurable thresholds and feature flags for all detection categories.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any


@dataclass
class DetectorConfig:
    """
    Configuration for DATAPIG detector

    All thresholds are configurable to adapt to different data domains.
    Feature flags enable/disable expensive detection methods.
    """

    # ========================================================================
    # Threshold Configuration
    # ========================================================================

    # Stale Data
    stale_threshold_days: int = 365  # 1 year default
    detect_future_dates: bool = True  # Flag impossible future timestamps

    # Outliers
    outlier_iqr_multiplier: float = 1.5  # IQR method: [Q1 - k*IQR, Q3 + k*IQR]
    outlier_method: str = "iqr"  # "iqr", "isolation_forest", "dbscan"

    # Sampling Bias
    imbalance_ratio_threshold: float = 10.0  # 10:1 majority:minority
    min_minority_class_pct: float = 10.0  # Minimum 10% representation

    # Distribution Shift
    rare_value_threshold: float = 0.01  # <1% frequency = rare
    min_category_diversity: int = 10  # Only flag if >10 unique values

    # Duplicates
    duplicate_method: str = "exact"  # "exact", "fuzzy", "subset"
    fuzzy_similarity_threshold: float = 0.9  # Levenshtein similarity >=90%
    duplicate_subset_columns: Optional[List[str]] = None  # Check specific columns

    # PII Detection
    pii_entropy_threshold: float = 4.0  # Shannon entropy >4.0 = potential secret
    pii_min_length: int = 16  # Minimum length for entropy-based detection

    # Schema Drift
    schema_detection_method: str = "first_row"  # "first_row", "statistical_mode", "explicit"
    explicit_schema: Optional[Dict[str, type]] = None  # Explicit schema if method="explicit"
    nullable_fields: Optional[List[str]] = None  # Fields that can be None

    # Label Noise
    label_noise_confidence_threshold: float = 0.5  # Only report if >50% contradictions

    # ========================================================================
    # Feature Flags
    # ========================================================================

    # Advanced detection methods (may be slower)
    enable_fuzzy_duplicates: bool = False  # Levenshtein distance matching
    enable_multivariate_outliers: bool = False  # Isolation Forest, DBSCAN
    enable_entropy_detection: bool = True  # High-entropy string detection
    enable_cardinality_checks: bool = True  # Too many unique values
    enable_null_pattern_analysis: bool = True  # Systematic missing data

    # New detection categories (Phase 2)
    enable_range_violations: bool = False  # Expected min/max bounds
    enable_temporal_ordering: bool = False  # Event sequence validation
    enable_unit_mismatches: bool = False  # Mixed units detection
    enable_encoding_issues: bool = False  # UTF-8 errors, mojibake

    # ========================================================================
    # Performance Tuning
    # ========================================================================

    sample_size: Optional[int] = None  # None = process all rows
    enable_caching: bool = True  # Cache intermediate results
    parallel_workers: int = 1  # Number of parallel detection threads
    batch_size: int = 1000  # Process in batches for large datasets

    # ========================================================================
    # Extensibility
    # ========================================================================

    custom_validators: List[Callable] = field(default_factory=list)  # User-defined validators
    custom_pii_patterns: Dict[str, str] = field(default_factory=dict)  # Custom regex patterns

    # Range violations (if enabled)
    expected_ranges: Dict[str, tuple] = field(default_factory=dict)  # {"field": (min, max)}

    # ========================================================================
    # Reporting
    # ========================================================================

    verbose: bool = False  # Print progress messages
    min_severity_to_report: str = "ENSIGN"  # CAPTAIN/COMMANDER/LIEUTENANT/ENSIGN


@dataclass
class PresetConfig:
    """Preset configurations for common use cases"""

    @staticmethod
    def default() -> DetectorConfig:
        """Default configuration (balanced)"""
        return DetectorConfig()

    @staticmethod
    def strict() -> DetectorConfig:
        """Strict configuration (maximum detection)"""
        return DetectorConfig(
            stale_threshold_days=180,  # 6 months
            outlier_iqr_multiplier=1.0,  # More sensitive
            imbalance_ratio_threshold=5.0,  # Stricter balance
            rare_value_threshold=0.05,  # <5% = rare
            enable_fuzzy_duplicates=True,
            enable_multivariate_outliers=True,
            enable_entropy_detection=True,
            enable_cardinality_checks=True,
            enable_null_pattern_analysis=True,
            min_severity_to_report="LIEUTENANT"  # Only medium+ issues
        )

    @staticmethod
    def lenient() -> DetectorConfig:
        """Lenient configuration (fewer false positives)"""
        return DetectorConfig(
            stale_threshold_days=730,  # 2 years
            outlier_iqr_multiplier=2.0,  # Less sensitive
            imbalance_ratio_threshold=20.0,  # Very unbalanced
            rare_value_threshold=0.001,  # <0.1% = rare
            enable_fuzzy_duplicates=False,
            enable_multivariate_outliers=False,
            enable_entropy_detection=False,
            min_severity_to_report="COMMANDER"  # Only high+ issues
        )

    @staticmethod
    def fast() -> DetectorConfig:
        """Fast configuration (performance-optimized)"""
        return DetectorConfig(
            enable_fuzzy_duplicates=False,
            enable_multivariate_outliers=False,
            enable_entropy_detection=False,
            enable_cardinality_checks=False,
            enable_null_pattern_analysis=False,
            sample_size=10000,  # Sample large datasets
            enable_caching=True,
            parallel_workers=4,  # Parallel processing
            batch_size=5000
        )

    @staticmethod
    def pii_focused() -> DetectorConfig:
        """PII-focused configuration (security audit)"""
        return DetectorConfig(
            pii_entropy_threshold=3.5,  # More sensitive
            pii_min_length=12,  # Shorter secrets
            enable_entropy_detection=True,
            custom_pii_patterns={
                "aws_key": r'AKIA[0-9A-Z]{16}',
                "jwt": r'eyJ[A-Za-z0-9-_=]+\.eyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_.+/=]*',
                "private_key": r'-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----',
                "database_url": r'(postgresql|mysql|mongodb)://[^\s]+',
            },
            min_severity_to_report="CAPTAIN"  # Only critical PII leaks
        )

    @staticmethod
    def ml_validation() -> DetectorConfig:
        """ML dataset validation configuration"""
        return DetectorConfig(
            imbalance_ratio_threshold=3.0,  # ML needs balanced classes
            rare_value_threshold=0.05,  # <5% = rare in ML
            enable_null_pattern_analysis=True,  # Missing data is critical
            enable_multivariate_outliers=True,  # Enable advanced outlier detection
            outlier_method="isolation_forest",  # ML-friendly outlier detection
            label_noise_confidence_threshold=0.3,  # Strict on label contradictions
            min_severity_to_report="LIEUTENANT"
        )


def create_config(preset: str = "default", **overrides) -> DetectorConfig:
    """
    Factory function to create configuration with presets + overrides

    Args:
        preset: "default", "strict", "lenient", "fast", "pii_focused", "ml_validation"
        **overrides: Override any config parameter

    Returns:
        DetectorConfig instance

    Example:
        config = create_config("strict", stale_threshold_days=90, verbose=True)
    """
    preset_map = {
        "default": PresetConfig.default,
        "strict": PresetConfig.strict,
        "lenient": PresetConfig.lenient,
        "fast": PresetConfig.fast,
        "pii_focused": PresetConfig.pii_focused,
        "ml_validation": PresetConfig.ml_validation,
    }

    if preset not in preset_map:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(preset_map.keys())}")

    base_config = preset_map[preset]()

    # Apply overrides
    for key, value in overrides.items():
        if hasattr(base_config, key):
            setattr(base_config, key, value)
        else:
            raise ValueError(f"Unknown config parameter: {key}")

    return base_config

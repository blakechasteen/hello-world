"""
Unit Tests for DATAPIG Configuration System

Tests configurable thresholds and preset configurations.

Author: HoloLoom Test Suite
Date: 2025-11-22
"""

import pytest
from HoloLoom.datapig.config import DetectorConfig, PresetConfig


# ============================================================================
# DetectorConfig Tests
# ============================================================================

def test_detector_config_defaults():
    """Test default configuration values"""
    config = DetectorConfig()

    assert config.stale_threshold_days == 365
    assert config.outlier_iqr_multiplier == 1.5
    assert config.imbalance_ratio_threshold == 10.0
    assert config.rare_value_threshold == 0.01
    assert config.enable_fuzzy_duplicates is False
    assert config.enable_entropy_detection is True


def test_detector_config_custom_values():
    """Test custom configuration values"""
    config = DetectorConfig(
        stale_threshold_days=180,
        outlier_iqr_multiplier=2.0,
        imbalance_ratio_threshold=5.0
    )

    assert config.stale_threshold_days == 180
    assert config.outlier_iqr_multiplier == 2.0
    assert config.imbalance_ratio_threshold == 5.0


# ============================================================================
# Preset Configurations Tests
# ============================================================================

def test_preset_default():
    """Test default preset configuration"""
    config = PresetConfig.default()

    assert config.stale_threshold_days == 365
    assert config.outlier_iqr_multiplier == 1.5
    assert config.imbalance_ratio_threshold == 10.0


def test_preset_strict():
    """Test strict preset (more sensitive thresholds)"""
    config = PresetConfig.strict()

    # Strict should have tighter thresholds
    assert config.stale_threshold_days < 365  # More aggressive stale detection
    assert config.outlier_iqr_multiplier < 1.5  # Detect more outliers
    assert config.imbalance_ratio_threshold < 10.0  # Detect more bias


def test_preset_lenient():
    """Test lenient preset (less sensitive thresholds)"""
    config = PresetConfig.lenient()

    # Lenient should have looser thresholds
    assert config.stale_threshold_days > 365  # Less aggressive stale detection
    assert config.outlier_iqr_multiplier > 1.5  # Detect fewer outliers
    assert config.imbalance_ratio_threshold > 10.0  # Detect less bias


def test_preset_fast():
    """Test fast preset (performance-optimized)"""
    config = PresetConfig.fast()

    # Fast should disable expensive checks
    assert config.enable_fuzzy_duplicates is False
    assert config.enable_entropy_detection is False


def test_preset_pii_focused():
    """Test PII-focused preset"""
    config = PresetConfig.pii_focused()

    # PII-focused should enable all PII detection
    assert config.enable_entropy_detection is True


def test_preset_ml_validation():
    """Test ML validation preset"""
    config = PresetConfig.ml_validation()

    # ML validation should be sensitive to bias and noise
    assert config.imbalance_ratio_threshold <= 10.0


# ============================================================================
# Config Integration Tests
# ============================================================================

def test_config_with_detector():
    """Test using config with DataPigDetector"""
    from HoloLoom.datapig import DataPigDetector

    config = PresetConfig.strict()
    detector = DataPigDetector(enable_verbose=False)

    # Detector should be created successfully
    assert detector is not None


def test_all_presets_valid():
    """Test that all presets produce valid configurations"""
    presets = [
        PresetConfig.default(),
        PresetConfig.strict(),
        PresetConfig.lenient(),
        PresetConfig.fast(),
        PresetConfig.pii_focused(),
        PresetConfig.ml_validation(),
    ]

    for config in presets:
        # All configs should have valid threshold values
        assert config.stale_threshold_days > 0
        assert config.outlier_iqr_multiplier > 0
        assert config.imbalance_ratio_threshold > 0
        assert 0 < config.rare_value_threshold < 1

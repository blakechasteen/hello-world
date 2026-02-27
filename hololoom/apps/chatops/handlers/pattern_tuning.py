#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pattern Detection Tuning System
================================
Configurable thresholds and patterns for proactive agent detection.

Features:
- Tunable confidence thresholds
- Custom pattern registration
- A/B testing framework
- Performance metrics
- Pattern effectiveness tracking

Usage:
    tuner = PatternTuner()

    # Adjust thresholds
    tuner.set_threshold("decision", confidence=0.7)

    # Add custom pattern
    tuner.add_pattern("decision", r"we should (.+)")

    # Evaluate effectiveness
    metrics = tuner.evaluate(test_messages)
"""

import logging
import re
import json
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class PatternConfig:
    """Configuration for a pattern category."""
    name: str
    patterns: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.6
    weight: float = 1.0
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionMetrics:
    """Metrics for pattern detection performance."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        """Calculate precision (TP / (TP + FP))."""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall (TP / (TP + FN))."""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score (2 * P * R / (P + R))."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives
        }


# ============================================================================
# Default Pattern Configurations
# ============================================================================

DEFAULT_PATTERNS = {
    "decision": PatternConfig(
        name="decision",
        patterns=[
            r"let['\s]+s\s+(use|go\s+with|do|implement|choose)\s+(.+)",
            r"we['\s]+(decided|agreed|will|should)\s+(.+)",
            r"(decision|agreed):\s*(.+)",
            r"final\s+decision",
            r"we['\s]+re\s+going\s+with\s+(.+)",
            r"(settled|confirmed|approved):\s*(.+)",
        ],
        confidence_threshold=0.7,
        weight=1.0,
        enabled=True
    ),

    "action_item": PatternConfig(
        name="action_item",
        patterns=[
            r"(todo|to\s+do|task):\s*(.+)",
            r"(need|needs)\s+to\s+(.+)",
            r"(should|must|will)\s+(.+)",
            r"@(\w+)[,\s]+(please|can\s+you)\s+(.+)",
            r"action\s+item:\s*(.+)",
            r"(assign|assigned)\s+(to\s+)?@?(\w+)",
        ],
        confidence_threshold=0.6,
        weight=1.0,
        enabled=True
    ),

    "question": PatternConfig(
        name="question",
        patterns=[
            r".+\?",
            r"(what|when|where|who|why|how)\s+.+",
            r"(can|could|would|should)\s+(you|we|anyone)\s+.+",
        ],
        confidence_threshold=0.5,
        weight=0.8,
        enabled=True
    ),

    "agreement": PatternConfig(
        name="agreement",
        patterns=[
            r"(agree|agreed|yes|yep|yeah|correct|exactly)",
            r"\+1|👍",
            r"sounds\s+good",
            r"makes\s+sense",
        ],
        confidence_threshold=0.6,
        weight=0.7,
        enabled=True
    ),

    "disagreement": PatternConfig(
        name="disagreement",
        patterns=[
            r"(disagree|no|nope|nah|incorrect)",
            r"-1|👎",
            r"(not\s+sure|don['\s]t\s+think)",
            r"(concern|worried|issue)\s+with",
        ],
        confidence_threshold=0.6,
        weight=0.7,
        enabled=True
    ),

    "urgent": PatternConfig(
        name="urgent",
        patterns=[
            r"(urgent|asap|immediately|critical|emergency)",
            r"🚨|⚠️",
            r"(need|require).+(now|asap|urgent)",
        ],
        confidence_threshold=0.8,
        weight=1.5,
        enabled=True
    ),
}


# ============================================================================
# Pattern Tuner
# ============================================================================

class PatternTuner:
    """
    Manages pattern detection configuration and tuning.

    Features:
    - Load/save pattern configurations
    - Adjust confidence thresholds
    - Add/remove patterns
    - A/B testing
    - Performance tracking

    Usage:
        tuner = PatternTuner()

        # Adjust threshold
        tuner.set_threshold("decision", 0.75)

        # Add custom pattern
        tuner.add_pattern("decision", r"we agreed on (.+)")

        # Save configuration
        tuner.save("pattern_config.json")
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize pattern tuner.

        Args:
            config_path: Optional path to load configuration from
        """
        self.patterns: Dict[str, PatternConfig] = {}
        self.metrics: Dict[str, DetectionMetrics] = {}

        # Load from file or use defaults
        if config_path and Path(config_path).exists():
            self.load(config_path)
        else:
            self.patterns = DEFAULT_PATTERNS.copy()

        logger.info(f"PatternTuner initialized with {len(self.patterns)} categories")

    # ========================================================================
    # Configuration Management
    # ========================================================================

    def set_threshold(self, category: str, confidence: float) -> None:
        """
        Set confidence threshold for a pattern category.

        Args:
            category: Pattern category name
            confidence: New threshold (0.0 to 1.0)
        """
        if category not in self.patterns:
            raise ValueError(f"Unknown category: {category}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {confidence}")

        self.patterns[category].confidence_threshold = confidence
        logger.info(f"Set {category} threshold to {confidence}")

    def set_weight(self, category: str, weight: float) -> None:
        """
        Set weight for a pattern category.

        Args:
            category: Pattern category name
            weight: New weight (typically 0.1 to 2.0)
        """
        if category not in self.patterns:
            raise ValueError(f"Unknown category: {category}")

        self.patterns[category].weight = weight
        logger.info(f"Set {category} weight to {weight}")

    def add_pattern(self, category: str, pattern: str) -> None:
        """
        Add a new pattern to a category.

        Args:
            category: Pattern category name
            pattern: Regex pattern string
        """
        if category not in self.patterns:
            raise ValueError(f"Unknown category: {category}")

        # Test pattern is valid
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")

        if pattern not in self.patterns[category].patterns:
            self.patterns[category].patterns.append(pattern)
            logger.info(f"Added pattern to {category}: {pattern}")

    def remove_pattern(self, category: str, pattern: str) -> None:
        """Remove a pattern from a category."""
        if category not in self.patterns:
            raise ValueError(f"Unknown category: {category}")

        if pattern in self.patterns[category].patterns:
            self.patterns[category].patterns.remove(pattern)
            logger.info(f"Removed pattern from {category}: {pattern}")

    def enable_category(self, category: str, enabled: bool = True) -> None:
        """Enable or disable a pattern category."""
        if category not in self.patterns:
            raise ValueError(f"Unknown category: {category}")

        self.patterns[category].enabled = enabled
        logger.info(f"{'Enabled' if enabled else 'Disabled'} category: {category}")

    # ========================================================================
    # Pattern Detection
    # ========================================================================

    def detect(
        self,
        text: str,
        category: str,
        return_confidence: bool = False
    ) -> bool | Tuple[bool, float]:
        """
        Detect if text matches patterns in category.

        Args:
            text: Text to check
            category: Pattern category
            return_confidence: Return (match, confidence) tuple

        Returns:
            Boolean match or (match, confidence) tuple
        """
        if category not in self.patterns:
            return (False, 0.0) if return_confidence else False

        config = self.patterns[category]

        # Skip if disabled
        if not config.enabled:
            return (False, 0.0) if return_confidence else False

        # Check patterns
        max_confidence = 0.0
        for pattern in config.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Simple confidence: pattern match gives base confidence
                # Could be enhanced with ML scoring
                confidence = min(1.0, config.weight * 0.8)
                max_confidence = max(max_confidence, confidence)

        # Check threshold
        matches = max_confidence >= config.confidence_threshold

        if return_confidence:
            return matches, max_confidence
        return matches

    def detect_all(
        self,
        text: str,
        enabled_only: bool = True
    ) -> Dict[str, Tuple[bool, float]]:
        """
        Detect all pattern categories in text.

        Args:
            text: Text to check
            enabled_only: Only check enabled categories

        Returns:
            Dict of category -> (match, confidence)
        """
        results = {}

        for category, config in self.patterns.items():
            if enabled_only and not config.enabled:
                continue

            matches, confidence = self.detect(text, category, return_confidence=True)
            results[category] = (matches, confidence)

        return results

    # ========================================================================
    # Metrics & Evaluation
    # ========================================================================

    def record_result(
        self,
        category: str,
        predicted: bool,
        actual: bool
    ) -> None:
        """
        Record a detection result for metrics.

        Args:
            category: Pattern category
            predicted: What detector predicted
            actual: Ground truth
        """
        if category not in self.metrics:
            self.metrics[category] = DetectionMetrics()

        metrics = self.metrics[category]

        if predicted and actual:
            metrics.true_positives += 1
        elif predicted and not actual:
            metrics.false_positives += 1
        elif not predicted and actual:
            metrics.false_negatives += 1
        else:
            metrics.true_negatives += 1

    def get_metrics(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detection metrics.

        Args:
            category: Specific category or None for all

        Returns:
            Metrics dictionary
        """
        if category:
            if category not in self.metrics:
                return {}
            return self.metrics[category].to_dict()

        # All categories
        return {
            cat: metrics.to_dict()
            for cat, metrics in self.metrics.items()
        }

    def evaluate(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, DetectionMetrics]:
        """
        Evaluate detector on test cases.

        Args:
            test_cases: List of {"text": str, "category": str, "expected": bool}

        Returns:
            Metrics per category
        """
        # Reset metrics
        self.metrics = {}

        for case in test_cases:
            text = case["text"]
            category = case["category"]
            expected = case["expected"]

            predicted = self.detect(text, category)
            self.record_result(category, predicted, expected)

        return self.metrics

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Output file path
        """
        config_data = {
            "patterns": {
                name: {
                    "patterns": cfg.patterns,
                    "confidence_threshold": cfg.confidence_threshold,
                    "weight": cfg.weight,
                    "enabled": cfg.enabled,
                    "metadata": cfg.metadata
                }
                for name, cfg in self.patterns.items()
            },
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Saved configuration to {path}")

    def load(self, path: str) -> None:
        """
        Load configuration from JSON file.

        Args:
            path: Input file path
        """
        with open(path, 'r') as f:
            config_data = json.load(f)

        self.patterns = {}
        for name, cfg_dict in config_data["patterns"].items():
            self.patterns[name] = PatternConfig(
                name=name,
                patterns=cfg_dict["patterns"],
                confidence_threshold=cfg_dict["confidence_threshold"],
                weight=cfg_dict["weight"],
                enabled=cfg_dict["enabled"],
                metadata=cfg_dict.get("metadata", {})
            )

        logger.info(f"Loaded configuration from {path}")

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        return {
            "categories": len(self.patterns),
            "total_patterns": sum(len(cfg.patterns) for cfg in self.patterns.values()),
            "enabled_categories": sum(1 for cfg in self.patterns.values() if cfg.enabled),
            "categories_detail": {
                name: {
                    "patterns": len(cfg.patterns),
                    "threshold": cfg.confidence_threshold,
                    "weight": cfg.weight,
                    "enabled": cfg.enabled
                }
                for name, cfg in self.patterns.items()
            }
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("="*80)
    print("Pattern Detection Tuning Demo")
    print("="*80)
    print()

    # Create tuner
    tuner = PatternTuner()

    # Show initial config
    print("Initial Configuration:")
    summary = tuner.get_summary()
    print(f"  Categories: {summary['categories']}")
    print(f"  Total patterns: {summary['total_patterns']}")
    print()

    for category, details in summary['categories_detail'].items():
        print(f"  {category}:")
        print(f"    Patterns: {details['patterns']}")
        print(f"    Threshold: {details['threshold']}")
        print(f"    Weight: {details['weight']}")
        print(f"    Enabled: {details['enabled']}")
    print()

    # Test detection
    print("Testing Detection:")
    test_messages = [
        "Let's use Matrix for the chatops integration",
        "We decided to implement threading support",
        "TODO: Add unit tests for the bot",
        "What about rate limiting?",
        "I agree with that approach",
        "🚨 Critical bug in production!"
    ]

    for msg in test_messages:
        print(f"\n  Message: {msg}")
        results = tuner.detect_all(msg)
        for category, (match, confidence) in results.items():
            if match:
                print(f"    ✓ {category}: {confidence:.2f}")
    print()

    # Tune thresholds
    print("Tuning Thresholds:")
    tuner.set_threshold("decision", 0.75)
    tuner.set_threshold("action_item", 0.65)
    tuner.set_weight("urgent", 2.0)
    print("  ✓ Adjusted thresholds and weights")
    print()

    # Add custom pattern
    print("Adding Custom Pattern:")
    tuner.add_pattern("decision", r"we should (.+)")
    print("  ✓ Added custom decision pattern")
    print()

    # Evaluate
    print("Evaluation on Test Cases:")
    test_cases = [
        {"text": "Let's use Matrix", "category": "decision", "expected": True},
        {"text": "TODO: implement bot", "category": "action_item", "expected": True},
        {"text": "What about this?", "category": "question", "expected": True},
        {"text": "Just a comment", "category": "decision", "expected": False},
    ]

    metrics = tuner.evaluate(test_cases)
    for category, metric in metrics.items():
        print(f"\n  {category}:")
        print(f"    Precision: {metric.precision:.2%}")
        print(f"    Recall: {metric.recall:.2%}")
        print(f"    F1 Score: {metric.f1_score:.2%}")
    print()

    # Save configuration
    print("Saving Configuration:")
    tuner.save("demo_pattern_config.json")
    print("  ✓ Saved to demo_pattern_config.json")
    print()

    print("✓ Demo complete!")

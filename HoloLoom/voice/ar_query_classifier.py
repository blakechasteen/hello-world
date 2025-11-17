"""
AR Query Classifier - Adaptive routing for Elle AR queries
===========================================================
Classifies AR queries by complexity and AR-specific types with confidence scoring.

Author: Claude Code (Agent Q)
Date: November 17, 2025
Phase: Agent Swarm Wave 5 - Adaptive Routing Integration
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class ARComplexity(Enum):
    """AR query complexity levels."""
    SIMPLE = "simple"          # "Show me hive 5", "Next hive"
    STANDARD = "standard"      # "What's the health status?", "Compare hives"
    COMPLEX = "complex"        # "Show me all unhealthy hives", "Find patterns"
    RESEARCH = "research"      # "Analyze trends", "Explain why..."


class ARQueryType(Enum):
    """AR-specific query types."""
    GESTURE_COMMAND = "gesture_command"      # Gesture-driven command
    VISUAL_QUERY = "visual_query"            # "What's this?", "Explain that"
    SPATIAL_REFERENCE = "spatial_reference"  # "Over there", "That one"
    MULTIMODAL = "multimodal"                # Combination of voice+gesture+vision
    VOICE_ONLY = "voice_only"                # Pure voice command


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ARClassificationResult:
    """Result of AR query classification."""
    query: str
    complexity: ARComplexity
    ar_type: ARQueryType
    confidence: float
    reasoning: str = ""
    metadata: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "complexity": self.complexity.value,
            "ar_type": self.ar_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


@dataclass
class ARPatternRule:
    """Pattern rule for AR query classification."""
    pattern: str  # Regex pattern
    complexity: ARComplexity
    ar_type: ARQueryType
    confidence_boost: float = 0.0
    priority: int = 0
    examples: List[str] = field(default_factory=list)


# ============================================================================
# AR Query Classifier
# ============================================================================

class ARQueryClassifier:
    """
    Adaptive classifier for Elle AR queries.

    Features:
    - 4 complexity levels: SIMPLE, STANDARD, COMPLEX, RESEARCH
    - 5 AR-specific query types: GESTURE_COMMAND, VISUAL_QUERY, SPATIAL_REFERENCE, MULTIMODAL, VOICE_ONLY
    - Confidence scoring based on pattern matching + feature analysis
    - Automatic logging for pattern mining
    - Integration with HoloLoom adaptive learning system

    Usage:
        classifier = ARQueryClassifier(enable_logging=True)
        result = classifier.classify(
            query="Show me all unhealthy hives",
            context={"has_gesture": False, "has_vision": False}
        )
        print(f"Complexity: {result.complexity.value}, Confidence: {result.confidence:.2f}")
    """

    def __init__(
        self,
        enable_logging: bool = True,
        log_dir: Optional[str] = None,
        custom_patterns: Optional[List[ARPatternRule]] = None
    ):
        """
        Initialize AR query classifier.

        Args:
            enable_logging: Enable classification logging for pattern mining
            log_dir: Directory for classification logs (default: ./ar_logs)
            custom_patterns: Optional custom pattern rules
        """
        self.enable_logging = enable_logging
        self.log_dir = Path(log_dir) if log_dir else Path("./ar_logs")
        if self.enable_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.log_dir / f"classifications_{datetime.now():%Y%m%d}.jsonl"

        # Statistics
        self.stats = defaultdict(int)

        # Load patterns
        self.patterns = self._initialize_patterns()
        if custom_patterns:
            self.patterns.extend(custom_patterns)

        # Sort patterns by priority (higher priority first)
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

    def _initialize_patterns(self) -> List[ARPatternRule]:
        """Initialize default AR pattern rules."""
        patterns = []

        # ========================================
        # SIMPLE Patterns (Basic commands)
        # ========================================

        # Navigation commands
        patterns.append(ARPatternRule(
            pattern=r'\b(next|previous|back|forward)\s+(hive|bee|colony)\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.15,
            priority=10,
            examples=["next hive", "previous colony", "back"]
        ))

        # Direct selection
        patterns.append(ARPatternRule(
            pattern=r'\bshow\s+(me\s+)?(hive|bee|colony)\s+\d+\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.15,
            priority=10,
            examples=["show me hive 5", "show hive 12"]
        ))

        # Spatial references
        patterns.append(ARPatternRule(
            pattern=r'\b(this|that|here|there|over\s+there)\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.SPATIAL_REFERENCE,
            confidence_boost=0.20,
            priority=15,
            examples=["this one", "that hive", "over there"]
        ))

        # Gesture commands
        patterns.append(ARPatternRule(
            pattern=r'\b(select|choose|pick)\s+(this|that)\b',
            complexity=ARComplexity.SIMPLE,
            ar_type=ARQueryType.GESTURE_COMMAND,
            confidence_boost=0.20,
            priority=12,
            examples=["select this", "pick that one"]
        ))

        # ========================================
        # STANDARD Patterns (Questions, Status)
        # ========================================

        # What questions
        patterns.append(ARPatternRule(
            pattern=r'\bwhat\s+is\s+(the\s+)?(health|status|condition|state)\b',
            complexity=ARComplexity.STANDARD,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.12,
            priority=8,
            examples=["what is the health status", "what's the condition"]
        ))

        # Visual queries
        patterns.append(ARPatternRule(
            pattern=r'\b(what\'?s|what\s+is)\s+(this|that)\b',
            complexity=ARComplexity.STANDARD,
            ar_type=ARQueryType.VISUAL_QUERY,
            confidence_boost=0.18,
            priority=10,
            examples=["what's this", "what is that"]
        ))

        # Comparison queries
        patterns.append(ARPatternRule(
            pattern=r'\bcompare\s+',
            complexity=ARComplexity.STANDARD,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.10,
            priority=7,
            examples=["compare hives", "compare health levels"]
        ))

        # Status queries
        patterns.append(ARPatternRule(
            pattern=r'\b(how|show)\s+(many|much)\b',
            complexity=ARComplexity.STANDARD,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.10,
            priority=7,
            examples=["how many bees", "show me how much honey"]
        ))

        # ========================================
        # COMPLEX Patterns (Multi-entity, Filtering)
        # ========================================

        # All/every queries
        patterns.append(ARPatternRule(
            pattern=r'\b(all|every)\s+\w+',
            complexity=ARComplexity.COMPLEX,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.15,
            priority=9,
            examples=["all unhealthy hives", "every colony"]
        ))

        # Filter queries
        patterns.append(ARPatternRule(
            pattern=r'\b(show|find)\s+(me\s+)?(all\s+)?\w+\s+(with|that|where)\b',
            complexity=ARComplexity.COMPLEX,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.12,
            priority=8,
            examples=["show me all hives with low health", "find bees that are swarming"]
        ))

        # Pattern detection
        patterns.append(ARPatternRule(
            pattern=r'\b(find|detect|identify)\s+(pattern|trend|anomal)',
            complexity=ARComplexity.COMPLEX,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.15,
            priority=9,
            examples=["find patterns in bee behavior", "detect anomalies"]
        ))

        # Multi-step commands
        patterns.append(ARPatternRule(
            pattern=r'\b(then|and then|after that)\b',
            complexity=ARComplexity.COMPLEX,
            ar_type=ARQueryType.MULTIMODAL,
            confidence_boost=0.12,
            priority=8,
            examples=["show this hive and then compare it", "select that and then analyze"]
        ))

        # ========================================
        # RESEARCH Patterns (Analysis, Explanation)
        # ========================================

        # Why questions
        patterns.append(ARPatternRule(
            pattern=r'\bwhy\s+(is|are|do|does)\b',
            complexity=ARComplexity.RESEARCH,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.18,
            priority=10,
            examples=["why is this hive unhealthy", "why are the bees swarming"]
        ))

        # Explain queries
        patterns.append(ARPatternRule(
            pattern=r'\b(explain|describe|analyze)\b',
            complexity=ARComplexity.RESEARCH,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.18,
            priority=10,
            examples=["explain the health decline", "analyze trends"]
        ))

        # Trend analysis
        patterns.append(ARPatternRule(
            pattern=r'\b(trend|history|over\s+time|timeline)\b',
            complexity=ARComplexity.RESEARCH,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.15,
            priority=9,
            examples=["show trends over time", "analyze historical data"]
        ))

        # Root cause
        patterns.append(ARPatternRule(
            pattern=r'\b(root\s+cause|cause\s+of|reason\s+for)\b',
            complexity=ARComplexity.RESEARCH,
            ar_type=ARQueryType.VOICE_ONLY,
            confidence_boost=0.18,
            priority=10,
            examples=["find root cause", "what's the reason for the decline"]
        ))

        return patterns

    def classify(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> ARClassificationResult:
        """
        Classify AR query by complexity and type.

        Args:
            query: Query text
            context: Optional context dict with:
                - has_gesture: bool (gesture input detected)
                - has_vision: bool (visual input detected)
                - has_spatial: bool (spatial reference detected)
                - gaze_target: str (what user is looking at)

        Returns:
            ARClassificationResult with complexity, type, and confidence
        """
        context = context or {}
        query_lower = query.lower().strip()

        # Initialize scoring
        complexity_scores = defaultdict(float)
        ar_type_scores = defaultdict(float)
        matched_patterns = []

        # Pattern matching
        for pattern_rule in self.patterns:
            if re.search(pattern_rule.pattern, query_lower):
                matched_patterns.append(pattern_rule)

                # Add base score
                complexity_scores[pattern_rule.complexity] += 1.0
                ar_type_scores[pattern_rule.ar_type] += 1.0

                # Add confidence boost
                complexity_scores[pattern_rule.complexity] += pattern_rule.confidence_boost
                ar_type_scores[pattern_rule.ar_type] += pattern_rule.confidence_boost

        # Context-based AR type detection
        if context.get("has_gesture", False):
            ar_type_scores[ARQueryType.GESTURE_COMMAND] += 0.5
            ar_type_scores[ARQueryType.MULTIMODAL] += 0.3

        if context.get("has_vision", False):
            ar_type_scores[ARQueryType.VISUAL_QUERY] += 0.5
            ar_type_scores[ARQueryType.MULTIMODAL] += 0.3

        if context.get("has_spatial", False):
            ar_type_scores[ARQueryType.SPATIAL_REFERENCE] += 0.5

        # Feature-based complexity scoring
        query_features = self._extract_features(query_lower, context)

        # Word count complexity
        word_count = len(query_lower.split())
        if word_count <= 3:
            complexity_scores[ARComplexity.SIMPLE] += 0.3
        elif word_count <= 7:
            complexity_scores[ARComplexity.STANDARD] += 0.3
        elif word_count <= 12:
            complexity_scores[ARComplexity.COMPLEX] += 0.3
        else:
            complexity_scores[ARComplexity.RESEARCH] += 0.3

        # Question word complexity
        if query_features["has_why"]:
            complexity_scores[ARComplexity.RESEARCH] += 0.4
        elif query_features["has_what"]:
            complexity_scores[ARComplexity.STANDARD] += 0.2
        elif query_features["has_how"]:
            complexity_scores[ARComplexity.STANDARD] += 0.3

        # Multi-entity detection
        if query_features["has_quantifier"]:
            complexity_scores[ARComplexity.COMPLEX] += 0.3

        # Causal reasoning
        if query_features["has_causation"]:
            complexity_scores[ARComplexity.RESEARCH] += 0.4

        # Select best complexity
        if complexity_scores:
            complexity = max(complexity_scores.items(), key=lambda x: x[1])[0]
            max_complexity_score = complexity_scores[complexity]
            total_complexity_score = sum(complexity_scores.values())
            complexity_confidence = max_complexity_score / total_complexity_score if total_complexity_score > 0 else 0.5
        else:
            # Default fallback
            complexity = ARComplexity.STANDARD
            complexity_confidence = 0.3

        # Select best AR type
        if ar_type_scores:
            ar_type = max(ar_type_scores.items(), key=lambda x: x[1])[0]
            max_ar_type_score = ar_type_scores[ar_type]
            total_ar_type_score = sum(ar_type_scores.values())
            ar_type_confidence = max_ar_type_score / total_ar_type_score if total_ar_type_score > 0 else 0.5
        else:
            # Default fallback
            ar_type = ARQueryType.VOICE_ONLY
            ar_type_confidence = 0.3

        # Combined confidence (weighted average)
        confidence = (complexity_confidence * 0.6 + ar_type_confidence * 0.4)

        # Build reasoning
        reasoning_parts = []
        if matched_patterns:
            reasoning_parts.append(f"Matched {len(matched_patterns)} pattern(s)")
        reasoning_parts.append(f"Word count: {word_count}")
        if query_features["has_why"]:
            reasoning_parts.append("Contains 'why' (research)")
        if query_features["has_quantifier"]:
            reasoning_parts.append("Contains quantifier (complex)")
        if context.get("has_gesture"):
            reasoning_parts.append("Gesture input detected")
        if context.get("has_vision"):
            reasoning_parts.append("Visual input detected")

        reasoning = "; ".join(reasoning_parts)

        # Create result
        result = ARClassificationResult(
            query=query,
            complexity=complexity,
            ar_type=ar_type,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "matched_patterns": len(matched_patterns),
                "word_count": word_count,
                "features": query_features,
                "context": context,
                "complexity_scores": {k.value: v for k, v in complexity_scores.items()},
                "ar_type_scores": {k.value: v for k, v in ar_type_scores.items()}
            }
        )

        # Log classification
        if self.enable_logging:
            self._log_classification(result)

        # Update statistics
        self.stats["total"] += 1
        self.stats[f"complexity_{complexity.value}"] += 1
        self.stats[f"ar_type_{ar_type.value}"] += 1

        return result

    def _extract_features(self, query: str, context: Dict) -> Dict:
        """Extract linguistic features from query."""
        features = {
            "has_why": bool(re.search(r'\bwhy\b', query)),
            "has_what": bool(re.search(r'\bwhat\b', query)),
            "has_how": bool(re.search(r'\bhow\b', query)),
            "has_where": bool(re.search(r'\bwhere\b', query)),
            "has_when": bool(re.search(r'\bwhen\b', query)),
            "has_quantifier": bool(re.search(r'\b(all|every|many|most|some)\b', query)),
            "has_causation": bool(re.search(r'\b(because|cause|reason|due\s+to)\b', query)),
            "has_comparison": bool(re.search(r'\b(compare|versus|vs|than)\b', query)),
            "has_negation": bool(re.search(r'\b(not|no|never|none)\b', query)),
            "has_imperative": bool(re.search(r'\b(show|find|get|tell|explain)\b', query)),
        }
        return features

    def _log_classification(self, result: ARClassificationResult) -> None:
        """Log classification result to JSONL file."""
        try:
            with open(self.log_file, "a") as f:
                log_entry = result.to_dict()
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log classification: {e}")

    def get_statistics(self) -> Dict:
        """Get classification statistics."""
        return dict(self.stats)

    def add_custom_pattern(self, pattern: ARPatternRule) -> None:
        """Add a custom pattern rule."""
        self.patterns.append(pattern)
        # Re-sort by priority
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

    def load_patterns_from_file(self, filepath: str) -> None:
        """Load custom patterns from JSON file."""
        try:
            with open(filepath, "r") as f:
                patterns_data = json.load(f)

            for p in patterns_data:
                pattern = ARPatternRule(
                    pattern=p["pattern"],
                    complexity=ARComplexity(p["complexity"]),
                    ar_type=ARQueryType(p["ar_type"]),
                    confidence_boost=p.get("confidence_boost", 0.0),
                    priority=p.get("priority", 0),
                    examples=p.get("examples", [])
                )
                self.add_custom_pattern(pattern)

            logger.info(f"Loaded {len(patterns_data)} custom patterns from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load patterns from {filepath}: {e}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_ar_classifier(
    enable_logging: bool = True,
    log_dir: Optional[str] = None
) -> ARQueryClassifier:
    """
    Factory function to create AR query classifier.

    Args:
        enable_logging: Enable classification logging
        log_dir: Directory for logs

    Returns:
        ARQueryClassifier instance
    """
    return ARQueryClassifier(
        enable_logging=enable_logging,
        log_dir=log_dir
    )

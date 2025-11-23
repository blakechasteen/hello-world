# Option A: BossPig Business Slop Detector - Comprehensive Roadmap

**Duration**: 2 weeks (10 working days)
**Team Size**: 1 developer (concurrent with Options B & C)
**Effort**: 40 hours total (4 hours/day)
**Status**: Ready to begin
**Date Created**: 2025-11-20

---

## Executive Summary

BossPig is a business document quality analyzer that detects 15 categories of "business slop" - jargon, vague language, buzzwords, and other low-quality writing patterns. It provides real-time scoring (0-100) and actionable feedback.

**Value Proposition**:
- **Unique**: No competitors have AI slop detection for business docs
- **Monetizable**: SaaS tiers already designed ($50-$500/month)
- **Quick to market**: 2-week MVP, leverages existing HoloLoom infrastructure
- **Real problem**: Business writing degraded by AI-generated slop

---

## Architecture Overview

```
Business Document (PDF, DOCX, MD)
    ↓
[SpinningWheel Adapter] (existing - HoloLoom)
    ↓
[Text Extraction & Chunking]
    ↓
[BossPig Detector] (15 slop categories)
    ├─ Jargon Detector
    ├─ Vague Commitment Detector
    ├─ Buzzword Detector
    ├─ Passive Voice Detector
    ├─ Weasel Words Detector
    ├─ Superlatives Detector
    ├─ Empty Phrases Detector
    ├─ Corporate Speak Detector
    ├─ Hedge Words Detector
    ├─ Filler Content Detector
    ├─ Redundancy Detector
    ├─ Name Dropping Detector
    ├─ False Urgency Detector
    ├─ Vague Metrics Detector
    └─ AI Fingerprints Detector
    ↓
[Quality Scorer] (0-100)
    ↓
[Feedback Generator]
    ↓
[CLI Output / JSON Report]
```

---

## Phase 1: Foundation (Days 1-2) - 8 hours

### Day 1: Project Setup & Core Infrastructure (4 hours)

**Task 1.1: Directory Structure (30 min)**
```
bosspig/
├── __init__.py
├── detector.py          # Main detection engine
├── scorer.py            # Quality scoring system
├── categories/          # Individual detectors
│   ├── __init__.py
│   ├── jargon.py
│   ├── vague_commitments.py
│   ├── buzzwords.py
│   └── ... (12 more)
├── cli.py              # Command-line interface
├── config.py           # Configuration
├── utils.py            # Shared utilities
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   ├── test_scorer.py
│   └── test_categories/
└── demo_documents/     # Sample docs for testing
```

**Verification**:
- [ ] All directories created
- [ ] __init__.py in each package
- [ ] README.md with architecture overview

**Task 1.2: Configuration System (1 hour)**

Create `bosspig/config.py`:
```python
@dataclass
class BossPigConfig:
    """Configuration for BossPig detector."""

    # Detection thresholds
    jargon_threshold: float = 0.7
    vague_threshold: float = 0.6
    buzzword_threshold: float = 0.8

    # Scoring weights (must sum to 1.0)
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        'jargon': 0.10,
        'vague_commitments': 0.12,
        'buzzwords': 0.08,
        'passive_voice': 0.07,
        'weasel_words': 0.09,
        'superlatives': 0.06,
        'empty_phrases': 0.08,
        'corporate_speak': 0.10,
        'hedge_words': 0.06,
        'filler_content': 0.07,
        'redundancy': 0.05,
        'name_dropping': 0.04,
        'false_urgency': 0.04,
        'vague_metrics': 0.03,
        'ai_fingerprints': 0.01,
    })

    # Output settings
    output_format: str = 'text'  # text, json, html
    show_suggestions: bool = True
    show_examples: bool = True
    verbose: bool = False
```

**Verification**:
- [ ] Config weights sum to 1.0
- [ ] All 15 categories represented
- [ ] Valid defaults set

**Task 1.3: Core Protocol Definitions (1.5 hours)**

Create `bosspig/detector.py` with protocols:
```python
from typing import Protocol, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

class SlopCategory(Enum):
    """15 categories of business slop."""
    JARGON = "jargon"
    VAGUE_COMMITMENTS = "vague_commitments"
    BUZZWORDS = "buzzwords"
    PASSIVE_VOICE = "passive_voice"
    WEASEL_WORDS = "weasel_words"
    SUPERLATIVES = "superlatives"
    EMPTY_PHRASES = "empty_phrases"
    CORPORATE_SPEAK = "corporate_speak"
    HEDGE_WORDS = "hedge_words"
    FILLER_CONTENT = "filler_content"
    REDUNDANCY = "redundancy"
    NAME_DROPPING = "name_dropping"
    FALSE_URGENCY = "false_urgency"
    VAGUE_METRICS = "vague_metrics"
    AI_FINGERPRINTS = "ai_fingerprints"

@dataclass
class SlopDetection:
    """Single instance of detected slop."""
    category: SlopCategory
    text: str
    span: tuple[int, int]
    confidence: float
    severity: float  # 0.0-1.0
    suggestion: str
    example: str

@dataclass
class DocumentAnalysis:
    """Complete analysis of a business document."""
    document_id: str
    total_words: int
    total_sentences: int
    detections: List[SlopDetection]
    quality_score: float  # 0-100
    category_scores: Dict[SlopCategory, float]
    overall_assessment: str
    top_issues: List[str]
    suggestions: List[str]

class CategoryDetector(Protocol):
    """Protocol for individual slop category detectors."""

    def detect(self, text: str) -> List[SlopDetection]:
        """Detect instances of this slop category."""
        ...

    def get_severity(self, detection: str) -> float:
        """Calculate severity (0.0-1.0) for detected instance."""
        ...
```

**Verification**:
- [ ] All 15 categories in enum
- [ ] Protocol defines clear interface
- [ ] Dataclasses have all required fields

**Task 1.4: Main Detector Engine (1 hour)**

Create core detection orchestrator in `bosspig/detector.py`:
```python
class BossPigDetector:
    """Main business slop detection engine."""

    def __init__(self, config: BossPigConfig):
        self.config = config
        self.detectors: Dict[SlopCategory, CategoryDetector] = {}
        self._load_detectors()

    def _load_detectors(self):
        """Load all 15 category detectors."""
        from bosspig.categories import (
            JargonDetector, VagueCommitmentDetector, BuzzwordDetector,
            # ... load all 15
        )

        self.detectors[SlopCategory.JARGON] = JargonDetector()
        self.detectors[SlopCategory.VAGUE_COMMITMENTS] = VagueCommitmentDetector()
        # ... register all 15

    def analyze(self, text: str, document_id: str = None) -> DocumentAnalysis:
        """Analyze business document for slop."""
        # Preprocessing
        words = text.split()
        sentences = text.split('.')

        # Run all detectors
        all_detections = []
        for category, detector in self.detectors.items():
            detections = detector.detect(text)
            all_detections.extend(detections)

        # Score document
        from bosspig.scorer import QualityScorer
        scorer = QualityScorer(self.config)
        quality_score = scorer.score(all_detections, len(words))
        category_scores = scorer.score_by_category(all_detections)

        # Generate assessment
        assessment = self._generate_assessment(quality_score, category_scores)
        top_issues = self._identify_top_issues(category_scores)
        suggestions = self._generate_suggestions(all_detections)

        return DocumentAnalysis(
            document_id=document_id or "unknown",
            total_words=len(words),
            total_sentences=len(sentences),
            detections=all_detections,
            quality_score=quality_score,
            category_scores=category_scores,
            overall_assessment=assessment,
            top_issues=top_issues,
            suggestions=suggestions
        )
```

**Verification**:
- [ ] All 15 detectors registered
- [ ] analyze() returns complete DocumentAnalysis
- [ ] Error handling for missing detectors

### Day 2: Quality Scoring System (4 hours)

**Task 2.1: Scorer Implementation (2 hours)**

Create `bosspig/scorer.py`:
```python
class QualityScorer:
    """Calculates quality scores (0-100) from detections."""

    def __init__(self, config: BossPigConfig):
        self.config = config

    def score(self, detections: List[SlopDetection], total_words: int) -> float:
        """Calculate overall quality score (0-100)."""

        # Start at 100 (perfect)
        score = 100.0

        # Penalize based on detections
        for detection in detections:
            # Penalty = severity × weight × frequency_factor
            category = detection.category
            weight = self.config.category_weights[category.value]
            severity = detection.severity

            penalty = severity * weight * 10  # Max 10 points per detection
            score -= penalty

        # Apply frequency penalty (too many issues = exponential penalty)
        slop_density = len(detections) / max(total_words, 1)
        if slop_density > 0.05:  # More than 5% of words are slop
            frequency_penalty = (slop_density - 0.05) * 50
            score -= frequency_penalty

        # Clamp to [0, 100]
        return max(0.0, min(100.0, score))

    def score_by_category(
        self,
        detections: List[SlopDetection]
    ) -> Dict[SlopCategory, float]:
        """Calculate per-category scores."""

        category_detections = {}
        for detection in detections:
            if detection.category not in category_detections:
                category_detections[detection.category] = []
            category_detections[detection.category].append(detection)

        category_scores = {}
        for category in SlopCategory:
            if category not in category_detections:
                category_scores[category] = 100.0  # Perfect if no detections
            else:
                # Score = 100 - (avg_severity × count × 10)
                dets = category_detections[category]
                avg_severity = sum(d.severity for d in dets) / len(dets)
                penalty = avg_severity * len(dets) * 10
                category_scores[category] = max(0.0, 100.0 - penalty)

        return category_scores

    def get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade."""
        if score >= 90: return "A (Excellent)"
        if score >= 80: return "B (Good)"
        if score >= 70: return "C (Acceptable)"
        if score >= 60: return "D (Poor)"
        return "F (Unacceptable)"
```

**Verification**:
- [ ] Score always in [0, 100]
- [ ] Category weights properly applied
- [ ] Frequency penalty works correctly
- [ ] Grade mapping accurate

**Task 2.2: Assessment Generator (1 hour)**

Add to `bosspig/detector.py`:
```python
def _generate_assessment(
    self,
    quality_score: float,
    category_scores: Dict[SlopCategory, float]
) -> str:
    """Generate human-readable overall assessment."""

    grade = QualityScorer.get_grade(quality_score)

    # Find weakest categories
    sorted_cats = sorted(
        category_scores.items(),
        key=lambda x: x[1]
    )
    weakest = [cat.value for cat, score in sorted_cats[:3] if score < 70]

    # Build assessment
    if quality_score >= 90:
        return f"Excellent ({grade}). This document demonstrates clear, professional writing with minimal business slop."
    elif quality_score >= 70:
        assessment = f"Good ({grade}). Generally professional"
        if weakest:
            assessment += f", but watch for {', '.join(weakest)}"
        return assessment + "."
    elif quality_score >= 50:
        assessment = f"Needs improvement ({grade}). Significant issues with {', '.join(weakest[:2])}"
        return assessment + ". Consider major revision."
    else:
        return f"Poor ({grade}). This document is heavily laden with business slop across multiple categories. Recommend complete rewrite."

def _identify_top_issues(
    self,
    category_scores: Dict[SlopCategory, float]
) -> List[str]:
    """Identify top 5 problem categories."""

    sorted_cats = sorted(
        category_scores.items(),
        key=lambda x: x[1]
    )

    return [
        f"{cat.value.replace('_', ' ').title()}: {score:.0f}/100"
        for cat, score in sorted_cats[:5]
    ]

def _generate_suggestions(
    self,
    detections: List[SlopDetection]
) -> List[str]:
    """Generate top 10 actionable suggestions."""

    # Group by category
    by_category = {}
    for det in detections:
        if det.category not in by_category:
            by_category[det.category] = []
        by_category[det.category].append(det)

    # Get most severe detection from each category
    suggestions = []
    for category, dets in sorted(
        by_category.items(),
        key=lambda x: -max(d.severity for d in x[1])
    ):
        most_severe = max(dets, key=lambda d: d.severity)
        suggestions.append(
            f"{category.value.replace('_', ' ').title()}: "
            f"{most_severe.suggestion}"
        )

    return suggestions[:10]  # Top 10
```

**Verification**:
- [ ] Assessment matches score range
- [ ] Top issues sorted by severity
- [ ] Suggestions actionable
- [ ] No duplicate suggestions

**Task 2.3: Test Suite (1 hour)**

Create `bosspig/tests/test_scorer.py`:
```python
def test_perfect_score():
    """Empty document should score 100."""
    scorer = QualityScorer(BossPigConfig())
    score = scorer.score([], total_words=100)
    assert score == 100.0

def test_single_detection_penalty():
    """Single high-severity detection should reduce score."""
    detection = SlopDetection(
        category=SlopCategory.JARGON,
        text="synergize",
        span=(0, 9),
        confidence=0.95,
        severity=0.9,
        suggestion="Use 'work together' instead",
        example="The teams will work together on this project."
    )

    scorer = QualityScorer(BossPigConfig())
    score = scorer.score([detection], total_words=10)

    # Should lose points but not fail completely
    assert 80 < score < 100

def test_frequency_penalty():
    """High slop density should trigger exponential penalty."""
    # 10 detections in 100 words = 10% density (should trigger penalty)
    detections = [
        SlopDetection(
            category=SlopCategory.BUZZWORDS,
            text="innovative",
            span=(i*10, i*10+10),
            confidence=0.8,
            severity=0.7,
            suggestion="Be specific",
            example="Specific improvement"
        )
        for i in range(10)
    ]

    scorer = QualityScorer(BossPigConfig())
    score = scorer.score(detections, total_words=100)

    # Should be significantly penalized
    assert score < 50

def test_category_scores():
    """Category scores should be independent."""
    detections = [
        SlopDetection(
            category=SlopCategory.JARGON,
            text="synergize",
            span=(0, 9),
            confidence=0.9,
            severity=0.9,
            suggestion="...",
            example="..."
        ),
        SlopDetection(
            category=SlopCategory.BUZZWORDS,
            text="disruptive",
            span=(10, 20),
            confidence=0.8,
            severity=0.8,
            suggestion="...",
            example="..."
        )
    ]

    scorer = QualityScorer(BossPigConfig())
    cat_scores = scorer.score_by_category(detections)

    # Jargon should be penalized
    assert cat_scores[SlopCategory.JARGON] < 100
    # Buzzwords should be penalized
    assert cat_scores[SlopCategory.BUZZWORDS] < 100
    # Passive voice should be perfect (no detections)
    assert cat_scores[SlopCategory.PASSIVE_VOICE] == 100
```

**Verification**:
- [ ] All tests pass
- [ ] Edge cases covered
- [ ] Penalties calculated correctly

---

## Phase 2: Category Detectors (Days 3-7) - 20 hours

**Daily cadence**: 3 detectors per day × 5 days = 15 detectors

### Day 3: Core Detectors (Jargon, Vague Commitments, Buzzwords)

**Task 3.1: Jargon Detector (2 hours)**

Create `bosspig/categories/jargon.py`:
```python
class JargonDetector:
    """Detects business jargon and unnecessarily complex language."""

    # Comprehensive jargon dictionary
    JARGON_TERMS = {
        # High severity (obvious jargon)
        'synergize': 0.9,
        'leverage': 0.85,
        'paradigm': 0.9,
        'ecosystem': 0.8,
        'holistic': 0.8,
        'granular': 0.75,
        'actionable': 0.7,
        'operationalize': 0.9,
        'monetize': 0.7,
        'ideate': 0.85,

        # Medium severity (overused business terms)
        'utilize': 0.6,
        'facilitate': 0.6,
        'implement': 0.5,
        'strategic': 0.5,
        'optimize': 0.6,
        'streamline': 0.6,
        'scalable': 0.5,
        'robust': 0.5,

        # Add 50+ more terms...
    }

    # Simpler alternatives
    SUGGESTIONS = {
        'synergize': "work together",
        'leverage': "use",
        'paradigm': "model or approach",
        'utilize': "use",
        'facilitate': "help or enable",
        # ... for all terms
    }

    EXAMPLES = {
        'synergize': "The teams will work together on this initiative.",
        'leverage': "We will use our existing customer base.",
        # ... for all terms
    }

    def detect(self, text: str) -> List[SlopDetection]:
        """Detect jargon in text."""
        detections = []

        # Lowercase for matching
        lower_text = text.lower()

        for term, severity in self.JARGON_TERMS.items():
            # Find all occurrences
            start = 0
            while True:
                pos = lower_text.find(term, start)
                if pos == -1:
                    break

                # Create detection
                detections.append(SlopDetection(
                    category=SlopCategory.JARGON,
                    text=text[pos:pos+len(term)],
                    span=(pos, pos+len(term)),
                    confidence=0.95,  # High confidence for dictionary match
                    severity=severity,
                    suggestion=f"Replace with '{self.SUGGESTIONS[term]}'",
                    example=self.EXAMPLES[term]
                ))

                start = pos + 1

        return detections

    def get_severity(self, detection: str) -> float:
        """Get severity for specific jargon."""
        return self.JARGON_TERMS.get(detection.lower(), 0.5)
```

**Verification**:
- [ ] Detects all jargon terms
- [ ] Provides appropriate suggestions
- [ ] Severity levels accurate
- [ ] Case-insensitive matching

**Task 3.2: Vague Commitment Detector (2 hours)**

Create `bosspig/categories/vague_commitments.py`:
```python
class VagueCommitmentDetector:
    """Detects vague, non-committal language."""

    VAGUE_PATTERNS = [
        # Time vagueness
        (r'\bsoon\b', 0.7, "Specify when (e.g., 'by Q2' or 'within 3 weeks')"),
        (r'\beventually\b', 0.8, "Provide timeline"),
        (r'\bin the near future\b', 0.9, "Give specific date"),
        (r'\bdown the road\b', 0.85, "State timeframe"),

        # Outcome vagueness
        (r'\bwe hope to\b', 0.8, "State definite plan: 'We will...'"),
        (r'\bwe plan to consider\b', 0.9, "Either commit or don't mention"),
        (r'\bwe may\b', 0.7, "Be definitive: 'We will' or 'We won't'"),
        (r'\bpotentially\b', 0.6, "State likelihood or remove"),

        # Action vagueness
        (r'\bexplore opportunities\b', 0.85, "Specify what opportunities"),
        (r'\binvestigate further\b', 0.7, "State investigation scope"),
        (r'\btake appropriate action\b', 0.9, "Define specific actions"),

        # Add 30+ more patterns...
    ]

    def detect(self, text: str) -> List[SlopDetection]:
        """Detect vague commitments."""
        import re

        detections = []

        for pattern, severity, suggestion in self.VAGUE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detections.append(SlopDetection(
                    category=SlopCategory.VAGUE_COMMITMENTS,
                    text=match.group(),
                    span=(match.start(), match.end()),
                    confidence=0.9,
                    severity=severity,
                    suggestion=suggestion,
                    example=self._get_example(match.group())
                ))

        return detections

    def _get_example(self, vague_phrase: str) -> str:
        """Get concrete example for vague phrase."""
        examples = {
            'soon': "by March 15th",
            'eventually': "within 6 months",
            'we hope to': "we will",
            # ... for all patterns
        }

        for key, example in examples.items():
            if key in vague_phrase.lower():
                return f"Instead: '{example}'"

        return "Be specific and commit."
```

**Verification**:
- [ ] Regex patterns compile
- [ ] All patterns detected
- [ ] Suggestions actionable
- [ ] Examples concrete

**Task 3.3: Buzzword Detector (1.5 hours)**

Create `bosspig/categories/buzzwords.py`:
```python
class BuzzwordDetector:
    """Detects overused buzzwords and trendy terms."""

    BUZZWORDS = {
        # Tech buzzwords
        'disruptive': 0.9,
        'innovative': 0.8,
        'revolutionary': 0.85,
        'cutting-edge': 0.8,
        'next-generation': 0.75,
        'state-of-the-art': 0.7,
        'game-changer': 0.9,
        'paradigm shift': 0.95,

        # Business buzzwords
        'best-in-class': 0.85,
        'world-class': 0.8,
        'industry-leading': 0.75,
        'thought leader': 0.9,
        'value-add': 0.8,
        'win-win': 0.85,

        # Modern buzzwords
        'digital transformation': 0.7,
        'agile': 0.6,
        'data-driven': 0.6,
        'customer-centric': 0.65,

        # Add 50+ more...
    }

    ALTERNATIVES = {
        'disruptive': "new approach or different method",
        'innovative': "new (if truly new, otherwise remove)",
        'game-changer': "significant improvement",
        # ... for all buzzwords
    }

    def detect(self, text: str) -> List[SlopDetection]:
        """Detect buzzwords."""
        detections = []
        lower_text = text.lower()

        for buzzword, severity in self.BUZZWORDS.items():
            start = 0
            while True:
                pos = lower_text.find(buzzword, start)
                if pos == -1:
                    break

                detections.append(SlopDetection(
                    category=SlopCategory.BUZZWORDS,
                    text=text[pos:pos+len(buzzword)],
                    span=(pos, pos+len(buzzword)),
                    confidence=0.95,
                    severity=severity,
                    suggestion=f"Use '{self.ALTERNATIVES.get(buzzword, 'specific description')}' instead",
                    example=f"Be specific about what makes it {buzzword}"
                ))

                start = pos + 1

        return detections
```

**Verification**:
- [ ] All buzzwords detected
- [ ] Alternatives provided
- [ ] Multi-word buzzwords work

**Task 3.4: Day 3 Testing (0.5 hours)**

Run comprehensive tests:
```bash
pytest bosspig/tests/test_categories/test_jargon.py -v
pytest bosspig/tests/test_categories/test_vague_commitments.py -v
pytest bosspig/tests/test_categories/test_buzzwords.py -v
```

**Verification**:
- [ ] All 3 detectors pass tests
- [ ] No false positives
- [ ] Coverage >80%

### Day 4: Language Detectors (Passive Voice, Weasel Words, Superlatives)

**Task 4.1-4.3**: Similar structure to Day 3
- Passive Voice: regex + spaCy patterns
- Weasel Words: "some", "several", "various", etc.
- Superlatives: "best", "greatest", "most", etc.

**Time**: 2 hours each + 0.5 hour testing

### Day 5: Corporate Speak (Empty Phrases, Corporate Speak, Hedge Words)

**Task 5.1-5.3**: Similar structure
- Empty Phrases: "at the end of the day", "circle back", etc.
- Corporate Speak: "touch base", "take offline", etc.
- Hedge Words: "arguably", "somewhat", "fairly", etc.

**Time**: 2 hours each + 0.5 hour testing

### Day 6: Content Quality (Filler, Redundancy, Name Dropping)

**Task 6.1-6.3**: Similar structure
- Filler Content: Paragraph-level analysis for low information density
- Redundancy: Repetitive phrases and concepts
- Name Dropping: Excessive references without context

**Time**: 2 hours each + 0.5 hour testing

### Day 7: Advanced Detectors (False Urgency, Vague Metrics, AI Fingerprints)

**Task 7.1: False Urgency Detector (2.5 hours)**

Most sophisticated detector - analyzes urgency language patterns:
```python
class FalseUrgencyDetector:
    """Detects manufactured urgency without substance."""

    URGENCY_MARKERS = [
        (r'\bASAP\b', 0.7),
        (r'\bimmediate(?:ly)?\b', 0.6),
        (r'\btime-sensitive\b', 0.7),
        (r'\bact now\b', 0.8),
        (r'\bdon\'t miss out\b', 0.9),
    ]

    def detect(self, text: str) -> List[SlopDetection]:
        # Detect urgency markers
        # Check if followed by concrete deadline
        # If no deadline → false urgency
        ...
```

**Task 7.2: Vague Metrics Detector (2 hours)**

Detects metrics without specifics:
- "significant improvement" → how much?
- "increased revenue" → by what percentage?
- "many customers" → how many?

**Task 7.3: AI Fingerprints Detector (2 hours)**

Detects common AI writing patterns:
- "delve into"
- "it's important to note that"
- "in today's fast-paced world"
- Over-formal structure
- Repetitive sentence patterns

**Task 7.4: Day 7 Testing (1 hour)**

Complete test suite for all 15 detectors.

**Verification**:
- [ ] All 15 detectors implemented
- [ ] All tests passing
- [ ] No memory leaks
- [ ] Performance acceptable (<1s per page)

---

## Phase 3: Integration & CLI (Days 8-9) - 8 hours

### Day 8: CLI Interface (4 hours)

**Task 8.1: Argument Parser (1 hour)**

Create `bosspig/cli.py`:
```python
import argparse
from pathlib import Path

def create_parser():
    parser = argparse.ArgumentParser(
        description="BossPig - Business Slop Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single document
  python -m bosspig analyze document.pdf

  # Analyze directory
  python -m bosspig analyze documents/ --recursive

  # Custom output format
  python -m bosspig analyze doc.docx --format json --output report.json

  # Verbose mode with suggestions
  python -m bosspig analyze doc.md --verbose --suggestions
        """
    )

    subparsers = parser.add_subparsers(dest='command')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze documents')
    analyze_parser.add_argument('input', help='Document or directory path')
    analyze_parser.add_argument('--recursive', '-r', action='store_true')
    analyze_parser.add_argument('--format', choices=['text', 'json', 'html'], default='text')
    analyze_parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    analyze_parser.add_argument('--verbose', '-v', action='store_true')
    analyze_parser.add_argument('--suggestions', '-s', action='store_true')
    analyze_parser.add_argument('--threshold', type=float, default=70.0)

    # Config command
    config_parser = subparsers.add_parser('config', help='Show configuration')

    # Version command
    version_parser = subparsers.add_parser('version', help='Show version')

    return parser
```

**Task 8.2: Document Processing Pipeline (2 hours)**

Integrate with HoloLoom SpinningWheel:
```python
from HoloLoom.spinningWheel import (
    PDFSpinner,
    DOCXSpinner,
    MarkdownSpinner
)

def process_document(file_path: Path) -> str:
    """Extract text from document using HoloLoom spinners."""

    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        spinner = PDFSpinner()
        shards = spinner.spin({'path': str(file_path)})
    elif suffix in ['.docx', '.doc']:
        spinner = DOCXSpinner()
        shards = spinner.spin({'path': str(file_path)})
    elif suffix in ['.md', '.markdown']:
        spinner = MarkdownSpinner()
        shards = spinner.spin({'path': str(file_path)})
    else:
        # Plain text
        text = file_path.read_text()
        return text

    # Combine shards
    return '\n\n'.join(shard.content for shard in shards)
```

**Task 8.3: Output Formatters (1 hour)**

Create text, JSON, and HTML formatters:
```python
class TextFormatter:
    """Human-readable text output."""

    def format(self, analysis: DocumentAnalysis) -> str:
        output = []
        output.append("="*60)
        output.append(f"BossPig Analysis: {analysis.document_id}")
        output.append("="*60)
        output.append(f"Quality Score: {analysis.quality_score:.1f}/100")
        output.append(f"Grade: {QualityScorer.get_grade(analysis.quality_score)}")
        output.append(f"Words: {analysis.total_words:,}")
        output.append(f"Issues Found: {len(analysis.detections)}")
        output.append("")
        output.append("Overall Assessment:")
        output.append(f"  {analysis.overall_assessment}")
        output.append("")
        output.append("Top Issues:")
        for issue in analysis.top_issues:
            output.append(f"  • {issue}")
        output.append("")
        output.append("Suggestions:")
        for i, suggestion in enumerate(analysis.suggestions, 1):
            output.append(f"  {i}. {suggestion}")

        return '\n'.join(output)

class JSONFormatter:
    """Machine-readable JSON output."""

    def format(self, analysis: DocumentAnalysis) -> str:
        import json

        # Convert to dict
        data = {
            'document_id': analysis.document_id,
            'quality_score': analysis.quality_score,
            'total_words': analysis.total_words,
            'total_sentences': analysis.total_sentences,
            'detections': [
                {
                    'category': d.category.value,
                    'text': d.text,
                    'span': d.span,
                    'confidence': d.confidence,
                    'severity': d.severity,
                    'suggestion': d.suggestion
                }
                for d in analysis.detections
            ],
            'category_scores': {
                cat.value: score
                for cat, score in analysis.category_scores.items()
            },
            'assessment': analysis.overall_assessment,
            'top_issues': analysis.top_issues,
            'suggestions': analysis.suggestions
        }

        return json.dumps(data, indent=2)
```

**Verification**:
- [ ] CLI accepts all arguments
- [ ] Document processing works for PDF/DOCX/MD
- [ ] All formatters produce correct output

### Day 9: Testing & Documentation (4 hours)

**Task 9.1: Integration Tests (2 hours)**

Create `bosspig/tests/test_integration.py`:
```python
def test_end_to_end_analysis():
    """Test complete pipeline from document to report."""

    # Create test document
    test_text = """
    We plan to leverage our synergies to drive innovation in the near future.
    This paradigm shift will be disruptive and game-changing.
    Our world-class team will synergize to deliver value-add solutions.
    """

    # Run detection
    detector = BossPigDetector(BossPigConfig())
    analysis = detector.analyze(test_text, "test_doc")

    # Verify detections
    assert len(analysis.detections) > 0
    assert analysis.quality_score < 70  # Should be poor

    # Verify categories detected
    detected_cats = {d.category for d in analysis.detections}
    assert SlopCategory.JARGON in detected_cats
    assert SlopCategory.BUZZWORDS in detected_cats
    assert SlopCategory.VAGUE_COMMITMENTS in detected_cats

def test_cli_analyze():
    """Test CLI analyze command."""
    import subprocess

    result = subprocess.run(
        ['python', '-m', 'bosspig', 'analyze', 'test.md'],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0
    assert 'Quality Score:' in result.stdout
```

**Task 9.2: User Documentation (2 hours)**

Create `bosspig/README.md`:
```markdown
# BossPig Business Slop Detector

Detects and scores 15 categories of business writing slop.

## Installation

```bash
pip install bosspig  # When published
# OR
git clone https://github.com/yourorg/bosspig
cd bosspig
pip install -e .
```

## Quick Start

```bash
# Analyze single document
python -m bosspig analyze document.pdf

# Analyze with suggestions
python -m bosspig analyze document.docx --suggestions

# JSON output
python -m bosspig analyze document.md --format json --output report.json
```

## 15 Slop Categories

1. **Jargon**: synergize, leverage, paradigm
2. **Vague Commitments**: "soon", "we hope to"
3. **Buzzwords**: disruptive, innovative
4. **Passive Voice**: "was done by"
5. **Weasel Words**: "some", "various"
6. **Superlatives**: "best", "greatest"
7. **Empty Phrases**: "at the end of the day"
8. **Corporate Speak**: "touch base", "circle back"
9. **Hedge Words**: "arguably", "somewhat"
10. **Filler Content**: Low information density
11. **Redundancy**: Repetitive concepts
12. **Name Dropping**: Excessive references
13. **False Urgency**: "ASAP" without deadline
14. **Vague Metrics**: "significant" without numbers
15. **AI Fingerprints**: "delve into", "it's important to note"

## Scoring

- **90-100**: Excellent (A)
- **80-89**: Good (B)
- **70-79**: Acceptable (C)
- **60-69**: Poor (D)
- **0-59**: Unacceptable (F)

## Python API

```python
from bosspig import BossPigDetector, BossPigConfig

detector = BossPigDetector(BossPigConfig())
analysis = detector.analyze(text, document_id="my_doc")

print(f"Score: {analysis.quality_score}/100")
print(f"Issues: {len(analysis.detections)}")
```

## Configuration

Customize in `bosspig/config.py` or via environment variables:

```python
config = BossPigConfig(
    jargon_threshold=0.8,  # Stricter
    output_format='json',
    show_suggestions=True
)
```

## Development

```bash
# Run tests
pytest bosspig/tests/ -v

# Type checking
mypy bosspig/

# Linting
ruff check bosspig/
```
```

**Verification**:
- [ ] All integration tests pass
- [ ] Documentation complete
- [ ] Examples work

---

## Phase 4: Polish & Launch (Day 10) - 4 hours

### Day 10: Final Testing & Launch Prep

**Task 10.1: Performance Testing (1 hour)**

Benchmark on real documents:
```python
def test_performance():
    """Ensure <1s per page."""

    # Test with 10-page document
    long_text = sample_text * 500  # ~10 pages

    detector = BossPigDetector(BossPigConfig())

    start = time.time()
    analysis = detector.analyze(long_text)
    duration = time.time() - start

    assert duration < 10.0  # <1s per page
    print(f"Analyzed {len(long_text.split())} words in {duration:.2f}s")
```

**Task 10.2: Demo Documents (1 hour)**

Create 5 demo documents showing different quality levels:
- `demo_excellent.md` - Score 95+
- `demo_good.md` - Score 80-90
- `demo_acceptable.md` - Score 70-79
- `demo_poor.md` - Score 50-69
- `demo_terrible.md` - Score <50

**Task 10.3: Launch Checklist (1 hour)**

Create `LAUNCH_CHECKLIST.md`:
```markdown
# BossPig Launch Checklist

## Code Quality
- [ ] All 15 detectors implemented
- [ ] All tests passing (>80% coverage)
- [ ] No linting errors
- [ ] Type hints complete
- [ ] Documentation complete

## Performance
- [ ] <1s per page analysis
- [ ] Memory usage <500MB
- [ ] No memory leaks

## Functionality
- [ ] CLI works for PDF/DOCX/MD
- [ ] All output formats work (text/json/html)
- [ ] Scoring accurate
- [ ] Suggestions actionable

## Distribution
- [ ] setup.py configured
- [ ] PyPI metadata complete
- [ ] README with examples
- [ ] LICENSE file
- [ ] CHANGELOG started

## Marketing
- [ ] Demo video recorded
- [ ] Landing page ready
- [ ] SaaS tiers defined
- [ ] Pricing calculator working
```

**Task 10.4: Release (1 hour)**

```bash
# Final tests
pytest bosspig/tests/ -v --cov=bosspig

# Build package
python setup.py sdist bdist_wheel

# Test install
pip install dist/bosspig-0.1.0-py3-none-any.whl

# Verify
python -m bosspig --version
python -m bosspig analyze demo_documents/demo_terrible.md
```

**Verification**:
- [ ] Package builds
- [ ] Installation works
- [ ] CLI functional
- [ ] Ready for beta users

---

## Success Metrics

### Day 10 Deliverables

1. **Working MVP**:
   - [ ] 15 category detectors
   - [ ] Quality scoring (0-100)
   - [ ] CLI interface
   - [ ] 3 output formats

2. **Quality Targets**:
   - [ ] >80% test coverage
   - [ ] <1s per page performance
   - [ ] Accurate scoring (validated with test docs)
   - [ ] Actionable suggestions

3. **Documentation**:
   - [ ] README with examples
   - [ ] API documentation
   - [ ] User guide
   - [ ] Developer guide

4. **Distribution**:
   - [ ] PyPI package
   - [ ] GitHub repository
   - [ ] Demo documents
   - [ ] Marketing materials

---

## Risk Mitigation

### Technical Risks

**Risk 1**: Detector accuracy
- **Mitigation**: Extensive testing with real business documents
- **Fallback**: Manual tuning of thresholds

**Risk 2**: Performance issues
- **Mitigation**: Profile early, optimize hot paths
- **Fallback**: Parallel processing for large documents

**Risk 3**: False positives
- **Mitigation**: Confidence thresholds, user feedback
- **Fallback**: Allowlist for domain-specific terms

### Schedule Risks

**Risk 1**: Detector complexity underestimated
- **Mitigation**: 20 hours budgeted for 15 detectors (>1hr each)
- **Fallback**: Simplify detectors, release v1.1 with improvements

**Risk 2**: Integration issues
- **Mitigation**: Use existing HoloLoom spinners
- **Fallback**: Support fewer formats initially (PDF only)

---

## Next Steps After Launch

### Week 3-4: Iteration

1. **User Feedback**:
   - Collect beta user feedback
   - Track false positive rate
   - Measure user satisfaction

2. **Improvements**:
   - Fine-tune thresholds
   - Add user-requested categories
   - Improve suggestions

3. **Marketing**:
   - Blog post announcing launch
   - Demo video on social media
   - Reach out to writing tools reviewers

### Months 2-3: SaaS Build

1. **Web Interface**:
   - Upload documents
   - Real-time analysis
   - History tracking

2. **API**:
   - REST API for integrations
   - Webhooks for CI/CD
   - Bulk analysis

3. **Monetization**:
   - Stripe integration
   - Tier enforcement
   - Usage tracking

---

## Appendix A: Detector Template

Template for implementing new detectors:

```python
"""
{Category} Detector
===================

Detects {description}.

Severity: {0.0-1.0 range explanation}
Confidence: {typically 0.8-0.95 for pattern matching}
"""

from typing import List
import re

from bosspig.detector import SlopDetection, SlopCategory

class {Category}Detector:
    """{One-line description}."""

    # Patterns with severity scores
    PATTERNS = [
        (r'{regex}', {severity}, "{suggestion}"),
        # Add 10-20 patterns
    ]

    # Examples for suggestions
    EXAMPLES = {
        '{pattern_key}': "{concrete example}",
    }

    def detect(self, text: str) -> List[SlopDetection]:
        """Detect {category} in text."""
        detections = []

        for pattern, severity, suggestion in self.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                detections.append(SlopDetection(
                    category=SlopCategory.{CATEGORY},
                    text=match.group(),
                    span=(match.start(), match.end()),
                    confidence=0.90,  # Adjust based on pattern precision
                    severity=severity,
                    suggestion=suggestion,
                    example=self._get_example(match.group())
                ))

        return detections

    def _get_example(self, detected_text: str) -> str:
        """Get concrete example for detected text."""
        # Look up in EXAMPLES or generate
        for key, example in self.EXAMPLES.items():
            if key in detected_text.lower():
                return example
        return "Provide specific, concrete language."

    def get_severity(self, detection: str) -> float:
        """Calculate severity for specific detection."""
        # Override if severity varies by context
        return 0.7  # Default
```

---

## Appendix B: Test Template

Template for testing detectors:

```python
"""
Tests for {Category} Detector
"""

import pytest
from bosspig.categories.{category} import {Category}Detector
from bosspig.detector import SlopCategory

def test_basic_detection():
    """Test basic {category} detection."""
    detector = {Category}Detector()

    text = "{sample text with slop}"
    detections = detector.detect(text)

    assert len(detections) > 0
    assert all(d.category == SlopCategory.{CATEGORY} for d in detections)

def test_severity_scoring():
    """Test severity varies by pattern."""
    detector = {Category}Detector()

    # High severity example
    high_severity = detector.detect("{high severity text}")
    # Low severity example
    low_severity = detector.detect("{low severity text}")

    assert high_severity[0].severity > low_severity[0].severity

def test_no_false_positives():
    """Test clean text produces no detections."""
    detector = {Category}Detector()

    clean_text = "{clean, professional text}"
    detections = detector.detect(clean_text)

    assert len(detections) == 0

def test_suggestions_provided():
    """Test all detections have actionable suggestions."""
    detector = {Category}Detector()

    text = "{sample text with slop}"
    detections = detector.detect(text)

    for detection in detections:
        assert detection.suggestion
        assert len(detection.suggestion) > 10  # Meaningful suggestion
        assert detection.example  # Concrete example provided
```

---

**End of BossPig Roadmap**

Total estimated effort: 40 hours over 10 days
Completion criteria: Working MVP with 15 detectors, CLI, and documentation

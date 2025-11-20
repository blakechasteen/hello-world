"""
Importance Scoring Framework
=============================

Reusable importance scoring for all HoloLoom spinners.

Provides:
- Standardized signal calculation
- Domain-agnostic scoring algorithms
- Explainable importance assessment
- Temporal decay functions
- Authority/credibility metrics

Author: Claude Code
Date: November 2025
"""

import re
import time
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from datetime import datetime

from .protocol import ImportanceSignals, ImportanceScore


# ============================================================================
# Signal Calculators
# ============================================================================

class LengthScorer:
    """Score based on content length (substantive vs. noise)."""

    def __init__(
        self,
        min_length: int = 50,
        optimal_length: int = 300,
        max_length: int = 5000
    ):
        """
        Initialize length scorer.

        Args:
            min_length: Minimum substantive length (chars)
            optimal_length: Optimal length for full score
            max_length: Maximum before diminishing returns
        """
        self.min_length = min_length
        self.optimal_length = optimal_length
        self.max_length = max_length

    def score(self, text: str) -> float:
        """
        Score text length.

        Returns:
            0.0-1.0 score
                - 0.0: Very short (<min_length)
                - 0.5: Half of optimal
                - 1.0: At or above optimal length
        """
        length = len(text)

        if length < self.min_length:
            # Very short: linear scale 0.0 → 0.3
            return min(0.3, length / self.min_length * 0.3)

        elif length < self.optimal_length:
            # Below optimal: linear scale 0.3 → 1.0
            progress = (length - self.min_length) / (self.optimal_length - self.min_length)
            return 0.3 + progress * 0.7

        elif length < self.max_length:
            # Above optimal: full score
            return 1.0

        else:
            # Very long: diminishing returns
            excess = length - self.max_length
            penalty = min(0.3, excess / self.max_length * 0.3)
            return max(0.7, 1.0 - penalty)


class TechnicalScorer:
    """Score based on technical term density."""

    def __init__(
        self,
        technical_terms: Optional[Set[str]] = None,
        domain_specific: bool = True
    ):
        """
        Initialize technical scorer.

        Args:
            technical_terms: Set of technical terms (None = use defaults)
            domain_specific: Use domain-specific terms
        """
        self.technical_terms = technical_terms or self._default_terms()
        self.domain_specific = domain_specific

    def _default_terms(self) -> Set[str]:
        """Default technical terms (HoloLoom domain)."""
        return {
            # AI/ML
            'policy', 'thompson', 'sampling', 'embedding', 'neural',
            'attention', 'transformer', 'reinforcement', 'bandit',
            'bayesian', 'gradient', 'optimization', 'convergence',

            # HoloLoom specific
            'memory', 'shard', 'orchestrator', 'spinner', 'motif',
            'knowledge', 'graph', 'vector', 'retrieval', 'weaving',
            'matryoshka', 'spectral', 'warp', 'loom', 'fusion',

            # Programming
            'algorithm', 'function', 'class', 'method', 'api',
            'protocol', 'interface', 'implementation', 'pipeline',
            'architecture', 'framework', 'library', 'dependency'
        }

    def score(self, text: str) -> float:
        """
        Score technical content density.

        Returns:
            0.0-1.0 score based on term frequency
        """
        text_lower = text.lower()

        # Count matches
        matches = sum(1 for term in self.technical_terms if term in text_lower)

        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return 0.0

        density = matches / max(words / 50, 1)  # Per 50 words

        # Map to 0.0-1.0
        # 0 matches = 0.0, 5+ matches = 1.0
        return min(1.0, density / 5.0)

    def add_terms(self, terms: List[str]):
        """Add domain-specific terms."""
        self.technical_terms.update(t.lower() for t in terms)


class StructuralScorer:
    """Score based on content structure and formatting."""

    def score(self, text: str) -> float:
        """
        Score structural quality.

        Checks:
        - Paragraphs (multiple line breaks)
        - Bullet points
        - Numbered lists
        - Code blocks
        - Headers/sections

        Returns:
            0.0-1.0 score
        """
        score = 0.0

        # Paragraphs (multiple line breaks)
        if '\n\n' in text:
            score += 0.2

        # Bullet points or lists
        if re.search(r'^\s*[-*•]\s', text, re.MULTILINE):
            score += 0.2
        if re.search(r'^\s*\d+\.\s', text, re.MULTILINE):
            score += 0.2

        # Code blocks
        if '```' in text or re.search(r'^\s{4,}', text, re.MULTILINE):
            score += 0.2

        # Headers (markdown or plain)
        if re.search(r'^#+\s', text, re.MULTILINE) or re.search(r'^[A-Z][a-z]+:$', text, re.MULTILINE):
            score += 0.2

        return min(1.0, score)


class AuthorityScorer:
    """Score based on source authority/credibility."""

    def __init__(self, authority_map: Optional[Dict[str, float]] = None):
        """
        Initialize authority scorer.

        Args:
            authority_map: Mapping of source patterns to scores
                           (e.g., {"@expert": 0.9, "@intern": 0.5})
        """
        self.authority_map = authority_map or {}

    def score(
        self,
        source: str,
        author: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> float:
        """
        Score source authority.

        Args:
            source: Source identifier (file path, URL, username)
            author: Author name (if available)
            metadata: Additional metadata

        Returns:
            0.0-1.0 authority score
        """
        # Check authority map
        for pattern, score in self.authority_map.items():
            if pattern in source or (author and pattern in author):
                return score

        # Default heuristics
        score = 0.5  # Neutral default

        # URL-based
        if source.startswith('http'):
            # Academic/official sources
            if any(domain in source for domain in ['.edu', '.gov', '.org']):
                score = 0.8
            # Commercial
            elif any(domain in source for domain in ['.com', '.io']):
                score = 0.6

        # File-based
        elif 'official' in source.lower() or 'docs' in source.lower():
            score = 0.8

        # Metadata hints
        if metadata:
            if metadata.get('verified', False):
                score = min(1.0, score + 0.2)
            if metadata.get('upvotes', 0) > 100:
                score = min(1.0, score + 0.1)

        return score

    def add_authority(self, pattern: str, score: float):
        """Add authority mapping."""
        self.authority_map[pattern] = score


class RecencyScorer:
    """Score based on temporal decay."""

    def __init__(
        self,
        half_life_days: float = 30.0,
        max_age_days: float = 365.0
    ):
        """
        Initialize recency scorer.

        Args:
            half_life_days: Days until score halves (exponential decay)
            max_age_days: Maximum age to consider (older = 0.0)
        """
        self.half_life_days = half_life_days
        self.max_age_days = max_age_days

    def score(self, timestamp: Optional[float] = None) -> float:
        """
        Score based on age.

        Args:
            timestamp: Unix timestamp (None = use current time)

        Returns:
            0.0-1.0 score (1.0 = very recent, 0.0 = very old)
        """
        if timestamp is None:
            return 1.0  # Assume current

        # Calculate age in days
        age_seconds = time.time() - timestamp
        age_days = age_seconds / 86400

        # Too old
        if age_days > self.max_age_days:
            return 0.0

        # Exponential decay
        # score = 0.5^(age_days / half_life_days)
        decay_factor = age_days / self.half_life_days
        score = 0.5 ** decay_factor

        return max(0.0, min(1.0, score))


class EngagementScorer:
    """Score based on engagement metrics (likes, shares, replies)."""

    def score(
        self,
        likes: int = 0,
        shares: int = 0,
        replies: int = 0,
        views: int = 0
    ) -> float:
        """
        Score engagement.

        Args:
            likes: Number of likes/upvotes
            shares: Number of shares/retweets
            replies: Number of replies/comments
            views: Number of views

        Returns:
            0.0-1.0 engagement score
        """
        # Weighted engagement
        engagement = (
            likes * 1.0 +
            shares * 3.0 +  # Shares = stronger signal
            replies * 2.0 +  # Replies = engagement
            views * 0.01     # Views are weak signal
        )

        # Map to 0.0-1.0
        # 0 = 0.0, 100+ = 1.0
        return min(1.0, engagement / 100.0)


class NoiseDetector:
    """Detect noise patterns (greetings, spam, duplicates)."""

    def __init__(self):
        """Initialize noise detector."""
        self.noise_patterns = [
            r'\b(hi|hello|hey|thanks|thank you|ok|okay|bye|cheers)\b',
            r'^(lol|lmao|omg|wtf)$',
            r'(!!!|???|\.\.\.)$',  # Excessive punctuation
            r'\b(click here|buy now|limited time)\b',  # Spam
        ]

    def detect(self, text: str) -> float:
        """
        Detect noise and return penalty.

        Args:
            text: Text to check

        Returns:
            Penalty (0.0 to -1.0, where -1.0 = pure noise)
        """
        text_lower = text.lower().strip()

        # Very short + greeting = high noise
        if len(text) < 30:
            for pattern in self.noise_patterns[:4]:
                if re.search(pattern, text_lower):
                    return -0.5  # Heavy penalty

        # Spam patterns
        spam_matches = sum(
            1 for pattern in self.noise_patterns[4:]
            if re.search(pattern, text_lower)
        )
        if spam_matches > 0:
            return -0.3 * spam_matches

        # All caps (shouting)
        if text.isupper() and len(text) > 10:
            return -0.2

        # Excessive repetition
        words = text.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:
                return -0.3  # High repetition

        return 0.0  # No noise detected


# ============================================================================
# Importance Scorer (Main Class)
# ============================================================================

class ImportanceScorer:
    """
    Unified importance scorer using all signals.

    Combines:
    - Length (substantive content)
    - Technical terms (domain relevance)
    - Structure (formatting quality)
    - Authority (source credibility)
    - Recency (time decay)
    - Engagement (social signals)
    - Noise detection (spam filtering)
    """

    def __init__(
        self,
        technical_terms: Optional[Set[str]] = None,
        authority_map: Optional[Dict[str, float]] = None,
        custom_scorers: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize importance scorer.

        Args:
            technical_terms: Custom technical terms
            authority_map: Source authority mappings
            custom_scorers: Custom scoring functions
        """
        self.length_scorer = LengthScorer()
        self.technical_scorer = TechnicalScorer(technical_terms)
        self.structural_scorer = StructuralScorer()
        self.authority_scorer = AuthorityScorer(authority_map)
        self.recency_scorer = RecencyScorer()
        self.engagement_scorer = EngagementScorer()
        self.noise_detector = NoiseDetector()

        self.custom_scorers = custom_scorers or {}

    def score(
        self,
        text: str,
        source: str = "unknown",
        timestamp: Optional[float] = None,
        author: Optional[str] = None,
        engagement: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict] = None
    ) -> ImportanceScore:
        """
        Compute complete importance score.

        Args:
            text: Content to score
            source: Source identifier
            timestamp: Unix timestamp (None = current)
            author: Author name
            engagement: Engagement metrics (likes, shares, etc.)
            metadata: Additional metadata

        Returns:
            ImportanceScore with signals and explanation
        """
        # Calculate signals
        signals = ImportanceSignals(
            length_score=self.length_scorer.score(text),
            technical_score=self.technical_scorer.score(text),
            structural_score=self.structural_scorer.score(text),
            authority_score=self.authority_scorer.score(source, author, metadata),
            recency_score=self.recency_scorer.score(timestamp),
            engagement_score=self.engagement_scorer.score(
                **(engagement or {})
            ),
            noise_penalty=self.noise_detector.detect(text)
        )

        # Custom signals
        for name, scorer_func in self.custom_scorers.items():
            try:
                signals.custom_signals[name] = scorer_func(text, metadata)
            except Exception as e:
                print(f"Warning: Custom scorer '{name}' failed: {e}")

        # Compute total
        total_score = signals.compute_total()

        # Generate explanation
        reason = signals.explain()

        return ImportanceScore(
            score=total_score,
            signals=signals,
            reason=reason
        )

    def score_batch(
        self,
        items: List[Dict]
    ) -> List[ImportanceScore]:
        """
        Score multiple items in batch.

        Args:
            items: List of dicts with 'text', 'source', 'timestamp', etc.

        Returns:
            List of ImportanceScore results
        """
        results = []

        for item in items:
            score = self.score(
                text=item.get('text', ''),
                source=item.get('source', 'unknown'),
                timestamp=item.get('timestamp'),
                author=item.get('author'),
                engagement=item.get('engagement'),
                metadata=item.get('metadata')
            )
            results.append(score)

        return results

    def add_technical_terms(self, terms: List[str]):
        """Add domain-specific technical terms."""
        self.technical_scorer.add_terms(terms)

    def add_authority(self, pattern: str, score: float):
        """Add authority mapping."""
        self.authority_scorer.add_authority(pattern, score)


# ============================================================================
# Preset Configurations
# ============================================================================

def create_chat_scorer() -> ImportanceScorer:
    """
    Create scorer optimized for chat/conversation history.

    High weight on:
    - Technical terms (domain expertise)
    - Engagement (replies, reactions)
    - Low tolerance for greetings/noise
    """
    technical_terms = {
        'policy', 'algorithm', 'implementation', 'refactor',
        'bug', 'feature', 'test', 'deploy', 'production',
        'performance', 'optimization', 'architecture'
    }

    authority_map = {
        '@senior': 0.9,
        '@lead': 0.85,
        '@architect': 0.9,
        '@junior': 0.5,
        '@intern': 0.4
    }

    return ImportanceScorer(
        technical_terms=technical_terms,
        authority_map=authority_map
    )


def create_git_scorer() -> ImportanceScorer:
    """
    Create scorer optimized for Git commits.

    High weight on:
    - Structural quality (well-formatted commits)
    - Technical content (code references)
    - Recency (recent commits more relevant)
    """
    technical_terms = {
        'fix', 'add', 'update', 'refactor', 'remove',
        'bug', 'feature', 'test', 'docs', 'merge',
        'breaking', 'deprecated', 'improve', 'optimize'
    }

    return ImportanceScorer(technical_terms=technical_terms)


def create_email_scorer() -> ImportanceScorer:
    """
    Create scorer optimized for email.

    High weight on:
    - Length (substantive vs. quick replies)
    - Authority (sender importance)
    - Engagement (thread depth)
    """
    authority_map = {
        'ceo@': 1.0,
        'vp@': 0.9,
        'director@': 0.8,
        'manager@': 0.7,
        'noreply@': 0.1
    }

    return ImportanceScorer(authority_map=authority_map)


def create_document_scorer() -> ImportanceScorer:
    """
    Create scorer optimized for documents (PDFs, articles).

    High weight on:
    - Length (comprehensive documents)
    - Structure (well-organized)
    - Technical content (domain-specific)
    """
    return ImportanceScorer()  # Use defaults


# ============================================================================
# Utilities
# ============================================================================

def explain_score(importance: ImportanceScore) -> str:
    """
    Generate detailed explanation of importance score.

    Args:
        importance: ImportanceScore to explain

    Returns:
        Multi-line explanation
    """
    lines = [
        f"Importance Score: {importance.score:.2f}",
        f"Reason: {importance.reason}",
        "",
        "Signal Breakdown:"
    ]

    signals = importance.signals
    signal_names = [
        ('Length', signals.length_score),
        ('Technical', signals.technical_score),
        ('Structural', signals.structural_score),
        ('Authority', signals.authority_score),
        ('Recency', signals.recency_score),
        ('Engagement', signals.engagement_score),
        ('Noise Penalty', signals.noise_penalty)
    ]

    for name, value in signal_names:
        bar = '█' * int(value * 20) if value > 0 else '░' * int(abs(value) * 20)
        lines.append(f"  {name:12s}: {value:+.2f} {bar}")

    # Custom signals
    for name, value in signals.custom_signals.items():
        bar = '█' * int(value * 20)
        lines.append(f"  {name:12s}: {value:+.2f} {bar}")

    return '\n'.join(lines)

"""
Source Credibility Scoring System

Evaluates research source quality based on multiple factors:
- Peer review status
- Publication venue prestige
- Author credentials (h-index, affiliation)
- Recency
- Citation count
- Primary vs secondary source
- Methodological rigor

Integrates with CitationManager to provide quality scores for all sources.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class CredibilityLevel(Enum):
    """Source credibility levels"""
    EXCELLENT = "excellent"      # 0.9-1.0: Top-tier peer-reviewed, recent
    GOOD = "good"                # 0.7-0.9: Peer-reviewed, established venue
    MODERATE = "moderate"        # 0.5-0.7: Credible but not ideal
    QUESTIONABLE = "questionable" # 0.3-0.5: Lacks key quality indicators
    POOR = "poor"                # 0.0-0.3: Not recommended


@dataclass
class CredibilityMetrics:
    """
    Detailed credibility breakdown

    Attributes:
        overall_score: Overall credibility (0.0-1.0)
        level: Credibility level classification
        peer_reviewed: Is peer-reviewed
        venue_prestige: Journal/venue quality (0.0-1.0)
        author_credibility: Author h-index/reputation (0.0-1.0)
        recency_score: How recent (0.0-1.0)
        citation_score: Citation count impact (0.0-1.0)
        methodology_score: Research quality (0.0-1.0)
        is_primary_source: Primary research vs review
        warnings: Quality warnings
        recommendations: Improvement suggestions
    """
    overall_score: float
    level: CredibilityLevel
    peer_reviewed: bool = False
    venue_prestige: float = 0.5
    author_credibility: float = 0.5
    recency_score: float = 0.5
    citation_score: float = 0.5
    methodology_score: float = 0.5
    is_primary_source: bool = False
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        lines.append(f"Credibility: {self.level.value.upper()} ({self.overall_score:.2f})")
        lines.append(f"\nBreakdown:")
        lines.append(f"  Peer-reviewed: {'✓ Yes' if self.peer_reviewed else '✗ No'}")
        lines.append(f"  Venue prestige: {self._format_score(self.venue_prestige)}")
        lines.append(f"  Author credibility: {self._format_score(self.author_credibility)}")
        lines.append(f"  Recency: {self._format_score(self.recency_score)}")
        lines.append(f"  Citations: {self._format_score(self.citation_score)}")
        lines.append(f"  Methodology: {self._format_score(self.methodology_score)}")
        lines.append(f"  Primary source: {'✓ Yes' if self.is_primary_source else '✗ No'}")

        if self.warnings:
            lines.append(f"\n⚠ Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.recommendations:
            lines.append(f"\n→ Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def _format_score(self, score: float) -> str:
        """Format score with visual indicator"""
        if score >= 0.8:
            return f"{'█' * 5} {score:.2f} (Excellent)"
        elif score >= 0.6:
            return f"{'█' * 4}{'░'} {score:.2f} (Good)"
        elif score >= 0.4:
            return f"{'█' * 3}{'░' * 2} {score:.2f} (Moderate)"
        elif score >= 0.2:
            return f"{'█' * 2}{'░' * 3} {score:.2f} (Low)"
        else:
            return f"{'█'}{'░' * 4} {score:.2f} (Very low)"


class SourceCredibilityScorer:
    """
    Evaluate research source credibility

    Uses multiple signals to provide comprehensive quality assessment:
    - Journal quality databases
    - Author h-index lookup (via Semantic Scholar, Google Scholar APIs)
    - Citation count analysis
    - Methodological quality indicators
    - Recency and relevance

    Usage:
        scorer = SourceCredibilityScorer()

        # Score a source
        metrics = scorer.score_source(
            title="Machine Learning Paper",
            authors=["Smith, J.", "Jones, M."],
            journal="Nature",
            year=2023,
            doi="10.1038/example"
        )

        print(metrics.get_summary())

        # Batch scoring
        scores = scorer.score_corpus(research_corpus)
        high_quality = [s for s in scores if s.overall_score > 0.7]
    """

    def __init__(self):
        """Initialize credibility scorer"""
        # Load journal quality databases
        self.journal_rankings = self._load_journal_rankings()
        self.predatory_journals = self._load_predatory_journals()

        # Citation databases (would integrate with APIs in production)
        self.citation_cache: Dict[str, int] = {}

    def score_source(self,
                    title: str,
                    authors: List[str],
                    journal: Optional[str] = None,
                    year: Optional[int] = None,
                    doi: Optional[str] = None,
                    source_type: str = "journal_article",
                    content: Optional[str] = None,
                    url: Optional[str] = None) -> CredibilityMetrics:
        """
        Score a research source

        Args:
            title: Source title
            authors: Author list
            journal: Journal/venue name
            year: Publication year
            doi: Digital Object Identifier
            source_type: Type of source
            content: Full text content (for methodology analysis)
            url: Source URL

        Returns:
            CredibilityMetrics with detailed breakdown
        """
        # Score components
        peer_reviewed = self._is_peer_reviewed(journal, source_type, content)
        venue_prestige = self._score_venue_prestige(journal, source_type, url)
        author_credibility = self._score_author_credibility(authors)
        recency_score = self._score_recency(year)
        citation_score = self._score_citations(doi, title, year)
        methodology_score = self._score_methodology(content, source_type)
        is_primary = self._is_primary_source(content, source_type)

        # Calculate overall score (weighted average)
        weights = {
            'peer_reviewed': 0.25,      # Most important
            'venue_prestige': 0.20,
            'author_credibility': 0.15,
            'recency': 0.10,
            'citations': 0.10,
            'methodology': 0.15,
            'primary_source': 0.05,
        }

        overall = (
            (1.0 if peer_reviewed else 0.0) * weights['peer_reviewed'] +
            venue_prestige * weights['venue_prestige'] +
            author_credibility * weights['author_credibility'] +
            recency_score * weights['recency'] +
            citation_score * weights['citations'] +
            methodology_score * weights['methodology'] +
            (1.0 if is_primary else 0.5) * weights['primary_source']
        )

        # Classify level
        if overall >= 0.9:
            level = CredibilityLevel.EXCELLENT
        elif overall >= 0.7:
            level = CredibilityLevel.GOOD
        elif overall >= 0.5:
            level = CredibilityLevel.MODERATE
        elif overall >= 0.3:
            level = CredibilityLevel.QUESTIONABLE
        else:
            level = CredibilityLevel.POOR

        # Generate warnings and recommendations
        warnings = self._generate_warnings(
            peer_reviewed, venue_prestige, recency_score, journal
        )
        recommendations = self._generate_recommendations(
            level, peer_reviewed, venue_prestige, is_primary
        )

        return CredibilityMetrics(
            overall_score=overall,
            level=level,
            peer_reviewed=peer_reviewed,
            venue_prestige=venue_prestige,
            author_credibility=author_credibility,
            recency_score=recency_score,
            citation_score=citation_score,
            methodology_score=methodology_score,
            is_primary_source=is_primary,
            warnings=warnings,
            recommendations=recommendations,
        )

    def score_corpus(self, corpus) -> List[Tuple[str, CredibilityMetrics]]:
        """
        Score all sources in a research corpus

        Args:
            corpus: ResearchCorpus with sources

        Returns:
            List of (source_id, metrics) tuples sorted by score
        """
        scores = []

        for source in corpus.sources:
            if source.citation:
                metrics = self.score_source(
                    title=source.citation.title,
                    authors=source.citation.authors,
                    journal=source.citation.journal,
                    year=source.citation.year,
                    doi=source.citation.doi,
                    source_type=source.citation.source_type.value,
                    content=source.content,
                    url=source.citation.url,
                )
                scores.append((source.id, metrics))

        # Sort by overall score (descending)
        scores.sort(key=lambda x: x[1].overall_score, reverse=True)
        return scores

    # ========================================================================
    # Scoring methods
    # ========================================================================

    def _is_peer_reviewed(self,
                         journal: Optional[str],
                         source_type: str,
                         content: Optional[str]) -> bool:
        """Check if source is peer-reviewed"""
        # Check journal database
        if journal:
            journal_lower = journal.lower()

            # Check predatory journal list
            if any(pred in journal_lower for pred in self.predatory_journals):
                return False

            # Check known peer-reviewed journals
            if journal_lower in self.journal_rankings:
                return True

            # Check common indicators
            peer_reviewed_indicators = [
                'nature', 'science', 'cell', 'lancet', 'jama', 'nejm',
                'ieee', 'acm', 'springer', 'elsevier', 'wiley',
                'proceedings', 'journal of', 'transactions on'
            ]

            if any(indicator in journal_lower for indicator in peer_reviewed_indicators):
                return True

        # Check source type
        if source_type in ['journal_article', 'conference']:
            return True

        # Check content for peer review indicators
        if content:
            content_lower = content[:2000].lower()
            if 'peer review' in content_lower or 'peer-reviewed' in content_lower:
                return True

        return False

    def _score_venue_prestige(self,
                              journal: Optional[str],
                              source_type: str,
                              url: Optional[str]) -> float:
        """Score venue/journal prestige"""
        if not journal:
            # Domain-based scoring for web sources
            if url:
                domain = url.split('/')[2] if '/' in url else url

                high_prestige_domains = {
                    '.gov': 0.9,
                    '.edu': 0.8,
                    'nih.gov': 1.0,
                    'cdc.gov': 0.95,
                    'who.int': 0.95,
                    'arxiv.org': 0.7,
                    'biorxiv.org': 0.7,
                }

                for domain_pattern, score in high_prestige_domains.items():
                    if domain_pattern in domain:
                        return score

            return 0.3  # Unknown venue

        journal_lower = journal.lower()

        # Check rankings database
        if journal_lower in self.journal_rankings:
            return self.journal_rankings[journal_lower]

        # Top-tier journals (Nature, Science, etc.)
        top_tier = ['nature', 'science', 'cell', 'lancet', 'jama', 'nejm', 'pnas']
        if any(t in journal_lower for t in top_tier):
            return 0.95

        # Major publishers (reputable but not top-tier)
        major_publishers = ['springer', 'elsevier', 'wiley', 'ieee', 'acm']
        if any(pub in journal_lower for pub in major_publishers):
            return 0.75

        # Generic journals
        if 'journal' in journal_lower or 'proceedings' in journal_lower:
            return 0.6

        return 0.5

    def _score_author_credibility(self, authors: List[str]) -> float:
        """Score author credibility"""
        # Would integrate with Semantic Scholar, Google Scholar APIs in production
        # For MVP, use heuristics

        if not authors or authors == ["Unknown Author"]:
            return 0.2

        # Assume established authors have published multiple papers
        # This is a placeholder - real implementation would look up h-index
        author_count = len(authors)

        if author_count >= 5:
            # Large author list suggests collaborative research
            return 0.7
        elif author_count >= 3:
            return 0.6
        elif author_count == 2:
            return 0.5
        else:
            return 0.4

    def _score_recency(self, year: Optional[int]) -> float:
        """Score based on publication recency"""
        if not year:
            return 0.3

        current_year = datetime.now().year
        age = current_year - year

        if age < 0:
            # Future date (error)
            return 0.0
        elif age <= 2:
            # Very recent (last 2 years)
            return 1.0
        elif age <= 5:
            # Recent (3-5 years)
            return 0.8
        elif age <= 10:
            # Moderately recent (6-10 years)
            return 0.6
        elif age <= 20:
            # Older but still relevant (11-20 years)
            return 0.4
        else:
            # Very old (>20 years)
            return 0.2

    def _score_citations(self,
                        doi: Optional[str],
                        title: str,
                        year: Optional[int]) -> float:
        """Score based on citation count"""
        # Would integrate with citation APIs in production
        # For MVP, use estimates

        if doi and doi in self.citation_cache:
            citation_count = self.citation_cache[doi]
        else:
            # Estimate based on age
            if year:
                age = datetime.now().year - year
                # Older papers accumulate more citations
                citation_count = max(0, age * 5)  # ~5 citations per year
            else:
                citation_count = 0

        # Score based on citation count
        if citation_count >= 100:
            return 1.0
        elif citation_count >= 50:
            return 0.9
        elif citation_count >= 20:
            return 0.7
        elif citation_count >= 10:
            return 0.5
        elif citation_count >= 5:
            return 0.3
        else:
            return 0.1

    def _score_methodology(self,
                          content: Optional[str],
                          source_type: str) -> float:
        """Score methodological rigor"""
        if not content:
            return 0.5  # Unknown

        content_lower = content[:3000].lower()

        # Check for methodological indicators
        methodology_indicators = {
            'methods': 0.2,
            'methodology': 0.2,
            'participants': 0.15,
            'sample size': 0.1,
            'statistical analysis': 0.15,
            'data collection': 0.1,
            'randomized': 0.1,
            'controlled': 0.1,
            'significant': 0.1,
        }

        score = 0.0
        for indicator, weight in methodology_indicators.items():
            if indicator in content_lower:
                score += weight

        # Cap at 1.0
        return min(1.0, score)

    def _is_primary_source(self,
                          content: Optional[str],
                          source_type: str) -> bool:
        """Check if source is primary research"""
        if not content:
            return source_type == 'journal_article'

        content_lower = content[:3000].lower()

        # Primary research indicators
        primary_indicators = [
            'methods', 'participants', 'results', 'data collection',
            'we conducted', 'we found', 'our study'
        ]

        # Secondary research indicators
        secondary_indicators = [
            'review', 'meta-analysis', 'survey of', 'previous research',
            'literature review', 'systematic review'
        ]

        primary_score = sum(1 for ind in primary_indicators if ind in content_lower)
        secondary_score = sum(1 for ind in secondary_indicators if ind in content_lower)

        return primary_score > secondary_score

    # ========================================================================
    # Helper methods
    # ========================================================================

    def _load_journal_rankings(self) -> Dict[str, float]:
        """Load journal quality rankings"""
        # Would load from database in production
        # For MVP, hardcoded top journals
        return {
            'nature': 0.98,
            'science': 0.98,
            'cell': 0.95,
            'lancet': 0.95,
            'jama': 0.92,
            'new england journal of medicine': 0.95,
            'pnas': 0.90,
            'nature communications': 0.85,
            'science advances': 0.85,
            'plos one': 0.70,
            'biorxiv': 0.65,  # Preprints
            'arxiv': 0.65,
        }

    def _load_predatory_journals(self) -> Set[str]:
        """Load list of predatory journals"""
        # Would load from Beall's list or similar in production
        # For MVP, placeholder
        return {
            'international journal of advanced research',
            'global journal of',
            'universal journal of',
        }

    def _generate_warnings(self,
                          peer_reviewed: bool,
                          venue_prestige: float,
                          recency_score: float,
                          journal: Optional[str]) -> List[str]:
        """Generate quality warnings"""
        warnings = []

        if not peer_reviewed:
            warnings.append("Source is not peer-reviewed")

        if venue_prestige < 0.4:
            warnings.append("Low-prestige venue or unknown journal")

        if recency_score < 0.3:
            warnings.append("Source is quite old (>20 years)")

        if journal and any(pred in journal.lower() for pred in self.predatory_journals):
            warnings.append("⚠ CAUTION: Possible predatory journal")

        return warnings

    def _generate_recommendations(self,
                                 level: CredibilityLevel,
                                 peer_reviewed: bool,
                                 venue_prestige: float,
                                 is_primary: bool) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []

        if level in [CredibilityLevel.POOR, CredibilityLevel.QUESTIONABLE]:
            recommendations.append("Consider replacing with higher-quality source")

        if not peer_reviewed:
            recommendations.append("Seek peer-reviewed alternative")

        if venue_prestige < 0.5:
            recommendations.append("Look for publications in established journals")

        if not is_primary:
            recommendations.append("Consider citing original research instead of review")

        return recommendations


# Factory function
def create_credibility_scorer() -> SourceCredibilityScorer:
    """Create a new credibility scorer"""
    return SourceCredibilityScorer()

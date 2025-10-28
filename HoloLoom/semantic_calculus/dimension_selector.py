#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 SMART DIMENSION SELECTOR
===========================
Intelligently selects optimal subset of dimensions from 244D space.

Purpose:
    FUSED mode needs a middle ground between FAST (16D) and RESEARCH (244D).
    This module selects the "best" 36 dimensions based on:

    1. **Category Balance** - Ensure all categories represented
    2. **Discriminative Power** - Dimensions with high variance
    3. **Importance Weights** - Based on usage patterns
    4. **Domain Relevance** - Narrative vs dialogue vs technical

Selection Strategies:
    - BALANCED: Equal representation from each category
    - DISCRIMINATIVE: Highest variance dimensions
    - NARRATIVE: Optimized for story analysis
    - DIALOGUE: Optimized for conversation
    - HYBRID: Weighted combination (default)
"""

# Force UTF-8 encoding for Windows console
import sys
import io
if sys.platform == "win32":
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from .dimensions import (
        SemanticDimension,
        STANDARD_DIMENSIONS,
        EXTENDED_244_DIMENSIONS,
    )
except ImportError:
    from dimensions import (
        SemanticDimension,
        STANDARD_DIMENSIONS,
        EXTENDED_244_DIMENSIONS,
    )


class SelectionStrategy(Enum):
    """Strategy for selecting dimensions."""
    BALANCED = "balanced"           # Equal from each category
    DISCRIMINATIVE = "discriminative"  # Highest variance
    NARRATIVE = "narrative"         # Optimized for stories
    DIALOGUE = "dialogue"          # Optimized for conversations
    HYBRID = "hybrid"              # Weighted combination (default)


@dataclass
class DimensionScore:
    """Scoring information for a dimension."""
    dimension: SemanticDimension
    category: str
    variance_score: float = 0.0
    balance_score: float = 0.0
    domain_score: float = 0.0
    total_score: float = 0.0


class SmartDimensionSelector:
    """
    Intelligently selects optimal dimension subset.

    Example:
        >>> selector = SmartDimensionSelector()
        >>> dims_36 = selector.select(
        ...     n_dimensions=36,
        ...     strategy=SelectionStrategy.HYBRID
        ... )
        >>> print(f"Selected {len(dims_36)} dimensions")
    """

    # Category definitions
    CATEGORIES = {
        'Core': [
            'Warmth', 'Valence', 'Arousal', 'Intensity',
            'Formality', 'Directness', 'Power', 'Generosity',
            'Certainty', 'Complexity', 'Concreteness', 'Familiarity',
            'Agency', 'Stability', 'Urgency', 'Completion'
        ],
        'Narrative': [
            'Heroism', 'Transformation', 'Conflict', 'Mystery', 'Sacrifice',
            'Wisdom', 'Courage', 'Redemption', 'Destiny', 'Honor', 'Loyalty',
            'Quest', 'Transcendence', 'Shadow', 'Initiation', 'Rebirth'
        ],
        'Emotional': [
            'Authenticity', 'Vulnerability', 'Trust', 'Hope', 'Grief', 'Shame',
            'Compassion', 'Rage', 'Longing', 'Awe', 'Jealousy', 'Guilt',
            'Pride', 'Disgust', 'Ecstasy', 'Dread'
        ],
        'Archetypal': [
            'Hero-Archetype', 'Mentor-Archetype', 'Shadow-Archetype',
            'Trickster-Archetype', 'Mother-Archetype', 'Father-Archetype',
            'Child-Archetype', 'Anima-Animus', 'Self-Archetype',
            'Threshold-Guardian', 'Herald', 'Ally', 'Shapeshifter',
            'Oracle', 'Ruler', 'Lover'
        ],
        'Philosophical': [
            'Freedom', 'Meaning', 'Being', 'Essence', 'Absurdity',
            'Time-Consciousness', 'Death-Awareness', 'Anxiety',
            'Responsibility', 'Care', 'Truth-Aletheia'
        ],
        'Theme': [
            'Love-Hate', 'War-Peace', 'Fate-Free-Will', 'Order-Chaos',
            'Mortality-Immortality', 'Knowledge-Ignorance',
            'Appearance-Reality', 'Nature-Culture'
        ],
        'Plot': [
            'Irony', 'Hubris', 'Nemesis', 'Hamartia', 'Catastrophe',
            'Suspense', 'Climax', 'Reversal', 'Recognition'
        ],
    }

    def __init__(self, all_dimensions: Optional[List[SemanticDimension]] = None):
        """
        Initialize selector.

        Args:
            all_dimensions: Full dimension set (defaults to EXTENDED_244_DIMENSIONS)
        """
        self.all_dimensions = all_dimensions or EXTENDED_244_DIMENSIONS
        self._categorize_dimensions()

    def _categorize_dimensions(self):
        """Build mapping of dimensions to categories."""
        self.dim_to_category = {}

        for dim in self.all_dimensions:
            # Find which category this dimension belongs to
            category = 'Other'
            for cat_name, cat_dims in self.CATEGORIES.items():
                if dim.name in cat_dims:
                    category = cat_name
                    break
            self.dim_to_category[dim.name] = category

    def compute_variance_scores(
        self,
        embed_fn: Callable,
        sample_texts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute variance score for each dimension.

        Higher variance = more discriminative power.

        Args:
            embed_fn: Embedding function
            sample_texts: Optional texts to test on (uses defaults if None)

        Returns:
            Dict mapping dimension name -> variance score
        """
        if sample_texts is None:
            # Default diverse sample
            sample_texts = [
                "love", "hate", "fear", "joy", "anger", "peace",
                "war", "birth", "death", "hope", "despair",
                "hero", "villain", "mentor", "trickster",
                "beginning", "middle", "end", "transformation"
            ]

        # Get embeddings
        embeddings = np.array([embed_fn(text) for text in sample_texts])

        # Compute projections for each dimension
        variances = {}
        for dim in self.all_dimensions:
            if not hasattr(dim, 'axis') or dim.axis is None:
                # Need to learn axis first
                dim.learn_axis(embed_fn)

            # Project all samples onto this dimension
            projections = [dim.project(emb) for emb in embeddings]

            # Compute variance
            variance = np.var(projections)
            variances[dim.name] = float(variance)

        return variances

    def compute_balance_scores(self) -> Dict[str, float]:
        """
        Compute balance scores based on category representation.

        Ensures we don't over-select from any single category.

        Returns:
            Dict mapping dimension name -> balance score
        """
        # Count dimensions per category
        category_counts = {}
        for dim in self.all_dimensions:
            cat = self.dim_to_category[dim.name]
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Balance score is inverse of category size
        # (prefer dimensions from underrepresented categories)
        balance_scores = {}
        for dim in self.all_dimensions:
            cat = self.dim_to_category[dim.name]
            count = category_counts[cat]
            balance_scores[dim.name] = 1.0 / count

        return balance_scores

    def compute_domain_scores(
        self,
        domain: str = 'narrative'
    ) -> Dict[str, float]:
        """
        Compute domain-specific importance scores.

        Args:
            domain: 'narrative', 'dialogue', 'technical', or 'general'

        Returns:
            Dict mapping dimension name -> domain relevance score
        """
        # Domain-specific weights
        domain_weights = {
            'narrative': {
                'Core': 0.5,
                'Narrative': 1.0,
                'Emotional': 0.8,
                'Archetypal': 0.9,
                'Philosophical': 0.6,
                'Theme': 0.8,
                'Plot': 0.9,
                'Other': 0.3,
            },
            'dialogue': {
                'Core': 1.0,
                'Narrative': 0.4,
                'Emotional': 0.9,
                'Archetypal': 0.3,
                'Philosophical': 0.5,
                'Theme': 0.4,
                'Plot': 0.3,
                'Other': 0.5,
            },
            'technical': {
                'Core': 1.0,
                'Narrative': 0.2,
                'Emotional': 0.3,
                'Archetypal': 0.1,
                'Philosophical': 0.7,
                'Theme': 0.3,
                'Plot': 0.1,
                'Other': 0.8,
            },
            'general': {
                'Core': 0.8,
                'Narrative': 0.6,
                'Emotional': 0.7,
                'Archetypal': 0.5,
                'Philosophical': 0.6,
                'Theme': 0.5,
                'Plot': 0.4,
                'Other': 0.5,
            },
        }

        weights = domain_weights.get(domain, domain_weights['general'])

        scores = {}
        for dim in self.all_dimensions:
            cat = self.dim_to_category[dim.name]
            scores[dim.name] = weights.get(cat, 0.5)

        return scores

    def select(
        self,
        n_dimensions: int = 36,
        strategy: SelectionStrategy = SelectionStrategy.HYBRID,
        embed_fn: Optional[Callable] = None,
        domain: str = 'narrative',
        weights: Optional[Dict[str, float]] = None
    ) -> List[SemanticDimension]:
        """
        Select optimal subset of dimensions.

        Args:
            n_dimensions: Number of dimensions to select (default 36)
            strategy: Selection strategy
            embed_fn: Embedding function (required for DISCRIMINATIVE)
            domain: Domain for domain-specific scoring
            weights: Custom weights for HYBRID strategy
                    {variance: 0.4, balance: 0.3, domain: 0.3}

        Returns:
            List of selected SemanticDimension objects
        """
        print(f"🎯 Smart Dimension Selection: {n_dimensions}D from {len(self.all_dimensions)}D")
        print(f"   Strategy: {strategy.value}")
        print(f"   Domain: {domain}")

        # Compute scores
        variance_scores = {}
        if strategy in [SelectionStrategy.DISCRIMINATIVE, SelectionStrategy.HYBRID]:
            if embed_fn is None:
                print("   ⚠️  No embed_fn provided, using balance-only scoring")
                strategy = SelectionStrategy.BALANCED
            else:
                print("   Computing variance scores...")
                variance_scores = self.compute_variance_scores(embed_fn)

        balance_scores = self.compute_balance_scores()
        domain_scores = self.compute_domain_scores(domain)

        # Create dimension scores
        dim_scores = []
        for dim in self.all_dimensions:
            score = DimensionScore(
                dimension=dim,
                category=self.dim_to_category[dim.name],
                variance_score=variance_scores.get(dim.name, 0.0),
                balance_score=balance_scores[dim.name],
                domain_score=domain_scores[dim.name],
            )
            dim_scores.append(score)

        # Apply strategy
        if strategy == SelectionStrategy.BALANCED:
            # Pure category balance
            for score in dim_scores:
                score.total_score = score.balance_score

        elif strategy == SelectionStrategy.DISCRIMINATIVE:
            # Pure variance
            for score in dim_scores:
                score.total_score = score.variance_score

        elif strategy == SelectionStrategy.NARRATIVE:
            # Narrative-optimized: 60% domain, 40% balance
            for score in dim_scores:
                score.total_score = 0.6 * score.domain_score + 0.4 * score.balance_score

        elif strategy == SelectionStrategy.DIALOGUE:
            # Dialogue-optimized: 70% domain, 30% variance
            domain_dialogue = self.compute_domain_scores('dialogue')
            for score in dim_scores:
                score.domain_score = domain_dialogue[score.dimension.name]
                score.total_score = 0.7 * score.domain_score + 0.3 * score.variance_score

        elif strategy == SelectionStrategy.HYBRID:
            # Default hybrid: balanced combination
            if weights is None:
                weights = {'variance': 0.35, 'balance': 0.30, 'domain': 0.35}

            for score in dim_scores:
                score.total_score = (
                    weights.get('variance', 0.35) * score.variance_score +
                    weights.get('balance', 0.30) * score.balance_score +
                    weights.get('domain', 0.35) * score.domain_score
                )

        # Sort by total score
        dim_scores.sort(key=lambda x: x.total_score, reverse=True)

        # Select top N
        selected = [score.dimension for score in dim_scores[:n_dimensions]]

        # Print summary
        print(f"\n   ✅ Selected {len(selected)} dimensions:")
        category_dist = {}
        for score in dim_scores[:n_dimensions]:
            cat = score.category
            category_dist[cat] = category_dist.get(cat, 0) + 1

        for cat, count in sorted(category_dist.items()):
            print(f"      {cat}: {count}")

        print(f"\n   Top 10 selected dimensions:")
        for i, score in enumerate(dim_scores[:10], 1):
            print(f"      {i:2d}. {score.dimension.name:<25} "
                  f"[{score.category}] score={score.total_score:.3f}")

        return selected

    def get_selection_report(
        self,
        selected: List[SemanticDimension]
    ) -> Dict:
        """
        Generate detailed report on selection.

        Args:
            selected: List of selected dimensions

        Returns:
            Report dictionary with statistics
        """
        # Category distribution
        category_dist = {}
        for dim in selected:
            cat = self.dim_to_category[dim.name]
            category_dist[cat] = category_dist.get(cat, 0) + 1

        # Full dimension list
        dim_list = [
            {
                'name': dim.name,
                'category': self.dim_to_category[dim.name],
                'positive': dim.positive_exemplars,
                'negative': dim.negative_exemplars,
            }
            for dim in selected
        ]

        return {
            'n_selected': len(selected),
            'n_total': len(self.all_dimensions),
            'category_distribution': category_dist,
            'dimensions': dim_list,
        }


def create_fused_36d_selection(
    embed_fn: Optional[Callable] = None,
    strategy: SelectionStrategy = SelectionStrategy.HYBRID,
    domain: str = 'narrative'
) -> List[SemanticDimension]:
    """
    Convenience function to create optimal 36D selection for FUSED mode.

    Args:
        embed_fn: Optional embedding function for variance scoring
        strategy: Selection strategy (default HYBRID)
        domain: Domain context (default 'narrative')

    Returns:
        List of 36 SemanticDimension objects

    Example:
        >>> from HoloLoom.embedding.spectral import create_embedder
        >>> embed_model = create_embedder(sizes=[384])
        >>> embed_fn = lambda text: embed_model.encode([text])[0]
        >>> dims_36 = create_fused_36d_selection(embed_fn)
        >>> print(f"Selected {len(dims_36)} dimensions for FUSED mode")
    """
    selector = SmartDimensionSelector()
    return selector.select(
        n_dimensions=36,
        strategy=strategy,
        embed_fn=embed_fn,
        domain=domain
    )


if __name__ == "__main__":
    # Demo: Show selection for different strategies
    print("=" * 80)
    print("🎯 SMART DIMENSION SELECTOR DEMO")
    print("=" * 80)

    selector = SmartDimensionSelector()

    # Test different strategies
    strategies = [
        SelectionStrategy.BALANCED,
        SelectionStrategy.NARRATIVE,
    ]

    for strategy in strategies:
        print(f"\n{'='*80}")
        dims = selector.select(n_dimensions=36, strategy=strategy)
        print(f"{'='*80}\n")

    print("\n✅ Demo complete!")

"""
Semantic Cluster Labeling

Uses HoloLoom's 228D semantic dimensions to generate meaningful,
interpretable labels for clusters.

Instead of "Cluster 0", you get "Machine Learning / Technical".
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings

# Try to import semantic dimensions
try:
    from HoloLoom.semantic_calculus.dimensions import (
        SemanticDimension,
        STANDARD_DIMENSIONS
    )
    _HAVE_SEMANTIC = True
except ImportError:
    _HAVE_SEMANTIC = False
    STANDARD_DIMENSIONS = []


# Domain categories for cluster labeling
DOMAIN_CATEGORIES = {
    "technology": {
        "keywords": ["software", "code", "programming", "algorithm", "computer",
                     "data", "system", "network", "api", "database", "cloud",
                     "machine", "learning", "ai", "neural", "model"],
        "subcategories": {
            "ai_ml": ["machine learning", "neural", "ai", "deep learning", "model", "training"],
            "web": ["web", "http", "api", "frontend", "backend", "javascript"],
            "data": ["data", "database", "sql", "analytics", "pipeline"],
            "devops": ["cloud", "docker", "kubernetes", "deploy", "ci/cd"],
        }
    },
    "science": {
        "keywords": ["research", "study", "experiment", "theory", "hypothesis",
                     "analysis", "method", "result", "finding", "evidence"],
        "subcategories": {
            "physics": ["physics", "quantum", "energy", "particle", "wave"],
            "biology": ["biology", "cell", "gene", "protein", "organism"],
            "chemistry": ["chemistry", "molecule", "reaction", "compound"],
        }
    },
    "business": {
        "keywords": ["business", "company", "market", "customer", "product",
                     "strategy", "growth", "revenue", "team", "management"],
        "subcategories": {
            "marketing": ["marketing", "brand", "campaign", "customer", "audience"],
            "finance": ["finance", "investment", "revenue", "profit", "budget"],
            "operations": ["operations", "process", "efficiency", "workflow"],
        }
    },
    "creative": {
        "keywords": ["design", "create", "art", "story", "write", "music",
                     "visual", "creative", "content", "media"],
        "subcategories": {
            "writing": ["write", "story", "content", "article", "blog"],
            "design": ["design", "visual", "ui", "ux", "graphic"],
            "media": ["video", "audio", "music", "film", "photo"],
        }
    }
}


@dataclass
class LabelCandidate:
    """A candidate label for a cluster."""
    primary: str
    secondary: Optional[str] = None
    confidence: float = 0.0
    domain: Optional[str] = None

    def __str__(self):
        if self.secondary:
            return f"{self.primary} / {self.secondary}"
        return self.primary


def label_clusters(
    clusters: List[Any],
    embeddings: np.ndarray,
    texts: List[str],
    use_semantic: bool = True,
    use_domain: bool = True
) -> List[Any]:
    """
    Generate semantic labels for clusters.

    Args:
        clusters: List of ClusterResult objects
        embeddings: All embeddings (n_samples x dim)
        texts: All text items
        use_semantic: Use 228D semantic dimensions for labeling
        use_domain: Use domain category detection

    Returns:
        Clusters with updated labels
    """
    for cluster in clusters:
        # Get cluster texts
        cluster_texts = cluster.items
        cluster_text_combined = ' '.join(cluster_texts).lower()

        # Generate label candidates
        candidates = []

        # 1. Domain-based labeling
        if use_domain:
            domain_label = _detect_domain(cluster_text_combined)
            if domain_label:
                candidates.append(domain_label)

        # 2. Semantic dimension-based labeling
        if use_semantic and _HAVE_SEMANTIC and cluster.centroid is not None:
            semantic_label = _get_semantic_label(cluster.centroid)
            if semantic_label:
                candidates.append(semantic_label)

        # 3. Keyword extraction fallback
        keyword_label = _extract_keywords(cluster_texts)
        if keyword_label:
            candidates.append(keyword_label)

        # Select best label
        if candidates:
            # Sort by confidence
            candidates.sort(key=lambda x: x.confidence, reverse=True)
            best = candidates[0]
            cluster.label = str(best)
        else:
            cluster.label = f"Group {clusters.index(cluster) + 1}"

    return clusters


def _detect_domain(text: str) -> Optional[LabelCandidate]:
    """Detect domain category from text."""
    text_lower = text.lower()

    best_domain = None
    best_score = 0
    best_subcategory = None

    for domain, info in DOMAIN_CATEGORIES.items():
        # Count keyword matches
        score = sum(1 for kw in info["keywords"] if kw in text_lower)

        if score > best_score:
            best_score = score
            best_domain = domain

            # Check subcategories
            for subcat, keywords in info.get("subcategories", {}).items():
                sub_score = sum(1 for kw in keywords if kw in text_lower)
                if sub_score >= 2:  # Require at least 2 matches
                    best_subcategory = subcat.replace("_", " ").title()
                    break

    if best_domain and best_score >= 2:
        confidence = min(best_score / 10, 1.0)
        return LabelCandidate(
            primary=best_domain.title(),
            secondary=best_subcategory,
            confidence=confidence,
            domain=best_domain
        )

    return None


def _get_semantic_label(centroid: np.ndarray) -> Optional[LabelCandidate]:
    """Generate label from semantic dimension projections."""
    if not _HAVE_SEMANTIC or not STANDARD_DIMENSIONS:
        return None

    # Project centroid onto semantic dimensions
    projections = {}

    for dim in STANDARD_DIMENSIONS:
        if dim.axis is not None:
            # Ensure dimensions match
            if len(centroid) == len(dim.axis):
                proj = float(np.dot(centroid, dim.axis))
                projections[dim.name] = proj

    if not projections:
        return None

    # Find most distinctive dimensions (highest absolute values)
    sorted_dims = sorted(projections.items(), key=lambda x: abs(x[1]), reverse=True)

    if not sorted_dims:
        return None

    # Get top dimension and its polarity
    top_dim, top_val = sorted_dims[0]

    # Map dimension to human-readable label
    polarity = "Positive" if top_val > 0 else "Negative"

    # Get secondary dimension if available
    secondary = None
    if len(sorted_dims) > 1:
        second_dim, second_val = sorted_dims[1]
        if abs(second_val) > 0.3:  # Only include if significant
            secondary = f"{second_dim.title()}"

    confidence = min(abs(top_val), 1.0)

    return LabelCandidate(
        primary=top_dim.title(),
        secondary=secondary,
        confidence=confidence * 0.8  # Slightly lower confidence for semantic
    )


def _extract_keywords(texts: List[str], top_n: int = 3) -> Optional[LabelCandidate]:
    """Extract representative keywords from cluster texts."""
    import re
    from collections import Counter

    # Expanded stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'and',
        'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither', 'not',
        'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also', 'now',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'any', 'this',
        'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your',
        'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
        'what', 'which', 'who', 'whom', 'whose', 'get', 'got', 'getting',
        'use', 'using', 'used', 'make', 'made', 'making', 'like', 'just',
        'know', 'think', 'see', 'want', 'way', 'look', 'first', 'new', 'one',
        'two', 'three', 'thing', 'things', 'time', 'year', 'years', 'day',
        'work', 'working', 'people', 'really', 'well', 'even', 'back', 'good'
    }

    # Extract all words
    all_text = ' '.join(texts).lower()
    words = re.findall(r'\b[a-z]{3,}\b', all_text)

    # Filter and count
    filtered = [w for w in words if w not in stopwords]
    counter = Counter(filtered)

    if not counter:
        return None

    # Get top words
    top_words = [word for word, count in counter.most_common(top_n)]

    if not top_words:
        return None

    # Calculate confidence based on word frequency
    total_words = len(filtered)
    top_count = sum(counter[w] for w in top_words)
    confidence = min(top_count / max(total_words, 1) * 2, 0.9)

    primary = top_words[0].title()
    secondary = top_words[1].title() if len(top_words) > 1 else None

    return LabelCandidate(
        primary=primary,
        secondary=secondary,
        confidence=confidence
    )


def generate_cluster_summary(
    clusters: List[Any],
    include_examples: int = 3
) -> str:
    """
    Generate a human-readable summary of clustering results.

    Args:
        clusters: List of ClusterResult objects
        include_examples: Number of example items per cluster

    Returns:
        Formatted summary string
    """
    lines = [
        f"Found {len(clusters)} clusters:",
        ""
    ]

    for i, cluster in enumerate(clusters, 1):
        lines.append(f"{i}. {cluster.label}")
        lines.append(f"   Size: {cluster.size} items")
        lines.append(f"   Confidence: {cluster.confidence:.1%}")

        # Show examples
        examples = cluster.items[:include_examples]
        if examples:
            lines.append("   Examples:")
            for ex in examples:
                # Truncate long examples
                ex_short = ex[:60] + "..." if len(ex) > 60 else ex
                lines.append(f"     - {ex_short}")

        lines.append("")

    return '\n'.join(lines)

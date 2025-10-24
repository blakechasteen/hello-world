"""
Reverse Query System
====================

Instead of "query → memories", this does "memory → queries".

Given a memory, find:
1. What queries would retrieve it?
2. What are the best keywords for finding it?
3. What related concepts lead to it?
4. How is it connected to other memories?

Use cases:
- Discoverability: "How would I find this again?"
- SEO for memory: "What should I search for?"
- Knowledge mapping: "What concepts link here?"
- Query suggestion: "People who viewed X also searched for..."
"""

from typing import List, Optional, Dict, Set, Tuple
from dataclasses import dataclass, field
from collections import Counter
import re


@dataclass
class ReverseQueryResult:
    """Result of reverse query analysis."""

    # Queries that would retrieve this memory
    exact_queries: List[str]           # Guaranteed to return this
    likely_queries: List[Tuple[str, float]]  # High probability (query, score)
    possible_queries: List[Tuple[str, float]]  # Lower probability

    # Keywords
    primary_keywords: List[str]        # Core topics
    secondary_keywords: List[str]      # Related concepts
    rare_keywords: List[str]           # Unique distinguishing terms

    # Concepts
    concepts: List[str]                # Extracted concepts
    entities: List[str]                # Named entities

    # Connections
    related_memories: List[str]        # Similar/related memory IDs
    incoming_links: List[str]          # What links to this?
    outgoing_links: List[str]          # What does this link to?

    # Metadata
    discoverability_score: float       # How easy to find (0-1)
    uniqueness_score: float            # How unique/rare (0-1)


class ReverseQueryEngine:
    """Engine for reverse query analysis."""

    def __init__(self, memory_store):
        self.store = memory_store

        # Build indices
        self.term_to_memories: Dict[str, Set[str]] = {}  # term → memory_ids
        self.memory_to_terms: Dict[str, Set[str]] = {}   # memory_id → terms
        self.term_frequencies: Counter = Counter()       # Global term counts

    async def analyze(self, memory_id: str, memory_text: str) -> ReverseQueryResult:
        """
        Analyze a memory to find what queries would retrieve it.

        Args:
            memory_id: Memory identifier
            memory_text: Memory content

        Returns:
            ReverseQueryResult with analysis
        """

        # Extract terms from memory
        terms = self._extract_terms(memory_text)

        # Find exact match queries (rare terms)
        exact = self._find_exact_queries(terms)

        # Find likely queries (combinations of common terms)
        likely = self._find_likely_queries(terms)

        # Find possible queries (broader concepts)
        possible = self._find_possible_queries(terms)

        # Categorize keywords
        primary, secondary, rare = self._categorize_keywords(terms)

        # Extract concepts and entities
        concepts = self._extract_concepts(memory_text)
        entities = self._extract_entities(memory_text)

        # Find related memories
        related = await self._find_related_memories(memory_id, terms)

        # Calculate scores
        discoverability = self._calculate_discoverability(terms)
        uniqueness = self._calculate_uniqueness(terms)

        return ReverseQueryResult(
            exact_queries=exact,
            likely_queries=likely,
            possible_queries=possible,
            primary_keywords=primary,
            secondary_keywords=secondary,
            rare_keywords=rare,
            concepts=concepts,
            entities=entities,
            related_memories=related,
            incoming_links=[],  # Would populate from graph
            outgoing_links=[],
            discoverability_score=discoverability,
            uniqueness_score=uniqueness
        )

    def _extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text."""
        # Tokenize
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Split into words
        words = text.split()

        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
            'who', 'when', 'where', 'why', 'how'
        }

        terms = [w for w in words if w not in stop_words and len(w) > 2]

        return terms

    def _find_exact_queries(self, terms: List[str]) -> List[str]:
        """
        Find queries that would exactly match this memory.

        Uses rare terms that uniquely identify this content.
        """
        # Find rarest terms (most distinctive)
        term_counts = Counter(terms)

        # Score terms by rarity (inverse document frequency)
        rare_terms = []
        for term in set(terms):
            # Lower global frequency = more distinctive
            global_freq = self.term_frequencies.get(term, 1)
            if global_freq < 5:  # Appears in < 5 documents
                rare_terms.append(term)

        # Generate exact match queries
        exact = []

        # Single rare term queries
        for term in rare_terms[:5]:
            exact.append(term)

        # Combinations of top terms
        top_terms = [t for t, _ in term_counts.most_common(3)]
        if len(top_terms) >= 2:
            exact.append(' '.join(top_terms[:2]))

        return exact

    def _find_likely_queries(self, terms: List[str]) -> List[Tuple[str, float]]:
        """
        Find queries likely to retrieve this memory.

        Returns: List of (query, probability) tuples
        """
        term_counts = Counter(terms)
        likely = []

        # Top single-word queries
        for term, count in term_counts.most_common(10):
            # Probability based on term frequency in this doc vs others
            score = self._calculate_query_score(term, count)
            likely.append((term, score))

        # Two-word combinations
        top_terms = [t for t, _ in term_counts.most_common(5)]
        for i in range(len(top_terms)):
            for j in range(i + 1, len(top_terms)):
                query = f"{top_terms[i]} {top_terms[j]}"
                score = self._calculate_query_score(query, 1)
                likely.append((query, score))

        # Sort by score
        likely.sort(key=lambda x: x[1], reverse=True)

        return likely[:10]

    def _find_possible_queries(self, terms: List[str]) -> List[Tuple[str, float]]:
        """
        Find broader queries that might retrieve this.

        Returns: List of (query, probability) tuples
        """
        # Extract bigrams and trigrams
        ngrams = self._extract_ngrams(terms, n=2) + self._extract_ngrams(terms, n=3)

        possible = []
        for ngram in ngrams[:20]:
            query = ' '.join(ngram)
            score = self._calculate_query_score(query, 1) * 0.7  # Lower confidence
            possible.append((query, score))

        # Sort by score
        possible.sort(key=lambda x: x[1], reverse=True)

        return possible[:10]

    def _categorize_keywords(
        self,
        terms: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Categorize keywords by importance.

        Returns:
            (primary, secondary, rare) keyword lists
        """
        term_counts = Counter(terms)

        # Primary: Most frequent (core topics)
        primary = [t for t, _ in term_counts.most_common(5)]

        # Secondary: Medium frequency (related concepts)
        all_terms = term_counts.most_common()
        secondary = [t for t, c in all_terms[5:15]]

        # Rare: Low frequency but distinctive
        rare = []
        for term, count in term_counts.items():
            global_freq = self.term_frequencies.get(term, 0)
            if global_freq < 3 and count >= 1:
                rare.append(term)

        return primary, secondary, rare[:10]

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract high-level concepts from text."""
        # Simple version: capitalized words/phrases
        concepts = []

        # Find capitalized words (potential concepts)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # Count occurrences
        concept_counts = Counter(words)

        # Return top concepts
        concepts = [c for c, _ in concept_counts.most_common(10)]

        return concepts

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities."""
        # Simple version: capitalized sequences
        # Real version would use NER
        entities = []

        # Multi-word capitalized phrases
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        matches = re.findall(pattern, text)

        entities = list(set(matches))

        return entities[:10]

    async def _find_related_memories(
        self,
        memory_id: str,
        terms: List[str]
    ) -> List[str]:
        """Find memories with similar term profiles."""
        # This would query the actual store
        # Placeholder for now
        return []

    def _calculate_query_score(self, query: str, local_count: int) -> float:
        """
        Calculate probability that query would retrieve this memory.

        Uses TF-IDF like scoring.
        """
        # Term frequency in this document
        tf = local_count

        # Inverse document frequency (global rarity)
        global_freq = self.term_frequencies.get(query, 1)
        idf = 1.0 / (1 + global_freq)

        # Combine
        score = tf * idf

        # Normalize to 0-1
        return min(1.0, score / 10.0)

    def _calculate_discoverability(self, terms: List[str]) -> float:
        """
        How easy is this memory to find?

        High = common terms, appears in many contexts
        Low = rare terms, narrow topic
        """
        if not terms:
            return 0.0

        # Average term frequency
        total_freq = sum(self.term_frequencies.get(t, 0) for t in set(terms))
        avg_freq = total_freq / len(set(terms))

        # Normalize (higher = more discoverable)
        score = min(1.0, avg_freq / 100.0)

        return score

    def _calculate_uniqueness(self, terms: List[str]) -> float:
        """
        How unique/rare is this memory?

        High = rare terms, few similar documents
        Low = common terms, many similar documents
        """
        if not terms:
            return 0.0

        # Count rare terms
        rare_count = sum(1 for t in set(terms) if self.term_frequencies.get(t, 0) < 5)

        # Percentage of terms that are rare
        uniqueness = rare_count / max(len(set(terms)), 1)

        return uniqueness

    def _extract_ngrams(self, terms: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from terms."""
        ngrams = []
        for i in range(len(terms) - n + 1):
            ngrams.append(tuple(terms[i:i+n]))
        return ngrams

    def update_index(self, memory_id: str, terms: List[str]):
        """Update reverse index with new memory."""

        # Update term → memories mapping
        for term in set(terms):
            if term not in self.term_to_memories:
                self.term_to_memories[term] = set()
            self.term_to_memories[term].add(memory_id)

        # Update memory → terms mapping
        self.memory_to_terms[memory_id] = set(terms)

        # Update global term frequencies
        self.term_frequencies.update(terms)

    def suggest_improvements(self, memory_id: str, text: str) -> Dict[str, List[str]]:
        """
        Suggest how to make this memory more discoverable.

        Returns suggestions for:
        - Tags to add
        - Keywords to emphasize
        - Related content to link
        """
        terms = self._extract_terms(text)
        concepts = self._extract_concepts(text)

        suggestions = {
            'add_tags': [],
            'add_keywords': [],
            'create_links': []
        }

        # Suggest tags based on top concepts
        suggestions['add_tags'] = concepts[:5]

        # Suggest keywords from rare but relevant terms
        for term in set(terms):
            freq = self.term_frequencies.get(term, 0)
            if 3 <= freq <= 10:  # Sweet spot: not too rare, not too common
                suggestions['add_keywords'].append(term)

        suggestions['add_keywords'] = suggestions['add_keywords'][:5]

        return suggestions


# Convenience functions
async def what_queries_find_this(memory_id: str, text: str, engine: ReverseQueryEngine) -> List[str]:
    """Quick function: what queries would find this memory?"""
    result = await engine.analyze(memory_id, text)
    return result.exact_queries + [q for q, _ in result.likely_queries[:5]]


async def how_discoverable(memory_id: str, text: str, engine: ReverseQueryEngine) -> float:
    """Quick function: how easy is this to find? (0-1)"""
    result = await engine.analyze(memory_id, text)
    return result.discoverability_score


async def make_more_findable(memory_id: str, text: str, engine: ReverseQueryEngine) -> Dict:
    """Quick function: how to make this more discoverable?"""
    return engine.suggest_improvements(memory_id, text)

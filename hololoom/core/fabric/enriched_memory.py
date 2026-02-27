#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enriched Memory
===============
Enhanced memory structure with extracted patterns, entities, and relationships.

Turns raw conversation shards into structured knowledge ready for synthesis.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ReasoningType(Enum):
    """Type of reasoning in a memory."""
    QUESTION = "question"           # User asking for information
    ANSWER = "answer"               # System providing information
    DECISION = "decision"           # Choice or judgment
    FACT = "fact"                   # Statement of truth
    EXPLANATION = "explanation"     # How/why something works
    COMPARISON = "comparison"       # A vs B
    PROCEDURE = "procedure"         # How to do X
    OBSERVATION = "observation"     # Noticed pattern
    HYPOTHESIS = "hypothesis"       # Potential explanation
    REFLECTION = "reflection"       # Meta-cognitive insight


@dataclass
class EnrichedMemory:
    """
    Enhanced memory with extracted structure.

    Takes raw conversation and extracts:
    - Entities (people, places, concepts)
    - Relationships (X relates to Y)
    - Reasoning patterns (question, decision, fact)
    - Topics and themes
    - Temporal context
    - Dependencies (related memories)
    """
    # Core fields (from MemoryShard)
    id: str
    text: str
    timestamp: datetime
    importance: float

    # Enrichment fields
    reasoning_type: ReasoningType
    entities: List[str] = field(default_factory=list)
    relationships: List[Tuple[str, str, str]] = field(default_factory=list)  # (subject, predicate, object)
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # IDs of related memories

    # Conversation-specific
    user_input: Optional[str] = None
    system_output: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'text': self.text,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'reasoning_type': self.reasoning_type.value,
            'entities': self.entities,
            'relationships': [(s, p, o) for s, p, o in self.relationships],
            'topics': self.topics,
            'keywords': self.keywords,
            'dependencies': self.dependencies,
            'user_input': self.user_input,
            'system_output': self.system_output,
            'metadata': self.metadata
        }


class MemoryEnricher:
    """
    Enriches raw memories with extracted patterns.

    Extracts:
    - Entities (capitalized terms, domain concepts)
    - Relationships (X → Y connections)
    - Reasoning type (question, decision, fact, etc.)
    - Topics (themes, domains)
    - Keywords (important terms)
    """

    def __init__(self, domain_terms: Optional[List[str]] = None):
        """
        Initialize enricher.

        Args:
            domain_terms: Domain-specific terms to recognize
        """
        self.domain_terms = set(domain_terms or [])

        # Add default HoloLoom terms
        self.domain_terms.update([
            'hololoom', 'thompson', 'sampling', 'policy', 'memory',
            'shard', 'orchestrator', 'spinner', 'embedding', 'vector',
            'graph', 'knowledge', 'retrieval', 'neural', 'bandit',
            'matryoshka', 'spectral', 'motif', 'context', 'decision'
        ])

    def enrich(self, raw_memory: Dict[str, Any]) -> EnrichedMemory:
        """
        Enrich a raw memory with structure.

        Args:
            raw_memory: Raw memory dict with text, timestamp, etc.

        Returns:
            EnrichedMemory with extracted patterns
        """
        # Extract basic fields
        memory_id = raw_memory.get('id', 'unknown')
        text = raw_memory.get('text', '')
        timestamp = raw_memory.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        importance = raw_memory.get('importance', 0.5)

        # Parse conversation structure
        user_input, system_output = self._parse_conversation(text)

        # Extract entities
        entities = self._extract_entities(text)

        # Extract relationships
        relationships = self._extract_relationships(text, entities)

        # Determine reasoning type
        reasoning_type = self._classify_reasoning(user_input or text, system_output or '')

        # Extract topics
        topics = self._extract_topics(text)

        # Extract keywords
        keywords = self._extract_keywords(text)

        return EnrichedMemory(
            id=memory_id,
            text=text,
            timestamp=timestamp,
            importance=importance,
            reasoning_type=reasoning_type,
            entities=entities,
            relationships=relationships,
            topics=topics,
            keywords=keywords,
            user_input=user_input,
            system_output=system_output,
            metadata=raw_memory.get('metadata', {})
        )

    def _parse_conversation(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse conversation turn structure."""
        # Look for "User:" and "System:" markers
        user_match = re.search(r'User:\s*(.+?)(?=System:|Importance:|$)', text, re.DOTALL)
        system_match = re.search(r'System:\s*(.+?)(?=Importance:|$)', text, re.DOTALL)

        user_input = user_match.group(1).strip() if user_match else None
        system_output = system_match.group(1).strip() if system_match else None

        return user_input, system_output

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (capitalized terms, domain concepts)."""
        entities = set()

        # Capitalized words (proper nouns)
        caps_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        caps_words = re.findall(caps_pattern, text)

        # Filter common words
        common = {'The', 'This', 'That', 'These', 'Those', 'It', 'User', 'System'}
        for word in caps_words:
            if word not in common and len(word) > 2:
                entities.add(word)

        # Domain terms (case insensitive)
        text_lower = text.lower()
        for term in self.domain_terms:
            if term in text_lower:
                entities.add(term.title())

        return sorted(list(entities))[:20]  # Top 20

    def _extract_relationships(self, text: str,
                              entities: List[str]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities.

        Returns triples: (subject, predicate, object)
        """
        relationships = []

        if len(entities) < 2:
            return relationships

        # Simple pattern matching for relationships
        patterns = [
            (r'(\w+)\s+is\s+(?:a|an)\s+(\w+)', 'IS_A'),
            (r'(\w+)\s+uses\s+(\w+)', 'USES'),
            (r'(\w+)\s+relates to\s+(\w+)', 'RELATES_TO'),
            (r'(\w+)\s+combines\s+(\w+)', 'COMBINES'),
            (r'(\w+)\s+(?:contains|includes)\s+(\w+)', 'CONTAINS'),
            (r'(\w+)\s+(?:works with|integrates with)\s+(\w+)', 'WORKS_WITH'),
        ]

        for pattern, predicate in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).title()
                obj = match.group(2).title()

                # Only include if both are recognized entities or domain terms
                if (subject in entities or subject.lower() in self.domain_terms) and \
                   (obj in entities or obj.lower() in self.domain_terms):
                    relationships.append((subject, predicate, obj))

        return relationships[:10]  # Top 10

    def _classify_reasoning(self, user_input: str, system_output: str) -> ReasoningType:
        """Classify the type of reasoning in this memory."""
        text = (user_input or '') + ' ' + (system_output or '')
        text_lower = text.lower()

        # Question indicators
        if '?' in user_input or any(w in text_lower for w in ['what', 'how', 'why', 'when', 'where']):
            if system_output and len(system_output) > 50:
                return ReasoningType.ANSWER
            return ReasoningType.QUESTION

        # Decision indicators
        if any(w in text_lower for w in ['decide', 'choose', 'should', 'recommend', 'prefer']):
            return ReasoningType.DECISION

        # Explanation indicators
        if any(w in text_lower for w in ['because', 'therefore', 'thus', 'explain', 'reason']):
            return ReasoningType.EXPLANATION

        # Comparison indicators
        if any(w in text_lower for w in ['versus', 'vs', 'compared to', 'difference between']):
            return ReasoningType.COMPARISON

        # Procedure indicators
        if any(w in text_lower for w in ['step', 'first', 'then', 'next', 'finally', 'how to']):
            return ReasoningType.PROCEDURE

        # Fact indicators (definitive statements)
        if any(w in text_lower for w in ['is', 'are', 'was', 'were', 'consists of']):
            return ReasoningType.FACT

        # Default
        return ReasoningType.OBSERVATION

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics/themes from text."""
        topics = set()

        # Topic clusters (groups of related terms)
        topic_clusters = {
            'memory': ['memory', 'recall', 'storage', 'retrieval', 'shard'],
            'policy': ['policy', 'decision', 'thompson', 'sampling', 'bandit'],
            'embedding': ['embedding', 'vector', 'semantic', 'matryoshka'],
            'graph': ['graph', 'knowledge', 'neo4j', 'node', 'edge', 'relationship'],
            'orchestration': ['orchestrator', 'pipeline', 'workflow', 'coordination'],
            'spinner': ['spinner', 'spinning', 'wheel', 'input', 'adapter'],
            'neural': ['neural', 'network', 'model', 'training', 'learning']
        }

        text_lower = text.lower()

        for topic, terms in topic_clusters.items():
            if any(term in text_lower for term in terms):
                topics.add(topic)

        return sorted(list(topics))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords (domain terms, repeated words)."""
        keywords = set()

        # Domain terms present
        text_lower = text.lower()
        for term in self.domain_terms:
            if term in text_lower:
                keywords.add(term)

        # Frequent words (more than once, longer than 4 chars)
        words = re.findall(r'\b[a-z]{5,}\b', text_lower)
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count > 1:
                keywords.add(word)

        return sorted(list(keywords))[:15]  # Top 15

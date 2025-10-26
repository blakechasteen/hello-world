# -*- coding: utf-8 -*-
"""
Neo4j Enricher
==============

Graph-based context enrichment using Neo4j knowledge graph.

Provides:
- Entity lookup in knowledge graph
- Related entity discovery
- Relationship traversal
- Graph-based context expansion

Requirements:
    pip install neo4j

Usage:
    from HoloLoom.spinningWheel.enrichment import Neo4jEnricher

    enricher = Neo4jEnricher(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    result = await enricher.enrich("Inspected hive Jodi today")
    # Returns related entities from knowledge graph

Created: 2025-10-26
"""

from typing import Dict, List, Any, Optional
from .base import BaseEnricher, EnrichmentResult


class Neo4jEnricher(BaseEnricher):
    """
    Enrich text with knowledge graph context from Neo4j.

    Capabilities:
    - Lookup entities mentioned in text
    - Find related entities via graph traversal
    - Extract relationship context
    - Provide domain-specific context from KG

    Example:
        >>> enricher = Neo4jEnricher(uri="bolt://localhost:7687")
        >>> result = await enricher.enrich("Hive Jodi showing signs of varroa")
        >>> result.data
        {
            'entities_found': ['Hive Jodi', 'varroa'],
            'related_entities': ['treatment_oxalic_acid', 'inspection_protocol'],
            'relationships': [
                {'from': 'Hive Jodi', 'to': 'location_east', 'type': 'LOCATED_AT'},
                {'from': 'varroa', 'to': 'treatment_oxalic_acid', 'type': 'TREATED_WITH'}
            ],
            'context': 'Historical data shows Hive Jodi last treated 30 days ago'
        }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.uri = self.config.get('uri', 'bolt://localhost:7687')
        self.user = self.config.get('user', 'neo4j')
        self.password = self.config.get('password')
        self.database = self.config.get('database', 'neo4j')

        # Connection
        self.driver = None
        self._init_connection()

    def _init_connection(self):
        """Initialize Neo4j connection."""
        if not self.password:
            # No password provided - run in mock mode
            return

        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").single()

        except ImportError:
            print("Warning: neo4j package not installed. Install with: pip install neo4j")
            self.driver = None
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j at {self.uri}: {e}")
            self.driver = None

    async def enrich(self, text: str) -> EnrichmentResult:
        """
        Enrich text with knowledge graph context.

        Args:
            text: Input text to enrich

        Returns:
            EnrichmentResult with graph-based context
        """
        if not self.driver:
            # Mock mode - return empty enrichment
            return EnrichmentResult(
                enricher_type='neo4j',
                data={
                    'entities_found': [],
                    'related_entities': [],
                    'relationships': [],
                    'mock_mode': True
                },
                confidence=0.0,
                metadata={'status': 'no_connection'}
            )

        # Extract entities from text (simple heuristic)
        entities = self._extract_entity_candidates(text)

        # Query Neo4j for each entity
        entities_found = []
        related_entities = []
        relationships = []

        with self.driver.session(database=self.database) as session:
            for entity_name in entities:
                # Look up entity in graph
                result = session.run(
                    """
                    MATCH (e:Entity {name: $name})
                    RETURN e
                    """,
                    name=entity_name
                )

                if result.peek():
                    entities_found.append(entity_name)

                    # Find related entities (1-hop traversal)
                    related_result = session.run(
                        """
                        MATCH (e:Entity {name: $name})-[r]-(related:Entity)
                        RETURN related.name as name, type(r) as rel_type
                        LIMIT 5
                        """,
                        name=entity_name
                    )

                    for record in related_result:
                        related_name = record['name']
                        rel_type = record['rel_type']

                        if related_name not in related_entities:
                            related_entities.append(related_name)

                        relationships.append({
                            'from': entity_name,
                            'to': related_name,
                            'type': rel_type
                        })

        # Build context summary
        context = self._build_context_summary(entities_found, related_entities, relationships)

        return EnrichmentResult(
            enricher_type='neo4j',
            data={
                'entities_found': entities_found,
                'related_entities': related_entities,
                'relationships': relationships,
                'context': context
            },
            confidence=1.0 if entities_found else 0.5,
            metadata={
                'uri': self.uri,
                'entity_candidates': entities,
                'match_rate': len(entities_found) / len(entities) if entities else 0.0
            }
        )

    def _extract_entity_candidates(self, text: str) -> List[str]:
        """
        Extract potential entity names from text.

        Uses simple heuristics:
        - Capitalized words
        - Multi-word proper nouns
        - Domain-specific patterns (Hive XYZ, etc.)
        """
        import re

        candidates = []

        # Pattern 1: "Hive XYZ" style entities
        hive_pattern = r'\b(?:Hive|hive)\s+([A-Z][a-z]+)\b'
        candidates.extend(re.findall(hive_pattern, text))

        # Pattern 2: Capitalized words (potential proper nouns)
        cap_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        # Filter common words
        common_words = {'The', 'This', 'That', 'I', 'We', 'They', 'It', 'A', 'An'}
        candidates.extend([w for w in cap_words if w not in common_words])

        # Pattern 3: Multi-word proper nouns
        multiword_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        candidates.extend(re.findall(multiword_pattern, text))

        # Deduplicate
        return list(set(candidates))

    def _build_context_summary(
        self,
        entities_found: List[str],
        related_entities: List[str],
        relationships: List[Dict[str, str]]
    ) -> str:
        """Build human-readable context summary."""
        if not entities_found:
            return "No entities found in knowledge graph"

        summary = f"Found {len(entities_found)} entities in knowledge graph: {', '.join(entities_found)}. "

        if related_entities:
            summary += f"Related entities: {', '.join(related_entities[:3])}. "

        if relationships:
            rel_types = list(set(r['type'] for r in relationships))
            summary += f"Relationships: {', '.join(rel_types)}."

        return summary

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Convenience function
async def enrich_with_neo4j(
    text: str,
    uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: Optional[str] = None
) -> EnrichmentResult:
    """
    Quick function to enrich text with Neo4j context.

    Args:
        text: Input text
        uri: Neo4j connection URI
        user: Neo4j username
        password: Neo4j password

    Returns:
        EnrichmentResult with graph context

    Example:
        result = await enrich_with_neo4j(
            "Hive Jodi needs treatment",
            password="mypassword"
        )
    """
    enricher = Neo4jEnricher({
        'uri': uri,
        'user': user,
        'password': password
    })

    result = await enricher.enrich(text)
    enricher.close()

    return result

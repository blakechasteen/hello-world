"""
Analogical Reasoning Engine

Layer 3 of Cognitive Architecture - Transfer knowledge across domains.

Analogy: Learning by structural similarity.
- Structure mapping: Atom ↔ Solar system
- Knowledge transfer: Heart ↔ Pump
- Case-based reasoning: Solve new problem using similar past problem

Core Algorithm:
1. Find structural mapping between source and target domains
2. Transfer knowledge via the mapping
3. Adapt transferred knowledge to target context

Research Alignment:
- Gentner (1983): "Structure-Mapping Theory"
- Hofstadter & Mitchell (1994): "Copycat" program
- Holyoak & Thagard (1989): "Analogical mapping by constraint satisfaction"
- Forbus et al. (2011): "Structure-Mapping Engine (SME)"

Public API:
    AnalogicalMapping: Correspondence between domains
    AnalogicalReasoner: Main reasoning engine
    StructureMapper: Finds domain alignments
    KnowledgeTransferer: Transfers facts/rules via mapping
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Callable
from collections import defaultdict
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Entity:
    """
    Entity in a domain.

    Represents objects, concepts, or components.

    Attributes:
        name: Entity identifier
        properties: Attributes (e.g., {"mass": "large", "color": "red"})
        entity_type: Category (optional)
    """
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    entity_type: Optional[str] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.name == other.name

    def __repr__(self) -> str:
        props = ', '.join(f"{k}={v}" for k, v in list(self.properties.items())[:3])
        return f"Entity({self.name}, {props})"


@dataclass
class Relation:
    """
    Relation between entities.

    Represents relationships, functions, or interactions.

    Attributes:
        relation_type: Type of relation (e.g., "orbits", "attracts")
        entities: Participating entities (ordered)
        properties: Relation properties (optional)
    """
    relation_type: str
    entities: Tuple[Entity, ...]
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.relation_type, self.entities))

    def __eq__(self, other):
        return (isinstance(other, Relation) and
                self.relation_type == other.relation_type and
                self.entities == other.entities)

    def __repr__(self) -> str:
        entity_names = ', '.join(e.name for e in self.entities)
        return f"{self.relation_type}({entity_names})"


@dataclass
class Domain:
    """
    Structured domain representation.

    Contains entities and their relationships.

    Attributes:
        name: Domain name (e.g., "solar_system", "atom")
        entities: Set of domain entities
        relations: Set of domain relations
        facts: Additional factual knowledge
    """
    name: str
    entities: Set[Entity] = field(default_factory=set)
    relations: Set[Relation] = field(default_factory=set)
    facts: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: Entity):
        """Add entity to domain."""
        self.entities.add(entity)

    def add_relation(self, relation: Relation):
        """Add relation to domain."""
        self.relations.add(relation)

    def get_entity(self, name: str) -> Optional[Entity]:
        """Find entity by name."""
        for e in self.entities:
            if e.name == name:
                return e
        return None

    def get_relations_for(self, entity: Entity) -> List[Relation]:
        """Find all relations involving entity."""
        return [r for r in self.relations if entity in r.entities]

    def __repr__(self) -> str:
        return f"Domain({self.name}, {len(self.entities)} entities, {len(self.relations)} relations)"


@dataclass
class AnalogicalMapping:
    """
    Mapping between source and target domains.

    Represents correspondence: source entity/relation → target entity/relation.

    Attributes:
        source_domain: Source domain name
        target_domain: Target domain name
        entity_mappings: Entity correspondences {source → target}
        relation_mappings: Relation correspondences {source → target}
        score: Mapping quality [0, 1]
        justification: Why this mapping is valid
    """
    source_domain: str
    target_domain: str
    entity_mappings: Dict[Entity, Entity] = field(default_factory=dict)
    relation_mappings: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    justification: List[str] = field(default_factory=list)

    def map_entity(self, source: Entity) -> Optional[Entity]:
        """Map source entity to target."""
        return self.entity_mappings.get(source)

    def map_relation(self, source_relation_type: str) -> Optional[str]:
        """Map source relation type to target."""
        return self.relation_mappings.get(source_relation_type)

    def reverse(self) -> 'AnalogicalMapping':
        """Create reverse mapping (target → source)."""
        return AnalogicalMapping(
            source_domain=self.target_domain,
            target_domain=self.source_domain,
            entity_mappings={v: k for k, v in self.entity_mappings.items()},
            relation_mappings={v: k for k, v in self.relation_mappings.items()},
            score=self.score,
            justification=self.justification
        )

    def __repr__(self) -> str:
        return (f"AnalogicalMapping({self.source_domain} → {self.target_domain}, "
                f"{len(self.entity_mappings)} entities, "
                f"score={self.score:.3f})")


# ============================================================================
# Structure Mapping
# ============================================================================

class StructureMapper:
    """
    Finds structural correspondences between domains.

    Implements simplified version of Gentner's Structure-Mapping Engine (SME).

    Algorithm:
    1. Generate candidate entity mappings (match properties)
    2. Check relation compatibility (structural consistency)
    3. Score mappings (structural + semantic similarity)
    4. Select best consistent mapping
    """

    def __init__(self,
                 semantic_similarity: Optional[Callable[[Entity, Entity], float]] = None,
                 relation_similarity: Optional[Callable[[str, str], float]] = None):
        """
        Initialize structure mapper.

        Args:
            semantic_similarity: Function computing entity similarity
            relation_similarity: Function computing relation similarity
        """
        self.semantic_similarity = semantic_similarity or self._default_entity_similarity
        self.relation_similarity = relation_similarity or self._default_relation_similarity

    def find_mapping(self,
                    source: Domain,
                    target: Domain,
                    max_mappings: int = 1) -> List[AnalogicalMapping]:
        """
        Find structural mappings between domains.

        Main entry point for structure mapping.

        Args:
            source: Source domain
            target: Target domain
            max_mappings: Maximum mappings to return

        Returns:
            Ranked list of mappings (best first)
        """
        logger.info(f"Finding analogical mapping: {source.name} → {target.name}")

        # 1. Generate candidate entity mappings
        candidate_mappings = self._generate_candidate_entity_mappings(source, target)

        logger.info(f"Generated {len(candidate_mappings)} candidate entity mappings")

        # 2. Find relation-consistent mappings
        consistent_mappings = []
        for entity_map in candidate_mappings:
            relation_map = self._find_relation_mapping(source, target, entity_map)
            if relation_map:
                mapping = AnalogicalMapping(
                    source_domain=source.name,
                    target_domain=target.name,
                    entity_mappings=entity_map,
                    relation_mappings=relation_map
                )
                consistent_mappings.append(mapping)

        logger.info(f"Found {len(consistent_mappings)} relation-consistent mappings")

        # 3. Score and rank mappings
        for mapping in consistent_mappings:
            mapping.score = self._score_mapping(mapping, source, target)
            mapping.justification = self._generate_justification(mapping, source, target)

        consistent_mappings.sort(key=lambda m: m.score, reverse=True)

        logger.info(f"Returning top {min(max_mappings, len(consistent_mappings))} mappings")

        return consistent_mappings[:max_mappings]

    def _generate_candidate_entity_mappings(self,
                                           source: Domain,
                                           target: Domain) -> List[Dict[Entity, Entity]]:
        """
        Generate candidate entity correspondences.

        Uses greedy matching based on semantic similarity.
        """
        source_entities = list(source.entities)
        target_entities = list(target.entities)

        # Compute similarity matrix
        similarities = {}
        for s_ent in source_entities:
            for t_ent in target_entities:
                sim = self.semantic_similarity(s_ent, t_ent)
                if sim > 0.3:  # Threshold
                    similarities[(s_ent, t_ent)] = sim

        # Greedy matching (simple version)
        # More sophisticated: Hungarian algorithm
        mappings = []

        # One-to-one mapping (bijection)
        used_targets = set()
        entity_map = {}

        # Sort by similarity
        sorted_pairs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        for (s_ent, t_ent), sim in sorted_pairs:
            if s_ent not in entity_map and t_ent not in used_targets:
                entity_map[s_ent] = t_ent
                used_targets.add(t_ent)

        if entity_map:
            mappings.append(entity_map)

        return mappings if mappings else [{}]

    def _find_relation_mapping(self,
                              source: Domain,
                              target: Domain,
                              entity_mapping: Dict[Entity, Entity]) -> Optional[Dict[str, str]]:
        """
        Find relation mapping consistent with entity mapping.

        Checks structural consistency: if source relation R(A, B),
        then target should have R'(A', B') where A → A', B → B'.
        """
        relation_map = {}

        for source_rel in source.relations:
            # Check if all entities in relation are mapped
            mapped_entities = []
            for ent in source_rel.entities:
                if ent in entity_mapping:
                    mapped_entities.append(entity_mapping[ent])
                else:
                    # Entity not mapped → skip this relation
                    break

            if len(mapped_entities) == len(source_rel.entities):
                # Find target relation with same structure
                target_rel = self._find_corresponding_relation(
                    target,
                    tuple(mapped_entities),
                    source_rel.relation_type
                )

                if target_rel:
                    relation_map[source_rel.relation_type] = target_rel.relation_type

        return relation_map if relation_map else None

    def _find_corresponding_relation(self,
                                    target: Domain,
                                    entities: Tuple[Entity, ...],
                                    source_relation_type: str) -> Optional[Relation]:
        """Find target relation matching entity pattern."""
        for target_rel in target.relations:
            if target_rel.entities == entities:
                # Check if relation types are compatible
                if self.relation_similarity(source_relation_type, target_rel.relation_type) > 0.5:
                    return target_rel
        return None

    def _score_mapping(self,
                      mapping: AnalogicalMapping,
                      source: Domain,
                      target: Domain) -> float:
        """
        Score mapping quality.

        Combines:
        - Structural consistency (relation preservation)
        - Semantic similarity (entity/relation matching)
        - Coverage (how much of source is mapped)

        Returns:
            Score in [0, 1]
        """
        # Structural consistency
        n_relations_preserved = len(mapping.relation_mappings)
        total_relations = len(source.relations)
        structural_score = n_relations_preserved / max(total_relations, 1)

        # Coverage
        n_entities_mapped = len(mapping.entity_mappings)
        total_entities = len(source.entities)
        coverage_score = n_entities_mapped / max(total_entities, 1)

        # Semantic similarity (average)
        semantic_scores = []
        for s_ent, t_ent in mapping.entity_mappings.items():
            sim = self.semantic_similarity(s_ent, t_ent)
            semantic_scores.append(sim)

        semantic_score = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0

        # Combine (weighted average)
        score = 0.4 * structural_score + 0.3 * coverage_score + 0.3 * semantic_score

        return score

    def _generate_justification(self,
                               mapping: AnalogicalMapping,
                               source: Domain,
                               target: Domain) -> List[str]:
        """Generate human-readable justification for mapping."""
        justifications = []

        # Entity mappings
        for s_ent, t_ent in list(mapping.entity_mappings.items())[:3]:
            justifications.append(f"{s_ent.name} ↔ {t_ent.name}")

        # Relation preservation
        for s_rel, t_rel in list(mapping.relation_mappings.items())[:3]:
            justifications.append(f"{s_rel}() ↔ {t_rel}()")

        return justifications

    # Default similarity functions

    def _default_entity_similarity(self, e1: Entity, e2: Entity) -> float:
        """Default entity similarity based on shared properties."""
        if not e1.properties or not e2.properties:
            return 0.3  # Neutral

        shared = 0
        total = 0

        all_keys = set(e1.properties.keys()) | set(e2.properties.keys())

        for key in all_keys:
            total += 1
            if key in e1.properties and key in e2.properties:
                if e1.properties[key] == e2.properties[key]:
                    shared += 1

        return shared / max(total, 1)

    def _default_relation_similarity(self, r1: str, r2: str) -> float:
        """Default relation similarity (string matching)."""
        if r1 == r2:
            return 1.0

        # Simple substring matching
        r1_lower = r1.lower()
        r2_lower = r2.lower()

        if r1_lower in r2_lower or r2_lower in r1_lower:
            return 0.7

        return 0.3


# ============================================================================
# Knowledge Transfer
# ============================================================================

class KnowledgeTransferer:
    """
    Transfers knowledge from source to target via mapping.

    Given:
    - Analogical mapping
    - Source domain knowledge

    Infers:
    - Corresponding target domain knowledge
    """

    def transfer_fact(self,
                     source_fact: Dict[str, Any],
                     mapping: AnalogicalMapping) -> Optional[Dict[str, Any]]:
        """
        Transfer single fact via mapping.

        Args:
            source_fact: Fact in source domain
            mapping: Domain correspondence

        Returns:
            Transferred fact in target domain
        """
        target_fact = {}

        for key, value in source_fact.items():
            # If value is entity, map it
            if isinstance(value, Entity):
                mapped = mapping.map_entity(value)
                if mapped:
                    target_fact[key] = mapped
                else:
                    return None  # Cannot transfer (unmapped entity)
            else:
                # Keep non-entity values
                target_fact[key] = value

        return target_fact if target_fact else None

    def transfer_relation(self,
                         source_relation: Relation,
                         mapping: AnalogicalMapping) -> Optional[Relation]:
        """
        Transfer relation via mapping.

        Args:
            source_relation: Relation in source domain
            mapping: Domain correspondence

        Returns:
            Transferred relation in target domain
        """
        # Map entities
        mapped_entities = []
        for ent in source_relation.entities:
            mapped = mapping.map_entity(ent)
            if mapped:
                mapped_entities.append(mapped)
            else:
                return None  # Cannot transfer

        # Map relation type
        mapped_rel_type = mapping.map_relation(source_relation.relation_type)
        if not mapped_rel_type:
            return None

        return Relation(
            relation_type=mapped_rel_type,
            entities=tuple(mapped_entities),
            properties=source_relation.properties.copy()
        )

    def transfer_all(self,
                    source_domain: Domain,
                    mapping: AnalogicalMapping) -> Domain:
        """
        Transfer entire domain via mapping.

        Args:
            source_domain: Source domain
            mapping: Correspondence

        Returns:
            Target domain with transferred knowledge
        """
        target_domain = Domain(name=mapping.target_domain)

        # Transfer entities
        for source_ent in source_domain.entities:
            target_ent = mapping.map_entity(source_ent)
            if target_ent:
                target_domain.add_entity(target_ent)

        # Transfer relations
        for source_rel in source_domain.relations:
            target_rel = self.transfer_relation(source_rel, mapping)
            if target_rel:
                target_domain.add_relation(target_rel)

        return target_domain


# ============================================================================
# Case-Based Reasoning
# ============================================================================

@dataclass
class Case:
    """
    Past problem-solution pair.

    Attributes:
        problem: Problem description (as domain)
        solution: Solution (facts, actions, etc.)
        outcome: How well solution worked
        context: Additional context
    """
    problem: Domain
    solution: Dict[str, Any]
    outcome: float = 1.0  # Success measure [0, 1]
    context: Dict[str, Any] = field(default_factory=dict)


class CaseLibrary:
    """
    Library of past cases for case-based reasoning.

    Stores problem-solution pairs for reuse.
    """

    def __init__(self):
        self.cases: List[Case] = []

    def add_case(self, case: Case):
        """Add case to library."""
        self.cases.append(case)

    def find_similar(self,
                    problem: Domain,
                    mapper: StructureMapper,
                    max_cases: int = 5) -> List[Tuple[Case, AnalogicalMapping]]:
        """
        Find similar past cases.

        Args:
            problem: Current problem to solve
            mapper: Structure mapper for similarity
            max_cases: Maximum cases to return

        Returns:
            List of (case, mapping) pairs sorted by similarity
        """
        similarities = []

        for case in self.cases:
            mappings = mapper.find_mapping(case.problem, problem, max_mappings=1)
            if mappings:
                mapping = mappings[0]
                similarity = mapping.score * case.outcome  # Weight by success
                similarities.append((case, mapping, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[2], reverse=True)

        return [(case, mapping) for case, mapping, _ in similarities[:max_cases]]


# ============================================================================
# Analogical Reasoner
# ============================================================================

class AnalogicalReasoner:
    """
    Main analogical reasoning engine.

    Combines structure mapping, knowledge transfer, and case-based reasoning.

    Usage:
        reasoner = AnalogicalReasoner()
        mapping = reasoner.find_analogy(source, target)
        transferred = reasoner.transfer_knowledge(source, mapping)
        solution = reasoner.solve_by_analogy(problem, case_library)
    """

    def __init__(self):
        self.mapper = StructureMapper()
        self.transferer = KnowledgeTransferer()
        self.case_library = CaseLibrary()

    def find_analogy(self,
                    source: Domain,
                    target: Domain) -> Optional[AnalogicalMapping]:
        """
        Find best analogical mapping between domains.

        Args:
            source: Familiar domain
            target: New domain

        Returns:
            Best mapping or None
        """
        mappings = self.mapper.find_mapping(source, target, max_mappings=1)
        return mappings[0] if mappings else None

    def transfer_knowledge(self,
                          source: Domain,
                          mapping: AnalogicalMapping) -> Domain:
        """
        Transfer knowledge from source to target via mapping.

        Args:
            source: Source domain with knowledge
            mapping: Domain correspondence

        Returns:
            Target domain with transferred knowledge
        """
        return self.transferer.transfer_all(source, mapping)

    def solve_by_analogy(self,
                        problem: Domain,
                        max_adaptations: int = 3) -> Optional[Dict[str, Any]]:
        """
        Solve new problem using similar past cases.

        Case-based reasoning:
        1. Retrieve similar cases from library
        2. Find best analogy
        3. Transfer solution
        4. Adapt to current problem

        Args:
            problem: New problem to solve
            max_adaptations: Maximum solutions to try

        Returns:
            Adapted solution or None
        """
        logger.info(f"Solving by analogy: {problem.name}")

        # Find similar cases
        similar_cases = self.case_library.find_similar(
            problem,
            self.mapper,
            max_cases=max_adaptations
        )

        if not similar_cases:
            logger.warning("No similar cases found")
            return None

        # Use best case
        best_case, best_mapping = similar_cases[0]

        logger.info(f"Using case: {best_case.problem.name} (similarity={best_mapping.score:.3f})")

        # Transfer solution
        # (In real system, would adapt solution to differences)
        adapted_solution = best_case.solution.copy()

        return adapted_solution

    def add_case(self, case: Case):
        """Add case to library for future reuse."""
        self.case_library.add_case(case)


# ============================================================================
# Helper Functions
# ============================================================================

def create_entity(name: str, **properties) -> Entity:
    """Convenience function to create entity."""
    return Entity(name=name, properties=properties)


def create_relation(relation_type: str, *entities, **properties) -> Relation:
    """Convenience function to create relation."""
    return Relation(
        relation_type=relation_type,
        entities=tuple(entities),
        properties=properties
    )


def create_domain(name: str) -> Domain:
    """Convenience function to create domain."""
    return Domain(name=name)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core classes
    'Entity',
    'Relation',
    'Domain',
    'AnalogicalMapping',
    'StructureMapper',
    'KnowledgeTransferer',
    'Case',
    'CaseLibrary',
    'AnalogicalReasoner',

    # Helper functions
    'create_entity',
    'create_relation',
    'create_domain',
]

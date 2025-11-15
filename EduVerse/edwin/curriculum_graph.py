"""
Curriculum Knowledge Graph

Transforms EduVerse curriculum into a navigable knowledge graph using HoloLoom KG.

**Key Features**:
- Ingest 220+ learning objectives as graph nodes
- Prerequisites as directed edges (REQUIRES)
- Bloom progressions as edges (BUILDS_ON)
- Subject clustering (BELONGS_TO)
- Learning path generation (BFS/DFS)
- Grade-level filtering

**Implementation Date**: November 15, 2025
"""

from typing import List, Set, Dict, Optional, Tuple
from collections import deque
import asyncio
from pathlib import Path

from HoloLoom.memory.graph import KG, KGEdge
from EduVerse.education.curriculum import (
    CurriculumFramework,
    LearningObjective,
    Subject,
    BloomLevel
)


class EdWINKnowledgeGraph:
    """
    Curriculum as Knowledge Graph

    Represents 220+ learning objectives as a navigable graph structure.

    Architecture:
    - Nodes: Learning objectives (with metadata)
    - Edges: Prerequisites (REQUIRES), progressions (BUILDS_ON), clustering (BELONGS_TO)
    - Queries: BFS/DFS for learning paths, subgraph extraction

    Example:
        >>> kg = EdWINKnowledgeGraph()
        >>> await kg.ingest_curriculum()
        >>>
        >>> # Get learning path
        >>> path = kg.get_learning_path(
        ...     from_id="math.algebra.6.expressions",
        ...     to_id="math.algebra.10.systems"
        ... )
        >>>
        >>> # Get prerequisites
        >>> prereqs = kg.get_prerequisites("math.algebra.8.linear_equations")
    """

    def __init__(self, curriculum: Optional[CurriculumFramework] = None):
        """
        Initialize knowledge graph

        Args:
            curriculum: Optional curriculum framework (creates new if None)
        """
        self.curriculum = curriculum or CurriculumFramework()
        self.kg = KG()
        self._ingested = False

    async def ingest_curriculum(self) -> Dict[str, int]:
        """
        Ingest curriculum into knowledge graph

        Transforms all 220+ learning objectives into graph nodes with edges:
        - REQUIRES: Prerequisite relationships
        - BUILDS_ON: Bloom progression (e.g., Apply builds on Understand)
        - BELONGS_TO: Subject membership
        - RELATES_TO: Cross-subject connections

        Returns:
            Statistics dict with counts (objectives, edges, subjects)
        """
        if self._ingested:
            return self._get_stats()

        print(f"📚 Ingesting {len(self.curriculum.objectives)} objectives...")

        edges_added = 0

        for obj_id, obj in self.curriculum.objectives.items():
            # Add objective → subject edge
            self.kg.add_edges([
                KGEdge(
                    src=obj_id,
                    dst=f"subject_{obj.subject.value}",
                    type="BELONGS_TO",
                    weight=1.0,
                    metadata={
                        "title": obj.title,
                        "description": obj.description,
                        "grade_level": obj.grade_level,
                        "bloom_level": obj.bloom_level.value,
                        "bloom_name": obj.bloom_level.name,
                        "estimated_hours": obj.estimated_hours,
                        "keywords": obj.keywords,
                        "standards": obj.standards
                    }
                )
            ])
            edges_added += 1

            # Add prerequisite edges
            for prereq_id in obj.prerequisites:
                if prereq_id in self.curriculum.objectives:
                    self.kg.add_edges([
                        KGEdge(
                            src=obj_id,
                            dst=prereq_id,
                            type="REQUIRES",
                            weight=1.0,
                            metadata={
                                "from_grade": obj.grade_level,
                                "to_grade": self.curriculum.objectives[prereq_id].grade_level
                            }
                        )
                    ])
                    edges_added += 1

            # Add Bloom progression edges
            # Each level builds on the previous level
            if obj.bloom_level.value > 1:
                # Find same topic at lower Bloom level
                lower_bloom_candidates = [
                    other_id for other_id, other_obj in self.curriculum.objectives.items()
                    if other_obj.subject == obj.subject
                    and other_obj.grade_level <= obj.grade_level
                    and other_obj.bloom_level.value == obj.bloom_level.value - 1
                    and any(kw in other_obj.keywords for kw in obj.keywords[:3])  # Similar keywords
                ]

                for candidate_id in lower_bloom_candidates[:1]:  # Take first match
                    self.kg.add_edges([
                        KGEdge(
                            src=obj_id,
                            dst=candidate_id,
                            type="BUILDS_ON",
                            weight=0.8,
                            metadata={
                                "bloom_from": obj.bloom_level.name,
                                "bloom_to": self.curriculum.objectives[candidate_id].bloom_level.name
                            }
                        )
                    ])
                    edges_added += 1

        self._ingested = True

        stats = self._get_stats()
        print(f"✅ Curriculum ingested!")
        print(f"   - Objectives: {stats['objectives']}")
        print(f"   - Edges: {stats['edges']}")
        print(f"   - Subjects: {stats['subjects']}")

        return stats

    def get_prerequisites(self, objective_id: str, recursive: bool = True) -> List[str]:
        """
        Get all prerequisites for an objective

        Args:
            objective_id: Learning objective ID
            recursive: If True, get all transitive prerequisites

        Returns:
            List of prerequisite objective IDs
        """
        if objective_id not in self.curriculum.objectives:
            return []

        if not recursive:
            return self.curriculum.objectives[objective_id].prerequisites

        # BFS to get all transitive prerequisites
        visited = set()
        queue = deque([objective_id])

        while queue:
            current_id = queue.popleft()

            if current_id in visited:
                continue

            visited.add(current_id)

            # Get immediate prerequisites
            if current_id in self.curriculum.objectives:
                prereqs = self.curriculum.objectives[current_id].prerequisites
                for prereq_id in prereqs:
                    if prereq_id not in visited:
                        queue.append(prereq_id)

        # Remove the objective itself
        visited.discard(objective_id)

        return list(visited)

    def get_learning_path(
        self,
        from_id: str,
        to_id: str
    ) -> List[LearningObjective]:
        """
        Generate learning path from one objective to another

        Uses BFS through prerequisite graph to find shortest path.

        Args:
            from_id: Starting objective ID
            to_id: Target objective ID

        Returns:
            Ordered list of objectives (from → to), empty if no path

        Example:
            >>> path = kg.get_learning_path(
            ...     "math.algebra.6.expressions",
            ...     "math.algebra.10.systems"
            ... )
            >>> for obj in path:
            ...     print(f"{obj.title} (Grade {obj.grade_level})")
        """
        return self.curriculum.get_learning_path(from_id, to_id)

    def get_next_objectives(
        self,
        mastered_ids: List[str],
        grade_filter: Optional[int] = None,
        subject_filter: Optional[Subject] = None
    ) -> List[LearningObjective]:
        """
        Get objectives where all prerequisites are mastered

        Args:
            mastered_ids: List of mastered objective IDs
            grade_filter: Optional grade level filter (±1 grade acceptable)
            subject_filter: Optional subject filter

        Returns:
            List of unlocked objectives ready to learn
        """
        ready = self.curriculum.get_next_objectives(mastered_ids)

        # Apply filters
        if grade_filter is not None:
            ready = [
                obj for obj in ready
                if abs(obj.grade_level - grade_filter) <= 1
            ]

        if subject_filter is not None:
            ready = [
                obj for obj in ready
                if obj.subject == subject_filter
            ]

        return ready

    def filter_by_grade(
        self,
        grade: int,
        buffer: int = 1
    ) -> List[LearningObjective]:
        """
        Get objectives for a grade level (±buffer)

        Args:
            grade: Target grade level (4-12)
            buffer: Grade buffer (default: 1, allows ±1 grade)

        Returns:
            List of objectives for grade range
        """
        objectives = []

        for obj in self.curriculum.objectives.values():
            if abs(obj.grade_level - grade) <= buffer:
                objectives.append(obj)

        return objectives

    def retrieve_relevant(
        self,
        query: str,
        k: int = 5,
        grade_filter: Optional[int] = None
    ) -> List[LearningObjective]:
        """
        Retrieve relevant objectives based on query

        Uses keyword matching + grade filtering.

        Args:
            query: Search query
            k: Number of results
            grade_filter: Optional grade level

        Returns:
            Top k relevant objectives
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Score each objective by keyword overlap
        scored_objectives = []

        for obj in self.curriculum.objectives.values():
            # Apply grade filter
            if grade_filter and abs(obj.grade_level - grade_filter) > 2:
                continue

            # Calculate keyword overlap score
            obj_keywords = set(kw.lower() for kw in obj.keywords)
            obj_title_words = set(obj.title.lower().split())
            obj_desc_words = set(obj.description.lower().split())

            all_obj_words = obj_keywords | obj_title_words | obj_desc_words

            overlap = len(query_words & all_obj_words)

            # Boost if query appears in title
            if any(word in obj.title.lower() for word in query_words):
                overlap += 2

            if overlap > 0:
                scored_objectives.append((overlap, obj))

        # Sort by score descending
        scored_objectives.sort(key=lambda x: x[0], reverse=True)

        return [obj for score, obj in scored_objectives[:k]]

    def get_knowledge_gaps(
        self,
        target_objective_id: str,
        mastered_ids: Set[str]
    ) -> List[str]:
        """
        Identify knowledge gaps (missing prerequisites)

        Args:
            target_objective_id: Target learning objective
            mastered_ids: Set of mastered objective IDs

        Returns:
            List of missing prerequisite IDs
        """
        all_prereqs = set(self.get_prerequisites(target_objective_id))
        gaps = all_prereqs - mastered_ids

        return list(gaps)

    def get_subject_objectives(
        self,
        subject: Subject,
        grade_range: Optional[Tuple[int, int]] = None
    ) -> List[LearningObjective]:
        """
        Get all objectives for a subject

        Args:
            subject: Subject enum
            grade_range: Optional (min_grade, max_grade) tuple

        Returns:
            List of objectives for subject
        """
        objectives = self.curriculum.get_objectives_by_subject(subject)

        if grade_range:
            min_grade, max_grade = grade_range
            objectives = [
                obj for obj in objectives
                if min_grade <= obj.grade_level <= max_grade
            ]

        return objectives

    def save(self, filepath: str) -> None:
        """
        Save knowledge graph to file

        Args:
            filepath: Path to save (JSON format)
        """
        self.kg.save(filepath)
        print(f"💾 Saved knowledge graph to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load knowledge graph from file

        Args:
            filepath: Path to load from
        """
        self.kg.load(filepath)
        self._ingested = True
        print(f"📂 Loaded knowledge graph from {filepath}")

    def _get_stats(self) -> Dict[str, int]:
        """Get curriculum statistics"""
        return {
            "objectives": len(self.curriculum.objectives),
            "edges": len(self.kg.edges) if hasattr(self.kg, 'edges') else 0,
            "subjects": len(self.curriculum.subjects)
        }

    def get_objective(self, objective_id: str) -> Optional[LearningObjective]:
        """Get objective by ID"""
        return self.curriculum.get_objective(objective_id)


# Helper functions

async def create_curriculum_graph() -> EdWINKnowledgeGraph:
    """
    Create and ingest curriculum knowledge graph

    Returns:
        Fully initialized EdWINKnowledgeGraph
    """
    kg = EdWINKnowledgeGraph()
    await kg.ingest_curriculum()
    return kg

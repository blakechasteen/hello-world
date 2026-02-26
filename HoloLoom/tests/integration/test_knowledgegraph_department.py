#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Knowledge Graph Department.

Tests:
-----
1. Component Tests (12 tests)
   - GraphConstructor: Text extraction, triple construction, graph merging
   - GraphReasoner: Path finding, subgraph extraction, queries
   - GraphEvolver: Add, update, merge, prune
   - GraphVerifier: Consistency checking, violation detection

2. Department Integration Tests (6 tests)
   - Graph construction task
   - Path finding task
   - Subgraph extraction task
   - Graph evolution task
   - Graph verification task

3. DS-STAR Protocol Tests (2 tests)
   - Verification workflow
   - Refinement workflow

4. Performance Tests (2 tests)
   - Graph construction performance
   - Path finding performance

Total: 22 test scenarios
"""

import pytest
from typing import List
import asyncio

from HoloLoom.apps.departments.knowledgegraph import (
    KnowledgeGraphDepartment,
    GraphConstructor,
    GraphReasoner,
    GraphEvolver,
    GraphVerifier,
    Triple,
    GraphQuery,
    EvolutionOperation
)
from HoloLoom.departments import DepartmentRequest, DepartmentResponse, DepartmentRegistry
from HoloLoom.memory.graph import KG, KGEdge


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_text() -> str:
    """Create sample text for graph construction."""
    return """
    Varroa mites are parasites that attack honey bees.
    Honey bees produce honey in hives.
    Colony collapse affects honey bee populations.
    Varroa mites cause colony collapse.
    Beekeepers monitor bee health.
    """


@pytest.fixture
def sample_kg() -> KG:
    """Create a sample knowledge graph."""
    kg = KG()
    edges = [
        KGEdge("varroa_mites", "honey_bees", "ATTACKS", 1.0),
        KGEdge("honey_bees", "honey", "PRODUCES", 0.9),
        KGEdge("honey_bees", "hive", "LOCATED_IN", 0.95),
        KGEdge("varroa_mites", "colony_collapse", "CAUSES", 0.85),
        KGEdge("colony_collapse", "bee_population", "AFFECTS", 0.9),
        KGEdge("beekeepers", "bee_health", "MONITORS", 0.8)
    ]
    kg.add_edges(edges)
    return kg


# ============================================================================
# Component Tests: GraphConstructor
# ============================================================================

def test_graph_constructor_initialization():
    """Test GraphConstructor initialization."""
    constructor = GraphConstructor(enable_spacy=False)
    assert constructor.enable_spacy == False


def test_graph_constructor_extract_triples(sample_text):
    """Test triple extraction from text."""
    constructor = GraphConstructor(enable_spacy=False)
    triples = constructor.extract_triples(sample_text)

    # Should extract some triples (regex-based)
    assert len(triples) > 0

    # Check triple structure
    for triple in triples:
        assert isinstance(triple, Triple)
        assert isinstance(triple.subject, str)
        assert isinstance(triple.predicate, str)
        assert isinstance(triple.object, str)
        assert 0.0 <= triple.weight <= 1.0


def test_graph_constructor_construct_from_text(sample_text):
    """Test graph construction from text."""
    constructor = GraphConstructor(enable_spacy=False)
    kg = constructor.construct_from_text(sample_text)

    # Should create non-empty graph
    assert len(kg.get_all_entities()) > 0
    assert len(kg.get_all_edges()) > 0


def test_graph_constructor_merge_graphs(sample_kg):
    """Test merging two graphs."""
    constructor = GraphConstructor()

    # Create a second graph
    kg2 = KG()
    kg2.add_edges([
        KGEdge("new_entity", "another_entity", "RELATED_TO", 0.7)
    ])

    # Merge
    merged = constructor.merge_graphs(sample_kg, kg2)

    # Should have entities from both graphs
    all_entities = merged.get_all_entities()
    assert "varroa_mites" in all_entities
    assert "new_entity" in all_entities


# ============================================================================
# Component Tests: GraphReasoner
# ============================================================================

def test_graph_reasoner_initialization():
    """Test GraphReasoner initialization."""
    reasoner = GraphReasoner()
    assert reasoner is not None


def test_graph_reasoner_find_paths(sample_kg):
    """Test path finding between entities."""
    reasoner = GraphReasoner()
    paths = reasoner.find_paths(
        sample_kg,
        start_entity="varroa_mites",
        end_entity="bee_population",
        max_hops=5
    )

    # Should find at least one path
    assert len(paths) > 0

    # Check path structure
    path = paths[0]
    assert path.entities[0] == "varroa_mites"
    assert path.entities[-1] == "bee_population"
    assert len(path.relations) == len(path.entities) - 1
    assert path.hops > 0


def test_graph_reasoner_extract_subgraph(sample_kg):
    """Test subgraph extraction."""
    reasoner = GraphReasoner()
    subgraph = reasoner.extract_subgraph(
        sample_kg,
        seed_entities=["varroa_mites"],
        max_hops=2
    )

    # Should extract subgraph
    assert len(subgraph.entities) > 0
    assert len(subgraph.edges) > 0
    assert "varroa_mites" in subgraph.entities


def test_graph_reasoner_query_graph(sample_kg):
    """Test graph pattern matching."""
    reasoner = GraphReasoner()
    query = GraphQuery(
        pattern=[Triple("*", "CAUSES", "*")],
        max_results=10
    )

    results = reasoner.query_graph(sample_kg, query)

    # Should find CAUSES relations
    assert len(results) > 0
    for result in results:
        assert result["predicate"] == "CAUSES"


def test_graph_reasoner_find_communities(sample_kg):
    """Test community detection."""
    reasoner = GraphReasoner()
    communities = reasoner.find_communities(sample_kg, min_community_size=2)

    # Should find at least one community
    assert len(communities) >= 1

    # Check community structure
    for community in communities:
        assert isinstance(community, set)
        assert len(community) >= 2


# ============================================================================
# Component Tests: GraphEvolver
# ============================================================================

def test_graph_evolver_initialization():
    """Test GraphEvolver initialization."""
    evolver = GraphEvolver(prune_threshold=0.3)
    assert evolver.prune_threshold == 0.3


def test_graph_evolver_add_relation(sample_kg):
    """Test adding relation to graph."""
    evolver = GraphEvolver()
    record = evolver.add_relation(
        sample_kg,
        subject="new_entity",
        predicate="RELATES_TO",
        obj="varroa_mites",
        weight=0.8
    )

    # Should succeed
    assert record.success
    assert record.operation == EvolutionOperation.ADD_RELATION

    # Entity should be in graph
    assert "new_entity" in sample_kg.get_all_entities()


def test_graph_evolver_prune_low_weight(sample_kg):
    """Test pruning low-weight edges."""
    # Add low-weight edge
    sample_kg.add_edges([
        KGEdge("low_weight_1", "low_weight_2", "WEAK_LINK", 0.1)
    ])

    evolver = GraphEvolver(prune_threshold=0.3)
    record = evolver.prune_low_weight_edges(sample_kg)

    # Should remove low-weight edges
    assert record.success
    assert record.details["removed_count"] >= 1


def test_graph_evolver_history():
    """Test evolution history tracking."""
    evolver = GraphEvolver(max_history=5)
    kg = KG()

    # Perform multiple operations
    for i in range(7):
        evolver.add_relation(kg, f"entity_{i}", "RELATES_TO", f"entity_{i+1}")

    # History should be limited
    history = evolver.get_evolution_history(limit=10)
    assert len(history) <= 5


# ============================================================================
# Component Tests: GraphVerifier
# ============================================================================

def test_graph_verifier_initialization():
    """Test GraphVerifier initialization."""
    verifier = GraphVerifier()
    assert verifier is not None


def test_graph_verifier_verify_consistent_graph(sample_kg):
    """Test verification of consistent graph."""
    verifier = GraphVerifier()
    report = verifier.verify(sample_kg)

    # Report should have valid structure
    assert isinstance(report.is_consistent, bool)
    assert 0.0 <= report.consistency_score <= 1.0
    assert report.num_entities > 0
    assert report.num_relations > 0


def test_graph_verifier_detect_duplicates():
    """Test detection of duplicate entities."""
    kg = KG()
    kg.add_edges([
        KGEdge("bee", "hive", "LOCATED_IN", 1.0),
        KGEdge("honey_bee", "hive", "LOCATED_IN", 1.0)  # Possible duplicate
    ])

    verifier = GraphVerifier()
    report = verifier.verify(kg)

    # Should detect potential duplicate
    duplicate_violations = [
        v for v in report.violations
        if "duplicate" in v.description.lower()
    ]
    assert len(duplicate_violations) > 0


# ============================================================================
# Department Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_department_initialization():
    """Test KnowledgeGraphDepartment initialization."""
    async with KnowledgeGraphDepartment() as dept:
        assert dept.name == "knowledge_graph"
        assert isinstance(dept.graph_constructor, GraphConstructor)
        assert isinstance(dept.graph_reasoner, GraphReasoner)
        assert isinstance(dept.graph_evolver, GraphEvolver)
        assert isinstance(dept.graph_verifier, GraphVerifier)


@pytest.mark.asyncio
async def test_department_construct_graph_task(sample_text):
    """Test graph construction task through department."""
    async with KnowledgeGraphDepartment() as dept:
        request = DepartmentRequest(
            task_id="kg_001",
            task_type="construct_graph",
            parameters={"text": sample_text, "merge_with_existing": False}
        )

        response = await dept.execute(request)

        assert response.success
        assert response.department_name == "knowledge_graph"
        assert "num_entities" in response.result
        assert "num_relations" in response.result
        assert response.result["num_entities"] > 0


@pytest.mark.asyncio
async def test_department_find_paths_task():
    """Test path finding task through department."""
    # Create department with pre-populated graph
    kg = KG()
    kg.add_edges([
        KGEdge("A", "B", "RELATES_TO", 1.0),
        KGEdge("B", "C", "RELATES_TO", 1.0),
        KGEdge("C", "D", "RELATES_TO", 1.0)
    ])

    async with KnowledgeGraphDepartment(knowledge_graph=kg) as dept:
        request = DepartmentRequest(
            task_id="kg_002",
            task_type="find_paths",
            parameters={
                "start_entity": "A",
                "end_entity": "D",
                "max_hops": 5
            }
        )

        response = await dept.execute(request)

        assert response.success
        assert "num_paths" in response.result
        assert response.result["num_paths"] > 0


@pytest.mark.asyncio
async def test_department_extract_subgraph_task(sample_kg):
    """Test subgraph extraction task through department."""
    async with KnowledgeGraphDepartment(knowledge_graph=sample_kg) as dept:
        request = DepartmentRequest(
            task_id="kg_003",
            task_type="extract_subgraph",
            parameters={
                "seed_entities": ["varroa_mites"],
                "max_hops": 2
            }
        )

        response = await dept.execute(request)

        assert response.success
        assert "num_entities" in response.result
        assert response.result["num_entities"] > 0


@pytest.mark.asyncio
async def test_department_evolve_graph_task(sample_kg):
    """Test graph evolution task through department."""
    async with KnowledgeGraphDepartment(knowledge_graph=sample_kg) as dept:
        request = DepartmentRequest(
            task_id="kg_004",
            task_type="evolve_graph",
            parameters={
                "operation": "add_relation",
                "subject": "new_entity",
                "predicate": "RELATES_TO",
                "object": "varroa_mites",
                "weight": 0.8
            }
        )

        response = await dept.execute(request)

        assert response.success
        assert response.result["success"]


@pytest.mark.asyncio
async def test_department_verify_graph_task(sample_kg):
    """Test graph verification task through department."""
    async with KnowledgeGraphDepartment(knowledge_graph=sample_kg) as dept:
        request = DepartmentRequest(
            task_id="kg_005",
            task_type="verify_graph",
            parameters={}
        )

        response = await dept.execute(request)

        assert response.success
        assert "consistency_score" in response.result
        assert 0.0 <= response.result["consistency_score"] <= 1.0


@pytest.mark.asyncio
async def test_department_invalid_task_type():
    """Test handling invalid task type."""
    async with KnowledgeGraphDepartment() as dept:
        request = DepartmentRequest(
            task_id="kg_999",
            task_type="invalid_task",
            parameters={}
        )

        response = await dept.execute(request)

        assert not response.success
        assert response.error_message is not None


# ============================================================================
# DS-STAR Protocol Tests
# ============================================================================

@pytest.mark.asyncio
async def test_verification_workflow(sample_text):
    """Test verification workflow."""
    async with KnowledgeGraphDepartment() as dept:
        # Construct graph
        request = DepartmentRequest(
            task_id="verify_001",
            task_type="construct_graph",
            parameters={"text": sample_text}
        )

        response = await dept.execute(request)
        assert response.success

        # Verify result
        is_sufficient, confidence, explanation = await dept.verify(request, response.result)

        assert isinstance(is_sufficient, bool)
        assert 0.0 <= confidence <= 1.0
        assert len(explanation) > 0


@pytest.mark.asyncio
async def test_refinement_workflow():
    """Test refinement workflow."""
    kg = KG()
    kg.add_edges([KGEdge("A", "B", "RELATES_TO", 1.0)])

    async with KnowledgeGraphDepartment(knowledge_graph=kg) as dept:
        # Initial request (no path exists)
        request = DepartmentRequest(
            task_id="refine_001",
            task_type="find_paths",
            parameters={
                "start_entity": "A",
                "end_entity": "Z",  # Doesn't exist
                "max_hops": 3
            }
        )

        initial_result = {"num_paths": 0, "confidence": 0.0}

        # Refine result (increases max_hops)
        refined_result = await dept.refine(request, initial_result)

        # Refinement should return a result
        assert refined_result is not None


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_graph_construction_performance(sample_text):
    """Test graph construction performance (<200ms)."""
    import time

    constructor = GraphConstructor(enable_spacy=False)

    start = time.time()
    kg = constructor.construct_from_text(sample_text)
    duration_ms = (time.time() - start) * 1000

    assert duration_ms < 200, f"Construction took {duration_ms:.2f}ms (target: <200ms)"
    assert len(kg.get_all_entities()) > 0


@pytest.mark.asyncio
async def test_path_finding_performance(sample_kg):
    """Test path finding performance (<50ms for small graphs)."""
    import time

    reasoner = GraphReasoner()

    start = time.time()
    paths = reasoner.find_paths(
        sample_kg,
        start_entity="varroa_mites",
        end_entity="bee_population",
        max_hops=5
    )
    duration_ms = (time.time() - start) * 1000

    assert duration_ms < 50, f"Path finding took {duration_ms:.2f}ms (target: <50ms)"


# ============================================================================
# Health Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_health_check():
    """Test health monitoring."""
    async with KnowledgeGraphDepartment() as dept:
        health = await dept.health_check()

        assert "status" in health
        assert "total_tasks" in health
        assert "successful_tasks" in health
        assert "success_rate" in health
        assert "graph_stats" in health
        assert "components" in health

        # All components should be operational
        assert health["components"]["graph_constructor"] == "operational"
        assert health["components"]["graph_reasoner"] == "operational"
        assert health["components"]["graph_evolver"] == "operational"
        assert health["components"]["graph_verifier"] == "operational"


# ============================================================================
# Lifecycle Management
# ============================================================================

@pytest.mark.asyncio
async def test_context_manager_lifecycle():
    """Test async context manager lifecycle."""
    dept = KnowledgeGraphDepartment()

    async with dept:
        assert dept.name == "knowledge_graph"
        # Department should be operational within context

    # After exit, department should be cleaned up
    # (No explicit check needed, just verify no exceptions)

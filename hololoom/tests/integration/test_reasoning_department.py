#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Reasoning Department.

Tests the complete integration of advanced reasoning capabilities:
- Multi-hop graph traversal
- Causal inference (direct and indirect)
- Counterfactual analysis
- DS-STAR verification workflow
- Registry integration

Run with:
    pytest HoloLoom/tests/integration/test_reasoning_department.py -v
"""

import pytest
import asyncio
from typing import List

from hololoom.departments import (
    DepartmentRegistry,
    DepartmentRequest
)
from hololoom.apps.departments.reasoning import (
    ReasoningDepartment,
    MultiHopReasoner,
    CausalInferenceEngine,
    CounterfactualAnalyzer
)
from hololoom.memory.graph import KG, KGEdge


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for testing."""
    kg = KG()

    # Beekeeping domain knowledge
    edges = [
        # Pest relationships
        KGEdge("varroa_mites", "honey_bees", "ATTACKS", 0.95),
        KGEdge("varroa_mites", "bee_immunity", "WEAKENS", 0.90),
        KGEdge("bee_immunity", "colony_health", "AFFECTS", 0.85),
        KGEdge("colony_health", "colony_collapse", "LEADS_TO", 0.80),

        # Hive management
        KGEdge("hive_ventilation", "hive_temperature", "REGULATES", 0.90),
        KGEdge("hive_ventilation", "humidity", "CONTROLS", 0.85),
        KGEdge("humidity", "mold_growth", "INFLUENCES", 0.75),
        KGEdge("mold_growth", "colony_health", "IMPACTS", 0.70),

        # Bee behavior
        KGEdge("honey_bees", "waggle_dance", "PERFORMS", 0.95),
        KGEdge("waggle_dance", "food_source_location", "COMMUNICATES", 0.90),
        KGEdge("food_source_location", "foraging_efficiency", "DETERMINES", 0.85),
        KGEdge("foraging_efficiency", "colony_health", "AFFECTS", 0.80),

        # Cross-domain connections
        KGEdge("colony_health", "honey_production", "DETERMINES", 0.90),
        KGEdge("honey_production", "beekeeper_income", "AFFECTS", 0.85),

        # Causal chains
        KGEdge("pesticide_exposure", "bee_navigation", "IMPAIRS", 0.85),
        KGEdge("bee_navigation", "foraging_efficiency", "AFFECTS", 0.80),
    ]

    kg.add_edges(edges)
    return kg


@pytest.fixture
async def reasoning_dept(sample_knowledge_graph):
    """Create Reasoning Department instance."""
    dept = ReasoningDepartment(max_hops=5, min_causal_strength=0.5)

    # Inject knowledge graph for testing
    dept._knowledge_graph = sample_knowledge_graph

    await dept.initialize()
    yield dept
    await dept.close()


@pytest.fixture
async def registry_with_reasoning(reasoning_dept):
    """Create registry with Reasoning Department."""
    async with DepartmentRegistry() as registry:
        await registry.register(reasoning_dept)
        yield registry


# ============================================================================
# Test Multi-Hop Reasoning
# ============================================================================

@pytest.mark.asyncio
async def test_multi_hop_reasoner_direct_path(sample_knowledge_graph):
    """Test finding direct path between entities."""
    reasoner = MultiHopReasoner(max_hops=3)

    paths = reasoner.find_paths(
        sample_knowledge_graph,
        "varroa_mites",
        "colony_collapse",
        max_hops=5
    )

    # Should find path: varroa_mites → ... → colony_collapse
    assert len(paths) > 0
    assert paths[0].total_hops > 0
    assert paths[0].final_answer == "colony_collapse"


@pytest.mark.asyncio
async def test_multi_hop_reasoner_common_connections(sample_knowledge_graph):
    """Test finding common connections between multiple entities."""
    reasoner = MultiHopReasoner(max_hops=3)

    chains = reasoner.find_common_connections(
        sample_knowledge_graph,
        ["varroa_mites", "hive_ventilation"],
        max_hops=4
    )

    # Should find "colony_health" as common connection
    assert len(chains) > 0

    # Check that colony_health is in results
    common_entities = [chain.final_answer for chain in chains]
    assert "colony_health" in common_entities


@pytest.mark.asyncio
async def test_multi_hop_path_caching(sample_knowledge_graph):
    """Test that path finding uses caching."""
    reasoner = MultiHopReasoner(max_hops=3)

    # First call (should cache)
    paths1 = reasoner.find_paths(
        sample_knowledge_graph,
        "varroa_mites",
        "colony_collapse",
        max_hops=5
    )

    # Second call (should use cache)
    paths2 = reasoner.find_paths(
        sample_knowledge_graph,
        "varroa_mites",
        "colony_collapse",
        max_hops=5
    )

    # Results should be identical (from cache)
    assert len(paths1) == len(paths2)
    assert paths1[0].total_hops == paths2[0].total_hops


# ============================================================================
# Test Causal Inference
# ============================================================================

@pytest.mark.asyncio
async def test_causal_inference_direct(sample_knowledge_graph):
    """Test direct causal relationship detection."""
    engine = CausalInferenceEngine(min_strength=0.5)

    causal_rel = engine.infer_causality(
        sample_knowledge_graph,
        "varroa_mites",
        "honey_bees"
    )

    # Should find direct causal link
    assert causal_rel is not None
    assert causal_rel.cause == "varroa_mites"
    assert causal_rel.effect == "honey_bees"
    assert causal_rel.strength > 0.8  # Strong direct link


@pytest.mark.asyncio
async def test_causal_inference_indirect(sample_knowledge_graph):
    """Test indirect causal relationship detection."""
    engine = CausalInferenceEngine(min_strength=0.5)

    causal_rel = engine.infer_causality(
        sample_knowledge_graph,
        "varroa_mites",
        "colony_collapse"
    )

    # Should find indirect causal chain
    assert causal_rel is not None
    assert causal_rel.cause == "varroa_mites"
    assert causal_rel.effect == "colony_collapse"
    assert causal_rel.strength >= 0.5  # Indirect = lower strength


@pytest.mark.asyncio
async def test_causal_inference_find_all(sample_knowledge_graph):
    """Test finding all causal relations for an entity."""
    engine = CausalInferenceEngine(min_strength=0.5)

    relations = engine.find_all_causal_relations(
        sample_knowledge_graph,
        "colony_health",
        max_depth=2
    )

    # colony_health should have both causes and effects
    assert len(relations["causes"]) > 0  # What colony_health causes
    assert len(relations["effects"]) > 0  # What causes colony_health


# ============================================================================
# Test Counterfactual Analysis
# ============================================================================

@pytest.mark.asyncio
async def test_counterfactual_remove(sample_knowledge_graph):
    """Test counterfactual analysis with entity removal."""
    engine = CausalInferenceEngine(min_strength=0.5)
    analyzer = CounterfactualAnalyzer()

    scenario = analyzer.analyze_counterfactual(
        sample_knowledge_graph,
        "remove",
        "varroa_mites",
        engine
    )

    # Should predict effects on downstream entities
    assert "varroa_mites" in scenario.original_condition
    assert "remove" in scenario.modified_condition.lower()
    assert len(scenario.affected_entities) > 0
    assert scenario.confidence > 0.5


@pytest.mark.asyncio
async def test_counterfactual_increase(sample_knowledge_graph):
    """Test counterfactual analysis with increase modification."""
    engine = CausalInferenceEngine(min_strength=0.5)
    analyzer = CounterfactualAnalyzer()

    scenario = analyzer.analyze_counterfactual(
        sample_knowledge_graph,
        "increase",
        "hive_ventilation",
        engine
    )

    # Should predict amplified effects
    assert "increase" in scenario.modified_condition.lower()
    assert len(scenario.affected_entities) > 0
    assert len(scenario.reasoning) > 0


@pytest.mark.asyncio
async def test_counterfactual_comparison(sample_knowledge_graph):
    """Test comparing multiple counterfactual scenarios."""
    engine = CausalInferenceEngine(min_strength=0.5)
    analyzer = CounterfactualAnalyzer()

    # Create multiple scenarios
    scenarios = [
        analyzer.analyze_counterfactual(sample_knowledge_graph, "remove", "varroa_mites", engine),
        analyzer.analyze_counterfactual(sample_knowledge_graph, "increase", "hive_ventilation", engine),
        analyzer.analyze_counterfactual(sample_knowledge_graph, "decrease", "pesticide_exposure", engine)
    ]

    # Compare scenarios
    comparison = analyzer.compare_scenarios(scenarios)

    assert comparison["best_scenario"] is not None
    assert comparison["worst_scenario"] is not None
    assert len(comparison["summary"]) == 3


# ============================================================================
# Test Reasoning Department Integration
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_department_initialization():
    """Test Reasoning Department initializes correctly."""
    dept = ReasoningDepartment()

    assert dept.name == "reasoning"
    assert dept.domain == "advanced"
    assert dept.version == "1.0.0"
    assert "multi_hop_query" in dept.supported_tasks
    assert "causal_inference" in dept.supported_tasks
    assert "counterfactual" in dept.supported_tasks

    await dept.initialize()
    assert dept._initialized == True

    await dept.close()


@pytest.mark.asyncio
async def test_multi_hop_query_task(reasoning_dept):
    """Test multi-hop query task execution."""
    request = DepartmentRequest(
        task_id="reason_001",
        task_type="multi_hop_query",
        parameters={
            "start_entity": "varroa_mites",
            "end_entity": "colony_collapse",
            "max_hops": 5
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify response
    assert response.task_id == "reason_001"
    assert response.result is not None
    assert "paths_found" in response.result
    assert response.result["paths_found"] > 0
    assert "reasoning_chains" in response.result
    assert response.confidence.score > 0.6


@pytest.mark.asyncio
async def test_causal_inference_task(reasoning_dept):
    """Test causal inference task execution."""
    request = DepartmentRequest(
        task_id="causal_001",
        task_type="causal_inference",
        parameters={
            "entity_a": "varroa_mites",
            "entity_b": "colony_collapse"
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify response
    assert response.result is not None
    assert "causal_relation" in response.result

    if response.result["causal_relation"]:
        assert "cause" in response.result["causal_relation"]
        assert "effect" in response.result["causal_relation"]
        assert "strength" in response.result["causal_relation"]


@pytest.mark.asyncio
async def test_causal_inference_all_relations(reasoning_dept):
    """Test finding all causal relations for an entity."""
    request = DepartmentRequest(
        task_id="causal_002",
        task_type="causal_inference",
        parameters={
            "entity_a": "colony_health"
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify response
    assert response.result is not None
    assert "causes" in response.result
    assert "effects" in response.result
    assert response.result["causes_count"] >= 0
    assert response.result["effects_count"] >= 0


@pytest.mark.asyncio
async def test_counterfactual_task(reasoning_dept):
    """Test counterfactual analysis task."""
    request = DepartmentRequest(
        task_id="counter_001",
        task_type="counterfactual",
        parameters={
            "modification": "remove",
            "entity": "varroa_mites"
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify response
    assert response.result is not None
    assert "predicted_outcome" in response.result
    assert "affected_entities" in response.result
    assert "reasoning" in response.result
    assert response.confidence.score > 0.5


@pytest.mark.asyncio
async def test_explain_reasoning_task(reasoning_dept):
    """Test reasoning explanation task."""
    request = DepartmentRequest(
        task_id="explain_001",
        task_type="explain_reasoning",
        parameters={
            "conclusion": "Colony collapse is caused by multiple factors",
            "entity": "colony_health"
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify response
    assert response.result is not None
    assert "explanations" in response.result
    assert "conclusion" in response.result


# ============================================================================
# Test DS-STAR Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_verify_sufficient(reasoning_dept):
    """Test verification of sufficient reasoning."""
    # Execute reasoning
    request = DepartmentRequest(
        task_id="verify_001",
        task_type="multi_hop_query",
        parameters={
            "start_entity": "varroa_mites",
            "end_entity": "colony_collapse",
            "max_hops": 5
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify
    verification = await reasoning_dept.verify(response)

    # Should be sufficient if paths found
    if response.result.get("paths_found", 0) > 0:
        assert verification.sufficient == True
        assert verification.confidence_valid == True
        assert verification.reasoning_sound == True


@pytest.mark.asyncio
async def test_reasoning_verify_insufficient(reasoning_dept):
    """Test verification of insufficient reasoning."""
    # Create low-confidence response
    request = DepartmentRequest(
        task_id="verify_002",
        task_type="multi_hop_query",
        parameters={
            "start_entity": "nonexistent_entity_1",
            "end_entity": "nonexistent_entity_2",
            "max_hops": 5
        }
    )

    response = await reasoning_dept.execute(request)

    # Verify (should be insufficient)
    verification = await reasoning_dept.verify(response)

    # Low confidence = insufficient
    assert verification.sufficient == False or response.confidence.score < 0.7


@pytest.mark.asyncio
async def test_reasoning_refine(reasoning_dept):
    """Test refinement of reasoning."""
    # Execute with limited hops
    request = DepartmentRequest(
        task_id="refine_001",
        task_type="multi_hop_query",
        parameters={
            "start_entity": "varroa_mites",
            "end_entity": "beekeeper_income",
            "max_hops": 2  # Deliberately limited
        }
    )

    response = await reasoning_dept.execute(request)
    initial_paths = response.result.get("paths_found", 0)

    # Verify
    verification = await reasoning_dept.verify(response)

    # Refine if needed
    if not verification.sufficient:
        refined = await reasoning_dept.refine(request, response, verification)

        # Refined should have more hops allowed
        assert refined.result.get("paths_found", 0) >= initial_paths


# ============================================================================
# Test Registry Integration
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_registry_registration(registry_with_reasoning):
    """Test Reasoning Department registration."""
    registry = registry_with_reasoning

    # Check registration
    dept = registry.get_department("reasoning")
    assert dept is not None
    assert dept.name == "reasoning"

    # Check discovery by domain
    depts_by_domain = registry.find_by_domain("advanced")
    assert len(depts_by_domain) > 0
    assert any(d.name == "reasoning" for d in depts_by_domain)


@pytest.mark.asyncio
async def test_reasoning_registry_routing(registry_with_reasoning):
    """Test routing requests through registry."""
    registry = registry_with_reasoning

    # Route multi-hop query
    request = DepartmentRequest(
        task_id="route_001",
        task_type="multi_hop_query",
        parameters={
            "start_entity": "varroa_mites",
            "end_entity": "colony_health",
            "max_hops": 3
        }
    )

    response = await registry.route_request(request)

    # Should get valid response
    assert response is not None
    assert response.task_id == "route_001"
    assert response.result.get("paths_found", 0) >= 0


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_invalid_task_type(reasoning_dept):
    """Test error handling for invalid task type."""
    request = DepartmentRequest(
        task_id="invalid_task",
        task_type="unsupported_task",
        parameters={}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await reasoning_dept.execute(request)


@pytest.mark.asyncio
async def test_reasoning_missing_parameters(reasoning_dept):
    """Test error handling for missing parameters."""
    request = DepartmentRequest(
        task_id="missing_params",
        task_type="multi_hop_query",
        parameters={"max_hops": 5}  # Missing start_entity, end_entity
    )

    with pytest.raises(ValueError, match="Missing required parameter"):
        await reasoning_dept.execute(request)


# ============================================================================
# Test Performance
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_performance_multi_hop(reasoning_dept):
    """Test multi-hop reasoning performance."""
    request = DepartmentRequest(
        task_id="perf_001",
        task_type="multi_hop_query",
        parameters={
            "start_entity": "varroa_mites",
            "end_entity": "colony_collapse",
            "max_hops": 5
        }
    )

    response = await reasoning_dept.execute(request)

    # Should complete in reasonable time
    assert response.execution_time_ms < 5000  # <5 seconds


@pytest.mark.asyncio
async def test_reasoning_performance_causal(reasoning_dept):
    """Test causal inference performance."""
    request = DepartmentRequest(
        task_id="perf_002",
        task_type="causal_inference",
        parameters={
            "entity_a": "colony_health"
        }
    )

    response = await reasoning_dept.execute(request)

    # Should complete in reasonable time
    assert response.execution_time_ms < 3000  # <3 seconds


# ============================================================================
# Test Health Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_health_check(reasoning_dept):
    """Test department health monitoring."""
    # Execute some operations
    for i in range(3):
        request = DepartmentRequest(
            task_id=f"health_req_{i}",
            task_type="multi_hop_query",
            parameters={
                "start_entity": "varroa_mites",
                "end_entity": "colony_health",
                "max_hops": 3
            }
        )
        await reasoning_dept.execute(request)

    # Check health
    health = await reasoning_dept.health_check()

    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert health["name"] == "reasoning"
    assert health["performance"]["total_requests"] >= 3


# ============================================================================
# Test Lifecycle
# ============================================================================

@pytest.mark.asyncio
async def test_reasoning_lifecycle_context_manager():
    """Test Reasoning Department lifecycle."""
    async with ReasoningDepartment() as dept:
        assert dept._initialized == True

        # Execute request
        kg = KG()
        kg.add_edges([
            KGEdge("a", "b", "CAUSES", 0.9),
            KGEdge("b", "c", "LEADS_TO", 0.8)
        ])
        dept._knowledge_graph = kg

        request = DepartmentRequest(
            task_id="lifecycle_test",
            task_type="multi_hop_query",
            parameters={
                "start_entity": "a",
                "end_entity": "c",
                "max_hops": 3
            }
        )

        response = await dept.execute(request)
        assert response is not None

    # After exit, should be closed
    assert dept._closed == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

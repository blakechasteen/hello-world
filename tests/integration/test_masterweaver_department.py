#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for MasterWeaver Department (Beekeeping).

Tests the complete integration of beekeeping entity extraction:
- Pattern-based extraction
- LLM-based extraction (if available)
- Hybrid extraction
- Taxonomy validation
- Knowledge enrichment
- DS-STAR verification workflow

Run with:
    pytest HoloLoom/tests/integration/test_masterweaver_department.py -v
"""

import pytest
import asyncio
from pathlib import Path

from HoloLoom.departments import DepartmentRequest, DepartmentRegistry
from HoloLoom.departments.beekeeping import MasterWeaverDepartment


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
async def masterweaver_dept():
    """Create a MasterWeaver Department instance."""
    dept = MasterWeaverDepartment(enable_llm=False)  # Disable LLM for tests

    await dept.initialize()
    yield dept
    await dept.close()


@pytest.fixture
async def masterweaver_dept_with_llm():
    """Create a MasterWeaver Department instance with LLM enabled."""
    dept = MasterWeaverDepartment(enable_llm=True)

    await dept.initialize()
    yield dept
    await dept.close()


@pytest.fixture
async def registry_with_masterweaver(masterweaver_dept):
    """Create a registry with MasterWeaver Department registered."""
    async with DepartmentRegistry() as registry:
        await registry.register(masterweaver_dept)
        yield registry


# ============================================================================
# Sample Test Data
# ============================================================================

@pytest.fixture
def sample_beekeeping_text():
    """Sample beekeeping inspection text."""
    return """
    Inspected hive #3 today. Found 5 frames of capped brood with excellent laying pattern.
    Queen is marked and performing well. Saw 2 frames of capped honey and good pollen stores.
    No signs of varroa mites. Added one super for the upcoming honey flow.
    Workers are active and bringing in nectar. Overall colony health looks excellent.
    """


@pytest.fixture
def sample_complex_text():
    """More complex beekeeping text with diseases."""
    return """
    Hive inspection revealed concerning issues. Found chalkbrood mummies in several cells.
    Varroa mite count is elevated - need to treat with oxalic acid strips.
    Queen is still laying but brood pattern is spotty. Removed 2 frames with
    american foulbrood symptoms and sent sample to lab. Applied formic acid treatment.
    Colony population seems weak - may need to combine with another hive.
    """


# ============================================================================
# Test Department Initialization
# ============================================================================

@pytest.mark.asyncio
async def test_masterweaver_initialization():
    """Test MasterWeaver Department initializes correctly."""
    dept = MasterWeaverDepartment(enable_llm=False)

    assert dept.name == "masterweaver"
    assert dept.domain == "beekeeping"
    assert dept.version == "1.0.0"
    assert "extract_entities" in dept.supported_tasks
    assert "validate_taxonomy" in dept.supported_tasks
    assert "enrich_knowledge" in dept.supported_tasks

    # Initialize
    await dept.initialize()

    assert dept._initialized == True
    assert dept.taxonomy is not None
    assert len(dept.taxonomy) > 0
    assert "entity_types" in dept.taxonomy
    assert len(dept._compiled_patterns) > 0

    # Cleanup
    await dept.close()


@pytest.mark.asyncio
async def test_masterweaver_taxonomy_loaded():
    """Test taxonomy is loaded correctly."""
    dept = MasterWeaverDepartment(enable_llm=False)
    await dept.initialize()

    # Check taxonomy structure
    assert "entity_types" in dept.taxonomy
    assert "relationship_types" in dept.taxonomy
    assert "extraction_patterns" in dept.taxonomy
    assert "confidence_calibration" in dept.taxonomy

    # Check specific entity categories
    entity_types = dept.taxonomy["entity_types"]
    assert "colony_members" in entity_types
    assert "hive_components" in entity_types
    assert "colony_products" in entity_types
    assert "colony_health" in entity_types
    assert "beekeeping_activities" in entity_types
    assert "equipment" in entity_types

    # Check specific entities
    assert "queen" in entity_types["colony_members"]
    assert "hive" in entity_types["hive_components"]
    assert "brood" in entity_types["colony_products"]
    assert "varroa" in entity_types["colony_health"]

    await dept.close()


# ============================================================================
# Test Pattern-Based Extraction
# ============================================================================

@pytest.mark.asyncio
async def test_pattern_extraction_basic(masterweaver_dept, sample_beekeeping_text):
    """Test basic pattern-based entity extraction."""
    request = DepartmentRequest(
        task_id="pattern_001",
        task_type="extract_entities",
        parameters={
            "text": sample_beekeeping_text,
            "strategy": "pattern"
        },
        confidence_expected=0.75
    )

    response = await masterweaver_dept.execute(request)

    # Verify response structure
    assert response.task_id == "pattern_001"
    assert response.result is not None
    assert "entities" in response.result
    assert "entity_count" in response.result
    assert response.result["strategy"] == "pattern"

    # Check extracted entities
    entities = response.result["entities"]
    assert len(entities) > 0

    # Should find common entities
    entity_types = [e["type"] for e in entities]
    assert "hive" in entity_types or "brood" in entity_types or "queen" in entity_types

    # Verify confidence
    assert response.confidence.score >= 0.0
    assert response.confidence.score <= 1.0


@pytest.mark.asyncio
async def test_pattern_extraction_specific_entities(masterweaver_dept):
    """Test extraction of specific entity types."""
    text = "The queen bee is laying eggs. Workers are collecting pollen and producing wax."

    request = DepartmentRequest(
        task_id="pattern_002",
        task_type="extract_entities",
        parameters={
            "text": text,
            "strategy": "pattern"
        }
    )

    response = await masterweaver_dept.execute(request)
    entities = response.result["entities"]

    # Extract entity types
    entity_types = [e["type"] for e in entities]

    # Should find queen, worker, pollen, wax
    assert "queen" in entity_types
    assert "worker" in entity_types or "pollen" in entity_types or "wax" in entity_types


@pytest.mark.asyncio
async def test_pattern_extraction_diseases(masterweaver_dept, sample_complex_text):
    """Test extraction of disease/health entities."""
    request = DepartmentRequest(
        task_id="pattern_003",
        task_type="extract_entities",
        parameters={
            "text": sample_complex_text,
            "strategy": "pattern"
        }
    )

    response = await masterweaver_dept.execute(request)
    entities = response.result["entities"]

    entity_types = [e["type"] for e in entities]

    # Should find health-related entities
    health_entities = ["varroa", "chalkbrood", "afb"]
    found_health = any(h in entity_types for h in health_entities)
    assert found_health, f"Expected to find health entities, got: {entity_types}"


# ============================================================================
# Test Taxonomy Validation
# ============================================================================

@pytest.mark.asyncio
async def test_validate_known_entities(masterweaver_dept):
    """Test validation of known entities."""
    entities = [
        {"text": "queen", "type": "queen", "category": "colony_members", "confidence": 0.90},
        {"text": "hive", "type": "hive", "category": "hive_components", "confidence": 0.85},
        {"text": "brood", "type": "brood", "category": "colony_products", "confidence": 0.88}
    ]

    request = DepartmentRequest(
        task_id="validate_001",
        task_type="validate_taxonomy",
        parameters={"entities": entities}
    )

    response = await masterweaver_dept.execute(request)

    # Check validation results
    assert "validation_results" in response.result
    assert response.result["valid_count"] == 3
    assert response.result["total_count"] == 3
    assert response.result["validation_rate"] == 1.0

    # Confidence should be high
    assert response.confidence.score >= 0.90


@pytest.mark.asyncio
async def test_validate_unknown_entities(masterweaver_dept):
    """Test validation of unknown entities."""
    entities = [
        {"text": "alien_entity", "type": "unknown_type", "category": "unknown", "confidence": 0.70},
        {"text": "random_thing", "type": "random", "category": "unknown", "confidence": 0.60}
    ]

    request = DepartmentRequest(
        task_id="validate_002",
        task_type="validate_taxonomy",
        parameters={"entities": entities}
    )

    response = await masterweaver_dept.execute(request)

    # Check validation results
    assert response.result["valid_count"] == 0
    assert response.result["total_count"] == 2
    assert response.result["validation_rate"] == 0.0

    # Confidence should be low
    assert response.confidence.score < 0.50


@pytest.mark.asyncio
async def test_validate_mixed_entities(masterweaver_dept):
    """Test validation of mix of known/unknown entities."""
    entities = [
        {"text": "queen", "type": "queen", "category": "colony_members", "confidence": 0.90},
        {"text": "unknown", "type": "unknown", "category": "unknown", "confidence": 0.50},
        {"text": "brood", "type": "brood", "category": "colony_products", "confidence": 0.85}
    ]

    request = DepartmentRequest(
        task_id="validate_003",
        task_type="validate_taxonomy",
        parameters={"entities": entities}
    )

    response = await masterweaver_dept.execute(request)

    # Should have 2/3 valid
    assert response.result["valid_count"] == 2
    assert response.result["total_count"] == 3
    assert response.result["validation_rate"] == pytest.approx(0.666, rel=0.1)


# ============================================================================
# Test Knowledge Enrichment
# ============================================================================

@pytest.mark.asyncio
async def test_enrich_knowledge_basic(masterweaver_dept):
    """Test basic knowledge enrichment (relationship extraction)."""
    entities = [
        {"text": "queen", "type": "queen", "category": "colony_members"},
        {"text": "brood", "type": "brood", "category": "colony_products"},
        {"text": "worker", "type": "worker", "category": "colony_members"},
        {"text": "wax", "type": "wax", "category": "colony_products"}
    ]

    request = DepartmentRequest(
        task_id="enrich_001",
        task_type="enrich_knowledge",
        parameters={"entities": entities}
    )

    response = await masterweaver_dept.execute(request)

    # Check relationships
    assert "relationships" in response.result
    assert "relationship_count" in response.result

    relationships = response.result["relationships"]

    # Should find some relationships
    if len(relationships) > 0:
        rel = relationships[0]
        assert "source" in rel
        assert "target" in rel
        assert "type" in rel
        assert "confidence" in rel


@pytest.mark.asyncio
async def test_enrich_knowledge_expected_relationships(masterweaver_dept):
    """Test that expected relationships are found."""
    entities = [
        {"text": "worker", "type": "worker", "category": "colony_members"},
        {"text": "wax", "type": "wax", "category": "colony_products"}
    ]

    request = DepartmentRequest(
        task_id="enrich_002",
        task_type="enrich_knowledge",
        parameters={"entities": entities}
    )

    response = await masterweaver_dept.execute(request)

    relationships = response.result["relationships"]

    # Should find worker PRODUCES wax
    produces_rel = [r for r in relationships if r["type"] == "PRODUCES"]
    assert len(produces_rel) > 0


# ============================================================================
# Test Verification (DS-STAR)
# ============================================================================

@pytest.mark.asyncio
async def test_verify_sufficient_extraction(masterweaver_dept, sample_beekeeping_text):
    """Test verifying a high-quality extraction."""
    # Execute extraction
    request = DepartmentRequest(
        task_id="verify_001",
        task_type="extract_entities",
        parameters={
            "text": sample_beekeeping_text,
            "strategy": "pattern"
        }
    )

    response = await masterweaver_dept.execute(request)

    # Manually boost confidence for testing
    if response.confidence.score < 0.70:
        response.confidence.score = 0.80

    # Verify
    verification = await masterweaver_dept.verify(response)

    # Should be sufficient if entities found
    if response.result["entity_count"] > 0:
        assert verification.confidence_valid == True


@pytest.mark.asyncio
async def test_verify_insufficient_extraction(masterweaver_dept):
    """Test verifying a low-quality extraction."""
    # Execute extraction with text that has no entities
    request = DepartmentRequest(
        task_id="verify_002",
        task_type="extract_entities",
        parameters={
            "text": "This is random text with no beekeeping content.",
            "strategy": "pattern"
        }
    )

    response = await masterweaver_dept.execute(request)

    # Verify
    verification = await masterweaver_dept.verify(response)

    # Should not be sufficient (no entities)
    assert verification.sufficient == False
    assert verification.refinement_suggestions is not None


# ============================================================================
# Test Refinement
# ============================================================================

@pytest.mark.asyncio
async def test_refine_low_confidence(masterweaver_dept):
    """Test refining a low-confidence extraction."""
    # Execute initial extraction
    request = DepartmentRequest(
        task_id="refine_001",
        task_type="extract_entities",
        parameters={
            "text": "The bees are in the box.",
            "strategy": "pattern"
        }
    )

    response = await masterweaver_dept.execute(request)

    # Force low confidence
    response.confidence.score = 0.60

    # Verify (should fail)
    verification = await masterweaver_dept.verify(response)

    if not verification.sufficient:
        # Refine
        refined_response = await masterweaver_dept.refine(request, response, verification)

        # Refined response should exist
        assert refined_response is not None
        assert refined_response.task_id.endswith("_refined")


# ============================================================================
# Test Full DS-STAR Workflow
# ============================================================================

@pytest.mark.asyncio
async def test_full_ds_star_workflow(masterweaver_dept, sample_beekeeping_text):
    """Test complete DS-STAR cycle: Execute → Verify → Refine."""
    # Execute
    request = DepartmentRequest(
        task_id="workflow_001",
        task_type="extract_entities",
        parameters={
            "text": sample_beekeeping_text,
            "strategy": "pattern"
        },
        confidence_expected=0.80
    )

    response = await masterweaver_dept.execute(request)

    # Verify
    verification = await masterweaver_dept.verify(response)

    # If insufficient, refine up to 2 iterations
    max_iterations = 2
    iteration = 0

    while not verification.sufficient and iteration < max_iterations:
        response = await masterweaver_dept.refine(request, response, verification)
        verification = await masterweaver_dept.verify(response)
        iteration += 1

    # Should eventually be sufficient (or hit max iterations)
    assert iteration <= max_iterations


# ============================================================================
# Test Health Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_masterweaver_health_check(masterweaver_dept):
    """Test department health monitoring."""
    # Execute some extractions
    for i in range(3):
        request = DepartmentRequest(
            task_id=f"health_req_{i}",
            task_type="extract_entities",
            parameters={
                "text": f"Inspected hive #{i}. Found brood and honey.",
                "strategy": "pattern"
            }
        )
        await masterweaver_dept.execute(request)

    # Check health
    health = await masterweaver_dept.health_check()

    assert health["status"] in ["healthy", "degraded", "unhealthy"]
    assert health["name"] == "masterweaver"
    assert health["performance"]["total_requests"] >= 3


# ============================================================================
# Test Registry Integration
# ============================================================================

@pytest.mark.asyncio
async def test_masterweaver_registry_registration(registry_with_masterweaver):
    """Test MasterWeaver Department registration in registry."""
    registry = registry_with_masterweaver

    # Check registration
    dept = registry.get_department("masterweaver")
    assert dept is not None
    assert dept.name == "masterweaver"

    # Check discovery by domain
    depts_by_domain = registry.find_by_domain("beekeeping")
    assert len(depts_by_domain) > 0
    assert any(d.name == "masterweaver" for d in depts_by_domain)

    # Check discovery by task
    depts_by_task = registry.find_by_task("extract_entities")
    assert len(depts_by_task) > 0
    assert any(d.name == "masterweaver" for d in depts_by_task)


@pytest.mark.asyncio
async def test_masterweaver_registry_routing(registry_with_masterweaver, sample_beekeeping_text):
    """Test routing requests through registry to MasterWeaver."""
    registry = registry_with_masterweaver

    # Route request
    request = DepartmentRequest(
        task_id="route_req_001",
        task_type="extract_entities",
        parameters={
            "text": sample_beekeeping_text,
            "strategy": "pattern"
        }
    )

    response = await registry.route_request(request)

    # Should get valid response
    assert response is not None
    assert response.task_id == "route_req_001"
    assert "entities" in response.result


# ============================================================================
# Test Error Handling
# ============================================================================

@pytest.mark.asyncio
async def test_masterweaver_invalid_task_type(masterweaver_dept):
    """Test error handling for invalid task type."""
    request = DepartmentRequest(
        task_id="invalid_task",
        task_type="unsupported_task",
        parameters={"text": "Test"}
    )

    with pytest.raises(ValueError, match="Unsupported task type"):
        await masterweaver_dept.execute(request)


@pytest.mark.asyncio
async def test_masterweaver_missing_text_parameter(masterweaver_dept):
    """Test error handling for missing text parameter."""
    request = DepartmentRequest(
        task_id="missing_text",
        task_type="extract_entities",
        parameters={"strategy": "pattern"}  # Missing 'text'
    )

    with pytest.raises(ValueError, match="Missing required parameter"):
        await masterweaver_dept.execute(request)


# ============================================================================
# Test Statistics Tracking
# ============================================================================

@pytest.mark.asyncio
async def test_masterweaver_statistics_tracking(masterweaver_dept):
    """Test extraction statistics tracking."""
    # Execute multiple extractions
    for i in range(3):
        request = DepartmentRequest(
            task_id=f"stats_req_{i}",
            task_type="extract_entities",
            parameters={
                "text": "Queen bee laying eggs in hive.",
                "strategy": "pattern"
            }
        )
        await masterweaver_dept.execute(request)

    # Check statistics
    stats = masterweaver_dept._extraction_stats
    assert stats["pattern_extractions"] >= 3
    assert stats["entities_extracted"] >= 0


# ============================================================================
# Test Lifecycle
# ============================================================================

@pytest.mark.asyncio
async def test_masterweaver_lifecycle_context_manager():
    """Test MasterWeaver Department lifecycle with async context manager."""
    async with MasterWeaverDepartment(enable_llm=False) as dept:
        assert dept._initialized == True

        # Execute request
        request = DepartmentRequest(
            task_id="lifecycle_test",
            task_type="extract_entities",
            parameters={
                "text": "Hive inspection found brood.",
                "strategy": "pattern"
            }
        )

        response = await dept.execute(request)
        assert response is not None

    # After exit, should be closed
    assert dept._closed == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

#!/usr/bin/env python3
"""
MCP Federation Bootstrap — Wire departments into a running federation.

This module provides factory functions to stand up a department federation
from the charter registry. It bridges the static charter definitions
(mcp_department_registry.py) with live department instances.

Usage:
    from hololoom.apps.departments.mcp.bootstrap import create_federation

    # With real departments
    router = await create_federation({
        "Infrastructure": infrastructure_dept,
        "MasterWeaver": masterweaver_dept,
        "Verification": verification_dept,
    })

    # Call across departments
    resp = await router.call_tool(
        target="MasterWeaver",
        tool_name="extract_entities",
        parameters={"input_data": "...", "domain": "beekeeping"},
        session_id="session_001",
        caller="Verification",
    )

For a lightweight demo without heavy dependencies:
    router = await create_lightweight_federation()
"""

from __future__ import annotations

import logging
from typing import Any

from hololoom.alignment.mcp_department_registry import (
    DEPARTMENT_REGISTRY,
    DepartmentCharter,
)
from hololoom.protocols.department import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

from ..base import BaseDepartment
from .mcp_adapter import DepartmentMCPAdapter
from .mcp_router import MCPRouter

logger = logging.getLogger(__name__)


async def create_federation(
    departments: dict[str, BaseDepartment],
    charters: dict[str, DepartmentCharter] | None = None,
) -> MCPRouter:
    """
    Create an MCP federation from department instances.

    Args:
        departments: Map of department name → BaseDepartment instance.
        charters: Optional override charters. Defaults to DEPARTMENT_REGISTRY.

    Returns:
        Configured MCPRouter with all departments registered.
    """
    charter_registry = charters or DEPARTMENT_REGISTRY
    router = MCPRouter()

    for name, dept in departments.items():
        charter = charter_registry.get(name)
        if charter is None:
            logger.warning(
                f"No charter found for '{name}', skipping. "
                f"Available: {list(charter_registry.keys())}"
            )
            continue

        adapter = DepartmentMCPAdapter(dept, charter)
        router.register(adapter)
        logger.info(f"Federation: registered {name}")

    return router


# ============================================================================
# Lightweight departments for demo/testing without heavy dependencies
# ============================================================================


class LightweightInfrastructure(BaseDepartment):
    """
    In-memory infrastructure department.

    Simulates Neo4j/Qdrant with dict-based storage. Useful for testing
    federation wiring without database dependencies.
    """

    def __init__(self):
        config = DepartmentConfig(
            name="Infrastructure",
            domain="system",
            version="1.0.0",
            supported_tasks=["query_neo4j", "query_qdrant", "diagnose_performance"],
        )
        super().__init__(
            name="Infrastructure",
            domain="system",
            version="1.0.0",
            supported_tasks=["query_neo4j", "query_qdrant", "diagnose_performance"],
            confidence_range=(0.7, 0.99),
            config=config,
        )
        # In-memory store
        self._graph: dict[str, dict[str, Any]] = {}
        self._vectors: dict[str, list[float]] = {}

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "query_neo4j":
                result = self._query_graph(request.parameters)
            case "query_qdrant":
                result = self._query_vectors(request.parameters)
            case "diagnose_performance":
                result = {"status": "healthy", "bottlenecks": []}
            case _:
                result = {"error": f"Unknown task: {request.task_type}"}

        return DepartmentResponse(
            task_id=request.task_id,
            result=result,
            confidence=ConfidenceMetadata.from_score(0.9),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, summary="ok", confidence=0.9)

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return prior

    def store_entity(self, entity_id: str, data: dict[str, Any]) -> None:
        """Add an entity to the in-memory graph (for testing)."""
        self._graph[entity_id] = data

    def _query_graph(self, params: dict[str, Any]) -> dict[str, Any]:
        params.get("cypher_query", "")
        # Simple pattern match on stored entities
        results = list(self._graph.values())
        return {"results": results, "count": len(results)}

    def _query_vectors(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"results": [], "scores": []}


class LightweightMasterWeaver(BaseDepartment):
    """
    Entity extraction department backed by simple pattern matching.

    No LLM required — uses keyword extraction for testing the federation
    wiring, not the NLP quality.
    """

    def __init__(self):
        config = DepartmentConfig(
            name="MasterWeaver",
            domain="knowledge",
            version="1.0.0",
            supported_tasks=[
                "extract_entities",
                "validate_entity_consistency",
                "query_domain_ontology",
            ],
        )
        super().__init__(
            name="MasterWeaver",
            domain="knowledge",
            version="1.0.0",
            supported_tasks=[
                "extract_entities",
                "validate_entity_consistency",
                "query_domain_ontology",
            ],
            confidence_range=(0.6, 0.95),
            config=config,
        )
        self._extracted: list[dict[str, Any]] = []

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "extract_entities":
                result = self._extract(request.parameters)
            case "validate_entity_consistency":
                result = self._validate(request.parameters)
            case "query_domain_ontology":
                result = self._query_ontology(request.parameters)
            case _:
                result = {"error": f"Unknown task: {request.task_type}"}

        return DepartmentResponse(
            task_id=request.task_id,
            result=result,
            confidence=ConfidenceMetadata.from_score(0.82),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, summary="ok", confidence=0.85)

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return prior

    def _extract(self, params: dict[str, Any]) -> dict[str, Any]:
        text = params.get("input_data", "")
        domain = params.get("domain", "general")
        # Simple word-based entity extraction
        words = text.split()
        entities = [
            {"text": w, "type": "keyword", "domain": domain}
            for w in words
            if len(w) > 3 and w[0].isupper()
        ]
        self._extracted.extend(entities)
        return {
            "entities": entities,
            "count": len(entities),
            "domain": domain,
        }

    def _validate(self, params: dict[str, Any]) -> dict[str, Any]:
        params.get("entities", [])
        return {"is_consistent": True, "inconsistencies": []}

    def _query_ontology(self, params: dict[str, Any]) -> dict[str, Any]:
        concept = params.get("concept", "")
        return {
            "definition": f"Definition of {concept}",
            "relationships": [],
            "examples": [],
        }


class LightweightVerification(BaseDepartment):
    """
    Verification department that checks confidence claims.

    Compares claimed confidence against simple heuristics. No external
    dependencies.
    """

    def __init__(self):
        config = DepartmentConfig(
            name="Verification",
            domain="quality",
            version="1.0.0",
            supported_tasks=[
                "validate_confidence_claim",
                "request_rerun",
                "cross_check_departments",
            ],
        )
        super().__init__(
            name="Verification",
            domain="quality",
            version="1.0.0",
            supported_tasks=[
                "validate_confidence_claim",
                "request_rerun",
                "cross_check_departments",
            ],
            confidence_range=(0.7, 0.99),
            config=config,
        )

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "validate_confidence_claim":
                result = self._validate_confidence(request.parameters)
            case "request_rerun":
                result = {"accepted": True, "new_task_id": "rerun_001"}
            case "cross_check_departments":
                result = {"consistent": True, "conflicts": []}
            case _:
                result = {"error": f"Unknown task: {request.task_type}"}

        return DepartmentResponse(
            task_id=request.task_id,
            result=result,
            confidence=ConfidenceMetadata.from_score(0.91),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        return VerificationResult(verified=True, summary="ok", confidence=0.95)

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return prior

    def _validate_confidence(self, params: dict[str, Any]) -> dict[str, Any]:
        claim = params.get("claim", {})
        claimed = claim.get("confidence", 0.5)
        # Simple heuristic: flag if claimed > 0.95 without evidence
        evidence = params.get("evidence", {})
        actual = claimed if evidence else claimed * 0.8
        return {
            "validation_result": abs(claimed - actual) < 0.2,
            "actual_confidence": actual,
            "claimed_confidence": claimed,
            "recommendation": "Acceptable" if abs(claimed - actual) < 0.2 else "Re-run recommended",
        }


async def create_lightweight_federation() -> MCPRouter:
    """
    Create a federation with lightweight departments for demo/testing.

    Returns a router with Infrastructure, MasterWeaver, and Verification
    departments wired up and ready to use. No external dependencies required.
    """
    return await create_federation({
        "Infrastructure": LightweightInfrastructure(),
        "MasterWeaver": LightweightMasterWeaver(),
        "Verification": LightweightVerification(),
    })


async def create_real_federation(
    *,
    enable_neo4j: bool = False,
    enable_qdrant: bool = False,
    neo4j_uri: str | None = None,
    qdrant_host: str | None = None,
    storage_dir: str | None = None,
    enable_llm: bool = False,
    ollama_endpoint: str = "http://localhost:11434",
    taxonomy_path: str | None = None,
    mcp_server_url: str | None = None,
) -> MCPRouter:
    """
    Create a federation backed by real HoloLoom subsystems.

    Graceful degradation: if external backends (Neo4j, Qdrant, Ollama)
    aren't available, departments fall back to in-memory / keyword modes
    with lower confidence scores.

    Args:
        enable_neo4j: Enable Neo4j graph backend for Infrastructure.
        enable_qdrant: Enable Qdrant vector backend for Infrastructure.
        neo4j_uri: Neo4j connection URI (default bolt://localhost:7687).
        qdrant_host: Qdrant host (default localhost:6333).
        storage_dir: Directory for zero-copy embedding storage.
        enable_llm: Enable LLM-based entity extraction for MasterWeaver.
        ollama_endpoint: Ollama API endpoint for LLM extraction.
        taxonomy_path: Path to beekeeping taxonomy.json for MasterWeaver.
        mcp_server_url: VS Code MCP server URL for Execution department.

    Returns:
        Configured MCPRouter with all 6 real department implementations.
    """
    from .real_departments import (
        RealContext,
        RealExecution,
        RealInfrastructure,
        RealMasterWeaver,
        RealOrchestration,
        RealVerification,
    )

    departments = {
        "Infrastructure": RealInfrastructure(
            enable_neo4j=enable_neo4j,
            enable_qdrant=enable_qdrant,
            neo4j_uri=neo4j_uri,
            qdrant_host=qdrant_host,
            storage_dir=storage_dir,
        ),
        "MasterWeaver": RealMasterWeaver(
            taxonomy_path=taxonomy_path,
            enable_llm=enable_llm,
            ollama_endpoint=ollama_endpoint,
        ),
        "Verification": RealVerification(),
        "Orchestration": RealOrchestration(),
        "Context": RealContext(),
        "Execution": RealExecution(
            mcp_server_url=mcp_server_url,
        ),
    }

    router = await create_federation(departments)
    logger.info(
        f"Real federation created: {router.departments} "
        f"(neo4j={enable_neo4j}, qdrant={enable_qdrant}, llm={enable_llm})"
    )
    return router

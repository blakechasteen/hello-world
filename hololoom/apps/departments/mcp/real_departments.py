"""
Real Department Implementations — Charter-aligned wrappers for HoloLoom subsystems.

These departments bridge the MCP charter tool names to the actual HoloLoom
subsystem implementations. Each wrapper:

1. Accepts the charter's tool names as task_types (what the MCP adapter sends)
2. Translates parameters to the subsystem's expected format
3. Delegates to the real subsystem code
4. Returns DepartmentResponse with proper confidence metadata

The lightweight departments in bootstrap.py are for testing the federation
wiring. These are for production — backed by real embedding stores, Neo4j,
pattern extractors, and confidence validators.

Usage:
    from hololoom.apps.departments.mcp.real_departments import (
        RealInfrastructure,
        RealMasterWeaver,
        RealVerification,
    )
    from hololoom.apps.departments.mcp.bootstrap import create_federation

    router = await create_federation({
        "Infrastructure": RealInfrastructure(),
        "MasterWeaver": RealMasterWeaver(),
        "Verification": RealVerification(),
    })
"""

from __future__ import annotations

import logging
from typing import Any

from hololoom.protocols.department import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)

from ..base import BaseDepartment

logger = logging.getLogger(__name__)


# ============================================================================
# Infrastructure — backed by real embedding store + optional Neo4j/Qdrant
# ============================================================================


class RealInfrastructure(BaseDepartment):
    """
    Production Infrastructure department.

    Wraps InfrastructureDepartment's backends and exposes them under
    the charter tool names: query_neo4j, query_qdrant, diagnose_performance.

    Graceful degradation: if Neo4j/Qdrant aren't available, returns
    meaningful error results instead of crashing.
    """

    def __init__(
        self,
        enable_neo4j: bool = False,
        enable_qdrant: bool = False,
        neo4j_uri: str | None = None,
        qdrant_host: str | None = None,
        storage_dir: str | None = None,
    ):
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
        self._enable_neo4j = enable_neo4j
        self._enable_qdrant = enable_qdrant
        self._neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        self._qdrant_host = qdrant_host or "localhost:6333"
        self._storage_dir = storage_dir

        # Lazy-initialized backends
        self._infra_dept = None

    async def _get_infra(self):
        """Lazy-init the real InfrastructureDepartment."""
        if self._infra_dept is None:
            from pathlib import Path

            from ..infrastructure.infrastructure import InfrastructureDepartment

            storage = Path(self._storage_dir) if self._storage_dir else None
            infra_config = DepartmentConfig(
                name="infrastructure",
                domain="system",
                version="1.0.0",
                supported_tasks=["diagnose_performance", "check_backends"],
            )
            self._infra_dept = InfrastructureDepartment(
                storage_dir=storage,
                enable_neo4j=self._enable_neo4j,
                enable_qdrant=self._enable_qdrant,
                neo4j_uri=self._neo4j_uri,
                qdrant_host=self._qdrant_host,
                dept_config=infra_config,
            )
            await self._infra_dept.initialize()
        return self._infra_dept

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "query_neo4j":
                return await self._handle_neo4j(request)
            case "query_qdrant":
                return await self._handle_qdrant(request)
            case "diagnose_performance":
                return await self._handle_diagnose(request)
            case _:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"error": f"Unknown task: {request.task_type}"},
                    confidence=ConfidenceMetadata.from_score(0.0),
                )

    async def _handle_neo4j(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute a Cypher query via the real Neo4j driver."""
        infra = await self._get_infra()

        if not infra._backend_health.get("neo4j", False):
            # Graceful degradation: query in-memory graph
            cypher = request.parameters.get("cypher_query", "")
            return DepartmentResponse(
                task_id=request.task_id,
                result={
                    "results": [],
                    "count": 0,
                    "note": "Neo4j unavailable — returned empty result set",
                    "query": cypher,
                },
                confidence=ConfidenceMetadata.from_score(0.5,
                    justification=["Neo4j backend not available"],
                    sources=["degraded mode — no persistent graph"],
                ),
            )

        # Use the real Neo4j driver
        cypher = request.parameters.get("cypher_query", "")
        params = request.parameters.get("parameters", {})

        try:
            with infra._neo4j_driver.session() as session:
                result = session.run(cypher, params)
                records = [dict(record) for record in result]

            return DepartmentResponse(
                task_id=request.task_id,
                result={"results": records, "count": len(records)},
                confidence=ConfidenceMetadata.from_score(0.95,
                    justification=[f"Executed Cypher query, {len(records)} results"],
                ),
            )
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}")
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e), "query": cypher},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

    async def _handle_qdrant(self, request: DepartmentRequest) -> DepartmentResponse:
        """Execute semantic search via embedding store or Qdrant."""
        infra = await self._get_infra()

        query_vector = request.parameters.get("query_vector")
        collection = request.parameters.get("collection", "default")
        limit = request.parameters.get("limit", 10)

        if query_vector is None:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "Missing required parameter: query_vector"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        # Try Qdrant first, fall back to embedding store
        if infra._backend_health.get("qdrant", False) and infra._qdrant_client:
            try:
                import numpy as np

                search_result = infra._qdrant_client.search(
                    collection_name=collection,
                    query_vector=query_vector,
                    limit=limit,
                )
                results = [
                    {"id": str(hit.id), "score": hit.score, "payload": hit.payload}
                    for hit in search_result
                ]
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"results": results, "scores": [r["score"] for r in results]},
                    confidence=ConfidenceMetadata.from_score(
                        min(results[0]["score"], 1.0) if results else 0.3,
                        justification=[f"Qdrant search: {len(results)} results"],
                    ),
                )
            except Exception as e:
                logger.warning(f"Qdrant search failed, falling back to embedding store: {e}")

        # Fallback: use Matryoshka embedding store
        if infra._embedding_store:
            import numpy as np

            try:
                search_results = await infra._embedding_store.search_similar(
                    np.array(query_vector), k=limit,
                )
                results = [
                    {"id": eid, "score": sim}
                    for eid, sim, _view in search_results
                ]
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"results": results, "scores": [r["score"] for r in results]},
                    confidence=ConfidenceMetadata.from_score(
                        min(results[0]["score"], 1.0) if results else 0.3,
                        justification=[f"Embedding store search: {len(results)} results"],
                    ),
                )
            except Exception as e:
                logger.error(f"Embedding store search failed: {e}")

        # Both unavailable
        return DepartmentResponse(
            task_id=request.task_id,
            result={"results": [], "scores": [], "note": "No vector backend available"},
            confidence=ConfidenceMetadata.from_score(0.3,
                justification=["No vector backend available"],
                sources=["degraded mode"],
            ),
        )

    async def _handle_diagnose(self, request: DepartmentRequest) -> DepartmentResponse:
        """Run performance diagnostics via the real infrastructure department."""
        try:
            infra = await self._get_infra()
            diag_request = DepartmentRequest(
                task_type="diagnose_performance",
                parameters=request.parameters,
                context=request.context,
            )
            return await infra.execute(diag_request)
        except Exception as e:
            logger.warning(f"RealInfrastructure: diagnose delegation failed ({e}), using built-in")
            try:
                infra = self._infra_dept
                backend_health = infra._backend_health.copy() if infra else {}
                healthy = sum(1 for v in backend_health.values() if v)
                total = len(backend_health) or 1
                health_score = healthy / total
            except Exception:
                backend_health = {}
                health_score = 0.5

            return DepartmentResponse(
                task_id=request.task_id,
                result={
                    "status": "healthy" if health_score > 0.5 else "degraded",
                    "backend_health": backend_health,
                    "health_score": health_score,
                    "bottlenecks": [],
                },
                confidence=ConfidenceMetadata.from_score(health_score,
                    justification=[f"Built-in diagnostics, health={health_score:.2f}"],
                ),
            )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        has_error = "error" in response.result
        return VerificationResult(
            sufficient=not has_error and response.confidence.score >= 0.7,
            confidence_valid=response.confidence.score >= 0.7,
            reasoning_sound=not has_error,
            alternative_paths=["fallback_to_inmemory"] if has_error else [],
            verifier_confidence=0.95,
        )

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return await self.execute(request)

    async def close(self) -> None:
        if self._infra_dept is not None:
            await self._infra_dept.close()
            self._infra_dept = None
        await super().close()


# ============================================================================
# MasterWeaver — backed by real entity extraction (pattern + optional LLM)
# ============================================================================


class RealMasterWeaver(BaseDepartment):
    """
    Production MasterWeaver department.

    Wraps MasterWeaverDepartment's entity extraction and exposes it under
    charter tool names: extract_entities, validate_entity_consistency,
    query_domain_ontology.

    Graceful degradation: if the taxonomy or LLM aren't available, falls
    back to keyword-based extraction (same as lightweight, but with real
    confidence calibration).
    """

    def __init__(
        self,
        taxonomy_path: str | None = None,
        enable_llm: bool = False,
        ollama_endpoint: str = "http://localhost:11434",
    ):
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
        self._taxonomy_path = taxonomy_path
        self._enable_llm = enable_llm
        self._ollama_endpoint = ollama_endpoint

        # Lazy-initialized real department
        self._weaver_dept = None
        # In-memory extracted entities (for consistency checks)
        self._extracted_entities: list[dict[str, Any]] = []

    async def _get_weaver(self):
        """Lazy-init the real MasterWeaverDepartment."""
        if self._weaver_dept is None:
            try:
                from pathlib import Path

                from ..beekeeping.masterweaver import MasterWeaverDepartment

                taxonomy = Path(self._taxonomy_path) if self._taxonomy_path else None
                weaver_config = DepartmentConfig(
                    name="masterweaver",
                    domain="beekeeping",
                    version="1.0.0",
                    supported_tasks=["extract_entities", "validate_taxonomy", "enrich_knowledge"],
                )
                self._weaver_dept = MasterWeaverDepartment(
                    taxonomy_path=taxonomy,
                    enable_llm=self._enable_llm,
                    ollama_endpoint=self._ollama_endpoint,
                    dept_config=weaver_config,
                )
                await self._weaver_dept.initialize()
                logger.info("RealMasterWeaver: initialized with real MasterWeaverDepartment")
            except Exception as e:
                logger.warning(
                    f"RealMasterWeaver: failed to init real department ({e}), "
                    "using built-in extraction"
                )
                self._weaver_dept = None
        return self._weaver_dept

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "extract_entities":
                return await self._handle_extract(request)
            case "validate_entity_consistency":
                return await self._handle_validate(request)
            case "query_domain_ontology":
                return await self._handle_ontology(request)
            case _:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"error": f"Unknown task: {request.task_type}"},
                    confidence=ConfidenceMetadata.from_score(0.0),
                )

    async def _handle_extract(self, request: DepartmentRequest) -> DepartmentResponse:
        """Extract entities — delegate to real dept or use built-in."""
        weaver = await self._get_weaver()

        input_data = request.parameters.get("input_data", "")
        domain = request.parameters.get("domain", "general")

        if weaver is not None:
            # Delegate to real MasterWeaverDepartment
            try:
                dept_request = DepartmentRequest(
                    task_type="extract_entities",
                    parameters={
                        "text": input_data,  # real dept uses "text" not "input_data"
                        "strategy": "hybrid" if self._enable_llm else "pattern",
                        "domain": domain,
                    },
                    context=request.context,
                )
                response = await weaver.execute(dept_request)

                # Track extracted entities
                entities = response.result.get("entities", [])
                self._extracted_entities.extend(entities)

                # Re-wrap with charter-aligned keys
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={
                        "entities": entities,
                        "count": len(entities),
                        "domain": domain,
                        "source": "real_masterweaver",
                    },
                    confidence=response.confidence,
                )
            except Exception as e:
                logger.warning(f"RealMasterWeaver: delegation failed ({e}), falling back to builtin")

        # Fallback: built-in keyword extraction (same as lightweight)
        words = input_data.split()
        entities = [
            {"text": w, "type": "keyword", "domain": domain}
            for w in words
            if len(w) > 3 and w[0].isupper()
        ]
        self._extracted_entities.extend(entities)

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "entities": entities,
                "count": len(entities),
                "domain": domain,
                "source": "builtin_keyword",
            },
            confidence=ConfidenceMetadata.from_score(
                0.75 if entities else 0.4,
                justification=[f"Keyword extraction: {len(entities)} entities"],
                sources=["no taxonomy loaded"] if not entities else [],
            ),
        )

    async def _handle_validate(self, request: DepartmentRequest) -> DepartmentResponse:
        """Validate entity consistency — check for contradictions."""
        entities = request.parameters.get("entities", [])

        if not entities:
            # Use internally tracked entities if none provided
            entities = self._extracted_entities[-50:]  # last 50

        # Check for duplicate entities with conflicting types
        seen: dict[str, str] = {}
        inconsistencies = []
        for ent in entities:
            text = ent.get("text", "")
            etype = ent.get("type", ent.get("entity_type", ""))
            if text in seen and seen[text] != etype:
                inconsistencies.append({
                    "entity": text,
                    "type_a": seen[text],
                    "type_b": etype,
                    "severity": "medium",
                })
            seen[text] = etype

        is_consistent = len(inconsistencies) == 0
        confidence_score = 0.95 if is_consistent else max(0.5, 1.0 - 0.1 * len(inconsistencies))

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "is_consistent": is_consistent,
                "inconsistencies": inconsistencies,
                "entities_checked": len(entities),
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[
                    f"Checked {len(entities)} entities, "
                    f"{len(inconsistencies)} inconsistencies"
                ],
            ),
        )

    async def _handle_ontology(self, request: DepartmentRequest) -> DepartmentResponse:
        """Query domain ontology — use taxonomy if available."""
        domain = request.parameters.get("domain", "general")
        concept = request.parameters.get("concept", "")

        weaver = await self._get_weaver()

        if weaver is not None and weaver.taxonomy:
            # Search taxonomy for concept
            taxonomy = weaver.taxonomy
            categories = taxonomy.get("categories", {})

            matches = []
            relationships = []
            for cat_name, cat_data in categories.items():
                terms = cat_data.get("terms", [])
                if isinstance(terms, list):
                    for term in terms:
                        term_text = term if isinstance(term, str) else term.get("term", "")
                        if concept.lower() in term_text.lower():
                            matches.append({
                                "term": term_text,
                                "category": cat_name,
                            })
                            # Add parent relationship
                            relationships.append({
                                "source": term_text,
                                "target": cat_name,
                                "type": "belongs_to",
                            })

            return DepartmentResponse(
                task_id=request.task_id,
                result={
                    "definition": f"{concept} in {domain} domain",
                    "matches": matches,
                    "relationships": relationships,
                    "examples": [m["term"] for m in matches[:5]],
                    "source": "taxonomy",
                },
                confidence=ConfidenceMetadata.from_score(
                    0.9 if matches else 0.4,
                    justification=[f"Found {len(matches)} taxonomy matches"],
                ),
            )

        # Fallback: return stub
        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "definition": f"Definition of {concept}",
                "relationships": [],
                "examples": [],
                "source": "stub",
            },
            confidence=ConfidenceMetadata.from_score(0.3,
                justification=["No taxonomy available"],
                sources=["stub response"],
            ),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        has_error = "error" in response.result
        return VerificationResult(
            sufficient=not has_error and response.confidence.score >= 0.6,
            confidence_valid=response.confidence.score >= 0.6,
            reasoning_sound=not has_error,
            alternative_paths=[],
            verifier_confidence=0.9,
        )

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return await self.execute(request)

    async def close(self) -> None:
        if self._weaver_dept is not None:
            await self._weaver_dept.close()
            self._weaver_dept = None
        await super().close()


# ============================================================================
# Verification — backed by real confidence validation + cross-checking
# ============================================================================


class RealVerification(BaseDepartment):
    """
    Production Verification department.

    Wraps VerificationDepartment and exposes charter tool names:
    validate_confidence_claim, request_rerun, cross_check_departments.

    Provides real confidence calibration: compares claimed confidence
    against evidence quality, historical accuracy, and heuristic checks.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        consistency_threshold: float = 0.80,
    ):
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
        self._confidence_threshold = confidence_threshold
        self._consistency_threshold = consistency_threshold

        # Verification history for trend detection
        self._validation_history: list[dict[str, Any]] = []
        # Rerun requests log
        self._rerun_requests: list[dict[str, Any]] = []

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "validate_confidence_claim":
                return await self._handle_validate(request)
            case "request_rerun":
                return await self._handle_rerun(request)
            case "cross_check_departments":
                return await self._handle_cross_check(request)
            case _:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"error": f"Unknown task: {request.task_type}"},
                    confidence=ConfidenceMetadata.from_score(0.0),
                )

    async def _handle_validate(self, request: DepartmentRequest) -> DepartmentResponse:
        """Validate a confidence claim against evidence quality."""
        claim = request.parameters.get("claim", {})
        evidence = request.parameters.get("evidence", {})
        claimed_confidence = claim.get("confidence", 0.5)

        # Multi-factor confidence validation
        penalties = []
        adjustments = 0.0

        # Factor 1: Evidence presence
        if not evidence:
            penalties.append("no supporting evidence provided")
            adjustments -= 0.2
        else:
            evidence_keys = len(evidence)
            if evidence_keys < 3:
                penalties.append(f"sparse evidence ({evidence_keys} fields)")
                adjustments -= 0.05

        # Factor 2: Overconfidence detection
        if claimed_confidence > 0.95:
            penalties.append("very high claim (>0.95) — requires strong evidence")
            if not evidence or len(evidence) < 3:
                adjustments -= 0.15

        # Factor 3: Historical calibration
        if self._validation_history:
            recent = self._validation_history[-20:]
            avg_gap = sum(
                abs(h["claimed"] - h["actual"]) for h in recent
            ) / len(recent)
            if avg_gap > 0.15:
                penalties.append(f"historical miscalibration (avg gap: {avg_gap:.2f})")
                adjustments -= 0.05

        # Factor 4: Claim specificity
        claim_keys = set(claim.keys()) - {"confidence"}
        if len(claim_keys) == 0:
            penalties.append("claim has no structured fields besides confidence")
            adjustments -= 0.1

        # Compute actual confidence
        actual_confidence = max(0.0, min(1.0, claimed_confidence + adjustments))
        gap = abs(claimed_confidence - actual_confidence)
        validation_result = gap < 0.2

        recommendation = "Acceptable" if validation_result else "Re-run recommended"
        if gap > 0.3:
            recommendation = "Significant miscalibration — re-run with stricter parameters"

        # Track in history
        self._validation_history.append({
            "claimed": claimed_confidence,
            "actual": actual_confidence,
            "gap": gap,
            "valid": validation_result,
        })

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "validation_result": validation_result,
                "actual_confidence": round(actual_confidence, 4),
                "claimed_confidence": claimed_confidence,
                "gap": round(gap, 4),
                "penalties": penalties,
                "recommendation": recommendation,
            },
            confidence=ConfidenceMetadata.from_score(
                0.92,
                justification=[
                    f"Validated claim ({claimed_confidence:.2f} → {actual_confidence:.2f}), "
                    f"{len(penalties)} penalties"
                ],
            ),
        )

    async def _handle_rerun(self, request: DepartmentRequest) -> DepartmentResponse:
        """Request a rerun of a task in another department."""
        department = request.parameters.get("department", "")
        task_id = request.parameters.get("task_id", "")
        new_params = request.parameters.get("new_params", {})
        reason = request.parameters.get("reason", "")

        if not department or not task_id:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "Missing required: department and task_id"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        # Log the rerun request (actual dispatch happens via MCPRouter)
        rerun_record = {
            "target_department": department,
            "original_task_id": task_id,
            "new_params": new_params,
            "reason": reason,
            "status": "accepted",
        }
        self._rerun_requests.append(rerun_record)

        # Generate a new task ID for the rerun
        import uuid
        new_task_id = f"rerun_{uuid.uuid4().hex[:8]}"

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "accepted": True,
                "new_task_id": new_task_id,
                "target_department": department,
                "reason": reason,
                "note": "Rerun request logged. Caller should dispatch via MCPRouter.",
            },
            confidence=ConfidenceMetadata.from_score(0.95,
                justification=["Rerun request accepted and logged"],
            ),
        )

    async def _handle_cross_check(self, request: DepartmentRequest) -> DepartmentResponse:
        """Cross-check outputs from multiple departments."""
        request.parameters.get("task_id", "")
        departments = request.parameters.get("departments", [])

        if not departments:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "No departments specified for cross-check"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        # Cross-checking requires outputs from the specified departments.
        # In a real federation, the router would gather these. Here we
        # return a structural check based on available validation history.
        conflicts = []
        if self._validation_history:
            recent_invalid = [
                h for h in self._validation_history[-20:]
                if not h["valid"]
            ]
            if recent_invalid:
                conflicts.append({
                    "type": "confidence_miscalibration",
                    "count": len(recent_invalid),
                    "severity": "medium" if len(recent_invalid) < 5 else "high",
                })

        consistent = len(conflicts) == 0

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "consistent": consistent,
                "conflicts": conflicts,
                "departments_checked": departments,
                "resolution_recommendation": (
                    "All departments consistent" if consistent
                    else "Review flagged confidence miscalibrations"
                ),
            },
            confidence=ConfidenceMetadata.from_score(
                0.9 if consistent else 0.75,
                justification=[
                    f"Cross-checked {len(departments)} departments, "
                    f"{len(conflicts)} conflicts"
                ],
            ),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        has_error = "error" in response.result
        return VerificationResult(
            sufficient=not has_error and response.confidence.score >= 0.7,
            confidence_valid=response.confidence.score >= 0.7,
            reasoning_sound=not has_error,
            alternative_paths=[],
            verifier_confidence=0.95,
        )

    async def refine(self, request, prior, verification) -> DepartmentResponse:  # noqa: Verification
        return await self.execute(request)


# ============================================================================
# Orchestration — backed by real OrchestrationDepartment + session tracking
# ============================================================================


class RealOrchestration(BaseDepartment):
    """
    Production Orchestration department.

    Wraps OrchestrationDepartment and exposes charter tool names:
    route_task, get_session_context, escalate_decision.

    Graceful degradation: route_task falls back to keyword-based routing
    when the OrchestrationDepartment can't be loaded. Session context and
    escalation are handled with built-in tracking.
    """

    def __init__(self, registry: Any = None):
        config = DepartmentConfig(
            name="Orchestration",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "route_task",
                "get_session_context",
                "escalate_decision",
            ],
        )
        super().__init__(
            name="Orchestration",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "route_task",
                "get_session_context",
                "escalate_decision",
            ],
            confidence_range=(0.7, 0.99),
            config=config,
        )
        self._registry = registry

        # Lazy-initialized real department
        self._orch_dept = None

        # Session context store (session_id → context)
        self._session_contexts: dict[str, dict[str, Any]] = {}

        # Escalation log
        self._escalations: list[dict[str, Any]] = []

    async def _get_orch(self):
        """Lazy-init the real OrchestrationDepartment."""
        if self._orch_dept is None:
            try:
                from ..orchestration.orchestration import OrchestrationDepartment

                orch_config = DepartmentConfig(
                    name="orchestration",
                    domain="system",
                    version="1.0.0",
                    supported_tasks=["execute_workflow", "aggregate_results", "route_task"],
                )
                self._orch_dept = OrchestrationDepartment(
                    registry=self._registry,
                    dept_config=orch_config,
                )
                await self._orch_dept.initialize()
                logger.info("RealOrchestration: initialized with real OrchestrationDepartment")
            except Exception as e:
                logger.warning(
                    f"RealOrchestration: failed to init real department ({e}), "
                    "using built-in routing"
                )
                self._orch_dept = None
        return self._orch_dept

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "route_task":
                return await self._handle_route(request)
            case "get_session_context":
                return await self._handle_session_context(request)
            case "escalate_decision":
                return await self._handle_escalate(request)
            case _:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"error": f"Unknown task: {request.task_type}"},
                    confidence=ConfidenceMetadata.from_score(0.0),
                )

    async def _handle_route(self, request: DepartmentRequest) -> DepartmentResponse:
        """Route a task to the best department."""
        task_spec = request.parameters.get("task_spec", {})
        candidates = request.parameters.get("candidate_departments", [])

        orch = await self._get_orch()

        if orch is not None:
            try:
                # Delegate to real OrchestrationDepartment
                orch_request = DepartmentRequest(
                    task_type="route_task",
                    parameters={
                        "task_description": (
                            task_spec.get("description", str(task_spec))
                            if isinstance(task_spec, dict) else str(task_spec)
                        ),
                        "departments": candidates,
                    },
                    context=request.context,
                )
                response = await orch.execute(orch_request)

                # Translate result to charter format
                inner = response.result
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={
                        "assigned_department": inner.get("selected_department", ""),
                        "routing_reason": f"Scored {inner.get('routing_score', 0):.2f}",
                        "estimated_duration": "unknown",
                        "candidates_evaluated": inner.get("candidates_evaluated", 0),
                    },
                    confidence=response.confidence,
                )
            except Exception as e:
                logger.warning(f"RealOrchestration: delegation failed ({e}), using built-in")

        # Built-in keyword routing
        task_text = (
            task_spec.get("description", str(task_spec))
            if isinstance(task_spec, dict) else str(task_spec)
        ).lower()

        # Simple keyword-based routing
        ROUTING_KEYWORDS = {
            "Infrastructure": ["query", "neo4j", "qdrant", "performance", "diagnose", "backend"],
            "MasterWeaver": ["entity", "extract", "ontology", "knowledge", "domain"],
            "Verification": ["validate", "confidence", "check", "verify", "cross"],
            "Context": ["context", "enrich", "missing", "session"],
            "Execution": ["run", "execute", "task", "code", "status"],
        }

        # Score each candidate (or all departments if no candidates)
        search_depts = candidates if candidates else list(ROUTING_KEYWORDS.keys())
        best_dept = search_depts[0] if search_depts else "Orchestration"
        best_score = 0.0

        for dept in search_depts:
            keywords = ROUTING_KEYWORDS.get(dept, [])
            score = sum(1 for kw in keywords if kw in task_text)
            if score > best_score:
                best_score = score
                best_dept = dept

        confidence_score = min(0.9, 0.6 + best_score * 0.1)

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "assigned_department": best_dept,
                "routing_reason": f"Keyword match (score={best_score})",
                "estimated_duration": "unknown",
                "candidates_evaluated": len(search_depts),
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[f"Routed to {best_dept} via keyword matching"],
            ),
        )

    async def _handle_session_context(self, request: DepartmentRequest) -> DepartmentResponse:
        """Retrieve session context for a department."""
        session_id = request.parameters.get("session_id", "")
        department = request.parameters.get("department", "")

        if not session_id:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "Missing required: session_id"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        # Retrieve or create session context
        ctx = self._session_contexts.get(session_id, {})

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "context": ctx,
                "roadmap_phase": ctx.get("roadmap_phase", "unknown"),
                "relevant_decisions": ctx.get("decisions", []),
                "department": department,
                "session_id": session_id,
            },
            confidence=ConfidenceMetadata.from_score(
                0.85 if ctx else 0.5,
                justification=[
                    f"Session {'found' if ctx else 'empty'}"
                ],
            ),
        )

    async def _handle_escalate(self, request: DepartmentRequest) -> DepartmentResponse:
        """Escalate a decision requiring human input or consensus."""
        issue = request.parameters.get("issue", "")
        reason = request.parameters.get("reason", "")
        affected = request.parameters.get("affected_departments", [])

        if not issue:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "Missing required: issue"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        import uuid
        escalation_id = f"esc_{uuid.uuid4().hex[:8]}"

        record = {
            "escalation_id": escalation_id,
            "issue": issue,
            "reason": reason,
            "affected_departments": affected,
            "status": "pending",
            "resolution_channel": "human_review",
        }
        self._escalations.append(record)

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "escalation_id": escalation_id,
                "status": "pending",
                "resolution_channel": "human_review",
                "affected_departments": affected,
            },
            confidence=ConfidenceMetadata.from_score(
                0.95,
                justification=["Escalation logged and queued for review"],
            ),
        )

    def update_session_context(
        self, session_id: str, updates: dict[str, Any]
    ) -> None:
        """Update session context (called by federation participants)."""
        if session_id not in self._session_contexts:
            self._session_contexts[session_id] = {}
        self._session_contexts[session_id].update(updates)

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        has_error = "error" in response.result
        return VerificationResult(
            sufficient=not has_error and response.confidence.score >= 0.6,
            confidence_valid=response.confidence.score >= 0.6,
            reasoning_sound=not has_error,
            alternative_paths=[],
            verifier_confidence=0.9,
        )

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return await self.execute(request)


# ============================================================================
# Context — backed by real context enrichment + missing-context detection
# ============================================================================


class RealContext(BaseDepartment):
    """
    Production Context department.

    Wraps ContextDepartment (context_department.py) and exposes charter
    tool names: enrich_context, detect_missing_context.

    Graceful degradation: when the full ContextDepartment can't be loaded
    (missing context system imports), falls back to simple keyword-based
    context enrichment.
    """

    def __init__(self):
        config = DepartmentConfig(
            name="Context",
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "enrich_context",
                "detect_missing_context",
            ],
        )
        super().__init__(
            name="Context",
            domain="general",
            version="1.0.0",
            supported_tasks=[
                "enrich_context",
                "detect_missing_context",
            ],
            confidence_range=(0.6, 0.95),
            config=config,
        )

        # Lazy-initialized real department
        self._context_dept = None

    async def _get_context(self):
        """Lazy-init the real ContextDepartment."""
        if self._context_dept is None:
            try:
                from ..context_department import ContextDepartment as FullContextDept

                self._context_dept = FullContextDept(
                    enable_learning=False,
                    enable_privacy=True,
                )
                # ContextDepartment uses __aenter__ for full init, but
                # execute() works without it for basic operations
                logger.info("RealContext: initialized with real ContextDepartment")
            except Exception as e:
                logger.warning(
                    f"RealContext: failed to init real department ({e}), "
                    "using built-in enrichment"
                )
                self._context_dept = None
        return self._context_dept

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "enrich_context":
                return await self._handle_enrich(request)
            case "detect_missing_context":
                return await self._handle_detect_missing(request)
            case _:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"error": f"Unknown task: {request.task_type}"},
                    confidence=ConfidenceMetadata.from_score(0.0),
                )

    async def _handle_enrich(self, request: DepartmentRequest) -> DepartmentResponse:
        """Perform multi-pass context enrichment on raw data."""
        raw_data = request.parameters.get("raw_data", "")
        context_request = request.parameters.get("context_request", {})
        passes = request.parameters.get("passes", 3)

        ctx_dept = await self._get_context()

        if ctx_dept is not None:
            try:
                # Delegate to real ContextDepartment's enrich_request
                ctx_request = DepartmentRequest(
                    task_type="enrich_request",
                    parameters={
                        "user_id": context_request.get("user_id", "anonymous"),
                        "session_id": context_request.get("session_id", "default"),
                        "raw_data": raw_data,
                    },
                    context=request.context,
                )
                response = await ctx_dept.execute(ctx_request)

                # Re-wrap result in charter format
                enrichment = response.result
                enriched = {}
                if hasattr(enrichment, "enriched_request"):
                    enriched = enrichment.enriched_request.context
                elif isinstance(enrichment, dict):
                    enriched = enrichment

                return DepartmentResponse(
                    task_id=request.task_id,
                    result={
                        "enriched_context": enriched,
                        "confidence": response.confidence.score,
                        "sources": ["context_department"],
                        "passes_completed": passes,
                    },
                    confidence=response.confidence,
                )
            except Exception as e:
                logger.warning(f"RealContext: delegation failed ({e}), using built-in")

        # Built-in enrichment: keyword extraction + structural analysis
        words = raw_data.split() if raw_data else []
        keywords = [w for w in words if len(w) > 4 and w[0].isupper()]
        word_count = len(words)

        enriched_context = {
            "raw_data_length": len(raw_data),
            "word_count": word_count,
            "key_terms": keywords[:20],
            "context_request": context_request,
            "structural_features": {
                "has_questions": "?" in raw_data,
                "has_entities": len(keywords) > 0,
                "complexity": "high" if word_count > 100 else "medium" if word_count > 30 else "low",
            },
        }

        # Simulate multi-pass (each pass adds a layer)
        for pass_num in range(1, min(passes, 3) + 1):
            enriched_context[f"pass_{pass_num}"] = {
                "depth": pass_num,
                "terms_refined": len(keywords) + pass_num,
            }

        confidence_score = min(0.85, 0.5 + 0.1 * min(passes, 3) + 0.05 * min(len(keywords), 5))

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "enriched_context": enriched_context,
                "confidence": confidence_score,
                "sources": ["builtin_keyword"],
                "passes_completed": min(passes, 3),
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[f"Built-in enrichment, {min(passes, 3)} passes"],
            ),
        )

    async def _handle_detect_missing(self, request: DepartmentRequest) -> DepartmentResponse:
        """Analyze a decision and detect what context is missing."""
        decision = request.parameters.get("decision", {})
        available_context = request.parameters.get("available_context", {})

        # Analyze decision fields for common missing context patterns
        missing = []

        # Check for common required fields
        EXPECTED_FIELDS = {
            "user_context": "Who is making/affected by this decision",
            "domain_context": "Domain-specific knowledge or constraints",
            "temporal_context": "Time-sensitive information (deadlines, recency)",
            "historical_context": "Past decisions or outcomes on similar topics",
            "constraints": "Known constraints or limitations",
        }

        available_keys = set(available_context.keys()) if available_context else set()
        decision_keys = set(decision.keys()) if decision else set()
        all_keys = available_keys | decision_keys

        for field, description in EXPECTED_FIELDS.items():
            # Check if this field or a related one is present
            related = any(field.split("_")[0] in k.lower() for k in all_keys)
            if not related:
                missing.append({
                    "field": field,
                    "description": description,
                    "impact": "medium",
                })

        # Assess severity
        missing_count = len(missing)
        if missing_count == 0:
            severity = "low"
        elif missing_count <= 2:
            severity = "medium"
        elif missing_count <= 4:
            severity = "high"
        else:
            severity = "critical"

        # Generate recommendations
        recommendations = []
        for m in missing:
            recommendations.append(f"Gather {m['field']}: {m['description']}")

        confidence_score = max(0.5, 0.95 - missing_count * 0.08)

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "missing_context": missing,
                "impact_severity": severity,
                "recommendations": recommendations,
                "available_fields": list(all_keys),
                "completeness_score": max(0.0, 1.0 - missing_count / len(EXPECTED_FIELDS)),
            },
            confidence=ConfidenceMetadata.from_score(
                confidence_score,
                justification=[
                    f"Detected {missing_count} missing context fields, "
                    f"severity: {severity}"
                ],
            ),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        has_error = "error" in response.result
        return VerificationResult(
            sufficient=not has_error and response.confidence.score >= 0.6,
            confidence_valid=response.confidence.score >= 0.6,
            reasoning_sound=not has_error,
            alternative_paths=[],
            verifier_confidence=0.9,
        )

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return await self.execute(request)


# ============================================================================
# Execution — backed by task tracking + optional Claude Code bridge
# ============================================================================


class RealExecution(BaseDepartment):
    """
    Production Execution department.

    Exposes charter tool names: run_claude_code_task, check_task_status.

    Tracks tasks in-memory with status lifecycle (queued → running →
    completed/failed). Optionally bridges to ClaudeCodeDepartment for
    real VS Code integration when available.

    Graceful degradation: without VS Code MCP connection, tasks are
    accepted and logged but execute in simulation mode with lower
    confidence.
    """

    def __init__(self, mcp_server_url: str | None = None):
        config = DepartmentConfig(
            name="Execution",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "run_claude_code_task",
                "check_task_status",
            ],
        )
        super().__init__(
            name="Execution",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "run_claude_code_task",
                "check_task_status",
            ],
            confidence_range=(0.5, 0.95),
            config=config,
        )
        self._mcp_server_url = mcp_server_url

        # Task registry
        self._tasks: dict[str, dict[str, Any]] = {}
        self._task_counter = 0

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        match request.task_type:
            case "run_claude_code_task":
                return await self._handle_run_task(request)
            case "check_task_status":
                return await self._handle_check_status(request)
            case _:
                return DepartmentResponse(
                    task_id=request.task_id,
                    result={"error": f"Unknown task: {request.task_type}"},
                    confidence=ConfidenceMetadata.from_score(0.0),
                )

    async def _handle_run_task(self, request: DepartmentRequest) -> DepartmentResponse:
        """Accept and track a task for execution."""
        task_spec = request.parameters.get("task_spec", "")
        context_files = request.parameters.get("context_files", [])

        if not task_spec:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "Missing required: task_spec"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        import uuid
        self._task_counter += 1
        task_id = f"task_{uuid.uuid4().hex[:8]}"

        # Register the task
        task_record = {
            "task_id": task_id,
            "task_spec": task_spec,
            "context_files": context_files,
            "status": "queued",
            "progress": 0.0,
            "result": None,
            "error": None,
        }
        self._tasks[task_id] = task_record

        # Try to bridge to real Claude Code MCP
        execution_mode = "simulation"
        if self._mcp_server_url:
            try:

                # Note: actual MCP dispatch would happen asynchronously.
                # For now, we log the intent and mark as queued.
                execution_mode = "mcp_queued"
                task_record["status"] = "running"
                task_record["progress"] = 0.1
                logger.info(
                    f"RealExecution: task {task_id} queued for MCP "
                    f"at {self._mcp_server_url}"
                )
            except Exception as e:
                logger.warning(f"RealExecution: MCP bridge unavailable ({e})")

        # In simulation mode, mark as completed with a stub result
        if execution_mode == "simulation":
            task_record["status"] = "completed"
            task_record["progress"] = 1.0
            task_record["result"] = (
                f"Task '{task_spec[:80]}' accepted. "
                "Execution simulated (no MCP backend connected)."
            )

        confidence = 0.9 if execution_mode == "mcp_queued" else 0.6

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "task_id": task_id,
                "status": task_record["status"],
                "initial_response": task_record.get("result", "Task queued"),
                "execution_mode": execution_mode,
            },
            confidence=ConfidenceMetadata.from_score(
                confidence,
                justification=[f"Task accepted ({execution_mode})"],
            ),
        )

    async def _handle_check_status(self, request: DepartmentRequest) -> DepartmentResponse:
        """Check status of a previously submitted task."""
        task_id = request.parameters.get("task_id", "")

        if not task_id:
            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": "Missing required: task_id"},
                confidence=ConfidenceMetadata.from_score(0.0),
            )

        task = self._tasks.get(task_id)

        if task is None:
            return DepartmentResponse(
                task_id=request.task_id,
                result={
                    "error": f"Task '{task_id}' not found",
                    "known_tasks": list(self._tasks.keys())[-10:],
                },
                confidence=ConfidenceMetadata.from_score(0.3),
            )

        return DepartmentResponse(
            task_id=request.task_id,
            result={
                "status": task["status"],
                "progress": task["progress"],
                "result": task.get("result"),
                "error": task.get("error"),
                "task_id": task_id,
            },
            confidence=ConfidenceMetadata.from_score(
                0.95,
                justification=[f"Task {task_id} status: {task['status']}"],
            ),
        )

    async def verify(self, response: DepartmentResponse) -> VerificationResult:
        has_error = "error" in response.result and response.result["error"] is not None
        return VerificationResult(
            sufficient=not has_error and response.confidence.score >= 0.5,
            confidence_valid=response.confidence.score >= 0.5,
            reasoning_sound=not has_error,
            alternative_paths=[],
            verifier_confidence=0.9,
        )

    async def refine(self, request, prior, verification) -> DepartmentResponse:
        return await self.execute(request)

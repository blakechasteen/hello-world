#!/usr/bin/env python3
"""
Infrastructure Department - Low-level infrastructure services.

Provides shared infrastructure for all departments:
- Zero-copy embedding storage
- Neo4j + Qdrant integration
- Performance diagnostics
- Resource management

Design Philosophy:
"Share infrastructure, scale horizontally."

Supported Tasks:
- "store_embeddings": Store embeddings in zero-copy store
- "retrieve_embeddings": Retrieve embeddings from store
- "search_embeddings": Search similar embeddings
- "diagnose_performance": Run performance diagnostics
- "check_backends": Check Neo4j + Qdrant health
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..base import BaseDepartment
from ..protocol import (
    ConfidenceMetadata,
    DepartmentConfig,
    DepartmentRequest,
    DepartmentResponse,
    VerificationResult,
)
from .zero_copy import MatryoshkaEmbeddingStore

logger = logging.getLogger(__name__)


# ============================================================================
# Infrastructure Department
# ============================================================================

class InfrastructureDepartment(BaseDepartment):
    """
    Infrastructure Department - Low-level shared services.

    Provides infrastructure services that all departments can use:
    1. Zero-copy embedding storage (no memory duplication)
    2. Performance diagnostics (latency profiling)
    3. Backend health checks (Neo4j, Qdrant)
    4. Resource management (connection pooling)

    Example:

        # Create department
        async with InfrastructureDepartment(storage_dir="./data") as dept:
            # Store embeddings
            request = DepartmentRequest(
                task_id="store_001",
                task_type="store_embeddings",
                parameters={
                    "embeddings": [
                        {"id": "doc_1", "vector": [0.1, 0.2, ...], "tags": ["context"]},
                        {"id": "doc_2", "vector": [0.3, 0.4, ...], "tags": ["beekeeping"]}
                    ],
                    "scale": 384
                }
            )

            response = await dept.execute(request)

            # Search similar
            search_request = DepartmentRequest(
                task_id="search_001",
                task_type="search_embeddings",
                parameters={
                    "query_vector": [0.15, 0.25, ...],
                    "k": 10,
                    "scale": 384
                }
            )

            results = await dept.execute(search_request)
    """

    def __init__(
        self,
        storage_dir: Path | None = None,
        scales: list[int] = [96, 192, 384],
        max_embeddings: int = 100000,
        enable_neo4j: bool = False,
        enable_qdrant: bool = False,
        neo4j_uri: str | None = None,
        qdrant_host: str | None = None,
        dept_config: DepartmentConfig | None = None
    ):
        """
        Initialize Infrastructure Department.

        Args:
            storage_dir: Directory for zero-copy embedding storage
            scales: Matryoshka scales to support (default: [96, 192, 384])
            max_embeddings: Maximum embeddings to store
            enable_neo4j: Enable Neo4j integration
            enable_qdrant: Enable Qdrant integration
            neo4j_uri: Neo4j connection URI
            qdrant_host: Qdrant host
            dept_config: Department configuration
        """
        # Initialize base department
        super().__init__(
            name="infrastructure",
            domain="system",
            version="1.0.0",
            supported_tasks=[
                "store_embeddings",
                "retrieve_embeddings",
                "search_embeddings",
                "diagnose_performance",
                "check_backends"
            ],
            confidence_range=(0.80, 0.99),
            config=dept_config or DepartmentConfig(name="infrastructure", domain="infrastructure")
        )

        # Storage configuration
        if storage_dir is None:
            storage_dir = Path.cwd() / "hololoom_data" / "embeddings"

        self.storage_dir = Path(storage_dir)
        self.scales = scales
        self.max_embeddings = max_embeddings

        # Embedding store (lazy initialization)
        self._embedding_store: MatryoshkaEmbeddingStore | None = None

        # Backend configuration
        self.enable_neo4j = enable_neo4j
        self.enable_qdrant = enable_qdrant
        self.neo4j_uri = neo4j_uri or "bolt://localhost:7687"
        self.qdrant_host = qdrant_host or "localhost:6333"

        # Backend clients (lazy initialization)
        self._neo4j_driver: Any | None = None
        self._qdrant_client: Any | None = None

        # Performance tracking
        self._operation_latencies: dict[str, list[float]] = {}
        self._backend_health: dict[str, bool] = {
            "embedding_store": False,
            "neo4j": False,
            "qdrant": False
        }

        logger.info(
            f"Infrastructure Department initialized "
            f"(storage: {storage_dir}, scales: {scales})"
        )

    # ========================================================================
    # Lifecycle Management
    # ========================================================================

    async def initialize(self) -> None:
        """Initialize infrastructure services."""
        if self._initialized:
            return

        # Initialize embedding store
        logger.info("Initializing zero-copy embedding store")
        self._embedding_store = MatryoshkaEmbeddingStore(
            storage_dir=self.storage_dir,
            scales=self.scales,
            max_embeddings=self.max_embeddings
        )
        await self._embedding_store.initialize()
        self._backend_health["embedding_store"] = True

        # Initialize Neo4j (if enabled)
        if self.enable_neo4j:
            await self._initialize_neo4j()

        # Initialize Qdrant (if enabled)
        if self.enable_qdrant:
            await self._initialize_qdrant()

        await super().initialize()

        logger.info("Infrastructure Department initialized successfully")

    async def close(self) -> None:
        """Cleanup infrastructure resources."""
        # Close embedding store
        if self._embedding_store is not None:
            await self._embedding_store.close()
            self._embedding_store = None

        # Close Neo4j
        if self._neo4j_driver is not None:
            self._neo4j_driver.close()
            self._neo4j_driver = None

        # Close Qdrant
        if self._qdrant_client is not None:
            # Qdrant client doesn't require explicit close
            self._qdrant_client = None

        await super().close()

        logger.info("Infrastructure Department closed")

    # ========================================================================
    # Core Department Methods
    # ========================================================================

    async def execute(
        self,
        request: DepartmentRequest
    ) -> DepartmentResponse:
        """
        Execute an infrastructure task.

        Supported task types:
        - "store_embeddings": Store embeddings in zero-copy store
        - "retrieve_embeddings": Retrieve embeddings from store
        - "search_embeddings": Search similar embeddings
        - "diagnose_performance": Run performance diagnostics
        - "check_backends": Check backend health

        Args:
            request: Standardized request with task details

        Returns:
            DepartmentResponse with result and confidence
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        # Validate task type
        if request.task_type not in self.supported_tasks:
            raise ValueError(
                f"Unsupported task type: {request.task_type}. "
                f"Supported: {self.supported_tasks}"
            )

        # Route to specific handler
        try:
            if request.task_type == "store_embeddings":
                result, confidence = await self._store_embeddings(request)
            elif request.task_type == "retrieve_embeddings":
                result, confidence = await self._retrieve_embeddings(request)
            elif request.task_type == "search_embeddings":
                result, confidence = await self._search_embeddings(request)
            elif request.task_type == "diagnose_performance":
                result, confidence = await self._diagnose_performance(request)
            elif request.task_type == "check_backends":
                result, confidence = await self._check_backends(request)
            else:
                raise ValueError(f"Unknown task type: {request.task_type}")

            # Build learning signals
            learning_signals = {
                "task_type": request.task_type,
                "outcome": "success",
                "confidence": confidence.score
            }

            # Build response
            response = DepartmentResponse(
                task_id=request.task_id,
                result=result,
                confidence=confidence,
                detail_level=request.context_preference,
                reasoning=self._build_reasoning(result, request) if request.context_preference != "minimal" else None,
                learning_signals=learning_signals,
                session_state=await self._build_session_state(request.session_id, result),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )

            # Record success
            latency_ms = response.execution_time_ms
            self._record_request(success=True, latency_ms=latency_ms, confidence=confidence.score)
            self._record_operation_latency(request.task_type, latency_ms)

            return response

        except Exception as e:
            # Record error
            self._record_error(e)

            # Return error response
            error_confidence = ConfidenceMetadata.from_score(
                0.0,
                justification=[],
                uncertainty_sources=[f"Execution failed: {str(e)}"]
            )

            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._record_request(success=False, latency_ms=latency_ms, confidence=0.0)

            return DepartmentResponse(
                task_id=request.task_id,
                result={"error": str(e)},
                confidence=error_confidence,
                execution_time_ms=latency_ms
            )

    async def verify(
        self,
        response: DepartmentResponse
    ) -> VerificationResult:
        """
        Verify infrastructure operation quality.

        Checks:
        1. Confidence validity (score ≥ 0.80 for infrastructure)
        2. Operation success (no errors)
        3. Backend health (all backends healthy)

        Args:
            response: Response to verify

        Returns:
            VerificationResult with pass/fail checks
        """
        # Check 1: Confidence threshold
        confidence_threshold = 0.80
        confidence_valid = response.confidence.score >= confidence_threshold

        # Check 2: Operation success
        result = response.result
        has_error = "error" in result
        operation_success = not has_error

        # Check 3: Backend health (only check enabled backends)
        enabled_backends = {"embedding_store": True}
        if self.enable_neo4j:
            enabled_backends["neo4j"] = True
        if self.enable_qdrant:
            enabled_backends["qdrant"] = True

        all_backends_healthy = all(
            self._backend_health.get(backend, False)
            for backend in enabled_backends.keys()
        )

        # Determine if sufficient
        sufficient = confidence_valid and operation_success and all_backends_healthy

        # Generate refinement suggestions if insufficient
        refinement_suggestions = None
        if not sufficient:
            refinement_suggestions = {}

            if not confidence_valid:
                refinement_suggestions["retry"] = True
                refinement_suggestions["reason"] = "Low confidence - retry operation"

            if has_error:
                refinement_suggestions["check_error"] = True
                refinement_suggestions["reason"] = f"Operation failed: {result.get('error')}"

            if not all_backends_healthy:
                unhealthy = [k for k, v in self._backend_health.items() if not v]
                refinement_suggestions["check_backends"] = unhealthy
                refinement_suggestions["reason"] = f"Unhealthy backends: {unhealthy}"

        # Identify alternative paths
        alternative_paths = []
        if not sufficient:
            if not all_backends_healthy:
                alternative_paths.append("fallback_to_inmemory")
            if has_error:
                alternative_paths.append("retry_with_backoff")

        return VerificationResult(
            sufficient=sufficient,
            confidence_valid=confidence_valid,
            reasoning_sound=operation_success,
            alternative_paths=alternative_paths,
            refinement_suggestions=refinement_suggestions,
            escalation_needed=response.confidence.score < 0.5,
            escalation_reason="Very low confidence" if response.confidence.score < 0.5 else None,
            verifier_confidence=0.95  # High confidence in infrastructure verification
        )

    async def refine(
        self,
        request: DepartmentRequest,
        prior_response: DepartmentResponse,
        verification: VerificationResult
    ) -> DepartmentResponse:
        """
        Refine infrastructure operation.

        Refinement strategies:
        1. Retry with exponential backoff
        2. Fallback to in-memory if backends unavailable
        3. Clear cache and retry

        Args:
            request: Original request
            prior_response: Previous insufficient response
            verification: Verification results with suggestions

        Returns:
            Refined DepartmentResponse
        """
        logger.info(
            f"Refining infrastructure operation for task {request.task_id} "
            f"(prior confidence: {prior_response.confidence.score:.2f})"
        )

        suggestions = verification.refinement_suggestions or {}

        # Apply refinement strategy
        if suggestions.get("retry"):
            # Retry with exponential backoff
            await asyncio.sleep(0.1)  # 100ms backoff
            return await self.execute(request)

        elif suggestions.get("check_backends"):
            # Re-check backend health
            await self._check_all_backends()
            return await self.execute(request)

        else:
            # Default: just retry
            return await self.execute(request)

    # ========================================================================
    # Task Handlers
    # ========================================================================

    async def _store_embeddings(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Store embeddings in zero-copy store."""
        embeddings = request.parameters.get("embeddings", [])
        scale = request.parameters.get("scale", 384)

        if not embeddings:
            raise ValueError("Missing required parameter: 'embeddings'")

        if scale not in self.scales:
            raise ValueError(f"Unsupported scale: {scale}. Supported: {self.scales}")

        # Store each embedding
        stored_count = 0
        failed_count = 0

        for emb_data in embeddings:
            id = emb_data.get("id")
            vector = np.array(emb_data.get("vector"))
            tags = emb_data.get("tags", [])

            try:
                success = await self._embedding_store.stores[scale].add_embedding(
                    id, vector, tags
                )

                if success:
                    stored_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                logger.error(f"Failed to store embedding {id}: {e}")
                failed_count += 1

        # Calculate confidence
        total = len(embeddings)
        success_rate = stored_count / total if total > 0 else 0.0

        confidence = ConfidenceMetadata.from_score(
            success_rate,
            justification=[f"Stored {stored_count}/{total} embeddings"],
            uncertainty_sources=[f"{failed_count} failures"] if failed_count > 0 else []
        )

        result = {
            "stored_count": stored_count,
            "failed_count": failed_count,
            "total_count": total,
            "success_rate": success_rate,
            "scale": scale
        }

        return result, confidence

    async def _retrieve_embeddings(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Retrieve embeddings from zero-copy store."""
        ids = request.parameters.get("ids", [])
        scale = request.parameters.get("scale", 384)

        if not ids:
            raise ValueError("Missing required parameter: 'ids'")

        if scale not in self.scales:
            raise ValueError(f"Unsupported scale: {scale}. Supported: {self.scales}")

        # Retrieve embeddings
        embeddings = []
        found_count = 0
        missing_count = 0

        for id in ids:
            view = await self._embedding_store.stores[scale].get_embedding(id)

            if view:
                embeddings.append({
                    "id": view.id,
                    "vector": view.vector.tolist(),
                    "metadata": {
                        "dimension": view.metadata.dimension,
                        "tags": view.metadata.tags,
                        "access_count": view.metadata.access_count
                    }
                })
                found_count += 1
            else:
                missing_count += 1

        # Calculate confidence
        total = len(ids)
        found_rate = found_count / total if total > 0 else 0.0

        confidence = ConfidenceMetadata.from_score(
            found_rate,
            justification=[f"Found {found_count}/{total} embeddings"],
            uncertainty_sources=[f"{missing_count} missing"] if missing_count > 0 else []
        )

        result = {
            "embeddings": embeddings,
            "found_count": found_count,
            "missing_count": missing_count,
            "total_count": total,
            "found_rate": found_rate,
            "scale": scale
        }

        return result, confidence

    async def _search_embeddings(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Search similar embeddings."""
        query_vector = request.parameters.get("query_vector")
        k = request.parameters.get("k", 10)
        scale = request.parameters.get("scale", 384)
        tags = request.parameters.get("tags")

        if query_vector is None:
            raise ValueError("Missing required parameter: 'query_vector'")

        query_vector = np.array(query_vector)

        if scale not in self.scales:
            raise ValueError(f"Unsupported scale: {scale}. Supported: {self.scales}")

        # Search similar
        results = await self._embedding_store.search_similar(
            query_vector,
            k=k,
            scale=scale,
            tags=tags
        )

        # Build result
        search_results = [
            {
                "id": id,
                "similarity": similarity,
                "metadata": {
                    "dimension": view.metadata.dimension,
                    "tags": view.metadata.tags
                }
            }
            for id, similarity, view in results
        ]

        # Calculate confidence (based on top similarity)
        if results:
            top_similarity = results[0][1]
            confidence_score = min(top_similarity, 1.0)
        else:
            confidence_score = 0.3

        confidence = ConfidenceMetadata.from_score(
            confidence_score,
            justification=[f"Found {len(results)} similar embeddings"],
            uncertainty_sources=[] if results else ["No results found"]
        )

        result = {
            "results": search_results,
            "result_count": len(results),
            "k": k,
            "scale": scale
        }

        return result, confidence

    async def _diagnose_performance(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Run performance diagnostics."""
        # Get embedding store stats
        store_stats = self._embedding_store.get_stats_all_scales()

        # Get operation latencies
        latency_stats = {
            task_type: {
                "count": len(latencies),
                "mean_ms": np.mean(latencies) if latencies else 0.0,
                "p50_ms": np.percentile(latencies, 50) if latencies else 0.0,
                "p95_ms": np.percentile(latencies, 95) if latencies else 0.0,
                "p99_ms": np.percentile(latencies, 99) if latencies else 0.0
            }
            for task_type, latencies in self._operation_latencies.items()
        }

        # Get backend health
        backend_health = self._backend_health.copy()

        # Calculate overall health score
        healthy_count = sum(1 for v in backend_health.values() if v)
        total_count = len(backend_health)
        health_score = healthy_count / total_count if total_count > 0 else 0.0

        confidence = ConfidenceMetadata.from_score(
            health_score,
            justification=[f"{healthy_count}/{total_count} backends healthy"],
            uncertainty_sources=[]
        )

        result = {
            "store_stats": store_stats,
            "latency_stats": latency_stats,
            "backend_health": backend_health,
            "health_score": health_score
        }

        return result, confidence

    async def _check_backends(
        self,
        request: DepartmentRequest
    ) -> tuple[dict[str, Any], ConfidenceMetadata]:
        """Check backend health."""
        await self._check_all_backends()

        # Calculate health score
        healthy_count = sum(1 for v in self._backend_health.values() if v)
        total_count = len(self._backend_health)
        health_score = healthy_count / total_count if total_count > 0 else 0.0

        confidence = ConfidenceMetadata.from_score(
            health_score,
            justification=[f"{healthy_count}/{total_count} backends healthy"],
            uncertainty_sources=[]
        )

        result = {
            "backend_health": self._backend_health.copy(),
            "healthy_count": healthy_count,
            "total_count": total_count,
            "health_score": health_score
        }

        return result, confidence

    # ========================================================================
    # Backend Initialization and Health Checks
    # ========================================================================

    async def _initialize_neo4j(self) -> None:
        """Initialize Neo4j driver."""
        try:
            from neo4j import GraphDatabase

            self._neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=("neo4j", "password")  # TODO: Make configurable
            )

            # Test connection
            with self._neo4j_driver.session() as session:
                result = session.run("RETURN 1")
                result.single()

            self._backend_health["neo4j"] = True
            logger.info("Neo4j initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Neo4j: {e}")
            self._backend_health["neo4j"] = False

    async def _initialize_qdrant(self) -> None:
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient

            self._qdrant_client = QdrantClient(host=self.qdrant_host)

            # Test connection
            collections = self._qdrant_client.get_collections()

            self._backend_health["qdrant"] = True
            logger.info("Qdrant initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Qdrant: {e}")
            self._backend_health["qdrant"] = False

    async def _check_all_backends(self) -> None:
        """Check health of all backends."""
        # Check embedding store
        if self._embedding_store:
            self._backend_health["embedding_store"] = True
        else:
            self._backend_health["embedding_store"] = False

        # Check Neo4j
        if self._neo4j_driver:
            try:
                with self._neo4j_driver.session() as session:
                    result = session.run("RETURN 1")
                    result.single()
                self._backend_health["neo4j"] = True
            except Exception as e:
                logger.warning(f"Neo4j health check failed: {e}")
                self._backend_health["neo4j"] = False

        # Check Qdrant
        if self._qdrant_client:
            try:
                collections = self._qdrant_client.get_collections()
                self._backend_health["qdrant"] = True
            except Exception as e:
                logger.warning(f"Qdrant health check failed: {e}")
                self._backend_health["qdrant"] = False

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _record_operation_latency(self, task_type: str, latency_ms: float) -> None:
        """Record operation latency for diagnostics."""
        if task_type not in self._operation_latencies:
            self._operation_latencies[task_type] = []

        self._operation_latencies[task_type].append(latency_ms)

        # Keep only recent latencies (last 1000)
        if len(self._operation_latencies[task_type]) > 1000:
            self._operation_latencies[task_type] = self._operation_latencies[task_type][-1000:]

    def _build_reasoning(
        self,
        result: dict[str, Any],
        request: DepartmentRequest
    ) -> dict[str, Any]:
        """Build reasoning dict for response."""
        return {
            "task_type": request.task_type,
            "backend_health": self._backend_health.copy()
        }

    async def _build_session_state(
        self,
        session_id: str | None,
        result: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build session state update."""
        if not session_id:
            return None

        return {
            "last_operation": result
        }

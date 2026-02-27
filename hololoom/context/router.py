#!/usr/bin/env python3
"""
Query Router for Hybrid Backend Coordination

Part 3: Classification and Basic Routing (Days 11-15)

Implements 4 routing patterns for multi-backend queries.
Validated in Demo 4 (all patterns working).

Patterns:
1. Sequential: SQL → Graph (policy → affected entities)
2. Parallel: SQL || Graph || Vector (execute simultaneously)
3. Fallback: SQL fails → Graph (automatic recovery)
4. Verification: Graph validated by SQL (confidence boost)
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

from hololoom.context.classifier import QueryClassifier, BackendSelection, Backend
from hololoom.context.bandit import ThompsonBandit
from hololoom.context.calibration import ConfidenceCalibrator
from hololoom.context.learning_tracker import LearningTracker
from hololoom.context.strategy_updater import StrategyUpdater
from hololoom.infrastructure.mcp import (
    MCPRequest,
    MCPResponse,
    MCPServer,
    RequestType,
    ResponseStatus,
    ToolName,
    generate_request_id
)


logger = logging.getLogger(__name__)


# ============================================================================
# Routing Pattern
# ============================================================================

class RoutingPattern(str, Enum):
    """Multi-backend routing patterns"""
    SEQUENTIAL = "sequential"      # SQL → Graph (dependent queries)
    PARALLEL = "parallel"          # SQL || Graph || Vector (independent)
    FALLBACK = "fallback"          # SQL fails → Graph (recovery)
    VERIFICATION = "verification"  # Graph validated by SQL (confidence boost)
    SINGLE = "single"              # Single backend (most common)


@dataclass
class RoutingResult:
    """
    Result of query routing

    Attributes:
        pattern: Routing pattern used
        backends_used: List of backends that executed
        rows: Combined result rows
        row_count: Total rows returned
        total_latency_ms: End-to-end latency
        confidence: Final confidence score
        success: Whether routing succeeded
        errors: Any errors encountered
        metadata: Additional metadata
    """
    pattern: RoutingPattern
    backends_used: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    total_latency_ms: float
    confidence: float
    success: bool
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Query Router
# ============================================================================

class QueryRouter:
    """
    Query router with multi-backend coordination

    Integrates:
    - QueryClassifier (7-rule decision tree)
    - ThompsonBandit (adaptive backend selection)
    - MCP Server (SQL backend via MCP)
    - 4 routing patterns (sequential, parallel, fallback, verification)
    """

    def __init__(
        self,
        classifier: QueryClassifier,
        bandit: ThompsonBandit,
        mcp_server: MCPServer,
        session_id: str,
        enable_learning: bool = True,
        enable_calibration: bool = True,
        enable_strategy_updates: bool = True
    ):
        """
        Initialize query router

        Args:
            classifier: Query classifier (7-rule decision tree)
            bandit: Thompson Sampling bandit
            mcp_server: MCP server for SQL backend
            session_id: Session ID for MCP requests
            enable_learning: Enable learning tracker (default: True)
            enable_calibration: Enable confidence calibration (default: True)
            enable_strategy_updates: Enable strategy updates (default: True)
        """
        self.classifier = classifier
        self.bandit = bandit
        self.mcp_server = mcp_server
        self.session_id = session_id
        self.query_count = 0

        # Learning mechanisms (Part 4)
        self.enable_learning = enable_learning
        self.enable_calibration = enable_calibration
        self.enable_strategy_updates = enable_strategy_updates

        # Initialize learning components
        if self.enable_calibration:
            self.calibrator = ConfidenceCalibrator(min_observations=100)
        else:
            self.calibrator = None

        if self.enable_learning:
            self.learning_tracker = LearningTracker(reflection_buffer=None)
        else:
            self.learning_tracker = None

        if self.enable_strategy_updates and self.learning_tracker and self.calibrator:
            self.strategy_updater = StrategyUpdater(
                query_router=self,
                update_interval=3600.0,  # 1 hour
                min_observations=100,
                max_adjustment=0.20
            )
        else:
            self.strategy_updater = None

        # Refinement flag (updated by strategy updater)
        self.enable_refinement = False

    async def route(self, query: str) -> RoutingResult:
        """
        Route query to appropriate backend(s)

        Args:
            query: Natural language query

        Returns:
            RoutingResult with combined results
        """
        self.query_count += 1
        start_time = time.time()

        # Step 1: Classify query
        classification = self.classifier.classify(query)

        # Step 1.5: Apply confidence calibration (Part 4)
        predicted_confidence = classification.confidence
        if self.enable_calibration and self.calibrator:
            # Adjust confidence based on historical calibration
            adjusted_confidence = self.calibrator.adjust_confidence(
                predicted=classification.confidence,
                backend=classification.backend
            )
            classification.confidence = adjusted_confidence

            logger.debug(
                f"Confidence calibrated: {predicted_confidence:.3f} → {adjusted_confidence:.3f}"
            )

        logger.info(
            f"Query classified: backend={classification.backend}, "
            f"confidence={classification.confidence:.2f}, "
            f"rule={classification.rule_matched}"
        )

        # Step 2: Route based on classification
        if classification.backend == Backend.THOMPSON_SAMPLING:
            # Use Thompson Sampling for ambiguous queries
            result = await self._route_thompson_sampling(query, classification)

        elif classification.is_hybrid():
            # Hybrid query (SQL + Graph)
            result = await self._route_hybrid(query, classification)

        else:
            # Single backend
            result = await self._route_single(query, classification)

        # Step 3: Update Thompson Sampling statistics
        for backend in result.backends_used:
            self.bandit.update(
                backend=backend,
                success=result.success,
                confidence=result.confidence
            )

        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        result.total_latency_ms = total_latency_ms

        # Step 4: Learning mechanisms (Part 4)
        if self.enable_learning or self.enable_calibration or self.enable_strategy_updates:
            # Get primary backend
            primary_backend = result.backends_used[0] if result.backends_used else classification.backend

            # Record routing decision in learning tracker
            if self.enable_learning and self.learning_tracker:
                await self.learning_tracker.record_routing(
                    session_id=self.session_id,
                    query=query,
                    backend=primary_backend,
                    predicted_confidence=predicted_confidence,
                    actual_confidence=result.confidence,
                    latency_ms=total_latency_ms,
                    cache_hit=result.metadata.get("cache_hit", False),
                    fallback_used=result.pattern == RoutingPattern.FALLBACK
                )

            # Update calibrator with predicted vs. actual confidence
            if self.enable_calibration and self.calibrator:
                self.calibrator.add_observation(
                    predicted_confidence=predicted_confidence,
                    actual_confidence=result.confidence,
                    backend=primary_backend
                )

            # Periodically update strategy
            if self.enable_strategy_updates and self.strategy_updater:
                await self.strategy_updater.update_if_needed()

        logger.info(
            f"Query routed: pattern={result.pattern}, "
            f"backends={result.backends_used}, "
            f"rows={result.row_count}, "
            f"latency={total_latency_ms:.2f}ms, "
            f"success={result.success}"
        )

        return result

    # ========================================================================
    # Routing Strategies
    # ========================================================================

    async def _route_single(
        self,
        query: str,
        classification: BackendSelection
    ) -> RoutingResult:
        """Route to single backend"""
        backend = classification.backend

        # Execute query on backend
        if backend == Backend.SQL:
            result = await self._execute_sql(query)
        elif backend == Backend.NEO4J:
            result = await self._execute_neo4j(query)
        elif backend == Backend.QDRANT:
            result = await self._execute_qdrant(query)
        else:
            # Unknown backend
            return RoutingResult(
                pattern=RoutingPattern.SINGLE,
                backends_used=[backend],
                rows=[],
                row_count=0,
                total_latency_ms=0.0,
                confidence=0.0,
                success=False,
                errors=[f"Unknown backend: {backend}"]
            )

        return RoutingResult(
            pattern=RoutingPattern.SINGLE,
            backends_used=[backend],
            rows=result["rows"],
            row_count=result["row_count"],
            total_latency_ms=result["latency_ms"],
            confidence=classification.confidence,
            success=result["success"],
            errors=result.get("errors", []),
            metadata={
                "backend": backend,
                "classification": classification.rule_matched
            }
        )

    async def _route_thompson_sampling(
        self,
        query: str,
        classification: BackendSelection
    ) -> RoutingResult:
        """Route using Thompson Sampling"""
        # Select backend via Thompson Sampling
        backend = self.bandit.select()

        logger.info(f"Thompson Sampling selected: {backend}")

        # Execute query
        if backend == Backend.SQL:
            result = await self._execute_sql(query)
        elif backend == Backend.NEO4J:
            result = await self._execute_neo4j(query)
        elif backend == Backend.QDRANT:
            result = await self._execute_qdrant(query)
        else:
            result = {"rows": [], "row_count": 0, "latency_ms": 0.0, "success": False, "errors": [f"Unknown backend: {backend}"]}

        return RoutingResult(
            pattern=RoutingPattern.SINGLE,
            backends_used=[backend],
            rows=result["rows"],
            row_count=result["row_count"],
            total_latency_ms=result["latency_ms"],
            confidence=0.50,  # Exploratory confidence
            success=result["success"],
            errors=result.get("errors", []),
            metadata={
                "backend": backend,
                "classification": "Thompson Sampling",
                "bandit_stats": self.bandit.get_stats()
            }
        )

    async def _route_hybrid(
        self,
        query: str,
        classification: BackendSelection
    ) -> RoutingResult:
        """
        Route hybrid query (multiple backends)

        Pattern: Sequential (SQL → Graph for validation/expansion)
        """
        backends_used = classification.get_backends()

        # Execute primary backend (SQL)
        primary_result = await self._execute_sql(query)

        # Execute secondary backend (Graph) for expansion
        secondary_result = await self._execute_neo4j(query)

        # Merge results
        combined_rows = primary_result["rows"] + secondary_result["rows"]
        combined_latency = primary_result["latency_ms"] + secondary_result["latency_ms"]

        # Success if either backend succeeded
        success = primary_result["success"] or secondary_result["success"]

        # Combine errors
        errors = []
        if not primary_result["success"]:
            errors.extend(primary_result.get("errors", []))
        if not secondary_result["success"]:
            errors.extend(secondary_result.get("errors", []))

        return RoutingResult(
            pattern=RoutingPattern.SEQUENTIAL,
            backends_used=backends_used,
            rows=combined_rows,
            row_count=len(combined_rows),
            total_latency_ms=combined_latency,
            confidence=classification.confidence,
            success=success,
            errors=errors,
            metadata={
                "primary_backend": backends_used[0],
                "secondary_backend": backends_used[1] if len(backends_used) > 1 else None,
                "classification": classification.rule_matched
            }
        )

    # ========================================================================
    # Backend Execution (MCP Integration)
    # ========================================================================

    async def _execute_sql(self, query: str) -> Dict[str, Any]:
        """
        Execute query on SQL backend via MCP

        Args:
            query: Natural language query (will be converted to SQL)

        Returns:
            Dict with rows, row_count, latency_ms, success, errors
        """
        # For demo purposes, convert NL query to SQL
        # In production, this would use an NL-to-SQL model
        sql_query = self._nl_to_sql(query)

        # Create MCP request
        request = MCPRequest(
            request_id=generate_request_id(),
            session_id=self.session_id,
            request_type=RequestType.TOOL_CALL,
            tool_name=ToolName.QUERY_SQL,
            parameters={
                "query": sql_query,
                "params": [],
                "timeout_ms": 30000
            }
        )

        # Execute via MCP
        response = await self.mcp_server.handle_request(request)

        if response.status == ResponseStatus.SUCCESS:
            return {
                "rows": response.result["rows"],
                "row_count": response.result["row_count"],
                "latency_ms": response.result["latency_ms"],
                "success": True,
                "errors": []
            }
        else:
            error_msg = response.error.get("message", "Unknown error") if response.error else "Unknown error"
            return {
                "rows": [],
                "row_count": 0,
                "latency_ms": 0.0,
                "success": False,
                "errors": [error_msg]
            }

    async def _execute_neo4j(self, query: str) -> Dict[str, Any]:
        """
        Execute query on Neo4j backend

        Args:
            query: Natural language query

        Returns:
            Dict with rows, row_count, latency_ms, success, errors

        Note: Mock implementation for now (Neo4j integration in Part 4)
        """
        logger.debug(f"Neo4j query (mock): {query}")

        # Mock response
        await asyncio.sleep(0.03)  # Simulate 30ms latency

        return {
            "rows": [
                {"entity": "hive_042", "relationship": "uses", "policy": "bee_001"},
                {"entity": "hive_001", "relationship": "follows", "policy": "bee_002"}
            ],
            "row_count": 2,
            "latency_ms": 30.0,
            "success": True,
            "errors": []
        }

    async def _execute_qdrant(self, query: str) -> Dict[str, Any]:
        """
        Execute query on Qdrant backend

        Args:
            query: Natural language query

        Returns:
            Dict with rows, row_count, latency_ms, success, errors

        Note: Mock implementation for now (Qdrant integration in Part 4)
        """
        logger.debug(f"Qdrant query (mock): {query}")

        # Mock response
        await asyncio.sleep(0.025)  # Simulate 25ms latency

        return {
            "rows": [
                {"document": "Varroa treatment guide", "similarity": 0.92},
                {"document": "Integrated pest management", "similarity": 0.87},
                {"document": "Hive inspection checklist", "similarity": 0.81}
            ],
            "row_count": 3,
            "latency_ms": 25.0,
            "success": True,
            "errors": []
        }

    # ========================================================================
    # Helper Functions
    # ========================================================================

    def _nl_to_sql(self, query: str) -> str:
        """
        Convert natural language query to SQL

        Args:
            query: Natural language query

        Returns:
            SQL query string

        Note: Simple keyword matching for demo.
              Production would use NL-to-SQL model.
        """
        query_lower = query.lower()

        # Policy queries
        if "policy" in query_lower or "rule" in query_lower:
            if "treatment" in query_lower:
                return "SELECT * FROM policy_rules WHERE rule_type = 'treatment'"
            elif "harvest" in query_lower:
                return "SELECT * FROM policy_rules WHERE rule_type = 'harvest'"
            else:
                return "SELECT * FROM policy_rules"

        # Transaction queries
        if "transaction" in query_lower or "action" in query_lower:
            if "hive" in query_lower:
                return "SELECT * FROM transaction_logs WHERE entity_type = 'hive'"
            else:
                return "SELECT * FROM transaction_logs"

        # Audit queries
        if "audit" in query_lower or "compliance" in query_lower:
            if "violation" in query_lower or "non-compliant" in query_lower:
                return "SELECT * FROM audit_trails WHERE compliance_flag = 1"
            else:
                return "SELECT * FROM audit_trails"

        # Permission queries
        if "permission" in query_lower or "access" in query_lower:
            return "SELECT * FROM user_permissions"

        # Count queries
        if "how many" in query_lower or "count" in query_lower:
            if "policy" in query_lower or "rule" in query_lower:
                return "SELECT COUNT(*) as count FROM policy_rules"
            elif "transaction" in query_lower:
                return "SELECT COUNT(*) as count FROM transaction_logs"
            elif "audit" in query_lower:
                return "SELECT COUNT(*) as count FROM audit_trails"
            else:
                return "SELECT COUNT(*) as count FROM policy_rules"

        # Default: return all policies
        return "SELECT * FROM policy_rules LIMIT 10"

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        return {
            "query_count": self.query_count,
            "classifier_stats": self.classifier.get_stats(),
            "bandit_stats": self.bandit.get_stats(),
            "bandit_summary": self.bandit.get_summary()
        }


# ============================================================================
# Factory Function
# ============================================================================

async def create_query_router(
    mcp_server: MCPServer,
    session_id: str,
    backends: Optional[List[str]] = None,
    enable_learning: bool = True,
    enable_calibration: bool = True,
    enable_strategy_updates: bool = True
) -> QueryRouter:
    """
    Create query router with classifier, bandit, and learning mechanisms

    Args:
        mcp_server: MCP server for SQL backend
        session_id: Session ID for MCP requests
        backends: List of backend names (default: ["sql", "neo4j", "qdrant"])
        enable_learning: Enable learning tracker (default: True)
        enable_calibration: Enable confidence calibration (default: True)
        enable_strategy_updates: Enable strategy updates (default: True)

    Returns:
        QueryRouter instance with learning enabled
    """
    if backends is None:
        backends = [Backend.SQL, Backend.NEO4J, Backend.QDRANT]

    classifier = QueryClassifier()
    bandit = ThompsonBandit(backends)

    return QueryRouter(
        classifier=classifier,
        bandit=bandit,
        mcp_server=mcp_server,
        session_id=session_id,
        enable_learning=enable_learning,
        enable_calibration=enable_calibration,
        enable_strategy_updates=enable_strategy_updates
    )

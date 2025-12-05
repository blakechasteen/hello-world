#!/usr/bin/env python3
"""
Enhanced HoloLoom Integration for Promptly Matrix Bot

Integrates HoloLoom's complete 9-step weaving cycle with:
- Dynamic memory backend (Yarn Graph / KG)
- Proper async lifecycle management
- Production hardening (circuit breakers, rate limiting, metrics)
- Real-time WebSocket updates for dashboard
- Neo4j knowledge graph support

Features:
- Full knowledge graph memory (NetworkX or Neo4j)
- Multi-scale Matryoshka embeddings (96-192-384D)
- Thompson Sampling exploration
- Spectral graph features
- Complete provenance tracking
- Recursive learning
- Production-grade reliability

Usage:
    from bot.hololoom_integration_enhanced import EnhancedHoloLoomBot

    bot = EnhancedHoloLoomBot(config_mode="FAST")

    # Use as async context manager (recommended)
    async with bot:
        response = await bot.weave(
            query="What is Thompson Sampling?",
            user_id="@alice:matrix.org",
            complexity="FAST"
        )

        print(response.text)
        print(f"Confidence: {response.confidence}")
"""

import logging
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Add HoloLoom to path
HOLOLOOM_PATH = Path(__file__).parent.parent.parent / "HoloLoom"
if HOLOLOOM_PATH.exists():
    sys.path.insert(0, str(HOLOLOOM_PATH.parent))
else:
    logger.warning(f"HoloLoom not found at {HOLOLOOM_PATH}")


@dataclass
class WeavingResponse:
    """Response from HoloLoom weaving"""
    text: str                          # Response text
    confidence: float                  # Confidence score [0-1]
    tool_used: str                     # Tool that was selected
    context_depth: int                 # Graph traversal depth
    memories_retrieved: int            # Number of memories used
    latency_ms: float                  # Total latency
    cache_hit: bool = False           # Was response cached?
    provenance: Dict[str, Any] = field(default_factory=dict)  # Full trace
    metadata: Dict[str, Any] = field(default_factory=dict)    # Additional metadata

    # Stage-by-stage breakdown (for dashboard visualization)
    stages: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary"""
        cache_str = " (cached)" if self.cache_hit else ""
        return (
            f"Confidence: {self.confidence:.2%}{cache_str}\n"
            f"Tool: {self.tool_used}\n"
            f"Context: {self.memories_retrieved} memories, depth {self.context_depth}\n"
            f"Latency: {self.latency_ms:.0f}ms"
        )


class EnhancedHoloLoomBot:
    """
    Enhanced HoloLoom-integrated bot with production features.

    Features:
    - Dynamic memory backend (Yarn Graph / KG)
    - Async context manager support
    - Production hardening (circuit breakers, rate limiting)
    - Real-time WebSocket updates
    - Neo4j knowledge graph support

    Usage:
        bot = EnhancedHoloLoomBot(config_mode="FAST")

        async with bot:
            response = await bot.weave(
                query="Explain recursion",
                user_id="@alice:matrix.org"
            )
    """

    def __init__(
        self,
        config_mode: str = "FAST",
        enable_production_hardening: bool = True,
        enable_neo4j: bool = False,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        websocket_callback: Optional[Callable] = None
    ):
        """
        Initialize Enhanced HoloLoom bot.

        Args:
            config_mode: Default config mode (BARE/FAST/FUSED)
            enable_production_hardening: Enable circuit breakers, rate limiting, metrics
            enable_neo4j: Use Neo4j for knowledge graph (production)
            neo4j_uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            websocket_callback: Optional callback for real-time updates
        """
        self.config_mode = config_mode
        self.enable_production_hardening = enable_production_hardening
        self.enable_neo4j = enable_neo4j
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.websocket_callback = websocket_callback

        # Core components
        self.orchestrator = None
        self.config = None
        self.yarn_graph = None  # KG instance
        self.memory_backend = None
        self.initialized = False

        # Production components
        self.circuit_breaker = None
        self.rate_limiter = None

    async def __aenter__(self):
        """Async context manager entry - initialize bot"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources"""
        await self.cleanup()
        return False

    async def initialize(self):
        """
        Initialize HoloLoom components.

        Sets up:
        - Configuration
        - Memory backend (Yarn Graph or Neo4j)
        - Orchestrator with production hardening
        - Circuit breakers and rate limiting
        """
        if self.initialized:
            return

        try:
            # Import HoloLoom components
            from HoloLoom.config import Config, MemoryBackend
            from HoloLoom.weaving_orchestrator import WeavingOrchestrator
            from HoloLoom.memory.graph import KG
            from HoloLoom.documentation.types import MemoryShard

            logger.info(f"Initializing Enhanced HoloLoom bot (mode: {self.config_mode})")

            # Create config
            if self.config_mode == "BARE":
                self.config = Config.bare()
            elif self.config_mode == "FUSED":
                self.config = Config.fused()
            else:  # FAST (default)
                self.config = Config.fast()

            # Configure memory backend
            if self.enable_neo4j and self.neo4j_uri:
                logger.info("Initializing Neo4j knowledge graph backend")
                self.config.memory_backend = MemoryBackend.HYBRID
                # Memory backend will be created by orchestrator
            else:
                logger.info("Initializing NetworkX in-memory knowledge graph")
                # Create Yarn Graph (KG instance)
                self.yarn_graph = KG()

                # Populate with initial bot knowledge
                self._populate_yarn_graph()

            # Create stage callback for real-time updates
            stage_callback = self._create_stage_callback() if self.websocket_callback else None

            # Create orchestrator with production hardening
            self.orchestrator = WeavingOrchestrator(
                cfg=self.config,
                yarn_graph=self.yarn_graph,
                enable_reflection=True,
                enable_complexity_auto_detect=True,
                enable_semantic_cache=True,
                enable_production_hardening=self.enable_production_hardening,
                enable_circuit_breakers=True,
                circuit_breaker_threshold=5,
                rate_limit_qps=100.0,
                enable_health_checks=True,
                stage_callback=stage_callback
            )

            # Enter orchestrator context
            await self.orchestrator.__aenter__()

            self.initialized = True
            logger.info(f"Enhanced HoloLoom bot initialized successfully")
            logger.info(f"  Config: {self.config_mode}")
            logger.info(f"  Production Hardening: {self.enable_production_hardening}")
            logger.info(f"  Memory Backend: {'Neo4j' if self.enable_neo4j else 'NetworkX'}")

        except ImportError as e:
            logger.error(f"Failed to import HoloLoom: {e}")
            logger.error("Make sure HoloLoom is in the parent directory")
            raise

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced HoloLoom bot: {e}")
            raise

    def _populate_yarn_graph(self):
        """Populate Yarn Graph with initial bot knowledge"""
        try:
            from HoloLoom.memory.graph import KGEdge

            # Define bot knowledge as edges
            knowledge_edges = [
                # Bot capabilities
                KGEdge("promptly", "dspy", "USES", 1.0),
                KGEdge("promptly", "prompt_optimization", "DOES", 1.0),
                KGEdge("dspy", "machine_learning", "IS_A", 1.0),

                # Schema builder
                KGEdge("schema_builder", "json_schema", "CREATES", 1.0),
                KGEdge("schema_builder", "pydantic", "CREATES", 1.0),
                KGEdge("schema_builder", "dspy_signature", "CREATES", 1.0),

                # Verification
                KGEdge("verification", "hallucination_detection", "DOES", 1.0),
                KGEdge("verification", "fact_checking", "DOES", 1.0),

                # Refinement
                KGEdge("refinement", "elegance", "STRATEGY", 1.0),
                KGEdge("refinement", "verify", "STRATEGY", 1.0),
                KGEdge("elegance", "clarity", "IMPROVES", 1.0),
                KGEdge("elegance", "simplicity", "IMPROVES", 1.0),
                KGEdge("verify", "accuracy", "IMPROVES", 1.0),
                KGEdge("verify", "completeness", "IMPROVES", 1.0),

                # Thompson Sampling
                KGEdge("thompson_sampling", "bayesian", "USES", 1.0),
                KGEdge("thompson_sampling", "exploration", "BALANCES", 1.0),
                KGEdge("thompson_sampling", "exploitation", "BALANCES", 1.0),
                KGEdge("bayesian", "posterior_distribution", "COMPUTES", 1.0),
            ]

            # Add edges to Yarn Graph
            self.yarn_graph.add_edges(knowledge_edges)

            logger.info(f"Populated Yarn Graph with {len(knowledge_edges)} knowledge edges")

        except Exception as e:
            logger.warning(f"Failed to populate Yarn Graph: {e}")

    def _create_stage_callback(self) -> Callable:
        """Create callback for real-time stage updates"""
        async def stage_update(stage_name: str, status: str, metadata: Dict[str, Any]):
            """Called when weaving stage completes"""
            if self.websocket_callback:
                try:
                    await self.websocket_callback({
                        "type": "weaving_stage",
                        "stage": stage_name,
                        "status": status,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.warning(f"WebSocket callback failed: {e}")

        return stage_update

    async def weave(
        self,
        query: str,
        user_id: str,
        room_id: Optional[str] = None,
        complexity: str = "FAST"
    ) -> WeavingResponse:
        """
        Process query through HoloLoom weaving cycle.

        Args:
            query: User query
            user_id: User ID
            room_id: Room ID (optional)
            complexity: Complexity level (LITE/FAST/FULL/RESEARCH)

        Returns:
            WeavingResponse with answer and metadata
        """
        if not self.initialized:
            await self.initialize()

        import time
        start_time = time.time()

        try:
            from HoloLoom.documentation.types import Query as HoloLoomQuery

            # Notify start
            if self.websocket_callback:
                await self.websocket_callback({
                    "type": "weaving_start",
                    "query_text": query,
                    "user_id": user_id,
                    "complexity": complexity,
                    "timestamp": datetime.now().isoformat()
                })

            # Create query object
            hololoom_query = HoloLoomQuery(
                text=query,
                metadata={
                    "user_id": user_id,
                    "room_id": room_id,
                    "complexity": complexity
                }
            )

            # Weave (full 9-step cycle)
            spacetime = await self.orchestrator.weave(hololoom_query)

            # Extract response
            latency_ms = (time.time() - start_time) * 1000

            # Extract stage breakdown if available
            stages = []
            if hasattr(spacetime, 'trace') and hasattr(spacetime.trace, 'stages'):
                stages = [
                    {
                        "name": stage.name,
                        "status": stage.status,
                        "duration_ms": stage.duration_ms,
                        "metadata": stage.metadata
                    }
                    for stage in spacetime.trace.stages
                ]

            response = WeavingResponse(
                text=spacetime.response,
                confidence=spacetime.confidence,
                tool_used=spacetime.metadata.get('tool_used', 'unknown'),
                context_depth=spacetime.metadata.get('context_depth', 0),
                memories_retrieved=spacetime.metadata.get('memories_retrieved', 0),
                latency_ms=latency_ms,
                cache_hit=spacetime.metadata.get('cache_hit', False),
                provenance={
                    'trace': spacetime.trace.summary() if hasattr(spacetime, 'trace') else {},
                    'decision_rationale': spacetime.metadata.get('decision_rationale', '')
                },
                metadata=spacetime.metadata,
                stages=stages
            )

            logger.info(
                f"Weaving complete: query='{query[:50]}...', "
                f"confidence={response.confidence:.2f}, "
                f"latency={latency_ms:.0f}ms"
            )

            # Notify completion
            if self.websocket_callback:
                await self.websocket_callback({
                    "type": "weaving_complete",
                    "response": response.text,
                    "confidence": response.confidence,
                    "tool_used": response.tool_used,
                    "latency_ms": latency_ms,
                    "timestamp": datetime.now().isoformat()
                })

            return response

        except Exception as e:
            logger.error(f"Weaving failed: {e}")

            # Notify error
            if self.websocket_callback:
                await self.websocket_callback({
                    "type": "weaving_error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

            # Fallback response
            return WeavingResponse(
                text=f"Error processing query: {str(e)}",
                confidence=0.0,
                tool_used="error",
                context_depth=0,
                memories_retrieved=0,
                latency_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)}
            )

    async def add_memory(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        user_id: Optional[str] = None
    ):
        """
        Add memory to knowledge graph.

        Args:
            text: Memory text
            entities: Extracted entities
            user_id: User who created memory
        """
        if not self.initialized:
            await self.initialize()

        try:
            from HoloLoom.memory.graph import KGEdge

            # Extract entities if not provided
            if not entities:
                # Simple entity extraction (in production, use NER)
                entities = [word for word in text.split() if word[0].isupper()]

            # Create edges between entities
            edges = []
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    edges.append(KGEdge(entity1, entity2, "MENTIONS", 0.8))

            # Add to Yarn Graph
            if self.yarn_graph:
                self.yarn_graph.add_edges(edges)
                logger.info(f"Added memory with {len(edges)} edges: {text[:50]}...")

        except Exception as e:
            logger.error(f"Failed to add memory: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.

        Returns:
            Dictionary with memory count, config, health status, etc.
        """
        if not self.initialized:
            return {
                "initialized": False,
                "error": "Not initialized"
            }

        try:
            stats = {
                "initialized": True,
                "config_mode": self.config_mode,
                "production_hardening": self.enable_production_hardening,
                "memory_backend": "Neo4j" if self.enable_neo4j else "NetworkX"
            }

            # Memory statistics
            if self.yarn_graph:
                stats["memory"] = {
                    "nodes": self.yarn_graph.num_nodes(),
                    "edges": self.yarn_graph.num_edges()
                }

            # Production hardening statistics
            if self.enable_production_hardening and self.orchestrator:
                if hasattr(self.orchestrator, 'health_checker'):
                    health = await self.orchestrator.health_checker.check_health()
                    stats["health"] = {
                        "status": health.status.value,
                        "components": {
                            name: component.status.value
                            for name, component in health.components.items()
                        }
                    }

            return stats

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                "initialized": True,
                "error": str(e)
            }

    async def cleanup(self):
        """Cleanup resources"""
        if self.orchestrator:
            try:
                await self.orchestrator.__aexit__(None, None, None)
                logger.info("Orchestrator cleaned up successfully")
            except Exception as e:
                logger.error(f"Failed to cleanup orchestrator: {e}")

        self.initialized = False


# Backward compatibility - alias to original class name
HoloLoomBot = EnhancedHoloLoomBot

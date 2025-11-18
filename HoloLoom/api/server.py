"""
HoloLoom Production API Server
================================

FastAPI server exposing all Week 5-7 memory systems.

Endpoints:
- Health & Stats
- Memory operations (experience, recall)
- Consolidation (trigger, status)
- Semantic transition (detect patterns, promote)
- Temporal evolution (query, history, summary)
- Curiosity engine (suggestions)
- Graph reasoning (multi-hop queries)

Author: HoloLoom Team
Date: 2025-11-18 (Week 8B)
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Import models
from HoloLoom.api.models import *
from HoloLoom.api.middleware import setup_middleware, APIKeyAuth

# Import HoloLoom memory systems
from HoloLoom import HoloLoom
from HoloLoom.config import Config
from HoloLoom.memory.consolidation import (
    SleepBasedConsolidation,
    ConsolidationStrategy as ConsolidationStrategyEnum
)
from HoloLoom.memory.semantic_transition import (
    SemanticTransitionEngine,
    SemanticTransitionConfig
)
from HoloLoom.memory.temporal_evolution import (
    TemporalEvolutionTracker,
    UnderstandingState as UnderstandingStateEnum
)
from HoloLoom.memory.curiosity import CuriosityEngine, CuriosityConfig
from HoloLoom.memory.graph_reasoning import GraphReasoner
from HoloLoom.memory.graph import KG

logger = logging.getLogger(__name__)


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Request metrics
REQUEST_COUNT = Counter(
    'hololoom_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'hololoom_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

# Memory metrics
MEMORY_COUNT = Gauge(
    'hololoom_memories_total',
    'Total memories stored',
    ['scope']
)

CONSOLIDATION_COUNT = Counter(
    'hololoom_consolidations_total',
    'Total consolidations',
    ['strategy']
)

SEMANTIC_CONCEPTS = Gauge(
    'hololoom_semantic_concepts_total',
    'Total semantic concepts'
)

# System health
SYSTEM_HEALTH = Gauge(
    'hololoom_system_health',
    'System health (1=healthy, 0=unhealthy)',
    ['component']
)


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state."""

    def __init__(self):
        self.loom: Optional[HoloLoom] = None
        self.consolidation: Optional[SleepBasedConsolidation] = None
        self.semantic_engine: Optional[SemanticTransitionEngine] = None
        self.temporal_tracker: Optional[TemporalEvolutionTracker] = None
        self.curiosity_engine: Optional[CuriosityEngine] = None
        self.graph_reasoner: Optional[GraphReasoner] = None
        self.kg: Optional[KG] = None
        self.background_tasks: Dict[str, asyncio.Task] = {}
        self.stats = {
            "consolidation": {
                "total_consolidations": 0,
                "total_episodes_processed": 0,
                "total_facts_created": 0,
                "total_time_ms": 0.0,
                "last_consolidation": None
            },
            "semantic": {
                "total_patterns_detected": 0,
                "total_concepts_created": 0,
                "total_pattern_frequency": 0,
                "last_transition": None
            },
            "curiosity": {
                "total_suggestions": 0,
                "suggestions_by_type": {t.value: 0 for t in ExplorationSuggestionType},
                "suggestions_followed": 0
            }
        }


app_state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle (startup/shutdown).

    Initializes HoloLoom and all memory systems on startup,
    cleans up on shutdown.
    """
    logger.info("Starting HoloLoom API server...")

    # Initialize HoloLoom
    config = Config.fused()
    app_state.loom = HoloLoom(cfg=config)
    await app_state.loom.__aenter__()

    # Initialize knowledge graph
    app_state.kg = KG()

    # Initialize consolidation
    app_state.consolidation = SleepBasedConsolidation(
        loom=app_state.loom,
        kg=app_state.kg,
        consolidation_interval_minutes=60.0
    )

    # Initialize semantic transition
    semantic_config = SemanticTransitionConfig(
        enabled=True,
        background_interval_seconds=300.0
    )
    app_state.semantic_engine = SemanticTransitionEngine(
        loom=app_state.loom,
        kg=app_state.kg,
        config=semantic_config
    )

    # Initialize temporal tracker
    app_state.temporal_tracker = TemporalEvolutionTracker(
        loom=app_state.loom
    )

    # Initialize curiosity engine
    curiosity_config = CuriosityConfig(enabled=True)
    app_state.curiosity_engine = CuriosityEngine(
        kg=app_state.kg,
        config=curiosity_config
    )

    # Initialize graph reasoner
    app_state.graph_reasoner = GraphReasoner(
        kg=app_state.kg,
        retriever=None  # Will use KG-only reasoning
    )

    # Start background consolidation
    if hasattr(app_state.consolidation, 'start_background_consolidation'):
        task = asyncio.create_task(
            app_state.consolidation.start_background_consolidation()
        )
        app_state.background_tasks['consolidation'] = task

    # Start background semantic transition
    if hasattr(app_state.semantic_engine, 'start_background_transition'):
        task = asyncio.create_task(
            app_state.semantic_engine.start_background_transition()
        )
        app_state.background_tasks['semantic'] = task

    # Set system health
    SYSTEM_HEALTH.labels(component='consolidation').set(1)
    SYSTEM_HEALTH.labels(component='semantic').set(1)
    SYSTEM_HEALTH.labels(component='temporal').set(1)
    SYSTEM_HEALTH.labels(component='curiosity').set(1)
    SYSTEM_HEALTH.labels(component='graph').set(1)

    logger.info("HoloLoom API server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down HoloLoom API server...")

    # Cancel background tasks
    for task_name, task in app_state.background_tasks.items():
        logger.info(f"Cancelling background task: {task_name}")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Cleanup HoloLoom
    if app_state.loom:
        await app_state.loom.__aexit__(None, None, None)

    logger.info("HoloLoom API server shut down")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="HoloLoom Memory API",
    description="Production API for HoloLoom's advanced memory systems (Weeks 5-7)",
    version="1.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Setup authentication
api_key_auth = APIKeyAuth()


# ============================================================================
# Dependencies
# ============================================================================

async def get_api_key(key: str = Depends(api_key_auth)) -> Optional[str]:
    """Dependency for API key authentication."""
    return key


# ============================================================================
# Health & Metrics Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns system status and component health.
    """
    systems = {
        "consolidation": "ok" if app_state.consolidation else "unavailable",
        "semantic_transition": "ok" if app_state.semantic_engine else "unavailable",
        "temporal_evolution": "ok" if app_state.temporal_tracker else "unavailable",
        "curiosity": "ok" if app_state.curiosity_engine else "unavailable",
        "graph_reasoning": "ok" if app_state.graph_reasoner else "unavailable"
    }

    all_ok = all(status == "ok" for status in systems.values())

    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        systems=systems,
        timestamp=datetime.now()
    )


@app.get("/metrics", tags=["Metrics"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    return PlainTextResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/api/v1/stats", response_model=SystemStatsResponse, tags=["Stats"])
async def get_system_stats(api_key: str = Depends(get_api_key)):
    """
    Get complete system statistics.

    Requires authentication.
    """
    # Consolidation stats
    consolidation_stats = ConsolidationStatsResponse(
        total_consolidations=app_state.stats["consolidation"]["total_consolidations"],
        total_episodes_processed=app_state.stats["consolidation"]["total_episodes_processed"],
        total_facts_created=app_state.stats["consolidation"]["total_facts_created"],
        average_consolidation_time_ms=(
            app_state.stats["consolidation"]["total_time_ms"] /
            max(1, app_state.stats["consolidation"]["total_consolidations"])
        ),
        last_consolidation=app_state.stats["consolidation"]["last_consolidation"]
    )

    # Semantic stats
    semantic_stats = SemanticStatsResponse(
        total_patterns_detected=app_state.stats["semantic"]["total_patterns_detected"],
        total_concepts_created=app_state.stats["semantic"]["total_concepts_created"],
        average_pattern_frequency=(
            app_state.stats["semantic"]["total_pattern_frequency"] /
            max(1, app_state.stats["semantic"]["total_patterns_detected"])
        ),
        episodic_to_semantic_ratio=0.0,  # TODO: Calculate
        last_transition=app_state.stats["semantic"]["last_transition"]
    )

    # Curiosity stats
    curiosity_stats = CuriosityStatsResponse(
        total_suggestions_generated=app_state.stats["curiosity"]["total_suggestions"],
        suggestions_by_type=app_state.stats["curiosity"]["suggestions_by_type"],
        average_importance=0.7,  # TODO: Track
        suggestions_followed=app_state.stats["curiosity"]["suggestions_followed"],
        follow_rate=(
            app_state.stats["curiosity"]["suggestions_followed"] /
            max(1, app_state.stats["curiosity"]["total_suggestions"])
        )
    )

    # Graph stats
    graph_stats = {
        "total_nodes": app_state.kg.node_count() if app_state.kg else 0,
        "total_edges": app_state.kg.edge_count() if app_state.kg else 0
    }

    return SystemStatsResponse(
        consolidation=consolidation_stats,
        semantic=semantic_stats,
        curiosity=curiosity_stats,
        graph=graph_stats
    )


# ============================================================================
# Memory Operations
# ============================================================================

@app.post("/api/v1/experience", response_model=ExperienceResponse, tags=["Memory"])
async def create_experience(
    request: ExperienceRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Store new episodic memory.

    Creates a new memory in the specified scope (SESSION, AGENT, USER, GLOBAL).
    """
    start_time = time.time()

    try:
        # Create memory
        result = await app_state.loom.experience(request.content)

        # Update metrics
        MEMORY_COUNT.labels(scope=request.scope.value).inc()

        # Track timing
        duration_ms = (time.time() - start_time) * 1000
        REQUEST_DURATION.labels(endpoint='/api/v1/experience').observe(duration_ms / 1000)

        return ExperienceResponse(
            memory_id=result.id,
            scope=request.scope,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Failed to create experience: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create memory: {str(e)}"
        )


@app.post("/api/v1/recall", response_model=RecallResponse, tags=["Memory"])
async def recall_memories(
    request: RecallRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Retrieve relevant memories.

    Searches memories using hybrid retrieval (BM25 + semantic similarity).
    """
    start_time = time.time()

    try:
        # Recall memories
        memories = await app_state.loom.recall(
            request.query,
            k=request.max_results
        )

        # Convert to response format
        memory_results = [
            MemoryResult(
                memory_id=mem.id,
                content=mem.text,
                scope=MemoryScope.SESSION,  # TODO: Get from memory
                relevance_score=1.0,  # TODO: Get from recall
                timestamp=datetime.now(),
                metadata=None if not request.include_metadata else {}
            )
            for mem in memories
        ]

        duration_ms = (time.time() - start_time) * 1000

        return RecallResponse(
            query=request.query,
            memories=memory_results,
            count=len(memory_results),
            total_available=len(memory_results),  # TODO: Get actual count
            latency_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Failed to recall memories: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to recall memories: {str(e)}"
        )


# ============================================================================
# Consolidation Endpoints
# ============================================================================

@app.post("/api/v1/consolidation/trigger", response_model=ConsolidationResponse, tags=["Consolidation"])
async def trigger_consolidation(
    request: ConsolidationTriggerRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Manually trigger consolidation.

    Queues a background consolidation job.
    """
    job_id = str(uuid.uuid4())

    async def run_consolidation():
        """Background consolidation task."""
        try:
            start_time = time.time()

            # Run consolidation
            result = await app_state.consolidation.consolidate_during_idle()

            duration_ms = (time.time() - start_time) * 1000

            # Update stats
            app_state.stats["consolidation"]["total_consolidations"] += 1
            app_state.stats["consolidation"]["total_episodes_processed"] += result.input_episodes
            app_state.stats["consolidation"]["total_facts_created"] += result.output_facts
            app_state.stats["consolidation"]["total_time_ms"] += duration_ms
            app_state.stats["consolidation"]["last_consolidation"] = datetime.now()

            # Update metrics
            CONSOLIDATION_COUNT.labels(strategy=result.strategy.value).inc()

            logger.info(f"Consolidation complete: {result.input_episodes} → {result.output_facts}")

        except Exception as e:
            logger.error(f"Consolidation failed: {e}", exc_info=True)

    background_tasks.add_task(run_consolidation)

    return ConsolidationResponse(
        status="queued",
        job_id=job_id,
        estimated_duration_ms=5000.0  # Estimate
    )


# ============================================================================
# Semantic Transition Endpoints
# ============================================================================

@app.post("/api/v1/semantic/detect-patterns", response_model=PatternDetectionResponse, tags=["Semantic"])
async def detect_patterns(
    request: PatternDetectionRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Detect episodic patterns.

    Analyzes recent episodic memories to find recurring patterns.
    """
    start_time = time.time()

    try:
        # Detect patterns
        patterns = await app_state.semantic_engine.detect_patterns(
            threshold=request.threshold,
            similarity_threshold=request.similarity_threshold,
            window_days=request.window_days
        )

        # Convert to response format
        pattern_results = [
            EpisodicPattern(
                pattern_id=p.pattern_id,
                pattern_type=p.pattern_type,
                frequency=p.frequency,
                similarity_score=p.similarity_score,
                representative_query=p.representative_query,
                common_entities=p.common_entities,
                common_motifs=p.common_motifs,
                episodic_memory_ids=p.episodic_memory_ids,
                first_seen=p.first_seen,
                last_seen=p.last_seen
            )
            for p in patterns[:request.max_patterns]
        ]

        duration_ms = (time.time() - start_time) * 1000

        # Update stats
        app_state.stats["semantic"]["total_patterns_detected"] += len(pattern_results)
        app_state.stats["semantic"]["total_pattern_frequency"] += sum(
            p.frequency for p in patterns
        )

        return PatternDetectionResponse(
            patterns=pattern_results,
            count=len(pattern_results),
            detection_time_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Pattern detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pattern detection failed: {str(e)}"
        )


@app.post("/api/v1/semantic/promote", response_model=PromotePatternResponse, tags=["Semantic"])
async def promote_pattern(
    request: PromotePatternRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Promote pattern to semantic concept.

    Queues a background job to create semantic concept from episodic pattern.
    """
    job_id = str(uuid.uuid4())

    async def run_promotion():
        """Background promotion task."""
        try:
            # Promote pattern
            concept = await app_state.semantic_engine.promote_to_semantic(
                request.pattern_id,
                concept_name=request.concept_name
            )

            # Update stats
            app_state.stats["semantic"]["total_concepts_created"] += 1
            app_state.stats["semantic"]["last_transition"] = datetime.now()

            # Update metrics
            SEMANTIC_CONCEPTS.inc()

            logger.info(f"Pattern promoted: {request.pattern_id} → {concept.concept_id}")

        except Exception as e:
            logger.error(f"Pattern promotion failed: {e}", exc_info=True)

    background_tasks.add_task(run_promotion)

    return PromotePatternResponse(
        status="queued",
        pattern_id=request.pattern_id,
        job_id=job_id
    )


# ============================================================================
# Temporal Evolution Endpoints
# ============================================================================

@app.post("/api/v1/temporal/query", response_model=TemporalQueryResponse, tags=["Temporal"])
async def temporal_query(
    request: TemporalQueryRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Query understanding at specific time.

    Returns snapshot of understanding state at the requested timestamp.
    """
    try:
        snapshot = await app_state.temporal_tracker.query_at_time(
            request.concept,
            request.timestamp
        )

        if snapshot:
            return TemporalQueryResponse(
                concept=request.concept,
                timestamp=request.timestamp,
                snapshot=UnderstandingSnapshot(
                    concept=snapshot.concept,
                    state=UnderstandingState(snapshot.state.value),
                    memory_count=snapshot.memory_count,
                    confidence=snapshot.confidence,
                    timestamp=snapshot.timestamp,
                    notable_memories=snapshot.notable_memories,
                    metadata=snapshot.metadata
                ),
                found=True
            )
        else:
            return TemporalQueryResponse(
                concept=request.concept,
                timestamp=request.timestamp,
                snapshot=None,
                found=False
            )

    except Exception as e:
        logger.error(f"Temporal query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Temporal query failed: {str(e)}"
        )


@app.get("/api/v1/temporal/history/{concept}", response_model=ConceptHistoryResponse, tags=["Temporal"])
async def get_concept_history(
    concept: str,
    api_key: str = Depends(get_api_key)
):
    """
    Get complete evolution history for a concept.

    Returns all state transitions, memory timeline, and milestones.
    """
    try:
        history = await app_state.temporal_tracker.get_concept_history(concept)

        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No history found for concept: {concept}"
            )

        # Convert to response format
        state_transitions = [
            {
                "timestamp": t.timestamp.isoformat(),
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "trigger": t.trigger
            }
            for t in history.state_transitions
        ]

        memory_timeline = [
            {
                "timestamp": m.timestamp.isoformat(),
                "memory_id": m.memory_id,
                "confidence": m.confidence
            }
            for m in history.memory_timeline
        ]

        milestones = [
            {
                "type": m.milestone_type,
                "timestamp": m.timestamp.isoformat(),
                "description": m.description
            }
            for m in history.milestones
        ]

        return ConceptHistoryResponse(
            concept=concept,
            first_learned=history.first_learned,
            current_state=UnderstandingState(history.current_state.value),
            state_transitions=state_transitions,
            memory_timeline=memory_timeline,
            milestones=milestones,
            total_memories=history.total_memories
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get concept history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get concept history: {str(e)}"
        )


@app.post("/api/v1/temporal/summary", response_model=EvolutionSummaryResponse, tags=["Temporal"])
async def get_evolution_summary(
    request: EvolutionSummaryRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Get evolution summary for specified period.

    Returns aggregate statistics about learning progress.
    """
    try:
        summary = await app_state.temporal_tracker.get_evolution_summary(
            days=request.days
        )

        return EvolutionSummaryResponse(
            period_days=request.days,
            total_concepts=summary.total_concepts,
            concepts_by_state={
                UnderstandingState(k.value): v
                for k, v in summary.concepts_by_state.items()
            },
            new_concepts=summary.new_concepts,
            mastered_concepts=summary.mastered_concepts,
            forgotten_concepts=summary.forgotten_concepts,
            top_learning_areas=summary.top_learning_areas,
            learning_velocity=summary.learning_velocity
        )

    except Exception as e:
        logger.error(f"Failed to get evolution summary: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get evolution summary: {str(e)}"
        )


# ============================================================================
# Curiosity Engine Endpoints
# ============================================================================

@app.post("/api/v1/curiosity/suggest", response_model=CuriositySuggestionResponse, tags=["Curiosity"])
async def get_exploration_suggestions(
    request: CuriositySuggestionRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Get proactive exploration suggestions.

    Returns suggestions for interesting concepts to explore.
    """
    start_time = time.time()

    try:
        suggestions = await app_state.curiosity_engine.suggest_exploration(
            limit=request.limit
        )

        # Filter by importance threshold
        suggestions = [
            s for s in suggestions
            if s.importance >= request.importance_threshold
        ]

        # Convert to response format
        suggestion_results = [
            ExplorationSuggestion(
                type=ExplorationSuggestionType(s.type),
                concept=s.concept,
                reason=s.reason,
                importance=s.importance,
                suggested_query=s.suggested_query,
                expected_benefit=s.expected_benefit,
                metadata=s.metadata,
                created_at=s.created_at
            )
            for s in suggestions
        ]

        duration_ms = (time.time() - start_time) * 1000

        # Update stats
        app_state.stats["curiosity"]["total_suggestions"] += len(suggestion_results)
        for suggestion in suggestion_results:
            app_state.stats["curiosity"]["suggestions_by_type"][suggestion.type.value] += 1

        return CuriositySuggestionResponse(
            suggestions=suggestion_results,
            count=len(suggestion_results),
            generation_time_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Failed to generate suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


# ============================================================================
# Graph Reasoning Endpoints
# ============================================================================

@app.post("/api/v1/graph/multi-hop", response_model=MultiHopQueryResponse, tags=["Graph"])
async def multi_hop_query(
    request: MultiHopQueryRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Multi-hop graph reasoning.

    Traverses knowledge graph to find connected entities and memories.
    """
    start_time = time.time()

    try:
        # Perform multi-hop query
        result = await app_state.graph_reasoner.multi_hop_query(
            request.query,
            max_hops=request.max_hops,
            max_results=request.max_results
        )

        # Convert to response format
        result_items = [
            MultiHopResultItem(
                memory_id=mem.id,
                content=mem.text,
                relevance_score=score,
                hop_distance=0,  # TODO: Track hop distance
                reasoning_path=None  # TODO: Include path
            )
            for mem, score in result.all_results[:request.max_results]
        ]

        duration_ms = (time.time() - start_time) * 1000

        return MultiHopQueryResponse(
            query=request.query,
            max_hops=request.max_hops,
            results=result_items,
            total_paths_explored=len(result.reasoning_paths),
            query_entities=result.query_entities,
            latency_ms=duration_ms
        )

    except Exception as e:
        logger.error(f"Multi-hop query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-hop query failed: {str(e)}"
        )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "HoloLoom.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

"""
Helper Functions for API Endpoints
===================================

Common utility functions shared across API endpoint handlers.

Functions:
    load_memory_shards: Load fallback memory shards
    load_from_persistent_backend: Load from Neo4j/Qdrant
    get_orchestrator: Get or create orchestrator instance
    format_verification: Format verification response
    format_steps: Format reasoning steps response
"""

import logging
from typing import Optional, List, Dict

from hololoom.agentic import create_agentic_orchestrator
from hololoom.protocols.types import MemoryShard

from ..models import VerificationResponse, ReasoningStepResponse
from ..state import state

logger = logging.getLogger(__name__)


def load_memory_shards() -> List[MemoryShard]:
    """
    Load memory shards from data source (fallback when persistent backend unavailable).

    Returns:
        Example shards for development/fallback
    """
    return [
        MemoryShard(
            id="example_1",
            text="HoloLoom is a neural decision-making system with multi-scale embeddings.",
            episode="hololoom_basics",
            entities=["hololoom", "embeddings"],
            motifs=["definition"]
        )
    ]


async def load_from_persistent_backend() -> List[MemoryShard]:
    """
    Load memories from persistent backend (Neo4j/Qdrant).

    Returns:
        List of MemoryShard objects loaded from storage
    """
    if not state.memory_backend:
        return []

    try:
        from hololoom.memory.protocol import MemoryQuery

        query = MemoryQuery(
            text="",  # Empty query = retrieve all
            limit=1000
        )

        result = await state.memory_backend.retrieve(query)

        # Convert Memory objects to MemoryShard objects
        shards = []
        for memory in result.memories:
            shard = MemoryShard(
                id=memory.id,
                text=memory.text,
                episode=memory.context.get("episode", "default"),
                entities=memory.context.get("entities", []),
                motifs=memory.context.get("motifs", []),
                metadata=memory.metadata
            )
            shards.append(shard)

        return shards

    except Exception as e:
        logger.error(f"Failed to load from persistent backend: {e}")
        return []


async def get_orchestrator():
    """Get or create global orchestrator instance."""
    if state.orchestrator is None:
        logger.info("Initializing agentic orchestrator...")
        state.orchestrator = await create_agentic_orchestrator(
            state.config,
            state.shards,
            enable_verification=True,
            enable_goal_tracking=True,
            audit_trail=state.audit_trail,
            monitor=state.monitor
        )
        logger.info("Orchestrator ready!")

    return state.orchestrator


def format_verification(verification) -> Optional[VerificationResponse]:
    """Format verification result for API response."""
    if verification is None:
        return None

    return VerificationResponse(
        verified=verification.verified,
        confidence=verification.confidence,
        contradictions=verification.contradictions,
        supporting_evidence=verification.supporting_evidence,
        suggested_refinements=verification.suggested_refinements
    )


def format_steps(steps: List[Dict]) -> List[ReasoningStepResponse]:
    """Format reasoning steps for API response."""
    return [
        ReasoningStepResponse(
            type=step.get("type", "unknown"),
            query=step.get("query"),
            confidence=step.get("confidence"),
            finding=step.get("finding"),
            completed=step.get("completed"),
            tool=step.get("tool_used") or step.get("tool")
        )
        for step in steps
    ]

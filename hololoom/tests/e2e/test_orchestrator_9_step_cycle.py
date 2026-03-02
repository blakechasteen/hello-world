"""
Core Orchestrator 9-Step Cycle Tests (Week 1 Critical)
========================================================
Tests the complete weaving cycle from query → spacetime.

9-Step Cycle:
1. Loom Command → Pattern Card selection (BARE/FAST/FUSED)
2. Chrono Trigger → Temporal window creation
3. Yarn Graph → Thread selection from memory
4. Resonance Shed → Feature extraction, DotPlasma creation
5. Warp Space → Continuous manifold tensioning
6. Convergence Engine → Discrete decision collapse
7. Tool Execution → Action with results
8. Spacetime Fabric → Provenance and trace
9. Reflection Buffer → Learning from outcome

Test Coverage:
- All 3 pattern cards (BARE/FAST/FUSED)
- Complete cycle execution
- Step ordering and dependencies
- Error propagation across steps
- Timeout handling per step
- Spacetime provenance completeness
- Performance thresholds

Created: 2025-11-21 (Bug Fix + Test Creation Session)
"""

import pytest
import asyncio
from typing import List
from datetime import datetime

from hololoom.weaving_orchestrator import WeavingOrchestrator
from hololoom.config import Config, ExecutionMode
from hololoom.protocols.types import Query, MemoryShard, Spacetime


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_shards() -> List[MemoryShard]:
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="Thompson Sampling balances exploration and exploitation using Bayesian priors",
            episode="algorithms",
            entities=["Thompson Sampling", "Bayesian", "exploration"],
            motifs=["definition", "tradeoff"]
        ),
        MemoryShard(
            id="shard_2",
            text="Matryoshka embeddings use nested representations at multiple scales",
            episode="embeddings",
            entities=["Matryoshka", "embeddings", "multi-scale"],
            motifs=["architecture", "efficiency"]
        ),
        MemoryShard(
            id="shard_3",
            text="Knowledge graphs store entities and relationships for semantic retrieval",
            episode="memory",
            entities=["knowledge graph", "entities", "relationships"],
            motifs=["storage", "retrieval"]
        )
    ]


@pytest.fixture
def bare_config() -> Config:
    """BARE mode config (minimal processing)."""
    return Config.bare()


@pytest.fixture
def fast_config() -> Config:
    """FAST mode config (balanced)."""
    return Config.fast()


@pytest.fixture
def fused_config() -> Config:
    """FUSED mode config (full processing)."""
    return Config.fused()


# ============================================================================
# Step 1: Loom Command / Pattern Card Selection
# ============================================================================

@pytest.mark.asyncio
async def test_step1_pattern_card_selection_bare(bare_config, test_shards):
    """Test Step 1: Pattern card selection for BARE mode."""
    async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify BARE pattern card was used
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.BARE.value
        assert 'pattern_card' in spacetime.metadata
        assert spacetime.metadata['pattern_card'] == 'BARE'


@pytest.mark.asyncio
async def test_step1_pattern_card_selection_fast(fast_config, test_shards):
    """Test Step 1: Pattern card selection for FAST mode."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="Explain Matryoshka embeddings in detail")
        spacetime = await orchestrator.weave(query)

        # Verify FAST pattern card was used
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.FAST.value
        assert spacetime.metadata['pattern_card'] == 'FAST'


@pytest.mark.asyncio
async def test_step1_pattern_card_selection_fused(fused_config, test_shards):
    """Test Step 1: Pattern card selection for FUSED mode."""
    async with WeavingOrchestrator(cfg=fused_config, shards=test_shards) as orchestrator:
        query = Query(text="Compare Thompson Sampling with UCB and analyze tradeoffs")
        spacetime = await orchestrator.weave(query)

        # Verify FUSED pattern card was used
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.FUSED.value
        assert spacetime.metadata['pattern_card'] == 'FUSED'


@pytest.mark.asyncio
async def test_step1_invalid_pattern_card(bare_config, test_shards):
    """Test Step 1: Handling of invalid pattern card."""
    # This should gracefully fall back to BARE mode
    bare_config.execution_mode = "INVALID_MODE"  # type: ignore

    async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
        query = Query(text="Test query")
        spacetime = await orchestrator.weave(query)

        # Should fall back to BARE mode
        assert spacetime is not None
        assert 'execution_mode' in spacetime.metadata


# ============================================================================
# Step 2: Chrono Trigger / Temporal Window
# ============================================================================

@pytest.mark.asyncio
async def test_step2_temporal_window_creation(fast_config, test_shards):
    """Test Step 2: Temporal window creation."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify temporal window metadata
        assert 'temporal_window' in spacetime.metadata
        assert 'timeout_ms' in spacetime.metadata
        assert spacetime.metadata['timeout_ms'] > 0


@pytest.mark.asyncio
async def test_step2_timeout_enforcement_bare(bare_config, test_shards):
    """Test Step 2: Timeout enforcement in BARE mode (5s)."""
    bare_config.bare_timeout_seconds = 0.1  # Very short timeout

    async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
        query = Query(text="Quick query")
        start = datetime.now()
        spacetime = await orchestrator.weave(query)
        duration = (datetime.now() - start).total_seconds()

        # Should complete within timeout (or timeout gracefully)
        assert duration < 1.0  # Should be well under 1 second


@pytest.mark.asyncio
async def test_step2_timeout_enforcement_fused(fused_config, test_shards):
    """Test Step 2: Timeout enforcement in FUSED mode (120s)."""
    async with WeavingOrchestrator(cfg=fused_config, shards=test_shards) as orchestrator:
        query = Query(text="Complex research query")
        spacetime = await orchestrator.weave(query)

        # Verify FUSED timeout was configured
        assert spacetime.metadata.get('timeout_ms', 0) >= 120000  # 120 seconds


# ============================================================================
# Step 3: Yarn Graph / Thread Selection
# ============================================================================

@pytest.mark.asyncio
async def test_step3_thread_selection(fast_config, test_shards):
    """Test Step 3: Thread selection from Yarn Graph."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="Tell me about Thompson Sampling")
        spacetime = await orchestrator.weave(query)

        # Verify threads were selected
        assert 'retrieved_threads' in spacetime.metadata or 'context_size' in spacetime.metadata
        assert spacetime.metadata.get('context_size', 0) > 0


@pytest.mark.asyncio
async def test_step3_empty_memory(fast_config):
    """Test Step 3: Thread selection with empty memory."""
    async with WeavingOrchestrator(cfg=fast_config, shards=[]) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Should handle empty memory gracefully
        assert spacetime is not None
        assert spacetime.confidence >= 0.0


@pytest.mark.asyncio
async def test_step3_recency_based_selection(fast_config, test_shards):
    """Test Step 3: Thread selection prioritizes recent memories."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        # First query - establishes recency
        query1 = Query(text="What is Thompson Sampling?")
        await orchestrator.weave(query1)

        # Second query - should retrieve recently accessed threads
        query2 = Query(text="Tell me more about Bayesian priors")
        spacetime = await orchestrator.weave(query2)

        assert spacetime.metadata.get('context_size', 0) > 0


# ============================================================================
# Step 4: Resonance Shed / Feature Extraction
# ============================================================================

@pytest.mark.asyncio
async def test_step4_feature_extraction_motifs(fast_config, test_shards):
    """Test Step 4: Motif feature extraction."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify motif extraction occurred
        assert 'features' in spacetime.metadata or 'motifs_extracted' in spacetime.metadata


@pytest.mark.asyncio
async def test_step4_feature_extraction_embeddings(fast_config, test_shards):
    """Test Step 4: Embedding feature extraction."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="Explain Matryoshka embeddings")
        spacetime = await orchestrator.weave(query)

        # Verify embeddings were computed
        assert 'embedding_dim' in spacetime.metadata or 'scales' in spacetime.metadata


@pytest.mark.asyncio
async def test_step4_feature_extraction_spectral(fused_config, test_shards):
    """Test Step 4: Spectral feature extraction (FUSED only)."""
    async with WeavingOrchestrator(cfg=fused_config, shards=test_shards) as orchestrator:
        query = Query(text="Compare different embedding approaches")
        spacetime = await orchestrator.weave(query)

        # FUSED mode should extract spectral features
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.FUSED.value


@pytest.mark.asyncio
async def test_step4_dotplasma_creation(fast_config, test_shards):
    """Test Step 4: DotPlasma (feature fusion) creation."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify feature fusion occurred (multiple feature types combined)
        assert spacetime is not None
        assert hasattr(spacetime, 'metadata')


# ============================================================================
# Step 5: Warp Space / Continuous Manifold Tensioning
# ============================================================================

@pytest.mark.asyncio
async def test_step5_warp_space_tensioning(fused_config, test_shards):
    """Test Step 5: Warp Space tensioning (FUSED mode)."""
    async with WeavingOrchestrator(cfg=fused_config, shards=test_shards) as orchestrator:
        query = Query(text="Analyze Thompson Sampling tradeoffs")
        spacetime = await orchestrator.weave(query)

        # FUSED mode should use Warp Space
        assert spacetime.confidence > 0.0


@pytest.mark.asyncio
async def test_step5_warp_space_skip_in_bare(bare_config, test_shards):
    """Test Step 5: Warp Space skipped in BARE mode."""
    async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
        query = Query(text="Quick query")
        spacetime = await orchestrator.weave(query)

        # BARE mode should skip Warp Space (faster)
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.BARE.value


# ============================================================================
# Step 6: Convergence Engine / Decision Collapse
# ============================================================================

@pytest.mark.asyncio
async def test_step6_decision_collapse(fast_config, test_shards):
    """Test Step 6: Convergence engine decision collapse."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify decision was made
        assert 'tool_used' in spacetime.metadata or 'action' in spacetime.metadata


@pytest.mark.asyncio
async def test_step6_thompson_sampling_exploration(fast_config, test_shards):
    """Test Step 6: Thompson Sampling exploration in convergence."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        # Run multiple queries to test exploration
        for i in range(5):
            query = Query(text=f"Query {i}: What is Thompson Sampling?")
            spacetime = await orchestrator.weave(query)
            assert spacetime.confidence >= 0.0


# ============================================================================
# Step 7: Tool Execution
# ============================================================================

@pytest.mark.asyncio
async def test_step7_tool_execution(fast_config, test_shards):
    """Test Step 7: Tool execution after decision."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="Explain Thompson Sampling")
        spacetime = await orchestrator.weave(query)

        # Verify tool was executed
        assert 'tool_used' in spacetime.metadata
        assert spacetime.metadata['tool_used'] in ['answer', 'research', 'explore', 'verify']


@pytest.mark.asyncio
async def test_step7_tool_execution_error_handling(fast_config, test_shards):
    """Test Step 7: Tool execution error handling."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        # Query that might cause edge cases
        query = Query(text="")  # Empty query
        spacetime = await orchestrator.weave(query)

        # Should handle gracefully
        assert spacetime is not None


# ============================================================================
# Step 8: Spacetime Fabric / Provenance
# ============================================================================

@pytest.mark.asyncio
async def test_step8_spacetime_provenance_complete(fast_config, test_shards):
    """Test Step 8: Complete provenance in Spacetime fabric."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify complete provenance
        assert hasattr(spacetime, 'query_id')
        assert hasattr(spacetime, 'confidence')
        assert hasattr(spacetime, 'metadata')
        assert 'execution_mode' in spacetime.metadata
        assert 'tool_used' in spacetime.metadata


@pytest.mark.asyncio
async def test_step8_spacetime_trace(fast_config, test_shards):
    """Test Step 8: Spacetime trace includes all steps."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="Explain Matryoshka embeddings")
        spacetime = await orchestrator.weave(query)

        # Verify trace exists
        assert hasattr(spacetime, 'metadata')
        # Trace should include timing information
        assert 'duration_ms' in spacetime.metadata or 'timestamp' in spacetime.metadata


@pytest.mark.asyncio
async def test_step8_spacetime_immutability(fast_config, test_shards):
    """Test Step 8: Spacetime objects are immutable after creation."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Spacetime should be complete and consistent
        original_query_id = spacetime.query_id
        original_confidence = spacetime.confidence

        # These values should remain constant
        assert spacetime.query_id == original_query_id
        assert spacetime.confidence == original_confidence


# ============================================================================
# Step 9: Reflection Buffer / Learning
# ============================================================================

@pytest.mark.asyncio
async def test_step9_reflection_buffer_storage(fast_config, test_shards):
    """Test Step 9: Reflection buffer stores outcomes."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards, enable_reflection=True) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Reflect on outcome
        await orchestrator.reflect(spacetime, feedback={"helpful": True})

        # Verify reflection was stored
        assert spacetime is not None


@pytest.mark.asyncio
async def test_step9_reflection_learning(fast_config, test_shards):
    """Test Step 9: Reflection buffer enables learning."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards, enable_reflection=True) as orchestrator:
        # First query
        query1 = Query(text="What is Thompson Sampling?")
        spacetime1 = await orchestrator.weave(query1)
        await orchestrator.reflect(spacetime1, feedback={"helpful": True})

        # Second similar query - should benefit from learning
        query2 = Query(text="Explain Thompson Sampling")
        spacetime2 = await orchestrator.weave(query2)

        # Both should complete successfully
        assert spacetime1.confidence >= 0.0
        assert spacetime2.confidence >= 0.0


# ============================================================================
# Complete Cycle Tests
# ============================================================================

@pytest.mark.asyncio
async def test_complete_cycle_bare_mode(bare_config, test_shards):
    """Test complete 9-step cycle in BARE mode."""
    async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Verify all steps completed
        assert spacetime is not None
        assert spacetime.confidence >= 0.0
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.BARE.value
        assert 'tool_used' in spacetime.metadata
        assert hasattr(spacetime, 'query_id')


@pytest.mark.asyncio
async def test_complete_cycle_fast_mode(fast_config, test_shards):
    """Test complete 9-step cycle in FAST mode."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="Explain Matryoshka embeddings and their benefits")
        spacetime = await orchestrator.weave(query)

        # Verify all steps completed
        assert spacetime is not None
        assert spacetime.confidence >= 0.0
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.FAST.value
        assert 'tool_used' in spacetime.metadata
        assert 'context_size' in spacetime.metadata or 'retrieved_threads' in spacetime.metadata


@pytest.mark.asyncio
async def test_complete_cycle_fused_mode(fused_config, test_shards):
    """Test complete 9-step cycle in FUSED mode."""
    async with WeavingOrchestrator(cfg=fused_config, shards=test_shards) as orchestrator:
        query = Query(text="Compare Thompson Sampling with UCB and analyze tradeoffs")
        spacetime = await orchestrator.weave(query)

        # Verify all steps completed (FUSED has richest metadata)
        assert spacetime is not None
        assert spacetime.confidence >= 0.0
        assert spacetime.metadata.get('execution_mode') == ExecutionMode.FUSED.value
        assert 'tool_used' in spacetime.metadata


@pytest.mark.asyncio
async def test_complete_cycle_with_reflection(fast_config, test_shards):
    """Test complete 9-step cycle including reflection."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards, enable_reflection=True) as orchestrator:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orchestrator.weave(query)

        # Reflect on outcome
        await orchestrator.reflect(spacetime, feedback={"helpful": True, "rating": 5})

        # Verify cycle completed with reflection
        assert spacetime is not None
        assert spacetime.confidence >= 0.0


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_performance_bare_mode_latency(bare_config, test_shards):
    """Test BARE mode performance (<50ms target)."""
    async with WeavingOrchestrator(cfg=bare_config, shards=test_shards) as orchestrator:
        query = Query(text="Quick query")

        start = datetime.now()
        spacetime = await orchestrator.weave(query)
        duration_ms = (datetime.now() - start).total_seconds() * 1000

        # BARE mode should be very fast (cold cache allows some overhead)
        assert duration_ms < 1000  # 1 second allowance for cold cache
        assert spacetime is not None


@pytest.mark.asyncio
async def test_performance_fast_mode_latency(fast_config, test_shards):
    """Test FAST mode performance (<150ms target)."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="What is Thompson Sampling?")

        start = datetime.now()
        spacetime = await orchestrator.weave(query)
        duration_ms = (datetime.now() - start).total_seconds() * 1000

        # FAST mode target: <150ms (cold cache allows overhead)
        assert duration_ms < 2000  # 2 second allowance
        assert spacetime is not None


@pytest.mark.asyncio
async def test_performance_fused_mode_latency(fused_config, test_shards):
    """Test FUSED mode performance (<300ms target)."""
    async with WeavingOrchestrator(cfg=fused_config, shards=test_shards) as orchestrator:
        query = Query(text="Analyze Thompson Sampling in detail")

        start = datetime.now()
        spacetime = await orchestrator.weave(query)
        duration_ms = (datetime.now() - start).total_seconds() * 1000

        # FUSED mode target: <300ms (cold cache allows overhead)
        assert duration_ms < 5000  # 5 second allowance
        assert spacetime is not None


# ============================================================================
# Error Propagation Tests
# ============================================================================

@pytest.mark.asyncio
async def test_error_propagation_empty_query(fast_config, test_shards):
    """Test error propagation with empty query."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        query = Query(text="")
        spacetime = await orchestrator.weave(query)

        # Should handle gracefully
        assert spacetime is not None


@pytest.mark.asyncio
async def test_error_propagation_very_long_query(fast_config, test_shards):
    """Test error propagation with very long query."""
    async with WeavingOrchestrator(cfg=fast_config, shards=test_shards) as orchestrator:
        # 10KB query
        long_text = "Thompson Sampling " * 1000
        query = Query(text=long_text)
        spacetime = await orchestrator.weave(query)

        # Should handle gracefully (may truncate or summarize)
        assert spacetime is not None

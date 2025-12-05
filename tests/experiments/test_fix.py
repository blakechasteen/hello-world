"""Quick test of v1.0 validation fixes"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.config import Config
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.loom import command as loom_cmd


async def test_single_query():
    """Test a single query with fixed scales"""
    print("Testing v1.0 validation fixes...")

    # Create test shard
    shards = [
        MemoryShard(
            id="test",
            text="Thompson Sampling is a Bayesian approach",
            episode="test",
            entities=[],
            motifs=[]
        )
    ]

    # Configure
    config = Config.fused()
    scales = [768]
    config.scales = scales
    config.fusion_weights = {768: 1.0}
    config.loom_pattern = "bare"

    # Create embedder
    embedder = MatryoshkaEmbeddings(
        base_model_name="nomic-ai/nomic-embed-text-v1.5",
        sizes=scales
    )

    # Override pattern scales
    for pattern_spec in [loom_cmd.BARE_PATTERN, loom_cmd.FAST_PATTERN, loom_cmd.FUSED_PATTERN]:
        pattern_spec.scales = scales
        pattern_spec.fusion_weights = {768: 1.0}

    # Test weaving
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        orchestrator.embedder = embedder

        query = Query(text="What is Thompson Sampling?")
        print(f"\nWeaving: {query.text}")

        spacetime = await orchestrator.weave(query)

        print(f"✓ Success!")
        print(f"  Confidence: {spacetime.confidence}")
        print(f"  Tool: {spacetime.tool_used}")
        print(f"  Response: {spacetime.response[:100]}...")


if __name__ == "__main__":
    asyncio.run(test_single_query())

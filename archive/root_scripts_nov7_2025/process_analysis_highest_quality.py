"""
Process AGENT_SWARM_ANALYSIS.md with Highest Quality Settings

Uses HoloLoom's FUSED configuration with all optimizations enabled:
- Phase 5 compositional cache
- Full multi-scale embeddings (96, 192, 384 dimensions)
- Complete feature extraction
- Thompson Sampling for exploration
"""

import asyncio
from pathlib import Path
from HoloLoom.config import Config
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.documentation.types import Query, MemoryShard
import re

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """Chunk text by paragraphs with overlap."""
    # Split by double newline (paragraphs)
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para_size = len(para)

        if current_size + para_size > chunk_size and current_chunk:
            # Save current chunk
            chunks.append('\n\n'.join(current_chunk))

            # Keep last paragraph for overlap
            if overlap > 0 and len(current_chunk) > 0:
                current_chunk = [current_chunk[-1]]
                current_size = len(current_chunk[0])
            else:
                current_chunk = []
                current_size = 0

        current_chunk.append(para)
        current_size += para_size

    # Add final chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks

async def process_analysis_document():
    """Process the agent swarm analysis with highest quality settings."""

    # Read the analysis document
    analysis_path = Path("AGENT_SWARM_ANALYSIS.md")
    content = analysis_path.read_text(encoding='utf-8')

    print("=" * 80)
    print("PROCESSING AGENT_SWARM_ANALYSIS.md")
    print("Configuration: FUSED (Highest Quality)")
    print("=" * 80)
    print()

    # Create highest quality config
    config = Config.fused()

    # Enable all optimizations
    config.enable_linguistic_gate = True
    config.linguistic_mode = "both"
    config.use_compositional_cache = True
    config.enable_alignment = True
    # Disable safety guardrails for analysis queries (testing mode)
    config.enable_safety_guardrails = False

    print("[CONFIG] Configuration Settings:")
    print(f"  - Mode: FUSED (highest quality)")
    print(f"  - Scales: {config.scales}")
    print(f"  - Phase 5 Linguistic Gate: {config.enable_linguistic_gate}")
    print(f"  - Compositional Cache: {config.use_compositional_cache}")
    print(f"  - Alignment Framework: {config.enable_alignment}")
    print(f"  - Safety Guardrails: Testing mode (analysis queries bypassed)")
    print()

    # Create memory shards from the document
    print("[SHARDS] Creating memory shards from document...")

    # Chunk the text
    chunks = chunk_text(content, chunk_size=1500, overlap=200)

    # Create memory shards
    shards = []
    for i, chunk in enumerate(chunks):
        shard = MemoryShard(
            id=f"analysis_shard_{i}",
            text=chunk,
            episode="agent_swarm_analysis",
            entities=[],  # Would be extracted by entity recognition
            motifs=[],  # Would be extracted by motif detection
            metadata={
                'source': 'AGENT_SWARM_ANALYSIS.md',
                'chunk_id': i,
                'total_chunks': len(chunks),
                'importance_score': 0.8  # High importance for analysis document
            }
        )
        shards.append(shard)

    print(f"  [OK] Created {len(shards)} memory shards")
    print()

    # Print shard summaries
    print("[SHARDS] Shard Summaries:")
    for i, shard in enumerate(shards[:10], 1):  # Show first 10
        preview = shard.text[:100].replace('\n', ' ')
        # Remove Unicode characters that cause encoding issues
        preview = preview.encode('ascii', 'replace').decode('ascii')
        importance = shard.metadata.get('importance_score', 0.0)
        print(f"  {i}. [{importance:.2f}] {preview}...")

    if len(shards) > 10:
        print(f"  ... and {len(shards) - 10} more shards")
    print()

    # Create orchestrator with highest quality settings
    # (Guardrails automatically created in testing mode via config)
    print("[ORCHESTRATOR] Initializing WeavingOrchestrator...")
    async with WeavingOrchestrator(cfg=config, shards=shards) as orchestrator:
        print("  [OK] Orchestrator ready")
        print()

        # Process key queries about the analysis
        queries = [
            "What are the critical bottlenecks in HoloLoom?",
            "What optimizations should be prioritized?",
            "What is the completion status of the 9-layer architecture?",
            "What are the most impactful quick wins?",
            "How does Phase 5 compositional cache perform?"
        ]

        print("[QUERIES] Processing Key Queries:")
        print("=" * 80)
        print()

        results = []
        for query_text in queries:
            print(f"Query: {query_text}")
            print("-" * 80)

            query = Query(text=query_text)
            spacetime = await orchestrator.weave(query)

            # Extract results
            # Spacetime might have 'response' or 'text' attribute
            response_text = getattr(spacetime, 'text', None) or getattr(spacetime, 'response', 'No response available')

            result = {
                'query': query_text,
                'response': response_text,
                'confidence': spacetime.confidence,
                'tool_used': spacetime.metadata.get('tool_used', 'unknown'),
                'duration_ms': spacetime.metadata.get('duration_ms', 0.0),
                'cache_hit': spacetime.metadata.get('cache_hit', False)
            }
            results.append(result)

            # Print result
            print(f"Response: {str(result['response'])[:200]}...")
            print(f"Confidence: {result['confidence']:.3f}")
            print(f"Tool: {result['tool_used']}")
            print(f"Duration: {result['duration_ms']:.1f}ms")
            print(f"Cache: {'HIT [OK]' if result['cache_hit'] else 'MISS'}")
            print()

        print("=" * 80)
        print()

        # Summary statistics
        print("[STATS] Processing Summary:")
        print(f"  Total queries: {len(results)}")
        print(f"  Avg confidence: {sum(r['confidence'] for r in results) / len(results):.3f}")
        print(f"  Avg duration: {sum(r['duration_ms'] for r in results) / len(results):.1f}ms")
        cache_hits = sum(1 for r in results if r['cache_hit'])
        print(f"  Cache hit rate: {cache_hits}/{len(results)} ({cache_hits/len(results)*100:.1f}%)")
        print()

        # Extract key insights
        print("[INSIGHTS] Key Insights from Document:")
        print("=" * 80)

        insights_query = Query(
            text="Extract the top 5 actionable recommendations from this analysis"
        )
        insights_spacetime = await orchestrator.weave(insights_query)
        print(getattr(insights_spacetime, 'response', 'No response available'))
        print()

        # Architecture status
        print("[STATUS] Architecture Status:")
        print("=" * 80)

        status_query = Query(
            text="What is the overall completion status and what are the integration gaps?"
        )
        status_spacetime = await orchestrator.weave(status_query)
        print(getattr(status_spacetime, 'response', 'No response available'))
        print()

        # Performance analysis
        print("[PERFORMANCE] Performance Bottlenecks:")
        print("=" * 80)

        perf_query = Query(
            text="What are the critical performance bottlenecks and their fixes?"
        )
        perf_spacetime = await orchestrator.weave(perf_query)
        print(getattr(perf_spacetime, 'response', 'No response available'))
        print()

    print("=" * 80)
    print("[OK] PROCESSING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(process_analysis_document())

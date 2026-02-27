"""
End-to-End Test: Perplexity-Style Web Search
=============================================
Complete workflow from query -> search -> cite -> response
"""
import asyncio
from hololoom.agentic.web_research import WebResearchOrchestrator
from hololoom.config import Config
from hololoom.documentation.types import MemoryShard

async def main():
    print("=" * 80)
    print("END-TO-END WEB SEARCH TEST")
    print("=" * 80)
    print()
    
    # Setup
    print("Setting up system...")
    config = Config.fast()
    config.enable_agentic_reasoning = True
    config.enable_alignment = False  # Skip guardrails for demo
    
    # Sample memory
    shards = [
        MemoryShard(
            id="seed_1",
            text="Thompson Sampling is a Bayesian approach to multi-armed bandits.",
            entities=["Thompson Sampling", "Bayesian", "bandits"],
            motifs=["exploration", "exploitation"],
            metadata={"source": "seed"}
        )
    ]
    
    print("Config: FAST mode")
    print("Agentic reasoning: ENABLED")
    print("Memory: 1 seed shard")
    print()
    
    # Create orchestrator
    print("Creating WebResearchOrchestrator...")
    async with await WebResearchOrchestrator.create(
        config=config,
        shards=shards,
        enable_web_search=True,
        search_provider="mock"  # Use mock for deterministic results
    ) as orchestrator:
        print("Orchestrator ready")
        print()
        
        # Run query
        query = "What is Thompson Sampling and how does it work?"
        print(f"Query: {query}")
        print()
        
        print("Searching web + synthesizing response...")
        result = await orchestrator.research_web(
            query=query,
            max_web_results=5,
            enable_citations=True
        )
        
        # Display results
        print()
        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print()
        
        print(f"Total Duration: {result.total_duration_ms:.1f}ms")
        print(f"Web Search Time: {result.web_search_time_ms:.1f}ms")
        print(f"Search Results Found: {len(result.search_results)}")
        print(f"Memory Shards Created: {len(result.memory_shards)}")
        print()
        
        # Search results
        print("Top Search Results:")
        print("-" * 80)
        for i, sr in enumerate(result.search_results[:3], 1):
            print(f"[{i}] {sr.title}")
            print(f"    URL: {sr.url}")
            print(f"    Score: {sr.final_score:.3f}")
            print(f"    Snippet: {sr.snippet[:100]}...")
            print()
        
        # Response with citations
        print("Synthesized Response (with citations):")
        print("-" * 80)
        print(result.cited_response)
        print()
        
        # Verify citations present
        has_citations = any(f"[{i}]" in result.cited_response for i in range(1, 4))
        print(f"Citations present: {has_citations}")
        print()
        
        # Spacetime metadata
        print("Spacetime Metadata:")
        print(f"   - Confidence: {result.spacetime.confidence:.3f}")
        print(f"   - Tool used: {result.spacetime.metadata.get('tool_used', 'N/A')}")
        print(f"   - Duration: {result.spacetime.metadata.get('duration_ms', 0):.1f}ms")
        print()
        
        print("=" * 80)
        print("END-TO-END TEST COMPLETE!")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())

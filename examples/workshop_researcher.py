import asyncio
import sys
import os
from pathlib import Path

# Fix: Ensure we can import HoloLoom from the parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom import HoloLoom
import logging

# Configure logging to show the "weaving" process
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ResearcherDemo")

async def run_researcher_demo():
    print("\n" + "="*80)
    print("🚀 HOLOLOOM WORKFLOW DEMO: THE AUTOMATED RESEARCHER")
    print("="*80)
    print("Goal: Flow data through [Store] -> [Embed] -> [Recall]\n")

    # Fix: Use the async factory method
    loom = await HoloLoom.create(pattern="bare")
    
    # 1. THE DATA (Simulating multiple shards of knowledge)
    research_papers = [
        "Paper A: Neural networks can achieve multi-scale awareness through temporal weaving.",
        "Paper B: Matryoshka embeddings allow for adaptive vector dimensions in agentic recall.",
        "Paper C: Safety guardrails are essential for autonomous tool execution in distributed systems."
    ]

    print("Step 1: [Memory Store] Ingesting research papers...")
    for paper in research_papers:
        await loom.ingest_text(paper)
    print("✓ Ingestion complete.\n")

    print("Step 2: [Embedder] system has automatically vectorized the shards.")
    # The embedder is available on the weaver as 'embedder'
    # encode() is a sync method in MatryoshkaEmbeddings
    sample_vec = loom.weaver.embedder.encode(["Temporal Weaving"])[0]
    print(f"✓ Verified embedding vector for 'Temporal Weaving' (Dim: {len(sample_vec)})\n")

    print("Step 3: [Memory Search] Performing cross-context recall...")
    query = "How do we achieve safe autonomous awareness?"
    results = await loom.recall(query, limit=3)
    
    print(f"\n🔍 Query: '{query}'")
    print("-" * 40)
    for i, res in enumerate(results):
        print(f"[{i+1}] Relevance: {getattr(res, 'relevance', 'N/A'):.4f}")
        print(f"    Content: {res.text[:100]}...")
    print("-" * 40)

    print("\n✅ Researcher Demo Complete: The agents successfully flowed knowledge from ingestion to discovery.")

if __name__ == "__main__":
    asyncio.run(run_researcher_demo())

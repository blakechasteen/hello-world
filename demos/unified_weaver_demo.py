#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Weaver API Demo
========================
Demonstrates the clean, modern mythRL.Weaver interface.

This is the recommended API for all new code!

Author: mythRL Team
Date: 2025-10-27
"""

import asyncio
from mythRL import Weaver


async def main():
    print("=" * 80)
    print("UNIFIED WEAVER API DEMO")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Example 1: Quick Start - Fast Mode
    # ========================================================================
    print("[Example 1] Quick Start - Fast Mode")
    print("-" * 80)
    
    knowledge = """
    HoloLoom is a revolutionary neural decision-making system that combines
    symbolic and continuous representations through a weaving metaphor.
    
    Key components include:
    - Loom Command: Pattern selection
    - Chrono Trigger: Temporal control
    - Yarn Graph: Thread storage
    - Resonance Shed: Feature extraction
    - Warp Space: Continuous manifold
    - Convergence Engine: Decision collapse
    - Spacetime Fabric: Provenance tracking
    
    HoloLoom uses Thompson Sampling for exploration-exploitation balance
    and Matryoshka embeddings for multi-scale semantic representation.
    """
    
    weaver = await Weaver.create(mode='fast', knowledge=knowledge)
    print(f"✓ Created Weaver: {weaver}")
    print()
    
    # Query 1
    print("[Query 1] What is HoloLoom?")
    result = await weaver.query("What is HoloLoom?")
    print(f"Response: {result.response[:200]}...")
    print(f"Tool: {result.tool}, Confidence: {result.confidence:.2f}, Duration: {result.duration_ms:.1f}ms")
    print()
    
    # Query 2
    print("[Query 2] What components does it have?")
    result = await weaver.query("What components does it have?")
    print(f"Response: {result.response[:200]}...")
    print(f"Context used: {result.context_count} shards")
    print()
    
    await weaver.close()
    
    # ========================================================================
    # Example 2: Conversational Mode with Chat
    # ========================================================================
    print()
    print("[Example 2] Conversational Mode")
    print("-" * 80)
    
    weaver = await Weaver.create(mode='fast', knowledge=knowledge)
    
    # First message
    result = await weaver.chat("Tell me about Thompson Sampling")
    print(f"User: Tell me about Thompson Sampling")
    print(f"Weaver: {result.response[:150]}...")
    print()
    
    # Follow-up (maintains context)
    result = await weaver.chat("What about Matryoshka embeddings?")
    print(f"User: What about Matryoshka embeddings?")
    print(f"Weaver: {result.response[:150]}...")
    print()
    
    await weaver.close()
    
    # ========================================================================
    # Example 3: Different Modes
    # ========================================================================
    print()
    print("[Example 3] Comparing Modes")
    print("-" * 80)
    
    modes = ['lite', 'fast', 'full']
    question = "What is HoloLoom?"
    
    for mode in modes:
        weaver = await Weaver.create(mode=mode, knowledge=knowledge)
        result = await weaver.query(question)
        print(f"{mode.upper():8s} → {result.duration_ms:6.1f}ms (pattern={result.pattern})")
        await weaver.close()
    
    print()
    
    # ========================================================================
    # Example 4: Dynamic Knowledge Ingestion
    # ========================================================================
    print()
    print("[Example 4] Dynamic Knowledge Ingestion")
    print("-" * 80)
    
    weaver = await Weaver.create(mode='fast', knowledge=knowledge)
    print(f"Initial shards: {len(weaver._shards)}")
    
    # Ingest new knowledge
    new_knowledge = """
    Thompson Sampling is a probabilistic algorithm for the multi-armed bandit
    problem. It samples from posterior distributions to balance exploration
    and exploitation, achieving optimal regret bounds.
    """
    
    count = await weaver.ingest(new_knowledge)
    print(f"✓ Ingested {count} new shards")
    print(f"Total shards: {len(weaver._shards)}")
    print()
    
    # Query with new knowledge
    result = await weaver.query("Explain Thompson Sampling in detail")
    print(f"Response: {result.response[:200]}...")
    print()
    
    await weaver.close()
    
    # ========================================================================
    # Example 5: Context Manager Pattern
    # ========================================================================
    print()
    print("[Example 5] Context Manager Pattern")
    print("-" * 80)
    
    async with await Weaver.create(mode='fast', knowledge=knowledge) as weaver:
        result = await weaver.query("What is the weaving metaphor?")
        print(f"Response: {result.response[:150]}...")
        print("✓ Weaver automatically closed via context manager")
    
    print()
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  ✓ Simple API: Weaver.create() + query()")
    print("  ✓ Multiple modes: lite/fast/full/research")
    print("  ✓ Conversational: chat() maintains context")
    print("  ✓ Dynamic ingestion: Add knowledge on the fly")
    print("  ✓ Clean results: WeaverResult with all metadata")
    print()
    print("For production use:")
    print("  from mythRL import Weaver")
    print("  weaver = await Weaver.create(mode='full', memory_backend='neo4j_qdrant')")
    print("  result = await weaver.query('Your question')")
    print()
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Pipeline Demo (Simplified)
=====================================
Complete data flow without complex imports.

Demonstrates: Text → Memories → Store → Query → Response
"""

import asyncio
import sys
import io
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct module loading
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load hybrid memory store
hybrid_module = load_module("hybrid_neo4j_qdrant", "HoloLoom/memory/stores/hybrid_neo4j_qdrant.py")
HybridNeo4jQdrant = hybrid_module.HybridNeo4jQdrant
Memory = hybrid_module.Memory
MemoryQuery = hybrid_module.MemoryQuery
Strategy = hybrid_module.Strategy

# Load LoomCommand
loom_module = load_module("loom_command", "HoloLoom/loom/command.py")
LoomCommand = loom_module.LoomCommand
PatternCard = loom_module.PatternCard


# ============================================================================
# Simple Text Chunker (inline, no imports)
# ============================================================================

def chunk_text_by_paragraph(text: str, chunk_size: int = 300) -> List[str]:
    """Split text by paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_size = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        para_size = len(para)

        if para_size > chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            chunks.append(para)
        elif current_size + para_size > chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_size = para_size
        else:
            current_chunk.append(para)
            current_size += para_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def extract_entities(text: str) -> List[str]:
    """Extract capitalized words as entities."""
    common_words = {
        'The', 'This', 'That', 'These', 'Those', 'I', 'You', 'We', 'They',
        'A', 'An', 'It', 'He', 'She', 'What', 'When', 'Where', 'Why', 'How'
    }
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities = [w for w in words if w not in common_words and len(w) > 2]
    return list(set(entities))[:20]


# ============================================================================
# Text Ingestion Pipeline
# ============================================================================

class TextIngestionPipeline:
    """Simple pipeline: Text → Chunks → Memories → Store."""

    def __init__(self, memory_store: HybridNeo4jQdrant, loom_command: LoomCommand):
        self.memory = memory_store
        self.loom = loom_command

    async def ingest(
        self,
        text: str,
        source: str,
        user_id: str = "blake",
        chunk_size: int = 300
    ) -> List[Memory]:
        """Ingest text → memories."""

        print(f"\n{'='*80}")
        print(f"INGESTING: {source}")
        print(f"{'='*80}\n")

        # Step 1: Chunk text
        print("1️⃣  Chunking text by paragraph")
        chunks = chunk_text_by_paragraph(text, chunk_size)
        print(f"   → Created {len(chunks)} chunks")
        print(f"   → Avg size: {sum(len(c) for c in chunks) // len(chunks)} chars")

        # Step 2: Create memories
        print(f"\n2️⃣  Creating Memory objects")
        memories = []
        for i, chunk_text in enumerate(chunks):
            entities = extract_entities(chunk_text)
            tags = entities[:5] + ['text', f'chunk_{i}']

            mem_id = hashlib.md5(f"{source}_{i}".encode()).hexdigest()[:16]

            memory = Memory(
                id=mem_id,
                text=chunk_text,
                timestamp=datetime.now(),
                context={'user_id': user_id, 'tags': tags},
                metadata={
                    'source': source,
                    'chunk_index': i,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'entities': entities
                }
            )
            memories.append(memory)

        print(f"   → Created {len(memories)} memories")

        # Show sample
        if memories:
            mem = memories[0]
            print(f"   → Sample:")
            print(f"      ID: {mem.id}")
            print(f"      Text: {mem.text[:60]}...")
            print(f"      Tags: {mem.context.get('tags', [])[:3]}")

        # Step 3: Store
        print(f"\n3️⃣  Storing to Neo4j + Qdrant")
        stored_ids = []
        for mem in memories:
            mem_id = await self.memory.store(mem)
            stored_ids.append(mem_id)

        print(f"   → Stored {len(stored_ids)} memories")
        print(f"   → Neo4j: Graph + symbolic vectors")
        print(f"   → Qdrant: Semantic vectors")

        print(f"\n{'='*80}")
        print(f"✓ INGESTION COMPLETE")
        print(f"{'='*80}\n")

        return memories

    async def query(
        self,
        query_text: str,
        user_id: str = "blake",
        user_preference: str = None
    ) -> dict:
        """Query with pattern selection."""

        print(f"\n{'='*80}")
        print(f"QUERYING: {query_text[:60]}...")
        print(f"{'='*80}\n")

        # Step 1: Select pattern
        print("1️⃣  Pattern Selection")
        pattern = self.loom.select_pattern(query_text, user_preference)
        print(f"   → Pattern: {pattern.card.value.upper()}")

        # Step 2: Determine strategy
        print(f"\n2️⃣  Strategy Selection")
        if pattern.card == PatternCard.BARE:
            strategy = Strategy.GRAPH
            limit = 3
        elif pattern.card == PatternCard.FAST:
            strategy = Strategy.SEMANTIC
            limit = 5
        else:
            strategy = Strategy.FUSED
            limit = 7

        print(f"   → Strategy: {strategy}")
        print(f"   → Limit: {limit}")

        # Step 3: Retrieve
        print(f"\n3️⃣  Retrieval")
        query_obj = MemoryQuery(text=query_text, user_id=user_id, limit=limit)
        result = await self.memory.retrieve(query_obj, strategy)

        print(f"   → Retrieved: {len(result.memories)} memories")

        # Step 4: Show results
        print(f"\n4️⃣  Results")
        for i, (mem, score) in enumerate(zip(result.memories[:3], result.scores[:3]), 1):
            print(f"   {i}. [{score:.3f}] {mem.text[:70]}...")

        if len(result.memories) > 3:
            print(f"   ... and {len(result.memories) - 3} more")

        print(f"\n{'='*80}")
        print(f"✓ QUERY COMPLETE")
        print(f"{'='*80}\n")

        return {
            'pattern': pattern.card.value,
            'strategy': strategy,
            'memories': result.memories,
            'scores': result.scores
        }


# ============================================================================
# Demo
# ============================================================================

async def main():
    print("\n" + "="*80)
    print("🚀 END-TO-END PIPELINE DEMO")
    print("   Text → Memories → Store → Query")
    print("="*80 + "\n")

    # Initialize
    print("Initializing...")
    memory = HybridNeo4jQdrant(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="hololoom123",
        qdrant_url="http://localhost:6333"
    )

    loom = LoomCommand(default_pattern=PatternCard.FAST, auto_select=True)

    pipeline = TextIngestionPipeline(memory_store=memory, loom_command=loom)

    print("✓ Ready!\n")

    # ========================================================================
    # PHASE 1: INGEST
    # ========================================================================

    print("\n" + "#"*80)
    print("PHASE 1: DATA INGESTION")
    print("#"*80)

    beekeeping_text = """
# Winter Preparation for Weak Hives

## Critical Steps

Weak colonies require special attention. Hives with fewer than 8 frames of brood need close monitoring.

## Insulation

Insulation wraps are essential for weak hives. They help maintain internal temperature during cold months. The north apiary has high cold exposure and requires extra protection.

## Feeding

Sugar fondant should be placed above the cluster starting in November. Weak colonies need supplemental feeding through winter. Check stores every 2-3 weeks.

## Hive Jodi

Hive Jodi is located in the north apiary which has high cold exposure. This hive shows weakness with only 8 frames of brood. Special attention required.

## Equipment

Mouse guards should be installed before first frost. Entrance reducers help weak hives defend against robbing. Top insulation reduces heat loss.

## Monitoring

Weekly checks through October. Bi-weekly checks November-December. Monthly checks January-March. Look for activity on warm days.
"""

    memories = await pipeline.ingest(
        text=beekeeping_text,
        source="winter_prep.md",
        chunk_size=300
    )

    print(f"\n📊 Summary: {len(memories)} memories stored")

    # ========================================================================
    # PHASE 2: QUERY
    # ========================================================================

    print("\n\n" + "#"*80)
    print("PHASE 2: RETRIEVAL")
    print("#"*80)

    test_queries = [
        ("What does Hive Jodi need?", None, "Auto-select"),
        ("insulation", "bare", "BARE mode"),
        ("How do I prepare weak hives?", "fused", "FUSED mode")
    ]

    results = []
    for query_text, pref, desc in test_queries:
        print(f"\n\n{'─'*80}")
        print(f"TEST: {desc}")
        print(f"{'─'*80}")

        result = await pipeline.query(query_text, user_preference=pref)
        results.append(result)

        print(f"\n📊 Analysis:")
        print(f"   Pattern: {result['pattern'].upper()}")
        print(f"   Memories: {len(result['memories'])}")
        print(f"   Avg score: {sum(result['scores']) / len(result['scores']):.3f}")

    # ========================================================================
    # PHASE 3: SUMMARY
    # ========================================================================

    print("\n\n" + "#"*80)
    print("PHASE 3: SUMMARY")
    print("#"*80)

    # Health check
    print("\n🏥 Memory Health:")
    health = await memory.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Neo4j: {health['neo4j']['memories']} memories")
    print(f"   Qdrant: {health['qdrant']['memories']} memories")

    # Metrics
    print("\n✨ Quality:")
    all_scores = [score for r in results for score in r['scores']]
    avg = sum(all_scores) / len(all_scores)
    print(f"   Avg relevance: {avg:.3f}")
    print(f"   Highly relevant (>0.4): {sum(1 for s in all_scores if s > 0.4)}/{len(all_scores)}")

    # Cleanup
    memory.close()

    # Done
    print("\n\n" + "="*80)
    print("✅ PIPELINE COMPLETE!")
    print("="*80)
    print("\nDemonstrated:")
    print("1. ✅ Text → Chunks (paragraph-based)")
    print("2. ✅ Chunks → Memories (with entities/tags)")
    print("3. ✅ Memories → Store (Neo4j + Qdrant)")
    print("4. ✅ Pattern selection (BARE/FAST/FUSED)")
    print("5. ✅ Strategy retrieval (GRAPH/SEMANTIC/FUSED)")
    print("\nResult:")
    print(f"   {len(memories)} memories ingested")
    print(f"   {len(results)} queries executed")
    print(f"   {avg:.1%} average relevance")
    print(f"   {health['neo4j']['memories']} total in graph")
    print("\n✨ Full HoloLoom data pipeline OPERATIONAL! ✨")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())

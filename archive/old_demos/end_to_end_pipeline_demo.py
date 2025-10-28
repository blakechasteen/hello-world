#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Pipeline Demo
========================
Complete data flow: Text → Shards → Memories → Store → Query → Response

This demonstrates the FULL HoloLoom MVP pipeline:
1. TextSpinner: Raw text → MemoryShards (chunking, entity extraction)
2. Memory Creation: MemoryShards → Memory objects (with embeddings)
3. Storage: Dual-write to Neo4j (graph) + Qdrant (vectors)
4. Retrieval: Pattern card → Strategy → Context
5. Full Cycle: Query → Pattern → Memory → Response

This is the complete "data ingestion + retrieval" pipeline.
"""

import asyncio
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import List

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct module loading to avoid package import hell
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load BaseSpinner first (for text spinner dependency)
base_spinner_module = load_module("base_spinner", "HoloLoom/spinningWheel/base.py")
sys.modules['HoloLoom.spinning_wheel.base'] = base_spinner_module

# Load TextSpinner
text_spinner_module = load_module("text_spinner", "HoloLoom/spinningWheel/text.py")
TextSpinner = text_spinner_module.TextSpinner
TextSpinnerConfig = text_spinner_module.TextSpinnerConfig
MemoryShard = text_spinner_module.MemoryShard

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
# MemoryShard → Memory Converter
# ============================================================================

class ShardToMemoryConverter:
    """
    Converts MemoryShards (from spinners) to Memory objects (for storage).

    MemoryShard: Raw ingested data with entities/motifs
    Memory: Structured memory with timestamp, user_id, tags
    """

    @staticmethod
    def convert(shard: MemoryShard, user_id: str = "blake") -> Memory:
        """
        Convert MemoryShard → Memory.

        Args:
            shard: MemoryShard from spinner
            user_id: User identifier

        Returns:
            Memory object ready for storage
        """
        # Extract tags from entities and metadata
        tags = []

        # Add entities as tags
        if shard.entities:
            tags.extend(shard.entities[:5])  # Top 5 entities

        # Add source type as tag
        if shard.metadata and 'format' in shard.metadata:
            tags.append(shard.metadata['format'])

        # Add chunk info as tag if chunked
        if shard.metadata and 'chunk_by' in shard.metadata:
            tags.append(f"chunked_{shard.metadata['chunk_by']}")

        # Build memory metadata from shard metadata
        memory_metadata = {}
        if shard.metadata:
            memory_metadata = {
                'shard_id': shard.id,
                'episode': shard.episode,
                'source': shard.metadata.get('source', 'unknown'),
                'char_count': shard.metadata.get('char_count', len(shard.text)),
                'word_count': shard.metadata.get('word_count', len(shard.text.split())),
            }

            # Preserve chunk info
            if 'chunk_index' in shard.metadata:
                memory_metadata['chunk_index'] = shard.metadata['chunk_index']
            if 'chunk_by' in shard.metadata:
                memory_metadata['chunk_by'] = shard.metadata['chunk_by']

        memory = Memory(
            id=shard.id,
            text=shard.text,
            user_id=user_id,
            timestamp=datetime.now(),
            tags=tags,
            metadata=memory_metadata
        )

        return memory


# ============================================================================
# Complete End-to-End Pipeline
# ============================================================================

class EndToEndPipeline:
    """
    Complete pipeline: Text → Shards → Memories → Store → Query.

    This is the FULL data ingestion + retrieval flow.
    """

    def __init__(
        self,
        memory_store: HybridNeo4jQdrant,
        loom_command: LoomCommand,
        text_spinner_config: TextSpinnerConfig = None
    ):
        """
        Initialize pipeline.

        Args:
            memory_store: Hybrid Neo4j + Qdrant store
            loom_command: Pattern card selector
            text_spinner_config: Optional TextSpinner config
        """
        self.memory = memory_store
        self.loom = loom_command
        self.converter = ShardToMemoryConverter()

        # Default TextSpinner config
        if text_spinner_config is None:
            text_spinner_config = TextSpinnerConfig(
                chunk_by='paragraph',
                chunk_size=500,
                extract_entities=True,
                enable_enrichment=False
            )

        self.spinner = TextSpinner(text_spinner_config)

    async def ingest_text(
        self,
        text: str,
        source: str,
        user_id: str = "blake",
        metadata: dict = None
    ) -> List[Memory]:
        """
        Ingest text document into memory store.

        Flow:
        1. Text → TextSpinner → MemoryShards
        2. MemoryShards → Memory objects
        3. Memory objects → Neo4j + Qdrant

        Args:
            text: Raw text content
            source: Source identifier (filename, URL, etc.)
            user_id: User identifier
            metadata: Additional metadata

        Returns:
            List of Memory objects stored
        """
        print(f"\n{'='*80}")
        print(f"INGESTING TEXT: {source}")
        print(f"{'='*80}\n")

        # Step 1: Text → Shards
        print("1️⃣  TextSpinner: Text → MemoryShards")
        raw_data = {
            'text': text,
            'source': source,
            'metadata': metadata or {}
        }

        shards = await self.spinner.spin(raw_data)
        print(f"   → Created {len(shards)} shards")
        print(f"   → Total chars: {sum(len(s.text) for s in shards)}")
        print(f"   → Avg shard size: {sum(len(s.text) for s in shards) // len(shards)} chars")

        # Step 2: Shards → Memory objects
        print(f"\n2️⃣  Converter: MemoryShards → Memory objects")
        memories = [self.converter.convert(shard, user_id) for shard in shards]
        print(f"   → Converted {len(memories)} memories")

        # Show sample
        if memories:
            mem = memories[0]
            print(f"   → Sample memory:")
            print(f"      ID: {mem.id}")
            print(f"      Text: {mem.text[:60]}...")
            print(f"      Tags: {mem.tags[:3]}...")
            print(f"      Metadata: {list(mem.metadata.keys())}")

        # Step 3: Store → Neo4j + Qdrant
        print(f"\n3️⃣  Storage: Dual-write to Neo4j + Qdrant")
        stored_ids = []
        for mem in memories:
            mem_id = await self.memory.store(mem)
            stored_ids.append(mem_id)

        print(f"   → Stored {len(stored_ids)} memories")
        print(f"   → Neo4j: Graph relationships")
        print(f"   → Qdrant: Vector embeddings")

        print(f"\n{'='*80}")
        print(f"✓ INGESTION COMPLETE: {len(memories)} memories stored")
        print(f"{'='*80}\n")

        return memories

    async def query_with_pattern(
        self,
        query_text: str,
        user_id: str = "blake",
        user_preference: str = None
    ) -> dict:
        """
        Query with pattern card selection.

        Flow:
        1. LoomCommand selects pattern
        2. Pattern determines memory strategy
        3. Retrieve from Neo4j + Qdrant
        4. Return context

        Args:
            query_text: Query text
            user_id: User identifier
            user_preference: Optional pattern preference

        Returns:
            Dict with pattern, strategy, memories, scores
        """
        print(f"\n{'='*80}")
        print(f"QUERYING: {query_text[:60]}...")
        print(f"{'='*80}\n")

        # Step 1: Select pattern
        print("1️⃣  Pattern Selection")
        pattern = self.loom.select_pattern(
            query_text=query_text,
            user_preference=user_preference
        )
        print(f"   → Pattern: {pattern.card.value.upper()}")

        # Step 2: Determine strategy
        print(f"\n2️⃣  Strategy Determination")
        if pattern.card == PatternCard.BARE:
            strategy = Strategy.GRAPH
            limit = 3
        elif pattern.card == PatternCard.FAST:
            strategy = Strategy.SEMANTIC
            limit = 5
        else:  # FUSED
            strategy = Strategy.FUSED
            limit = 7

        print(f"   → Strategy: {strategy}")
        print(f"   → Limit: {limit} memories")

        # Step 3: Retrieve
        print(f"\n3️⃣  Memory Retrieval")
        query = MemoryQuery(text=query_text, user_id=user_id, limit=limit)
        result = await self.memory.retrieve(query, strategy)

        print(f"   → Retrieved: {len(result.memories)} memories")
        print(f"   → Strategy used: {result.strategy_used}")

        # Show results
        print(f"\n4️⃣  Retrieved Context")
        for i, (mem, score) in enumerate(zip(result.memories[:3], result.scores[:3]), 1):
            print(f"   {i}. [{score:.3f}] {mem.text[:70]}...")

        if len(result.memories) > 3:
            print(f"   ... and {len(result.memories) - 3} more")

        print(f"\n{'='*80}")
        print(f"✓ QUERY COMPLETE: {len(result.memories)} memories retrieved")
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
    """Run end-to-end pipeline demo."""

    print("\n" + "="*80)
    print("🚀 END-TO-END PIPELINE DEMO")
    print("   Text → Shards → Memories → Store → Query → Response")
    print("="*80 + "\n")

    # Initialize components
    print("Initializing components...")
    memory = HybridNeo4jQdrant(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="hololoom123",
        qdrant_url="http://localhost:6333"
    )

    loom = LoomCommand(
        default_pattern=PatternCard.FAST,
        auto_select=True
    )

    # TextSpinner config: paragraph chunking
    spinner_config = TextSpinnerConfig(
        chunk_by='paragraph',
        chunk_size=300,
        extract_entities=True,
        enable_enrichment=False
    )

    pipeline = EndToEndPipeline(
        memory_store=memory,
        loom_command=loom,
        text_spinner_config=spinner_config
    )

    print("✓ Initialization complete!\n")

    # ========================================================================
    # PHASE 1: INGEST BEEKEEPING KNOWLEDGE
    # ========================================================================

    print("\n" + "#"*80)
    print("PHASE 1: DATA INGESTION")
    print("#"*80)

    beekeeping_text = """
# Beekeeping Winter Preparation Guide

## Weak Hive Management

Weak colonies require special attention during winter preparation. Hives with fewer than 8 frames of brood should be monitored closely.

### Critical Steps for Weak Hives

Insulation is essential for weak hives. Wrapping hives with insulation material helps maintain internal temperature. The north apiary experiences high cold exposure and requires extra protection.

### Feeding Strategy

Sugar fondant should be placed above the cluster starting in November. Weak colonies need supplemental feeding through late fall and winter. Check sugar stores every 2-3 weeks.

### Location Considerations

Hive Jodi is located in the north apiary which has particularly high cold exposure. This hive has shown signs of weakness with only 8 frames of brood. Special attention is required.

## Equipment Needs

Mouse guards should be installed before first frost. Entrance reducers help weak hives defend against robbing. Top insulation reduces heat loss through the crown board.

### Ventilation Balance

Proper ventilation prevents moisture buildup while maintaining warmth. Upper entrances allow moisture to escape without creating drafts through the cluster.

## Monitoring Schedule

Weekly checks through October. Bi-weekly checks through November and December. Monthly checks January through March. Look for signs of activity on warm days.
"""

    # Ingest the text
    memories = await pipeline.ingest_text(
        text=beekeeping_text,
        source="winter_preparation_guide.md",
        user_id="blake",
        metadata={
            'category': 'beekeeping',
            'topic': 'winter_prep',
            'season': 'fall',
            'year': 2025
        }
    )

    print(f"\n📊 Ingestion Summary:")
    print(f"   Total memories stored: {len(memories)}")
    print(f"   Source: winter_preparation_guide.md")
    print(f"   Format: Markdown document, paragraph-chunked")

    # ========================================================================
    # PHASE 2: QUERY WITH DIFFERENT PATTERNS
    # ========================================================================

    print("\n\n" + "#"*80)
    print("PHASE 2: RETRIEVAL WITH PATTERN CARDS")
    print("#"*80)

    test_queries = [
        {
            "query": "What does weak Hive Jodi need for winter?",
            "preference": None,  # Auto-select
            "description": "Auto-select pattern (should be FAST)"
        },
        {
            "query": "insulation",
            "preference": "bare",  # Force BARE
            "description": "Force BARE pattern (fast graph-only)"
        },
        {
            "query": "How should I prepare weak hives for winter with proper feeding, insulation, and monitoring?",
            "preference": "fused",  # Force FUSED
            "description": "Force FUSED pattern (comprehensive hybrid)"
        }
    ]

    results = []
    for i, test in enumerate(test_queries, 1):
        print(f"\n\n{'─'*80}")
        print(f"QUERY {i}/{len(test_queries)}: {test['description']}")
        print(f"{'─'*80}")

        result = await pipeline.query_with_pattern(
            query_text=test['query'],
            user_preference=test['preference']
        )

        results.append(result)

        # Analyze relevance
        print(f"\n📊 Result Analysis:")
        print(f"   Pattern used: {result['pattern'].upper()}")
        print(f"   Strategy: {result['strategy']}")
        print(f"   Memories: {len(result['memories'])}")
        print(f"   Avg score: {sum(result['scores']) / len(result['scores']):.3f}")
        print(f"   Top score: {max(result['scores']):.3f}")

    # ========================================================================
    # PHASE 3: SUMMARY AND STATISTICS
    # ========================================================================

    print("\n\n" + "#"*80)
    print("PHASE 3: SESSION SUMMARY")
    print("#"*80)

    # Memory health
    print("\n🏥 Memory Store Health:")
    health = await memory.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Neo4j memories: {health['neo4j']['memories']}")
    print(f"   Qdrant memories: {health['qdrant']['memories']}")

    # Pattern distribution
    print("\n📈 Pattern Selection:")
    patterns_used = [r['pattern'] for r in results]
    for pattern in set(patterns_used):
        count = patterns_used.count(pattern)
        print(f"   {pattern.upper()}: {count} queries")

    # Strategy distribution
    print("\n🎯 Strategy Usage:")
    strategies_used = [r['strategy'] for r in results]
    for strategy in set(strategies_used):
        count = strategies_used.count(strategy)
        print(f"   {strategy}: {count} queries")

    # Quality metrics
    print("\n✨ Quality Metrics:")
    all_scores = [score for r in results for score in r['scores']]
    avg_score = sum(all_scores) / len(all_scores)
    max_score = max(all_scores)
    min_score = min(all_scores)
    print(f"   Avg relevance: {avg_score:.3f}")
    print(f"   Max relevance: {max_score:.3f}")
    print(f"   Min relevance: {min_score:.3f}")
    print(f"   Highly relevant (>0.4): {sum(1 for s in all_scores if s > 0.4)} / {len(all_scores)}")

    # Cleanup
    memory.close()

    # ========================================================================
    # CONCLUSION
    # ========================================================================

    print("\n\n" + "="*80)
    print("✅ END-TO-END PIPELINE COMPLETE!")
    print("="*80)
    print("\nWhat We Demonstrated:")
    print("1. ✅ Text ingestion → MemoryShards (TextSpinner)")
    print("2. ✅ MemoryShards → Memory objects (Converter)")
    print("3. ✅ Dual-write to Neo4j + Qdrant (HybridStore)")
    print("4. ✅ Pattern card selection (LoomCommand)")
    print("5. ✅ Strategy-based retrieval (GRAPH/SEMANTIC/FUSED)")
    print("6. ✅ Context → Features → Response (simulated)")
    print("\nResult:")
    print(f"   {len(memories)} memories ingested from markdown document")
    print(f"   {len(results)} queries executed with different patterns")
    print(f"   {avg_score:.1%} average retrieval relevance")
    print(f"   {health['neo4j']['memories']} total memories in graph")
    print(f"   {health['qdrant']['memories']} total memories in vectors")
    print("\n✨ The full HoloLoom data pipeline is OPERATIONAL! ✨")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())

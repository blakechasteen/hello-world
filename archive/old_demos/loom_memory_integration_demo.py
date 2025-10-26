#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loom Memory Integration Demo
=============================
Demonstrates how Pattern Cards determine memory retrieval strategy
and enforce token budgets in the full loom cycle.

This is the MVP integration showing:
1. LoomCommand selects Pattern Card (BARE/FAST/FUSED)
2. Pattern Card determines memory strategy
3. Memory retrieval respects token budgets
4. Results fed into loom cycle

Architecture:
    User Query
        ↓
    LoomCommand (selects Pattern Card)
        ↓
    Pattern Card → Memory Strategy + Token Budget
        ↓
    HybridMemoryStore (retrieves with strategy)
        ↓
    Token budget enforced
        ↓
    Context → Features → Policy → Response
"""

import asyncio
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

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

# Load LoomCommand
loom_module = load_module("loom_command", "HoloLoom/loom/command.py")
LoomCommand = loom_module.LoomCommand
PatternCard = loom_module.PatternCard
PatternSpec = loom_module.PatternSpec

# Load hybrid memory store
hybrid_module = load_module("hybrid_neo4j_qdrant", "HoloLoom/memory/stores/hybrid_neo4j_qdrant.py")
HybridNeo4jQdrant = hybrid_module.HybridNeo4jQdrant
Memory = hybrid_module.Memory
MemoryQuery = hybrid_module.MemoryQuery
RetrievalResult = hybrid_module.RetrievalResult
Strategy = hybrid_module.Strategy


# ============================================================================
# Pattern Card → Memory Strategy Mapping
# ============================================================================

class MemoryConfig:
    """
    Memory configuration derived from Pattern Card.

    Maps Pattern Card execution mode to:
    - Memory retrieval strategy
    - Result limit
    - Token budget
    """

    @staticmethod
    def from_pattern_card(pattern: PatternSpec) -> Dict[str, Any]:
        """
        Derive memory configuration from pattern card.

        Args:
            pattern: PatternSpec from LoomCommand

        Returns:
            Dict with strategy, limit, max_tokens
        """
        if pattern.card == PatternCard.BARE:
            return {
                "strategy": Strategy.GRAPH,  # Fast symbolic only
                "limit": 3,
                "max_tokens": 500,
                "description": "BARE mode: Minimal graph retrieval"
            }

        elif pattern.card == PatternCard.FAST:
            return {
                "strategy": Strategy.SEMANTIC,  # Vector similarity
                "limit": 5,
                "max_tokens": 1000,
                "description": "FAST mode: Semantic vector search"
            }

        else:  # FUSED
            return {
                "strategy": Strategy.FUSED,  # Hybrid fusion
                "limit": 7,
                "max_tokens": 2000,
                "description": "FUSED mode: Hybrid graph + semantic"
            }


# ============================================================================
# Token Budget Enforcement
# ============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation).

    Real implementation would use tiktoken or similar.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 chars per token
    return len(text) // 4


def enforce_token_budget(
    memories: List[Memory],
    scores: List[float],
    max_tokens: int
) -> tuple[List[Memory], List[float], int]:
    """
    Enforce token budget by truncating results.

    Args:
        memories: Retrieved memories
        scores: Relevance scores
        max_tokens: Maximum allowed tokens

    Returns:
        (truncated_memories, truncated_scores, actual_tokens)
    """
    truncated_memories = []
    truncated_scores = []
    total_tokens = 0

    for mem, score in zip(memories, scores):
        mem_tokens = estimate_tokens(mem.text)

        if total_tokens + mem_tokens <= max_tokens:
            truncated_memories.append(mem)
            truncated_scores.append(score)
            total_tokens += mem_tokens
        else:
            # Budget exceeded, stop here
            break

    return truncated_memories, truncated_scores, total_tokens


# ============================================================================
# Loom Cycle with Memory Integration
# ============================================================================

class LoomWithMemory:
    """
    Loom cycle integrating Pattern Cards with Memory.

    This is the MVP showing complete flow:
    Query → Pattern → Strategy → Memory → Features → Response
    """

    def __init__(
        self,
        memory_store: HybridNeo4jQdrant,
        loom_command: LoomCommand
    ):
        """
        Initialize loom with memory.

        Args:
            memory_store: Hybrid memory store
            loom_command: Loom command for pattern selection
        """
        self.memory = memory_store
        self.loom = loom_command

        # Cycle statistics
        self.cycle_history: List[Dict[str, Any]] = []

    async def process_query(
        self,
        query_text: str,
        user_id: str = "blake",
        user_preference: str = None
    ) -> Dict[str, Any]:
        """
        Process query through full loom cycle.

        Steps:
        1. LoomCommand selects Pattern Card
        2. Derive memory configuration from pattern
        3. Retrieve memories with selected strategy
        4. Enforce token budget
        5. Extract features (simulated)
        6. Make decision (simulated)
        7. Generate response (simulated)

        Args:
            query_text: User query
            user_id: User identifier
            user_preference: Optional pattern preference

        Returns:
            Dict with cycle results
        """
        print(f"\n{'='*80}")
        print(f"LOOM CYCLE: {query_text[:60]}...")
        print(f"{'='*80}\n")

        # Step 1: Select Pattern Card
        print("1️⃣  Pattern Selection")
        pattern = self.loom.select_pattern(
            query_text=query_text,
            user_preference=user_preference
        )
        print(f"   → Pattern: {pattern.card.value.upper()}")
        print(f"   → Scales: {pattern.scales}")
        print(f"   → Quality target: {pattern.quality_target:.1f}")
        print(f"   → Timeout: {pattern.pipeline_timeout:.1f}s")

        # Step 2: Derive memory configuration
        print(f"\n2️⃣  Memory Configuration")
        mem_config = MemoryConfig.from_pattern_card(pattern)
        print(f"   → Strategy: {mem_config['strategy']}")
        print(f"   → Limit: {mem_config['limit']} memories")
        print(f"   → Token budget: {mem_config['max_tokens']} tokens")
        print(f"   → {mem_config['description']}")

        # Step 3: Retrieve memories
        print(f"\n3️⃣  Memory Retrieval")
        query = MemoryQuery(
            text=query_text,
            user_id=user_id,
            limit=mem_config['limit']
        )

        start_time = datetime.now()
        result = await self.memory.retrieve(
            query=query,
            strategy=mem_config['strategy']
        )
        retrieval_ms = (datetime.now() - start_time).total_seconds() * 1000

        print(f"   → Retrieved: {len(result.memories)} memories")
        print(f"   → Strategy used: {result.strategy_used}")
        print(f"   → Latency: {retrieval_ms:.1f}ms")

        # Step 4: Enforce token budget
        print(f"\n4️⃣  Token Budget Enforcement")
        before_count = len(result.memories)
        truncated_mems, truncated_scores, actual_tokens = enforce_token_budget(
            result.memories,
            result.scores,
            mem_config['max_tokens']
        )
        after_count = len(truncated_mems)

        print(f"   → Before: {before_count} memories")
        print(f"   → After: {after_count} memories")
        print(f"   → Actual tokens: {actual_tokens} / {mem_config['max_tokens']}")

        budget_status = "✓ WITHIN BUDGET" if actual_tokens <= mem_config['max_tokens'] else "✗ EXCEEDED"
        print(f"   → Status: {budget_status}")

        # Step 5: Show retrieved context
        print(f"\n5️⃣  Retrieved Context")
        for i, (mem, score) in enumerate(zip(truncated_mems[:3], truncated_scores[:3]), 1):
            tokens = estimate_tokens(mem.text)
            print(f"   {i}. [{score:.3f}] (~{tokens} tokens)")
            print(f"      {mem.text[:70]}...")

        if len(truncated_mems) > 3:
            print(f"   ... and {len(truncated_mems) - 3} more")

        # Step 6-7: Simulated feature extraction and decision making
        print(f"\n6️⃣  Feature Extraction (simulated)")
        print(f"   → Motifs: {pattern.motif_mode}")
        print(f"   → Embeddings: {pattern.scales}")
        print(f"   → Spectral: {'enabled' if pattern.enable_spectral else 'disabled'}")

        print(f"\n7️⃣  Decision Making (simulated)")
        print(f"   → Policy complexity: {pattern.policy_complexity}")
        print(f"   → Transformer layers: {pattern.n_transformer_layers}")
        print(f"   → Attention heads: {pattern.n_attention_heads}")

        # Record cycle
        cycle_result = {
            "query": query_text,
            "pattern": pattern.card.value,
            "strategy": mem_config['strategy'],
            "memories_retrieved": before_count,
            "memories_used": after_count,
            "actual_tokens": actual_tokens,
            "token_budget": mem_config['max_tokens'],
            "within_budget": actual_tokens <= mem_config['max_tokens'],
            "retrieval_latency_ms": retrieval_ms,
            "top_scores": truncated_scores[:3]
        }

        self.cycle_history.append(cycle_result)

        print(f"\n{'='*80}")
        print(f"✓ CYCLE COMPLETE")
        print(f"{'='*80}\n")

        return cycle_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get cycle statistics."""
        if not self.cycle_history:
            return {"total_cycles": 0}

        # Token efficiency stats
        total_tokens = sum(c['actual_tokens'] for c in self.cycle_history)
        avg_tokens = total_tokens / len(self.cycle_history)
        budget_compliance = sum(1 for c in self.cycle_history if c['within_budget'])

        # Latency stats
        total_latency = sum(c['retrieval_latency_ms'] for c in self.cycle_history)
        avg_latency = total_latency / len(self.cycle_history)

        # Pattern distribution
        pattern_counts = {}
        for cycle in self.cycle_history:
            pattern = cycle['pattern']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        return {
            "total_cycles": len(self.cycle_history),
            "avg_tokens_per_cycle": avg_tokens,
            "budget_compliance_rate": budget_compliance / len(self.cycle_history),
            "avg_retrieval_latency_ms": avg_latency,
            "pattern_distribution": pattern_counts
        }


# ============================================================================
# Demo
# ============================================================================

async def main():
    """Run loom memory integration demo."""

    print("\n" + "="*80)
    print("🚀 LOOM MEMORY INTEGRATION DEMO")
    print("   Pattern Cards → Memory Strategy → Token Budgets")
    print("="*80 + "\n")

    # Initialize memory store
    print("Initializing hybrid memory store...")
    memory = HybridNeo4jQdrant(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="hololoom123",
        qdrant_url="http://localhost:6333"
    )

    # Initialize loom command
    print("Initializing loom command...")
    loom = LoomCommand(
        default_pattern=PatternCard.FAST,
        auto_select=True
    )

    # Create integrated loom
    integrated_loom = LoomWithMemory(
        memory_store=memory,
        loom_command=loom
    )

    print("\n✓ Initialization complete!\n")

    # Test queries with different patterns
    test_cases = [
        {
            "query": "What winter prep does weak Hive Jodi need?",
            "preference": None,  # Auto-select (should choose FAST)
            "expected": "FAST"
        },
        {
            "query": "sugar",  # Short query
            "preference": None,  # Auto-select (should choose BARE)
            "expected": "BARE"
        },
        {
            "query": "Comprehensive winter strategy for all weak hives based on historical patterns, including insulation, feeding, ventilation, and mouse guard installation timing",
            "preference": None,  # Long query, but auto-select still picks FAST
            "expected": "FAST"
        },
        {
            "query": "What winter prep does weak Hive Jodi need?",
            "preference": "fused",  # Force FUSED
            "expected": "FUSED"
        }
    ]

    print("="*80)
    print("TEST CASES")
    print("="*80)

    for i, test in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'#'*80}")

        result = await integrated_loom.process_query(
            query_text=test['query'],
            user_preference=test['preference']
        )

        # Validate expectations
        if result['pattern'] == test['expected'].lower():
            print(f"✓ Pattern selection correct: {test['expected']}")
        else:
            print(f"⚠️  Expected {test['expected']}, got {result['pattern']}")

    # Show statistics
    print("\n\n" + "="*80)
    print("📊 SESSION STATISTICS")
    print("="*80 + "\n")

    stats = integrated_loom.get_statistics()
    print(f"Total cycles: {stats['total_cycles']}")
    print(f"Avg tokens per cycle: {stats['avg_tokens_per_cycle']:.1f}")
    print(f"Budget compliance: {stats['budget_compliance_rate']*100:.1f}%")
    print(f"Avg retrieval latency: {stats['avg_retrieval_latency_ms']:.1f}ms")
    print(f"\nPattern distribution:")
    for pattern, count in stats['pattern_distribution'].items():
        print(f"  {pattern.upper()}: {count}")

    # Memory health check
    print("\n" + "="*80)
    print("🏥 MEMORY HEALTH CHECK")
    print("="*80 + "\n")

    health = await memory.health_check()
    print(f"Status: {health['status']}")
    print(f"Neo4j memories: {health['neo4j']['memories']}")
    print(f"Qdrant memories: {health['qdrant']['memories']}")

    # Cleanup
    memory.close()

    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Pattern Cards automatically determine memory strategy")
    print("2. Token budgets are enforced per execution mode")
    print("3. BARE: Fast graph-only (~500 tokens)")
    print("4. FAST: Semantic vectors (~1000 tokens)")
    print("5. FUSED: Hybrid fusion (~2000 tokens)")
    print("6. System adapts retrieval to available compute")
    print("\nNext: Integrate with full orchestrator pipeline")
    print("="*80 + "\n")


if __name__ == '__main__':
    asyncio.run(main())

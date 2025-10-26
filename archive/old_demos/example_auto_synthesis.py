#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Synthesis from Conversational Memory
==========================================
Demonstrates automatic training data synthesis from accumulated conversations.

Workflow:
  1. Have conversations with ConversationalAutoLoom
  2. Important exchanges are auto-filtered and remembered
  3. Periodically mine patterns from accumulated signal
  4. Synthesize training data
  5. Export for fine-tuning or few-shot prompting

The self-improving loop: Chat → Filter → Mine → Train → Improve
"""

import asyncio
import sys
import io
from pathlib import Path
from datetime import datetime

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Direct imports
import importlib.util

def load_module(module_name, file_path, dependencies=None):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    if dependencies:
        for dep_name, dep_module in dependencies.items():
            sys.modules[dep_name] = dep_module
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load synthesis modules
base_dir = Path(__file__).parent / "HoloLoom" / "synthesis"
enriched_mod = load_module("enriched_memory", base_dir / "enriched_memory.py")
sys.modules["HoloLoom.synthesis.enriched_memory"] = enriched_mod

extractor_mod = load_module(
    "pattern_extractor",
    base_dir / "pattern_extractor.py",
    dependencies={"HoloLoom.synthesis.enriched_memory": enriched_mod}
)
sys.modules["HoloLoom.synthesis.pattern_extractor"] = extractor_mod

synthesizer_mod = load_module(
    "data_synthesizer",
    base_dir / "data_synthesizer.py",
    dependencies={
        "HoloLoom.synthesis.enriched_memory": enriched_mod,
        "HoloLoom.synthesis.pattern_extractor": extractor_mod
    }
)

# Import classes
MemoryEnricher = enriched_mod.MemoryEnricher
PatternExtractor = extractor_mod.PatternExtractor
DataSynthesizer = synthesizer_mod.DataSynthesizer
SynthesisConfig = synthesizer_mod.SynthesisConfig


# ============================================================================
# Mock Conversational System (Simulates ConversationalAutoLoom)
# ============================================================================

class MockConversationTurn:
    """Mock conversation turn."""
    def __init__(self, turn_id, user_input, system_output, importance, timestamp):
        self.turn_id = turn_id
        self.user_input = user_input
        self.system_output = system_output
        self.importance_score = importance
        self.timestamp = timestamp

    def to_text(self):
        return f"""
Conversation Turn {self.turn_id} ({self.timestamp})

User: {self.user_input}

System: {self.system_output}

Importance: {self.importance_score:.2f}
"""


class MockConversationalLoom:
    """Mock conversational system for demo."""
    def __init__(self):
        self.conversation_history = []
        self.turn_counter = 0

    def add_turn(self, user_input, system_output, importance):
        """Add a conversation turn."""
        turn = MockConversationTurn(
            turn_id=self.turn_counter,
            user_input=user_input,
            system_output=system_output,
            importance=importance,
            timestamp=datetime.now().isoformat()
        )
        self.conversation_history.append(turn)
        self.turn_counter += 1

    def get_history(self, min_importance=0.0):
        """Get filtered conversation history."""
        return [
            turn for turn in self.conversation_history
            if turn.importance_score >= min_importance
        ]


# ============================================================================
# Synthesis Integration
# ============================================================================

async def mine_training_data(
    loom: MockConversationalLoom,
    min_importance: float = 0.4,
    min_confidence: float = 0.5,
    output_file: str = "auto_synthesis.jsonl"
):
    """
    Mine training data from conversational memory.

    Args:
        loom: Conversational system with accumulated history
        min_importance: Minimum importance threshold for filtering
        min_confidence: Minimum pattern confidence for synthesis
        output_file: Output JSONL file

    Returns:
        Number of training examples generated
    """

    print("=" * 70)
    print("MINING TRAINING DATA FROM CONVERSATIONS")
    print("=" * 70)
    print()

    # Step 1: Get filtered signal (important turns only)
    important_turns = loom.get_history(min_importance=min_importance)

    print(f"Conversation Stats:")
    print(f"  Total Turns: {len(loom.conversation_history)}")
    print(f"  Important Turns: {len(important_turns)} (>= {min_importance} importance)")
    print(f"  Signal Rate: {len(important_turns)/len(loom.conversation_history)*100:.1f}%")
    print()

    if not important_turns:
        print("No important conversations to mine!")
        return 0

    # Step 2: Convert to raw memory format
    raw_memories = [
        {
            'id': f"conv_{turn.turn_id}",
            'text': turn.to_text(),
            'timestamp': turn.timestamp,
            'importance': turn.importance_score,
            'metadata': {
                'user_input': turn.user_input,
                'system_output': turn.system_output
            }
        }
        for turn in important_turns
    ]

    # Step 3: Enrich memories
    print("Enriching memories...")
    enricher = MemoryEnricher()
    enriched_memories = [enricher.enrich(m) for m in raw_memories]
    print(f"✓ Enriched {len(enriched_memories)} memories")
    print()

    # Step 4: Extract patterns
    print("Extracting patterns...")
    extractor = PatternExtractor(min_confidence=min_confidence)
    patterns = extractor.extract_patterns(enriched_memories)

    # Count by type
    pattern_counts = {}
    for p in patterns:
        ptype = p.pattern_type.value
        pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1

    print(f"✓ Extracted {len(patterns)} patterns:")
    for ptype, count in pattern_counts.items():
        print(f"  {ptype}: {count}")
    print()

    if not patterns:
        print("No patterns extracted!")
        return 0

    # Step 5: Synthesize training data
    print("Synthesizing training examples...")
    config = SynthesisConfig(
        include_reasoning=True,
        include_context=True,
        min_confidence=min_confidence,
        system_prompt="You are a helpful AI assistant trained on high-quality technical conversations."
    )
    synthesizer = DataSynthesizer(config)
    examples = synthesizer.synthesize(patterns)

    print(f"✓ Generated {len(examples)} training examples")
    print()

    # Step 6: Get statistics
    stats = synthesizer.export_statistics(examples)

    print("Training Data Quality:")
    print(f"  Total Examples: {stats['total_examples']}")
    print(f"  Average Length: {stats['avg_length']:.0f} chars")
    print(f"  Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"  High Confidence (>= 0.7): {stats['high_confidence_count']} ({stats['high_confidence_count']/stats['total_examples']*100:.1f}%)")
    print()

    # Step 7: Export
    output_path = Path(__file__).parent / "synthesis_output" / output_file
    output_path.parent.mkdir(exist_ok=True)

    synthesizer.export_jsonl(examples, str(output_path), format='alpaca')
    print(f"✓ Exported to: {output_path}")
    print()

    return len(examples)


async def demo_periodic_synthesis():
    """Demonstrate periodic synthesis from ongoing conversations."""

    print("\n" + "=" * 70)
    print("AUTO-SYNTHESIS DEMO: Self-Improving Conversational AI")
    print("=" * 70)
    print()

    # Create mock conversational system
    loom = MockConversationalLoom()

    # Simulate some conversations
    conversations = [
        # SIGNAL - Technical questions
        ("What is Thompson Sampling?", "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem...", 0.88),
        ("How does the policy engine work?", "The policy engine uses neural networks with transformer blocks and cross-attention...", 0.75),
        ("Explain Matryoshka embeddings", "Matryoshka embeddings are hierarchical representations where smaller dimensions nest inside larger ones...", 0.82),

        # NOISE - Trivial exchanges
        ("thanks", "You're welcome!", 0.18),
        ("ok", "Great!", 0.12),

        # SIGNAL - More technical
        ("Compare BARE and FUSED modes", "BARE mode is minimal processing while FUSED uses all features. BARE is ~5x faster but FUSED has higher quality...", 0.68),
        ("If I increase embedding dimension, what happens?", "If you increase embedding dimension, retrieval quality improves but computational cost increases...", 0.65),

        # NOISE
        ("hi", "Hello! How can I help?", 0.15),

        # SIGNAL
        ("How to set up the knowledge graph?", "Here are the steps: 1) Install NetworkX, 2) Create KG instance, 3) Add entities...", 0.78),
    ]

    print("PHASE 1: Having Conversations")
    print("-" * 70)
    for user_in, sys_out, importance in conversations:
        loom.add_turn(user_in, sys_out, importance)
        status = "SIGNAL ✓" if importance >= 0.4 else "NOISE ✗"
        print(f"[{status}] User: {user_in[:40]}... (importance: {importance:.2f})")

    print()
    print(f"Total conversations: {len(conversations)}")
    signal_count = sum(1 for _, _, imp in conversations if imp >= 0.4)
    print(f"Signal: {signal_count} | Noise: {len(conversations) - signal_count}")
    print()

    # Phase 2: Mine training data
    print("\nPHASE 2: Mining Training Data")
    print("-" * 70)
    example_count = await mine_training_data(
        loom,
        min_importance=0.4,
        min_confidence=0.5,
        output_file="auto_synthesis_demo.jsonl"
    )

    # Phase 3: Show the self-improving loop
    print("=" * 70)
    print("THE SELF-IMPROVING LOOP")
    print("=" * 70)
    print(f"""
    1. Chat with ConversationalAutoLoom
       → {len(conversations)} total exchanges
       → {signal_count} important (>= 0.4)
       → {len(conversations) - signal_count} noise filtered

    2. Automatic filtering
       → Signal: {signal_count/len(conversations)*100:.1f}% remembered
       → Noise: {(len(conversations)-signal_count)/len(conversations)*100:.1f}% discarded

    3. Pattern mining
       → Extracted learnable patterns from signal
       → Q&A pairs, reasoning chains, definitions

    4. Training data synthesis
       → Generated {example_count} training examples
       → Average confidence: high
       → Ready for fine-tuning

    5. [Future] Fine-tune local model
       → Train on YOUR reasoning patterns
       → Domain-specific knowledge captured
       → Self-supervised learning

    6. [Future] Deploy improved model
       → Better at YOUR domain
       → Understands YOUR problem-solving approach
       → Continuous improvement from usage

    The loop: Chat → Filter → Mine → Train → Improve → Repeat
    """)


async def demo_incremental_synthesis():
    """Show incremental synthesis as conversations accumulate."""

    print("\n" + "=" * 70)
    print("INCREMENTAL SYNTHESIS: Mining as You Go")
    print("=" * 70)
    print()

    loom = MockConversationalLoom()

    # Simulate conversation batches
    batches = [
        # Batch 1: Initial conversations
        [
            ("What is Thompson Sampling?", "Thompson Sampling is...", 0.88),
            ("How does it work?", "It maintains Beta distributions...", 0.75),
            ("thanks", "You're welcome!", 0.18),
        ],
        # Batch 2: More conversations
        [
            ("Explain Matryoshka embeddings", "Matryoshka embeddings are...", 0.82),
            ("ok", "Great!", 0.12),
            ("Compare BARE and FUSED", "BARE is minimal, FUSED is full...", 0.68),
        ],
        # Batch 3: Even more
        [
            ("How to set up knowledge graph?", "Here are the steps...", 0.78),
            ("What about spectral features?", "Spectral features use graph Laplacian...", 0.72),
        ]
    ]

    total_examples = 0

    for batch_num, batch in enumerate(batches, 1):
        print(f"\nBATCH {batch_num}: Adding {len(batch)} conversations")
        print("-" * 70)

        # Add conversations
        for user_in, sys_out, importance in batch:
            loom.add_turn(user_in, sys_out, importance)
            status = "✓" if importance >= 0.4 else "✗"
            print(f"  [{status}] {user_in[:50]}...")

        # Mine training data from accumulated conversations
        print(f"\n  Mining training data from {len(loom.conversation_history)} total conversations...")
        example_count = await mine_training_data(
            loom,
            min_importance=0.4,
            min_confidence=0.5,
            output_file=f"incremental_batch_{batch_num}.jsonl"
        )

        total_examples += example_count
        print(f"  ✓ Total training examples so far: {total_examples}")
        print()

    print("=" * 70)
    print(f"FINAL RESULT: {total_examples} training examples from {len(loom.conversation_history)} conversations")
    print("=" * 70)


async def main():
    """Run auto-synthesis demonstrations."""

    print("\n" + "=" * 70)
    print("AUTO-SYNTHESIS: Self-Improving AI from Conversation")
    print("=" * 70)

    # Demo 1: Periodic synthesis
    await demo_periodic_synthesis()

    # Demo 2: Incremental synthesis
    await demo_incremental_synthesis()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The auto-synthesis system enables continuous learning:

1. **Automatic Filtering**: ConversationalAutoLoom filters signal from noise
   - Important exchanges (>= 0.4) are remembered
   - Trivial exchanges are discarded

2. **Pattern Mining**: Extract learnable patterns from accumulated signal
   - Q&A pairs
   - Reasoning chains
   - Definitions and procedures

3. **Training Data Synthesis**: Convert patterns to training examples
   - Alpaca/ChatML format
   - High confidence (>= 0.7)
   - YOUR reasoning patterns

4. **Continuous Improvement**: Periodically fine-tune on accumulated data
   - Daily/weekly synthesis runs
   - Incremental training
   - Domain adaptation

The result: An AI that learns from YOUR conversations, captures YOUR
reasoning, and continuously improves from meaningful interactions.

Signal → Patterns → Training → Intelligence → Better Conversations → Repeat 🎯
    """)


if __name__ == '__main__':
    asyncio.run(main())

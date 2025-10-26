#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesis Pipeline Demo
=======================
Complete demonstration of extracting training data from filtered conversations.

Flow:
  Raw Conversations (with importance scores)
    ↓
  Filter Signal (>= 0.4 importance)
    ↓
  Enrich Memories (extract entities, relationships, reasoning)
    ↓
  Extract Patterns (Q&A, reasoning chains, causal, decisions)
    ↓
  Synthesize Training Data (Alpaca/ChatML format)
    ↓
  Export to JSONL for fine-tuning

The gold mine: YOUR filtered signal → Training data → Intelligence
"""

import asyncio
import sys
import io
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Direct imports to avoid path issues
import importlib.util

def load_module(module_name, file_path, dependencies=None):
    """Load a Python module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    # Pre-load dependencies into sys.modules so relative imports work
    if dependencies:
        for dep_name, dep_module in dependencies.items():
            sys.modules[dep_name] = dep_module

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load synthesis modules in dependency order
base_dir = Path(__file__).parent / "HoloLoom" / "synthesis"

# 1. Load enriched_memory first (no dependencies)
enriched_mod = load_module("enriched_memory", base_dir / "enriched_memory.py")
sys.modules["HoloLoom.synthesis.enriched_memory"] = enriched_mod

# 2. Load pattern_extractor (depends on enriched_memory)
extractor_mod = load_module(
    "pattern_extractor",
    base_dir / "pattern_extractor.py",
    dependencies={"HoloLoom.synthesis.enriched_memory": enriched_mod}
)
sys.modules["HoloLoom.synthesis.pattern_extractor"] = extractor_mod

# 3. Load data_synthesizer (depends on pattern_extractor)
synthesizer_mod = load_module(
    "data_synthesizer",
    base_dir / "data_synthesizer.py",
    dependencies={
        "HoloLoom.synthesis.enriched_memory": enriched_mod,
        "HoloLoom.synthesis.pattern_extractor": extractor_mod
    }
)

# Import classes
EnrichedMemory = enriched_mod.EnrichedMemory
ReasoningType = enriched_mod.ReasoningType
MemoryEnricher = enriched_mod.MemoryEnricher

Pattern = extractor_mod.Pattern
PatternType = extractor_mod.PatternType
PatternExtractor = extractor_mod.PatternExtractor

TrainingExample = synthesizer_mod.TrainingExample
SynthesisConfig = synthesizer_mod.SynthesisConfig
DataSynthesizer = synthesizer_mod.DataSynthesizer


# ============================================================================
# Sample Filtered Conversations (Signal Only, >= 0.4 importance)
# ============================================================================

SAMPLE_CONVERSATIONS = [
    {
        'turn_id': 1,
        'user_input': 'What is Thompson Sampling and how does it work?',
        'system_output': 'Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It works by maintaining Beta distributions for each action, sampling from these distributions to select actions, and updating the distributions based on observed rewards. This naturally balances exploration and exploitation.',
        'importance': 0.88,
        'timestamp': '2025-10-23T10:15:00'
    },
    {
        'turn_id': 3,
        'user_input': 'How does the policy engine decide which tool to use?',
        'system_output': 'The policy engine uses a neural network with transformer blocks and cross-attention to context. It processes the query features, attends to relevant memory shards, and outputs logits for each tool. The decision is then made using Thompson Sampling for exploration-exploitation balance.',
        'importance': 0.75,
        'timestamp': '2025-10-23T10:17:00'
    },
    {
        'turn_id': 5,
        'user_input': 'If I increase the embedding dimension, what happens to retrieval quality?',
        'system_output': 'If you increase the embedding dimension, retrieval quality generally improves because the model can capture more nuanced semantic information. However, this comes with increased computational cost and memory usage.',
        'importance': 0.68,
        'timestamp': '2025-10-23T10:20:00'
    },
    {
        'turn_id': 7,
        'user_input': 'Given that I want fast responses, should I use BARE or FAST mode?',
        'system_output': 'Based on your need for fast responses, you should use BARE mode. It uses regex-only motifs, single-scale embeddings, and simple policy, which minimizes latency. FAST mode adds neural features and multi-scale processing, which is slower.',
        'importance': 0.72,
        'timestamp': '2025-10-23T10:23:00'
    },
    {
        'turn_id': 9,
        'user_input': 'Compare FAST mode and FUSED mode',
        'system_output': 'FAST mode is balanced with 2 scales, hybrid motifs, and neural policy - good for most use cases. FUSED mode uses all 3 scales, full NLP, spectral features, and hierarchical policy - highest quality but slowest. FAST is ~2x faster than FUSED while maintaining good accuracy.',
        'importance': 0.65,
        'timestamp': '2025-10-23T10:25:00'
    },
    {
        'turn_id': 11,
        'user_input': 'How to set up the knowledge graph?',
        'system_output': 'Here are the steps:\n1. Install NetworkX: pip install networkx\n2. Create KG instance: from HoloLoom.memory.graph import KG; kg = KG()\n3. Add entities: kg.add_entity("HoloLoom", entity_type="system")\n4. Add relationships: kg.add_relationship("HoloLoom", "uses", "Thompson Sampling")\n5. Query the graph: subgraph = kg.get_subgraph(["HoloLoom"], depth=2)',
        'importance': 0.78,
        'timestamp': '2025-10-23T10:28:00'
    },
    {
        'turn_id': 13,
        'user_input': 'What is Matryoshka embedding?',
        'system_output': 'Matryoshka embedding is a hierarchical embedding representation where smaller dimensions are nested inside larger ones. In HoloLoom, we use 96, 192, and 384 dimensions, where the 96-dim embedding is the first 96 values of the 384-dim embedding. This allows multi-scale retrieval and efficient storage.',
        'importance': 0.82,
        'timestamp': '2025-10-23T10:30:00'
    },
]


async def demo_enrichment():
    """Step 1: Enrich raw memories with extracted structure."""

    print("=" * 70)
    print("STEP 1: MEMORY ENRICHMENT")
    print("=" * 70)
    print("\nExtracting entities, relationships, and reasoning patterns...\n")

    enricher = MemoryEnricher()
    enriched_memories: List[EnrichedMemory] = []

    for conv in SAMPLE_CONVERSATIONS:
        # Convert conversation to raw memory dict
        raw_memory = {
            'id': f"conv_{conv['turn_id']}",
            'text': f"User: {conv['user_input']}\n\nSystem: {conv['system_output']}",
            'timestamp': conv['timestamp'],
            'importance': conv['importance'],
            'metadata': {
                'user_input': conv['user_input'],
                'system_output': conv['system_output']
            }
        }

        # Enrich
        enriched = enricher.enrich(raw_memory)
        enriched_memories.append(enriched)

        # Display
        print(f"Memory {enriched.id} (Importance: {enriched.importance:.2f})")
        print(f"  Type: {enriched.reasoning_type.value}")
        print(f"  Entities: {', '.join(enriched.entities[:5])}")
        print(f"  Topics: {', '.join(enriched.topics[:3])}")
        print(f"  Relationships: {len(enriched.relationships)}")
        print()

    print(f"✓ Enriched {len(enriched_memories)} memories\n")
    return enriched_memories


async def demo_pattern_extraction(enriched_memories: List[EnrichedMemory]):
    """Step 2: Extract learnable patterns from enriched memories."""

    print("=" * 70)
    print("STEP 2: PATTERN EXTRACTION")
    print("=" * 70)
    print("\nMining Q&A pairs, reasoning chains, causal relationships...\n")

    extractor = PatternExtractor(min_confidence=0.5)
    patterns = extractor.extract_patterns(enriched_memories)

    # Group by pattern type
    by_type: Dict[PatternType, List[Pattern]] = {}
    for pattern in patterns:
        if pattern.pattern_type not in by_type:
            by_type[pattern.pattern_type] = []
        by_type[pattern.pattern_type].append(pattern)

    # Display
    for ptype, pattern_list in by_type.items():
        print(f"{ptype.value.upper()}: {len(pattern_list)} patterns")

        # Show first example
        if pattern_list:
            p = pattern_list[0]
            print(f"  Example (confidence: {p.confidence:.2f}):")

            if ptype == PatternType.QA_PAIR:
                print(f"    Q: {p.content['question'][:60]}...")
                print(f"    A: {p.content['answer'][:60]}...")
            elif ptype == PatternType.REASONING_CHAIN:
                print(f"    Premise: {p.content['premise'][:50]}...")
                print(f"    Steps: {len(p.content['steps'])}")
                print(f"    Conclusion: {p.content['conclusion'][:50]}...")
            elif ptype == PatternType.CAUSAL:
                print(f"    Cause: {p.content['cause'][:50]}...")
                print(f"    Effect: {p.content['effect'][:50]}...")
            elif ptype == PatternType.DECISION:
                print(f"    Context: {p.content['context'][:50]}...")
                print(f"    Decision: {p.content['decision'][:50]}...")
            elif ptype == PatternType.COMPARISON:
                print(f"    Items: {p.content['item_a']} vs {p.content['item_b']}")
            elif ptype == PatternType.PROCEDURE:
                print(f"    Task: {p.content['task'][:50]}...")
                print(f"    Steps: {len(p.content['steps'])}")
            elif ptype == PatternType.DEFINITION:
                print(f"    Term: {p.content['term']}")
                print(f"    Definition: {p.content['definition'][:50]}...")

            print()

    print(f"✓ Extracted {len(patterns)} total patterns\n")
    return patterns


async def demo_synthesis(patterns: List[Pattern]):
    """Step 3: Synthesize training data from patterns."""

    print("=" * 70)
    print("STEP 3: DATA SYNTHESIS")
    print("=" * 70)
    print("\nConverting patterns to training examples...\n")

    config = SynthesisConfig(
        include_reasoning=True,
        include_context=True,
        min_confidence=0.5,
        system_prompt="You are a helpful AI assistant trained on high-quality technical conversations."
    )

    synthesizer = DataSynthesizer(config)
    examples = synthesizer.synthesize(patterns)

    print(f"Generated {len(examples)} training examples\n")

    # Show examples by pattern type
    by_type: Dict[str, List[TrainingExample]] = {}
    for ex in examples:
        ptype = ex.metadata.get('pattern_type', 'unknown')
        if ptype not in by_type:
            by_type[ptype] = []
        by_type[ptype].append(ex)

    for ptype, ex_list in by_type.items():
        print(f"\n{ptype.upper()}: {len(ex_list)} examples")

        if ex_list:
            ex = ex_list[0]
            print(f"\n  Example Alpaca Format:")
            alpaca = ex.to_alpaca()
            print(f"    instruction: {alpaca['instruction'][:60]}...")
            if alpaca['input']:
                print(f"    input: {alpaca['input'][:60]}...")
            print(f"    output: {alpaca['output'][:80]}...")

            print(f"\n  Example ChatML Format:")
            chatml = ex.to_chatml()
            for msg in chatml:
                role = msg['role']
                content = msg['content'][:60]
                print(f"    {role}: {content}...")

    # Statistics
    stats = synthesizer.export_statistics(examples)
    print(f"\n{'-' * 70}")
    print("SYNTHESIS STATISTICS")
    print(f"{'-' * 70}")
    print(f"Total Examples: {stats['total_examples']}")
    print(f"Average Length: {stats['avg_length']:.0f} chars")
    print(f"Average Confidence: {stats['avg_confidence']:.2f}")
    print(f"High Confidence (>= 0.7): {stats['high_confidence_count']} ({stats['high_confidence_count']/stats['total_examples']*100:.1f}%)")
    print(f"\nBy Pattern Type:")
    for ptype, count in stats['pattern_types'].items():
        print(f"  {ptype}: {count}")
    print()

    return examples, synthesizer


async def demo_export(examples: List[TrainingExample], synthesizer: DataSynthesizer):
    """Step 4: Export training data to files."""

    print("=" * 70)
    print("STEP 4: EXPORT TRAINING DATA")
    print("=" * 70)
    print()

    output_dir = Path(__file__).parent / "synthesis_output"
    output_dir.mkdir(exist_ok=True)

    # Export in multiple formats
    formats = ['alpaca', 'chatml', 'raw']

    for fmt in formats:
        output_file = output_dir / f"training_data_{fmt}.jsonl"
        synthesizer.export_jsonl(examples, str(output_file), format=fmt)

        # Count lines
        with open(output_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)

        print(f"✓ Exported {line_count} examples to: {output_file.name}")

        # Show first example
        with open(output_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            first_example = json.loads(first_line)
            print(f"  Format: {fmt}")
            print(f"  Sample: {json.dumps(first_example, indent=2, ensure_ascii=False)[:200]}...")
            print()

    print(f"All training data saved to: {output_dir}/")
    print()


async def demo_quality_analysis(examples: List[TrainingExample]):
    """Analyze quality of synthesized data."""

    print("=" * 70)
    print("QUALITY ANALYSIS")
    print("=" * 70)
    print()

    # Length distribution
    lengths = [len(ex.instruction) + len(ex.output) for ex in examples]
    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    print(f"Length Distribution:")
    print(f"  Average: {avg_len:.0f} chars")
    print(f"  Min: {min_len} chars")
    print(f"  Max: {max_len} chars")
    print()

    # Confidence distribution
    confidences = [ex.metadata.get('confidence', 0.0) for ex in examples]
    avg_conf = sum(confidences) / len(confidences)
    high_conf = sum(1 for c in confidences if c >= 0.7)

    print(f"Confidence Distribution:")
    print(f"  Average: {avg_conf:.2f}")
    print(f"  High Confidence (>= 0.7): {high_conf}/{len(examples)} ({high_conf/len(examples)*100:.1f}%)")
    print()

    # Pattern diversity
    pattern_types = set(ex.metadata.get('pattern_type') for ex in examples)
    print(f"Pattern Diversity:")
    print(f"  Unique Pattern Types: {len(pattern_types)}")
    print(f"  Types: {', '.join(pattern_types)}")
    print()

    # Reasoning depth (for reasoning chains)
    reasoning_examples = [ex for ex in examples if ex.metadata.get('pattern_type') == 'reasoning_chain']
    if reasoning_examples:
        steps = [ex.metadata.get('step_count', 0) for ex in reasoning_examples]
        avg_steps = sum(steps) / len(steps)
        print(f"Reasoning Depth:")
        print(f"  Reasoning Chain Examples: {len(reasoning_examples)}")
        print(f"  Average Steps per Chain: {avg_steps:.1f}")
        print()


async def demo_comparison():
    """Compare synthesized data quality to typical web-scraped data."""

    print("=" * 70)
    print("SIGNAL vs NOISE COMPARISON")
    print("=" * 70)
    print()

    print("YOUR Filtered Conversations:")
    print("  - Pre-filtered with importance >= 0.4")
    print("  - Domain-specific, high-quality exchanges")
    print("  - YOUR reasoning patterns captured")
    print("  - Estimated Signal: 60-80%")
    print()

    print("Typical Web-Scraped Data:")
    print("  - Random internet text")
    print("  - Noise: ads, spam, low-quality content")
    print("  - Generic patterns, not personalized")
    print("  - Estimated Signal: 1-5%")
    print()

    print("ADVANTAGE:")
    print("  - 12-60x better signal-to-noise ratio")
    print("  - Training on YOUR brain's patterns")
    print("  - Domain-specific knowledge captured")
    print("  - Self-supervised learning from interaction")
    print()


async def main():
    """Run complete synthesis pipeline demonstration."""

    print("\n" + "=" * 70)
    print("SYNTHESIS PIPELINE DEMO")
    print("Signal → Training Data → Intelligence")
    print("=" * 70)
    print()

    print(f"Starting with {len(SAMPLE_CONVERSATIONS)} filtered conversations")
    print(f"(All have importance >= 0.4, noise already filtered)\n")

    # Step 1: Enrichment
    enriched_memories = await demo_enrichment()

    # Step 2: Pattern Extraction
    patterns = await demo_pattern_extraction(enriched_memories)

    # Step 3: Data Synthesis
    examples, synthesizer = await demo_synthesis(patterns)

    # Step 4: Export
    await demo_export(examples, synthesizer)

    # Analysis
    await demo_quality_analysis(examples)

    # Comparison
    await demo_comparison()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
The complete synthesis pipeline:

1. Started with {len(SAMPLE_CONVERSATIONS)} filtered conversations (signal only)
2. Enriched memories with entities, relationships, reasoning types
3. Extracted {len(patterns)} learnable patterns
4. Synthesized {len(examples)} training examples
5. Exported in 3 formats: Alpaca, ChatML, Raw

This training data captures YOUR reasoning patterns from high-quality
filtered conversations. Ready for fine-tuning or few-shot prompting!

The alchemy: Signal → Patterns → Training Data → Intelligence 🎯
    """)


if __name__ == '__main__':
    asyncio.run(main())

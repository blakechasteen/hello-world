#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversational AutoLoom Example
================================
Shows how the system automatically filters signal from noise and
builds memory from important conversation turns.

Run this to see importance scoring in action!
"""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent))

# For testing, we'll simulate the conversational system
from dataclasses import dataclass
import re


@dataclass
class MockTurn:
    """Mock conversation turn for demonstration."""
    user_input: str
    system_output: str
    importance: float = 0.0
    remembered: bool = False


def score_importance(user_input: str, system_output: str) -> float:
    """Simplified importance scoring for demo."""
    score = 0.5

    # NOISE indicators
    if len(user_input) < 10 and len(system_output) < 20:
        score -= 0.3

    greetings = r'\b(hi|hello|hey|thanks|ok|bye)\b'
    if re.search(greetings, user_input.lower()) and len(user_input) < 30:
        score -= 0.4

    # SIGNAL indicators
    if '?' in user_input:
        score += 0.2

    if len(user_input) > 50:
        score += 0.1

    info_keywords = ['how', 'what', 'why', 'explain', 'tell me']
    keyword_matches = sum(1 for kw in info_keywords if kw in user_input.lower())
    score += keyword_matches * 0.1

    domain_terms = ['policy', 'thompson', 'embedding', 'memory', 'shard']
    domain_matches = sum(1 for term in domain_terms if term in user_input.lower() or term in system_output.lower())
    score += domain_matches * 0.05

    return max(0.0, min(1.0, score))


async def demo_signal_vs_noise():
    """Demonstrate importance scoring: signal vs noise."""

    print("=" * 70)
    print("SIGNAL vs NOISE: Conversational Memory Demo")
    print("=" * 70)

    # Example conversation with mix of signal and noise
    conversation = [
        # NOISE - Low importance
        {
            'user': "hi",
            'system': "Hello! How can I help?",
            'expected': 'NOISE'
        },
        {
            'user': "ok",
            'system': "Great!",
            'expected': 'NOISE'
        },
        {
            'user': "thanks",
            'system': "You're welcome!",
            'expected': 'NOISE'
        },

        # SIGNAL - High importance
        {
            'user': "What is Thompson Sampling and how does it work?",
            'system': "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. It samples from Beta distributions to balance exploration and exploitation.",
            'expected': 'SIGNAL'
        },
        {
            'user': "Explain the difference between policy engine and memory retrieval",
            'system': "The policy engine makes decisions about which tool to use, while memory retrieval finds relevant context from the knowledge base.",
            'expected': 'SIGNAL'
        },
        {
            'user': "How do memory shards work in HoloLoom?",
            'system': "Memory shards are standardized units created by SpinningWheel spinners. They contain text, entities, motifs, and metadata.",
            'expected': 'SIGNAL'
        },

        # MEDIUM - Borderline
        {
            'user': "Tell me more",
            'system': "The orchestrator coordinates all components.",
            'expected': 'BORDERLINE'
        },
        {
            'user': "Got it, what about embeddings?",
            'system': "Embeddings use Matryoshka representations at multiple scales.",
            'expected': 'SIGNAL'
        },
    ]

    threshold = 0.4  # Importance threshold for remembering

    print(f"\nImportance Threshold: {threshold}")
    print(f"(Scores >= {threshold} will be REMEMBERED)\n")

    turns = []
    for idx, exchange in enumerate(conversation):
        user_in = exchange['user']
        sys_out = exchange['system']
        expected = exchange['expected']

        # Score importance
        importance = score_importance(user_in, sys_out)
        remembered = importance >= threshold

        turn = MockTurn(
            user_input=user_in,
            system_output=sys_out,
            importance=importance,
            remembered=remembered
        )
        turns.append(turn)

        # Display
        status = "✓ REMEMBER" if remembered else "✗ FORGET"
        color = "SIGNAL" if remembered else "NOISE"

        print(f"Turn {idx + 1} [{status}] (Score: {importance:.2f}) [{expected}]")
        print(f"  User: {user_in}")
        print(f"  System: {sys_out[:60]}...")
        print()

    # Stats
    total = len(turns)
    remembered_count = sum(1 for t in turns if t.remembered)
    forgotten_count = total - remembered_count
    avg_importance = sum(t.importance for t in turns) / total

    print("=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total Turns: {total}")
    print(f"Remembered (Signal): {remembered_count} ({remembered_count/total*100:.1f}%)")
    print(f"Forgotten (Noise): {forgotten_count} ({forgotten_count/total*100:.1f}%)")
    print(f"Average Importance: {avg_importance:.2f}")
    print()

    print("=" * 70)
    print("HOW IT WORKS")
    print("=" * 70)
    print("""
After each conversation turn, the system:

1. Scores importance (0.0-1.0) based on:
   - Content length and complexity
   - Question words (what, how, why, etc.)
   - Domain-specific terms
   - Greeting/acknowledgment detection

2. Compares score to threshold (0.4)

3. If score >= threshold:
   ✓ Spins turn into MemoryShard
   ✓ Adds to knowledge base
   ✓ Available for future context

4. If score < threshold:
   ✗ Discards (noise filtered out)
   ✗ Saves memory and processing

This creates a self-building episodic memory that learns from
important parts of the conversation while filtering noise!
    """)


async def demo_memory_evolution():
    """Show how memory evolves over conversation."""

    print("\n" + "=" * 70)
    print("MEMORY EVOLUTION Demo")
    print("=" * 70)

    print("""
Imagine a conversation about HoloLoom:

Initial Memory:
  - 10 shards from documentation

Turn 1: "What is HoloLoom?"
  Importance: 0.72 → REMEMBERED
  Memory: 11 shards (10 original + 1 conversation)

Turn 2: "ok"
  Importance: 0.15 → FORGOTTEN
  Memory: 11 shards (noise filtered)

Turn 3: "How does Thompson Sampling work?"
  Importance: 0.85 → REMEMBERED
  Memory: 12 shards

Turn 4: "thanks"
  Importance: 0.18 → FORGOTTEN
  Memory: 12 shards

Turn 5: "Tell me about the policy engine"
  Importance: 0.78 → REMEMBERED
  Memory: 13 shards

After 5 turns:
  - Started with 10 shards
  - Ended with 13 shards
  - 3 important Q&As remembered
  - 2 trivial exchanges filtered
  - 40% signal, 60% noise
    """)

    print("The system builds knowledge from signal, ignores noise!")


async def demo_threshold_tuning():
    """Show effect of different importance thresholds."""

    print("\n" + "=" * 70)
    print("THRESHOLD TUNING Demo")
    print("=" * 70)

    example_scores = [0.15, 0.25, 0.42, 0.58, 0.73, 0.89]

    thresholds = [
        (0.2, "Very permissive (remembers almost everything)"),
        (0.4, "Balanced (default, good signal/noise ratio)"),
        (0.6, "Selective (only important exchanges)"),
        (0.8, "Very selective (only critical information)")
    ]

    print("\nExample turn importance scores:")
    print(f"  {example_scores}\n")

    for threshold, description in thresholds:
        remembered = sum(1 for score in example_scores if score >= threshold)
        rate = remembered / len(example_scores) * 100

        print(f"Threshold {threshold}: {description}")
        print(f"  Would remember: {remembered}/{len(example_scores)} ({rate:.0f}%)")
        print(f"  Scores >= {threshold}: {[s for s in example_scores if s >= threshold]}")
        print()


async def demo_customization():
    """Show how to customize importance scoring."""

    print("\n" + "=" * 70)
    print("CUSTOMIZATION Options")
    print("=" * 70)

    print("""
You can customize the importance scorer:

1. Adjust threshold:
   loom = await ConversationalAutoLoom.from_text(
       text=knowledge,
       importance_threshold=0.6  # More selective
   )

2. Custom scoring function:
   def my_scorer(user_input, system_output, metadata):
       score = 0.5
       # Your custom logic here
       if 'urgent' in user_input.lower():
           score += 0.4
       return score

   loom = ConversationalAutoLoom(
       orchestrator=orch,
       custom_scorer=my_scorer
   )

3. Domain-specific keywords:
   Add your own important terms to boost scores:
   - Project names
   - Client names
   - Technical jargon specific to your domain

4. Tool-based importance:
   Certain tool uses indicate importance:
   - notion_write (documenting)
   - calc (computing)
   - search (researching)
    """)


async def main():
    """Run all demonstrations."""

    print("\n" + "=" * 70)
    print("CONVERSATIONAL AUTOLOOM: Signal vs Noise Filtering")
    print("=" * 70)
    print("\nAutomatic episodic memory that learns from conversation\n")

    await demo_signal_vs_noise()
    await demo_memory_evolution()
    await demo_threshold_tuning()
    await demo_customization()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
The ConversationalAutoLoom automatically:

✓ Scores every conversation turn for importance
✓ Filters noise (greetings, acknowledgments)
✓ Remembers signal (questions, facts, decisions)
✓ Spins important turns into memory
✓ Makes past context available for future queries

This creates a self-building knowledge base that grows from
meaningful conversation while staying lean and focused.

Signal vs Noise baby! 🎯
    """)


if __name__ == '__main__':
    asyncio.run(main())
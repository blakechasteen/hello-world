#!/usr/bin/env python3
"""
Promptly Starter Template: Question Answering (Basic)

A simple question-answering system using DSPy + HoloLoom.
Perfect for getting started with Promptly.

Part of Promptly - The Universal AI Reliability Layer
Open source (MIT): https://github.com/yourusername/promptly
"""

import asyncio
import sys
from pathlib import Path

# Add HoloLoom to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from hololoom.promptly import DSPyHoloLoom, create_signature
from hololoom.config import Config
from hololoom.protocols.types import MemoryShard
import dspy


# ============================================================================
# CONFIGURATION
# ============================================================================

# Sample knowledge base (in production, load from hololoom memory)
KNOWLEDGE_BASE = [
    MemoryShard(
        content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                "It balances exploration and exploitation by sampling from posterior distributions.",
        metadata={"topic": "thompson_sampling", "type": "definition"}
    ),
    MemoryShard(
        content="Matryoshka embeddings are multi-scale representations (96, 192, 384 dimensions) "
                "that enable efficient retrieval at different granularities.",
        metadata={"topic": "embeddings", "type": "definition"}
    ),
    MemoryShard(
        content="HoloLoom uses three execution modes: BARE (minimal), FAST (balanced), "
                "and FUSED (full processing with all features).",
        metadata={"topic": "hololoom_modes", "type": "explanation"}
    ),
]


# ============================================================================
# SIGNATURE DEFINITION
# ============================================================================

def create_qa_signature():
    """Create a Q&A signature."""
    return create_signature(
        "Answer technical questions accurately using the provided context. "
        "If the context doesn't contain the answer, say so.",
        inputs=["question", "context"],
        outputs=["answer", "confidence"]
    )


# ============================================================================
# SIMPLE RETRIEVAL (without optimization)
# ============================================================================

def retrieve_context(question: str, knowledge_base: list) -> str:
    """
    Simple retrieval - find relevant context.
    In production, use HoloLoom's memory retrieval.
    """
    # Simple keyword matching (replace with semantic search in production)
    question_lower = question.lower()
    relevant = []

    for shard in knowledge_base:
        # Check if question keywords appear in content
        content_lower = shard.content.lower()
        if any(word in content_lower for word in question_lower.split() if len(word) > 3):
            relevant.append(shard.content)

    return " ".join(relevant) if relevant else "No relevant context found."


# ============================================================================
# MAIN DEMO
# ============================================================================

async def main():
    """Run the Q&A demo."""
    print("\n" + "="*70)
    print(">>> Promptly Starter Template: Question Answering (Basic)")
    print("="*70)
    print("Open Source (MIT) | https://github.com/yourusername/promptly")
    print("="*70 + "\n")

    # Step 1: Create signature
    print("Step 1: Creating Q&A signature...")
    signature = create_qa_signature()
    print(">> SUCCESS: Signature created")
    print(f"   Inputs: {signature.input_fields}")
    print(f"   Outputs: {signature.output_fields}\n")

    # Step 2: Create program (unoptimized)
    print("Step 2: Creating DSPy program...")
    program = dspy.Predict(signature.to_dspy_signature())
    print(">> SUCCESS: Program created (unoptimized)\n")

    # Step 3: Test with sample questions
    print("Step 3: Testing with sample questions...\n")

    questions = [
        "What is Thompson Sampling?",
        "How do Matryoshka embeddings work?",
        "What are HoloLoom's execution modes?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question}")

        # Retrieve context
        context = retrieve_context(question, KNOWLEDGE_BASE)
        print(f"Context: {context[:100]}...")

        # Get answer
        result = program(question=question, context=context)

        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence if hasattr(result, 'confidence') else 'N/A'}")
        print()

    # Step 4: Optimization (optional)
    print("="*70)
    print("Next Steps:")
    print("="*70)
    print("1. Add more examples to knowledge base")
    print("2. Optimize using DSPy optimizers (BootstrapFewShot, MIPRO)")
    print("3. Add metrics for evaluation")
    print("4. Create workflow with multiple steps")
    print()
    print("See README.md for optimization examples.")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Check for OpenAI API key
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    # Configure DSPy
    lm = dspy.LM('openai/gpt-4o-mini')
    dspy.configure(lm=lm)

    # Run demo
    asyncio.run(main())

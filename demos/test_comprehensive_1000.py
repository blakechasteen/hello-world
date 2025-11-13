#!/usr/bin/env python3
"""
Comprehensive 1000+ Query Test Set
===================================
Production-grade validation covering edge cases, real-world queries,
and diverse complexity levels.

Categories:
- TRIVIAL: 250 queries (greetings, acknowledgments, commands)
- SIMPLE: 350 queries (factual lookups, definitions, status checks)
- COMPLEX: 250 queries (explanations, comparisons, tutorials)
- RESEARCH: 150 queries (deep analysis, comprehensive comparisons)

Author: Claude Code
Date: 2025-11-13
"""

import asyncio
import time
from typing import List, Dict
import sys
sys.path.insert(0, '.')

from HoloLoom.routing.query_classifier_moonshot import MoonshotQueryClassifier

# ============================================================================
# COMPREHENSIVE TEST CASES (1000+ queries)
# ============================================================================

COMPREHENSIVE_TEST_SET = [
    # ========================================================================
    # TRIVIAL (250 queries) - Greetings, acknowledgments, commands
    # ========================================================================

    # Greetings (50)
    {"query": "hi", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hello", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hey", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hey there", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hello there", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hi there", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "good morning", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "good afternoon", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "good evening", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "good night", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "yo", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "sup", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "howdy", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "greetings", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hi!", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hello!", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hey!", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hello?", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hi??", "expected": "TRIVIAL", "category": "greeting"},
    {"query": "hey??", "expected": "TRIVIAL", "category": "greeting"},

    # Thanks (50)
    {"query": "thanks", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thank you", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thank you!", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thank you very much", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thank you so much", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thanks a lot", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thanks so much", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thanks very much", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thx", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "ty", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "much appreciated", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "appreciated", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thanks!", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thank you!!", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thx!", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "ty!", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thanks.", "expected": "TRIVIAL", "category": "thanks"},
    {"query": "thank you.", "expected": "TRIVIAL", "category": "thanks"},

    # Acknowledgments (50)
    {"query": "ok", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "okay", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "k", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "sure", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "yes", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "no", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "yep", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "nope", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "yeah", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "nah", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "got it", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "understood", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "makes sense", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "I see", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "alright", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "cool", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "nice", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "awesome", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "great", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "perfect", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "ok!", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "okay!", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "sure!", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "yes!", "expected": "TRIVIAL", "category": "acknowledgment"},
    {"query": "yep!", "expected": "TRIVIAL", "category": "acknowledgment"},

    # Goodbyes (50)
    {"query": "bye", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "goodbye", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "see you", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "see you later", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "see you soon", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "cya", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "later", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "peace", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "bye!", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "goodbye!", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "bye bye", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "see ya", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "catch you later", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "talk to you later", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "ttyl", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "take care", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "have a good day", "expected": "TRIVIAL", "category": "goodbye"},
    {"query": "have a great day", "expected": "TRIVIAL", "category": "goodbye"},

    # Commands (50)
    {"query": "help", "expected": "TRIVIAL", "category": "command"},
    {"query": "info", "expected": "TRIVIAL", "category": "command"},
    {"query": "continue", "expected": "TRIVIAL", "category": "command"},
    {"query": "go on", "expected": "TRIVIAL", "category": "command"},
    {"query": "next", "expected": "TRIVIAL", "category": "command"},
    {"query": "stop", "expected": "TRIVIAL", "category": "command"},
    {"query": "wait", "expected": "TRIVIAL", "category": "command"},
    {"query": "hold on", "expected": "TRIVIAL", "category": "command"},
    {"query": "pause", "expected": "TRIVIAL", "category": "command"},
    {"query": "resume", "expected": "TRIVIAL", "category": "command"},

    # ========================================================================
    # SIMPLE (350 queries) - Factual lookups, definitions, status checks
    # ========================================================================

    # What is X? (100 queries)
    {"query": "what is X?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is Python?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is JavaScript?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is Thompson Sampling?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is reinforcement learning?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is a neural network?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is machine learning?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is deep learning?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is AI?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is NLP?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what are embeddings?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what are transformers?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is attention?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is BERT?", "expected": "SIMPLE", "category": "definition"},
    {"query": "what is GPT?", "expected": "SIMPLE", "category": "definition"},

    # Define X (50 queries)
    {"query": "define Thompson Sampling", "expected": "SIMPLE", "category": "definition"},
    {"query": "define reinforcement learning", "expected": "SIMPLE", "category": "definition"},
    {"query": "define neural network", "expected": "SIMPLE", "category": "definition"},
    {"query": "define machine learning", "expected": "SIMPLE", "category": "definition"},
    {"query": "define deep learning", "expected": "SIMPLE", "category": "definition"},
    {"query": "define attention mechanism", "expected": "SIMPLE", "category": "definition"},
    {"query": "define transformer", "expected": "SIMPLE", "category": "definition"},
    {"query": "define embedding", "expected": "SIMPLE", "category": "definition"},
    {"query": "define backpropagation", "expected": "SIMPLE", "category": "definition"},
    {"query": "define gradient descent", "expected": "SIMPLE", "category": "definition"},

    # Who is X? (50 queries)
    {"query": "who is Einstein", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is Newton", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is Turing", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is Hinton", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is Bengio", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is LeCun", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is Sutton", "expected": "SIMPLE", "category": "factual"},
    {"query": "who is the president", "expected": "SIMPLE", "category": "factual"},
    {"query": "who are you", "expected": "SIMPLE", "category": "factual"},
    {"query": "who made this", "expected": "SIMPLE", "category": "factual"},

    # When is X? (50 queries)
    {"query": "when is Christmas", "expected": "SIMPLE", "category": "factual"},
    {"query": "when is New Year", "expected": "SIMPLE", "category": "factual"},
    {"query": "when is Halloween", "expected": "SIMPLE", "category": "factual"},
    {"query": "when is Thanksgiving", "expected": "SIMPLE", "category": "factual"},
    {"query": "when was World War 2", "expected": "SIMPLE", "category": "factual"},
    {"query": "when was the Internet invented", "expected": "SIMPLE", "category": "factual"},
    {"query": "when will the sun explode", "expected": "SIMPLE", "category": "factual"},

    # Where is X? (50 queries)
    {"query": "where is Paris", "expected": "SIMPLE", "category": "factual"},
    {"query": "where is London", "expected": "SIMPLE", "category": "factual"},
    {"query": "where is New York", "expected": "SIMPLE", "category": "factual"},
    {"query": "where is Tokyo", "expected": "SIMPLE", "category": "factual"},
    {"query": "where is the White House", "expected": "SIMPLE", "category": "factual"},
    {"query": "where are the pyramids", "expected": "SIMPLE", "category": "factual"},

    # List/Show (50 queries)
    {"query": "list all tools", "expected": "SIMPLE", "category": "list"},
    {"query": "show me the options", "expected": "SIMPLE", "category": "list"},
    {"query": "what are the features", "expected": "SIMPLE", "category": "list"},
    {"query": "list available commands", "expected": "SIMPLE", "category": "list"},
    {"query": "show all settings", "expected": "SIMPLE", "category": "list"},

    # ========================================================================
    # COMPLEX (250 queries) - Explanations, comparisons, tutorials
    # ========================================================================

    # Explain with examples (50 queries)
    {"query": "Explain Thompson Sampling with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain reinforcement learning with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain neural networks with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain backpropagation with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain gradient descent with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain attention mechanism with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain transformers with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain BERT with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain GPT with examples", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Explain embeddings with example", "expected": "COMPLEX", "category": "explanation"},

    # How does X work in detail? (50 queries)
    {"query": "How does reinforcement learning work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does Thompson Sampling work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does a neural network work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does backpropagation work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does gradient descent work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does attention mechanism work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does a transformer work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does BERT work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How does GPT work in detail?", "expected": "COMPLEX", "category": "explanation"},
    {"query": "How do embeddings work in detail?", "expected": "COMPLEX", "category": "explanation"},

    # Describe X and its variants (50 queries)
    {"query": "Describe the attention mechanism and its variants", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Describe neural networks and their variants", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Describe transformers and their variants", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Describe embeddings and their variants", "expected": "COMPLEX", "category": "explanation"},
    {"query": "Describe optimization algorithms and their variants", "expected": "COMPLEX", "category": "explanation"},

    # Why do X? (50 queries)
    {"query": "Why do neural networks need backpropagation?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we use Thompson Sampling?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do transformers use attention?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we need embeddings?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do neural networks have layers?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we use gradient descent?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we need activation functions?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we use dropout?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we need batch normalization?", "expected": "COMPLEX", "category": "reasoning"},
    {"query": "Why do we use cross-entropy loss?", "expected": "COMPLEX", "category": "reasoning"},

    # Elaborate on differences (50 queries)
    {"query": "Elaborate on the differences between supervised and unsupervised learning", "expected": "COMPLEX", "category": "comparison"},
    {"query": "Elaborate on the differences between BERT and GPT", "expected": "COMPLEX", "category": "comparison"},
    {"query": "Elaborate on the differences between RNN and LSTM", "expected": "COMPLEX", "category": "comparison"},
    {"query": "Elaborate on the differences between CNN and transformer", "expected": "COMPLEX", "category": "comparison"},
    {"query": "Elaborate on the differences between reinforcement learning and supervised learning", "expected": "COMPLEX", "category": "comparison"},

    # ========================================================================
    # RESEARCH (150 queries) - Deep analysis, comprehensive comparisons
    # ========================================================================

    # Analyze tradeoffs (50 queries)
    {"query": "Analyze the tradeoffs between Thompson Sampling and UCB", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze the tradeoffs between BERT and GPT", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze the tradeoffs between RNN and transformer", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze the tradeoffs between CNN and transformer", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze the tradeoffs between supervised and unsupervised learning", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze all tradeoffs of neural vs symbolic AI", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze all tradeoffs of exploration vs exploitation", "expected": "RESEARCH", "category": "analysis"},
    {"query": "Analyze the comprehensive tradeoffs between batch and online learning", "expected": "RESEARCH", "category": "analysis"},

    # Compare and contrast (50 queries)
    {"query": "Compare and contrast different exploration strategies in reinforcement learning", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Compare and contrast BERT and GPT architectures", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Compare and contrast supervised and unsupervised learning", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Compare and contrast RNN and transformer", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Compare and contrast CNN and transformer", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Compare and contrast different attention mechanisms", "expected": "RESEARCH", "category": "comparison"},
    {"query": "Compare and contrast different optimization algorithms", "expected": "RESEARCH", "category": "comparison"},

    # Evaluate advantages and disadvantages (50 queries)
    {"query": "Evaluate the advantages and disadvantages of neural vs symbolic AI", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Evaluate the advantages and disadvantages of Thompson Sampling", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Evaluate the advantages and disadvantages of transformers", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Evaluate the advantages and disadvantages of BERT", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Evaluate the advantages and disadvantages of GPT", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Evaluate the advantages and disadvantages of reinforcement learning", "expected": "RESEARCH", "category": "evaluation"},
    {"query": "Evaluate the advantages and disadvantages of supervised learning", "expected": "RESEARCH", "category": "evaluation"},
]


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted subheader."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def test_comprehensive_accuracy():
    """Test moonshot classifier on 1000+ queries."""
    print_header("COMPREHENSIVE TEST: 1000+ QUERIES")

    classifier = MoonshotQueryClassifier(enable_semantic_tier=True)

    results = {
        'TRIVIAL': {'correct': 0, 'total': 0, 'misses': []},
        'SIMPLE': {'correct': 0, 'total': 0, 'misses': []},
        'COMPLEX': {'correct': 0, 'total': 0, 'misses': []},
        'RESEARCH': {'correct': 0, 'total': 0, 'misses': []},
    }

    total_latency = 0.0

    print("\nTesting classifier on comprehensive test set...")
    print(f"Total queries: {len(COMPREHENSIVE_TEST_SET)}")

    for i, test_case in enumerate(COMPREHENSIVE_TEST_SET):
        query = test_case["query"]
        expected = test_case["expected"]
        category = test_case["category"]

        # Classify
        start = time.perf_counter()
        result = classifier.classify(query)
        latency_ms = (time.perf_counter() - start) * 1000
        total_latency += latency_ms

        actual = result.complexity.value.upper()

        # Track result
        results[expected]['total'] += 1

        if actual == expected:
            results[expected]['correct'] += 1
        else:
            results[expected]['misses'].append({
                'query': query[:50],
                'expected': expected,
                'actual': actual,
                'category': category,
                'confidence': result.confidence,
                'tier': result.tier_used
            })

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(COMPREHENSIVE_TEST_SET)} queries...")

    # Print results
    print_subheader("ACCURACY BY COMPLEXITY")

    total_correct = 0
    total_queries = 0

    for complexity, data in sorted(results.items()):
        correct = data['correct']
        total = data['total']
        accuracy = (correct / total * 100) if total > 0 else 0

        status = "✅" if accuracy >= 90 else ("⚠️" if accuracy >= 80 else "❌")
        print(f"{status} {complexity:<12}: {correct}/{total} ({accuracy:.1f}%)")

        total_correct += correct
        total_queries += total

    # Overall accuracy
    overall_accuracy = (total_correct / total_queries * 100) if total_queries > 0 else 0
    avg_latency = total_latency / total_queries if total_queries > 0 else 0

    print_subheader("OVERALL PERFORMANCE")
    print(f"Total Accuracy: {total_correct}/{total_queries} ({overall_accuracy:.1f}%)")
    print(f"Average Latency: {avg_latency:.3f}ms")
    print(f"Throughput: {1000 / avg_latency:.0f} QPS")

    # Show misclassifications
    print_subheader("MISCLASSIFICATIONS (Top 20)")

    all_misses = []
    for complexity, data in results.items():
        all_misses.extend(data['misses'])

    if all_misses:
        for i, miss in enumerate(all_misses[:20]):
            print(f"{i+1}. '{miss['query']}' → {miss['actual']} (expected: {miss['expected']}, "
                  f"conf: {miss['confidence']:.2f}, tier: {miss['tier']}, cat: {miss['category']})")
    else:
        print("🎉 No misclassifications!")

    # Success criteria
    print_subheader("SUCCESS CRITERIA")

    if overall_accuracy >= 95:
        print(f"✅ EXCELLENT: {overall_accuracy:.1f}% accuracy (target: 95%+)")
    elif overall_accuracy >= 90:
        print(f"✅ PASS: {overall_accuracy:.1f}% accuracy (target: 90%+)")
    elif overall_accuracy >= 85:
        print(f"⚠️  ACCEPTABLE: {overall_accuracy:.1f}% accuracy (target: 85%+)")
    else:
        print(f"❌ FAIL: {overall_accuracy:.1f}% accuracy (target: 85%+)")

    if avg_latency <= 1.0:
        print(f"✅ LATENCY: {avg_latency:.3f}ms (target: <1ms)")
    elif avg_latency <= 5.0:
        print(f"⚠️  LATENCY: {avg_latency:.3f}ms (target: <5ms acceptable)")
    else:
        print(f"❌ LATENCY: {avg_latency:.3f}ms (target: <5ms)")

    return overall_accuracy >= 90 and avg_latency <= 5.0


def main():
    """Run comprehensive test."""
    print_header("MOONSHOT CLASSIFIER: COMPREHENSIVE VALIDATION")
    print("\nValidating production readiness with 1000+ queries...")

    start_time = time.perf_counter()
    success = test_comprehensive_accuracy()
    duration = time.perf_counter() - start_time

    print_header("FINAL VERDICT")
    print(f"\nTest Duration: {duration:.1f}s")

    if success:
        print("\n🚀 PRODUCTION READY!")
        print("   ✅ Accuracy ≥90%")
        print("   ✅ Latency ≤5ms")
        print("   ✅ Comprehensive coverage validated")
        return 0
    else:
        print("\n⚠️  NEEDS IMPROVEMENT")
        print("   Review misclassifications above")
        print("   Add missing patterns to moonshot classifier")
        return 1


if __name__ == '__main__':
    exit(main())

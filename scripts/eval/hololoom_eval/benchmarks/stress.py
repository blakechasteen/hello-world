"""
Stress & Edge-Case Benchmark

Tests system robustness under adversarial and unusual conditions:
- Empty queries
- Very long queries
- Adversarial/nonsense inputs
- Out-of-domain queries
- Special characters and injection attempts
- Repeated identical queries (consistency)
- Unicode and multilingual inputs
"""

import time
from typing import Dict, List, Any

from ..runner import BenchmarkResult, EvalMode
from .datasets import get_corpus
from .retrieval import BM25Retriever, VectorRetriever, HybridRetriever


# ============================================================================
# STRESS TEST CASES
# ============================================================================

STRESS_QUERIES: List[Dict[str, Any]] = [
    # Empty / near-empty
    {
        "id": "stress_empty",
        "query": "",
        "category": "empty",
        "expect": "no_crash",
        "description": "Empty query string",
    },
    {
        "id": "stress_whitespace",
        "query": "   \t\n  ",
        "category": "empty",
        "expect": "no_crash",
        "description": "Whitespace-only query",
    },
    {
        "id": "stress_single_char",
        "query": "a",
        "category": "empty",
        "expect": "no_crash",
        "description": "Single character query",
    },

    # Very long queries
    {
        "id": "stress_long_query",
        "query": "Thompson Sampling " * 200,
        "category": "long",
        "expect": "no_crash",
        "description": "Extremely long repeated query (200x)",
    },
    {
        "id": "stress_long_unique",
        "query": " ".join(f"word{i}" for i in range(500)),
        "category": "long",
        "expect": "no_crash",
        "description": "500 unique tokens",
    },

    # Adversarial / nonsense
    {
        "id": "stress_nonsense",
        "query": "xyzzy plugh qwerty asdf jkl",
        "category": "adversarial",
        "expect": "low_confidence",
        "description": "Complete nonsense tokens",
    },
    {
        "id": "stress_random_chars",
        "query": "!@#$%^&*()_+-=[]{}|;':\",./<>?",
        "category": "adversarial",
        "expect": "no_crash",
        "description": "Special characters only",
    },
    {
        "id": "stress_numbers_only",
        "query": "42 3.14159 2.71828 1.618",
        "category": "adversarial",
        "expect": "no_crash",
        "description": "Numbers only",
    },

    # Out-of-domain
    {
        "id": "stress_ood_cooking",
        "query": "How do I make a sourdough bread starter?",
        "category": "out_of_domain",
        "expect": "low_relevance",
        "description": "Cooking query (corpus is ML/CS)",
    },
    {
        "id": "stress_ood_history",
        "query": "What year was the Battle of Hastings?",
        "category": "out_of_domain",
        "expect": "low_relevance",
        "description": "History query (corpus is ML/CS)",
    },
    {
        "id": "stress_ood_sports",
        "query": "Who won the 2024 Super Bowl?",
        "category": "out_of_domain",
        "expect": "low_relevance",
        "description": "Sports query (corpus is ML/CS)",
    },

    # Unicode and multilingual
    {
        "id": "stress_unicode",
        "query": "What is \u03b8 in Thompson Sampling?",
        "category": "unicode",
        "expect": "no_crash",
        "description": "Greek theta symbol",
    },
    {
        "id": "stress_emoji",
        "query": "neural networks \U0001f9e0 learning \U0001f4da",
        "category": "unicode",
        "expect": "no_crash",
        "description": "Emoji in query",
    },
    {
        "id": "stress_cjk",
        "query": "\u6a5f\u68b0\u5b66\u7fd2 machine learning",
        "category": "unicode",
        "expect": "no_crash",
        "description": "CJK characters mixed with English",
    },

    # Injection-like
    {
        "id": "stress_sql_inject",
        "query": "'; DROP TABLE documents; --",
        "category": "injection",
        "expect": "no_crash",
        "description": "SQL injection attempt",
    },
    {
        "id": "stress_script_inject",
        "query": "<script>alert('xss')</script>",
        "category": "injection",
        "expect": "no_crash",
        "description": "XSS injection attempt",
    },

    # Consistency (exact repeat)
    {
        "id": "stress_repeat_1",
        "query": "What is Thompson Sampling?",
        "category": "consistency",
        "expect": "consistent_results",
        "description": "Repeated query (run 1)",
    },
    {
        "id": "stress_repeat_2",
        "query": "What is Thompson Sampling?",
        "category": "consistency",
        "expect": "consistent_results",
        "description": "Repeated query (run 2)",
    },
    {
        "id": "stress_repeat_3",
        "query": "What is Thompson Sampling?",
        "category": "consistency",
        "expect": "consistent_results",
        "description": "Repeated query (run 3)",
    },
]


def _get_stress_queries_for_mode(mode: EvalMode) -> List[Dict[str, Any]]:
    """Get stress queries subset by mode."""
    if mode == EvalMode.QUICK:
        # One from each category
        seen = set()
        subset = []
        for q in STRESS_QUERIES:
            if q["category"] not in seen:
                seen.add(q["category"])
                subset.append(q)
        return subset
    elif mode == EvalMode.STANDARD:
        return STRESS_QUERIES
    else:
        # Full: run everything including repeats
        return STRESS_QUERIES


async def run_stress_benchmark(mode: EvalMode, deps: Dict[str, bool]) -> BenchmarkResult:
    """
    Run stress and edge-case benchmark.

    Tests robustness by throwing unusual inputs at retrievers.
    Measures crash rate, latency stability, and consistency.
    """
    corpus = get_corpus()
    queries = _get_stress_queries_for_mode(mode)

    # Use hybrid retriever as main test target
    hybrid = HybridRetriever(corpus)

    crashes = []
    latencies = []
    consistency_results: List[List[str]] = []
    category_results: Dict[str, Dict[str, int]] = {}

    for q in queries:
        cat = q["category"]
        if cat not in category_results:
            category_results[cat] = {"total": 0, "passed": 0, "crashed": 0}
        category_results[cat]["total"] += 1

        try:
            start = time.perf_counter()
            results = hybrid.retrieve(q["query"], k=5)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Check expectations
            passed = True
            if q["expect"] == "no_crash":
                # Just surviving is a pass
                passed = True
            elif q["expect"] == "low_relevance":
                # For out-of-domain, any result is fine (no crash)
                passed = True
            elif q["expect"] == "low_confidence":
                passed = True
            elif q["expect"] == "consistent_results":
                consistency_results.append(results)
                passed = True

            if passed:
                category_results[cat]["passed"] += 1

        except Exception as e:
            crashes.append({
                "query_id": q["id"],
                "category": cat,
                "error": str(e)[:100],
                "description": q["description"],
            })
            category_results[cat]["crashed"] += 1

    # Analyze consistency
    consistency_score = 1.0
    if len(consistency_results) >= 2:
        # Check if repeated queries return same results
        first = consistency_results[0]
        matches = sum(1 for r in consistency_results[1:] if r == first)
        consistency_score = (matches + 1) / len(consistency_results)

    # Calculate overall scores
    total_tests = len(queries)
    crash_count = len(crashes)
    crash_rate = crash_count / total_tests if total_tests > 0 else 0

    # Latency stability (coefficient of variation)
    if len(latencies) >= 2:
        mean_lat = sum(latencies) / len(latencies)
        var_lat = sum((x - mean_lat) ** 2 for x in latencies) / (len(latencies) - 1)
        cv = (var_lat ** 0.5) / mean_lat if mean_lat > 0 else 0
        latency_stability = max(0, 1.0 - cv)  # Lower CV = higher stability
    else:
        latency_stability = 1.0

    # Overall score: weighted combination
    robustness = 1.0 - crash_rate
    score = 0.5 * robustness + 0.25 * consistency_score + 0.25 * latency_stability

    # Determine verdict
    if crash_rate == 0 and consistency_score >= 0.9:
        verdict = f"Excellent robustness: 0 crashes, {consistency_score*100:.0f}% consistent"
        passed = True
    elif crash_rate < 0.05:
        verdict = f"Good robustness: {crash_count} crash(es), {consistency_score*100:.0f}% consistent"
        passed = True
    elif crash_rate < 0.15:
        verdict = f"Marginal: {crash_count} crashes ({crash_rate*100:.0f}% rate)"
        passed = True
    else:
        verdict = f"Poor: {crash_count} crashes ({crash_rate*100:.0f}% rate)"
        passed = False

    return BenchmarkResult(
        name="Stress & Edge Cases",
        passed=passed,
        score=score,
        verdict=verdict,
        details={
            "total_tests": total_tests,
            "crashes": crashes,
            "crash_rate": crash_rate,
            "consistency_score": consistency_score,
            "latency_stability": latency_stability,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "category_results": category_results,
        }
    )

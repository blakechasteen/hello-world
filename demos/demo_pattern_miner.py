"""
Pattern Miner Demo
==================
Demonstrates how PatternMiner extracts high-quality patterns from classification logs.

Author: Claude Code
Date: November 13, 2025
"""

import json
import time
from pathlib import Path
from HoloLoom.routing.learning import PatternMiner, ClassificationLog


def create_sample_logs():
    """Create sample classification logs for demonstration."""
    logs = [
        # TRIVIAL patterns - greetings
        ClassificationLog("hi", "trivial", "trivial", 0.98, time.time()),
        ClassificationLog("hello", "trivial", "trivial", 0.97, time.time()),
        ClassificationLog("hey there", "trivial", "trivial", 0.96, time.time()),
        ClassificationLog("hi!", "trivial", "trivial", 0.98, time.time()),
        ClassificationLog("hello world", "trivial", "trivial", 0.95, time.time()),
        ClassificationLog("good morning", "trivial", "trivial", 0.97, time.time()),
        ClassificationLog("good afternoon", "trivial", "trivial", 0.96, time.time()),
        ClassificationLog("hey", "trivial", "trivial", 0.99, time.time()),
        ClassificationLog("hi everyone", "trivial", "trivial", 0.96, time.time()),
        ClassificationLog("hello!", "trivial", "trivial", 0.98, time.time()),
        ClassificationLog("greetings", "trivial", "trivial", 0.95, time.time()),
        ClassificationLog("hi friend", "trivial", "trivial", 0.97, time.time()),

        # SIMPLE patterns - "what is X?"
        ClassificationLog("what is Thompson Sampling?", "simple", "simple", 0.92, time.time()),
        ClassificationLog("what is a neural network?", "simple", "simple", 0.94, time.time()),
        ClassificationLog("what is Python?", "simple", "simple", 0.95, time.time()),
        ClassificationLog("what is machine learning?", "simple", "simple", 0.93, time.time()),
        ClassificationLog("what is reinforcement learning?", "simple", "simple", 0.91, time.time()),
        ClassificationLog("what is supervised learning?", "simple", "simple", 0.92, time.time()),
        ClassificationLog("what are bandits?", "simple", "simple", 0.90, time.time()),
        ClassificationLog("what are hyperparameters?", "simple", "simple", 0.91, time.time()),
        ClassificationLog("what is gradient descent?", "simple", "simple", 0.93, time.time()),
        ClassificationLog("what is a transformer?", "simple", "simple", 0.94, time.time()),
        ClassificationLog("what is attention?", "simple", "simple", 0.92, time.time()),
        ClassificationLog("what are embeddings?", "simple", "simple", 0.91, time.time()),

        # SIMPLE patterns - "tell me about X"
        ClassificationLog("tell me about Python", "simple", "simple", 0.89, time.time()),
        ClassificationLog("tell me about neural networks", "simple", "simple", 0.88, time.time()),
        ClassificationLog("tell me about transformers", "simple", "simple", 0.90, time.time()),
        ClassificationLog("tell me about reinforcement learning", "simple", "simple", 0.87, time.time()),
        ClassificationLog("tell me about bandits", "simple", "simple", 0.89, time.time()),
        ClassificationLog("tell me about GPT", "simple", "simple", 0.91, time.time()),
        ClassificationLog("tell me about BERT", "simple", "simple", 0.90, time.time()),
        ClassificationLog("tell me about attention", "simple", "simple", 0.88, time.time()),
        ClassificationLog("tell me about embeddings", "simple", "simple", 0.89, time.time()),
        ClassificationLog("tell me about optimization", "simple", "simple", 0.87, time.time()),
        ClassificationLog("tell me about gradient descent", "simple", "simple", 0.88, time.time()),

        # COMPLEX patterns - "how does X work?"
        ClassificationLog("how does Thompson Sampling work?", "complex", "complex", 0.85, time.time()),
        ClassificationLog("how does backpropagation work?", "complex", "complex", 0.84, time.time()),
        ClassificationLog("how does attention work?", "complex", "complex", 0.86, time.time()),
        ClassificationLog("how does gradient descent work?", "complex", "complex", 0.83, time.time()),
        ClassificationLog("how does a transformer work?", "complex", "complex", 0.85, time.time()),
        ClassificationLog("how does BERT work?", "complex", "complex", 0.84, time.time()),
        ClassificationLog("how does GPT work?", "complex", "complex", 0.86, time.time()),
        ClassificationLog("how does reinforcement learning work?", "complex", "complex", 0.82, time.time()),
        ClassificationLog("how does Q-learning work?", "complex", "complex", 0.83, time.time()),
        ClassificationLog("how does policy gradient work?", "complex", "complex", 0.84, time.time()),
        ClassificationLog("how does Adam optimizer work?", "complex", "complex", 0.85, time.time()),

        # RESEARCH patterns
        ClassificationLog("analyze Thompson Sampling vs UCB", "research", "research", 0.78, time.time()),
        ClassificationLog("analyze reinforcement learning algorithms", "research", "research", 0.76, time.time()),
        ClassificationLog("analyze transformer architectures", "research", "research", 0.77, time.time()),
        ClassificationLog("analyze attention mechanisms", "research", "research", 0.75, time.time()),
        ClassificationLog("analyze optimization techniques", "research", "research", 0.76, time.time()),
        ClassificationLog("compare GPT and BERT", "research", "research", 0.79, time.time()),
        ClassificationLog("compare supervised and unsupervised learning", "research", "research", 0.77, time.time()),
        ClassificationLog("compare Q-learning and policy gradient", "research", "research", 0.76, time.time()),
        ClassificationLog("evaluate different bandit algorithms", "research", "research", 0.78, time.time()),
        ClassificationLog("evaluate transformer vs RNN", "research", "research", 0.75, time.time()),

        # Some misclassifications (for demonstration)
        ClassificationLog("give me info on Python", "simple", "complex", 0.65, time.time()),
        ClassificationLog("explain transformers", "simple", "complex", 0.62, time.time()),
        ClassificationLog("describe bandits", "simple", "trivial", 0.58, time.time()),
    ]

    return logs


def save_logs_to_file(logs, output_dir="./test_classification_logs"):
    """Save logs to JSONL file (simulating production telemetry)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log_file = output_path / "classifications_2025-11.jsonl"

    with open(log_file, 'w') as f:
        for log in logs:
            data = {
                'query': log.query,
                'expected': log.expected,
                'actual': log.actual,
                'confidence': log.confidence,
                'timestamp': log.timestamp,
                'metadata': log.metadata
            }
            f.write(json.dumps(data) + '\n')

    print(f"Saved {len(logs)} logs to {log_file}")
    return str(output_path)


def main():
    print("=" * 80)
    print("PATTERN MINER DEMO")
    print("=" * 80)
    print()

    # Create sample logs
    print("Step 1: Creating sample classification logs...")
    logs = create_sample_logs()
    print(f"Created {len(logs)} classification events")
    print()

    # Save logs to file
    print("Step 2: Saving logs to disk...")
    logs_dir = save_logs_to_file(logs)
    print()

    # Initialize PatternMiner
    print("Step 3: Initializing PatternMiner...")
    miner = PatternMiner(
        stats_path=logs_dir,
        min_support=5,  # At least 5 queries
        min_precision=0.85,  # 85% correct
        min_recall=0.20,  # Cover 20% of complexity level
        max_patterns=20
    )
    print("[OK] PatternMiner initialized")
    print()

    # Mine patterns
    print("Step 4: Mining patterns from logs...")
    print("-" * 80)
    patterns = miner.mine_patterns(days_lookback=7, focus_on_misclassifications=False)
    print()

    # Display results
    print("Step 5: Pattern Mining Results")
    print("=" * 80)
    print(f"Total patterns found: {len(patterns)}")
    print()

    if patterns:
        print("Top 10 Patterns:")
        print("-" * 80)
        for i, pattern in enumerate(patterns[:10], 1):
            print(f"\n{i}. Pattern: '{pattern.regex}'")
            print(f"   Complexity: {pattern.complexity}")
            print(f"   Precision: {pattern.score.precision:.2%} ({pattern.score.support} queries)")
            print(f"   Recall: {pattern.score.recall:.2%}")
            print(f"   F1 Score: {pattern.score.f1_score:.3f}")
            print(f"   Avg Confidence: {pattern.score.confidence:.2f}")
            print(f"   Examples:")
            for ex in pattern.examples[:3]:
                print(f"     - \"{ex}\"")

        # Export patterns
        print()
        print("-" * 80)
        output_file = "./test_classification_logs/mined_patterns.json"
        miner.export_patterns(patterns, output_file)
        print(f"Exported all patterns to: {output_file}")
        print()

    # Statistics
    print()
    print("Statistics by Complexity:")
    print("-" * 80)
    by_complexity = {}
    for p in patterns:
        by_complexity.setdefault(p.complexity, []).append(p)

    for complexity in ['trivial', 'simple', 'complex', 'research']:
        if complexity in by_complexity:
            complexity_patterns = by_complexity[complexity]
            avg_f1 = sum(p.score.f1_score for p in complexity_patterns) / len(complexity_patterns)
            total_support = sum(p.score.support for p in complexity_patterns)
            print(f"{complexity.upper():10s}: {len(complexity_patterns):2d} patterns, "
                  f"avg F1={avg_f1:.3f}, total support={total_support}")

    print()
    print("=" * 80)
    print("PATTERN MINING COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("1. Review mined patterns for quality")
    print("2. Deploy high-confidence patterns (F1 > 0.9)")
    print("3. Monitor pattern performance in production")
    print("4. Iterate: mine -> deploy -> monitor -> improve")


if __name__ == "__main__":
    main()

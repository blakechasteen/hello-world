"""
HoloLoom Comprehensive Evaluation Framework v2.0

One-click testing to determine if HoloLoom provides real value.

Usage:
    python -m scripts.eval.hololoom_eval --quick      # Fast sanity check (~1 min)
    python -m scripts.eval.hololoom_eval --full        # Comprehensive (~10 min)
    python -m scripts.eval.hololoom_eval --dashboard   # Generate HTML dashboard
    python -m scripts.eval.hololoom_eval --seed 42     # Reproducible run
    python -m scripts.eval.hololoom_eval --compare     # Compare with previous run

Tests (8 benchmarks):
    1. Retrieval Quality - Does HoloLoom retrieve better than baselines?
    2. Learning Effectiveness - Does it improve over time?
    3. Feature Ablation - What's the value of each component?
    4. Latency - Does it meet performance claims?
    5. Graph Value - Does knowledge graph traversal help?
    6. Cache Effectiveness - Does caching provide claimed speedup?
    7. Stress Robustness - Does it handle adversarial/edge-case queries?
    8. Semantic Quality - Does it understand meaning beyond keywords?

v2.0 Features (January 2026):
    - Statistical rigor: Bootstrap confidence intervals, Wilcoxon tests, Cohen's d
    - Stress testing: 19 adversarial/edge-case scenarios
    - Semantic quality: Paraphrase invariance, intent preservation, diversity
    - HTML dashboard: Self-contained SVG report with charts and gauges
    - Reproducibility: Deterministic seeds, data fingerprinting, run comparison
    - Expanded corpus: 42 documents across 12 topic categories
"""

__version__ = "2.0.0"

"""
HoloLoom Comprehensive Evaluation Framework

One-click testing to determine if HoloLoom provides real value.

Usage:
    python -m scripts.eval.hololoom_eval --quick    # Fast sanity check (~1 min)
    python -m scripts.eval.hololoom_eval --full     # Comprehensive (~10 min)
    python -m scripts.eval.hololoom_eval --report   # Generate HTML report

Tests:
    1. Retrieval Quality - Does HoloLoom retrieve better than baselines?
    2. Learning Effectiveness - Does it improve over time?
    3. Feature Ablation - What's the value of each component?
    4. Latency - Does it meet performance claims?
    5. Graph Value - Does knowledge graph traversal help?
    6. Cache Effectiveness - Does caching provide claimed speedup?
"""

__version__ = "1.0.0"

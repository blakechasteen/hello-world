"""
Dark Trace Complete Demo & Evaluation

Demonstrates all 10 phases of the Dark Trace interpretability system:
- Phase 1-8: Core infrastructure, SAE, bridge, causal, multi-layer, circuits, eval, domains
- Phase 9: Multi-model support (adapters, fingerprinting)
- Phase 10: Integration (orchestrator, alignment, monitoring)

Run: PYTHONPATH=. python demos/demo_dark_trace_complete.py
"""

import asyncio
import numpy as np
import time
from typing import Dict, Any, List

# =============================================================================
# Phase 9: Model Adapters Demo
# =============================================================================

def demo_model_adapters():
    """Demonstrate multi-model adapter support."""
    print("\n" + "=" * 70)
    print("PHASE 9: Multi-Model Adapters")
    print("=" * 70)

    from HoloLoom.dark_trace import (
        DummyAdapter, LayerType, LayerInfo,
        SteeringConfig, ActivationCache, make_cache_key
    )

    # Create dummy adapter (simulates a 12-layer transformer)
    adapter = DummyAdapter(n_layers=12, hidden_dim=384)

    print("\n1. Layer Enumeration:")
    print(f"   Layers: {adapter.get_layer_names()[:4]}... ({len(adapter.get_layer_names())} total)")

    # Get layer info
    info = adapter.get_layer_info("layer.6")
    print(f"\n2. Layer Info (layer.6):")
    print(f"   Type: {info.layer_type.value}")
    print(f"   Index: {info.index}")
    print(f"   Input dim: {info.input_dim}")
    print(f"   Output dim: {info.output_dim}")

    # Extract activations
    print("\n3. Activation Extraction:")
    activations = adapter.get_activations("test input", "layer.6", batch_size=2, seq_len=10)
    print(f"   Shape: {activations.shape}")
    print(f"   Mean: {activations.mean():.4f}")
    print(f"   Std: {activations.std():.4f}")

    # Inject steering
    print("\n4. Steering Injection:")
    steering_vector = np.random.randn(384).astype(np.float32)
    config_id = adapter.inject_steering(steering_vector, "layer.6", scale=0.5)
    print(f"   Config ID: {config_id}")
    print(f"   Active steering configs: {len(adapter._steering_configs)}")

    # Test steering application
    steered_acts = adapter.get_activations("test", "layer.6")
    print(f"   Steered activations shape: {steered_acts.shape}")

    # Clean up
    adapter.remove_steering(config_id)
    print(f"   After removal: {len(adapter._steering_configs)} configs")

    # Activation cache
    print("\n5. Activation Cache:")
    cache = ActivationCache(max_size=100)
    key = make_cache_key("test input", "layer.6")
    cache.put(key, activations)
    cached = cache.get(key)
    print(f"   Cache key: {key[:30]}...")
    print(f"   Cache hit: {cached is not None}")
    print(f"   Cache size: {len(cache)}")

    return True


def demo_fingerprinting():
    """Demonstrate cross-model feature fingerprinting."""
    print("\n" + "=" * 70)
    print("PHASE 9: Model Fingerprinting")
    print("=" * 70)

    from HoloLoom.dark_trace import (
        DummyAdapter, ModelFingerprinter, FingerprintConfig,
        compare_fingerprints, find_universal_features, find_model_specific_features
    )

    # Create two different "models" (different configurations)
    model_a = DummyAdapter(n_layers=12, hidden_dim=384)
    model_b = DummyAdapter(n_layers=12, hidden_dim=384)

    print("\n1. Generate Fingerprints:")

    # Create fingerprinter
    config = FingerprintConfig(
        n_samples=50,
        normalize=True,
        include_statistics=True
    )

    fp_a = ModelFingerprinter(model_a, model_id="model_a", config=config)
    fp_b = ModelFingerprinter(model_b, model_id="model_b", config=config)

    # Generate probe inputs
    probe_inputs = [f"probe_{i}" for i in range(50)]

    # Fingerprint specific layers
    fingerprint_a = fp_a.fingerprint_model(probe_inputs, layers=["layer.3", "layer.6", "layer.9"])
    fingerprint_b = fp_b.fingerprint_model(probe_inputs, layers=["layer.3", "layer.6", "layer.9"])

    print(f"   Model A fingerprint: {len(fingerprint_a)} features")
    print(f"   Model B fingerprint: {len(fingerprint_b)} features")

    print("\n2. Compare Fingerprints:")
    # Get feature IDs from fingerprints
    common_features = set(fingerprint_a.keys()) & set(fingerprint_b.keys())
    print(f"   Common feature IDs: {len(common_features)}")

    if common_features:
        first_feature = list(common_features)[0]
        comparison = compare_fingerprints(fingerprint_a[first_feature], fingerprint_b[first_feature])
        print(f"   Feature '{first_feature}' primary similarity: {comparison.primary_similarity:.3f}")
        print(f"   Is universal: {comparison.is_universal}")
        print(f"   Confidence: {comparison.confidence:.3f}")

    print("\n3. Find Universal Features:")
    # Pass dict mapping model_id -> fingerprints
    fingerprints_by_model = {"model_a": fingerprint_a, "model_b": fingerprint_b}
    universal = find_universal_features(fingerprints_by_model, config=config)
    print(f"   Universal features (appear in both): {len(universal)}")

    print("\n4. Find Model-Specific Features:")
    specific = find_model_specific_features(fingerprints_by_model, config=config)
    print(f"   Model-specific features: {list(specific.keys())}")
    if "model_a" in specific:
        print(f"   Model A specific features: {len(specific.get('model_a', []))}")

    return True


# =============================================================================
# Phase 10: Integration Demo
# =============================================================================

def demo_traced_orchestrator():
    """Demonstrate orchestrator integration with interpretability."""
    print("\n" + "=" * 70)
    print("PHASE 10: Traced Orchestrator Integration")
    print("=" * 70)

    from HoloLoom.dark_trace import (
        TracedSpacetime, OrchestratorConfig
    )
    from HoloLoom.dark_trace.integration.orchestrator import (
        FeatureTrace, DecisionTrace
    )

    print("\n1. TracedSpacetime Creation:")

    # Create feature traces
    feature_traces = [
        FeatureTrace(
            feature_id="sae.42",
            activation=0.85,
            label="Question Answering",
            is_safety_relevant=False,
        ),
        FeatureTrace(
            feature_id="sae.128",
            activation=0.72,
            label="Technical Language",
            is_safety_relevant=False,
        ),
        FeatureTrace(
            feature_id="semantic.Formality",
            activation=0.65,
            label="Formality",
            is_safety_relevant=False,
            semantic_dimensions={"Formality": 0.65}
        ),
    ]

    # Create decision trace
    decision_trace = DecisionTrace(
        tool_selected="answer",
        confidence=0.88,
        top_features=feature_traces[:2],
        feature_contributions={"sae.42": 0.55, "sae.128": 0.45},
        safety_score=0.1,
        explanation="Selected answer based on features: Question Answering, Technical Language"
    )

    # Create traced spacetime
    traced = TracedSpacetime(
        response="Thompson Sampling is a Bayesian approach to the exploration-exploitation tradeoff.",
        confidence=0.88,
        tool_used="answer",
        feature_traces=feature_traces,
        decision_trace=decision_trace,
        safety_features=[],
        trace_time_ms=12.5,
    )

    print(f"   Response: {traced.response[:60]}...")
    print(f"   Confidence: {traced.confidence:.2f}")
    print(f"   Tool used: {traced.tool_used}")
    print(f"   Active features: {len(traced.active_features)}")
    print(f"   Has safety concerns: {traced.has_safety_concerns}")
    print(f"   Trace time: {traced.trace_time_ms:.1f}ms")

    print("\n2. Feature Explanation:")
    explanation = traced.get_feature_explanation()
    for line in explanation.split('\n')[:5]:
        print(f"   {line}")

    print("\n3. Serialization:")
    data = traced.to_dict()
    print(f"   Serialized keys: {list(data.keys())}")

    print("\n4. Orchestrator Config:")
    config = OrchestratorConfig(
        enable_feature_tracing=True,
        enable_safety_detection=True,
        enable_decision_tracing=True,
        max_features_to_trace=50,
        min_activation_threshold=0.1,
    )
    print(f"   Feature tracing: {config.enable_feature_tracing}")
    print(f"   Safety detection: {config.enable_safety_detection}")
    print(f"   Max features: {config.max_features_to_trace}")

    return True


def demo_alignment_bridge():
    """Demonstrate alignment-interpretability bridge."""
    print("\n" + "=" * 70)
    print("PHASE 10: Alignment Bridge")
    print("=" * 70)

    from HoloLoom.dark_trace import (
        AlignmentBridge, SafetyFeatureMapping, AlignmentConfig,
        create_alignment_bridge
    )
    from HoloLoom.dark_trace.integration.alignment import SafetyDimension

    print("\n1. Safety Feature Mapping:")
    mapping = SafetyFeatureMapping()
    # Add specific feature mappings
    mapping.add_mapping("sae.deception_42", SafetyDimension.DECEPTION)
    mapping.add_mapping("sae.helpful_12", SafetyDimension.GOAL_TRANSPARENCY)
    mapping.add_mapping("semantic.harm", SafetyDimension.HARM_POTENTIAL)

    print(f"   Feature-to-dimension mappings: {len(mapping.feature_to_dimension)}")
    print(f"   Dimensions tracked: {list(mapping.feature_to_dimension.values())}")
    print(f"   Deception threshold: {mapping.dimension_thresholds.get(SafetyDimension.DECEPTION, 0)}")

    print("\n2. Alignment Config:")
    config = AlignmentConfig(
        block_threshold=0.7,
        escalate_threshold=0.5,
        min_confidence=0.3,
        use_semantic_analysis=True,
        use_sae_features=True,
        log_evaluations=True,
    )
    print(f"   Block threshold: {config.block_threshold}")
    print(f"   Escalate threshold: {config.escalate_threshold}")
    print(f"   Use semantic analysis: {config.use_semantic_analysis}")
    print(f"   Use SAE features: {config.use_sae_features}")

    print("\n3. Alignment Bridge Creation:")
    bridge = create_alignment_bridge(config=config)
    print(f"   Bridge created: {type(bridge).__name__}")
    print(f"   Default mappings loaded: {len(bridge.mapping.feature_to_dimension)}")

    # Add custom mappings to the bridge's internal mapping
    bridge.mapping.add_mapping("sae.custom_safety", SafetyDimension.DECEPTION)
    print(f"   After custom mapping: {len(bridge.mapping.feature_to_dimension)} mappings")

    print("\n4. Safety Evaluation:")
    # Simulate feature activations (as numpy array for evaluate_safety)
    activations = np.random.normal(0.3, 0.1, (1, 10, 384))  # Low activation
    evaluation = bridge.evaluate_safety(activations)
    print(f"   Risk level: {evaluation.risk_level.value}")
    print(f"   Is blocked: {evaluation.is_blocked}")
    print(f"   Confidence: {evaluation.confidence:.3f}")
    if evaluation.reason:
        print(f"   Reason: {evaluation.reason[:60]}...")
    if evaluation.recommendation:
        print(f"   Recommendation: {evaluation.recommendation[:60]}...")

    return True


def demo_monitoring():
    """Demonstrate production monitoring and drift detection."""
    print("\n" + "=" * 70)
    print("PHASE 10: Production Monitoring")
    print("=" * 70)

    from HoloLoom.dark_trace import (
        InterpretabilityMonitor, FeatureStatistics, DriftDetector,
        MonitorConfig, create_monitor
    )

    print("\n1. Feature Statistics (Welford's Algorithm):")
    stats = FeatureStatistics(feature_id="sae.42")

    # Simulate observations
    np.random.seed(42)
    baseline_values = np.random.normal(0.5, 0.1, 100)
    for val in baseline_values:
        stats.update(val)

    print(f"   Feature: {stats.feature_id}")
    print(f"   Observations: {stats.n_observations}")
    print(f"   Mean: {stats.mean:.4f}")
    print(f"   Std: {stats.std:.4f}")
    print(f"   Min: {stats.min_activation:.4f}")
    print(f"   Max: {stats.max_activation:.4f}")
    print(f"   Sparsity: {stats.sparsity:.4f}")

    # Set baseline for drift detection
    stats.set_baseline()
    print(f"   Baseline set: mean={stats.baseline_mean:.4f}, std={stats.baseline_std:.4f}")

    print("\n2. Monitor Config:")
    config = MonitorConfig(
        recent_window_size=1000,
        mean_drift_threshold=0.1,
        enable_alerting=True,
        enable_drift_detection=True,
    )
    print(f"   Window size: {config.recent_window_size}")
    print(f"   Drift threshold: {config.mean_drift_threshold}")
    print(f"   Alerting enabled: {config.enable_alerting}")

    print("\n3. Interpretability Monitor:")
    monitor = create_monitor(config=config)

    # Record some activations
    for i in range(10):
        activations = np.random.randn(5, 384).astype(np.float32) * 0.5 + 0.3
        monitor.record(activations, metadata={"batch_id": i})

    stats_dict = monitor.get_statistics()
    print(f"   Observation count: {stats_dict['observation_count']}")
    print(f"   Features tracked: {stats_dict['features_tracked']}")
    print(f"   Avg record time: {stats_dict['avg_record_time_ms']:.3f}ms")

    print("\n4. Health Summary:")
    health = monitor.get_health_summary()
    print(f"   Status: {health['status']}")
    print(f"   Observation count: {health['observation_count']}")
    print(f"   Features tracked: {health['features_tracked']}")

    print("\n5. Drift Detection:")
    detector = DriftDetector(
        mean_threshold=0.2,
        variance_threshold=0.3,
        sparsity_threshold=0.1
    )

    # Create feature statistics for drift test
    feature_stats = {}
    for i in range(10):
        fs = FeatureStatistics(feature_id=f"feature_{i}")
        # Baseline values
        for val in np.random.normal(0.5, 0.1, 50):
            fs.update(val)
        fs.set_baseline()
        # Add drift for some features
        if i < 3:
            for val in np.random.normal(0.8, 0.15, 20):  # Mean shift
                fs.update(val)
        feature_stats[f"feature_{i}"] = fs

    drift_report = detector.detect_drift(feature_stats)
    print(f"   Has drift: {drift_report.has_drift}")
    print(f"   Drifted features: {len(drift_report.drifted_features)}")
    if drift_report.drifted_features:
        print(f"   Drift types: {[d.drift_type.value for d in drift_report.drifted_features[:3]]}")

    return True


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_performance():
    """Evaluate Dark Trace performance characteristics."""
    print("\n" + "=" * 70)
    print("PERFORMANCE EVALUATION")
    print("=" * 70)

    from HoloLoom.dark_trace import (
        DummyAdapter, create_monitor, FeatureStatistics
    )

    results = {}

    # 1. Activation extraction latency
    print("\n1. Activation Extraction Latency:")
    adapter = DummyAdapter(n_layers=12, hidden_dim=384)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        adapter.get_activations("test", "layer.6")
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    results['activation_extraction_ms'] = avg_time
    print(f"   Average: {avg_time:.3f}ms")
    print(f"   P99: {np.percentile(times, 99):.3f}ms")

    # 2. Monitor record latency
    print("\n2. Monitor Record Latency:")
    monitor = create_monitor()

    times = []
    for _ in range(100):
        activations = np.random.randn(10, 384).astype(np.float32)
        start = time.perf_counter()
        monitor.record(activations)
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    results['monitor_record_ms'] = avg_time
    print(f"   Average: {avg_time:.3f}ms")
    print(f"   P99: {np.percentile(times, 99):.3f}ms")

    # 3. Feature statistics update latency
    print("\n3. Feature Statistics Update Latency:")
    stats = FeatureStatistics(feature_id="test")

    times = []
    for i in range(10000):
        start = time.perf_counter()
        stats.update(np.random.randn())
        times.append((time.perf_counter() - start) * 1000)

    avg_time = np.mean(times)
    results['stats_update_ms'] = avg_time
    print(f"   Average: {avg_time:.6f}ms")
    print(f"   Updates/sec: {1000/avg_time:.0f}")

    # 4. Memory usage estimate
    print("\n4. Memory Usage:")
    import sys

    # Create substantial data
    adapter = DummyAdapter(n_layers=12, hidden_dim=768)
    activations = adapter.get_activations("test", "layer.6", batch_size=32, seq_len=512)

    mem_mb = activations.nbytes / (1024 * 1024)
    results['activation_memory_mb'] = mem_mb
    print(f"   Single activation (32x512x768): {mem_mb:.2f}MB")

    # Summary
    print("\n" + "-" * 50)
    print("Performance Summary:")
    print(f"   Activation extraction: {results['activation_extraction_ms']:.3f}ms")
    print(f"   Monitor record: {results['monitor_record_ms']:.3f}ms")
    print(f"   Stats update: {results['stats_update_ms']*1000:.3f}µs")
    print(f"   Memory per batch: {results['activation_memory_mb']:.2f}MB")

    # Assessment
    print("\n" + "-" * 50)
    print("Assessment:")

    total_overhead = results['activation_extraction_ms'] + results['monitor_record_ms']
    if total_overhead < 1.0:
        print(f"   [OK] Total overhead ({total_overhead:.3f}ms) is EXCELLENT (<1ms)")
    elif total_overhead < 5.0:
        print(f"   [OK] Total overhead ({total_overhead:.3f}ms) is GOOD (<5ms)")
    else:
        print(f"   [!] Total overhead ({total_overhead:.3f}ms) may impact latency")

    return results


def run_full_demo():
    """Run the complete Dark Trace demonstration."""
    print("=" * 70)
    print("DARK TRACE: Complete Interpretability Suite Demo")
    print("HoloLoom's Responsibility & Interpretability System")
    print("=" * 70)
    print()
    print("Mission: Illuminate what was previously dark through tracing.")
    print("Every decision, activation, and representation is transparent.")

    results = {
        'phase_9_adapters': False,
        'phase_9_fingerprinting': False,
        'phase_10_orchestrator': False,
        'phase_10_alignment': False,
        'phase_10_monitoring': False,
    }

    try:
        results['phase_9_adapters'] = demo_model_adapters()
    except Exception as e:
        print(f"\n[X] Phase 9 Adapters failed: {e}")

    try:
        results['phase_9_fingerprinting'] = demo_fingerprinting()
    except Exception as e:
        print(f"\n[X] Phase 9 Fingerprinting failed: {e}")

    try:
        results['phase_10_orchestrator'] = demo_traced_orchestrator()
    except Exception as e:
        print(f"\n[X] Phase 10 Orchestrator failed: {e}")

    try:
        results['phase_10_alignment'] = demo_alignment_bridge()
    except Exception as e:
        print(f"\n[X] Phase 10 Alignment failed: {e}")

    try:
        results['phase_10_monitoring'] = demo_monitoring()
    except Exception as e:
        print(f"\n[X] Phase 10 Monitoring failed: {e}")

    # Performance evaluation
    perf_results = evaluate_performance()

    # Final summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    print(f"\nComponent Tests: {passed}/{total} passed")
    for name, status in results.items():
        icon = "[OK]" if status else "[X]"
        print(f"   {icon} {name.replace('_', ' ').title()}")

    print(f"\nPerformance:")
    print(f"   Activation extraction: {perf_results.get('activation_extraction_ms', 0):.3f}ms")
    print(f"   Monitor overhead: {perf_results.get('monitor_record_ms', 0):.3f}ms")

    if passed == total:
        print("\n" + "=" * 70)
        print("[OK] ALL DARK TRACE COMPONENTS OPERATIONAL")
        print("  Interpretability is not a feature - it is the existential purpose.")
        print("=" * 70)
    else:
        print(f"\n[!] {total - passed} component(s) need attention")

    return passed == total


if __name__ == "__main__":
    success = run_full_demo()
    exit(0 if success else 1)

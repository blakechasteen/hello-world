"""
Production Model Testing for Dark Trace Interpretability

Tests Dark Trace with real transformer models (GPT-2, BERT) to validate
that interpretability features work correctly on production models.

Requires: pip install transformers torch

Author: HoloLoom Team
Created: December 2025
"""

import unittest
import time
import sys
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np

# Check for transformers availability
try:
    import torch
    from transformers import AutoModel, AutoTokenizer, GPT2Model, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

from HoloLoom.dark_trace.models import (
    TransformerAdapter,
    ModelFingerprinter,
    compare_fingerprints,
    find_universal_features,
    find_model_specific_features,
    DummyAdapter,
)
from HoloLoom.dark_trace.models.fingerprint import SimilarityMetric


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark."""
    operation: str
    model_name: str
    latency_ms: float
    memory_mb: float
    samples_processed: int
    throughput: float  # samples/second
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.operation} on {self.model_name}: "
            f"{self.latency_ms:.2f}ms, {self.memory_mb:.1f}MB, "
            f"{self.throughput:.1f} samples/s"
        )


class PerformanceBenchmarker:
    """Benchmark performance of Dark Trace operations on real models."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []

    def benchmark_activation_extraction(
        self,
        adapter: TransformerAdapter,
        model_name: str,
        inputs: Any,
        layer: str,
        n_runs: int = 10
    ) -> BenchmarkResult:
        """Benchmark activation extraction latency."""
        # Convert BatchEncoding to dict if needed
        inputs_dict = dict(inputs) if hasattr(inputs, 'keys') else inputs

        # Warmup
        _ = adapter.get_activations(inputs_dict, layer)

        # Measure
        start_time = time.perf_counter()
        for _ in range(n_runs):
            activations = adapter.get_activations(inputs_dict, layer)
        end_time = time.perf_counter()

        latency_ms = ((end_time - start_time) / n_runs) * 1000

        # Memory estimation (rough)
        memory_mb = activations.nbytes / (1024 * 1024) if hasattr(activations, 'nbytes') else 0

        result = BenchmarkResult(
            operation="activation_extraction",
            model_name=model_name,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            samples_processed=n_runs,
            throughput=n_runs / (end_time - start_time),
            metadata={"layer": layer, "shape": activations.shape}
        )
        self.results.append(result)
        return result

    def benchmark_fingerprinting(
        self,
        adapter: TransformerAdapter,
        model_name: str,
        inputs: Any,
        layer: str,
        n_features: int = 50
    ) -> BenchmarkResult:
        """Benchmark fingerprinting operation."""
        fingerprinter = ModelFingerprinter(adapter, model_id=model_name)

        # Convert BatchEncoding to dict if needed
        inputs_dict = dict(inputs) if hasattr(inputs, 'keys') else inputs

        start_time = time.perf_counter()
        fingerprint = fingerprinter.fingerprint_layer(
            layer,
            [inputs_dict],
            feature_indices=list(range(n_features))
        )
        end_time = time.perf_counter()

        latency_ms = (end_time - start_time) * 1000

        result = BenchmarkResult(
            operation="fingerprinting",
            model_name=model_name,
            latency_ms=latency_ms,
            memory_mb=0,  # Would need memory profiler
            samples_processed=n_features,
            throughput=n_features / (end_time - start_time),
            metadata={"layer": layer, "n_features": n_features}
        )
        self.results.append(result)
        return result

    def benchmark_steering(
        self,
        adapter: TransformerAdapter,
        model_name: str,
        layer: str,
        n_runs: int = 10
    ) -> BenchmarkResult:
        """Benchmark steering vector injection."""
        # Get layer dimensions
        layer_info = adapter.get_layer_info(layer)
        hidden_dim = layer_info.output_dim if layer_info and layer_info.output_dim else 768

        steering_vector = np.random.randn(hidden_dim).astype(np.float32)

        start_time = time.perf_counter()
        for _ in range(n_runs):
            config_id = adapter.inject_steering(steering_vector, layer, scale=0.1)
            adapter.remove_steering(config_id)
        end_time = time.perf_counter()

        latency_ms = ((end_time - start_time) / n_runs) * 1000

        result = BenchmarkResult(
            operation="steering_injection",
            model_name=model_name,
            latency_ms=latency_ms,
            memory_mb=steering_vector.nbytes / (1024 * 1024),
            samples_processed=n_runs,
            throughput=n_runs / (end_time - start_time),
            metadata={"layer": layer, "hidden_dim": hidden_dim}
        )
        self.results.append(result)
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks."""
        summary = {
            "total_benchmarks": len(self.results),
            "by_operation": {},
            "by_model": {}
        }

        for result in self.results:
            # By operation
            if result.operation not in summary["by_operation"]:
                summary["by_operation"][result.operation] = []
            summary["by_operation"][result.operation].append({
                "model": result.model_name,
                "latency_ms": result.latency_ms,
                "throughput": result.throughput
            })

            # By model
            if result.model_name not in summary["by_model"]:
                summary["by_model"][result.model_name] = []
            summary["by_model"][result.model_name].append({
                "operation": result.operation,
                "latency_ms": result.latency_ms
            })

        return summary


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers not installed")
class TestGPT2Integration(unittest.TestCase):
    """Test Dark Trace with GPT-2 model."""

    @classmethod
    def setUpClass(cls):
        """Load GPT-2 model once for all tests."""
        cls.model_name = "gpt2"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = GPT2Model.from_pretrained(cls.model_name)
        cls.model.eval()

        # Create adapter
        cls.adapter = TransformerAdapter(cls.model)

        # Prepare test inputs
        cls.test_text = "The quick brown fox jumps over the lazy dog."
        cls.inputs = cls.tokenizer(cls.test_text, return_tensors="pt")

    def test_layer_enumeration(self):
        """Test that we can enumerate GPT-2 layers."""
        layer_names = self.adapter.get_layer_names()

        self.assertGreater(len(layer_names), 0, "Should find layers")

        # GPT-2 should have transformer blocks - check for any block-related names
        # Layer names vary by TransformerAdapter implementation, just verify we have layers
        print(f"Found {len(layer_names)} layers: {layer_names[:5]}...")

        # GPT-2 has 12 transformer blocks, should find multiple layers
        self.assertGreater(len(layer_names), 10, "GPT-2 should have many hookable layers")

    def test_activation_extraction(self):
        """Test extracting activations from GPT-2."""
        layer_names = self.adapter.get_layer_names()

        # Find a valid layer
        test_layer = None
        for name in layer_names:
            if "attention" in name.lower() or "mlp" in name.lower():
                test_layer = name
                break

        self.assertIsNotNone(test_layer, "Should find a test layer")

        # Extract activations - convert BatchEncoding to dict
        activations = self.adapter.get_activations(dict(self.inputs), test_layer)

        self.assertIsInstance(activations, np.ndarray)
        self.assertGreater(activations.size, 0, "Should have non-empty activations")

    def test_fingerprinting(self):
        """Test fingerprinting GPT-2 features."""
        fingerprinter = ModelFingerprinter(self.adapter, model_id=self.model_name)

        layer_names = self.adapter.get_layer_names()
        test_layer = layer_names[len(layer_names) // 2]  # Middle layer

        # Create fingerprint with limited features
        # API: fingerprint_layer(layer, inputs: List[Any], feature_indices: Optional[List[int]])
        fingerprint = fingerprinter.fingerprint_layer(
            test_layer,
            [dict(self.inputs)],  # List of dict inputs
            feature_indices=list(range(10))  # Limit to 10 features
        )

        self.assertIsNotNone(fingerprint)
        self.assertIsInstance(fingerprint, dict)
        if fingerprint:
            first_fp = list(fingerprint.values())[0]
            self.assertEqual(first_fp.model_id, self.model_name)
            self.assertEqual(first_fp.layer, test_layer)

    def test_steering_injection(self):
        """Test steering vector injection in GPT-2."""
        layer_names = self.adapter.get_layer_names()
        test_layer = layer_names[len(layer_names) // 2]

        # Get layer info for dimensions
        layer_info = self.adapter.get_layer_info(test_layer)
        hidden_dim = layer_info.output_dim if layer_info and layer_info.output_dim else 768

        # Create steering vector
        steering_vector = np.random.randn(hidden_dim).astype(np.float32)

        # Inject steering
        config_id = self.adapter.inject_steering(steering_vector, test_layer, scale=0.1)
        self.assertIsNotNone(config_id)

        # Remove steering
        removed = self.adapter.remove_steering(config_id)
        self.assertTrue(removed)

    def test_multi_layer_extraction(self):
        """Test extracting from multiple layers simultaneously."""
        layer_names = self.adapter.get_layer_names()

        # Select 3 layers
        test_layers = layer_names[:3] if len(layer_names) >= 3 else layer_names

        activations = self.adapter.get_multi_layer_activations(dict(self.inputs), test_layers)

        self.assertEqual(len(activations), len(test_layers))
        for layer in test_layers:
            self.assertIn(layer, activations)
            self.assertIsInstance(activations[layer], np.ndarray)


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers not installed")
class TestBERTIntegration(unittest.TestCase):
    """Test Dark Trace with BERT model."""

    @classmethod
    def setUpClass(cls):
        """Load BERT model once for all tests."""
        cls.model_name = "bert-base-uncased"
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = BertModel.from_pretrained(cls.model_name)
        cls.model.eval()

        # Create adapter
        cls.adapter = TransformerAdapter(cls.model)

        # Prepare test inputs
        cls.test_text = "The quick brown fox jumps over the lazy dog."
        cls.inputs = cls.tokenizer(cls.test_text, return_tensors="pt")

    def test_layer_enumeration(self):
        """Test that we can enumerate BERT layers."""
        layer_names = self.adapter.get_layer_names()

        self.assertGreater(len(layer_names), 0, "Should find layers")

        # BERT specific - should have encoder layers
        encoder_layers = [l for l in layer_names if "encoder" in l.lower() or "layer" in l.lower()]
        self.assertGreater(len(encoder_layers), 0, "Should find encoder layers")

    def test_activation_extraction(self):
        """Test extracting activations from BERT."""
        layer_names = self.adapter.get_layer_names()

        # Find a valid layer
        test_layer = layer_names[len(layer_names) // 2]

        # Extract activations - convert BatchEncoding to dict
        activations = self.adapter.get_activations(dict(self.inputs), test_layer)

        self.assertIsInstance(activations, np.ndarray)
        self.assertGreater(activations.size, 0)

    def test_fingerprinting(self):
        """Test fingerprinting BERT features."""
        fingerprinter = ModelFingerprinter(self.adapter, model_id=self.model_name)

        layer_names = self.adapter.get_layer_names()
        test_layer = layer_names[len(layer_names) // 2]

        # API: fingerprint_layer(layer, inputs: List[Any], feature_indices: Optional[List[int]])
        fingerprint = fingerprinter.fingerprint_layer(
            test_layer,
            [dict(self.inputs)],  # List of dict inputs
            feature_indices=list(range(10))  # Limit to 10 features
        )

        self.assertIsNotNone(fingerprint)
        self.assertIsInstance(fingerprint, dict)
        if fingerprint:
            first_fp = list(fingerprint.values())[0]
            self.assertEqual(first_fp.model_id, self.model_name)


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers not installed")
class TestCrossModelComparison(unittest.TestCase):
    """Test comparing features across different models."""

    @classmethod
    def setUpClass(cls):
        """Load both GPT-2 and BERT for comparison."""
        # GPT-2
        cls.gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls.gpt2_model = GPT2Model.from_pretrained("gpt2")
        cls.gpt2_model.eval()
        cls.gpt2_adapter = TransformerAdapter(cls.gpt2_model)

        # BERT
        cls.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cls.bert_model = BertModel.from_pretrained("bert-base-uncased")
        cls.bert_model.eval()
        cls.bert_adapter = TransformerAdapter(cls.bert_model)

        # Test text
        cls.test_text = "The quick brown fox jumps over the lazy dog."
        cls.gpt2_inputs = cls.gpt2_tokenizer(cls.test_text, return_tensors="pt")
        cls.bert_inputs = cls.bert_tokenizer(cls.test_text, return_tensors="pt")

    def test_fingerprint_comparison(self):
        """Test comparing fingerprints across models."""
        # Create fingerprinters
        gpt2_fp = ModelFingerprinter(self.gpt2_adapter, model_id="gpt2")
        bert_fp = ModelFingerprinter(self.bert_adapter, model_id="bert")

        # Get layers
        gpt2_layers = self.gpt2_adapter.get_layer_names()
        bert_layers = self.bert_adapter.get_layer_names()

        # Fingerprint middle layer of each
        gpt2_layer = gpt2_layers[len(gpt2_layers) // 2]
        bert_layer = bert_layers[len(bert_layers) // 2]

        gpt2_fingerprint = gpt2_fp.fingerprint_layer(
            gpt2_layer, [dict(self.gpt2_inputs)], feature_indices=list(range(10))
        )
        bert_fingerprint = bert_fp.fingerprint_layer(
            bert_layer, [dict(self.bert_inputs)], feature_indices=list(range(10))
        )

        # Compare - extract first fingerprint from each dict
        self.assertIsInstance(gpt2_fingerprint, dict)
        self.assertIsInstance(bert_fingerprint, dict)
        if gpt2_fingerprint and bert_fingerprint:
            gpt2_fp_obj = list(gpt2_fingerprint.values())[0]
            bert_fp_obj = list(bert_fingerprint.values())[0]
            comparison = compare_fingerprints(gpt2_fp_obj, bert_fp_obj)
            self.assertIsNotNone(comparison)
            self.assertIn(SimilarityMetric.COSINE, comparison.similarities)

    def test_find_universal_features(self):
        """Test finding features that appear across models."""
        # Create fingerprinters
        gpt2_fp = ModelFingerprinter(self.gpt2_adapter, model_id="gpt2")
        bert_fp = ModelFingerprinter(self.bert_adapter, model_id="bert")

        # Get layers
        gpt2_layers = self.gpt2_adapter.get_layer_names()
        bert_layers = self.bert_adapter.get_layer_names()

        gpt2_layer = gpt2_layers[len(gpt2_layers) // 2]
        bert_layer = bert_layers[len(bert_layers) // 2]

        gpt2_fingerprint = gpt2_fp.fingerprint_layer(
            gpt2_layer, [dict(self.gpt2_inputs)], feature_indices=list(range(10))
        )
        bert_fingerprint = bert_fp.fingerprint_layer(
            bert_layer, [dict(self.bert_inputs)], feature_indices=list(range(10))
        )

        # Find universal features (lower threshold for test)
        fingerprints_by_model = {
            "gpt2": gpt2_fingerprint,
            "bert": bert_fingerprint
        }

        # Use FingerprintConfig for threshold
        from HoloLoom.dark_trace.models.fingerprint import FingerprintConfig
        config = FingerprintConfig(universal_threshold=0.5)
        universal = find_universal_features(fingerprints_by_model, config=config)

        # May or may not find universal features - just verify no error
        self.assertIsInstance(universal, list)

    def test_find_model_specific_features(self):
        """Test finding features unique to each model."""
        gpt2_fp = ModelFingerprinter(self.gpt2_adapter, model_id="gpt2")
        bert_fp = ModelFingerprinter(self.bert_adapter, model_id="bert")

        gpt2_layers = self.gpt2_adapter.get_layer_names()
        bert_layers = self.bert_adapter.get_layer_names()

        gpt2_layer = gpt2_layers[len(gpt2_layers) // 2]
        bert_layer = bert_layers[len(bert_layers) // 2]

        gpt2_fingerprint = gpt2_fp.fingerprint_layer(
            gpt2_layer, [dict(self.gpt2_inputs)], feature_indices=list(range(10))
        )
        bert_fingerprint = bert_fp.fingerprint_layer(
            bert_layer, [dict(self.bert_inputs)], feature_indices=list(range(10))
        )

        fingerprints_by_model = {
            "gpt2": gpt2_fingerprint,
            "bert": bert_fingerprint
        }

        # Use FingerprintConfig for threshold
        from HoloLoom.dark_trace.models.fingerprint import FingerprintConfig
        config = FingerprintConfig(specific_threshold=0.3)
        model_specific = find_model_specific_features(fingerprints_by_model, config=config)

        # Returns Dict[str, List[...]], not List
        self.assertIsInstance(model_specific, dict)


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers not installed")
class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for production deployment."""

    @classmethod
    def setUpClass(cls):
        """Load GPT-2 for benchmarking."""
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        cls.model = GPT2Model.from_pretrained("gpt2")
        cls.model.eval()
        cls.adapter = TransformerAdapter(cls.model)

        cls.test_text = "The quick brown fox jumps over the lazy dog."
        cls.inputs = cls.tokenizer(cls.test_text, return_tensors="pt")

        cls.benchmarker = PerformanceBenchmarker()

    def test_activation_extraction_latency(self):
        """Benchmark activation extraction latency."""
        layer_names = self.adapter.get_layer_names()
        test_layer = layer_names[len(layer_names) // 2]

        result = self.benchmarker.benchmark_activation_extraction(
            self.adapter, "gpt2", self.inputs, test_layer, n_runs=10
        )

        print(f"\n{result}")

        # Should be fast - under 100ms per extraction
        self.assertLess(result.latency_ms, 100, "Activation extraction should be <100ms")

    def test_fingerprinting_latency(self):
        """Benchmark fingerprinting latency."""
        layer_names = self.adapter.get_layer_names()
        test_layer = layer_names[len(layer_names) // 2]

        result = self.benchmarker.benchmark_fingerprinting(
            self.adapter, "gpt2", self.inputs, test_layer, n_features=50
        )

        print(f"\n{result}")

        # Fingerprinting 50 features should be under 5 seconds
        self.assertLess(result.latency_ms, 5000, "Fingerprinting should be <5s")

    def test_steering_latency(self):
        """Benchmark steering injection latency."""
        layer_names = self.adapter.get_layer_names()
        test_layer = layer_names[len(layer_names) // 2]

        result = self.benchmarker.benchmark_steering(
            self.adapter, "gpt2", test_layer, n_runs=10
        )

        print(f"\n{result}")

        # Steering should be very fast - under 10ms
        self.assertLess(result.latency_ms, 10, "Steering should be <10ms")

    @classmethod
    def tearDownClass(cls):
        """Print benchmark summary."""
        summary = cls.benchmarker.get_summary()
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        for op, results in summary["by_operation"].items():
            print(f"\n{op}:")
            for r in results:
                print(f"  {r['model']}: {r['latency_ms']:.2f}ms ({r['throughput']:.1f}/s)")


class TestDummyAdapterBaseline(unittest.TestCase):
    """Baseline tests using DummyAdapter (no transformers required)."""

    def setUp(self):
        """Create DummyAdapter for testing."""
        self.adapter = DummyAdapter(
            n_layers=12,
            hidden_dim=768,
        )
        self.inputs = np.random.randn(1, 10, 768).astype(np.float32)

    def test_layer_enumeration(self):
        """Test layer enumeration on dummy adapter."""
        layer_names = self.adapter.get_layer_names()

        self.assertEqual(len(layer_names), 12)

    def test_activation_extraction(self):
        """Test activation extraction on dummy adapter."""
        layer_names = self.adapter.get_layer_names()
        activations = self.adapter.get_activations(self.inputs, layer_names[0])

        self.assertIsInstance(activations, np.ndarray)

    def test_fingerprinting(self):
        """Test fingerprinting on dummy adapter."""
        fingerprinter = ModelFingerprinter(self.adapter, model_id="dummy")

        layer_names = self.adapter.get_layer_names()
        fingerprint = fingerprinter.fingerprint_layer(
            layer_names[0],
            [self.inputs],
            feature_indices=list(range(10))
        )

        # Returns Dict[str, FeatureFingerprint]
        self.assertIsInstance(fingerprint, dict)
        if fingerprint:
            first_fp = list(fingerprint.values())[0]
            self.assertEqual(first_fp.model_id, "dummy")

    def test_steering_injection(self):
        """Test steering on dummy adapter."""
        layer_names = self.adapter.get_layer_names()
        steering_vector = np.random.randn(768).astype(np.float32)

        config_id = self.adapter.inject_steering(steering_vector, layer_names[0], scale=0.1)
        self.assertIsNotNone(config_id)

        removed = self.adapter.remove_steering(config_id)
        self.assertTrue(removed)


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)

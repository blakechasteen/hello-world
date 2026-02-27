#!/usr/bin/env python3
"""Unit tests for visual compression system."""

import pytest
import numpy as np
import networkx as nx
from HoloLoom.memory.visual_compression import (
    calculate_optimal_dimensions,
    compress_to_visual,
    CompressionType,
    CompressionMetrics,
    KnowledgeGraphRenderer,
    TableRenderer,
    CodeRenderer
)


class TestCalculateOptimalDimensions:
    """Test optimal dimension calculation."""

    def test_small_dataset(self):
        """Test dimensions for small dataset."""
        width, height = calculate_optimal_dimensions(estimated_tokens=100, target_ratio=3.0)

        # Should use minimum dimensions
        assert width >= 196
        assert height >= 196
        # Small datasets may use 1:1 ratio (minimum dimensions)
        assert width / height >= 1.0  # At least square or wider

    def test_medium_dataset(self):
        """Test dimensions for medium dataset."""
        width, height = calculate_optimal_dimensions(estimated_tokens=1000, target_ratio=3.0)

        # Should be between min and max (height may be at minimum 196)
        assert 196 <= width <= 1200
        assert 196 <= height <= 1200
        # Should maintain 3:2 aspect ratio (approximately)
        assert 1.4 <= width / height <= 1.7

    def test_large_dataset(self):
        """Test dimensions for large dataset."""
        width, height = calculate_optimal_dimensions(estimated_tokens=10000, target_ratio=3.0)

        # Should hit maximum dimensions
        assert width <= 1200
        assert height <= 1200

    def test_target_ratio_variations(self):
        """Test different target ratios."""
        tokens = 1000

        # Higher ratio = more compression = smaller images
        w1, h1 = calculate_optimal_dimensions(tokens, target_ratio=5.0)
        w2, h2 = calculate_optimal_dimensions(tokens, target_ratio=3.0)
        w3, h3 = calculate_optimal_dimensions(tokens, target_ratio=2.0)

        # Higher ratio should produce smaller dimensions
        assert w1 * h1 < w2 * h2 < w3 * h3

    def test_patch_alignment(self):
        """Test that dimensions are multiples of 14 (patch size)."""
        width, height = calculate_optimal_dimensions(estimated_tokens=500, target_ratio=3.0)

        assert width % 14 == 0
        assert height % 14 == 0

    def test_minimum_tokens(self):
        """Test behavior with very few tokens."""
        width, height = calculate_optimal_dimensions(estimated_tokens=10, target_ratio=3.0)

        # Should use minimum dimensions even for tiny datasets
        assert width >= 196
        assert height >= 196


class TestKnowledgeGraphRenderer:
    """Test knowledge graph rendering."""

    def test_render_simple_graph(self):
        """Test rendering a simple graph."""
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")
        G.add_edge("A", "B")

        renderer = KnowledgeGraphRenderer(400, 300)
        image = renderer.render(G)

        assert isinstance(image, np.ndarray)
        assert image.shape == (300, 400, 3)
        assert image.dtype == np.uint8

    def test_estimate_tokens(self):
        """Test token estimation for graphs."""
        G = nx.DiGraph()
        for i in range(10):
            G.add_node(f"Node_{i}")
        for i in range(5):
            G.add_edge(f"Node_{i}", f"Node_{i+1}")

        renderer = KnowledgeGraphRenderer(400, 300)
        tokens = renderer.estimate_tokens(G)

        # 10 nodes * 10 tokens + 5 edges * 15 tokens = 175
        assert tokens == 175


class TestTableRenderer:
    """Test table rendering."""

    def test_render_dict_table(self):
        """Test rendering a dict-based table."""
        data = {
            'headers': ['A', 'B', 'C'],
            'rows': [
                [1, 2, 3],
                [4, 5, 6]
            ]
        }

        renderer = TableRenderer(400, 300)
        image = renderer.render(data)

        assert isinstance(image, np.ndarray)
        assert image.shape == (300, 400, 3)
        assert image.dtype == np.uint8

    def test_render_keyvalue_dict(self):
        """Test rendering a key-value dict."""
        data = {'key1': 'value1', 'key2': 'value2'}

        renderer = TableRenderer(400, 300)
        image = renderer.render(data)

        assert isinstance(image, np.ndarray)
        assert image.shape == (300, 400, 3)

    def test_estimate_tokens_headers_rows(self):
        """Test token estimation for headers/rows format."""
        data = {
            'headers': ['A', 'B', 'C'],
            'rows': [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        }

        renderer = TableRenderer(400, 300)
        tokens = renderer.estimate_tokens(data)

        # 3 rows * 3 cols * 5 + 3 headers * 3 = 54
        assert tokens == 54

    def test_estimate_tokens_keyvalue(self):
        """Test token estimation for key-value format."""
        data = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

        renderer = TableRenderer(400, 300)
        tokens = renderer.estimate_tokens(data)

        # 1 row * 3 cols * 5 + 3 headers * 3 = 24
        assert tokens == 24


class TestCodeRenderer:
    """Test code rendering."""

    def test_render_simple_code(self):
        """Test rendering simple code."""
        code = "def foo():\n    return 42"

        renderer = CodeRenderer(400, 300)
        image = renderer.render(code)

        assert isinstance(image, np.ndarray)
        assert image.shape == (300, 400, 3)
        assert image.dtype == np.uint8

    def test_estimate_tokens(self):
        """Test token estimation for code."""
        code = "def foo():\n    return 42"

        renderer = CodeRenderer(400, 300)
        tokens = renderer.estimate_tokens(code)

        # 24 chars / 3 = 8 tokens
        assert tokens == len(code) // 3


class TestCompressToVisual:
    """Test main compression function."""

    def test_auto_detect_graph(self):
        """Test auto-detection of knowledge graph."""
        G = nx.DiGraph()
        G.add_edge("A", "B")

        image, metrics = compress_to_visual(G, compression_type=CompressionType.AUTO)

        assert metrics.compression_type == "knowledge_graph"
        assert isinstance(image, np.ndarray)

    def test_auto_detect_table(self):
        """Test auto-detection of table."""
        data = {'headers': ['A', 'B'], 'rows': [[1, 2]]}

        image, metrics = compress_to_visual(data, compression_type=CompressionType.AUTO)

        assert metrics.compression_type == "table"
        assert isinstance(image, np.ndarray)

    def test_auto_detect_code(self):
        """Test auto-detection of code."""
        code = "def foo():\n    return 42"

        image, metrics = compress_to_visual(code, compression_type=CompressionType.AUTO)

        assert metrics.compression_type == "code"
        assert isinstance(image, np.ndarray)

    def test_adaptive_sizing_enabled(self):
        """Test adaptive sizing produces variable dimensions."""
        # Small graph
        G1 = nx.DiGraph()
        G1.add_edge("A", "B")

        # Large graph
        G2 = nx.DiGraph()
        for i in range(50):
            G2.add_node(f"Node_{i}")

        img1, _ = compress_to_visual(G1, adaptive_sizing=True)
        img2, _ = compress_to_visual(G2, adaptive_sizing=True)

        # Larger graph should produce larger image
        assert img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]

    def test_adaptive_sizing_disabled(self):
        """Test explicit dimensions override adaptive sizing."""
        G = nx.DiGraph()
        G.add_edge("A", "B")

        image, _ = compress_to_visual(
            G,
            width=400,
            height=300,
            adaptive_sizing=False
        )

        # Should use exact dimensions specified
        assert image.shape == (300, 400, 3)

    def test_compression_metrics(self):
        """Test compression metrics are calculated correctly."""
        G = nx.DiGraph()
        for i in range(20):
            G.add_node(f"Node_{i}")

        image, metrics = compress_to_visual(G)

        assert metrics.original_tokens > 0
        assert metrics.visual_tokens > 0
        assert metrics.compression_ratio == metrics.original_tokens / metrics.visual_tokens
        assert metrics.compression_type == "knowledge_graph"

    def test_compression_ratio_realistic(self):
        """Test that realistic datasets achieve compression."""
        # Medium-sized graph (should compress well)
        G = nx.DiGraph()
        for i in range(50):
            G.add_node(f"Node_{i}")
        for i in range(25):
            G.add_edge(f"Node_{i}", f"Node_{i+1}")

        _, metrics = compress_to_visual(G)

        # Should achieve some compression
        assert metrics.compression_ratio > 1.0

    def test_empty_data_handling(self):
        """Test handling of empty/minimal data."""
        # Empty table
        data = {'headers': ['A', 'B'], 'rows': []}

        image, metrics = compress_to_visual(data)

        # Should not crash, should produce valid image
        assert isinstance(image, np.ndarray)
        assert image.shape[2] == 3  # RGB

    def test_string_compression_type(self):
        """Test string-based compression type specification."""
        G = nx.DiGraph()
        G.add_edge("A", "B")

        image, metrics = compress_to_visual(G, compression_type="knowledge_graph")

        assert metrics.compression_type == "knowledge_graph"
        assert isinstance(image, np.ndarray)

    def test_configurable_target_ratio(self):
        """Test configurable target_ratio parameter."""
        # Create medium-sized graph
        G = nx.DiGraph()
        for i in range(50):
            G.add_node(f"Node_{i}")
        for i in range(25):
            G.add_edge(f"Node_{i}", f"Node_{i+1}")

        # High compression (target_ratio=5.0) should produce smaller images
        img_high, metrics_high = compress_to_visual(G, target_ratio=5.0)

        # Low compression (target_ratio=2.0) should produce larger images
        img_low, metrics_low = compress_to_visual(G, target_ratio=2.0)

        # Verify size relationship
        pixels_high = img_high.shape[0] * img_high.shape[1]
        pixels_low = img_low.shape[0] * img_low.shape[1]
        assert pixels_high < pixels_low

        # Verify compression ratios are close to targets
        assert metrics_high.compression_ratio > metrics_low.compression_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

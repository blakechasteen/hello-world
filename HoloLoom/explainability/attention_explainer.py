"""
Attention Visualization - What did the model focus on?

Visualizes attention patterns to understand what parts of the input
the model pays attention to when making decisions.

Research:
- Bahdanau et al. (2015): Neural Machine Translation by Jointly Learning to Align and Translate
- Vaswani et al. (2017): Attention is All You Need (Transformer)
- Selvaraju et al. (2017): Grad-CAM - Gradient-weighted Class Activation Mapping
- Vig (2019): A Multiscale Visualization of Attention in the Transformer Model
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class AttentionPattern(Enum):
    """Common attention patterns"""
    UNIFORM = "uniform"  # Equal attention everywhere
    FOCUSED = "focused"  # Strong focus on few elements
    LOCAL = "local"  # Attends to nearby elements
    GLOBAL = "global"  # Attends across full sequence
    HIERARCHICAL = "hierarchical"  # Multi-level attention
    SPARSE = "sparse"  # Few strong attention points


@dataclass
class AttentionHeatmap:
    """Attention heatmap visualization data"""
    queries: List[str]  # Query elements
    keys: List[str]  # Key elements
    weights: List[List[float]]  # Attention weights [queries x keys]

    # Metadata
    pattern: AttentionPattern
    head_id: Optional[int] = None  # Multi-head attention head ID
    layer_id: Optional[int] = None  # Layer ID
    entropy: Optional[float] = None  # Attention entropy (uncertainty)
    max_weight: float = 1.0
    min_weight: float = 0.0

    def __post_init__(self):
        """Compute statistics"""
        if self.entropy is None and NUMPY_AVAILABLE:
            # Compute attention entropy
            weights_array = np.array(self.weights)
            entropies = []
            for row in weights_array:
                # Normalize to probability distribution
                prob = row / (row.sum() + 1e-10)
                # Shannon entropy: -Σ p log p
                entropy = -np.sum(prob * np.log(prob + 1e-10))
                entropies.append(entropy)
            self.entropy = float(np.mean(entropies))

    def top_k_attended(self, k: int = 5) -> List[Tuple[str, str, float]]:
        """Return top-k most attended (query, key, weight) pairs"""
        attended = []
        for i, query in enumerate(self.queries):
            for j, key in enumerate(self.keys):
                weight = self.weights[i][j]
                attended.append((query, key, weight))

        # Sort by weight descending
        attended.sort(key=lambda x: x[2], reverse=True)
        return attended[:k]


class AttentionExplainer:
    """
    Explain model decisions through attention visualization.

    Works with transformer-based models and any model with attention mechanisms.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        """
        Args:
            model: Model with attention mechanisms
            num_heads: Number of attention heads (multi-head attention)
            num_layers: Number of attention layers
        """
        self.model = model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self._attention_cache = {}  # Cache attention weights

    def extract_attention(
        self,
        input_tokens: List[str],
        output_tokens: Optional[List[str]] = None,
        layer: Optional[int] = None,
        head: Optional[int] = None
    ) -> List[AttentionHeatmap]:
        """
        Extract attention weights from model.

        Args:
            input_tokens: Input sequence tokens
            output_tokens: Output sequence tokens (for encoder-decoder)
            layer: Specific layer to visualize (None = all layers)
            head: Specific head to visualize (None = all heads)

        Returns:
            List of attention heatmaps
        """
        if self.model is None:
            # Generate synthetic attention for demo
            return self._synthetic_attention(input_tokens, output_tokens)

        # Extract from real model
        if TORCH_AVAILABLE and hasattr(self.model, 'attention_weights'):
            return self._extract_from_pytorch(input_tokens, output_tokens, layer, head)
        else:
            return self._synthetic_attention(input_tokens, output_tokens)

    def _extract_from_pytorch(
        self,
        input_tokens: List[str],
        output_tokens: Optional[List[str]],
        layer: Optional[int],
        head: Optional[int]
    ) -> List[AttentionHeatmap]:
        """Extract attention from PyTorch model"""
        # Assume model exposes attention_weights attribute
        # Shape: [batch, num_layers, num_heads, seq_len, seq_len]
        attention_weights = self.model.attention_weights

        if not TORCH_AVAILABLE:
            return []

        heatmaps = []

        if isinstance(attention_weights, torch.Tensor):
            # Convert to numpy
            attn_np = attention_weights.detach().cpu().numpy()

            # Extract specific layer/head or all
            batch_size, num_layers, num_heads, seq_len, _ = attn_np.shape

            layers_to_viz = [layer] if layer is not None else range(num_layers)
            heads_to_viz = [head] if head is not None else range(num_heads)

            for l in layers_to_viz:
                for h in heads_to_viz:
                    weights = attn_np[0, l, h, :, :].tolist()  # [seq_len, seq_len]

                    # Determine pattern
                    pattern = self._classify_attention_pattern(weights)

                    heatmap = AttentionHeatmap(
                        queries=input_tokens,
                        keys=output_tokens or input_tokens,
                        weights=weights,
                        pattern=pattern,
                        layer_id=l,
                        head_id=h
                    )
                    heatmaps.append(heatmap)

        return heatmaps

    def _synthetic_attention(
        self,
        input_tokens: List[str],
        output_tokens: Optional[List[str]] = None
    ) -> List[AttentionHeatmap]:
        """
        Generate synthetic attention for demonstration.

        Useful when real model attention is not available.
        """
        keys = output_tokens or input_tokens
        n_queries = len(input_tokens)
        n_keys = len(keys)

        # Create different attention patterns
        import random

        # Pattern 1: Diagonal (local attention)
        diagonal_weights = [[0.0] * n_keys for _ in range(n_queries)]
        for i in range(n_queries):
            for j in range(n_keys):
                # Higher weight for nearby positions
                distance = abs(i - j)
                diagonal_weights[i][j] = max(0, 1.0 - 0.2 * distance)

        # Normalize rows to sum to 1
        diagonal_weights = self._normalize_attention(diagonal_weights)

        heatmap1 = AttentionHeatmap(
            queries=input_tokens,
            keys=keys,
            weights=diagonal_weights,
            pattern=AttentionPattern.LOCAL,
            layer_id=0,
            head_id=0
        )

        # Pattern 2: Focused (sparse attention)
        focused_weights = [[0.1] * n_keys for _ in range(n_queries)]
        for i in range(n_queries):
            # Focus on 1-2 key positions
            focus_idx = random.randint(0, n_keys - 1)
            focused_weights[i][focus_idx] = 0.8

        focused_weights = self._normalize_attention(focused_weights)

        heatmap2 = AttentionHeatmap(
            queries=input_tokens,
            keys=keys,
            weights=focused_weights,
            pattern=AttentionPattern.FOCUSED,
            layer_id=0,
            head_id=1
        )

        # Pattern 3: Global (uniform attention)
        uniform_weights = [[1.0 / n_keys] * n_keys for _ in range(n_queries)]

        heatmap3 = AttentionHeatmap(
            queries=input_tokens,
            keys=keys,
            weights=uniform_weights,
            pattern=AttentionPattern.GLOBAL,
            layer_id=1,
            head_id=0
        )

        return [heatmap1, heatmap2, heatmap3]

    def _normalize_attention(self, weights: List[List[float]]) -> List[List[float]]:
        """Normalize attention weights to sum to 1 per query"""
        normalized = []
        for row in weights:
            total = sum(row)
            if total > 0:
                normalized.append([w / total for w in row])
            else:
                normalized.append([1.0 / len(row)] * len(row))
        return normalized

    def _classify_attention_pattern(self, weights: List[List[float]]) -> AttentionPattern:
        """Classify attention pattern based on weight distribution"""
        if not NUMPY_AVAILABLE:
            return AttentionPattern.UNIFORM

        weights_array = np.array(weights)

        # Compute statistics
        max_weights = weights_array.max(axis=1)  # Max per query
        mean_max = max_weights.mean()

        # Check sparsity (how many weights > 0.1)
        sparsity = (weights_array > 0.1).sum() / weights_array.size

        # Check locality (diagonal dominance)
        diagonal_sum = np.trace(weights_array)
        off_diagonal_sum = weights_array.sum() - diagonal_sum
        locality = diagonal_sum / (diagonal_sum + off_diagonal_sum + 1e-10)

        # Classify
        if mean_max > 0.7:
            return AttentionPattern.FOCUSED
        elif sparsity < 0.3:
            return AttentionPattern.SPARSE
        elif locality > 0.5:
            return AttentionPattern.LOCAL
        elif 0.4 < mean_max < 0.6:
            return AttentionPattern.UNIFORM
        else:
            return AttentionPattern.GLOBAL

    def visualize_attention_text(
        self,
        heatmap: AttentionHeatmap,
        top_k: int = 5
    ) -> str:
        """
        Generate text-based attention visualization.

        Args:
            heatmap: Attention heatmap to visualize
            top_k: Number of top attended pairs to show

        Returns:
            Text representation of attention
        """
        lines = []
        lines.append(f"=== Attention Heatmap (Layer {heatmap.layer_id}, Head {heatmap.head_id}) ===")
        lines.append(f"Pattern: {heatmap.pattern.value}")
        lines.append(f"Entropy: {heatmap.entropy:.3f}" if heatmap.entropy else "")
        lines.append("")

        # Top-k attended pairs
        lines.append(f"Top {top_k} Attended Pairs:")
        for query, key, weight in heatmap.top_k_attended(top_k):
            # Visual bar
            bar_len = int(weight * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            lines.append(f"  {query:15s} → {key:15s} [{bar}] {weight:.3f}")

        lines.append("")

        # Attention matrix (simplified)
        lines.append("Attention Matrix (rows=queries, cols=keys):")
        lines.append("      " + " ".join(f"{k[:4]:>5s}" for k in heatmap.keys[:10]))  # Max 10 cols

        for i, query in enumerate(heatmap.queries[:10]):  # Max 10 rows
            row_str = f"{query[:4]:>5s} "
            for j in range(min(10, len(heatmap.keys))):
                weight = heatmap.weights[i][j]
                # Use Unicode blocks for heatmap
                if weight > 0.7:
                    char = "█"
                elif weight > 0.5:
                    char = "▓"
                elif weight > 0.3:
                    char = "▒"
                elif weight > 0.1:
                    char = "░"
                else:
                    char = " "
                row_str += f"  {char}   "
            lines.append(row_str)

        return "\n".join(lines)

    def analyze_attention_flow(
        self,
        heatmaps: List[AttentionHeatmap]
    ) -> Dict[str, Any]:
        """
        Analyze attention flow across layers/heads.

        Args:
            heatmaps: List of attention heatmaps from different layers/heads

        Returns:
            Analysis summary
        """
        analysis = {
            'num_layers': max(h.layer_id for h in heatmaps if h.layer_id is not None) + 1,
            'num_heads': max(h.head_id for h in heatmaps if h.head_id is not None) + 1,
            'patterns': {},
            'avg_entropy': 0.0,
            'most_attended_tokens': [],
        }

        # Pattern distribution
        for heatmap in heatmaps:
            pattern = heatmap.pattern.value
            analysis['patterns'][pattern] = analysis['patterns'].get(pattern, 0) + 1

        # Average entropy
        entropies = [h.entropy for h in heatmaps if h.entropy is not None]
        if entropies:
            analysis['avg_entropy'] = sum(entropies) / len(entropies)

        # Most attended tokens (aggregate across all heatmaps)
        token_attention = {}
        for heatmap in heatmaps:
            for i, key in enumerate(heatmap.keys):
                # Sum attention received by this key
                total_attention = sum(heatmap.weights[j][i] for j in range(len(heatmap.queries)))
                token_attention[key] = token_attention.get(key, 0) + total_attention

        # Top-10 most attended
        analysis['most_attended_tokens'] = sorted(
            token_attention.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return analysis


def visualize_attention(
    input_tokens: List[str],
    output_tokens: Optional[List[str]] = None,
    model: Optional[Any] = None
) -> str:
    """
    Convenience function to visualize attention.

    Args:
        input_tokens: Input sequence
        output_tokens: Output sequence (optional)
        model: Model with attention (optional)

    Returns:
        Text visualization of attention
    """
    explainer = AttentionExplainer(model=model)
    heatmaps = explainer.extract_attention(input_tokens, output_tokens)

    output = []
    output.append("=" * 80)
    output.append("ATTENTION VISUALIZATION")
    output.append("=" * 80)
    output.append("")

    # Visualize each heatmap
    for heatmap in heatmaps[:3]:  # Show first 3
        output.append(explainer.visualize_attention_text(heatmap, top_k=5))
        output.append("")

    # Overall analysis
    analysis = explainer.analyze_attention_flow(heatmaps)
    output.append("=" * 80)
    output.append("ATTENTION ANALYSIS")
    output.append("=" * 80)
    output.append(f"Layers: {analysis['num_layers']}, Heads: {analysis['num_heads']}")
    output.append(f"Average Entropy: {analysis['avg_entropy']:.3f}")
    output.append("")
    output.append("Pattern Distribution:")
    for pattern, count in analysis['patterns'].items():
        output.append(f"  {pattern:15s}: {count:3d}")
    output.append("")
    output.append("Most Attended Tokens:")
    for token, attention in analysis['most_attended_tokens']:
        bar_len = int((attention / analysis['most_attended_tokens'][0][1]) * 40)
        bar = "█" * bar_len
        output.append(f"  {token:20s} [{bar:40s}] {attention:.2f}")

    return "\n".join(output)

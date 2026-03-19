from __future__ import annotations
"""
Dark Trace SAE: Auto-Labeling Pipeline

LLM-powered feature labeling with semantic alignment scoring.
Based on Anthropic's methodology for interpretable feature descriptions.

Components:
- ActivationPatternAnalyzer: Find top-k activating examples per feature
- SemanticAlignmentScorer: Correlation with 244 semantic axes
- LabelCache: Persistent storage for labels
- FeatureLabeler: Main labeling orchestrator

Research Foundation:
- Anthropic's feature labeling methodology (2023-2024)
- Semantic alignment with interpretable dimensions
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Optional: UnifiedLLMClient for multi-provider support
try:
    from hololoom.llm.unified_client import LLMConfig, UnifiedLLMClient
    HAS_UNIFIED_CLIENT = True
except ImportError:
    HAS_UNIFIED_CLIENT = False
    UnifiedLLMClient = None
    LLMConfig = None

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureLabel:
    """Label for a single SAE feature."""

    feature_idx: int
    description: str
    confidence: float  # 0.0-1.0 confidence in label

    # Supporting evidence
    top_activating_examples: list[str] = field(default_factory=list)
    activation_pattern: str = ""  # Summary of activation pattern

    # Semantic alignment
    semantic_alignments: dict[str, float] = field(default_factory=dict)  # axis -> correlation
    primary_semantic_axis: str | None = None
    semantic_alignment_score: float = 0.0

    # Metadata
    labeled_at: str | None = None
    labeler_model: str = "unknown"
    version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "feature_idx": self.feature_idx,
            "description": self.description,
            "confidence": self.confidence,
            "top_activating_examples": self.top_activating_examples,
            "activation_pattern": self.activation_pattern,
            "semantic_alignments": self.semantic_alignments,
            "primary_semantic_axis": self.primary_semantic_axis,
            "semantic_alignment_score": self.semantic_alignment_score,
            "labeled_at": self.labeled_at,
            "labeler_model": self.labeler_model,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureLabel":
        """Deserialize from dictionary."""
        return cls(
            feature_idx=data["feature_idx"],
            description=data["description"],
            confidence=data["confidence"],
            top_activating_examples=data.get("top_activating_examples", []),
            activation_pattern=data.get("activation_pattern", ""),
            semantic_alignments=data.get("semantic_alignments", {}),
            primary_semantic_axis=data.get("primary_semantic_axis"),
            semantic_alignment_score=data.get("semantic_alignment_score", 0.0),
            labeled_at=data.get("labeled_at"),
            labeler_model=data.get("labeler_model", "unknown"),
            version=data.get("version", "1.0"),
        )


# =============================================================================
# Activation Pattern Analyzer
# =============================================================================

class ActivationPatternAnalyzer:
    """
    Analyze activation patterns to find top-k activating examples.

    For each feature, finds the inputs that maximally activate it,
    providing evidence for what concept the feature represents.
    """

    def __init__(
        self,
        top_k: int = 10,
        min_activation_threshold: float = 0.1,
    ):
        self.top_k = top_k
        self.min_activation_threshold = min_activation_threshold

        # Storage for examples and their activations
        self._example_texts: list[str] = []
        self._activations: torch.Tensor | None = None  # [n_examples, n_features]

    def collect_activations(
        self,
        sae: torch.nn.Module,
        activations: torch.Tensor,  # [n_examples, input_dim]
        example_texts: list[str],
    ) -> None:
        """
        Collect activation patterns from a batch of examples.

        Args:
            sae: The SAE model
            activations: Input activations [n_examples, input_dim]
            example_texts: Corresponding text for each example
        """
        sae.eval()
        with torch.no_grad():
            _, latents, _ = sae(activations)  # [n_examples, n_features]

            if self._activations is None:
                self._activations = latents.cpu()
            else:
                self._activations = torch.cat([self._activations, latents.cpu()], dim=0)

            self._example_texts.extend(example_texts)

    def get_top_activating_examples(
        self,
        feature_idx: int,
    ) -> tuple[list[str], list[float]]:
        """
        Get top-k examples that maximally activate a feature.

        Args:
            feature_idx: Index of the feature

        Returns:
            Tuple of (examples, activation_values)
        """
        if self._activations is None or len(self._example_texts) == 0:
            return [], []

        # Get activations for this feature
        feature_acts = self._activations[:, feature_idx]  # [n_examples]

        # Filter by minimum threshold
        mask = feature_acts >= self.min_activation_threshold
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) == 0:
            return [], []

        # Get top-k from valid indices
        valid_acts = feature_acts[valid_indices]
        k = min(self.top_k, len(valid_acts))
        top_k_rel_indices = torch.topk(valid_acts, k).indices
        top_k_indices = valid_indices[top_k_rel_indices]

        examples = [self._example_texts[i] for i in top_k_indices.tolist()]
        act_values = feature_acts[top_k_indices].tolist()

        return examples, act_values

    def get_activation_pattern_summary(
        self,
        feature_idx: int,
    ) -> str:
        """
        Generate a summary of the activation pattern for a feature.

        Returns description of when/how the feature activates.
        """
        if self._activations is None:
            return "No activation data collected."

        feature_acts = self._activations[:, feature_idx]

        # Compute statistics
        total_examples = len(feature_acts)
        active_count = (feature_acts > 0).sum().item()
        active_ratio = active_count / total_examples if total_examples > 0 else 0

        mean_when_active = feature_acts[feature_acts > 0].mean().item() if active_count > 0 else 0
        max_activation = feature_acts.max().item()

        # Categorize sparsity
        if active_ratio < 0.01:
            sparsity_desc = "very sparse (activates rarely)"
        elif active_ratio < 0.05:
            sparsity_desc = "sparse (activates occasionally)"
        elif active_ratio < 0.20:
            sparsity_desc = "moderately sparse"
        else:
            sparsity_desc = "frequently active"

        summary = (
            f"Feature {feature_idx}: {sparsity_desc}. "
            f"Active in {active_count}/{total_examples} examples ({active_ratio:.1%}). "
            f"Mean activation (when active): {mean_when_active:.3f}, max: {max_activation:.3f}."
        )

        return summary

    def clear(self) -> None:
        """Clear collected data."""
        self._example_texts = []
        self._activations = None


# =============================================================================
# Semantic Alignment Scorer
# =============================================================================

class SemanticAlignmentScorer:
    """
    Score feature alignment with semantic dimensions.

    Uses correlation between feature activations and semantic axis projections
    to find which interpretable dimensions each feature relates to.
    """

    def __init__(
        self,
        semantic_axes: dict[str, np.ndarray] | None = None,
        n_top_alignments: int = 5,
    ):
        """
        Args:
            semantic_axes: Dict mapping axis names to their direction vectors
            n_top_alignments: Number of top alignments to report
        """
        self.semantic_axes = semantic_axes or {}
        self.n_top_alignments = n_top_alignments

        # Storage for semantic projections
        self._semantic_projections: dict[str, np.ndarray] | None = None  # axis -> [n_examples]
        self._feature_activations: np.ndarray | None = None  # [n_examples, n_features]

    def set_semantic_axes(self, axes: dict[str, np.ndarray]) -> None:
        """Set semantic axes for alignment scoring."""
        self.semantic_axes = axes

    def collect_projections(
        self,
        activations: torch.Tensor,  # [n_examples, embedding_dim]
        feature_activations: torch.Tensor,  # [n_examples, n_features]
    ) -> None:
        """
        Collect semantic projections and feature activations.

        Args:
            activations: Original embeddings before SAE
            feature_activations: SAE latent activations
        """
        activations_np = activations.detach().cpu().numpy()
        features_np = feature_activations.detach().cpu().numpy()

        # Project onto semantic axes
        projections = {}
        for axis_name, axis_direction in self.semantic_axes.items():
            # Normalize direction
            axis_norm = axis_direction / (np.linalg.norm(axis_direction) + 1e-8)
            # Project
            proj = activations_np @ axis_norm  # [n_examples]
            projections[axis_name] = proj

        if self._semantic_projections is None:
            self._semantic_projections = projections
            self._feature_activations = features_np
        else:
            # Append
            for axis_name in projections:
                self._semantic_projections[axis_name] = np.concatenate([
                    self._semantic_projections[axis_name],
                    projections[axis_name]
                ])
            self._feature_activations = np.concatenate([
                self._feature_activations,
                features_np
            ], axis=0)

    def score_feature_alignment(
        self,
        feature_idx: int,
    ) -> dict[str, float]:
        """
        Score a feature's alignment with all semantic axes.

        Returns dict mapping axis names to correlation coefficients.
        """
        if self._feature_activations is None or self._semantic_projections is None:
            return {}

        feature_acts = self._feature_activations[:, feature_idx]

        alignments = {}
        for axis_name, projections in self._semantic_projections.items():
            # Compute Pearson correlation
            corr = self._pearson_correlation(feature_acts, projections)
            alignments[axis_name] = float(corr)

        return alignments

    def get_top_alignments(
        self,
        feature_idx: int,
    ) -> list[tuple[str, float]]:
        """
        Get top-k semantic alignments for a feature.

        Returns list of (axis_name, correlation) tuples sorted by abs(correlation).
        """
        alignments = self.score_feature_alignment(feature_idx)

        # Sort by absolute correlation
        sorted_alignments = sorted(
            alignments.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_alignments[:self.n_top_alignments]

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) < 2:
            return 0.0

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def clear(self) -> None:
        """Clear collected data."""
        self._semantic_projections = None
        self._feature_activations = None


# =============================================================================
# Label Cache
# =============================================================================

class LabelCache:
    """
    Persistent cache for feature labels.

    Stores labels to disk with versioning and integrity checks.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_file: str = "feature_labels.json",
    ):
        self.cache_dir = cache_dir or Path(".cache/sae_labels")
        self.cache_file = cache_file
        self.cache_path = self.cache_dir / cache_file

        # In-memory cache
        self._labels: dict[int, FeatureLabel] = {}
        self._metadata: dict[str, Any] = {
            "version": "1.0",
            "created_at": None,
            "updated_at": None,
            "model_hash": None,
        }

        # Load existing cache if available
        self._load_cache()

    def _load_cache(self) -> None:
        """Load cache from disk if exists."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)

                self._metadata = data.get("metadata", self._metadata)

                labels_data = data.get("labels", {})
                for idx_str, label_dict in labels_data.items():
                    idx = int(idx_str)
                    self._labels[idx] = FeatureLabel.from_dict(label_dict)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Failed to load label cache: {e}")

    def save(self) -> None:
        """Save cache to disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._metadata["updated_at"] = datetime.now().isoformat()
        if self._metadata["created_at"] is None:
            self._metadata["created_at"] = self._metadata["updated_at"]

        data = {
            "metadata": self._metadata,
            "labels": {
                str(idx): label.to_dict()
                for idx, label in self._labels.items()
            }
        }

        with open(self.cache_path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, feature_idx: int) -> FeatureLabel | None:
        """Get label for a feature if cached."""
        return self._labels.get(feature_idx)

    def set(self, label: FeatureLabel) -> None:
        """Cache a label."""
        self._labels[label.feature_idx] = label

    def has(self, feature_idx: int) -> bool:
        """Check if feature has a cached label."""
        return feature_idx in self._labels

    def get_all(self) -> dict[int, FeatureLabel]:
        """Get all cached labels."""
        return self._labels.copy()

    def clear(self) -> None:
        """Clear all cached labels."""
        self._labels = {}

    def set_model_hash(self, model_hash: str) -> None:
        """Set the hash of the model these labels are for."""
        self._metadata["model_hash"] = model_hash

    def is_valid_for_model(self, model_hash: str) -> bool:
        """Check if cache is valid for a given model hash."""
        return self._metadata.get("model_hash") == model_hash

    @property
    def size(self) -> int:
        """Number of cached labels."""
        return len(self._labels)


# =============================================================================
# Feature Labeler (Main Orchestrator)
# =============================================================================

class FeatureLabeler:
    """
    LLM-powered feature labeling with semantic alignment.

    Orchestrates the full labeling pipeline:
    1. Collect activations and find top-k examples per feature
    2. Generate LLM descriptions for each feature
    3. Score semantic alignment with 244 axes
    4. Cache results for persistence

    Usage:
        labeler = FeatureLabeler(llm_provider="ollama")

        # Collect training examples
        for batch in dataloader:
            labeler.collect_examples(sae, batch.activations, batch.texts)

        # Label features
        labels = await labeler.label_features(sae, feature_indices=[0, 1, 2])
    """

    # Default prompt template for feature labeling
    DEFAULT_PROMPT_TEMPLATE = """You are an expert in neural network interpretability.
Your task is to describe what concept a sparse autoencoder feature represents.

This feature activates most strongly for the following examples:
{examples}

The feature has the following activation pattern:
{activation_pattern}

Based on these examples and pattern, provide a concise description (1-2 sentences)
of what concept or pattern this feature detects. Focus on the common theme.

Description:"""

    def __init__(
        self,
        llm_provider: str = "ollama",
        llm_model: str = "llama3.2:3b",
        llm_call_fn: Callable[[str], str] | None = None,
        unified_client: Optional["UnifiedLLMClient"] = None,
        semantic_axes: dict[str, np.ndarray] | None = None,
        cache_dir: Path | None = None,
        top_k_examples: int = 10,
        prompt_template: str | None = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ):
        """
        Args:
            llm_provider: LLM provider ("ollama", "anthropic", "openai", "google")
            llm_model: Model name for the provider
            llm_call_fn: Optional custom function for LLM calls
            unified_client: Optional UnifiedLLMClient instance (recommended)
            semantic_axes: Dict of semantic axis name -> direction vector
            cache_dir: Directory for label cache
            top_k_examples: Number of top examples per feature
            prompt_template: Custom prompt template
            max_tokens: Maximum tokens for LLM generation
            temperature: LLM sampling temperature
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_call_fn = llm_call_fn
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
        self.max_tokens = max_tokens
        self.temperature = temperature

        # UnifiedLLMClient for multi-provider support with fallback
        self._unified_client = unified_client
        if self._unified_client is None and HAS_UNIFIED_CLIENT:
            # Auto-create unified client with configured provider
            try:
                primary_config = LLMConfig(provider=llm_provider, model=llm_model)
                # Fallback to Ollama if primary fails
                fallbacks = []
                if llm_provider != "ollama":
                    fallbacks.append(LLMConfig(provider="ollama", model="llama3.2:3b"))
                self._unified_client = UnifiedLLMClient(
                    primary=primary_config,
                    fallbacks=fallbacks,
                    enable_cost_tracking=True
                )
                logger.info(f"Created UnifiedLLMClient with {llm_provider}/{llm_model}")
            except Exception as e:
                logger.warning(f"Failed to create UnifiedLLMClient: {e}. Using legacy method.")
                self._unified_client = None

        # Components
        self.analyzer = ActivationPatternAnalyzer(top_k=top_k_examples)
        self.alignment_scorer = SemanticAlignmentScorer(semantic_axes=semantic_axes)
        self.cache = LabelCache(cache_dir=cache_dir)

    def collect_examples(
        self,
        sae: torch.nn.Module,
        activations: torch.Tensor,
        example_texts: list[str],
    ) -> None:
        """
        Collect activation examples for labeling.

        Call this multiple times to accumulate examples from training data.

        Args:
            sae: The SAE model
            activations: Input activations [batch, input_dim]
            example_texts: Text corresponding to each activation
        """
        sae.eval()
        with torch.no_grad():
            _, latents, _ = sae(activations)

        # Collect for both analyzer and alignment scorer
        self.analyzer.collect_activations(sae, activations, example_texts)
        self.alignment_scorer.collect_projections(activations, latents)

    async def label_features(
        self,
        sae: torch.nn.Module,
        feature_indices: list[int] | None = None,
        force_relabel: bool = False,
        batch_size: int = 10,
    ) -> dict[int, FeatureLabel]:
        """
        Label features using LLM and semantic alignment.

        Args:
            sae: The SAE model (for hash validation)
            feature_indices: Which features to label (None = all)
            force_relabel: Relabel even if cached
            batch_size: Number of features to label at once

        Returns:
            Dict mapping feature index to FeatureLabel
        """
        # Determine which features to label
        n_features = sae.latent_dim if hasattr(sae, 'latent_dim') else sae.decoder.weight.shape[0]

        if feature_indices is None:
            feature_indices = list(range(n_features))

        # Filter out already cached (unless force_relabel)
        if not force_relabel:
            feature_indices = [
                idx for idx in feature_indices
                if not self.cache.has(idx)
            ]

        # Label in batches
        results: dict[int, FeatureLabel] = {}

        for i in range(0, len(feature_indices), batch_size):
            batch_indices = feature_indices[i:i + batch_size]

            # Label each feature in batch
            for idx in batch_indices:
                label = await self._label_single_feature(idx)
                results[idx] = label
                self.cache.set(label)

            # Save cache periodically
            if i % (batch_size * 5) == 0:
                self.cache.save()

        # Final save
        self.cache.save()

        # Merge with cached labels
        all_labels = self.cache.get_all()
        all_labels.update(results)

        return {
            idx: all_labels[idx]
            for idx in (feature_indices if force_relabel else all_labels.keys())
        }

    async def _label_single_feature(self, feature_idx: int) -> FeatureLabel:
        """Label a single feature."""
        # Get top activating examples
        examples, act_values = self.analyzer.get_top_activating_examples(feature_idx)
        activation_pattern = self.analyzer.get_activation_pattern_summary(feature_idx)

        # Get semantic alignments
        alignments = self.alignment_scorer.score_feature_alignment(feature_idx)
        top_alignments = self.alignment_scorer.get_top_alignments(feature_idx)

        # Primary semantic axis (highest absolute correlation)
        primary_axis = top_alignments[0][0] if top_alignments else None
        alignment_score = abs(top_alignments[0][1]) if top_alignments else 0.0

        # Generate LLM description
        description = await self._generate_description(
            examples, act_values, activation_pattern
        )

        # Estimate confidence based on evidence quality
        confidence = self._estimate_confidence(
            n_examples=len(examples),
            alignment_score=alignment_score,
            description_length=len(description),
        )

        return FeatureLabel(
            feature_idx=feature_idx,
            description=description,
            confidence=confidence,
            top_activating_examples=examples[:5],  # Store top 5
            activation_pattern=activation_pattern,
            semantic_alignments=alignments,
            primary_semantic_axis=primary_axis,
            semantic_alignment_score=alignment_score,
            labeled_at=datetime.now().isoformat(),
            labeler_model=f"{self.llm_provider}/{self.llm_model}",
        )

    async def _generate_description(
        self,
        examples: list[str],
        act_values: list[float],
        activation_pattern: str,
    ) -> str:
        """Generate LLM description for a feature."""
        if not examples:
            return "Feature with no strongly activating examples found."

        # Format examples with activation values
        examples_text = "\n".join([
            f"- (activation: {val:.3f}) {text[:200]}..."
            if len(text) > 200 else f"- (activation: {val:.3f}) {text}"
            for text, val in zip(examples[:5], act_values[:5])
        ])

        # Build prompt
        prompt = self.prompt_template.format(
            examples=examples_text,
            activation_pattern=activation_pattern,
        )

        # Call LLM
        if self.llm_call_fn:
            description = self.llm_call_fn(prompt)
        else:
            description = await self._default_llm_call(prompt)

        return description.strip()

    async def _default_llm_call(self, prompt: str) -> str:
        """
        Default LLM call implementation using UnifiedLLMClient.

        Uses multi-provider client with automatic fallback if available,
        otherwise falls back to direct ollama calls.
        """
        # Try UnifiedLLMClient first (supports multiple providers with fallback)
        if self._unified_client is not None:
            try:
                response = await self._unified_client.complete(
                    prompt=prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                logger.debug(
                    f"LLM call: {response.model}, "
                    f"{response.output_tokens} tokens, "
                    f"${response.cost_estimate.total_cost_usd:.4f}" if response.cost_estimate else ""
                )
                return response.content
            except Exception as e:
                logger.warning(f"UnifiedLLMClient failed: {e}. Trying legacy method.")

        # Fallback to direct ollama call (legacy behavior)
        try:
            import ollama
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response["message"]["content"]
        except ImportError:
            return "[LLM not available - install ollama package or use UnifiedLLMClient]"
        except Exception as e:
            return f"[LLM error: {e}]"

    def _estimate_confidence(
        self,
        n_examples: int,
        alignment_score: float,
        description_length: int,
    ) -> float:
        """Estimate confidence in a label based on evidence quality."""
        # Base confidence from number of examples
        example_conf = min(n_examples / 10.0, 1.0)  # Max at 10 examples

        # Boost from semantic alignment
        alignment_conf = alignment_score  # Already 0-1

        # Penalty for very short or generic descriptions
        desc_conf = 1.0 if description_length > 20 else 0.5

        # Weighted combination
        confidence = (
            0.5 * example_conf +
            0.3 * alignment_conf +
            0.2 * desc_conf
        )

        return float(np.clip(confidence, 0.0, 1.0))

    def clear_collected_data(self) -> None:
        """Clear all collected activation data."""
        self.analyzer.clear()
        self.alignment_scorer.clear()

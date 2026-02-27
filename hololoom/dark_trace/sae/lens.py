"""
Dark Trace SAE: SAELens - TraceLens Implementation

Implements the TraceLens protocol for SAE-based interpretability,
enabling unified orchestration with other interpretability lenses.

Methods:
- project(): Map activations to SAE latent space
- get_active(): Get top-k active features with labels
- steer(): Generate steering vectors via decoder weights
- explain(): Human-readable feature explanations

Research Foundation:
- SAE: Anthropic's "Towards Monosemanticity" (2023)
- Steering: Decoder-based activation steering
- Unified: TraceLens protocol from Dark Trace
"""

from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn.functional as F

from hololoom.dark_trace.protocol import (
    TraceLens,
    BaseLens,
    LensType,
    Feature,
    FeatureSource,
    FeatureActivation,
    SteeringVector,
    SafetyFlag,
)

from hololoom.dark_trace.sae.labeler import FeatureLabeler, FeatureLabel


class SAELens(BaseLens):
    """
    SAE-based interpretability lens implementing TraceLens protocol.

    Wraps a trained SAE (SparseAutoEncoder or ResearchSAE) and provides:
    - Feature projection (encoder)
    - Steering (decoder-based)
    - Explanations (with optional auto-labeling)

    Usage:
        sae = ResearchSAE(input_dim=384, expansion_factor=16)
        lens = SAELens(sae)

        # Project activations
        features = lens.project(activations)

        # Get active features
        active = lens.get_active(activations, top_k=10)

        # Generate steering vector
        steering = lens.steer({"sae.42": 0.8, "sae.100": -0.5})

        # Explain activations
        explanation = lens.explain(activations, verbosity=2)
    """

    def __init__(
        self,
        sae: torch.nn.Module,
        labeler: Optional[FeatureLabeler] = None,
        labels: Optional[Dict[int, FeatureLabel]] = None,
        name: str = "SAELens",
        register_features: bool = True,
    ):
        """
        Initialize SAELens from a trained SAE.

        Args:
            sae: Trained SAE model (SparseAutoEncoder or ResearchSAE)
            labeler: Optional FeatureLabeler for auto-labeling
            labels: Optional pre-computed labels {feature_idx: FeatureLabel}
            name: Name for this lens
            register_features: Whether to pre-register all features
        """
        # Extract dimensions from SAE
        self.sae = sae
        input_dim = self._get_input_dim(sae)
        n_features = self._get_latent_dim(sae)

        super().__init__(
            input_dim=input_dim,
            n_features=n_features,
            lens_type=LensType.SAE,
            name=name,
        )

        self.labeler = labeler
        self._labels = labels or {}

        # Register features
        if register_features:
            self._register_all_features()

    def _get_input_dim(self, sae: torch.nn.Module) -> int:
        """Extract input dimension from SAE."""
        if hasattr(sae, 'input_dim'):
            return sae.input_dim
        elif hasattr(sae, 'encoder'):
            return sae.encoder.in_features
        else:
            raise ValueError("Cannot determine SAE input dimension")

    def _get_latent_dim(self, sae: torch.nn.Module) -> int:
        """Extract latent dimension from SAE."""
        if hasattr(sae, 'latent_dim'):
            return sae.latent_dim
        elif hasattr(sae, 'encoder'):
            return sae.encoder.out_features
        else:
            raise ValueError("Cannot determine SAE latent dimension")

    def _register_all_features(self) -> None:
        """Register all SAE features in the feature registry."""
        for idx in range(self._n_features):
            label_info = self._labels.get(idx)

            feature = Feature(
                id=f"sae.{idx}",
                source=FeatureSource.SAE,
                index=idx,
                label=label_info.description if label_info else f"SAE Feature {idx}",
                description=label_info.activation_pattern if label_info else None,
                exemplars=label_info.top_activating_examples if label_info else [],
                metadata={
                    "semantic_alignments": label_info.semantic_alignments if label_info else {},
                    "primary_semantic_axis": label_info.primary_semantic_axis if label_info else None,
                }
            )

            self.register_feature(feature)

    def set_labels(self, labels: Dict[int, FeatureLabel]) -> None:
        """
        Set feature labels and update registry.

        Args:
            labels: Dict mapping feature index to FeatureLabel
        """
        self._labels = labels
        self._register_all_features()

    # =========================================================================
    # TraceLens Protocol Implementation
    # =========================================================================

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Project activations into SAE feature space.

        Args:
            activations: Input activations [batch, seq, dim] or [batch, dim] or [dim]

        Returns:
            SAE latent activations with same batch structure
        """
        self.sae.eval()

        # Handle different input shapes
        original_shape = activations.shape
        original_dim = activations.dim()

        # Flatten to [N, input_dim]
        if original_dim == 1:
            flat = activations.unsqueeze(0)  # [1, dim]
        elif original_dim == 2:
            flat = activations  # [batch, dim]
        elif original_dim == 3:
            batch, seq, dim = activations.shape
            flat = activations.reshape(-1, dim)  # [batch*seq, dim]
        else:
            raise ValueError(f"Unsupported activation shape: {original_shape}")

        # Project through SAE encoder
        with torch.no_grad():
            _, latents, _ = self.sae(flat)

        # Restore shape
        if original_dim == 1:
            return latents.squeeze(0)  # [n_features]
        elif original_dim == 2:
            return latents  # [batch, n_features]
        else:
            return latents.reshape(batch, seq, -1)  # [batch, seq, n_features]

    def get_active(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
        top_k: Optional[int] = None,
    ) -> List[FeatureActivation]:
        """
        Get active features above threshold.

        Args:
            activations: Input activations
            threshold: Minimum activation threshold
            top_k: Return only top-k features (optional)

        Returns:
            List of FeatureActivation sorted by activation (descending)
        """
        # Project to feature space
        feature_acts = self.project(activations)

        # Handle batched input by taking mean across batch
        if feature_acts.dim() > 1:
            feature_acts = feature_acts.mean(dim=0)  # [n_features]

        # Filter by threshold
        active_mask = feature_acts > threshold
        active_indices = torch.where(active_mask)[0]
        active_values = feature_acts[active_indices]

        # Sort by activation value (descending)
        sorted_order = torch.argsort(active_values, descending=True)
        active_indices = active_indices[sorted_order]
        active_values = active_values[sorted_order]

        # Limit to top_k if specified
        if top_k is not None:
            active_indices = active_indices[:top_k]
            active_values = active_values[:top_k]

        # Build FeatureActivation list
        result = []
        for idx, val in zip(active_indices.tolist(), active_values.tolist()):
            feature = self.get_feature(f"sae.{idx}")
            if feature is None:
                # Create minimal feature if not registered
                feature = Feature.from_sae(idx, val)

            result.append(FeatureActivation(
                feature=feature,
                activation=val,
                confidence=1.0,  # SAE features are deterministic
            ))

        return result

    def steer(
        self,
        goals: Dict[str, float],
        input_dim: Optional[int] = None,
    ) -> SteeringVector:
        """
        Generate steering vector from feature goals.

        Uses SAE decoder weights: steering = sum(goal[i] * decoder_col[i])

        Args:
            goals: Feature goals {"sae.42": 0.8, "sae.100": -0.5}
            input_dim: Unused (determined by SAE)

        Returns:
            SteeringVector for adding to activations
        """
        # Get decoder weights [n_features, input_dim]
        decoder = self.sae.decoder.weight.detach()  # [input_dim, n_features] - transposed for Linear

        # Build steering vector
        steering = torch.zeros(self._input_dim, device=decoder.device, dtype=decoder.dtype)

        for feature_id, target_strength in goals.items():
            # Parse feature ID
            if not feature_id.startswith("sae."):
                continue

            try:
                idx = int(feature_id.split(".")[1])
            except (ValueError, IndexError):
                continue

            if idx < 0 or idx >= self._n_features:
                continue

            # Add decoder column scaled by target strength
            # For nn.Linear: weight is [out_features, in_features] = [input_dim, n_features]
            steering += target_strength * decoder[:, idx]

        return SteeringVector(
            vector=steering,
            features=goals,
            source=LensType.SAE,
            target_description=f"Steer {len(goals)} SAE features",
            safety_validated=False,
        )

    def explain(
        self,
        activations: torch.Tensor,
        verbosity: int = 1,
        include_features: Optional[List[str]] = None,
    ) -> str:
        """
        Generate human-readable explanation of activations.

        Args:
            activations: Input activations to explain
            verbosity: Detail level (1=brief, 2=moderate, 3=detailed)
            include_features: Only explain these features (optional)

        Returns:
            Human-readable explanation string
        """
        active = self.get_active(activations, threshold=0.1, top_k=10)

        if include_features:
            active = [a for a in active if a.feature.id in include_features]

        if not active:
            return f"{self.name}: No significant SAE features detected."

        lines = [f"{self.name} Analysis ({self._n_features} features):"]

        # Brief: top features with labels
        if verbosity >= 1:
            lines.append("\nTop Active Features:")
            for act in active[:5]:
                label = act.feature.label or act.feature.id
                lines.append(f"  • {label}: {act.activation:.3f}")

        # Moderate: add semantic alignments
        if verbosity >= 2:
            lines.append("\nSemantic Alignments:")
            for act in active[:5]:
                alignments = act.feature.metadata.get("semantic_alignments", {})
                if alignments:
                    top_align = max(alignments.items(), key=lambda x: abs(x[1]))
                    lines.append(f"  {act.feature.id}: correlates with {top_align[0]} ({top_align[1]:.3f})")
                else:
                    lines.append(f"  {act.feature.id}: no semantic alignment data")

        # Detailed: add exemplars and full info
        if verbosity >= 3:
            lines.append("\nFeature Details:")
            for act in active[:5]:
                lines.append(f"\n  {act.feature.id}:")
                lines.append(f"    Label: {act.feature.label}")
                if act.feature.description:
                    lines.append(f"    Pattern: {act.feature.description}")
                if act.feature.exemplars:
                    lines.append(f"    Exemplars: {act.feature.exemplars[:2]}")
                lines.append(f"    Sparsity: {act.feature.sparsity:.3f}")
                lines.append(f"    Safety: {act.feature.safety_flag.name}")

        return "\n".join(lines)

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def ablate(
        self,
        activations: torch.Tensor,
        feature_ids: List[str],
    ) -> torch.Tensor:
        """
        Ablate (remove) specific features from activations.

        Uses SAE to project, zero out features, and reconstruct.

        Args:
            activations: Original activations
            feature_ids: Features to ablate (e.g., ["sae.42", "sae.100"])

        Returns:
            Reconstructed activations without ablated features
        """
        self.sae.eval()

        # Get feature indices
        indices = []
        for fid in feature_ids:
            if fid.startswith("sae."):
                try:
                    idx = int(fid.split(".")[1])
                    if 0 <= idx < self._n_features:
                        indices.append(idx)
                except (ValueError, IndexError):
                    pass

        if not indices:
            return activations

        with torch.no_grad():
            # Project to latent space
            _, latents, _ = self.sae(activations)

            # Zero out ablated features
            for idx in indices:
                latents[:, idx] = 0.0

            # Reconstruct
            reconstructed = self.sae.decoder(latents)
            reconstructed = reconstructed + self.sae.decoder.bias

        return reconstructed

    def inject(
        self,
        activations: torch.Tensor,
        feature_values: Dict[str, float],
    ) -> torch.Tensor:
        """
        Inject specific feature values into activations.

        Args:
            activations: Original activations
            feature_values: {feature_id: value} to inject

        Returns:
            Modified activations with injected features
        """
        # Generate steering vector and add to activations
        steering = self.steer(feature_values)
        return activations + steering.vector

    def patch(
        self,
        source_activations: torch.Tensor,
        target_activations: torch.Tensor,
        feature_ids: List[str],
    ) -> torch.Tensor:
        """
        Patch features from source to target activations.

        Args:
            source_activations: Source of feature values
            target_activations: Target to patch into
            feature_ids: Features to transfer

        Returns:
            Target activations with patched features
        """
        self.sae.eval()

        # Get feature indices
        indices = []
        for fid in feature_ids:
            if fid.startswith("sae."):
                try:
                    idx = int(fid.split(".")[1])
                    if 0 <= idx < self._n_features:
                        indices.append(idx)
                except (ValueError, IndexError):
                    pass

        if not indices:
            return target_activations

        with torch.no_grad():
            # Get source latents
            _, source_latents, _ = self.sae(source_activations)

            # Get target latents
            _, target_latents, _ = self.sae(target_activations)

            # Patch specific features
            for idx in indices:
                target_latents[:, idx] = source_latents[:, idx]

            # Reconstruct
            reconstructed = self.sae.decoder(target_latents)
            reconstructed = reconstructed + self.sae.decoder.bias

        return reconstructed

    def get_decoder_directions(
        self,
        feature_ids: List[str],
    ) -> torch.Tensor:
        """
        Get decoder directions for specified features.

        Useful for visualization and analysis.

        Args:
            feature_ids: Features to get directions for

        Returns:
            Tensor [n_features, input_dim] of decoder directions
        """
        decoder = self.sae.decoder.weight.detach()  # [input_dim, n_features]

        directions = []
        for fid in feature_ids:
            if fid.startswith("sae."):
                try:
                    idx = int(fid.split(".")[1])
                    if 0 <= idx < self._n_features:
                        directions.append(decoder[:, idx])
                except (ValueError, IndexError):
                    pass

        if not directions:
            return torch.zeros(0, self._input_dim)

        return torch.stack(directions)  # [n, input_dim]

"""
Feature engineering for contextual bandits.

Combines context (from HoloLoom embeddings) and action features into
input vectors for the reward model.
"""

import numpy as np
import numpy.typing as npt
from typing import Callable
from HoloLoom.bandits.neural_ts.types import Context, Action


class ContextActionFeaturizer:
    """
    Featurizer that concatenates context and action features.

    Handles three modes:
    1. Context-only: No action features (all actions treated identically)
    2. Context + action: Concatenate context and action features
    3. Custom: User-provided featurization function

    Attributes:
        context_dim: Expected context dimension
        action_dim: Expected action dimension (0 for context-only)
        transform: Optional custom transformation function

    Example:
        >>> # Context-only mode
        >>> feat = ContextActionFeaturizer(context_dim=512, action_dim=0)
        >>> ctx = Context(id="ctx1", x=np.random.randn(512))
        >>> actions = [Action(id="tool_a"), Action(id="tool_b")]
        >>> features = feat(ctx, actions)
        >>> print(features.shape)  # [2, 512] - same context repeated
        >>>
        >>> # Context + action mode
        >>> feat = ContextActionFeaturizer(context_dim=512, action_dim=8)
        >>> actions = [
        ...     Action(id="tool_a", a=np.random.randn(8)),
        ...     Action(id="tool_b", a=np.random.randn(8))
        ... ]
        >>> features = feat(ctx, actions)
        >>> print(features.shape)  # [2, 520] - context + action features
    """

    def __init__(
        self,
        context_dim: int,
        action_dim: int = 0,
        transform: Callable[[npt.NDArray], npt.NDArray] | None = None,
    ):
        """
        Initialize featurizer.

        Args:
            context_dim: Expected context feature dimension
            action_dim: Expected action feature dimension (0 for context-only)
            transform: Optional custom transformation (applied after concatenation)

        Raises:
            ValueError: If dimensions are negative
        """
        if context_dim <= 0:
            raise ValueError(f"context_dim must be positive, got {context_dim}")
        if action_dim < 0:
            raise ValueError(f"action_dim must be non-negative, got {action_dim}")

        self.context_dim = context_dim
        self.action_dim = action_dim
        self.transform = transform

    def __call__(
        self,
        ctx: Context,
        actions: list[Action],
    ) -> npt.NDArray[np.floating]:
        """
        Featurize context-action pairs.

        Args:
            ctx: Context (contains x of shape [context_dim])
            actions: List of actions (each with optional a of shape [action_dim])

        Returns:
            Feature matrix, shape [len(actions), context_dim + action_dim]

        Raises:
            ValueError: If context or action dimensions don't match config

        Example:
            >>> ctx = Context(id="ctx1", x=np.ones(512))
            >>> actions = [Action(id=f"tool_{i}", a=np.ones(8)) for i in range(5)]
            >>> features = featurizer(ctx, actions)
            >>> print(features.shape)  # [5, 520]
        """
        # Validate context
        if ctx.x.shape[0] != self.context_dim:
            raise ValueError(
                f"Context dimension mismatch: expected {self.context_dim}, "
                f"got {ctx.x.shape[0]}"
            )

        # Build feature matrix
        features_list = []

        for action in actions:
            if self.action_dim == 0:
                # Context-only mode
                if action.a is not None:
                    import warnings
                    warnings.warn(
                        f"Action {action.id} has features but action_dim=0. "
                        f"Ignoring action features."
                    )
                feat = ctx.x
            else:
                # Context + action mode
                if action.a is None:
                    raise ValueError(
                        f"Action {action.id} missing features but action_dim={self.action_dim}"
                    )
                if action.a.shape[0] != self.action_dim:
                    raise ValueError(
                        f"Action {action.id} dimension mismatch: "
                        f"expected {self.action_dim}, got {action.a.shape[0]}"
                    )

                feat = np.concatenate([ctx.x, action.a])

            features_list.append(feat)

        features = np.stack(features_list)  # [len(actions), feature_dim]

        # Apply optional transform
        if self.transform is not None:
            features = self.transform(features)

        return features

    @property
    def feature_dim(self) -> int:
        """Total feature dimension (context_dim + action_dim)."""
        return self.context_dim + self.action_dim

    def __repr__(self) -> str:
        has_transform = self.transform is not None
        return (
            f"ContextActionFeaturizer(context_dim={self.context_dim}, "
            f"action_dim={self.action_dim}, transform={has_transform})"
        )


def create_loom_featurizer(
    embedding_dim: int = 384,
    action_dim: int = 0,
    normalize: bool = False,
) -> ContextActionFeaturizer:
    """
    Factory for HoloLoom-specific featurizer.

    Creates featurizer that works with HoloLoom's Matryoshka embeddings.
    Optionally normalizes features to unit norm.

    Args:
        embedding_dim: HoloLoom embedding dimension (96, 192, or 384)
        action_dim: Action feature dimension (0 for context-only)
        normalize: Whether to L2-normalize features

    Returns:
        Configured ContextActionFeaturizer

    Example:
        >>> # For use with HoloLoom's 384-dim embeddings, no action features
        >>> feat = create_loom_featurizer(embedding_dim=384, action_dim=0)
        >>>
        >>> # With action features and normalization
        >>> feat = create_loom_featurizer(
        ...     embedding_dim=384,
        ...     action_dim=16,
        ...     normalize=True
        ... )
    """
    transform = None

    if normalize:
        def l2_normalize(x: npt.NDArray) -> npt.NDArray:
            """L2 normalize each row."""
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
            return x / norms

        transform = l2_normalize

    return ContextActionFeaturizer(
        context_dim=embedding_dim,
        action_dim=action_dim,
        transform=transform,
    )

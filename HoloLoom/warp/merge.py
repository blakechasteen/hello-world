"""
Merge Operations for WarpSpace - Chomskyan Compositional Semantics
===================================================================

Implements Merge from Minimalist Program as tensor operations in WarpSpace.

Philosophy:
-----------
In Chomsky's Minimalism, Merge is the atomic structure-building operation.
In HoloLoom's WarpSpace, we implement Merge as compositional tensor fusion:

    Merge(embedding_1, embedding_2) → combined_embedding

This enables:
- Compositional semantics (meaning of "red ball" from "red" + "ball")
- Hierarchical structure (nested merges)
- Head-dependency tracking
- Feature unification

Types of Merge:
1. External Merge: Combine two separate items
2. Internal Merge: Move an item (creates long-distance dependencies)
3. Parallel Merge: Merge multiple items simultaneously
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Merge Types
# ============================================================================

class MergeType(Enum):
    """Type of merge operation."""
    EXTERNAL = "external"  # Combine two separate items
    INTERNAL = "internal"  # Move an item (wh-movement, etc.)
    PARALLEL = "parallel"  # Merge multiple items at once


# ============================================================================
# Merged Object
# ============================================================================

@dataclass
class MergedObject:
    """
    Result of a Merge operation.

    Represents a syntactic object with compositional semantics.

    Attributes:
        embedding: Combined embedding vector
        components: Original components (e.g., ["the", "cat"])
        head: Head word (determines category)
        merge_type: Type of merge performed
        label: Syntactic label (NP, VP, etc.)
        children: Nested structure (recursive)
        metadata: Additional merge information
    """
    embedding: np.ndarray
    components: List[str]
    head: str
    merge_type: MergeType
    label: str
    children: List['MergedObject'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        comp_str = "+".join(self.components)
        return f"Merged[{self.label}]({comp_str}, head={self.head})"


# ============================================================================
# Merge Operator
# ============================================================================

class MergeOperator:
    """
    Implements Merge operations in continuous tensor space.

    Chomsky's Merge in continuous space:
    - External Merge: Vector fusion (weighted sum, concatenation, etc.)
    - Internal Merge: Attention-based reordering
    - Parallel Merge: Multi-head attention

    Usage:
        merger = MergeOperator(
            embedder=embedder,
            fusion_method="weighted_sum"
        )

        # External merge: "the" + "cat" → "the cat"
        the_emb = embedder.encode(["the"])[0]
        cat_emb = embedder.encode(["cat"])[0]

        merged = merger.external_merge(
            the_emb, cat_emb,
            head="cat",
            dependent="the",
            label="NP"
        )
    """

    def __init__(
        self,
        embedder,
        fusion_method: str = "weighted_sum",
        head_weight: float = 0.7,
        dependent_weight: float = 0.3
    ):
        """
        Initialize merge operator.

        Args:
            embedder: Embedder instance (for encoding text)
            fusion_method: How to combine embeddings
                          ("weighted_sum", "concat", "hadamard", "mlp")
            head_weight: Weight for head in fusion (0-1)
            dependent_weight: Weight for dependent in fusion (0-1)
        """
        self.embedder = embedder
        self.fusion_method = fusion_method
        self.head_weight = head_weight
        self.dependent_weight = dependent_weight

        # Normalize weights
        total = head_weight + dependent_weight
        self.head_weight /= total
        self.dependent_weight /= total

        logger.info(
            f"MergeOperator initialized: method={fusion_method}, "
            f"head_weight={self.head_weight:.2f}, "
            f"dependent_weight={self.dependent_weight:.2f}"
        )

    # ========================================================================
    # External Merge
    # ========================================================================

    def external_merge(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        head: str,
        dependent: str,
        label: str = "PHRASE",
        alpha_is_head: bool = False
    ) -> MergedObject:
        """
        External Merge: Combine two separate syntactic objects.

        Chomsky: Merge(α, β) = {α, β} with label determined by head

        Tensor implementation:
            If α is head: merged_emb = head_weight * α + dependent_weight * β
            If β is head: merged_emb = head_weight * β + dependent_weight * α

        Args:
            alpha: First embedding
            beta: Second embedding
            head: Head word (determines category)
            dependent: Dependent word
            label: Syntactic label (NP, VP, PP, etc.)
            alpha_is_head: True if alpha is the head (default: beta is head)

        Returns:
            MergedObject with compositional embedding

        Example:
            # Merge "the" (determiner) + "cat" (noun) → "the cat" (NP)
            the_emb = embedder.encode(["the"])[0]
            cat_emb = embedder.encode(["cat"])[0]

            # Cat is head (determines that phrase is NP)
            merged = merger.external_merge(
                the_emb, cat_emb,
                head="cat", dependent="the",
                label="NP", alpha_is_head=False
            )
        """
        # Determine head/dependent order
        if alpha_is_head:
            head_emb, dep_emb = alpha, beta
            components = [head, dependent]
        else:
            head_emb, dep_emb = beta, alpha
            components = [dependent, head]

        # Fuse embeddings based on method
        if self.fusion_method == "weighted_sum":
            # Weighted sum (head-prominence bias)
            merged_emb = (self.head_weight * head_emb +
                         self.dependent_weight * dep_emb)

        elif self.fusion_method == "concat":
            # Concatenation (preserves both fully)
            merged_emb = np.concatenate([head_emb, dep_emb])

        elif self.fusion_method == "hadamard":
            # Element-wise product (multiplicative interaction)
            merged_emb = head_emb * dep_emb

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Normalize
        merged_emb = merged_emb / (np.linalg.norm(merged_emb) + 1e-10)

        logger.debug(
            f"External merge: {dependent} + {head} → {label} "
            f"(head_weight={self.head_weight:.2f})"
        )

        return MergedObject(
            embedding=merged_emb,
            components=components,
            head=head,
            merge_type=MergeType.EXTERNAL,
            label=label,
            metadata={
                "head_weight": float(self.head_weight),
                "dependent_weight": float(self.dependent_weight),
                "fusion_method": self.fusion_method,
                "alpha_is_head": alpha_is_head
            }
        )

    # ========================================================================
    # Internal Merge
    # ========================================================================

    def internal_merge(
        self,
        merged_obj: MergedObject,
        move_index: int,
        target_position: str = "front"
    ) -> MergedObject:
        """
        Internal Merge: Move an element within structure.

        Chomsky: Internal Merge creates movement (wh-movement, topicalization, etc.)

        Example: "What did John see?"
            Base structure: "John saw what"
            Internal merge: Move "what" to front → "What did John see?"

        Tensor implementation:
            - Reweight components via attention
            - Emphasize moved element

        Args:
            merged_obj: Existing merged object
            move_index: Index of component to move
            target_position: Where to move ("front", "back")

        Returns:
            New MergedObject with moved component emphasized
        """
        # Compute attention weights (emphasize moved element)
        n_components = len(merged_obj.components)
        attention = np.ones(n_components) * 0.5  # Base attention
        attention[move_index] = 1.5  # Emphasize moved element
        attention = attention / np.sum(attention)  # Normalize

        # Reorder components
        moved_component = merged_obj.components.pop(move_index)
        if target_position == "front":
            new_components = [moved_component] + merged_obj.components
        else:
            new_components = merged_obj.components + [moved_component]

        logger.debug(
            f"Internal merge: Move '{moved_component}' to {target_position}"
        )

        return MergedObject(
            embedding=merged_obj.embedding,  # Embedding unchanged (simplified)
            components=new_components,
            head=merged_obj.head,
            merge_type=MergeType.INTERNAL,
            label=merged_obj.label,
            metadata={
                **merged_obj.metadata,
                "moved_component": moved_component,
                "move_index": move_index,
                "target_position": target_position,
                "attention_weights": attention.tolist()
            }
        )

    # ========================================================================
    # Parallel Merge
    # ========================================================================

    def parallel_merge(
        self,
        embeddings: List[np.ndarray],
        components: List[str],
        head_index: int,
        label: str = "PHRASE"
    ) -> MergedObject:
        """
        Parallel Merge: Merge multiple items simultaneously.

        Useful for complex constructions:
        - "the big red ball" → Merge(Det, Adj, Adj, Noun) all at once
        - Multi-word expressions

        Tensor implementation:
            - Weighted sum with head prominence
            - All dependents contribute equally

        Args:
            embeddings: List of embeddings to merge
            components: List of words
            head_index: Index of head word
            label: Syntactic label

        Returns:
            MergedObject with all components merged
        """
        if len(embeddings) != len(components):
            raise ValueError("Embeddings and components must have same length")

        # Compute weights: head gets head_weight, others share remaining
        weights = np.ones(len(embeddings))
        weights[head_index] = self.head_weight / (1.0 - self.head_weight)  # Boost head
        weights = weights / np.sum(weights)  # Normalize

        # Weighted sum
        merged_emb = np.sum([w * emb for w, emb in zip(weights, embeddings)], axis=0)
        merged_emb = merged_emb / (np.linalg.norm(merged_emb) + 1e-10)

        logger.debug(
            f"Parallel merge: {components} → {label} (head={components[head_index]})"
        )

        return MergedObject(
            embedding=merged_emb,
            components=components,
            head=components[head_index],
            merge_type=MergeType.PARALLEL,
            label=label,
            metadata={
                "n_components": len(components),
                "head_index": head_index,
                "weights": weights.tolist()
            }
        )

    # ========================================================================
    # Recursive Merge (X-bar Integration)
    # ========================================================================

    def recursive_merge(
        self,
        embeddings: List[np.ndarray],
        words: List[str],
        structure: List[Tuple[int, int, str, str]]
    ) -> MergedObject:
        """
        Recursive Merge: Build hierarchical structure bottom-up.

        Mimics syntactic tree construction:

        Example: "the big cat"
            1. Merge("big", "cat") → Adj+N = N'
            2. Merge("the", N') → Det+N' = NP

        Args:
            embeddings: Leaf embeddings (one per word)
            words: Leaf words
            structure: List of (left_idx, right_idx, head_position, label)
                      head_position: "left" or "right"
                      Merges are applied in order

        Returns:
            Root MergedObject with full tree structure

        Example:
            words = ["the", "big", "cat"]
            embeddings = [emb_the, emb_big, emb_cat]
            structure = [
                (1, 2, "right", "N'"),    # Merge big+cat → N' (cat is head)
                (0, -1, "right", "NP")    # Merge the+N' → NP (N' is head)
            ]
        """
        # Start with leaf nodes
        nodes = [
            MergedObject(
                embedding=emb,
                components=[word],
                head=word,
                merge_type=MergeType.EXTERNAL,
                label=f"LEX({word})",
                metadata={"is_leaf": True}
            )
            for emb, word in zip(embeddings, words)
        ]

        # Apply merges sequentially
        for left_idx, right_idx, head_pos, label in structure:
            # Get nodes to merge
            if right_idx == -1:  # Use last merged node
                right_idx = len(nodes) - 1

            left_node = nodes[left_idx]
            right_node = nodes[right_idx]

            # Determine head
            alpha_is_head = (head_pos == "left")

            # External merge
            merged = self.external_merge(
                left_node.embedding,
                right_node.embedding,
                head=left_node.head if alpha_is_head else right_node.head,
                dependent=right_node.head if alpha_is_head else left_node.head,
                label=label,
                alpha_is_head=alpha_is_head
            )

            # Add children for tree structure
            merged.children = [left_node, right_node]

            # Append to nodes
            nodes.append(merged)

        # Return root (last merged node)
        return nodes[-1]


# ============================================================================
# Visualization
# ============================================================================

def visualize_merge_tree(merged_obj: MergedObject, indent: int = 0) -> str:
    """
    Visualize merge tree structure.

    Args:
        merged_obj: Root of merge tree
        indent: Indentation level

    Returns:
        String representation of tree
    """
    prefix = "  " * indent
    components_str = "+".join(merged_obj.components)
    result = f"{prefix}{merged_obj.label}: {components_str} (head={merged_obj.head})\n"

    for child in merged_obj.children:
        result += visualize_merge_tree(child, indent + 1)

    return result


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MERGE OPERATOR DEMO")
    print("=" * 80)
    print()

    # Mock embedder for demo
    class MockEmbedder:
        def encode(self, texts):
            """Generate deterministic embeddings for demo."""
            embeddings = []
            for text in texts:
                # Use hash for deterministic but varied embeddings
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed)
                emb = rng.normal(0, 1, 384)
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                embeddings.append(emb)
            return np.array(embeddings)

    embedder = MockEmbedder()
    merger = MergeOperator(embedder, fusion_method="weighted_sum")

    # Example 1: External Merge (simple)
    print("Example 1: External Merge")
    print("-" * 80)

    the_emb = embedder.encode(["the"])[0]
    cat_emb = embedder.encode(["cat"])[0]

    merged = merger.external_merge(
        the_emb, cat_emb,
        head="cat",
        dependent="the",
        label="NP"
    )

    print(f"Result: {merged}")
    print(f"Embedding shape: {merged.embedding.shape}")
    print(f"Components: {merged.components}")
    print(f"Head: {merged.head}")
    print()

    # Example 2: Parallel Merge (multi-word)
    print("Example 2: Parallel Merge")
    print("-" * 80)

    words = ["the", "big", "red", "ball"]
    embeddings = embedder.encode(words)

    merged_parallel = merger.parallel_merge(
        embeddings,
        words,
        head_index=3,  # "ball" is head
        label="NP"
    )

    print(f"Result: {merged_parallel}")
    print(f"Components: {merged_parallel.components}")
    print(f"Head: {merged_parallel.head}")
    print()

    # Example 3: Recursive Merge (hierarchical)
    print("Example 3: Recursive Merge")
    print("-" * 80)

    words = ["the", "big", "cat"]
    embeddings = embedder.encode(words)

    # Structure: first merge big+cat, then the+(big cat)
    structure = [
        (1, 2, "right", "N'"),   # big + cat → N' (cat is head)
        (0, -1, "right", "NP")   # the + N' → NP
    ]

    merged_recursive = merger.recursive_merge(embeddings, words, structure)

    print("Merge tree:")
    print(visualize_merge_tree(merged_recursive))

    print("[SUCCESS] Merge operator operational!")

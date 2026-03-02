"""
Multi-Modal Fusion

Combines features from multiple modalities into unified representation.
Uses attention mechanisms for intelligent feature weighting.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from .protocol import (
    ProcessedInput,
    ModalityType,
    MultiModalFusionProtocol
)


class MultiModalFusion:
    """
    Multi-modal fusion using attention mechanisms.
    
    Combines features from different modalities (text, image, audio, structured)
    into a unified representation.
    
    Strategies:
    - 'attention': Learned attention weights for each modality
    - 'concat': Simple concatenation with normalization
    - 'average': Weighted average based on confidence
    - 'max': Take maximum values across modalities
    """
    
    def __init__(self, target_dim: int = 512):
        """
        Initialize multi-modal fusion.
        
        Args:
            target_dim: Target dimension for aligned embeddings
        """
        self.target_dim = target_dim
    
    async def fuse(
        self,
        inputs: List[ProcessedInput],
        strategy: str = "attention"
    ) -> ProcessedInput:
        """
        Fuse multiple modalities into unified representation.
        
        Args:
            inputs: List of processed inputs from different modalities
            strategy: Fusion strategy ('attention', 'concat', 'average', 'max')
        
        Returns:
            Fused ProcessedInput with MULTIMODAL modality
        """
        if not inputs:
            raise ValueError("No inputs to fuse")
        
        if len(inputs) == 1:
            # Single modality, just return it
            return inputs[0]
        
        # Align embeddings to same dimension
        aligned_inputs = self.align_embeddings(inputs)
        
        # Select fusion strategy
        if strategy == "attention":
            fused_embedding = self._fuse_attention(aligned_inputs)
        elif strategy == "concat":
            fused_embedding = self._fuse_concat(aligned_inputs)
        elif strategy == "average":
            fused_embedding = self._fuse_average(aligned_inputs)
        elif strategy == "max":
            fused_embedding = self._fuse_max(aligned_inputs)
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        # Combine content descriptions
        content_parts = [inp.content for inp in inputs]
        combined_content = " | ".join(content_parts)
        
        # Combine features
        combined_features = {}
        for inp in inputs:
            modality_key = inp.modality.value
            combined_features[modality_key] = inp.features.get(modality_key, {})
        
        # Calculate combined confidence
        confidences = [inp.confidence for inp in inputs]
        combined_confidence = np.mean(confidences)
        
        # Get sources
        sources = [inp.source for inp in inputs if inp.source]
        combined_source = ", ".join(sources) if sources else None
        
        return ProcessedInput(
            modality=ModalityType.MULTIMODAL,
            content=combined_content[:500],  # Limit to 500 chars
            embedding=fused_embedding,
            confidence=combined_confidence,
            source=combined_source,
            features=combined_features
        )
    
    def align_embeddings(
        self,
        inputs: List[ProcessedInput]
    ) -> List[ProcessedInput]:
        """
        Align embeddings across modalities to same dimension.
        
        Uses simple projection (PCA-like) or padding/truncation.
        
        Args:
            inputs: List of processed inputs
        
        Returns:
            List of inputs with aligned_embeddings populated
        """
        # Collect all embeddings
        embeddings = []
        for inp in inputs:
            if inp.embedding is not None:
                embeddings.append(inp.embedding)
        
        if not embeddings:
            # No embeddings to align
            return inputs
        
        # Find max dimension
        max_dim = max(emb.shape[0] for emb in embeddings)
        target_dim = min(self.target_dim, max_dim)
        
        # Align each input
        for inp in inputs:
            if inp.embedding is not None:
                aligned = self._align_single_embedding(inp.embedding, target_dim)
                inp.aligned_embeddings[inp.modality] = aligned
            else:
                # Create zero embedding if missing
                inp.aligned_embeddings[inp.modality] = np.zeros(target_dim)
        
        return inputs
    
    def _align_single_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """Align single embedding to target dimension."""
        current_dim = embedding.shape[0]
        
        if current_dim == target_dim:
            return embedding
        elif current_dim > target_dim:
            # Truncate (or use PCA in production)
            return embedding[:target_dim]
        else:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = embedding
            return padded
    
    def _fuse_attention(self, inputs: List[ProcessedInput]) -> np.ndarray:
        """
        Fuse embeddings using attention mechanism.
        
        Simple attention: weights based on confidence scores.
        For production, use learned attention.
        """
        # Get aligned embeddings
        embeddings = []
        weights = []
        
        for inp in inputs:
            emb = inp.aligned_embeddings.get(inp.modality)
            if emb is not None:
                embeddings.append(emb)
                # Use confidence as attention weight
                weights.append(inp.confidence)
        
        if not embeddings:
            return np.zeros(self.target_dim)
        
        # Normalize weights (softmax)
        weights = np.array(weights)
        weights = np.exp(weights) / np.sum(np.exp(weights))
        
        # Weighted sum
        fused = np.zeros_like(embeddings[0])
        for emb, weight in zip(embeddings, weights):
            fused += weight * emb
        
        # Normalize
        norm = np.linalg.norm(fused)
        if norm > 0:
            fused = fused / norm
        
        return fused
    
    def _fuse_concat(self, inputs: List[ProcessedInput]) -> np.ndarray:
        """
        Fuse embeddings by concatenation.
        
        Concatenates all embeddings and projects to target_dim.
        """
        embeddings = []
        
        for inp in inputs:
            emb = inp.aligned_embeddings.get(inp.modality)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros(self.target_dim)
        
        # Concatenate
        concatenated = np.concatenate(embeddings)
        
        # Project to target dimension (simple truncation or repeat)
        if len(concatenated) > self.target_dim:
            return concatenated[:self.target_dim]
        elif len(concatenated) < self.target_dim:
            # Pad
            padded = np.zeros(self.target_dim)
            padded[:len(concatenated)] = concatenated
            return padded
        else:
            return concatenated
    
    def _fuse_average(self, inputs: List[ProcessedInput]) -> np.ndarray:
        """
        Fuse embeddings by confidence-weighted average.
        """
        embeddings = []
        confidences = []
        
        for inp in inputs:
            emb = inp.aligned_embeddings.get(inp.modality)
            if emb is not None:
                embeddings.append(emb)
                confidences.append(inp.confidence)
        
        if not embeddings:
            return np.zeros(self.target_dim)
        
        # Weighted average
        confidences = np.array(confidences)
        confidences = confidences / confidences.sum()
        
        fused = np.zeros_like(embeddings[0])
        for emb, conf in zip(embeddings, confidences):
            fused += conf * emb
        
        return fused
    
    def _fuse_max(self, inputs: List[ProcessedInput]) -> np.ndarray:
        """
        Fuse embeddings by taking maximum values.
        
        Takes element-wise maximum across all embeddings.
        """
        embeddings = []
        
        for inp in inputs:
            emb = inp.aligned_embeddings.get(inp.modality)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.zeros(self.target_dim)
        
        # Element-wise max
        stacked = np.stack(embeddings, axis=0)
        fused = np.max(stacked, axis=0)
        
        return fused
    
    def compute_cross_modal_similarity(
        self,
        input1: ProcessedInput,
        input2: ProcessedInput
    ) -> float:
        """
        Compute similarity between two modalities.
        
        Uses cosine similarity of aligned embeddings.
        
        Args:
            input1: First processed input
            input2: Second processed input
        
        Returns:
            Similarity score (0.0-1.0)
        """
        emb1 = input1.aligned_embeddings.get(input1.modality)
        emb2 = input2.aligned_embeddings.get(input2.modality)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Convert to 0-1 range
        return (similarity + 1.0) / 2.0

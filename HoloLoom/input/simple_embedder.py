"""
Simple TF-IDF-based Embedder

Fallback embedder when sentence transformers are not available.
Uses TF-IDF for basic semantic representation.
"""

import numpy as np
import hashlib
from typing import Union, List
from collections import Counter


class SimpleEmbedder:
    """
    Simple TF-IDF-style embedder for text.
    
    Generates fixed-dimension embeddings using:
    - Token hashing for vocabulary independence
    - TF-IDF weighting for semantic importance
    - Cosine-normalized outputs
    """
    
    def __init__(self, dimension: int = 512, n_features: int = 10000):
        """
        Initialize simple embedder.
        
        Args:
            dimension: Output embedding dimension
            n_features: Number of hash buckets for token hashing
        """
        self.dimension = dimension
        self.n_features = n_features
        self.idf = {}  # Token IDF scores
        self.doc_count = 0
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text to fixed-dimension embedding.
        
        Args:
            text: Input text or list of texts
        
        Returns:
            Embedding vector(s)
        """
        if isinstance(text, list):
            return np.array([self._encode_single(t) for t in text])
        else:
            return self._encode_single(text)
    
    def _encode_single(self, text: str) -> np.ndarray:
        """Encode single text."""
        # Tokenize
        tokens = self._tokenize(text)
        
        if not tokens:
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Count tokens
        token_counts = Counter(tokens)
        
        # Create hashed embedding
        embedding = np.zeros(self.dimension, dtype=np.float32)
        
        for token, count in token_counts.items():
            # Hash token to dimension index
            hash_val = int(hashlib.md5(token.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            
            # TF-IDF weight (simplified - no corpus IDF)
            tf = count / len(tokens)
            idf = 1.0  # Default IDF
            
            embedding[idx] += tf * idf
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Lowercase and split
        text = text.lower()
        
        # Remove punctuation
        text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
        
        # Split and filter
        tokens = [t for t in text.split() if len(t) > 2]
        
        return tokens


class StructuredEmbedder:
    """
    Simple embedder for structured data.
    
    Converts JSON/dict structures to fixed-dimension embeddings.
    """
    
    def __init__(self, dimension: int = 128):
        """
        Initialize structured embedder.
        
        Args:
            dimension: Output embedding dimension
        """
        self.dimension = dimension
    
    def encode(self, data: Union[dict, list]) -> np.ndarray:
        """
        Encode structured data to embedding.
        
        Args:
            data: Dictionary or list
        
        Returns:
            Embedding vector
        """
        # Flatten structure
        features = self._extract_features(data)
        
        # Create hash-based embedding
        embedding = np.zeros(self.dimension, dtype=np.float32)
        
        for key, value in features.items():
            # Hash key-value pair
            feature_str = f"{key}:{value}"
            hash_val = int(hashlib.md5(feature_str.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            
            embedding[idx] += 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        
        return embedding
    
    def _extract_features(self, data, prefix='') -> dict:
        """Recursively extract features from nested structure."""
        features = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    features.update(self._extract_features(value, new_prefix))
                else:
                    features[new_prefix] = str(value)
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                
                if isinstance(item, (dict, list)):
                    features.update(self._extract_features(item, new_prefix))
                else:
                    features[new_prefix] = str(item)
        
        else:
            features[prefix or 'value'] = str(data)
        
        return features

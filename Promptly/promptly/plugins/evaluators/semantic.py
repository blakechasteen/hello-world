#!/usr/bin/env python3
"""
Semantic Similarity Evaluator Plugin

Evaluates output based on semantic similarity using various methods.
"""

from typing import Dict, Any, Optional
import re
from ..base import BaseEvaluator


class SemanticSimilarityEvaluator(BaseEvaluator):
    """
    Evaluates text based on semantic similarity.

    Supports multiple backends:
    - TF-IDF cosine similarity (default, no dependencies)
    - sentence-transformers (if available)
    - OpenAI embeddings (if API key provided)

    Falls back gracefully if advanced dependencies not available.
    """

    def __init__(self, backend: str = "tfidf", model_name: str = None, api_key: str = None):
        """
        Initialize semantic evaluator

        Args:
            backend: 'tfidf', 'sentence-transformers', or 'openai'
            model_name: Model name for sentence-transformers
            api_key: API key for OpenAI
        """
        super().__init__(
            name="semantic",
            description="Semantic similarity evaluator using embeddings"
        )
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self._model = None

        self._init_backend()

    def _init_backend(self):
        """Initialize the selected backend"""
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.model_name or 'all-MiniLM-L6-v2'
                self._model = SentenceTransformer(model_name)
                self._similarity_func = self._sentence_transformer_similarity
            except ImportError:
                print("Warning: sentence-transformers not available, falling back to TF-IDF")
                self.backend = "tfidf"
                self._similarity_func = self._tfidf_similarity

        elif self.backend == "openai":
            if not self.api_key:
                print("Warning: OpenAI API key not provided, falling back to TF-IDF")
                self.backend = "tfidf"
                self._similarity_func = self._tfidf_similarity
            else:
                self._similarity_func = self._openai_similarity

        else:  # tfidf
            self._similarity_func = self._tfidf_similarity

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate semantic similarity between actual and expected

        Args:
            actual: The actual output text
            expected: The expected text or reference
            context: Optional context (can include threshold settings)

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        if not actual or not expected:
            return 0.0

        similarity = self._similarity_func(actual, expected)

        # Apply threshold if provided in context
        if context and 'threshold' in context:
            threshold = context['threshold']
            if similarity < threshold:
                return 0.0

        return max(0.0, min(1.0, similarity))

    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity (no external dependencies)"""
        # Tokenize
        words1 = self._tokenize(text1)
        words2 = self._tokenize(text2)

        # Build vocabulary
        vocab = sorted(set(words1 + words2))
        if not vocab:
            return 0.0

        # Create TF-IDF vectors
        vec1 = self._tfidf_vector(words1, vocab)
        vec2 = self._tfidf_vector(words2, vocab)

        # Cosine similarity
        return self._cosine_similarity(vec1, vec2)

    def _sentence_transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence-transformers"""
        from sentence_transformers import util

        # Encode texts
        embeddings = self._model.encode([text1, text2])

        # Calculate cosine similarity
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity[0][0])

    def _openai_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using OpenAI embeddings"""
        try:
            import openai
            openai.api_key = self.api_key

            # Get embeddings
            response1 = openai.Embedding.create(
                input=text1,
                model="text-embedding-ada-002"
            )
            response2 = openai.Embedding.create(
                input=text2,
                model="text-embedding-ada-002"
            )

            emb1 = response1['data'][0]['embedding']
            emb2 = response2['data'][0]['embedding']

            # Cosine similarity
            return self._cosine_similarity(emb1, emb2)

        except Exception as e:
            print(f"Warning: OpenAI embedding failed ({e}), falling back to TF-IDF")
            return self._tfidf_similarity(text1, text2)

    def _tokenize(self, text: str) -> list:
        """Simple tokenization"""
        # Lowercase and split on non-alphanumeric
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def _tfidf_vector(self, words: list, vocab: list) -> list:
        """Create simple TF-IDF vector"""
        import math

        # Term frequency
        tf = {}
        for word in words:
            tf[word] = tf.get(word, 0) + 1

        # Normalize by document length
        doc_length = len(words)
        for word in tf:
            tf[word] = tf[word] / doc_length if doc_length > 0 else 0

        # Create vector (using simplified IDF = 1 for single document comparison)
        vector = [tf.get(word, 0) for word in vocab]

        return vector

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """Calculate cosine similarity between two vectors"""
        if len(vec1) != len(vec2):
            return 0.0

        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed metrics including similarity score and backend info"""
        score = self.evaluate(actual, expected, context)

        metrics = {
            "score": score,
            "backend": self.backend,
            "actual_length": len(actual),
            "expected_length": len(expected),
            "actual_word_count": len(actual.split()),
            "expected_word_count": len(expected.split())
        }

        if self.backend == "sentence-transformers" and self._model:
            metrics["model_name"] = self.model_name or 'all-MiniLM-L6-v2'

        return metrics


class ExactMatchEvaluator(BaseEvaluator):
    """Simple exact match evaluator"""

    def __init__(self, case_sensitive: bool = False, ignore_whitespace: bool = True):
        super().__init__(
            name="exact_match",
            description="Exact string matching evaluator"
        )
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate exact match

        Args:
            actual: The actual output text
            expected: The expected text
            context: Optional context

        Returns:
            float: 1.0 if exact match, 0.0 otherwise
        """
        if not actual or not expected:
            return 0.0

        actual_text = actual
        expected_text = expected

        if self.ignore_whitespace:
            actual_text = re.sub(r'\s+', ' ', actual_text).strip()
            expected_text = re.sub(r'\s+', ' ', expected_text).strip()

        if not self.case_sensitive:
            actual_text = actual_text.lower()
            expected_text = expected_text.lower()

        return 1.0 if actual_text == expected_text else 0.0

#!/usr/bin/env python3
"""
NLP Metrics Evaluator Plugins

Implements standard NLP evaluation metrics:
- BLEU (machine translation quality)
- ROUGE (summarization quality)
- Cosine similarity (semantic similarity via embeddings)
- Perplexity (language model quality)
- Named Entity Overlap (entity extraction quality)
"""

from typing import Dict, Any, Optional, List, Set
import re
import math
from collections import Counter
from ..base import BaseEvaluator


class BLEUEvaluator(BaseEvaluator):
    """
    BLEU (Bilingual Evaluation Understudy) score evaluator.

    Measures n-gram overlap between generated and reference text.
    Commonly used for machine translation evaluation.
    """

    def __init__(self, max_n: int = 4, smooth: bool = True):
        """
        Initialize BLEU evaluator

        Args:
            max_n: Maximum n-gram size (default 4 for BLEU-4)
            smooth: Apply smoothing for zero counts
        """
        super().__init__(
            name="bleu",
            description=f"BLEU-{max_n} score for translation/generation quality"
        )
        self.max_n = max_n
        self.smooth = smooth

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate BLEU score

        Args:
            actual: Generated text (hypothesis)
            expected: Reference text
            context: Optional context (can contain multiple references)

        Returns:
            float: BLEU score between 0.0 and 1.0
        """
        # Tokenize
        hypothesis = self._tokenize(actual)
        references = [self._tokenize(expected)]

        # Handle multiple references if provided
        if context and 'references' in context:
            references = [self._tokenize(ref) for ref in context['references']]

        if not hypothesis or not any(references):
            return 0.0

        # Calculate BLEU score
        return self._calculate_bleu(hypothesis, references)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from tokens"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams

    def _calculate_bleu(self, hypothesis: List[str], references: List[List[str]]) -> float:
        """Calculate BLEU score"""
        # Calculate precision for each n-gram size
        precisions = []

        for n in range(1, self.max_n + 1):
            hyp_ngrams = self._get_ngrams(hypothesis, n)
            max_ref_counts = Counter()

            # Get maximum counts across all references
            for reference in references:
                ref_ngrams = self._get_ngrams(reference, n)
                for ngram in ref_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

            # Calculate clipped counts
            clipped_counts = 0
            total_counts = 0

            for ngram in hyp_ngrams:
                clipped_counts += min(hyp_ngrams[ngram], max_ref_counts[ngram])
                total_counts += hyp_ngrams[ngram]

            # Calculate precision with smoothing if needed
            if total_counts == 0:
                precision = 0.0
            elif clipped_counts == 0 and self.smooth:
                precision = 1.0 / (2 ** n * total_counts)  # Smoothing
            else:
                precision = clipped_counts / total_counts

            precisions.append(precision)

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0

        # Brevity penalty
        hyp_len = len(hypothesis)
        ref_len = min(len(ref) for ref in references)  # Closest reference length

        if hyp_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0

        return bp * geo_mean

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed BLEU metrics"""
        hypothesis = self._tokenize(actual)
        references = [self._tokenize(expected)]

        if context and 'references' in context:
            references = [self._tokenize(ref) for ref in context['references']]

        # Calculate per-n-gram precisions
        precisions = {}
        for n in range(1, self.max_n + 1):
            hyp_ngrams = self._get_ngrams(hypothesis, n)
            max_ref_counts = Counter()

            for reference in references:
                ref_ngrams = self._get_ngrams(reference, n)
                for ngram in ref_ngrams:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])

            clipped = sum(min(hyp_ngrams[ng], max_ref_counts[ng]) for ng in hyp_ngrams)
            total = sum(hyp_ngrams.values())
            precisions[f'precision_{n}'] = clipped / total if total > 0 else 0.0

        return {
            'score': self.evaluate(actual, expected, context),
            'hypothesis_length': len(hypothesis),
            'reference_length': min(len(ref) for ref in references),
            **precisions
        }


class ROUGEEvaluator(BaseEvaluator):
    """
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score evaluator.

    Measures recall of n-grams and longest common subsequence.
    Commonly used for summarization evaluation.
    """

    def __init__(self, variant: str = "rouge-l", beta: float = 1.0):
        """
        Initialize ROUGE evaluator

        Args:
            variant: ROUGE variant ('rouge-1', 'rouge-2', 'rouge-l')
            beta: F-measure beta parameter (1.0 = F1, higher favors recall)
        """
        super().__init__(
            name=f"rouge_{variant.split('-')[1]}",
            description=f"{variant.upper()} score for summarization quality"
        )
        self.variant = variant.lower()
        self.beta = beta

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate ROUGE score

        Args:
            actual: Generated summary
            expected: Reference summary
            context: Optional context (can contain multiple references)

        Returns:
            float: ROUGE F-measure between 0.0 and 1.0
        """
        # Tokenize
        hypothesis = self._tokenize(actual)
        reference = self._tokenize(expected)

        if not hypothesis or not reference:
            return 0.0

        if self.variant == 'rouge-1':
            return self._rouge_n(hypothesis, reference, 1)
        elif self.variant == 'rouge-2':
            return self._rouge_n(hypothesis, reference, 2)
        elif self.variant == 'rouge-l':
            return self._rouge_l(hypothesis, reference)
        else:
            raise ValueError(f"Unknown ROUGE variant: {self.variant}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def _rouge_n(self, hypothesis: List[str], reference: List[str], n: int) -> float:
        """Calculate ROUGE-N score"""
        hyp_ngrams = self._get_ngrams(hypothesis, n)
        ref_ngrams = self._get_ngrams(reference, n)

        # Calculate overlap
        overlap = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in ref_ngrams)
        ref_count = sum(ref_ngrams.values())
        hyp_count = sum(hyp_ngrams.values())

        if ref_count == 0 or hyp_count == 0:
            return 0.0

        recall = overlap / ref_count
        precision = overlap / hyp_count

        # F-measure
        if precision + recall == 0:
            return 0.0

        beta_sq = self.beta ** 2
        f_measure = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

        return f_measure

    def _rouge_l(self, hypothesis: List[str], reference: List[str]) -> float:
        """Calculate ROUGE-L score (based on LCS)"""
        lcs_length = self._lcs_length(hypothesis, reference)

        if len(hypothesis) == 0 or len(reference) == 0:
            return 0.0

        recall = lcs_length / len(reference)
        precision = lcs_length / len(hypothesis)

        if precision + recall == 0:
            return 0.0

        beta_sq = self.beta ** 2
        f_measure = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)

        return f_measure

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed ROUGE metrics"""
        hypothesis = self._tokenize(actual)
        reference = self._tokenize(expected)

        metrics = {
            'score': self.evaluate(actual, expected, context),
            'variant': self.variant,
            'hypothesis_length': len(hypothesis),
            'reference_length': len(reference)
        }

        if self.variant == 'rouge-l':
            lcs = self._lcs_length(hypothesis, reference)
            metrics['lcs_length'] = lcs
            metrics['lcs_recall'] = lcs / len(reference) if len(reference) > 0 else 0.0
            metrics['lcs_precision'] = lcs / len(hypothesis) if len(hypothesis) > 0 else 0.0

        return metrics


class CosineSimilarityEvaluator(BaseEvaluator):
    """
    Cosine similarity evaluator using sentence embeddings.

    Measures semantic similarity via embedding vectors.
    Supports multiple embedding backends.
    """

    def __init__(self, backend: str = "tfidf", model_name: Optional[str] = None):
        """
        Initialize cosine similarity evaluator

        Args:
            backend: Embedding backend ('tfidf', 'sentence-transformers', 'openai')
            model_name: Model name for sentence-transformers
        """
        super().__init__(
            name="cosine_similarity",
            description="Cosine similarity using embeddings"
        )
        self.backend = backend
        self.model_name = model_name
        self._model = None
        self._init_backend()

    def _init_backend(self):
        """Initialize embedding backend"""
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self.model_name = self.model_name or 'all-MiniLM-L6-v2'
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                print("Warning: sentence-transformers not available, falling back to TF-IDF")
                self.backend = "tfidf"

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate cosine similarity"""
        if not actual or not expected:
            return 0.0

        if self.backend == "sentence-transformers" and self._model:
            return self._sentence_transformer_similarity(actual, expected)
        else:
            return self._tfidf_similarity(actual, expected)

    def _sentence_transformer_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using sentence transformers"""
        from sentence_transformers import util
        embeddings = self._model.encode([text1, text2])
        similarity = util.cos_sim(embeddings[0], embeddings[1])
        return float(similarity[0][0])

    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity"""
        words1 = text1.lower().split()
        words2 = text2.lower().split()

        vocab = sorted(set(words1 + words2))
        if not vocab:
            return 0.0

        vec1 = self._tfidf_vector(words1, vocab)
        vec2 = self._tfidf_vector(words2, vocab)

        return self._cosine_sim(vec1, vec2)

    def _tfidf_vector(self, words: List[str], vocab: List[str]) -> List[float]:
        """Create TF-IDF vector"""
        tf = Counter(words)
        total = len(words)
        return [tf.get(word, 0) / total if total > 0 else 0 for word in vocab]

    def _cosine_sim(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot / (mag1 * mag2) if mag1 > 0 and mag2 > 0 else 0.0


class PerplexityEvaluator(BaseEvaluator):
    """
    Perplexity evaluator for language model quality.

    Measures how well a probability model predicts a sample.
    Lower perplexity = better model fit.
    """

    def __init__(self, model_name: str = "gpt2"):
        """
        Initialize perplexity evaluator

        Args:
            model_name: Language model to use (requires transformers library)
        """
        super().__init__(
            name="perplexity",
            description="Perplexity score for language model quality"
        )
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._init_model()

    def _init_model(self):
        """Initialize language model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._model.eval()

            # Use GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()

        except ImportError:
            print("Warning: transformers library not available. Install with: pip install transformers")
        except Exception as e:
            print(f"Warning: Could not load model {self.model_name}: {e}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate perplexity score (normalized to 0-1 range)

        Args:
            actual: Text to evaluate
            expected: Not used (perplexity is intrinsic metric)
            context: Optional context

        Returns:
            float: Normalized perplexity score (1.0 = low perplexity, 0.0 = high perplexity)
        """
        if not self._model or not actual:
            return 0.5

        try:
            import torch

            # Tokenize
            inputs = self._tokenizer(actual, return_tensors="pt", truncation=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Calculate loss (cross-entropy)
            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            # Perplexity = exp(loss)
            perplexity = math.exp(loss)

            # Normalize to 0-1 range (lower perplexity = higher score)
            # Using sigmoid-like transformation
            normalized = 1.0 / (1.0 + math.log1p(perplexity))

            return max(0.0, min(1.0, normalized))

        except Exception as e:
            print(f"Warning: Perplexity calculation failed ({e})")
            return 0.5

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed perplexity metrics"""
        if not self._model or not actual:
            return {'score': 0.5, 'perplexity': None}

        try:
            import torch

            inputs = self._tokenizer(actual, return_tensors="pt", truncation=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()

            perplexity = math.exp(loss)

            return {
                'score': self.evaluate(actual, expected, context),
                'perplexity': perplexity,
                'cross_entropy_loss': loss,
                'model': self.model_name,
                'token_count': len(inputs['input_ids'][0])
            }

        except Exception as e:
            return {'score': 0.5, 'perplexity': None, 'error': str(e)}


class NamedEntityOverlapEvaluator(BaseEvaluator):
    """
    Named Entity Overlap evaluator.

    Measures overlap of named entities between generated and reference text.
    Useful for information extraction and entity-centric tasks.
    """

    def __init__(self, entity_types: Optional[List[str]] = None, case_sensitive: bool = False):
        """
        Initialize NER overlap evaluator

        Args:
            entity_types: Entity types to consider (e.g., ['PERSON', 'ORG', 'LOC'])
            case_sensitive: Whether entity matching is case-sensitive
        """
        super().__init__(
            name="ner_overlap",
            description="Named entity overlap evaluator"
        )
        self.entity_types = entity_types
        self.case_sensitive = case_sensitive
        self._nlp = None
        self._init_nlp()

    def _init_nlp(self):
        """Initialize NLP pipeline for NER"""
        try:
            import spacy
            # Try to load English model
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model not found. Download with: python -m spacy download en_core_web_sm")
                # Fallback to pattern-based extraction
                self._nlp = None
        except ImportError:
            print("Warning: spaCy not available. Install with: pip install spacy")
            self._nlp = None

    def _extract_entities_spacy(self, text: str) -> Set[tuple]:
        """Extract named entities using spaCy"""
        doc = self._nlp(text)
        entities = set()

        for ent in doc.ents:
            if self.entity_types is None or ent.label_ in self.entity_types:
                entity_text = ent.text if self.case_sensitive else ent.text.lower()
                entities.add((entity_text, ent.label_))

        return entities

    def _extract_entities_pattern(self, text: str) -> Set[tuple]:
        """Extract entities using simple patterns (fallback)"""
        entities = set()

        # Capitalized words (likely proper nouns)
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                entity_text = word if self.case_sensitive else word.lower()
                entities.add((entity_text, 'UNKNOWN'))

        return entities

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate named entity overlap

        Args:
            actual: Generated text
            expected: Reference text
            context: Optional context

        Returns:
            float: F1 score of entity overlap
        """
        if not actual or not expected:
            return 0.0

        # Extract entities
        if self._nlp:
            actual_entities = self._extract_entities_spacy(actual)
            expected_entities = self._extract_entities_spacy(expected)
        else:
            actual_entities = self._extract_entities_pattern(actual)
            expected_entities = self._extract_entities_pattern(expected)

        if not expected_entities:
            return 1.0 if not actual_entities else 0.0

        # Calculate overlap
        overlap = actual_entities & expected_entities

        if not overlap:
            return 0.0

        precision = len(overlap) / len(actual_entities) if actual_entities else 0.0
        recall = len(overlap) / len(expected_entities) if expected_entities else 0.0

        # F1 score
        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed NER overlap metrics"""
        if self._nlp:
            actual_entities = self._extract_entities_spacy(actual)
            expected_entities = self._extract_entities_spacy(expected)
        else:
            actual_entities = self._extract_entities_pattern(actual)
            expected_entities = self._extract_entities_pattern(expected)

        overlap = actual_entities & expected_entities

        precision = len(overlap) / len(actual_entities) if actual_entities else 0.0
        recall = len(overlap) / len(expected_entities) if expected_entities else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'score': f1,
            'precision': precision,
            'recall': recall,
            'actual_entity_count': len(actual_entities),
            'expected_entity_count': len(expected_entities),
            'overlap_count': len(overlap),
            'actual_entities': [{'text': e[0], 'type': e[1]} for e in actual_entities],
            'expected_entities': [{'text': e[0], 'type': e[1]} for e in expected_entities],
            'missing_entities': [{'text': e[0], 'type': e[1]} for e in (expected_entities - actual_entities)],
            'extra_entities': [{'text': e[0], 'type': e[1]} for e in (actual_entities - expected_entities)]
        }

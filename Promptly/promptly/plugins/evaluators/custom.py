#!/usr/bin/env python3
"""
Custom Metrics Evaluator Plugins

Implements specialized evaluation metrics:
- Length-based scoring
- Readability metrics (Flesch-Kincaid, etc.)
- Sentiment analysis
- Toxicity detection
- JSON schema validation
"""

from typing import Dict, Any, Optional, List
import re
import json
import math
from ..base import BaseEvaluator


class LengthEvaluator(BaseEvaluator):
    """
    Length-based evaluator.

    Scores text based on length constraints and target ranges.
    Useful for controlling output verbosity.
    """

    def __init__(self, target_length: Optional[int] = None,
                 min_length: Optional[int] = None,
                 max_length: Optional[int] = None,
                 unit: str = "chars"):
        """
        Initialize length evaluator

        Args:
            target_length: Target length (ideal)
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length
            unit: Unit of measurement ('chars', 'words', 'sentences')
        """
        super().__init__(
            name="length",
            description=f"Length-based evaluator (unit: {unit})"
        )
        self.target_length = target_length
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit

    def _measure_length(self, text: str) -> int:
        """Measure text length in specified unit"""
        if self.unit == "chars":
            return len(text)
        elif self.unit == "words":
            return len(text.split())
        elif self.unit == "sentences":
            # Simple sentence split on .!?
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
        else:
            raise ValueError(f"Unknown unit: {self.unit}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate based on length

        Args:
            actual: Text to evaluate
            expected: Reference text (used for target length if not specified)
            context: Optional context with length constraints

        Returns:
            float: Score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        # Get length constraints from context if provided
        target = context.get('target_length', self.target_length) if context else self.target_length
        min_len = context.get('min_length', self.min_length) if context else self.min_length
        max_len = context.get('max_length', self.max_length) if context else self.max_length

        # If no constraints specified, use expected text length as target
        if target is None and min_len is None and max_len is None:
            target = self._measure_length(expected)

        actual_length = self._measure_length(actual)

        # Hard constraints
        if min_len is not None and actual_length < min_len:
            # Penalty for being too short
            ratio = actual_length / min_len
            return max(0.0, ratio * 0.5)  # Max 50% score if too short

        if max_len is not None and actual_length > max_len:
            # Penalty for being too long
            ratio = max_len / actual_length
            return max(0.0, ratio * 0.5)  # Max 50% score if too long

        # Soft target (if specified)
        if target is not None:
            # Score based on distance from target
            if actual_length == target:
                return 1.0

            # Use Gaussian-like scoring
            diff = abs(actual_length - target)
            tolerance = target * 0.2  # 20% tolerance
            score = math.exp(-(diff ** 2) / (2 * (tolerance ** 2)))
            return max(0.0, min(1.0, score))

        # If only min/max specified and within bounds
        return 1.0

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed length metrics"""
        return {
            'score': self.evaluate(actual, expected, context),
            'actual_length': self._measure_length(actual),
            'expected_length': self._measure_length(expected),
            'unit': self.unit,
            'target_length': self.target_length,
            'min_length': self.min_length,
            'max_length': self.max_length
        }


class ReadabilityEvaluator(BaseEvaluator):
    """
    Readability metrics evaluator.

    Calculates various readability scores:
    - Flesch Reading Ease
    - Flesch-Kincaid Grade Level
    - Automated Readability Index (ARI)
    """

    def __init__(self, metric: str = "flesch_reading_ease",
                 target_score: Optional[float] = None):
        """
        Initialize readability evaluator

        Args:
            metric: Readability metric ('flesch_reading_ease', 'flesch_kincaid_grade', 'ari')
            target_score: Target readability score (optional)
        """
        super().__init__(
            name=f"readability_{metric}",
            description=f"Readability evaluator using {metric}"
        )
        self.metric = metric
        self.target_score = target_score

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count (simple heuristic)"""
        word = word.lower()
        vowels = "aeiouy"
        syllables = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e'):
            syllables -= 1

        # Ensure at least one syllable
        return max(1, syllables)

    def _analyze_text(self, text: str) -> Dict[str, int]:
        """Analyze text for readability metrics"""
        # Count sentences
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if s.strip()])

        # Count words
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)

        # Count syllables
        syllable_count = sum(self._count_syllables(word) for word in words)

        # Count characters
        char_count = sum(len(word) for word in words)

        return {
            'sentences': max(1, sentence_count),
            'words': max(1, word_count),
            'syllables': max(1, syllable_count),
            'characters': max(1, char_count)
        }

    def _flesch_reading_ease(self, stats: Dict[str, int]) -> float:
        """
        Calculate Flesch Reading Ease score.

        Score ranges:
        90-100: Very Easy (5th grade)
        80-90: Easy (6th grade)
        70-80: Fairly Easy (7th grade)
        60-70: Standard (8th-9th grade)
        50-60: Fairly Difficult (10th-12th grade)
        30-50: Difficult (College)
        0-30: Very Difficult (College graduate)
        """
        avg_sentence_length = stats['words'] / stats['sentences']
        avg_syllables_per_word = stats['syllables'] / stats['words']

        score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

        return max(0.0, min(100.0, score))

    def _flesch_kincaid_grade(self, stats: Dict[str, int]) -> float:
        """
        Calculate Flesch-Kincaid Grade Level.

        Returns US grade level (e.g., 8.0 = 8th grade)
        """
        avg_sentence_length = stats['words'] / stats['sentences']
        avg_syllables_per_word = stats['syllables'] / stats['words']

        grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59

        return max(0.0, grade)

    def _automated_readability_index(self, stats: Dict[str, int]) -> float:
        """
        Calculate Automated Readability Index (ARI).

        Returns US grade level
        """
        avg_chars_per_word = stats['characters'] / stats['words']
        avg_words_per_sentence = stats['words'] / stats['sentences']

        ari = 4.71 * avg_chars_per_word + 0.5 * avg_words_per_sentence - 21.43

        return max(0.0, ari)

    def calculate_readability(self, text: str) -> float:
        """Calculate readability score based on selected metric"""
        stats = self._analyze_text(text)

        if self.metric == "flesch_reading_ease":
            return self._flesch_reading_ease(stats)
        elif self.metric == "flesch_kincaid_grade":
            return self._flesch_kincaid_grade(stats)
        elif self.metric == "ari":
            return self._automated_readability_index(stats)
        else:
            raise ValueError(f"Unknown readability metric: {self.metric}")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate readability

        Args:
            actual: Text to evaluate
            expected: Reference text (for target score if not specified)
            context: Optional context with target score

        Returns:
            float: Normalized score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        actual_score = self.calculate_readability(actual)

        # Get target score
        target = context.get('target_score', self.target_score) if context else self.target_score

        if target is None:
            # Use expected text readability as target
            target = self.calculate_readability(expected) if expected else actual_score

        # Normalize based on metric
        if self.metric == "flesch_reading_ease":
            # Higher is better, normalize to 0-1
            # If target specified, score based on distance
            if target:
                diff = abs(actual_score - target)
                tolerance = 20  # ±20 points tolerance
                score = math.exp(-(diff ** 2) / (2 * (tolerance ** 2)))
                return max(0.0, min(1.0, score))
            else:
                # Just normalize to 0-1 range
                return actual_score / 100.0
        else:
            # Grade level metrics (lower is better for general audience)
            # If target specified, score based on distance
            if target:
                diff = abs(actual_score - target)
                tolerance = 2.0  # ±2 grade levels tolerance
                score = math.exp(-(diff ** 2) / (2 * (tolerance ** 2)))
                return max(0.0, min(1.0, score))
            else:
                # Prefer 6-10th grade level (6-10 score)
                if 6 <= actual_score <= 10:
                    return 1.0
                elif actual_score < 6:
                    return actual_score / 6.0
                else:
                    return max(0.0, 1.0 - (actual_score - 10) / 10.0)

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed readability metrics"""
        stats = self._analyze_text(actual)
        actual_score = self.calculate_readability(actual)
        expected_score = self.calculate_readability(expected) if expected else None

        return {
            'score': self.evaluate(actual, expected, context),
            'readability_score': actual_score,
            'expected_readability': expected_score,
            'metric': self.metric,
            **stats
        }


class SentimentEvaluator(BaseEvaluator):
    """
    Sentiment analysis evaluator.

    Evaluates sentiment polarity and subjectivity.
    """

    def __init__(self, backend: str = "simple", target_polarity: Optional[str] = None):
        """
        Initialize sentiment evaluator

        Args:
            backend: Sentiment backend ('simple', 'textblob', 'transformers')
            target_polarity: Target sentiment ('positive', 'negative', 'neutral')
        """
        super().__init__(
            name="sentiment",
            description="Sentiment analysis evaluator"
        )
        self.backend = backend
        self.target_polarity = target_polarity
        self._analyzer = None
        self._init_backend()

    def _init_backend(self):
        """Initialize sentiment backend"""
        if self.backend == "textblob":
            try:
                from textblob import TextBlob
                self._analyzer = TextBlob
            except ImportError:
                print("Warning: textblob not available, falling back to simple sentiment")
                self.backend = "simple"
        elif self.backend == "transformers":
            try:
                from transformers import pipeline
                self._analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            except ImportError:
                print("Warning: transformers not available, falling back to simple sentiment")
                self.backend = "simple"

    def _simple_sentiment(self, text: str) -> Dict[str, float]:
        """Simple lexicon-based sentiment analysis"""
        # Simple positive/negative word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'best',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'perfect', 'outstanding',
            'brilliant', 'awesome', 'beautiful', 'nice', 'positive', 'success'
        }

        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'dislike',
            'angry', 'sad', 'disappointed', 'poor', 'negative', 'fail', 'failure',
            'wrong', 'error', 'problem', 'issue', 'difficult', 'hard'
        }

        words = text.lower().split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        total = pos_count + neg_count

        if total == 0:
            polarity = 0.0  # Neutral
        else:
            polarity = (pos_count - neg_count) / total

        return {
            'polarity': polarity,  # -1 to 1
            'subjectivity': min(1.0, total / max(1, len(words))),  # 0 to 1
            'positive_words': pos_count,
            'negative_words': neg_count
        }

    def _textblob_sentiment(self, text: str) -> Dict[str, float]:
        """TextBlob sentiment analysis"""
        blob = self._analyzer(text)
        return {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        }

    def _transformers_sentiment(self, text: str) -> Dict[str, float]:
        """Transformers-based sentiment analysis"""
        result = self._analyzer(text[:512])[0]  # Truncate to max length
        label = result['label']
        score = result['score']

        # Convert to polarity score
        polarity = score if label == 'POSITIVE' else -score

        return {
            'polarity': polarity,
            'label': label,
            'confidence': score
        }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment"""
        if not text:
            return {'polarity': 0.0, 'subjectivity': 0.0}

        if self.backend == "textblob" and self._analyzer:
            return self._textblob_sentiment(text)
        elif self.backend == "transformers" and self._analyzer:
            return self._transformers_sentiment(text)
        else:
            return self._simple_sentiment(text)

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate sentiment match

        Args:
            actual: Text to evaluate
            expected: Reference text
            context: Optional context with target sentiment

        Returns:
            float: Score between 0.0 and 1.0
        """
        if not actual:
            return 0.0

        actual_sentiment = self.analyze_sentiment(actual)
        target_polarity = context.get('target_polarity', self.target_polarity) if context else self.target_polarity

        if target_polarity:
            # Score based on target polarity
            actual_pol = actual_sentiment['polarity']

            if target_polarity == 'positive':
                # Score increases with positive polarity
                return max(0.0, (actual_pol + 1) / 2)
            elif target_polarity == 'negative':
                # Score increases with negative polarity
                return max(0.0, (1 - actual_pol) / 2)
            else:  # neutral
                # Score decreases with distance from 0
                return max(0.0, 1.0 - abs(actual_pol))
        else:
            # Compare with expected sentiment
            expected_sentiment = self.analyze_sentiment(expected)
            pol_diff = abs(actual_sentiment['polarity'] - expected_sentiment['polarity'])
            return max(0.0, 1.0 - pol_diff / 2)

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed sentiment metrics"""
        actual_sentiment = self.analyze_sentiment(actual)
        expected_sentiment = self.analyze_sentiment(expected) if expected else {}

        return {
            'score': self.evaluate(actual, expected, context),
            'actual_sentiment': actual_sentiment,
            'expected_sentiment': expected_sentiment,
            'backend': self.backend
        }


class ToxicityEvaluator(BaseEvaluator):
    """
    Toxicity detection evaluator.

    Detects toxic, offensive, or harmful content.
    """

    def __init__(self, threshold: float = 0.5, backend: str = "simple"):
        """
        Initialize toxicity evaluator

        Args:
            threshold: Toxicity threshold (0.0 to 1.0)
            backend: Detection backend ('simple', 'detoxify')
        """
        super().__init__(
            name="toxicity",
            description="Toxicity detection evaluator"
        )
        self.threshold = threshold
        self.backend = backend
        self._detector = None
        self._init_backend()

    def _init_backend(self):
        """Initialize toxicity detector"""
        if self.backend == "detoxify":
            try:
                from detoxify import Detoxify
                self._detector = Detoxify('original')
            except ImportError:
                print("Warning: detoxify not available, falling back to simple detection")
                self.backend = "simple"

    def _simple_toxicity(self, text: str) -> Dict[str, float]:
        """Simple keyword-based toxicity detection"""
        # Simple toxic word list (placeholder - expand as needed)
        toxic_words = {
            'hate', 'kill', 'stupid', 'idiot', 'dumb', 'worst',
            'terrible', 'awful', 'horrible', 'disgusting', 'pathetic'
        }

        words = text.lower().split()
        toxic_count = sum(1 for word in words if word in toxic_words)
        total_words = len(words)

        toxicity_score = min(1.0, toxic_count / max(1, total_words) * 10)

        return {
            'toxicity': toxicity_score,
            'toxic_word_count': toxic_count
        }

    def _detoxify_analysis(self, text: str) -> Dict[str, float]:
        """Detoxify-based toxicity analysis"""
        results = self._detector.predict(text)
        return {
            'toxicity': results.get('toxicity', 0.0),
            'severe_toxicity': results.get('severe_toxicity', 0.0),
            'obscene': results.get('obscene', 0.0),
            'threat': results.get('threat', 0.0),
            'insult': results.get('insult', 0.0),
            'identity_attack': results.get('identity_attack', 0.0)
        }

    def analyze_toxicity(self, text: str) -> Dict[str, float]:
        """Analyze text toxicity"""
        if not text:
            return {'toxicity': 0.0}

        if self.backend == "detoxify" and self._detector:
            return self._detoxify_analysis(text)
        else:
            return self._simple_toxicity(text)

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Evaluate toxicity (lower toxicity = higher score)

        Args:
            actual: Text to evaluate
            expected: Not used (toxicity is intrinsic metric)
            context: Optional context

        Returns:
            float: Score between 0.0 and 1.0 (1.0 = non-toxic, 0.0 = toxic)
        """
        if not actual:
            return 1.0  # Empty text is non-toxic

        toxicity = self.analyze_toxicity(actual)
        toxicity_score = toxicity['toxicity']

        # Invert score (lower toxicity = higher score)
        return max(0.0, 1.0 - toxicity_score)

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed toxicity metrics"""
        toxicity = self.analyze_toxicity(actual)

        return {
            'score': self.evaluate(actual, expected, context),
            'toxicity_analysis': toxicity,
            'is_toxic': toxicity['toxicity'] > self.threshold,
            'threshold': self.threshold,
            'backend': self.backend
        }


class JSONSchemaEvaluator(BaseEvaluator):
    """
    JSON schema validation evaluator.

    Validates JSON output against a schema.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize JSON schema evaluator

        Args:
            schema: JSON schema to validate against
        """
        super().__init__(
            name="json_schema",
            description="JSON schema validation evaluator"
        )
        self.schema = schema
        self._validator = None
        self._init_validator()

    def _init_validator(self):
        """Initialize JSON schema validator"""
        try:
            import jsonschema
            self._validator = jsonschema
        except ImportError:
            print("Warning: jsonschema not available. Install with: pip install jsonschema")

    def evaluate(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Validate JSON against schema

        Args:
            actual: JSON string to validate
            expected: Expected JSON or schema
            context: Optional context with schema

        Returns:
            float: 1.0 if valid, 0.0 if invalid
        """
        if not actual:
            return 0.0

        # Get schema from context or use provided schema
        schema = context.get('schema', self.schema) if context else self.schema

        # Try to parse expected as schema if no schema provided
        if schema is None and expected:
            try:
                schema = json.loads(expected)
            except json.JSONDecodeError:
                pass

        # Try to parse actual as JSON
        try:
            data = json.loads(actual)
        except json.JSONDecodeError:
            return 0.0  # Invalid JSON

        # If no schema, just check if valid JSON
        if schema is None:
            return 1.0

        # Validate against schema
        if self._validator:
            try:
                self._validator.validate(data, schema)
                return 1.0
            except self._validator.ValidationError:
                return 0.0
        else:
            # Basic validation without jsonschema library
            # Just check if JSON is parseable
            return 1.0

    def get_metrics(self, actual: str, expected: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get detailed JSON validation metrics"""
        schema = context.get('schema', self.schema) if context else self.schema

        metrics = {
            'score': self.evaluate(actual, expected, context),
            'has_schema': schema is not None
        }

        # Try to parse JSON and get additional info
        try:
            data = json.loads(actual)
            metrics['is_valid_json'] = True
            metrics['data_type'] = type(data).__name__

            if isinstance(data, dict):
                metrics['key_count'] = len(data)
                metrics['keys'] = list(data.keys())
            elif isinstance(data, list):
                metrics['item_count'] = len(data)

        except json.JSONDecodeError as e:
            metrics['is_valid_json'] = False
            metrics['parse_error'] = str(e)

        # Validate against schema if available
        if self._validator and schema:
            try:
                data = json.loads(actual)
                self._validator.validate(data, schema)
                metrics['schema_valid'] = True
            except json.JSONDecodeError:
                metrics['schema_valid'] = False
                metrics['validation_error'] = 'Invalid JSON'
            except self._validator.ValidationError as e:
                metrics['schema_valid'] = False
                metrics['validation_error'] = str(e.message)

        return metrics

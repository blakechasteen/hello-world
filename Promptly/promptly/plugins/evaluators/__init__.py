"""
Promptly Evaluator Plugins

Advanced evaluation plugins including:
- LLM-based evaluators (OpenAI, Anthropic, Ollama)
- NLP metrics (BLEU, ROUGE, cosine similarity, perplexity, NER overlap)
- Custom metrics (length, readability, sentiment, toxicity, JSON validation)
- Composite evaluators (weighted, voting, cascading)
"""

# Original evaluators
from .keyword import KeywordEvaluator
from .semantic import SemanticSimilarityEvaluator, ExactMatchEvaluator

# LLM-based evaluators
from .llm import (
    LLMJudgeEvaluator,
    OpenAIEvaluator,
    AnthropicEvaluator,
    OllamaEvaluator,
    LLMPairwiseEvaluator
)

# NLP metrics evaluators
from .nlp_metrics import (
    BLEUEvaluator,
    ROUGEEvaluator,
    CosineSimilarityEvaluator,
    PerplexityEvaluator,
    NamedEntityOverlapEvaluator
)

# Custom metrics evaluators
from .custom import (
    LengthEvaluator,
    ReadabilityEvaluator,
    SentimentEvaluator,
    ToxicityEvaluator,
    JSONSchemaEvaluator
)

# Composite evaluators
from .composite import (
    WeightedAverageEvaluator,
    VotingEnsembleEvaluator,
    CascadingEvaluator,
    MinMaxEvaluator,
    ConditionalEvaluator,
    ThresholdedEvaluator
)

__all__ = [
    # Original
    'KeywordEvaluator',
    'SemanticSimilarityEvaluator',
    'ExactMatchEvaluator',

    # LLM-based
    'LLMJudgeEvaluator',
    'OpenAIEvaluator',
    'AnthropicEvaluator',
    'OllamaEvaluator',
    'LLMPairwiseEvaluator',

    # NLP metrics
    'BLEUEvaluator',
    'ROUGEEvaluator',
    'CosineSimilarityEvaluator',
    'PerplexityEvaluator',
    'NamedEntityOverlapEvaluator',

    # Custom metrics
    'LengthEvaluator',
    'ReadabilityEvaluator',
    'SentimentEvaluator',
    'ToxicityEvaluator',
    'JSONSchemaEvaluator',

    # Composite
    'WeightedAverageEvaluator',
    'VotingEnsembleEvaluator',
    'CascadingEvaluator',
    'MinMaxEvaluator',
    'ConditionalEvaluator',
    'ThresholdedEvaluator'
]

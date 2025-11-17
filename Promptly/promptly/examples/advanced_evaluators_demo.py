#!/usr/bin/env python3
"""
Advanced Evaluators Demo

Comprehensive examples demonstrating all advanced evaluator plugins.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plugins.evaluators import *


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(evaluator_name, score, metrics=None):
    """Print evaluation results"""
    print(f"[{evaluator_name}] Score: {score:.3f}")
    if metrics:
        print(f"  Metrics: {metrics}")
    print()


# ============================================================================
# 1. LLM-BASED EVALUATORS
# ============================================================================

def demo_llm_evaluators():
    """Demonstrate LLM-based evaluators"""
    print_section("1. LLM-BASED EVALUATORS")

    actual = "The capital of France is Paris, which is known for the Eiffel Tower and rich cultural heritage."
    expected = "Paris is the capital of France."

    # OpenAI Evaluator (requires API key)
    print("1.1 OpenAI GPT-4 as Judge")
    print("-" * 40)
    try:
        evaluator = OpenAIEvaluator(
            model="gpt-4-turbo-preview",
            criteria=["accuracy", "completeness", "clarity"],
            temperature=0.1
        )
        score = evaluator.evaluate(actual, expected)
        metrics = evaluator.get_metrics(actual, expected)
        print_result("OpenAI GPT-4", score, metrics)
    except Exception as e:
        print(f"Skipping OpenAI evaluator: {e}\n")

    # Anthropic Evaluator (requires API key)
    print("1.2 Anthropic Claude as Judge")
    print("-" * 40)
    try:
        evaluator = AnthropicEvaluator(
            model="claude-3-5-sonnet-20241022",
            criteria=["factual accuracy", "relevance", "coherence"],
            temperature=0.1
        )
        score = evaluator.evaluate(actual, expected)
        metrics = evaluator.get_metrics(actual, expected)
        print_result("Anthropic Claude", score, metrics)
    except Exception as e:
        print(f"Skipping Anthropic evaluator: {e}\n")

    # Ollama Evaluator (requires local Ollama server)
    print("1.3 Local LLM via Ollama")
    print("-" * 40)
    try:
        evaluator = OllamaEvaluator(
            model="llama3",
            criteria=["accuracy", "conciseness"],
            temperature=0.1
        )
        score = evaluator.evaluate(actual, expected)
        metrics = evaluator.get_metrics(actual, expected)
        print_result("Ollama Llama3", score, metrics)
    except Exception as e:
        print(f"Skipping Ollama evaluator: {e}\n")

    # Pairwise Comparison (requires base LLM evaluator)
    print("1.4 LLM Pairwise Comparison")
    print("-" * 40)
    try:
        base_evaluator = OllamaEvaluator(model="llama3")
        pairwise = LLMPairwiseEvaluator(base_evaluator)

        output_a = "Paris is the capital of France."
        output_b = "The capital of France is Paris, famous for the Eiffel Tower."
        reference = "What is the capital of France?"

        result = pairwise.compare(output_a, output_b, reference)
        print(f"Winner: {result['winner']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reasoning: {result['reasoning']}\n")
    except Exception as e:
        print(f"Skipping pairwise comparison: {e}\n")


# ============================================================================
# 2. NLP METRICS EVALUATORS
# ============================================================================

def demo_nlp_metrics():
    """Demonstrate NLP metrics evaluators"""
    print_section("2. NLP METRICS EVALUATORS")

    # BLEU Score
    print("2.1 BLEU Score (Machine Translation Quality)")
    print("-" * 40)
    hypothesis = "the cat is on the mat"
    reference = "there is a cat on the mat"

    evaluator = BLEUEvaluator(max_n=4, smooth=True)
    score = evaluator.evaluate(hypothesis, reference)
    metrics = evaluator.get_metrics(hypothesis, reference)
    print_result("BLEU-4", score, metrics)

    # ROUGE Score
    print("2.2 ROUGE Score (Summarization Quality)")
    print("-" * 40)
    summary = "The quick brown fox jumps over the lazy dog."
    reference = "A fast brown fox leaps over a sleeping dog."

    for variant in ['rouge-1', 'rouge-2', 'rouge-l']:
        evaluator = ROUGEEvaluator(variant=variant, beta=1.0)
        score = evaluator.evaluate(summary, reference)
        print(f"  {variant.upper()}: {score:.3f}")
    print()

    # Cosine Similarity
    print("2.3 Cosine Similarity (Semantic Similarity)")
    print("-" * 40)
    text1 = "Machine learning is a subset of artificial intelligence."
    text2 = "AI includes machine learning as one of its components."

    evaluator = CosineSimilarityEvaluator(backend="tfidf")
    score = evaluator.evaluate(text1, text2)
    metrics = evaluator.get_metrics(text1, text2)
    print_result("Cosine Similarity", score, metrics)

    # Perplexity (requires transformers library)
    print("2.4 Perplexity Score (Language Model Quality)")
    print("-" * 40)
    try:
        evaluator = PerplexityEvaluator(model_name="gpt2")
        text = "The quick brown fox jumps over the lazy dog."
        score = evaluator.evaluate(text, "")
        metrics = evaluator.get_metrics(text, "")
        print_result("Perplexity", score, metrics)
    except Exception as e:
        print(f"Skipping perplexity evaluator: {e}\n")

    # Named Entity Overlap
    print("2.5 Named Entity Overlap (Entity Extraction Quality)")
    print("-" * 40)
    actual = "Apple Inc. is located in Cupertino, California. Tim Cook is the CEO."
    expected = "Apple is based in Cupertino. The CEO is Tim Cook."

    evaluator = NamedEntityOverlapEvaluator()
    score = evaluator.evaluate(actual, expected)
    metrics = evaluator.get_metrics(actual, expected)
    print_result("NER Overlap", score, metrics)


# ============================================================================
# 3. CUSTOM METRICS EVALUATORS
# ============================================================================

def demo_custom_metrics():
    """Demonstrate custom metrics evaluators"""
    print_section("3. CUSTOM METRICS EVALUATORS")

    # Length-based
    print("3.1 Length-based Scoring")
    print("-" * 40)
    text = "This is a sample text with some words."
    evaluator = LengthEvaluator(target_length=10, unit="words")
    score = evaluator.evaluate(text, "")
    metrics = evaluator.get_metrics(text, "")
    print_result("Length (target=10 words)", score, metrics)

    # Readability
    print("3.2 Readability Metrics")
    print("-" * 40)
    text = "The quick brown fox jumps over the lazy dog. It runs very fast through the forest."

    for metric in ['flesch_reading_ease', 'flesch_kincaid_grade', 'ari']:
        evaluator = ReadabilityEvaluator(metric=metric)
        score = evaluator.evaluate(text, "")
        metrics = evaluator.get_metrics(text, "")
        print(f"  {metric}: score={score:.3f}, readability={metrics['readability_score']:.1f}")
    print()

    # Sentiment Analysis
    print("3.3 Sentiment Analysis")
    print("-" * 40)
    positive_text = "This is amazing! I love it. Great work!"
    negative_text = "This is terrible. I hate it. Awful experience."
    neutral_text = "The product works as expected."

    evaluator = SentimentEvaluator(backend="simple", target_polarity="positive")

    for text, label in [(positive_text, "Positive"), (negative_text, "Negative"), (neutral_text, "Neutral")]:
        score = evaluator.evaluate(text, "")
        metrics = evaluator.get_metrics(text, "")
        polarity = metrics['actual_sentiment']['polarity']
        print(f"  {label} text: score={score:.3f}, polarity={polarity:.2f}")
    print()

    # Toxicity Detection
    print("3.4 Toxicity Detection")
    print("-" * 40)
    clean_text = "This is a helpful and constructive comment."
    toxic_text = "You are stupid and I hate everything about this."

    evaluator = ToxicityEvaluator(backend="simple", threshold=0.5)

    for text, label in [(clean_text, "Clean"), (toxic_text, "Toxic")]:
        score = evaluator.evaluate(text, "")
        metrics = evaluator.get_metrics(text, "")
        is_toxic = metrics['is_toxic']
        print(f"  {label} text: score={score:.3f}, is_toxic={is_toxic}")
    print()

    # JSON Schema Validation
    print("3.5 JSON Schema Validation")
    print("-" * 40)
    valid_json = '{"name": "John", "age": 30, "city": "New York"}'
    invalid_json = '{"name": "John", age: 30}'  # Invalid JSON

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
            "city": {"type": "string"}
        },
        "required": ["name", "age"]
    }

    evaluator = JSONSchemaEvaluator(schema=schema)

    for json_text, label in [(valid_json, "Valid"), (invalid_json, "Invalid")]:
        score = evaluator.evaluate(json_text, "")
        metrics = evaluator.get_metrics(json_text, "", context={'schema': schema})
        print(f"  {label} JSON: score={score:.3f}, is_valid={metrics.get('is_valid_json', False)}")
    print()


# ============================================================================
# 4. COMPOSITE EVALUATORS
# ============================================================================

def demo_composite_evaluators():
    """Demonstrate composite evaluators"""
    print_section("4. COMPOSITE EVALUATORS")

    actual = "Paris is the capital of France and is famous for the Eiffel Tower."
    expected = "Paris is the capital of France."

    # Weighted Average
    print("4.1 Weighted Average Evaluator")
    print("-" * 40)
    bleu = BLEUEvaluator(max_n=2)
    rouge = ROUGEEvaluator(variant='rouge-1')
    cosine = CosineSimilarityEvaluator(backend="tfidf")

    evaluator = WeightedAverageEvaluator([
        (bleu, 0.3),
        (rouge, 0.3),
        (cosine, 0.4)
    ])

    score = evaluator.evaluate(actual, expected)
    metrics = evaluator.get_metrics(actual, expected)
    print_result("Weighted Average", score, metrics['individual_scores'])

    # Voting Ensemble
    print("4.2 Voting Ensemble Evaluator")
    print("-" * 40)
    evaluator = VotingEnsembleEvaluator(
        evaluators=[bleu, rouge, cosine],
        threshold=0.5,
        voting_strategy="majority"
    )

    score = evaluator.evaluate(actual, expected)
    metrics = evaluator.get_metrics(actual, expected)
    print(f"  Pass votes: {metrics['pass_votes']}/{metrics['total_votes']}")
    print(f"  Final decision: {'PASS' if score == 1.0 else 'FAIL'}")
    print(f"  Individual votes: {metrics['votes']}\n")

    # Cascading Evaluator
    print("4.3 Cascading Evaluator (with short-circuit)")
    print("-" * 40)
    length_eval = LengthEvaluator(min_length=5, max_length=100, unit="words")
    toxicity_eval = ToxicityEvaluator(backend="simple")
    quality_eval = BLEUEvaluator()

    evaluator = CascadingEvaluator([
        (length_eval, 0.5),      # Must pass length check
        (toxicity_eval, 0.8),    # Must be non-toxic
        (quality_eval, 0.3)      # Then check quality
    ], stop_on_failure=True, aggregate="min")

    score = evaluator.evaluate(actual, expected)
    metrics = evaluator.get_metrics(actual, expected)
    print(f"  Final score: {score:.3f}")
    print(f"  Completed stages: {metrics['completed_stages']}/{metrics['total_stages']}")
    print(f"  All passed: {metrics['all_passed']}")
    print(f"  Stage results: {metrics['stage_results']}\n")

    # Min/Max Aggregation
    print("4.4 Min/Max Aggregation")
    print("-" * 40)
    evaluators_list = [bleu, rouge, cosine]

    for mode in ['min', 'max']:
        evaluator = MinMaxEvaluator(evaluators_list, mode=mode)
        score = evaluator.evaluate(actual, expected)
        metrics = evaluator.get_metrics(actual, expected)
        print(f"  {mode.upper()} score: {score:.3f}")
        print(f"  Individual scores: {metrics['scores']}")
    print()

    # Thresholded Evaluator
    print("4.5 Thresholded Evaluator")
    print("-" * 40)
    base_eval = CosineSimilarityEvaluator(backend="tfidf")

    for mode in ['binary', 'scale_above', 'scale_below']:
        evaluator = ThresholdedEvaluator(base_eval, threshold=0.5, mode=mode)
        score = evaluator.evaluate(actual, expected)
        metrics = evaluator.get_metrics(actual, expected)
        print(f"  {mode}: score={score:.3f}, base_score={metrics['base_score']:.3f}")
    print()

    # Conditional Evaluator
    print("4.6 Conditional Evaluator (Adaptive)")
    print("-" * 40)

    def length_condition(actual, expected, context):
        """Select evaluator based on text length"""
        word_count = len(actual.split())
        if word_count < 10:
            return 'short'
        elif word_count < 50:
            return 'medium'
        else:
            return 'long'

    evaluator = ConditionalEvaluator(
        evaluator_map={
            'short': ExactMatchEvaluator(),
            'medium': BLEUEvaluator(max_n=2),
            'long': BLEUEvaluator(max_n=4)
        },
        condition_func=length_condition,
        default_evaluator='medium'
    )

    short_text = "Hello world"
    medium_text = "The quick brown fox jumps over the lazy dog multiple times"
    long_text = " ".join(["This is a very long text with many words."] * 10)

    for text, label in [(short_text, "Short"), (medium_text, "Medium"), (long_text, "Long")]:
        score = evaluator.evaluate(text, expected)
        metrics = evaluator.get_metrics(text, expected)
        print(f"  {label}: selected={metrics['selected_evaluator']}, score={score:.3f}")
    print()


# ============================================================================
# 5. REAL-WORLD USE CASES
# ============================================================================

def demo_real_world_use_cases():
    """Demonstrate real-world evaluation scenarios"""
    print_section("5. REAL-WORLD USE CASES")

    # Use Case 1: Summarization Pipeline
    print("5.1 Summarization Quality Evaluation")
    print("-" * 40)
    summary = "AI is transforming industries through automation and data analysis."
    reference = "Artificial intelligence is revolutionizing multiple sectors by automating processes and analyzing large datasets."

    rouge_l = ROUGEEvaluator(variant='rouge-l')
    length_check = LengthEvaluator(min_length=10, max_length=30, unit="words")
    readability = ReadabilityEvaluator(metric='flesch_reading_ease')

    pipeline = WeightedAverageEvaluator([
        (rouge_l, 0.5),
        (length_check, 0.2),
        (readability, 0.3)
    ])

    score = pipeline.evaluate(summary, reference)
    metrics = pipeline.get_metrics(summary, reference)
    print(f"  Overall quality: {score:.3f}")
    print(f"  Component scores:")
    for name, data in metrics['individual_scores'].items():
        print(f"    {name}: {data['score']:.3f} (weight: {data['weight']:.1f})")
    print()

    # Use Case 2: Content Moderation
    print("5.2 Content Moderation Pipeline")
    print("-" * 40)
    comment = "This is a helpful comment with constructive feedback."

    toxicity = ToxicityEvaluator(backend="simple")
    sentiment = SentimentEvaluator(backend="simple")
    length = LengthEvaluator(min_length=5, max_length=500, unit="words")

    moderation = CascadingEvaluator([
        (toxicity, 0.7),     # Must be non-toxic (score >= 0.7)
        (length, 0.5),       # Must meet length requirements
        (sentiment, 0.3)     # Sentiment check (optional)
    ], stop_on_failure=True)

    score = moderation.evaluate(comment, "")
    metrics = moderation.get_metrics(comment, "")
    print(f"  Moderation result: {'APPROVED' if score > 0.5 else 'REJECTED'}")
    print(f"  All checks passed: {metrics['all_passed']}")
    print()

    # Use Case 3: A/B Testing Prompts
    print("5.3 A/B Testing Prompt Variants")
    print("-" * 40)
    reference = "Explain quantum computing simply."

    variant_a = "Quantum computing uses quantum bits that can be 0, 1, or both at once."
    variant_b = "Unlike classical computers using bits, quantum computers use qubits that exist in superposition."

    # Multi-dimensional evaluation
    bleu = BLEUEvaluator(max_n=4)
    readability = ReadabilityEvaluator(metric='flesch_kincaid_grade')
    length = LengthEvaluator(target_length=15, unit="words")

    composite = WeightedAverageEvaluator([
        (bleu, 0.4),
        (readability, 0.3),
        (length, 0.3)
    ])

    score_a = composite.evaluate(variant_a, reference)
    score_b = composite.evaluate(variant_b, reference)

    print(f"  Variant A: {score_a:.3f}")
    print(f"  Variant B: {score_b:.3f}")
    print(f"  Winner: {'Variant A' if score_a > score_b else 'Variant B'}")
    print()

    # Use Case 4: JSON API Response Validation
    print("5.4 JSON API Response Validation")
    print("-" * 40)
    response = '{"status": "success", "data": {"user_id": 123, "name": "John"}}'

    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "data": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "number"},
                    "name": {"type": "string"}
                },
                "required": ["user_id", "name"]
            }
        },
        "required": ["status", "data"]
    }

    json_validator = JSONSchemaEvaluator(schema=schema)
    score = json_validator.evaluate(response, "")
    metrics = json_validator.get_metrics(response, "", context={'schema': schema})

    print(f"  Validation result: {'VALID' if score == 1.0 else 'INVALID'}")
    print(f"  Is valid JSON: {metrics.get('is_valid_json', False)}")
    print(f"  Schema valid: {metrics.get('schema_valid', False)}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all demos"""
    print("\n" + "+" * 80)
    print("  ADVANCED EVALUATORS DEMO")
    print("  Comprehensive examples of all evaluator plugins")
    print("+" * 80)

    try:
        demo_llm_evaluators()
    except Exception as e:
        print(f"Error in LLM evaluators demo: {e}")

    try:
        demo_nlp_metrics()
    except Exception as e:
        print(f"Error in NLP metrics demo: {e}")

    try:
        demo_custom_metrics()
    except Exception as e:
        print(f"Error in custom metrics demo: {e}")

    try:
        demo_composite_evaluators()
    except Exception as e:
        print(f"Error in composite evaluators demo: {e}")

    try:
        demo_real_world_use_cases()
    except Exception as e:
        print(f"Error in real-world use cases demo: {e}")

    print("\n" + "+" * 80)
    print("  DEMO COMPLETE")
    print("+" * 80 + "\n")


if __name__ == "__main__":
    main()

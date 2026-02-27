"""
Shared Test Fixtures for Model Extension Evaluation Framework
==============================================================

Provides common fixtures for:
- Retrieval evaluation tests
- Calibration evaluation tests
- Learning effectiveness tests
- End-to-end integration tests

Author: HoloLoom Architecture Team
Date: December 2025
"""

import pytest
import random
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass

# Import eval types
from hololoom.model_extension.eval.retrieval_metrics import (
    RetrievalQuery,
    RetrievalResult,
    RetrievalEvaluator,
)
from hololoom.model_extension.eval.calibration_metrics import (
    Prediction,
    CalibrationEvaluator,
)
from hololoom.model_extension.eval.learning_metrics import (
    LearningInteraction,
    ThompsonSamplingState,
    LearningEvaluator,
)


# =============================================================================
# Random Seed for Reproducibility
# =============================================================================

@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility across all tests."""
    random.seed(42)
    yield


# =============================================================================
# Retrieval Fixtures
# =============================================================================

@pytest.fixture
def perfect_retrieval_query() -> RetrievalQuery:
    """Query with perfect retrieval (all relevant docs in top-k)."""
    return RetrievalQuery(
        query_id="perfect_q1",
        query_text="What is Thompson Sampling?",
        retrieved=[
            RetrievalResult(doc_id="doc1", score=0.95),
            RetrievalResult(doc_id="doc2", score=0.90),
            RetrievalResult(doc_id="doc3", score=0.85),
            RetrievalResult(doc_id="doc4", score=0.80),
            RetrievalResult(doc_id="doc5", score=0.75),
        ],
        relevant_doc_ids={"doc1", "doc2", "doc3"},
        relevance_grades={"doc1": 1.0, "doc2": 0.8, "doc3": 0.6},
    )


@pytest.fixture
def poor_retrieval_query() -> RetrievalQuery:
    """Query with poor retrieval (relevant docs not in top-k)."""
    return RetrievalQuery(
        query_id="poor_q1",
        query_text="Explain multi-armed bandits",
        retrieved=[
            RetrievalResult(doc_id="irrelevant1", score=0.90),
            RetrievalResult(doc_id="irrelevant2", score=0.85),
            RetrievalResult(doc_id="irrelevant3", score=0.80),
            RetrievalResult(doc_id="relevant1", score=0.75),  # Relevant but ranked 4th
            RetrievalResult(doc_id="irrelevant4", score=0.70),
        ],
        relevant_doc_ids={"relevant1", "relevant2", "relevant3"},
        relevance_grades={"relevant1": 1.0, "relevant2": 0.8, "relevant3": 0.6},
    )


@pytest.fixture
def mixed_retrieval_queries() -> List[RetrievalQuery]:
    """Mix of good and poor retrieval queries."""
    queries = []

    # Good retrieval
    for i in range(5):
        queries.append(RetrievalQuery(
            query_id=f"good_q{i}",
            query_text=f"Good query {i}",
            retrieved=[
                RetrievalResult(doc_id=f"rel_{i}_{j}", score=0.9 - j * 0.1)
                for j in range(5)
            ],
            relevant_doc_ids={f"rel_{i}_0", f"rel_{i}_1"},
        ))

    # Poor retrieval
    for i in range(5):
        queries.append(RetrievalQuery(
            query_id=f"poor_q{i}",
            query_text=f"Poor query {i}",
            retrieved=[
                RetrievalResult(doc_id=f"irrel_{i}_{j}", score=0.9 - j * 0.1)
                for j in range(5)
            ],
            relevant_doc_ids={f"rel_{i}_0", f"rel_{i}_1"},  # Not in retrieved
        ))

    return queries


@pytest.fixture
def retrieval_evaluator() -> RetrievalEvaluator:
    """Standard retrieval evaluator with default k values."""
    return RetrievalEvaluator(k_values=[1, 3, 5, 10])


# =============================================================================
# Calibration Fixtures
# =============================================================================

@pytest.fixture
def well_calibrated_predictions() -> List[Prediction]:
    """Predictions from a well-calibrated model."""
    predictions = []

    # For each confidence bin, accuracy should match
    for conf_bin in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for _ in range(20):
            # Add noise but keep average accuracy ≈ confidence
            is_correct = random.random() < conf_bin
            conf = conf_bin + random.uniform(-0.05, 0.05)
            conf = max(0.0, min(1.0, conf))
            predictions.append(Prediction(
                confidence=conf,
                correct=is_correct,
            ))

    return predictions


@pytest.fixture
def overconfident_predictions() -> List[Prediction]:
    """Predictions from an overconfident model."""
    predictions = []

    # High confidence but lower accuracy
    for _ in range(50):
        conf = random.uniform(0.8, 0.95)  # High confidence
        is_correct = random.random() < 0.5  # But only 50% accurate
        predictions.append(Prediction(
            confidence=conf,
            correct=is_correct,
        ))

    # Low confidence with reasonable accuracy
    for _ in range(50):
        conf = random.uniform(0.2, 0.4)
        is_correct = random.random() < conf
        predictions.append(Prediction(
            confidence=conf,
            correct=is_correct,
        ))

    return predictions


@pytest.fixture
def underconfident_predictions() -> List[Prediction]:
    """Predictions from an underconfident model."""
    predictions = []

    # Low confidence but high accuracy
    for _ in range(50):
        conf = random.uniform(0.2, 0.4)  # Low confidence
        is_correct = random.random() < 0.8  # But 80% accurate
        predictions.append(Prediction(
            confidence=conf,
            correct=is_correct,
        ))

    # High confidence with reasonable accuracy
    for _ in range(50):
        conf = random.uniform(0.7, 0.9)
        is_correct = random.random() < conf
        predictions.append(Prediction(
            confidence=conf,
            correct=is_correct,
        ))

    return predictions


@pytest.fixture
def calibration_evaluator() -> CalibrationEvaluator:
    """Standard calibration evaluator with 10 bins."""
    return CalibrationEvaluator(num_bins=10)


# =============================================================================
# Learning Fixtures
# =============================================================================

@pytest.fixture
def improving_interactions() -> List[LearningInteraction]:
    """Interactions showing improvement over time."""
    interactions = []

    for i in range(200):
        # Quality improves over time
        base_quality = 0.4 + 0.3 * (i / 200)  # 0.4 → 0.7
        noise = random.uniform(-0.1, 0.1)
        quality = max(0.0, min(1.0, base_quality + noise))

        interactions.append(LearningInteraction(
            query_id=f"q_{i}",
            quality_score=quality,
            confidence=quality + random.uniform(-0.05, 0.05),
            patterns_discovered=1 if i % 20 == 0 else 0,
            timestamp=float(i),
        ))

    return interactions


@pytest.fixture
def stagnant_interactions() -> List[LearningInteraction]:
    """Interactions showing no improvement (stagnant learning)."""
    interactions = []

    for i in range(200):
        # Quality stays constant
        quality = 0.5 + random.uniform(-0.1, 0.1)

        interactions.append(LearningInteraction(
            query_id=f"q_{i}",
            quality_score=quality,
            confidence=quality,
            patterns_discovered=0,
            timestamp=float(i),
        ))

    return interactions


@pytest.fixture
def degrading_interactions() -> List[LearningInteraction]:
    """Interactions showing degradation (catastrophic forgetting)."""
    interactions = []

    for i in range(200):
        # Quality degrades over time
        base_quality = 0.8 - 0.4 * (i / 200)  # 0.8 → 0.4
        noise = random.uniform(-0.1, 0.1)
        quality = max(0.0, min(1.0, base_quality + noise))

        interactions.append(LearningInteraction(
            query_id=f"q_{i}",
            quality_score=quality,
            confidence=quality,
            patterns_discovered=0,
            timestamp=float(i),
        ))

    return interactions


@pytest.fixture
def thompson_sampling_states() -> List[ThompsonSamplingState]:
    """Thompson Sampling states for multiple arms."""
    return [
        ThompsonSamplingState(arm_id="arm1", alpha=10, beta=5, pulls=15),  # Good arm
        ThompsonSamplingState(arm_id="arm2", alpha=5, beta=10, pulls=15),  # Bad arm
        ThompsonSamplingState(arm_id="arm3", alpha=7, beta=7, pulls=14),   # Medium arm
    ]


@pytest.fixture
def learning_evaluator() -> LearningEvaluator:
    """Standard learning evaluator."""
    return LearningEvaluator(window_size=50, target_quality_percentile=0.9)


# =============================================================================
# Golden Dataset Fixtures
# =============================================================================

@pytest.fixture
def golden_retrieval_dataset() -> List[Dict]:
    """Golden dataset for retrieval evaluation."""
    return [
        {
            "query": "What is Thompson Sampling?",
            "relevant_docs": ["ts_intro", "ts_tutorial", "bayesian_bandits"],
            "relevance_grades": {"ts_intro": 1.0, "ts_tutorial": 0.8, "bayesian_bandits": 0.5},
        },
        {
            "query": "Explain multi-armed bandits",
            "relevant_docs": ["mab_overview", "mab_algorithms", "exploration_exploitation"],
            "relevance_grades": {"mab_overview": 1.0, "mab_algorithms": 0.9, "exploration_exploitation": 0.6},
        },
        {
            "query": "UCB algorithm explanation",
            "relevant_docs": ["ucb_paper", "ucb_tutorial", "confidence_bounds"],
            "relevance_grades": {"ucb_paper": 1.0, "ucb_tutorial": 0.7, "confidence_bounds": 0.5},
        },
    ]


@pytest.fixture
def golden_qa_pairs() -> List[Dict]:
    """Golden Q&A pairs for quality evaluation."""
    return [
        {
            "query": "What is the time complexity of Thompson Sampling?",
            "expected_answer_contains": ["O(1)", "constant", "sample"],
            "min_confidence": 0.7,
        },
        {
            "query": "How does Thompson Sampling balance exploration and exploitation?",
            "expected_answer_contains": ["posterior", "sample", "uncertainty"],
            "min_confidence": 0.6,
        },
    ]


# =============================================================================
# Utility Functions for Tests
# =============================================================================

def generate_retrieval_queries(
    n_queries: int,
    n_retrieved: int = 10,
    n_relevant: int = 3,
    quality: str = "mixed",
) -> List[RetrievalQuery]:
    """
    Generate synthetic retrieval queries.

    Args:
        n_queries: Number of queries to generate
        n_retrieved: Number of retrieved docs per query
        n_relevant: Number of relevant docs per query
        quality: "good", "poor", or "mixed"
    """
    queries = []

    for i in range(n_queries):
        # Determine if this query has good or poor retrieval
        if quality == "good":
            is_good = True
        elif quality == "poor":
            is_good = False
        else:
            is_good = random.random() > 0.5

        # Generate relevant doc IDs
        relevant_ids = {f"rel_{i}_{j}" for j in range(n_relevant)}

        # Generate retrieved docs
        if is_good:
            # Good retrieval: relevant docs first
            retrieved = [
                RetrievalResult(doc_id=f"rel_{i}_{j}", score=0.9 - j * 0.1)
                for j in range(min(n_relevant, n_retrieved))
            ]
            # Pad with irrelevant
            for j in range(len(retrieved), n_retrieved):
                retrieved.append(
                    RetrievalResult(doc_id=f"irrel_{i}_{j}", score=0.5 - j * 0.05)
                )
        else:
            # Poor retrieval: irrelevant docs first
            retrieved = [
                RetrievalResult(doc_id=f"irrel_{i}_{j}", score=0.9 - j * 0.1)
                for j in range(n_retrieved)
            ]

        queries.append(RetrievalQuery(
            query_id=f"q_{i}",
            query_text=f"Query {i}",
            retrieved=retrieved,
            relevant_doc_ids=relevant_ids,
        ))

    return queries


def generate_predictions(
    n_predictions: int,
    calibration: str = "well",
) -> List[Prediction]:
    """
    Generate synthetic predictions.

    Args:
        n_predictions: Number of predictions to generate
        calibration: "well", "over", or "under"
    """
    predictions = []

    for _ in range(n_predictions):
        conf = random.random()

        if calibration == "well":
            # Well-calibrated: accuracy ≈ confidence
            is_correct = random.random() < conf
        elif calibration == "over":
            # Overconfident: accuracy < confidence
            is_correct = random.random() < conf * 0.6
        else:
            # Underconfident: accuracy > confidence
            is_correct = random.random() < min(conf * 1.5, 1.0)

        predictions.append(Prediction(confidence=conf, correct=is_correct))

    return predictions

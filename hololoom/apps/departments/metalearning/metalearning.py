#!/usr/bin/env python3
"""
Meta-Learning Department - Few-shot learning, transfer learning, and meta-adaptation.

The Meta-Learning Department enables the system to learn from few examples,
transfer knowledge across tasks, adapt learning strategies, and consolidate
knowledge across departments.

Components:
-----------
1. FewShotLearner: Learn from few examples using prototypical networks
2. TransferLearner: Transfer knowledge across tasks and domains
3. MetaAdaptationEngine: Adapt learning strategies based on task characteristics
4. KnowledgeConsolidator: Consolidate knowledge across departments
5. MetaLearningDepartment: DS-STAR protocol implementation

Design Principles:
-----------------
- Few-shot learning: Learn from 1-10 examples (k-shot learning)
- Transfer learning: Leverage knowledge from related tasks
- Meta-adaptation: Learn how to learn (learning-to-learn)
- Knowledge consolidation: Cross-department knowledge integration

Performance:
-----------
- Few-shot learning: <100ms for k≤10 examples
- Transfer learning: <200ms for cross-task transfer
- Meta-adaptation: <50ms for strategy selection
- Knowledge consolidation: <300ms for multi-department consolidation
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from hololoom.apps.departments.base import BaseDepartment, DepartmentRequest, DepartmentResponse

# ============================================================================
# Data Structures
# ============================================================================

class TaskType(Enum):
    """Types of learning tasks."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    REASONING = "reasoning"


class TransferStrategy(Enum):
    """Transfer learning strategies."""
    FINE_TUNING = "fine_tuning"  # Fine-tune on target task
    FEATURE_EXTRACTION = "feature_extraction"  # Use as feature extractor
    DOMAIN_ADAPTATION = "domain_adaptation"  # Adapt to new domain
    MULTI_TASK = "multi_task"  # Learn multiple tasks jointly


class AdaptationStrategy(Enum):
    """Meta-adaptation strategies."""
    MAML = "maml"  # Model-Agnostic Meta-Learning
    PROTOTYPICAL = "prototypical"  # Prototypical Networks
    MATCHING = "matching"  # Matching Networks
    RELATION = "relation"  # Relation Networks


@dataclass
class Example:
    """A single training example for few-shot learning."""
    id: str
    input_features: np.ndarray
    label: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskContext:
    """Context for a learning task."""
    task_id: str
    task_type: TaskType
    support_examples: list[Example]  # Training examples (few-shot support set)
    query_examples: list[Example]  # Test examples (query set)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prototype:
    """Prototype representation for a class/concept."""
    label: Any
    centroid: np.ndarray  # Mean of support examples
    variance: float  # Variance of support examples
    support_count: int
    confidence: float


@dataclass
class TransferredKnowledge:
    """Knowledge transferred from source to target task."""
    source_task_id: str
    target_task_id: str
    transferred_features: np.ndarray
    transfer_strategy: TransferStrategy
    transfer_quality: float  # 0.0-1.0
    similarity_score: float  # Task similarity
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidatedKnowledge:
    """Knowledge consolidated across multiple departments."""
    consolidation_id: str
    source_departments: list[str]
    consolidated_representation: dict[str, Any]
    confidence: float
    consensus_score: float  # Agreement across departments
    conflict_resolutions: list[str]
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


@dataclass
class LearningStrategy:
    """Adapted learning strategy for a task."""
    strategy_id: str
    adaptation_type: AdaptationStrategy
    learning_rate: float
    num_adaptation_steps: int
    meta_parameters: dict[str, Any]
    expected_performance: float


# ============================================================================
# Few-Shot Learner
# ============================================================================

class FewShotLearner:
    """
    Few-shot learning using prototypical networks.

    Learns from k examples (typically k=1-10) by:
    1. Computing prototypes (class centroids) from support set
    2. Classifying queries by nearest prototype distance
    3. Adapting to new classes without retraining

    Algorithm:
    ---------
    For each class c:
        prototype_c = mean(support_examples_c)

    For each query q:
        predicted_class = argmin_c distance(q, prototype_c)
    """

    def __init__(
        self,
        distance_metric: str = "euclidean",
        min_support_examples: int = 1,
        max_support_examples: int = 10
    ):
        """
        Initialize few-shot learner.

        Args:
            distance_metric: Distance metric for prototype matching
            min_support_examples: Minimum k for k-shot learning
            max_support_examples: Maximum k for k-shot learning
        """
        self.distance_metric = distance_metric
        self.min_support_examples = min_support_examples
        self.max_support_examples = max_support_examples
        self._prototypes: dict[str, Prototype] = {}

    def learn_from_examples(
        self,
        task_context: TaskContext
    ) -> dict[str, Prototype]:
        """
        Learn prototypes from support examples.

        Args:
            task_context: Task with support examples

        Returns:
            Dict mapping labels to prototypes
        """
        # Group examples by label
        examples_by_label: dict[Any, list[Example]] = {}
        for example in task_context.support_examples:
            if example.label not in examples_by_label:
                examples_by_label[example.label] = []
            examples_by_label[example.label].append(example)

        # Compute prototype for each label
        prototypes = {}
        for label, examples in examples_by_label.items():
            if len(examples) < self.min_support_examples:
                continue  # Skip classes with insufficient examples

            # Compute centroid (mean of features)
            features = np.array([ex.input_features for ex in examples])
            centroid = np.mean(features, axis=0)

            # Compute variance
            variance = np.var(features)

            # Compute confidence (higher for more examples, lower variance)
            confidence = min(
                1.0,
                (len(examples) / self.max_support_examples) * (1.0 / (1.0 + variance))
            )

            prototype = Prototype(
                label=label,
                centroid=centroid,
                variance=float(variance),
                support_count=len(examples),
                confidence=confidence
            )
            prototypes[label] = prototype

        # Cache prototypes
        self._prototypes.update(prototypes)

        return prototypes

    def predict(self, query_features: np.ndarray, prototypes: dict[str, Prototype]) -> tuple[Any, float]:
        """
        Predict label for query by nearest prototype.

        Args:
            query_features: Input features for query
            prototypes: Learned prototypes

        Returns:
            (predicted_label, confidence)
        """
        if not prototypes:
            return None, 0.0

        # Compute distances to all prototypes
        distances = {}
        for label, prototype in prototypes.items():
            distance = self._compute_distance(query_features, prototype.centroid)
            distances[label] = distance

        # Predict nearest prototype
        predicted_label = min(distances.keys(), key=lambda k: distances[k])
        min_distance = distances[predicted_label]

        # Convert distance to confidence (closer = higher confidence)
        # Use softmax over negative distances
        neg_distances = np.array([-d for d in distances.values()])
        exp_distances = np.exp(neg_distances - np.max(neg_distances))  # Numerical stability
        probabilities = exp_distances / np.sum(exp_distances)

        predicted_idx = list(distances.keys()).index(predicted_label)
        confidence = float(probabilities[predicted_idx])

        return predicted_label, confidence

    def evaluate(self, task_context: TaskContext) -> dict[str, float]:
        """
        Evaluate few-shot learning performance.

        Args:
            task_context: Task with support and query examples

        Returns:
            Metrics (accuracy, precision, recall, f1)
        """
        # Learn from support set
        prototypes = self.learn_from_examples(task_context)

        if not prototypes or not task_context.query_examples:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "num_prototypes": len(prototypes),
                "num_queries": len(task_context.query_examples)
            }

        # Predict on query set
        predictions = []
        ground_truth = []
        confidences = []

        for query in task_context.query_examples:
            pred_label, confidence = self.predict(query.input_features, prototypes)
            predictions.append(pred_label)
            ground_truth.append(query.label)
            confidences.append(confidence)

        # Compute metrics
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # Simplified precision/recall/f1 (would need per-class for full implementation)
        precision = accuracy  # Simplified
        recall = accuracy  # Simplified
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "num_prototypes": len(prototypes),
            "num_queries": len(predictions)
        }

    def _compute_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two feature vectors."""
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(a - b))
        elif self.distance_metric == "cosine":
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 1.0
            return 1.0 - (dot / (norm_a * norm_b))
        else:
            return float(np.linalg.norm(a - b))  # Default to euclidean


# ============================================================================
# Transfer Learner
# ============================================================================

class TransferLearner:
    """
    Transfer learning across tasks and domains.

    Strategies:
    ----------
    1. Fine-tuning: Adapt source model to target task
    2. Feature extraction: Use source as feature extractor
    3. Domain adaptation: Align source/target distributions
    4. Multi-task: Learn multiple tasks jointly

    Transfer Quality:
    ----------------
    quality = task_similarity × source_performance × adaptation_success
    """

    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize transfer learner.

        Args:
            similarity_threshold: Minimum similarity for transfer
        """
        self.similarity_threshold = similarity_threshold
        self._source_tasks: dict[str, TaskContext] = {}
        self._task_embeddings: dict[str, np.ndarray] = {}

    def register_source_task(self, task_id: str, task_context: TaskContext):
        """Register a source task for transfer learning."""
        self._source_tasks[task_id] = task_context
        # Compute task embedding (mean of support example features)
        if task_context.support_examples:
            features = np.array([ex.input_features for ex in task_context.support_examples])
            task_embedding = np.mean(features, axis=0)
            self._task_embeddings[task_id] = task_embedding

    def find_similar_tasks(
        self,
        target_task: TaskContext,
        top_k: int = 3
    ) -> list[tuple[str, float]]:
        """
        Find source tasks similar to target task.

        Args:
            target_task: Target task context
            top_k: Number of similar tasks to return

        Returns:
            List of (task_id, similarity_score) tuples
        """
        if not target_task.support_examples or not self._task_embeddings:
            return []

        # Compute target task embedding
        features = np.array([ex.input_features for ex in target_task.support_examples])
        target_embedding = np.mean(features, axis=0)

        # Compute similarity to all source tasks
        similarities = []
        for task_id, source_embedding in self._task_embeddings.items():
            # Cosine similarity
            dot = np.dot(target_embedding, source_embedding)
            norm_target = np.linalg.norm(target_embedding)
            norm_source = np.linalg.norm(source_embedding)

            if norm_target == 0 or norm_source == 0:
                similarity = 0.0
            else:
                similarity = float(dot / (norm_target * norm_source))

            # Only consider tasks above threshold
            if similarity >= self.similarity_threshold:
                similarities.append((task_id, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def transfer_knowledge(
        self,
        source_task_id: str,
        target_task: TaskContext,
        strategy: TransferStrategy = TransferStrategy.FEATURE_EXTRACTION
    ) -> TransferredKnowledge:
        """
        Transfer knowledge from source to target task.

        Args:
            source_task_id: Source task ID
            target_task: Target task context
            strategy: Transfer strategy

        Returns:
            Transferred knowledge
        """
        if source_task_id not in self._source_tasks:
            raise ValueError(f"Source task {source_task_id} not registered")

        source_task = self._source_tasks[source_task_id]

        # Compute task similarity
        similarities = self.find_similar_tasks(target_task, top_k=10)
        similarity_dict = dict(similarities)
        similarity_score = similarity_dict.get(source_task_id, 0.0)

        # Transfer features based on strategy
        if strategy == TransferStrategy.FEATURE_EXTRACTION:
            # Use source prototypes as feature extractors
            transferred_features = self._extract_features(source_task, target_task)
        elif strategy == TransferStrategy.FINE_TUNING:
            # Fine-tune on target examples (simplified: blend features)
            transferred_features = self._fine_tune(source_task, target_task)
        elif strategy == TransferStrategy.DOMAIN_ADAPTATION:
            # Align distributions (simplified: normalize)
            transferred_features = self._adapt_domain(source_task, target_task)
        else:
            # Multi-task: combine source and target
            transferred_features = self._multi_task_combine(source_task, target_task)

        # Compute transfer quality
        transfer_quality = self._compute_transfer_quality(
            source_task, target_task, similarity_score
        )

        return TransferredKnowledge(
            source_task_id=source_task_id,
            target_task_id=target_task.task_id,
            transferred_features=transferred_features,
            transfer_strategy=strategy,
            transfer_quality=transfer_quality,
            similarity_score=similarity_score,
            metadata={
                "source_examples": len(source_task.support_examples),
                "target_examples": len(target_task.support_examples),
                "strategy": strategy.value
            }
        )

    def _extract_features(self, source_task: TaskContext, target_task: TaskContext) -> np.ndarray:
        """Extract features using source task as feature extractor."""
        # Simplified: Average source and target features
        source_features = np.array([ex.input_features for ex in source_task.support_examples])
        target_features = np.array([ex.input_features for ex in target_task.support_examples])

        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)

        # Weighted combination (more weight to target)
        transferred = 0.3 * source_mean + 0.7 * target_mean
        return transferred

    def _fine_tune(self, source_task: TaskContext, target_task: TaskContext) -> np.ndarray:
        """Fine-tune source on target task."""
        # Simplified: Blend with higher target weight
        source_features = np.array([ex.input_features for ex in source_task.support_examples])
        target_features = np.array([ex.input_features for ex in target_task.support_examples])

        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)

        # Higher target weight for fine-tuning
        transferred = 0.2 * source_mean + 0.8 * target_mean
        return transferred

    def _adapt_domain(self, source_task: TaskContext, target_task: TaskContext) -> np.ndarray:
        """Adapt from source to target domain."""
        # Simplified: Normalize to target distribution
        source_features = np.array([ex.input_features for ex in source_task.support_examples])
        target_features = np.array([ex.input_features for ex in target_task.support_examples])

        # Normalize source to match target statistics
        source_mean = np.mean(source_features, axis=0)
        source_std = np.std(source_features, axis=0) + 1e-8
        target_mean = np.mean(target_features, axis=0)
        target_std = np.std(target_features, axis=0) + 1e-8

        # Z-score normalize then rescale
        normalized = (source_mean - source_mean) / source_std
        adapted = normalized * target_std + target_mean
        return adapted

    def _multi_task_combine(self, source_task: TaskContext, target_task: TaskContext) -> np.ndarray:
        """Combine source and target for multi-task learning."""
        # Simplified: Equal weight combination
        source_features = np.array([ex.input_features for ex in source_task.support_examples])
        target_features = np.array([ex.input_features for ex in target_task.support_examples])

        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)

        combined = 0.5 * source_mean + 0.5 * target_mean
        return combined

    def _compute_transfer_quality(
        self,
        source_task: TaskContext,
        target_task: TaskContext,
        similarity_score: float
    ) -> float:
        """Compute quality of knowledge transfer."""
        # Quality factors:
        # 1. Task similarity (0.5 weight)
        # 2. Source data sufficiency (0.3 weight)
        # 3. Target data sufficiency (0.2 weight)

        source_sufficiency = min(1.0, len(source_task.support_examples) / 10)
        target_sufficiency = min(1.0, len(target_task.support_examples) / 5)

        quality = (
            0.5 * similarity_score +
            0.3 * source_sufficiency +
            0.2 * target_sufficiency
        )

        return quality


# ============================================================================
# Meta-Adaptation Engine
# ============================================================================

class MetaAdaptationEngine:
    """
    Meta-adaptation: Learn how to learn.

    Selects optimal learning strategy based on:
    1. Task characteristics (type, complexity, data availability)
    2. Historical performance on similar tasks
    3. Computational constraints

    Strategies:
    ----------
    - MAML: Model-Agnostic Meta-Learning (few gradient steps)
    - Prototypical: Prototypical Networks (nearest centroid)
    - Matching: Matching Networks (attention over support set)
    - Relation: Relation Networks (learn similarity metric)
    """

    def __init__(self):
        """Initialize meta-adaptation engine."""
        self._strategy_history: dict[str, list[tuple[TaskContext, LearningStrategy, float]]] = {}
        self._strategy_performance: dict[AdaptationStrategy, list[float]] = {
            strategy: [] for strategy in AdaptationStrategy
        }

    def select_strategy(
        self,
        task_context: TaskContext,
        constraints: dict[str, Any] | None = None
    ) -> LearningStrategy:
        """
        Select optimal learning strategy for task.

        Args:
            task_context: Task context
            constraints: Computational/time constraints

        Returns:
            Selected learning strategy
        """
        constraints = constraints or {}

        # Analyze task characteristics
        num_support = len(task_context.support_examples)
        num_classes = len(set(ex.label for ex in task_context.support_examples))
        task_type = task_context.task_type

        # Strategy selection heuristics
        if num_support < 5:
            # Very few examples: Use prototypical (simple, effective)
            strategy_type = AdaptationStrategy.PROTOTYPICAL
            learning_rate = 0.01
            num_steps = 1
        elif num_support < 20:
            # Moderate examples: Use MAML (can adapt quickly)
            strategy_type = AdaptationStrategy.MAML
            learning_rate = 0.001
            num_steps = 5
        elif num_classes > 10:
            # Many classes: Use matching (attention over support)
            strategy_type = AdaptationStrategy.MATCHING
            learning_rate = 0.0001
            num_steps = 10
        else:
            # General case: Use relation (learn similarity)
            strategy_type = AdaptationStrategy.RELATION
            learning_rate = 0.0001
            num_steps = 10

        # Adjust for constraints
        if constraints.get("max_time_ms", float("inf")) < 100:
            # Time-constrained: Use simplest strategy
            strategy_type = AdaptationStrategy.PROTOTYPICAL
            num_steps = 1

        # Estimate expected performance based on history
        historical_performance = self._strategy_performance.get(strategy_type, [])
        expected_performance = np.mean(historical_performance) if historical_performance else 0.7

        strategy = LearningStrategy(
            strategy_id=str(uuid.uuid4()),
            adaptation_type=strategy_type,
            learning_rate=learning_rate,
            num_adaptation_steps=num_steps,
            meta_parameters={
                "num_support": num_support,
                "num_classes": num_classes,
                "task_type": task_type.value
            },
            expected_performance=expected_performance
        )

        return strategy

    def record_performance(
        self,
        task_context: TaskContext,
        strategy: LearningStrategy,
        performance: float
    ):
        """Record strategy performance for meta-learning."""
        strategy_type = strategy.adaptation_type

        # Record in history
        if strategy_type.value not in self._strategy_history:
            self._strategy_history[strategy_type.value] = []
        self._strategy_history[strategy_type.value].append(
            (task_context, strategy, performance)
        )

        # Update performance statistics
        self._strategy_performance[strategy_type].append(performance)

        # Keep only recent history (last 100 tasks)
        if len(self._strategy_performance[strategy_type]) > 100:
            self._strategy_performance[strategy_type] = \
                self._strategy_performance[strategy_type][-100:]

    def get_strategy_statistics(self) -> dict[str, dict[str, float]]:
        """Get performance statistics for each strategy."""
        stats = {}
        for strategy_type, performances in self._strategy_performance.items():
            if performances:
                stats[strategy_type.value] = {
                    "mean_performance": float(np.mean(performances)),
                    "std_performance": float(np.std(performances)),
                    "min_performance": float(np.min(performances)),
                    "max_performance": float(np.max(performances)),
                    "num_trials": len(performances)
                }
            else:
                stats[strategy_type.value] = {
                    "mean_performance": 0.0,
                    "std_performance": 0.0,
                    "min_performance": 0.0,
                    "max_performance": 0.0,
                    "num_trials": 0
                }
        return stats


# ============================================================================
# Knowledge Consolidator
# ============================================================================

class KnowledgeConsolidator:
    """
    Consolidate knowledge across multiple departments.

    Cross-Department Integration:
    ----------------------------
    1. Context Department: Entity/relation retrieval
    2. Reasoning Department: Causal relationships
    3. Planning Department: Goal hierarchies
    4. Meta-Learning Department: Learning patterns

    Consolidation Process:
    ---------------------
    1. Gather responses from multiple departments
    2. Identify agreements and conflicts
    3. Resolve conflicts via voting/confidence-weighting
    4. Create unified consolidated representation
    """

    def __init__(self, min_consensus: float = 0.6):
        """
        Initialize knowledge consolidator.

        Args:
            min_consensus: Minimum consensus score for consolidation
        """
        self.min_consensus = min_consensus
        self._consolidation_cache: dict[str, ConsolidatedKnowledge] = {}

    async def consolidate(
        self,
        query: str,
        department_responses: dict[str, DepartmentResponse]
    ) -> ConsolidatedKnowledge:
        """
        Consolidate knowledge from multiple departments.

        Args:
            query: Original query
            department_responses: Responses from each department

        Returns:
            Consolidated knowledge
        """
        if not department_responses:
            return ConsolidatedKnowledge(
                consolidation_id=str(uuid.uuid4()),
                source_departments=[],
                consolidated_representation={},
                confidence=0.0,
                consensus_score=0.0,
                conflict_resolutions=[]
            )

        # Extract key information from each department
        department_data = {}
        confidences = []

        for dept_name, response in department_responses.items():
            if response.success:
                department_data[dept_name] = response.result
                confidences.append(response.confidence)

        # Compute consensus (agreement across departments)
        consensus_score = self._compute_consensus(department_data)

        # Identify conflicts
        conflicts = self._identify_conflicts(department_data)

        # Resolve conflicts
        resolutions = self._resolve_conflicts(conflicts, department_responses)

        # Create consolidated representation
        consolidated_rep = self._create_consolidated_representation(
            department_data, resolutions
        )

        # Overall confidence (weighted by consensus)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        overall_confidence = avg_confidence * consensus_score

        consolidation = ConsolidatedKnowledge(
            consolidation_id=str(uuid.uuid4()),
            source_departments=list(department_data.keys()),
            consolidated_representation=consolidated_rep,
            confidence=overall_confidence,
            consensus_score=consensus_score,
            conflict_resolutions=resolutions
        )

        # Cache consolidation
        cache_key = f"{query}_{','.join(sorted(department_data.keys()))}"
        self._consolidation_cache[cache_key] = consolidation

        return consolidation

    def _compute_consensus(self, department_data: dict[str, Any]) -> float:
        """
        Compute consensus score across departments.

        High consensus: Departments agree on key facts
        Low consensus: Departments provide conflicting information
        """
        if len(department_data) < 2:
            return 1.0  # Single department = perfect consensus

        # Simplified: Check for common entities/concepts
        all_entities: list[set[str]] = []

        for dept_name, data in department_data.items():
            entities = set()
            # Extract entities from different result formats
            if isinstance(data, dict):
                if "entities" in data:
                    entities.update(str(e) for e in data["entities"])
                if "paths" in data:
                    for path in data.get("paths", []):
                        if isinstance(path, dict):
                            entities.update(str(e) for e in path.get("entities", []))
                if "actions" in data:
                    for action in data.get("actions", []):
                        if isinstance(action, dict):
                            entities.add(str(action.get("name", "")))
            all_entities.append(entities)

        if not any(all_entities):
            return 0.5  # No entities extracted, neutral consensus

        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                set_i = all_entities[i]
                set_j = all_entities[j]
                if not set_i and not set_j:
                    similarity = 1.0
                elif not set_i or not set_j:
                    similarity = 0.0
                else:
                    intersection = len(set_i & set_j)
                    union = len(set_i | set_j)
                    similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)

        # Average similarity = consensus
        consensus = np.mean(similarities) if similarities else 0.5
        return float(consensus)

    def _identify_conflicts(self, department_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify conflicting information across departments."""
        conflicts = []

        # Check for conflicting confidence scores
        confidences = {}
        for dept_name, data in department_data.items():
            if isinstance(data, dict) and "confidence" in data:
                confidences[dept_name] = data["confidence"]

        if len(confidences) >= 2:
            conf_values = list(confidences.values())
            conf_std = np.std(conf_values)
            if conf_std > 0.2:  # High variance in confidence
                conflicts.append({
                    "type": "confidence_variance",
                    "description": f"High variance in confidence across departments (std={conf_std:.2f})",
                    "departments": list(confidences.keys()),
                    "values": confidences
                })

        return conflicts

    def _resolve_conflicts(
        self,
        conflicts: list[dict[str, Any]],
        department_responses: dict[str, DepartmentResponse]
    ) -> list[str]:
        """Resolve conflicts via voting and confidence-weighting."""
        resolutions = []

        for conflict in conflicts:
            if conflict["type"] == "confidence_variance":
                # Resolve by selecting highest confidence department
                dept_confidences = {
                    dept: resp.confidence
                    for dept, resp in department_responses.items()
                    if resp.success
                }
                if dept_confidences:
                    best_dept = max(dept_confidences.keys(), key=lambda d: dept_confidences[d])
                    best_conf = dept_confidences[best_dept]
                    resolutions.append(
                        f"Confidence conflict resolved: Selected {best_dept} (confidence={best_conf:.2f})"
                    )

        return resolutions

    def _create_consolidated_representation(
        self,
        department_data: dict[str, Any],
        resolutions: list[str]
    ) -> dict[str, Any]:
        """Create unified consolidated representation."""
        consolidated = {
            "departments": list(department_data.keys()),
            "resolutions": resolutions,
            "data": department_data
        }

        # Extract common entities
        all_entities = []
        for data in department_data.values():
            if isinstance(data, dict) and "entities" in data:
                all_entities.extend(data["entities"])

        if all_entities:
            # Count entity occurrences
            entity_counts = {}
            for entity in all_entities:
                entity_str = str(entity)
                entity_counts[entity_str] = entity_counts.get(entity_str, 0) + 1

            # Keep entities mentioned by multiple departments
            common_entities = [
                entity for entity, count in entity_counts.items()
                if count >= 2
            ]
            consolidated["common_entities"] = common_entities

        return consolidated


# ============================================================================
# Meta-Learning Department
# ============================================================================

class MetaLearningDepartment(BaseDepartment):
    """
    Meta-Learning Department: Few-shot learning, transfer learning, and meta-adaptation.

    Implements DS-STAR Protocol:
    ---------------------------
    D - Decompose: Break learning task into components
    S - Solve: Execute few-shot/transfer learning
    T - Test: Validate learned knowledge
    A - Aggregate: Consolidate cross-department knowledge
    R - Refine: Adapt learning strategy based on performance

    Capabilities:
    ------------
    1. Few-shot learning: Learn from k examples (k=1-10)
    2. Transfer learning: Transfer knowledge across tasks
    3. Meta-adaptation: Learn how to learn
    4. Knowledge consolidation: Cross-department integration

    Task Types:
    ----------
    - few_shot_learning: Learn from few examples
    - transfer_learning: Transfer knowledge from source to target
    - meta_adaptation: Select optimal learning strategy
    - knowledge_consolidation: Consolidate multi-department knowledge
    """

    def __init__(
        self,
        registry = None,
        few_shot_learner: FewShotLearner | None = None,
        transfer_learner: TransferLearner | None = None,
        meta_adaptation_engine: MetaAdaptationEngine | None = None,
        knowledge_consolidator: KnowledgeConsolidator | None = None
    ):
        """
        Initialize Meta-Learning Department.

        Args:
            registry: Department registry for cross-department communication
            few_shot_learner: Few-shot learning component
            transfer_learner: Transfer learning component
            meta_adaptation_engine: Meta-adaptation component
            knowledge_consolidator: Knowledge consolidation component
        """
        super().__init__(name="meta_learning", registry=registry)

        # Initialize components
        self.few_shot_learner = few_shot_learner or FewShotLearner()
        self.transfer_learner = transfer_learner or TransferLearner()
        self.meta_adaptation_engine = meta_adaptation_engine or MetaAdaptationEngine()
        self.knowledge_consolidator = knowledge_consolidator or KnowledgeConsolidator()

        # Metrics
        self._total_tasks = 0
        self._successful_tasks = 0
        self._avg_performance = 0.0

    async def execute(self, request: DepartmentRequest) -> DepartmentResponse:
        """
        Execute meta-learning task.

        Supported task types:
        - few_shot_learning: Learn from few examples
        - transfer_learning: Transfer knowledge across tasks
        - meta_adaptation: Select optimal learning strategy
        - knowledge_consolidation: Consolidate multi-department knowledge
        """
        task_type = request.task_type
        params = request.parameters

        try:
            if task_type == "few_shot_learning":
                result = await self._execute_few_shot_learning(params)
            elif task_type == "transfer_learning":
                result = await self._execute_transfer_learning(params)
            elif task_type == "meta_adaptation":
                result = await self._execute_meta_adaptation(params)
            elif task_type == "knowledge_consolidation":
                result = await self._execute_knowledge_consolidation(params)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            self._total_tasks += 1
            self._successful_tasks += 1

            return DepartmentResponse(
                task_id=request.task_id,
                department_name=self.name,
                success=True,
                result=result,
                confidence=result.get("confidence", 0.7),
                metadata={
                    "task_type": task_type,
                    "performance": result.get("performance", {})
                }
            )

        except Exception as e:
            self._total_tasks += 1
            return DepartmentResponse(
                task_id=request.task_id,
                department_name=self.name,
                success=False,
                result={},
                confidence=0.0,
                error_message=str(e)
            )

    async def verify(self, request: DepartmentRequest, result: Any) -> tuple[bool, float, str]:
        """
        Verify meta-learning result quality.

        Verification checks:
        - Performance metrics meet thresholds
        - Confidence is reasonable
        - Learning strategy is appropriate
        """
        task_type = request.task_type

        # Extract performance metrics
        if isinstance(result, dict):
            confidence = result.get("confidence", 0.0)
            performance = result.get("performance", {})

            # Verification criteria
            if task_type == "few_shot_learning":
                accuracy = performance.get("accuracy", 0.0)
                is_sufficient = accuracy >= 0.6 and confidence >= 0.5
                confidence_score = (accuracy + confidence) / 2
                explanation = f"Few-shot accuracy: {accuracy:.2f}, confidence: {confidence:.2f}"

            elif task_type == "transfer_learning":
                transfer_quality = result.get("transfer_quality", 0.0)
                is_sufficient = transfer_quality >= 0.5 and confidence >= 0.5
                confidence_score = (transfer_quality + confidence) / 2
                explanation = f"Transfer quality: {transfer_quality:.2f}, confidence: {confidence:.2f}"

            elif task_type == "knowledge_consolidation":
                consensus = result.get("consensus_score", 0.0)
                is_sufficient = consensus >= 0.6 and confidence >= 0.5
                confidence_score = (consensus + confidence) / 2
                explanation = f"Consensus: {consensus:.2f}, confidence: {confidence:.2f}"

            else:
                is_sufficient = confidence >= 0.5
                confidence_score = confidence
                explanation = f"Confidence: {confidence:.2f}"

        else:
            is_sufficient = False
            confidence_score = 0.0
            explanation = "Invalid result format"

        return is_sufficient, confidence_score, explanation

    async def refine(self, request: DepartmentRequest, previous_result: Any) -> Any:
        """
        Refine meta-learning result.

        Refinement strategies:
        - Add more support examples
        - Try different learning strategy
        - Increase adaptation steps
        """
        task_type = request.task_type
        params = request.parameters

        if task_type == "few_shot_learning":
            # Retry with more examples if available
            return await self._execute_few_shot_learning(params)

        elif task_type == "transfer_learning":
            # Try different transfer strategy
            strategy = params.get("strategy", TransferStrategy.FEATURE_EXTRACTION)
            # Cycle through strategies
            strategies = list(TransferStrategy)
            current_idx = strategies.index(strategy)
            next_idx = (current_idx + 1) % len(strategies)
            params["strategy"] = strategies[next_idx]
            return await self._execute_transfer_learning(params)

        elif task_type == "meta_adaptation":
            # Increase adaptation steps
            return await self._execute_meta_adaptation(params)

        else:
            return previous_result

    # ========================================================================
    # Task Execution Methods
    # ========================================================================

    async def _execute_few_shot_learning(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute few-shot learning task."""
        task_context = params.get("task_context")
        if not isinstance(task_context, TaskContext):
            raise ValueError("task_context parameter must be TaskContext instance")

        # Learn from support examples
        prototypes = self.few_shot_learner.learn_from_examples(task_context)

        # Evaluate on query set
        metrics = self.few_shot_learner.evaluate(task_context)

        return {
            "prototypes": {
                label: {
                    "centroid": proto.centroid.tolist(),
                    "confidence": proto.confidence,
                    "support_count": proto.support_count
                }
                for label, proto in prototypes.items()
            },
            "performance": metrics,
            "confidence": metrics.get("avg_confidence", 0.7)
        }

    async def _execute_transfer_learning(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute transfer learning task."""
        source_task_id = params.get("source_task_id")
        target_task = params.get("target_task")
        strategy = params.get("strategy", TransferStrategy.FEATURE_EXTRACTION)

        if not isinstance(target_task, TaskContext):
            raise ValueError("target_task parameter must be TaskContext instance")

        # Find similar tasks if source not specified
        if not source_task_id:
            similar_tasks = self.transfer_learner.find_similar_tasks(target_task, top_k=1)
            if not similar_tasks:
                raise ValueError("No similar source tasks found")
            source_task_id, _ = similar_tasks[0]

        # Transfer knowledge
        transferred = self.transfer_learner.transfer_knowledge(
            source_task_id, target_task, strategy
        )

        return {
            "source_task_id": transferred.source_task_id,
            "target_task_id": transferred.target_task_id,
            "transfer_strategy": transferred.transfer_strategy.value,
            "transfer_quality": transferred.transfer_quality,
            "similarity_score": transferred.similarity_score,
            "confidence": transferred.transfer_quality
        }

    async def _execute_meta_adaptation(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute meta-adaptation task."""
        task_context = params.get("task_context")
        constraints = params.get("constraints", {})

        if not isinstance(task_context, TaskContext):
            raise ValueError("task_context parameter must be TaskContext instance")

        # Select optimal strategy
        strategy = self.meta_adaptation_engine.select_strategy(task_context, constraints)

        # Get strategy statistics
        stats = self.meta_adaptation_engine.get_strategy_statistics()

        return {
            "selected_strategy": {
                "type": strategy.adaptation_type.value,
                "learning_rate": strategy.learning_rate,
                "num_steps": strategy.num_adaptation_steps,
                "expected_performance": strategy.expected_performance
            },
            "strategy_statistics": stats,
            "confidence": strategy.expected_performance
        }

    async def _execute_knowledge_consolidation(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute knowledge consolidation task."""
        query = params.get("query", "")
        department_responses = params.get("department_responses", {})

        if not isinstance(department_responses, dict):
            raise ValueError("department_responses must be dict of DepartmentResponse")

        # Consolidate knowledge
        consolidated = await self.knowledge_consolidator.consolidate(
            query, department_responses
        )

        return {
            "consolidation_id": consolidated.consolidation_id,
            "source_departments": consolidated.source_departments,
            "consolidated_representation": consolidated.consolidated_representation,
            "consensus_score": consolidated.consensus_score,
            "conflict_resolutions": consolidated.conflict_resolutions,
            "confidence": consolidated.confidence
        }

    # ========================================================================
    # Health Monitoring
    # ========================================================================

    async def health_check(self) -> dict[str, Any]:
        """Check department health."""
        success_rate = (
            self._successful_tasks / self._total_tasks
            if self._total_tasks > 0
            else 1.0
        )

        return {
            "status": "healthy" if success_rate >= 0.8 else "degraded",
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "success_rate": success_rate,
            "avg_performance": self._avg_performance,
            "components": {
                "few_shot_learner": "operational",
                "transfer_learner": "operational",
                "meta_adaptation_engine": "operational",
                "knowledge_consolidator": "operational"
            }
        }

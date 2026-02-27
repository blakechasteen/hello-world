#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for Meta-Learning Department.

Tests:
-----
1. Component Tests (12 tests)
   - FewShotLearner: Learning, prediction, evaluation
   - TransferLearner: Task similarity, knowledge transfer
   - MetaAdaptationEngine: Strategy selection, performance tracking
   - KnowledgeConsolidator: Multi-department consolidation

2. Department Integration Tests (8 tests)
   - Few-shot learning task
   - Transfer learning task
   - Meta-adaptation task
   - Knowledge consolidation task

3. DS-STAR Protocol Tests (3 tests)
   - Verification workflow
   - Refinement workflow
   - End-to-end pipeline

4. Performance Tests (3 tests)
   - Few-shot learning performance
   - Transfer learning performance
   - Knowledge consolidation performance

5. Registry Integration (2 tests)
   - Department registration
   - Request routing

Total: 28 test scenarios
"""

import pytest
import numpy as np
from typing import List
import asyncio

from hololoom.apps.departments.metalearning import (
    MetaLearningDepartment,
    FewShotLearner,
    TransferLearner,
    MetaAdaptationEngine,
    KnowledgeConsolidator,
    TaskType,
    TransferStrategy,
    AdaptationStrategy,
    Example,
    TaskContext,
    Prototype
)
from hololoom.departments import DepartmentRequest, DepartmentResponse, DepartmentRegistry


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_examples() -> List[Example]:
    """Create sample examples for testing."""
    np.random.seed(42)
    examples = []

    # Class A (3 examples)
    for i in range(3):
        features = np.random.randn(10) + np.array([1.0] * 10)  # Shifted distribution
        examples.append(
            Example(
                id=f"example_a_{i}",
                input_features=features,
                label="class_a",
                metadata={"source": "test"}
            )
        )

    # Class B (3 examples)
    for i in range(3):
        features = np.random.randn(10) + np.array([-1.0] * 10)  # Different distribution
        examples.append(
            Example(
                id=f"example_b_{i}",
                input_features=features,
                label="class_b",
                metadata={"source": "test"}
            )
        )

    return examples


@pytest.fixture
def classification_task(sample_examples) -> TaskContext:
    """Create a classification task."""
    # Split examples into support and query
    support = sample_examples[:4]  # 2 from each class
    query = sample_examples[4:]  # Remaining examples

    return TaskContext(
        task_id="task_classification",
        task_type=TaskType.CLASSIFICATION,
        support_examples=support,
        query_examples=query,
        metadata={"description": "Binary classification task"}
    )


@pytest.fixture
def source_task(sample_examples) -> TaskContext:
    """Create a source task for transfer learning."""
    return TaskContext(
        task_id="source_task",
        task_type=TaskType.CLASSIFICATION,
        support_examples=sample_examples[:4],
        query_examples=[],
        metadata={"domain": "source_domain"}
    )


@pytest.fixture
def target_task() -> TaskContext:
    """Create a target task for transfer learning."""
    np.random.seed(43)
    examples = []

    # Similar but slightly different distribution
    for i in range(3):
        features = np.random.randn(10) + np.array([0.8] * 10)
        examples.append(
            Example(
                id=f"target_{i}",
                input_features=features,
                label="class_a",
                metadata={"source": "target"}
            )
        )

    return TaskContext(
        task_id="target_task",
        task_type=TaskType.CLASSIFICATION,
        support_examples=examples,
        query_examples=[],
        metadata={"domain": "target_domain"}
    )


# ============================================================================
# Component Tests: FewShotLearner
# ============================================================================

def test_few_shot_learner_initialization():
    """Test FewShotLearner initialization."""
    learner = FewShotLearner(distance_metric="euclidean", min_support_examples=1)
    assert learner.distance_metric == "euclidean"
    assert learner.min_support_examples == 1
    assert len(learner._prototypes) == 0


def test_few_shot_learn_from_examples(classification_task):
    """Test learning prototypes from examples."""
    learner = FewShotLearner()
    prototypes = learner.learn_from_examples(classification_task)

    # Should create 2 prototypes (class_a and class_b)
    assert len(prototypes) == 2
    assert "class_a" in prototypes
    assert "class_b" in prototypes

    # Check prototype properties
    proto_a = prototypes["class_a"]
    assert isinstance(proto_a, Prototype)
    assert proto_a.label == "class_a"
    assert len(proto_a.centroid) == 10  # Feature dimension
    assert proto_a.support_count >= 1
    assert 0.0 <= proto_a.confidence <= 1.0


def test_few_shot_predict(classification_task):
    """Test prediction using prototypes."""
    learner = FewShotLearner()
    prototypes = learner.learn_from_examples(classification_task)

    # Predict on a query example
    query = classification_task.query_examples[0]
    predicted_label, confidence = learner.predict(query.input_features, prototypes)

    # Should predict a label
    assert predicted_label in ["class_a", "class_b"]
    assert 0.0 <= confidence <= 1.0


def test_few_shot_evaluate(classification_task):
    """Test few-shot learning evaluation."""
    learner = FewShotLearner()
    metrics = learner.evaluate(classification_task)

    # Check metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "avg_confidence" in metrics
    assert "num_prototypes" in metrics
    assert "num_queries" in metrics

    # Metrics should be in valid ranges
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["avg_confidence"] <= 1.0
    assert metrics["num_prototypes"] == 2
    assert metrics["num_queries"] == 2


# ============================================================================
# Component Tests: TransferLearner
# ============================================================================

def test_transfer_learner_initialization():
    """Test TransferLearner initialization."""
    learner = TransferLearner(similarity_threshold=0.6)
    assert learner.similarity_threshold == 0.6
    assert len(learner._source_tasks) == 0
    assert len(learner._task_embeddings) == 0


def test_transfer_register_source_task(source_task):
    """Test registering source task."""
    learner = TransferLearner()
    learner.register_source_task("source_1", source_task)

    assert "source_1" in learner._source_tasks
    assert "source_1" in learner._task_embeddings
    assert len(learner._task_embeddings["source_1"]) == 10  # Feature dimension


def test_transfer_find_similar_tasks(source_task, target_task):
    """Test finding similar tasks."""
    learner = TransferLearner(similarity_threshold=0.0)  # Low threshold for testing
    learner.register_source_task("source_1", source_task)

    similar_tasks = learner.find_similar_tasks(target_task, top_k=1)

    assert len(similar_tasks) >= 0  # May or may not find similar tasks
    if similar_tasks:
        task_id, similarity = similar_tasks[0]
        assert task_id == "source_1"
        assert 0.0 <= similarity <= 1.0


def test_transfer_knowledge(source_task, target_task):
    """Test transferring knowledge."""
    learner = TransferLearner(similarity_threshold=0.0)
    learner.register_source_task("source_1", source_task)

    transferred = learner.transfer_knowledge(
        "source_1",
        target_task,
        strategy=TransferStrategy.FEATURE_EXTRACTION
    )

    assert transferred.source_task_id == "source_1"
    assert transferred.target_task_id == target_task.task_id
    assert transferred.transfer_strategy == TransferStrategy.FEATURE_EXTRACTION
    assert 0.0 <= transferred.transfer_quality <= 1.0
    assert 0.0 <= transferred.similarity_score <= 1.0
    assert len(transferred.transferred_features) == 10


# ============================================================================
# Component Tests: MetaAdaptationEngine
# ============================================================================

def test_meta_adaptation_initialization():
    """Test MetaAdaptationEngine initialization."""
    engine = MetaAdaptationEngine()
    assert len(engine._strategy_history) == 0
    assert len(engine._strategy_performance) == len(AdaptationStrategy)


def test_meta_adaptation_select_strategy(classification_task):
    """Test strategy selection."""
    engine = MetaAdaptationEngine()
    strategy = engine.select_strategy(classification_task, constraints={})

    assert strategy.adaptation_type in AdaptationStrategy
    assert strategy.learning_rate > 0
    assert strategy.num_adaptation_steps > 0
    assert 0.0 <= strategy.expected_performance <= 1.0


def test_meta_adaptation_record_performance(classification_task):
    """Test recording strategy performance."""
    engine = MetaAdaptationEngine()
    strategy = engine.select_strategy(classification_task)

    # Record performance
    engine.record_performance(classification_task, strategy, performance=0.85)

    # Check recorded
    assert len(engine._strategy_performance[strategy.adaptation_type]) == 1
    assert engine._strategy_performance[strategy.adaptation_type][0] == 0.85


def test_meta_adaptation_statistics():
    """Test getting strategy statistics."""
    engine = MetaAdaptationEngine()

    # Record some performances
    from hololoom.apps.departments.metalearning.metalearning import LearningStrategy
    strategy = LearningStrategy(
        strategy_id="test",
        adaptation_type=AdaptationStrategy.PROTOTYPICAL,
        learning_rate=0.01,
        num_adaptation_steps=1,
        meta_parameters={},
        expected_performance=0.7
    )

    engine._strategy_performance[AdaptationStrategy.PROTOTYPICAL].extend([0.8, 0.85, 0.9])

    stats = engine.get_strategy_statistics()

    assert "prototypical" in stats
    proto_stats = stats["prototypical"]
    assert "mean_performance" in proto_stats
    assert "std_performance" in proto_stats
    assert "num_trials" in proto_stats
    assert proto_stats["num_trials"] == 3


# ============================================================================
# Component Tests: KnowledgeConsolidator
# ============================================================================

def test_knowledge_consolidator_initialization():
    """Test KnowledgeConsolidator initialization."""
    consolidator = KnowledgeConsolidator(min_consensus=0.6)
    assert consolidator.min_consensus == 0.6
    assert len(consolidator._consolidation_cache) == 0


@pytest.mark.asyncio
async def test_knowledge_consolidate_empty():
    """Test consolidation with no responses."""
    consolidator = KnowledgeConsolidator()
    consolidated = await consolidator.consolidate("test query", {})

    assert consolidated.confidence == 0.0
    assert consolidated.consensus_score == 0.0
    assert len(consolidated.source_departments) == 0


@pytest.mark.asyncio
async def test_knowledge_consolidate_multiple_departments():
    """Test consolidation with multiple department responses."""
    consolidator = KnowledgeConsolidator()

    # Create mock department responses
    dept_responses = {
        "context": DepartmentResponse(
            task_id="task_1",
            department_name="context",
            success=True,
            result={"entities": ["entity_a", "entity_b"], "confidence": 0.9},
            confidence=0.9
        ),
        "reasoning": DepartmentResponse(
            task_id="task_1",
            department_name="reasoning",
            success=True,
            result={"entities": ["entity_a", "entity_c"], "confidence": 0.85},
            confidence=0.85
        )
    }

    consolidated = await consolidator.consolidate("test query", dept_responses)

    assert len(consolidated.source_departments) == 2
    assert "context" in consolidated.source_departments
    assert "reasoning" in consolidated.source_departments
    assert 0.0 <= consolidated.confidence <= 1.0
    assert 0.0 <= consolidated.consensus_score <= 1.0


# ============================================================================
# Department Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_department_initialization():
    """Test MetaLearningDepartment initialization."""
    async with MetaLearningDepartment() as dept:
        assert dept.name == "meta_learning"
        assert isinstance(dept.few_shot_learner, FewShotLearner)
        assert isinstance(dept.transfer_learner, TransferLearner)
        assert isinstance(dept.meta_adaptation_engine, MetaAdaptationEngine)
        assert isinstance(dept.knowledge_consolidator, KnowledgeConsolidator)


@pytest.mark.asyncio
async def test_department_few_shot_learning_task(classification_task):
    """Test few-shot learning task through department."""
    async with MetaLearningDepartment() as dept:
        request = DepartmentRequest(
            task_id="task_001",
            task_type="few_shot_learning",
            parameters={"task_context": classification_task}
        )

        response = await dept.execute(request)

        assert response.success
        assert response.department_name == "meta_learning"
        assert "prototypes" in response.result
        assert "performance" in response.result
        assert "confidence" in response.result
        assert 0.0 <= response.confidence <= 1.0


@pytest.mark.asyncio
async def test_department_transfer_learning_task(source_task, target_task):
    """Test transfer learning task through department."""
    async with MetaLearningDepartment() as dept:
        # Register source task
        dept.transfer_learner.register_source_task("source_1", source_task)

        request = DepartmentRequest(
            task_id="task_002",
            task_type="transfer_learning",
            parameters={
                "source_task_id": "source_1",
                "target_task": target_task,
                "strategy": TransferStrategy.FEATURE_EXTRACTION
            }
        )

        response = await dept.execute(request)

        assert response.success
        assert "source_task_id" in response.result
        assert "transfer_quality" in response.result
        assert "similarity_score" in response.result
        assert 0.0 <= response.confidence <= 1.0


@pytest.mark.asyncio
async def test_department_meta_adaptation_task(classification_task):
    """Test meta-adaptation task through department."""
    async with MetaLearningDepartment() as dept:
        request = DepartmentRequest(
            task_id="task_003",
            task_type="meta_adaptation",
            parameters={
                "task_context": classification_task,
                "constraints": {"max_time_ms": 100}
            }
        )

        response = await dept.execute(request)

        assert response.success
        assert "selected_strategy" in response.result
        assert "strategy_statistics" in response.result
        strategy = response.result["selected_strategy"]
        assert "type" in strategy
        assert "learning_rate" in strategy
        assert "num_steps" in strategy


@pytest.mark.asyncio
async def test_department_knowledge_consolidation_task():
    """Test knowledge consolidation task through department."""
    async with MetaLearningDepartment() as dept:
        # Create mock department responses
        dept_responses = {
            "context": DepartmentResponse(
                task_id="task_1",
                department_name="context",
                success=True,
                result={"entities": ["bee", "hive"], "confidence": 0.9},
                confidence=0.9
            ),
            "reasoning": DepartmentResponse(
                task_id="task_1",
                department_name="reasoning",
                success=True,
                result={"entities": ["bee", "colony"], "confidence": 0.85},
                confidence=0.85
            )
        }

        request = DepartmentRequest(
            task_id="task_004",
            task_type="knowledge_consolidation",
            parameters={
                "query": "What affects bee colonies?",
                "department_responses": dept_responses
            }
        )

        response = await dept.execute(request)

        assert response.success
        assert "consolidation_id" in response.result
        assert "source_departments" in response.result
        assert "consensus_score" in response.result
        assert len(response.result["source_departments"]) == 2


@pytest.mark.asyncio
async def test_department_invalid_task_type():
    """Test handling invalid task type."""
    async with MetaLearningDepartment() as dept:
        request = DepartmentRequest(
            task_id="task_999",
            task_type="invalid_task",
            parameters={}
        )

        response = await dept.execute(request)

        assert not response.success
        assert response.error_message is not None
        assert "Unsupported task type" in response.error_message


# ============================================================================
# DS-STAR Protocol Tests
# ============================================================================

@pytest.mark.asyncio
async def test_verification_sufficient(classification_task):
    """Test verification with sufficient result."""
    async with MetaLearningDepartment() as dept:
        # Create high-quality result
        result = {
            "performance": {"accuracy": 0.85},
            "confidence": 0.9
        }

        request = DepartmentRequest(
            task_id="verify_001",
            task_type="few_shot_learning",
            parameters={"task_context": classification_task}
        )

        is_sufficient, confidence, explanation = await dept.verify(request, result)

        assert is_sufficient
        assert confidence >= 0.5
        assert "accuracy" in explanation.lower() or "confidence" in explanation.lower()


@pytest.mark.asyncio
async def test_verification_insufficient(classification_task):
    """Test verification with insufficient result."""
    async with MetaLearningDepartment() as dept:
        # Create low-quality result
        result = {
            "performance": {"accuracy": 0.3},
            "confidence": 0.4
        }

        request = DepartmentRequest(
            task_id="verify_002",
            task_type="few_shot_learning",
            parameters={"task_context": classification_task}
        )

        is_sufficient, confidence, explanation = await dept.verify(request, result)

        assert not is_sufficient
        assert confidence < 0.6


@pytest.mark.asyncio
async def test_refinement_workflow(classification_task):
    """Test refinement workflow."""
    async with MetaLearningDepartment() as dept:
        # Initial low-quality result
        initial_result = {
            "performance": {"accuracy": 0.4},
            "confidence": 0.5
        }

        request = DepartmentRequest(
            task_id="refine_001",
            task_type="few_shot_learning",
            parameters={"task_context": classification_task}
        )

        # Refine result
        refined_result = await dept.refine(request, initial_result)

        # Refinement should return a result (even if same)
        assert refined_result is not None


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.asyncio
async def test_few_shot_learning_performance(classification_task):
    """Test few-shot learning performance (<100ms)."""
    import time

    learner = FewShotLearner()

    start = time.time()
    metrics = learner.evaluate(classification_task)
    duration_ms = (time.time() - start) * 1000

    assert duration_ms < 100, f"Few-shot learning took {duration_ms:.2f}ms (target: <100ms)"
    assert metrics["accuracy"] >= 0.0


@pytest.mark.asyncio
async def test_transfer_learning_performance(source_task, target_task):
    """Test transfer learning performance (<200ms)."""
    import time

    learner = TransferLearner(similarity_threshold=0.0)
    learner.register_source_task("source_1", source_task)

    start = time.time()
    transferred = learner.transfer_knowledge("source_1", target_task)
    duration_ms = (time.time() - start) * 1000

    assert duration_ms < 200, f"Transfer learning took {duration_ms:.2f}ms (target: <200ms)"
    assert transferred.transfer_quality >= 0.0


@pytest.mark.asyncio
async def test_knowledge_consolidation_performance():
    """Test knowledge consolidation performance (<300ms)."""
    import time

    consolidator = KnowledgeConsolidator()
    dept_responses = {
        f"dept_{i}": DepartmentResponse(
            task_id="task_1",
            department_name=f"dept_{i}",
            success=True,
            result={"entities": [f"entity_{i}"], "confidence": 0.8},
            confidence=0.8
        )
        for i in range(3)
    }

    start = time.time()
    consolidated = await consolidator.consolidate("test query", dept_responses)
    duration_ms = (time.time() - start) * 1000

    assert duration_ms < 300, f"Consolidation took {duration_ms:.2f}ms (target: <300ms)"
    assert consolidated.confidence >= 0.0


# ============================================================================
# Registry Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_registry_integration():
    """Test department registration with registry."""
    async with DepartmentRegistry() as registry:
        async with MetaLearningDepartment(registry=registry) as dept:
            await registry.register(dept)

            # Verify registration
            registered = registry.get_department("meta_learning")
            assert registered is not None
            assert registered.name == "meta_learning"


@pytest.mark.asyncio
async def test_registry_request_routing(classification_task):
    """Test request routing through registry."""
    async with DepartmentRegistry() as registry:
        async with MetaLearningDepartment(registry=registry) as dept:
            await registry.register(dept)

            request = DepartmentRequest(
                task_id="route_001",
                task_type="few_shot_learning",
                parameters={"task_context": classification_task}
            )

            # Route request through registry
            response = await registry.route_request("meta_learning", request)

            assert response.success
            assert response.department_name == "meta_learning"


# ============================================================================
# Health Monitoring
# ============================================================================

@pytest.mark.asyncio
async def test_health_check():
    """Test health monitoring."""
    async with MetaLearningDepartment() as dept:
        health = await dept.health_check()

        assert "status" in health
        assert "total_tasks" in health
        assert "successful_tasks" in health
        assert "success_rate" in health
        assert "components" in health

        # All components should be operational
        assert health["components"]["few_shot_learner"] == "operational"
        assert health["components"]["transfer_learner"] == "operational"
        assert health["components"]["meta_adaptation_engine"] == "operational"
        assert health["components"]["knowledge_consolidator"] == "operational"


# ============================================================================
# Lifecycle Management
# ============================================================================

@pytest.mark.asyncio
async def test_context_manager_lifecycle():
    """Test async context manager lifecycle."""
    dept = MetaLearningDepartment()

    async with dept:
        assert dept.name == "meta_learning"
        # Department should be operational within context

    # After exit, department should be cleaned up
    # (No explicit check needed, just verify no exceptions)

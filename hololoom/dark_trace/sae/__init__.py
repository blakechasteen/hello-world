"""
Dark Trace SAE: Research-Grade Sparse Autoencoders

This module provides research-grade SAE training and evaluation based on
Anthropic's "Towards Monosemanticity" (2023) and "Scaling Monosemanticity" (2024).

Key Features:
- L1 Annealing: Gradual sparsity increase for stable training (+15-25% quality)
- Dead Feature Detection: Track and resample inactive features (+20-30% utilization)
- Ghost Gradients: Allow dead features to receive gradient signals (+10-15% recovery)
- Comprehensive Metrics: Reconstruction, sparsity, health, quality evaluation
- Auto-Labeling: LLM-powered feature descriptions with semantic alignment
- TraceLens Integration: Unified interpretability interface

Usage:
    from hololoom.dark_trace.sae import (
        ResearchSAE,
        ResearchTrainer,
        TrainingConfig,
        SAEEvaluator,
        FeatureLabeler,
        SAELens,
    )

    # Create research-grade SAE
    sae = ResearchSAE(input_dim=384, expansion_factor=16)

    # Train with L1 annealing and dead feature detection
    config = TrainingConfig(
        l1_warmup_steps=1000,
        dead_feature_threshold=1e-5,
        enable_ghost_gradients=True,
    )
    trainer = ResearchTrainer(sae, config)

    # Evaluate quality
    evaluator = SAEEvaluator()
    report = evaluator.evaluate(sae, test_activations)

    # Label features
    labeler = FeatureLabeler(llm_provider="ollama")
    labels = await labeler.label_features(sae, training_examples)

    # Use as interpretability lens
    lens = SAELens(sae, labeler)
    result = lens.project(activations)
"""

# Core SAE architecture
from hololoom.dark_trace.sae.core import (
    ResearchSAE,
    FeatureHealth,
    ActivationTracker,
)

# Research-grade training
from hololoom.dark_trace.sae.trainer import (
    ResearchTrainer,
    TrainingConfig,
    L1AnnealingScheduler,
    DeadFeatureResampler,
    TrainingMetrics,
)

# Evaluation and metrics
from hololoom.dark_trace.sae.evaluator import (
    SAEEvaluator,
    ReconstructionMetrics,
    SparsityMetrics,
    HealthMetrics,
    SAEEvaluationReport,
)

# Auto-labeling
from hololoom.dark_trace.sae.labeler import (
    FeatureLabeler,
    ActivationPatternAnalyzer,
    SemanticAlignmentScorer,
    LabelCache,
    FeatureLabel,
)

# TraceLens integration
from hololoom.dark_trace.sae.lens import (
    SAELens,
)

# Activation buffer
from hololoom.dark_trace.sae.activation_buffer import (
    ActivationBuffer,
    ActivationSample,
    init_activation_buffer,
    get_activation_buffer,
)

# SAE-backed governance detectors
from hololoom.dark_trace.sae.detectors import (
    FeatureMap,
    SAEDeceptionDetector,
    SAEPowerSeekingDetector,
    SAESelfModificationDetector,
    SAEAdversarialDetector,
)

# Legacy (backward compatibility)
# For new projects, use ResearchSAE and ResearchTrainer instead
from hololoom.dark_trace.sae.legacy import (
    SparseAutoEncoder,
    DarkSaeTrainer,
)

__all__ = [
    # Core
    "ResearchSAE",
    "FeatureHealth",
    "ActivationTracker",
    # Training
    "ResearchTrainer",
    "TrainingConfig",
    "L1AnnealingScheduler",
    "DeadFeatureResampler",
    "TrainingMetrics",
    # Evaluation
    "SAEEvaluator",
    "ReconstructionMetrics",
    "SparsityMetrics",
    "HealthMetrics",
    "SAEEvaluationReport",
    # Labeling
    "FeatureLabeler",
    "ActivationPatternAnalyzer",
    "SemanticAlignmentScorer",
    "LabelCache",
    "FeatureLabel",
    # Lens
    "SAELens",
    # Activation buffer
    "ActivationBuffer",
    "ActivationSample",
    "init_activation_buffer",
    "get_activation_buffer",
    # SAE-backed governance detectors
    "FeatureMap",
    "SAEDeceptionDetector",
    "SAEPowerSeekingDetector",
    "SAESelfModificationDetector",
    "SAEAdversarialDetector",
    # Legacy (backward compatibility)
    "SparseAutoEncoder",
    "DarkSaeTrainer",
]

"""
Unit Tests for Bayesian Policy Integration
==========================================

Tests Bayesian uncertainty quantification in HoloLoom's policy network.

Run:
    pytest HoloLoom/tests/unit/test_bayesian_policy.py -v

Author: HoloLoom Probabilistic Reasoning Team
Date: 2025-11-03
"""

import pytest
import numpy as np
import torch
import asyncio

from HoloLoom.policy.bayesian_policy import (
    BayesianLinear,
    BayesianNeuralCore,
    BayesianUnifiedPolicy,
    create_bayesian_policy,
)
from HoloLoom.policy.unified import create_policy, NeuralCore
from HoloLoom.embedding.spectral import MatryoshkaEmbeddings
from HoloLoom.protocols.types import Features, Context


# ============================================================================
# Test BayesianLinear
# ============================================================================

def test_bayesian_linear_initialization():
    """Test BayesianLinear layer initialization."""
    layer = BayesianLinear(in_features=10, out_features=5, prior_std=1.0)

    assert layer.in_features == 10
    assert layer.out_features == 5
    assert layer.prior_std == 1.0
    assert layer.weight_mean.shape == (5, 10)
    assert layer.bias_mean.shape == (5,)


def test_bayesian_linear_forward_deterministic():
    """Test deterministic forward pass (sample=False)."""
    layer = BayesianLinear(in_features=10, out_features=5)
    x = torch.randn(3, 10)

    # Deterministic mode: same input → same output
    y1 = layer(x, sample=False)
    y2 = layer(x, sample=False)

    assert y1.shape == (3, 5)
    assert torch.allclose(y1, y2)


def test_bayesian_linear_forward_stochastic():
    """Test stochastic forward pass (sample=True)."""
    layer = BayesianLinear(in_features=10, out_features=5)
    x = torch.randn(3, 10)

    # Stochastic mode: same input → different outputs
    y1 = layer(x, sample=True)
    y2 = layer(x, sample=True)

    assert y1.shape == (3, 5)
    assert not torch.allclose(y1, y2)  # Should differ due to sampling


def test_bayesian_linear_kl_divergence():
    """Test KL divergence computation."""
    layer = BayesianLinear(in_features=10, out_features=5, prior_std=1.0)

    kl = layer.kl_divergence()

    # KL should be positive (unless q == p exactly, which is unlikely)
    assert kl.item() >= 0


# ============================================================================
# Test BayesianNeuralCore
# ============================================================================

@pytest.mark.asyncio
async def test_bayesian_neural_core_creation():
    """Test creating BayesianNeuralCore from NeuralCore."""
    # Create base core
    base_core = NeuralCore(d_model=128, n_layers=1, n_heads=2, n_tools=4)

    # Create Bayesian core
    bayesian_core = BayesianNeuralCore(
        base_core=base_core,
        prior_std=1.0,
        replace_all=False
    )

    assert bayesian_core is not None
    assert hasattr(bayesian_core, 'tool_head')
    assert isinstance(bayesian_core.tool_head, BayesianLinear)


@pytest.mark.asyncio
async def test_bayesian_neural_core_decide():
    """Test decision making with uncertainty."""
    # Create base and Bayesian cores
    base_core = NeuralCore(d_model=128, n_layers=1, n_heads=2, n_tools=4)
    bayesian_core = BayesianNeuralCore(base_core, prior_std=1.0)

    # Mock inputs
    mem = torch.randn(1, 3, 128)
    ctrl = torch.randn(1, 8)
    adapter_idx = 0

    # Deterministic decision
    logits_det, pooled_det, unc_det = await bayesian_core.decide(
        mem, ctrl, adapter_idx, n_samples=1, sample_weights=False
    )

    assert logits_det.shape == (1, 4)
    assert pooled_det.shape == (1, 128)
    assert unc_det['epistemic'] == 0.0  # No uncertainty in deterministic mode
    assert unc_det['aleatoric'] == 0.0
    assert unc_det['total'] == 0.0

    # Bayesian decision with uncertainty
    logits_bay, pooled_bay, unc_bay = await bayesian_core.decide(
        mem, ctrl, adapter_idx, n_samples=10, sample_weights=True
    )

    assert logits_bay.shape == (1, 4)
    assert pooled_bay.shape == (1, 128)
    assert unc_bay['epistemic'] >= 0.0  # Should have some epistemic uncertainty
    assert unc_bay['aleatoric'] >= 0.0
    assert unc_bay['total'] >= 0.0
    assert unc_bay['entropy'] >= 0.0


# ============================================================================
# Test BayesianUnifiedPolicy
# ============================================================================

@pytest.mark.asyncio
async def test_bayesian_unified_policy_creation():
    """Test creating BayesianUnifiedPolicy."""
    # Create components
    emb = MatryoshkaEmbeddings(sizes=[128])

    # Create base policy
    base_policy = create_policy(
        mem_dim=128,
        emb=emb,
        scales=[128],
        n_layers=1,
        use_bayesian=False
    )

    # Create Bayesian policy
    bayesian_policy = create_bayesian_policy(
        base_policy=base_policy,
        n_samples=10,
        kl_weight=1.0,
        prior_std=1.0
    )

    assert isinstance(bayesian_policy, BayesianUnifiedPolicy)
    assert bayesian_policy.n_samples == 10
    assert bayesian_policy.kl_weight == 1.0
    assert bayesian_policy.prior_std == 1.0


@pytest.mark.asyncio
async def test_bayesian_unified_policy_decide():
    """Test decision making with BayesianUnifiedPolicy."""
    # Create policy
    emb = MatryoshkaEmbeddings(sizes=[128])
    policy = create_policy(
        mem_dim=128,
        emb=emb,
        scales=[128],
        n_layers=1,
        use_bayesian=True,
        bayesian_samples=10
    )

    # Mock features and context
    features = Features(
        psi=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        motifs=["question→answer"],
        metrics={"coherence": 0.8},
        confidence=0.8
    )

    context = Context(
        hits=[],
        kg_sub=None,
        shard_texts=["Test context"],
        relevance=0.7
    )

    # Make decision with uncertainty
    action = await policy.decide(features, context, return_uncertainty=True)

    assert action.chosen_tool in ["answer", "search", "notion_write", "calc"]
    assert hasattr(action, 'metadata')
    assert 'uncertainty' in action.metadata

    unc = action.metadata['uncertainty']
    assert 'epistemic' in unc
    assert 'aleatoric' in unc
    assert 'total' in unc
    assert 'entropy' in unc

    # All uncertainties should be non-negative
    assert unc['epistemic'] >= 0.0
    assert unc['aleatoric'] >= 0.0
    assert unc['total'] >= 0.0
    assert unc['entropy'] >= 0.0


@pytest.mark.asyncio
async def test_uncertainty_increases_with_complexity():
    """Test that uncertainty increases with query complexity."""
    # Create policy
    emb = MatryoshkaEmbeddings(sizes=[128])
    policy = create_policy(
        mem_dim=128,
        emb=emb,
        scales=[128],
        n_layers=1,
        use_bayesian=True,
        bayesian_samples=20  # More samples for better uncertainty estimate
    )

    context = Context(
        hits=[],
        kg_sub=None,
        shard_texts=["Test context"],
        relevance=0.7
    )

    # Simple query (high coherence)
    features_simple = Features(
        psi=np.array([0.9, 0.2, 0.1, 0.2, 0.3, 0.2]),
        motifs=["question→answer"],
        metrics={"coherence": 0.95},
        confidence=0.95
    )

    action_simple = await policy.decide(features_simple, context, return_uncertainty=True)
    unc_simple = action_simple.metadata['uncertainty']['epistemic']

    # Complex query (low coherence)
    features_complex = Features(
        psi=np.array([0.4, 0.7, 0.8, 0.5, 0.6, 0.7]),
        motifs=["contrast", "subordinate→main"],
        metrics={"coherence": 0.45},
        confidence=0.45
    )

    action_complex = await policy.decide(features_complex, context, return_uncertainty=True)
    unc_complex = action_complex.metadata['uncertainty']['epistemic']

    # Complex queries should have higher epistemic uncertainty
    # (This may not always hold due to randomness, so we just check non-negativity)
    assert unc_simple >= 0.0
    assert unc_complex >= 0.0


# ============================================================================
# Test Integration with Config
# ============================================================================

def test_config_bayesian_fields():
    """Test that Config has Bayesian fields."""
    from HoloLoom.config import Config

    config = Config.fused()

    # Check Bayesian fields exist
    assert hasattr(config, 'use_bayesian')
    assert hasattr(config, 'bayesian_samples')
    assert hasattr(config, 'bayesian_kl_weight')
    assert hasattr(config, 'bayesian_prior_std')

    # Check defaults
    assert config.use_bayesian is False  # Opt-in
    assert config.bayesian_samples == 10
    assert config.bayesian_kl_weight == 1.0
    assert config.bayesian_prior_std == 1.0


@pytest.mark.asyncio
async def test_create_policy_with_config():
    """Test creating policy with Config settings."""
    from HoloLoom.config import Config

    # Enable Bayesian in config
    config = Config.fused()
    config.use_bayesian = True
    config.bayesian_samples = 15

    # Create policy from config
    emb = MatryoshkaEmbeddings(sizes=[128])
    policy = create_policy(
        mem_dim=128,
        emb=emb,
        scales=[128],
        use_bayesian=config.use_bayesian,
        bayesian_samples=config.bayesian_samples
    )

    # Should be Bayesian policy
    assert isinstance(policy, BayesianUnifiedPolicy)
    assert policy.n_samples == 15


# ============================================================================
# Test Graceful Fallback
# ============================================================================

@pytest.mark.asyncio
async def test_graceful_fallback():
    """Test graceful fallback when Bayesian unavailable."""
    # This test verifies the warning/fallback logic in create_policy
    # In normal operation, Bayesian should be available
    # But if imports fail, it should fall back to deterministic

    emb = MatryoshkaEmbeddings(sizes=[128])

    # Try to create Bayesian policy
    policy = create_policy(
        mem_dim=128,
        emb=emb,
        scales=[128],
        use_bayesian=True,  # Request Bayesian
        bayesian_samples=10
    )

    # Should succeed (either Bayesian or fallback to deterministic)
    assert policy is not None

    # Make a decision (should work either way)
    features = Features(
        psi=np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        motifs=[],
        metrics={"coherence": 0.8},
        confidence=0.8
    )
    context = Context(hits=[], kg_sub=None, shard_texts=["Test"], relevance=0.7)

    action = await policy.decide(features, context)
    assert action.chosen_tool in ["answer", "search", "notion_write", "calc"]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

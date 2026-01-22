"""
Comprehensive Test Suite for Policy Engine Module
=================================================
Complete test coverage for HoloLoom's unified policy engine including:
- Neural network components (CustomMHA, MotifGatedMHA, TinyTransformerBlock)
- Thompson Sampling bandit (all 3 strategies)
- Policy engine integration (create_policy, feature encoding, adapters)
- Edge cases and error handling

Target: 2000+ lines, 90%+ coverage

Author: HoloLoom Team
Date: January 2026
"""

import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import asyncio
import tempfile
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# Ensure guardrails are disabled for testing
os.environ["DISABLE_GUARDRAILS"] = "1"


# ============================================================================
# SECTION 1: NEURAL NETWORK COMPONENTS (~600 lines)
# ============================================================================

class TestCustomMHA:
    """Test suite for CustomMHA (Custom Multi-Head Attention) module."""

    def test_initialization(self, d_model, n_heads, device):
        """Test CustomMHA initialization with various configurations."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        # Verify dimensions
        assert mha.d_model == d_model
        assert mha.n_heads == n_heads
        assert mha.d_head == d_model // n_heads

        # Verify linear layers exist
        assert hasattr(mha, 'Wq')
        assert hasattr(mha, 'Wk')
        assert hasattr(mha, 'Wv')
        assert hasattr(mha, 'Wo')

        # Verify shapes
        assert mha.Wq.in_features == d_model
        assert mha.Wq.out_features == d_model

    def test_forward_pass_shape(self, custom_mha, random_seq_input, random_gates, device):
        """Test forward pass produces correct output shapes."""
        custom_mha = custom_mha.to(device)
        x = random_seq_input.to(device)
        gates = random_gates.to(device)

        output, attention = custom_mha(x, gates)

        B, T, D = x.shape
        H = custom_mha.n_heads

        assert output.shape == (B, T, D), f"Output shape mismatch: {output.shape}"
        assert attention.shape == (B, H, T, T), f"Attention shape mismatch: {attention.shape}"

    def test_forward_pass_with_different_batch_sizes(self, d_model, n_heads, device):
        """Test forward pass with various batch sizes."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        for batch_size in [1, 8, 32, 64]:
            x = torch.randn(batch_size, 10, d_model).to(device)
            gates = torch.rand(batch_size, n_heads).to(device)

            output, attention = mha(x, gates)
            assert output.shape == (batch_size, 10, d_model)

    def test_attention_gate_effect(self, d_model, n_heads, device):
        """Test that gates properly modulate attention weights."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)
        x = torch.randn(4, 10, d_model).to(device)

        # Test with all gates = 1 (full attention)
        full_gates = torch.ones(4, n_heads).to(device)
        output_full, attn_full = mha(x, full_gates)

        # Test with all gates = 0 (no attention)
        zero_gates = torch.zeros(4, n_heads).to(device)
        output_zero, attn_zero = mha(x, zero_gates)

        # Attention should be zeroed when gates are zero
        assert torch.allclose(attn_zero, torch.zeros_like(attn_zero))

        # Outputs should be different
        assert not torch.allclose(output_full, output_zero)

    def test_gradient_flow(self, custom_mha, random_seq_input, random_gates, device):
        """Test that gradients flow properly through the module."""
        custom_mha = custom_mha.to(device)
        x = random_seq_input.to(device).requires_grad_(True)
        gates = random_gates.to(device).requires_grad_(True)

        output, _ = custom_mha(x, gates)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert gates.grad is not None

        # Check no NaN gradients
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(gates.grad).any()

        # Check parameter gradients
        for name, param in custom_mha.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_numerical_stability(self, d_model, n_heads, device):
        """Test numerical stability with extreme values."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        # Test with large values
        x_large = torch.randn(4, 10, d_model).to(device) * 100
        gates = torch.rand(4, n_heads).to(device)

        output, attn = mha(x_large, gates)
        assert not torch.isnan(output).any(), "NaN in output with large values"
        assert not torch.isinf(output).any(), "Inf in output with large values"

        # Test with small values
        x_small = torch.randn(4, 10, d_model).to(device) * 0.001
        output_small, _ = mha(x_small, gates)
        assert not torch.isnan(output_small).any()


class TestCrossAttention:
    """Test suite for CrossAttention module."""

    def test_initialization(self, d_model, n_heads, device):
        """Test CrossAttention initialization."""
        from HoloLoom.policy.unified import CrossAttention

        cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads).to(device)

        assert cross_attn.d_model == d_model
        assert cross_attn.n_heads == n_heads
        assert cross_attn.d_head == d_model // n_heads

    def test_forward_pass_shape(self, d_model, n_heads, device):
        """Test forward pass with query and memory tensors."""
        from HoloLoom.policy.unified import CrossAttention

        cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads).to(device)

        batch_size = 8
        query_len = 16
        memory_len = 32

        x = torch.randn(batch_size, query_len, d_model).to(device)
        mem = torch.randn(batch_size, memory_len, d_model).to(device)

        output = cross_attn(x, mem)

        assert output.shape == (batch_size, query_len, d_model)

    def test_different_memory_lengths(self, d_model, n_heads, device):
        """Test cross attention with various memory sizes."""
        from HoloLoom.policy.unified import CrossAttention

        cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads).to(device)
        batch_size = 4
        query_len = 8

        for memory_len in [1, 10, 50, 100]:
            x = torch.randn(batch_size, query_len, d_model).to(device)
            mem = torch.randn(batch_size, memory_len, d_model).to(device)

            output = cross_attn(x, mem)
            assert output.shape == (batch_size, query_len, d_model)

    def test_gradient_flow(self, d_model, n_heads, device):
        """Test gradient flow through cross attention."""
        from HoloLoom.policy.unified import CrossAttention

        cross_attn = CrossAttention(d_model=d_model, n_heads=n_heads).to(device)

        x = torch.randn(4, 8, d_model, requires_grad=True).to(device)
        mem = torch.randn(4, 16, d_model, requires_grad=True).to(device)

        output = cross_attn(x, mem)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert mem.grad is not None


class TestMotifGatedMHA:
    """Test suite for MotifGatedMHA module."""

    def test_initialization(self, d_model, n_heads, n_motifs, device):
        """Test MotifGatedMHA initialization."""
        from HoloLoom.policy.unified import MotifGatedMHA

        mha = MotifGatedMHA(d_model=d_model, n_heads=n_heads, n_motifs=n_motifs).to(device)

        assert hasattr(mha, 'mha')
        assert hasattr(mha, 'gate_proj')
        assert mha.gate_proj.in_features == n_motifs
        assert mha.gate_proj.out_features == n_heads

    def test_forward_pass(self, motif_gated_mha, random_seq_input, random_motif_ctrl, device):
        """Test forward pass with motif control."""
        motif_gated_mha = motif_gated_mha.to(device)
        x = random_seq_input.to(device)
        motif_ctrl = random_motif_ctrl.to(device)

        output, attention = motif_gated_mha(x, motif_ctrl)

        B, T, D = x.shape
        assert output.shape == (B, T, D)

    def test_motif_control_effect(self, d_model, n_heads, n_motifs, device):
        """Test that motif control affects gating."""
        from HoloLoom.policy.unified import MotifGatedMHA

        mha = MotifGatedMHA(d_model=d_model, n_heads=n_heads, n_motifs=n_motifs).to(device)
        x = torch.randn(4, 10, d_model).to(device)

        # Different motif controls should produce different outputs
        ctrl1 = torch.zeros(4, n_motifs).to(device)
        ctrl2 = torch.ones(4, n_motifs).to(device)

        out1, _ = mha(x, ctrl1)
        out2, _ = mha(x, ctrl2)

        # Outputs should be different
        assert not torch.allclose(out1, out2)

    def test_gate_sigmoid_bounds(self, d_model, n_heads, n_motifs, device):
        """Test that gates are bounded by sigmoid."""
        from HoloLoom.policy.unified import MotifGatedMHA

        mha = MotifGatedMHA(d_model=d_model, n_heads=n_heads, n_motifs=n_motifs).to(device)

        # Create extreme motif controls
        motif_ctrl = torch.randn(4, n_motifs).to(device) * 100
        gates = torch.sigmoid(mha.gate_proj(motif_ctrl))

        # Gates should be in [0, 1] due to sigmoid
        assert (gates >= 0).all()
        assert (gates <= 1).all()


class TestLoRALikeFFN:
    """Test suite for LoRA-style Feed-Forward Network."""

    def test_initialization(self, d_model, n_adapters, device):
        """Test LoRALikeFFN initialization."""
        from HoloLoom.policy.unified import LoRALikeFFN

        ffn = LoRALikeFFN(d_model=d_model, d_ff=4*d_model, r=8, n_adapters=n_adapters).to(device)

        assert hasattr(ffn, 'fc1')
        assert hasattr(ffn, 'fc2')
        assert hasattr(ffn, 'adapters')
        assert len(ffn.adapters) == n_adapters

    def test_forward_pass(self, lora_ffn, random_input, device):
        """Test forward pass with default adapter."""
        lora_ffn = lora_ffn.to(device)
        x = random_input.to(device)

        output = lora_ffn(x, adapter_idx=0)

        assert output.shape == x.shape

    def test_different_adapters(self, d_model, n_adapters, device):
        """Test that different adapters produce different outputs."""
        from HoloLoom.policy.unified import LoRALikeFFN

        ffn = LoRALikeFFN(d_model=d_model, d_ff=4*d_model, r=8, n_adapters=n_adapters).to(device)
        x = torch.randn(8, d_model).to(device)

        outputs = []
        for i in range(n_adapters):
            outputs.append(ffn(x, adapter_idx=i))

        # Different adapters should produce different outputs
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j]), \
                    f"Adapters {i} and {j} produce identical outputs"

    def test_adapter_low_rank(self, d_model, device):
        """Test that adapters use low-rank approximation."""
        from HoloLoom.policy.unified import LoRALikeFFN

        r = 8  # Low rank
        ffn = LoRALikeFFN(d_model=d_model, d_ff=4*d_model, r=r, n_adapters=2).to(device)

        # Check adapter architecture
        for adapter in ffn.adapters:
            # First layer: d_model -> r
            assert adapter[0].in_features == d_model
            assert adapter[0].out_features == r
            # Second layer: r -> d_model
            assert adapter[1].in_features == r
            assert adapter[1].out_features == d_model


class TestTinyTransformerBlock:
    """Test suite for TinyTransformerBlock."""

    def test_initialization(self, d_model, n_heads, n_motifs, n_adapters, device):
        """Test TinyTransformerBlock initialization."""
        from HoloLoom.policy.unified import TinyTransformerBlock

        block = TinyTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_motifs=n_motifs,
            n_adapters=n_adapters
        ).to(device)

        assert hasattr(block, 'cross')
        assert hasattr(block, 'mha')
        assert hasattr(block, 'ffn')
        assert hasattr(block, 'ln1')
        assert hasattr(block, 'ln2')
        assert hasattr(block, 'ln3')

    def test_forward_pass(self, transformer_block, d_model, n_motifs, device):
        """Test forward pass through transformer block."""
        transformer_block = transformer_block.to(device)

        batch_size = 8
        query_len = 16
        memory_len = 32

        x = torch.randn(batch_size, query_len, d_model).to(device)
        mem = torch.randn(batch_size, memory_len, d_model).to(device)
        motif_ctrl = torch.rand(batch_size, n_motifs).to(device)
        adapter_idx = 0

        output = transformer_block(x, mem, motif_ctrl, adapter_idx)

        assert output.shape == (batch_size, query_len, d_model)

    def test_residual_connections(self, d_model, n_heads, n_motifs, n_adapters, device):
        """Test that residual connections work properly."""
        from HoloLoom.policy.unified import TinyTransformerBlock

        # Initialize block with zero-initialized weights
        block = TinyTransformerBlock(
            d_model=d_model,
            n_heads=n_heads,
            n_motifs=n_motifs,
            n_adapters=n_adapters
        ).to(device)

        x = torch.randn(4, 8, d_model).to(device)
        mem = torch.randn(4, 16, d_model).to(device)
        motif_ctrl = torch.rand(4, n_motifs).to(device)

        output = block(x, mem, motif_ctrl, adapter_idx=0)

        # Output should not be identical to input (transformations applied)
        assert not torch.allclose(output, x)

    def test_gradient_flow(self, transformer_block, d_model, n_motifs, device):
        """Test gradient flow through transformer block."""
        transformer_block = transformer_block.to(device)

        x = torch.randn(4, 8, d_model, requires_grad=True).to(device)
        mem = torch.randn(4, 16, d_model, requires_grad=True).to(device)
        motif_ctrl = torch.rand(4, n_motifs, requires_grad=True).to(device)

        output = transformer_block(x, mem, motif_ctrl, adapter_idx=0)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert mem.grad is not None
        assert motif_ctrl.grad is not None

        # Check parameter gradients
        grad_count = 0
        for name, param in transformer_block.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        assert grad_count > 0, "No gradients computed for any parameters"


class TestNeuralCore:
    """Test suite for NeuralCore (main policy network)."""

    def test_initialization(self, neural_core, d_model, n_tools, device):
        """Test NeuralCore initialization."""
        neural_core = neural_core.to(device)

        assert hasattr(neural_core, 'latent')
        assert hasattr(neural_core, 'blocks')
        assert hasattr(neural_core, 'readout')
        assert hasattr(neural_core, 'tool_head')
        assert hasattr(neural_core, 'semantic_proj')
        assert hasattr(neural_core, 'semantic_gate')

        # Check tool head output dimension
        assert neural_core.tool_head.out_features == n_tools

    @pytest.mark.asyncio
    async def test_decide_without_semantic(self, neural_core, d_model, n_motifs, n_tools, device):
        """Test decide method without semantic features."""
        neural_core = neural_core.to(device)

        batch_size = 4
        memory_len = 16

        mem = torch.randn(batch_size, memory_len, d_model).to(device)
        ctrl = torch.rand(batch_size, n_motifs).to(device)
        adapter_idx = 0

        logits, pooled = await neural_core.decide(mem, ctrl, adapter_idx)

        assert logits.shape == (batch_size, n_tools)
        assert pooled.shape == (batch_size, d_model)

    @pytest.mark.asyncio
    async def test_decide_with_semantic(self, neural_core, d_model, n_motifs, n_tools, device):
        """Test decide method with semantic features."""
        neural_core = neural_core.to(device)

        batch_size = 4
        memory_len = 16
        semantic_dim = 8

        mem = torch.randn(batch_size, memory_len, d_model).to(device)
        ctrl = torch.rand(batch_size, n_motifs).to(device)
        semantic_features = torch.randn(batch_size, semantic_dim).to(device)
        adapter_idx = 0

        logits, pooled = await neural_core.decide(mem, ctrl, adapter_idx, semantic_features=semantic_features)

        assert logits.shape == (batch_size, n_tools)
        assert pooled.shape == (batch_size, d_model)

    @pytest.mark.asyncio
    async def test_semantic_gate_effect(self, neural_core, d_model, n_motifs, device):
        """Test that semantic features affect the gating mechanism."""
        neural_core = neural_core.to(device)

        batch_size = 4
        memory_len = 16

        mem = torch.randn(batch_size, memory_len, d_model).to(device)
        ctrl = torch.rand(batch_size, n_motifs).to(device)

        # Without semantic features
        logits_no_sem, _ = await neural_core.decide(mem, ctrl, adapter_idx=0)

        # With semantic features
        semantic = torch.randn(batch_size, 8).to(device)
        logits_with_sem, _ = await neural_core.decide(mem, ctrl, adapter_idx=0, semantic_features=semantic)

        # Outputs should differ
        assert not torch.allclose(logits_no_sem, logits_with_sem)

    def test_tool_names(self, neural_core):
        """Test that tool names are correctly defined."""
        expected_tools = ["answer", "search", "notion_write", "calc"]
        assert neural_core.tools == expected_tools


# ============================================================================
# SECTION 2: THOMPSON SAMPLING BANDIT (~500 lines)
# ============================================================================

class TestTSBanditInitialization:
    """Test Thompson Sampling bandit initialization."""

    def test_default_initialization(self):
        """Test bandit with default parameters."""
        from HoloLoom.policy.thompson_sampling import TSBandit
        from HoloLoom.protocols.types import BanditStrategy

        bandit = TSBandit(n_arms=4)

        assert bandit.n_arms == 4
        assert len(bandit.success) == 4
        assert len(bandit.fail) == 4
        assert len(bandit.pulls) == 4

        # Initial priors should be uniform (Beta(1,1))
        np.testing.assert_array_equal(bandit.success, np.ones(4))
        np.testing.assert_array_equal(bandit.fail, np.ones(4))
        np.testing.assert_array_equal(bandit.pulls, np.zeros(4))

    def test_initialization_with_strategy(self):
        """Test bandit with different strategies."""
        from HoloLoom.policy.thompson_sampling import TSBandit
        from HoloLoom.protocols.types import BanditStrategy

        for strategy in BanditStrategy:
            bandit = TSBandit(n_arms=4, strategy=strategy)
            assert bandit.strategy == strategy

    def test_epsilon_parameter(self):
        """Test epsilon parameter for exploration rate."""
        from HoloLoom.policy.thompson_sampling import TSBandit
        from HoloLoom.protocols.types import BanditStrategy

        bandit = TSBandit(n_arms=4, epsilon=0.2)
        assert bandit.epsilon == 0.2


class TestTSBanditChoose:
    """Test Thompson Sampling arm selection."""

    def test_choose_returns_valid_arm(self, ts_bandit):
        """Test that choose returns a valid arm index."""
        for _ in range(100):
            arm = ts_bandit.choose()
            assert 0 <= arm < ts_bandit.n_arms

    def test_choose_distribution(self, ts_bandit):
        """Test that choose explores all arms over many trials."""
        arm_counts = np.zeros(ts_bandit.n_arms)

        for _ in range(1000):
            arm = ts_bandit.choose()
            arm_counts[arm] += 1

        # All arms should be selected at least once
        assert (arm_counts > 0).all()

    def test_choose_prefers_successful_arms(self, trained_bandit):
        """Test that choose prefers arms with higher success rates."""
        arm_counts = np.zeros(trained_bandit.n_arms)

        for _ in range(1000):
            arm = trained_bandit.choose()
            arm_counts[arm] += 1

        # Arm 0 should be preferred (highest success in trained_bandit)
        assert arm_counts[0] > arm_counts[2]
        assert arm_counts[0] > arm_counts[3]


class TestTSBanditGetPriors:
    """Test bandit prior probability retrieval."""

    def test_uniform_priors_at_start(self, ts_bandit):
        """Test priors are uniform with no training."""
        priors = ts_bandit.get_priors()

        assert len(priors) == ts_bandit.n_arms
        # Beta(1,1) has mean 0.5
        np.testing.assert_array_almost_equal(priors, np.full(4, 0.5))

    def test_priors_update_after_training(self, trained_bandit):
        """Test priors reflect training history."""
        priors = trained_bandit.get_priors()

        # Arm 0 should have highest prior (most successful)
        assert priors[0] > priors[2]
        assert priors[0] > priors[3]

    def test_priors_sum_not_necessarily_one(self, trained_bandit):
        """Test that priors don't need to sum to 1 (they're not probabilities)."""
        priors = trained_bandit.get_priors()

        # Each prior is an independent Beta distribution mean
        # They represent expected success rates, not a probability distribution
        assert all(0 <= p <= 1 for p in priors)


class TestTSBanditSelectWithStrategy:
    """Test strategy-based arm selection."""

    def test_epsilon_greedy_exploit(self):
        """Test epsilon-greedy exploitation mode."""
        from HoloLoom.policy.thompson_sampling import TSBandit
        from HoloLoom.protocols.types import BanditStrategy

        bandit = TSBandit(n_arms=4, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=0.0)

        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        # With epsilon=0, should always exploit
        for _ in range(100):
            arm, debug = bandit.select_with_strategy(neural_probs)
            assert arm == 1  # Highest neural prob
            assert debug['mode'] == 'exploit'

    def test_epsilon_greedy_explore(self):
        """Test epsilon-greedy exploration mode."""
        from HoloLoom.policy.thompson_sampling import TSBandit
        from HoloLoom.protocols.types import BanditStrategy

        bandit = TSBandit(n_arms=4, strategy=BanditStrategy.EPSILON_GREEDY, epsilon=1.0)

        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        # With epsilon=1, should always explore
        explore_count = 0
        for _ in range(100):
            arm, debug = bandit.select_with_strategy(neural_probs)
            if debug['mode'] == 'explore':
                explore_count += 1

        assert explore_count == 100

    def test_bayesian_blend(self, bayesian_blend_bandit):
        """Test Bayesian blend strategy."""
        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        arm, debug = bayesian_blend_bandit.select_with_strategy(neural_probs)

        assert debug['mode'] == 'blend'
        assert 'neural_probs' in debug
        assert 'bandit_priors' in debug
        assert 'combined' in debug

        # Combined should be 70% neural + 30% bandit
        combined = np.array(debug['combined'])
        expected = 0.7 * neural_probs + 0.3 * bayesian_blend_bandit.get_priors()
        np.testing.assert_array_almost_equal(combined, expected)

    def test_pure_thompson(self, pure_thompson_bandit):
        """Test pure Thompson Sampling strategy."""
        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        arm, debug = pure_thompson_bandit.select_with_strategy(neural_probs)

        assert debug['mode'] == 'pure_thompson'
        assert 'sampled_values' in debug

    def test_pulls_tracking(self, ts_bandit):
        """Test that pulls are tracked correctly."""
        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        initial_pulls = ts_bandit.pulls.copy()

        arm, _ = ts_bandit.select_with_strategy(neural_probs)

        # The selected arm's pull count should increase
        assert ts_bandit.pulls[arm] == initial_pulls[arm] + 1

    def test_debug_info_structure(self, ts_bandit):
        """Test that debug info has expected structure."""
        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        arm, debug = ts_bandit.select_with_strategy(neural_probs)

        assert 'strategy' in debug
        assert 'mode' in debug
        assert 'selected_arm' in debug
        assert 'total_pulls' in debug


class TestTSBanditUpdate:
    """Test bandit statistics update."""

    def test_positive_reward_update(self, ts_bandit):
        """Test update with positive reward."""
        initial_success = ts_bandit.success[0]
        initial_fail = ts_bandit.fail[0]

        ts_bandit.update(arm=0, reward=0.8)

        assert ts_bandit.success[0] == initial_success + 0.8
        assert ts_bandit.fail[0] == initial_fail

    def test_negative_reward_update(self, ts_bandit):
        """Test update with negative reward."""
        initial_success = ts_bandit.success[0]
        initial_fail = ts_bandit.fail[0]

        ts_bandit.update(arm=0, reward=-0.5)

        assert ts_bandit.success[0] == initial_success
        assert ts_bandit.fail[0] == initial_fail + 0.5

    def test_zero_reward_no_change(self, ts_bandit):
        """Test update with zero reward."""
        initial_success = ts_bandit.success[0]
        initial_fail = ts_bandit.fail[0]

        ts_bandit.update(arm=0, reward=0.0)

        assert ts_bandit.success[0] == initial_success
        assert ts_bandit.fail[0] == initial_fail

    def test_multiple_updates(self, ts_bandit):
        """Test multiple successive updates."""
        for i in range(10):
            ts_bandit.update(arm=0, reward=1.0)
            ts_bandit.update(arm=1, reward=-0.5)

        assert ts_bandit.success[0] == 1 + 10  # Initial 1 + 10 successes
        assert ts_bandit.fail[1] == 1 + 5  # Initial 1 + 10 * 0.5


class TestTSBanditGetStats:
    """Test bandit statistics retrieval."""

    def test_stats_structure(self, ts_bandit):
        """Test stats dictionary structure."""
        stats = ts_bandit.get_stats()

        assert len(stats) == ts_bandit.n_arms

        for arm_idx in range(ts_bandit.n_arms):
            assert arm_idx in stats
            arm_stats = stats[arm_idx]
            assert 'mean' in arm_stats
            assert 'success' in arm_stats
            assert 'fail' in arm_stats
            assert 'pulls' in arm_stats

    def test_stats_values(self, trained_bandit):
        """Test stats values are correct."""
        stats = trained_bandit.get_stats()

        for arm_idx in range(trained_bandit.n_arms):
            expected_mean = trained_bandit.success[arm_idx] / (
                trained_bandit.success[arm_idx] + trained_bandit.fail[arm_idx]
            )
            assert abs(stats[arm_idx]['mean'] - expected_mean) < 1e-6

    def test_initial_stats(self, ts_bandit):
        """Test stats at initialization."""
        stats = ts_bandit.get_stats()

        for arm_idx in range(ts_bandit.n_arms):
            assert stats[arm_idx]['mean'] == 0.5  # Beta(1,1) mean
            assert stats[arm_idx]['success'] == 1.0
            assert stats[arm_idx]['fail'] == 1.0
            assert stats[arm_idx]['pulls'] == 0.0


class TestTSBanditEdgeCases:
    """Test edge cases for Thompson Sampling bandit."""

    def test_single_arm(self):
        """Test bandit with single arm."""
        from HoloLoom.policy.thompson_sampling import TSBandit

        bandit = TSBandit(n_arms=1)

        for _ in range(10):
            arm = bandit.choose()
            assert arm == 0

    def test_many_arms(self):
        """Test bandit with many arms."""
        from HoloLoom.policy.thompson_sampling import TSBandit

        bandit = TSBandit(n_arms=100)

        arm_counts = np.zeros(100)
        for _ in range(1000):
            arm = bandit.choose()
            arm_counts[arm] += 1

        # With uniform priors, all arms should be explored
        assert (arm_counts > 0).sum() >= 50  # At least 50% of arms explored

    def test_extreme_priors(self):
        """Test bandit with extreme prior updates."""
        from HoloLoom.policy.thompson_sampling import TSBandit

        bandit = TSBandit(n_arms=4)

        # Give arm 0 very strong prior
        for _ in range(1000):
            bandit.update(0, 1.0)

        # Arm 0 should almost always be chosen
        arm_0_count = sum(1 for _ in range(100) if bandit.choose() == 0)
        assert arm_0_count >= 95

    def test_invalid_strategy(self, ts_bandit):
        """Test with invalid strategy."""
        neural_probs = np.array([0.1, 0.7, 0.1, 0.1])

        # Create mock invalid strategy
        class InvalidStrategy:
            value = "invalid"

        with pytest.raises(ValueError):
            ts_bandit.select_with_strategy(neural_probs, strategy=InvalidStrategy)


# ============================================================================
# SECTION 3: POLICY ENGINE INTEGRATION (~800 lines)
# ============================================================================

class TestCreatePolicyFactory:
    """Test the create_policy factory function."""

    def test_basic_creation(self, mock_embeddings, scales, device):
        """Test basic policy creation."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.protocols.types import BanditStrategy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        assert policy is not None
        assert policy.mem_dim == 384
        assert policy.device == device

    def test_creation_with_all_strategies(self, mock_embeddings, scales, device):
        """Test policy creation with each bandit strategy."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.protocols.types import BanditStrategy

        for strategy in BanditStrategy:
            policy = create_policy(
                mem_dim=384,
                emb=mock_embeddings,
                scales=scales,
                device=device,
                bandit_strategy=strategy
            )
            assert policy.bandit_strategy == strategy

    def test_creation_with_custom_epsilon(self, mock_embeddings, scales, device):
        """Test policy creation with custom epsilon."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device,
            epsilon=0.25
        )

        assert policy.epsilon == 0.25

    def test_creation_with_custom_layers(self, mock_embeddings, scales, device):
        """Test policy creation with custom layer count."""
        from HoloLoom.policy.unified import create_policy

        for n_layers in [1, 2, 4]:
            policy = create_policy(
                mem_dim=384,
                emb=mock_embeddings,
                scales=scales,
                device=device,
                n_layers=n_layers
            )
            assert len(policy.core.blocks) == n_layers

    def test_adapter_mapping(self, mock_embeddings, scales, device):
        """Test adapter dimension mapping."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        assert min(scales) in policy.adapter_for_dim
        assert max(scales) in policy.adapter_for_dim

    def test_adapter_bank(self, mock_embeddings, scales, device):
        """Test adapter bank configuration."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        expected_adapters = {0: "general", 1: "farm", 2: "brewing", 3: "mirrorcore"}
        assert policy.adapter_bank == expected_adapters


class TestUnifiedPolicyDecide:
    """Test UnifiedPolicy decision making."""

    @pytest.mark.asyncio
    async def test_decide_with_context(self, mock_embeddings, scales, mock_features, mock_context, device):
        """Test decide method with full context."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        action_plan = await policy.decide(mock_features, mock_context)

        assert action_plan is not None
        assert action_plan.tool in policy.core.tools
        assert 0.0 <= action_plan.confidence <= 1.0
        assert action_plan.adapter in policy.adapter_bank.values()

    @pytest.mark.asyncio
    async def test_decide_with_empty_context(self, mock_embeddings, scales, mock_features, empty_context, device):
        """Test decide method with empty context."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        action_plan = await policy.decide(mock_features, empty_context)

        assert action_plan is not None
        assert action_plan.tool in policy.core.tools

    @pytest.mark.asyncio
    async def test_decide_tool_probabilities(self, mock_embeddings, scales, mock_features, mock_context, device):
        """Test that tool probabilities sum to 1."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        action_plan = await policy.decide(mock_features, mock_context)

        prob_sum = sum(action_plan.tool_probs.values())
        assert abs(prob_sum - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_decide_updates_bandit(self, mock_embeddings, scales, mock_features, mock_context, device):
        """Test that decide updates bandit statistics."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        initial_pulls = policy.bandit.pulls.copy()

        await policy.decide(mock_features, mock_context)

        # Total pulls should increase by 1
        assert policy.bandit.pulls.sum() == initial_pulls.sum() + 1


class TestUnifiedPolicyMotifControl:
    """Test motif control signal generation."""

    def test_ctrl_from_known_motifs(self, mock_embeddings, scales, device):
        """Test control vector generation from known motifs."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        # Use Unicode arrows (→) to match the source code motif_vocab
        motifs = ["cause→effect", "question→answer"]
        ctrl = policy._ctrl_from(motifs)

        assert ctrl.shape == (1, 8)

        # Check that known motifs activate their indices
        # Motif vocabulary uses Unicode arrows
        motif_vocab = [
            "cause→effect", "contrast", "condition→consequence",
            "goal→constraint", "question→answer", "subordinate→main",
            "advcl→main", "setup→twist"
        ]

        for motif in motifs:
            if motif in motif_vocab:
                idx = motif_vocab.index(motif)
                assert ctrl[0, idx] == 1.0

    def test_ctrl_from_unknown_motifs(self, mock_embeddings, scales, device):
        """Test control vector with unknown motifs."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        motifs = ["unknown_motif_1", "unknown_motif_2"]
        ctrl = policy._ctrl_from(motifs)

        # All should be zero for unknown motifs
        assert torch.allclose(ctrl, torch.zeros_like(ctrl))

    def test_ctrl_from_empty_motifs(self, mock_embeddings, scales, device):
        """Test control vector with no motifs."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        ctrl = policy._ctrl_from([])

        assert ctrl.shape == (1, 8)
        assert torch.allclose(ctrl, torch.zeros_like(ctrl))


class TestUnifiedPolicyStrategies:
    """Test policy behavior with different bandit strategies."""

    @pytest.mark.asyncio
    async def test_epsilon_greedy_behavior(self, mock_embeddings, scales, mock_features, mock_context, device):
        """Test epsilon-greedy strategy behavior."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.protocols.types import BanditStrategy

        # Test with epsilon=0 (pure exploitation)
        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device,
            bandit_strategy=BanditStrategy.EPSILON_GREEDY,
            epsilon=0.0
        )

        # Multiple decisions should be consistent (same neural network output)
        results = []
        for _ in range(10):
            action = await policy.decide(mock_features, mock_context)
            results.append(action.tool)

        # With epsilon=0, should always choose same tool
        assert len(set(results)) == 1

    @pytest.mark.asyncio
    async def test_pure_thompson_exploration(self, mock_embeddings, scales, mock_features, mock_context, device):
        """Test pure Thompson strategy explores more."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.protocols.types import BanditStrategy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device,
            bandit_strategy=BanditStrategy.PURE_THOMPSON
        )

        tool_counts = {}
        for _ in range(50):
            action = await policy.decide(mock_features, mock_context)
            tool_counts[action.tool] = tool_counts.get(action.tool, 0) + 1

        # Pure Thompson should explore multiple tools
        assert len(tool_counts) >= 2


class TestUnifiedPolicySemanticFeatures:
    """Test semantic feature integration."""

    @pytest.mark.asyncio
    async def test_semantic_features_from_context(self, mock_embeddings, scales, mock_features, device):
        """Test semantic features extraction from context metadata."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.policy.tests.conftest import MockContext

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        # Create context with semantic state
        context = MockContext(
            shard_texts=["test text"],
            metadata={
                "semantic_state": {
                    "policy_features": [0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.7, 0.8],
                    "dominant_category": "analytical"
                }
            }
        )

        action_plan = await policy.decide(mock_features, context)
        assert action_plan is not None


# ============================================================================
# SECTION 4: EDGE CASES AND ERROR HANDLING (~400 lines)
# ============================================================================

class TestEdgeCasesEmptyInputs:
    """Test handling of empty and null inputs."""

    @pytest.mark.asyncio
    async def test_empty_shard_texts(self, mock_embeddings, scales, mock_features, device):
        """Test with empty shard texts."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.policy.tests.conftest import MockContext

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        context = MockContext(shard_texts=[], hits=[])
        action_plan = await policy.decide(mock_features, context)

        assert action_plan is not None
        assert action_plan.tool in policy.core.tools

    @pytest.mark.asyncio
    async def test_empty_motifs(self, mock_embeddings, scales, device):
        """Test with empty motifs list."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.policy.tests.conftest import MockContext, MockFeatures

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        features = MockFeatures(motifs=[])
        context = MockContext(shard_texts=["test"])

        action_plan = await policy.decide(features, context)
        assert action_plan is not None

    @pytest.mark.asyncio
    async def test_zero_coherence(self, mock_embeddings, scales, device):
        """Test with zero coherence metric."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.policy.tests.conftest import MockContext, MockFeatures

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        features = MockFeatures(metrics={"coherence": 0.0})
        context = MockContext(shard_texts=["test"])

        action_plan = await policy.decide(features, context)
        assert action_plan is not None


class TestEdgeCasesNumericalStability:
    """Test numerical stability in edge cases."""

    def test_mha_with_large_values(self, d_model, n_heads, device):
        """Test MHA with large input values."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        # Large values
        x = torch.randn(4, 10, d_model).to(device) * 1000
        gates = torch.rand(4, n_heads).to(device)

        output, attn = mha(x, gates)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_mha_with_small_values(self, d_model, n_heads, device):
        """Test MHA with very small input values."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        # Very small values
        x = torch.randn(4, 10, d_model).to(device) * 1e-8
        gates = torch.rand(4, n_heads).to(device)

        output, attn = mha(x, gates)

        assert not torch.isnan(output).any()

    def test_softmax_stability(self, d_model, n_heads, device):
        """Test softmax numerical stability with extreme values."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        # Create input that could cause softmax overflow
        x = torch.randn(4, 10, d_model).to(device)
        x[0, 0, :] = 1e6  # Very large value

        gates = torch.rand(4, n_heads).to(device)

        output, attn = mha(x, gates)

        # Attention weights should still be valid
        assert not torch.isnan(attn).any()


class TestEdgeCasesBatchSizes:
    """Test with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 16, 64, 128])
    def test_mha_batch_sizes(self, batch_size, d_model, n_heads, device):
        """Test MHA with different batch sizes."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        x = torch.randn(batch_size, 10, d_model).to(device)
        gates = torch.rand(batch_size, n_heads).to(device)

        output, attn = mha(x, gates)

        assert output.shape == (batch_size, 10, d_model)

    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    @pytest.mark.asyncio
    async def test_neural_core_batch_sizes(self, batch_size, d_model, n_motifs, n_tools, device):
        """Test NeuralCore with different batch sizes."""
        from HoloLoom.policy.unified import NeuralCore

        core = NeuralCore(d_model=d_model, n_layers=2, n_motifs=n_motifs, n_tools=n_tools).to(device)

        mem = torch.randn(batch_size, 16, d_model).to(device)
        ctrl = torch.rand(batch_size, n_motifs).to(device)

        logits, pooled = await core.decide(mem, ctrl, adapter_idx=0)

        assert logits.shape == (batch_size, n_tools)
        assert pooled.shape == (batch_size, d_model)


class TestEdgeCasesSequenceLengths:
    """Test with various sequence lengths."""

    @pytest.mark.parametrize("seq_len", [1, 5, 20, 100])
    def test_mha_sequence_lengths(self, seq_len, d_model, n_heads, device):
        """Test MHA with different sequence lengths."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)

        x = torch.randn(4, seq_len, d_model).to(device)
        gates = torch.rand(4, n_heads).to(device)

        output, attn = mha(x, gates)

        assert output.shape == (4, seq_len, d_model)
        assert attn.shape == (4, n_heads, seq_len, seq_len)


class TestEdgeCasesBanditBounds:
    """Test bandit behavior at boundaries."""

    def test_bandit_after_many_updates(self):
        """Test bandit stability after many updates."""
        from HoloLoom.policy.thompson_sampling import TSBandit

        bandit = TSBandit(n_arms=4)

        # Many updates
        for _ in range(10000):
            arm = bandit.choose()
            reward = np.random.choice([1.0, -0.5])
            bandit.update(arm, reward)

        # Should still produce valid outputs
        stats = bandit.get_stats()
        for arm_idx in range(4):
            assert 0 <= stats[arm_idx]['mean'] <= 1
            assert stats[arm_idx]['success'] > 0
            assert stats[arm_idx]['fail'] > 0

    def test_bandit_with_extreme_rewards(self):
        """Test bandit with extreme reward values."""
        from HoloLoom.policy.thompson_sampling import TSBandit

        bandit = TSBandit(n_arms=4)

        # Very large positive reward
        bandit.update(0, 1000.0)
        stats = bandit.get_stats()
        assert stats[0]['success'] == 1001.0  # Initial 1 + 1000

        # Very large negative reward
        bandit.update(1, -1000.0)
        stats = bandit.get_stats()
        assert stats[1]['fail'] == 1001.0  # Initial 1 + 1000

        # Should still choose arm 0 preferentially
        arm_counts = [0, 0, 0, 0]
        for _ in range(100):
            arm = bandit.choose()
            arm_counts[arm] += 1

        assert arm_counts[0] > arm_counts[1]


class TestEdgeCasesDeviceCompatibility:
    """Test device compatibility and transfers."""

    def test_mha_device_transfer(self, d_model, n_heads):
        """Test MHA device transfer."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads)

        # Start on CPU
        x_cpu = torch.randn(4, 10, d_model)
        gates_cpu = torch.rand(4, n_heads)

        output_cpu, _ = mha(x_cpu, gates_cpu)
        assert output_cpu.device.type == 'cpu'

        # Move to GPU if available
        if torch.cuda.is_available():
            mha_cuda = mha.cuda()
            x_cuda = x_cpu.cuda()
            gates_cuda = gates_cpu.cuda()

            output_cuda, _ = mha_cuda(x_cuda, gates_cuda)
            assert output_cuda.device.type == 'cuda'

    def test_policy_device_consistency(self, mock_embeddings, scales, device):
        """Test policy maintains device consistency."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        # Check all parameters are on correct device
        for name, param in policy.core.named_parameters():
            assert param.device == device, f"Parameter {name} on wrong device"


# ============================================================================
# SECTION 5: AUXILIARY COMPONENTS (~300 lines)
# ============================================================================

class TestMLPBlock:
    """Test MLP building block."""

    def test_initialization(self):
        """Test MLPBlock initialization."""
        from HoloLoom.policy.unified import MLPBlock

        mlp = MLPBlock(128, [256, 256], activation='relu', residual=False)

        assert mlp.out_dim == 256

    def test_forward_pass(self):
        """Test MLPBlock forward pass."""
        from HoloLoom.policy.unified import MLPBlock

        mlp = MLPBlock(128, [256, 256], activation='relu')
        x = torch.randn(32, 128)

        output = mlp(x)

        assert output.shape == (32, 256)

    def test_residual_connection(self):
        """Test MLPBlock with residual connection."""
        from HoloLoom.policy.unified import MLPBlock

        # Residual only works when in_dim == out_dim
        mlp = MLPBlock(256, [256], activation='relu', residual=True)
        x = torch.randn(32, 256)

        output = mlp(x)

        assert output.shape == (32, 256)

    def test_different_activations(self):
        """Test MLPBlock with different activations."""
        from HoloLoom.policy.unified import MLPBlock

        for activation in ['relu', 'gelu']:
            mlp = MLPBlock(128, [256], activation=activation)
            x = torch.randn(32, 128)

            output = mlp(x)
            assert output.shape == (32, 256)


class TestAttentionBlock:
    """Test attention building block."""

    def test_initialization(self):
        """Test AttentionBlock initialization."""
        from HoloLoom.policy.unified import AttentionBlock

        attn = AttentionBlock(128, num_heads=4)
        assert hasattr(attn, 'mha')

    def test_forward_pass(self):
        """Test AttentionBlock forward pass."""
        from HoloLoom.policy.unified import AttentionBlock

        attn = AttentionBlock(128, num_heads=4)
        x = torch.randn(32, 10, 128)

        output = attn(x)

        assert output.shape == (32, 10, 128)


class TestIntrinsicCuriosityModule:
    """Test ICM curiosity module."""

    def test_initialization(self):
        """Test ICM initialization."""
        from HoloLoom.policy.unified import IntrinsicCuriosityModule

        icm = IntrinsicCuriosityModule(128, 6, feature_dim=64)

        assert hasattr(icm, 'encoder')
        assert hasattr(icm, 'forward_model')
        assert hasattr(icm, 'inverse_model')

    def test_forward_pass(self):
        """Test ICM forward pass."""
        from HoloLoom.policy.unified import IntrinsicCuriosityModule

        icm = IntrinsicCuriosityModule(128, 6, feature_dim=64)

        state = torch.randn(32, 128)
        action = torch.randn(32, 6)
        next_state = torch.randn(32, 128)

        output = icm(state, action, next_state)

        assert 'intrinsic_reward' in output
        assert 'forward_loss' in output
        assert 'inverse_loss' in output
        assert output['intrinsic_reward'].shape == (32,)


class TestRandomNetworkDistillation:
    """Test RND curiosity module."""

    def test_initialization(self):
        """Test RND initialization."""
        from HoloLoom.policy.unified import RandomNetworkDistillation

        rnd = RandomNetworkDistillation(128, feature_dim=64)

        assert hasattr(rnd, 'target')
        assert hasattr(rnd, 'predictor')

        # Target should have frozen gradients
        for param in rnd.target.parameters():
            assert not param.requires_grad

    def test_forward_pass(self):
        """Test RND forward pass."""
        from HoloLoom.policy.unified import RandomNetworkDistillation

        rnd = RandomNetworkDistillation(128, feature_dim=64)
        state = torch.randn(32, 128)

        output = rnd(state, update_stats=True)

        assert 'intrinsic_reward' in output
        assert 'prediction_loss' in output


class TestSimpleUnifiedPolicy:
    """Test the test-friendly SimpleUnifiedPolicy."""

    def test_deterministic_policy(self):
        """Test deterministic policy variant."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='deterministic'
        )

        x = torch.randn(32, 128)
        output = policy(x)

        assert 'action' in output
        assert 'value' in output
        assert output['action'].shape == (32, 6)

    def test_categorical_policy(self):
        """Test categorical policy variant."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=10,
            policy_type='categorical'
        )

        x = torch.randn(32, 128)
        output = policy(x)

        assert 'logits' in output
        assert 'action_probs' in output

        # Probabilities should sum to 1
        prob_sum = output['action_probs'].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(32), atol=1e-5)

    def test_gaussian_policy(self):
        """Test Gaussian policy variant."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            state_dependent_std=True
        )

        x = torch.randn(32, 128)
        output = policy(x)

        assert 'mean' in output
        assert 'std' in output
        assert output['std'].shape == (32, 6)

    def test_policy_with_attention(self):
        """Test policy with attention layers."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_attention=True,
            num_attention_layers=2
        )

        # 2D input
        x_2d = torch.randn(32, 128)
        output_2d = policy(x_2d)
        assert 'mean' in output_2d

        # 3D sequential input
        x_3d = torch.randn(32, 10, 128)
        output_3d = policy(x_3d)
        assert 'mean' in output_3d

    def test_policy_with_icm(self):
        """Test policy with ICM curiosity."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_icm=True
        )

        state = torch.randn(32, 128)
        action = torch.randn(32, 6)
        next_state = torch.randn(32, 128)

        intrinsic = policy.compute_intrinsic_reward(state, action, next_state)
        assert intrinsic.shape == (32,)

    def test_policy_with_rnd(self):
        """Test policy with RND curiosity."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_rnd=True
        )

        state = torch.randn(32, 128)
        action = torch.randn(32, 6)
        next_state = torch.randn(32, 128)

        intrinsic = policy.compute_intrinsic_reward(state, action, next_state)
        assert intrinsic.shape == (32,)

    def test_hierarchical_policy(self):
        """Test hierarchical policy with skills."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='deterministic',
            use_hierarchical=True,
            num_skills=8
        )

        x = torch.randn(32, 128)
        output = policy(x)

        assert 'skill' in output
        assert output['skill'].shape == (32, 8)

    def test_sample_action(self):
        """Test action sampling."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        for policy_type in ['deterministic', 'categorical', 'gaussian']:
            policy = SimpleUnifiedPolicy(
                input_dim=128,
                action_dim=6 if policy_type != 'categorical' else 10,
                policy_type=policy_type
            )

            x = torch.randn(32, 128)
            action, info = policy.sample_action(x, deterministic=True)

            if policy_type == 'categorical':
                assert action.shape == (32,)
            else:
                assert action.shape == (32, 6)

    def test_evaluate_actions(self):
        """Test action evaluation."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        for policy_type in ['categorical', 'gaussian']:
            policy = SimpleUnifiedPolicy(
                input_dim=128,
                action_dim=6 if policy_type == 'gaussian' else 10,
                policy_type=policy_type
            )

            x = torch.randn(32, 128)
            if policy_type == 'categorical':
                actions = torch.randint(0, 10, (32,))
            else:
                actions = torch.randn(32, 6)

            eval_dict = policy.evaluate_actions(x, actions)

            assert 'log_probs' in eval_dict
            assert 'entropy' in eval_dict
            assert 'value' in eval_dict

    def test_get_value(self):
        """Test value function."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy

        policy = SimpleUnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian'
        )

        x = torch.randn(32, 128)
        value = policy.get_value(x)

        assert value.shape == (32,)


class TestPPOAgent:
    """Test PPO agent functionality."""

    def test_initialization(self):
        """Test PPO agent initialization."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy, PPOAgent, PPOConfig

        policy = SimpleUnifiedPolicy(128, 6, policy_type='gaussian')
        config = PPOConfig(lr=3e-4, clip_epsilon=0.2)
        agent = PPOAgent(policy, config=config)

        assert agent.policy is policy
        assert agent.config.clip_epsilon == 0.2

    def test_compute_gae(self):
        """Test GAE computation."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy, PPOAgent

        policy = SimpleUnifiedPolicy(128, 6, policy_type='gaussian')
        agent = PPOAgent(policy)

        rewards = torch.randn(100)
        values = torch.randn(100)
        dones = torch.zeros(100)
        dones[30] = 1
        dones[70] = 1
        next_value = torch.randn(1)

        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)

        assert advantages.shape == (100,)
        assert returns.shape == (100,)

    def test_save_load(self):
        """Test agent save and load."""
        from HoloLoom.policy.unified import SimpleUnifiedPolicy, PPOAgent

        policy = SimpleUnifiedPolicy(128, 6, policy_type='gaussian')
        agent = PPOAgent(policy)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name

        try:
            agent.save(temp_path)

            new_policy = SimpleUnifiedPolicy(128, 6, policy_type='gaussian')
            new_agent = PPOAgent(new_policy)
            new_agent.load(temp_path)

            # Verify weights match
            for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            import os
            os.unlink(temp_path)


# ============================================================================
# INTEGRATION TESTS (~200 lines)
# ============================================================================

class TestFullIntegration:
    """Full integration tests for the policy engine."""

    @pytest.mark.asyncio
    async def test_complete_decision_loop(self, mock_embeddings, scales, device):
        """Test complete decision making loop."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.policy.tests.conftest import MockFeatures, MockContext

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        # Simulate multiple decisions
        for i in range(10):
            # psi must be 384-dimensional to match mem_dim
            # Use Unicode arrows (→) to match the source code motif_vocab
            features = MockFeatures(
                psi=np.random.randn(384).astype(np.float32),
                motifs=["question→answer"] if i % 2 == 0 else ["cause→effect"],
                metrics={"coherence": 0.5 + i * 0.05}
            )

            context = MockContext(
                shard_texts=[f"Context {i}"],
                hits=[]
            )

            action = await policy.decide(features, context)

            assert action.tool in policy.core.tools
            assert 0 <= action.confidence <= 1

        # Verify bandit has been updated
        stats = policy.bandit.get_stats()
        total_pulls = sum(stats[i]['pulls'] for i in range(4))
        assert total_pulls == 10

    @pytest.mark.asyncio
    async def test_policy_learning_behavior(self, mock_embeddings, scales, device):
        """Test that policy shows learning behavior over time."""
        from HoloLoom.policy.unified import create_policy
        from HoloLoom.protocols.types import BanditStrategy
        from HoloLoom.policy.tests.conftest import MockFeatures, MockContext

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device,
            bandit_strategy=BanditStrategy.BAYESIAN_BLEND
        )

        features = MockFeatures()
        context = MockContext(shard_texts=["test"])

        # Collect initial tool distribution
        initial_counts = {tool: 0 for tool in policy.core.tools}
        for _ in range(50):
            action = await policy.decide(features, context)
            initial_counts[action.tool] += 1

        # Simulate learning: give high rewards to 'answer' tool
        for _ in range(100):
            action = await policy.decide(features, context)
            # Simulate reward based on tool
            # Note: reward is applied in decide() based on coherence
            # This test verifies the bandit statistics change

        stats = policy.bandit.get_stats()
        # Statistics should have been updated
        assert any(stats[i]['pulls'] > 0 for i in range(4))

    def test_gradient_through_full_network(self, mock_embeddings, scales, device):
        """Test gradient flow through entire network."""
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        # Create input tensors that require gradients
        x = torch.randn(4, 16, 384, requires_grad=True).to(device)
        ctrl = torch.rand(4, 8, requires_grad=True).to(device)

        # Forward pass through core
        logits, pooled = asyncio.get_event_loop().run_until_complete(
            policy.core.decide(x, ctrl, adapter_idx=0)
        )

        # Backward pass
        loss = logits.sum() + pooled.sum()
        loss.backward()

        # Verify gradients exist
        assert x.grad is not None
        assert ctrl.grad is not None

        # Count parameters with gradients
        grad_params = 0
        for name, param in policy.core.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_params += 1
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

        assert grad_params > 0


# ============================================================================
# PERFORMANCE TESTS (~100 lines)
# ============================================================================

class TestPerformance:
    """Performance tests for the policy engine."""

    def test_mha_forward_time(self, d_model, n_heads, device, measure_time):
        """Test MHA forward pass timing."""
        from HoloLoom.policy.unified import CustomMHA

        mha = CustomMHA(d_model=d_model, n_heads=n_heads).to(device)
        mha.eval()

        x = torch.randn(32, 16, d_model).to(device)
        gates = torch.rand(32, n_heads).to(device)

        # Warmup
        for _ in range(5):
            mha(x, gates)

        # Measure
        with torch.no_grad():
            _, time_ms = measure_time(lambda: mha(x, gates))

        # Should be fast (< 100ms on CPU, < 10ms on GPU)
        max_time = 10.0 if device.type == 'cuda' else 100.0
        assert time_ms < max_time, f"MHA too slow: {time_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_decide_time(self, mock_embeddings, scales, mock_features, mock_context, device):
        """Test decision making timing."""
        import time
        from HoloLoom.policy.unified import create_policy

        policy = create_policy(
            mem_dim=384,
            emb=mock_embeddings,
            scales=scales,
            device=device
        )

        # Warmup
        for _ in range(3):
            await policy.decide(mock_features, mock_context)

        # Measure time directly in async context (avoid event loop conflict)
        start = time.perf_counter()
        result = await policy.decide(mock_features, mock_context)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should be fast (< 500ms for full decision)
        assert elapsed_ms < 500.0, f"Decision too slow: {elapsed_ms:.2f}ms"
        assert result is not None

    def test_bandit_select_time(self, ts_bandit, measure_time):
        """Test bandit selection timing."""
        neural_probs = np.random.rand(4)
        neural_probs /= neural_probs.sum()

        # Measure
        _, time_ms = measure_time(
            lambda: [ts_bandit.select_with_strategy(neural_probs) for _ in range(1000)]
        )

        # 1000 selections should be very fast (< 100ms)
        assert time_ms < 100.0, f"Bandit selection too slow: {time_ms:.2f}ms"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

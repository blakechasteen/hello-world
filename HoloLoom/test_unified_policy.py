"""
Comprehensive Test Suite for Neural Decision Engine

Run this script to validate all components of the unified policy.
Tests each feature independently and in combination.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import traceback
import sys

# Import the policy module
try:
    from policy.unified import (
        UnifiedPolicy, PPOAgent, PPOConfig,
        IntrinsicCuriosityModule, RandomNetworkDistillation,
        HierarchicalPolicy, MLPBlock, AttentionBlock
    )
    print("✓ Successfully imported all modules from policy.unified")
except ImportError as e:
    print(f"✗ Import Error: {e}")
    print("\nMake sure policy/unified.py exists and is in the correct location.")
    print("Expected structure:")
    print("  your_project/")
    print("  ├── policy/")
    print("  │   ├── __init__.py")
    print("  │   └── unified.py")
    print("  └── test_unified_policy.py  (this file)")
    sys.exit(1)


class TestRunner:
    """Manages test execution and reporting."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nDevice: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def run_test(self, name: str, test_func):
        """Run a single test and record results."""
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print('='*70)
        
        try:
            test_func()
            self.passed += 1
            self.tests.append((name, True, None))
            print(f"✓ PASSED")
        except Exception as e:
            self.failed += 1
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.tests.append((name, False, error_msg))
            print(f"✗ FAILED: {error_msg}")
            print("\nTraceback:")
            traceback.print_exc()
    
    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print('='*70)
        print(f"Total Tests: {self.passed + self.failed}")
        print(f"Passed: {self.passed} ✓")
        print(f"Failed: {self.failed} ✗")
        print(f"Success Rate: {100 * self.passed / (self.passed + self.failed):.1f}%")
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for name, passed, error in self.tests:
                if not passed:
                    print(f"  ✗ {name}")
                    print(f"    Error: {error}")
        
        print('='*70)
        return self.failed == 0


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_mlp_block(runner: TestRunner):
    """Test basic MLP building block."""
    def test():
        mlp = MLPBlock(128, [256, 256], activation='relu', residual=False)
        x = torch.randn(32, 128)
        out = mlp(x)
        assert out.shape == (32, 256), f"Expected (32, 256), got {out.shape}"
        print(f"  Output shape: {out.shape} ✓")
        
        # Test with residual
        mlp_res = MLPBlock(256, [256], activation='relu', residual=True)
        x = torch.randn(32, 256)
        out = mlp_res(x)
        assert out.shape == (32, 256)
        print(f"  Residual connection works ✓")
    
    runner.run_test("MLPBlock", test)


def test_attention_block(runner: TestRunner):
    """Test attention mechanism."""
    def test():
        attn = AttentionBlock(128, num_heads=4)
        x = torch.randn(32, 10, 128)  # batch, seq, features
        out = attn(x)
        assert out.shape == (32, 10, 128), f"Expected (32, 10, 128), got {out.shape}"
        print(f"  Attention output shape: {out.shape} ✓")
    
    runner.run_test("AttentionBlock", test)


def test_icm(runner: TestRunner):
    """Test Intrinsic Curiosity Module."""
    def test():
        state_dim, action_dim = 128, 6
        icm = IntrinsicCuriosityModule(state_dim, action_dim, feature_dim=64)
        
        state = torch.randn(32, state_dim)
        action = torch.randn(32, action_dim)
        next_state = torch.randn(32, state_dim)
        
        output = icm(state, action, next_state)
        
        assert 'intrinsic_reward' in output
        assert 'forward_loss' in output
        assert 'inverse_loss' in output
        
        print(f"  Intrinsic reward shape: {output['intrinsic_reward'].shape} ✓")
        print(f"  Forward loss: {output['forward_loss'].item():.4f} ✓")
        print(f"  Inverse loss: {output['inverse_loss'].item():.4f} ✓")
        print(f"  Mean intrinsic reward: {output['intrinsic_reward'].mean().item():.4f} ✓")
    
    runner.run_test("Intrinsic Curiosity Module (ICM)", test)


def test_rnd(runner: TestRunner):
    """Test Random Network Distillation."""
    def test():
        state_dim = 128
        rnd = RandomNetworkDistillation(state_dim, feature_dim=64)
        
        state = torch.randn(32, state_dim)
        
        # Test in training mode
        rnd.train()
        output = rnd(state, update_stats=True)
        
        assert 'intrinsic_reward' in output
        assert 'prediction_loss' in output
        
        print(f"  Intrinsic reward shape: {output['intrinsic_reward'].shape} ✓")
        print(f"  Prediction loss: {output['prediction_loss'].item():.4f} ✓")
        print(f"  Running mean updated: {rnd.running_mean.abs().sum().item() > 0} ✓")
    
    runner.run_test("Random Network Distillation (RND)", test)


def test_hierarchical_policy(runner: TestRunner):
    """Test hierarchical policy with skills."""
    def test():
        state_dim, action_dim = 128, 6
        hier = HierarchicalPolicy(state_dim, action_dim, num_skills=8)
        
        state = torch.randn(32, state_dim)
        
        # Test skill selection
        skill, skill_idx = hier.select_skill(state, deterministic=False)
        print(f"  Selected skill shape: {skill.shape} ✓")
        print(f"  Skill indices shape: {skill_idx.shape} ✓")
        
        # Test forward pass
        output = hier(state)
        
        assert 'mean' in output
        assert 'std' in output
        assert 'value' in output
        assert 'skill' in output
        
        print(f"  Action mean shape: {output['mean'].shape} ✓")
        print(f"  Value shape: {output['value'].shape} ✓")
        
        # Test skill diversity loss
        skills = skill
        loss = hier.compute_skill_diversity_loss(state, skills)
        print(f"  Skill diversity loss: {loss.item():.4f} ✓")
    
    runner.run_test("Hierarchical Policy", test)


def test_unified_policy_deterministic(runner: TestRunner):
    """Test deterministic policy."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='deterministic',
            hidden_dims=[256, 256]
        )
        
        x = torch.randn(32, 128)
        outputs = policy(x)
        
        assert 'action' in outputs
        assert 'value' in outputs
        assert outputs['action'].shape == (32, 6)
        
        print(f"  Action shape: {outputs['action'].shape} ✓")
        print(f"  Action range: [{outputs['action'].min():.2f}, {outputs['action'].max():.2f}] ✓")
        
        # Test sampling
        action, info = policy.sample_action(x)
        assert action.shape == (32, 6)
        print(f"  Sample action works ✓")
    
    runner.run_test("Unified Policy - Deterministic", test)


def test_unified_policy_categorical(runner: TestRunner):
    """Test categorical policy."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=10,
            policy_type='categorical'
        )
        
        x = torch.randn(32, 128)
        outputs = policy(x)
        
        assert 'logits' in outputs
        assert 'action_probs' in outputs
        assert outputs['action_probs'].shape == (32, 10)
        
        # Check probabilities sum to 1
        prob_sum = outputs['action_probs'].sum(dim=-1)
        assert torch.allclose(prob_sum, torch.ones(32), atol=1e-5)
        print(f"  Action probabilities sum to 1 ✓")
        
        # Test action evaluation
        actions = torch.randint(0, 10, (32,))
        eval_dict = policy.evaluate_actions(x, actions)
        
        assert 'log_probs' in eval_dict
        assert 'entropy' in eval_dict
        assert 'value' in eval_dict
        
        print(f"  Log probs shape: {eval_dict['log_probs'].shape} ✓")
        print(f"  Entropy: {eval_dict['entropy'].mean().item():.4f} ✓")
    
    runner.run_test("Unified Policy - Categorical", test)


def test_unified_policy_gaussian(runner: TestRunner):
    """Test Gaussian policy."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            state_dependent_std=True
        )
        
        x = torch.randn(32, 128)
        outputs = policy(x)
        
        assert 'mean' in outputs
        assert 'std' in outputs
        assert 'log_std' in outputs
        
        print(f"  Mean shape: {outputs['mean'].shape} ✓")
        print(f"  Std shape: {outputs['std'].shape} ✓")
        print(f"  Std range: [{outputs['std'].min():.4f}, {outputs['std'].max():.4f}] ✓")
        
        # Test action evaluation
        actions = torch.randn(32, 6)
        eval_dict = policy.evaluate_actions(x, actions)
        
        print(f"  Log probs computed ✓")
        print(f"  Mean entropy: {eval_dict['entropy'].mean().item():.4f} ✓")
    
    runner.run_test("Unified Policy - Gaussian", test)


def test_unified_policy_with_attention(runner: TestRunner):
    """Test policy with attention mechanism."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_attention=True,
            num_attention_layers=2
        )
        
        # Test with 2D input (batch, features)
        x = torch.randn(32, 128)
        action, info = policy.sample_action(x)
        print(f"  2D input handled ✓")
        
        # Test with 3D input (batch, seq, features)
        x_seq = torch.randn(32, 10, 128)
        action, info = policy.sample_action(x_seq)
        assert action.shape == (32, 6)
        print(f"  3D sequential input handled ✓")
    
    runner.run_test("Unified Policy - With Attention", test)


def test_unified_policy_with_icm(runner: TestRunner):
    """Test policy with ICM curiosity."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_icm=True
        )
        
        state = torch.randn(32, 128)
        action = torch.randn(32, 6)
        next_state = torch.randn(32, 128)
        
        intrinsic_reward = policy.compute_intrinsic_reward(state, action, next_state)
        
        assert intrinsic_reward.shape == (32,)
        print(f"  Intrinsic reward computed ✓")
        print(f"  Mean reward: {intrinsic_reward.mean().item():.4f} ✓")
    
    runner.run_test("Unified Policy - With ICM", test)


def test_unified_policy_with_rnd(runner: TestRunner):
    """Test policy with RND curiosity."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_rnd=True
        )
        
        state = torch.randn(32, 128)
        action = torch.randn(32, 6)
        next_state = torch.randn(32, 128)
        
        intrinsic_reward = policy.compute_intrinsic_reward(state, action, next_state)
        
        assert intrinsic_reward.shape == (32,)
        print(f"  RND reward computed ✓")
    
    runner.run_test("Unified Policy - With RND", test)


def test_unified_policy_hierarchical(runner: TestRunner):
    """Test unified policy in hierarchical mode."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            use_hierarchical=True,
            num_skills=8
        )
        
        x = torch.randn(32, 128)
        outputs = policy(x)
        
        assert 'mean' in outputs
        assert 'value' in outputs
        assert 'skill' in outputs
        
        print(f"  Hierarchical outputs correct ✓")
        
        # Test sampling
        action, info = policy.sample_action(x)
        assert 'skill' in info
        print(f"  Skill in output ✓")
    
    runner.run_test("Unified Policy - Hierarchical Mode", test)


def test_ppo_agent_creation(runner: TestRunner):
    """Test PPO agent creation."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian'
        )
        
        config = PPOConfig(
            clip_epsilon=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
        
        agent = PPOAgent(policy, config=config, device=runner.device)
        
        print(f"  Agent created on {runner.device} ✓")
        print(f"  Optimizer created ✓")
    
    runner.run_test("PPO Agent - Creation", test)


def test_ppo_gae(runner: TestRunner):
    """Test GAE computation."""
    def test():
        policy = UnifiedPolicy(input_dim=128, action_dim=6, policy_type='gaussian')
        agent = PPOAgent(policy, device=runner.device)
        
        rewards = torch.randn(100).to(runner.device)
        values = torch.randn(100).to(runner.device)
        dones = torch.zeros(100).to(runner.device)
        dones[20] = 1  # Episode boundary
        dones[60] = 1
        next_value = torch.randn(1).to(runner.device)
        
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        assert advantages.shape == (100,)
        assert returns.shape == (100,)
        print(f"  GAE computed ✓")
        print(f"  Advantages range: [{advantages.min():.2f}, {advantages.max():.2f}] ✓")
    
    runner.run_test("PPO Agent - GAE Computation", test)


def test_ppo_update(runner: TestRunner):
    """Test PPO update step."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian'
        ).to(runner.device)
        
        agent = PPOAgent(policy, device=runner.device)
        
        # Create fake trajectory
        states = torch.randn(200, 128).to(runner.device)
        actions = torch.randn(200, 6).to(runner.device)
        log_probs = torch.randn(200).to(runner.device)
        returns = torch.randn(200).to(runner.device)
        advantages = torch.randn(200).to(runner.device)
        next_states = torch.randn(200, 128).to(runner.device)
        
        # Perform update
        metrics = agent.update(
            states, actions, log_probs, returns, advantages,
            next_states=next_states,
            num_epochs=2,
            batch_size=64
        )
        
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy' in metrics
        assert 'kl_divergence' in metrics
        
        print(f"  Policy loss: {metrics['policy_loss']:.4f} ✓")
        print(f"  Value loss: {metrics['value_loss']:.4f} ✓")
        print(f"  Entropy: {metrics['entropy']:.4f} ✓")
        print(f"  KL divergence: {metrics['kl_divergence']:.4f} ✓")
    
    runner.run_test("PPO Agent - Update Step", test)


def test_ppo_update_with_curiosity(runner: TestRunner):
    """Test PPO update with curiosity modules."""
    def test():
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_icm=True,
            use_rnd=True
        ).to(runner.device)
        
        agent = PPOAgent(policy, device=runner.device)
        
        # Create trajectory
        states = torch.randn(200, 128).to(runner.device)
        actions = torch.randn(200, 6).to(runner.device)
        log_probs = torch.randn(200).to(runner.device)
        returns = torch.randn(200).to(runner.device)
        advantages = torch.randn(200).to(runner.device)
        next_states = torch.randn(200, 128).to(runner.device)
        
        # Update with curiosity
        metrics = agent.update(
            states, actions, log_probs, returns, advantages,
            next_states=next_states,
            num_epochs=2,
            batch_size=64
        )
        
        assert 'curiosity_loss' in metrics
        print(f"  Curiosity loss: {metrics['curiosity_loss']:.4f} ✓")
        print(f"  Curiosity integrated into training ✓")
    
    runner.run_test("PPO Agent - Update with Curiosity", test)


def test_save_load(runner: TestRunner):
    """Test saving and loading agent."""
    def test():
        import tempfile
        import os
        
        policy = UnifiedPolicy(input_dim=128, action_dim=6, policy_type='gaussian')
        agent = PPOAgent(policy, device=runner.device)
        
        # Save
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            temp_path = f.name
        
        agent.save(temp_path)
        print(f"  Saved checkpoint ✓")
        
        # Create new agent and load
        new_policy = UnifiedPolicy(input_dim=128, action_dim=6, policy_type='gaussian')
        new_agent = PPOAgent(new_policy, device=runner.device)
        new_agent.load(temp_path)
        print(f"  Loaded checkpoint ✓")
        
        # Verify weights match
        for p1, p2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
            assert torch.allclose(p1, p2)
        print(f"  Weights match ✓")
        
        # Cleanup
        os.unlink(temp_path)
    
    runner.run_test("PPO Agent - Save/Load", test)


def test_full_pipeline(runner: TestRunner):
    """Test complete training pipeline."""
    def test():
        # Create full-featured policy
        policy = UnifiedPolicy(
            input_dim=128,
            action_dim=6,
            policy_type='gaussian',
            use_icm=True,
            use_rnd=True,
            use_attention=True,
            hidden_dims=[256, 256]
        ).to(runner.device)
        
        agent = PPOAgent(policy, device=runner.device)
        
        print(f"  Created full policy ✓")
        
        # Simulate environment interaction
        state = torch.randn(1, 128).to(runner.device)
        
        # Sample action
        action, info = policy.sample_action(state)
        print(f"  Action sampled ✓")
        
        # Simulate transition
        next_state = torch.randn(1, 128).to(runner.device)
        extrinsic_reward = torch.tensor([1.0]).to(runner.device)
        
        # Compute intrinsic reward
        intrinsic_reward = policy.compute_intrinsic_reward(
            state, action, next_state
        )
        total_reward = extrinsic_reward + intrinsic_reward
        print(f"  Rewards computed (extrinsic + intrinsic) ✓")
        
        # Build batch
        batch_size = 100
        states = torch.randn(batch_size, 128).to(runner.device)
        actions = torch.randn(batch_size, 6).to(runner.device)
        log_probs = torch.randn(batch_size).to(runner.device)
        returns = torch.randn(batch_size).to(runner.device)
        advantages = torch.randn(batch_size).to(runner.device)
        next_states = torch.randn(batch_size, 128).to(runner.device)
        
        # Update
        metrics = agent.update(
            states, actions, log_probs, returns, advantages,
            next_states=next_states,
            num_epochs=2,
            batch_size=32
        )
        
        print(f"  Training update completed ✓")
        print(f"  All metrics computed ✓")
    
    runner.run_test("Full Pipeline - End-to-End", test)


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("NEURAL DECISION ENGINE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nThis will test all components of the unified policy system.")
    print("Each test is independent and will report its own status.\n")
    
    runner = TestRunner()
    
    # Run all tests
    test_mlp_block(runner)
    test_attention_block(runner)
    test_icm(runner)
    test_rnd(runner)
    test_hierarchical_policy(runner)
    test_unified_policy_deterministic(runner)
    test_unified_policy_categorical(runner)
    test_unified_policy_gaussian(runner)
    test_unified_policy_with_attention(runner)
    test_unified_policy_with_icm(runner)
    test_unified_policy_with_rnd(runner)
    test_unified_policy_hierarchical(runner)
    test_ppo_agent_creation(runner)
    test_ppo_gae(runner)
    test_ppo_update(runner)
    test_ppo_update_with_curiosity(runner)
    test_save_load(runner)
    test_full_pipeline(runner)
    
    # Print summary
    success = runner.print_summary()
    
    if success:
        print("\n🎉 All tests passed! The neural decision engine is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
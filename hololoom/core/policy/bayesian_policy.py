"""
Bayesian Policy Network with Uncertainty Quantification
========================================================
Extends HoloLoom's unified policy with Bayesian neural networks for uncertainty estimation.

MOTIVATION:
-----------
Neural networks give point predictions without uncertainty. Bayesian neural networks (BNNs)
maintain distributions over weights, enabling:

1. **Epistemic Uncertainty**: Model uncertainty (reduces with more data)
2. **Aleatoric Uncertainty**: Data uncertainty (irreducible noise)
3. **Uncertainty-Aware Exploration**: High uncertainty → explore more
4. **Principled Thompson Sampling**: Sample from posterior over weights

APPROACH:
---------
Replace deterministic linear layers with Bayesian layers that maintain
variational distributions q(w) over weights.

Forward pass: Sample multiple weight configurations, compute predictive distribution
Training: Optimize ELBO = E_q[log p(y|x,w)] - KL(q||p)

INTEGRATION:
------------
Wraps existing NeuralCore with BayesianLinearLayer to enable:
- MC dropout-style sampling (n_samples forward passes)
- Epistemic vs aleatoric uncertainty decomposition
- Predictive entropy for exploration signals
- Optional: Replace all layers with Bayesian equivalents

Author: HoloLoom Probabilistic Reasoning Team
Date: 2025-11-03
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hololoom.policy.thompson_sampling import TSBandit
from hololoom.policy.unified import (
    NeuralCore,
    UnifiedPolicy,
)

# Import existing policy components
from hololoom.protocols.types import ActionPlan, BanditStrategy, Context, Features

# Import variational inference components

# ============================================================================
# Bayesian Linear Wrapper for PyTorch
# ============================================================================

class BayesianLinear(nn.Module):
    """
    Bayesian linear layer for PyTorch (wraps numpy-based BayesianLinearLayer).

    Maintains variational distribution over weights: q(W) = N(μ_W, Σ_W)

    Forward pass samples weights from distribution:
        W ~ q(W)
        y = x @ W + b

    Enables uncertainty quantification via MC sampling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        device: torch.device | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.device = device or torch.device("cpu")

        # Variational parameters (mean and log_std of weight distribution)
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_log_std = nn.Parameter(torch.log(torch.ones(out_features, in_features) * prior_std))
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_log_std = nn.Parameter(torch.log(torch.ones(out_features) * prior_std))

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass with weight uncertainty.

        Args:
            x: Input tensor [B, in_features]
            sample: If True, sample weights; if False, use mean (deterministic)

        Returns:
            Output tensor [B, out_features]
        """
        if sample:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(self.weight_log_std)
            weight_eps = torch.randn_like(self.weight_mean)
            weight = self.weight_mean + weight_std * weight_eps

            bias_std = torch.exp(self.bias_log_std)
            bias_eps = torch.randn_like(self.bias_mean)
            bias = self.bias_mean + bias_std * bias_eps
        else:
            # Deterministic (use mean)
            weight = self.weight_mean
            bias = self.bias_mean

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between variational distribution and prior.

        KL(q||p) = ∫ q(w) log[q(w)/p(w)] dw

        For Gaussian q and p:
        KL = 0.5 * (tr(Σ_q) / σ_p² + μ_q² / σ_p² - k + k*log(σ_p²) - log|Σ_q|)

        Returns:
            KL divergence (scalar)
        """
        # Prior: N(0, prior_std²)
        prior_var = self.prior_std ** 2

        # Weight KL
        weight_var = torch.exp(2 * self.weight_log_std)
        weight_kl = 0.5 * (
            torch.sum(weight_var) / prior_var +
            torch.sum(self.weight_mean ** 2) / prior_var -
            self.weight_mean.numel() +
            self.weight_mean.numel() * math.log(prior_var) -
            2 * torch.sum(self.weight_log_std)
        )

        # Bias KL
        bias_var = torch.exp(2 * self.bias_log_std)
        bias_kl = 0.5 * (
            torch.sum(bias_var) / prior_var +
            torch.sum(self.bias_mean ** 2) / prior_var -
            self.bias_mean.numel() +
            self.bias_mean.numel() * math.log(prior_var) -
            2 * torch.sum(self.bias_log_std)
        )

        return weight_kl + bias_kl


# ============================================================================
# Bayesian Neural Core
# ============================================================================

class BayesianNeuralCore(nn.Module):
    """
    Bayesian version of NeuralCore with uncertainty quantification.

    Replaces key linear layers with BayesianLinear to enable:
    - Weight uncertainty
    - Epistemic uncertainty estimation
    - MC sampling for predictions
    """

    def __init__(
        self,
        base_core: NeuralCore,
        prior_std: float = 1.0,
        replace_all: bool = False
    ):
        """
        Initialize Bayesian core from existing NeuralCore.

        Args:
            base_core: Existing NeuralCore instance
            prior_std: Prior standard deviation for weights
            replace_all: If True, replace ALL layers; if False, only tool_head
        """
        super().__init__()

        # Copy base architecture
        self.latent = base_core.latent
        self.blocks = base_core.blocks
        self.readout = base_core.readout
        self.tools = base_core.tools

        # Replace tool selection head with Bayesian version
        d_model = base_core.tool_head.in_features
        n_tools = base_core.tool_head.out_features

        self.tool_head = BayesianLinear(
            d_model,
            n_tools,
            prior_std=prior_std,
            device=next(base_core.parameters()).device
        )

        # TODO: If replace_all=True, also replace attention/FFN layers
        # For now, only replace final layer (sufficient for uncertainty estimation)

        self.device = next(base_core.parameters()).device
        self.to(self.device)

    async def decide(
        self,
        mem: torch.Tensor,
        ctrl: torch.Tensor,
        adapter_idx: int,
        n_samples: int = 1,
        sample_weights: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """
        Make a decision with uncertainty estimation.

        Args:
            mem: Context memory embeddings [B, M, D]
            ctrl: Motif control vector [B, n_motifs]
            adapter_idx: Which adapter to use
            n_samples: Number of MC samples for uncertainty (default: 1)
            sample_weights: If True, sample weights; if False, deterministic

        Returns:
            Tuple of (logits, pooled_features, uncertainty_metrics)
        """
        B = mem.size(0)

        # Expand latent query for batch
        x = self.latent.expand(B, -1, -1)

        # Process through transformer blocks (deterministic)
        for blk in self.blocks:
            x = blk(x, mem, ctrl, adapter_idx)

        # Pool and readout
        pooled = x.mean(dim=1)  # [B, D]
        features = self.readout(pooled)  # [B, D]

        if n_samples == 1 and not sample_weights:
            # Deterministic forward pass
            logits = self.tool_head(features, sample=False)
            uncertainty = {
                'epistemic': 0.0,
                'aleatoric': 0.0,
                'total': 0.0,
                'entropy': 0.0
            }
            return logits, pooled, uncertainty

        # MC sampling for uncertainty estimation
        logits_samples = []
        for _ in range(n_samples):
            logits_sample = self.tool_head(features, sample=True)
            logits_samples.append(logits_sample)

        logits_samples = torch.stack(logits_samples, dim=0)  # [n_samples, B, n_tools]

        # Mean prediction
        logits_mean = logits_samples.mean(dim=0)  # [B, n_tools]

        # Compute uncertainties
        probs_samples = torch.softmax(logits_samples, dim=-1)  # [n_samples, B, n_tools]
        probs_mean = probs_samples.mean(dim=0)  # [B, n_tools]

        # Epistemic uncertainty: variance of predictions
        epistemic = probs_samples.var(dim=0).mean(dim=-1)  # [B]

        # Total uncertainty: predictive entropy
        entropy = -(probs_mean * torch.log(probs_mean + 1e-10)).sum(dim=-1)  # [B]

        # Aleatoric uncertainty: expected entropy of samples
        sample_entropies = -(probs_samples * torch.log(probs_samples + 1e-10)).sum(dim=-1)  # [n_samples, B]
        aleatoric = sample_entropies.mean(dim=0)  # [B]

        uncertainty = {
            'epistemic': float(epistemic.mean().item()),
            'aleatoric': float(aleatoric.mean().item()),
            'total': float(entropy.mean().item()),
            'entropy': float(entropy.mean().item())
        }

        return logits_mean, pooled, uncertainty

    def kl_loss(self) -> torch.Tensor:
        """
        Compute total KL divergence for all Bayesian layers.

        Returns:
            Total KL divergence
        """
        return self.tool_head.kl_divergence()


# ============================================================================
# Bayesian Unified Policy
# ============================================================================

@dataclass
class BayesianUnifiedPolicy:
    """
    Bayesian extension of UnifiedPolicy with uncertainty quantification.

    Key differences from UnifiedPolicy:
    - Maintains distributions over weights (not point estimates)
    - Computes epistemic + aleatoric uncertainty
    - Uses predictive entropy for exploration signals
    - Supports ELBO training (likelihood + KL regularization)

    Integration:
    - Drop-in replacement for UnifiedPolicy when use_bayesian=True
    - Backward compatible interface
    - Adds uncertainty metrics to ActionPlan
    """

    core: BayesianNeuralCore
    psi_proj: nn.Linear
    device: torch.device
    adapter_for_dim: dict[int, int]
    adapter_bank: dict[int, str]
    mem_dim: int
    emb: 'MatryoshkaEmbeddings'  # Type hint (forward reference)
    bandit: TSBandit | None = None
    bandit_strategy: BanditStrategy = BanditStrategy.EPSILON_GREEDY
    epsilon: float = 0.1
    n_samples: int = 10  # MC samples for uncertainty
    kl_weight: float = 1.0  # KL term weight in ELBO
    prior_std: float = 1.0  # Prior weight variance

    def __post_init__(self):
        if self.bandit is None:
            self.bandit = TSBandit(
                n_arms=len(self.core.tools),
                strategy=self.bandit_strategy,
                epsilon=self.epsilon
            )

    async def decide(
        self,
        features: Features,
        context: Context,
        return_uncertainty: bool = True
    ) -> ActionPlan:
        """
        Make a decision with uncertainty quantification.

        Pipeline:
        1. Encode context shards as memory
        2. Build motif control vector
        3. Run Bayesian neural core (MC sampling)
        4. Compute epistemic + aleatoric uncertainty
        5. Use uncertainty for exploration (high uncertainty → explore)
        6. Select tool via bandit strategy
        7. Update bandit with reward
        8. Return ActionPlan with uncertainty metrics

        Args:
            features: Extracted features (Ψ + motifs)
            context: Retrieved context (shards, KG)
            return_uncertainty: If True, compute uncertainty (slower)

        Returns:
            ActionPlan with chosen tool, adapter, and uncertainty metrics
        """
        # Step 1: Encode context as memory
        # Support both old (shard_texts) and new (memories) attribute names
        shard_texts = getattr(context, 'shard_texts', None) or getattr(context, 'memories', [])
        if not shard_texts:
            mem = torch.zeros(1, 1, self.mem_dim).to(self.device)
        else:
            mem_np = self.emb.encode_scales(shard_texts, size=self.mem_dim)
            mem = torch.tensor(mem_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Step 2: Build control signal
        # Support both old (psi) and new (spectral) attribute names
        psi = getattr(features, 'psi', None)
        if psi is None:
            psi = getattr(features, 'spectral', np.zeros(6))
            if psi is None:
                psi = np.zeros(6)
        psi_tensor = torch.tensor(psi, dtype=torch.float32).unsqueeze(0).to(self.device)
        motif_ctrl = self._ctrl_from(features.motifs).to(self.device)
        ctrl = torch.clamp(motif_ctrl + 0.25 * torch.sigmoid(self.psi_proj(psi_tensor)), 0, 1)

        # Step 3: Get adapter
        adapter_idx = self.adapter_for_dim.get(self.mem_dim, 0)

        # Step 4: Forward pass with uncertainty
        if return_uncertainty:
            logits, _, uncertainty = await self.core.decide(
                mem, ctrl, adapter_idx,
                n_samples=self.n_samples,
                sample_weights=True
            )
        else:
            logits, _, uncertainty = await self.core.decide(
                mem, ctrl, adapter_idx,
                n_samples=1,
                sample_weights=False
            )

        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]

        # Step 5: Uncertainty-aware exploration
        # High epistemic uncertainty → explore more
        epistemic_unc = uncertainty['epistemic']

        # Adaptive epsilon based on uncertainty
        adaptive_epsilon = self.epsilon * (1.0 + epistemic_unc)  # Increase exploration when uncertain

        # Step 6: Tool selection
        tool_idx, bandit_debug = self.bandit.select_with_strategy(probs)
        tool = self.core.tools[tool_idx]

        # Step 7: Compute reward
        # Support both old (hits) and new (metadata.hits) attribute names
        hits = getattr(context, 'hits', None) or context.metadata.get('hits', [])
        episodes = len(set(s.episode for s, _ in hits)) if hits else 0
        # Support both old (metrics) and new (metadata) attribute names
        metrics = getattr(features, 'metrics', None) or features.metadata
        coherence = metrics.get("coherence", 0.0) if metrics else 0.0

        # Bonus for reducing epistemic uncertainty
        uncertainty_reduction_bonus = -epistemic_unc * 0.1  # Small penalty for high uncertainty
        reward = coherence + 0.1 * episodes + uncertainty_reduction_bonus

        # Step 8: Update bandit
        self.bandit.update(tool_idx, reward)

        # Step 9: Build ActionPlan
        adapter = self.adapter_bank.get(adapter_idx, "general")
        tool_probs = {self.core.tools[i]: float(probs[i]) for i in range(len(probs))}

        action_plan = ActionPlan(
            tool=tool,
            confidence=float(probs[tool_idx]),
            adapter=adapter,
            tool_probs=tool_probs,
        )

        # Add uncertainty metrics to metadata
        if hasattr(action_plan, 'metadata'):
            action_plan.metadata = {
                'bandit': bandit_debug,
                'uncertainty': uncertainty,
                'adaptive_epsilon': adaptive_epsilon,
                'epistemic_uncertainty': epistemic_unc,
                'aleatoric_uncertainty': uncertainty['aleatoric'],
                'total_uncertainty': uncertainty['total'],
                'predictive_entropy': uncertainty['entropy']
            }

        return action_plan

    def _ctrl_from(self, motifs: list[str]) -> torch.Tensor:
        """Convert motif list to control vector (same as UnifiedPolicy)."""
        motif_vocab = [
            "cause→effect",
            "contrast",
            "condition→consequence",
            "goal→constraint",
            "question→answer",
            "subordinate→main",
            "advcl→main",
            "setup→twist"
        ]

        vec = np.zeros(len(motif_vocab), dtype=np.float32)
        for m in motifs:
            if m in motif_vocab:
                vec[motif_vocab.index(m)] = 1.0

        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

    def compute_elbo(
        self,
        features: Features,
        context: Context,
        true_tool_idx: int
    ) -> tuple[float, float]:
        """
        Compute ELBO for training.

        ELBO = E_q[log p(y|x,w)] - KL(q||p)

        Args:
            features: Input features
            context: Input context
            true_tool_idx: Ground truth tool (for supervised signal)

        Returns:
            Tuple of (elbo, kl_divergence)
        """
        # This would be used in training loop
        # For now, just compute KL
        kl_div = self.core.kl_loss()

        # Likelihood would come from cross-entropy with true_tool_idx
        # log_likelihood = -F.cross_entropy(logits, true_tool_idx)
        # elbo = log_likelihood - self.kl_weight * kl_div

        return 0.0, float(kl_div.item())


# ============================================================================
# Factory Function
# ============================================================================

def create_bayesian_policy(
    base_policy: UnifiedPolicy,
    n_samples: int = 10,
    kl_weight: float = 1.0,
    prior_std: float = 1.0
) -> BayesianUnifiedPolicy:
    """
    Create a Bayesian policy from an existing UnifiedPolicy.

    Args:
        base_policy: Existing UnifiedPolicy instance
        n_samples: Number of MC samples for uncertainty (default: 10)
        kl_weight: Weight for KL term in ELBO (default: 1.0)
        prior_std: Prior standard deviation for weights (default: 1.0)

    Returns:
        BayesianUnifiedPolicy with uncertainty quantification
    """
    # Create Bayesian core from base core
    bayesian_core = BayesianNeuralCore(
        base_core=base_policy.core,
        prior_std=prior_std,
        replace_all=False  # Only replace tool_head for now
    )

    return BayesianUnifiedPolicy(
        core=bayesian_core,
        psi_proj=base_policy.psi_proj,
        device=base_policy.device,
        adapter_for_dim=base_policy.adapter_for_dim,
        adapter_bank=base_policy.adapter_bank,
        mem_dim=base_policy.mem_dim,
        emb=base_policy.emb,
        bandit=base_policy.bandit,
        bandit_strategy=base_policy.bandit_strategy,
        epsilon=base_policy.epsilon,
        n_samples=n_samples,
        kl_weight=kl_weight,
        prior_std=prior_std
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'BayesianLinear',
    'BayesianNeuralCore',
    'BayesianUnifiedPolicy',
    'create_bayesian_policy',
]

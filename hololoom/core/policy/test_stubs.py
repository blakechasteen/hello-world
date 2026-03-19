"""
Test-Compatible Policy Stubs
==============================
Lightweight implementations of policy components for the test suite.

These provide the minimal API tests expect so the module can be imported
and exercised in isolation.

Extracted from unified.py (March 2026 Refactor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    """Simple MLP block used in tests."""
    def __init__(self, in_dim: int, hidden_dims: list[int], activation: str = 'relu', residual: bool = False):
        super().__init__()
        layers = []
        prev = in_dim
        act = nn.ReLU if activation == 'relu' else nn.GELU
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act())
            prev = h
        self.net = nn.Sequential(*layers)
        self.out_dim = prev
        self.residual = residual and (in_dim == self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.residual:
            out = out + x
        return out


class AttentionBlock(nn.Module):
    """Lightweight attention block that wraps nn.MultiheadAttention."""
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(0, 1)
        out, _ = self.mha(x_t, x_t, x_t)
        return out.transpose(0, 1)


class IntrinsicCuriosityModule(nn.Module):
    """Minimal ICM: encoder + forward / inverse models returning losses and reward."""
    def __init__(self, state_dim: int, action_dim: int, feature_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(state_dim, feature_dim), nn.ReLU())
        self.forward_model = nn.Sequential(nn.Linear(feature_dim + action_dim, feature_dim), nn.ReLU())
        self.inverse_model = nn.Sequential(nn.Linear(feature_dim * 2, action_dim))
        self.mse = nn.MSELoss()

    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(state)
        z_next = self.encoder(next_state)

        pred_next = self.forward_model(torch.cat([z, action], dim=-1))
        forward_loss = self.mse(pred_next, z_next)

        pred_action = self.inverse_model(torch.cat([z, z_next], dim=-1))
        inverse_loss = self.mse(pred_action, action)

        intrinsic_reward = ((z_next - pred_next).pow(2).mean(dim=1))

        return {
            'intrinsic_reward': intrinsic_reward,
            'forward_loss': forward_loss,
            'inverse_loss': inverse_loss
        }

    __call__ = forward


class RandomNetworkDistillation(nn.Module):
    """Simple RND: fixed random target network + predictor."""
    def __init__(self, state_dim: int, feature_dim: int = 64):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(state_dim, feature_dim), nn.ReLU())
        for p in self.target.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(nn.Linear(state_dim, feature_dim), nn.ReLU())
        self.running_mean = torch.zeros(1)
        self.mse = nn.MSELoss()

    def forward(self, state: torch.Tensor, update_stats: bool = False) -> dict[str, torch.Tensor]:
        tgt = self.target(state).detach()
        pred = self.predictor(state)
        loss = self.mse(pred, tgt)
        intrinsic = ((pred - tgt).pow(2).mean(dim=1))

        if update_stats:
            m = intrinsic.mean().detach()
            self.running_mean = 0.99 * self.running_mean + 0.01 * m

        return {
            'intrinsic_reward': intrinsic,
            'prediction_loss': loss
        }

    __call__ = forward


class HierarchicalPolicy(nn.Module):
    """Minimal hierarchical policy with skill selection."""
    def __init__(self, state_dim: int, action_dim: int, num_skills: int = 8):
        super().__init__()
        self.skill_head = nn.Linear(state_dim, num_skills)
        self.action_head = nn.Linear(state_dim, action_dim)
        self.value_head = nn.Linear(state_dim, 1)

    def select_skill(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.skill_head(state)
        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            idx = torch.argmax(probs, dim=-1)
        else:
            idx = torch.multinomial(probs, num_samples=1).squeeze(-1)

        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, idx.unsqueeze(-1), 1.0)
        return one_hot, idx

    def forward(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        mean = self.action_head(state)
        std = torch.zeros_like(mean)
        value = self.value_head(state).squeeze(-1)
        skill, _ = self.select_skill(state, deterministic=True)
        return {'mean': mean, 'std': std, 'value': value, 'skill': skill}

    def compute_skill_diversity_loss(self, state: torch.Tensor, skills: torch.Tensor) -> torch.Tensor:
        return torch.var(skills, dim=0).mean()


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_epsilon: float = 0.2
    epochs: int = 4
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01


class PPOAgent:
    def __init__(self, policy: nn.Module, config: PPOConfig = None, device: str = 'cpu', **kwargs):
        self.policy = policy
        self.device = device
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr)

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, next_value: torch.Tensor, gamma: float = 0.99, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            next_v = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_v * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(self, *args: Any, **kwargs: Any) -> dict[str, float]:
        return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'kl_divergence': 0.0, 'curiosity_loss': 0.0}

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path))


class SimpleUnifiedPolicy(nn.Module):
    """Simple, test-friendly UnifiedPolicy that matches the test expectations."""
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        policy_type: str = 'deterministic',
        hidden_dims: list[int] = [256, 256],
        state_dependent_std: bool = False,
        use_attention: bool = False,
        num_attention_layers: int = 0,
        use_icm: bool = False,
        use_rnd: bool = False,
        use_hierarchical: bool = False,
        num_skills: int = 8
    ):
        super().__init__()
        self.policy_type = policy_type
        self.use_icm = use_icm
        self.use_rnd = use_rnd
        self.use_hierarchical = use_hierarchical

        self.mlp = MLPBlock(input_dim, hidden_dims, activation='relu', residual=False)
        last = self.mlp.out_dim if hasattr(self.mlp, 'out_dim') else hidden_dims[-1]

        self.action_head = nn.Linear(last, action_dim)
        self.logit_head = nn.Linear(last, action_dim)
        self.value_head = nn.Linear(last, 1)

        if state_dependent_std:
            self.log_std_head = nn.Linear(last, action_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.use_attention = use_attention
        if use_attention and num_attention_layers > 0:
            self.attn_layers = nn.ModuleList([AttentionBlock(last) for _ in range(num_attention_layers)])
        else:
            self.attn_layers = None

        self.icm = IntrinsicCuriosityModule(input_dim, action_dim, feature_dim=64) if use_icm else None
        self.rnd = RandomNetworkDistillation(input_dim, feature_dim=64) if use_rnd else None

        if use_hierarchical:
            self.skill_head = nn.Linear(last, num_skills)
        else:
            self.skill_head = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and self.attn_layers is not None:
            B, T, F = x.shape
            x_flat = x.view(B * T, F)
            h = self.mlp(x_flat).view(B, T, -1)
            for att in self.attn_layers:
                h = att(h)
            h = h.mean(dim=1)
            return h
        elif x.dim() == 3 and self.attn_layers is None:
            return self.mlp(x.mean(dim=1))
        else:
            return self.mlp(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._encode(x)

        if self.policy_type == 'deterministic':
            action = torch.tanh(self.action_head(h))
            value = self.value_head(h).squeeze(-1)
            out = {'action': action, 'value': value}
            out['mean'] = self.action_head(h)

        elif self.policy_type == 'categorical':
            logits = self.logit_head(h)
            probs = torch.softmax(logits, dim=-1)
            value = self.value_head(h).squeeze(-1)
            out = {'logits': logits, 'action_probs': probs, 'value': value}
            out['mean'] = self.action_head(h)

        elif self.policy_type == 'gaussian':
            mean = self.action_head(h)
            if hasattr(self, 'log_std_head'):
                log_std = self.log_std_head(h)
                std = torch.exp(log_std)
            else:
                log_std = self.log_std.unsqueeze(0).expand(h.size(0), -1)
                std = torch.exp(log_std)
            value = self.value_head(h).squeeze(-1)
            out = {'mean': mean, 'std': std, 'log_std': log_std, 'value': value}

        else:
            raise ValueError(f'Unknown policy_type: {self.policy_type}')

        if self.skill_head is not None:
            out['skill'] = torch.softmax(self.skill_head(h), dim=-1)

        return out

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self._encode(x)
        if self.policy_type == 'categorical':
            logits = self.logit_head(h)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
            entropy = -(log_probs * torch.softmax(logits, dim=-1)).sum(dim=-1)
            value = self.value_head(h).squeeze(-1)
            return {'log_probs': selected, 'entropy': entropy, 'value': value}
        elif self.policy_type == 'gaussian':
            mean = self.action_head(h)
            if hasattr(self, 'log_std_head'):
                log_std = self.log_std_head(h)
            else:
                log_std = self.log_std.unsqueeze(0).expand(h.size(0), -1)
            std = torch.exp(log_std)
            var = std ** 2
            log_probs = -0.5 * (((actions - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi))
            log_probs = log_probs.sum(dim=-1)
            entropy = 0.5 * (log_std * 2 + math.log(2 * math.pi) + 1).sum(dim=-1)
            value = self.value_head(h).squeeze(-1)
            return {'log_probs': log_probs, 'entropy': entropy, 'value': value}
        else:
            raise NotImplementedError

    def sample_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
        out = self.forward(x)
        if self.policy_type == 'deterministic':
            info = {}
            if 'skill' in out:
                info['skill'] = out['skill']
            return out['action'], info
        elif self.policy_type == 'categorical':
            probs = out['action_probs']
            if deterministic:
                sample = torch.argmax(probs, dim=-1)
            else:
                sample = torch.multinomial(probs, num_samples=1).squeeze(-1)
            info = {'probs': probs}
            if 'skill' in out:
                info['skill'] = out['skill']
            return sample, info
        elif self.policy_type == 'gaussian':
            mean = out['mean']
            std = out['std']
            if deterministic:
                sample = mean
            else:
                eps = torch.randn_like(mean)
                sample = mean + eps * std
            info = {'mean': mean, 'std': std}
            if 'skill' in out:
                info['skill'] = out['skill']
            return sample, info

    def compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        if self.icm is not None:
            out = self.icm(state, action, next_state)
            return out['intrinsic_reward']
        if self.rnd is not None:
            out = self.rnd(state, update_stats=False)
            return out['intrinsic_reward']
        return torch.zeros(state.size(0))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        h = self._encode(x)
        return self.value_head(h).squeeze(-1)

"""
Neural Decision Engine - Unified Policy Network

A flexible neural network-based decision engine supporting multiple policy types:
- Deterministic policies for continuous/discrete actions
- Stochastic policies with probability distributions
- Multi-headed architectures for complex decision spaces
- Attention mechanisms for sequential decision-making
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, MultivariateNormal
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class MLPBlock(nn.Module):
    """Multi-layer perceptron building block with residual connections."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 activation: str = 'relu', dropout: float = 0.0,
                 use_layer_norm: bool = True, residual: bool = False):
        super().__init__()
        
        self.residual = residual and (input_dim == hidden_dims[-1])
        layers = []
        
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        if self.residual:
            out = out + x
        return out


class AttentionBlock(nn.Module):
    """Multi-head attention for processing sequential inputs."""
    
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class UnifiedPolicy(nn.Module):
    """
    Unified neural decision engine supporting multiple policy types.
    
    Args:
        input_dim: Dimension of input observations
        action_dim: Dimension of action space
        hidden_dims: List of hidden layer dimensions
        policy_type: Type of policy ('deterministic', 'categorical', 'gaussian', 'multi_headed')
        use_attention: Whether to use attention mechanism for sequential inputs
        num_attention_layers: Number of attention layers if use_attention=True
        state_dependent_std: For Gaussian policies, whether std depends on state
        min_std: Minimum standard deviation for Gaussian policies
        max_std: Maximum standard deviation for Gaussian policies
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        policy_type: str = 'gaussian',
        use_attention: bool = False,
        num_attention_layers: int = 2,
        activation: str = 'relu',
        dropout: float = 0.0,
        state_dependent_std: bool = True,
        min_std: float = 1e-6,
        max_std: float = 1.0,
        use_layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.policy_type = policy_type
        self.use_attention = use_attention
        self.state_dependent_std = state_dependent_std
        self.min_std = min_std
        self.max_std = max_std
        
        # Input processing
        if use_attention:
            self.attention_layers = nn.ModuleList([
                AttentionBlock(input_dim, num_heads=4, dropout=dropout)
                for _ in range(num_attention_layers)
            ])
            feature_dim = input_dim
        else:
            feature_dim = hidden_dims[-1]
        
        # Feature extraction backbone
        self.backbone = MLPBlock(
            input_dim, 
            hidden_dims,
            activation=activation,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        
        # Policy heads based on type
        if policy_type == 'deterministic':
            self.action_head = nn.Linear(feature_dim, action_dim)
        
        elif policy_type == 'categorical':
            self.action_head = nn.Linear(feature_dim, action_dim)
        
        elif policy_type == 'gaussian':
            self.mean_head = nn.Linear(feature_dim, action_dim)
            
            if state_dependent_std:
                self.std_head = nn.Linear(feature_dim, action_dim)
            else:
                self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        elif policy_type == 'multi_headed':
            # Multiple output heads for complex decision spaces
            self.continuous_head = nn.Linear(feature_dim, action_dim // 2)
            self.discrete_head = nn.Linear(feature_dim, action_dim // 2)
            self.std_head = nn.Linear(feature_dim, action_dim // 2)
        
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        
        # Value head (useful for actor-critic architectures)
        self.value_head = nn.Linear(feature_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            x: Input observations [batch_size, (seq_len,) input_dim]
            mask: Optional attention mask for sequential inputs
        
        Returns:
            Dictionary containing policy outputs and value estimate
        """
        # Handle sequential inputs with attention
        if self.use_attention:
            # Ensure 3D input for attention: [batch, seq, features]
            if x.dim() == 2:
                x = x.unsqueeze(1)
            
            for attn_layer in self.attention_layers:
                x = attn_layer(x, mask)
            
            # Pool sequence dimension
            x = x.mean(dim=1)
        
        # Feature extraction
        features = self.backbone(x)
        
        # Value estimation
        value = self.value_head(features)
        
        # Policy-specific outputs
        outputs = {'value': value, 'features': features}
        
        if self.policy_type == 'deterministic':
            action = torch.tanh(self.action_head(features))
            outputs['action'] = action
        
        elif self.policy_type == 'categorical':
            logits = self.action_head(features)
            outputs['logits'] = logits
            outputs['action_probs'] = F.softmax(logits, dim=-1)
        
        elif self.policy_type == 'gaussian':
            mean = self.mean_head(features)
            
            if self.state_dependent_std:
                log_std = self.std_head(features)
                log_std = torch.clamp(log_std, np.log(self.min_std), np.log(self.max_std))
            else:
                log_std = self.log_std.expand_as(mean)
            
            std = torch.exp(log_std)
            
            outputs['mean'] = mean
            outputs['std'] = std
            outputs['log_std'] = log_std
        
        elif self.policy_type == 'multi_headed':
            continuous_action = torch.tanh(self.continuous_head(features))
            discrete_logits = self.discrete_head(features)
            log_std = self.std_head(features)
            
            outputs['continuous_action'] = continuous_action
            outputs['discrete_logits'] = discrete_logits
            outputs['discrete_probs'] = F.softmax(discrete_logits, dim=-1)
            outputs['std'] = torch.exp(log_std)
        
        return outputs
    
    def sample_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict]:
        """
        Sample an action from the policy.
        
        Args:
            x: Input observation
            deterministic: Whether to use deterministic (mean) action
        
        Returns:
            Tuple of (action, info_dict)
        """
        with torch.no_grad():
            outputs = self.forward(x)
        
        if self.policy_type == 'deterministic':
            action = outputs['action']
            info = {'value': outputs['value']}
        
        elif self.policy_type == 'categorical':
            if deterministic:
                action = outputs['action_probs'].argmax(dim=-1)
            else:
                dist = Categorical(outputs['action_probs'])
                action = dist.sample()
            
            info = {
                'value': outputs['value'],
                'action_probs': outputs['action_probs']
            }
        
        elif self.policy_type == 'gaussian':
            if deterministic:
                action = outputs['mean']
            else:
                dist = Normal(outputs['mean'], outputs['std'])
                action = dist.sample()
            
            info = {
                'value': outputs['value'],
                'mean': outputs['mean'],
                'std': outputs['std']
            }
        
        elif self.policy_type == 'multi_headed':
            continuous = outputs['continuous_action']
            
            if deterministic:
                discrete = outputs['discrete_probs'].argmax(dim=-1)
            else:
                dist = Categorical(outputs['discrete_probs'])
                discrete = dist.sample()
            
            action = {
                'continuous': continuous,
                'discrete': discrete
            }
            info = {'value': outputs['value']}
        
        return action, info
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions.
        
        Args:
            x: Input observations
            actions: Actions to evaluate
        
        Returns:
            Dictionary with log_probs, entropy, and value
        """
        outputs = self.forward(x)
        
        if self.policy_type == 'categorical':
            dist = Categorical(outputs['action_probs'])
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        
        elif self.policy_type == 'gaussian':
            dist = Normal(outputs['mean'], outputs['std'])
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        else:
            raise NotImplementedError(f"Action evaluation not implemented for {self.policy_type}")
        
        return {
            'log_probs': log_probs,
            'entropy': entropy,
            'value': outputs['value'].squeeze(-1)
        }
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value estimate for given state."""
        with torch.no_grad():
            outputs = self.forward(x)
        return outputs['value']


class EnsemblePolicy(nn.Module):
    """Ensemble of multiple policies for robust decision-making."""
    
    def __init__(self, num_policies: int, **policy_kwargs):
        super().__init__()
        self.policies = nn.ModuleList([
            UnifiedPolicy(**policy_kwargs) for _ in range(num_policies)
        ])
        self.num_policies = num_policies
    
    def forward(self, x: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """Forward pass through all policies."""
        return [policy(x) for policy in self.policies]
    
    def sample_action(self, x: torch.Tensor, deterministic: bool = False, 
                     aggregate: str = 'mean') -> Tuple[torch.Tensor, Dict]:
        """
        Sample action using ensemble aggregation.
        
        Args:
            x: Input observation
            deterministic: Use deterministic actions
            aggregate: Aggregation method ('mean', 'vote', 'random')
        """
        actions, infos = [], []
        
        for policy in self.policies:
            action, info = policy.sample_action(x, deterministic)
            actions.append(action)
            infos.append(info)
        
        if aggregate == 'mean':
            if isinstance(actions[0], dict):
                aggregated = {k: torch.stack([a[k] for a in actions]).mean(0) 
                            for k in actions[0].keys()}
            else:
                aggregated = torch.stack(actions).mean(0)
        
        elif aggregate == 'vote':
            # For discrete actions
            aggregated = torch.mode(torch.stack(actions), dim=0)[0]
        
        elif aggregate == 'random':
            idx = np.random.randint(self.num_policies)
            aggregated = actions[idx]
        
        return aggregated, {'ensemble_infos': infos}


# Example usage and testing
if __name__ == "__main__":
    # Test different policy types
    batch_size = 32
    input_dim = 128
    action_dim = 6
    
    print("Testing Gaussian Policy:")
    gaussian_policy = UnifiedPolicy(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        policy_type='gaussian',
        state_dependent_std=True
    )
    
    x = torch.randn(batch_size, input_dim)
    action, info = gaussian_policy.sample_action(x)
    print(f"  Action shape: {action.shape}")
    print(f"  Value shape: {info['value'].shape}")
    
    print("\nTesting Categorical Policy:")
    categorical_policy = UnifiedPolicy(
        input_dim=input_dim,
        action_dim=action_dim,
        policy_type='categorical'
    )
    action, info = categorical_policy.sample_action(x)
    print(f"  Action shape: {action.shape}")
    
    print("\nTesting with Attention:")
    attention_policy = UnifiedPolicy(
        input_dim=input_dim,
        action_dim=action_dim,
        policy_type='gaussian',
        use_attention=True,
        num_attention_layers=2
    )
    x_seq = torch.randn(batch_size, 10, input_dim)  # Sequential input
    action, info = attention_policy.sample_action(x_seq)
    print(f"  Action shape: {action.shape}")
    
    print("\nTesting Ensemble:")
    ensemble = EnsemblePolicy(
        num_policies=5,
        input_dim=input_dim,
        action_dim=action_dim,
        policy_type='gaussian'
    )
    action, info = ensemble.sample_action(x, aggregate='mean')
    print(f"  Ensemble action shape: {action.shape}")
    
    print("\n✓ All tests passed!")
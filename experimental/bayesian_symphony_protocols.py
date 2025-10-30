#!/usr/bin/env python3
"""
Bayesian Symphony: Different Bayesian Approaches Throughout mythRL
==================================================================
Because Thompson Sampling everywhere would be boring! Each layer gets its own
sophisticated Bayesian approach tailored to its specific decision-making needs.

Bayesian Architecture Map:
--------------------------
1. Thompson Sampling + MCTS: Decision flux capacitor (already implemented)
2. Variational Bayes: WarpSpace manifold uncertainty
3. Bayesian Neural Networks: Feature extraction with epistemic uncertainty
4. Hierarchical Bayes: Multi-level memory organization
5. Empirical Bayes: Pattern selection hyperparameter learning
6. Non-parametric Bayes: Emergent pattern discovery
7. Gaussian Process: Temporal reasoning smooth interpolation
8. Bayesian Optimization: Hyperparameter tuning for protocols

Plus: Token budgeting with Bayesian resource allocation!
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import math
from abc import ABC, abstractmethod

# Import base mythRL types
try:
    from dev.protocol_modules_mythrl import ComplexityLevel, ProvenceTrace
except ImportError:
    # Fallback definitions
    class ComplexityLevel(Enum):
        LITE = 3
        FAST = 5
        FULL = 7
        RESEARCH = 9
    
    @dataclass
    class ProvenceTrace:
        operation_id: str
        complexity_level: ComplexityLevel
        modules_invoked: List[str] = field(default_factory=list)
        protocol_calls: List[Dict] = field(default_factory=list)


# ============================================================================
# 1. VARIATIONAL BAYESIAN WARPSPACE
# ============================================================================

class VariationalWarpSpace:
    """
    WarpSpace using Variational Bayes for manifold uncertainty quantification.
    
    Instead of point estimates, maintains probability distributions over
    manifold parameters with variational inference.
    """
    
    def __init__(self, base_dim: int = 384, latent_dim: int = 64, num_components: int = 8):
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.num_components = num_components
        
        # Variational parameters for manifold distribution
        self.manifold_mean = torch.zeros(latent_dim, base_dim)
        self.manifold_logvar = torch.zeros(latent_dim, base_dim)
        
        # Mixture weights (Dirichlet prior)
        self.mixture_alpha = torch.ones(num_components)
        
        # Precision matrix (Wishart prior)
        self.precision_mean = torch.eye(latent_dim)
        self.precision_df = latent_dim + 2  # Degrees of freedom
    
    async def tensor_operation_variational(
        self, 
        data: Dict, 
        complexity: ComplexityLevel,
        operation_type: str = "transform"
    ) -> Dict:
        """
        Perform tensor operations with variational uncertainty quantification.
        
        Args:
            data: Input tensor data
            complexity: Complexity level
            operation_type: Type of operation to perform
            
        Returns:
            Result with uncertainty estimates
        """
        start_time = time.perf_counter()
        
        # Extract tensors
        if 'embeddings' in data:
            x = torch.tensor(data['embeddings'], dtype=torch.float32)
        else:
            x = torch.randn(1, self.base_dim)  # Mock data
        
        # Variational inference for manifold parameters
        samples = []
        kl_divergences = []
        
        num_samples = self._get_num_samples(complexity)
        
        for _ in range(num_samples):
            # Sample manifold parameters from variational posterior
            epsilon = torch.randn_like(self.manifold_mean)
            sampled_manifold = self.manifold_mean + torch.exp(0.5 * self.manifold_logvar) * epsilon
            
            # Transform data through sampled manifold
            if operation_type == "transform":
                transformed = x @ sampled_manifold.T
            elif operation_type == "topology":
                # Topological transformation with uncertainty
                transformed = self._topology_transform_uncertain(x, sampled_manifold)
            elif operation_type == "impossible" and complexity == ComplexityLevel.RESEARCH:
                # Impossible math with variational uncertainty
                transformed = self._impossible_math_variational(x, sampled_manifold)
            else:
                transformed = x @ sampled_manifold.T
            
            samples.append(transformed)
            
            # Compute KL divergence
            kl = self._compute_kl_divergence()
            kl_divergences.append(kl)
        
        # Aggregate samples
        stacked_samples = torch.stack(samples)
        mean_result = stacked_samples.mean(dim=0)
        var_result = stacked_samples.var(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = var_result.mean().item()
        
        # Aleatoric uncertainty (data uncertainty) 
        aleatoric_uncertainty = self._estimate_aleatoric_uncertainty(x)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'transformed_data': mean_result.numpy().tolist(),
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': epistemic_uncertainty + aleatoric_uncertainty,
            'kl_divergence': torch.stack(kl_divergences).mean().item(),
            'num_samples': num_samples,
            'confidence': 1.0 / (1.0 + epistemic_uncertainty),
            'duration_ms': duration_ms,
            'manifold_params': {
                'mean_norm': self.manifold_mean.norm().item(),
                'var_norm': torch.exp(self.manifold_logvar).norm().item()
            }
        }
    
    def _get_num_samples(self, complexity: ComplexityLevel) -> int:
        """Get number of variational samples based on complexity."""
        return {
            ComplexityLevel.LITE: 3,
            ComplexityLevel.FAST: 8,
            ComplexityLevel.FULL: 15,
            ComplexityLevel.RESEARCH: 25
        }[complexity]
    
    def _topology_transform_uncertain(self, x: torch.Tensor, manifold: torch.Tensor) -> torch.Tensor:
        """Topological transformation with manifold uncertainty."""
        # Project to manifold
        projected = x @ manifold.T
        
        # Apply non-linear topology
        return torch.tanh(projected) + 0.1 * torch.sin(projected * 3.14159)
    
    def _impossible_math_variational(self, x: torch.Tensor, manifold: torch.Tensor) -> torch.Tensor:
        """Impossible mathematical operations with variational uncertainty."""
        # Complex number operations
        complex_proj = (x @ manifold.T).to(torch.complex64)
        
        # Impossible: Take logarithm of negative numbers (analytically continued)
        result = torch.log(complex_proj + 0j)
        
        # Return real part with imaginary uncertainty
        return result.real + 0.1 * result.imag
    
    def _compute_kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between variational posterior and prior."""
        # KL(q(θ) || p(θ)) for multivariate Gaussian
        kl = -0.5 * torch.sum(1 + self.manifold_logvar - self.manifold_mean.pow(2) - self.manifold_logvar.exp())
        return kl
    
    def _estimate_aleatoric_uncertainty(self, x: torch.Tensor) -> float:
        """Estimate aleatoric (data) uncertainty."""
        return 0.05 * x.var().item()
    
    def update_variational_params(self, gradients: Dict):
        """Update variational parameters using gradients."""
        if 'mean_grad' in gradients:
            self.manifold_mean -= 0.01 * gradients['mean_grad']
        if 'logvar_grad' in gradients:
            self.manifold_logvar -= 0.01 * gradients['logvar_grad']


# ============================================================================
# 2. BAYESIAN NEURAL NETWORKS FOR FEATURE EXTRACTION  
# ============================================================================

class BayesianFeatureExtractor:
    """
    Feature extraction using Bayesian Neural Networks with epistemic uncertainty.
    
    Each weight is a distribution, not a point estimate.
    """
    
    def __init__(self, input_dim: int = 384, hidden_dims: List[int] = [256, 128], output_dim: int = 96):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Bayesian weight parameters (mean and variance)
        self.layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layer = {
                'weight_mean': torch.randn(dims[i], dims[i+1]) * 0.1,
                'weight_logvar': torch.full((dims[i], dims[i+1]), -2.0),  # Start with low variance
                'bias_mean': torch.zeros(dims[i+1]),
                'bias_logvar': torch.full((dims[i+1],), -2.0)
            }
            self.layers.append(layer)
        
        # Prior parameters
        self.prior_weight_var = 1.0
        self.prior_bias_var = 1.0
    
    async def extract_features_bayesian(
        self, 
        embeddings: Union[List[List[float]], np.ndarray], 
        complexity: ComplexityLevel
    ) -> Dict:
        """
        Extract features with Bayesian uncertainty quantification.
        
        Args:
            embeddings: Input embeddings
            complexity: Complexity level for sampling strategy
            
        Returns:
            Features with uncertainty estimates
        """
        start_time = time.perf_counter()
        
        # Convert to tensor
        if isinstance(embeddings, list):
            x = torch.tensor(embeddings, dtype=torch.float32)
        else:
            x = torch.tensor(embeddings, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Monte Carlo sampling for Bayesian inference
        num_samples = self._get_mc_samples(complexity)
        feature_samples = []
        kl_losses = []
        
        for sample_idx in range(num_samples):
            # Sample weights from posterior
            sampled_weights = []
            layer_kl = 0.0
            
            for layer in self.layers:
                # Sample weights
                weight_eps = torch.randn_like(layer['weight_mean'])
                weight = layer['weight_mean'] + torch.exp(0.5 * layer['weight_logvar']) * weight_eps
                
                # Sample biases  
                bias_eps = torch.randn_like(layer['bias_mean'])
                bias = layer['bias_mean'] + torch.exp(0.5 * layer['bias_logvar']) * bias_eps
                
                sampled_weights.append({'weight': weight, 'bias': bias})
                
                # Compute KL divergence for this layer
                kl_w = self._compute_weight_kl(layer['weight_mean'], layer['weight_logvar'])
                kl_b = self._compute_bias_kl(layer['bias_mean'], layer['bias_logvar'])
                layer_kl += kl_w + kl_b
            
            # Forward pass with sampled weights
            h = x
            for i, layer_weights in enumerate(sampled_weights):
                h = h @ layer_weights['weight'] + layer_weights['bias']
                if i < len(sampled_weights) - 1:  # No activation on final layer
                    h = torch.relu(h)
            
            feature_samples.append(h)
            kl_losses.append(layer_kl)
        
        # Aggregate samples
        stacked_features = torch.stack(feature_samples)
        mean_features = stacked_features.mean(dim=0)
        var_features = stacked_features.var(dim=0)
        
        # Uncertainty metrics
        epistemic_uncertainty = var_features.mean().item()
        predictive_entropy = self._compute_predictive_entropy(stacked_features)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'features': mean_features.squeeze().numpy().tolist(),
            'feature_uncertainty': var_features.squeeze().numpy().tolist(),
            'epistemic_uncertainty': epistemic_uncertainty,
            'predictive_entropy': predictive_entropy,
            'kl_divergence': torch.stack(kl_losses).mean().item(),
            'confidence': 1.0 / (1.0 + epistemic_uncertainty),
            'num_samples': num_samples,
            'duration_ms': duration_ms,
            'bayesian_metrics': {
                'weight_posterior_entropy': self._compute_posterior_entropy(),
                'effective_num_params': self._compute_effective_params()
            }
        }
    
    def _get_mc_samples(self, complexity: ComplexityLevel) -> int:
        """Get number of Monte Carlo samples based on complexity."""
        return {
            ComplexityLevel.LITE: 5,
            ComplexityLevel.FAST: 12,
            ComplexityLevel.FULL: 20,
            ComplexityLevel.RESEARCH: 35
        }[complexity]
    
    def _compute_weight_kl(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for weight distribution."""
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(
            (mean.pow(2) + var) / self.prior_weight_var - 1 - logvar + math.log(self.prior_weight_var)
        )
        return kl
    
    def _compute_bias_kl(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for bias distribution."""
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(
            (mean.pow(2) + var) / self.prior_bias_var - 1 - logvar + math.log(self.prior_bias_var)
        )
        return kl
    
    def _compute_predictive_entropy(self, samples: torch.Tensor) -> float:
        """Compute predictive entropy across samples."""
        # Convert to probabilities
        probs = torch.softmax(samples, dim=-1)
        mean_probs = probs.mean(dim=0)
        
        # Compute entropy
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum().item()
        return entropy
    
    def _compute_posterior_entropy(self) -> float:
        """Compute entropy of weight posteriors."""
        total_entropy = 0.0
        for layer in self.layers:
            # Entropy of multivariate Gaussian
            entropy = 0.5 * torch.sum(layer['weight_logvar'] + math.log(2 * math.pi * math.e))
            total_entropy += entropy.item()
        return total_entropy
    
    def _compute_effective_params(self) -> float:
        """Compute effective number of parameters (uncertainty-weighted)."""
        total_params = 0.0
        effective_params = 0.0
        
        for layer in self.layers:
            layer_params = layer['weight_mean'].numel() + layer['bias_mean'].numel()
            layer_uncertainty = torch.exp(layer['weight_logvar']).mean() + torch.exp(layer['bias_logvar']).mean()
            
            total_params += layer_params
            effective_params += layer_params / (1.0 + layer_uncertainty.item())
        
        return effective_params


# ============================================================================
# 3. HIERARCHICAL BAYESIAN MEMORY
# ============================================================================

class HierarchicalBayesianMemory:
    """
    Multi-level memory organization using Hierarchical Bayesian modeling.
    
    Memory shards are organized in a hierarchy with shared hyperpriors.
    """
    
    def __init__(self, levels: int = 3, clusters_per_level: List[int] = [5, 15, 50]):
        self.levels = levels
        self.clusters_per_level = clusters_per_level
        
        # Hierarchical parameters
        self.hyperpriors = []
        self.level_priors = []
        self.shard_posteriors = []
        
        # Initialize hierarchy
        for level in range(levels):
            # Global hyperprior at each level
            hyperprior = {
                'relevance_alpha': 2.0,
                'relevance_beta': 2.0,
                'importance_mu': 0.5,
                'importance_sigma': 0.3
            }
            self.hyperpriors.append(hyperprior)
            
            # Cluster-level priors
            level_clusters = []
            for cluster in range(clusters_per_level[level]):
                cluster_prior = {
                    'relevance_alpha': hyperprior['relevance_alpha'] + np.random.gamma(1, 0.1),
                    'relevance_beta': hyperprior['relevance_beta'] + np.random.gamma(1, 0.1),
                    'importance_mu': hyperprior['importance_mu'] + np.random.normal(0, 0.1),
                    'importance_sigma': hyperprior['importance_sigma']
                }
                level_clusters.append(cluster_prior)
            self.level_priors.append(level_clusters)
        
        # Memory storage
        self.memory_shards = []
        self.shard_assignments = []  # Which cluster each shard belongs to
    
    async def retrieve_hierarchical(
        self, 
        query: str, 
        threshold: float = 0.7,
        complexity: ComplexityLevel = ComplexityLevel.FAST
    ) -> Dict:
        """
        Hierarchical Bayesian retrieval with uncertainty propagation.
        
        Args:
            query: Query string
            threshold: Relevance threshold
            complexity: Complexity level for inference
            
        Returns:
            Retrieved shards with hierarchical uncertainty
        """
        start_time = time.perf_counter()
        
        # Mock query embedding
        query_embedding = np.random.randn(384)
        
        # Hierarchical inference
        level_results = []
        total_uncertainty = 0.0
        
        for level in range(self.levels):
            level_shards = self._get_level_shards(level)
            
            if not level_shards:
                continue
            
            # Bayesian inference at this level
            level_result = await self._infer_level_relevance(
                query_embedding, 
                level_shards, 
                level,
                complexity
            )
            
            level_results.append(level_result)
            total_uncertainty += level_result['uncertainty']
        
        # Hierarchical aggregation
        aggregated_shards = []
        confidence_scores = []
        
        for level_result in level_results:
            for shard in level_result['shards']:
                if shard['hierarchical_relevance'] >= threshold:
                    aggregated_shards.append(shard)
                    confidence_scores.append(shard['confidence'])
        
        # Sort by hierarchical relevance
        aggregated_shards.sort(key=lambda x: x['hierarchical_relevance'], reverse=True)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'shards': aggregated_shards[:20],  # Top 20
            'hierarchical_uncertainty': total_uncertainty,
            'level_contributions': [len(lr['shards']) for lr in level_results],
            'confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'inference_levels': len(level_results),
            'duration_ms': duration_ms,
            'bayesian_evidence': self._compute_model_evidence(level_results)
        }
    
    async def _infer_level_relevance(
        self, 
        query_embedding: np.ndarray, 
        level_shards: List[Dict],
        level: int,
        complexity: ComplexityLevel
    ) -> Dict:
        """Perform Bayesian inference at a specific level."""
        
        # Get priors for this level
        level_priors = self.level_priors[level]
        
        # Inference parameters based on complexity
        num_samples = {
            ComplexityLevel.LITE: 10,
            ComplexityLevel.FAST: 25,
            ComplexityLevel.FULL: 50,
            ComplexityLevel.RESEARCH: 100
        }[complexity]
        
        inferred_shards = []
        level_uncertainty = 0.0
        
        for shard in level_shards:
            # Get cluster assignment
            cluster_idx = shard.get('cluster', 0) % len(level_priors)
            cluster_prior = level_priors[cluster_idx]
            
            # Bayesian inference for relevance
            relevance_samples = []
            importance_samples = []
            
            for _ in range(num_samples):
                # Sample relevance from Beta distribution
                relevance = np.random.beta(
                    cluster_prior['relevance_alpha'],
                    cluster_prior['relevance_beta']
                )
                
                # Sample importance from Gaussian
                importance = np.random.normal(
                    cluster_prior['importance_mu'],
                    cluster_prior['importance_sigma']
                )
                
                relevance_samples.append(relevance)
                importance_samples.append(max(0, importance))  # Truncate at 0
            
            # Aggregate samples
            mean_relevance = np.mean(relevance_samples)
            var_relevance = np.var(relevance_samples)
            mean_importance = np.mean(importance_samples)
            
            # Hierarchical relevance combines level-specific and global factors
            global_factor = self._compute_global_factor(shard, query_embedding)
            hierarchical_relevance = 0.7 * mean_relevance + 0.3 * global_factor
            
            inferred_shard = {
                'id': shard.get('id', f'shard_{len(inferred_shards)}'),
                'content': shard.get('content', 'Sample memory content'),
                'hierarchical_relevance': hierarchical_relevance,
                'level': level,
                'cluster': cluster_idx,
                'confidence': 1.0 / (1.0 + var_relevance),
                'importance': mean_importance,
                'uncertainty': var_relevance
            }
            
            inferred_shards.append(inferred_shard)
            level_uncertainty += var_relevance
        
        return {
            'shards': inferred_shards,
            'uncertainty': level_uncertainty,
            'level': level,
            'num_samples': num_samples
        }
    
    def _get_level_shards(self, level: int) -> List[Dict]:
        """Get shards assigned to a specific level."""
        # Mock data for demonstration
        num_shards = min(10, self.clusters_per_level[level])
        shards = []
        
        for i in range(num_shards):
            shard = {
                'id': f'level_{level}_shard_{i}',
                'content': f'Memory content at level {level}, shard {i}',
                'embedding': np.random.randn(384),
                'cluster': i % self.clusters_per_level[level],
                'level': level
            }
            shards.append(shard)
        
        return shards
    
    def _compute_global_factor(self, shard: Dict, query_embedding: np.ndarray) -> float:
        """Compute global relevance factor."""
        # Simple cosine similarity
        shard_embedding = shard.get('embedding', np.random.randn(384))
        similarity = np.dot(query_embedding, shard_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(shard_embedding)
        )
        return max(0, similarity)
    
    def _compute_model_evidence(self, level_results: List[Dict]) -> float:
        """Compute Bayesian model evidence."""
        # Marginal likelihood approximation
        total_evidence = 0.0
        
        for level_result in level_results:
            level_evidence = 0.0
            for shard in level_result['shards']:
                # Evidence = P(data|model) approximated by confidence
                level_evidence += shard['confidence']
            
            total_evidence += level_evidence / max(1, len(level_result['shards']))
        
        return total_evidence / max(1, len(level_results))


# ============================================================================
# 4. EMPIRICAL BAYESIAN PATTERN SELECTION
# ============================================================================

class EmpiricalBayesianPatternSelector:
    """
    Pattern selection using Empirical Bayes for hyperparameter learning.
    
    Learns optimal hyperparameters from data automatically.
    """
    
    def __init__(self, num_patterns: int = 12):
        self.num_patterns = num_patterns
        self.pattern_names = [
            "analytical", "creative", "factual", "conversational",
            "technical", "explanatory", "comparative", "sequential",
            "causal", "temporal", "spatial", "emergent"
        ]
        
        # Empirical Bayes parameters
        self.pattern_counts = np.ones(num_patterns)  # Success counts
        self.total_counts = np.ones(num_patterns)    # Total attempts
        
        # Hyperparameters (learned from data)
        self.hyperalpha = 1.0  # Will be learned
        self.hyperbeta = 1.0   # Will be learned
        
        # Pattern history for learning
        self.selection_history = []
        self.success_history = []
    
    async def select_pattern_empirical(
        self, 
        query: str, 
        features: Dict,
        complexity: ComplexityLevel
    ) -> Dict:
        """
        Select pattern using Empirical Bayesian approach.
        
        Args:
            query: Input query
            features: Extracted features
            complexity: Complexity level
            
        Returns:
            Selected pattern with uncertainty
        """
        start_time = time.perf_counter()
        
        # Update hyperparameters using empirical Bayes
        self._update_hyperparameters()
        
        # Compute pattern probabilities using learned hyperparameters
        pattern_probs = []
        pattern_uncertainties = []
        
        for i in range(self.num_patterns):
            # Posterior Beta distribution parameters
            alpha_post = self.hyperalpha + self.pattern_counts[i]
            beta_post = self.hyperbeta + (self.total_counts[i] - self.pattern_counts[i])
            
            # Mean and variance of Beta distribution
            mean_prob = alpha_post / (alpha_post + beta_post)
            var_prob = (alpha_post * beta_post) / (
                (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
            )
            
            pattern_probs.append(mean_prob)
            pattern_uncertainties.append(var_prob)
        
        # Feature-based adjustment
        feature_scores = self._compute_feature_scores(query, features)
        
        # Combine empirical priors with feature scores
        combined_scores = []
        for i in range(self.num_patterns):
            # Weighted combination: empirical prior + feature evidence
            weight_empirical = 0.6  # Empirical Bayes weight
            weight_features = 0.4   # Feature weight
            
            combined_score = (
                weight_empirical * pattern_probs[i] + 
                weight_features * feature_scores[i]
            )
            combined_scores.append(combined_score)
        
        # Select pattern based on complexity
        if complexity == ComplexityLevel.LITE:
            # Greedy selection
            selected_idx = np.argmax(combined_scores)
        else:
            # Uncertainty-aware selection
            selected_idx = self._uncertainty_aware_selection(
                combined_scores, 
                pattern_uncertainties, 
                complexity
            )
        
        selected_pattern = self.pattern_names[selected_idx]
        
        # Record selection for future learning
        self.selection_history.append(selected_idx)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        return {
            'selected_pattern': selected_pattern,
            'pattern_idx': selected_idx,
            'confidence': combined_scores[selected_idx],
            'uncertainty': pattern_uncertainties[selected_idx],
            'pattern_probabilities': dict(zip(self.pattern_names, combined_scores)),
            'empirical_hyperparams': {
                'alpha': self.hyperalpha,
                'beta': self.hyperbeta
            },
            'selection_entropy': self._compute_selection_entropy(combined_scores),
            'duration_ms': duration_ms
        }
    
    def _update_hyperparameters(self):
        """Update hyperparameters using empirical Bayes (Type-II ML)."""
        if len(self.selection_history) < 5:
            return  # Need some data first
        
        # Method of moments estimation for Beta hyperparameters
        success_rates = self.pattern_counts / self.total_counts
        
        # Clip to avoid numerical issues
        success_rates = np.clip(success_rates, 0.01, 0.99)
        
        mean_rate = np.mean(success_rates)
        var_rate = np.var(success_rates)
        
        # Method of moments for Beta distribution
        if var_rate > 0 and mean_rate > 0 and mean_rate < 1:
            common_factor = mean_rate * (1 - mean_rate) / var_rate - 1
            self.hyperalpha = mean_rate * common_factor
            self.hyperbeta = (1 - mean_rate) * common_factor
            
            # Ensure reasonable bounds
            self.hyperalpha = max(0.1, min(10.0, self.hyperalpha))
            self.hyperbeta = max(0.1, min(10.0, self.hyperbeta))
    
    def _compute_feature_scores(self, query: str, features: Dict) -> List[float]:
        """Compute pattern scores based on features."""
        scores = []
        query_lower = query.lower()
        
        # Simple feature-based scoring
        for pattern in self.pattern_names:
            score = 0.1  # Base score
            
            # Query-based heuristics
            if pattern == "analytical" and any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
                score += 0.4
            elif pattern == "creative" and any(word in query_lower for word in ["create", "imagine", "brainstorm"]):
                score += 0.4
            elif pattern == "factual" and any(word in query_lower for word in ["what", "who", "when", "where"]):
                score += 0.4
            elif pattern == "technical" and any(word in query_lower for word in ["code", "implement", "algorithm"]):
                score += 0.4
            
            # Feature-based scoring
            if 'confidence' in features:
                score += 0.2 * features['confidence']
            
            scores.append(min(1.0, score))
        
        return scores
    
    def _uncertainty_aware_selection(
        self, 
        scores: List[float], 
        uncertainties: List[float],
        complexity: ComplexityLevel
    ) -> int:
        """Select pattern considering uncertainty."""
        
        # Thompson sampling with uncertainty weighting
        if complexity == ComplexityLevel.RESEARCH:
            # High exploration
            exploration_weight = 0.3
        else:
            # Moderate exploration
            exploration_weight = 0.1
        
        adjusted_scores = []
        for i in range(len(scores)):
            # Add uncertainty bonus for exploration
            uncertainty_bonus = exploration_weight * uncertainties[i]
            adjusted_score = scores[i] + uncertainty_bonus
            adjusted_scores.append(adjusted_score)
        
        return np.argmax(adjusted_scores)
    
    def _compute_selection_entropy(self, scores: List[float]) -> float:
        """Compute entropy of selection distribution."""
        # Normalize to probabilities
        scores_arr = np.array(scores)
        probs = scores_arr / scores_arr.sum()
        
        # Compute entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy
    
    def update_pattern_success(self, pattern_idx: int, success: bool):
        """Update pattern statistics based on outcome."""
        self.total_counts[pattern_idx] += 1
        if success:
            self.pattern_counts[pattern_idx] += 1
        
        self.success_history.append(success)


# ============================================================================
# 5. TOKEN BUDGETING WITH BAYESIAN RESOURCE ALLOCATION
# ============================================================================

@dataclass
class TokenBudget:
    """Smart token budget with Bayesian resource allocation."""
    total_budget: int
    allocated: Dict[str, int] = field(default_factory=dict)
    used: Dict[str, int] = field(default_factory=dict)
    uncertainty_buffer: int = 0
    
    def remaining(self) -> int:
        """Get remaining tokens."""
        return self.total_budget - sum(self.used.values()) - self.uncertainty_buffer


class BayesianTokenBudgetManager:
    """
    Intelligent token budget management using Bayesian resource allocation.
    
    Learns optimal allocation strategies from usage patterns.
    """
    
    def __init__(self, base_budget: int = 8000):
        self.base_budget = base_budget
        
        # Bayesian parameters for budget allocation
        self.protocol_usage_history = {}  # Protocol -> list of token usages
        self.complexity_multipliers = {
            ComplexityLevel.LITE: 0.5,
            ComplexityLevel.FAST: 1.0,
            ComplexityLevel.FULL: 1.8,
            ComplexityLevel.RESEARCH: 3.5
        }
        
        # Learned parameters (Gamma distributions for token usage)
        self.protocol_params = {}  # Protocol -> {alpha, beta} for Gamma distribution
        
        # Default token estimates per protocol
        self.default_estimates = {
            'pattern_selection': 50,
            'feature_extraction': 100,
            'memory_backend': 200,
            'decision_engine': 80,
            'warpspace': 150,
            'tool_execution': 300,
            'synthesis_bridge': 120,
            'temporal_windows': 60
        }
    
    def create_complexity_budget(self, complexity: ComplexityLevel, context_size: int = 0) -> TokenBudget:
        """
        Create budget allocation based on complexity and context.
        
        Args:
            complexity: Complexity level
            context_size: Additional context tokens needed
            
        Returns:
            TokenBudget with Bayesian allocation
        """
        # Base budget adjusted for complexity
        multiplier = self.complexity_multipliers[complexity]
        adjusted_budget = int(self.base_budget * multiplier) + context_size
        
        # Create budget
        budget = TokenBudget(
            total_budget=adjusted_budget,
            uncertainty_buffer=self._compute_uncertainty_buffer(complexity)
        )
        
        # Bayesian allocation based on learned usage patterns
        allocation = self._bayesian_allocation(complexity, adjusted_budget - budget.uncertainty_buffer)
        budget.allocated = allocation
        
        return budget
    
    def _bayesian_allocation(self, complexity: ComplexityLevel, available_budget: int) -> Dict[str, int]:
        """Allocate budget using Bayesian predictions."""
        allocation = {}
        total_predicted = 0
        
        # Get active protocols for this complexity level
        active_protocols = self._get_active_protocols(complexity)
        
        # Predict usage for each protocol using Bayesian inference
        predictions = {}
        for protocol in active_protocols:
            predicted_usage = self._predict_protocol_usage(protocol, complexity)
            predictions[protocol] = predicted_usage
            total_predicted += predicted_usage
        
        # Allocate proportionally with safety margins
        safety_margin = 1.2  # 20% buffer
        
        for protocol in active_protocols:
            if total_predicted > 0:
                proportion = predictions[protocol] / total_predicted
                allocated_tokens = int(available_budget * proportion * safety_margin)
            else:
                allocated_tokens = self.default_estimates.get(protocol, 100)
            
            allocation[protocol] = allocated_tokens
        
        return allocation
    
    def _predict_protocol_usage(self, protocol: str, complexity: ComplexityLevel) -> int:
        """Predict token usage for protocol using Bayesian inference."""
        
        if protocol not in self.protocol_params:
            # No history - use default with uncertainty
            base_estimate = self.default_estimates.get(protocol, 100)
            complexity_factor = self.complexity_multipliers[complexity]
            return int(base_estimate * complexity_factor)
        
        # Use learned Gamma distribution parameters
        alpha, beta = self.protocol_params[protocol]['alpha'], self.protocol_params[protocol]['beta']
        
        # Mean of Gamma distribution
        mean_usage = alpha / beta
        
        # Adjust for complexity
        complexity_factor = self.complexity_multipliers[complexity]
        predicted_usage = mean_usage * complexity_factor
        
        return max(10, int(predicted_usage))
    
    def _get_active_protocols(self, complexity: ComplexityLevel) -> List[str]:
        """Get list of active protocols for complexity level."""
        base_protocols = ['pattern_selection', 'feature_extraction', 'memory_backend', 'tool_execution']
        
        if complexity.value >= ComplexityLevel.FAST.value:
            base_protocols.extend(['temporal_windows'])
        
        if complexity.value >= ComplexityLevel.FULL.value:
            base_protocols.extend(['decision_engine', 'synthesis_bridge'])
        
        if complexity.value >= ComplexityLevel.RESEARCH.value:
            base_protocols.extend(['warpspace'])
        
        return base_protocols
    
    def _compute_uncertainty_buffer(self, complexity: ComplexityLevel) -> int:
        """Compute uncertainty buffer based on complexity."""
        base_buffer = 100
        complexity_factor = self.complexity_multipliers[complexity]
        return int(base_buffer * complexity_factor)
    
    def update_usage(self, protocol: str, tokens_used: int):
        """Update usage statistics for Bayesian learning."""
        
        if protocol not in self.protocol_usage_history:
            self.protocol_usage_history[protocol] = []
        
        self.protocol_usage_history[protocol].append(tokens_used)
        
        # Update Bayesian parameters if we have enough data
        if len(self.protocol_usage_history[protocol]) >= 3:
            self._update_bayesian_params(protocol)
    
    def _update_bayesian_params(self, protocol: str):
        """Update Bayesian parameters using observed data."""
        usage_data = self.protocol_usage_history[protocol]
        
        # Use method of moments to estimate Gamma parameters
        sample_mean = np.mean(usage_data)
        sample_var = np.var(usage_data)
        
        if sample_var > 0 and sample_mean > 0:
            # Method of moments for Gamma distribution
            beta = sample_mean / sample_var
            alpha = sample_mean * beta
            
            # Store parameters
            self.protocol_params[protocol] = {
                'alpha': max(0.1, alpha),
                'beta': max(0.001, beta)
            }
    
    def get_budget_analytics(self) -> Dict:
        """Get budget analytics and predictions."""
        analytics = {
            'total_protocols_tracked': len(self.protocol_usage_history),
            'learned_parameters': len(self.protocol_params),
            'average_accuracy': self._compute_prediction_accuracy(),
            'protocol_statistics': {}
        }
        
        for protocol, history in self.protocol_usage_history.items():
            analytics['protocol_statistics'][protocol] = {
                'samples': len(history),
                'mean_usage': np.mean(history),
                'std_usage': np.std(history),
                'has_learned_params': protocol in self.protocol_params
            }
        
        return analytics
    
    def _compute_prediction_accuracy(self) -> float:
        """Compute prediction accuracy for learned protocols."""
        if not self.protocol_params:
            return 0.0
        
        accuracies = []
        for protocol in self.protocol_params:
            if len(self.protocol_usage_history[protocol]) >= 5:
                # Use last 20% of data as test set
                history = self.protocol_usage_history[protocol]
                test_size = max(1, len(history) // 5)
                train_data = history[:-test_size]
                test_data = history[-test_size:]
                
                # Compute accuracy on test data
                predicted = self._predict_protocol_usage(protocol, ComplexityLevel.FAST)
                actual_mean = np.mean(test_data)
                
                if actual_mean > 0:
                    error = abs(predicted - actual_mean) / actual_mean
                    accuracy = max(0, 1 - error)
                    accuracies.append(accuracy)
        
        return np.mean(accuracies) if accuracies else 0.0


# ============================================================================
# DEMONSTRATION FUNCTION
# ============================================================================

async def demonstrate_bayesian_symphony():
    """Demonstrate all Bayesian approaches working together."""
    
    print("🎼 BAYESIAN SYMPHONY DEMONSTRATION 🎼")
    print("=" * 80)
    
    # 1. Variational WarpSpace
    print("\n1. 🌌 VARIATIONAL BAYESIAN WARPSPACE")
    print("-" * 50)
    
    vb_warpspace = VariationalWarpSpace()
    warp_result = await vb_warpspace.tensor_operation_variational(
        {'embeddings': np.random.randn(384).tolist()},
        ComplexityLevel.FULL,
        "topology"
    )
    
    print(f"   Epistemic Uncertainty: {warp_result['epistemic_uncertainty']:.4f}")
    print(f"   Total Uncertainty: {warp_result['total_uncertainty']:.4f}")
    print(f"   KL Divergence: {warp_result['kl_divergence']:.4f}")
    print(f"   Confidence: {warp_result['confidence']:.3f}")
    
    # 2. Bayesian Neural Networks
    print("\n2. 🧠 BAYESIAN NEURAL FEATURE EXTRACTION")
    print("-" * 50)
    
    bnn_extractor = BayesianFeatureExtractor()
    bnn_result = await bnn_extractor.extract_features_bayesian(
        np.random.randn(384).tolist(),
        ComplexityLevel.FULL
    )
    
    print(f"   Epistemic Uncertainty: {bnn_result['epistemic_uncertainty']:.4f}")
    print(f"   Predictive Entropy: {bnn_result['predictive_entropy']:.4f}")
    print(f"   Effective Parameters: {bnn_result['bayesian_metrics']['effective_num_params']:.1f}")
    print(f"   Confidence: {bnn_result['confidence']:.3f}")
    
    # 3. Hierarchical Bayesian Memory
    print("\n3. 🏗️ HIERARCHICAL BAYESIAN MEMORY")
    print("-" * 50)
    
    hb_memory = HierarchicalBayesianMemory()
    memory_result = await hb_memory.retrieve_hierarchical(
        "test query",
        threshold=0.6,
        complexity=ComplexityLevel.FULL
    )
    
    print(f"   Retrieved Shards: {len(memory_result['shards'])}")
    print(f"   Hierarchical Uncertainty: {memory_result['hierarchical_uncertainty']:.4f}")
    print(f"   Level Contributions: {memory_result['level_contributions']}")
    print(f"   Bayesian Evidence: {memory_result['bayesian_evidence']:.4f}")
    
    # 4. Empirical Bayesian Pattern Selection
    print("\n4. 🎯 EMPIRICAL BAYESIAN PATTERN SELECTION")
    print("-" * 50)
    
    eb_selector = EmpiricalBayesianPatternSelector()
    pattern_result = await eb_selector.select_pattern_empirical(
        "analyze the complex relationships",
        {'confidence': 0.8},
        ComplexityLevel.FULL
    )
    
    print(f"   Selected Pattern: {pattern_result['selected_pattern']}")
    print(f"   Selection Entropy: {pattern_result['selection_entropy']:.4f}")
    print(f"   Empirical Alpha: {pattern_result['empirical_hyperparams']['alpha']:.3f}")
    print(f"   Empirical Beta: {pattern_result['empirical_hyperparams']['beta']:.3f}")
    
    # 5. Bayesian Token Budgeting
    print("\n5. 💰 BAYESIAN TOKEN BUDGET MANAGEMENT")
    print("-" * 50)
    
    budget_manager = BayesianTokenBudgetManager()
    budget = budget_manager.create_complexity_budget(ComplexityLevel.FULL, context_size=500)
    
    print(f"   Total Budget: {budget.total_budget} tokens")
    print(f"   Uncertainty Buffer: {budget.uncertainty_buffer} tokens")
    print(f"   Allocated Protocols: {len(budget.allocated)}")
    print(f"   Remaining Budget: {budget.remaining()} tokens")
    
    # Simulate usage and learning
    budget_manager.update_usage('feature_extraction', 120)
    budget_manager.update_usage('memory_backend', 250)
    
    analytics = budget_manager.get_budget_analytics()
    print(f"   Tracked Protocols: {analytics['total_protocols_tracked']}")
    print(f"   Learned Parameters: {analytics['learned_parameters']}")
    
    print("\n" + "=" * 80)
    print("🎼 BAYESIAN SYMPHONY COMPLETE! 🎼")
    print("✨ Different Bayesian approaches at every layer! ✨")
    print("🎯 Thompson Sampling + MCTS + Variational + Hierarchical + Empirical!")
    print("💰 Plus intelligent token budgeting with Bayesian resource allocation!")


if __name__ == "__main__":
    asyncio.run(demonstrate_bayesian_symphony())
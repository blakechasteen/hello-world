"""
Learned Routing with Thompson Sampling

Uses multi-armed bandit to learn optimal backend selection from performance data.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ThompsonBandit:
    """
    Thompson Sampling multi-armed bandit for backend selection.
    
    Uses Beta distribution for each backend:
    - Alpha: successes + 1
    - Beta: failures + 1
    
    Naturally balances exploration (try new backends) vs exploitation (use best backend).
    """
    
    backends: List[str]
    alpha: np.ndarray = field(default=None)
    beta: np.ndarray = field(default=None)
    
    def __post_init__(self):
        """Initialize alpha/beta parameters."""
        n_backends = len(self.backends)
        if self.alpha is None:
            self.alpha = np.ones(n_backends, dtype=float)
        if self.beta is None:
            self.beta = np.ones(n_backends, dtype=float)
    
    def select(self) -> str:
        """
        Select backend using Thompson Sampling.
        
        Returns:
            Selected backend name
        """
        # Sample from Beta distribution for each backend
        samples = np.random.beta(self.alpha, self.beta)
        
        # Select backend with highest sample
        idx = np.argmax(samples)
        return self.backends[idx]
    
    def update(self, backend: str, success: bool):
        """
        Update bandit parameters based on outcome.
        
        Args:
            backend: Backend that was selected
            success: Whether the query was successful
        """
        idx = self.backends.index(backend)
        
        if success:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1
    
    def get_stats(self) -> Dict:
        """Get statistics for each backend."""
        stats = {}
        
        for i, backend in enumerate(self.backends):
            # Expected value of Beta distribution
            expected_success_rate = self.alpha[i] / (self.alpha[i] + self.beta[i])
            
            # Total observations
            total = (self.alpha[i] + self.beta[i] - 2)  # Subtract initial prior
            
            stats[backend] = {
                'expected_success_rate': float(expected_success_rate),
                'total_observations': int(total),
                'alpha': float(self.alpha[i]),
                'beta': float(self.beta[i])
            }
        
        return stats
    
    def save(self, path: Path):
        """Save bandit parameters to file."""
        data = {
            'backends': self.backends,
            'alpha': self.alpha.tolist(),
            'beta': self.beta.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'ThompsonBandit':
        """Load bandit parameters from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            backends=data['backends'],
            alpha=np.array(data['alpha']),
            beta=np.array(data['beta'])
        )


class LearnedRouter:
    """
    Learned routing policy using Thompson Sampling.
    
    Maintains separate bandits for each query type to learn specialized routing.
    """
    
    def __init__(
        self,
        backends: List[str],
        query_types: List[str],
        storage_path: Optional[Path] = None
    ):
        """
        Initialize learned router.
        
        Args:
            backends: List of available backends
            query_types: List of query types ('factual', 'analytical', 'creative', 'conversational')
            storage_path: Path to store bandit parameters
        """
        self.backends = backends
        self.query_types = query_types
        self.storage_path = storage_path or Path(__file__).parent / 'bandit_params.json'
        
        # Create bandit for each query type
        self.bandits: Dict[str, ThompsonBandit] = {}
        for query_type in query_types:
            self.bandits[query_type] = ThompsonBandit(backends=backends)
        
        # Load saved parameters if available
        self._load_params()
    
    def select_backend(self, query_type: str) -> str:
        """
        Select backend for a query using Thompson Sampling.
        
        Args:
            query_type: Type of query
        
        Returns:
            Selected backend name
        """
        # Default to 'conversational' if unknown type
        if query_type not in self.bandits:
            query_type = 'conversational'
        
        return self.bandits[query_type].select()
    
    def update(self, query_type: str, backend: str, success: bool):
        """
        Update routing policy based on outcome.
        
        Args:
            query_type: Type of query
            backend: Backend that was selected
            success: Whether the query was successful
        """
        # Default to 'conversational' if unknown type
        if query_type not in self.bandits:
            query_type = 'conversational'
        
        self.bandits[query_type].update(backend, success)
        
        # Save updated parameters
        self._save_params()
    
    def get_stats(self) -> Dict:
        """Get statistics for all query types and backends."""
        stats = {}
        for query_type, bandit in self.bandits.items():
            stats[query_type] = bandit.get_stats()
        return stats
    
    def _save_params(self):
        """Save all bandit parameters."""
        data = {}
        for query_type, bandit in self.bandits.items():
            data[query_type] = {
                'alpha': bandit.alpha.tolist(),
                'beta': bandit.beta.tolist()
            }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_params(self):
        """Load saved bandit parameters."""
        if not self.storage_path.exists():
            return
        
        with open(self.storage_path, 'r') as f:
            data = json.load(f)
        
        for query_type, params in data.items():
            if query_type in self.bandits:
                self.bandits[query_type].alpha = np.array(params['alpha'])
                self.bandits[query_type].beta = np.array(params['beta'])

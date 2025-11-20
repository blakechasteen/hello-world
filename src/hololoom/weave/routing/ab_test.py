"""
A/B Testing Framework for Routing Strategies

Compares multiple routing strategies (learned vs rule-based) to validate improvements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
import random
from datetime import datetime
import json
from pathlib import Path


@dataclass
class StrategyVariant:
    """Single strategy variant in an A/B test."""
    
    name: str
    weight: float  # Probability of selecting this variant (0.0-1.0)
    strategy_fn: Callable[[str, str], str]  # (query_type, complexity) -> backend
    
    # Outcome tracking
    total_queries: int = 0
    total_latency_ms: float = 0.0
    total_relevance: float = 0.0
    successes: int = 0
    
    def record_outcome(
        self,
        latency_ms: float,
        relevance_score: float,
        success: bool
    ):
        """Record outcome of a query routed by this variant."""
        self.total_queries += 1
        self.total_latency_ms += latency_ms
        self.total_relevance += relevance_score
        if success:
            self.successes += 1
    
    def get_metrics(self) -> Dict:
        """Get performance metrics for this variant."""
        if self.total_queries == 0:
            return {
                'total_queries': 0,
                'avg_latency_ms': 0.0,
                'avg_relevance': 0.0,
                'success_rate': 0.0
            }
        
        return {
            'total_queries': self.total_queries,
            'avg_latency_ms': self.total_latency_ms / self.total_queries,
            'avg_relevance': self.total_relevance / self.total_queries,
            'success_rate': self.successes / self.total_queries
        }


class ABTestRouter:
    """
    A/B testing router for comparing routing strategies.
    
    Randomly selects variants based on weights and tracks performance.
    """
    
    def __init__(
        self,
        variants: List[StrategyVariant],
        storage_path: Optional[Path] = None
    ):
        """
        Initialize A/B test router.
        
        Args:
            variants: List of strategy variants to test
            storage_path: Path to store test results
        """
        self.variants = variants
        self.storage_path = storage_path or Path(__file__).parent / 'ab_test_results.json'
        
        # Normalize weights
        total_weight = sum(v.weight for v in variants)
        for variant in variants:
            variant.weight /= total_weight
        
        # Load previous results if available
        self._load_results()
    
    def select_variant(self) -> StrategyVariant:
        """Select variant based on weights."""
        # Weighted random selection
        r = random.random()
        cumulative = 0.0
        
        for variant in self.variants:
            cumulative += variant.weight
            if r <= cumulative:
                return variant
        
        # Fallback to first variant
        return self.variants[0]
    
    def route(self, query_type: str, complexity: str) -> tuple[str, str]:
        """
        Route query and return (backend, variant_name).
        
        Args:
            query_type: Type of query
            complexity: Complexity level
        
        Returns:
            Tuple of (backend, variant_name)
        """
        variant = self.select_variant()
        backend = variant.strategy_fn(query_type, complexity)
        return backend, variant.name
    
    def record_outcome(
        self,
        variant_name: str,
        latency_ms: float,
        relevance_score: float,
        success: bool
    ):
        """
        Record outcome for a specific variant.
        
        Args:
            variant_name: Name of variant that was selected
            latency_ms: Query latency
            relevance_score: Relevance score (0.0-1.0)
            success: Whether query was successful
        """
        for variant in self.variants:
            if variant.name == variant_name:
                variant.record_outcome(latency_ms, relevance_score, success)
                break
        
        # Save updated results
        self._save_results()
    
    def get_results(self) -> Dict:
        """Get A/B test results for all variants."""
        results = {}
        for variant in self.variants:
            results[variant.name] = variant.get_metrics()
        return results
    
    def get_winner(self) -> str:
        """
        Get the variant with the best performance.
        
        Uses composite score: 0.4*relevance + 0.3*success_rate - 0.3*normalized_latency
        """
        if not self.variants or all(v.total_queries == 0 for v in self.variants):
            return None
        
        # Calculate composite scores
        scores = {}
        max_latency = max(
            v.total_latency_ms / v.total_queries
            for v in self.variants
            if v.total_queries > 0
        )
        
        for variant in self.variants:
            if variant.total_queries == 0:
                continue
            
            metrics = variant.get_metrics()
            normalized_latency = metrics['avg_latency_ms'] / max_latency if max_latency > 0 else 0
            
            score = (
                0.4 * metrics['avg_relevance'] +
                0.3 * metrics['success_rate'] -
                0.3 * normalized_latency
            )
            
            scores[variant.name] = score
        
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _save_results(self):
        """Save A/B test results to file."""
        data = {
            'variants': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for variant in self.variants:
            data['variants'][variant.name] = {
                'weight': variant.weight,
                'total_queries': variant.total_queries,
                'total_latency_ms': variant.total_latency_ms,
                'total_relevance': variant.total_relevance,
                'successes': variant.successes,
                'metrics': variant.get_metrics()
            }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_results(self):
        """Load previous A/B test results."""
        if not self.storage_path.exists():
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for variant in self.variants:
                if variant.name in data['variants']:
                    variant_data = data['variants'][variant.name]
                    variant.total_queries = variant_data['total_queries']
                    variant.total_latency_ms = variant_data['total_latency_ms']
                    variant.total_relevance = variant_data['total_relevance']
                    variant.successes = variant_data['successes']
        except Exception:
            pass  # Ignore errors loading previous results

"""
Routing Integration for WeavingOrchestrator

Adds learned routing capabilities to the orchestrator.
"""

from typing import Optional, Dict
from pathlib import Path
import time

from .learned import LearnedRouter
from .metrics import RoutingMetrics, MetricsCollector
from .ab_test import ABTestRouter, StrategyVariant


def classify_query_type(query_text: str) -> str:
    """
    Classify query into one of four types.
    
    Args:
        query_text: Query text
    
    Returns:
        Query type: 'factual', 'analytical', 'creative', 'conversational'
    """
    query_lower = query_text.lower()
    
    # Factual: who/what/when/where questions
    if any(word in query_lower for word in ['who is', 'what is', 'when did', 'where is', 'define']):
        return 'factual'
    
    # Analytical: why/how questions, comparisons
    if any(word in query_lower for word in ['why', 'how', 'compare', 'analyze', 'explain']):
        return 'analytical'
    
    # Creative: imagine, create, design
    if any(word in query_lower for word in ['imagine', 'create', 'design', 'invent', 'story']):
        return 'creative'
    
    # Default: conversational
    return 'conversational'


def rule_based_routing(query_type: str, complexity: str) -> str:
    """
    Original rule-based routing strategy.
    
    Args:
        query_type: Type of query
        complexity: Complexity level
    
    Returns:
        Backend name
    """
    # Simple rules based on complexity
    if complexity == 'RESEARCH':
        return 'HYBRID'  # Use all capabilities for deep research
    elif complexity == 'FULL':
        return 'NEO4J_QDRANT'  # Graph + vector for full queries
    elif complexity == 'FAST':
        return 'NETWORKX'  # In-memory for fast queries
    else:  # LITE
        return 'NETWORKX'  # Lightweight for simple queries


class RoutingOrchestrator:
    """
    Intelligent routing orchestrator.
    
    Wraps learned routing with metrics collection and A/B testing.
    """
    
    def __init__(
        self,
        backends: list[str],
        query_types: list[str],
        enable_ab_test: bool = False,
        storage_dir: Optional[Path] = None
    ):
        """
        Initialize routing orchestrator.
        
        Args:
            backends: Available memory backends
            query_types: Query types to learn
            enable_ab_test: Whether to run A/B testing
            storage_dir: Directory to store routing data
        """
        self.backends = backends
        self.query_types = query_types
        self.enable_ab_test = enable_ab_test
        
        # Storage paths
        if storage_dir is None:
            storage_dir = Path(__file__).parent
        
        # Create learned router
        self.learned_router = LearnedRouter(
            backends=backends,
            query_types=query_types,
            storage_path=storage_dir / 'bandit_params.json'
        )
        
        # Create metrics collector
        self.metrics_collector = MetricsCollector(
            storage_path=storage_dir / 'metrics.jsonl'
        )
        
        # Create A/B test router if enabled
        self.ab_test_router = None
        if enable_ab_test:
            # Define strategy variants
            variants = [
                StrategyVariant(
                    name='learned',
                    weight=0.5,
                    strategy_fn=lambda qt, c: self.learned_router.select_backend(qt)
                ),
                StrategyVariant(
                    name='rule_based',
                    weight=0.5,
                    strategy_fn=rule_based_routing
                )
            ]
            
            self.ab_test_router = ABTestRouter(
                variants=variants,
                storage_path=storage_dir / 'ab_test_results.json'
            )
    
    def select_backend(self, query: str, complexity: str) -> tuple[str, str]:
        """
        Select backend for query.
        
        Args:
            query: Query text
            complexity: Complexity level
        
        Returns:
            Tuple of (backend, strategy_used)
        """
        query_type = classify_query_type(query)
        
        if self.enable_ab_test and self.ab_test_router:
            # Use A/B testing
            backend, strategy = self.ab_test_router.route(query_type, complexity)
        else:
            # Use learned routing directly
            backend = self.learned_router.select_backend(query_type)
            strategy = 'learned'
        
        return backend, strategy
    
    def record_outcome(
        self,
        query: str,
        complexity: str,
        backend_selected: str,
        strategy_used: str,
        latency_ms: float,
        relevance_score: float,
        confidence: float,
        success: bool,
        memory_size: int = 0
    ):
        """
        Record routing outcome and update models.
        
        Args:
            query: Query text
            complexity: Complexity level
            backend_selected: Backend that was selected
            strategy_used: Strategy that was used ('learned' or 'rule_based')
            latency_ms: Query latency
            relevance_score: Relevance score (0.0-1.0)
            confidence: Confidence score (0.0-1.0)
            success: Whether query was successful
            memory_size: Number of shards in memory
        """
        query_type = classify_query_type(query)
        
        # Record metrics
        metrics = RoutingMetrics(
            query=query,
            query_type=query_type,
            complexity=complexity,
            backend_selected=backend_selected,
            latency_ms=latency_ms,
            relevance_score=relevance_score,
            confidence=confidence,
            success=success,
            memory_size=memory_size
        )
        self.metrics_collector.record(metrics)
        
        # Update learned router (always, for continuous learning)
        self.learned_router.update(query_type, backend_selected, success)
        
        # Update A/B test if enabled
        if self.enable_ab_test and self.ab_test_router:
            self.ab_test_router.record_outcome(
                variant_name=strategy_used,
                latency_ms=latency_ms,
                relevance_score=relevance_score,
                success=success
            )
    
    def get_routing_stats(self) -> Dict:
        """Get comprehensive routing statistics."""
        stats = {
            'learned_router': self.learned_router.get_stats(),
            'backend_stats': self.metrics_collector.get_all_stats(),
            'total_queries': len(self.metrics_collector.metrics)
        }
        
        if self.enable_ab_test and self.ab_test_router:
            stats['ab_test'] = {
                'results': self.ab_test_router.get_results(),
                'winner': self.ab_test_router.get_winner()
            }
        
        return stats

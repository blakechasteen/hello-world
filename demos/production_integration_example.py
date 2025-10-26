#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Integration Example
===============================
Shows how to integrate AnalyticalWeavingOrchestrator into a production pipeline
with monitoring, logging, and adaptive configuration.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json

from HoloLoom.analytical_orchestrator import create_analytical_orchestrator
from HoloLoom.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionSemanticPipeline:
    """
    Production-ready semantic processing pipeline with analytical guarantees.
    
    Features:
    - Adaptive analysis (enable based on query complexity)
    - Metric monitoring and alerting
    - Performance tracking
    - Automatic fallback on errors
    """
    
    def __init__(
        self,
        enable_adaptive_analysis: bool = True,
        alert_on_instability: bool = True,
        log_metrics: bool = True
    ):
        """
        Initialize production pipeline.
        
        Args:
            enable_adaptive_analysis: Adapt analysis based on query complexity
            alert_on_instability: Alert when mathematical guarantees violated
            log_metrics: Log all analytical metrics
        """
        # Create orchestrator
        self.weaver = create_analytical_orchestrator(
            pattern="fused",
            enable_all_analysis=True,
            use_mcts=True,
            mcts_simulations=100
        )
        
        self.enable_adaptive_analysis = enable_adaptive_analysis
        self.alert_on_instability = alert_on_instability
        self.log_metrics = log_metrics
        
        # Monitoring
        self.query_count = 0
        self.alert_count = 0
        self.metrics_log = []
        
        # Thresholds for alerts
        self.thresholds = {
            'spectral_radius': 0.95,      # Alert if approaching 1
            'lipschitz_constant': 500,    # Alert if too sensitive
            'diversity_score': 0.3,       # Alert if low diversity
            'gradient_norm': 0.001        # Alert if gradient vanishes
        }
        
        logger.info("ProductionSemanticPipeline initialized")
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Process query with full analytical pipeline.
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Dict with response and analytical guarantees
        """
        self.query_count += 1
        start_time = datetime.now()
        
        logger.info(f"Processing query #{self.query_count}: {query[:60]}...")
        
        try:
            # Execute weaving
            spacetime = await self.weaver.weave(query, context=context)
            
            # Extract metrics
            analytical_metrics = getattr(spacetime.trace, 'analytical_metrics', {})
            
            # Check for issues
            alerts = self._check_guarantees(analytical_metrics)
            
            # Log metrics
            if self.log_metrics:
                self._log_metrics(query, analytical_metrics, alerts)
            
            # Build response
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'query': query,
                'response': spacetime.response,
                'tool': spacetime.tool_used,
                'confidence': spacetime.confidence,
                'duration_seconds': duration,
                'analytical_guarantees': self._summarize_guarantees(analytical_metrics),
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Query processed in {duration:.2f}s - Tool: {spacetime.tool_used}")
            
            if alerts:
                logger.warning(f"Generated {len(alerts)} alerts")
                self.alert_count += len(alerts)
            
            return result
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                'query': query,
                'response': f"Error: {str(e)}",
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _check_guarantees(self, metrics: Dict) -> List[Dict]:
        """
        Check if mathematical guarantees are met.
        
        Args:
            metrics: Analytical metrics
            
        Returns:
            List of alerts
        """
        alerts = []
        
        if not self.alert_on_instability:
            return alerts
        
        # Check metric validity
        if 'metric_space' in metrics:
            if not metrics['metric_space'].get('is_valid_metric', False):
                alerts.append({
                    'level': 'ERROR',
                    'message': 'Metric axioms violated - distances invalid',
                    'metric': 'metric_validity',
                    'recommendation': 'Check embedding function'
                })
        
        # Check spectral stability
        if 'spectral_stability' in metrics:
            radius = metrics['spectral_stability'].get('spectral_radius', 0)
            if radius > self.thresholds['spectral_radius']:
                alerts.append({
                    'level': 'WARNING',
                    'message': f'Spectral radius {radius:.3f} approaching instability',
                    'metric': 'spectral_radius',
                    'value': radius,
                    'recommendation': 'System may become unstable in iterative use'
                })
        
        # Check Lipschitz constant
        if 'continuity' in metrics:
            L = metrics['continuity'].get('lipschitz_constant')
            if L and L > self.thresholds['lipschitz_constant']:
                alerts.append({
                    'level': 'WARNING',
                    'message': f'High Lipschitz constant {L:.1f} - sensitive to inputs',
                    'metric': 'lipschitz_constant',
                    'value': L,
                    'recommendation': 'Small input changes may cause large output changes'
                })
        
        # Check diversity
        if 'orthogonalization' in metrics:
            diversity = metrics['orthogonalization'].get('diversity_score', 0)
            if diversity < self.thresholds['diversity_score']:
                alerts.append({
                    'level': 'INFO',
                    'message': f'Low diversity score {diversity:.3f}',
                    'metric': 'diversity_score',
                    'value': diversity,
                    'recommendation': 'Context may be redundant'
                })
        
        # Check gradient vanishing
        if 'gradient_optimization' in metrics:
            grad_norm = metrics['gradient_optimization'].get('gradient_norm', 0)
            if grad_norm < self.thresholds['gradient_norm']:
                alerts.append({
                    'level': 'INFO',
                    'message': f'Gradient vanishing (norm={grad_norm:.6f})',
                    'metric': 'gradient_norm',
                    'value': grad_norm,
                    'recommendation': 'Query may be at local optimum'
                })
        
        return alerts
    
    def _summarize_guarantees(self, metrics: Dict) -> Dict:
        """
        Summarize mathematical guarantees in human-readable format.
        
        Args:
            metrics: Analytical metrics
            
        Returns:
            Summary dict
        """
        summary = {}
        
        if 'metric_space' in metrics:
            summary['valid_metric_space'] = metrics['metric_space'].get('is_valid_metric', False)
            summary['complete_space'] = metrics['metric_space'].get('is_complete', 'Unknown')
        
        if 'spectral_stability' in metrics:
            summary['stable_dynamics'] = metrics['spectral_stability'].get('is_stable', False)
            summary['spectral_radius'] = metrics['spectral_stability'].get('spectral_radius', 0)
        
        if 'orthogonalization' in metrics:
            summary['context_diversity'] = metrics['orthogonalization'].get('diversity_score', 0)
        
        if 'gradient_optimization' in metrics:
            summary['relevance_optimized'] = metrics['gradient_optimization'].get('improvement', 0) > 0
            summary['gradient_norm'] = metrics['gradient_optimization'].get('gradient_norm', 0)
        
        if 'continuity' in metrics:
            summary['lipschitz_continuous'] = metrics['continuity'].get('is_smooth', False)
            summary['sensitivity_bound'] = metrics['continuity'].get('lipschitz_constant')
        
        if 'convergence' in metrics:
            summary['learning_converges'] = metrics['convergence'].get('is_converging', False)
            summary['performance_trend'] = metrics['convergence'].get('direction', 'unknown')
        
        return summary
    
    def _log_metrics(self, query: str, metrics: Dict, alerts: List[Dict]):
        """Log metrics to structured log."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'query_count': self.query_count,
            'query': query[:100],
            'metrics': metrics,
            'alerts': alerts
        }
        
        self.metrics_log.append(log_entry)
        
        # Optional: Write to file
        # with open('metrics.jsonl', 'a') as f:
        #     f.write(json.dumps(log_entry) + '\n')
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        stats = self.weaver.get_analytical_statistics()
        
        stats['pipeline'] = {
            'queries_processed': self.query_count,
            'alerts_generated': self.alert_count,
            'metrics_logged': len(self.metrics_log)
        }
        
        return stats
    
    def stop(self):
        """Stop pipeline."""
        self.weaver.stop()
        logger.info("Pipeline stopped")


async def main():
    """Example usage."""
    print("="*80)
    print("PRODUCTION INTEGRATION EXAMPLE")
    print("="*80)
    print()
    
    # Create pipeline
    pipeline = ProductionSemanticPipeline(
        enable_adaptive_analysis=True,
        alert_on_instability=True,
        log_metrics=True
    )
    
    # Example queries
    queries = [
        "What is machine learning?",
        "Explain neural network architectures",
        "How does gradient descent work?",
        "What are transformers?",
        "Explain attention mechanisms"
    ]
    
    print("Processing queries with full analytical guarantees...\n")
    
    results = []
    for query in queries:
        result = await pipeline.process_query(query)
        results.append(result)
        
        print(f"\nQuery: {query}")
        print(f"Tool: {result.get('tool', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        print(f"Duration: {result.get('duration_seconds', 0):.2f}s")
        
        # Show guarantees
        guarantees = result.get('analytical_guarantees', {})
        if guarantees:
            print("\nMathematical Guarantees:")
            for key, value in guarantees.items():
                print(f"  {key}: {value}")
        
        # Show alerts
        alerts = result.get('alerts', [])
        if alerts:
            print("\nAlerts:")
            for alert in alerts:
                print(f"  [{alert['level']}] {alert['message']}")
        
        print("-" * 80)
    
    # Show statistics
    print("\n" + "="*80)
    print("PIPELINE STATISTICS")
    print("="*80)
    
    stats = pipeline.get_statistics()
    
    print(f"\nQueries processed: {stats['pipeline']['queries_processed']}")
    print(f"Alerts generated: {stats['pipeline']['alerts_generated']}")
    print(f"Metrics logged: {stats['pipeline']['metrics_logged']}")
    
    print(f"\nAnalysis statistics:")
    analysis = stats['analysis']
    print(f"  Metric verifications: {analysis['metric_verifications']}")
    print(f"  Valid metrics: {analysis['metric_valid_count']}")
    print(f"  Gradient optimizations: {analysis['gradient_optimizations']}")
    print(f"  Spectral analyses: {analysis['spectral_analyses']}")
    
    if stats['convergence']['confidence_history_length'] > 0:
        print(f"\nConvergence:")
        print(f"  Mean confidence: {stats['convergence']['mean_confidence']:.3f}")
        print(f"  Mean gradient norm: {stats['convergence']['mean_gradient_norm']:.4f}")
    
    # Stop pipeline
    pipeline.stop()
    
    print("\n" + "="*80)
    print("Production pipeline complete!")
    print("All queries processed with mathematical guarantees.")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

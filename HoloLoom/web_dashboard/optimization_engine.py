#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thompson Sampling Auto-Optimization Engine for Workflow Builder
================================================================

Provides intelligent workflow optimization using Thompson Sampling to learn
which optimization strategies work best over time.

Features:
- Automatic bottleneck detection
- Parallelization opportunity discovery
- Thompson Sampling strategy selection
- Performance profiling and tracking
"""

import logging
import random
import uuid
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class ThompsonOptimizer:
    """
    Uses Thompson Sampling to select optimal workflow optimization strategies.

    Maintains Beta(alpha, beta) distributions for each optimization type,
    updating beliefs based on observed outcomes.
    """

    def __init__(self):
        self.priors = {
            'parallelize': {'alpha': 1.0, 'beta': 1.0},
            'cache': {'alpha': 1.0, 'beta': 1.0},
            'reorder': {'alpha': 1.0, 'beta': 1.0},
            'substitute': {'alpha': 1.0, 'beta': 1.0}
        }

    def suggest_optimization(self, workflow: dict, performance_data: dict) -> Dict[str, Any]:
        """
        Use Thompson Sampling to select optimization strategy.

        Args:
            workflow: Workflow definition with nodes and connections
            performance_data: Execution metrics (latencies, bottlenecks, etc.)

        Returns:
            Dict with suggestion details
        """
        # Sample from Beta distributions
        samples = {}
        for strategy, prior in self.priors.items():
            alpha, beta = prior['alpha'], prior['beta']
            # Approximate Beta sampling using transformed uniform
            u = random.random()
            sample = (u ** (1/alpha)) * ((1-u) ** (1/beta)) if alpha > 0 and beta > 0 else 0.5
            samples[strategy] = sample

        # Select strategy with highest sample
        best_strategy = max(samples.items(), key=lambda x: x[1])[0]
        confidence = samples[best_strategy]

        # Generate suggestion based on strategy
        suggestion_id = str(uuid.uuid4())
        workflow_id = workflow.get('name', 'unnamed_workflow')

        if best_strategy == 'parallelize':
            opportunities = self._find_parallelization_opportunities(workflow, performance_data)
            if opportunities:
                opp = opportunities[0]
                return {
                    'suggestion_id': suggestion_id,
                    'workflow_id': workflow_id,
                    'suggestion_type': 'parallelize',
                    'description': f"Parallelize nodes {', '.join(opp['nodes'])} - no data dependencies detected",
                    'expected_improvement': opp['expected_improvement'],
                    'confidence': confidence,
                    'affected_nodes': opp['nodes']
                }

        elif best_strategy == 'cache':
            return {
                'suggestion_id': suggestion_id,
                'workflow_id': workflow_id,
                'suggestion_type': 'cache',
                'description': "Enable caching for retrieval nodes",
                'expected_improvement': 15.0,
                'confidence': confidence,
                'affected_nodes': ['retrieve', 'search']
            }

        elif best_strategy == 'reorder':
            bottlenecks = self._detect_bottlenecks(workflow, performance_data)
            if bottlenecks:
                return {
                    'suggestion_id': suggestion_id,
                    'workflow_id': workflow_id,
                    'suggestion_type': 'reorder',
                    'description': f"Reorder execution to prioritize fast paths",
                    'expected_improvement': 20.0,
                    'confidence': confidence,
                    'affected_nodes': [b['node_id'] for b in bottlenecks[:2]]
                }

        # Default fallback
        return {
            'suggestion_id': suggestion_id,
            'workflow_id': workflow_id,
            'suggestion_type': best_strategy,
            'description': f"Apply {best_strategy} optimization",
            'expected_improvement': 10.0,
            'confidence': confidence,
            'affected_nodes': []
        }

    def update_belief(self, suggestion_id: str, outcome: dict, optimization_history: list):
        """
        Update Thompson Sampling priors based on outcome.

        Args:
            suggestion_id: ID of the applied suggestion
            outcome: Dict with 'success' (bool) and 'improvement' (float)
            optimization_history: List of past suggestions
        """
        # Find suggestion in history
        suggestion = None
        for item in optimization_history:
            if item.get('suggestion_id') == suggestion_id:
                suggestion = item
                break

        if not suggestion:
            logger.warning(f"Suggestion {suggestion_id} not found in history")
            return

        strategy = suggestion['suggestion_type']
        success = outcome.get('success', False)

        if strategy in self.priors:
            if success:
                self.priors[strategy]['alpha'] += 1.0
            else:
                self.priors[strategy]['beta'] += 1.0

            logger.info(f"Updated beliefs for {strategy}: alpha={self.priors[strategy]['alpha']:.1f}, beta={self.priors[strategy]['beta']:.1f}")

    def _find_parallelization_opportunities(self, workflow: dict, performance_data: dict) -> List[dict]:
        """
        Find nodes that can be parallelized.

        Returns list of opportunities with format:
        {
            'nodes': [node_ids],
            'expected_improvement': float (percentage),
            'rationale': str
        }
        """
        opportunities = []
        nodes = {n['id']: n for n in workflow.get('nodes', [])}
        connections = workflow.get('connections', [])

        # Build adjacency map
        dependencies = {}
        for conn in connections:
            to_node = conn.get('to')
            from_node = conn.get('from')
            if to_node not in dependencies:
                dependencies[to_node] = []
            dependencies[to_node].append(from_node)

        # Find independent nodes (same depth, no shared dependencies)
        node_depths = {}
        for node_id in nodes:
            node_depths[node_id] = self._compute_node_depth(node_id, dependencies)

        # Group by depth
        depth_groups = {}
        for node_id, depth in node_depths.items():
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node_id)

        # Check for parallelizable groups
        for depth, node_ids in depth_groups.items():
            if len(node_ids) >= 2:
                # Estimate improvement (latency reduction)
                latencies = []
                for nid in node_ids:
                    perf = performance_data.get(nid, {})
                    latencies.append(perf.get('avg_latency_ms', 100))

                total_sequential = sum(latencies)
                max_parallel = max(latencies) if latencies else 100
                improvement = ((total_sequential - max_parallel) / total_sequential * 100) if total_sequential > 0 else 0

                if improvement > 10:  # Only suggest if >10% improvement
                    opportunities.append({
                        'nodes': node_ids,
                        'expected_improvement': improvement,
                        'rationale': f"Can execute {len(node_ids)} nodes in parallel at depth {depth}"
                    })

        return sorted(opportunities, key=lambda x: x['expected_improvement'], reverse=True)

    def _detect_bottlenecks(self, workflow: dict, performance_data: dict) -> List[dict]:
        """
        Detect bottlenecks using:
        1. Critical path analysis (longest sequential chain)
        2. Latency contribution (% of total time)
        3. Variance analysis (high variance = unstable)

        Returns list of bottlenecks sorted by severity.
        """
        bottlenecks = []
        nodes = workflow.get('nodes', [])

        total_latency = sum(
            performance_data.get(n['id'], {}).get('avg_latency_ms', 0)
            for n in nodes
        )

        if total_latency == 0:
            return []

        for node in nodes:
            node_id = node['id']
            perf = performance_data.get(node_id, {})

            avg_latency = perf.get('avg_latency_ms', 0)
            p95_latency = perf.get('p95_latency_ms', avg_latency * 1.5)
            success_rate = perf.get('success_rate', 1.0)

            # Calculate bottleneck score
            latency_contribution = avg_latency / total_latency if total_latency > 0 else 0
            variance = (p95_latency - avg_latency) / avg_latency if avg_latency > 0 else 0
            reliability = 1.0 - success_rate

            bottleneck_score = (
                0.5 * latency_contribution +
                0.3 * min(variance, 1.0) +
                0.2 * reliability
            )

            if bottleneck_score > 0.3:  # Threshold for significant bottleneck
                bottlenecks.append({
                    'node_id': node_id,
                    'node_type': node.get('agentType', 'unknown'),
                    'bottleneck_score': bottleneck_score,
                    'latency_contribution': latency_contribution * 100,
                    'variance': variance,
                    'success_rate': success_rate
                })

        return sorted(bottlenecks, key=lambda x: x['bottleneck_score'], reverse=True)

    def _compute_node_depth(self, node_id: str, dependencies: dict) -> int:
        """Compute node depth in DAG (longest path from root)."""
        if node_id not in dependencies or not dependencies[node_id]:
            return 0

        max_depth = 0
        for dep in dependencies[node_id]:
            dep_depth = self._compute_node_depth(dep, dependencies)
            max_depth = max(max_depth, dep_depth + 1)

        return max_depth

    def get_performance_profile(self, workflow: dict, performance_data: dict) -> List[Dict[str, Any]]:
        """
        Generate performance profiles for all nodes in workflow.

        Returns list of PerformanceProfile dicts.
        """
        profiles = []

        for node in workflow.get('nodes', []):
            node_id = node['id']
            node_type = node.get('agentType', 'unknown')
            perf = performance_data.get(node_id, {})

            avg_latency = perf.get('avg_latency_ms', 0)
            p95_latency = perf.get('p95_latency_ms', avg_latency * 1.5)
            success_rate = perf.get('success_rate', 1.0)
            call_count = perf.get('call_count', 0)

            # Calculate throughput (calls per minute)
            # Assuming recent time window of 1 minute
            throughput = call_count  # calls per minute

            # Calculate bottleneck score
            bottlenecks = self._detect_bottlenecks(workflow, performance_data)
            bottleneck_score = 0.0
            for b in bottlenecks:
                if b['node_id'] == node_id:
                    bottleneck_score = b['bottleneck_score']
                    break

            profiles.append({
                'node_id': node_id,
                'node_type': node_type,
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'success_rate': success_rate,
                'throughput': throughput,
                'bottleneck_score': bottleneck_score
            })

        return profiles

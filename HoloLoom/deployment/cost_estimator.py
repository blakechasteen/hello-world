#!/usr/bin/env python3
"""
Cost Estimator
=============

Estimates monthly deployment costs for different platforms.

Considers:
- Compute costs (dyno/VM/function)
- Database costs
- Storage costs
- LLM API costs (estimated)
- Network costs

Created: November 2025
"""

from typing import Dict, Any


class CostEstimator:
    """
    Estimate monthly deployment costs.

    Provides realistic cost estimates for each platform based on typical usage.
    """

    # Platform base costs (monthly USD)
    PLATFORM_COSTS = {
        'heroku': {
            'compute': 7,      # Hobby dyno
            'database': 9,     # Postgres Basic
            'storage': 0,
            'name': 'Heroku Hobby'
        },
        'railway': {
            'compute': 5,      # Starter plan
            'database': 0,     # Included
            'storage': 0,
            'name': 'Railway Starter'
        },
        'fly': {
            'compute': 1.94,   # shared-cpu-1x
            'database': 0,     # Free tier
            'storage': 0.15,   # Per GB
            'name': 'Fly.io Shared CPU'
        },
        'aws_lambda': {
            'compute': 0.20,   # Per 1M requests
            'database': 0,     # On-demand
            'storage': 0.023,  # Per GB
            'name': 'AWS Lambda'
        },
        'local': {
            'compute': 0,      # Free
            'database': 0,     # Free
            'storage': 0,      # Free
            'name': 'Local Docker'
        }
    }

    # Workflow complexity multipliers
    WORKFLOW_COMPLEXITY = {
        'simple': 1.0,      # Basic automation (email triage)
        'moderate': 1.5,    # Multi-step workflows (report generation)
        'complex': 2.5,     # Advanced workflows (code review)
    }

    def estimate(self, platform: str, workflow_id: str) -> Dict[str, Any]:
        """
        Estimate monthly cost for deploying a workflow.

        Args:
            platform: Deployment platform
            workflow_id: Workflow ID

        Returns:
            Cost estimate with breakdown
        """
        # Get base platform costs
        base_costs = self.PLATFORM_COSTS.get(platform.lower(), self.PLATFORM_COSTS['local'])

        # Determine workflow complexity
        complexity = self._determine_complexity(workflow_id)
        complexity_multiplier = self.WORKFLOW_COMPLEXITY.get(complexity, 1.0)

        # Calculate component costs
        monthly_base = base_costs['compute']
        monthly_database = base_costs['database']
        monthly_storage = base_costs['storage']
        monthly_llm = self._estimate_llm_costs(workflow_id, complexity)

        # Apply complexity multiplier to compute costs
        monthly_compute = monthly_base * complexity_multiplier

        # Total monthly cost
        monthly_total = monthly_compute + monthly_database + monthly_storage + monthly_llm

        return {
            'platform': platform,
            'workflow_id': workflow_id,
            'complexity': complexity,
            'monthly_base': monthly_base,
            'monthly_compute': monthly_compute,
            'monthly_database': monthly_database,
            'monthly_storage': monthly_storage,
            'monthly_llm': monthly_llm,
            'monthly_total': monthly_total,
            'breakdown': {
                'Compute': f'${monthly_compute:.2f}',
                'Database': f'${monthly_database:.2f}',
                'Storage': f'${monthly_storage:.2f}',
                'LLM API': f'${monthly_llm:.2f}',
            },
            'notes': self._get_cost_notes(platform, monthly_total)
        }

    def _determine_complexity(self, workflow_id: str) -> str:
        """Determine workflow complexity based on ID."""
        # Simple heuristic based on common workflow patterns
        simple_workflows = ['inbox-triage', 'email-digest', 'calendar-optimization']
        complex_workflows = ['code-review', 'bug-triage', 'security-scan']

        if any(s in workflow_id for s in simple_workflows):
            return 'simple'
        elif any(c in workflow_id for c in complex_workflows):
            return 'complex'
        else:
            return 'moderate'

    def _estimate_llm_costs(self, workflow_id: str, complexity: str) -> float:
        """
        Estimate monthly LLM API costs.

        Based on typical usage patterns:
        - Simple: 100 requests/day @ $0.001/request = $3/month
        - Moderate: 200 requests/day @ $0.002/request = $12/month
        - Complex: 500 requests/day @ $0.003/request = $45/month
        """
        llm_costs = {
            'simple': 3.0,
            'moderate': 12.0,
            'complex': 45.0
        }

        return llm_costs.get(complexity, 12.0)

    def _get_cost_notes(self, platform: str, monthly_total: float) -> list:
        """Get platform-specific cost notes."""
        notes = []

        if platform == 'heroku':
            notes.append("Free tier available (512MB RAM, no DB)")
            notes.append("Hobby tier recommended for light production use")
            notes.append(f"Estimated annual cost: ${monthly_total * 12:.2f}")

        elif platform == 'railway':
            notes.append("$5/month free credit (covers starter plan)")
            notes.append("Great for side projects and MVPs")
            notes.append("Upgrade to Team ($20/month) for production")

        elif platform == 'fly':
            notes.append("Free tier: 3 shared-cpu-1x VMs")
            notes.append("Very cost-effective for global deployment")
            notes.append("Scale up to dedicated-cpu ($29/month) for high traffic")

        elif platform == 'aws_lambda':
            notes.append("Pay per use (no idle costs)")
            notes.append("Free tier: 1M requests/month")
            notes.append("Best for intermittent workflows")

        elif platform == 'local':
            notes.append("No cloud costs")
            notes.append("Uses your local machine resources")
            notes.append("Perfect for development and testing")

        return notes

    def compare_platforms(self, workflow_id: str) -> Dict[str, Any]:
        """
        Compare costs across all platforms for a workflow.

        Returns:
            Cost comparison table
        """
        comparisons = {}

        for platform in self.PLATFORM_COSTS.keys():
            estimate = self.estimate(platform, workflow_id)
            comparisons[platform] = {
                'monthly_total': estimate['monthly_total'],
                'annual_total': estimate['monthly_total'] * 12,
                'breakdown': estimate['breakdown'],
                'notes': estimate['notes']
            }

        # Sort by cost (ascending)
        sorted_platforms = sorted(
            comparisons.items(),
            key=lambda x: x[1]['monthly_total']
        )

        return {
            'workflow_id': workflow_id,
            'platforms': dict(sorted_platforms),
            'cheapest': sorted_platforms[0][0],
            'most_expensive': sorted_platforms[-1][0],
            'cost_range': {
                'min': sorted_platforms[0][1]['monthly_total'],
                'max': sorted_platforms[-1][1]['monthly_total']
            }
        }

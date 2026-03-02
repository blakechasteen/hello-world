"""
Cross-Department Workflow Examples

This package provides real-world examples of multi-department coordination
using the HoloLoom B2B Framework.

Available Workflows:
-------------------
1. Research & Analysis Pipeline
   - Context enrichment → RAG retrieval → Planning decomposition
   - Use case: Complex research with implementation planning

2. Deployment with Health Monitoring
   - Permission validation → Deployment → Health checks → Audit trail
   - Use case: Production deployments with rollback capability

3. Intelligent Query Routing
   - Context analysis → Auto routing → Learning from outcomes
   - Use case: Adaptive query handling

4. Performance Monitoring & Auto-Scaling
   - Resource monitoring → Performance analysis → Auto-scaling
   - Use case: Infrastructure management

5. Complete B2B Customer Onboarding
   - Profile creation → Planning → Provisioning → Knowledge base setup
   - Use case: Enterprise customer onboarding

Usage:
------
    from hololoom.apps.departments.examples.workflow_examples import (
        research_workflow_example,
        deployment_workflow_example,
        intelligent_routing_workflow_example,
        monitoring_scaling_workflow_example,
        customer_onboarding_workflow_example,
    )

    # Run individual workflow
    result = await research_workflow_example()

    # Or run all workflows
    from hololoom.apps.departments.examples.workflow_examples import main
    results = await main()

Author: HoloLoom B2B Framework
Date: November 2025
"""

from hololoom.apps.departments.examples.workflow_examples import (
    research_workflow_example,
    deployment_workflow_example,
    intelligent_routing_workflow_example,
    monitoring_scaling_workflow_example,
    customer_onboarding_workflow_example,
    main as run_all_workflows,
)

__all__ = [
    "research_workflow_example",
    "deployment_workflow_example",
    "intelligent_routing_workflow_example",
    "monitoring_scaling_workflow_example",
    "customer_onboarding_workflow_example",
    "run_all_workflows",
]

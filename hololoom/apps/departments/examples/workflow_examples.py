"""
Cross-Department Workflow Examples

This module demonstrates real-world workflows that coordinate multiple departments
(RAG, Planning, Orchestration, Infrastructure, Context) to solve complex problems.

Each example is executable and shows:
- Department coordination patterns
- Error handling
- Performance monitoring
- Context-aware behavior
- Learning and adaptation

Author: HoloLoom B2B Framework
Date: November 2025
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from HoloLoom.apps.departments.protocol import DepartmentRequest, DepartmentResponse
from HoloLoom.apps.departments.rag_department import RAGDepartment
from HoloLoom.apps.departments.planning_department import PlanningDepartment
from HoloLoom.apps.departments.orchestration_department import OrchestrationDepartment
from HoloLoom.apps.departments.infrastructure_department import InfrastructureDepartment
from HoloLoom.apps.departments.context_department import ContextDepartment


# ===== Workflow 1: Research & Analysis Pipeline =====

async def research_workflow_example():
    """
    Complex research workflow demonstrating:
    1. Context Department enriches user query with preferences and history
    2. Orchestration Department routes to appropriate departments
    3. RAG Department retrieves relevant information
    4. Planning Department decomposes into research steps
    5. Results aggregated and returned with full context

    Use Case: "Research Thompson Sampling and create implementation plan"
    """
    print("\n" + "="*80)
    print("WORKFLOW 1: RESEARCH & ANALYSIS PIPELINE")
    print("="*80 + "\n")

    # Initialize departments
    context_dept = ContextDepartment()
    rag_dept = RAGDepartment()
    planning_dept = PlanningDepartment()
    orchestration_dept = OrchestrationDepartment(
        department_registry={
            "context": context_dept,
            "rag": rag_dept,
            "planning": planning_dept,
        }
    )

    user_id = "researcher_001"
    query = "Research Thompson Sampling and create an implementation plan"

    # Step 1: Enrich request with context
    print("Step 1: Context enrichment...")
    context_request = DepartmentRequest(
        task_type="enrich_request",
        parameters={
            "user_id": user_id,
            "request": {
                "query": query,
                "domain": "machine_learning",
            },
        },
    )
    context_response = await context_dept.execute(context_request)
    # ContextDepartment returns ContextEnrichment dataclass
    enrichment = context_response.result
    enriched_request_obj = enrichment.enriched_request  # DepartmentRequest object
    print(f"[+] Context enriched with {len(enrichment.context_sources)} sources")
    print(f"  User preferences: {enriched_request_obj.context.get('user_preferences', {})}")

    # Step 2: Parallel retrieval and planning
    print("\nStep 2: Parallel RAG retrieval and goal decomposition...")
    parallel_request = DepartmentRequest(
        task_type="parallel_execution",
        parameters={
            "departments": [
                {
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "parameters": {
                        "query": query,
                        "max_sources": 10,
                        "mode": "research",
                    },
                },
                {
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "parameters": {
                        "goal": "Create Thompson Sampling implementation plan",
                        "context": enriched_request_obj.context,
                    },
                },
            ],
        },
    )

    parallel_response = await orchestration_dept.execute(parallel_request)
    rag_result = parallel_response.result["results"][0]
    planning_result = parallel_response.result["results"][1]

    print(f"[+] RAG retrieved {len(rag_result.result.get('sources', []))} sources")
    print(f"[+] Planning decomposed into {len(planning_result.result['sub_goals'])} steps")

    # Step 3: Create execution plan
    print("\nStep 3: Create detailed execution plan...")
    planning_request = DepartmentRequest(
        task_type="plan_execution",
        parameters={
            "goal": "Implement Thompson Sampling",
            "sub_goals": planning_result.result["sub_goals"],
            "context": {
                "sources": rag_result.result.get("sources", []),
                "user_preferences": enriched_request_obj.context.get("user_preferences", {}),
            },
        },
    )

    execution_plan = await planning_dept.execute(planning_request)
    print(f"[+] Execution plan created with {len(execution_plan.result['execution_order'])} steps")

    # Step 4: Track session for future reference
    print("\nStep 4: Track session history...")
    session_request = DepartmentRequest(
        task_type="track_session",
        parameters={
            "user_id": user_id,
            "message": query,
            "metadata": {
                "workflow": "research",
                "sources_retrieved": len(rag_result.result.get("sources", [])),
                "steps_planned": len(execution_plan.result["execution_order"]),
            },
        },
    )

    await context_dept.execute(session_request)
    print("[+] Session tracked")

    # Final summary
    print("\n" + "-"*80)
    print("RESEARCH WORKFLOW COMPLETE")
    print(f"Query: {query}")
    print(f"Sources retrieved: {len(rag_result.result.get('sources', []))}")
    print(f"Implementation steps: {len(execution_plan.result['execution_order'])}")
    print(f"Total confidence: {execution_plan.confidence.score:.2f}")
    print("-"*80 + "\n")

    return {
        "query": query,
        "sources": rag_result.result.get("sources", []),
        "execution_plan": execution_plan.result,
        "confidence": execution_plan.confidence.score,
    }


# ===== Workflow 2: Deployment with Health Monitoring =====

async def deployment_workflow_example():
    """
    Production deployment workflow demonstrating:
    1. Context Department validates user permissions
    2. Planning Department creates deployment plan with rollback strategy
    3. Infrastructure Department deploys service and monitors health
    4. Orchestration Department coordinates sequential execution
    5. Context Department logs audit trail

    Use Case: "Deploy microservice with health monitoring and rollback plan"
    """
    print("\n" + "="*80)
    print("WORKFLOW 2: DEPLOYMENT WITH HEALTH MONITORING")
    print("="*80 + "\n")

    # Initialize departments
    context_dept = ContextDepartment()
    planning_dept = PlanningDepartment()
    infrastructure_dept = InfrastructureDepartment()
    orchestration_dept = OrchestrationDepartment(
        department_registry={
            "context": context_dept,
            "planning": planning_dept,
            "infrastructure": infrastructure_dept,
        }
    )

    user_id = "devops_engineer_42"
    service_name = "payment-processor-v2"

    # Step 1: Validate permissions
    print("Step 1: Validate deployment permissions...")
    permission_request = DepartmentRequest(
        task_type="enforce_privacy",
        parameters={
            "user_id": user_id,
            "operation": "deploy_service",
            "resource_id": service_name,
            "privacy_level": "RESTRICTED",
        },
    )

    permission_response = await context_dept.execute(permission_request)

    if not permission_response.result["allowed"]:
        print(f"[x] Permission denied: {permission_response.result['reason']}")
        return {"status": "permission_denied"}

    print(f"[+] Permission granted for {user_id}")

    # Step 2: Create deployment plan with rollback strategy
    print("\nStep 2: Create deployment plan with rollback...")
    deployment_plan_request = DepartmentRequest(
        task_type="goal_decomposition",
        parameters={
            "goal": f"Deploy {service_name} with zero downtime",
            "context": {
                "service": service_name,
                "environment": "production",
                "strategy": "blue_green",
            },
        },
    )

    deployment_plan = await planning_dept.execute(deployment_plan_request)
    print(f"[+] Deployment plan created with {len(deployment_plan.result['sub_goals'])} steps")

    # Step 3: Execute deployment workflow (sequential)
    print("\nStep 3: Execute deployment workflow...")
    workflow_request = DepartmentRequest(
        task_type="sequential_workflow",
        parameters={
            "steps": [
                {
                    "name": "health_check_before",
                    "department_id": "infrastructure",
                    "task_type": "check_health",
                    "parameters": {
                        "services": [service_name],
                    },
                },
                {
                    "name": "deploy",
                    "department_id": "infrastructure",
                    "task_type": "deploy_service",
                    "parameters": {
                        "service_name": service_name,
                        "version": "2.0.0",
                        "strategy": "blue_green",
                    },
                },
                {
                    "name": "health_check_after",
                    "department_id": "infrastructure",
                    "task_type": "check_health",
                    "parameters": {
                        "services": [service_name],
                    },
                },
                {
                    "name": "monitor_resources",
                    "department_id": "infrastructure",
                    "task_type": "monitor_resources",
                    "parameters": {},
                },
            ],
        },
    )

    workflow_response = await orchestration_dept.execute(workflow_request)

    # Check results
    steps_completed = len([s for s in workflow_response.result["step_results"] if s["success"]])
    total_steps = len(workflow_response.result["step_results"])

    print(f"[+] Workflow completed: {steps_completed}/{total_steps} steps successful")

    # Step 4: Log audit trail
    print("\nStep 4: Log deployment audit trail...")
    audit_request = DepartmentRequest(
        task_type="track_session",
        parameters={
            "user_id": user_id,
            "message": f"Deployed {service_name} v2.0.0",
            "metadata": {
                "workflow": "deployment",
                "service": service_name,
                "steps_completed": steps_completed,
                "total_duration_ms": workflow_response.metadata.get("execution_time_ms", 0),
            },
        },
    )

    await context_dept.execute(audit_request)
    print("[+] Audit trail logged")

    # Final summary
    print("\n" + "-"*80)
    print("DEPLOYMENT WORKFLOW COMPLETE")
    print(f"Service: {service_name}")
    print(f"Steps completed: {steps_completed}/{total_steps}")
    print(f"Confidence: {workflow_response.confidence.score:.2f}")
    print("-"*80 + "\n")

    return {
        "service": service_name,
        "status": "deployed" if steps_completed == total_steps else "partial_failure",
        "steps_completed": steps_completed,
        "confidence": workflow_response.confidence.score,
    }


# ===== Workflow 3: Intelligent Query Routing =====

async def intelligent_routing_workflow_example():
    """
    Adaptive query routing workflow demonstrating:
    1. Context Department analyzes user patterns and preferences
    2. Orchestration Department uses AUTO routing to select best department
    3. Departments process query
    4. Context Department learns from outcome

    Use Case: Handle diverse user queries with intelligent routing
    """
    print("\n" + "="*80)
    print("WORKFLOW 3: INTELLIGENT QUERY ROUTING")
    print("="*80 + "\n")

    # Initialize departments
    context_dept = ContextDepartment()
    rag_dept = RAGDepartment()
    planning_dept = PlanningDepartment()
    orchestration_dept = OrchestrationDepartment(
        department_registry={
            "context": context_dept,
            "rag": rag_dept,
            "planning": planning_dept,
        }
    )

    user_id = "analyst_007"

    # Test queries with different characteristics
    queries = [
        "What is Thompson Sampling?",  # Simple factual → RAG
        "Create a 6-month roadmap for ML infrastructure",  # Complex planning → Planning
        "Explain the tradeoffs between exploration and exploitation",  # Analytical → RAG
    ]

    results = []

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 60)

        # Step 1: Enrich with context
        context_request = DepartmentRequest(
            task_type="enrich_request",
            parameters={
                "user_id": user_id,
                "request": {"query": query},
            },
        )
        context_response = await context_dept.execute(context_request)
        # ContextDepartment returns ContextEnrichment dataclass
        enrichment = context_response.result
        enriched_context = enrichment.enriched_request.context

        # Step 2: AUTO route based on task type
        # Orchestration will analyze query and select best department
        routing_request = DepartmentRequest(
            task_type="route_task",
            parameters={
                "routing_strategy": "auto",
                "task_type": "question_answering" if "?" in query else "goal_decomposition",
                "query": query,
                "context": enriched_context,
            },
        )

        routing_response = await orchestration_dept.execute(routing_request)
        routed_dept = routing_response.metadata.get("routed_to", "unknown")

        print(f"[+] Routed to: {routed_dept}")
        print(f"  Confidence: {routing_response.confidence.score:.2f}")

        # Step 3: Update user preferences based on outcome
        preference_request = DepartmentRequest(
            task_type="update_preferences",
            parameters={
                "user_id": user_id,
                "preference_key": "query_types",
                "preference_value": {
                    "query": query[:50],
                    "routed_to": routed_dept,
                    "confidence": routing_response.confidence.score,
                },
            },
        )

        await context_dept.execute(preference_request)

        results.append({
            "query": query,
            "routed_to": routed_dept,
            "confidence": routing_response.confidence.score,
        })

    # Final summary
    print("\n" + "-"*80)
    print("INTELLIGENT ROUTING WORKFLOW COMPLETE")
    print(f"Queries processed: {len(queries)}")
    print(f"Average confidence: {sum(r['confidence'] for r in results) / len(results):.2f}")
    print("\nRouting breakdown:")
    for r in results:
        print(f"  • {r['query'][:50]:<50} → {r['routed_to']:<15} ({r['confidence']:.2f})")
    print("-"*80 + "\n")

    return results


# ===== Workflow 4: Performance Monitoring & Auto-Scaling =====

async def monitoring_scaling_workflow_example():
    """
    Infrastructure monitoring and auto-scaling workflow demonstrating:
    1. Infrastructure Department monitors system resources
    2. Context Department tracks performance patterns
    3. Planning Department creates scaling strategy
    4. Infrastructure Department executes scaling
    5. Orchestration Department coordinates the pipeline

    Use Case: Auto-scale based on resource utilization trends
    """
    print("\n" + "="*80)
    print("WORKFLOW 4: PERFORMANCE MONITORING & AUTO-SCALING")
    print("="*80 + "\n")

    # Initialize departments
    context_dept = ContextDepartment()
    planning_dept = PlanningDepartment()
    infrastructure_dept = InfrastructureDepartment()
    orchestration_dept = OrchestrationDepartment(
        department_registry={
            "context": context_dept,
            "planning": planning_dept,
            "infrastructure": infrastructure_dept,
        }
    )

    service_name = "api-gateway"

    # Step 1: Monitor resources
    print("Step 1: Monitor system resources...")
    monitor_request = DepartmentRequest(
        task_type="monitor_resources",
        parameters={},
    )

    monitor_response = await infrastructure_dept.execute(monitor_request)
    metrics = monitor_response.result["metrics"]

    print(f"[+] Resources monitored:")
    print(f"  CPU: {metrics['cpu_percent']:.1f}%")
    print(f"  Memory: {metrics['memory_percent']:.1f}%")
    print(f"  Disk: {metrics['disk_percent']:.1f}%")

    # Step 2: Analyze performance
    print("\nStep 2: Analyze performance trends...")
    perf_request = DepartmentRequest(
        task_type="analyze_performance",
        parameters={
            "service_name": service_name,
        },
    )

    perf_response = await infrastructure_dept.execute(perf_request)
    analysis = perf_response.result

    print(f"[+] Performance analysis:")
    print(f"  Latency p95: {analysis.get('latency_p95', 0):.1f}ms")
    print(f"  Error rate: {analysis.get('error_rate', 0):.2%}")

    # Step 3: Determine if scaling needed
    print("\nStep 3: Check if scaling needed...")
    scale_request = DepartmentRequest(
        task_type="auto_scale",
        parameters={
            "service_name": service_name,
            "current_instances": 3,
            "metrics": metrics,
        },
    )

    scale_response = await infrastructure_dept.execute(scale_request)
    scaling = scale_response.result

    if scaling["should_scale"]:
        print(f"[+] Scaling recommended: {scaling['recommendation']}")
        print(f"  Suggested instances: {scaling['suggested_instances']}")
        print(f"  Reason: {scaling['reason']}")

        # Step 4: Create scaling plan
        print("\nStep 4: Create scaling execution plan...")
        plan_request = DepartmentRequest(
            task_type="plan_execution",
            parameters={
                "goal": f"Scale {service_name} to {scaling['suggested_instances']} instances",
                "sub_goals": [
                    "Verify current health",
                    "Add new instances",
                    "Wait for warmup",
                    "Verify scaled health",
                ],
                "context": {"metrics": metrics, "analysis": analysis},
            },
        )

        plan_response = await planning_dept.execute(plan_request)
        print(f"[+] Scaling plan created with {len(plan_response.result['execution_order'])} steps")

        # Step 5: Track scaling event
        print("\nStep 5: Log scaling event...")
        context_request = DepartmentRequest(
            task_type="track_session",
            parameters={
                "user_id": "system",
                "message": f"Auto-scaled {service_name}",
                "metadata": {
                    "workflow": "auto_scaling",
                    "from_instances": 3,
                    "to_instances": scaling["suggested_instances"],
                    "reason": scaling["reason"],
                },
            },
        )

        await context_dept.execute(context_request)
        print("[+] Scaling event logged")
    else:
        print("[+] No scaling needed - resources within normal range")

    # Final summary
    print("\n" + "-"*80)
    print("MONITORING & SCALING WORKFLOW COMPLETE")
    print(f"Service: {service_name}")
    print(f"CPU: {metrics['cpu_percent']:.1f}% | Memory: {metrics['memory_percent']:.1f}%")
    print(f"Scaling needed: {scaling['should_scale']}")
    print("-"*80 + "\n")

    return {
        "service": service_name,
        "metrics": metrics,
        "scaling": scaling,
    }


# ===== Workflow 5: Complete B2B Customer Onboarding =====

async def customer_onboarding_workflow_example():
    """
    End-to-end B2B customer onboarding workflow demonstrating:
    1. Context Department creates customer profile with privacy settings
    2. Planning Department creates 90-day onboarding plan
    3. Infrastructure Department provisions resources
    4. RAG Department populates initial knowledge base
    5. Orchestration Department coordinates entire onboarding

    Use Case: Onboard new enterprise customer with custom configuration
    """
    print("\n" + "="*80)
    print("WORKFLOW 5: COMPLETE B2B CUSTOMER ONBOARDING")
    print("="*80 + "\n")

    # Initialize departments
    context_dept = ContextDepartment()
    rag_dept = RAGDepartment()
    planning_dept = PlanningDepartment()
    infrastructure_dept = InfrastructureDepartment()
    orchestration_dept = OrchestrationDepartment(
        department_registry={
            "context": context_dept,
            "rag": rag_dept,
            "planning": planning_dept,
            "infrastructure": infrastructure_dept,
        }
    )

    customer_id = "acme_corp"
    customer_tier = "platinum"

    # Comprehensive onboarding workflow
    print("Executing comprehensive onboarding workflow...")
    workflow_request = DepartmentRequest(
        task_type="sequential_workflow",
        parameters={
            "steps": [
                # Step 1: Create customer profile
                {
                    "name": "create_profile",
                    "department_id": "context",
                    "task_type": "update_preferences",
                    "parameters": {
                        "user_id": customer_id,
                        "preference_key": "tier",
                        "preference_value": customer_tier,
                    },
                },

                # Step 2: Set privacy level
                {
                    "name": "configure_privacy",
                    "department_id": "context",
                    "task_type": "enforce_privacy",
                    "parameters": {
                        "user_id": customer_id,
                        "operation": "read_sensitive_data",
                        "resource_id": "customer_data",
                        "privacy_level": "RESTRICTED",
                    },
                },

                # Step 3: Create 90-day plan
                {
                    "name": "create_onboarding_plan",
                    "department_id": "planning",
                    "task_type": "goal_decomposition",
                    "parameters": {
                        "goal": f"Successfully onboard {customer_id} ({customer_tier} tier)",
                        "context": {
                            "customer": customer_id,
                            "tier": customer_tier,
                            "duration": "90_days",
                        },
                    },
                },

                # Step 4: Provision infrastructure
                {
                    "name": "provision_resources",
                    "department_id": "infrastructure",
                    "task_type": "deploy_service",
                    "parameters": {
                        "service_name": f"{customer_id}_instance",
                        "version": "1.0.0",
                        "strategy": "create",
                    },
                },

                # Step 5: Health check
                {
                    "name": "verify_deployment",
                    "department_id": "infrastructure",
                    "task_type": "check_health",
                    "parameters": {
                        "services": [f"{customer_id}_instance"],
                    },
                },

                # Step 6: Initialize knowledge base
                {
                    "name": "setup_knowledge_base",
                    "department_id": "rag",
                    "task_type": "retrieve_context",
                    "parameters": {
                        "query": f"Initialize knowledge base for {customer_tier} customer",
                        "max_sources": 20,
                        "mode": "direct",
                    },
                },
            ],
        },
    )

    workflow_response = await orchestration_dept.execute(workflow_request)

    # Analyze results
    step_results = workflow_response.result["step_results"]
    successful_steps = [s for s in step_results if s["success"]]

    print(f"\n[+] Onboarding workflow completed: {len(successful_steps)}/{len(step_results)} steps successful\n")

    # Print step-by-step results
    for i, step in enumerate(step_results, 1):
        status = "[+]" if step["success"] else "[x]"
        print(f"{status} Step {i}: {step['name']:<25} (confidence: {step.get('confidence', 0):.2f})")

    # Final summary
    print("\n" + "-"*80)
    print("B2B ONBOARDING WORKFLOW COMPLETE")
    print(f"Customer: {customer_id} ({customer_tier} tier)")
    print(f"Steps completed: {len(successful_steps)}/{len(step_results)}")
    print(f"Overall confidence: {workflow_response.confidence.score:.2f}")
    print(f"Execution time: {workflow_response.metadata.get('execution_time_ms', 0):.1f}ms")
    print("-"*80 + "\n")

    return {
        "customer_id": customer_id,
        "tier": customer_tier,
        "steps_completed": len(successful_steps),
        "total_steps": len(step_results),
        "confidence": workflow_response.confidence.score,
    }


# ===== Main Demo Runner =====

async def main():
    """Run all workflow examples."""
    print("\n" + "="*80)
    print("CROSS-DEPARTMENT WORKFLOW EXAMPLES")
    print("Demonstrating HoloLoom B2B Framework Multi-Department Coordination")
    print("="*80)

    # Run all workflows
    results = {}

    try:
        results["research"] = await research_workflow_example()
    except Exception as e:
        print(f"Research workflow failed: {e}")

    try:
        results["deployment"] = await deployment_workflow_example()
    except Exception as e:
        print(f"Deployment workflow failed: {e}")

    try:
        results["routing"] = await intelligent_routing_workflow_example()
    except Exception as e:
        print(f"Routing workflow failed: {e}")

    try:
        results["monitoring"] = await monitoring_scaling_workflow_example()
    except Exception as e:
        print(f"Monitoring workflow failed: {e}")

    try:
        results["onboarding"] = await customer_onboarding_workflow_example()
    except Exception as e:
        print(f"Onboarding workflow failed: {e}")

    # Final summary
    print("\n" + "="*80)
    print("ALL WORKFLOWS COMPLETE")
    print("="*80)
    print(f"\nTotal workflows executed: {len(results)}")
    print("\nWorkflow Summary:")
    for name, result in results.items():
        print(f"  • {name.title():<20} - {result.get('status', 'completed')}")
    print("\n" + "="*80 + "\n")

    return results


if __name__ == "__main__":
    asyncio.run(main())

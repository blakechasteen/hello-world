#!/usr/bin/env python3
"""
HoloLoom Workflow Features Demo
================================

Demonstrates all workflow enhancements:
1. Custom agent registry
2. Domain-specific templates
3. AI workflow generation
4. Analytics dashboard
5. Collaborative editing (simulated)

Usage:
    PYTHONPATH=. python demos/demo_workflow_features.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.workflows import (
    get_registry,
    WorkflowTemplates,
    AIWorkflowGenerator,
    WorkflowAnalytics,
    CollaborationManager,
    init_workflows
)


async def demo_agent_registry():
    """Demo: Custom Agent Registry"""
    print("\n" + "=" * 80)
    print("1. CUSTOM AGENT REGISTRY")
    print("=" * 80)

    registry = get_registry()

    # Show built-in agents
    print(f"\n📊 Total agents: {len(registry.agents)}")

    print("\n🔍 Agents by category:")
    from HoloLoom.workflows.agent_registry import AgentCategory
    for category in AgentCategory:
        agents = registry.list_by_category(category)
        if agents:
            print(f"\n{category.value.upper()} ({len(agents)}):")
            for agent in agents[:3]:  # Show first 3
                print(f"  • {agent.name} ({agent.agent_id})")
                print(f"    {agent.description[:60]}..." if len(agent.description) > 60 else f"    {agent.description}")

    # Register custom agent
    print("\n✨ Registering custom agent...")

    @registry.register(
        'image_generator',
        name='Image Generator',
        category='custom',
        inputs=['prompt'],
        outputs=['image'],
        color='#ff6b6b',
        icon='🎨',
        description='Generate images from text prompts using DALL-E'
    )
    async def image_generator(inputs, config):
        prompt = inputs.get('prompt', '')
        model = config.get('model', 'dall-e-3')
        return {
            'image_url': f'https://example.com/generated/{prompt[:20]}.png',
            'model': model,
            'prompt': prompt
        }

    print(f"   ✓ Registered 'image_generator' agent")
    print(f"   Total agents now: {len(registry.agents)}")

    # Search agents
    print("\n🔎 Search for 'code' agents:")
    results = registry.search('code')
    for agent in results[:3]:
        print(f"  • {agent.name}")

    return registry


async def demo_workflow_templates():
    """Demo: Domain-Specific Templates"""
    print("\n" + "=" * 80)
    print("2. WORKFLOW TEMPLATES")
    print("=" * 80)

    templates = WorkflowTemplates()

    print(f"\n📦 Total templates: {len(templates.templates)}")

    print("\n📋 Available templates:")
    for template in templates.list_all():
        print(f"\n• {template.name} ({template.difficulty})")
        print(f"  Category: {template.category.value}")
        print(f"  {template.description}")
        print(f"  Nodes: {len(template.nodes)} | Connections: {len(template.connections)}")
        print(f"  Use cases:")
        for uc in template.use_cases[:2]:
            print(f"    - {uc}")

    # Export template
    print("\n💾 Export RAG Research template:")
    rag_workflow = templates.export_to_json('rag_research')
    workflow_obj = json.loads(rag_workflow)
    print(f"   Workflow: {workflow_obj['name']}")
    print(f"   Nodes: {len(workflow_obj['nodes'])}")
    print(f"   Connections: {len(workflow_obj['connections'])}")

    # Clone and customize
    print("\n🔧 Clone and customize template:")
    customized = templates.clone_and_customize(
        'code_review',
        'Custom Code Review',
        {
            'node_1': {'language': 'typescript', 'checks': ['quality', 'security', 'performance']}
        }
    )
    print(f"   Customized: {customized['name']}")
    print(f"   Modified node configs: 1")

    return templates


async def demo_ai_workflow_generation():
    """Demo: AI Workflow Generator"""
    print("\n" + "=" * 80)
    print("3. AI WORKFLOW GENERATION")
    print("=" * 80)

    generator = AIWorkflowGenerator()

    # Example 1: Code analysis
    print("\n🤖 Example 1: Generate workflow from description")
    print("   Description: 'Analyze Python code for security issues and suggest fixes'")

    workflow1 = await generator.generate(
        "Analyze Python code for security issues, use an LLM to suggest fixes, "
        "and save the results in JSON format"
    )

    print(f"\n   ✓ Generated workflow: {workflow1['name']}")
    print(f"   Nodes: {len(workflow1['nodes'])}")
    print(f"   Connections: {len(workflow1['connections'])}")
    print(f"   Pipeline:")
    for i, node in enumerate(workflow1['nodes'], 1):
        print(f"     {i}. {node['agentType']}")

    # Example 2: Research pipeline
    print("\n🤖 Example 2: Complex research workflow")
    print("   Description: 'Research with multiple sources, synthesis, and refinement'")

    workflow2 = await generator.generate(
        "Build a research workflow that searches multiple sources, "
        "synthesizes the findings, and refines the answer with error handling"
    )

    print(f"\n   ✓ Generated workflow: {workflow2['name']}")
    print(f"   Nodes: {len(workflow2['nodes'])}")
    print(f"   Pipeline:")
    for i, node in enumerate(workflow2['nodes'], 1):
        print(f"     {i}. {node['agentType']}")

    # Example 3: Workflow refinement
    print("\n🔄 Example 3: Refine existing workflow")
    print("   Refinement: 'Add error handling and make it parallel'")

    refined = await generator.refine(workflow2, "Add error handling and make it parallel")

    print(f"\n   ✓ Refined workflow")
    print(f"   Nodes before: {len(workflow2['nodes'])} → after: {len(refined['nodes'])}")
    print(f"   Added: {len(refined['nodes']) - len(workflow2['nodes'])} nodes")

    # Validate
    print("\n✅ Validation:")
    valid, errors = generator.validate(refined)
    print(f"   Valid: {valid}")
    if errors:
        for error in errors:
            print(f"   ⚠️  {error}")
    else:
        print(f"   No errors found!")

    return generator, refined


async def demo_workflow_analytics():
    """Demo: Workflow Analytics"""
    print("\n" + "=" * 80)
    print("4. WORKFLOW ANALYTICS")
    print("=" * 80)

    analytics = WorkflowAnalytics()

    # Simulate executions
    print("\n📊 Simulating 30 workflow executions...")

    import random

    for i in range(30):
        analytics.track_execution({
            'execution_id': f'exec_{i}',
            'workflow_id': 'demo_workflow',
            'workflow_name': 'Demo Workflow',
            'total_duration_ms': random.uniform(150, 600),
            'nodes_executed': 5,
            'nodes_failed': random.choice([0, 0, 0, 0, 1]),  # 80% success
            'node_durations': {
                'node_1': random.uniform(20, 50),
                'node_2': random.uniform(30, 100),
                'node_3': random.uniform(250, 400),  # Bottleneck!
                'node_4': random.uniform(10, 30),
                'node_5': random.uniform(5, 20)
            },
            'node_statuses': {
                'node_1': 'success',
                'node_2': 'success',
                'node_3': 'success',
                'node_4': 'success',
                'node_5': random.choice(['success', 'success', 'success', 'failure'])
            },
            'status': random.choice(['success', 'success', 'success', 'success', 'failure'])
        })

    # Get metrics
    print("\n📈 Metrics Summary:")
    metrics = analytics.get_metrics('demo_workflow')

    print(f"\n   Total Executions: {metrics.total_executions}")
    print(f"   Successes: {metrics.successes} ({metrics.successes / metrics.total_executions * 100:.1f}%)")
    print(f"   Failures: {metrics.failures}")
    print(f"   Avg Duration: {metrics.avg_duration_ms:.0f}ms")
    print(f"   P95 Duration: {metrics.p95_duration_ms:.0f}ms")

    print(f"\n   ⚠️  Bottleneck Nodes: {len(metrics.bottleneck_nodes)}")
    for node_id in metrics.bottleneck_nodes:
        node_m = metrics.node_metrics[node_id]
        print(f"      • {node_id}: {node_m.avg_duration_ms:.0f}ms ({node_m.avg_duration_ms / metrics.avg_duration_ms * 100:.0f}% of total)")

    # Export metrics
    print("\n💾 Export metrics as JSON:")
    exported = analytics.export_metrics('demo_workflow')
    print(f"   Success rate: {exported['success_rate']:.1%}")
    print(f"   Avg duration: {exported['avg_duration_ms']:.0f}ms")

    # Generate dashboard
    print("\n🎨 Generating HTML dashboard...")
    html = analytics.generate_dashboard('demo_workflow')

    output_file = Path('demos/output/workflow_analytics_demo.html')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html)

    print(f"   ✓ Saved to: {output_file}")
    print(f"   Dashboard size: {len(html)} bytes")

    return analytics


async def demo_collaborative_editing():
    """Demo: Collaborative Editing (simulated)"""
    print("\n" + "=" * 80)
    print("5. COLLABORATIVE EDITING")
    print("=" * 80)

    manager = CollaborationManager()

    # Create session
    session = manager.get_or_create_session('demo_workflow')

    print(f"\n👥 Collaborative session created: demo_workflow")

    # Mock WebSocket
    class MockWebSocket:
        def __init__(self, name):
            self.name = name

        async def send_json(self, message):
            if message['type'] != 'initial_state':  # Skip verbose messages
                print(f"   → [{self.name}] {message['type']}: {message.get('user_id', 'system')}")

    # Add users
    print("\n📥 Users joining session:")
    await session.add_user("alice", "Alice", MockWebSocket("Alice"))
    await session.add_user("bob", "Bob", MockWebSocket("Bob"))
    await session.add_user("carol", "Carol", MockWebSocket("Carol"))

    print(f"\n   Active users: {len(session.users)}")
    for user in session.users.values():
        print(f"      • {user.display_name} ({user.color})")

    # Simulate operations
    print("\n⚙️  Applying operations:")

    # Alice adds nodes
    print("\n   1. Alice adds 2 nodes:")
    await session.apply_operation({
        'type': 'add_node',
        'user_id': 'alice',
        'data': {
            'node': {
                'id': 'node_1',
                'agentType': 'hololoom',
                'x': 100,
                'y': 200,
                'config': {'pattern': 'fast'}
            }
        }
    })

    await session.apply_operation({
        'type': 'add_node',
        'user_id': 'alice',
        'data': {
            'node': {
                'id': 'node_2',
                'agentType': 'response',
                'x': 300,
                'y': 200,
                'config': {'format': 'markdown'}
            }
        }
    })

    # Bob adds connection
    print("\n   2. Bob connects the nodes:")
    await session.apply_operation({
        'type': 'add_connection',
        'user_id': 'bob',
        'data': {
            'connection': {
                'id': 'conn_1',
                'from': 'node_1',
                'to': 'node_2'
            }
        }
    })

    # Carol moves node
    print("\n   3. Carol repositions a node:")
    await session.apply_operation({
        'type': 'move_node',
        'user_id': 'carol',
        'data': {
            'node_id': 'node_2',
            'x': 400,
            'y': 250
        }
    })

    # Update cursors
    print("\n   4. Cursor movements:")
    await session.update_cursor('alice', 150, 220)
    await session.update_cursor('bob', 350, 200)

    # Show final state
    print("\n📊 Final workflow state:")
    state = session.get_workflow_state()
    print(f"   Nodes: {len(state['nodes'])}")
    print(f"   Connections: {len(state['connections'])}")
    print(f"   Total operations: {len(session.operations)}")

    # Operation history
    print("\n📜 Operation history:")
    for op_data in session.get_operation_history():
        print(f"   • {op_data['op_type']} by {op_data['user_id']} at {op_data['timestamp'][-12:]}")

    # Active sessions
    print("\n🌐 Active sessions:")
    active = manager.get_active_sessions()
    for session_info in active:
        print(f"   • {session_info['workflow_id']}: {session_info['user_count']} users, {session_info['operation_count']} ops")

    return manager


async def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("🚀 HOLOLOOM WORKFLOW FEATURES DEMO")
    print("=" * 80)
    print("\nThis demo showcases all workflow enhancements:")
    print("  1. Custom Agent Registry")
    print("  2. Domain-Specific Templates")
    print("  3. AI Workflow Generation")
    print("  4. Workflow Analytics")
    print("  5. Collaborative Editing")

    try:
        # Initialize
        print("\n🔧 Initializing workflow system...")
        system = init_workflows()
        print(f"   ✓ Registry: {len(system['registry'].agents)} agents")
        print(f"   ✓ Templates: {len(system['templates'].templates)} templates")
        print(f"   ✓ Analytics: Ready")
        print(f"   ✓ Collaboration: Ready")

        # Run demos
        registry = await demo_agent_registry()
        templates = await demo_workflow_templates()
        generator, workflow = await demo_ai_workflow_generation()
        analytics = await demo_workflow_analytics()
        manager = await demo_collaborative_editing()

        # Summary
        print("\n" + "=" * 80)
        print("✅ DEMO COMPLETE")
        print("=" * 80)

        print("\n📊 Summary:")
        print(f"   • Total agents: {len(registry.agents)}")
        print(f"   • Total templates: {len(templates.templates)}")
        print(f"   • Workflows generated: 3")
        print(f"   • Analytics tracked: 30 executions")
        print(f"   • Collaborative operations: {len(manager.sessions['demo_workflow'].operations)}")

        print("\n📁 Output files:")
        print(f"   • demos/output/workflow_analytics_demo.html")

        print("\n🎯 Next steps:")
        print("   1. Start workflow executor: python HoloLoom/web_dashboard/workflow_executor.py")
        print("   2. Open workflow_builder.html in browser")
        print("   3. Try the AI workflow generator with natural language")
        print("   4. Monitor performance with analytics dashboard")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

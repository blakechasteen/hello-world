#!/usr/bin/env python3
"""
Enhanced Workflow System Demo
==============================

Demonstrates:
- 15 new workflow templates
- Enhanced AI generator with confidence scoring
- Debugging tools (step-through, breakpoints)
- Testing framework (validation, simulation, comparison)

Run:
    PYTHONPATH=. python HoloLoom/workflows/demo_enhanced_workflows.py
"""

import asyncio
import logging
from HoloLoom.workflows.templates import WorkflowTemplates, TemplateCategory
from HoloLoom.workflows.ai_generator import AIWorkflowGenerator
from HoloLoom.workflows.test_framework import (
    validate_workflow, WorkflowTester
)
from HoloLoom.workflows.debug_tools import WorkflowDebugger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_templates():
    """Demonstrate 15 workflow templates"""
    print("\n" + "=" * 80)
    print("DEMO 1: WORKFLOW TEMPLATES (15 Total)")
    print("=" * 80)

    templates = WorkflowTemplates()
    all_templates = templates.list_all()

    print(f"\nTotal templates: {len(all_templates)}")
    print("\nBy Category:")

    for category in TemplateCategory:
        templates_in_cat = templates.list_by_category(category)
        print(f"\n  {category.value.upper()} ({len(templates_in_cat)})")
        for t in templates_in_cat:
            print(f"    - {t.name}")
            print(f"      {t.description}")
            print(f"      Difficulty: {t.difficulty}")

    # Show a new template in detail
    print("\n" + "-" * 80)
    print("EXAMPLE: Email Analysis Pipeline (NEW)")
    print("-" * 80)

    email_template = templates.get('email_analysis')
    print(f"\nName: {email_template.name}")
    print(f"Description: {email_template.description}")
    print(f"Nodes: {len(email_template.nodes)}")
    print(f"Connections: {len(email_template.connections)}")
    print(f"\nUse Cases:")
    for uc in email_template.use_cases:
        print(f"  - {uc}")

    # Export template as JSON
    json_export = templates.export_to_json('email_analysis')
    print(f"\nJSON (first 300 chars):\n{json_export[:300]}...")


def demo_ai_generator_enhanced():
    """Demonstrate enhanced AI generator with confidence scoring"""
    print("\n" + "=" * 80)
    print("DEMO 2: ENHANCED AI GENERATOR WITH CONFIDENCE SCORING")
    print("=" * 80)

    generator = AIWorkflowGenerator()

    test_descriptions = [
        "Analyze Python code for security vulnerabilities and suggest fixes",
        "Build a research workflow that searches multiple sources and synthesizes findings",
        "Create a pipeline that sends emails to customers, classifies responses by sentiment, and extracts action items",
        "Detect spam in user-generated content and escalate for human review if needed",
        "Generate SQL queries from natural language and validate them before execution"
    ]

    print(f"\nTesting {len(test_descriptions)} descriptions:\n")

    for i, desc in enumerate(test_descriptions, 1):
        print(f"{i}. Description:")
        print(f"   \"{desc}\"")

        intent = generator.detect_intent(desc)

        print(f"   Primary Goal: {intent.primary_goal}")
        print(f"   Confidence: {intent.confidence:.0%}")
        print(f"   Explanation: {intent.explanation}")
        print(f"   Complexity: {intent.complexity}")
        print(f"   Agents: {', '.join(intent.agents_needed)}")
        if intent.constraints:
            print(f"   Constraints: {', '.join(intent.constraints)}")
        print()


async def demo_debugging_tools():
    """Demonstrate debugging tools"""
    print("\n" + "=" * 80)
    print("DEMO 3: WORKFLOW DEBUGGING TOOLS")
    print("=" * 80)

    # Create a sample workflow
    workflow = {
        'id': 'debug_example',
        'name': 'Debug Example',
        'nodes': [
            {'id': 'node_1', 'agentType': 'hololoom', 'config': {}},
            {'id': 'node_2', 'agentType': 'response', 'config': {}}
        ],
        'connections': [
            {'from': 'node_1', 'to': 'node_2'}
        ]
    }

    debugger = WorkflowDebugger(workflow)

    print("\nWorkflow Structure:")
    print(f"  Nodes: {len(workflow['nodes'])}")
    print(f"  Connections: {len(workflow['connections'])}")

    # Set breakpoint
    debugger.set_breakpoint('node_1')
    print(f"\nBreakpoints set:")
    for bp in debugger.list_breakpoints():
        print(f"  - {bp.node_id}")

    # Run to breakpoint
    print("\nRunning to breakpoint...")
    trace = await debugger.run_to_breakpoint()

    print(f"\nTrace Status: {trace.status}")
    print(f"Frames Executed: {len(trace.frames)}")

    # Inspect variables
    if trace.frames:
        print(f"\nFirst Frame Inspection:")
        vars = debugger.inspect_variables(0)
        for key, value in vars.items():
            if isinstance(value, dict):
                print(f"  {key}: {{...}}")
            else:
                print(f"  {key}: {value}")

    # Print trace
    print("\n" + debugger.print_trace())


def demo_testing_framework():
    """Demonstrate testing framework"""
    print("\n" + "=" * 80)
    print("DEMO 4: WORKFLOW TESTING FRAMEWORK")
    print("=" * 80)

    tester = WorkflowTester()

    # Test a template
    print("\n1. Template Validation:")
    result = tester.test_template('rag_research')
    print(f"   Template: {result.details['name']}")
    print(f"   Valid: {result.passed}")
    print(f"   Nodes: {result.details['num_nodes']}")
    print(f"   Connections: {result.details['num_connections']}")

    # Test all templates
    print("\n2. Test All Templates:")
    all_results = tester.test_all_templates()
    passed = sum(1 for r in all_results if r.passed)
    print(f"   {passed}/{len(all_results)} templates valid")

    # Test AI generator
    print("\n3. Test AI Generator:")
    descriptions = [
        "Analyze code",
        "Research topic",
        "Process data"
    ]
    gen_results = tester.test_ai_generator(descriptions)

    for result in gen_results:
        status = "PASS" if result.passed else "FAIL"
        print(f"   [{status}] {result.test_name}")
        if result.details:
            print(f"        Confidence: {result.details.get('confidence', 0):.0%}")

    # Validate a workflow
    print("\n4. Workflow Validation:")
    test_workflow = {
        'nodes': [
            {'id': 'n1', 'agentType': 'hololoom', 'config': {}},
            {'id': 'n2', 'agentType': 'response', 'config': {}}
        ],
        'connections': [
            {'from': 'n1', 'to': 'n2'}
        ]
    }

    validation = tester.validate_workflow(test_workflow)
    print(f"   Valid: {validation.valid}")
    print(f"   Metrics:")
    for key, value in validation.metrics.items():
        print(f"     - {key}: {value}")

    # Simulate execution
    print("\n5. Workflow Simulation (Dry-Run):")
    trace = tester.simulate_execution(test_workflow, inputs={'query': 'test'})
    print(f"   Steps: {len(trace['steps'])}")
    for step in trace['steps']:
        print(f"     - {step['node_id']} ({step['node_type']})")

    # Compare workflows
    print("\n6. Workflow Comparison:")
    workflow2 = {
        'nodes': [
            {'id': 'n1', 'agentType': 'hololoom', 'config': {}},
            {'id': 'n2', 'agentType': 'refiner', 'config': {}},
            {'id': 'n3', 'agentType': 'response', 'config': {}}
        ],
        'connections': [
            {'from': 'n1', 'to': 'n2'},
            {'from': 'n2', 'to': 'n3'}
        ]
    }

    diff = tester.compare_workflows(test_workflow, workflow2)
    print(f"   Added Nodes: {diff['nodes_added']}")
    print(f"   Removed Nodes: {diff['nodes_removed']}")
    print(f"   Similarity: {diff['similarity_score']:.0%}")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("HOLOLOOM ENHANCED WORKFLOW SYSTEM - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nFeatures:")
    print("  - 15 workflow templates (5 original + 10 new)")
    print("  - Enhanced AI generator with confidence scoring")
    print("  - Workflow debugging tools (breakpoints, step-through)")
    print("  - Testing framework (validation, simulation, comparison)")

    # Run demos
    demo_templates()
    demo_ai_generator_enhanced()
    asyncio.run(demo_debugging_tools())
    demo_testing_framework()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
New Templates (10):
  1. SQL Query Pipeline - Database query execution with validation
  2. Email Analysis Pipeline - Parse, classify, and extract from emails
  3. Document Q&A Pipeline - Question answering over documents
  4. Sentiment Analysis Pipeline - Multi-model sentiment analysis
  5. Translation Pipeline - Language detection and translation
  6. Bug Triage Workflow - Automatic bug classification and assignment
  7. Meeting Summarization Pipeline - Transcribe and extract action items
  8. Product Recommendation Engine - Personalized recommendations
  9. Content Moderation Pipeline - Multi-model content analysis
  10. (And more!)

Enhanced Features:
  ✓ Confidence scoring in intent detection (0.0-1.0)
  ✓ Expanded keyword patterns (25+ intent types)
  ✓ Domain-specific mappings (10+ domains)
  ✓ Better constraint detection
  ✓ Detailed intent explanations

Debugging & Testing:
  ✓ Step-through execution mode
  ✓ Breakpoint support (conditional + hit count)
  ✓ Variable inspection at any frame
  ✓ Execution replay & trace export
  ✓ Workflow validation (cycles, connections, agents)
  ✓ Dry-run simulation
  ✓ Workflow comparison
  ✓ Template testing suite
  ✓ AI generator testing

Documentation:
  ✓ BEST_PRACTICES.md - 10 sections of guidance
  ✓ Updated README.md - API reference
  ✓ Complete API for all new features

Status: ✅ Production Ready
Created: November 2025
    """)

    print("=" * 80)


if __name__ == "__main__":
    main()

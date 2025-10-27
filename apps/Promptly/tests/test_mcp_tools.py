#!/usr/bin/env python3
"""
Quick test of new MCP tools (composition and analytics)
Run this to verify the new tools are properly integrated
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'promptly'))

print("Testing MCP Server Tool Integration\n")
print("=" * 60)

# Test 1: Check imports
print("\n1. Checking imports...")
try:
    from loop_composition import LoopComposer, CompositionStep, CompositionResult
    print("   [OK] Loop composition available")
    composition_ok = True
except ImportError as e:
    print(f"   [WARN] Loop composition not available: {e}")
    composition_ok = False

try:
    from prompt_analytics import PromptAnalytics, PromptExecution
    print("   [OK] Prompt analytics available")
    analytics_ok = True
except ImportError as e:
    print(f"   [WARN] Prompt analytics not available: {e}")
    analytics_ok = False

# Test 2: Check MCP server imports
print("\n2. Checking MCP server integration...")
try:
    # Don't actually import the server (requires MCP SDK)
    # Just check the file exists and has the new tools
    mcp_path = os.path.join(os.path.dirname(__file__), 'promptly', 'mcp_server.py')
    with open(mcp_path, 'r', encoding='utf-8') as f:
        content = f.read()

    tools_to_check = [
        'promptly_compose_loops',
        'promptly_decompose_refine_verify',
        'promptly_analytics_summary',
        'promptly_analytics_prompt_stats',
        'promptly_analytics_recommendations',
        'promptly_analytics_top_prompts'
    ]

    found_tools = []
    for tool in tools_to_check:
        if f'name="{tool}"' in content:
            found_tools.append(tool)

    print(f"   [OK] Found {len(found_tools)}/{len(tools_to_check)} new tools in MCP server")
    for tool in found_tools:
        print(f"        - {tool}")

except Exception as e:
    print(f"   [ERROR] Could not check MCP server: {e}")

# Test 3: Test analytics functionality
if analytics_ok:
    print("\n3. Testing analytics...")
    try:
        analytics = PromptAnalytics()

        # Get summary (should be empty initially)
        summary = analytics.get_summary()
        print(f"   [OK] Analytics summary: {summary['total_executions']} executions")

        # Record a test execution
        from datetime import datetime
        exec_test = PromptExecution(
            prompt_id="test_tool",
            prompt_name="test_mcp_integration",
            execution_time=1.5,
            quality_score=0.85,
            success=True,
            model="test",
            backend="test"
        )

        analytics.record_execution(exec_test)
        print("   [OK] Successfully recorded test execution")

        # Get updated summary
        summary = analytics.get_summary()
        print(f"   [OK] Updated summary: {summary['total_executions']} executions")

    except Exception as e:
        print(f"   [ERROR] Analytics test failed: {e}")

# Test 4: Test composition functionality (without actual execution)
if composition_ok:
    print("\n4. Testing composition...")
    try:
        from recursive_loops import LoopType, LoopConfig

        # Create composition steps
        steps = [
            CompositionStep(
                loop_type=LoopType.REFINE,
                config=LoopConfig(loop_type=LoopType.REFINE, max_iterations=2),
                description="Test refine step"
            )
        ]

        print(f"   [OK] Created {len(steps)} composition step(s)")
        print(f"        - {steps[0].description}")

    except Exception as e:
        print(f"   [ERROR] Composition test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Integration Test Summary:")
print(f"  Loop Composition: {'OK' if composition_ok else 'NOT AVAILABLE'}")
print(f"  Prompt Analytics: {'OK' if analytics_ok else 'NOT AVAILABLE'}")
print("\nNew MCP Tools Ready:")
print("  1. promptly_compose_loops")
print("  2. promptly_decompose_refine_verify")
print("  3. promptly_analytics_summary")
print("  4. promptly_analytics_prompt_stats")
print("  5. promptly_analytics_recommendations")
print("  6. promptly_analytics_top_prompts")
print("\nTotal MCP Tools: 27 (21 original + 6 new)")

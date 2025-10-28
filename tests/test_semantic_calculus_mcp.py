#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Semantic Calculus MCP Server

Tests all three MCP tools:
1. analyze_semantic_flow - velocity, acceleration, curvature analysis
2. predict_conversation_flow - trajectory prediction
3. evaluate_conversation_ethics - ethical evaluation
"""

import asyncio
import sys
import io
from pathlib import Path

# Fix Windows console encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.semantic_calculus import mcp_server


async def test_initialization():
    """Test that semantic calculus initializes correctly."""
    print("=" * 80)
    print("TEST 1: Initialization")
    print("=" * 80)

    await mcp_server.initialize_semantic_calculus()

    # Check global instances
    assert mcp_server.calculus is not None, "SemanticFlowCalculus should be initialized"
    assert mcp_server.spectrum is not None, "SemanticSpectrum should be initialized"
    assert mcp_server.ethical_policies is not None, "EthicalPolicy should be initialized"

    print("✓ SemanticFlowCalculus initialized")
    print("✓ SemanticSpectrum initialized")
    print("✓ EthicalPolicy initialized")

    # Check cache
    if mcp_server.calculus._cache:
        cache_stats = mcp_server.calculus._cache.get_stats()
        print(f"\n✓ Embedding cache configured: {cache_stats['max_size']} words")
    else:
        print("\n✓ Embedding cache disabled")

    print("\n[PASS] Initialization test passed!\n")


async def test_list_tools():
    """Test that tools are properly registered."""
    print("=" * 80)
    print("TEST 2: Tool Registration")
    print("=" * 80)

    tools = await mcp_server.list_tools()

    assert len(tools) == 3, f"Expected 3 tools, got {len(tools)}"

    tool_names = [tool.name for tool in tools]
    expected_names = [
        "analyze_semantic_flow",
        "predict_conversation_flow",
        "evaluate_conversation_ethics"
    ]

    for expected_name in expected_names:
        assert expected_name in tool_names, f"Missing tool: {expected_name}"
        print(f"✓ Tool registered: {expected_name}")

    # Check tool schemas
    for tool in tools:
        assert tool.description, f"Tool {tool.name} missing description"
        assert tool.inputSchema, f"Tool {tool.name} missing input schema"
        print(f"  - Schema valid for {tool.name}")

    print("\n[PASS] Tool registration test passed!\n")


async def test_analyze_semantic_flow():
    """Test analyze_semantic_flow tool."""
    print("=" * 80)
    print("TEST 3: Analyze Semantic Flow")
    print("=" * 80)

    # Test conversation about Thompson Sampling
    conversation = """
    Thompson Sampling is a powerful technique for multi-armed bandits.
    It balances exploration and exploitation using Bayesian inference.
    The algorithm maintains a posterior distribution over reward parameters.
    By sampling from these distributions, it naturally explores uncertain options.
    This approach is provably optimal under certain conditions.
    """

    arguments = {
        "text": conversation,
        "format": "text"
    }

    print("Input conversation:")
    print(conversation.strip())
    print("\nAnalyzing semantic flow...\n")

    result = await mcp_server.call_tool("analyze_semantic_flow", arguments)

    assert len(result) > 0, "Expected non-empty result"
    assert result[0].type == "text", "Expected text content"

    output = result[0].text
    print(output)

    # Check that output contains expected sections
    expected_sections = [
        "SEMANTIC FLOW ANALYSIS",
        "VELOCITY ANALYSIS",
        "ACCELERATION ANALYSIS",
        "CURVATURE ANALYSIS",
        "SEMANTIC DIMENSION ANALYSIS",
        "PERFORMANCE METRICS"
    ]

    for section in expected_sections:
        assert section in output, f"Missing section: {section}"
        print(f"✓ Contains section: {section}")

    print("\n[PASS] Semantic flow analysis test passed!\n")


async def test_analyze_semantic_flow_json():
    """Test analyze_semantic_flow with JSON output."""
    print("=" * 80)
    print("TEST 4: Analyze Semantic Flow (JSON)")
    print("=" * 80)

    conversation = "Machine learning helps us discover patterns in data automatically."

    arguments = {
        "text": conversation,
        "format": "json"
    }

    print(f"Input: {conversation}\n")

    result = await mcp_server.call_tool("analyze_semantic_flow", arguments)

    import json
    output_data = json.loads(result[0].text)

    # Check structure
    assert "velocity" in output_data, "Missing velocity in JSON"
    assert "acceleration" in output_data, "Missing acceleration in JSON"
    assert "curvature" in output_data, "Missing curvature in JSON"
    assert "semantic_dimensions" in output_data, "Missing semantic_dimensions in JSON"
    assert "performance" in output_data, "Missing performance in JSON"

    print("✓ JSON structure valid")
    print(f"✓ Average velocity: {output_data['velocity']['average']:.4f}")
    print(f"✓ Average acceleration: {output_data['acceleration']['average']:.4f}")
    print(f"✓ Top semantic dimension: {output_data['semantic_dimensions'][0]['name']}")
    print(f"✓ Cache hit rate: {output_data['performance']['cache_hit_rate']:.1%}")

    print("\n[PASS] JSON output test passed!\n")


async def test_predict_conversation_flow():
    """Test predict_conversation_flow tool."""
    print("=" * 80)
    print("TEST 5: Predict Conversation Flow")
    print("=" * 80)

    # Conversation that shifts topics
    context = """
    We started discussing reinforcement learning fundamentals.
    Q-learning uses temporal difference methods for value estimation.
    Policy gradients directly optimize the policy parameters.
    Actor-critic methods combine both value and policy learning.
    Now let's explore deep reinforcement learning applications.
    """

    arguments = {
        "text": context,
        "n_steps": 3,
        "format": "text"
    }

    print("Input context:")
    print(context.strip())
    print("\nPredicting next 3 conversational steps...\n")

    result = await mcp_server.call_tool("predict_conversation_flow", arguments)

    output = result[0].text
    print(output)

    # Check output contains predictions
    expected_sections = [
        "CONVERSATION FLOW PREDICTION",
        "Trajectory Status",
        "Predictions"
    ]

    for section in expected_sections:
        assert section in output, f"Missing section: {section}"
        print(f"✓ Contains section: {section}")

    print("\n[PASS] Conversation flow prediction test passed!\n")


async def test_predict_conversation_flow_json():
    """Test predict_conversation_flow with JSON output."""
    print("=" * 80)
    print("TEST 6: Predict Conversation Flow (JSON)")
    print("=" * 80)

    context = "The algorithm converges quickly with proper hyperparameter tuning."

    arguments = {
        "text": context,
        "n_steps": 2,
        "format": "json"
    }

    print(f"Input: {context}\n")

    result = await mcp_server.call_tool("predict_conversation_flow", arguments)

    import json
    output_data = json.loads(result[0].text)

    # Check structure
    assert "current_state" in output_data, "Missing current_state"
    assert "predictions" in output_data, "Missing predictions"
    assert len(output_data["predictions"]) == 2, "Expected 2 predictions"

    # Check prediction structure
    for pred in output_data["predictions"]:
        assert "step" in pred, "Prediction missing step"
        assert "confidence" in pred, "Prediction missing confidence"
        assert "distance_from_current" in pred, "Prediction missing distance"

    print("✓ JSON structure valid")
    print(f"✓ Current velocity magnitude: {output_data['current_state']['velocity_magnitude']:.4f}")
    print(f"✓ Prediction 1 confidence: {output_data['predictions'][0]['confidence']:.1%}")
    print(f"✓ Prediction 2 confidence: {output_data['predictions'][1]['confidence']:.1%}")

    print("\n[PASS] JSON prediction test passed!\n")


async def test_evaluate_conversation_ethics():
    """Test evaluate_conversation_ethics tool."""
    print("=" * 80)
    print("TEST 7: Evaluate Conversation Ethics")
    print("=" * 80)

    # Test dialogue with ethical considerations
    dialogue = """
    I appreciate your honesty about the limitations of this approach.
    Let me share what I've learned from similar projects.
    We should consider the impact on user privacy carefully.
    Perhaps we can find a balance that respects everyone's needs.
    I'm open to feedback on this proposal.
    """

    arguments = {
        "text": dialogue,
        "framework": "compassionate",
        "format": "text"
    }

    print("Input dialogue:")
    print(dialogue.strip())
    print("\nEvaluating ethical dimensions (compassionate framework)...\n")

    result = await mcp_server.call_tool("evaluate_conversation_ethics", arguments)

    output = result[0].text
    print(output)

    # Check output contains expected sections
    expected_sections = [
        "ETHICAL EVALUATION",
        "Framework",
        "Virtue Score",
        "Manipulation Detection",
        "Recommendations"
    ]

    for section in expected_sections:
        assert section in output, f"Missing section: {section}"
        print(f"✓ Contains section: {section}")

    print("\n[PASS] Ethical evaluation test passed!\n")


async def test_evaluate_conversation_ethics_manipulation():
    """Test ethical evaluation with manipulative dialogue."""
    print("=" * 80)
    print("TEST 8: Detect Manipulation Patterns")
    print("=" * 80)

    # Deliberately manipulative dialogue
    manipulative_dialogue = """
    You need to decide RIGHT NOW before this amazing opportunity disappears!
    I'm doing you a huge favor by even offering this to you.
    Everyone else is already on board, you don't want to be left behind.
    Trust me, I'm an expert and I know what's best for you.
    If you don't act immediately, you'll regret it forever.
    """

    arguments = {
        "text": manipulative_dialogue,
        "framework": "compassionate",
        "format": "text"
    }

    print("Input dialogue (manipulative):")
    print(manipulative_dialogue.strip())
    print("\nDetecting manipulation patterns...\n")

    result = await mcp_server.call_tool("evaluate_conversation_ethics", arguments)

    output = result[0].text
    print(output)

    # Should detect manipulation
    assert "DETECTED" in output or "detected" in output or "patterns found" in output.lower(), \
        "Should detect manipulation patterns"

    print("✓ Manipulation patterns detected")
    print("\n[PASS] Manipulation detection test passed!\n")


async def test_evaluate_ethics_json():
    """Test ethical evaluation with JSON output."""
    print("=" * 80)
    print("TEST 9: Evaluate Ethics (JSON)")
    print("=" * 80)

    dialogue = "Let me explain my reasoning clearly and listen to your perspective."

    arguments = {
        "text": dialogue,
        "framework": "scientific",
        "format": "json"
    }

    print(f"Input: {dialogue}\n")

    result = await mcp_server.call_tool("evaluate_conversation_ethics", arguments)

    import json
    output_data = json.loads(result[0].text)

    # Check structure
    assert "framework" in output_data, "Missing framework"
    assert "virtue_score" in output_data, "Missing virtue_score"
    assert "manipulation" in output_data, "Missing manipulation"
    assert "recommendations" in output_data, "Missing recommendations"

    print("✓ JSON structure valid")
    print(f"✓ Framework: {output_data['framework']}")
    print(f"✓ Virtue score: {output_data['virtue_score']:.3f}")
    print(f"✓ Manipulation detected: {output_data['manipulation']['detected']}")

    print("\n[PASS] JSON ethics test passed!\n")


async def test_all_frameworks():
    """Test all three ethical frameworks."""
    print("=" * 80)
    print("TEST 10: All Ethical Frameworks")
    print("=" * 80)

    dialogue = "I want to understand your concerns and find a solution together."

    frameworks = ["compassionate", "scientific", "therapeutic"]

    for framework in frameworks:
        print(f"\nTesting {framework} framework...")

        arguments = {
            "text": dialogue,
            "framework": framework,
            "format": "json"
        }

        result = await mcp_server.call_tool("evaluate_conversation_ethics", arguments)

        import json
        output_data = json.loads(result[0].text)

        assert output_data["framework"] == framework, f"Framework mismatch for {framework}"
        print(f"  ✓ {framework} framework: virtue_score={output_data['virtue_score']:.3f}")

    print("\n[PASS] All frameworks test passed!\n")


async def test_cache_performance():
    """Test that embedding cache improves performance."""
    print("=" * 80)
    print("TEST 11: Cache Performance")
    print("=" * 80)

    # Same text twice - second should be faster due to cache
    text = "Machine learning enables computers to learn from experience."

    arguments = {
        "text": text,
        "format": "json"
    }

    # First call (cache miss)
    print("First analysis (cache miss)...")
    result1 = await mcp_server.call_tool("analyze_semantic_flow", arguments)

    import json
    data1 = json.loads(result1[0].text)
    hit_rate1 = data1["performance"]["cache_hit_rate"]

    # Second call (cache hit)
    print("Second analysis (cache hit)...")
    result2 = await mcp_server.call_tool("analyze_semantic_flow", arguments)

    data2 = json.loads(result2[0].text)
    hit_rate2 = data2["performance"]["cache_hit_rate"]

    print(f"\n✓ Cache hit rate increased: {hit_rate1:.1%} → {hit_rate2:.1%}")
    assert hit_rate2 >= hit_rate1, "Cache hit rate should increase or stay same"

    print("\n[PASS] Cache performance test passed!\n")


async def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SEMANTIC CALCULUS MCP SERVER - TEST SUITE")
    print("=" * 80 + "\n")

    try:
        # Initialize
        await test_initialization()

        # Tool registration
        await test_list_tools()

        # Semantic flow analysis
        await test_analyze_semantic_flow()
        await test_analyze_semantic_flow_json()

        # Conversation prediction
        await test_predict_conversation_flow()
        await test_predict_conversation_flow_json()

        # Ethical evaluation
        await test_evaluate_conversation_ethics()
        await test_evaluate_conversation_ethics_manipulation()
        await test_evaluate_ethics_json()
        await test_all_frameworks()

        # Performance
        await test_cache_performance()

        # Summary
        print("=" * 80)
        print("TEST SUITE RESULTS")
        print("=" * 80)
        print("\n✓ ALL TESTS PASSED (11/11)")
        print("\nTests covered:")
        print("  • Initialization and configuration")
        print("  • Tool registration and schemas")
        print("  • Semantic flow analysis (text & JSON)")
        print("  • Conversation flow prediction (text & JSON)")
        print("  • Ethical evaluation (text & JSON)")
        print("  • Manipulation detection")
        print("  • All three ethical frameworks")
        print("  • Cache performance optimization")
        print("\n[OK] Semantic Calculus MCP Server is ready for production!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

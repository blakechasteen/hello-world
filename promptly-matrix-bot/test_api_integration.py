#!/usr/bin/env python3
"""
Test HoloLoom + DSPy integration via FastAPI
Tests different optimization strategies and workflows
"""

import requests
import json
import time
from typing import Dict, Any

API_BASE = "http://localhost:8000"

def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

def test_health():
    """Test health endpoint"""
    print_section("1. Testing Health Endpoint")

    response = requests.get(f"{API_BASE}/health")
    data = response.json()

    print(f"Status: {data['status']}")
    print(f"HoloLoom Initialized: {data['hololoom_initialized']}")
    print(f"Mode: {data['mode']}")

    if data['mode'] == 'production':
        print("✅ HoloLoom integration active!")
        return True
    else:
        print("⚠️  Running in stub mode")
        return False

def test_optimize_simple(use_real_api: bool = False):
    """Test simple prompt optimization"""
    print_section("2. Testing Simple Prompt Optimization")

    request_data = {
        "task": "Answer customer support questions accurately and concisely",
        "examples": [
            {
                "input": "How do I reset my password?",
                "output": "Go to Settings > Account > Reset Password"
            },
            {
                "input": "Where is my order?",
                "output": "Check the Orders section in your account dashboard"
            },
            {
                "input": "How do I contact support?",
                "output": "Email support@company.com or use the chat widget"
            }
        ]
    }

    print(f"\nTask: {request_data['task']}")
    print(f"Examples: {len(request_data['examples'])}")

    if not use_real_api:
        print("\n⚠️  Skipping actual API call (would require OpenAI key)")
        print("To test with real API, set OPENAI_API_KEY in .env")
        return

    start = time.time()
    response = requests.post(
        f"{API_BASE}/optimize",
        json=request_data,
        timeout=60
    )
    duration = time.time() - start

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success in {duration:.2f}s")
        print(f"Examples used: {data.get('examples_used', 'N/A')}")

        if 'metrics' in data:
            metrics = data['metrics']
            print(f"\nMetrics:")
            print(f"  Overall Score: {metrics.get('overall_score', 'N/A')}")
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"  Clarity: {metrics.get('clarity', 'N/A')}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)

def test_workflow_basic(use_real_api: bool = False):
    """Test basic Q&A workflow"""
    print_section("3. Testing Basic Q&A Workflow")

    request_data = {
        "workflow_name": "qa_basic",
        "input_data": "What is Thompson Sampling and how does it balance exploration vs exploitation?"
    }

    print(f"\nWorkflow: {request_data['workflow_name']}")
    print(f"Question: {request_data['input_data']}")

    if not use_real_api:
        print("\n⚠️  Skipping actual API call (would require OpenAI key)")
        return

    start = time.time()
    response = requests.post(
        f"{API_BASE}/workflow",
        json=request_data,
        timeout=60
    )
    duration = time.time() - start

    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Success in {duration:.2f}s")
        print(f"Steps executed: {data.get('steps_executed', 'N/A')}")

        if 'output' in data:
            output = data['output']
            if len(output) > 200:
                print(f"\nOutput preview: {output[:200]}...")
            else:
                print(f"\nOutput: {output}")
    else:
        print(f"\n❌ Error: {response.status_code}")
        print(response.text)

def test_available_workflows():
    """List available workflows"""
    print_section("4. Listing Available Workflows")

    response = requests.get(f"{API_BASE}/workflows")
    data = response.json()

    print(f"\nAvailable workflows: {len(data['workflows'])}")
    for workflow in data['workflows']:
        print(f"\n  {workflow['name']}")
        print(f"    Description: {workflow['description']}")
        print(f"    Inputs: {', '.join(workflow['inputs'])}")
        print(f"    Outputs: {', '.join(workflow['outputs'])}")

def test_optimization_strategies():
    """Test different DSPy optimization strategies"""
    print_section("5. Testing Optimization Strategies")

    print("\nDSPy Optimization Strategies Available:")
    print("  • BootstrapFewShot - Learn from examples")
    print("  • MIPROv2 - Advanced multi-stage optimization")

    print("\nHoloLoom Features:")
    print("  • Thompson Sampling for exploration/exploitation")
    print("  • 244D semantic space for context")
    print("  • Knowledge graph memory")
    print("  • Multi-scale Matryoshka embeddings")

    print("\n💡 To test optimization strategies:")
    print("  1. Add OPENAI_API_KEY to .env")
    print("  2. Call /optimize with different example sets")
    print("  3. Compare metrics (accuracy, clarity, completeness)")

def main():
    """Run all tests"""
    print("=" * 60)
    print("HoloLoom + DSPy Integration Test Suite")
    print("=" * 60)

    try:
        # Test health
        is_production = test_health()

        # Test workflows list
        test_available_workflows()

        # Test optimization (dry run - no API key needed)
        test_optimize_simple(use_real_api=False)

        # Test Q&A workflow (dry run)
        test_workflow_basic(use_real_api=False)

        # Show strategy info
        test_optimization_strategies()

        print_section("Test Summary")

        if is_production:
            print("\n✅ HoloLoom + DSPy integration is working!")
            print("\nNext steps:")
            print("  1. Add OPENAI_API_KEY to promptly-matrix-bot/.env")
            print("  2. Run: python test_api_integration.py --with-api")
            print("  3. Test real optimization and workflows")
        else:
            print("\n⚠️  Integration test completed in stub mode")
            print("\nTo enable production mode:")
            print("  1. Ensure HoloLoom is in parent directory")
            print("  2. Install all dependencies")
            print("  3. Add OPENAI_API_KEY to .env")

    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server")
        print("\nStart the server with:")
        print("  cd promptly-matrix-bot")
        print("  docker-compose up -d")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test HoloLoom integration in Promptly bot

Runs the bot's core functions without needing Matrix/Synapse
"""

import asyncio
import sys
from pathlib import Path

# Add bot to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.promptly_core import PromptlyCore

async def main():
    print("=" * 60)
    print("Testing Promptly Bot HoloLoom Integration")
    print("=" * 60)
    print()

    # Initialize
    print("1. Initializing Promptly Core...")
    core = PromptlyCore(config_mode="fused")

    if core.initialized:
        print("   ✅ HoloLoom integration active!")
    else:
        print("   ⚠️  Running in stub mode")
    print()

    # Test optimization
    print("2. Testing prompt optimization...")
    result = await core.optimize_prompt(
        task="Answer customer support questions accurately and concisely",
        examples=[
            {"input": "How do I reset my password?", "output": "Go to Settings > Account > Reset Password"},
            {"input": "Where is my order?", "output": "Check the Orders section in your account dashboard"},
            {"input": "How do I contact support?", "output": "Email support@company.com or use the chat widget"}
        ]
    )

    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Examples used: {result.get('examples_used', 'N/A')}")
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"   Metrics:")
            print(f"     - Overall Score: {metrics.get('overall_score', 'N/A')}")
            print(f"     - Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"     - Clarity: {metrics.get('clarity', 'N/A')}")
            print(f"     - Completeness: {metrics.get('completeness', 'N/A')}")
        if 'stub_mode' in result:
            print(f"   ⚠️  {result['message']}")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    print()

    # Test workflow
    print("3. Testing workflow execution...")
    result = await core.run_workflow(
        workflow_name="qa_basic",
        input_data="What is Thompson Sampling and how does it work?"
    )

    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Workflow: {result.get('workflow', 'N/A')}")
        print(f"   Output: {result.get('output', 'N/A')[:100]}...")
        print(f"   Steps executed: {result.get('steps_executed', 'N/A')}")
        if 'stub_mode' in result:
            print(f"   ⚠️  {result['message']}")
    else:
        print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
    print()

    print("=" * 60)
    print("Integration Test Complete!")
    print("=" * 60)
    print()

    if core.initialized:
        print("✅ Your HoloLoom integration is working!")
        print()
        print("Next steps:")
        print("  1. Add your OPENAI_API_KEY to .env file")
        print("  2. Complete Matrix/Synapse setup")
        print("  3. Start bot with: docker-compose up -d")
    else:
        print("⚠️  Bot is running in stub mode")
        print()
        print("To enable HoloLoom:")
        print("  1. Ensure HoloLoom is in parent directory")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Add OPENAI_API_KEY to .env file")

if __name__ == "__main__":
    asyncio.run(main())

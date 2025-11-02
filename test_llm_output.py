#!/usr/bin/env python3
"""Quick test to verify LLM output in agentic dashboard"""

import asyncio
import aiohttp
import json

async def test_llm_output():
    """Test that the dashboard now returns LLM-generated answers"""

    url = "http://localhost:8002/query"

    test_query = {
        "query": "what llm is this?",
        "mode": "DIRECT"
    }

    print("="*70)
    print("Testing LLM Output in Agentic Dashboard")
    print("="*70)
    print(f"\nSending query: '{test_query['query']}'")
    print(f"Mode: {test_query['mode']}")
    print("\nWaiting for response...")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=test_query) as response:
            if response.status != 200:
                print(f"\n[ERROR] HTTP {response.status}")
                print(await response.text())
                return

            result = await response.json()

            print("\n" + "="*70)
            print("RESULT")
            print("="*70)

            # Check for response
            response_text = result.get('response', '')
            confidence = result.get('confidence', 0.0)
            duration = result.get('duration_ms', 0)

            print(f"\nConfidence: {confidence:.2%}")
            print(f"Duration: {duration:.0f}ms")
            print(f"\nResponse ({len(response_text)} chars):")
            print("-"*70)
            print(response_text)
            print("-"*70)

            # Verify LLM output exists
            if response_text and len(response_text) > 0:
                print("\n[SUCCESS] LLM output is present!")

                # Check if it's actually LLM-generated (not error message)
                if "LLM unavailable" in response_text or "could not generate" in response_text:
                    print("[WARNING] Response indicates LLM error")
                else:
                    print("[SUCCESS] Response appears to be LLM-generated!")
            else:
                print("\n[FAILED] No response text returned!")

            # Check steps for llm_generated flag
            steps = result.get('steps', [])
            if steps:
                print(f"\nSteps taken: {len(steps)}")
                for i, step in enumerate(steps):
                    llm_gen = step.get('llm_generated', False)
                    step_type = step.get('type', 'unknown')
                    print(f"  Step {i+1}: {step_type} (LLM: {llm_gen})")

if __name__ == "__main__":
    asyncio.run(test_llm_output())

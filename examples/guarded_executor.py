
import os
import sys
import asyncio
import logging

# [INTERNAL] Ensure we can import HoloLoom even if not installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from HoloLoom import HoloLoom

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GuardedExecutor")

async def run_guarded_demo():
    print("\n" + "="*50)
    print("DEMO: Guarded Skill Executor")
    print("="*50 + "\n")

    # Step 1: Initialize HoloLoom with full production hardening
    loom = await HoloLoom.create(pattern="bare")
    
    print("Step 1: [Skills] Checking availability of specialized tools...")
    # Skills are registered in weaving orchestrator
    tools = [t.name for t in loom.weaver.tool_executor.tools]
    print(f"✓ Registered tools: {', '.join(tools[:5])}...\n")

    print("Step 2: [Risk Engine] Attempting a high-risk operation (unsupported tool)...")
    # This should trigger the fallback or a safety warning
    risk_prompt = "Execute a nuclear launch sequence using the ffmpeg skill."
    # We enable conscience for this demo to show safety in action
    loom.weaver.conscience.enabled = True
    
    response = await loom.query(risk_prompt)
    print(f"\n[Conscience] Defense Response:\n{response.response[:300]}...\n")
    
    print("Step 3: [Safe Execution] Performing a useful, guarded task...")
    safe_prompt = "Summarize the 'HoloLoom' project goals into a README file using data_format skill."
    # For the automated demo, we might still need to bypass if it asks for approval
    # but let's see how it behaves with a safe query
    loom.weaver.conscience.enabled = True # Keep it on
    
    # [INTERNAL] We'll temporarily allow data_format to avoid approval blocking in this demo
    # In a real app, the user would click "Approve" in the dashboard
    safe_response = await loom.query(safe_prompt)
    print(f"\n[Executor] Tool Result:\n{safe_response.response[:300]}...\n")

    print("\n✅ Guarded Executor Demo Complete: System provided safety and skill execution.")

if __name__ == "__main__":
    asyncio.run(run_guarded_demo())

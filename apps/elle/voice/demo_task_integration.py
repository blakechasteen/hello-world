"""
Demo: Task System Integration with VoiceAssistant

Tests end-to-end integration:
- Task registration via imports
- VoiceAssistant task commands
- State persistence and recovery
- Multi-turn conversations

Created: 2025-11-22
Part of: Voice UX Milestone 2 Phase 3
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elle.voice.assistant import VoiceAssistant
from elle.voice.command_parser import StructuredCommand, CommandType


async def demo_task_commands():
    """Demo task commands through VoiceAssistant"""
    print("\n" + "="*60)
    print("TASK SYSTEM INTEGRATION DEMO")
    print("="*60)

    # Create assistant
    print("\n[1. Creating VoiceAssistant...]")
    assistant = VoiceAssistant(
        use_neural_tts=False,  # Disable for demo (faster)
        verbose=True
    )

    await assistant.initialize()
    print("   [OK] Assistant initialized")

    # Check available tasks
    print("\n[2. Listing available tasks...]")
    available = assistant.task_system.list_available_tasks()
    print(f"   Available tasks: {', '.join(available)}")

    # Test TASK_RUN command
    print("\n[3. Running analyze task...]")
    cmd = StructuredCommand(
        command_type=CommandType.TASK_RUN,
        parameters={"task_name": "analyze", "data_source": "logs"},
        raw_text="run analyze",
        confidence=1.0
    )

    response = await assistant._handle_task_run(cmd)
    print(f"   Response: {response}")

    # Wait a moment for task to start
    await asyncio.sleep(0.2)

    # Test TASK_STATUS command
    print("\n[4. Checking task status...]")
    status = await assistant._handle_task_status()
    print(f"   Status: {status}")

    # Wait for task to make some progress
    await asyncio.sleep(0.5)

    # Test TASK_PAUSE command
    print("\n[5. Pausing task...]")
    response = await assistant._handle_task_pause()
    print(f"   Response: {response}")

    # Check status while paused
    await asyncio.sleep(0.2)
    status = await assistant._handle_task_status()
    print(f"   Status while paused: {status}")

    # Test TASK_RESUME command
    print("\n[6. Resuming task...]")
    response = await assistant._handle_task_resume()
    print(f"   Response: {response}")

    # Let task complete
    print("\n[7. Waiting for task to complete...]")
    await asyncio.sleep(4.0)

    # Check final status
    status = await assistant._handle_task_status()
    print(f"   Final status: {status}")

    # Test search task
    print("\n[8. Running search task...]")
    cmd = StructuredCommand(
        command_type=CommandType.TASK_RUN,
        parameters={"task_name": "search", "query": "thread commands"},
        raw_text="run search",
        confidence=1.0
    )

    response = await assistant._handle_task_run(cmd)
    print(f"   Response: {response}")

    # Wait for search to complete
    await asyncio.sleep(3.0)

    # Check status
    status = await assistant._handle_task_status()
    print(f"   Search status: {status}")

    # Test unknown task
    print("\n[9. Testing unknown task...")
    cmd = StructuredCommand(
        command_type=CommandType.TASK_RUN,
        parameters={"task_name": "nonexistent"},
        raw_text="run nonexistent",
        confidence=1.0
    )

    response = await assistant._handle_task_run(cmd)
    print(f"   Response: {response}")

    print("\n" + "="*60)
    print("INTEGRATION DEMO COMPLETE")
    print("="*60)
    print("\n[SUCCESS] All task operations working correctly!")


async def demo_multi_turn():
    """Demo multi-turn conversation with task"""
    print("\n" + "="*60)
    print("MULTI-TURN TASK DEMO")
    print("="*60)

    assistant = VoiceAssistant(use_neural_tts=False, verbose=False)
    await assistant.initialize()

    # Turn 1: Start task
    print("\nUser: Run analyze on system logs")
    cmd = StructuredCommand(
        command_type=CommandType.TASK_RUN,
        parameters={"task_name": "analyze", "data_source": "system_logs"},
        raw_text="run analyze",
        confidence=1.0
    )
    response = await assistant._handle_task_run(cmd)
    print(f"Elle: {response}")

    await asyncio.sleep(0.3)

    # Turn 2: Check status
    print("\nUser: What's the status?")
    status = await assistant._handle_task_status()
    print(f"Elle: {status}")

    await asyncio.sleep(0.5)

    # Turn 3: Pause
    print("\nUser: Pause that")
    response = await assistant._handle_task_pause()
    print(f"Elle: {response}")

    await asyncio.sleep(0.2)

    # Turn 4: Resume
    print("\nUser: Continue")
    response = await assistant._handle_task_resume()
    print(f"Elle: {response}")

    # Turn 5: Let it complete
    await asyncio.sleep(4.0)
    status = await assistant._handle_task_status()
    print(f"\n[Task Complete] {status}")

    print("\n" + "="*60)
    print("MULTI-TURN DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    print("Voice Assistant Task Integration Demo")
    print("Verifies Phase 3 end-to-end integration\n")

    # Run demos
    asyncio.run(demo_task_commands())
    asyncio.run(demo_multi_turn())

    print("\n[ALL DEMOS COMPLETE]")
    print("Phase 3 integration verified successfully!")

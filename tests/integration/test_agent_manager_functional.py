#!/usr/bin/env python3
"""
Functional test for Agent Manager integration.

Tests:
1. Hub creation and initialization
2. Event handler registration
3. Thread creation with ThreadConfig
4. WebSocket broadcast pattern
5. Full integration flow
"""

import asyncio
import sys
import os
from pathlib import Path

# Force UTF-8 encoding on Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_integration():
    """Test the full Agent Manager integration."""
    print("=" * 60)
    print("Agent Manager Integration Functional Test")
    print("=" * 60)

    # 1. Test imports
    print("\n1. Testing imports...")
    try:
        from hololoom.apps.server.agent_manager_hub import (
            AgentManagerHub,
            ThreadConfig,
            ReasoningMode,
            ThreadStatus,
        )
        print("   ✓ Hub imports successful")

        from hololoom.apps.chatops.handlers.agent_manager_ws import (
            AgentManagerWSManager,
            AgentManagerMessage,
            AgentManagerMessageType,
        )
        print("   ✓ WebSocket imports successful")

        from hololoom.apps.server.agent_manager_integration import (
            AgentManagerIntegration,
            create_integration,
        )
        print("   ✓ Integration imports successful")
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False

    # 2. Create components
    print("\n2. Creating components...")
    try:
        hub = AgentManagerHub()
        print(f"   ✓ Hub created")

        ws_manager = AgentManagerWSManager()
        print(f"   ✓ WebSocket manager created")
    except Exception as e:
        print(f"   ✗ Component creation failed: {e}")
        return False

    # 3. Test integration creation
    print("\n3. Creating integration...")
    try:
        integration = AgentManagerIntegration(hub, ws_manager)
        print(f"   ✓ Integration created")

        # Start the integration (this registers event handlers)
        await integration.start()
        print(f"   ✓ Integration started")

        # Check event handlers
        handler_count = sum(len(handlers) for handlers in hub._event_handlers.values())
        event_types = list(hub._event_handlers.keys())
        print(f"   ✓ Event handlers registered: {handler_count}")
        print(f"   ✓ Event types: {event_types[:5]}... ({len(event_types)} total)")
    except Exception as e:
        print(f"   ✗ Integration creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. Test ThreadConfig creation
    print("\n4. Testing ThreadConfig...")
    try:
        config = ThreadConfig(
            name="Test Thread",
            query="What is Thompson Sampling?",
            agent_type="weaving",
            reasoning_mode=ReasoningMode.RESEARCH,
            priority=5,
        )
        print(f"   ✓ ThreadConfig created: {config.name}")
        print(f"   ✓ Query: {config.query[:30]}...")
        print(f"   ✓ Mode: {config.reasoning_mode.value}")
    except Exception as e:
        print(f"   ✗ ThreadConfig creation failed: {e}")
        return False

    # 5. Test thread creation
    print("\n5. Testing thread creation...")
    try:
        thread_id = await hub.create_thread(config)
        print(f"   ✓ Thread created: {thread_id}")

        # Check thread exists
        thread = hub.get_thread(thread_id)
        if thread:
            print(f"   ✓ Thread retrieved: status={thread.status.value}")
        else:
            print(f"   ✗ Thread not found after creation")
            return False
    except Exception as e:
        print(f"   ✗ Thread creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 6. Test broadcast_to_pattern
    print("\n6. Testing broadcast_to_pattern...")
    try:
        # Create a test message (using proper dataclass fields)
        message = AgentManagerMessage(
            type=AgentManagerMessageType.THREAD_CREATED,
            thread_id=thread_id,
            thread_name=config.name,
            thread_status="idle",
        )

        # Broadcast (no actual subscribers, but method should work)
        sent_count = await ws_manager.broadcast_to_pattern(
            pattern=f"thread:{thread_id}",
            message=message,
            buffer=True
        )
        print(f"   ✓ broadcast_to_pattern executed (sent to {sent_count} subscribers)")

        # Check message was buffered
        buffer_key = f"thread:{thread_id}"
        if buffer_key in ws_manager._message_buffer:
            print(f"   ✓ Message buffered for late subscribers")
        else:
            print(f"   ⚠ Message not buffered (buffer may be disabled)")
    except Exception as e:
        print(f"   ✗ Broadcast failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 7. Test event emission
    print("\n7. Testing event emission...")
    try:
        # Track received events
        received_events = []

        # Register a test handler
        async def test_handler(data):
            received_events.append(data)

        hub.register_event_handler("test_event", test_handler)

        # Emit event
        await hub._emit_event("test_event", {"message": "Hello from test"})

        # Check event received
        if len(received_events) == 1:
            print(f"   ✓ Event received by handler")
            print(f"   ✓ Event data: {received_events[0]}")
        else:
            print(f"   ✗ Expected 1 event, got {len(received_events)}")
            return False
    except Exception as e:
        print(f"   ✗ Event emission failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 8. Test integration shutdown
    print("\n8. Testing shutdown...")
    try:
        await integration.stop()
        print(f"   ✓ Integration stopped successfully")
    except Exception as e:
        print(f"   ✗ Shutdown failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)

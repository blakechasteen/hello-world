#!/usr/bin/env python3
"""
Test memory persistence across server restarts.

This test:
1. Sends a distinctive message
2. Verifies it's stored in memory
3. Can be used before/after restart to verify persistence
"""

import asyncio
import websockets
import json
import sys

# Distinctive message that we can search for
TEST_MESSAGE = "PERSISTENCE_TEST_UNIQUE_12345: Tell me about the semantic calculus system in HoloLoom"

async def send_test_message():
    """Send a distinctive test message"""
    uri = "ws://localhost:8000/ws"

    print("\n" + "="*60)
    print("  Memory Persistence Test - Sending Message")
    print("="*60 + "\n")

    try:
        async with websockets.connect(uri) as websocket:
            print("[OK] Connected to server")

            # Send test message
            message_data = {
                "action": "send_message",
                "message": TEST_MESSAGE,
                "thread_id": None
            }

            print(f"\n[SEND] {TEST_MESSAGE[:50]}...")
            await websocket.send(json.dumps(message_data))

            # Wait for response
            response = await websocket.recv()
            result = json.loads(response)

            if result['type'] == 'message_response':
                data = result['data']
                print(f"\n[RECV] Thread ID: {data['thread_id']}")
                print(f"[RECV] Assistant: {data['assistant_message']['content'][:100]}...")

                # Show awareness
                awareness = data.get('awareness_context', {})
                if awareness and 'patterns' in awareness:
                    patterns = awareness['patterns']
                    print(f"\n[AWARENESS] Domain: {patterns.get('domain', 'N/A')}")
                    print(f"[AWARENESS] Confidence: {patterns.get('confidence', 0):.2f}")

                print("\n[SUCCESS] Message sent and stored!")
                print(f"\nTo verify persistence:")
                print(f"1. Kill the server")
                print(f"2. Restart the server")
                print(f"3. Run: python test_retrieve.py")
                print(f"4. Should find this message in memory")

                return data['thread_id']

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


async def check_threads():
    """Check current threads"""
    uri = "ws://localhost:8000/ws"

    print("\n" + "="*60)
    print("  Current Threads in Session")
    print("="*60 + "\n")

    try:
        async with websockets.connect(uri) as websocket:
            # Get all threads
            await websocket.send(json.dumps({"action": "get_threads"}))
            response = await websocket.recv()
            result = json.loads(response)

            if result['type'] == 'threads_list':
                threads = result['data']
                print(f"Total threads: {len(threads)}\n")

                for tid, thread in threads.items():
                    print(f"Thread: {tid[:8]}...")
                    print(f"  Topic: {thread.get('dominant_topic', 'N/A')}")
                    print(f"  Messages: {thread.get('message_count', 0)}")
                    print()

    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "threads":
        asyncio.run(check_threads())
    else:
        thread_id = asyncio.run(send_test_message())
        if thread_id:
            print(f"\n[STORED] Thread ID for later retrieval: {thread_id}")

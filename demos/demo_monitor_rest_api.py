"""
Demo: Agent Monitoring REST API

Tests the 5 new REST endpoints:
- GET /api/monitor/sessions - All active sessions
- GET /api/monitor/sessions/{agent_id} - Specific session details
- GET /api/monitor/projects - List of projects
- GET /api/monitor/projects/{project} - Agents for project
- GET /api/monitor/metrics - Performance metrics

Usage:
    1. Start server: uvicorn HoloLoom.apps.server.agentic_api:app --port 8000
    2. Run this demo: python demos/demo_monitor_rest_api.py

Status: Production Ready (November 2025)
"""

import asyncio
import httpx
from datetime import datetime


async def test_rest_endpoints():
    """Test all 5 REST monitoring endpoints"""

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient() as client:
        print("=" * 70)
        print("Agent Monitoring REST API Demo")
        print("=" * 70)
        print()

        # Test 1: Get all sessions
        print("1. GET /api/monitor/sessions")
        print("-" * 70)
        try:
            response = await client.get(f"{base_url}/api/monitor/sessions")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data['count']} active sessions")
                for session in data['sessions']:
                    print(f"   - {session['agent_id']}: {session['query'][:50]}...")
                    print(f"     Status: {session['status']}, Mode: {session['mode']}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("   Make sure server is running: uvicorn HoloLoom.apps.server.agentic_api:app --port 8000")
            return

        print()

        # Test 2: Get projects
        print("2. GET /api/monitor/projects")
        print("-" * 70)
        try:
            response = await client.get(f"{base_url}/api/monitor/projects")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data['count']} projects")
                for project in data['projects']:
                    print(f"   - {project}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")

        print()

        # Test 3: Get metrics
        print("3. GET /api/monitor/metrics")
        print("-" * 70)
        try:
            response = await client.get(f"{base_url}/api/monitor/metrics")
            if response.status_code == 200:
                metrics = response.json()
                print("✅ Success:")
                print(f"   Total started: {metrics['total_agents_started']}")
                print(f"   Total completed: {metrics['total_agents_completed']}")
                print(f"   Total failed: {metrics['total_agents_failed']}")
                print(f"   Active agents: {metrics['active_agents']}")
                print(f"   Avg latency: {metrics['avg_latency_ms']:.1f}ms")
                print(f"   Success rate: {metrics['success_rate']:.1%}")
                print(f"   WebSocket connections: {metrics['ws_connections']}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")

        print()

        # Test 4: Create a test agent
        print("4. POST /query (create test agent)")
        print("-" * 70)
        try:
            query_data = {
                "text": "What is Thompson Sampling?",
                "mode": "verify"
            }
            response = await client.post(f"{base_url}/query", json=query_data)
            if response.status_code == 200:
                result = response.json()
                agent_id = result.get('agent_id', 'unknown')
                print(f"✅ Agent created: {agent_id}")
                print(f"   Response: {result['response'][:100]}...")
                print(f"   Confidence: {result['confidence']:.2f}")
                print()

                # Test 5: Get specific session details
                print("5. GET /api/monitor/sessions/{agent_id}")
                print("-" * 70)
                response = await client.get(f"{base_url}/api/monitor/sessions/{agent_id}")
                if response.status_code == 200:
                    session = response.json()
                    print("✅ Success:")
                    print(f"   Agent: {session['agent_id']}")
                    print(f"   Project: {session['project']}")
                    print(f"   Query: {session['query'][:60]}...")
                    print(f"   Status: {session['status']}")
                    print(f"   Mode: {session['mode']}")
                    print(f"   Duration: {session.get('total_duration_ms', 'N/A')}ms")

                    # Check if tree exists
                    if session.get('tree'):
                        tree = session['tree']
                        print(f"   Tree root: {tree.get('node_id', 'N/A')}")
                        print(f"   Tree children: {len(tree.get('children', []))}")
                elif response.status_code == 404:
                    print(f"⚠️  Session {agent_id} not found (may have been cleaned up)")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")

                print()

                # Test 6: Get project agents
                print("6. GET /api/monitor/projects/mythRL")
                print("-" * 70)
                response = await client.get(f"{base_url}/api/monitor/projects/mythRL")
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Success: {data['count']} agents in mythRL")
                    for agent in data['agents'][:3]:  # Show first 3
                        print(f"   - {agent['agent_id']}: {agent['query'][:40]}...")
                        print(f"     Status: {agent['status']}")
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")

            else:
                print(f"❌ Query failed {response.status_code}: {response.text}")
        except Exception as e:
            print(f"❌ Request failed: {e}")

        print()
        print("=" * 70)
        print("REST API Test Complete")
        print("=" * 70)


async def test_with_curl():
    """Print curl commands for manual testing"""
    print()
    print("Manual Testing with curl:")
    print("=" * 70)
    print()
    print("# Get all sessions")
    print("curl http://localhost:8000/api/monitor/sessions")
    print()
    print("# Get specific session")
    print("curl http://localhost:8000/api/monitor/sessions/{agent_id}")
    print()
    print("# Get projects")
    print("curl http://localhost:8000/api/monitor/projects")
    print()
    print("# Get project agents")
    print("curl http://localhost:8000/api/monitor/projects/mythRL")
    print()
    print("# Get metrics")
    print("curl http://localhost:8000/api/monitor/metrics")
    print()


async def main():
    """Run REST API tests"""
    await test_rest_endpoints()
    await test_with_curl()


if __name__ == "__main__":
    asyncio.run(main())

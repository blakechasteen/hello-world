#!/usr/bin/env python3
"""Test HoloLoom weaving endpoint."""

import requests
import json
import time

# Test health endpoint
print("Testing health endpoint...")
response = requests.get("http://localhost:8765/health")
print(f"Health: {json.dumps(response.json(), indent=2)}\n")

# Test HoloLoom weaving
print("Testing HoloLoom weaving endpoint...")
weaving_request = {
    "query": "What is Thompson Sampling?",
    "pattern": "fast",
    "complexity": "auto"
}

response = requests.post(
    "http://localhost:8765/execute/hololoom",
    json=weaving_request
)

result = response.json()
print(f"Execution queued: {json.dumps(result, indent=2)}\n")

execution_id = result["execution_id"]

# Poll for status
print(f"Polling status for execution {execution_id}...")
for i in range(20):
    time.sleep(1)
    status_response = requests.get(f"http://localhost:8765/execute/status/{execution_id}")
    status = status_response.json()

    print(f"[{i+1}] Status: {status['status']}, Progress: {status['progress']:.1%}, Step: {status['current_step']}")

    if status["status"] in ["completed", "failed"]:
        print(f"\nFinal result:")
        print(json.dumps(status, indent=2))
        break

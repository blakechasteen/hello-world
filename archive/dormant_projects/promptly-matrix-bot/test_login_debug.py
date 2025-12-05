#!/usr/bin/env python3
"""Debug Matrix login to see exact request/response"""

import asyncio
import getpass
from nio import AsyncClient, LoginResponse, LoginError

async def test_login():
    print("=== Matrix Login Debug ===\n")

    # Get credentials
    username = input("Enter username (just 'promptlybot' without @ or :matrix.org): ")
    password = getpass.getpass("Enter password: ")

    print(f"\nAttempting login...")
    print(f"Homeserver: https://matrix.org")
    print(f"Username: {username}")
    print(f"Password: {'*' * len(password)}")

    # Create client
    client = AsyncClient("https://matrix.org", username)

    # Attempt login
    response = await client.login(password, device_name="Debug Test")

    print(f"\n=== Response ===")
    print(f"Type: {type(response).__name__}")

    if isinstance(response, LoginResponse):
        print("✅ SUCCESS!")
        print(f"User ID: {response.user_id}")
        print(f"Device ID: {response.device_id}")
        print(f"Access Token: {response.access_token[:30]}...")
    elif isinstance(response, LoginError):
        print("❌ FAILED!")
        print(f"Status Code: {response.status_code}")
        print(f"Message: {response.message}")
        if hasattr(response, 'retry_after_ms'):
            print(f"Retry After: {response.retry_after_ms}ms")

    await client.close()

if __name__ == "__main__":
    asyncio.run(test_login())
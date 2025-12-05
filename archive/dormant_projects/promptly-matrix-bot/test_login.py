#!/usr/bin/env python3
"""Test Matrix login with different formats"""

import asyncio
import getpass
from nio import AsyncClient, LoginResponse

async def test_login():
    print("Testing Matrix login formats...")

    password = getpass.getpass("Enter bot password: ")

    # Test 1: localpart only
    print("\n=== Test 1: AsyncClient with localpart 'promptlybot' ===")
    client1 = AsyncClient("https://matrix.org", "promptlybot")
    response1 = await client1.login(password, device_name="Test Bot 1")
    print(f"Result: {type(response1).__name__}")
    print(f"Response: {response1}")
    await client1.close()

    if isinstance(response1, LoginResponse):
        print("✅ SUCCESS! This format works.")
        print(f"User ID: {response1.user_id}")
        print(f"Access Token: {response1.access_token[:20]}...")
        return

    # Test 2: full user_id
    print("\n=== Test 2: AsyncClient with full user_id '@promptlybot:matrix.org' ===")
    client2 = AsyncClient("https://matrix.org", "@promptlybot:matrix.org")
    response2 = await client2.login(password, device_name="Test Bot 2")
    print(f"Result: {type(response2).__name__}")
    print(f"Response: {response2}")
    await client2.close()

    if isinstance(response2, LoginResponse):
        print("✅ SUCCESS! This format works.")
        print(f"User ID: {response2.user_id}")
        print(f"Access Token: {response2.access_token[:20]}...")
        return

if __name__ == "__main__":
    asyncio.run(test_login())

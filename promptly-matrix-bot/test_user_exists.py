#!/usr/bin/env python3
"""Check if a Matrix user exists by trying to fetch profile"""

import asyncio
import aiohttp

async def check_user_exists(user_id):
    """Check if a Matrix user exists by fetching their profile"""

    # Extract homeserver from user_id
    if ':' in user_id:
        homeserver = user_id.split(':')[1]
    else:
        homeserver = 'matrix.org'

    # Construct profile URL
    url = f"https://{homeserver}/_matrix/client/r0/profile/{user_id}"

    print(f"Checking if user exists: {user_id}")
    print(f"URL: {url}\n")

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            status = response.status
            data = await response.json()

            print(f"Status Code: {status}")
            print(f"Response: {data}\n")

            if status == 200:
                print("✅ User EXISTS!")
                if 'displayname' in data:
                    print(f"Display Name: {data['displayname']}")
                return True
            elif status == 404:
                print("❌ User DOES NOT EXIST on this homeserver")
                print("Double-check the username or homeserver!")
                return False
            else:
                print(f"⚠️ Unexpected response: {status}")
                return None

if __name__ == "__main__":
    # Test user IDs
    test_users = [
        "@promptlybot:matrix.org",
    ]

    for user_id in test_users:
        asyncio.run(check_user_exists(user_id))
        print("=" * 50)
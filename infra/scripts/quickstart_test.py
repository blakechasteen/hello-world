"""
Quick test to verify HoloLoom is working
Run: PYTHONPATH=. python quickstart_test.py
"""
import asyncio
from HoloLoom import HoloLoom

async def test():
    print("🚀 Testing HoloLoom...")

    async with HoloLoom() as loom:
        # Test 1: Can we form memories?
        print("\n✓ Step 1: Forming memory...")
        await loom.experience("Thompson Sampling balances exploration and exploitation")

        # Test 2: Can we recall?
        print("✓ Step 2: Recalling memory...")
        memories = await loom.recall("What is Thompson Sampling?")

        if memories:
            print(f"✓ Step 3: Found {len(memories)} memories!")
            print(f"  Content preview: {memories[0].content[:100]}...")
            print("\n✅ SUCCESS! HoloLoom is operational.\n")
            return True
        else:
            print("❌ FAILED: No memories retrieved")
            return False

if __name__ == "__main__":
    success = asyncio.run(test())
    if success:
        print("🎉 You're ready to use HoloLoom with your own data!")
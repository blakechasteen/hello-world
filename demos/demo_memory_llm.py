"""
Demo: MemoryLLM - An LLM that remembers.

Usage:
    python demos/demo_memory_llm.py
"""

from HoloLoom.memory_llm import MemoryLLM

def main():
    print("=" * 50)
    print("MemoryLLM Demo")
    print("=" * 50)

    # Create memory-only instance (no LLM needed for demo)
    llm = MemoryLLM()
    print(f"\nCreated: {llm}")

    # Store some knowledge
    print("\n--- Storing memories ---")
    llm.remember("Thompson Sampling is a Bayesian approach to the multi-armed bandit problem")
    llm.remember("It balances exploration and exploitation using probability distributions")
    llm.remember("UCB (Upper Confidence Bound) is another bandit algorithm")
    llm.remember("Python is a programming language with simple syntax")
    llm.remember("Machine learning uses data to make predictions")
    print(f"Stored 5 memories")

    # Recall relevant memories
    print("\n--- Recalling 'bandit algorithms' ---")
    results = llm.recall("bandit algorithms")
    for mem in results:
        print(f"  [{mem.score:.2f}] {mem.text[:60]}...")

    print("\n--- Recalling 'programming' ---")
    results = llm.recall("programming")
    for mem in results:
        print(f"  [{mem.score:.2f}] {mem.text[:60]}...")

    # Show that irrelevant queries return less
    print("\n--- Recalling 'weather' (irrelevant) ---")
    results = llm.recall("weather")
    print(f"  Found {len(results)} relevant memories")

    # Chat mode (without actual LLM)
    print("\n--- Chat mode (memory-only) ---")
    response = llm("What do you know about Thompson Sampling?")
    print(f"Response: {response}")

    print(f"\n{llm}")
    print("\nDone!")


if __name__ == "__main__":
    main()

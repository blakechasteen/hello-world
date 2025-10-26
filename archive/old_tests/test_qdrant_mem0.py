"""
Quick test of mem0 with Qdrant
================================

Run: python test_qdrant_mem0.py

Prerequisites:
- Qdrant running: docker ps | findstr qdrant
- OpenAI API key: set OPENAI_API_KEY=sk-...
"""

from mem0 import Memory

# Configure mem0 to use Qdrant
config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "collection_name": "my_memories"
        }
    }
}

print("Initializing mem0 with Qdrant...")
m = Memory.from_config(config)
print("OK - Connected to Qdrant\n")

# Test it
print("Storing a memory...")
result = m.add([
    {"role": "user", "content": "I'm Blake and I keep bees. I have three hives named Jodi, Aurora, and Luna."}
], user_id="blake")

print(f"Stored! Extracted {len(result.get('results', []))} memories\n")

# Search
print("Searching...")
results = m.search("What are Blake's hives?", user_id="blake")

print("\nResults:")
for mem in results.get('results', []):
    print(f"  - {mem.get('memory', 'N/A')}")

print("\nSuccess! Data is now in Qdrant at http://localhost:6333/dashboard")

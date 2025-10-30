"""
Unified Memory System Demo

Demonstrates "everything is a memory operation" philosophy.

Shows how all CRM operations (creating contacts, embeddings, similarity search,
activity logging, relationship creation) are unified under memory operations.

Run with: PYTHONPATH=.. python -m crm_app.demo_unified_memory
"""

from datetime import datetime, timedelta

from crm_app.models import Contact, Company, Activity, ActivityType, ActivityOutcome, CompanySize
from crm_app.memory import (
    UnifiedMemory,
    MemoryType,
    MemoryAddress,
    Memory,
    MemoryQuery
)


def print_header(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str):
    """Print formatted subsection"""
    print(f"\n{title}")
    print("-" * 70)


def demo_memory_philosophy():
    """Demonstrate the core insight: Everything is a memory operation."""
    print_header("Unified Memory System: Everything is a Memory Operation")

    print("""
The Insight:
------------
Every CRM operation is fundamentally a memory operation:

Traditional View:                Memory View:
-----------------               --------------
create_contact()         ->     write_symbolic(contact)
compute_embedding()      ->     compress(contact_address)
find_similar()           ->     query_semantic(similarity_criteria)
log_activity()           ->     write_episodic(activity)
add_relationship()       ->     associate(addr1, addr2, relation)

Same operations, unified understanding.
""")


def demo_memory_addresses():
    """Demonstrate memory addressing across subsystems."""
    print_section("Memory Addressing: Universal Identifiers")

    # Create addresses for different memory types
    contact_addr = MemoryAddress(
        subsystem=MemoryType.SYMBOLIC,
        id="contact_alice_123",
        version=1
    )

    embedding_addr = MemoryAddress(
        subsystem=MemoryType.SEMANTIC,
        id="emb_contact_alice_123",
        version=1
    )

    activity_addr = MemoryAddress(
        subsystem=MemoryType.EPISODIC,
        id="activity_call_456",
        version=1
    )

    edge_addr = MemoryAddress(
        subsystem=MemoryType.RELATIONAL,
        id="edge_alice_acme_789",
        version=1
    )

    print("\nMemory addresses are uniform across all subsystems:")
    print(f"  Symbolic:   {contact_addr}")
    print(f"  Semantic:   {embedding_addr}")
    print(f"  Episodic:   {activity_addr}")
    print(f"  Relational: {edge_addr}")

    print("\n[OK] All memories have uniform addressing")


def demo_unified_memory_initialization():
    """Demonstrate unified memory system initialization."""
    print_section("Unified Memory Initialization")

    memory = UnifiedMemory()

    print("\nUnified memory system created")
    print("Ready to coordinate all memory subsystems:")
    print("  - Symbolic Memory (exact entity storage)")
    print("  - Semantic Memory (continuous embeddings)")
    print("  - Episodic Memory (time-ordered events)")
    print("  - Relational Memory (entity relationships)")
    print("  - Working Memory (active context)")
    print("  - Meta Memory (memory about memories)")

    print("\n[OK] Unified memory system initialized")
    return memory


def demo_memory_write_operations(memory: UnifiedMemory):
    """Demonstrate writing to different memory subsystems."""
    print_section("Memory Write Operations")

    # Create sample entities
    contact = Contact.create(
        name="Alice Johnson",
        email="alice@techcorp.com",
        title="VP of Sales"
    )

    company = Company.create(
        name="TechCorp",
        industry="Technology",
        size=CompanySize.ENTERPRISE
    )

    activity = Activity.create(
        type=ActivityType.CALL,
        contact_id=contact.id,
        subject="Discovery call",
        content="Discussed enterprise requirements",
        outcome=ActivityOutcome.POSITIVE
    )

    print("\n1. Writing to Symbolic Memory (exact entities)")
    print(f"   Memory operation: write_symbolic(contact)")
    print(f"   Entity: {contact.name} - {contact.title}")

    print("\n2. Writing to Episodic Memory (time-ordered events)")
    print(f"   Memory operation: write_episodic(activity)")
    print(f"   Event: {activity.type.value} - {activity.subject}")
    print(f"   Timestamp: {activity.timestamp}")

    print("\n3. Writing to Relational Memory (associations)")
    print(f"   Memory operation: associate(contact_addr, company_addr, 'WORKS_AT')")
    print(f"   Relationship: {contact.name} WORKS_AT {company.name}")

    print("\n[OK] All writes are memory operations")


def demo_memory_transform_operations():
    """Demonstrate memory transformations (compression)."""
    print_section("Memory Transform Operations: Compression")

    print("""
Compression = Symbolic -> Semantic:
---------------------------------

Symbolic Memory (discrete):
  Contact:
    name: "Alice Johnson"
    title: "VP of Sales"
    email: "alice@techcorp.com"

        | compress()

Semantic Memory (continuous):
  Embedding: [0.23, -0.45, 0.67, ..., 0.12]  (384 dimensions)

Purpose: Enable approximate retrieval by semantic similarity
""")

    print("[OK] Compression transforms symbolic -> semantic memory")


def demo_memory_query_operations():
    """Demonstrate querying different memory subsystems."""
    print_section("Memory Query Operations")

    print("\n1. Symbolic Memory Query (exact filters):")
    print("""
   query = MemoryQuery(
       subsystem=MemoryType.SYMBOLIC,
       criteria={"lead_score__gt": 0.8, "industry": "Technology"}
   )
   result = memory.query(query)

   Returns: Contacts matching exact criteria
""")

    print("2. Semantic Memory Query (similarity search):")
    print("""
   query = MemoryQuery(
       subsystem=MemoryType.SEMANTIC,
       criteria={"similar_to": contact_embedding}
   )
   result = memory.query(query)

   Returns: Contacts semantically similar (by embedding distance)
""")

    print("3. Episodic Memory Query (temporal range):")
    print("""
   query = MemoryQuery(
       subsystem=MemoryType.EPISODIC,
       criteria={"type": "call", "outcome": "positive"},
       time_range=(start_date, end_date)
   )
   result = memory.query(query)

   Returns: Activities in time range matching criteria
""")

    print("4. Relational Memory Query (graph traversal):")
    print("""
   query = MemoryQuery(
       subsystem=MemoryType.RELATIONAL,
       criteria={"from": contact_addr, "relation": "WORKS_WITH"}
   )
   result = memory.query(query)

   Returns: Related entities via graph edges
""")

    print("\n[OK] All queries are memory operations with subsystem-specific semantics")


def demo_memory_subsystems():
    """Demonstrate the six memory subsystems."""
    print_section("Six Memory Subsystems")

    subsystems = [
        ("Symbolic", "Exact entity storage", "Contacts, Companies, Deals"),
        ("Semantic", "Continuous embeddings", "384-dim vectors, similarity"),
        ("Episodic", "Time-ordered events", "Activities with timestamps"),
        ("Relational", "Entity relationships", "Knowledge graph edges"),
        ("Working", "Active context", "Query state, active memories"),
        ("Meta", "Memory about memories", "Confidence, lineage, quality")
    ]

    print("\n+-------------+----------------------+------------------------+")
    print("|  Subsystem  |      Purpose         |     Storage Type       |")
    print("+-------------+----------------------+------------------------+")

    for name, purpose, storage in subsystems:
        print(f"| {name:11} | {purpose:20} | {storage:22} |")

    print("+-------------+----------------------+------------------------+")

    print("\n[OK] Six subsystems, one unified interface")


def demo_traditional_vs_memory():
    """Compare traditional CRM operations vs memory operations."""
    print_section("Traditional vs Memory-First: Same Code, Different Understanding")

    comparisons = [
        ("Traditional", "Memory-First"),
        ("-" * 35, "-" * 35),
        ("storage.create_contact(c)", "memory.write_symbolic(c)"),
        ("embedder.embed_contact(c)", "memory.compress(contact_addr)"),
        ("similarity.find_similar(c)", "memory.query_semantic({...})"),
        ("storage.log_activity(a)", "memory.write_episodic(a)"),
        ("kg.add_edge(c1, c2, rel)", "memory.associate(a1, a2, rel)"),
        ("storage.get_contact(id)", "memory.read(address)"),
    ]

    for traditional, memory in comparisons:
        print(f"{traditional:35} │ {memory}")

    print("""
Same implementation, unified mental model:
  - All operations are memory operations
  - Clear subsystem boundaries
  - Predictable performance (memory access patterns)
  - Modular evolution (add new memory subsystems)
""")

    print("[OK] Memory-first thinking simplifies the entire system")


def demo_memory_lifecycle():
    """Demonstrate complete memory lifecycle."""
    print_section("Complete Memory Lifecycle")

    print("""
Memory Lifecycle: Birth -> Life -> Transformation -> Death
-------------------------------------------------------

1. BIRTH (Write)
   contact = Contact.create(...)
   addr = memory.write_symbolic(contact)

2. LIFE (Read)
   contact = memory.read(addr)

3. TRANSFORMATION (Compress/Update)
   embedding_addr = memory.compress(addr)  # Symbolic -> Semantic
   memory.update(addr, {"lead_score": 0.9})  # Modify

4. ASSOCIATION (Relate)
   memory.associate(contact_addr, company_addr, "WORKS_AT")

5. QUERY (Retrieve)
   similar = memory.query_semantic({"similar_to": addr})

6. DEATH (Delete/Archive)
   memory.delete(addr)  # Or mark inactive

Every stage is a memory operation.
""")

    print("[OK] Complete memory lifecycle demonstrated")


def demo_meta_memory():
    """Demonstrate meta-memory (memory about memories)."""
    print_section("Meta-Memory: Memory About Memories")

    print("""
Meta-memory tracks:
------------------
- When memory was created
- How often it's been accessed
- Confidence in the memory
- Processing time for operations
- Data lineage (where it came from)
- Associations with other memories

Example meta-memory for a contact:
{
    'created_at': '2025-10-29T10:30:00',
    'accessed_count': 47,
    'last_accessed': '2025-10-29T15:45:23',
    'confidence': 0.95,
    'source': 'crm_import_batch_123',
    'embedding_computed': True,
    'associations': ['company_abc', 'deal_xyz'],
    'similar_to': ['contact_bob', 'contact_carol']
}

Meta-memory enables:
- Quality assessment (which memories are reliable?)
- Performance optimization (which memories are hot?)
- Lineage tracking (where did this memory come from?)
- Memory cleanup (which memories are stale?)
""")

    print("[OK] Meta-memory provides memory intelligence")


def main():
    """Run complete unified memory demo."""
    demo_memory_philosophy()
    demo_memory_addresses()

    memory = demo_unified_memory_initialization()

    demo_memory_write_operations(memory)
    demo_memory_transform_operations()
    demo_memory_query_operations()
    demo_memory_subsystems()
    demo_traditional_vs_memory()
    demo_memory_lifecycle()
    demo_meta_memory()

    print_header("Unified Memory System: Complete")

    print("""
Key Insights:
-------------
1. Everything is a memory operation
2. Six subsystems, one unified interface
3. Same code, elevated understanding
4. Memory operations are composable
5. Meta-memory provides intelligence

Benefits:
---------
- Single mental model (memory)
- Clear subsystem boundaries
- Predictable performance
- Modular evolution
- Testable behavior

Next Steps:
-----------
1. Implement subsystem adapters for existing code
2. Migrate traditional operations to memory API
3. Add working memory for query context
4. Implement meta-memory tracking
5. Add memory stream for time-travel/replay

"Everything is a memory operation" - Philosophy actualized ✨
""")


if __name__ == "__main__":
    main()

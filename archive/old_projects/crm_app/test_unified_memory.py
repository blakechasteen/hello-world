"""
Unified Memory System Tests
============================

Comprehensive test suite for unified memory architecture:
- Memory addressing across subsystems
- UnifiedMemory manager operations
- Memory transformations (compress, associate)
- Cross-subsystem queries
- Meta-memory tracking

"Everything is a memory operation" - validated.

Run with: PYTHONPATH=.. python -m crm_app.test_unified_memory
"""

import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from crm_app.models import Contact, Company, Activity, ActivityType, ActivityOutcome, CompanySize
from crm_app.memory import (
    UnifiedMemory,
    MemoryProtocol,
    MemoryType,
    MemoryAddress,
    Memory,
    MemoryQuery,
    MemoryResult
)


# ============================================================================
# Mock Memory Subsystem Implementations
# ============================================================================

class MockSymbolicMemory:
    """Mock symbolic memory subsystem for testing."""

    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.write_count = 0
        self.read_count = 0
        self.query_count = 0

    def write(self, memory: Memory) -> MemoryAddress:
        """Store symbolic memory."""
        self.memories[memory.address.id] = memory
        self.write_count += 1
        return memory.address

    def read(self, address: MemoryAddress) -> Optional[Memory]:
        """Read symbolic memory."""
        self.read_count += 1
        return self.memories.get(address.id)

    def query(self, query: MemoryQuery) -> MemoryResult:
        """Query symbolic memory with filters."""
        self.query_count += 1
        criteria = query.criteria

        # Simple filter implementation
        results = []
        scores = []

        for mem in self.memories.values():
            # Check filters
            match = True
            if "entity_type" in criteria:
                entity_type = mem.metadata.get("entity_type")
                if entity_type != criteria["entity_type"]:
                    match = False

            if match:
                results.append(mem)
                scores.append(1.0)  # Exact match

        return MemoryResult(
            memories=results[:query.limit],
            scores=scores[:query.limit],
            metadata={"subsystem": "symbolic"},
            total_found=len(results)
        )

    def update(self, address: MemoryAddress, updates: Dict[str, Any]) -> Memory:
        """Update symbolic memory."""
        if address.id in self.memories:
            memory = self.memories[address.id]
            # Update content if it's a dict-like object
            if hasattr(memory.content, "__dict__"):
                for key, value in updates.items():
                    setattr(memory.content, key, value)
            memory.metadata.update(updates)
            return memory
        return None

    def delete(self, address: MemoryAddress) -> bool:
        """Delete symbolic memory."""
        if address.id in self.memories:
            del self.memories[address.id]
            return True
        return False

    def associate(self, addr1: MemoryAddress, addr2: MemoryAddress, relation: str) -> MemoryAddress:
        """Create association (handled by relational subsystem)."""
        raise NotImplementedError("Symbolic memory doesn't handle associations")

    def compress(self, address: MemoryAddress) -> Memory:
        """Compress to semantic (handled by semantic subsystem)."""
        raise NotImplementedError("Symbolic memory doesn't handle compression")

    def stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "total_memories": len(self.memories),
            "writes": self.write_count,
            "reads": self.read_count,
            "queries": self.query_count
        }


class MockSemanticMemory:
    """Mock semantic memory subsystem for testing."""

    def __init__(self):
        self.memories: Dict[str, Memory] = {}
        self.compress_count = 0

    def write(self, memory: Memory) -> MemoryAddress:
        """Store semantic memory (embedding)."""
        self.memories[memory.address.id] = memory
        return memory.address

    def read(self, address: MemoryAddress) -> Optional[Memory]:
        """Read semantic memory."""
        return self.memories.get(address.id)

    def query(self, query: MemoryQuery) -> MemoryResult:
        """Query semantic memory (similarity search)."""
        criteria = query.criteria

        # Mock similarity search
        results = []
        scores = []

        if "similar_to" in criteria:
            target_addr = criteria["similar_to"]
            target_mem = self.memories.get(target_addr.id)

            if target_mem and isinstance(target_mem.content, np.ndarray):
                target_emb = target_mem.content

                # Compare with all embeddings
                for mem in self.memories.values():
                    if isinstance(mem.content, np.ndarray):
                        # Cosine similarity
                        sim = np.dot(target_emb, mem.content) / (
                            np.linalg.norm(target_emb) * np.linalg.norm(mem.content)
                        )
                        if sim >= query.min_confidence:
                            results.append(mem)
                            scores.append(float(sim))

        # Sort by score descending
        sorted_pairs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        if sorted_pairs:
            results, scores = zip(*sorted_pairs)
            results = list(results)[:query.limit]
            scores = list(scores)[:query.limit]
        else:
            results, scores = [], []

        return MemoryResult(
            memories=results,
            scores=scores,
            metadata={"subsystem": "semantic"},
            total_found=len(results)
        )

    def update(self, address: MemoryAddress, updates: Dict[str, Any]) -> Memory:
        """Update semantic memory."""
        if address.id in self.memories:
            memory = self.memories[address.id]
            memory.metadata.update(updates)
            return memory
        return None

    def delete(self, address: MemoryAddress) -> bool:
        """Delete semantic memory."""
        if address.id in self.memories:
            del self.memories[address.id]
            return True
        return False

    def associate(self, addr1: MemoryAddress, addr2: MemoryAddress, relation: str) -> MemoryAddress:
        """Create association (handled by relational subsystem)."""
        raise NotImplementedError("Semantic memory doesn't handle associations")

    def compress(self, address: MemoryAddress) -> Memory:
        """Compress symbolic memory to embedding."""
        self.compress_count += 1

        # Create mock embedding
        embedding = np.random.randn(384).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Create semantic memory address
        semantic_addr = MemoryAddress(
            subsystem=MemoryType.SEMANTIC,
            id=f"emb_{address.id}",
            version=1
        )

        # Create and store semantic memory
        semantic_mem = Memory(
            address=semantic_addr,
            content=embedding,
            metadata={
                "source_address": str(address),
                "compressed_at": datetime.now(timezone.utc).isoformat()
            }
        )

        self.memories[semantic_addr.id] = semantic_mem
        return semantic_mem

    def stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "total_embeddings": len(self.memories),
            "compressions": self.compress_count
        }


class MockEpisodicMemory:
    """Mock episodic memory subsystem for testing."""

    def __init__(self):
        self.memories: Dict[str, Memory] = {}

    def write(self, memory: Memory) -> MemoryAddress:
        """Store episodic memory (activity)."""
        self.memories[memory.address.id] = memory
        return memory.address

    def read(self, address: MemoryAddress) -> Optional[Memory]:
        """Read episodic memory."""
        return self.memories.get(address.id)

    def query(self, query: MemoryQuery) -> MemoryResult:
        """Query episodic memory (temporal range)."""
        criteria = query.criteria
        time_range = query.time_range

        results = []
        scores = []

        for mem in self.memories.values():
            # Filter by criteria
            match = True
            if "type" in criteria:
                activity = mem.content
                if hasattr(activity, "type") and activity.type.value != criteria["type"]:
                    match = False

            # Filter by time range
            if time_range and match:
                activity = mem.content
                if hasattr(activity, "timestamp"):
                    if not (time_range[0] <= activity.timestamp <= time_range[1]):
                        match = False

            if match:
                results.append(mem)
                scores.append(1.0)

        return MemoryResult(
            memories=results[:query.limit],
            scores=scores[:query.limit],
            metadata={"subsystem": "episodic"},
            total_found=len(results)
        )

    def update(self, address: MemoryAddress, updates: Dict[str, Any]) -> Memory:
        """Update episodic memory."""
        if address.id in self.memories:
            memory = self.memories[address.id]
            memory.metadata.update(updates)
            return memory
        return None

    def delete(self, address: MemoryAddress) -> bool:
        """Delete episodic memory."""
        if address.id in self.memories:
            del self.memories[address.id]
            return True
        return False

    def associate(self, addr1: MemoryAddress, addr2: MemoryAddress, relation: str) -> MemoryAddress:
        """Create association (handled by relational subsystem)."""
        raise NotImplementedError("Episodic memory doesn't handle associations")

    def compress(self, address: MemoryAddress) -> Memory:
        """Compress (not used for episodic)."""
        raise NotImplementedError("Episodic memory doesn't handle compression")

    def stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "total_activities": len(self.memories)
        }


class MockRelationalMemory:
    """Mock relational memory subsystem for testing."""

    def __init__(self):
        self.edges: Dict[str, Memory] = {}
        self.association_count = 0

    def write(self, memory: Memory) -> MemoryAddress:
        """Store relational memory (edge)."""
        self.edges[memory.address.id] = memory
        return memory.address

    def read(self, address: MemoryAddress) -> Optional[Memory]:
        """Read relational memory."""
        return self.edges.get(address.id)

    def query(self, query: MemoryQuery) -> MemoryResult:
        """Query relational memory (graph traversal)."""
        criteria = query.criteria

        results = []
        scores = []

        # Simple filtering by relation type
        for mem in self.edges.values():
            match = True
            if "relation" in criteria:
                if mem.metadata.get("relation") != criteria["relation"]:
                    match = False

            if "from" in criteria:
                if mem.metadata.get("from") != str(criteria["from"]):
                    match = False

            if match:
                results.append(mem)
                scores.append(1.0)

        return MemoryResult(
            memories=results[:query.limit],
            scores=scores[:query.limit],
            metadata={"subsystem": "relational"},
            total_found=len(results)
        )

    def update(self, address: MemoryAddress, updates: Dict[str, Any]) -> Memory:
        """Update relational memory."""
        if address.id in self.edges:
            memory = self.edges[address.id]
            memory.metadata.update(updates)
            return memory
        return None

    def delete(self, address: MemoryAddress) -> bool:
        """Delete relational memory."""
        if address.id in self.edges:
            del self.edges[address.id]
            return True
        return False

    def associate(self, addr1: MemoryAddress, addr2: MemoryAddress, relation: str) -> MemoryAddress:
        """Create association between memories."""
        self.association_count += 1

        # Create edge address
        edge_addr = MemoryAddress(
            subsystem=MemoryType.RELATIONAL,
            id=f"edge_{addr1.id}_{addr2.id}_{relation}",
            version=1
        )

        # Create edge memory
        edge_mem = Memory(
            address=edge_addr,
            content={"from": str(addr1), "to": str(addr2), "relation": relation},
            metadata={
                "from": str(addr1),
                "to": str(addr2),
                "relation": relation,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
        )

        self.edges[edge_addr.id] = edge_mem
        return edge_addr

    def compress(self, address: MemoryAddress) -> Memory:
        """Compress (not used for relational)."""
        raise NotImplementedError("Relational memory doesn't handle compression")

    def stats(self) -> Dict[str, Any]:
        """Return statistics."""
        return {
            "total_edges": len(self.edges),
            "associations": self.association_count
        }


# ============================================================================
# Test Functions
# ============================================================================

def test_memory_addressing():
    """Test universal memory addressing."""
    print("\n" + "="*70)
    print("TEST 1: Memory Addressing")
    print("="*70)

    # Create addresses for different subsystems
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
        id="edge_alice_acme_works_at",
        version=1
    )

    print("\nMemory addresses created:")
    print(f"  Symbolic:   {contact_addr}")
    print(f"  Semantic:   {embedding_addr}")
    print(f"  Episodic:   {activity_addr}")
    print(f"  Relational: {edge_addr}")

    # Verify format
    assert str(contact_addr) == "symbolic://contact_alice_123@v1"
    assert str(embedding_addr) == "semantic://emb_contact_alice_123@v1"
    assert str(activity_addr) == "episodic://activity_call_456@v1"
    assert str(edge_addr) == "relational://edge_alice_acme_works_at@v1"

    print("\n[OK] Universal addressing working")


def test_unified_memory_initialization():
    """Test UnifiedMemory initialization and subsystem registration."""
    print("\n" + "="*70)
    print("TEST 2: UnifiedMemory Initialization")
    print("="*70)

    # Create unified memory
    memory = UnifiedMemory()

    # Register mock subsystems
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())
    memory.register_subsystem(MemoryType.SEMANTIC, MockSemanticMemory())
    memory.register_subsystem(MemoryType.EPISODIC, MockEpisodicMemory())
    memory.register_subsystem(MemoryType.RELATIONAL, MockRelationalMemory())

    print("\nUnified memory initialized with subsystems:")
    print("  - Symbolic Memory (exact entity storage)")
    print("  - Semantic Memory (continuous embeddings)")
    print("  - Episodic Memory (time-ordered events)")
    print("  - Relational Memory (entity relationships)")

    # Check stats
    stats = memory.stats()
    assert 'meta' in stats
    assert 'subsystems' in stats
    assert len(stats['subsystems']) == 4

    print(f"\n[OK] UnifiedMemory initialized: {len(stats['subsystems'])} subsystems")
    return memory


def test_symbolic_write_read():
    """Test writing and reading symbolic memories."""
    print("\n" + "="*70)
    print("TEST 3: Symbolic Write/Read Operations")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())

    # Create contact
    contact = Contact.create(
        name="Alice Johnson",
        email="alice@techcorp.com",
        title="VP of Sales"
    )

    # Write to symbolic memory
    contact_addr = memory.write_symbolic(contact, metadata={"entity_type": "contact"})

    print(f"\nWritten contact to symbolic memory:")
    print(f"  Address: {contact_addr}")
    print(f"  Name: {contact.name}")
    print(f"  Title: {contact.title}")

    # Read back
    contact_mem = memory.read(contact_addr)
    assert contact_mem is not None
    assert contact_mem.content.name == "Alice Johnson"
    assert contact_mem.content.email == "alice@techcorp.com"

    print(f"\nRead contact from symbolic memory:")
    print(f"  Name: {contact_mem.content.name}")
    print(f"  Email: {contact_mem.content.email}")

    # Check meta-memory
    stats = memory.stats()
    assert stats['meta']['total_writes'] == 1
    assert stats['meta']['total_reads'] == 1

    print(f"\n[OK] Symbolic write/read working (writes: {stats['meta']['total_writes']}, reads: {stats['meta']['total_reads']})")
    return memory


def test_semantic_compression():
    """Test compressing symbolic memory to semantic representation."""
    print("\n" + "="*70)
    print("TEST 4: Memory Compression (Symbolic -> Semantic)")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())
    memory.register_subsystem(MemoryType.SEMANTIC, MockSemanticMemory())

    # Write contact to symbolic memory
    contact = Contact.create(
        name="Bob Smith",
        email="bob@example.com",
        title="CTO"
    )

    contact_addr = memory.write_symbolic(contact)
    print(f"\nSymbolic memory: {contact_addr}")
    print(f"  Entity: {contact.name} - {contact.title}")

    # Compress to semantic memory
    embedding_mem = memory.compress(contact_addr)

    print(f"\nCompressed to semantic memory:")
    print(f"  Address: {embedding_mem.address}")
    print(f"  Embedding shape: {embedding_mem.content.shape}")
    print(f"  Source: {embedding_mem.metadata['source_address']}")

    assert embedding_mem.address.subsystem == MemoryType.SEMANTIC
    assert isinstance(embedding_mem.content, np.ndarray)
    assert embedding_mem.content.shape == (384,)

    # Verify semantic subsystem has it
    semantic_read = memory.read(embedding_mem.address)
    assert semantic_read is not None
    assert np.array_equal(semantic_read.content, embedding_mem.content)

    print("\n[OK] Memory compression working (symbolic -> semantic)")
    return memory


def test_semantic_similarity_query():
    """Test semantic similarity search."""
    print("\n" + "="*70)
    print("TEST 5: Semantic Similarity Query")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())
    memory.register_subsystem(MemoryType.SEMANTIC, MockSemanticMemory())

    # Create multiple contacts and compress them
    contacts = [
        Contact.create(name="Alice", email="alice@tech.com", title="VP Sales", notes="Enterprise sales expert"),
        Contact.create(name="Bob", email="bob@tech.com", title="CTO", notes="Technology leader"),
        Contact.create(name="Carol", email="carol@tech.com", title="Sales Manager", notes="Sales professional")
    ]

    addresses = []
    for contact in contacts:
        addr = memory.write_symbolic(contact)
        emb_mem = memory.compress(addr)
        addresses.append(emb_mem.address)
        print(f"  Compressed: {contact.name} - {contact.title}")

    # Query for similar to Alice (VP Sales)
    print(f"\nQuerying for contacts similar to Alice (VP Sales)...")
    result = memory.query_semantic(
        criteria={"similar_to": addresses[0]},
        limit=3,
        min_similarity=0.0
    )

    print(f"\nFound {len(result.memories)} similar memories:")
    for i, (mem, score) in enumerate(zip(result.memories, result.scores)):
        print(f"  {i+1}. Address: {mem.address.id}")
        print(f"     Similarity: {score:.3f}")

    assert len(result.memories) > 0
    assert all(isinstance(score, float) for score in result.scores)

    print("\n[OK] Semantic similarity search working")
    return memory


def test_episodic_memory():
    """Test episodic memory (time-ordered activities)."""
    print("\n" + "="*70)
    print("TEST 6: Episodic Memory Operations")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.EPISODIC, MockEpisodicMemory())

    # Create activities
    contact = Contact.create(name="Alice", email="alice@test.com")

    activities = [
        Activity.create(
            type=ActivityType.CALL,
            contact_id=contact.id,
            subject="Discovery call",
            outcome=ActivityOutcome.POSITIVE
        ),
        Activity.create(
            type=ActivityType.EMAIL,
            contact_id=contact.id,
            subject="Follow-up email",
            outcome=ActivityOutcome.NEUTRAL
        ),
        Activity.create(
            type=ActivityType.CALL,
            contact_id=contact.id,
            subject="Demo call",
            outcome=ActivityOutcome.POSITIVE
        )
    ]

    # Store activities
    for activity in activities:
        addr = memory.write_episodic(activity)
        print(f"  Stored activity: {activity.type.value} - {activity.subject}")

    # Query for calls only
    print(f"\nQuerying for CALL activities...")
    result = memory.query_episodic(
        criteria={"type": "call"},
        limit=10
    )

    print(f"\nFound {len(result.memories)} call activities:")
    for mem in result.memories:
        activity = mem.content
        print(f"  - {activity.type.value}: {activity.subject} ({activity.outcome.value})")

    assert len(result.memories) == 2  # Should find 2 calls
    assert all(mem.content.type == ActivityType.CALL for mem in result.memories)

    print("\n[OK] Episodic memory working")
    return memory


def test_relational_associations():
    """Test creating associations between memories."""
    print("\n" + "="*70)
    print("TEST 7: Relational Associations")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())
    memory.register_subsystem(MemoryType.RELATIONAL, MockRelationalMemory())

    # Create entities
    contact = Contact.create(name="Alice", email="alice@techcorp.com", title="VP Sales")
    company = Company.create(name="TechCorp", industry="Technology", size=CompanySize.ENTERPRISE)

    # Store in symbolic memory
    contact_addr = memory.write_symbolic(contact, metadata={"entity_type": "contact"})
    company_addr = memory.write_symbolic(company, metadata={"entity_type": "company"})

    print(f"\nCreated entities:")
    print(f"  Contact: {contact_addr}")
    print(f"  Company: {company_addr}")

    # Create association
    edge_addr = memory.associate(contact_addr, company_addr, "WORKS_AT")

    print(f"\nCreated association:")
    print(f"  Edge: {edge_addr}")
    print(f"  Relation: WORKS_AT")
    print(f"  From: {contact.name}")
    print(f"  To: {company.name}")

    # Query for relationships
    result = memory.query(MemoryQuery(
        subsystem=MemoryType.RELATIONAL,
        criteria={"relation": "WORKS_AT"},
        limit=10
    ))

    print(f"\nFound {len(result.memories)} WORKS_AT relationships:")
    for mem in result.memories:
        print(f"  - {mem.metadata['from']} -> {mem.metadata['to']}")

    assert len(result.memories) == 1
    assert result.memories[0].metadata["relation"] == "WORKS_AT"

    print("\n[OK] Relational associations working")
    return memory


def test_cross_subsystem_operations():
    """Test operations spanning multiple subsystems."""
    print("\n" + "="*70)
    print("TEST 8: Cross-Subsystem Operations")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())
    memory.register_subsystem(MemoryType.SEMANTIC, MockSemanticMemory())
    memory.register_subsystem(MemoryType.EPISODIC, MockEpisodicMemory())
    memory.register_subsystem(MemoryType.RELATIONAL, MockRelationalMemory())

    # Create contact
    contact = Contact.create(
        name="David Chen",
        email="david@startup.io",
        title="Founder & CEO"
    )

    print("\nPerforming cross-subsystem operations:")

    # 1. Write to symbolic
    contact_addr = memory.write_symbolic(contact)
    print(f"  1. Symbolic write: {contact_addr}")

    # 2. Compress to semantic
    emb_mem = memory.compress(contact_addr)
    print(f"  2. Semantic compress: {emb_mem.address}")

    # 3. Log activity to episodic
    activity = Activity.create(
        type=ActivityType.MEETING,
        contact_id=contact.id,
        subject="Product demo",
        outcome=ActivityOutcome.POSITIVE
    )
    activity_addr = memory.write_episodic(activity)
    print(f"  3. Episodic write: {activity_addr}")

    # 4. Create association in relational
    company = Company.create(name="StartupIO", industry="Technology", size=CompanySize.SMALL)
    company_addr = memory.write_symbolic(company)
    edge_addr = memory.associate(contact_addr, company_addr, "FOUNDER_OF")
    print(f"  4. Relational associate: {edge_addr}")

    # Verify all subsystems have data
    stats = memory.stats()
    print(f"\nCross-subsystem statistics:")
    for subsystem, subsystem_stats in stats['subsystems'].items():
        print(f"  {subsystem}: {subsystem_stats}")

    # Check meta-memory
    print(f"\nMeta-memory:")
    print(f"  Total writes: {stats['meta']['total_writes']}")
    print(f"  Total reads: {stats['meta']['total_reads']}")

    assert stats['meta']['total_writes'] >= 3

    print("\n[OK] Cross-subsystem operations working")
    return memory


def test_meta_memory_tracking():
    """Test meta-memory statistics tracking."""
    print("\n" + "="*70)
    print("TEST 9: Meta-Memory Tracking")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())
    memory.register_subsystem(MemoryType.SEMANTIC, MockSemanticMemory())

    # Perform various operations
    contact = Contact.create(name="Test User", email="test@test.com")

    # Write
    addr = memory.write_symbolic(contact)

    # Read
    memory.read(addr)
    memory.read(addr)

    # Compress
    memory.compress(addr)

    # Query
    memory.query_symbolic(criteria={"entity_type": "contact"})

    # Get stats
    stats = memory.stats()

    print("\nMeta-memory statistics:")
    print(f"  Total writes: {stats['meta']['total_writes']}")
    print(f"  Total reads: {stats['meta']['total_reads']}")
    print(f"  Total queries: {stats['meta']['total_queries']}")

    print(f"\nSubsystem statistics:")
    for subsystem, sub_stats in stats['subsystems'].items():
        print(f"  {subsystem}:")
        for key, value in sub_stats.items():
            print(f"    - {key}: {value}")

    # Verify tracking
    assert stats['meta']['total_writes'] >= 1
    assert stats['meta']['total_reads'] >= 2
    assert stats['meta']['total_queries'] >= 1

    print("\n[OK] Meta-memory tracking working")
    return memory


def test_memory_lifecycle():
    """Test complete memory lifecycle (create, update, delete)."""
    print("\n" + "="*70)
    print("TEST 10: Memory Lifecycle (Create/Update/Delete)")
    print("="*70)

    memory = UnifiedMemory()
    memory.register_subsystem(MemoryType.SYMBOLIC, MockSymbolicMemory())

    # Create
    contact = Contact.create(
        name="Lifecycle Test",
        email="test@lifecycle.com",
        title="Engineer"
    )

    addr = memory.write_symbolic(contact, metadata={"version": 1})
    print(f"\n1. Created: {addr}")
    print(f"   Name: {contact.name}")
    print(f"   Title: {contact.title}")

    # Read
    mem = memory.read(addr)
    assert mem is not None
    print(f"\n2. Read: {addr}")
    print(f"   Name: {mem.content.name}")

    # Update
    updated = memory.update(addr, {"title": "Senior Engineer", "version": 2})
    print(f"\n3. Updated: {addr}")
    print(f"   New title: {updated.content.title}")

    # Delete
    deleted = memory.delete(addr)
    assert deleted == True
    print(f"\n4. Deleted: {addr}")

    # Verify deleted
    mem_after_delete = memory.read(addr)
    assert mem_after_delete is None
    print(f"   Verified deletion: memory not found")

    print("\n[OK] Memory lifecycle working")
    return memory


# ============================================================================
# Run All Tests
# ============================================================================

def run_all_tests():
    """Run complete unified memory test suite."""
    print("\n" + "="*70)
    print("UNIFIED MEMORY SYSTEM - TEST SUITE")
    print("="*70)
    print("Testing \"everything is a memory operation\"...")

    tests = [
        ("Memory Addressing", test_memory_addressing),
        ("UnifiedMemory Initialization", test_unified_memory_initialization),
        ("Symbolic Write/Read", test_symbolic_write_read),
        ("Semantic Compression", test_semantic_compression),
        ("Semantic Similarity Query", test_semantic_similarity_query),
        ("Episodic Memory", test_episodic_memory),
        ("Relational Associations", test_relational_associations),
        ("Cross-Subsystem Operations", test_cross_subsystem_operations),
        ("Meta-Memory Tracking", test_meta_memory_tracking),
        ("Memory Lifecycle", test_memory_lifecycle),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSING - Unified memory system working!")
        print("\n\"Everything is a memory operation\" - validated")
    else:
        print(f"\n[FAILED] {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)

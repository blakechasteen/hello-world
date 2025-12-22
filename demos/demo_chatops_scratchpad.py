#!/usr/bin/env python3
"""
Demo: ChatOps Scratch Pad System
================================

Demonstrates the ChatOps Scratch Pad for persistent artifact storage.

Features shown:
1. Store and retrieve artifacts
2. Listing artifacts with scopes
3. Sharing between users in a room
4. Session context management
5. Export functionality
6. Content deduplication
7. Rate limiting
8. Automatic type detection
9. Binary content handling
10. Quota enforcement

Created: December 2025
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

# Setup path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from HoloLoom.chatops.scratchpad import (
    ScratchPadConfig,
    ArtifactScope,
    ArtifactType,
)
from HoloLoom.chatops.scratchpad.manager import (
    ScratchPadManager,
    create_and_start_manager,
)


# ============================================================================
# Demo Utilities
# ============================================================================

def print_header(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def print_artifact(artifact, content: bytes = None) -> None:
    """Print artifact details."""
    print(f"  ID: {artifact.artifact_id}")
    print(f"  Name: {artifact.name}")
    print(f"  Type: {artifact.artifact_type.value}")
    print(f"  Scope: {artifact.scope.value}")
    print(f"  Size: {artifact.size_bytes} bytes")
    print(f"  Owner: {artifact.owner_user_id}")
    if content:
        if isinstance(content, bytes):
            try:
                text = content.decode('utf-8')
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"  Content: {preview}")
            except UnicodeDecodeError:
                print(f"  Content: <binary data, {len(content)} bytes>")


# ============================================================================
# Demo 1: Basic Store and Retrieve
# ============================================================================

async def demo_store_and_retrieve(manager: ScratchPadManager) -> None:
    """Demo 1: Basic store and retrieve operations."""
    print_header("Demo 1: Basic Store and Retrieve")

    user_id = "@alice:matrix.org"
    room_id = "!room123:matrix.org"

    # Store a simple text artifact
    print("Storing a text artifact...")
    result = await manager.store(
        name="hello_world.py",
        content="""#!/usr/bin/env python3
def hello():
    print("Hello, HoloLoom!")

if __name__ == "__main__":
    hello()
""",
        user_id=user_id,
        room_id=room_id,
    )

    if result.success:
        print("SUCCESS: Artifact stored!")
        print_artifact(result.artifact)
    else:
        print(f"ERROR: {result.error}")
        return

    artifact_id = result.artifact.artifact_id

    # Retrieve by ID
    print("\nRetrieving artifact by ID...")
    result = await manager.get(artifact_id, user_id, room_id)

    if result.success:
        print("SUCCESS: Artifact retrieved!")
        print_artifact(result.artifact, result.content)
    else:
        print(f"ERROR: {result.error}")

    # Retrieve by name
    print("\nRetrieving artifact by name...")
    result = await manager.get("hello_world.py", user_id, room_id)

    if result.success:
        print("SUCCESS: Found by name!")
        print(f"  ID: {result.artifact.artifact_id}")
    else:
        print(f"ERROR: {result.error}")


# ============================================================================
# Demo 2: Listing Artifacts
# ============================================================================

async def demo_listing(manager: ScratchPadManager) -> None:
    """Demo 2: Listing artifacts with different scopes."""
    print_header("Demo 2: Listing Artifacts")

    user_id = "@bob:matrix.org"
    room_id = "!room456:matrix.org"

    # Store multiple artifacts
    print("Storing multiple artifacts...")

    artifacts = [
        ("config.json", '{"key": "value", "debug": true}', ArtifactScope.USER),
        ("utils.py", "def add(a, b): return a + b", ArtifactScope.USER),
        ("shared_data.csv", "name,value\nalice,100\nbob,200", ArtifactScope.ROOM),
    ]

    for name, content, scope in artifacts:
        result = await manager.store(
            name=name,
            content=content,
            user_id=user_id,
            room_id=room_id,
            scope=scope,
        )
        status = "OK" if result.success else f"FAIL: {result.error}"
        print(f"  {name}: {status}")

    # List all user artifacts
    print("\nListing all accessible artifacts...")
    result = await manager.list(user_id, room_id)

    print(f"  Total: {result.total_count} artifacts")
    print(f"  Total size: {result.total_size_bytes} bytes")

    for artifact in result.artifacts:
        print(f"    - {artifact.name} ({artifact.scope.value}, {artifact.size_bytes} bytes)")

    # List only ROOM scope
    print("\nListing ROOM scope artifacts only...")
    result = await manager.list(user_id, room_id, scope_filter=ArtifactScope.ROOM)

    print(f"  Found: {result.total_count} ROOM artifacts")
    for artifact in result.artifacts:
        print(f"    - {artifact.name}")


# ============================================================================
# Demo 3: Sharing Between Users
# ============================================================================

async def demo_sharing(manager: ScratchPadManager) -> None:
    """Demo 3: Sharing artifacts between users in a room."""
    print_header("Demo 3: Sharing Between Users")

    alice = "@alice:matrix.org"
    bob = "@bob:matrix.org"
    room_id = "!shared_room:matrix.org"

    # Alice stores a private artifact
    print("Alice stores a private artifact...")
    result = await manager.store(
        name="alices_secret.txt",
        content="Alice's private notes",
        user_id=alice,
        room_id=room_id,
        scope=ArtifactScope.USER,  # Private to Alice
    )

    if not result.success:
        print(f"ERROR: {result.error}")
        return

    artifact_id = result.artifact.artifact_id
    print(f"  Stored: {result.artifact.name} (scope: USER)")

    # Bob tries to access (should fail)
    print("\nBob tries to access Alice's artifact...")
    result = await manager.get(artifact_id, bob, room_id)

    if not result.success:
        print(f"  Access denied: {result.error}")
    else:
        print("  Unexpected: Bob could access!")

    # Alice shares with room
    print("\nAlice shares the artifact with the room...")
    success, error = await manager.share(artifact_id, alice, room_id)

    if success:
        print("  Artifact shared! Scope changed to ROOM.")
    else:
        print(f"  ERROR: {error}")

    # Bob tries again (should succeed)
    print("\nBob tries to access again...")
    result = await manager.get(artifact_id, bob, room_id)

    if result.success:
        print(f"  SUCCESS: Bob can now access '{result.artifact.name}'")
    else:
        print(f"  ERROR: {result.error}")


# ============================================================================
# Demo 4: Session Context Management
# ============================================================================

async def demo_session_context(manager: ScratchPadManager) -> None:
    """Demo 4: Session context for multi-turn conversations."""
    print_header("Demo 4: Session Context Management")

    user_id = "@charlie:matrix.org"
    room_id = "!session_room:matrix.org"
    session_id = "session_abc123"

    # Store artifacts linked to a session
    print("Storing artifacts linked to a session...")

    # First artifact
    result = await manager.store(
        name="step1_query.txt",
        content="What is Thompson Sampling?",
        user_id=user_id,
        room_id=room_id,
        source_session_id=session_id,
    )
    print(f"  Step 1: {result.artifact.name if result.success else result.error}")

    # Second artifact
    result = await manager.store(
        name="step1_response.txt",
        content="Thompson Sampling is a Bayesian approach to the multi-armed bandit problem...",
        user_id=user_id,
        room_id=room_id,
        source_session_id=session_id,
    )
    print(f"  Step 2: {result.artifact.name if result.success else result.error}")

    # Third artifact
    result = await manager.store(
        name="step2_followup.txt",
        content="How does it compare to UCB?",
        user_id=user_id,
        room_id=room_id,
        source_session_id=session_id,
    )
    print(f"  Step 3: {result.artifact.name if result.success else result.error}")

    # Get session context
    print("\nRetrieving session context...")
    context = manager.get_session_context(session_id, user_id, room_id)

    print(f"  Session ID: {context.session_id}")
    print(f"  Artifacts in session: {len(context.artifact_references)}")

    for ref in context.artifact_references:
        print(f"    - {ref.name} ({ref.artifact_type.value})")

    # Clear session context
    print("\nClearing session context...")
    manager.clear_session_context(session_id, user_id, room_id)

    context = manager.get_session_context(session_id, user_id, room_id)
    print(f"  Artifacts after clear (reloaded from storage): {len(context.artifact_references)}")


# ============================================================================
# Demo 5: Export Functionality
# ============================================================================

async def demo_export(manager: ScratchPadManager) -> None:
    """Demo 5: Export artifacts to files and Matrix."""
    print_header("Demo 5: Export Functionality")

    user_id = "@dan:matrix.org"
    room_id = "!export_room:matrix.org"

    # Store an artifact
    print("Storing an artifact for export...")
    result = await manager.store(
        name="report.json",
        content='{"title": "Monthly Report", "data": [1, 2, 3]}',
        user_id=user_id,
        room_id=room_id,
    )

    if not result.success:
        print(f"ERROR: {result.error}")
        return

    print(f"  Stored: {result.artifact.name}")

    # Export to file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "exported_report.json"

        print(f"\nExporting to file: {output_path}")
        success, error = await manager.export_to_file(
            "report.json",
            user_id,
            room_id,
            output_path
        )

        if success:
            print(f"  SUCCESS: File created ({output_path.stat().st_size} bytes)")
            print(f"  Content: {output_path.read_text()[:100]}")
        else:
            print(f"  ERROR: {error}")

    # Get export data for Matrix
    print("\nGetting export data for Matrix upload...")
    content, filename, mimetype = await manager.get_export_data(
        "report.json",
        user_id,
        room_id
    )

    if content:
        print(f"  Filename: {filename}")
        print(f"  MIME type: {mimetype}")
        print(f"  Content size: {len(content)} bytes")
    else:
        print(f"  ERROR: {mimetype}")  # error is in mimetype position


# ============================================================================
# Demo 6: Content Deduplication
# ============================================================================

async def demo_deduplication(manager: ScratchPadManager) -> None:
    """Demo 6: Content deduplication via content-addressed storage."""
    print_header("Demo 6: Content Deduplication")

    user_id = "@eve:matrix.org"
    room_id = "!dedup_room:matrix.org"

    # Store same content with different names
    print("Storing identical content with different names...")

    content = """
This is some shared content that appears multiple times.
It should be deduplicated at the storage level.
Only one copy of the actual bytes should exist.
"""

    names = ["version1.txt", "version2.txt", "version3.txt"]

    for name in names:
        result = await manager.store(
            name=name,
            content=content,
            user_id=user_id,
            room_id=room_id,
        )
        status = "OK" if result.success else result.error
        print(f"  {name}: {status}")

    # Check storage stats
    print("\nStorage statistics:")
    stats = manager.get_stats()

    print(f"  Metadata entries: {stats.get('metadata_count', 'N/A')}")
    print(f"  Unique content files: {stats.get('content_count', 'N/A')}")
    print(f"  Total content size: {stats.get('total_content_bytes', 'N/A')} bytes")

    # Note: 3 metadata entries but only 1 unique content blob
    print("\n  Note: Multiple artifacts reference the same content hash!")


# ============================================================================
# Demo 7: Type Detection
# ============================================================================

async def demo_type_detection(manager: ScratchPadManager) -> None:
    """Demo 7: Automatic artifact type detection."""
    print_header("Demo 7: Automatic Type Detection")

    user_id = "@frank:matrix.org"
    room_id = "!types_room:matrix.org"

    # Store various content types
    print("Storing various content types (auto-detection)...")

    samples = [
        ("script.py", "def main(): pass"),
        ("data.json", '{"name": "test"}'),
        ("notes.txt", "Just some plain text notes"),
        ("document.md", "# Heading\n\nSome markdown content"),
        ("config.yaml", "key: value\nlist:\n  - item1"),
        ("code.js", "function hello() { return 42; }"),
    ]

    for name, content in samples:
        result = await manager.store(
            name=name,
            content=content,
            user_id=user_id,
            room_id=room_id,
        )

        if result.success:
            detected = result.artifact.artifact_type.value
            print(f"  {name:20} -> {detected}")
        else:
            print(f"  {name:20} -> ERROR: {result.error}")


# ============================================================================
# Demo 8: Binary Content
# ============================================================================

async def demo_binary_content(manager: ScratchPadManager) -> None:
    """Demo 8: Handling binary content (images, archives)."""
    print_header("Demo 8: Binary Content Handling")

    user_id = "@grace:matrix.org"
    room_id = "!binary_room:matrix.org"

    # Create a fake PNG (minimal valid PNG header)
    png_header = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D,  # IHDR length
        0x49, 0x48, 0x44, 0x52,  # IHDR type
        0x00, 0x00, 0x00, 0x01,  # width: 1
        0x00, 0x00, 0x00, 0x01,  # height: 1
        0x08, 0x02,              # bit depth, color type
        0x00, 0x00, 0x00,        # compression, filter, interlace
    ])

    print("Storing a PNG image (binary content)...")
    result = await manager.store(
        name="test_image.png",
        content=png_header,  # bytes, not str
        user_id=user_id,
        room_id=room_id,
    )

    if result.success:
        print(f"  SUCCESS: Stored {result.artifact.size_bytes} bytes")
        print(f"  Type: {result.artifact.artifact_type.value}")
    else:
        print(f"  ERROR: {result.error}")

    # Retrieve and verify
    print("\nRetrieving binary content...")
    result = await manager.get("test_image.png", user_id, room_id)

    if result.success:
        print(f"  Retrieved: {len(result.content)} bytes")
        print(f"  PNG header valid: {result.content[:8] == png_header[:8]}")
    else:
        print(f"  ERROR: {result.error}")


# ============================================================================
# Demo 9: Quota and Limits
# ============================================================================

async def demo_quotas(manager: ScratchPadManager) -> None:
    """Demo 9: Quota enforcement."""
    print_header("Demo 9: Quota and Limits")

    user_id = "@henry:matrix.org"
    room_id = "!quota_room:matrix.org"

    print(f"Manager quotas:")
    print(f"  Max artifacts per user: {manager.config.max_artifacts_per_user}")
    print(f"  Max artifact size: {manager.config.max_artifact_size_bytes // (1024*1024)} MB")
    print(f"  Max total storage: {manager.config.max_total_storage_bytes // (1024*1024*1024)} GB")

    # Store many small artifacts to approach quota
    print("\nStoring multiple artifacts...")

    for i in range(5):
        result = await manager.store(
            name=f"test_{i}.txt",
            content=f"Test content {i}",
            user_id=user_id,
            room_id=room_id,
        )
        status = "OK" if result.success else result.error
        print(f"  test_{i}.txt: {status}")

    # Check user artifact count
    result = await manager.list(user_id, room_id)
    print(f"\n  User has {result.total_count} artifacts ({result.total_size_bytes} bytes)")


# ============================================================================
# Demo 10: Delete and Cleanup
# ============================================================================

async def demo_delete(manager: ScratchPadManager) -> None:
    """Demo 10: Deleting artifacts."""
    print_header("Demo 10: Delete and Cleanup")

    user_id = "@iris:matrix.org"
    room_id = "!delete_room:matrix.org"

    # Store an artifact
    print("Storing an artifact...")
    result = await manager.store(
        name="to_delete.txt",
        content="This will be deleted",
        user_id=user_id,
        room_id=room_id,
    )

    if not result.success:
        print(f"ERROR: {result.error}")
        return

    artifact_id = result.artifact.artifact_id
    print(f"  Stored: {artifact_id}")

    # Verify it exists
    result = await manager.get(artifact_id, user_id, room_id)
    print(f"  Exists before delete: {result.success}")

    # Delete it
    print("\nDeleting artifact...")
    result = await manager.delete(artifact_id, user_id, room_id)

    if result.success:
        print(f"  SUCCESS: Deleted {result.artifact_id}")
    else:
        print(f"  ERROR: {result.error}")

    # Verify it's gone
    result = await manager.get(artifact_id, user_id, room_id)
    print(f"  Exists after delete: {result.success}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all demos."""
    print()
    print("*" * 70)
    print("  ChatOps Scratch Pad Demo")
    print("  Persistent Artifact Storage for Multi-Step Workflows")
    print("*" * 70)
    print()
    print("This demo shows the ChatOps Scratch Pad system for:")
    print("  - Passing context between !continue commands")
    print("  - Storing intermediate results in workflows")
    print("  - Sharing artifacts between users in a room")
    print()

    # Create manager with temporary storage
    # Use a dedicated demo directory (cleaned up at start, not end to avoid
    # Windows file locking issues with SQLite)
    import shutil
    demo_dir = Path(tempfile.gettempdir()) / "hololoom_scratchpad_demo"
    if demo_dir.exists():
        try:
            shutil.rmtree(demo_dir)
        except Exception:
            pass  # Ignore if previous cleanup failed
    demo_dir.mkdir(parents=True, exist_ok=True)

    config = ScratchPadConfig(
        metadata_db_path=str(demo_dir / "metadata.db"),
        cas_storage_path=str(demo_dir / "content"),
        audit_log_path=str(demo_dir / "audit.jsonl"),
        # Relaxed limits for demo
        max_artifacts_per_user=1000,
        rate_limit_requests=100,
        rate_limit_window_seconds=60,
    )

    manager = await create_and_start_manager(config)

    try:
        # Run all demos
        await demo_store_and_retrieve(manager)
        await demo_listing(manager)
        await demo_sharing(manager)
        await demo_session_context(manager)
        await demo_export(manager)
        await demo_deduplication(manager)
        await demo_type_detection(manager)
        await demo_binary_content(manager)
        await demo_quotas(manager)
        await demo_delete(manager)

        # Final stats
        print_header("Final Statistics")
        stats = manager.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    finally:
        await manager.stop()

    print()
    print("*" * 70)
    print("  DEMO COMPLETE")
    print("*" * 70)
    print()
    print("ChatOps Scratch Pad Commands:")
    print("  !scratch store <name>    - Store last weaving result")
    print("  !scratch get <name|id>   - Retrieve artifact")
    print("  !scratch list [scope]    - List artifacts")
    print("  !scratch delete <name>   - Delete artifact")
    print("  !scratch share <name>    - Share with room")
    print("  !scratch export <name>   - Upload to Matrix")
    print()
    print("See HoloLoom/chatops/README_SCRATCHPAD.md for full documentation.")
    print()


if __name__ == "__main__":
    asyncio.run(main())

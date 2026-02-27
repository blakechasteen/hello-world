#!/usr/bin/env python3
"""
Demo: Collaborative Spatial Session
=====================================

Interactive demo showing HoloLoom's Phase 4 collaborative spatial computing:
- Create collaborative session
- Join with multiple users (simulated)
- Gesture-to-action mappings
- Zone-based context awareness

Part of HoloLoom Phase 4: Spatial Computing Enhancement (November 2025)
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Spatial imports
from hololoom.spatial.collaborative_session import (
    CollaborativeSpatialSession,
    create_collaborative_spatial_session,
    create_and_start_session,
    SpatialActivityType,
    SpatialZoneType,
    GestureAction,
    SpatialContext,
    SpatialZone,
    SharedSpatialObject,
    GestureMapping
)
from hololoom.spatial.math_types import Vector3, Quaternion, Color, Transform
from hololoom.spatial.hand_tracking import HandGesture, GestureType
from hololoom.collaboration.session import SessionType


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---")


async def demo_session_creation():
    """Demo 1: Create and configure a collaborative spatial session."""
    print_header("Demo 1: Session Creation")

    # Create session using factory function
    session = create_collaborative_spatial_session(
        name="HoloLoom Knowledge Workshop",
        owner_id="user_alice",
        owner_name="Alice",
        session_type=SessionType.KNOWLEDGE_BASE
    )

    print(f"\n✓ Created session: {session.name}")
    print(f"  Session ID: {session.session_id}")
    print(f"  Owner: {session.owner_id}")
    print(f"  Type: {session.session_type.name}")

    # Show default zones
    print_subheader("Default Spatial Zones")
    for zone_id, zone in session.zones.items():
        print(f"  • {zone.name} ({zone.zone_type.name})")
        print(f"    Center: ({zone.center.x}, {zone.center.y}, {zone.center.z})")
        print(f"    Size: {zone.size.x}x{zone.size.y}x{zone.size.z}m")

    return session


async def demo_multi_user_join(session: CollaborativeSpatialSession):
    """Demo 2: Multiple users joining the session."""
    print_header("Demo 2: Multi-User Join")

    # Start the session
    await session.start()
    print("✓ Session started")

    # Simulate multiple users joining
    users = [
        ("user_alice", "Alice", "professional"),
        ("user_bob", "Bob", "casual"),
        ("user_charlie", "Charlie", "robot"),
        ("user_diana", "Diana", "hologram"),
    ]

    print_subheader("Users Joining Session")
    for user_id, display_name, preset in users:
        avatar, presence = await session.join(user_id, display_name, preset)
        print(f"  ✓ {display_name} joined with {preset} avatar")
        print(f"    Avatar ID: {avatar.avatar_id}")
        print(f"    Position: ({avatar.position.x:.1f}, {avatar.position.y:.1f}, {avatar.position.z:.1f})")

    # Show presence info
    print_subheader("Session Presence")
    for user_id, presence in session.presence_manager.presences.items():
        print(f"  • {presence.display_name} - {presence.activity_status.name}")

    return users


async def demo_zone_context_awareness(session: CollaborativeSpatialSession, users: List):
    """Demo 3: Zone-based context awareness."""
    print_header("Demo 3: Zone-Based Context Awareness")

    # Move users to different zones
    zone_positions = {
        "user_alice": (Vector3(0, 0, 0), "Main Stage"),           # Main stage
        "user_bob": (Vector3(8, 0, 0), "Whiteboard Zone"),        # Whiteboard
        "user_charlie": (Vector3(-8, 0, 0), "Knowledge Garden"),  # Knowledge garden
        "user_diana": (Vector3(0, 0, 8), "Discussion Area"),      # Discussion
    }

    print_subheader("Moving Users to Different Zones")
    for user_id, (position, zone_name) in zone_positions.items():
        # Update position in avatar
        avatar = session.avatar_manager.avatars.get(user_id)
        if avatar:
            avatar.position = position

        # Update context
        ctx = session.context_provider.update_context(
            user_id=user_id,
            position=position,
            rotation=Quaternion.identity()
        )

        print(f"  • {user_id}: moved to {zone_name}")
        print(f"    Position: ({position.x:.1f}, {position.y:.1f}, {position.z:.1f})")
        print(f"    Detected Zone: {ctx.current_zone.name if ctx.current_zone else 'None'}")

    # Show nearby users detection
    print_subheader("Nearby Users Detection")

    # Move Bob closer to Alice to test proximity
    bob_close_to_alice = Vector3(2, 0, 0)  # Within nearby_distance (3m)
    session.context_provider.update_context(
        "user_bob",
        bob_close_to_alice,
        Quaternion.identity()
    )

    alice_ctx = session.context_provider.get_context("user_alice")
    bob_ctx = session.context_provider.get_context("user_bob")

    print(f"  • Alice's nearby users: {alice_ctx.nearby_users if alice_ctx else []}")
    print(f"  • Bob's nearby users: {bob_ctx.nearby_users if bob_ctx else []}")

    # Demo knowledge query generation based on zone
    print_subheader("Context-Aware Knowledge Queries")
    for user_id, _ in users:
        query = session.context_provider.get_relevant_knowledge_query(user_id)
        ctx = session.context_provider.get_context(user_id)
        zone = ctx.current_zone.name if ctx and ctx.current_zone else "No Zone"
        print(f"  • {user_id} ({zone}): {query or 'No query generated'}")


async def demo_gesture_mappings(session: CollaborativeSpatialSession):
    """Demo 4: Gesture-to-action mappings."""
    print_header("Demo 4: Gesture-to-Action Mappings")

    # Show default gesture mappings
    print_subheader("Default Gesture Mappings")
    for gesture_type, mapping in session.gesture_bridge.mappings.items():
        target_req = " (requires target)" if mapping.requires_target else ""
        print(f"  • {gesture_type.name} → {mapping.action.name}{target_req}")
        print(f"    Cooldown: {mapping.cooldown_seconds}s")

    # Add custom gesture mapping
    print_subheader("Adding Custom Gesture Mapping")
    session.gesture_bridge.add_mapping(
        gesture_type=GestureType.FIST,
        action=GestureAction.PULL,
        requires_target=True,
        cooldown=1.0
    )
    print("  ✓ Added: FIST → PULL (requires target, 1.0s cooldown)")

    # Simulate gesture handling
    print_subheader("Simulating Gesture Actions")

    # Create a test gesture (wave)
    wave_gesture = HandGesture(
        gesture_type=GestureType.WAVE,
        confidence=0.95,
        hand="right"
    )

    # Handle the gesture
    result = await session.gesture_bridge.handle_gesture(
        user_id="user_alice",
        gesture=wave_gesture,
        target=None
    )
    print(f"  • Alice performed WAVE gesture")
    print(f"    Action triggered: {result}")

    # Point gesture (requires target)
    point_gesture = HandGesture(
        gesture_type=GestureType.POINT,
        confidence=0.90,
        hand="right"
    )

    # Without target (should fail)
    result_no_target = await session.gesture_bridge.handle_gesture(
        user_id="user_bob",
        gesture=point_gesture,
        target=None
    )
    print(f"  • Bob performed POINT gesture (no target)")
    print(f"    Action triggered: {result_no_target} (expected False)")

    # With target (should succeed)
    result_with_target = await session.gesture_bridge.handle_gesture(
        user_id="user_bob",
        gesture=point_gesture,
        target="object:knowledge_node_1"
    )
    print(f"  • Bob performed POINT gesture (with target)")
    print(f"    Action triggered: {result_with_target}")


async def demo_shared_objects(session: CollaborativeSpatialSession):
    """Demo 5: Shared spatial objects."""
    print_header("Demo 5: Shared Spatial Objects")

    # Create shared objects
    print_subheader("Creating Shared Objects")

    # Create a knowledge node object
    knowledge_obj = await session.create_shared_object(
        user_id="user_alice",
        name="Thompson Sampling Concept",
        object_type="knowledge_node",
        position=Vector3(0, 1.5, 0)
    )
    knowledge_obj.knowledge_node_id = "ts_001"
    knowledge_obj.memory_id = "mem_ts_001"
    print(f"  ✓ Created: {knowledge_obj.name}")
    print(f"    Type: {knowledge_obj.object_type}")
    print(f"    Owner: {knowledge_obj.owner_id}")

    # Create an annotation
    annotation_obj = await session.create_shared_object(
        user_id="user_bob",
        name="Important Note",
        object_type="annotation",
        position=Vector3(2, 1, 0)
    )
    print(f"  ✓ Created: {annotation_obj.name}")

    # Grab object
    print_subheader("Object Manipulation")
    grab_result = await session.grab_object("user_charlie", knowledge_obj.object_id)
    print(f"  • Charlie grabbed '{knowledge_obj.name}': {grab_result}")

    # Check Charlie's context
    charlie_ctx = session.context_provider.get_context("user_charlie")
    if charlie_ctx:
        print(f"    Charlie's held objects: {charlie_ctx.held_objects}")
        print(f"    Charlie's activity: {charlie_ctx.activity.name}")

    # Release object
    release_result = await session.release_object("user_charlie", knowledge_obj.object_id)
    print(f"  • Charlie released object: {release_result}")

    # Show session state summary
    print_subheader("Session State Summary")
    state = session.to_state()
    print(f"  • Users: {state['user_count']}")
    print(f"  • Shared Objects: {state['object_count']}")
    print(f"  • Zones: {len(state['zones'])}")


async def demo_activity_detection(session: CollaborativeSpatialSession):
    """Demo 6: Activity detection based on user behavior."""
    print_header("Demo 6: Activity Detection")

    print_subheader("Simulating Different Activities")

    # Idle activity (no movement for a while)
    print("\n  Scenario 1: Idle User")
    ctx = session.context_provider.update_context(
        "user_alice",
        Vector3(0, 0, 0),
        Quaternion.identity()
    )
    # Simulate time passing without movement
    ctx.activity_duration = 5.0  # 5 seconds idle
    print(f"    Alice's activity: {ctx.activity.name}")

    # Moving activity
    print("\n  Scenario 2: Exploring User")
    ctx = session.context_provider.update_context(
        "user_bob",
        Vector3(5, 0, 5),  # Moved significantly
        Quaternion.identity()
    )
    print(f"    Bob's activity: {ctx.activity.name}")

    # Drawing activity (with active tool)
    print("\n  Scenario 3: Drawing User")
    charlie_ctx = session.context_provider.get_context("user_charlie")
    if charlie_ctx:
        charlie_ctx.active_tools = ["whiteboard_brush"]
        new_ctx = session.context_provider.update_context(
            "user_charlie",
            charlie_ctx.position,
            Quaternion.identity()
        )
        new_ctx.active_tools = ["whiteboard_brush"]
        print(f"    Charlie's activity: {new_ctx.activity.name}")

    # Collaborating activity (near another user, looking at them)
    print("\n  Scenario 4: Collaborating Users")
    # Move Diana close to Alice
    session.context_provider.update_context(
        "user_diana",
        Vector3(1, 0, 0),  # Very close to Alice
        Quaternion.identity()
    )
    diana_ctx = session.context_provider.get_context("user_diana")
    if diana_ctx:
        diana_ctx.attention_history = ["user:user_alice", "user:user_alice", "user:user_alice"]
        print(f"    Diana's nearby users: {diana_ctx.nearby_users}")
        print(f"    Diana's attention history: {diana_ctx.attention_history[-3:]}")


async def demo_cleanup(session: CollaborativeSpatialSession, users: List):
    """Demo 7: Clean session teardown."""
    print_header("Demo 7: Session Cleanup")

    # Users leaving
    print_subheader("Users Leaving Session")
    for user_id, display_name, _ in users:
        await session.leave(user_id)
        print(f"  ✓ {display_name} left the session")

    # Stop session
    await session.stop()
    print("\n✓ Session stopped successfully")

    # Final state
    final_state = session.to_state()
    print(f"  Final user count: {final_state['user_count']}")


async def main():
    """Run the complete collaborative spatial session demo."""
    print("\n" + "=" * 60)
    print("  HoloLoom Phase 4: Collaborative Spatial Session Demo")
    print("  November 2025")
    print("=" * 60)

    try:
        # Demo 1: Create session
        session = await demo_session_creation()

        # Demo 2: Multi-user join
        users = await demo_multi_user_join(session)

        # Demo 3: Zone context awareness
        await demo_zone_context_awareness(session, users)

        # Demo 4: Gesture mappings
        await demo_gesture_mappings(session)

        # Demo 5: Shared objects
        await demo_shared_objects(session)

        # Demo 6: Activity detection
        await demo_activity_detection(session)

        # Demo 7: Cleanup
        await demo_cleanup(session, users)

        print("\n" + "=" * 60)
        print("  Demo Complete!")
        print("=" * 60)
        print("\nAll features demonstrated:")
        print("  ✓ Session creation with factory functions")
        print("  ✓ Multi-user join with avatar presets")
        print("  ✓ Zone-based context awareness")
        print("  ✓ Nearby user detection")
        print("  ✓ Context-aware knowledge queries")
        print("  ✓ Gesture-to-action mappings")
        print("  ✓ Shared object manipulation")
        print("  ✓ Activity detection")
        print("  ✓ Clean session teardown")

    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

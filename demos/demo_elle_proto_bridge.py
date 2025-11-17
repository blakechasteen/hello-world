"""
Demo: Elle ↔ Proto Bridge Integration

This demo shows the complete bidirectional bridge between Elle (AR observer)
and Proto (Matrix bot) via HoloLoom shared memory.

Workflow Demonstrated:
    1. Elle observes workshop via AR
    2. Elle stores observation in HoloLoom
    3. HoloLoom triggers notification to Proto
    4. Proto posts formatted update to Matrix
    5. User asks Proto for summary
    6. Proto retrieves from shared memory
    7. Proto displays formatted response

Created: 2025-11-17 (Week 4: Elle AR Bridge Integration)
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add elle to path
elle_path = Path(__file__).parent.parent / "elle"
sys.path.insert(0, str(elle_path))

# Add proto to path
proto_path = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(proto_path))

from elle.memory.proto_bridge import (
    ElleObservation,
    ElleProtoMemoryBridge,
    create_bridge
)
from proto.bot.elle_bridge import ProtoElleBridge
from proto.bot.elle_commands import ElleCommandHandler


async def demo_elle_side():
    """
    Demonstrate Elle side: Observing and storing.
    """
    print("=" * 70)
    print("ELLE SIDE: Observing Physical Space")
    print("=" * 70)
    print()

    # Create bridge
    async with create_bridge() as bridge:
        print("✅ Elle bridge initialized")
        print()

        # Simulate Elle observing workshop
        print("👁️  Elle observing workshop...")
        print("   User is standing at cluttered workbench")
        print("   Sawdust covering surfaces, tools scattered")
        print()

        # Create observation
        observation_1 = ElleObservation(
            observation_id="obs_workshop_20251117_1430",
            timestamp=datetime.now(),
            location="workshop",
            scene_summary="Workbench cluttered with sawdust and scattered tools",
            objects_observed=["shop_vac", "hand_tools", "workbench", "sawdust"],
            action_taken="Suggested: 'Start with the sawdust?'",
            user_response="User grabbed shop vac and started cleaning",
            tags=["cluttered", "needs_attention"],
            notify_proto=True,
        )

        # Store observation
        memory_id = await bridge.store_observation(observation_1)
        print(f"✅ Observation stored: {memory_id}")
        print()

        # Simulate 15 minutes later
        print("⏱️  15 minutes later...")
        print()

        # Create follow-up observation
        observation_2 = ElleObservation(
            observation_id="obs_workshop_20251117_1445",
            timestamp=datetime.now() + timedelta(minutes=15),
            location="workshop",
            scene_summary="Workbench cleared and organized, tools in proper locations",
            objects_observed=["workbench", "hand_tools", "shop_vac"],
            action_taken="Suggested: 'Looks good. Ready for the birdhouse?'",
            user_response="User acknowledged but mentioned missing cedar boards",
            deferred_tasks=[
                {
                    'description': 'Birdhouse project',
                    'reason': 'Need cedar boards (1x6, 6ft)',
                    'elle_note': 'Check toolbox for brad nails and wood glue',
                    'priority': 'medium',
                }
            ],
            tags=["productive", "organized", "deferred"],
            notify_proto=True,
        )

        memory_id_2 = await bridge.store_observation(observation_2)
        print(f"✅ Follow-up observation stored: {memory_id_2}")
        print()

        # Show what's stored
        print("📊 Observations in HoloLoom:")
        observations = await bridge.query_observations(
            query_type="location",
            location="workshop",
            limit=5
        )
        for obs in observations:
            time_str = obs.timestamp.strftime("%I:%M %p")
            print(f"  • {time_str}: {obs.scene_summary}")
        print()

        return observations


async def demo_proto_side():
    """
    Demonstrate Proto side: Querying and displaying.
    """
    print("=" * 70)
    print("PROTO SIDE: Querying Elle's Memory")
    print("=" * 70)
    print()

    # Create Proto bridge
    bridge = ProtoElleBridge()
    await bridge.start()

    print("✅ Proto bridge initialized")
    print()

    # Create command handler
    handler = ElleCommandHandler(bridge)

    # Simulate user commands
    commands = [
        ("workshop-summary", "User asks: @proto workshop-summary"),
        ("elle-status", "User asks: @proto elle-status"),
        ("ask-elle What did we accomplish in the workshop?", "User asks: @proto ask-elle What did we accomplish?"),
        ("recent-observations workshop", "User asks: @proto recent-observations workshop"),
        ("deferred-tasks", "User asks: @proto deferred-tasks"),
    ]

    for cmd, description in commands:
        print(f"💬 {description}")
        print("─" * 70)

        response = await bridge.handle_command(f"@proto {cmd}", "!demo:matrix.org")

        if response:
            print(response)
        else:
            print("(Not an Elle command)")

        print()
        print()

    await bridge.stop()


async def demo_full_workflow():
    """
    Demonstrate complete workflow: Elle → HoloLoom → Proto.
    """
    print()
    print("=" * 70)
    print("COMPLETE WORKFLOW: Elle → HoloLoom → Proto → Matrix")
    print("=" * 70)
    print()

    # Track notifications
    notifications_received = []

    async def notification_handler(observation: ElleObservation):
        """Handle notification from Elle."""
        notifications_received.append(observation)
        print(f"🔔 Proto received notification: {observation.observation_id}")
        print(f"   Location: {observation.location}")
        print(f"   Summary: {observation.scene_summary}")
        print()

    # Initialize bridges
    print("🔗 Initializing bridges...")
    print()

    # Elle bridge (with notification handler)
    elle_bridge = ElleProtoMemoryBridge(notification_handler=notification_handler)
    await elle_bridge.__aenter__()

    # Proto bridge (shares same HoloLoom)
    proto_bridge = ProtoElleBridge()
    await proto_bridge.start()

    print("✅ Both bridges connected to shared HoloLoom")
    print()

    # Step 1: Elle observes
    print("=" * 70)
    print("STEP 1: Elle Observes Workshop")
    print("=" * 70)
    print()

    observation = ElleObservation(
        observation_id="obs_demo_workflow",
        timestamp=datetime.now(),
        location="workshop",
        scene_summary="Completed workbench organization and tool storage",
        objects_observed=["workbench", "tool_chest", "pegboard", "hand_tools"],
        action_taken="Guided organization process over 30 minutes",
        user_response="Completed full cleanup and organization",
        deferred_tasks=[
            {
                'description': 'Install additional pegboard hooks',
                'reason': 'Need 1/4" hooks for smaller tools',
                'elle_note': 'Hardware store trip next weekend',
            },
            {
                'description': 'Label tool drawers',
                'reason': 'Improve findability',
                'elle_note': 'Label maker in office desk',
            },
        ],
        photo_ids=["photo_before_001", "photo_after_002"],
        tags=["productive", "organization", "deferred"],
        notify_proto=True,
    )

    # Store observation
    await elle_bridge.store_observation(observation)
    await asyncio.sleep(0.1)  # Let notification propagate

    print()

    # Step 2: Proto receives notification
    print("=" * 70)
    print("STEP 2: Proto Receives Notification")
    print("=" * 70)
    print()

    if notifications_received:
        print(f"✅ Proto received {len(notifications_received)} notification(s)")
        print()
    else:
        print("⚠️  No notifications received (handler may not be triggered in demo)")
        print()

    # Step 3: User queries Proto
    print("=" * 70)
    print("STEP 3: User Queries Proto from Matrix")
    print("=" * 70)
    print()

    print("💬 User: @proto workshop-summary")
    print("─" * 70)

    response = await proto_bridge.handle_command(
        "@proto workshop-summary",
        "!demo:matrix.org"
    )
    print(response)
    print()
    print()

    print("💬 User: @proto deferred-tasks workshop")
    print("─" * 70)

    response = await proto_bridge.handle_command(
        "@proto deferred-tasks workshop",
        "!demo:matrix.org"
    )
    print(response)
    print()

    # Cleanup
    await elle_bridge.__aexit__(None, None, None)
    await proto_bridge.stop()

    print()
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("✅ Successfully demonstrated:")
    print("   • Elle storing observations in HoloLoom")
    print("   • Proto querying observations from HoloLoom")
    print("   • Bidirectional bridge communication")
    print("   • Formatted Matrix output")
    print()


async def main():
    """Run all demos."""
    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "Elle ↔ Proto Bridge Demo" + " " * 29 + "║")
    print("║" + " " * 15 + "Week 4: AR Integration" + " " * 31 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    # Run demos
    await demo_elle_side()
    print()
    print()

    await demo_proto_side()
    print()
    print()

    await demo_full_workflow()


if __name__ == "__main__":
    asyncio.run(main())

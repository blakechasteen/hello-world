"""
SharedWorldState - Redis-backed shared world state for multiplayer

Manages synchronized world flags, active players, and NPC state across
all connected clients using Redis pub/sub and data structures.

Author: BigPlay Team
Created: 2025-11-17
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from redis import asyncio as aioredis


@dataclass
class WorldFlag:
    """Single world flag with metadata"""
    key: str
    value: bool
    set_by: str  # player_id or "system"
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "value": self.value,
            "set_by": self.set_by,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class NPCStateUpdate:
    """NPC state change event"""
    npc_id: str
    update_type: str  # "emotion", "location", "quest", "conversation"
    data: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "npc_id": self.npc_id,
            "update_type": self.update_type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class SharedWorldState:
    """
    Redis-backed shared world state for multiplayer games.

    Features:
    - World flags synchronized across all players
    - Active player tracking
    - NPC state broadcasting
    - Pub/sub event system

    Usage:
        async with SharedWorldState(redis_url="redis://localhost:6379") as world:
            # Set world flag
            await world.set_flag("gate_opened", True, player_id="player_123")

            # Get all flags
            flags = await world.get_all_flags()

            # Broadcast NPC state change
            await world.broadcast_npc_state("innkeeper", {
                "emotion": "happy",
                "valence": 0.7
            })
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize shared world state.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None

    async def __aenter__(self):
        """Async context manager entry - connect to Redis"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - disconnect from Redis"""
        await self.disconnect()

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        self.pubsub = self.redis.pubsub()

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()

    # ===== World Flags =====

    async def set_flag(self, flag: str, value: bool, player_id: str = "system") -> WorldFlag:
        """
        Set world flag and broadcast to all players.

        Args:
            flag: Flag name (e.g., "gate_opened", "boss_defeated")
            value: Flag value (True/False)
            player_id: Player who set the flag

        Returns:
            WorldFlag object with metadata
        """
        flag_obj = WorldFlag(
            key=flag,
            value=value,
            set_by=player_id,
            timestamp=datetime.now()
        )

        # Store in Redis hash
        await self.redis.hset(
            "world:flags",
            flag,
            json.dumps(flag_obj.to_dict())
        )

        # Broadcast to all players
        await self.redis.publish("world:updates", json.dumps({
            "type": "flag_changed",
            "flag": flag,
            "value": value,
            "set_by": player_id,
            "timestamp": flag_obj.timestamp.isoformat()
        }))

        return flag_obj

    async def get_flag(self, flag: str) -> Optional[WorldFlag]:
        """
        Get single world flag.

        Args:
            flag: Flag name

        Returns:
            WorldFlag or None if not set
        """
        data = await self.redis.hget("world:flags", flag)
        if not data:
            return None

        flag_dict = json.loads(data)
        return WorldFlag(
            key=flag_dict["key"],
            value=flag_dict["value"],
            set_by=flag_dict["set_by"],
            timestamp=datetime.fromisoformat(flag_dict["timestamp"])
        )

    async def get_all_flags(self) -> Dict[str, WorldFlag]:
        """
        Get all world flags.

        Returns:
            Dictionary of flag_name -> WorldFlag
        """
        all_flags = await self.redis.hgetall("world:flags")
        return {
            key: WorldFlag(**json.loads(value))
            for key, value in all_flags.items()
        }

    async def delete_flag(self, flag: str) -> bool:
        """
        Delete world flag.

        Args:
            flag: Flag name

        Returns:
            True if deleted, False if didn't exist
        """
        deleted = await self.redis.hdel("world:flags", flag)

        if deleted:
            # Broadcast deletion
            await self.redis.publish("world:updates", json.dumps({
                "type": "flag_deleted",
                "flag": flag,
                "timestamp": datetime.now().isoformat()
            }))

        return bool(deleted)

    # ===== Active Players =====

    async def add_player(self, player_id: str) -> int:
        """
        Add player to active players set.

        Args:
            player_id: Player identifier

        Returns:
            Total number of active players
        """
        await self.redis.sadd("world:active_players", player_id)

        # Set last seen timestamp
        await self.redis.hset(
            "world:player_heartbeats",
            player_id,
            datetime.now().isoformat()
        )

        count = await self.redis.scard("world:active_players")

        # Broadcast player joined
        await self.redis.publish("world:updates", json.dumps({
            "type": "player_joined",
            "player_id": player_id,
            "total_players": count,
            "timestamp": datetime.now().isoformat()
        }))

        return count

    async def remove_player(self, player_id: str) -> int:
        """
        Remove player from active players set.

        Args:
            player_id: Player identifier

        Returns:
            Remaining number of active players
        """
        await self.redis.srem("world:active_players", player_id)
        await self.redis.hdel("world:player_heartbeats", player_id)

        count = await self.redis.scard("world:active_players")

        # Broadcast player left
        await self.redis.publish("world:updates", json.dumps({
            "type": "player_left",
            "player_id": player_id,
            "total_players": count,
            "timestamp": datetime.now().isoformat()
        }))

        return count

    async def get_active_players(self) -> Set[str]:
        """
        Get all active player IDs.

        Returns:
            Set of player IDs
        """
        return await self.redis.smembers("world:active_players")

    async def player_heartbeat(self, player_id: str):
        """
        Update player's last-seen timestamp.

        Args:
            player_id: Player identifier
        """
        await self.redis.hset(
            "world:player_heartbeats",
            player_id,
            datetime.now().isoformat()
        )

    async def cleanup_stale_players(self, timeout_seconds: int = 300):
        """
        Remove players who haven't sent heartbeat in timeout period.

        Args:
            timeout_seconds: Timeout in seconds (default: 5 minutes)
        """
        heartbeats = await self.redis.hgetall("world:player_heartbeats")
        now = datetime.now()

        for player_id, last_seen_str in heartbeats.items():
            last_seen = datetime.fromisoformat(last_seen_str)
            if (now - last_seen).total_seconds() > timeout_seconds:
                await self.remove_player(player_id)

    # ===== NPC State Broadcasting =====

    async def broadcast_npc_state(
        self,
        npc_id: str,
        update_type: str,
        data: Dict[str, Any]
    ) -> NPCStateUpdate:
        """
        Broadcast NPC state change to all players.

        Args:
            npc_id: NPC identifier
            update_type: Type of update ("emotion", "location", "quest", "conversation")
            data: Update data dictionary

        Returns:
            NPCStateUpdate object
        """
        update = NPCStateUpdate(
            npc_id=npc_id,
            update_type=update_type,
            data=data,
            timestamp=datetime.now()
        )

        # Publish to npc:updates channel
        await self.redis.publish("npc:updates", json.dumps(update.to_dict()))

        # Store latest state in hash
        await self.redis.hset(
            f"npc:state:{npc_id}",
            update_type,
            json.dumps({
                "data": data,
                "timestamp": update.timestamp.isoformat()
            })
        )

        return update

    async def get_npc_state(self, npc_id: str) -> Dict[str, Any]:
        """
        Get current NPC state.

        Args:
            npc_id: NPC identifier

        Returns:
            Dictionary of update_type -> {data, timestamp}
        """
        state = await self.redis.hgetall(f"npc:state:{npc_id}")
        return {
            key: json.loads(value)
            for key, value in state.items()
        }

    # ===== World Events =====

    async def broadcast_world_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        exclude_player: Optional[str] = None
    ):
        """
        Broadcast arbitrary world event to all players.

        Args:
            event_type: Event type (e.g., "quest_completed", "boss_spawned")
            data: Event data
            exclude_player: Optional player to exclude from broadcast
        """
        await self.redis.publish("world:events", json.dumps({
            "type": event_type,
            "data": data,
            "exclude_player": exclude_player,
            "timestamp": datetime.now().isoformat()
        }))

    # ===== Pub/Sub Subscription =====

    async def subscribe_to_updates(self) -> aioredis.client.PubSub:
        """
        Subscribe to world updates channel.

        Returns:
            PubSub object for listening to updates
        """
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("world:updates", "npc:updates", "world:events")
        return pubsub

    async def listen_for_updates(self, callback):
        """
        Listen for world updates and call callback function.

        Args:
            callback: Async function to call with each message
        """
        pubsub = await self.subscribe_to_updates()

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    await callback(message["channel"], data)
        finally:
            await pubsub.close()

    # ===== Statistics =====

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get world state statistics.

        Returns:
            Dictionary with counts and metrics
        """
        return {
            "active_players": await self.redis.scard("world:active_players"),
            "total_flags": await self.redis.hlen("world:flags"),
            "tracked_npcs": len(await self.redis.keys("npc:state:*")),
            "timestamp": datetime.now().isoformat()
        }

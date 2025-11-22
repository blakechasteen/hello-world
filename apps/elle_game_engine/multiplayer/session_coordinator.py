"""
SessionCoordinator - Multi-player session and NPC lock coordination

Prevents conflicting NPC interactions, coordinates multi-player quests,
and handles race conditions in shared world state.

Author: BigPlay Team
Created: 2025-11-17
"""

import json
import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
from redis import asyncio as aioredis


@dataclass
class NPCLock:
    """NPC conversation lock information"""
    npc_id: str
    player_id: str
    locked_at: datetime
    expires_at: datetime

    def to_dict(self) -> dict:
        return {
            "npc_id": self.npc_id,
            "player_id": self.player_id,
            "locked_at": self.locked_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }


@dataclass
class MultiplayerQuest:
    """Multi-player quest coordination"""
    quest_id: str
    required_players: int
    current_players: Set[str]
    progress: Dict[str, int]
    started_at: datetime

    def is_complete(self) -> bool:
        """Check if all players have completed"""
        return all(
            progress >= 100
            for progress in self.progress.values()
        )

    def to_dict(self) -> dict:
        return {
            "quest_id": self.quest_id,
            "required_players": self.required_players,
            "current_players": list(self.current_players),
            "progress": self.progress,
            "started_at": self.started_at.isoformat()
        }


class SessionCoordinator:
    """
    Coordinates multi-player sessions and prevents conflicts.

    Features:
    - NPC conversation locking (exclusive access)
    - Multi-player quest coordination
    - Race condition prevention
    - Automatic lock expiration

    Usage:
        async with SessionCoordinator(redis_url="redis://localhost:6379") as coord:
            # Acquire NPC lock
            if await coord.acquire_npc_lock("innkeeper", "player_123"):
                # Player has exclusive access to NPC
                ...
                await coord.release_npc_lock("innkeeper")
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize session coordinator.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()

    async def connect(self):
        """Connect to Redis"""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis:
            await self.redis.close()

    # ===== NPC Locking =====

    async def acquire_npc_lock(
        self,
        npc_id: str,
        player_id: str,
        timeout: int = 30
    ) -> bool:
        """
        Acquire exclusive lock on NPC for conversation.

        Args:
            npc_id: NPC identifier
            player_id: Player requesting lock
            timeout: Lock timeout in seconds (default: 30)

        Returns:
            True if lock acquired, False if NPC is busy
        """
        lock_key = f"npc:lock:{npc_id}"

        # Try to acquire lock (NX = only set if not exists, EX = expiration)
        acquired = await self.redis.set(
            lock_key,
            json.dumps({
                "player_id": player_id,
                "locked_at": datetime.now().isoformat()
            }),
            nx=True,  # Only set if not exists
            ex=timeout  # Expire after timeout
        )

        if acquired:
            # Publish lock acquired event
            await self.redis.publish("npc:locks", json.dumps({
                "type": "lock_acquired",
                "npc_id": npc_id,
                "player_id": player_id,
                "timeout": timeout,
                "timestamp": datetime.now().isoformat()
            }))

        return bool(acquired)

    async def release_npc_lock(self, npc_id: str, player_id: Optional[str] = None) -> bool:
        """
        Release NPC lock.

        Args:
            npc_id: NPC identifier
            player_id: Optional player ID (for verification)

        Returns:
            True if lock was released
        """
        lock_key = f"npc:lock:{npc_id}"

        # If player_id provided, verify ownership before releasing
        if player_id:
            lock_data = await self.redis.get(lock_key)
            if lock_data:
                lock_info = json.loads(lock_data)
                if lock_info.get("player_id") != player_id:
                    # Lock owned by different player
                    return False

        # Delete lock
        deleted = await self.redis.delete(lock_key)

        if deleted:
            # Publish lock released event
            await self.redis.publish("npc:locks", json.dumps({
                "type": "lock_released",
                "npc_id": npc_id,
                "player_id": player_id,
                "timestamp": datetime.now().isoformat()
            }))

        return bool(deleted)

    async def is_npc_available(self, npc_id: str) -> bool:
        """
        Check if NPC is available for interaction.

        Args:
            npc_id: NPC identifier

        Returns:
            True if NPC is not locked
        """
        lock_key = f"npc:lock:{npc_id}"
        return not await self.redis.exists(lock_key)

    async def get_npc_lock_info(self, npc_id: str) -> Optional[Dict]:
        """
        Get NPC lock information.

        Args:
            npc_id: NPC identifier

        Returns:
            Lock info dict or None if not locked
        """
        lock_key = f"npc:lock:{npc_id}"
        lock_data = await self.redis.get(lock_key)

        if not lock_data:
            return None

        lock_info = json.loads(lock_data)
        ttl = await self.redis.ttl(lock_key)

        return {
            **lock_info,
            "remaining_seconds": ttl
        }

    async def get_locked_npcs(self) -> List[str]:
        """
        Get list of all currently locked NPCs.

        Returns:
            List of NPC IDs
        """
        keys = await self.redis.keys("npc:lock:*")
        return [key.replace("npc:lock:", "") for key in keys]

    # ===== Multi-Player Quest Coordination =====

    async def create_multiplayer_quest(
        self,
        quest_id: str,
        player_ids: List[str],
        required_players: Optional[int] = None
    ) -> MultiplayerQuest:
        """
        Create multi-player quest coordination.

        Args:
            quest_id: Quest identifier
            player_ids: List of player IDs participating
            required_players: Required number of players (default: len(player_ids))

        Returns:
            MultiplayerQuest object
        """
        if required_players is None:
            required_players = len(player_ids)

        quest = MultiplayerQuest(
            quest_id=quest_id,
            required_players=required_players,
            current_players=set(player_ids),
            progress={player_id: 0 for player_id in player_ids},
            started_at=datetime.now()
        )

        # Store in Redis
        quest_key = f"quest:multi:{quest_id}"
        await self.redis.set(quest_key, json.dumps(quest.to_dict()))
        await self.redis.expire(quest_key, 3600)  # 1 hour TTL

        # Add to player quest sets
        for player_id in player_ids:
            await self.redis.sadd(f"player:quests:{player_id}", quest_id)

        return quest

    async def update_quest_progress(
        self,
        quest_id: str,
        player_id: str,
        progress: int
    ) -> Optional[MultiplayerQuest]:
        """
        Update player's progress on multi-player quest.

        Args:
            quest_id: Quest identifier
            player_id: Player identifier
            progress: Progress percentage (0-100)

        Returns:
            Updated MultiplayerQuest or None if quest not found
        """
        quest_key = f"quest:multi:{quest_id}"
        quest_data = await self.redis.get(quest_key)

        if not quest_data:
            return None

        quest_dict = json.loads(quest_data)
        quest_dict["progress"][player_id] = progress

        # Update in Redis
        await self.redis.set(quest_key, json.dumps(quest_dict))

        # Publish progress update
        await self.redis.publish("quest:updates", json.dumps({
            "type": "progress_updated",
            "quest_id": quest_id,
            "player_id": player_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }))

        # Reconstruct MultiplayerQuest
        quest = MultiplayerQuest(
            quest_id=quest_dict["quest_id"],
            required_players=quest_dict["required_players"],
            current_players=set(quest_dict["current_players"]),
            progress=quest_dict["progress"],
            started_at=datetime.fromisoformat(quest_dict["started_at"])
        )

        # Check if quest is complete
        if quest.is_complete():
            await self._complete_multiplayer_quest(quest_id)

        return quest

    async def _complete_multiplayer_quest(self, quest_id: str):
        """Internal: Handle multi-player quest completion"""
        # Publish completion event
        await self.redis.publish("quest:updates", json.dumps({
            "type": "quest_completed",
            "quest_id": quest_id,
            "timestamp": datetime.now().isoformat()
        }))

        # Clean up
        quest_key = f"quest:multi:{quest_id}"
        quest_data = await self.redis.get(quest_key)

        if quest_data:
            quest_dict = json.loads(quest_data)
            for player_id in quest_dict["current_players"]:
                await self.redis.srem(f"player:quests:{player_id}", quest_id)

        await self.redis.delete(quest_key)

    async def get_multiplayer_quest(self, quest_id: str) -> Optional[MultiplayerQuest]:
        """
        Get multi-player quest information.

        Args:
            quest_id: Quest identifier

        Returns:
            MultiplayerQuest or None if not found
        """
        quest_key = f"quest:multi:{quest_id}"
        quest_data = await self.redis.get(quest_key)

        if not quest_data:
            return None

        quest_dict = json.loads(quest_data)
        return MultiplayerQuest(
            quest_id=quest_dict["quest_id"],
            required_players=quest_dict["required_players"],
            current_players=set(quest_dict["current_players"]),
            progress=quest_dict["progress"],
            started_at=datetime.fromisoformat(quest_dict["started_at"])
        )

    async def get_player_quests(self, player_id: str) -> List[str]:
        """
        Get all multi-player quests for a player.

        Args:
            player_id: Player identifier

        Returns:
            List of quest IDs
        """
        return list(await self.redis.smembers(f"player:quests:{player_id}"))

    # ===== Race Condition Prevention =====

    async def atomic_increment(self, key: str, amount: int = 1) -> int:
        """
        Atomically increment a counter.

        Args:
            key: Redis key
            amount: Amount to increment

        Returns:
            New value after increment
        """
        return await self.redis.incrby(key, amount)

    async def atomic_flag_set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Atomically set a flag if not already set.

        Args:
            key: Redis key
            value: Value to set
            ttl: Optional TTL in seconds

        Returns:
            True if set, False if already existed
        """
        if ttl:
            return bool(await self.redis.set(key, value, nx=True, ex=ttl))
        else:
            return bool(await self.redis.set(key, value, nx=True))

    async def compare_and_swap(
        self,
        key: str,
        expected_value: str,
        new_value: str
    ) -> bool:
        """
        Compare-and-swap operation using Lua script.

        Args:
            key: Redis key
            expected_value: Expected current value
            new_value: New value to set

        Returns:
            True if swap successful
        """
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            redis.call("set", KEYS[1], ARGV[2])
            return 1
        else
            return 0
        end
        """

        result = await self.redis.eval(
            lua_script,
            1,  # Number of keys
            key,
            expected_value,
            new_value
        )

        return bool(result)

    # ===== Cleanup =====

    async def cleanup_expired_locks(self):
        """
        Clean up any orphaned locks (should be automatic via TTL).
        This is a safety check.
        """
        locked_npcs = await self.get_locked_npcs()
        for npc_id in locked_npcs:
            ttl = await self.redis.ttl(f"npc:lock:{npc_id}")
            if ttl < 0:  # No expiration set (shouldn't happen)
                await self.release_npc_lock(npc_id)

    # ===== Statistics =====

    async def get_statistics(self) -> Dict:
        """
        Get session coordinator statistics.

        Returns:
            Dictionary with coordination stats
        """
        locked_npcs = await self.get_locked_npcs()
        multi_quests = len(await self.redis.keys("quest:multi:*"))

        return {
            "locked_npcs": len(locked_npcs),
            "locked_npc_ids": locked_npcs,
            "active_multiplayer_quests": multi_quests,
            "timestamp": datetime.now().isoformat()
        }

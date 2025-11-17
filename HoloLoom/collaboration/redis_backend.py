"""
Redis Backend for Distributed Collaboration
============================================
Redis integration for cross-server state synchronization.

Philosophy:
When multiple loom servers operate in parallel, they must share a common
memory. Redis provides the distributed warp space - a shared tensor field
where all servers can read and write coordination data.

Features:
- Session state persistence
- Pub/Sub message broadcasting
- Presence data caching
- Distributed locking
- Event streaming
"""

import logging
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try Redis import with graceful fallback
try:
    import redis.asyncio as aioredis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("redis not installed. Distributed collaboration unavailable.")
    REDIS_AVAILABLE = False
    Redis = None


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    key_prefix: str = "hololoom:"
    session_ttl: int = 86400  # 24 hours
    presence_ttl: int = 300   # 5 minutes


class RedisBackend:
    """
    Redis backend for distributed collaboration.

    Provides:
    - Session state storage
    - Pub/Sub messaging
    - Presence tracking
    - Distributed locks
    - Event streaming
    """

    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis backend.

        Args:
            config: Redis configuration
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError("redis not installed: pip install redis")

        self.config = config or RedisConfig()
        self.redis: Optional[Redis] = None
        self.pubsub = None
        self.subscriptions: Dict[str, Callable] = {}
        self._listener_task: Optional[asyncio.Task] = None

        logger.info(f"RedisBackend initialized: {self.config.host}:{self.config.port}")

    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(
            f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
            password=self.config.password,
            encoding="utf-8",
            decode_responses=True
        )

        logger.info("Connected to Redis")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._listener_task:
            self._listener_task.cancel()

        if self.pubsub:
            await self.pubsub.close()

        if self.redis:
            await self.redis.close()

        logger.info("Disconnected from Redis")

    def _make_key(self, *parts: str) -> str:
        """Create prefixed key."""
        return self.config.key_prefix + ":".join(parts)

    # ========================================================================
    # Session State Management
    # ========================================================================

    async def save_session(self, session_id: str, session_data: Dict[str, Any]):
        """
        Save session state to Redis.

        Args:
            session_id: Session ID
            session_data: Session data dictionary
        """
        key = self._make_key("session", session_id)
        data = json.dumps(session_data)

        await self.redis.setex(
            key,
            self.config.session_ttl,
            data
        )

        logger.debug(f"Saved session {session_id} to Redis")

    async def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session state from Redis.

        Args:
            session_id: Session ID

        Returns:
            Session data or None if not found
        """
        key = self._make_key("session", session_id)
        data = await self.redis.get(key)

        if data:
            logger.debug(f"Loaded session {session_id} from Redis")
            return json.loads(data)

        return None

    async def delete_session(self, session_id: str):
        """Delete session from Redis."""
        key = self._make_key("session", session_id)
        await self.redis.delete(key)

        logger.debug(f"Deleted session {session_id} from Redis")

    async def list_sessions(self) -> List[str]:
        """List all session IDs."""
        pattern = self._make_key("session", "*")
        keys = await self.redis.keys(pattern)

        # Extract session IDs from keys
        prefix = self._make_key("session", "")
        session_ids = [key.replace(prefix, "") for key in keys]

        return session_ids

    # ========================================================================
    # Presence Management
    # ========================================================================

    async def set_presence(
        self,
        session_id: str,
        user_id: str,
        presence_data: Dict[str, Any]
    ):
        """
        Set user presence in session.

        Args:
            session_id: Session ID
            user_id: User ID
            presence_data: Presence data
        """
        key = self._make_key("presence", session_id, user_id)
        data = json.dumps(presence_data)

        await self.redis.setex(
            key,
            self.config.presence_ttl,
            data
        )

        logger.debug(f"Set presence for {user_id} in session {session_id}")

    async def get_presence(
        self,
        session_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get user presence."""
        key = self._make_key("presence", session_id, user_id)
        data = await self.redis.get(key)

        if data:
            return json.loads(data)

        return None

    async def get_session_presence(self, session_id: str) -> Dict[str, Dict[str, Any]]:
        """Get all presence data for session."""
        pattern = self._make_key("presence", session_id, "*")
        keys = await self.redis.keys(pattern)

        presence = {}
        prefix = self._make_key("presence", session_id, "")

        for key in keys:
            user_id = key.replace(prefix, "")
            data = await self.redis.get(key)

            if data:
                presence[user_id] = json.loads(data)

        return presence

    async def remove_presence(self, session_id: str, user_id: str):
        """Remove user presence."""
        key = self._make_key("presence", session_id, user_id)
        await self.redis.delete(key)

        logger.debug(f"Removed presence for {user_id} in session {session_id}")

    # ========================================================================
    # Pub/Sub Messaging
    # ========================================================================

    async def publish(self, channel: str, message: Dict[str, Any]):
        """
        Publish message to channel.

        Args:
            channel: Channel name
            message: Message data
        """
        channel_key = self._make_key("channel", channel)
        data = json.dumps(message)

        await self.redis.publish(channel_key, data)

        logger.debug(f"Published to channel {channel}")

    async def subscribe(self, channel: str, callback: Callable):
        """
        Subscribe to channel.

        Args:
            channel: Channel name
            callback: Async callback function for messages
        """
        channel_key = self._make_key("channel", channel)

        if not self.pubsub:
            self.pubsub = self.redis.pubsub()

        await self.pubsub.subscribe(channel_key)
        self.subscriptions[channel] = callback

        # Start listener if not already running
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

        logger.info(f"Subscribed to channel {channel}")

    async def unsubscribe(self, channel: str):
        """Unsubscribe from channel."""
        channel_key = self._make_key("channel", channel)

        if self.pubsub:
            await self.pubsub.unsubscribe(channel_key)

        if channel in self.subscriptions:
            del self.subscriptions[channel]

        logger.info(f"Unsubscribed from channel {channel}")

    async def _listen(self):
        """Listen for pub/sub messages."""
        try:
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    channel_key = message["channel"]
                    data = json.loads(message["data"])

                    # Extract channel name
                    prefix = self._make_key("channel", "")
                    channel = channel_key.replace(prefix, "")

                    # Call callback
                    if channel in self.subscriptions:
                        callback = self.subscriptions[channel]
                        await callback(data)

        except asyncio.CancelledError:
            logger.info("Pub/sub listener cancelled")
        except Exception as e:
            logger.error(f"Pub/sub listener error: {e}")

    # ========================================================================
    # Distributed Locking
    # ========================================================================

    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking_timeout: Optional[int] = None
    ) -> bool:
        """
        Acquire distributed lock.

        Args:
            lock_name: Lock name
            timeout: Lock timeout in seconds
            blocking_timeout: How long to wait for lock (None = don't wait)

        Returns:
            True if lock acquired
        """
        key = self._make_key("lock", lock_name)

        # Try to acquire
        acquired = await self.redis.set(
            key,
            "locked",
            nx=True,  # Only set if doesn't exist
            ex=timeout
        )

        if acquired:
            logger.debug(f"Acquired lock: {lock_name}")
            return True

        # If blocking, wait and retry
        if blocking_timeout is not None:
            start = datetime.now(timezone.utc)

            while True:
                elapsed = (datetime.now(timezone.utc) - start).total_seconds()

                if elapsed >= blocking_timeout:
                    logger.debug(f"Lock timeout: {lock_name}")
                    return False

                await asyncio.sleep(0.1)

                acquired = await self.redis.set(
                    key,
                    "locked",
                    nx=True,
                    ex=timeout
                )

                if acquired:
                    logger.debug(f"Acquired lock after {elapsed:.2f}s: {lock_name}")
                    return True

        return False

    async def release_lock(self, lock_name: str):
        """Release distributed lock."""
        key = self._make_key("lock", lock_name)
        await self.redis.delete(key)

        logger.debug(f"Released lock: {lock_name}")

    # ========================================================================
    # Event Streaming
    # ========================================================================

    async def append_event(
        self,
        stream_name: str,
        event: Dict[str, Any],
        max_len: int = 10000
    ):
        """
        Append event to stream.

        Args:
            stream_name: Stream name
            event: Event data
            max_len: Maximum stream length
        """
        key = self._make_key("stream", stream_name)

        await self.redis.xadd(
            key,
            event,
            maxlen=max_len
        )

        logger.debug(f"Appended event to stream {stream_name}")

    async def read_events(
        self,
        stream_name: str,
        count: int = 100,
        last_id: str = "0"
    ) -> List[Dict[str, Any]]:
        """
        Read events from stream.

        Args:
            stream_name: Stream name
            count: Maximum events to read
            last_id: Read events after this ID

        Returns:
            List of events
        """
        key = self._make_key("stream", stream_name)

        results = await self.redis.xread(
            {key: last_id},
            count=count
        )

        events = []

        if results:
            for stream, messages in results:
                for msg_id, msg_data in messages:
                    events.append({
                        "id": msg_id,
                        "data": msg_data
                    })

        return events

    # ========================================================================
    # Utility
    # ========================================================================

    async def ping(self) -> bool:
        """Ping Redis to check connection."""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis stats."""
        info = await self.redis.info()

        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "0"),
            "total_commands_processed": info.get("total_commands_processed", 0),
            "uptime_in_seconds": info.get("uptime_in_seconds", 0)
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    async def demo():
        print("="*80)
        print("Redis Backend for Distributed Collaboration")
        print("="*80 + "\n")

        if not REDIS_AVAILABLE:
            print("Redis not installed: pip install redis")
            return

        # Create backend
        print("1. Connecting to Redis...")
        backend = RedisBackend()

        try:
            await backend.connect()
            print("   ✓ Connected")

            # Test ping
            print("\n2. Testing connection...")
            if await backend.ping():
                print("   ✓ Ping successful")

            # Save session
            print("\n3. Saving session...")
            session_data = {
                "name": "Test Session",
                "users": ["alice", "bob"],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            await backend.save_session("test-session", session_data)
            print("   ✓ Session saved")

            # Load session
            print("\n4. Loading session...")
            loaded = await backend.load_session("test-session")
            print(f"   ✓ Loaded: {loaded['name']}")

            # Set presence
            print("\n5. Setting presence...")
            presence_data = {
                "status": "online",
                "username": "Alice",
                "last_activity": datetime.now(timezone.utc).isoformat()
            }
            await backend.set_presence("test-session", "alice", presence_data)
            print("   ✓ Presence set")

            # Get presence
            print("\n6. Getting presence...")
            presence = await backend.get_session_presence("test-session")
            print(f"   ✓ Found {len(presence)} users")

            # Pub/Sub
            print("\n7. Testing Pub/Sub...")
            received_messages = []

            async def on_message(data):
                received_messages.append(data)
                print(f"   • Received: {data['text']}")

            await backend.subscribe("test-channel", on_message)

            # Give listener time to start
            await asyncio.sleep(0.1)

            await backend.publish("test-channel", {"text": "Hello, World!"})
            await backend.publish("test-channel", {"text": "Second message"})

            # Wait for messages
            await asyncio.sleep(0.5)
            print(f"   ✓ Received {len(received_messages)} messages")

            # Locking
            print("\n8. Testing distributed locks...")
            acquired = await backend.acquire_lock("test-lock", timeout=5)
            print(f"   ✓ Lock acquired: {acquired}")

            if acquired:
                await backend.release_lock("test-lock")
                print("   ✓ Lock released")

            # Stats
            print("\n9. Redis stats:")
            stats = await backend.get_stats()
            print(f"   • Clients: {stats['connected_clients']}")
            print(f"   • Memory: {stats['used_memory_human']}")
            print(f"   • Uptime: {stats['uptime_in_seconds']}s")

            # Cleanup
            print("\n10. Cleaning up...")
            await backend.delete_session("test-session")
            await backend.remove_presence("test-session", "alice")
            print("   ✓ Cleaned up")

        finally:
            await backend.disconnect()
            print("\n✓ Redis backend ready!")

    asyncio.run(demo())

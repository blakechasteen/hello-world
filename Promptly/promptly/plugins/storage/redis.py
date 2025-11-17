#!/usr/bin/env python3
"""
Redis Storage Backend Plugin

Production-grade Redis backend using redis-py with:
- In-memory storage for ultra-fast access
- Optional persistence (RDB snapshots + AOF)
- Pub/sub for collaborative editing notifications
- TTL support for temporary prompts
- Sorted sets for version ordering
- Redis Streams for audit logging
- Comprehensive error handling
"""

import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

try:
    import redis
    from redis.connection import ConnectionPool
    from redis.exceptions import (
        RedisError, ConnectionError as RedisConnectionError,
        TimeoutError, ResponseError
    )
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis-py not available. Install with: pip install redis")

from ..base import BaseStorageBackend


class RedisStorage(BaseStorageBackend):
    """
    Redis storage backend with high-performance features

    Features:
    - In-memory storage (sub-millisecond access)
    - Optional persistence (RDB + AOF)
    - Pub/sub for real-time notifications
    - TTL for temporary/ephemeral prompts
    - Sorted sets for efficient version queries
    - Pipeline support for batch operations
    - Redis Streams for audit trail
    - Clustering support (via redis-py-cluster)

    Connection string formats:
        redis://localhost:6379/0
        redis://:password@localhost:6379/0
        rediss://localhost:6380/0  (SSL/TLS)
        redis://localhost:6379/0?socket_timeout=5&socket_connect_timeout=5

    Example:
        storage = RedisStorage()
        storage.init_storage('redis://localhost:6379/0')

    Redis Key Structure:
        prompt:{name}:{branch}:{version}       - Individual prompt
        prompt:{name}:{branch}:latest          - Latest version pointer
        prompt:versions:{name}:{branch}        - Sorted set of versions
        branch:{name}                          - Branch metadata
        branch:list                            - Set of all branches
        evaluation:{prompt_name}:{timestamp}   - Evaluation results
        chain:{name}                           - Chain definition
        config:{key}                           - Configuration
    """

    def __init__(self, max_connections: int = 50, socket_timeout: int = 5,
                 socket_connect_timeout: int = 5, decode_responses: bool = True,
                 ttl_seconds: Optional[int] = None):
        """
        Initialize Redis storage backend

        Args:
            max_connections: Maximum connections in the pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            decode_responses: Decode responses from bytes to strings
            ttl_seconds: Default TTL for prompts (None = no expiration)
        """
        super().__init__(
            name="redis",
            description="Redis in-memory storage with pub/sub and persistence support"
        )

        if not REDIS_AVAILABLE:
            raise ImportError("redis-py required for Redis backend. Install: pip install redis")

        self.client: Optional[redis.Redis] = None
        self.pool: Optional[ConnectionPool] = None
        self.pubsub = None

        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.decode_responses = decode_responses
        self.default_ttl = ttl_seconds

    def init_storage(self, storage_path: str) -> None:
        """
        Initialize Redis connection

        Args:
            storage_path: Redis connection string (e.g., redis://localhost:6379/0)
        """
        try:
            # Create connection pool
            self.pool = ConnectionPool.from_url(
                storage_path,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                decode_responses=self.decode_responses
            )

            # Create Redis client
            self.client = redis.Redis(connection_pool=self.pool)

            # Test connection
            self.client.ping()

            # Initialize pub/sub
            self.pubsub = self.client.pubsub()

            # Initialize default data
            self._initialize_defaults()

        except (RedisConnectionError, TimeoutError) as e:
            raise ConnectionError(f"Could not connect to Redis: {e}")

    def _initialize_defaults(self) -> None:
        """Initialize default data (main branch, config)"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        # Create main branch if not exists
        if not self.client.exists('branch:main'):
            branch_data = {
                'name': 'main',
                'head_commit': 'init',
                'created_at': datetime.utcnow().isoformat(),
                'description': 'Main branch'
            }
            self.client.hset('branch:main', mapping=branch_data)
            self.client.sadd('branch:list', 'main')

        # Set current branch config
        if not self.client.exists('config:current_branch'):
            self.client.set('config:current_branch', 'main')

    def _generate_commit_hash(self, name: str, content: str, timestamp: str) -> str:
        """Generate a unique commit hash"""
        data = f"{name}:{content}:{timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def _prompt_key(self, name: str, branch: str, version: int) -> str:
        """Generate Redis key for a prompt"""
        return f"prompt:{name}:{branch}:{version}"

    def _latest_key(self, name: str, branch: str) -> str:
        """Generate Redis key for latest version pointer"""
        return f"prompt:{name}:{branch}:latest"

    def _versions_key(self, name: str, branch: str) -> str:
        """Generate Redis key for versions sorted set"""
        return f"prompt:versions:{name}:{branch}"

    def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish event to Redis pub/sub channel"""
        if not self.client:
            return

        channel = f"promptly:events:{event_type}"
        message = json.dumps(data)
        self.client.publish(channel, message)

    def save_prompt(self, prompt_data: Dict[str, Any]) -> str:
        """
        Save a prompt to Redis with optional TTL

        Args:
            prompt_data: Dictionary containing prompt information

        Returns:
            str: Commit hash
        """
        if not self.client:
            raise RuntimeError("Storage not initialized")

        name = prompt_data['name']
        content = prompt_data['content']
        branch = prompt_data.get('branch', 'main')
        metadata = prompt_data.get('metadata', {})
        ttl = prompt_data.get('ttl', self.default_ttl)

        timestamp = datetime.utcnow()
        commit_hash = self._generate_commit_hash(name, content, timestamp.isoformat())

        # Get latest version
        versions_key = self._versions_key(name, branch)
        latest_version = self.client.zrevrange(versions_key, 0, 0, withscores=True)
        version = int(latest_version[0][1]) + 1 if latest_version else 1

        # Create prompt data
        prompt = {
            'name': name,
            'content': content,
            'branch': branch,
            'version': version,
            'commit_hash': commit_hash,
            'created_at': timestamp.isoformat(),
            'metadata': json.dumps(metadata)
        }

        # Use pipeline for atomic operations
        pipe = self.client.pipeline()

        # Save prompt
        prompt_key = self._prompt_key(name, branch, version)
        pipe.hset(prompt_key, mapping=prompt)

        # Set TTL if specified
        if ttl:
            pipe.expire(prompt_key, ttl)

        # Update latest pointer
        latest_key = self._latest_key(name, branch)
        pipe.set(latest_key, version)
        if ttl:
            pipe.expire(latest_key, ttl)

        # Add to versions sorted set
        pipe.zadd(versions_key, {commit_hash: version})
        if ttl:
            pipe.expire(versions_key, ttl)

        # Update branch head
        pipe.hset(f'branch:{branch}', 'head_commit', commit_hash)

        # Add to commit history stream
        pipe.xadd(
            f'commits:{branch}',
            {
                'name': name,
                'commit_hash': commit_hash,
                'version': version,
                'timestamp': timestamp.isoformat()
            },
            maxlen=1000  # Keep last 1000 commits
        )

        # Execute pipeline
        pipe.execute()

        # Publish event
        self._publish_event('prompt_saved', {
            'name': name,
            'branch': branch,
            'version': version,
            'commit_hash': commit_hash
        })

        return commit_hash

    def get_prompt(self, name: str, branch: str = "main", version: Optional[int] = None,
                   commit_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt from Redis"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        if commit_hash:
            # Find by commit hash
            versions_key = self._versions_key(name, branch)
            result = self.client.zrangebyscore(versions_key, '-inf', '+inf')
            for hash_val in result:
                if hash_val == commit_hash:
                    version_score = self.client.zscore(versions_key, hash_val)
                    prompt_key = self._prompt_key(name, branch, int(version_score))
                    break
            else:
                return None
        elif version:
            prompt_key = self._prompt_key(name, branch, version)
        else:
            # Get latest version
            latest_key = self._latest_key(name, branch)
            latest_version = self.client.get(latest_key)
            if not latest_version:
                return None
            prompt_key = self._prompt_key(name, branch, int(latest_version))

        # Fetch prompt data
        prompt = self.client.hgetall(prompt_key)

        if not prompt:
            return None

        return {
            'name': prompt['name'],
            'content': prompt['content'],
            'branch': prompt['branch'],
            'version': int(prompt['version']),
            'commit_hash': prompt['commit_hash'],
            'created_at': prompt['created_at'],
            'metadata': json.loads(prompt.get('metadata', '{}'))
        }

    def list_prompts(self, branch: str = "main") -> List[Dict[str, Any]]:
        """List all prompts on a branch"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        # Scan for all prompts on this branch
        prompts = []
        cursor = '0'
        pattern = f"prompt:*:{branch}:latest"

        while cursor != 0:
            cursor, keys = self.client.scan(cursor, match=pattern, count=100)
            cursor = int(cursor) if isinstance(cursor, str) else cursor

            for key in keys:
                # Extract name from key: prompt:{name}:{branch}:latest
                parts = key.split(':')
                if len(parts) >= 4:
                    name = ':'.join(parts[1:-2])  # Handle names with colons
                    prompt = self.get_prompt(name, branch)
                    if prompt:
                        prompts.append({
                            'name': prompt['name'],
                            'version': prompt['version'],
                            'commit_hash': prompt['commit_hash'],
                            'created_at': prompt['created_at']
                        })

        return sorted(prompts, key=lambda x: x['name'])

    def create_branch(self, branch_name: str, from_branch: str = "main") -> None:
        """Create a new branch"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        # Check source branch exists
        if not self.client.exists(f'branch:{from_branch}'):
            raise Exception(f"Branch '{from_branch}' does not exist")

        # Check if target branch already exists
        if self.client.exists(f'branch:{branch_name}'):
            raise Exception(f"Branch '{branch_name}' already exists")

        # Get source branch data
        source_data = self.client.hgetall(f'branch:{from_branch}')

        # Create new branch
        branch_data = {
            'name': branch_name,
            'head_commit': source_data['head_commit'],
            'created_at': datetime.utcnow().isoformat(),
            'description': f"Branched from {from_branch}"
        }

        pipe = self.client.pipeline()
        pipe.hset(f'branch:{branch_name}', mapping=branch_data)
        pipe.sadd('branch:list', branch_name)

        # Copy latest prompts from source branch
        cursor = '0'
        pattern = f"prompt:*:{from_branch}:latest"

        while cursor != 0:
            cursor, keys = self.client.scan(cursor, match=pattern, count=100)
            cursor = int(cursor) if isinstance(cursor, str) else cursor

            for key in keys:
                parts = key.split(':')
                name = ':'.join(parts[1:-2])

                # Get latest version from source
                prompt = self.get_prompt(name, from_branch)
                if prompt:
                    # Save to new branch
                    new_prompt_data = {
                        'name': name,
                        'content': prompt['content'],
                        'branch': branch_name,
                        'metadata': prompt.get('metadata', {})
                    }
                    # This will be executed after pipe.execute()
                    # We need to do this separately to avoid nested pipelines
                    pass

        pipe.execute()

        # Now copy prompts (can't use save_prompt in pipeline)
        cursor = '0'
        pattern = f"prompt:*:{from_branch}:latest"
        while cursor != 0:
            cursor, keys = self.client.scan(cursor, match=pattern, count=100)
            cursor = int(cursor) if isinstance(cursor, str) else cursor

            for key in keys:
                parts = key.split(':')
                name = ':'.join(parts[1:-2])
                prompt = self.get_prompt(name, from_branch)
                if prompt:
                    self.save_prompt({
                        'name': name,
                        'content': prompt['content'],
                        'branch': branch_name,
                        'metadata': prompt.get('metadata', {})
                    })

        # Publish event
        self._publish_event('branch_created', {
            'branch': branch_name,
            'from': from_branch
        })

    def get_current_branch(self) -> str:
        """Get the current branch name"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        branch = self.client.get('config:current_branch')
        return branch if branch else 'main'

    def set_current_branch(self, branch_name: str) -> None:
        """Set the current branch"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        # Verify branch exists
        if not self.client.exists(f'branch:{branch_name}'):
            raise Exception(f"Branch '{branch_name}' does not exist")

        self.client.set('config:current_branch', branch_name)

    def get_commit_history(self, name: Optional[str] = None, branch: str = "main",
                          limit: int = 10) -> List[Dict[str, Any]]:
        """Get commit history from Redis Streams"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        # Read from commit stream
        stream_key = f'commits:{branch}'
        entries = self.client.xrevrange(stream_key, count=limit)

        commits = []
        for entry_id, data in entries:
            if name is None or data.get('name') == name:
                commits.append({
                    'commit_hash': data['commit_hash'],
                    'name': data['name'],
                    'version': int(data['version']),
                    'branch': branch,
                    'created_at': data['timestamp']
                })

        return commits[:limit]

    def save_evaluation(self, eval_data: Dict[str, Any]) -> None:
        """Save evaluation results"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        timestamp = datetime.utcnow().isoformat().replace(':', '-')
        eval_key = f"evaluation:{eval_data['prompt_name']}:{timestamp}"

        eval_doc = {
            'prompt_name': eval_data['prompt_name'],
            'commit_hash': eval_data['commit_hash'],
            'test_case': json.dumps(eval_data.get('test_case', {})),
            'expected': eval_data.get('expected', ''),
            'actual': eval_data.get('actual', ''),
            'score': str(eval_data.get('score', 0.0)),
            'metrics': json.dumps(eval_data.get('metrics', {})),
            'evaluator_name': eval_data.get('evaluator_name', ''),
            'created_at': datetime.utcnow().isoformat()
        }

        self.client.hset(eval_key, mapping=eval_doc)

        # Add to evaluations index
        self.client.zadd(
            f"evaluations:{eval_data['prompt_name']}",
            {eval_key: datetime.utcnow().timestamp()}
        )

    def save_chain(self, chain_data: Dict[str, Any]) -> None:
        """Save a prompt chain"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        chain_key = f"chain:{chain_data['name']}"

        if self.client.exists(chain_key):
            raise Exception(f"Chain '{chain_data['name']}' already exists")

        chain_doc = {
            'name': chain_data['name'],
            'steps': json.dumps(chain_data['steps']),
            'description': chain_data.get('description', ''),
            'created_at': datetime.utcnow().isoformat()
        }

        self.client.hset(chain_key, mapping=chain_doc)
        self.client.sadd('chain:list', chain_data['name'])

    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a prompt chain"""
        if not self.client:
            raise RuntimeError("Storage not initialized")

        chain_key = f"chain:{name}"
        chain = self.client.hgetall(chain_key)

        if not chain:
            return None

        return {
            'name': chain['name'],
            'steps': json.loads(chain['steps']),
            'description': chain.get('description', '')
        }

    def close(self) -> None:
        """Close the Redis connection"""
        if self.pubsub:
            self.pubsub.close()
            self.pubsub = None

        if self.client:
            self.client.close()
            self.client = None

        if self.pool:
            self.pool.disconnect()
            self.pool = None

    # Additional Redis-specific methods

    def subscribe_to_events(self, event_type: str, callback) -> None:
        """
        Subscribe to real-time events via pub/sub

        Args:
            event_type: Event type (e.g., 'prompt_saved', 'branch_created')
            callback: Function to call when event is received
        """
        if not self.pubsub:
            raise RuntimeError("Storage not initialized")

        channel = f"promptly:events:{event_type}"
        self.pubsub.subscribe(**{channel: callback})

    def set_prompt_ttl(self, name: str, branch: str, ttl_seconds: int) -> None:
        """
        Set TTL for a prompt (useful for temporary/ephemeral prompts)

        Args:
            name: Prompt name
            branch: Branch name
            ttl_seconds: Time to live in seconds
        """
        if not self.client:
            raise RuntimeError("Storage not initialized")

        latest_key = self._latest_key(name, branch)
        latest_version = self.client.get(latest_key)

        if latest_version:
            prompt_key = self._prompt_key(name, branch, int(latest_version))
            self.client.expire(prompt_key, ttl_seconds)
            self.client.expire(latest_key, ttl_seconds)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Redis statistics

        Returns:
            Dictionary with statistics
        """
        if not self.client:
            raise RuntimeError("Storage not initialized")

        info = self.client.info()
        memory_info = self.client.info('memory')

        # Count keys by pattern
        prompt_count = 0
        cursor = '0'
        while cursor != 0:
            cursor, keys = self.client.scan(cursor, match="prompt:*:*:latest", count=100)
            cursor = int(cursor) if isinstance(cursor, str) else cursor
            prompt_count += len(keys)

        branch_count = self.client.scard('branch:list')
        chain_count = self.client.scard('chain:list')

        return {
            'total_prompts': prompt_count,
            'total_branches': branch_count,
            'total_chains': chain_count,
            'redis_version': info.get('redis_version'),
            'connected_clients': info.get('connected_clients'),
            'used_memory_human': memory_info.get('used_memory_human'),
            'used_memory_peak_human': memory_info.get('used_memory_peak_human'),
            'uptime_days': info.get('uptime_in_days')
        }

    def backup_to_rdb(self) -> None:
        """
        Trigger a background RDB snapshot save

        Note: Requires appropriate Redis permissions
        """
        if not self.client:
            raise RuntimeError("Storage not initialized")

        try:
            self.client.bgsave()
        except ResponseError as e:
            print(f"Warning: Could not trigger RDB save: {e}")

    def flush_database(self, confirm: bool = False) -> None:
        """
        Flush entire Redis database (DANGEROUS!)

        Args:
            confirm: Must be True to execute
        """
        if not confirm:
            raise ValueError("Must set confirm=True to flush database")

        if not self.client:
            raise RuntimeError("Storage not initialized")

        self.client.flushdb()

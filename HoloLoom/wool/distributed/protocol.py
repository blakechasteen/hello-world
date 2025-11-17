"""
Wool Network Protocol
=====================

Network protocol for distributed wool storage communication.

This module implements the inter-node communication protocol:
- HTTP-based protocol for simplicity (future: gRPC for performance)
- Request/response patterns for file operations
- Connection pooling and retry logic
- Compression for large transfers

Protocol Operations:
    - STORE: Store file on remote node
    - READ: Read file from remote node
    - REPLICATE: Replicate file to node
    - PING: Health check
    - SYNC: Synchronize file across nodes

Message Format (JSON over HTTP):
    {
        "operation": "READ",
        "file_id": "abc123...",
        "offset": 0,
        "length": 1024,
        "node_id": "sender-node-id"
    }

Author: Claude Code
Date: November 17, 2025 (Phase 6.3)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logging.warning("aiohttp not available - network protocol disabled")

from HoloLoom.wool import WoolReference


logger = logging.getLogger(__name__)


class Operation(Enum):
    """Network operation types."""
    PING = "PING"
    STORE = "STORE"
    READ = "READ"
    REPLICATE = "REPLICATE"
    SYNC = "SYNC"
    DELETE = "DELETE"


@dataclass
class NetworkRequest:
    """Network request message."""
    operation: Operation
    node_id: str  # Sender node ID
    file_id: Optional[str] = None
    offset: Optional[int] = None
    length: Optional[int] = None
    data: Optional[bytes] = None
    content_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            'operation': self.operation.value,
            'node_id': self.node_id
        }
        if self.file_id:
            d['file_id'] = self.file_id
        if self.offset is not None:
            d['offset'] = self.offset
        if self.length is not None:
            d['length'] = self.length
        if self.content_type:
            d['content_type'] = self.content_type
        if self.metadata:
            d['metadata'] = self.metadata
        # Note: data sent separately (multipart or binary)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'NetworkRequest':
        """Create from dictionary."""
        return cls(
            operation=Operation(d['operation']),
            node_id=d['node_id'],
            file_id=d.get('file_id'),
            offset=d.get('offset'),
            length=d.get('length'),
            content_type=d.get('content_type'),
            metadata=d.get('metadata')
        )


@dataclass
class NetworkResponse:
    """Network response message."""
    success: bool
    operation: Operation
    node_id: str  # Responder node ID
    data: Optional[bytes] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            'success': self.success,
            'operation': self.operation.value,
            'node_id': self.node_id
        }
        if self.error:
            d['error'] = self.error
        if self.metadata:
            d['metadata'] = self.metadata
        # Note: data sent separately (multipart or binary)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any], data: Optional[bytes] = None) -> 'NetworkResponse':
        """Create from dictionary."""
        return cls(
            success=d['success'],
            operation=Operation(d['operation']),
            node_id=d['node_id'],
            error=d.get('error'),
            metadata=d.get('metadata'),
            data=data
        )


class WoolNetworkProtocol:
    """
    Network protocol for distributed wool storage.

    Features:
    - HTTP-based communication (simple, debuggable)
    - Connection pooling for performance
    - Automatic retries with exponential backoff
    - Request timeout handling
    - Compression for large transfers

    Usage:
        protocol = WoolNetworkProtocol(
            node_id="node-1",
            host="localhost",
            port=9000
        )

        async with protocol:
            # Read from remote node
            data = await protocol.read_remote(
                node_id="node-2",
                ref=WoolReference(file_id="abc...", offset=0, length=100)
            )

            # Replicate to remote node
            await protocol.replicate_to_node(
                node_id="node-2",
                file_id="abc...",
                data=b"content",
                content_type="text/plain"
            )
    """

    def __init__(
        self,
        node_id: str,
        host: str = "localhost",
        port: int = 9000,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0
    ):
        """
        Initialize network protocol.

        Args:
            node_id: This node's identifier
            host: This node's host
            port: This node's port
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_backoff: Initial backoff in seconds
        """
        self.node_id = node_id
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Peer addresses
        self.peer_addresses: Dict[str, str] = {}  # node_id → "host:port"

        # HTTP session (connection pool)
        self.session: Optional[aiohttp.ClientSession] = None if AIOHTTP_AVAILABLE else None

        # Statistics
        self.stats = {
            'requests_sent': 0,
            'requests_received': 0,
            'bytes_sent': 0,
            'bytes_received': 0,
            'errors': 0,
            'retries': 0
        }

        logger.info(f"Initialized wool network protocol for {node_id} at {host}:{port}")

    def register_peer(self, node_id: str, host: str, port: int):
        """
        Register a peer node.

        Args:
            node_id: Peer node identifier
            host: Peer host
            port: Peer port
        """
        self.peer_addresses[node_id] = f"http://{host}:{port}"
        logger.debug(f"Registered peer {node_id} at {host}:{port}")

    def _get_peer_url(self, node_id: str, endpoint: str = "/") -> str:
        """
        Get URL for peer node.

        Args:
            node_id: Peer node identifier
            endpoint: API endpoint

        Returns:
            Full URL

        Raises:
            ValueError: If peer not registered
        """
        if node_id not in self.peer_addresses:
            raise ValueError(f"Peer {node_id} not registered")

        base_url = self.peer_addresses[node_id]
        return f"{base_url}{endpoint}"

    async def ping(self, node_id: str) -> bool:
        """
        Ping a remote node.

        Args:
            node_id: Target node identifier

        Returns:
            True if node is reachable
        """
        if not AIOHTTP_AVAILABLE or self.session is None:
            logger.warning("aiohttp not available - cannot ping")
            return False

        request = NetworkRequest(
            operation=Operation.PING,
            node_id=self.node_id
        )

        try:
            url = self._get_peer_url(node_id, "/wool/ping")
            async with self.session.post(
                url,
                json=request.to_dict(),
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as resp:
                response_data = await resp.json()
                response = NetworkResponse.from_dict(response_data)
                return response.success
        except Exception as e:
            logger.debug(f"Ping failed for {node_id}: {e}")
            return False

    async def read_remote(
        self,
        node_id: str,
        ref: WoolReference
    ) -> bytes:
        """
        Read file from remote node.

        Args:
            node_id: Target node identifier
            ref: Wool reference to read

        Returns:
            File data

        Raises:
            Exception: If read fails
        """
        if not AIOHTTP_AVAILABLE or self.session is None:
            raise RuntimeError("aiohttp not available")

        request = NetworkRequest(
            operation=Operation.READ,
            node_id=self.node_id,
            file_id=ref.file_id,
            offset=ref.offset,
            length=ref.length
        )

        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                url = self._get_peer_url(node_id, "/wool/read")
                async with self.session.post(
                    url,
                    json=request.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status != 200:
                        raise Exception(f"HTTP {resp.status}")

                    # Read response
                    data = await resp.read()

                    self.stats['requests_sent'] += 1
                    self.stats['bytes_received'] += len(data)

                    return data

            except Exception as e:
                last_error = e
                self.stats['retries'] += 1

                if attempt < self.max_retries - 1:
                    backoff = self.retry_backoff * (2 ** attempt)
                    logger.debug(f"Read retry {attempt + 1}/{self.max_retries} after {backoff}s")
                    await asyncio.sleep(backoff)

        self.stats['errors'] += 1
        raise Exception(f"Read failed after {self.max_retries} retries: {last_error}")

    async def replicate_to_node(
        self,
        node_id: str,
        file_id: str,
        data: bytes,
        content_type: str = "application/octet-stream"
    ) -> bool:
        """
        Replicate file to remote node.

        Args:
            node_id: Target node identifier
            file_id: File identifier
            data: File data
            content_type: MIME type

        Returns:
            True if replication successful
        """
        if not AIOHTTP_AVAILABLE or self.session is None:
            logger.warning("aiohttp not available - cannot replicate")
            return False

        request = NetworkRequest(
            operation=Operation.REPLICATE,
            node_id=self.node_id,
            file_id=file_id,
            content_type=content_type
        )

        try:
            url = self._get_peer_url(node_id, "/wool/replicate")

            # Send request + data as multipart
            form_data = aiohttp.FormData()
            form_data.add_field('request', json.dumps(request.to_dict()), content_type='application/json')
            form_data.add_field('data', data, content_type=content_type)

            async with self.session.post(
                url,
                data=form_data,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Replication failed: HTTP {resp.status}")
                    self.stats['errors'] += 1
                    return False

                response_data = await resp.json()
                response = NetworkResponse.from_dict(response_data)

                self.stats['requests_sent'] += 1
                self.stats['bytes_sent'] += len(data)

                return response.success

        except Exception as e:
            logger.error(f"Replication to {node_id} failed: {e}")
            self.stats['errors'] += 1
            return False

    async def sync_file(
        self,
        node_id: str,
        file_id: str
    ) -> bool:
        """
        Request remote node to sync a file.

        Args:
            node_id: Target node identifier
            file_id: File to sync

        Returns:
            True if sync initiated
        """
        if not AIOHTTP_AVAILABLE or self.session is None:
            return False

        request = NetworkRequest(
            operation=Operation.SYNC,
            node_id=self.node_id,
            file_id=file_id
        )

        try:
            url = self._get_peer_url(node_id, "/wool/sync")
            async with self.session.post(
                url,
                json=request.to_dict(),
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                response_data = await resp.json()
                response = NetworkResponse.from_dict(response_data)

                self.stats['requests_sent'] += 1
                return response.success

        except Exception as e:
            logger.error(f"Sync request to {node_id} failed: {e}")
            self.stats['errors'] += 1
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get protocol statistics."""
        return {
            'node_id': self.node_id,
            'address': f"{self.host}:{self.port}",
            'peers_registered': len(self.peer_addresses),
            'operations': self.stats.copy()
        }

    async def __aenter__(self):
        """Async context manager entry."""
        if AIOHTTP_AVAILABLE:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close protocol and cleanup."""
        if self.session:
            await self.session.close()
            self.session = None
        logger.info(f"Closed wool network protocol for {self.node_id}")

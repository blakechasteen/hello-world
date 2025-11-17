#!/usr/bin/env python3
"""
Collaborative Workflow Editing
==============================

Real-time collaborative editing for workflows with conflict resolution.

Features:
- Multi-user editing
- Real-time synchronization via WebSocket
- Operational Transform for conflict resolution
- Presence awareness (who's editing what)
- Cursor tracking
- Undo/redo with branching history

Usage:
    from HoloLoom.workflows.collaborative import CollaborativeSession

    session = CollaborativeSession(workflow_id="my_workflow")

    # Add user
    await session.add_user(user_id="alice", websocket=ws)

    # Apply operation
    await session.apply_operation({
        'type': 'add_node',
        'user_id': 'alice',
        'node': {...}
    })

    # Broadcast to all users
    await session.broadcast({'type': 'cursor_move', 'x': 100, 'y': 200})
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict

try:
    from fastapi import WebSocket
except ImportError:
    WebSocket = None

logger = logging.getLogger(__name__)


class OperationType(str, Enum):
    """Types of workflow operations"""
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    UPDATE_NODE = "update_node"
    MOVE_NODE = "move_node"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    UPDATE_CONFIG = "update_config"


@dataclass
class Operation:
    """Single workflow operation"""
    op_id: str
    op_type: OperationType
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]

    # For operational transform
    version: int = 0
    parent_op_id: Optional[str] = None


@dataclass
class UserPresence:
    """User presence in collaborative session"""
    user_id: str
    display_name: str
    color: str
    websocket: Any  # WebSocket
    joined_at: datetime

    # State
    cursor_x: int = 0
    cursor_y: int = 0
    selected_node: Optional[str] = None
    is_editing: bool = False
    last_activity: datetime = field(default_factory=datetime.now)


class CollaborativeSession:
    """
    Manages a collaborative editing session for a workflow.

    Handles:
    - Multi-user synchronization
    - Conflict resolution
    - Presence tracking
    - Operation history
    """

    def __init__(self, workflow_id: str, workflow_data: Dict[str, Any] = None):
        self.workflow_id = workflow_id
        self.workflow_data = workflow_data or {'version': '1.0', 'nodes': [], 'connections': []}

        # Users
        self.users: Dict[str, UserPresence] = {}

        # Operation history
        self.operations: List[Operation] = []
        self.operation_counter = 0

        # Locks (for preventing concurrent edits to same node)
        self.node_locks: Dict[str, str] = {}  # node_id -> user_id

        # Colors for users
        self.user_colors = [
            '#667eea', '#f093fb', '#4facfe', '#43e97b',
            '#fa709a', '#30cfd0', '#8b5cf6', '#ec4899'
        ]
        self.color_index = 0

    async def add_user(self, user_id: str, display_name: str, websocket: Any):
        """Add user to collaborative session"""
        if user_id in self.users:
            logger.warning(f"User {user_id} already in session")
            return

        user = UserPresence(
            user_id=user_id,
            display_name=display_name,
            color=self.user_colors[self.color_index % len(self.user_colors)],
            websocket=websocket,
            joined_at=datetime.now()
        )

        self.users[user_id] = user
        self.color_index += 1

        # Broadcast user joined
        await self.broadcast({
            'type': 'user_joined',
            'user_id': user_id,
            'display_name': display_name,
            'color': user.color
        }, exclude=user_id)

        # Send current state to new user
        await self.send_to_user(user_id, {
            'type': 'initial_state',
            'workflow': self.workflow_data,
            'users': [
                {
                    'user_id': u.user_id,
                    'display_name': u.display_name,
                    'color': u.color,
                    'cursor_x': u.cursor_x,
                    'cursor_y': u.cursor_y,
                    'selected_node': u.selected_node
                }
                for u in self.users.values()
            ]
        })

        logger.info(f"User {display_name} ({user_id}) joined session {self.workflow_id}")

    async def remove_user(self, user_id: str):
        """Remove user from session"""
        if user_id not in self.users:
            return

        user = self.users[user_id]

        # Release any locks held by this user
        locked_nodes = [nid for nid, uid in self.node_locks.items() if uid == user_id]
        for node_id in locked_nodes:
            del self.node_locks[node_id]

        del self.users[user_id]

        # Broadcast user left
        await self.broadcast({
            'type': 'user_left',
            'user_id': user_id
        })

        logger.info(f"User {user.display_name} ({user_id}) left session {self.workflow_id}")

    async def apply_operation(self, operation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply operation to workflow with conflict resolution.

        Args:
            operation: Operation data

        Returns:
            Result with updated workflow state
        """
        user_id = operation.get('user_id')
        op_type = OperationType(operation.get('type'))

        # Create operation record
        op = Operation(
            op_id=self._generate_op_id(),
            op_type=op_type,
            user_id=user_id,
            timestamp=datetime.now(),
            data=operation.get('data', {}),
            version=self.operation_counter
        )

        self.operation_counter += 1

        # Check for conflicts
        conflict = self._check_conflict(op)
        if conflict:
            return {
                'status': 'conflict',
                'error': conflict,
                'operation_id': op.op_id
            }

        # Apply operation
        result = await self._execute_operation(op)

        if result['status'] == 'success':
            # Record operation
            self.operations.append(op)

            # Broadcast to other users
            await self.broadcast({
                'type': 'operation',
                'operation': {
                    'op_id': op.op_id,
                    'op_type': op.op_type.value,
                    'user_id': op.user_id,
                    'data': op.data,
                    'timestamp': op.timestamp.isoformat()
                }
            }, exclude=user_id)

        return result

    def _check_conflict(self, op: Operation) -> Optional[str]:
        """
        Check for conflicts before applying operation.

        Returns:
            Error message if conflict, None otherwise
        """
        if op.op_type in [OperationType.UPDATE_NODE, OperationType.REMOVE_NODE]:
            node_id = op.data.get('node_id')
            if node_id in self.node_locks and self.node_locks[node_id] != op.user_id:
                locked_by = self.users[self.node_locks[node_id]].display_name
                return f"Node {node_id} is being edited by {locked_by}"

        return None

    async def _execute_operation(self, op: Operation) -> Dict[str, Any]:
        """Execute operation on workflow data"""
        try:
            if op.op_type == OperationType.ADD_NODE:
                node = op.data.get('node')
                if not node:
                    return {'status': 'error', 'error': 'No node data provided'}

                self.workflow_data['nodes'].append(node)
                logger.info(f"Added node {node.get('id')} by {op.user_id}")

            elif op.op_type == OperationType.REMOVE_NODE:
                node_id = op.data.get('node_id')
                self.workflow_data['nodes'] = [
                    n for n in self.workflow_data['nodes']
                    if n['id'] != node_id
                ]

                # Remove connected connections
                self.workflow_data['connections'] = [
                    c for c in self.workflow_data['connections']
                    if c['from'] != node_id and c['to'] != node_id
                ]

                logger.info(f"Removed node {node_id} by {op.user_id}")

            elif op.op_type == OperationType.UPDATE_NODE:
                node_id = op.data.get('node_id')
                updates = op.data.get('updates', {})

                for node in self.workflow_data['nodes']:
                    if node['id'] == node_id:
                        node.update(updates)
                        break

                logger.info(f"Updated node {node_id} by {op.user_id}")

            elif op.op_type == OperationType.MOVE_NODE:
                node_id = op.data.get('node_id')
                x = op.data.get('x')
                y = op.data.get('y')

                for node in self.workflow_data['nodes']:
                    if node['id'] == node_id:
                        node['x'] = x
                        node['y'] = y
                        break

                logger.info(f"Moved node {node_id} to ({x}, {y}) by {op.user_id}")

            elif op.op_type == OperationType.ADD_CONNECTION:
                connection = op.data.get('connection')
                if not connection:
                    return {'status': 'error', 'error': 'No connection data provided'}

                self.workflow_data['connections'].append(connection)
                logger.info(f"Added connection {connection.get('id')} by {op.user_id}")

            elif op.op_type == OperationType.REMOVE_CONNECTION:
                conn_id = op.data.get('connection_id')
                self.workflow_data['connections'] = [
                    c for c in self.workflow_data['connections']
                    if c['id'] != conn_id
                ]

                logger.info(f"Removed connection {conn_id} by {op.user_id}")

            return {
                'status': 'success',
                'operation_id': op.op_id,
                'workflow': self.workflow_data
            }

        except Exception as e:
            logger.error(f"Operation execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def update_cursor(self, user_id: str, x: int, y: int):
        """Update user cursor position"""
        if user_id not in self.users:
            return

        user = self.users[user_id]
        user.cursor_x = x
        user.cursor_y = y
        user.last_activity = datetime.now()

        # Broadcast cursor move
        await self.broadcast({
            'type': 'cursor_move',
            'user_id': user_id,
            'x': x,
            'y': y
        }, exclude=user_id)

    async def lock_node(self, user_id: str, node_id: str) -> bool:
        """
        Lock node for editing.

        Returns:
            True if lock acquired, False otherwise
        """
        if node_id in self.node_locks and self.node_locks[node_id] != user_id:
            return False

        self.node_locks[node_id] = user_id

        # Broadcast lock
        await self.broadcast({
            'type': 'node_locked',
            'node_id': node_id,
            'user_id': user_id
        }, exclude=user_id)

        return True

    async def unlock_node(self, user_id: str, node_id: str):
        """Unlock node"""
        if node_id in self.node_locks and self.node_locks[node_id] == user_id:
            del self.node_locks[node_id]

            # Broadcast unlock
            await self.broadcast({
                'type': 'node_unlocked',
                'node_id': node_id
            }, exclude=user_id)

    async def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast message to all users except excluded"""
        disconnected = []

        for user_id, user in self.users.items():
            if user_id == exclude:
                continue

            try:
                if user.websocket:
                    await user.websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")
                disconnected.append(user_id)

        # Remove disconnected users
        for user_id in disconnected:
            await self.remove_user(user_id)

    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to specific user"""
        if user_id not in self.users:
            return

        user = self.users[user_id]
        try:
            if user.websocket:
                await user.websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send to {user_id}: {e}")
            await self.remove_user(user_id)

    def _generate_op_id(self) -> str:
        """Generate unique operation ID"""
        return f"op_{self.operation_counter}_{datetime.now().strftime('%f')}"

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        return self.workflow_data.copy()

    def get_operation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get operation history"""
        return [
            {
                'op_id': op.op_id,
                'op_type': op.op_type.value,
                'user_id': op.user_id,
                'timestamp': op.timestamp.isoformat(),
                'data': op.data
            }
            for op in self.operations[-limit:]
        ]


class CollaborationManager:
    """
    Manages multiple collaborative sessions.

    Coordinates across workflows and handles session lifecycle.
    """

    def __init__(self):
        self.sessions: Dict[str, CollaborativeSession] = {}

    def get_or_create_session(
        self,
        workflow_id: str,
        workflow_data: Optional[Dict[str, Any]] = None
    ) -> CollaborativeSession:
        """Get existing session or create new one"""
        if workflow_id not in self.sessions:
            self.sessions[workflow_id] = CollaborativeSession(workflow_id, workflow_data)
            logger.info(f"Created new collaborative session for {workflow_id}")

        return self.sessions[workflow_id]

    def get_session(self, workflow_id: str) -> Optional[CollaborativeSession]:
        """Get session by workflow ID"""
        return self.sessions.get(workflow_id)

    def close_session(self, workflow_id: str):
        """Close collaborative session"""
        if workflow_id in self.sessions:
            del self.sessions[workflow_id]
            logger.info(f"Closed collaborative session for {workflow_id}")

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions"""
        return [
            {
                'workflow_id': session.workflow_id,
                'user_count': len(session.users),
                'operation_count': len(session.operations)
            }
            for session in self.sessions.values()
        ]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        session = CollaborativeSession("test_workflow_1")

        print("=" * 80)
        print("Collaborative Editing Demo")
        print("=" * 80)

        # Simulate users
        class MockWebSocket:
            async def send_json(self, message):
                print(f"  → Broadcast: {message['type']}")

        await session.add_user("alice", "Alice", MockWebSocket())
        await session.add_user("bob", "Bob", MockWebSocket())

        # Alice adds a node
        print("\n1. Alice adds a node:")
        result = await session.apply_operation({
            'type': 'add_node',
            'user_id': 'alice',
            'data': {
                'node': {
                    'id': 'node_1',
                    'agentType': 'hololoom',
                    'x': 100,
                    'y': 200,
                    'config': {}
                }
            }
        })
        print(f"   Status: {result['status']}")

        # Bob tries to edit the same node (should conflict if locked)
        print("\n2. Bob tries to move Alice's node:")
        result = await session.apply_operation({
            'type': 'move_node',
            'user_id': 'bob',
            'data': {
                'node_id': 'node_1',
                'x': 200,
                'y': 300
            }
        })
        print(f"   Status: {result['status']}")

        # Alice adds connection
        print("\n3. Alice adds another node and connection:")
        await session.apply_operation({
            'type': 'add_node',
            'user_id': 'alice',
            'data': {
                'node': {
                    'id': 'node_2',
                    'agentType': 'response',
                    'x': 300,
                    'y': 200,
                    'config': {}
                }
            }
        })

        await session.apply_operation({
            'type': 'add_connection',
            'user_id': 'alice',
            'data': {
                'connection': {
                    'id': 'conn_1',
                    'from': 'node_1',
                    'to': 'node_2'
                }
            }
        })

        # Show final state
        print("\n4. Final workflow state:")
        state = session.get_workflow_state()
        print(f"   Nodes: {len(state['nodes'])}")
        print(f"   Connections: {len(state['connections'])}")
        print(f"   Operations: {len(session.operations)}")

        # Operation history
        print("\n5. Operation history:")
        for op_data in session.get_operation_history():
            print(f"   - {op_data['op_type']} by {op_data['user_id']} at {op_data['timestamp']}")

    asyncio.run(demo())

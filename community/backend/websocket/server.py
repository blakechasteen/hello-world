"""
WebSocket server for real-time features using Socket.IO

Features:
- Real-time messaging
- Live notifications
- Typing indicators
- Online presence
- Post/comment updates
- Redis adapter for horizontal scaling
"""
import socketio
import redis.asyncio as aioredis
from typing import Dict, Set, Optional
import logging
from uuid import UUID
import json

from core.config import settings

logger = logging.getLogger(__name__)

# Create Redis client for pub/sub
redis_client = aioredis.from_url(
    settings.REDIS_URL,
    encoding="utf-8",
    decode_responses=True
)

# Create Socket.IO server with Redis adapter for horizontal scaling
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins=settings.CORS_ORIGINS,
    logger=True,
    engineio_logger=True,
    # Redis manager for multi-instance support
    client_manager=socketio.AsyncRedisManager(settings.REDIS_URL),
)

# Create ASGI app
socket_app = socketio.ASGIApp(
    sio,
    socketio_path='/socket.io'
)

# Store user sessions: {session_id: user_id}
user_sessions: Dict[str, str] = {}

# Store user presence: {user_id: set of session_ids}
user_presence: Dict[str, Set[str]] = {}


@sio.event
async def connect(sid, environ, auth):
    """
    Handle client connection

    Auth should contain: {"token": "jwt_token"}
    """
    logger.info(f"Client connecting: {sid}")

    # Authenticate user
    if not auth or 'token' not in auth:
        logger.warning(f"Connection rejected for {sid}: No auth token")
        return False

    token = auth['token']

    # Verify JWT token
    from auth.security import decode_token

    payload = decode_token(token)
    if not payload or payload.get('type') != 'access':
        logger.warning(f"Connection rejected for {sid}: Invalid token")
        return False

    user_id = payload.get('sub')
    if not user_id:
        logger.warning(f"Connection rejected for {sid}: No user_id in token")
        return False

    # Store session
    user_sessions[sid] = user_id

    # Update presence
    if user_id not in user_presence:
        user_presence[user_id] = set()
    user_presence[user_id].add(sid)

    # Store in Redis for presence tracking
    await redis_client.sadd(f"online:users", user_id)
    await redis_client.hset(f"user:sessions:{user_id}", sid, "connected")

    logger.info(f"User {user_id} connected (session: {sid})")

    # Notify user's friends they're online
    await sio.emit('user_online', {'user_id': user_id}, room=sid)

    return True


@sio.event
async def disconnect(sid):
    """
    Handle client disconnection
    """
    user_id = user_sessions.get(sid)

    if user_id:
        logger.info(f"User {user_id} disconnected (session: {sid})")

        # Remove from presence
        if user_id in user_presence:
            user_presence[user_id].discard(sid)

            # If no more sessions, mark as offline
            if not user_presence[user_id]:
                del user_presence[user_id]
                await redis_client.srem("online:users", user_id)
                await sio.emit('user_offline', {'user_id': user_id})

        # Remove from Redis
        await redis_client.hdel(f"user:sessions:{user_id}", sid)

        # Remove session
        del user_sessions[sid]

    logger.info(f"Client disconnected: {sid}")


@sio.event
async def join_room(sid, data):
    """
    Join a room (community, conversation, post)

    data: {"room_type": "community|conversation|post", "room_id": "uuid"}
    """
    room_type = data.get('room_type')
    room_id = data.get('room_id')

    if not room_type or not room_id:
        return {'error': 'Missing room_type or room_id'}

    room_name = f"{room_type}:{room_id}"

    sio.enter_room(sid, room_name)
    logger.info(f"Session {sid} joined room {room_name}")

    return {'status': 'joined', 'room': room_name}


@sio.event
async def leave_room(sid, data):
    """
    Leave a room

    data: {"room_type": "...", "room_id": "..."}
    """
    room_type = data.get('room_type')
    room_id = data.get('room_id')

    if not room_type or not room_id:
        return {'error': 'Missing room_type or room_id'}

    room_name = f"{room_type}:{room_id}"

    sio.leave_room(sid, room_name)
    logger.info(f"Session {sid} left room {room_name}")

    return {'status': 'left', 'room': room_name}


@sio.event
async def send_message(sid, data):
    """
    Send a message to a conversation

    data: {
        "conversation_id": "uuid",
        "content": "message text",
        "reply_to_id": "uuid" (optional)
    }
    """
    user_id = user_sessions.get(sid)
    if not user_id:
        return {'error': 'Not authenticated'}

    conversation_id = data.get('conversation_id')
    content = data.get('content')

    if not conversation_id or not content:
        return {'error': 'Missing conversation_id or content'}

    # TODO: Save message to database
    # For now, just broadcast to conversation room

    message_data = {
        'message_id': str(UUID),  # Generate new UUID
        'conversation_id': conversation_id,
        'sender_id': user_id,
        'content': content,
        'created_at': 'timestamp',
    }

    room_name = f"conversation:{conversation_id}"
    await sio.emit('message_received', message_data, room=room_name)

    logger.info(f"Message sent in conversation {conversation_id} by user {user_id}")

    return {'status': 'sent', 'message': message_data}


@sio.event
async def typing_start(sid, data):
    """
    User started typing in a conversation

    data: {"conversation_id": "uuid"}
    """
    user_id = user_sessions.get(sid)
    if not user_id:
        return

    conversation_id = data.get('conversation_id')
    if not conversation_id:
        return

    room_name = f"conversation:{conversation_id}"

    # Broadcast to room (except sender)
    await sio.emit(
        'user_typing',
        {'user_id': user_id, 'conversation_id': conversation_id},
        room=room_name,
        skip_sid=sid
    )


@sio.event
async def typing_stop(sid, data):
    """
    User stopped typing

    data: {"conversation_id": "uuid"}
    """
    user_id = user_sessions.get(sid)
    if not user_id:
        return

    conversation_id = data.get('conversation_id')
    if not conversation_id:
        return

    room_name = f"conversation:{conversation_id}"

    await sio.emit(
        'user_stopped_typing',
        {'user_id': user_id, 'conversation_id': conversation_id},
        room=room_name,
        skip_sid=sid
    )


@sio.event
async def post_update(sid, data):
    """
    Real-time post update (for collaborative editing, live scores, etc.)

    data: {
        "post_id": "uuid",
        "update_type": "vote|comment|edit",
        "data": {...}
    }
    """
    post_id = data.get('post_id')
    update_type = data.get('update_type')

    if not post_id or not update_type:
        return

    room_name = f"post:{post_id}"

    await sio.emit(
        'post_updated',
        {
            'post_id': post_id,
            'update_type': update_type,
            'data': data.get('data', {})
        },
        room=room_name
    )


@sio.event
async def send_notification(sid, data):
    """
    Send a notification to a user

    data: {
        "user_id": "uuid",
        "notification_type": "...",
        "title": "...",
        "message": "...",
        "link": {...}
    }
    """
    # This would typically be called from backend services, not clients
    # But included for completeness

    target_user_id = data.get('user_id')
    if not target_user_id:
        return

    # Send to all sessions of the target user
    if target_user_id in user_presence:
        for user_sid in user_presence[target_user_id]:
            await sio.emit('notification', data, room=user_sid)


# Helper function to broadcast to a room (can be called from FastAPI)
async def broadcast_to_room(room: str, event: str, data: dict):
    """
    Broadcast an event to a room

    This can be called from FastAPI endpoints
    """
    await sio.emit(event, data, room=room)


# Helper function to send to a specific user
async def send_to_user(user_id: str, event: str, data: dict):
    """
    Send an event to all sessions of a specific user
    """
    if user_id in user_presence:
        for sid in user_presence[user_id]:
            await sio.emit(event, data, room=sid)


# Health check for WebSocket server
@sio.event
async def ping(sid):
    """
    Ping-pong for connection health check
    """
    return 'pong'


# Get online users count
async def get_online_users_count() -> int:
    """
    Get count of currently online users
    """
    return await redis_client.scard("online:users")


# Get online users
async def get_online_users() -> list:
    """
    Get list of currently online user IDs
    """
    return await redis_client.smembers("online:users")


# Check if user is online
async def is_user_online(user_id: str) -> bool:
    """
    Check if a specific user is online
    """
    return await redis_client.sismember("online:users", user_id)

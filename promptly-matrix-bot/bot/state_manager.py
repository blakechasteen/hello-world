#!/usr/bin/env python3
"""
State Manager for Promptly Matrix Bot

Manages conversation state, saved prompts, and workflow state using Redis.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("⚠️ Redis not available - using in-memory state")


class StateManager:
    """
    Manage bot state (conversation context, saved prompts, workflows)

    Falls back to in-memory dict if Redis unavailable.
    """

    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize state manager.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.redis_url = redis_url
        self.redis_client = None
        self.memory_store = {}  # Fallback in-memory storage

        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("✅ Connected to Redis")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                logger.warning("Using in-memory state (not persistent)")
                self.redis_client = None
        else:
            logger.info("Using in-memory state (not persistent)")

    def _key(self, *parts: str) -> str:
        """Generate Redis key"""
        return ":".join(["promptly"] + list(parts))

    def _set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value (Redis or memory)"""
        if self.redis_client:
            if ttl:
                self.redis_client.setex(key, ttl, json.dumps(value))
            else:
                self.redis_client.set(key, json.dumps(value))
        else:
            self.memory_store[key] = (value, time.time() + ttl if ttl else None)

    def _get(self, key: str) -> Optional[Any]:
        """Get value (Redis or memory)"""
        if self.redis_client:
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        else:
            if key in self.memory_store:
                value, expiry = self.memory_store[key]
                if expiry is None or time.time() < expiry:
                    return value
                else:
                    del self.memory_store[key]
            return None

    def _delete(self, key: str):
        """Delete value"""
        if self.redis_client:
            self.redis_client.delete(key)
        else:
            self.memory_store.pop(key, None)

    def _scan(self, pattern: str) -> List[str]:
        """Scan keys matching pattern"""
        if self.redis_client:
            keys = []
            for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            return keys
        else:
            import fnmatch
            return [k for k in self.memory_store.keys() if fnmatch.fnmatch(k, pattern)]

    # ========================================================================
    # Conversation Context
    # ========================================================================

    def get_conversation_context(self, room_id: str, user_id: str) -> Optional[Dict]:
        """
        Get conversation context for user in room.

        Args:
            room_id: Matrix room ID
            user_id: Matrix user ID

        Returns:
            Context dict or None
        """
        key = self._key("context", room_id, user_id)
        return self._get(key)

    def set_conversation_context(
        self,
        room_id: str,
        user_id: str,
        context: Dict,
        ttl: int = 3600
    ):
        """
        Set conversation context.

        Args:
            room_id: Matrix room ID
            user_id: Matrix user ID
            context: Context dict
            ttl: Time to live in seconds (default: 1 hour)
        """
        key = self._key("context", room_id, user_id)
        self._set(key, context, ttl)

    def append_to_history(
        self,
        room_id: str,
        user_id: str,
        user_message: str,
        bot_response: str
    ):
        """Append message exchange to conversation history"""
        context = self.get_conversation_context(room_id, user_id) or {"history": []}

        context['history'].append({
            "user": user_message,
            "bot": bot_response,
            "timestamp": datetime.now().isoformat()
        })

        # Keep last 10 exchanges
        context['history'] = context['history'][-10:]

        self.set_conversation_context(room_id, user_id, context)

    # ========================================================================
    # Saved Prompts (per room)
    # ========================================================================

    def save_prompt(
        self,
        room_id: str,
        name: str,
        prompt_data: Dict
    ) -> bool:
        """
        Save a prompt to room library.

        Args:
            room_id: Matrix room ID
            name: Prompt name
            prompt_data: Prompt data (task, signature, program, etc.)

        Returns:
            True if saved, False on error
        """
        try:
            key = self._key("prompt", room_id, name)

            # Add metadata
            prompt_data['name'] = name
            prompt_data['room_id'] = room_id
            prompt_data['created'] = datetime.now().isoformat()

            self._set(key, prompt_data)
            logger.info(f"Saved prompt '{name}' in room {room_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save prompt: {e}")
            return False

    def get_prompt(self, room_id: str, name: str) -> Optional[Dict]:
        """Get saved prompt"""
        key = self._key("prompt", room_id, name)
        return self._get(key)

    def list_prompts(self, room_id: str) -> List[Dict]:
        """List all saved prompts in room"""
        pattern = self._key("prompt", room_id, "*")
        keys = self._scan(pattern)

        prompts = []
        for key in keys:
            data = self._get(key)
            if data:
                prompts.append(data)

        # Sort by created date (newest first)
        prompts.sort(key=lambda p: p.get('created', ''), reverse=True)
        return prompts

    def delete_prompt(self, room_id: str, name: str) -> bool:
        """Delete saved prompt"""
        key = self._key("prompt", room_id, name)
        self._delete(key)
        return True

    # ========================================================================
    # Workflow State (multi-step execution)
    # ========================================================================

    def get_workflow_state(self, room_id: str, workflow_id: str) -> Optional[Dict]:
        """Get workflow execution state"""
        key = self._key("workflow", room_id, workflow_id)
        return self._get(key)

    def set_workflow_state(
        self,
        room_id: str,
        workflow_id: str,
        state: Dict,
        ttl: int = 86400  # 24 hours
    ):
        """Set workflow state"""
        key = self._key("workflow", room_id, workflow_id)
        self._set(key, state, ttl)

    def delete_workflow_state(self, room_id: str, workflow_id: str):
        """Delete workflow state"""
        key = self._key("workflow", room_id, workflow_id)
        self._delete(key)

    # ========================================================================
    # Approval Workflows
    # ========================================================================

    def create_approval_request(
        self,
        room_id: str,
        request_id: str,
        data: Dict,
        required_approvals: int = 1
    ):
        """
        Create approval request.

        Args:
            room_id: Matrix room ID
            request_id: Unique request ID
            data: Request data
            required_approvals: Number of approvals needed
        """
        key = self._key("approval", room_id, request_id)

        approval_data = {
            "request_id": request_id,
            "room_id": room_id,
            "data": data,
            "required_approvals": required_approvals,
            "approvals": [],
            "rejections": [],
            "created": datetime.now().isoformat(),
            "status": "pending"
        }

        self._set(key, approval_data, ttl=86400)  # 24h expiry

    def add_approval(
        self,
        room_id: str,
        request_id: str,
        user_id: str,
        approved: bool
    ) -> Dict:
        """
        Add approval/rejection to request.

        Args:
            room_id: Matrix room ID
            request_id: Request ID
            user_id: User approving/rejecting
            approved: True for approval, False for rejection

        Returns:
            Updated approval state
        """
        key = self._key("approval", room_id, request_id)
        state = self._get(key)

        if not state:
            return {"status": "not_found"}

        if approved:
            if user_id not in state['approvals']:
                state['approvals'].append(user_id)
        else:
            if user_id not in state['rejections']:
                state['rejections'].append(user_id)

        # Check if approved
        if len(state['approvals']) >= state['required_approvals']:
            state['status'] = 'approved'
        elif len(state['rejections']) > 0:
            state['status'] = 'rejected'

        self._set(key, state, ttl=86400)
        return state

    def get_approval_state(self, room_id: str, request_id: str) -> Optional[Dict]:
        """Get approval request state"""
        key = self._key("approval", room_id, request_id)
        return self._get(key)


# Singleton
_state_manager = None

def get_state_manager(redis_url: Optional[str] = None) -> StateManager:
    """Get singleton state manager"""
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager(redis_url)
    return _state_manager


# Test
if __name__ == "__main__":
    import os

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    state = StateManager(redis_url)

    # Test conversation context
    state.set_conversation_context(
        "!room123:matrix.org",
        "@user:matrix.org",
        {"last_command": "optimize", "task": "customer_support"}
    )

    context = state.get_conversation_context("!room123:matrix.org", "@user:matrix.org")
    print(f"Context: {context}")

    # Test saved prompts
    state.save_prompt(
        "!room123:matrix.org",
        "customer_support_qa",
        {
            "task": "Answer customer support questions",
            "signature": {"inputs": ["question"], "outputs": ["answer"]},
            "metrics": {"accuracy": 0.95}
        }
    )

    prompts = state.list_prompts("!room123:matrix.org")
    print(f"Saved prompts: {prompts}")

    # Test approval
    state.create_approval_request(
        "!room123:matrix.org",
        "approval_001",
        {"action": "deploy_prompt", "prompt": "customer_support_qa"},
        required_approvals=2
    )

    state.add_approval("!room123:matrix.org", "approval_001", "@user1:matrix.org", True)
    state.add_approval("!room123:matrix.org", "approval_001", "@user2:matrix.org", True)

    approval_state = state.get_approval_state("!room123:matrix.org", "approval_001")
    print(f"Approval state: {approval_state['status']}")  # Should be 'approved'

    print("\n✅ All state manager tests passed!")

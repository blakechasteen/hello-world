#!/usr/bin/env python3
"""
Team Shared Context System for Promptly Matrix Bot

Cross-room collaboration features:
- Shared prompt libraries
- Team-wide configurations
- Cross-room workflows
- Context inheritance
- Permission management

Features:
- Hierarchical context (team > room > user)
- Prompt library sharing
- Workflow templates
- Configuration inheritance
- Access control
- Context versioning

Usage:
    context = TeamContext(state)

    # Share prompt across team
    await context.share_prompt(
        name="customer_support_v1",
        prompt="...",
        scope="team",
        created_by="@alice:matrix.org"
    )

    # Get team context
    team_prompts = await context.get_team_prompts()

    # Inherit configuration
    config = await context.get_merged_config(
        room_id="!room:matrix.org",
        user_id="@alice:matrix.org"
    )
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ContextScope(Enum):
    """Context scope levels"""
    TEAM = "team"      # Shared across all rooms
    ROOM = "room"      # Specific to room
    USER = "user"      # Personal to user


class Permission(Enum):
    """Access permissions"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class SharedPrompt:
    """Shared prompt definition"""
    name: str
    prompt: str
    scope: ContextScope
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    permissions: Dict[str, Permission] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'prompt': self.prompt,
            'scope': self.scope.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'version': self.version,
            'tags': self.tags,
            'metadata': self.metadata,
            'permissions': {u: p.value for u, p in self.permissions.items()}
        }


@dataclass
class WorkflowTemplate:
    """Shared workflow template"""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    scope: ContextScope
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    permissions: Dict[str, Permission] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'steps': self.steps,
            'scope': self.scope.value,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags,
            'permissions': {u: p.value for u, p in self.permissions.items()}
        }


@dataclass
class TeamConfig:
    """Team-wide configuration"""
    team_name: str
    settings: Dict[str, Any] = field(default_factory=dict)
    default_model: str = "gpt-4"
    default_risk_threshold: float = 0.7
    approval_settings: Dict[str, Any] = field(default_factory=dict)
    admins: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'team_name': self.team_name,
            'settings': self.settings,
            'default_model': self.default_model,
            'default_risk_threshold': self.default_risk_threshold,
            'approval_settings': self.approval_settings,
            'admins': list(self.admins)
        }


class TeamContext:
    """
    Team shared context manager.

    Manages hierarchical context (team > room > user) with
    prompt libraries, workflows, and configurations.

    Usage:
        context = TeamContext(state)

        # Share prompt
        await context.share_prompt(
            name="support_prompt",
            prompt="You are a helpful assistant...",
            scope=ContextScope.TEAM,
            created_by="@alice:matrix.org"
        )

        # Get team prompts
        prompts = await context.get_team_prompts()

        # Grant permission
        await context.grant_permission(
            prompt_name="support_prompt",
            user_id="@bob:matrix.org",
            permission=Permission.WRITE
        )

        # Merged config (team + room + user)
        config = await context.get_merged_config(
            room_id="!room:matrix.org",
            user_id="@alice:matrix.org"
        )
    """

    def __init__(self, state=None, team_name: str = "default"):
        """
        Initialize team context.

        Args:
            state: State manager for persistence
            team_name: Team identifier
        """
        self.state = state
        self.team_name = team_name

        # In-memory storage (backed by Redis if available)
        self.prompts: Dict[str, SharedPrompt] = {}
        self.workflows: Dict[str, WorkflowTemplate] = {}
        self.team_config = TeamConfig(team_name=team_name)
        self.room_configs: Dict[str, Dict[str, Any]] = {}
        self.user_configs: Dict[str, Dict[str, Any]] = {}

        # Load from state
        self._load_from_state()

    def _load_from_state(self):
        """Load context from state"""
        if not self.state:
            return

        try:
            # Load prompts
            prompts_data = self.state.get(f"team:{self.team_name}:prompts")
            if prompts_data:
                for prompt_dict in json.loads(prompts_data):
                    prompt = self._dict_to_prompt(prompt_dict)
                    self.prompts[prompt.name] = prompt

            # Load workflows
            workflows_data = self.state.get(f"team:{self.team_name}:workflows")
            if workflows_data:
                for workflow_dict in json.loads(workflows_data):
                    workflow = self._dict_to_workflow(workflow_dict)
                    self.workflows[workflow.name] = workflow

            # Load team config
            config_data = self.state.get(f"team:{self.team_name}:config")
            if config_data:
                config_dict = json.loads(config_data)
                self.team_config = self._dict_to_config(config_dict)

            logger.info(
                f"Loaded team context: {len(self.prompts)} prompts, "
                f"{len(self.workflows)} workflows"
            )

        except Exception as e:
            logger.error(f"Failed to load team context: {e}")

    def _dict_to_prompt(self, d: Dict[str, Any]) -> SharedPrompt:
        """Convert dictionary to SharedPrompt"""
        return SharedPrompt(
            name=d['name'],
            prompt=d['prompt'],
            scope=ContextScope(d['scope']),
            created_by=d['created_by'],
            created_at=datetime.fromisoformat(d['created_at']),
            version=d.get('version', 1),
            tags=d.get('tags', []),
            metadata=d.get('metadata', {}),
            permissions={u: Permission(p) for u, p in d.get('permissions', {}).items()}
        )

    def _dict_to_workflow(self, d: Dict[str, Any]) -> WorkflowTemplate:
        """Convert dictionary to WorkflowTemplate"""
        return WorkflowTemplate(
            name=d['name'],
            description=d['description'],
            steps=d['steps'],
            scope=ContextScope(d['scope']),
            created_by=d['created_by'],
            created_at=datetime.fromisoformat(d['created_at']),
            tags=d.get('tags', []),
            permissions={u: Permission(p) for u, p in d.get('permissions', {}).items()}
        )

    def _dict_to_config(self, d: Dict[str, Any]) -> TeamConfig:
        """Convert dictionary to TeamConfig"""
        return TeamConfig(
            team_name=d['team_name'],
            settings=d.get('settings', {}),
            default_model=d.get('default_model', 'gpt-4'),
            default_risk_threshold=d.get('default_risk_threshold', 0.7),
            approval_settings=d.get('approval_settings', {}),
            admins=set(d.get('admins', []))
        )

    async def share_prompt(
        self,
        name: str,
        prompt: str,
        scope: ContextScope,
        created_by: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SharedPrompt:
        """
        Share prompt across team/room.

        Args:
            name: Prompt name
            prompt: Prompt text
            scope: Sharing scope (team/room/user)
            created_by: Creator user ID
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Created SharedPrompt
        """
        shared_prompt = SharedPrompt(
            name=name,
            prompt=prompt,
            scope=scope,
            created_by=created_by,
            tags=tags or [],
            metadata=metadata or {},
            permissions={created_by: Permission.ADMIN}
        )

        self.prompts[name] = shared_prompt

        # Persist to state
        await self._save_prompts()

        logger.info(f"Shared prompt '{name}' with scope {scope.value}")

        return shared_prompt

    async def share_workflow(
        self,
        name: str,
        description: str,
        steps: List[Dict[str, Any]],
        scope: ContextScope,
        created_by: str,
        tags: Optional[List[str]] = None
    ) -> WorkflowTemplate:
        """
        Share workflow template.

        Args:
            name: Workflow name
            description: Description
            steps: Workflow steps
            scope: Sharing scope
            created_by: Creator user ID
            tags: Optional tags

        Returns:
            Created WorkflowTemplate
        """
        workflow = WorkflowTemplate(
            name=name,
            description=description,
            steps=steps,
            scope=scope,
            created_by=created_by,
            tags=tags or [],
            permissions={created_by: Permission.ADMIN}
        )

        self.workflows[name] = workflow

        # Persist to state
        await self._save_workflows()

        logger.info(f"Shared workflow '{name}' with scope {scope.value}")

        return workflow

    async def get_team_prompts(
        self,
        user_id: Optional[str] = None
    ) -> List[SharedPrompt]:
        """
        Get team-level prompts (optionally filtered by permission).

        Args:
            user_id: Optional user ID for permission filtering

        Returns:
            List of accessible team prompts
        """
        prompts = [p for p in self.prompts.values() if p.scope == ContextScope.TEAM]

        # Filter by permission
        if user_id:
            prompts = [
                p for p in prompts
                if self._has_permission(p, user_id, Permission.READ)
            ]

        return prompts

    async def get_room_prompts(
        self,
        room_id: str,
        user_id: Optional[str] = None
    ) -> List[SharedPrompt]:
        """Get room-specific prompts"""
        prompts = [
            p for p in self.prompts.values()
            if p.scope == ContextScope.ROOM and
               p.metadata.get('room_id') == room_id
        ]

        if user_id:
            prompts = [
                p for p in prompts
                if self._has_permission(p, user_id, Permission.READ)
            ]

        return prompts

    async def get_prompt(
        self,
        name: str,
        user_id: Optional[str] = None
    ) -> Optional[SharedPrompt]:
        """
        Get prompt by name.

        Args:
            name: Prompt name
            user_id: Optional user ID for permission check

        Returns:
            SharedPrompt if found and accessible
        """
        prompt = self.prompts.get(name)

        if not prompt:
            return None

        # Check permission
        if user_id and not self._has_permission(prompt, user_id, Permission.READ):
            return None

        return prompt

    async def grant_permission(
        self,
        prompt_name: str,
        user_id: str,
        permission: Permission
    ):
        """Grant permission to user for prompt"""
        prompt = self.prompts.get(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_name}")

        prompt.permissions[user_id] = permission

        await self._save_prompts()

        logger.info(f"Granted {permission.value} permission on '{prompt_name}' to {user_id}")

    async def revoke_permission(
        self,
        prompt_name: str,
        user_id: str
    ):
        """Revoke permission from user"""
        prompt = self.prompts.get(prompt_name)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_name}")

        if user_id in prompt.permissions:
            del prompt.permissions[user_id]

        await self._save_prompts()

        logger.info(f"Revoked permission on '{prompt_name}' from {user_id}")

    def _has_permission(
        self,
        prompt: SharedPrompt,
        user_id: str,
        required: Permission
    ) -> bool:
        """Check if user has permission"""
        # Creator always has admin
        if prompt.created_by == user_id:
            return True

        # Team admins have access
        if user_id in self.team_config.admins:
            return True

        # Check explicit permissions
        user_perm = prompt.permissions.get(user_id)

        if not user_perm:
            return False

        # Permission hierarchy: ADMIN > WRITE > READ
        perm_levels = {Permission.READ: 1, Permission.WRITE: 2, Permission.ADMIN: 3}

        return perm_levels[user_perm] >= perm_levels[required]

    async def get_merged_config(
        self,
        room_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Get merged configuration (team < room < user).

        Team config is base, overridden by room, then user.

        Args:
            room_id: Room ID
            user_id: User ID

        Returns:
            Merged configuration
        """
        # Start with team config
        config = self.team_config.settings.copy()

        # Merge room config
        room_config = self.room_configs.get(room_id, {})
        config.update(room_config)

        # Merge user config
        user_config = self.user_configs.get(user_id, {})
        config.update(user_config)

        return config

    async def set_room_config(
        self,
        room_id: str,
        config: Dict[str, Any]
    ):
        """Set room-specific configuration"""
        self.room_configs[room_id] = config

        if self.state:
            self.state.set(
                f"team:{self.team_name}:room:{room_id}:config",
                json.dumps(config)
            )

        logger.info(f"Updated room config for {room_id}")

    async def set_user_config(
        self,
        user_id: str,
        config: Dict[str, Any]
    ):
        """Set user-specific configuration"""
        self.user_configs[user_id] = config

        if self.state:
            self.state.set(
                f"team:{self.team_name}:user:{user_id}:config",
                json.dumps(config)
            )

        logger.info(f"Updated user config for {user_id}")

    async def _save_prompts(self):
        """Save prompts to state"""
        if not self.state:
            return

        prompts_data = json.dumps([p.to_dict() for p in self.prompts.values()])
        self.state.set(f"team:{self.team_name}:prompts", prompts_data)

    async def _save_workflows(self):
        """Save workflows to state"""
        if not self.state:
            return

        workflows_data = json.dumps([w.to_dict() for w in self.workflows.values()])
        self.state.set(f"team:{self.team_name}:workflows", workflows_data)

    async def search_prompts(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        scope: Optional[ContextScope] = None,
        user_id: Optional[str] = None
    ) -> List[SharedPrompt]:
        """
        Search prompts.

        Args:
            query: Search query (matches name/prompt)
            tags: Optional tag filter
            scope: Optional scope filter
            user_id: Optional user for permission filtering

        Returns:
            Matching prompts
        """
        results = list(self.prompts.values())

        # Filter by query
        if query:
            query_lower = query.lower()
            results = [
                p for p in results
                if query_lower in p.name.lower() or query_lower in p.prompt.lower()
            ]

        # Filter by tags
        if tags:
            results = [
                p for p in results
                if any(tag in p.tags for tag in tags)
            ]

        # Filter by scope
        if scope:
            results = [p for p in results if p.scope == scope]

        # Filter by permission
        if user_id:
            results = [
                p for p in results
                if self._has_permission(p, user_id, Permission.READ)
            ]

        return results


# Singleton
_team_context = None


def get_team_context(state=None, team_name: str = "default") -> TeamContext:
    """Get singleton team context"""
    global _team_context
    if _team_context is None:
        _team_context = TeamContext(state, team_name)
    return _team_context


# Test
if __name__ == "__main__":
    import asyncio

    async def test():
        # Create team context
        context = TeamContext(team_name="test_team")

        # Share team prompt
        await context.share_prompt(
            name="customer_support",
            prompt="You are a helpful customer support agent...",
            scope=ContextScope.TEAM,
            created_by="@alice:matrix.org",
            tags=["support", "customer"]
        )

        # Share workflow
        await context.share_workflow(
            name="review_deploy",
            description="Code review + approval + deployment",
            steps=[
                {"type": "code-review", "params": {}},
                {"type": "approval", "params": {"risk": "high"}},
                {"type": "deploy", "params": {"target": "production"}}
            ],
            scope=ContextScope.TEAM,
            created_by="@alice:matrix.org",
            tags=["deployment"]
        )

        # Grant permission
        await context.grant_permission(
            "customer_support",
            "@bob:matrix.org",
            Permission.WRITE
        )

        # Get team prompts
        team_prompts = await context.get_team_prompts(user_id="@bob:matrix.org")
        print(f"Team prompts accessible to @bob: {len(team_prompts)}")
        for prompt in team_prompts:
            print(f"  - {prompt.name} (by {prompt.created_by})")

        # Search prompts
        search_results = await context.search_prompts(
            query="support",
            user_id="@bob:matrix.org"
        )
        print(f"\nSearch results for 'support': {len(search_results)}")

        # Merged config
        await context.set_room_config("!room:matrix.org", {"theme": "dark"})
        await context.set_user_config("@bob:matrix.org", {"notifications": "off"})

        config = await context.get_merged_config(
            room_id="!room:matrix.org",
            user_id="@bob:matrix.org"
        )
        print(f"\nMerged config: {config}")

        print("\nAll team context tests passed!")

    asyncio.run(test())

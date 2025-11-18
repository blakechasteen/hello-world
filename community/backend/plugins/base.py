"""
Plugin base class and protocol
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from uuid import UUID
from enum import Enum


class PluginType(str, Enum):
    """Plugin types"""

    CONTENT = "content"
    INTEGRATION = "integration"
    MODERATION = "moderation"
    ENGAGEMENT = "engagement"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    CUSTOMIZATION = "customization"
    ADMIN = "admin"


class HookType(str, Enum):
    """Available hook types"""

    # Content hooks
    BEFORE_POST_CREATE = "before_post_create"
    AFTER_POST_CREATE = "after_post_create"
    BEFORE_POST_UPDATE = "before_post_update"
    AFTER_POST_UPDATE = "after_post_update"
    BEFORE_POST_DELETE = "before_post_delete"
    AFTER_POST_DELETE = "after_post_delete"
    POST_RENDER = "post_render"
    POST_VOTE = "post_vote"

    # Comment hooks
    BEFORE_COMMENT_CREATE = "before_comment_create"
    AFTER_COMMENT_CREATE = "after_comment_create"
    COMMENT_VOTE = "comment_vote"

    # User hooks
    USER_REGISTER = "user_register"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_PROFILE_UPDATE = "user_profile_update"
    USER_REPUTATION_CHANGE = "user_reputation_change"

    # Community hooks
    COMMUNITY_CREATE = "community_create"
    COMMUNITY_JOIN = "community_join"
    COMMUNITY_LEAVE = "community_leave"

    # Moderation hooks
    CONTENT_REPORT = "content_report"
    CONTENT_APPROVE = "content_approve"
    CONTENT_REMOVE = "content_remove"
    USER_BAN = "user_ban"
    USER_UNBAN = "user_unban"

    # Notification hooks
    SEND_NOTIFICATION = "send_notification"


class PluginContext:
    """Context object passed to plugin hooks"""

    def __init__(
        self,
        user_id: Optional[UUID] = None,
        community_id: Optional[UUID] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.user_id = user_id
        self.community_id = community_id
        self.data = data or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get data from context"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """Set data in context"""
        self.data[key] = value


class PluginBase(ABC):
    """
    Base class for all plugins

    Plugins must inherit from this class and implement required methods
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin with configuration

        Args:
            config: Plugin configuration dictionary
        """
        self.config = config
        self.enabled = True
        self._hooks: Dict[HookType, List[Callable]] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Plugin description"""
        pass

    @property
    @abstractmethod
    def plugin_type(self) -> PluginType:
        """Plugin type"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize plugin resources

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Cleanup plugin resources

        Returns:
            bool: True if shutdown successful, False otherwise
        """
        pass

    def register_hook(self, hook_type: HookType, handler: Callable):
        """
        Register a hook handler

        Args:
            hook_type: Type of hook to register
            handler: Async callable that handles the hook
        """
        if hook_type not in self._hooks:
            self._hooks[hook_type] = []
        self._hooks[hook_type].append(handler)

    async def execute_hook(self, hook_type: HookType, context: PluginContext) -> Any:
        """
        Execute all handlers for a hook type

        Args:
            hook_type: Type of hook to execute
            context: Plugin context with data

        Returns:
            Modified context or result from handlers
        """
        if hook_type not in self._hooks:
            return context

        for handler in self._hooks[hook_type]:
            result = await handler(context)
            if result is not None:
                context = result

        return context

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def validate_permissions(self, required_permissions: List[str]) -> bool:
        """
        Validate that plugin has required permissions

        Args:
            required_permissions: List of required permission strings

        Returns:
            bool: True if all permissions granted
        """
        plugin_permissions = self.config.get("permissions", [])
        return all(perm in plugin_permissions for perm in required_permissions)

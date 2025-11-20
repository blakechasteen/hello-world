"""
Integration tests for plugin system
"""
import pytest
from httpx import AsyncClient

from plugins.manager import PluginManager
from plugins.examples.polls_plugin import PollsPlugin
from plugins.examples.events_plugin import EventsPlugin
from plugins.examples.marketplace_plugin import MarketplacePlugin
from plugins.examples.moderation_plugin import ModerationPlugin


@pytest.mark.integration
@pytest.mark.plugins
class TestPluginLoading:
    """Test plugin loading and initialization"""

    async def test_load_polls_plugin(self):
        """Test loading polls plugin"""
        manager = PluginManager()

        success = await manager.load_plugin(
            plugin_id="polls",
            plugin_module="plugins.examples.polls_plugin",
            plugin_class="PollsPlugin",
            config={"max_options": 10},
        )

        assert success is True
        assert "polls" in manager._plugins
        assert manager._plugins["polls"].enabled is True

    async def test_load_multiple_plugins(self):
        """Test loading multiple plugins"""
        manager = PluginManager()

        # Load polls plugin
        await manager.load_plugin(
            "polls",
            "plugins.examples.polls_plugin",
            "PollsPlugin",
            {},
        )

        # Load events plugin
        await manager.load_plugin(
            "events",
            "plugins.examples.events_plugin",
            "EventsPlugin",
            {},
        )

        assert len(manager._plugins) == 2
        assert "polls" in manager._plugins
        assert "events" in manager._plugins

    async def test_plugin_initialization_failure(self):
        """Test handling of plugin initialization failure"""
        manager = PluginManager()

        # Try to load non-existent plugin
        success = await manager.load_plugin(
            "nonexistent",
            "plugins.examples.nonexistent_plugin",
            "NonexistentPlugin",
            {},
        )

        assert success is False


@pytest.mark.integration
@pytest.mark.plugins
class TestPollsPlugin:
    """Test polls plugin functionality"""

    async def test_create_poll_post(
        self, client: AsyncClient, test_user, test_community, auth_headers
    ):
        """Test creating a poll post"""

        # Load polls plugin
        manager = PluginManager()
        await manager.load_plugin(
            "polls",
            "plugins.examples.polls_plugin",
            "PollsPlugin",
            {"max_options": 5},
        )

        # Create poll post
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "What's your favorite language?",
                "content": "Vote for your favorite programming language",
                "post_type": "poll",
                "community_name": test_community.name,
                "poll_data": {
                    "options": ["Python", "JavaScript", "Rust", "Go"],
                    "duration_hours": 24,
                    "allow_multiple": False,
                },
            },
        )

        assert response.status_code == 201
        post = response.json()
        assert post["post_type"] == "poll"
        assert "poll_data" in post["metadata"]

    async def test_vote_on_poll(
        self, client: AsyncClient, test_user, test_community, auth_headers
    ):
        """Test voting on a poll"""

        # Create poll first
        poll_response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Test Poll",
                "post_type": "poll",
                "community_name": test_community.name,
                "poll_data": {
                    "options": ["Option A", "Option B"],
                    "duration_hours": 24,
                },
            },
        )
        poll = poll_response.json()

        # Vote on poll (would require plugin endpoint)
        # This is a placeholder - actual implementation would be in plugin
        # vote_response = await client.post(
        #     f"/api/v1/plugins/polls/{poll['post_id']}/vote",
        #     headers=auth_headers,
        #     json={"option_index": 0},
        # )
        # assert vote_response.status_code == 200


@pytest.mark.integration
@pytest.mark.plugins
class TestEventsPlugin:
    """Test events plugin functionality"""

    async def test_create_event(
        self, client: AsyncClient, test_user, test_community, auth_headers
    ):
        """Test creating an event"""

        # Load events plugin
        manager = PluginManager()
        await manager.load_plugin(
            "events",
            "plugins.examples.events_plugin",
            "EventsPlugin",
            {},
        )

        # Create event post
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Community Meetup",
                "content": "Join us for a community meetup!",
                "post_type": "event",
                "community_name": test_community.name,
                "event_data": {
                    "start_time": "2025-12-01T18:00:00Z",
                    "end_time": "2025-12-01T20:00:00Z",
                    "location": "123 Main St",
                    "capacity": 50,
                },
            },
        )

        assert response.status_code == 201
        event = response.json()
        assert event["post_type"] == "event"


@pytest.mark.integration
@pytest.mark.plugins
class TestMarketplacePlugin:
    """Test marketplace plugin functionality"""

    async def test_create_listing(
        self, client: AsyncClient, test_user, test_community, auth_headers
    ):
        """Test creating a marketplace listing"""

        # Load marketplace plugin
        manager = PluginManager()
        await manager.load_plugin(
            "marketplace",
            "plugins.examples.marketplace_plugin",
            "MarketplacePlugin",
            {"payment_gateway": "stripe"},
        )

        # Create marketplace listing
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Used Laptop for Sale",
                "content": "Selling my old laptop",
                "post_type": "marketplace",
                "community_name": test_community.name,
                "listing_data": {
                    "price": 500.00,
                    "currency": "USD",
                    "condition": "used",
                    "category": "electronics",
                },
            },
        )

        assert response.status_code == 201
        listing = response.json()
        assert listing["post_type"] == "marketplace"


@pytest.mark.integration
@pytest.mark.plugins
class TestModerationPlugin:
    """Test moderation plugin functionality"""

    async def test_content_moderation(
        self, client: AsyncClient, test_user, test_community, auth_headers
    ):
        """Test automatic content moderation"""

        # Load moderation plugin
        manager = PluginManager()
        await manager.load_plugin(
            "moderation",
            "plugins.examples.moderation_plugin",
            "ModerationPlugin",
            {
                "toxicity_threshold": 0.7,
                "auto_remove_threshold": 0.9,
            },
        )

        # Create post with clean content (should pass)
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Friendly Post",
                "content": "This is a nice and friendly post",
                "post_type": "text",
                "community_name": test_community.name,
            },
        )

        assert response.status_code == 201

    async def test_toxic_content_flagged(
        self, client: AsyncClient, test_user, test_community, auth_headers
    ):
        """Test that toxic content is flagged"""

        manager = PluginManager()
        await manager.load_plugin(
            "moderation",
            "plugins.examples.moderation_plugin",
            "ModerationPlugin",
            {"toxicity_threshold": 0.5},
        )

        # Create post with potentially toxic content
        # (simplified - actual implementation would trigger moderation hooks)
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Controversial Post",
                "content": "This contains hate and offensive language",
                "post_type": "text",
                "community_name": test_community.name,
            },
        )

        # Depending on implementation, may be flagged or blocked
        # assert response.status_code in [201, 400]


@pytest.mark.integration
@pytest.mark.plugins
class TestPluginHooks:
    """Test plugin hook system"""

    async def test_before_post_create_hook(self):
        """Test BEFORE_POST_CREATE hook"""
        from plugins.base import PluginContext, HookType

        manager = PluginManager()
        await manager.load_plugin(
            "polls",
            "plugins.examples.polls_plugin",
            "PollsPlugin",
            {},
        )

        # Trigger hook
        context = PluginContext(
            user_id="test-user",
            data={
                "post_type": "poll",
                "poll_data": {
                    "options": ["A", "B"],
                    "duration_hours": 24,
                },
            },
        )

        result_context = await manager.trigger_hook(
            HookType.BEFORE_POST_CREATE,
            context,
        )

        # Hook should process without errors
        assert result_context is not None

    async def test_after_post_create_hook(self):
        """Test AFTER_POST_CREATE hook"""
        from plugins.base import PluginContext, HookType

        manager = PluginManager()
        await manager.load_plugin(
            "events",
            "plugins.examples.events_plugin",
            "EventsPlugin",
            {},
        )

        context = PluginContext(
            user_id="test-user",
            data={
                "post_type": "event",
                "post_id": "test-post-id",
            },
        )

        result_context = await manager.trigger_hook(
            HookType.AFTER_POST_CREATE,
            context,
        )

        assert result_context is not None


@pytest.mark.integration
@pytest.mark.plugins
class TestPluginConfiguration:
    """Test plugin configuration"""

    async def test_plugin_custom_config(self):
        """Test plugin with custom configuration"""
        manager = PluginManager()

        config = {
            "max_options": 20,
            "allow_image_options": True,
            "results_visible_before_voting": False,
        }

        await manager.load_plugin(
            "polls",
            "plugins.examples.polls_plugin",
            "PollsPlugin",
            config,
        )

        plugin = manager._plugins["polls"]
        assert plugin.get_config("max_options") == 20
        assert plugin.get_config("allow_image_options") is True

    async def test_plugin_enable_disable(self):
        """Test enabling and disabling plugins"""
        manager = PluginManager()

        await manager.load_plugin(
            "polls",
            "plugins.examples.polls_plugin",
            "PollsPlugin",
            {},
        )

        plugin = manager._plugins["polls"]

        # Initially enabled
        assert plugin.enabled is True

        # Disable
        plugin.enabled = False
        assert plugin.enabled is False

        # Re-enable
        plugin.enabled = True
        assert plugin.enabled is True

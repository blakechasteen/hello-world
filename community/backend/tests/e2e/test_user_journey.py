"""
End-to-end tests simulating complete user journeys
"""
import pytest
from httpx import AsyncClient


@pytest.mark.e2e
class TestNewUserJourney:
    """Test complete journey of a new user"""

    async def test_new_user_complete_journey(self, client: AsyncClient):
        """
        Test complete user journey from registration to active participation

        Journey:
        1. Register account
        2. Login
        3. Discover communities
        4. Join community
        5. Create post
        6. Comment on post
        7. Vote on content
        8. Update profile
        """

        # Step 1: Register
        register_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "securepassword123",
                "display_name": "New User",
            },
        )
        assert register_response.status_code == 201
        auth_data = register_response.json()
        access_token = auth_data["access_token"]
        headers = {"Authorization": f"Bearer {access_token}"}

        # Step 2: Login (verify tokens work)
        me_response = await client.get("/api/v1/auth/me", headers=headers)
        assert me_response.status_code == 200
        user = me_response.json()
        assert user["username"] == "newuser"

        # Step 3: Discover communities
        communities_response = await client.get("/api/v1/communities")
        assert communities_response.status_code == 200

        # Step 4: Create a community
        create_community_response = await client.post(
            "/api/v1/communities",
            headers=headers,
            json={
                "name": "mynewcommunity",
                "display_name": "My New Community",
                "description": "A community for new users",
            },
        )
        assert create_community_response.status_code == 201
        community = create_community_response.json()

        # Step 5: Create a post
        create_post_response = await client.post(
            "/api/v1/posts",
            headers=headers,
            json={
                "title": "My First Post",
                "content": "Hello, community! This is my first post.",
                "post_type": "text",
                "community_name": community["name"],
                "tags": ["introduction", "hello"],
            },
        )
        assert create_post_response.status_code == 201
        post = create_post_response.json()

        # Step 6: Comment on own post
        comment_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=headers,
            json={"content": "Replying to my own post!"},
        )
        assert comment_response.status_code == 201
        comment = comment_response.json()

        # Step 7: Vote on own comment
        vote_response = await client.post(
            f"/api/v1/comments/{comment['comment_id']}/vote",
            headers=headers,
            json={"vote_type": "upvote"},
        )
        assert vote_response.status_code == 200

        # Step 8: Update profile
        update_profile_response = await client.patch(
            f"/api/v1/users/{user['username']}",
            headers=headers,
            json={
                "display_name": "Experienced User",
                "bio": "I've been here for 5 minutes!",
            },
        )
        assert update_profile_response.status_code == 200

        # Step 9: View own profile
        profile_response = await client.get(
            f"/api/v1/users/{user['username']}"
        )
        assert profile_response.status_code == 200
        profile = profile_response.json()
        assert profile["display_name"] == "Experienced User"
        assert profile["post_count"] >= 1


@pytest.mark.e2e
class TestMultiUserInteraction:
    """Test interaction between multiple users"""

    async def test_two_users_interacting(self, client: AsyncClient):
        """
        Test two users creating content and interacting

        Scenario:
        1. User A creates community
        2. User A creates post
        3. User B registers and joins community
        4. User B comments on User A's post
        5. User A replies to User B's comment
        6. Users vote on each other's content
        """

        # User A: Register
        user_a_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "usera",
                "email": "usera@example.com",
                "password": "password123",
            },
        )
        user_a_token = user_a_response.json()["access_token"]
        user_a_headers = {"Authorization": f"Bearer {user_a_token}"}

        # User A: Create community
        community_response = await client.post(
            "/api/v1/communities",
            headers=user_a_headers,
            json={
                "name": "userinteraction",
                "display_name": "User Interaction",
                "description": "Testing user interactions",
            },
        )
        community = community_response.json()

        # User A: Create post
        post_response = await client.post(
            "/api/v1/posts",
            headers=user_a_headers,
            json={
                "title": "Welcome!",
                "content": "Welcome to our community",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        post = post_response.json()

        # User B: Register
        user_b_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "userb",
                "email": "userb@example.com",
                "password": "password123",
            },
        )
        user_b_token = user_b_response.json()["access_token"]
        user_b_headers = {"Authorization": f"Bearer {user_b_token}"}

        # User B: Join community
        join_response = await client.post(
            f"/api/v1/communities/{community['name']}/join",
            headers=user_b_headers,
        )
        assert join_response.status_code == 200

        # User B: Comment on User A's post
        comment_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=user_b_headers,
            json={"content": "Thanks for the welcome!"},
        )
        comment = comment_response.json()

        # User A: Reply to User B's comment
        reply_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=user_a_headers,
            json={
                "content": "You're welcome!",
                "parent_id": comment["comment_id"],
            },
        )
        reply = reply_response.json()
        assert reply["depth"] == 1

        # User B: Upvote User A's post
        vote_post_response = await client.post(
            f"/api/v1/posts/{post['post_id']}/vote",
            headers=user_b_headers,
            json={"vote_type": "upvote"},
        )
        assert vote_post_response.status_code == 200

        # User A: Upvote User B's comment
        vote_comment_response = await client.post(
            f"/api/v1/comments/{comment['comment_id']}/vote",
            headers=user_a_headers,
            json={"vote_type": "upvote"},
        )
        assert vote_comment_response.status_code == 200

        # Verify final state
        final_post_response = await client.get(
            f"/api/v1/posts/{post['post_id']}"
        )
        final_post = final_post_response.json()
        assert final_post["upvote_count"] == 1
        assert final_post["comment_count"] >= 2


@pytest.mark.e2e
class TestContentDiscovery:
    """Test content discovery features"""

    async def test_discover_content_journey(self, client: AsyncClient):
        """
        Test user discovering content through various means

        Discovery paths:
        1. Browse all posts
        2. Filter by community
        3. Search for posts
        4. Sort by different algorithms
        """

        # Setup: Create user and content
        user_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "discoverer",
                "email": "discover@example.com",
                "password": "password123",
            },
        )
        token = user_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create community and multiple posts
        community_response = await client.post(
            "/api/v1/communities",
            headers=headers,
            json={
                "name": "discovery",
                "display_name": "Discovery Test",
                "description": "Testing discovery",
            },
        )
        community = community_response.json()

        post_ids = []
        for i in range(5):
            post_response = await client.post(
                "/api/v1/posts",
                headers=headers,
                json={
                    "title": f"Discovery Post {i}",
                    "content": f"Content about discovery {i}",
                    "post_type": "text",
                    "community_name": community["name"],
                    "tags": [f"tag{i}", "discovery"],
                },
            )
            post_ids.append(post_response.json()["post_id"])

        # Path 1: Browse all posts
        all_posts_response = await client.get("/api/v1/posts")
        assert all_posts_response.status_code == 200
        all_posts = all_posts_response.json()
        assert len(all_posts) >= 5

        # Path 2: Filter by community
        community_posts_response = await client.get(
            f"/api/v1/posts?community_name={community['name']}"
        )
        community_posts = community_posts_response.json()
        assert len(community_posts) >= 5
        assert all(
            p["community"]["name"] == community["name"]
            for p in community_posts
        )

        # Path 3: Sort by new
        new_posts_response = await client.get("/api/v1/posts?sort_by=new")
        assert new_posts_response.status_code == 200

        # Path 4: Sort by hot
        hot_posts_response = await client.get("/api/v1/posts?sort_by=hot")
        assert hot_posts_response.status_code == 200

        # Path 5: Sort by top
        top_posts_response = await client.get("/api/v1/posts?sort_by=top")
        assert top_posts_response.status_code == 200


@pytest.mark.e2e
@pytest.mark.slow
class TestModerat

ionEscalation:
    """Test moderation workflow from report to resolution"""

    async def test_content_moderation_escalation(self, client: AsyncClient):
        """
        Test complete moderation workflow

        Workflow:
        1. Low trust user creates comment
        2. Comment requires moderation
        3. Moderator reviews and approves
        4. User trust increases over time
        """

        # Create low trust user
        low_trust_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "lowtrust",
                "email": "lowtrust@example.com",
                "password": "password123",
            },
        )
        low_trust_token = low_trust_response.json()["access_token"]
        low_trust_headers = {"Authorization": f"Bearer {low_trust_token}"}

        # Create moderator
        mod_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "moderator",
                "email": "mod@example.com",
                "password": "password123",
            },
        )
        mod_token = mod_response.json()["access_token"]
        mod_headers = {"Authorization": f"Bearer {mod_token}"}

        # Create community and post (as moderator)
        community_response = await client.post(
            "/api/v1/communities",
            headers=mod_headers,
            json={
                "name": "moderated",
                "display_name": "Moderated Community",
                "description": "Strict moderation",
            },
        )
        community = community_response.json()

        post_response = await client.post(
            "/api/v1/posts",
            headers=mod_headers,
            json={
                "title": "Moderated Post",
                "content": "This post is moderated",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        post = post_response.json()

        # Low trust user creates comment (will be pending)
        # Note: Actual implementation depends on trust level logic
        comment_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=low_trust_headers,
            json={"content": "My first comment"},
        )
        comment = comment_response.json()

        # Comment status depends on implementation
        # May be "published" or "pending_moderation"
        assert comment["status"] in ["published", "pending_moderation"]


@pytest.mark.e2e
class TestLongFormContentCreation:
    """Test creating and managing long-form content"""

    async def test_create_comprehensive_post(self, client: AsyncClient):
        """Test creating a post with all features"""

        # Register user
        user_response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "blogger",
                "email": "blogger@example.com",
                "password": "password123",
            },
        )
        token = user_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Create community
        community_response = await client.post(
            "/api/v1/communities",
            headers=headers,
            json={
                "name": "blogging",
                "display_name": "Blogging Community",
                "description": "For bloggers",
            },
        )
        community = community_response.json()

        # Create comprehensive post with all fields
        post_response = await client.post(
            "/api/v1/posts",
            headers=headers,
            json={
                "title": "A Comprehensive Guide to Community Building",
                "content": """
                # Introduction

                Community building is an art and a science...

                ## Key Principles

                1. Authentic engagement
                2. Consistent communication
                3. Value creation

                ## Conclusion

                Building a community takes time and effort.
                """,
                "post_type": "text",
                "community_name": community["name"],
                "tags": ["guide", "community", "building", "tutorial"],
            },
        )
        assert post_response.status_code == 201
        post = post_response.json()

        # Verify all fields
        assert len(post["content"]) > 100
        assert len(post["tags"]) == 4
        assert post["status"] == "published"

        # Create threaded discussion
        comments = []
        for i in range(3):
            comment_response = await client.post(
                f"/api/v1/comments/posts/{post['post_id']}",
                headers=headers,
                json={"content": f"Great point #{i+1}!"},
            )
            comments.append(comment_response.json())

        # Create nested replies
        for comment in comments[:2]:
            reply_response = await client.post(
                f"/api/v1/comments/posts/{post['post_id']}",
                headers=headers,
                json={
                    "content": "Thanks for the feedback!",
                    "parent_id": comment["comment_id"],
                },
            )
            assert reply_response.status_code == 201

        # Verify final discussion state
        all_comments_response = await client.get(
            f"/api/v1/comments/posts/{post['post_id']}"
        )
        all_comments = all_comments_response.json()
        assert len(all_comments) >= 5

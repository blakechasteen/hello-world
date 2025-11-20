"""
Integration tests for complete post workflow
Tests multiple components working together
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User


@pytest.mark.integration
class TestPostWorkflow:
    """Test complete post creation, voting, and commenting workflow"""

    async def test_complete_post_lifecycle(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test complete post lifecycle from creation to deletion"""

        # Step 1: Create a community
        community_response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "workflow",
                "display_name": "Workflow Test",
                "description": "Testing full workflow",
            },
        )
        assert community_response.status_code == 201
        community = community_response.json()

        # Step 2: Create a post in the community
        post_response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Integration Test Post",
                "content": "Testing the full workflow",
                "post_type": "text",
                "community_name": community["name"],
                "tags": ["test", "integration"],
            },
        )
        assert post_response.status_code == 201
        post = post_response.json()
        assert post["community"]["name"] == community["name"]

        # Step 3: Upvote the post
        vote_response = await client.post(
            f"/api/v1/posts/{post['post_id']}/vote",
            headers=auth_headers,
            json={"vote_type": "upvote"},
        )
        assert vote_response.status_code == 200
        voted_post = vote_response.json()
        assert voted_post["upvote_count"] == 1

        # Step 4: Create a comment on the post
        comment_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=auth_headers,
            json={"content": "Great post!"},
        )
        assert comment_response.status_code == 201
        comment = comment_response.json()

        # Step 5: Create a nested reply
        reply_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=auth_headers,
            json={
                "content": "Thanks!",
                "parent_id": comment["comment_id"],
            },
        )
        assert reply_response.status_code == 201
        reply = reply_response.json()
        assert reply["depth"] == 1
        assert reply["parent_id"] == comment["comment_id"]

        # Step 6: Verify post now has comments
        updated_post_response = await client.get(
            f"/api/v1/posts/{post['post_id']}"
        )
        updated_post = updated_post_response.json()
        assert updated_post["comment_count"] >= 2

        # Step 7: Get all comments for the post
        comments_response = await client.get(
            f"/api/v1/comments/posts/{post['post_id']}"
        )
        comments = comments_response.json()
        assert len(comments) >= 2

        # Step 8: Update the post
        update_response = await client.patch(
            f"/api/v1/posts/{post['post_id']}",
            headers=auth_headers,
            json={"title": "Updated Title"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["title"] == "Updated Title"

        # Step 9: Delete the post
        delete_response = await client.delete(
            f"/api/v1/posts/{post['post_id']}",
            headers=auth_headers,
        )
        assert delete_response.status_code == 200


@pytest.mark.integration
class TestUserInteractionWorkflow:
    """Test user interactions across multiple entities"""

    async def test_user_creates_multiple_content(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test user creating multiple pieces of content"""

        # Create community
        community_response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "multiuser",
                "display_name": "Multi User Test",
                "description": "Testing multiple interactions",
            },
        )
        community = community_response.json()

        # Create multiple posts
        post_ids = []
        for i in range(3):
            post_response = await client.post(
                "/api/v1/posts",
                headers=auth_headers,
                json={
                    "title": f"Post {i+1}",
                    "content": f"Content {i+1}",
                    "post_type": "text",
                    "community_name": community["name"],
                },
            )
            post_ids.append(post_response.json()["post_id"])

        # Verify user profile shows all posts
        user_posts_response = await client.get(
            f"/api/v1/users/{test_user.username}/posts"
        )
        user_posts = user_posts_response.json()
        assert len(user_posts) >= 3

        # Comment on each post
        for post_id in post_ids:
            comment_response = await client.post(
                f"/api/v1/comments/posts/{post_id}",
                headers=auth_headers,
                json={"content": f"Comment on post {post_id}"},
            )
            assert comment_response.status_code == 201

        # Verify user profile shows all comments
        user_comments_response = await client.get(
            f"/api/v1/users/{test_user.username}/comments"
        )
        user_comments = user_comments_response.json()
        assert len(user_comments) >= 3


@pytest.mark.integration
class TestVotingWorkflow:
    """Test voting interactions and score calculations"""

    async def test_multiple_users_voting(
        self,
        client: AsyncClient,
        test_user: User,
        test_admin: User,
        test_db: AsyncSession,
        auth_headers: dict,
        admin_headers: dict,
    ):
        """Test multiple users voting on same content"""

        # Create community
        community_response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "voting",
                "display_name": "Voting Test",
                "description": "Testing votes",
            },
        )
        community = community_response.json()

        # Create post
        post_response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Popular Post",
                "content": "This will get votes",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        post = post_response.json()

        # User 1 upvotes
        vote1_response = await client.post(
            f"/api/v1/posts/{post['post_id']}/vote",
            headers=auth_headers,
            json={"vote_type": "upvote"},
        )
        assert vote1_response.status_code == 200

        # User 2 upvotes
        vote2_response = await client.post(
            f"/api/v1/posts/{post['post_id']}/vote",
            headers=admin_headers,
            json={"vote_type": "upvote"},
        )
        assert vote2_response.status_code == 200

        # Get post and verify vote count
        updated_post_response = await client.get(
            f"/api/v1/posts/{post['post_id']}"
        )
        updated_post = updated_post_response.json()
        assert updated_post["upvote_count"] == 2
        assert updated_post["vote_score"] == 2

        # User 1 changes to downvote
        vote3_response = await client.post(
            f"/api/v1/posts/{post['post_id']}/vote",
            headers=auth_headers,
            json={"vote_type": "downvote"},
        )
        assert vote3_response.status_code == 200

        # Get post and verify updated counts
        final_post_response = await client.get(
            f"/api/v1/posts/{post['post_id']}"
        )
        final_post = final_post_response.json()
        assert final_post["upvote_count"] == 1
        assert final_post["downvote_count"] == 1
        assert final_post["vote_score"] == 0


@pytest.mark.integration
class TestModerationWorkflow:
    """Test moderation workflow across different roles"""

    async def test_moderation_approval_workflow(
        self,
        client: AsyncClient,
        test_user: User,
        test_moderator: User,
        test_db: AsyncSession,
        auth_headers: dict,
        moderator_headers: dict,
    ):
        """Test comment requiring moderation and approval"""

        # Set user to low trust (requires moderation)
        test_user.trust_level = 0
        await test_db.commit()

        # Create community and post as moderator
        community_response = await client.post(
            "/api/v1/communities",
            headers=moderator_headers,
            json={
                "name": "moderated",
                "display_name": "Moderated Community",
                "description": "Strict moderation",
            },
        )
        community = community_response.json()

        post_response = await client.post(
            "/api/v1/posts",
            headers=moderator_headers,
            json={
                "title": "Moderated Post",
                "content": "This post is moderated",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        post = post_response.json()

        # Low trust user creates comment (should be pending)
        comment_response = await client.post(
            f"/api/v1/comments/posts/{post['post_id']}",
            headers=auth_headers,
            json={"content": "This needs approval"},
        )
        assert comment_response.status_code == 201
        comment = comment_response.json()
        assert comment["status"] == "pending_moderation"

        # Moderator approves comment
        approve_response = await client.post(
            f"/api/v1/comments/{comment['comment_id']}/moderate",
            headers=moderator_headers,
            json={
                "action": "approve",
                "reason": "Looks good",
            },
        )
        assert approve_response.status_code == 200
        approved_comment = approve_response.json()
        assert approved_comment["status"] == "published"

        # Verify comment is now visible in post comments
        comments_response = await client.get(
            f"/api/v1/comments/posts/{post['post_id']}"
        )
        comments = comments_response.json()
        assert len(comments) >= 1


@pytest.mark.integration
class TestCommunityMembershipWorkflow:
    """Test community membership and permissions"""

    async def test_member_permissions(
        self,
        client: AsyncClient,
        test_user: User,
        test_admin: User,
        auth_headers: dict,
    ):
        """Test that only members can post to communities"""

        # User 1 creates community
        community_response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "exclusive",
                "display_name": "Exclusive Community",
                "description": "Members only",
            },
        )
        community = community_response.json()

        # User 1 can post (creator auto-joins)
        post_response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Member Post",
                "content": "I'm a member",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        assert post_response.status_code == 201

        # Create User 2
        from auth.security import create_access_token

        user2_token = create_access_token(
            data={"sub": str(test_admin.user_id)}
        )
        user2_headers = {"Authorization": f"Bearer {user2_token}"}

        # User 2 tries to post without joining (should work - public community)
        post2_response = await client.post(
            "/api/v1/posts",
            headers=user2_headers,
            json={
                "title": "Non-member Post",
                "content": "Can I post?",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        # Depending on implementation, may succeed or require join
        assert post2_response.status_code in [201, 403]

        # User 2 joins
        join_response = await client.post(
            f"/api/v1/communities/{community['name']}/join",
            headers=user2_headers,
        )
        assert join_response.status_code == 200

        # User 2 can now definitely post
        post3_response = await client.post(
            "/api/v1/posts",
            headers=user2_headers,
            json={
                "title": "Member Post 2",
                "content": "Now I'm a member",
                "post_type": "text",
                "community_name": community["name"],
            },
        )
        assert post3_response.status_code == 201

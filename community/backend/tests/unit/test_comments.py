"""
Unit tests for comment endpoints
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.post import Post
from models.comment import Comment


@pytest.mark.unit
@pytest.mark.api
class TestCreateComment:
    """Tests for creating comments"""

    async def test_create_comment(
        self,
        client: AsyncClient,
        test_post: Post,
        test_user: User,
        auth_headers: dict,
    ):
        """Test creating a comment"""
        response = await client.post(
            f"/api/v1/comments/posts/{test_post.post_id}",
            headers=auth_headers,
            json={"content": "This is my comment"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "This is my comment"
        assert data["author"]["username"] == "testuser"
        assert data["depth"] == 0
        assert data["path"] == "root"

    async def test_create_nested_comment(
        self,
        client: AsyncClient,
        test_post: Post,
        test_comment: Comment,
        auth_headers: dict,
    ):
        """Test creating a nested comment (reply)"""
        response = await client.post(
            f"/api/v1/comments/posts/{test_post.post_id}",
            headers=auth_headers,
            json={
                "content": "This is a reply",
                "parent_id": str(test_comment.comment_id),
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["content"] == "This is a reply"
        assert data["parent_id"] == str(test_comment.comment_id)
        assert data["depth"] == 1
        assert data["path"].startswith("root.")

    async def test_create_comment_without_auth(
        self, client: AsyncClient, test_post: Post
    ):
        """Test creating comment without authentication"""
        response = await client.post(
            f"/api/v1/comments/posts/{test_post.post_id}",
            json={"content": "Unauthorized comment"},
        )

        assert response.status_code == 403

    async def test_create_comment_on_locked_post(
        self,
        client: AsyncClient,
        test_post: Post,
        test_db: AsyncSession,
        auth_headers: dict,
    ):
        """Test creating comment on locked post"""
        # Lock the post
        test_post.locked = True
        await test_db.commit()

        response = await client.post(
            f"/api/v1/comments/posts/{test_post.post_id}",
            headers=auth_headers,
            json={"content": "Comment on locked post"},
        )

        assert response.status_code == 403

    async def test_create_comment_max_depth(
        self,
        client: AsyncClient,
        test_post: Post,
        test_db: AsyncSession,
        auth_headers: dict,
    ):
        """Test comment depth limit"""
        # Create nested comments up to max depth
        parent = None
        for i in range(10):
            comment = Comment(
                post_id=test_post.post_id,
                author_id=test_post.author_id,
                content=f"Comment {i}",
                parent_id=parent.comment_id if parent else None,
                path=f"{parent.path}.{parent.comment_id}" if parent else "root",
                depth=i,
                status="published",
            )
            test_db.add(comment)
            await test_db.commit()
            await test_db.refresh(comment)
            parent = comment

        # Try to add one more (should fail)
        response = await client.post(
            f"/api/v1/comments/posts/{test_post.post_id}",
            headers=auth_headers,
            json={
                "content": "Too deep",
                "parent_id": str(parent.comment_id),
            },
        )

        assert response.status_code == 400
        assert "max depth" in response.json()["detail"].lower()

    async def test_create_comment_auto_moderation(
        self,
        client: AsyncClient,
        test_post: Post,
        test_user: User,
        test_db: AsyncSession,
        auth_headers: dict,
    ):
        """Test auto-moderation for low trust users"""
        # Set user to low trust level
        test_user.trust_level = 0
        await test_db.commit()

        response = await client.post(
            f"/api/v1/comments/posts/{test_post.post_id}",
            headers=auth_headers,
            json={"content": "Comment from low trust user"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending_moderation"


@pytest.mark.unit
@pytest.mark.api
class TestGetComments:
    """Tests for getting comments"""

    async def test_get_post_comments(
        self, client: AsyncClient, test_post: Post, test_comment: Comment
    ):
        """Test getting comments for a post"""
        response = await client.get(
            f"/api/v1/comments/posts/{test_post.post_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(
            c["comment_id"] == str(test_comment.comment_id) for c in data
        )

    async def test_get_comments_sort_by_best(
        self, client: AsyncClient, test_post: Post, test_comment: Comment
    ):
        """Test getting comments sorted by best"""
        response = await client.get(
            f"/api/v1/comments/posts/{test_post.post_id}?sort_by=best"
        )

        assert response.status_code == 200

    async def test_get_comments_sort_by_new(
        self, client: AsyncClient, test_post: Post, test_comment: Comment
    ):
        """Test getting comments sorted by newest"""
        response = await client.get(
            f"/api/v1/comments/posts/{test_post.post_id}?sort_by=new"
        )

        assert response.status_code == 200

    async def test_get_comments_sort_by_top(
        self, client: AsyncClient, test_post: Post, test_comment: Comment
    ):
        """Test getting comments sorted by top"""
        response = await client.get(
            f"/api/v1/comments/posts/{test_post.post_id}?sort_by=top"
        )

        assert response.status_code == 200

    async def test_get_comments_sort_by_controversial(
        self, client: AsyncClient, test_post: Post, test_comment: Comment
    ):
        """Test getting comments sorted by controversial"""
        response = await client.get(
            f"/api/v1/comments/posts/{test_post.post_id}?sort_by=controversial"
        )

        assert response.status_code == 200

    async def test_get_single_comment(
        self, client: AsyncClient, test_comment: Comment
    ):
        """Test getting a single comment by ID"""
        response = await client.get(
            f"/api/v1/comments/{test_comment.comment_id}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["comment_id"] == str(test_comment.comment_id)
        assert data["content"] == test_comment.content

    async def test_get_nonexistent_comment(self, client: AsyncClient):
        """Test getting a non-existent comment"""
        import uuid

        fake_id = uuid.uuid4()
        response = await client.get(f"/api/v1/comments/{fake_id}")

        assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.api
class TestVoteComments:
    """Tests for voting on comments"""

    async def test_upvote_comment(
        self, client: AsyncClient, test_comment: Comment, auth_headers: dict
    ):
        """Test upvoting a comment"""
        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/vote",
            headers=auth_headers,
            json={"vote_type": "upvote"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["upvote_count"] == 1
        assert data["vote_score"] == 1

    async def test_downvote_comment(
        self, client: AsyncClient, test_comment: Comment, auth_headers: dict
    ):
        """Test downvoting a comment"""
        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/vote",
            headers=auth_headers,
            json={"vote_type": "downvote"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["downvote_count"] == 1
        assert data["vote_score"] == -1

    async def test_vote_without_auth(
        self, client: AsyncClient, test_comment: Comment
    ):
        """Test voting without authentication"""
        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/vote",
            json={"vote_type": "upvote"},
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestUpdateComment:
    """Tests for updating comments"""

    async def test_update_own_comment(
        self,
        client: AsyncClient,
        test_comment: Comment,
        auth_headers: dict,
    ):
        """Test updating own comment"""
        response = await client.patch(
            f"/api/v1/comments/{test_comment.comment_id}",
            headers=auth_headers,
            json={"content": "Updated comment content"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Updated comment content"
        assert data["edited"] is True

    async def test_update_other_user_comment(
        self, client: AsyncClient, test_comment: Comment, test_admin: User
    ):
        """Test user cannot update another user's comment"""
        from auth.security import create_access_token

        other_user_token = create_access_token(
            data={"sub": str(test_admin.user_id)}
        )

        response = await client.patch(
            f"/api/v1/comments/{test_comment.comment_id}",
            headers={"Authorization": f"Bearer {other_user_token}"},
            json={"content": "Hacked content"},
        )

        assert response.status_code in [403, 404]

    async def test_update_comment_without_auth(
        self, client: AsyncClient, test_comment: Comment
    ):
        """Test updating comment without authentication"""
        response = await client.patch(
            f"/api/v1/comments/{test_comment.comment_id}",
            json={"content": "Unauthorized update"},
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestDeleteComment:
    """Tests for deleting comments"""

    async def test_delete_own_comment(
        self,
        client: AsyncClient,
        test_comment: Comment,
        auth_headers: dict,
    ):
        """Test deleting own comment (soft delete)"""
        response = await client.delete(
            f"/api/v1/comments/{test_comment.comment_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200

    async def test_delete_other_user_comment(
        self, client: AsyncClient, test_comment: Comment, test_admin: User
    ):
        """Test user cannot delete another user's comment"""
        from auth.security import create_access_token

        other_user_token = create_access_token(
            data={"sub": str(test_admin.user_id)}
        )

        response = await client.delete(
            f"/api/v1/comments/{test_comment.comment_id}",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )

        assert response.status_code in [403, 404]


@pytest.mark.unit
@pytest.mark.api
class TestCommentModeration:
    """Tests for comment moderation"""

    async def test_approve_comment(
        self,
        client: AsyncClient,
        test_comment: Comment,
        test_db: AsyncSession,
        moderator_headers: dict,
    ):
        """Test moderator can approve pending comment"""
        # Set comment to pending
        test_comment.status = "pending_moderation"
        await test_db.commit()

        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/moderate",
            headers=moderator_headers,
            json={
                "action": "approve",
                "reason": "Looks good",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "published"

    async def test_remove_comment(
        self,
        client: AsyncClient,
        test_comment: Comment,
        moderator_headers: dict,
    ):
        """Test moderator can remove comment"""
        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/moderate",
            headers=moderator_headers,
            json={
                "action": "remove",
                "reason": "Violates rules",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "removed"

    async def test_delete_comment_admin_only(
        self,
        client: AsyncClient,
        test_comment: Comment,
        admin_headers: dict,
    ):
        """Test only admin can permanently delete comment"""
        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/moderate",
            headers=admin_headers,
            json={
                "action": "delete",
                "reason": "Spam",
            },
        )

        assert response.status_code == 200

    async def test_moderate_without_permissions(
        self,
        client: AsyncClient,
        test_comment: Comment,
        auth_headers: dict,
    ):
        """Test regular user cannot moderate comments"""
        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/moderate",
            headers=auth_headers,
            json={
                "action": "remove",
                "reason": "Test",
            },
        )

        assert response.status_code == 403

    async def test_restore_comment(
        self,
        client: AsyncClient,
        test_comment: Comment,
        test_db: AsyncSession,
        moderator_headers: dict,
    ):
        """Test moderator can restore removed comment"""
        # Remove comment first
        test_comment.status = "removed"
        await test_db.commit()

        response = await client.post(
            f"/api/v1/comments/{test_comment.comment_id}/moderate",
            headers=moderator_headers,
            json={
                "action": "restore",
                "reason": "False positive",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "published"

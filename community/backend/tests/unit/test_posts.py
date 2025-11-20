"""
Unit tests for post endpoints
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.community import Community
from models.post import Post


@pytest.mark.unit
@pytest.mark.api
class TestCreatePost:
    """Tests for creating posts"""

    async def test_create_text_post(
        self,
        client: AsyncClient,
        test_user: User,
        test_community: Community,
        auth_headers: dict,
    ):
        """Test creating a text post"""
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "New Test Post",
                "content": "This is the content of my new post",
                "post_type": "text",
                "community_name": test_community.name,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "New Test Post"
        assert data["post_type"] == "text"
        assert data["status"] == "published"
        assert data["author"]["username"] == "testuser"

    async def test_create_link_post(
        self,
        client: AsyncClient,
        test_user: User,
        test_community: Community,
        auth_headers: dict,
    ):
        """Test creating a link post"""
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Interesting Article",
                "post_type": "link",
                "link_url": "https://example.com/article",
                "community_name": test_community.name,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["post_type"] == "link"
        assert "link_url" in data["metadata"]

    async def test_create_post_without_auth(
        self, client: AsyncClient, test_community: Community
    ):
        """Test creating post without authentication"""
        response = await client.post(
            "/api/v1/posts",
            json={
                "title": "Unauthorized Post",
                "content": "Should fail",
                "post_type": "text",
            },
        )

        assert response.status_code == 403

    async def test_create_post_with_tags(
        self,
        client: AsyncClient,
        test_user: User,
        test_community: Community,
        auth_headers: dict,
    ):
        """Test creating post with tags"""
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Tagged Post",
                "content": "Post with tags",
                "post_type": "text",
                "tags": ["python", "programming", "tutorial"],
                "community_name": test_community.name,
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert "python" in data["tags"]
        assert len(data["tags"]) == 3

    async def test_create_post_to_nonexistent_community(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test creating post to non-existent community"""
        response = await client.post(
            "/api/v1/posts",
            headers=auth_headers,
            json={
                "title": "Post",
                "content": "Content",
                "post_type": "text",
                "community_name": "nonexistent",
            },
        )

        assert response.status_code == 404


@pytest.mark.unit
@pytest.mark.api
class TestGetPosts:
    """Tests for getting posts"""

    async def test_get_all_posts(
        self, client: AsyncClient, test_post: Post
    ):
        """Test getting all posts"""
        response = await client.get("/api/v1/posts")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(p["post_id"] == str(test_post.post_id) for p in data)

    async def test_get_posts_pagination(
        self,
        client: AsyncClient,
        test_user: User,
        test_community: Community,
        test_db: AsyncSession,
    ):
        """Test post pagination"""
        # Create 25 posts
        for i in range(25):
            post = Post(
                title=f"Post {i}",
                content=f"Content {i}",
                post_type="text",
                author_id=test_user.user_id,
                community_id=test_community.community_id,
                status="published",
            )
            test_db.add(post)
        await test_db.commit()

        # Get first page
        response = await client.get("/api/v1/posts?page=1&page_size=20")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 20

        # Get second page
        response = await client.get("/api/v1/posts?page=2&page_size=20")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 5

    async def test_get_posts_by_community(
        self, client: AsyncClient, test_post: Post, test_community: Community
    ):
        """Test getting posts filtered by community"""
        response = await client.get(
            f"/api/v1/posts?community_name={test_community.name}"
        )

        assert response.status_code == 200
        data = response.json()
        assert all(
            p["community"]["name"] == test_community.name for p in data
        )

    async def test_get_posts_sort_by_new(
        self, client: AsyncClient, test_post: Post
    ):
        """Test getting posts sorted by newest"""
        response = await client.get("/api/v1/posts?sort_by=new")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    async def test_get_posts_sort_by_hot(
        self, client: AsyncClient, test_post: Post
    ):
        """Test getting posts sorted by hot"""
        response = await client.get("/api/v1/posts?sort_by=hot")

        assert response.status_code == 200

    async def test_get_posts_sort_by_top(
        self, client: AsyncClient, test_post: Post
    ):
        """Test getting posts sorted by top"""
        response = await client.get("/api/v1/posts?sort_by=top")

        assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.api
class TestGetSinglePost:
    """Tests for getting a single post"""

    async def test_get_post_by_id(
        self, client: AsyncClient, test_post: Post
    ):
        """Test getting a single post by ID"""
        response = await client.get(f"/api/v1/posts/{test_post.post_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["post_id"] == str(test_post.post_id)
        assert data["title"] == test_post.title

    async def test_get_nonexistent_post(self, client: AsyncClient):
        """Test getting a non-existent post"""
        import uuid

        fake_id = uuid.uuid4()
        response = await client.get(f"/api/v1/posts/{fake_id}")

        assert response.status_code == 404

    async def test_get_post_includes_author(
        self, client: AsyncClient, test_post: Post
    ):
        """Test that post includes author information"""
        response = await client.get(f"/api/v1/posts/{test_post.post_id}")

        assert response.status_code == 200
        data = response.json()
        assert "author" in data
        assert data["author"]["username"] == "testuser"

    async def test_get_post_includes_community(
        self, client: AsyncClient, test_post: Post
    ):
        """Test that post includes community information"""
        response = await client.get(f"/api/v1/posts/{test_post.post_id}")

        assert response.status_code == 200
        data = response.json()
        assert "community" in data
        assert data["community"]["name"] == "testcommunity"


@pytest.mark.unit
@pytest.mark.api
class TestVotePosts:
    """Tests for voting on posts"""

    async def test_upvote_post(
        self, client: AsyncClient, test_post: Post, auth_headers: dict
    ):
        """Test upvoting a post"""
        response = await client.post(
            f"/api/v1/posts/{test_post.post_id}/vote",
            headers=auth_headers,
            json={"vote_type": "upvote"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["upvote_count"] == 1
        assert data["vote_score"] == 1

    async def test_downvote_post(
        self, client: AsyncClient, test_post: Post, auth_headers: dict
    ):
        """Test downvoting a post"""
        response = await client.post(
            f"/api/v1/posts/{test_post.post_id}/vote",
            headers=auth_headers,
            json={"vote_type": "downvote"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["downvote_count"] == 1
        assert data["vote_score"] == -1

    async def test_change_vote(
        self, client: AsyncClient, test_post: Post, auth_headers: dict
    ):
        """Test changing vote from upvote to downvote"""
        # First upvote
        await client.post(
            f"/api/v1/posts/{test_post.post_id}/vote",
            headers=auth_headers,
            json={"vote_type": "upvote"},
        )

        # Then downvote
        response = await client.post(
            f"/api/v1/posts/{test_post.post_id}/vote",
            headers=auth_headers,
            json={"vote_type": "downvote"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["downvote_count"] == 1
        assert data["upvote_count"] == 0

    async def test_vote_without_auth(
        self, client: AsyncClient, test_post: Post
    ):
        """Test voting without authentication"""
        response = await client.post(
            f"/api/v1/posts/{test_post.post_id}/vote",
            json={"vote_type": "upvote"},
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestUpdatePost:
    """Tests for updating posts"""

    async def test_update_own_post(
        self, client: AsyncClient, test_post: Post, test_user: User, auth_headers: dict
    ):
        """Test updating own post"""
        response = await client.patch(
            f"/api/v1/posts/{test_post.post_id}",
            headers=auth_headers,
            json={
                "title": "Updated Title",
                "content": "Updated content",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        assert data["content"] == "Updated content"

    async def test_update_other_user_post(
        self,
        client: AsyncClient,
        test_post: Post,
        test_admin: User,
        test_db: AsyncSession,
    ):
        """Test user cannot update another user's post"""
        # Create second user
        from auth.security import create_access_token

        second_user_token = create_access_token(
            data={"sub": str(test_admin.user_id)}
        )

        response = await client.patch(
            f"/api/v1/posts/{test_post.post_id}",
            headers={"Authorization": f"Bearer {second_user_token}"},
            json={
                "title": "Hacked Title",
            },
        )

        assert response.status_code in [403, 404]

    async def test_update_post_without_auth(
        self, client: AsyncClient, test_post: Post
    ):
        """Test updating post without authentication"""
        response = await client.patch(
            f"/api/v1/posts/{test_post.post_id}",
            json={"title": "Unauthorized Update"},
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestDeletePost:
    """Tests for deleting posts"""

    async def test_delete_own_post(
        self, client: AsyncClient, test_post: Post, auth_headers: dict
    ):
        """Test deleting own post"""
        response = await client.delete(
            f"/api/v1/posts/{test_post.post_id}",
            headers=auth_headers,
        )

        assert response.status_code == 200

    async def test_delete_other_user_post(
        self, client: AsyncClient, test_post: Post, test_admin: User
    ):
        """Test user cannot delete another user's post"""
        from auth.security import create_access_token

        second_user_token = create_access_token(
            data={"sub": str(test_admin.user_id)}
        )

        response = await client.delete(
            f"/api/v1/posts/{test_post.post_id}",
            headers={"Authorization": f"Bearer {second_user_token}"},
        )

        assert response.status_code in [403, 404]

    async def test_admin_can_delete_any_post(
        self, client: AsyncClient, test_post: Post, admin_headers: dict
    ):
        """Test admin can delete any post"""
        response = await client.delete(
            f"/api/v1/posts/{test_post.post_id}",
            headers=admin_headers,
        )

        assert response.status_code == 200

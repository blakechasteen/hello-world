"""
Unit tests for user endpoints
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.post import Post


@pytest.mark.unit
@pytest.mark.api
class TestGetUser:
    """Tests for getting user profile"""

    async def test_get_user_by_username(
        self, client: AsyncClient, test_user: User
    ):
        """Test getting user profile by username"""
        response = await client.get(f"/api/v1/users/{test_user.username}")

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["display_name"] == "Test User"
        assert "password_hash" not in data

    async def test_get_nonexistent_user(self, client: AsyncClient):
        """Test getting non-existent user"""
        response = await client.get("/api/v1/users/nonexistent")

        assert response.status_code == 404

    async def test_get_user_stats(
        self, client: AsyncClient, test_user: User
    ):
        """Test user profile includes stats"""
        response = await client.get(f"/api/v1/users/{test_user.username}")

        assert response.status_code == 200
        data = response.json()
        assert "reputation_score" in data
        assert "trust_level" in data
        assert "post_count" in data
        assert "comment_count" in data


@pytest.mark.unit
@pytest.mark.api
class TestUpdateUser:
    """Tests for updating user profile"""

    async def test_update_own_profile(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test user can update their own profile"""
        response = await client.patch(
            f"/api/v1/users/{test_user.username}",
            headers=auth_headers,
            json={
                "display_name": "Updated Name",
                "bio": "Updated bio",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Updated Name"
        assert data["bio"] == "Updated bio"

    async def test_update_other_user_profile(
        self, client: AsyncClient, test_user: User, test_admin: User, auth_headers: dict
    ):
        """Test user cannot update another user's profile"""
        response = await client.patch(
            f"/api/v1/users/{test_admin.username}",
            headers=auth_headers,
            json={
                "display_name": "Hacked Name",
            },
        )

        assert response.status_code == 403

    async def test_update_without_auth(
        self, client: AsyncClient, test_user: User
    ):
        """Test updating profile without authentication"""
        response = await client.patch(
            f"/api/v1/users/{test_user.username}",
            json={
                "display_name": "New Name",
            },
        )

        assert response.status_code == 403

    async def test_update_email(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test updating email address"""
        response = await client.patch(
            f"/api/v1/users/{test_user.username}",
            headers=auth_headers,
            json={
                "email": "newemail@example.com",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newemail@example.com"

    async def test_update_to_duplicate_email(
        self,
        client: AsyncClient,
        test_user: User,
        test_admin: User,
        auth_headers: dict,
    ):
        """Test updating to an email that's already taken"""
        response = await client.patch(
            f"/api/v1/users/{test_user.username}",
            headers=auth_headers,
            json={
                "email": test_admin.email,  # Already taken
            },
        )

        assert response.status_code == 400


@pytest.mark.unit
@pytest.mark.api
class TestUserPosts:
    """Tests for getting user's posts"""

    async def test_get_user_posts(
        self,
        client: AsyncClient,
        test_user: User,
        test_post: Post,
    ):
        """Test getting posts by a user"""
        response = await client.get(f"/api/v1/users/{test_user.username}/posts")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["title"] == "Test Post"
        assert data[0]["author"]["username"] == "testuser"

    async def test_get_user_posts_pagination(
        self,
        client: AsyncClient,
        test_user: User,
        test_db: AsyncSession,
        test_community,
    ):
        """Test pagination for user posts"""
        # Create multiple posts
        for i in range(15):
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
        response = await client.get(
            f"/api/v1/users/{test_user.username}/posts?page=1&page_size=10"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10

        # Get second page
        response = await client.get(
            f"/api/v1/users/{test_user.username}/posts?page=2&page_size=10"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 5

    async def test_get_user_posts_no_posts(
        self, client: AsyncClient, test_user: User
    ):
        """Test getting posts for user with no posts"""
        # Create new user with no posts
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "username": "noposts",
                "email": "noposts@example.com",
                "password": "password123",
            },
        )

        response = await client.get("/api/v1/users/noposts/posts")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0


@pytest.mark.unit
@pytest.mark.api
class TestUserComments:
    """Tests for getting user's comments"""

    async def test_get_user_comments(
        self,
        client: AsyncClient,
        test_user: User,
        test_comment,
    ):
        """Test getting comments by a user"""
        response = await client.get(
            f"/api/v1/users/{test_user.username}/comments"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["content"] == "This is a test comment"
        assert data[0]["author"]["username"] == "testuser"

    async def test_get_user_comments_pagination(
        self,
        client: AsyncClient,
        test_user: User,
        test_post: Post,
        test_db: AsyncSession,
    ):
        """Test pagination for user comments"""
        # Create multiple comments
        for i in range(25):
            from models.comment import Comment

            comment = Comment(
                post_id=test_post.post_id,
                author_id=test_user.user_id,
                content=f"Comment {i}",
                path="root",
                depth=0,
                status="published",
            )
            test_db.add(comment)
        await test_db.commit()

        # Get first page
        response = await client.get(
            f"/api/v1/users/{test_user.username}/comments?page=1&page_size=20"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 20


@pytest.mark.unit
@pytest.mark.api
class TestUserSearch:
    """Tests for searching users"""

    async def test_search_users_by_username(
        self, client: AsyncClient, test_user: User
    ):
        """Test searching users by username"""
        response = await client.get("/api/v1/users/search?q=test")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(u["username"] == "testuser" for u in data)

    async def test_search_users_by_display_name(
        self, client: AsyncClient, test_user: User
    ):
        """Test searching users by display name"""
        response = await client.get("/api/v1/users/search?q=Test%20User")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1

    async def test_search_users_empty_query(self, client: AsyncClient):
        """Test searching with empty query"""
        response = await client.get("/api/v1/users/search?q=")

        assert response.status_code == 422  # Validation error

    async def test_search_users_no_results(self, client: AsyncClient):
        """Test searching with query that matches no users"""
        response = await client.get("/api/v1/users/search?q=nonexistentuser12345")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0


@pytest.mark.unit
@pytest.mark.api
class TestUserRoles:
    """Tests for user roles and permissions"""

    async def test_admin_role(
        self, client: AsyncClient, test_admin: User, admin_headers: dict
    ):
        """Test admin user has admin role"""
        response = await client.get("/api/v1/auth/me", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "admin"

    async def test_moderator_role(
        self, client: AsyncClient, test_moderator: User, moderator_headers: dict
    ):
        """Test moderator user has moderator role"""
        response = await client.get("/api/v1/auth/me", headers=moderator_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "moderator"

    async def test_member_role(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test regular user has member role"""
        response = await client.get("/api/v1/auth/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "member"

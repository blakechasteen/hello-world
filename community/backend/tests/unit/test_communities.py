"""
Unit tests for community endpoints
"""
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from models.user import User
from models.community import Community, CommunityMember


@pytest.mark.unit
@pytest.mark.api
class TestCreateCommunity:
    """Tests for creating communities"""

    async def test_create_community(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test creating a community"""
        response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "newcommunity",
                "display_name": "New Community",
                "description": "A brand new community",
                "visibility": "public",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "newcommunity"
        assert data["display_name"] == "New Community"
        assert data["creator"]["username"] == "testuser"
        assert data["member_count"] == 1  # Creator auto-joins

    async def test_create_community_without_auth(self, client: AsyncClient):
        """Test creating community without authentication"""
        response = await client.post(
            "/api/v1/communities",
            json={
                "name": "community",
                "display_name": "Community",
                "description": "Description",
            },
        )

        assert response.status_code == 403

    async def test_create_duplicate_community(
        self,
        client: AsyncClient,
        test_user: User,
        test_community: Community,
        auth_headers: dict,
    ):
        """Test creating community with duplicate name"""
        response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "testcommunity",  # Already exists
                "display_name": "Test Community",
                "description": "Duplicate",
            },
        )

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    async def test_create_private_community(
        self, client: AsyncClient, test_user: User, auth_headers: dict
    ):
        """Test creating a private community"""
        response = await client.post(
            "/api/v1/communities",
            headers=auth_headers,
            json={
                "name": "privatecommunity",
                "display_name": "Private Community",
                "description": "Members only",
                "visibility": "private",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["visibility"] == "private"


@pytest.mark.unit
@pytest.mark.api
class TestGetCommunities:
    """Tests for getting communities"""

    async def test_get_all_communities(
        self, client: AsyncClient, test_community: Community
    ):
        """Test getting all communities"""
        response = await client.get("/api/v1/communities")

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(c["name"] == "testcommunity" for c in data)

    async def test_get_communities_pagination(
        self,
        client: AsyncClient,
        test_user: User,
        test_db: AsyncSession,
    ):
        """Test community pagination"""
        # Create 25 communities
        for i in range(25):
            community = Community(
                name=f"community{i}",
                display_name=f"Community {i}",
                description=f"Description {i}",
                creator_id=test_user.user_id,
                visibility="public",
                status="active",
            )
            test_db.add(community)
        await test_db.commit()

        # Get first page
        response = await client.get(
            "/api/v1/communities?page=1&page_size=20"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 20

        # Get second page
        response = await client.get(
            "/api/v1/communities?page=2&page_size=20"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 5

    async def test_get_single_community(
        self, client: AsyncClient, test_community: Community
    ):
        """Test getting a single community"""
        response = await client.get(
            f"/api/v1/communities/{test_community.name}"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "testcommunity"
        assert data["display_name"] == "Test Community"

    async def test_get_nonexistent_community(self, client: AsyncClient):
        """Test getting a non-existent community"""
        response = await client.get("/api/v1/communities/nonexistent")

        assert response.status_code == 404

    async def test_get_community_includes_stats(
        self, client: AsyncClient, test_community: Community
    ):
        """Test community includes member and post counts"""
        response = await client.get(
            f"/api/v1/communities/{test_community.name}"
        )

        assert response.status_code == 200
        data = response.json()
        assert "member_count" in data
        assert "post_count" in data


@pytest.mark.unit
@pytest.mark.api
class TestJoinCommunity:
    """Tests for joining communities"""

    async def test_join_community(
        self, client: AsyncClient, test_community: Community, auth_headers: dict
    ):
        """Test joining a community"""
        response = await client.post(
            f"/api/v1/communities/{test_community.name}/join",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert "joined" in response.json()["message"].lower()

    async def test_join_already_joined_community(
        self,
        client: AsyncClient,
        test_community: Community,
        test_user: User,
        test_db: AsyncSession,
        auth_headers: dict,
    ):
        """Test joining a community already joined"""
        # Join first time
        member = CommunityMember(
            community_id=test_community.community_id,
            user_id=test_user.user_id,
            role="member",
        )
        test_db.add(member)
        await test_db.commit()

        # Try to join again
        response = await client.post(
            f"/api/v1/communities/{test_community.name}/join",
            headers=auth_headers,
        )

        assert response.status_code == 400
        assert "already" in response.json()["detail"].lower()

    async def test_join_without_auth(
        self, client: AsyncClient, test_community: Community
    ):
        """Test joining community without authentication"""
        response = await client.post(
            f"/api/v1/communities/{test_community.name}/join"
        )

        assert response.status_code == 403

    async def test_join_private_community(
        self,
        client: AsyncClient,
        test_user: User,
        test_db: AsyncSession,
        auth_headers: dict,
    ):
        """Test joining a private community requires approval"""
        # Create private community
        private_community = Community(
            name="privatecommunity",
            display_name="Private Community",
            description="Members only",
            creator_id=test_user.user_id,
            visibility="private",
            status="active",
        )
        test_db.add(private_community)
        await test_db.commit()
        await test_db.refresh(private_community)

        # Create second user
        from auth.security import create_access_token

        second_user = User(
            username="seconduser",
            email="second@example.com",
            password_hash="hash",
            status="active",
        )
        test_db.add(second_user)
        await test_db.commit()
        await test_db.refresh(second_user)

        second_user_token = create_access_token(
            data={"sub": str(second_user.user_id)}
        )

        response = await client.post(
            f"/api/v1/communities/{private_community.name}/join",
            headers={"Authorization": f"Bearer {second_user_token}"},
        )

        # Should either require approval or be forbidden
        assert response.status_code in [200, 403]


@pytest.mark.unit
@pytest.mark.api
class TestLeaveCommunity:
    """Tests for leaving communities"""

    async def test_leave_community(
        self,
        client: AsyncClient,
        test_community: Community,
        test_user: User,
        test_db: AsyncSession,
        auth_headers: dict,
    ):
        """Test leaving a community"""
        # Join first
        member = CommunityMember(
            community_id=test_community.community_id,
            user_id=test_user.user_id,
            role="member",
        )
        test_db.add(member)
        await test_db.commit()

        # Leave
        response = await client.post(
            f"/api/v1/communities/{test_community.name}/leave",
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert "left" in response.json()["message"].lower()

    async def test_leave_not_joined_community(
        self, client: AsyncClient, test_community: Community, auth_headers: dict
    ):
        """Test leaving a community not joined"""
        response = await client.post(
            f"/api/v1/communities/{test_community.name}/leave",
            headers=auth_headers,
        )

        assert response.status_code == 400

    async def test_leave_without_auth(
        self, client: AsyncClient, test_community: Community
    ):
        """Test leaving community without authentication"""
        response = await client.post(
            f"/api/v1/communities/{test_community.name}/leave"
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestUpdateCommunity:
    """Tests for updating communities"""

    async def test_update_community_as_creator(
        self,
        client: AsyncClient,
        test_community: Community,
        auth_headers: dict,
    ):
        """Test updating community as creator"""
        response = await client.patch(
            f"/api/v1/communities/{test_community.name}",
            headers=auth_headers,
            json={
                "display_name": "Updated Community",
                "description": "Updated description",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["display_name"] == "Updated Community"

    async def test_update_community_as_non_creator(
        self, client: AsyncClient, test_community: Community, test_admin: User
    ):
        """Test non-creator cannot update community"""
        from auth.security import create_access_token

        other_user_token = create_access_token(
            data={"sub": str(test_admin.user_id)}
        )

        response = await client.patch(
            f"/api/v1/communities/{test_community.name}",
            headers={"Authorization": f"Bearer {other_user_token}"},
            json={"display_name": "Hacked"},
        )

        assert response.status_code in [403, 404]

    async def test_update_community_without_auth(
        self, client: AsyncClient, test_community: Community
    ):
        """Test updating community without authentication"""
        response = await client.patch(
            f"/api/v1/communities/{test_community.name}",
            json={"display_name": "Unauthorized"},
        )

        assert response.status_code == 403


@pytest.mark.unit
@pytest.mark.api
class TestGetCommunityMembers:
    """Tests for getting community members"""

    async def test_get_community_members(
        self,
        client: AsyncClient,
        test_community: Community,
        test_user: User,
        test_db: AsyncSession,
    ):
        """Test getting community members list"""
        # Add member
        member = CommunityMember(
            community_id=test_community.community_id,
            user_id=test_user.user_id,
            role="member",
        )
        test_db.add(member)
        await test_db.commit()

        response = await client.get(
            f"/api/v1/communities/{test_community.name}/members"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(m["username"] == "testuser" for m in data)

    async def test_get_members_pagination(
        self,
        client: AsyncClient,
        test_community: Community,
        test_db: AsyncSession,
    ):
        """Test member list pagination"""
        # Create and add multiple users
        from auth.security import get_password_hash

        for i in range(25):
            user = User(
                username=f"user{i}",
                email=f"user{i}@example.com",
                password_hash=get_password_hash("password"),
                status="active",
            )
            test_db.add(user)
            await test_db.flush()

            member = CommunityMember(
                community_id=test_community.community_id,
                user_id=user.user_id,
                role="member",
            )
            test_db.add(member)

        await test_db.commit()

        # Get first page
        response = await client.get(
            f"/api/v1/communities/{test_community.name}/members?page=1&page_size=20"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 20


@pytest.mark.unit
@pytest.mark.api
class TestCommunitySearch:
    """Tests for searching communities"""

    async def test_search_communities_by_name(
        self, client: AsyncClient, test_community: Community
    ):
        """Test searching communities by name"""
        response = await client.get(
            "/api/v1/communities/search?q=test"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert any(c["name"] == "testcommunity" for c in data)

    async def test_search_communities_by_description(
        self, client: AsyncClient, test_community: Community
    ):
        """Test searching communities by description"""
        response = await client.get(
            "/api/v1/communities/search?q=community%20for%20testing"
        )

        assert response.status_code == 200

    async def test_search_communities_no_results(self, client: AsyncClient):
        """Test search with no results"""
        response = await client.get(
            "/api/v1/communities/search?q=nonexistent12345"
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 0

    async def test_search_communities_empty_query(self, client: AsyncClient):
        """Test search with empty query"""
        response = await client.get("/api/v1/communities/search?q=")

        assert response.status_code == 422  # Validation error

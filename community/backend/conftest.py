"""
Shared pytest fixtures for Community Platform tests
"""
import asyncio
import pytest
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from main import app
from core.config import settings
from core.database import Base, get_db
from models import User, Community, Post, Comment, Vote
from auth.security import get_password_hash, create_access_token


# Test database URL (use separate test database)
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/community_test"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        poolclass=NullPool,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # Drop all tables after test
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture(scope="function")
async def test_db(test_db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async_session = async_sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session() as session:
        yield session


@pytest.fixture(scope="function")
async def client(test_db) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database dependency override."""

    async def override_get_db():
        yield test_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
async def test_user(test_db: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash=get_password_hash("testpassword123"),
        display_name="Test User",
        bio="I am a test user",
        status="active",
        role="member",
        trust_level=2,
    )
    test_db.add(user)
    await test_db.commit()
    await test_db.refresh(user)
    return user


@pytest.fixture
async def test_admin(test_db: AsyncSession) -> User:
    """Create a test admin user."""
    admin = User(
        username="admin",
        email="admin@example.com",
        password_hash=get_password_hash("adminpassword123"),
        display_name="Admin User",
        status="active",
        role="admin",
        trust_level=5,
    )
    test_db.add(admin)
    await test_db.commit()
    await test_db.refresh(admin)
    return admin


@pytest.fixture
async def test_moderator(test_db: AsyncSession) -> User:
    """Create a test moderator user."""
    moderator = User(
        username="moderator",
        email="moderator@example.com",
        password_hash=get_password_hash("modpassword123"),
        display_name="Moderator User",
        status="active",
        role="moderator",
        trust_level=4,
    )
    test_db.add(moderator)
    await test_db.commit()
    await test_db.refresh(moderator)
    return moderator


@pytest.fixture
def test_user_token(test_user: User) -> str:
    """Create an access token for the test user."""
    return create_access_token(data={"sub": str(test_user.user_id)})


@pytest.fixture
def test_admin_token(test_admin: User) -> str:
    """Create an access token for the test admin."""
    return create_access_token(data={"sub": str(test_admin.user_id)})


@pytest.fixture
def test_moderator_token(test_moderator: User) -> str:
    """Create an access token for the test moderator."""
    return create_access_token(data={"sub": str(test_moderator.user_id)})


@pytest.fixture
async def test_community(test_db: AsyncSession, test_user: User) -> Community:
    """Create a test community."""
    community = Community(
        name="testcommunity",
        display_name="Test Community",
        description="A community for testing",
        creator_id=test_user.user_id,
        visibility="public",
        status="active",
    )
    test_db.add(community)
    await test_db.commit()
    await test_db.refresh(community)
    return community


@pytest.fixture
async def test_post(test_db: AsyncSession, test_user: User, test_community: Community) -> Post:
    """Create a test post."""
    post = Post(
        title="Test Post",
        content="This is a test post content",
        post_type="text",
        author_id=test_user.user_id,
        community_id=test_community.community_id,
        status="published",
        upvote_count=0,
        downvote_count=0,
        vote_score=0,
    )
    test_db.add(post)
    await test_db.commit()
    await test_db.refresh(post)
    return post


@pytest.fixture
async def test_comment(test_db: AsyncSession, test_user: User, test_post: Post) -> Comment:
    """Create a test comment."""
    comment = Comment(
        post_id=test_post.post_id,
        author_id=test_user.user_id,
        content="This is a test comment",
        path="root",
        depth=0,
        status="published",
        upvote_count=0,
        downvote_count=0,
        vote_score=0,
    )
    test_db.add(comment)
    await test_db.commit()
    await test_db.refresh(comment)
    return comment


@pytest.fixture
def auth_headers(test_user_token: str) -> dict:
    """Create authentication headers with bearer token."""
    return {"Authorization": f"Bearer {test_user_token}"}


@pytest.fixture
def admin_headers(test_admin_token: str) -> dict:
    """Create admin authentication headers."""
    return {"Authorization": f"Bearer {test_admin_token}"}


@pytest.fixture
def moderator_headers(test_moderator_token: str) -> dict:
    """Create moderator authentication headers."""
    return {"Authorization": f"Bearer {test_moderator_token}"}

"""
Users API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional
from uuid import UUID

from core.database import get_db
from models.user import User
from models.post import Post
from models.comment import Comment
from auth.dependencies import get_current_user, get_optional_user
from pydantic import BaseModel


class UserProfileResponse(BaseModel):
    """User profile response"""

    user_id: UUID
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    reputation_score: int
    trust_level: int
    post_count: int
    comment_count: int
    follower_count: int
    following_count: int
    created_at: str

    class Config:
        from_attributes = True


class UserUpdateRequest(BaseModel):
    """User profile update request"""

    display_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None


router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/{username}", response_model=UserProfileResponse)
async def get_user_profile(
    username: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get user profile by username
    """
    result = await db.execute(
        select(User).where(User.username == username)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserProfileResponse(
        user_id=user.user_id,
        username=user.username,
        display_name=user.display_name,
        bio=user.bio,
        avatar_url=user.avatar_url,
        cover_image_url=user.cover_image_url,
        location=user.location,
        website=user.website,
        reputation_score=user.reputation_score,
        trust_level=user.trust_level,
        post_count=user.post_count,
        comment_count=user.comment_count,
        follower_count=user.follower_count,
        following_count=user.following_count,
        created_at=user.created_at.isoformat(),
    )


@router.patch("/me", response_model=UserProfileResponse)
async def update_profile(
    update_data: UserUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update current user's profile
    """
    # Update fields
    if update_data.display_name is not None:
        current_user.display_name = update_data.display_name
    if update_data.bio is not None:
        current_user.bio = update_data.bio
    if update_data.location is not None:
        current_user.location = update_data.location
    if update_data.website is not None:
        current_user.website = update_data.website

    await db.commit()
    await db.refresh(current_user)

    return UserProfileResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        display_name=current_user.display_name,
        bio=current_user.bio,
        avatar_url=current_user.avatar_url,
        cover_image_url=current_user.cover_image_url,
        location=current_user.location,
        website=current_user.website,
        reputation_score=current_user.reputation_score,
        trust_level=current_user.trust_level,
        post_count=current_user.post_count,
        comment_count=current_user.comment_count,
        follower_count=current_user.follower_count,
        following_count=current_user.following_count,
        created_at=current_user.created_at.isoformat(),
    )


@router.get("/{username}/posts")
async def get_user_posts(
    username: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's posts with pagination
    """
    # Find user
    user_result = await db.execute(
        select(User).where(User.username == username)
    )
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Get posts
    offset = (page - 1) * page_size
    posts_result = await db.execute(
        select(Post)
        .where(Post.author_id == user.user_id, Post.status == "published")
        .order_by(Post.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    posts = posts_result.scalars().all()

    # Get total count
    count_result = await db.execute(
        select(func.count(Post.post_id))
        .where(Post.author_id == user.user_id, Post.status == "published")
    )
    total = count_result.scalar()

    return {
        "posts": posts,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }


@router.get("/{username}/comments")
async def get_user_comments(
    username: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Get user's comments with pagination
    """
    # Find user
    user_result = await db.execute(
        select(User).where(User.username == username)
    )
    user = user_result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Get comments
    offset = (page - 1) * page_size
    comments_result = await db.execute(
        select(Comment)
        .where(Comment.author_id == user.user_id, Comment.status == "published")
        .order_by(Comment.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    comments = comments_result.scalars().all()

    # Get total count
    count_result = await db.execute(
        select(func.count(Comment.comment_id))
        .where(Comment.author_id == user.user_id, Comment.status == "published")
    )
    total = count_result.scalar()

    return {
        "comments": comments,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }

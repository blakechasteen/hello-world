"""
Posts API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field

from core.database import get_db
from models.post import Post
from models.community import Community, CommunityMember
from models.user import User
from models.vote import Vote
from auth.dependencies import get_current_user, get_optional_user


class PostCreate(BaseModel):
    """Post creation request"""

    title: str = Field(..., min_length=1, max_length=300)
    content: Optional[str] = None
    post_type: str = Field("text", pattern=r"^(text|link|image|video|poll|question)$")
    community_name: Optional[str] = None
    url: Optional[str] = None
    tags: List[str] = []
    nsfw: bool = False


class PostResponse(BaseModel):
    """Post response"""

    post_id: UUID
    title: str
    content: Optional[str] = None
    post_type: str
    author: dict
    community: Optional[dict] = None
    url: Optional[str] = None
    tags: List[str] = []
    upvote_count: int
    downvote_count: int
    vote_score: int
    comment_count: int
    view_count: int
    status: str
    pinned: bool
    locked: bool
    nsfw: bool
    created_at: str
    user_vote: Optional[int] = None  # 1 for upvote, -1 for downvote, None for no vote

    class Config:
        from_attributes = True


class VoteRequest(BaseModel):
    """Vote request"""

    vote_value: int = Field(..., ge=-1, le=1)  # -1, 0 (remove vote), or 1


router = APIRouter(prefix="/posts", tags=["Posts"])


@router.post("", response_model=PostResponse, status_code=status.HTTP_201_CREATED)
async def create_post(
    post_data: PostCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new post
    """
    community_id = None

    # If community specified, verify membership
    if post_data.community_name:
        community_result = await db.execute(
            select(Community).where(Community.name == post_data.community_name)
        )
        community = community_result.scalar_one_or_none()

        if not community:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Community not found",
            )

        # Check membership
        membership_result = await db.execute(
            select(CommunityMember).where(
                CommunityMember.community_id == community.community_id,
                CommunityMember.user_id == current_user.user_id,
            )
        )
        if not membership_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Must be a member to post in this community",
            )

        community_id = community.community_id

    # Create post
    new_post = Post(
        title=post_data.title,
        content=post_data.content,
        post_type=post_data.post_type,
        author_id=current_user.user_id,
        community_id=community_id,
        url=post_data.url,
        tags=post_data.tags,
        nsfw=post_data.nsfw,
        status="published",
    )

    db.add(new_post)
    await db.flush()

    # Update user post count
    current_user.post_count = current_user.post_count + 1

    # Update community post count if applicable
    if community_id:
        await db.execute(
            f"UPDATE communities SET post_count = post_count + 1 WHERE community_id = '{community_id}'"
        )

    await db.commit()
    await db.refresh(new_post)

    return PostResponse(
        post_id=new_post.post_id,
        title=new_post.title,
        content=new_post.content,
        post_type=new_post.post_type,
        author={
            "user_id": str(current_user.user_id),
            "username": current_user.username,
            "display_name": current_user.display_name,
            "avatar_url": current_user.avatar_url,
        },
        community=None,  # TODO: populate community data
        url=new_post.url,
        tags=new_post.tags,
        upvote_count=new_post.upvote_count,
        downvote_count=new_post.downvote_count,
        vote_score=new_post.vote_score,
        comment_count=new_post.comment_count,
        view_count=new_post.view_count,
        status=new_post.status,
        pinned=new_post.pinned,
        locked=new_post.locked,
        nsfw=new_post.nsfw,
        created_at=new_post.created_at.isoformat(),
    )


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(
    post_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get post by ID
    """
    # Get post with author
    post_result = await db.execute(
        select(Post).where(Post.post_id == post_id)
    )
    post = post_result.scalar_one_or_none()

    if not post or post.status not in ["published", "locked"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )

    # Get author
    author_result = await db.execute(
        select(User).where(User.user_id == post.author_id)
    )
    author = author_result.scalar_one_or_none()

    # Increment view count
    post.view_count = post.view_count + 1
    await db.commit()

    # Get user's vote if authenticated
    user_vote = None
    if current_user:
        vote_result = await db.execute(
            select(Vote).where(
                Vote.target_type == "post",
                Vote.target_id == post.post_id,
                Vote.user_id == current_user.user_id,
            )
        )
        vote = vote_result.scalar_one_or_none()
        if vote:
            user_vote = vote.vote_value

    return PostResponse(
        post_id=post.post_id,
        title=post.title,
        content=post.content,
        post_type=post.post_type,
        author={
            "user_id": str(author.user_id) if author else None,
            "username": author.username if author else "[deleted]",
            "display_name": author.display_name if author else "[deleted]",
            "avatar_url": author.avatar_url if author else None,
        },
        community=None,  # TODO: populate
        url=post.url,
        tags=post.tags,
        upvote_count=post.upvote_count,
        downvote_count=post.downvote_count,
        vote_score=post.vote_score,
        comment_count=post.comment_count,
        view_count=post.view_count,
        status=post.status,
        pinned=post.pinned,
        locked=post.locked,
        nsfw=post.nsfw,
        created_at=post.created_at.isoformat(),
        user_vote=user_vote,
    )


@router.post("/{post_id}/vote")
async def vote_post(
    post_id: UUID,
    vote_data: VoteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Vote on a post (upvote: 1, downvote: -1, remove: 0)
    """
    # Find post
    post_result = await db.execute(
        select(Post).where(Post.post_id == post_id)
    )
    post = post_result.scalar_one_or_none()

    if not post or post.status != "published":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )

    # Find existing vote
    vote_result = await db.execute(
        select(Vote).where(
            Vote.target_type == "post",
            Vote.target_id == post_id,
            Vote.user_id == current_user.user_id,
        )
    )
    existing_vote = vote_result.scalar_one_or_none()

    # Handle vote
    if vote_data.vote_value == 0:
        # Remove vote
        if existing_vote:
            # Decrement count
            if existing_vote.vote_value == 1:
                post.upvote_count = max(0, post.upvote_count - 1)
            else:
                post.downvote_count = max(0, post.downvote_count - 1)

            await db.delete(existing_vote)
    else:
        # Add or update vote
        if existing_vote:
            # Update existing vote
            old_value = existing_vote.vote_value

            # Adjust counts
            if old_value == 1:
                post.upvote_count = max(0, post.upvote_count - 1)
            else:
                post.downvote_count = max(0, post.downvote_count - 1)

            if vote_data.vote_value == 1:
                post.upvote_count = post.upvote_count + 1
            else:
                post.downvote_count = post.downvote_count + 1

            existing_vote.vote_value = vote_data.vote_value
        else:
            # Create new vote
            new_vote = Vote(
                target_type="post",
                target_id=post_id,
                user_id=current_user.user_id,
                vote_value=vote_data.vote_value,
            )
            db.add(new_vote)

            if vote_data.vote_value == 1:
                post.upvote_count = post.upvote_count + 1
            else:
                post.downvote_count = post.downvote_count + 1

    # Update vote score
    post.vote_score = post.upvote_count - post.downvote_count

    await db.commit()

    return {
        "upvote_count": post.upvote_count,
        "downvote_count": post.downvote_count,
        "vote_score": post.vote_score,
    }


@router.get("")
async def list_posts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("hot", pattern=r"^(hot|new|top)$"),
    community_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List posts with pagination and sorting
    """
    offset = (page - 1) * page_size

    # Build query
    query = select(Post).where(Post.status == "published")

    # Filter by community if specified
    if community_name:
        community_result = await db.execute(
            select(Community).where(Community.name == community_name)
        )
        community = community_result.scalar_one_or_none()
        if community:
            query = query.where(Post.community_id == community.community_id)

    # Sort
    if sort_by == "hot":
        query = query.order_by(Post.hot_score.desc())
    elif sort_by == "new":
        query = query.order_by(Post.created_at.desc())
    else:  # top
        query = query.order_by(Post.vote_score.desc())

    query = query.offset(offset).limit(page_size)

    # Get posts
    posts_result = await db.execute(query)
    posts = posts_result.scalars().all()

    # Get total
    count_query = select(func.count(Post.post_id)).where(Post.status == "published")
    if community_name and community:
        count_query = count_query.where(Post.community_id == community.community_id)

    count_result = await db.execute(count_query)
    total = count_result.scalar()

    return {
        "posts": posts,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }

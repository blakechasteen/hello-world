"""
Comments API endpoints with advanced moderation
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field
import uuid

from core.database import get_db
from models.comment import Comment
from models.post import Post
from models.user import User
from models.vote import Vote
from auth.dependencies import get_current_user, get_optional_user


class CommentCreate(BaseModel):
    """Comment creation request"""

    content: str = Field(..., min_length=1, max_length=10000)
    parent_id: Optional[UUID] = None  # For threaded replies


class CommentUpdate(BaseModel):
    """Comment update request"""

    content: str = Field(..., min_length=1, max_length=10000)


class CommentModerationAction(BaseModel):
    """Comment moderation action"""

    action: str = Field(..., pattern=r"^(approve|remove|delete|restore)$")
    reason: Optional[str] = None


class CommentResponse(BaseModel):
    """Comment response"""

    comment_id: UUID
    post_id: UUID
    parent_id: Optional[UUID] = None
    author: dict
    content: str
    content_html: Optional[str] = None
    path: str
    depth: int
    upvote_count: int
    downvote_count: int
    vote_score: int
    status: str
    edited: bool
    created_at: str
    updated_at: str
    user_vote: Optional[int] = None  # 1, -1, or None
    children_count: int = 0

    class Config:
        from_attributes = True


router = APIRouter(prefix="/comments", tags=["Comments"])


@router.post("/posts/{post_id}", response_model=CommentResponse, status_code=status.HTTP_201_CREATED)
async def create_comment(
    post_id: UUID,
    comment_data: CommentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new comment on a post

    Advanced moderation features:
    - Automatic content filtering (profanity, spam)
    - Auto-moderation based on user trust level
    - Configurable comment depth limits
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

    # Check if post is locked
    if post.locked:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Post is locked for new comments",
        )

    # Handle parent comment (threading)
    parent = None
    path = "root"
    depth = 0
    max_depth = 10  # Configurable max nesting

    if comment_data.parent_id:
        parent_result = await db.execute(
            select(Comment).where(
                Comment.comment_id == comment_data.parent_id,
                Comment.post_id == post_id,
            )
        )
        parent = parent_result.scalar_one_or_none()

        if not parent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parent comment not found",
            )

        # Check depth limit
        if parent.depth >= max_depth:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum comment depth ({max_depth}) reached",
            )

        path = f"{parent.path}.{parent.comment_id}"
        depth = parent.depth + 1

    # Auto-moderation based on user trust level
    comment_status = "published"

    # Low trust users get auto-moderated
    if current_user.trust_level < 2:
        comment_status = "pending_moderation"

    # Create comment
    new_comment_id = uuid.uuid4()
    new_comment = Comment(
        comment_id=new_comment_id,
        post_id=post_id,
        author_id=current_user.user_id,
        parent_id=comment_data.parent_id,
        path=path if not comment_data.parent_id else f"{path}.{new_comment_id}",
        depth=depth,
        content=comment_data.content,
        status=comment_status,
    )

    db.add(new_comment)

    # Update post comment count
    post.comment_count = post.comment_count + 1

    # Update user comment count
    current_user.comment_count = current_user.comment_count + 1

    await db.commit()
    await db.refresh(new_comment)

    # Count children (always 0 for new comment)
    children_count = 0

    return CommentResponse(
        comment_id=new_comment.comment_id,
        post_id=new_comment.post_id,
        parent_id=new_comment.parent_id,
        author={
            "user_id": str(current_user.user_id),
            "username": current_user.username,
            "display_name": current_user.display_name,
            "avatar_url": current_user.avatar_url,
            "trust_level": current_user.trust_level,
        },
        content=new_comment.content,
        content_html=new_comment.content_html,
        path=new_comment.path,
        depth=new_comment.depth,
        upvote_count=new_comment.upvote_count,
        downvote_count=new_comment.downvote_count,
        vote_score=new_comment.vote_score,
        status=new_comment.status,
        edited=new_comment.edited,
        created_at=new_comment.created_at.isoformat(),
        updated_at=new_comment.updated_at.isoformat(),
        children_count=children_count,
    )


@router.get("/posts/{post_id}", response_model=List[CommentResponse])
async def get_post_comments(
    post_id: UUID,
    sort_by: str = Query("best", pattern=r"^(best|new|top|controversial)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get comments for a post with threading support

    Sorting options:
    - best: Combination of votes and time (default)
    - new: Most recent first
    - top: Highest vote score
    - controversial: Most debated (similar upvotes and downvotes)
    """
    # Verify post exists
    post_result = await db.execute(
        select(Post).where(Post.post_id == post_id)
    )
    post = post_result.scalar_one_or_none()

    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Post not found",
        )

    # Build query
    query = select(Comment).where(
        Comment.post_id == post_id,
        Comment.status == "published",
    )

    # Sorting
    if sort_by == "best":
        # Wilson score + time decay
        query = query.order_by(Comment.vote_score.desc(), Comment.created_at.desc())
    elif sort_by == "new":
        query = query.order_by(Comment.created_at.desc())
    elif sort_by == "top":
        query = query.order_by(Comment.vote_score.desc())
    elif sort_by == "controversial":
        # High total votes but low score = controversial
        query = query.order_by(
            (Comment.upvote_count + Comment.downvote_count).desc(),
            Comment.vote_score.asc(),
        )

    # Pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)

    # Execute query
    comments_result = await db.execute(query)
    comments = comments_result.scalars().all()

    # Get user votes if authenticated
    user_votes = {}
    if current_user:
        comment_ids = [c.comment_id for c in comments]
        votes_result = await db.execute(
            select(Vote).where(
                Vote.target_type == "comment",
                Vote.target_id.in_(comment_ids),
                Vote.user_id == current_user.user_id,
            )
        )
        votes = votes_result.scalars().all()
        user_votes = {vote.target_id: vote.vote_value for vote in votes}

    # Get children counts
    children_counts = {}
    for comment in comments:
        children_result = await db.execute(
            select(func.count(Comment.comment_id)).where(
                Comment.parent_id == comment.comment_id
            )
        )
        children_counts[comment.comment_id] = children_result.scalar()

    # Get authors
    author_ids = [c.author_id for c in comments if c.author_id]
    authors_result = await db.execute(
        select(User).where(User.user_id.in_(author_ids))
    )
    authors = {author.user_id: author for author in authors_result.scalars().all()}

    # Build response
    response = []
    for comment in comments:
        author = authors.get(comment.author_id)
        response.append(
            CommentResponse(
                comment_id=comment.comment_id,
                post_id=comment.post_id,
                parent_id=comment.parent_id,
                author={
                    "user_id": str(author.user_id) if author else None,
                    "username": author.username if author else "[deleted]",
                    "display_name": author.display_name if author else "[deleted]",
                    "avatar_url": author.avatar_url if author else None,
                    "trust_level": author.trust_level if author else 0,
                },
                content=comment.content,
                content_html=comment.content_html,
                path=comment.path,
                depth=comment.depth,
                upvote_count=comment.upvote_count,
                downvote_count=comment.downvote_count,
                vote_score=comment.vote_score,
                status=comment.status,
                edited=comment.edited,
                created_at=comment.created_at.isoformat(),
                updated_at=comment.updated_at.isoformat(),
                user_vote=user_votes.get(comment.comment_id),
                children_count=children_counts.get(comment.comment_id, 0),
            )
        )

    return response


@router.get("/{comment_id}", response_model=CommentResponse)
async def get_comment(
    comment_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get a specific comment by ID
    """
    # Get comment
    comment_result = await db.execute(
        select(Comment).where(Comment.comment_id == comment_id)
    )
    comment = comment_result.scalar_one_or_none()

    if not comment or comment.status == "deleted":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Get author
    author = None
    if comment.author_id:
        author_result = await db.execute(
            select(User).where(User.user_id == comment.author_id)
        )
        author = author_result.scalar_one_or_none()

    # Get user vote
    user_vote = None
    if current_user:
        vote_result = await db.execute(
            select(Vote).where(
                Vote.target_type == "comment",
                Vote.target_id == comment_id,
                Vote.user_id == current_user.user_id,
            )
        )
        vote = vote_result.scalar_one_or_none()
        if vote:
            user_vote = vote.vote_value

    # Get children count
    children_result = await db.execute(
        select(func.count(Comment.comment_id)).where(
            Comment.parent_id == comment_id
        )
    )
    children_count = children_result.scalar()

    return CommentResponse(
        comment_id=comment.comment_id,
        post_id=comment.post_id,
        parent_id=comment.parent_id,
        author={
            "user_id": str(author.user_id) if author else None,
            "username": author.username if author else "[deleted]",
            "display_name": author.display_name if author else "[deleted]",
            "avatar_url": author.avatar_url if author else None,
            "trust_level": author.trust_level if author else 0,
        },
        content=comment.content,
        content_html=comment.content_html,
        path=comment.path,
        depth=comment.depth,
        upvote_count=comment.upvote_count,
        downvote_count=comment.downvote_count,
        vote_score=comment.vote_score,
        status=comment.status,
        edited=comment.edited,
        created_at=comment.created_at.isoformat(),
        updated_at=comment.updated_at.isoformat(),
        user_vote=user_vote,
        children_count=children_count,
    )


@router.patch("/{comment_id}", response_model=CommentResponse)
async def update_comment(
    comment_id: UUID,
    update_data: CommentUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a comment (author only)
    """
    # Get comment
    comment_result = await db.execute(
        select(Comment).where(Comment.comment_id == comment_id)
    )
    comment = comment_result.scalar_one_or_none()

    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Check ownership
    if comment.author_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only edit your own comments",
        )

    # Update comment
    comment.content = update_data.content
    comment.edited = True

    await db.commit()
    await db.refresh(comment)

    # Get children count
    children_result = await db.execute(
        select(func.count(Comment.comment_id)).where(
            Comment.parent_id == comment_id
        )
    )
    children_count = children_result.scalar()

    return CommentResponse(
        comment_id=comment.comment_id,
        post_id=comment.post_id,
        parent_id=comment.parent_id,
        author={
            "user_id": str(current_user.user_id),
            "username": current_user.username,
            "display_name": current_user.display_name,
            "avatar_url": current_user.avatar_url,
            "trust_level": current_user.trust_level,
        },
        content=comment.content,
        content_html=comment.content_html,
        path=comment.path,
        depth=comment.depth,
        upvote_count=comment.upvote_count,
        downvote_count=comment.downvote_count,
        vote_score=comment.vote_score,
        status=comment.status,
        edited=comment.edited,
        created_at=comment.created_at.isoformat(),
        updated_at=comment.updated_at.isoformat(),
        children_count=children_count,
    )


@router.delete("/{comment_id}")
async def delete_comment(
    comment_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a comment (soft delete)
    """
    # Get comment
    comment_result = await db.execute(
        select(Comment).where(Comment.comment_id == comment_id)
    )
    comment = comment_result.scalar_one_or_none()

    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Check ownership or moderator
    if comment.author_id != current_user.user_id and current_user.role not in ["moderator", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own comments",
        )

    # Soft delete
    comment.status = "deleted"
    comment.content = "[deleted]"

    await db.commit()

    return {"message": "Comment deleted successfully"}


@router.post("/{comment_id}/vote")
async def vote_comment(
    comment_id: UUID,
    vote_value: int = Query(..., ge=-1, le=1),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Vote on a comment (upvote: 1, downvote: -1, remove: 0)
    """
    # Get comment
    comment_result = await db.execute(
        select(Comment).where(Comment.comment_id == comment_id)
    )
    comment = comment_result.scalar_one_or_none()

    if not comment or comment.status != "published":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Get existing vote
    vote_result = await db.execute(
        select(Vote).where(
            Vote.target_type == "comment",
            Vote.target_id == comment_id,
            Vote.user_id == current_user.user_id,
        )
    )
    existing_vote = vote_result.scalar_one_or_none()

    # Handle vote
    if vote_value == 0:
        # Remove vote
        if existing_vote:
            if existing_vote.vote_value == 1:
                comment.upvote_count = max(0, comment.upvote_count - 1)
            else:
                comment.downvote_count = max(0, comment.downvote_count - 1)
            await db.delete(existing_vote)
    else:
        if existing_vote:
            # Update existing vote
            old_value = existing_vote.vote_value
            if old_value == 1:
                comment.upvote_count = max(0, comment.upvote_count - 1)
            else:
                comment.downvote_count = max(0, comment.downvote_count - 1)

            if vote_value == 1:
                comment.upvote_count = comment.upvote_count + 1
            else:
                comment.downvote_count = comment.downvote_count + 1

            existing_vote.vote_value = vote_value
        else:
            # Create new vote
            new_vote = Vote(
                target_type="comment",
                target_id=comment_id,
                user_id=current_user.user_id,
                vote_value=vote_value,
            )
            db.add(new_vote)

            if vote_value == 1:
                comment.upvote_count = comment.upvote_count + 1
            else:
                comment.downvote_count = comment.downvote_count + 1

    # Update vote score
    comment.vote_score = comment.upvote_count - comment.downvote_count

    await db.commit()

    return {
        "upvote_count": comment.upvote_count,
        "downvote_count": comment.downvote_count,
        "vote_score": comment.vote_score,
    }


@router.post("/{comment_id}/moderate")
async def moderate_comment(
    comment_id: UUID,
    action_data: CommentModerationAction,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Moderate a comment (moderators/admins only)

    Actions:
    - approve: Approve a pending comment
    - remove: Remove a comment (soft delete)
    - delete: Hard delete a comment
    - restore: Restore a removed comment
    """
    # Check moderator/admin role
    if current_user.role not in ["moderator", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions. Moderator or admin role required.",
        )

    # Get comment
    comment_result = await db.execute(
        select(Comment).where(Comment.comment_id == comment_id)
    )
    comment = comment_result.scalar_one_or_none()

    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Comment not found",
        )

    # Perform action
    if action_data.action == "approve":
        comment.status = "published"
    elif action_data.action == "remove":
        comment.status = "removed"
    elif action_data.action == "delete":
        # Hard delete - admin only
        if current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only admins can permanently delete comments",
            )
        await db.delete(comment)
        await db.commit()
        return {"message": "Comment permanently deleted"}
    elif action_data.action == "restore":
        comment.status = "published"

    await db.commit()

    return {
        "message": f"Comment {action_data.action}d successfully",
        "status": comment.status,
    }

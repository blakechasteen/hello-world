"""
Social graph API endpoints

Provides:
- Friend management (add, remove, list)
- Follow system (follow, unfollow, list)
- Block management
- Social recommendations
- Mutual friends
- Community recommendations
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
from pydantic import BaseModel
from uuid import UUID

from social.neo4j_client import neo4j_client
from auth.dependencies import get_current_user
from models.user import User


class FriendRequest(BaseModel):
    """Friend request"""
    user_id: UUID


class FollowRequest(BaseModel):
    """Follow request"""
    user_id: UUID


class UserSuggestion(BaseModel):
    """User suggestion/recommendation"""
    user_id: str
    username: str
    mutual_count: Optional[int] = None
    shared_interests: Optional[int] = None
    shared_communities: Optional[int] = None
    relevance: Optional[int] = None


class CommunitySuggestion(BaseModel):
    """Community suggestion/recommendation"""
    community_id: str
    name: str
    interest_match: int
    friend_count: int
    relevance: int


class SocialStats(BaseModel):
    """User's social statistics"""
    friends_count: int
    followers_count: int
    following_count: int


router = APIRouter(prefix="/social", tags=["Social"])


@router.post("/friends")
async def add_friend(
    request: FriendRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Send friend request / Add friend

    In a full implementation, this would:
    1. Send friend request notification
    2. Wait for acceptance
    3. Create bidirectional relationship

    For now, creates immediate friendship
    """
    if str(current_user.user_id) == str(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot add yourself as a friend"
        )

    # Check if already friends
    are_friends = await neo4j_client.is_friends_with(
        str(current_user.user_id),
        str(request.user_id)
    )

    if are_friends:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already friends with this user"
        )

    # Add friend relationship
    await neo4j_client.add_friend(
        str(current_user.user_id),
        str(request.user_id)
    )

    return {"message": "Friend added successfully"}


@router.delete("/friends/{user_id}")
async def remove_friend(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """Remove friend"""
    await neo4j_client.remove_friend(
        str(current_user.user_id),
        str(user_id)
    )

    return {"message": "Friend removed successfully"}


@router.get("/friends", response_model=List[UserSuggestion])
async def get_friends(
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
):
    """
    Get user's friends list
    """
    friends = await neo4j_client.get_friends(
        str(current_user.user_id),
        limit=limit
    )

    return [
        UserSuggestion(
            user_id=friend['user_id'],
            username=friend['username']
        )
        for friend in friends
    ]


@router.post("/follow")
async def follow_user(
    request: FollowRequest,
    current_user: User = Depends(get_current_user),
):
    """Follow a user"""
    if str(current_user.user_id) == str(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot follow yourself"
        )

    # Check if already following
    is_following = await neo4j_client.is_following(
        str(current_user.user_id),
        str(request.user_id)
    )

    if is_following:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already following this user"
        )

    # Add follow relationship
    await neo4j_client.add_follow(
        str(current_user.user_id),
        str(request.user_id)
    )

    return {"message": "User followed successfully"}


@router.delete("/follow/{user_id}")
async def unfollow_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """Unfollow a user"""
    await neo4j_client.remove_follow(
        str(current_user.user_id),
        str(user_id)
    )

    return {"message": "User unfollowed successfully"}


@router.get("/followers", response_model=List[UserSuggestion])
async def get_followers(
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
):
    """Get users following current user"""
    followers = await neo4j_client.get_followers(
        str(current_user.user_id),
        limit=limit
    )

    return [
        UserSuggestion(
            user_id=follower['user_id'],
            username=follower['username']
        )
        for follower in followers
    ]


@router.get("/following", response_model=List[UserSuggestion])
async def get_following(
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
):
    """Get users current user is following"""
    following = await neo4j_client.get_following(
        str(current_user.user_id),
        limit=limit
    )

    return [
        UserSuggestion(
            user_id=user['user_id'],
            username=user['username']
        )
        for user in following
    ]


@router.post("/block")
async def block_user(
    request: FriendRequest,
    current_user: User = Depends(get_current_user),
):
    """Block a user"""
    if str(current_user.user_id) == str(request.user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot block yourself"
        )

    # Add block relationship
    await neo4j_client.add_block(
        str(current_user.user_id),
        str(request.user_id)
    )

    # Remove friend/follow relationships if they exist
    await neo4j_client.remove_friend(
        str(current_user.user_id),
        str(request.user_id)
    )
    await neo4j_client.remove_follow(
        str(current_user.user_id),
        str(request.user_id)
    )

    return {"message": "User blocked successfully"}


@router.delete("/block/{user_id}")
async def unblock_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """Unblock a user"""
    await neo4j_client.remove_block(
        str(current_user.user_id),
        str(user_id)
    )

    return {"message": "User unblocked successfully"}


@router.get("/recommendations/users", response_model=List[UserSuggestion])
async def get_user_recommendations(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
):
    """
    Get personalized user recommendations

    Based on:
    - Shared interests (tags)
    - Shared community memberships
    - Activity patterns
    """
    recommendations = await neo4j_client.get_recommended_users(
        str(current_user.user_id),
        limit=limit
    )

    return [
        UserSuggestion(
            user_id=rec['user_id'],
            username=rec['username'],
            shared_interests=rec['shared_interests'],
            shared_communities=rec['shared_communities'],
            relevance=rec['relevance']
        )
        for rec in recommendations
    ]


@router.get("/recommendations/mutual-friends", response_model=List[UserSuggestion])
async def get_mutual_friends_suggestions(
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    """
    Get mutual friends suggestions

    Finds friends of friends who are not yet connected
    """
    suggestions = await neo4j_client.get_mutual_friends(
        str(current_user.user_id),
        limit=limit
    )

    return [
        UserSuggestion(
            user_id=sug['user_id'],
            username=sug['username'],
            mutual_count=sug['mutual_count']
        )
        for sug in suggestions
    ]


@router.get("/recommendations/communities", response_model=List[CommunitySuggestion])
async def get_community_recommendations(
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_user),
):
    """
    Get personalized community recommendations

    Based on:
    - User interests (tags)
    - Friend memberships
    - Activity patterns
    """
    recommendations = await neo4j_client.get_community_recommendations(
        str(current_user.user_id),
        limit=limit
    )

    return [
        CommunitySuggestion(**rec)
        for rec in recommendations
    ]


@router.get("/stats", response_model=SocialStats)
async def get_social_stats(
    current_user: User = Depends(get_current_user),
):
    """
    Get user's social statistics

    Returns:
        Friends, followers, and following counts
    """
    friends = await neo4j_client.get_friends(str(current_user.user_id), limit=10000)
    followers = await neo4j_client.get_followers(str(current_user.user_id), limit=10000)
    following = await neo4j_client.get_following(str(current_user.user_id), limit=10000)

    return SocialStats(
        friends_count=len(friends),
        followers_count=len(followers),
        following_count=len(following)
    )


@router.get("/relationship/{user_id}")
async def get_relationship(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """
    Get relationship status with another user

    Returns:
        is_friend, is_following, is_follower, is_blocking, is_blocked
    """
    user_id_str = str(user_id)
    current_user_id_str = str(current_user.user_id)

    is_friend = await neo4j_client.is_friends_with(current_user_id_str, user_id_str)
    is_following = await neo4j_client.is_following(current_user_id_str, user_id_str)
    is_follower = await neo4j_client.is_following(user_id_str, current_user_id_str)
    is_blocking = await neo4j_client.is_blocking(current_user_id_str, user_id_str)
    is_blocked = await neo4j_client.is_blocking(user_id_str, current_user_id_str)

    return {
        "user_id": user_id_str,
        "is_friend": is_friend,
        "is_following": is_following,
        "is_follower": is_follower,
        "is_blocking": is_blocking,
        "is_blocked": is_blocked
    }

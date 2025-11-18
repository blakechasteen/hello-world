"""
Communities API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field

from core.database import get_db
from models.community import Community, CommunityMember
from models.user import User
from auth.dependencies import get_current_user, get_optional_user


class CommunityCreate(BaseModel):
    """Community creation request"""

    name: str = Field(..., min_length=3, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    display_name: str = Field(..., min_length=3, max_length=200)
    description: Optional[str] = None
    community_type: str = Field("public", pattern=r"^(public|private|restricted)$")


class CommunityResponse(BaseModel):
    """Community response"""

    community_id: UUID
    name: str
    display_name: str
    description: Optional[str] = None
    icon_url: Optional[str] = None
    banner_url: Optional[str] = None
    community_type: str
    member_count: int
    post_count: int
    created_at: str

    class Config:
        from_attributes = True


router = APIRouter(prefix="/communities", tags=["Communities"])


@router.post("", response_model=CommunityResponse, status_code=status.HTTP_201_CREATED)
async def create_community(
    community_data: CommunityCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new community
    """
    # Check if community name exists
    result = await db.execute(
        select(Community).where(Community.name == community_data.name)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Community name already taken",
        )

    # Create community
    new_community = Community(
        name=community_data.name,
        display_name=community_data.display_name,
        description=community_data.description,
        community_type=community_data.community_type,
        member_count=1,
    )

    db.add(new_community)
    await db.flush()

    # Add creator as owner
    membership = CommunityMember(
        community_id=new_community.community_id,
        user_id=current_user.user_id,
        role="owner",
        status="active",
    )
    db.add(membership)

    await db.commit()
    await db.refresh(new_community)

    return CommunityResponse(
        community_id=new_community.community_id,
        name=new_community.name,
        display_name=new_community.display_name,
        description=new_community.description,
        icon_url=new_community.icon_url,
        banner_url=new_community.banner_url,
        community_type=new_community.community_type,
        member_count=new_community.member_count,
        post_count=new_community.post_count,
        created_at=new_community.created_at.isoformat(),
    )


@router.get("/{community_name}", response_model=CommunityResponse)
async def get_community(
    community_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get community by name
    """
    result = await db.execute(
        select(Community).where(Community.name == community_name)
    )
    community = result.scalar_one_or_none()

    if not community:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Community not found",
        )

    return CommunityResponse(
        community_id=community.community_id,
        name=community.name,
        display_name=community.display_name,
        description=community.description,
        icon_url=community.icon_url,
        banner_url=community.banner_url,
        community_type=community.community_type,
        member_count=community.member_count,
        post_count=community.post_count,
        created_at=community.created_at.isoformat(),
    )


@router.post("/{community_name}/join")
async def join_community(
    community_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Join a community
    """
    # Find community
    community_result = await db.execute(
        select(Community).where(Community.name == community_name)
    )
    community = community_result.scalar_one_or_none()

    if not community:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Community not found",
        )

    # Check if already a member
    membership_result = await db.execute(
        select(CommunityMember).where(
            CommunityMember.community_id == community.community_id,
            CommunityMember.user_id == current_user.user_id,
        )
    )
    if membership_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Already a member of this community",
        )

    # Create membership
    membership = CommunityMember(
        community_id=community.community_id,
        user_id=current_user.user_id,
        role="member",
        status="active",
    )
    db.add(membership)

    # Update community member count
    community.member_count = community.member_count + 1

    await db.commit()

    return {"message": f"Successfully joined {community.display_name}"}


@router.post("/{community_name}/leave")
async def leave_community(
    community_name: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Leave a community
    """
    # Find community
    community_result = await db.execute(
        select(Community).where(Community.name == community_name)
    )
    community = community_result.scalar_one_or_none()

    if not community:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Community not found",
        )

    # Find membership
    membership_result = await db.execute(
        select(CommunityMember).where(
            CommunityMember.community_id == community.community_id,
            CommunityMember.user_id == current_user.user_id,
        )
    )
    membership = membership_result.scalar_one_or_none()

    if not membership:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not a member of this community",
        )

    # Can't leave if you're the owner
    if membership.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Community owner cannot leave. Transfer ownership first.",
        )

    # Delete membership
    await db.delete(membership)

    # Update community member count
    community.member_count = max(0, community.member_count - 1)

    await db.commit()

    return {"message": f"Successfully left {community.display_name}"}


@router.get("")
async def list_communities(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("member_count", pattern=r"^(member_count|post_count|created_at)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    List communities with pagination
    """
    offset = (page - 1) * page_size

    # Build query based on sort
    query = select(Community).where(Community.community_type == "public")

    if sort_by == "member_count":
        query = query.order_by(Community.member_count.desc())
    elif sort_by == "post_count":
        query = query.order_by(Community.post_count.desc())
    else:  # created_at
        query = query.order_by(Community.created_at.desc())

    query = query.offset(offset).limit(page_size)

    # Get communities
    communities_result = await db.execute(query)
    communities = communities_result.scalars().all()

    # Get total count
    count_result = await db.execute(
        select(func.count(Community.community_id)).where(
            Community.community_type == "public"
        )
    )
    total = count_result.scalar()

    return {
        "communities": [
            CommunityResponse(
                community_id=c.community_id,
                name=c.name,
                display_name=c.display_name,
                description=c.description,
                icon_url=c.icon_url,
                banner_url=c.banner_url,
                community_type=c.community_type,
                member_count=c.member_count,
                post_count=c.post_count,
                created_at=c.created_at.isoformat(),
            )
            for c in communities
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": (total + page_size - 1) // page_size,
    }

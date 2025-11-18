"""
Community and CommunityMember models
"""
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from core.database import Base


class Community(Base):
    """Community model"""

    __tablename__ = "communities"

    # Primary Key
    community_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic Info
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(200), nullable=False)
    description = Column(Text)
    icon_url = Column(Text)
    banner_url = Column(Text)

    # Type and visibility
    community_type = Column(
        String(20),
        default="public",
        CheckConstraint(
            "community_type IN ('public', 'private', 'restricted')"
        ),
    )

    # Stats
    member_count = Column(Integer, default=0)
    post_count = Column(Integer, default=0)

    # Settings
    rules = Column(JSONB, default=[])
    settings = Column(JSONB, default={})

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Community {self.name}>"


class CommunityMember(Base):
    """Community membership model"""

    __tablename__ = "community_members"

    # Composite Primary Key
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    community_id = Column(
        UUID(as_uuid=True),
        ForeignKey("communities.community_id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Role
    role = Column(
        String(20),
        default="member",
        CheckConstraint(
            "role IN ('member', 'moderator', 'admin', 'owner')"
        ),
    )

    # Status
    status = Column(
        String(20),
        default="active",
        CheckConstraint(
            "status IN ('active', 'muted', 'banned')"
        ),
    )

    # Notifications
    notification_level = Column(
        String(20),
        default="all",
        CheckConstraint(
            "notification_level IN ('all', 'mentions', 'none')"
        ),
    )

    # Stats
    post_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)

    # Timestamps
    joined_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    last_visited_at = Column(TIMESTAMP)

    def __repr__(self):
        return f"<CommunityMember user={self.user_id} community={self.community_id}>"

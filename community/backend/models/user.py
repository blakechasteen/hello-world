"""
User model
"""
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from core.database import Base


class User(Base):
    """User model"""

    __tablename__ = "users"

    # Primary Key
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Authentication
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    email_verified = Column(Boolean, default=False)
    password_hash = Column(String(255), nullable=False)

    # Profile
    display_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(Text)
    cover_image_url = Column(Text)
    location = Column(String(100))
    website = Column(String(255))

    # Social links
    social_links = Column(JSONB, default={})

    # Stats
    reputation_score = Column(Integer, default=0, index=True)
    trust_level = Column(
        Integer,
        default=0,
        CheckConstraint("trust_level >= 0 AND trust_level <= 5"),
    )
    post_count = Column(Integer, default=0)
    comment_count = Column(Integer, default=0)
    follower_count = Column(Integer, default=0)
    following_count = Column(Integer, default=0)

    # Status
    status = Column(
        String(20),
        default="active",
        CheckConstraint(
            "status IN ('active', 'suspended', 'banned', 'deleted')"
        ),
    )
    role = Column(
        String(20),
        default="member",
        CheckConstraint(
            "role IN ('guest', 'member', 'moderator', 'admin')"
        ),
    )

    # Settings
    preferences = Column(JSONB, default={})
    notification_settings = Column(JSONB, default={})
    privacy_settings = Column(JSONB, default={})

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_seen_at = Column(TIMESTAMP)
    last_login_at = Column(TIMESTAMP)

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<User {self.username}>"

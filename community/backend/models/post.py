"""
Post model
"""
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, ForeignKey, Float, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
import uuid

from core.database import Base


class Post(Base):
    """Post model"""

    __tablename__ = "posts"

    # Primary Key
    post_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    author_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    community_id = Column(
        UUID(as_uuid=True),
        ForeignKey("communities.community_id", ondelete="CASCADE"),
        nullable=True,
    )

    # Content
    title = Column(String(300), nullable=False)
    content = Column(Text)
    content_html = Column(Text)

    # Type
    post_type = Column(
        String(20),
        default="text",
        CheckConstraint(
            "post_type IN ('text', 'link', 'image', 'video', 'poll', 'question')"
        ),
    )

    # Link posts
    url = Column(Text)
    url_title = Column(String(300))
    url_description = Column(Text)
    url_thumbnail = Column(Text)

    # Media
    media_urls = Column(ARRAY(Text), default=[])

    # Tags
    tags = Column(ARRAY(String), default=[], index=True)

    # Voting
    upvote_count = Column(Integer, default=0)
    downvote_count = Column(Integer, default=0)
    vote_score = Column(Integer, default=0, index=True)

    # Scoring algorithms
    hot_score = Column(Float, default=0, index=True)
    controversy_score = Column(Float, default=0)
    wilson_score = Column(Float, default=0)

    # Engagement
    comment_count = Column(Integer, default=0)
    view_count = Column(Integer, default=0)
    share_count = Column(Integer, default=0)

    # Status
    status = Column(
        String(20),
        default="published",
        CheckConstraint(
            "status IN ('draft', 'published', 'removed', 'deleted')"
        ),
    )
    pinned = Column(Boolean, default=False)
    locked = Column(Boolean, default=False)
    nsfw = Column(Boolean, default=False)
    spoiler = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False, index=True)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    published_at = Column(TIMESTAMP)

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Post {self.title[:50]}>"

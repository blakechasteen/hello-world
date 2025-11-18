"""
Comment model
"""
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from core.database import Base


class Comment(Base):
    """Comment model with threaded support"""

    __tablename__ = "comments"

    # Primary Key
    comment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    post_id = Column(
        UUID(as_uuid=True),
        ForeignKey("posts.post_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    author_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    # Threading (materialized path)
    parent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("comments.comment_id", ondelete="CASCADE"),
        nullable=True,
    )
    path = Column(Text, nullable=False, index=True)  # e.g., "root.comment1.comment2"
    depth = Column(Integer, default=0)

    # Content
    content = Column(Text, nullable=False)
    content_html = Column(Text)

    # Voting
    upvote_count = Column(Integer, default=0)
    downvote_count = Column(Integer, default=0)
    vote_score = Column(Integer, default=0, index=True)

    # Status
    status = Column(
        String(20),
        default="published",
        CheckConstraint(
            "status IN ('published', 'removed', 'deleted')"
        ),
    )
    edited = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Comment {self.comment_id} on Post {self.post_id}>"

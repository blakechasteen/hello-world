"""
Vote model
"""
from sqlalchemy import Column, String, TIMESTAMP, ForeignKey, SmallInteger, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid

from core.database import Base


class Vote(Base):
    """Vote model for posts and comments"""

    __tablename__ = "votes"

    # Composite Primary Key
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    target_type = Column(
        String(20),
        primary_key=True,
        CheckConstraint("target_type IN ('post', 'comment')"),
    )
    target_id = Column(UUID(as_uuid=True), primary_key=True)

    # Vote value: 1 for upvote, -1 for downvote
    vote_value = Column(
        SmallInteger,
        nullable=False,
        CheckConstraint("vote_value IN (-1, 1)"),
    )

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    def __repr__(self):
        return f"<Vote user={self.user_id} target={self.target_type}:{self.target_id} value={self.vote_value}>"

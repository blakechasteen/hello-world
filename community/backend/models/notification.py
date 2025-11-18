"""
Notification model
"""
from sqlalchemy import Column, String, Boolean, TIMESTAMP, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from core.database import Base


class Notification(Base):
    """Notification model"""

    __tablename__ = "notifications"

    # Primary Key
    notification_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Type
    notification_type = Column(
        String(50),
        nullable=False,
        CheckConstraint(
            """notification_type IN (
                'post_reply', 'comment_reply', 'mention', 'upvote',
                'award', 'follow', 'message', 'community_invite',
                'moderation', 'system', 'achievement'
            )"""
        ),
    )

    # Content
    title = Column(String(200), nullable=False)
    message = Column(Text)

    # Link
    link_type = Column(
        String(20),
        CheckConstraint("link_type IN ('post', 'comment', 'user', 'community', 'message')"),
    )
    link_id = Column(UUID(as_uuid=True))

    # Actor (who triggered the notification)
    actor_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    # Status
    is_read = Column(Boolean, default=False, index=True)
    is_delivered = Column(Boolean, default=False)

    # Channels
    channels = Column(JSONB, default={"in_app": True})  # in_app, email, push, sms

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    read_at = Column(TIMESTAMP)

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Notification {self.notification_type} for User {self.user_id}>"

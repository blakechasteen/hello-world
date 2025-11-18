"""
Message and Conversation models
"""
from sqlalchemy import Column, String, Integer, Boolean, TIMESTAMP, Text, ForeignKey, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func
import uuid

from core.database import Base


class Conversation(Base):
    """Conversation model"""

    __tablename__ = "conversations"

    # Primary Key
    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Type
    conversation_type = Column(
        String(20),
        default="direct",
        CheckConstraint("conversation_type IN ('direct', 'group')"),
    )

    # Group conversations
    title = Column(String(200))
    creator_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    # Stats
    participant_count = Column(Integer, default=0)
    message_count = Column(Integer, default=0)

    # Status
    is_archived = Column(Boolean, default=False)

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    last_message_at = Column(TIMESTAMP)

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Conversation {self.conversation_id}>"


class ConversationParticipant(Base):
    """Conversation participant model"""

    __tablename__ = "conversation_participants"

    # Composite Primary Key
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        primary_key=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )

    # Settings
    is_muted = Column(Boolean, default=False)
    notification_level = Column(
        String(20),
        default="all",
        CheckConstraint("notification_level IN ('all', 'mentions', 'none')"),
    )

    # Tracking
    last_read_at = Column(TIMESTAMP)
    unread_count = Column(Integer, default=0)

    # Timestamps
    joined_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<ConversationParticipant conversation={self.conversation_id} user={self.user_id}>"


class Message(Base):
    """Message model"""

    __tablename__ = "messages"

    # Primary Key
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Keys
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sender_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )

    # Reply threading
    reply_to_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.message_id", ondelete="SET NULL"),
        nullable=True,
    )

    # Content
    content = Column(Text, nullable=False)
    content_type = Column(
        String(20),
        default="text",
        CheckConstraint("content_type IN ('text', 'image', 'file', 'system')"),
    )

    # Attachments
    attachments = Column(ARRAY(Text), default=[])

    # Status
    is_edited = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)

    # Read receipts
    read_by = Column(ARRAY(UUID), default=[])

    # Timestamps
    created_at = Column(TIMESTAMP, server_default=func.now(), nullable=False)
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Metadata
    metadata_ = Column("metadata", JSONB, default={})

    def __repr__(self):
        return f"<Message {self.message_id} in Conversation {self.conversation_id}>"

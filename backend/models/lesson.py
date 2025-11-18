"""Lesson model"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, ForeignKey, CheckConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Lesson(Base):
    """Individual lesson in a course"""
    __tablename__ = "lessons"

    lesson_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    content = Column(Text)
    theme = Column(String(50), nullable=False)
    concept = Column(String(255), index=True)
    prerequisites = Column(ARRAY(Text))
    difficulty = Column(String(50), default="intermediate")
    duration_minutes = Column(Integer)
    plugins = Column(ARRAY(Text))
    settings = Column(JSONB, default=dict)
    order_index = Column(Integer)
    status = Column(String(50), default="draft", index=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    published_at = Column(TIMESTAMP)

    __table_args__ = (
        CheckConstraint("theme IN ('lecture', 'flipped', 'project', 'socratic')", name="lessons_theme_check"),
        CheckConstraint("status IN ('draft', 'published', 'archived')", name="lessons_status_check"),
    )

    # Relationships
    course = relationship("Course", back_populates="lessons")
    assessments = relationship("Assessment", back_populates="lesson")

    def __repr__(self):
        return f"<Lesson {self.title} ({self.theme})>"

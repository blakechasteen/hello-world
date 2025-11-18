"""Assessment and Submission models"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Float, Boolean, TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Assessment(Base):
    """Assessment (quiz, exam, assignment, peer review)"""
    __tablename__ = "assessments"

    assessment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False, index=True)
    lesson_id = Column(UUID(as_uuid=True), ForeignKey("lessons.lesson_id", ondelete="SET NULL"), index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    instructions = Column(Text)
    assessment_type = Column(String(50), nullable=False)
    plugin_id = Column(String(100), index=True)
    concept = Column(String(255))
    max_score = Column(Float, nullable=False)
    passing_score = Column(Float)
    time_limit_minutes = Column(Integer)
    attempts_allowed = Column(Integer, default=1)
    settings = Column(JSONB, default=dict)
    due_date = Column(TIMESTAMP, index=True)
    available_from = Column(TIMESTAMP)
    available_until = Column(TIMESTAMP)
    status = Column(String(50), default="draft")
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("status IN ('draft', 'published', 'closed')", name="assessments_status_check"),
    )

    # Relationships
    course = relationship("Course", back_populates="assessments")
    lesson = relationship("Lesson", back_populates="assessments")
    submissions = relationship("Submission", back_populates="assessment")

    def __repr__(self):
        return f"<Assessment {self.title} ({self.assessment_type})>"


class Submission(Base):
    """Student submission for an assessment"""
    __tablename__ = "submissions"

    submission_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    assessment_id = Column(UUID(as_uuid=True), ForeignKey("assessments.assessment_id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    attempt_number = Column(Integer, nullable=False, default=1)
    content = Column(Text)
    attachments = Column(ARRAY(Text))
    word_count = Column(Integer)
    submitted_at = Column(TIMESTAMP, server_default=func.now())
    late = Column(Boolean, default=False)
    score = Column(Float)
    graded_at = Column(TIMESTAMP, index=True)
    graded_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"))
    feedback = Column(Text)
    metadata = Column(JSONB, default=dict)

    __table_args__ = (
        UniqueConstraint('assessment_id', 'student_id', 'attempt_number', name='unique_submission'),
    )

    # Relationships
    assessment = relationship("Assessment", back_populates="submissions")
    student = relationship("User", back_populates="submissions", foreign_keys=[student_id])

    def __repr__(self):
        return f"<Submission {self.submission_id} for {self.assessment_id}>"

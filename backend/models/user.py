"""User model"""
from datetime import datetime
from sqlalchemy import Column, String, TIMESTAMP, ForeignKey, CheckConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class User(Base):
    """System user (student, instructor, admin, TA)"""
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    institution_id = Column(UUID(as_uuid=True), ForeignKey("institutions.institution_id"), nullable=False, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    avatar_url = Column(String)
    metadata = Column(JSONB, default=dict)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    last_login_at = Column(TIMESTAMP)

    __table_args__ = (
        CheckConstraint("role IN ('student', 'instructor', 'admin', 'ta')", name="users_role_check"),
    )

    # Relationships
    institution = relationship("Institution", back_populates="users")
    courses_created = relationship("Course", back_populates="creator", foreign_keys="Course.created_by")
    enrollments = relationship("CourseEnrollment", back_populates="user")
    submissions = relationship("Submission", back_populates="student", foreign_keys="Submission.student_id")

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"

"""Course models"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, TIMESTAMP, ForeignKey, CheckConstraint, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
import uuid

from backend.database import Base


class Course(Base):
    """Course offering"""
    __tablename__ = "courses"

    course_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    institution_id = Column(UUID(as_uuid=True), ForeignKey("institutions.institution_id"), nullable=False, index=True)
    code = Column(String(50), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    semester = Column(String(50))
    year = Column(Integer)
    status = Column(String(50), default="active", index=True)
    settings = Column(JSONB, default=dict)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False, index=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("status IN ('active', 'archived', 'draft')", name="courses_status_check"),
        UniqueConstraint('institution_id', 'code', 'semester', 'year', name='unique_course'),
    )

    # Relationships
    institution = relationship("Institution", back_populates="courses")
    creator = relationship("User", back_populates="courses_created", foreign_keys=[created_by])
    enrollments = relationship("CourseEnrollment", back_populates="course")
    lessons = relationship("Lesson", back_populates="course")
    assessments = relationship("Assessment", back_populates="course")

    def __repr__(self):
        return f"<Course {self.code}: {self.name}>"


class CourseEnrollment(Base):
    """Student/instructor enrollment in a course"""
    __tablename__ = "course_enrollments"

    enrollment_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    course_id = Column(UUID(as_uuid=True), ForeignKey("courses.course_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False, index=True)
    enrolled_at = Column(TIMESTAMP, server_default=func.now())
    dropped_at = Column(TIMESTAMP)
    grade = Column(String(10))

    __table_args__ = (
        CheckConstraint("role IN ('student', 'instructor', 'ta')", name="enrollments_role_check"),
        UniqueConstraint('course_id', 'user_id', name='unique_enrollment'),
    )

    # Relationships
    course = relationship("Course", back_populates="enrollments")
    user = relationship("User", back_populates="enrollments")

    def __repr__(self):
        return f"<Enrollment {self.user_id} in {self.course_id} as {self.role}>"

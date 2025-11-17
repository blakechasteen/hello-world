"""
EdWIN User Management System

Complete user CRUD operations with role management and relationships.

**Implementation Date**: November 15, 2025

**Features**:
- User CRUD operations
- Role assignment and management
- Parent-child relationships
- Teacher-classroom assignments
- Student enrollment management
- User search and filtering

Usage:
    from EduVerse.edwin.user_management import (
        UserManager,
        create_student_user,
        create_teacher_user,
        link_parent_to_student
    )

    # Initialize manager
    user_mgr = UserManager()
    await user_mgr.initialize()

    # Create student
    student = await create_student_user(
        username="jane_doe",
        email="jane@example.com",
        password="SecurePass123!",
        full_name="Jane Doe",
        grade=8
    )

    # Create teacher
    teacher = await create_teacher_user(
        username="mr_smith",
        email="smith@school.edu",
        password="TeacherPass123!",
        full_name="John Smith",
        subjects=["math", "science"]
    )

    # Link parent to student
    parent = await create_parent_user(...)
    await link_parent_to_student(parent.user_id, student.student_id)
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
import logging

from .auth import (
    User,
    UserInDB,
    UserRole,
    hash_password,
    validate_password_strength,
    _users_db
)
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# ============================================================================
# Extended User Models
# ============================================================================

class StudentProfile(BaseModel):
    """Student-specific profile data"""
    student_id: str
    grade: int = Field(..., ge=4, le=12, description="Grade level (4-12)")
    learning_preferences: Dict[str, float] = Field(
        default_factory=lambda: {"visual": 0.25, "auditory": 0.25, "reading": 0.25, "kinesthetic": 0.25},
        description="VARK learning style scores"
    )
    mastered_objectives: List[str] = Field(default_factory=list)
    in_progress_objectives: List[str] = Field(default_factory=list)
    total_xp: float = 0.0
    level: int = 1
    streak_days: int = 0
    parent_user_ids: List[str] = Field(default_factory=list, description="Parent user IDs")
    classroom_ids: List[str] = Field(default_factory=list, description="Enrolled classrooms")


class TeacherProfile(BaseModel):
    """Teacher-specific profile data"""
    teacher_id: str
    subjects: List[str] = Field(default_factory=list, description="Subjects taught")
    classroom_ids: List[str] = Field(default_factory=list, description="Classrooms managed")
    student_count: int = 0
    bio: Optional[str] = None
    office_hours: Optional[str] = None


class ParentProfile(BaseModel):
    """Parent-specific profile data"""
    parent_id: str
    children: List[str] = Field(default_factory=list, description="Student IDs of children")
    notification_preferences: Dict[str, bool] = Field(
        default_factory=lambda: {
            "progress_reports": True,
            "achievements": True,
            "assignments": True,
            "low_performance": True
        }
    )


class AdminProfile(BaseModel):
    """Admin-specific profile data"""
    admin_id: str
    permissions: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Admin permissions (* = all)"
    )


class Classroom(BaseModel):
    """Classroom model"""
    classroom_id: str
    name: str
    subject: str
    grade: int
    teacher_id: str
    student_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True


# ============================================================================
# Storage (In-Memory - Replace with Database)
# ============================================================================

# In-memory stores (will be replaced with Neo4j/PostgreSQL)
_students: Dict[str, StudentProfile] = {}
_teachers: Dict[str, TeacherProfile] = {}
_parents: Dict[str, ParentProfile] = {}
_admins: Dict[str, AdminProfile] = {}
_classrooms: Dict[str, Classroom] = {}

# Lookup indexes
_user_id_to_username: Dict[str, str] = {}
_email_to_username: Dict[str, str] = {}


# ============================================================================
# User Manager
# ============================================================================

class UserManager:
    """
    Centralized user management system

    Handles all user CRUD operations, relationships, and role management.
    """

    def __init__(self):
        """Initialize user manager"""
        self._initialized = False

    async def initialize(self):
        """Initialize user manager (load from database, etc.)"""
        if self._initialized:
            return

        logger.info("🔧 Initializing User Manager...")

        # TODO: Load users from database
        # For now, create default admin
        from .auth import create_default_admin
        await create_default_admin()

        self._initialized = True
        logger.info("✅ User Manager initialized")

    # ========================================================================
    # User CRUD Operations
    # ========================================================================

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str,
        role: UserRole,
        **profile_data
    ) -> User:
        """
        Create new user

        Args:
            username: Unique username
            email: Email address
            password: Plain text password
            full_name: Full name
            role: User role
            **profile_data: Role-specific profile data

        Returns:
            Created user

        Raises:
            HTTPException: If validation fails or user exists
        """
        # Validate password
        is_valid, error_msg = validate_password_strength(password)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        # Check if username exists
        if username in _users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists"
            )

        # Check if email exists
        if email in _email_to_username.values():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Generate IDs
        import secrets
        user_id = f"user_{secrets.token_hex(8)}"

        # Create base user
        hashed_password = hash_password(password)

        user = UserInDB(
            user_id=user_id,
            username=username,
            email=email,
            full_name=full_name,
            role=role,
            hashed_password=hashed_password,
            is_active=True,
            created_at=datetime.utcnow()
        )

        # Create role-specific profile
        if role == UserRole.STUDENT:
            student_id = f"student_{secrets.token_hex(8)}"
            user.student_id = student_id

            profile = StudentProfile(
                student_id=student_id,
                **profile_data
            )
            _students[student_id] = profile

        elif role == UserRole.TEACHER:
            teacher_id = f"teacher_{secrets.token_hex(8)}"
            user.teacher_id = teacher_id

            profile = TeacherProfile(
                teacher_id=teacher_id,
                **profile_data
            )
            _teachers[teacher_id] = profile

        elif role == UserRole.PARENT:
            parent_id = f"parent_{secrets.token_hex(8)}"

            profile = ParentProfile(
                parent_id=parent_id,
                **profile_data
            )
            _parents[parent_id] = profile

        elif role == UserRole.ADMIN:
            admin_id = f"admin_{secrets.token_hex(8)}"

            profile = AdminProfile(
                admin_id=admin_id,
                **profile_data
            )
            _admins[admin_id] = profile

        # Store user
        _users_db[username] = user
        _user_id_to_username[user_id] = username
        _email_to_username[email] = username

        logger.info(f"✅ Created user: {username} (role: {role}, id: {user_id})")

        return User(**user.dict(exclude={"hashed_password"}))

    async def get_user(self, username: str) -> Optional[User]:
        """Get user by username"""
        user = _users_db.get(username)
        if user:
            return User(**user.dict(exclude={"hashed_password"}))
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by user ID"""
        username = _user_id_to_username.get(user_id)
        if username:
            return await self.get_user(username)
        return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        username = _email_to_username.get(email)
        if username:
            return await self.get_user(username)
        return None

    async def update_user(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> User:
        """
        Update user

        Args:
            user_id: User ID
            updates: Fields to update

        Returns:
            Updated user
        """
        username = _user_id_to_username.get(user_id)
        if not username:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        user = _users_db[username]

        # Update allowed fields
        allowed_fields = {"full_name", "email", "is_active"}
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(user, field, value)

        _users_db[username] = user

        logger.info(f"✅ Updated user: {username}")

        return User(**user.dict(exclude={"hashed_password"}))

    async def delete_user(self, user_id: str) -> bool:
        """
        Delete user (soft delete - set is_active=False)

        Args:
            user_id: User ID

        Returns:
            True if successful
        """
        username = _user_id_to_username.get(user_id)
        if not username:
            return False

        user = _users_db[username]
        user.is_active = False
        _users_db[username] = user

        logger.info(f"✅ Deleted user: {username}")

        return True

    async def list_users(
        self,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[User]:
        """
        List users with filtering

        Args:
            role: Filter by role
            is_active: Filter by active status
            limit: Maximum results
            offset: Skip results

        Returns:
            List of users
        """
        users = []

        for user in _users_db.values():
            # Apply filters
            if role is not None and user.role != role:
                continue
            if is_active is not None and user.is_active != is_active:
                continue

            users.append(User(**user.dict(exclude={"hashed_password"})))

        # Pagination
        return users[offset:offset + limit]

    # ========================================================================
    # Profile Management
    # ========================================================================

    async def get_student_profile(self, student_id: str) -> Optional[StudentProfile]:
        """Get student profile"""
        return _students.get(student_id)

    async def get_teacher_profile(self, teacher_id: str) -> Optional[TeacherProfile]:
        """Get teacher profile"""
        return _teachers.get(teacher_id)

    async def get_parent_profile(self, parent_id: str) -> Optional[ParentProfile]:
        """Get parent profile"""
        return _parents.get(parent_id)

    async def update_student_profile(
        self,
        student_id: str,
        updates: Dict[str, Any]
    ) -> StudentProfile:
        """Update student profile"""
        if student_id not in _students:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        profile = _students[student_id]

        for field, value in updates.items():
            if hasattr(profile, field):
                setattr(profile, field, value)

        _students[student_id] = profile

        return profile

    # ========================================================================
    # Relationships
    # ========================================================================

    async def link_parent_to_student(self, parent_user_id: str, student_id: str):
        """
        Link parent to student

        Args:
            parent_user_id: Parent's user ID
            student_id: Student ID
        """
        # Get parent
        parent_user = await self.get_user_by_id(parent_user_id)
        if not parent_user or parent_user.role != UserRole.PARENT:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid parent user"
            )

        # Get student profile
        if student_id not in _students:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        # Find parent profile
        parent_profile = None
        for profile in _parents.values():
            # Find parent by checking user
            if student_id not in profile.children:
                profile.children.append(student_id)
                parent_profile = profile
                break

        # Update student
        student = _students[student_id]
        if parent_user_id not in student.parent_user_ids:
            student.parent_user_ids.append(parent_user_id)

        logger.info(f"✅ Linked parent {parent_user_id} to student {student_id}")

    async def enroll_student_in_classroom(self, student_id: str, classroom_id: str):
        """
        Enroll student in classroom

        Args:
            student_id: Student ID
            classroom_id: Classroom ID
        """
        if student_id not in _students:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Student not found"
            )

        if classroom_id not in _classrooms:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Classroom not found"
            )

        # Update student
        student = _students[student_id]
        if classroom_id not in student.classroom_ids:
            student.classroom_ids.append(classroom_id)

        # Update classroom
        classroom = _classrooms[classroom_id]
        if student_id not in classroom.student_ids:
            classroom.student_ids.append(student_id)

        # Update teacher
        teacher = _teachers.get(classroom.teacher_id)
        if teacher:
            teacher.student_count = len(classroom.student_ids)

        logger.info(f"✅ Enrolled student {student_id} in classroom {classroom_id}")

    # ========================================================================
    # Classroom Management
    # ========================================================================

    async def create_classroom(
        self,
        name: str,
        subject: str,
        grade: int,
        teacher_id: str
    ) -> Classroom:
        """
        Create classroom

        Args:
            name: Classroom name
            subject: Subject
            grade: Grade level
            teacher_id: Teacher ID

        Returns:
            Created classroom
        """
        if teacher_id not in _teachers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Teacher not found"
            )

        import secrets
        classroom_id = f"classroom_{secrets.token_hex(8)}"

        classroom = Classroom(
            classroom_id=classroom_id,
            name=name,
            subject=subject,
            grade=grade,
            teacher_id=teacher_id
        )

        _classrooms[classroom_id] = classroom

        # Update teacher
        teacher = _teachers[teacher_id]
        if classroom_id not in teacher.classroom_ids:
            teacher.classroom_ids.append(classroom_id)

        logger.info(f"✅ Created classroom: {name} (id: {classroom_id})")

        return classroom

    async def get_classroom(self, classroom_id: str) -> Optional[Classroom]:
        """Get classroom by ID"""
        return _classrooms.get(classroom_id)

    async def list_teacher_classrooms(self, teacher_id: str) -> List[Classroom]:
        """List classrooms for teacher"""
        return [
            classroom for classroom in _classrooms.values()
            if classroom.teacher_id == teacher_id
        ]

    async def list_student_classrooms(self, student_id: str) -> List[Classroom]:
        """List classrooms for student"""
        student = _students.get(student_id)
        if not student:
            return []

        return [
            _classrooms[cid] for cid in student.classroom_ids
            if cid in _classrooms
        ]


# ============================================================================
# Convenience Functions
# ============================================================================

# Global user manager instance
_user_manager: Optional[UserManager] = None


async def get_user_manager() -> UserManager:
    """Get global user manager instance"""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
        await _user_manager.initialize()
    return _user_manager


async def create_student_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    grade: int,
    learning_preferences: Optional[Dict[str, float]] = None
) -> User:
    """
    Create student user

    Args:
        username: Username
        email: Email
        password: Password
        full_name: Full name
        grade: Grade level (4-12)
        learning_preferences: VARK learning style scores

    Returns:
        Created user
    """
    mgr = await get_user_manager()

    return await mgr.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=UserRole.STUDENT,
        grade=grade,
        learning_preferences=learning_preferences or {}
    )


async def create_teacher_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    subjects: List[str],
    bio: Optional[str] = None
) -> User:
    """Create teacher user"""
    mgr = await get_user_manager()

    return await mgr.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=UserRole.TEACHER,
        subjects=subjects,
        bio=bio
    )


async def create_parent_user(
    username: str,
    email: str,
    password: str,
    full_name: str
) -> User:
    """Create parent user"""
    mgr = await get_user_manager()

    return await mgr.create_user(
        username=username,
        email=email,
        password=password,
        full_name=full_name,
        role=UserRole.PARENT
    )


async def link_parent_to_student(parent_user_id: str, student_id: str):
    """Link parent to student"""
    mgr = await get_user_manager()
    await mgr.link_parent_to_student(parent_user_id, student_id)

"""
EdWIN Authentication System

JWT-based authentication with role-based access control for EdWIN AI Tutor.

**Implementation Date**: November 15, 2025

**Features**:
- JWT token generation and validation
- OAuth2 password flow
- Role-based access control (Student, Teacher, Admin, Parent)
- Password hashing with bcrypt
- Token refresh mechanism
- Session management

**Roles & Permissions**:
- Student: Access own progress, ask questions, view assignments
- Teacher: View classroom, manage students, create assignments
- Admin: Platform-wide access, user management, system configuration
- Parent: View child's progress (read-only), receive notifications

Usage:
    from EduVerse.edwin.auth import (
        create_access_token,
        get_current_user,
        require_role,
        authenticate_user
    )

    # In FastAPI endpoints
    @app.post("/login")
    async def login(credentials: OAuth2PasswordRequestForm = Depends()):
        user = await authenticate_user(credentials.username, credentials.password)
        token = create_access_token(user.user_id, user.role)
        return {"access_token": token, "token_type": "bearer"}

    @app.get("/student/profile")
    async def get_profile(current_user: User = Depends(get_current_user)):
        # Only authenticated users can access
        return current_user.dict()

    @app.get("/admin/users")
    async def list_users(current_user: User = Depends(require_role("admin"))):
        # Only admins can access
        return await get_all_users()
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import secrets
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, EmailStr

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# JWT Configuration (override with environment variables in production)
SECRET_KEY = secrets.token_urlsafe(32)  # Generate secure random key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ============================================================================
# Enums
# ============================================================================

class UserRole(str, Enum):
    """User roles with hierarchical permissions"""
    STUDENT = "student"
    TEACHER = "teacher"
    PARENT = "parent"
    ADMIN = "admin"

    def __lt__(self, other):
        """Role hierarchy: STUDENT < PARENT < TEACHER < ADMIN"""
        hierarchy = {
            UserRole.STUDENT: 0,
            UserRole.PARENT: 1,
            UserRole.TEACHER: 2,
            UserRole.ADMIN: 3
        }
        return hierarchy[self] < hierarchy[other]


class PermissionType(str, Enum):
    """Permission types"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    MANAGE = "manage"


# ============================================================================
# Data Models
# ============================================================================

class User(BaseModel):
    """User model"""
    user_id: str = Field(..., description="Unique user ID")
    username: str = Field(..., description="Username")
    email: EmailStr = Field(..., description="Email address")
    full_name: str = Field(..., description="Full name")
    role: UserRole = Field(..., description="User role")
    is_active: bool = Field(True, description="Account active status")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    # Role-specific data
    student_id: Optional[str] = None  # For students
    teacher_id: Optional[str] = None  # For teachers
    parent_of: Optional[List[str]] = None  # For parents (list of student IDs)
    classroom_ids: Optional[List[str]] = None  # For teachers


class UserInDB(User):
    """User model with hashed password (stored in database)"""
    hashed_password: str


class Token(BaseModel):
    """Access token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until expiration
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    username: str
    role: UserRole
    exp: datetime


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str


# ============================================================================
# Password Utilities
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash

    Args:
        plain_password: Plain text password
        hashed_password: Hashed password from database

    Returns:
        True if password matches, False otherwise
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def validate_password_strength(password: str) -> tuple[bool, str]:
    """
    Validate password meets security requirements

    Requirements:
    - Minimum 8 characters
    - At least one uppercase letter
    - At least one lowercase letter
    - At least one digit
    - At least one special character

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"

    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"

    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"

    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    if not any(c in special_chars for c in password):
        return False, "Password must contain at least one special character"

    return True, ""


# ============================================================================
# JWT Token Management
# ============================================================================

def create_access_token(
    user_id: str,
    username: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token

    Args:
        user_id: User ID
        username: Username
        role: User role
        expires_delta: Token expiration time (default: ACCESS_TOKEN_EXPIRE_MINUTES)

    Returns:
        JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.utcnow() + expires_delta

    payload = {
        "sub": user_id,
        "username": username,
        "role": role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


def create_refresh_token(user_id: str) -> str:
    """
    Create JWT refresh token

    Args:
        user_id: User ID

    Returns:
        JWT refresh token string
    """
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "sub": user_id,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token


def decode_token(token: str) -> TokenData:
    """
    Decode and validate JWT token

    Args:
        token: JWT token string

    Returns:
        TokenData object

    Raises:
        HTTPException: If token is invalid or expired
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        user_id: str = payload.get("sub")
        username: str = payload.get("username")
        role_str: str = payload.get("role")
        exp_timestamp: int = payload.get("exp")

        if user_id is None or username is None or role_str is None:
            raise credentials_exception

        # Convert expiration timestamp to datetime
        exp = datetime.fromtimestamp(exp_timestamp)

        # Parse role
        try:
            role = UserRole(role_str)
        except ValueError:
            raise credentials_exception

        return TokenData(
            user_id=user_id,
            username=username,
            role=role,
            exp=exp
        )

    except JWTError as e:
        logger.error(f"JWT decode error: {e}")
        raise credentials_exception


# ============================================================================
# User Authentication
# ============================================================================

# In-memory user store (replace with database in production)
# This is a placeholder - will be replaced by user_management.py
_users_db: Dict[str, UserInDB] = {}


async def get_user(username: str) -> Optional[UserInDB]:
    """
    Retrieve user from database by username

    Args:
        username: Username

    Returns:
        User object or None if not found
    """
    # TODO: Replace with actual database query
    return _users_db.get(username)


async def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password

    Args:
        username: Username
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    user = await get_user(username)

    if not user:
        logger.warning(f"Login attempt for non-existent user: {username}")
        return None

    if not verify_password(password, user.hashed_password):
        logger.warning(f"Failed login attempt for user: {username}")
        return None

    if not user.is_active:
        logger.warning(f"Login attempt for inactive user: {username}")
        return None

    # Update last login
    user.last_login = datetime.utcnow()
    # TODO: Persist to database

    logger.info(f"Successful login: {username} (role: {user.role})")

    # Return user without password
    return User(**user.dict(exclude={"hashed_password"}))


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Get current authenticated user from JWT token

    Dependency for FastAPI endpoints requiring authentication.

    Args:
        token: JWT token from Authorization header

    Returns:
        Current user

    Raises:
        HTTPException: If authentication fails
    """
    token_data = decode_token(token)

    user = await get_user(token_data.username)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )

    return User(**user.dict(exclude={"hashed_password"}))


# ============================================================================
# Role-Based Access Control
# ============================================================================

def require_role(*allowed_roles: UserRole):
    """
    Dependency factory for role-based access control

    Usage:
        @app.get("/admin/dashboard")
        async def admin_dashboard(user: User = Depends(require_role(UserRole.ADMIN))):
            return {"message": "Admin dashboard"}

    Args:
        *allowed_roles: Roles allowed to access the endpoint

    Returns:
        FastAPI dependency function
    """
    async def check_role(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {', '.join(r.value for r in allowed_roles)}"
            )
        return current_user

    return check_role


def require_permission(permission: PermissionType, resource_type: str):
    """
    Dependency factory for permission-based access control

    Args:
        permission: Required permission type
        resource_type: Type of resource being accessed

    Returns:
        FastAPI dependency function
    """
    async def check_permission(current_user: User = Depends(get_current_user)) -> User:
        # Define role permissions
        role_permissions = {
            UserRole.STUDENT: {
                "profile": [PermissionType.READ, PermissionType.WRITE],
                "questions": [PermissionType.READ, PermissionType.WRITE],
                "progress": [PermissionType.READ],
            },
            UserRole.PARENT: {
                "child_progress": [PermissionType.READ],
                "notifications": [PermissionType.READ],
            },
            UserRole.TEACHER: {
                "classroom": [PermissionType.READ, PermissionType.WRITE, PermissionType.MANAGE],
                "students": [PermissionType.READ, PermissionType.WRITE],
                "assignments": [PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE],
            },
            UserRole.ADMIN: {
                # Admins have all permissions on all resources
                "*": [PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE, PermissionType.MANAGE],
            }
        }

        user_permissions = role_permissions.get(current_user.role, {})

        # Admin has access to everything
        if "*" in user_permissions:
            return current_user

        allowed_permissions = user_permissions.get(resource_type, [])

        if permission not in allowed_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {resource_type}"
            )

        return current_user

    return check_permission


def can_access_student_data(target_student_id: str):
    """
    Check if user can access specific student's data

    Rules:
    - Students can access their own data
    - Parents can access their children's data
    - Teachers can access their students' data
    - Admins can access all data

    Args:
        target_student_id: Student ID to check access for

    Returns:
        FastAPI dependency function
    """
    async def check_access(current_user: User = Depends(get_current_user)) -> User:
        # Admins have access to all
        if current_user.role == UserRole.ADMIN:
            return current_user

        # Students can access their own data
        if current_user.role == UserRole.STUDENT and current_user.student_id == target_student_id:
            return current_user

        # Parents can access their children's data
        if current_user.role == UserRole.PARENT:
            if current_user.parent_of and target_student_id in current_user.parent_of:
                return current_user

        # Teachers can access their students' data
        if current_user.role == UserRole.TEACHER:
            # TODO: Check if student is in teacher's classroom
            # For now, allow all teachers (will be refined with database)
            return current_user

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this student's data"
        )

    return check_access


# ============================================================================
# Session Management
# ============================================================================

class Session(BaseModel):
    """User session"""
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


# In-memory session store (replace with Redis in production)
_sessions: Dict[str, Session] = {}


def create_session(user_id: str, ip_address: Optional[str] = None, user_agent: Optional[str] = None) -> Session:
    """
    Create new user session

    Args:
        user_id: User ID
        ip_address: Client IP address
        user_agent: Client user agent

    Returns:
        Session object
    """
    session_id = secrets.token_urlsafe(32)
    now = datetime.utcnow()
    expires_at = now + timedelta(hours=24)

    session = Session(
        session_id=session_id,
        user_id=user_id,
        created_at=now,
        last_activity=now,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent
    )

    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    """Get session by ID"""
    session = _sessions.get(session_id)

    if session and session.expires_at < datetime.utcnow():
        # Session expired
        del _sessions[session_id]
        return None

    return session


def invalidate_session(session_id: str):
    """Invalidate (logout) session"""
    if session_id in _sessions:
        del _sessions[session_id]


def invalidate_user_sessions(user_id: str):
    """Invalidate all sessions for a user"""
    sessions_to_remove = [
        sid for sid, session in _sessions.items()
        if session.user_id == user_id
    ]

    for sid in sessions_to_remove:
        del _sessions[sid]


# ============================================================================
# Utility Functions
# ============================================================================

async def create_user(
    username: str,
    email: str,
    password: str,
    full_name: str,
    role: UserRole,
    **kwargs
) -> User:
    """
    Create new user (placeholder - will be implemented in user_management.py)

    Args:
        username: Username
        email: Email address
        password: Plain text password
        full_name: Full name
        role: User role
        **kwargs: Additional role-specific fields

    Returns:
        Created user

    Raises:
        HTTPException: If validation fails
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

    # Create user
    user_id = f"user_{secrets.token_hex(8)}"
    hashed_password = hash_password(password)

    user = UserInDB(
        user_id=user_id,
        username=username,
        email=email,
        full_name=full_name,
        role=role,
        hashed_password=hashed_password,
        **kwargs
    )

    _users_db[username] = user

    logger.info(f"Created user: {username} (role: {role})")

    return User(**user.dict(exclude={"hashed_password"}))


# ============================================================================
# Bootstrap Admin User
# ============================================================================

async def create_default_admin():
    """Create default admin user if none exists"""
    admin_username = "admin"

    if admin_username not in _users_db:
        await create_user(
            username=admin_username,
            email="admin@edwin.edu",
            password="Admin123!@#",  # Change in production!
            full_name="System Administrator",
            role=UserRole.ADMIN
        )
        logger.info("✅ Created default admin user (username: admin, password: Admin123!@#)")
        logger.warning("⚠️  CHANGE DEFAULT ADMIN PASSWORD IN PRODUCTION!")

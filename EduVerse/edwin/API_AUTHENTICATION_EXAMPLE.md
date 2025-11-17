# EdWIN API Authentication Integration

**Example of integrating authentication into existing EdWIN APIs**

**Implementation Date**: November 15, 2025

---

## Overview

This document shows how to integrate the authentication system into existing EdWIN API endpoints.

## Authentication Flow

```
1. User logs in → Receives JWT token
2. Client includes token in Authorization header
3. API validates token → Extracts user info
4. API checks permissions → Allows/denies access
```

---

## Login Endpoint

Add to `api.py`:

```python
from fastapi.security import OAuth2PasswordRequestForm
from EduVerse.edwin.auth import authenticate_user, create_access_token, Token

@app.post("/auth/login", response_model=Token)
async def login(credentials: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint - returns JWT token

    Requires:
        - username: Username or email
        - password: User password

    Returns:
        - access_token: JWT token
        - token_type: "bearer"
        - expires_in: Seconds until expiration
    """
    # Authenticate user
    user = await authenticate_user(credentials.username, credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token = create_access_token(
        user_id=user.user_id,
        username=user.username,
        role=user.role
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=24 * 60 * 60  # 24 hours
    )
```

---

## Protected Endpoints

### Example 1: Student Profile (Self-Access)

**Original** (no auth):
```python
@app.get("/student/{student_id}/profile")
async def get_profile(student_id: str):
    # Anyone can access any student profile
    return get_student_profile(student_id)
```

**With Authentication**:
```python
from EduVerse.edwin.auth import get_current_user, can_access_student_data, User

@app.get("/student/{student_id}/profile")
async def get_profile(
    student_id: str,
    current_user: User = Depends(can_access_student_data(student_id))
):
    """
    Get student profile

    Authorization:
    - Students can access their own profile
    - Parents can access their children's profiles
    - Teachers can access their students' profiles
    - Admins can access all profiles
    """
    return get_student_profile(student_id)
```

### Example 2: Teacher Dashboard (Role-Based)

**Original** (no auth):
```python
@app.get("/teacher/classroom/{classroom_id}")
async def get_classroom(classroom_id: str):
    # Anyone can access any classroom
    return get_classroom_data(classroom_id)
```

**With Authentication**:
```python
from EduVerse.edwin.auth import require_role, UserRole

@app.get("/teacher/classroom/{classroom_id}")
async def get_classroom(
    classroom_id: str,
    current_user: User = Depends(require_role(UserRole.TEACHER, UserRole.ADMIN))
):
    """
    Get classroom details

    Authorization: Teachers and Admins only
    """
    # Additional check: verify teacher owns this classroom
    if current_user.role == UserRole.TEACHER:
        teacher_classrooms = get_teacher_classrooms(current_user.teacher_id)
        if classroom_id not in teacher_classrooms:
            raise HTTPException(403, "You don't have access to this classroom")

    return get_classroom_data(classroom_id)
```

### Example 3: Admin Operations

**Original** (no auth):
```python
@app.post("/admin/users")
async def create_user_admin(user_data: UserCreate):
    # Anyone can create users
    return create_user(**user_data.dict())
```

**With Authentication**:
```python
@app.post("/admin/users")
async def create_user_admin(
    user_data: UserCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """
    Create new user (admin only)

    Authorization: Admins only
    """
    new_user = await create_user(**user_data.dict())

    # Audit log
    logger.info(f"ADMIN_ACTION: User {new_user.username} created by {current_user.username}")

    return new_user
```

---

## Question/Answer Endpoint

**Original** (no auth):
```python
@app.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    # Anyone can ask questions as any student
    return await edwin.teach(request.question)
```

**With Authentication**:
```python
@app.post("/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(can_access_student_data(request.student_id))
):
    """
    Ask a question to EdWIN AI tutor

    Authorization:
    - Students can ask questions for themselves
    - Parents can ask for their children
    - Teachers can ask for their students
    - Admins have full access
    """
    # Track who asked the question
    logger.info(f"QUESTION: student={request.student_id}, asked_by={current_user.username}")

    result = await edwin.teach(request.question)

    return result
```

---

## Progress Tracking

**Original** (no auth):
```python
@app.get("/student/{student_id}/progress")
async def get_progress(student_id: str):
    return get_student_progress(student_id)
```

**With Authentication**:
```python
@app.get("/student/{student_id}/progress")
async def get_progress(
    student_id: str,
    current_user: User = Depends(can_access_student_data(student_id))
):
    """
    Get student progress

    Authorization: Student/Parent/Teacher/Admin
    """
    progress = get_student_progress(student_id)

    # Redact sensitive data for parents
    if current_user.role == UserRole.PARENT:
        # Parents see summary only, not detailed answers
        progress = redact_for_parent(progress)

    return progress
```

---

## Rate Limiting

Add rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/question")
@limiter.limit("10/minute")  # 10 questions per minute
async def ask_question(
    request: Request,
    question: QuestionRequest,
    current_user: User = Depends(get_current_user)
):
    # Rate limited per IP
    ...
```

---

## Full Example: Updated API Structure

```python
# EduVerse/edwin/api.py

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from EduVerse.edwin.auth import (
    authenticate_user,
    create_access_token,
    get_current_user,
    require_role,
    can_access_student_data,
    User,
    UserRole,
    Token
)

app = FastAPI(title="EdWIN API", version="1.0.0")

# ==============================================================================
# Authentication Endpoints
# ==============================================================================

@app.post("/auth/login", response_model=Token)
async def login(credentials: OAuth2PasswordRequestForm = Depends()):
    """Login and receive JWT token"""
    user = await authenticate_user(credentials.username, credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )

    token = create_access_token(user.user_id, user.username, user.role)

    return Token(access_token=token, token_type="bearer", expires_in=24*60*60)


@app.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout (invalidate session)"""
    # Invalidate session if using session management
    return {"message": "Logged out successfully"}


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user


# ==============================================================================
# Student Endpoints (Protected)
# ==============================================================================

@app.post("/student", response_model=StudentResponse)
async def create_student(
    student: StudentCreate,
    current_user: User = Depends(require_role(UserRole.TEACHER, UserRole.ADMIN))
):
    """Create new student (teachers and admins only)"""
    # Create student logic
    ...


@app.get("/student/{student_id}/profile")
async def get_student_profile(
    student_id: str,
    current_user: User = Depends(can_access_student_data(student_id))
):
    """Get student profile (student/parent/teacher/admin)"""
    return get_profile(student_id)


@app.post("/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    current_user: User = Depends(can_access_student_data(request.student_id))
):
    """Ask question to AI tutor"""
    result = await edwin.teach(request.question)
    return result


# ==============================================================================
# Teacher Endpoints (Protected)
# ==============================================================================

@app.get("/teacher/classrooms")
async def list_classrooms(
    current_user: User = Depends(require_role(UserRole.TEACHER, UserRole.ADMIN))
):
    """List teacher's classrooms"""
    return get_teacher_classrooms(current_user.teacher_id)


@app.get("/teacher/classroom/{classroom_id}/students")
async def get_classroom_students(
    classroom_id: str,
    current_user: User = Depends(require_role(UserRole.TEACHER, UserRole.ADMIN))
):
    """Get students in classroom"""
    # Verify teacher owns classroom
    if current_user.role == UserRole.TEACHER:
        if not teacher_owns_classroom(current_user.teacher_id, classroom_id):
            raise HTTPException(403, "Access denied")

    return get_students_in_classroom(classroom_id)


# ==============================================================================
# Admin Endpoints (Protected)
# ==============================================================================

@app.get("/admin/users")
async def list_all_users(
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """List all users (admins only)"""
    return get_all_users()


@app.post("/admin/users")
async def create_user_admin(
    user_data: UserCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Create new user (admins only)"""
    new_user = await create_user(**user_data.dict())

    logger.info(f"ADMIN: User {new_user.username} created by {current_user.username}")

    return new_user
```

---

## Testing Authentication

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Login
response = requests.post(
    f"{BASE_URL}/auth/login",
    data={"username": "student1", "password": "Pass123!"}
)
token = response.json()["access_token"]

# 2. Use token in requests
headers = {"Authorization": f"Bearer {token}"}

# Get profile (authenticated)
response = requests.get(
    f"{BASE_URL}/student/student_001/profile",
    headers=headers
)
print(response.json())

# Ask question (authenticated)
response = requests.post(
    f"{BASE_URL}/question",
    headers=headers,
    json={
        "student_id": "student_001",
        "question": "What is photosynthesis?",
        "mode": "verify"
    }
)
print(response.json())
```

---

## Migration Checklist

- [ ] Add authentication imports to api.py
- [ ] Add login endpoint
- [ ] Add `Depends(get_current_user)` to all endpoints
- [ ] Add role-based checks where needed
- [ ] Add audit logging for sensitive operations
- [ ] Update API documentation
- [ ] Update client code to include tokens
- [ ] Test all endpoints with authentication
- [ ] Enable rate limiting

---

**Last Updated**: November 15, 2025

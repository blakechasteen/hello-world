"""
Pydantic schemas for authentication
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import UUID


class UserRegister(BaseModel):
    """User registration request"""

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    display_name: Optional[str] = Field(None, max_length=100)


class UserLogin(BaseModel):
    """User login request"""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenRefresh(BaseModel):
    """Token refresh request"""

    refresh_token: str


class PasswordChange(BaseModel):
    """Password change request"""

    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class PasswordReset(BaseModel):
    """Password reset request"""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""

    token: str
    new_password: str = Field(..., min_length=8, max_length=100)


class UserResponse(BaseModel):
    """User response (public profile)"""

    user_id: UUID
    username: str
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    reputation_score: int
    trust_level: int
    created_at: str

    class Config:
        from_attributes = True

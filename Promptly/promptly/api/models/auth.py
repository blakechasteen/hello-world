"""
Authentication API Models
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime


class Token(BaseModel):
    """JWT Token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = None
    scopes: list[str] = []


class User(BaseModel):
    """User model"""
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: bool = False
    scopes: list[str] = []


class UserLogin(BaseModel):
    """User login request"""
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)


class APIKeyCreate(BaseModel):
    """Create API key request"""
    name: str = Field(..., description="API key name/description")
    scopes: list[str] = Field(default=[], description="Permitted scopes")
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response"""
    key_id: str
    name: str
    api_key: str  # Only returned on creation
    scopes: list[str]
    created_at: datetime
    expires_at: Optional[datetime]


class APIKeyListItem(BaseModel):
    """API key list item (without actual key)"""
    key_id: str
    name: str
    scopes: list[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used: Optional[datetime]

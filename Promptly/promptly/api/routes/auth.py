"""
Authentication API Routes
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from ..models.auth import Token, User, UserLogin, APIKeyCreate, APIKeyResponse, APIKeyListItem
from ..middleware.auth import (
    create_access_token,
    generate_api_key,
    API_KEYS,
    USERS,
    get_current_user
)
from ..config import get_settings
from datetime import datetime
import uuid

settings = get_settings()
router = APIRouter()


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login and get JWT access token

    Authenticate with username/password and receive a JWT token.
    """
    # Verify credentials (in production, hash passwords and check database)
    user = USERS.get(form_data.username)

    if not user or form_data.password != "admin":  # Simplified for demo
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.scopes},
        expires_delta=access_token_expires
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/api-keys", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    key_request: APIKeyCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new API key

    Generates a new API key with specified scopes and expiration.
    """
    # Generate API key
    api_key = generate_api_key()
    key_id = str(uuid.uuid4())

    # Calculate expiration
    expires_at = None
    if key_request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=key_request.expires_in_days)

    # Store API key
    API_KEYS[api_key] = {
        "key_id": key_id,
        "name": key_request.name,
        "scopes": key_request.scopes,
        "created_at": datetime.utcnow(),
        "expires_at": expires_at,
        "created_by": current_user.username
    }

    return APIKeyResponse(
        key_id=key_id,
        name=key_request.name,
        api_key=api_key,  # Only returned on creation
        scopes=key_request.scopes,
        created_at=datetime.utcnow(),
        expires_at=expires_at
    )


@router.get("/api-keys", response_model=list[APIKeyListItem])
async def list_api_keys(current_user: User = Depends(get_current_user)):
    """
    List all API keys

    Returns all API keys (without the actual key value).
    """
    keys = []

    for key_data in API_KEYS.values():
        keys.append(
            APIKeyListItem(
                key_id=key_data["key_id"],
                name=key_data["name"],
                scopes=key_data["scopes"],
                created_at=key_data["created_at"],
                expires_at=key_data.get("expires_at"),
                last_used=key_data.get("last_used")
            )
        )

    return keys


@router.delete("/api-keys/{key_id}")
async def delete_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete an API key

    Revokes an API key.
    """
    # Find and delete key
    key_to_delete = None

    for api_key, key_data in API_KEYS.items():
        if key_data["key_id"] == key_id:
            key_to_delete = api_key
            break

    if not key_to_delete:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key '{key_id}' not found"
        )

    del API_KEYS[key_to_delete]

    return {"status": "success", "message": f"API key '{key_id}' deleted"}


@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information

    Returns information about the authenticated user.
    """
    return current_user

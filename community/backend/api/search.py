"""
Search API endpoints using Elasticsearch
"""
from fastapi import APIRouter, Query, Depends
from typing import Optional, List
from pydantic import BaseModel

from search.elasticsearch_client import es_client
from auth.dependencies import get_optional_user
from models.user import User


class SearchResponse(BaseModel):
    """Search response"""

    total: int
    hits: List[dict]
    page: int
    page_size: int
    pages: int = 0


class AutocompleteResponse(BaseModel):
    """Autocomplete response"""

    suggestions: List[str]


router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/posts", response_model=SearchResponse)
async def search_posts(
    q: str = Query(..., min_length=1, description="Search query"),
    community_id: Optional[str] = Query(None, description="Filter by community ID"),
    author_id: Optional[str] = Query(None, description="Filter by author ID"),
    post_type: Optional[str] = Query(None, description="Filter by post type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    sort_by: str = Query("relevance", pattern=r"^(relevance|recent|popular)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Search posts with full-text search

    Supports:
    - Full-text search across title, content, tags
    - Fuzzy matching for typos
    - Filters (community, author, type, tags)
    - Sorting (relevance, recent, popular)
    - Pagination
    - Highlighted results
    """
    # Build filters
    filters = {}
    if community_id:
        filters['community_id'] = community_id
    if author_id:
        filters['author_id'] = author_id
    if post_type:
        filters['post_type'] = post_type
    if tags:
        filters['tags'] = tags

    # Search
    results = await es_client.search_posts(
        query=q,
        filters=filters,
        sort_by=sort_by,
        page=page,
        page_size=page_size
    )

    return SearchResponse(**results)


@router.get("/users", response_model=SearchResponse)
async def search_users(
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Search users by username, display name, or bio
    """
    results = await es_client.search_users(
        query=q,
        page=page,
        page_size=page_size
    )

    return SearchResponse(**results, pages=0)  # Calculate pages if needed


@router.get("/communities", response_model=SearchResponse)
async def search_communities(
    q: str = Query(..., min_length=1, description="Search query"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """
    Search communities by name or description
    """
    results = await es_client.search_communities(
        query=q,
        page=page,
        page_size=page_size
    )

    return SearchResponse(**results, pages=0)


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete(
    q: str = Query(..., min_length=1, description="Partial query"),
    type: str = Query("posts", pattern=r"^(posts|users|communities)$"),
    limit: int = Query(10, ge=1, le=20),
):
    """
    Autocomplete suggestions for search

    Returns:
        List of suggestions based on partial query
    """
    suggestions = await es_client.autocomplete(
        query=q,
        index_type=type,
        limit=limit
    )

    return AutocompleteResponse(suggestions=suggestions)


@router.post("/index/posts/{post_id}")
async def index_post(post_id: str):
    """
    Manually index a post (admin/development use)
    """
    # TODO: Fetch post from database and index
    return {"message": "Post indexed (not implemented)"}


@router.delete("/index/posts/{post_id}")
async def delete_post_index(post_id: str):
    """
    Delete a post from search index (admin use)
    """
    # TODO: Delete from Elasticsearch
    return {"message": "Post removed from index (not implemented)"}

"""
Elasticsearch client for full-text search

Indexes:
- posts: Search posts by title, content, tags
- users: Search users by username, display_name, bio
- communities: Search communities by name, description
"""
from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Any, Optional
import logging
from uuid import UUID

from core.config import settings

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Elasticsearch client for full-text search
    """

    def __init__(self):
        """Initialize Elasticsearch client"""
        self.client = AsyncElasticsearch([settings.ELASTICSEARCH_URL])
        self.indices = {
            'posts': 'posts_index',
            'users': 'users_index',
            'communities': 'communities_index',
        }

    async def create_indices(self):
        """
        Create Elasticsearch indices with mappings
        """
        # Posts index
        posts_mapping = {
            "mappings": {
                "properties": {
                    "post_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "author_id": {"type": "keyword"},
                    "author_username": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "community_id": {"type": "keyword"},
                    "community_name": {"type": "keyword"},
                    "post_type": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    "vote_score": {"type": "integer"},
                    "hot_score": {"type": "float"},
                    "comment_count": {"type": "integer"},
                    "created_at": {"type": "date"},
                    "status": {"type": "keyword"},
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        }

        # Users index
        users_mapping = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "username": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "display_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "bio": {"type": "text"},
                    "reputation_score": {"type": "integer"},
                    "trust_level": {"type": "integer"},
                    "created_at": {"type": "date"},
                }
            }
        }

        # Communities index
        communities_mapping = {
            "mappings": {
                "properties": {
                    "community_id": {"type": "keyword"},
                    "name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "display_name": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    },
                    "description": {"type": "text"},
                    "community_type": {"type": "keyword"},
                    "member_count": {"type": "integer"},
                    "post_count": {"type": "integer"},
                    "created_at": {"type": "date"},
                }
            }
        }

        # Create indices
        for index_type, index_name in self.indices.items():
            if not await self.client.indices.exists(index=index_name):
                if index_type == 'posts':
                    await self.client.indices.create(index=index_name, body=posts_mapping)
                elif index_type == 'users':
                    await self.client.indices.create(index=index_name, body=users_mapping)
                elif index_type == 'communities':
                    await self.client.indices.create(index=index_name, body=communities_mapping)

                logger.info(f"Created Elasticsearch index: {index_name}")

    async def index_post(self, post_data: Dict[str, Any]):
        """
        Index a post document

        Args:
            post_data: Post data dictionary
        """
        try:
            await self.client.index(
                index=self.indices['posts'],
                id=str(post_data['post_id']),
                document=post_data
            )
            logger.debug(f"Indexed post: {post_data['post_id']}")
        except Exception as e:
            logger.error(f"Error indexing post: {e}")

    async def index_user(self, user_data: Dict[str, Any]):
        """
        Index a user document
        """
        try:
            await self.client.index(
                index=self.indices['users'],
                id=str(user_data['user_id']),
                document=user_data
            )
            logger.debug(f"Indexed user: {user_data['user_id']}")
        except Exception as e:
            logger.error(f"Error indexing user: {e}")

    async def index_community(self, community_data: Dict[str, Any]):
        """
        Index a community document
        """
        try:
            await self.client.index(
                index=self.indices['communities'],
                id=str(community_data['community_id']),
                document=community_data
            )
            logger.debug(f"Indexed community: {community_data['community_id']}")
        except Exception as e:
            logger.error(f"Error indexing community: {e}")

    async def search_posts(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "relevance",
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search posts with full-text search

        Args:
            query: Search query string
            filters: Optional filters (community_id, author_id, post_type, tags)
            sort_by: Sort order (relevance, recent, popular)
            page: Page number
            page_size: Results per page

        Returns:
            Search results with total count and hits
        """
        offset = (page - 1) * page_size

        # Build search query
        must_clauses = []

        # Full-text search
        if query:
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": ["title^3", "content", "tags^2"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            })

        # Filters
        filter_clauses = []
        if filters:
            if 'community_id' in filters:
                filter_clauses.append({"term": {"community_id": filters['community_id']}})
            if 'author_id' in filters:
                filter_clauses.append({"term": {"author_id": filters['author_id']}})
            if 'post_type' in filters:
                filter_clauses.append({"term": {"post_type": filters['post_type']}})
            if 'tags' in filters:
                filter_clauses.append({"terms": {"tags": filters['tags']}})

        # Only published posts
        filter_clauses.append({"term": {"status": "published"}})

        # Build bool query
        bool_query = {
            "bool": {
                "must": must_clauses if must_clauses else [{"match_all": {}}],
                "filter": filter_clauses
            }
        }

        # Sorting
        sort_order = []
        if sort_by == "recent":
            sort_order = [{"created_at": {"order": "desc"}}]
        elif sort_by == "popular":
            sort_order = [{"hot_score": {"order": "desc"}}]
        else:  # relevance (default)
            sort_order = ["_score", {"created_at": {"order": "desc"}}]

        # Execute search
        try:
            result = await self.client.search(
                index=self.indices['posts'],
                body={
                    "query": bool_query,
                    "sort": sort_order,
                    "from": offset,
                    "size": page_size,
                    "highlight": {
                        "fields": {
                            "title": {},
                            "content": {}
                        },
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"]
                    }
                }
            )

            # Extract results
            hits = []
            for hit in result['hits']['hits']:
                post = hit['_source']
                post['_score'] = hit['_score']

                # Add highlights
                if 'highlight' in hit:
                    post['_highlights'] = hit['highlight']

                hits.append(post)

            return {
                "total": result['hits']['total']['value'],
                "hits": hits,
                "page": page,
                "page_size": page_size,
                "pages": (result['hits']['total']['value'] + page_size - 1) // page_size
            }

        except Exception as e:
            logger.error(f"Error searching posts: {e}")
            return {"total": 0, "hits": [], "page": page, "page_size": page_size, "pages": 0}

    async def search_users(
        self,
        query: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search users
        """
        offset = (page - 1) * page_size

        search_query = {
            "multi_match": {
                "query": query,
                "fields": ["username^3", "display_name^2", "bio"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }

        try:
            result = await self.client.search(
                index=self.indices['users'],
                body={
                    "query": search_query,
                    "sort": ["_score", {"reputation_score": {"order": "desc"}}],
                    "from": offset,
                    "size": page_size
                }
            )

            hits = [hit['_source'] for hit in result['hits']['hits']]

            return {
                "total": result['hits']['total']['value'],
                "hits": hits,
                "page": page,
                "page_size": page_size
            }

        except Exception as e:
            logger.error(f"Error searching users: {e}")
            return {"total": 0, "hits": [], "page": page, "page_size": page_size}

    async def search_communities(
        self,
        query: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        Search communities
        """
        offset = (page - 1) * page_size

        search_query = {
            "multi_match": {
                "query": query,
                "fields": ["name^3", "display_name^2", "description"],
                "type": "best_fields",
                "fuzziness": "AUTO"
            }
        }

        try:
            result = await self.client.search(
                index=self.indices['communities'],
                body={
                    "query": search_query,
                    "sort": ["_score", {"member_count": {"order": "desc"}}],
                    "from": offset,
                    "size": page_size
                }
            )

            hits = [hit['_source'] for hit in result['hits']['hits']]

            return {
                "total": result['hits']['total']['value'],
                "hits": hits,
                "page": page,
                "page_size": page_size
            }

        except Exception as e:
            logger.error(f"Error searching communities: {e}")
            return {"total": 0, "hits": [], "page": page, "page_size": page_size}

    async def autocomplete(
        self,
        query: str,
        index_type: str = "posts",
        limit: int = 10
    ) -> List[str]:
        """
        Autocomplete suggestions

        Args:
            query: Partial query string
            index_type: Index to search (posts, users, communities)
            limit: Max suggestions

        Returns:
            List of suggestions
        """
        index = self.indices.get(index_type, self.indices['posts'])

        # Field to search based on index type
        field_map = {
            'posts': 'title',
            'users': 'username',
            'communities': 'name'
        }
        field = field_map.get(index_type, 'title')

        try:
            result = await self.client.search(
                index=index,
                body={
                    "query": {
                        "match_phrase_prefix": {
                            field: {
                                "query": query,
                                "max_expansions": limit
                            }
                        }
                    },
                    "size": limit,
                    "_source": [field]
                }
            )

            suggestions = [hit['_source'][field] for hit in result['hits']['hits']]
            return suggestions

        except Exception as e:
            logger.error(f"Error in autocomplete: {e}")
            return []

    async def close(self):
        """Close Elasticsearch client"""
        await self.client.close()


# Global Elasticsearch client instance
es_client = ElasticsearchClient()

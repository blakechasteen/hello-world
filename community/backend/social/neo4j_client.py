"""
Neo4j client for social graph operations

Manages:
- Friend relationships (bidirectional)
- Follow relationships (unidirectional)
- Block relationships
- Community memberships
- Interest tags
- Recommendations
"""
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional
import logging
from uuid import UUID

from core.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j client for social graph operations
    """

    def __init__(self):
        """Initialize Neo4j driver"""
        self.driver = AsyncGraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    async def close(self):
        """Close Neo4j driver"""
        await self.driver.close()

    async def create_user_node(self, user_id: str, username: str, metadata: Optional[Dict] = None):
        """
        Create a user node in the graph

        Args:
            user_id: User UUID
            username: Username
            metadata: Additional user metadata
        """
        async with self.driver.session() as session:
            query = """
            MERGE (u:User {user_id: $user_id})
            SET u.username = $username,
                u.created_at = datetime(),
                u.metadata = $metadata
            RETURN u
            """
            result = await session.run(
                query,
                user_id=user_id,
                username=username,
                metadata=metadata or {}
            )
            await result.consume()
            logger.debug(f"Created user node: {user_id}")

    async def add_friend(self, user_id: str, friend_id: str):
        """
        Create bidirectional friend relationship

        Args:
            user_id: User UUID
            friend_id: Friend UUID
        """
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})
            MATCH (f:User {user_id: $friend_id})
            MERGE (u)-[r1:FRIENDS_WITH {since: datetime()}]->(f)
            MERGE (f)-[r2:FRIENDS_WITH {since: datetime()}]->(u)
            RETURN r1, r2
            """
            result = await session.run(query, user_id=user_id, friend_id=friend_id)
            await result.consume()
            logger.info(f"Added friend relationship: {user_id} <-> {friend_id}")

    async def remove_friend(self, user_id: str, friend_id: str):
        """Remove friend relationship (bidirectional)"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[r:FRIENDS_WITH]-(f:User {user_id: $friend_id})
            DELETE r
            """
            result = await session.run(query, user_id=user_id, friend_id=friend_id)
            await result.consume()
            logger.info(f"Removed friend relationship: {user_id} <-> {friend_id}")

    async def add_follow(self, user_id: str, target_id: str):
        """
        Create unidirectional follow relationship

        Args:
            user_id: Follower UUID
            target_id: User being followed
        """
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})
            MATCH (t:User {user_id: $target_id})
            MERGE (u)-[r:FOLLOWS {since: datetime()}]->(t)
            RETURN r
            """
            result = await session.run(query, user_id=user_id, target_id=target_id)
            await result.consume()
            logger.info(f"Added follow: {user_id} -> {target_id}")

    async def remove_follow(self, user_id: str, target_id: str):
        """Remove follow relationship"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[r:FOLLOWS]->(t:User {user_id: $target_id})
            DELETE r
            """
            result = await session.run(query, user_id=user_id, target_id=target_id)
            await result.consume()
            logger.info(f"Removed follow: {user_id} -> {target_id}")

    async def add_block(self, user_id: str, blocked_id: str):
        """Block a user"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})
            MATCH (b:User {user_id: $blocked_id})
            MERGE (u)-[r:BLOCKS {since: datetime()}]->(b)
            RETURN r
            """
            result = await session.run(query, user_id=user_id, blocked_id=blocked_id)
            await result.consume()
            logger.info(f"Added block: {user_id} -> {blocked_id}")

    async def remove_block(self, user_id: str, blocked_id: str):
        """Unblock a user"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[r:BLOCKS]->(b:User {user_id: $blocked_id})
            DELETE r
            """
            result = await session.run(query, user_id=user_id, blocked_id=blocked_id)
            await result.consume()
            logger.info(f"Removed block: {user_id} -> {blocked_id}")

    async def get_friends(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all friends of a user

        Returns:
            List of friend user data
        """
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:FRIENDS_WITH]-(friend:User)
            RETURN friend.user_id as user_id, friend.username as username
            LIMIT $limit
            """
            result = await session.run(query, user_id=user_id, limit=limit)
            friends = []
            async for record in result:
                friends.append({
                    'user_id': record['user_id'],
                    'username': record['username']
                })
            return friends

    async def get_followers(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get users following this user"""
        async with self.driver.session() as session:
            query = """
            MATCH (follower:User)-[:FOLLOWS]->(u:User {user_id: $user_id})
            RETURN follower.user_id as user_id, follower.username as username
            LIMIT $limit
            """
            result = await session.run(query, user_id=user_id, limit=limit)
            followers = []
            async for record in result:
                followers.append({
                    'user_id': record['user_id'],
                    'username': record['username']
                })
            return followers

    async def get_following(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get users this user is following"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:FOLLOWS]->(following:User)
            RETURN following.user_id as user_id, following.username as username
            LIMIT $limit
            """
            result = await session.run(query, user_id=user_id, limit=limit)
            following = []
            async for record in result:
                following.append({
                    'user_id': record['user_id'],
                    'username': record['username']
                })
            return following

    async def get_mutual_friends(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Find mutual friends (friends of friends who are not yet friends)

        Returns:
            List of suggested users with mutual friend count
        """
        async with self.driver.session() as session:
            query = """
            MATCH (me:User {user_id: $user_id})-[:FRIENDS_WITH]-(friend)-[:FRIENDS_WITH]-(mutual)
            WHERE mutual.user_id <> $user_id
              AND NOT (me)-[:FRIENDS_WITH]-(mutual)
              AND NOT (me)-[:BLOCKS]->(mutual)
              AND NOT (mutual)-[:BLOCKS]->(me)
            RETURN mutual.user_id as user_id,
                   mutual.username as username,
                   COUNT(DISTINCT friend) as mutual_count
            ORDER BY mutual_count DESC
            LIMIT $limit
            """
            result = await session.run(query, user_id=user_id, limit=limit)
            suggestions = []
            async for record in result:
                suggestions.append({
                    'user_id': record['user_id'],
                    'username': record['username'],
                    'mutual_count': record['mutual_count']
                })
            return suggestions

    async def get_recommended_users(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Recommend users based on shared interests and community memberships

        Returns:
            List of recommended users with relevance scores
        """
        async with self.driver.session() as session:
            query = """
            MATCH (me:User {user_id: $user_id})
            MATCH (me)-[:INTERESTED_IN]->(tag:Tag)<-[:INTERESTED_IN]-(suggested:User)
            WHERE suggested.user_id <> $user_id
              AND NOT (me)-[:FOLLOWS]->(suggested)
              AND NOT (me)-[:BLOCKS]->(suggested)
            WITH suggested, COUNT(DISTINCT tag) as shared_interests
            OPTIONAL MATCH (me)-[:MEMBER_OF]->(community:Community)<-[:MEMBER_OF]-(suggested)
            WITH suggested, shared_interests, COUNT(DISTINCT community) as shared_communities
            RETURN suggested.user_id as user_id,
                   suggested.username as username,
                   shared_interests,
                   shared_communities,
                   (shared_interests * 2 + shared_communities) as relevance
            ORDER BY relevance DESC
            LIMIT $limit
            """
            result = await session.run(query, user_id=user_id, limit=limit)
            recommendations = []
            async for record in result:
                recommendations.append({
                    'user_id': record['user_id'],
                    'username': record['username'],
                    'shared_interests': record['shared_interests'],
                    'shared_communities': record['shared_communities'],
                    'relevance': record['relevance']
                })
            return recommendations

    async def get_community_recommendations(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recommend communities based on interests and friend memberships

        Returns:
            List of recommended communities
        """
        async with self.driver.session() as session:
            query = """
            MATCH (me:User {user_id: $user_id})
            MATCH (me)-[:INTERESTED_IN]->(tag:Tag)<-[:TAGGED_WITH]-(community:Community)
            WHERE NOT (me)-[:MEMBER_OF]->(community)
            WITH community, COUNT(DISTINCT tag) as interest_match
            OPTIONAL MATCH (me)-[:FRIENDS_WITH]-(friend)-[:MEMBER_OF]->(community)
            WITH community, interest_match, COUNT(DISTINCT friend) as friend_count
            RETURN community.community_id as community_id,
                   community.name as name,
                   interest_match,
                   friend_count,
                   (interest_match * 3 + friend_count) as relevance
            ORDER BY relevance DESC
            LIMIT $limit
            """
            result = await session.run(query, user_id=user_id, limit=limit)
            recommendations = []
            async for record in result:
                recommendations.append({
                    'community_id': record['community_id'],
                    'name': record['name'],
                    'interest_match': record['interest_match'],
                    'friend_count': record['friend_count'],
                    'relevance': record['relevance']
                })
            return recommendations

    async def is_friends_with(self, user_id: str, other_id: str) -> bool:
        """Check if two users are friends"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:FRIENDS_WITH]-(other:User {user_id: $other_id})
            RETURN COUNT(*) > 0 as are_friends
            """
            result = await session.run(query, user_id=user_id, other_id=other_id)
            record = await result.single()
            return record['are_friends'] if record else False

    async def is_following(self, user_id: str, target_id: str) -> bool:
        """Check if user is following target"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:FOLLOWS]->(t:User {user_id: $target_id})
            RETURN COUNT(*) > 0 as is_following
            """
            result = await session.run(query, user_id=user_id, target_id=target_id)
            record = await result.single()
            return record['is_following'] if record else False

    async def is_blocking(self, user_id: str, blocked_id: str) -> bool:
        """Check if user is blocking another user"""
        async with self.driver.session() as session:
            query = """
            MATCH (u:User {user_id: $user_id})-[:BLOCKS]->(b:User {user_id: $blocked_id})
            RETURN COUNT(*) > 0 as is_blocking
            """
            result = await session.run(query, user_id=user_id, blocked_id=blocked_id)
            record = await result.single()
            return record['is_blocking'] if record else False


# Global Neo4j client instance
neo4j_client = Neo4jClient()

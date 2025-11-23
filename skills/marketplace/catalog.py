"""
Unified Skill Catalog
======================

**Created**: November 22, 2025
**Purpose**: Browse, search, and discover skills across all sources

The catalog provides:
- Unified view of all skills (meta, domain, agentic)
- Rich metadata (ratings, usage stats, dependencies)
- Search and filtering (by category, tags, capabilities)
- Version tracking and compatibility
- Popularity metrics (downloads, usage)

Architecture:
```
SkillCatalog
├── Skills Database (SQLite)
│   ├── skill_id (primary key)
│   ├── name, version, category
│   ├── description, tags, capabilities
│   ├── author, homepage, license
│   ├── install_count, usage_count
│   ├── rating (1-5 stars)
│   └── created_at, updated_at
│
├── Search Engine
│   ├── Full-text search (name, description, tags)
│   ├── Filter by category, tags, capabilities
│   ├── Sort by popularity, rating, recency
│   └── Fuzzy matching
│
└── Metadata Cache
    ├── In-memory for fast lookups
    └── Periodic sync with database
```
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import json
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class SkillCategory(Enum):
    """Skill category."""
    META = "meta"
    DOMAIN = "domain"
    AGENTIC = "agentic"


class SortBy(Enum):
    """Sort criteria for catalog."""
    POPULARITY = "popularity"  # By install_count + usage_count
    RATING = "rating"  # By average rating
    RECENCY = "recency"  # By created_at
    NAME = "name"  # Alphabetical
    RELEVANCE = "relevance"  # By search score


@dataclass
class SkillMetadata:
    """
    Complete metadata for a skill in the catalog.

    This is richer than SkillManifest - includes metrics, ratings, etc.
    """
    # Identity
    skill_id: str  # Unique ID (e.g., "code_reviewer:1.0.0")
    name: str
    version: str
    category: SkillCategory

    # Description
    description: str
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)

    # Source
    source_type: str = "meta"  # "meta", "domain", "agentic"
    source_path: str = ""
    domain: str = ""  # For domain skills

    # Requirements
    requires_warp_space: bool = False
    requires_yarn_graph: bool = False
    requires_event_bus: bool = False
    requires_mission_control: bool = False
    requires_tools: List[str] = field(default_factory=list)
    python_version: str = ">=3.8"
    dependencies: List[str] = field(default_factory=list)

    # Author info
    author: str = ""
    author_email: str = ""
    homepage: str = ""
    repository: str = ""
    license: str = "MIT"

    # Metrics
    install_count: int = 0
    usage_count: int = 0
    rating: float = 0.0  # 0-5 stars
    rating_count: int = 0

    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str = ""

    # Compatibility
    compatible_with: List[str] = field(default_factory=list)  # Skill names
    incompatible_with: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['category'] = self.category.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillMetadata':
        """Create from dictionary."""
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = SkillCategory(data['category'])
        return cls(**data)


@dataclass
class SearchResult:
    """Search result with relevance score."""
    skill: SkillMetadata
    score: float  # 0.0-1.0 relevance score
    matched_fields: List[str]  # Which fields matched


# ============================================================================
# Skill Catalog
# ============================================================================

class SkillCatalog:
    """
    Unified catalog for all skills.

    Provides:
    - Browse all skills
    - Search by name, description, tags
    - Filter by category, capabilities
    - Sort by popularity, rating, recency
    - Track usage metrics
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize catalog.

        Args:
            db_path: Path to SQLite database (default: ./skills_catalog.db)
        """
        self.db_path = db_path or Path("./skills_catalog.db")
        self.conn: Optional[sqlite3.Connection] = None
        self._metadata_cache: Dict[str, SkillMetadata] = {}

    async def initialize(self):
        """Initialize catalog database."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        await self._create_tables()
        await self._load_cache()
        logger.info(f"Catalog initialized: {self.db_path}")

    async def close(self):
        """Close catalog database."""
        if self.conn:
            self.conn.close()
            self.conn = None

    async def _create_tables(self):
        """Create database tables."""
        cursor = self.conn.cursor()

        # Skills table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS skills (
                skill_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT,
                tags TEXT,  -- JSON array
                capabilities TEXT,  -- JSON array
                source_type TEXT,
                source_path TEXT,
                domain TEXT,
                requires_warp_space INTEGER,
                requires_yarn_graph INTEGER,
                requires_event_bus INTEGER,
                requires_mission_control INTEGER,
                requires_tools TEXT,  -- JSON array
                python_version TEXT,
                dependencies TEXT,  -- JSON array
                author TEXT,
                author_email TEXT,
                homepage TEXT,
                repository TEXT,
                license TEXT,
                install_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                rating_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                last_used_at TEXT,
                compatible_with TEXT,  -- JSON array
                incompatible_with TEXT  -- JSON array
            )
        """)

        # Ratings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                user_id TEXT,
                rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                comment TEXT,
                created_at TEXT,
                FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
            )
        """)

        # Usage history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS usage_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_id TEXT NOT NULL,
                session_id TEXT,
                success INTEGER,
                execution_time_ms REAL,
                tokens_used INTEGER,
                timestamp TEXT,
                FOREIGN KEY (skill_id) REFERENCES skills(skill_id)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_category ON skills(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_rating ON skills(rating DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_skills_install_count ON skills(install_count DESC)")

        self.conn.commit()

    async def _load_cache(self):
        """Load metadata cache from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM skills")

        for row in cursor.fetchall():
            metadata = self._row_to_metadata(row)
            self._metadata_cache[metadata.skill_id] = metadata

        logger.info(f"Loaded {len(self._metadata_cache)} skills into cache")

    def _row_to_metadata(self, row: sqlite3.Row) -> SkillMetadata:
        """Convert database row to metadata."""
        data = dict(row)

        # Parse JSON fields
        for field in ['tags', 'capabilities', 'requires_tools', 'dependencies',
                      'compatible_with', 'incompatible_with']:
            if data.get(field):
                data[field] = json.loads(data[field])
            else:
                data[field] = []

        # Convert booleans
        for field in ['requires_warp_space', 'requires_yarn_graph',
                      'requires_event_bus', 'requires_mission_control']:
            data[field] = bool(data.get(field, 0))

        # Convert category
        data['category'] = SkillCategory(data['category'])

        return SkillMetadata(**data)

    def _metadata_to_values(self, metadata: SkillMetadata) -> tuple:
        """Convert metadata to database values."""
        return (
            metadata.skill_id,
            metadata.name,
            metadata.version,
            metadata.category.value,
            metadata.description,
            json.dumps(metadata.tags),
            json.dumps(metadata.capabilities),
            metadata.source_type,
            metadata.source_path,
            metadata.domain,
            int(metadata.requires_warp_space),
            int(metadata.requires_yarn_graph),
            int(metadata.requires_event_bus),
            int(metadata.requires_mission_control),
            json.dumps(metadata.requires_tools),
            metadata.python_version,
            json.dumps(metadata.dependencies),
            metadata.author,
            metadata.author_email,
            metadata.homepage,
            metadata.repository,
            metadata.license,
            metadata.install_count,
            metadata.usage_count,
            metadata.rating,
            metadata.rating_count,
            metadata.created_at,
            metadata.updated_at,
            metadata.last_used_at,
            json.dumps(metadata.compatible_with),
            json.dumps(metadata.incompatible_with)
        )

    async def add_skill(self, metadata: SkillMetadata) -> bool:
        """
        Add skill to catalog.

        Args:
            metadata: Skill metadata

        Returns:
            True if added, False if already exists
        """
        if metadata.skill_id in self._metadata_cache:
            logger.warning(f"Skill already in catalog: {metadata.skill_id}")
            return False

        # Set timestamps
        if not metadata.created_at:
            metadata.created_at = datetime.now().isoformat()
        if not metadata.updated_at:
            metadata.updated_at = metadata.created_at

        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO skills VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, self._metadata_to_values(metadata))

        self.conn.commit()

        # Update cache
        self._metadata_cache[metadata.skill_id] = metadata

        logger.info(f"Added skill to catalog: {metadata.skill_id}")
        return True

    async def update_skill(self, metadata: SkillMetadata) -> bool:
        """
        Update skill in catalog.

        Args:
            metadata: Updated skill metadata

        Returns:
            True if updated, False if not found
        """
        if metadata.skill_id not in self._metadata_cache:
            logger.warning(f"Skill not in catalog: {metadata.skill_id}")
            return False

        # Update timestamp
        metadata.updated_at = datetime.now().isoformat()

        # Update database
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE skills SET
                name=?, version=?, category=?, description=?, tags=?, capabilities=?,
                source_type=?, source_path=?, domain=?,
                requires_warp_space=?, requires_yarn_graph=?, requires_event_bus=?, requires_mission_control=?,
                requires_tools=?, python_version=?, dependencies=?,
                author=?, author_email=?, homepage=?, repository=?, license=?,
                install_count=?, usage_count=?, rating=?, rating_count=?,
                created_at=?, updated_at=?, last_used_at=?,
                compatible_with=?, incompatible_with=?
            WHERE skill_id=?
        """, self._metadata_to_values(metadata)[1:] + (metadata.skill_id,))

        self.conn.commit()

        # Update cache
        self._metadata_cache[metadata.skill_id] = metadata

        logger.info(f"Updated skill in catalog: {metadata.skill_id}")
        return True

    async def remove_skill(self, skill_id: str) -> bool:
        """
        Remove skill from catalog.

        Args:
            skill_id: Skill ID to remove

        Returns:
            True if removed, False if not found
        """
        if skill_id not in self._metadata_cache:
            return False

        # Remove from database
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM skills WHERE skill_id=?", (skill_id,))
        cursor.execute("DELETE FROM ratings WHERE skill_id=?", (skill_id,))
        cursor.execute("DELETE FROM usage_history WHERE skill_id=?", (skill_id,))
        self.conn.commit()

        # Remove from cache
        del self._metadata_cache[skill_id]

        logger.info(f"Removed skill from catalog: {skill_id}")
        return True

    async def get_skill(self, skill_id: str) -> Optional[SkillMetadata]:
        """Get skill by ID."""
        return self._metadata_cache.get(skill_id)

    async def list_all(
        self,
        category: Optional[SkillCategory] = None,
        sort_by: SortBy = SortBy.NAME,
        limit: Optional[int] = None
    ) -> List[SkillMetadata]:
        """
        List all skills.

        Args:
            category: Filter by category
            sort_by: Sort criteria
            limit: Maximum results

        Returns:
            List of skill metadata
        """
        skills = list(self._metadata_cache.values())

        # Filter by category
        if category:
            skills = [s for s in skills if s.category == category]

        # Sort
        if sort_by == SortBy.NAME:
            skills.sort(key=lambda s: s.name.lower())
        elif sort_by == SortBy.POPULARITY:
            skills.sort(key=lambda s: s.install_count + s.usage_count, reverse=True)
        elif sort_by == SortBy.RATING:
            skills.sort(key=lambda s: s.rating, reverse=True)
        elif sort_by == SortBy.RECENCY:
            skills.sort(key=lambda s: s.created_at, reverse=True)

        # Limit
        if limit:
            skills = skills[:limit]

        return skills

    async def search(
        self,
        query: str,
        category: Optional[SkillCategory] = None,
        tags: Optional[List[str]] = None,
        capabilities: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[SearchResult]:
        """
        Search skills.

        Args:
            query: Search query (searches name, description, tags)
            category: Filter by category
            tags: Filter by tags (any match)
            capabilities: Filter by capabilities (any match)
            limit: Maximum results

        Returns:
            List of search results with relevance scores
        """
        results = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))

        for skill in self._metadata_cache.values():
            # Filter by category
            if category and skill.category != category:
                continue

            # Filter by tags
            if tags and not any(tag in skill.tags for tag in tags):
                continue

            # Filter by capabilities
            if capabilities and not any(cap in skill.capabilities for cap in capabilities):
                continue

            # Calculate relevance score
            score = 0.0
            matched_fields = []

            # Name match (highest weight)
            if query_lower in skill.name.lower():
                score += 1.0
                matched_fields.append('name')
            elif any(word in skill.name.lower() for word in query_words):
                score += 0.5
                matched_fields.append('name')

            # Description match
            if query_lower in skill.description.lower():
                score += 0.7
                matched_fields.append('description')
            elif any(word in skill.description.lower() for word in query_words):
                score += 0.3
                matched_fields.append('description')

            # Tags match
            if any(query_lower in tag.lower() for tag in skill.tags):
                score += 0.5
                matched_fields.append('tags')

            # Capabilities match
            if any(query_lower in cap.lower() for cap in skill.capabilities):
                score += 0.5
                matched_fields.append('capabilities')

            # Add result if score > 0
            if score > 0:
                results.append(SearchResult(
                    skill=skill,
                    score=score,
                    matched_fields=matched_fields
                ))

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        # Limit
        return results[:limit]

    async def record_install(self, skill_id: str) -> bool:
        """Record skill installation."""
        skill = await self.get_skill(skill_id)
        if not skill:
            return False

        skill.install_count += 1
        return await self.update_skill(skill)

    async def record_usage(
        self,
        skill_id: str,
        session_id: str,
        success: bool,
        execution_time_ms: float = 0.0,
        tokens_used: int = 0
    ) -> bool:
        """
        Record skill usage.

        Args:
            skill_id: Skill ID
            session_id: Session ID
            success: Whether execution succeeded
            execution_time_ms: Execution time
            tokens_used: Tokens used

        Returns:
            True if recorded
        """
        skill = await self.get_skill(skill_id)
        if not skill:
            return False

        # Update skill metrics
        skill.usage_count += 1
        skill.last_used_at = datetime.now().isoformat()
        await self.update_skill(skill)

        # Record in usage history
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO usage_history (skill_id, session_id, success, execution_time_ms, tokens_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (skill_id, session_id, int(success), execution_time_ms, tokens_used, datetime.now().isoformat()))

        self.conn.commit()
        return True

    async def add_rating(
        self,
        skill_id: str,
        rating: int,
        user_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """
        Add rating for skill.

        Args:
            skill_id: Skill ID
            rating: Rating (1-5 stars)
            user_id: Optional user ID
            comment: Optional comment

        Returns:
            True if added
        """
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be 1-5")

        skill = await self.get_skill(skill_id)
        if not skill:
            return False

        # Record rating
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO ratings (skill_id, user_id, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (skill_id, user_id, rating, comment, datetime.now().isoformat()))

        # Update skill average rating
        cursor.execute("""
            SELECT AVG(rating), COUNT(*) FROM ratings WHERE skill_id=?
        """, (skill_id,))
        avg_rating, count = cursor.fetchone()

        skill.rating = float(avg_rating)
        skill.rating_count = int(count)
        await self.update_skill(skill)

        self.conn.commit()
        return True

    async def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        return {
            'total_skills': len(self._metadata_cache),
            'by_category': {
                'meta': len([s for s in self._metadata_cache.values() if s.category == SkillCategory.META]),
                'domain': len([s for s in self._metadata_cache.values() if s.category == SkillCategory.DOMAIN]),
                'agentic': len([s for s in self._metadata_cache.values() if s.category == SkillCategory.AGENTIC])
            },
            'total_installs': sum(s.install_count for s in self._metadata_cache.values()),
            'total_usage': sum(s.usage_count for s in self._metadata_cache.values()),
            'avg_rating': sum(s.rating for s in self._metadata_cache.values()) / len(self._metadata_cache) if self._metadata_cache else 0.0
        }


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SkillCategory',
    'SortBy',
    'SkillMetadata',
    'SearchResult',
    'SkillCatalog',
]

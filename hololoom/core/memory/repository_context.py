"""
Repository Context Manager for Claude Code Agents
==================================================
Smart RAG system for managing multiple code repositories and controlling
agent access to specific codebases.

Features:
- Include/exclude repositories per agent
- Automatic code indexing with semantic search
- Repository-aware chunking (respects file/function boundaries)
- Fast retrieval with metadata filtering
- Access control and sandboxing

Usage:
    # Create repository manager
    repo_mgr = RepositoryContextManager()

    # Register repositories
    await repo_mgr.add_repository(
        name="hololoom",
        path="c:/Users/blake/OneDrive/Documents/mythRL/HoloLoom",
        tags=["python", "ml", "core"],
        access_level="public"
    )

    await repo_mgr.add_repository(
        name="COS",
        path="c:/Users/blake/OneDrive/Documents/mythRL/cos",
        tags=["business", "private"],
        access_level="private"
    )

    # Create agent with specific repository access
    agent_context = repo_mgr.create_agent_context(
        agent_id="frontend_agent",
        allowed_repos=["squad"],
        blocked_repos=["cos"],  # No access to business plans
        allowed_tags=["typescript", "react"]
    )

    # Query with agent context
    results = await agent_context.query(
        "How does the VS Code extension work?",
        limit=5
    )

Author: HoloLoom Team
Date: 2025-11-04
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

try:
    from .mcp_rag_server import (
        hybrid_search,
        init_memory,
        rag_query,
        rerank_results,
        semantic_chunk,
    )
    from .protocol import Memory, UnifiedMemoryInterface, create_unified_memory
except ImportError:
    # Fallback for standalone execution
    from mcp_rag_server import rag_query, semantic_chunk
    from protocol import UnifiedMemoryInterface, create_unified_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Access Control
# ============================================================================

class AccessLevel(Enum):
    """Repository access levels."""
    PUBLIC = "public"      # Anyone can access
    INTERNAL = "internal"  # Authenticated agents only
    PRIVATE = "private"    # Explicit permission required
    RESTRICTED = "restricted"  # Owner only


@dataclass
class RepositoryMetadata:
    """Metadata for a code repository."""

    name: str
    path: str
    tags: set[str] = field(default_factory=set)
    access_level: AccessLevel = AccessLevel.PUBLIC
    description: str = ""
    indexed_at: datetime | None = None
    file_count: int = 0
    line_count: int = 0
    last_modified: datetime | None = None

    # Patterns to ignore
    ignore_patterns: set[str] = field(default_factory=lambda: {
        "*.pyc", "__pycache__", "node_modules", ".git",
        ".venv", "venv", "*.egg-info", "dist", "build"
    })


@dataclass
class AgentContext:
    """Access context for a specific agent."""

    agent_id: str
    allowed_repos: set[str] = field(default_factory=set)  # Empty = all public
    blocked_repos: set[str] = field(default_factory=set)
    allowed_tags: set[str] = field(default_factory=set)   # Empty = all tags
    blocked_tags: set[str] = field(default_factory=set)
    access_level: AccessLevel = AccessLevel.PUBLIC

    def can_access_repo(self, repo: RepositoryMetadata) -> bool:
        """Check if agent can access this repository."""
        # Explicit block
        if repo.name in self.blocked_repos:
            return False

        # Access level check
        if repo.access_level == AccessLevel.RESTRICTED:
            return False

        if repo.access_level == AccessLevel.PRIVATE:
            return repo.name in self.allowed_repos

        # Public/Internal repos
        if self.allowed_repos:
            return repo.name in self.allowed_repos

        return True

    def can_access_tags(self, tags: set[str]) -> bool:
        """Check if agent can access content with these tags."""
        # Blocked tags
        if self.blocked_tags & tags:
            return False

        # Allowed tags (empty = all allowed)
        if self.allowed_tags:
            return bool(self.allowed_tags & tags)

        return True


# ============================================================================
# Code-Aware Chunking
# ============================================================================

def chunk_code_file(
    file_path: Path,
    content: str,
    max_chunk_size: int = 1000
) -> list[dict[str, Any]]:
    """
    Chunk code file respecting language semantics.

    Better than semantic_chunk() for code because it:
    - Preserves function/class boundaries
    - Includes relevant context (imports, class definition)
    - Maintains syntax highlighting metadata

    Args:
        file_path: Path to file
        content: File content
        max_chunk_size: Target chunk size

    Returns:
        List of chunk dicts with metadata
    """
    extension = file_path.suffix.lower()

    # Python: chunk by class/function
    if extension == '.py':
        return _chunk_python(file_path, content, max_chunk_size)

    # TypeScript/JavaScript: chunk by class/function
    elif extension in {'.ts', '.tsx', '.js', '.jsx'}:
        return _chunk_typescript(file_path, content, max_chunk_size)

    # Markdown: chunk by section
    elif extension in {'.md', '.mdx'}:
        return _chunk_markdown(file_path, content, max_chunk_size)

    # Fallback: semantic chunking
    else:
        chunks = semantic_chunk(content, max_chunk_size)
        return [
            {
                'text': chunk,
                'file_path': str(file_path),
                'extension': extension,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            for i, chunk in enumerate(chunks)
        ]


def _chunk_python(
    file_path: Path,
    content: str,
    max_chunk_size: int
) -> list[dict[str, Any]]:
    """Chunk Python file by class/function definitions."""
    chunks = []

    # Simple regex-based approach (could use AST for production)
    lines = content.split('\n')

    current_chunk = []
    current_type = None
    current_name = None
    imports = []

    for i, line in enumerate(lines):
        # Collect imports for context
        if line.strip().startswith(('import ', 'from ')):
            imports.append(line)

        # Detect class/function start
        if line.strip().startswith('class '):
            if current_chunk:
                chunks.append(_make_python_chunk(
                    file_path, current_chunk, current_type, current_name, imports, i
                ))
            current_chunk = [line]
            current_type = 'class'
            current_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()

        elif line.strip().startswith('def '):
            if current_chunk and len('\n'.join(current_chunk)) > max_chunk_size:
                chunks.append(_make_python_chunk(
                    file_path, current_chunk, current_type, current_name, imports, i
                ))
                current_chunk = [line]
            else:
                if current_chunk:
                    current_chunk.append(line)
                else:
                    current_chunk = [line]

            if current_type != 'class':  # Top-level function
                current_type = 'function'
                current_name = line.split('def ')[1].split('(')[0].strip()

        else:
            current_chunk.append(line)

    # Final chunk
    if current_chunk:
        chunks.append(_make_python_chunk(
            file_path, current_chunk, current_type, current_name, imports, len(lines)
        ))

    return chunks


def _make_python_chunk(
    file_path: Path,
    lines: list[str],
    chunk_type: str | None,
    name: str | None,
    imports: list[str],
    line_number: int
) -> dict[str, Any]:
    """Create Python chunk with metadata."""
    # Include imports for context
    context_lines = imports[:5] if imports else []  # Max 5 imports

    text = '\n'.join(context_lines + ['...'] + lines) if context_lines else '\n'.join(lines)

    return {
        'text': text,
        'file_path': str(file_path),
        'extension': '.py',
        'language': 'python',
        'chunk_type': chunk_type or 'code',
        'entity_name': name,
        'line_number': line_number,
        'has_imports': bool(imports)
    }


def _chunk_typescript(
    file_path: Path,
    content: str,
    max_chunk_size: int
) -> list[dict[str, Any]]:
    """Chunk TypeScript/JavaScript file."""
    # Similar to Python but with different patterns
    # For now, fallback to semantic chunking
    chunks = semantic_chunk(content, max_chunk_size)
    return [
        {
            'text': chunk,
            'file_path': str(file_path),
            'extension': file_path.suffix,
            'language': 'typescript',
            'chunk_index': i
        }
        for i, chunk in enumerate(chunks)
    ]


def _chunk_markdown(
    file_path: Path,
    content: str,
    max_chunk_size: int
) -> list[dict[str, Any]]:
    """Chunk Markdown by sections."""
    sections = []
    current_section = []
    current_heading = None

    for line in content.split('\n'):
        if line.startswith('#'):
            if current_section:
                sections.append((current_heading, '\n'.join(current_section)))
            current_section = [line]
            current_heading = line.strip('# ').strip()
        else:
            current_section.append(line)

    if current_section:
        sections.append((current_heading, '\n'.join(current_section)))

    return [
        {
            'text': text,
            'file_path': str(file_path),
            'extension': '.md',
            'language': 'markdown',
            'section_heading': heading,
            'chunk_index': i
        }
        for i, (heading, text) in enumerate(sections)
    ]


# ============================================================================
# Repository Context Manager
# ============================================================================

class RepositoryContextManager:
    """
    Manages multiple code repositories for agent access.

    Features:
    - Multi-repo indexing
    - Access control per agent
    - Tag-based filtering
    - Fast semantic search
    """

    def __init__(self, memory: UnifiedMemoryInterface | None = None):
        """
        Initialize repository manager.

        Args:
            memory: Optional memory backend (created if not provided)
        """
        self.memory = memory
        self.repositories: dict[str, RepositoryMetadata] = {}
        self.agent_contexts: dict[str, AgentContext] = {}

        # Memory ID -> Repository name mapping
        self._memory_to_repo: dict[str, str] = {}

    async def initialize(self):
        """Initialize memory backend if needed."""
        if self.memory is None:
            self.memory = await create_unified_memory(
                user_id="repository_manager",
                enable_mem0=False,
                enable_neo4j=True,
                enable_qdrant=True
            )
            logger.info("Repository manager initialized")

    async def add_repository(
        self,
        name: str,
        path: str,
        tags: set[str] | None = None,
        access_level: AccessLevel = AccessLevel.PUBLIC,
        description: str = "",
        auto_index: bool = True
    ) -> RepositoryMetadata:
        """
        Register a code repository.

        Args:
            name: Repository identifier
            path: Filesystem path
            tags: Repository tags for filtering
            access_level: Access control level
            description: Human-readable description
            auto_index: Automatically index files

        Returns:
            Repository metadata
        """
        repo_path = Path(path)

        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {path}")

        repo = RepositoryMetadata(
            name=name,
            path=str(repo_path),
            tags=tags or set(),
            access_level=access_level,
            description=description
        )

        self.repositories[name] = repo
        logger.info(f"Registered repository: {name} ({access_level.value})")

        if auto_index:
            await self.index_repository(name)

        return repo

    async def index_repository(
        self,
        name: str,
        file_extensions: set[str] | None = None
    ):
        """
        Index all files in repository.

        Args:
            name: Repository name
            file_extensions: File types to index (None = all text files)
        """
        if name not in self.repositories:
            raise ValueError(f"Repository not registered: {name}")

        repo = self.repositories[name]
        repo_path = Path(repo.path)

        # Default extensions
        if file_extensions is None:
            file_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.md', '.txt', '.json', '.yaml', '.yml'}

        logger.info(f"Indexing repository: {name}")

        indexed_files = 0
        total_lines = 0

        for file_path in repo_path.rglob('*'):
            # Skip ignored patterns
            if any(file_path.match(pattern) for pattern in repo.ignore_patterns):
                continue

            if not file_path.is_file():
                continue

            if file_path.suffix not in file_extensions:
                continue

            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')

                # Chunk file
                chunks = chunk_code_file(file_path, content)

                # Store each chunk
                for chunk in chunks:
                    # Combine repository tags with file metadata
                    chunk_tags = repo.tags.copy()
                    chunk_tags.add(f"repo:{name}")
                    chunk_tags.add(f"lang:{chunk.get('language', 'unknown')}")
                    chunk_tags.add(f"ext:{chunk['extension']}")

                    memory_id = await self.memory.store(
                        text=chunk['text'],
                        context={
                            'repository': name,
                            'file_path': chunk['file_path'],
                            'chunk_type': chunk.get('chunk_type', 'code'),
                            'entity_name': chunk.get('entity_name'),
                            'line_number': chunk.get('line_number', 0),
                        },
                        tags=list(chunk_tags)
                    )

                    self._memory_to_repo[memory_id] = name

                indexed_files += 1
                total_lines += len(content.split('\n'))

                if indexed_files % 10 == 0:
                    logger.info(f"Indexed {indexed_files} files...")

            except Exception as e:
                logger.warning(f"Failed to index {file_path}: {e}")

        # Update metadata
        repo.indexed_at = datetime.now()
        repo.file_count = indexed_files
        repo.line_count = total_lines

        logger.info(f"Indexed {indexed_files} files ({total_lines} lines) from {name}")

    def create_agent_context(
        self,
        agent_id: str,
        allowed_repos: set[str] | None = None,
        blocked_repos: set[str] | None = None,
        allowed_tags: set[str] | None = None,
        blocked_tags: set[str] | None = None,
        access_level: AccessLevel = AccessLevel.PUBLIC
    ) -> 'AgentQueryContext':
        """
        Create query context for specific agent.

        Args:
            agent_id: Agent identifier
            allowed_repos: Repos this agent can access (None = all public)
            blocked_repos: Repos this agent cannot access
            allowed_tags: Tags this agent can access (None = all)
            blocked_tags: Tags this agent cannot access
            access_level: Agent's access level

        Returns:
            AgentQueryContext for making queries
        """
        context = AgentContext(
            agent_id=agent_id,
            allowed_repos=allowed_repos or set(),
            blocked_repos=blocked_repos or set(),
            allowed_tags=allowed_tags or set(),
            blocked_tags=blocked_tags or set(),
            access_level=access_level
        )

        self.agent_contexts[agent_id] = context

        return AgentQueryContext(self, context)

    def list_repositories(
        self,
        agent_context: AgentContext | None = None
    ) -> list[RepositoryMetadata]:
        """
        List repositories accessible to agent.

        Args:
            agent_context: Agent context (None = list all)

        Returns:
            List of accessible repositories
        """
        if agent_context is None:
            return list(self.repositories.values())

        return [
            repo for repo in self.repositories.values()
            if agent_context.can_access_repo(repo)
        ]


class AgentQueryContext:
    """Query interface for specific agent with access control."""

    def __init__(self, manager: RepositoryContextManager, context: AgentContext):
        """
        Initialize agent query context.

        Args:
            manager: Repository manager
            context: Agent access context
        """
        self.manager = manager
        self.context = context

    async def query(
        self,
        query_text: str,
        limit: int = 5,
        repository: str | None = None,
        tags: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Query code with access control.

        Args:
            query_text: Search query
            limit: Max results
            repository: Specific repository (None = search all allowed)
            tags: Additional tag filters

        Returns:
            List of results with metadata
        """
        # Get accessible repositories
        accessible_repos = self.manager.list_repositories(self.context)

        if repository:
            # Filter to specific repo
            accessible_repos = [r for r in accessible_repos if r.name == repository]

            if not accessible_repos:
                logger.warning(f"Agent {self.context.agent_id} cannot access {repository}")
                return []

        # Build tag filter
        repo_names = {r.name for r in accessible_repos}
        tag_filter = {f"repo:{name}" for name in repo_names}

        if tags:
            # Check tag access
            if not self.context.can_access_tags(tags):
                logger.warning(f"Agent {self.context.agent_id} blocked from tags: {tags}")
                return []
            tag_filter.update(tags)

        # Query with RAG
        result = await rag_query(
            query_text,
            limit=limit,
            use_hyde=True,
            use_reranking=True
        )

        # Filter results by access control
        filtered_results = []
        for res in result['results']:
            # Check repository access
            repo_tag = next((t for t in res['tags'] if t.startswith('repo:')), None)
            if repo_tag:
                repo_name = repo_tag.split(':', 1)[1]
                if repo_name not in repo_names:
                    continue

            # Check tag access
            res_tags = set(res['tags'])
            if not self.context.can_access_tags(res_tags):
                continue

            filtered_results.append(res)

        logger.info(
            f"Agent {self.context.agent_id} query: '{query_text}' "
            f"→ {len(filtered_results)} results (filtered from {len(result['results'])})"
        )

        return filtered_results[:limit]

    def list_repositories(self) -> list[RepositoryMetadata]:
        """List repositories accessible to this agent."""
        return self.manager.list_repositories(self.context)


# ============================================================================
# Convenience Functions
# ============================================================================

async def create_repo_manager(
    repositories: dict[str, str] | None = None
) -> RepositoryContextManager:
    """
    Create and initialize repository manager.

    Args:
        repositories: Dict of {name: path} to auto-register

    Returns:
        Initialized RepositoryContextManager
    """
    manager = RepositoryContextManager()
    await manager.initialize()

    if repositories:
        for name, path in repositories.items():
            await manager.add_repository(name, path)

    return manager


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'AccessLevel',
    'RepositoryMetadata',
    'AgentContext',
    'RepositoryContextManager',
    'AgentQueryContext',
    'create_repo_manager',
]

"""
Workspace Scanner - Auto-index entire workspace into HoloLoom knowledge graph

Scans workspace directories and extracts:
- Code structure (functions, classes, imports)
- Comments (NOTE, TODO, FIXME)
- File relationships (imports, dependencies)

Respects .gitignore patterns for intelligent filtering.

Usage:
    from HoloLoom.spinningWheel import WorkspaceSpinner

    # Basic usage (creates MemoryShards)
    spinner = WorkspaceSpinner()
    shards = await spinner.spin_workspace("/path/to/workspace")

    # Integration with HoloLoom (stores in knowledge graph)
    from HoloLoom import HoloLoom
    async with HoloLoom() as loom:
        spinner = WorkspaceSpinner()
        stats = await spinner.index_to_hololoom(
            workspace_path="/path/to/workspace",
            hololoom_instance=loom,
            incremental=True  # Only re-index changed files
        )
        print(f"Indexed {stats['files']} files, {stats['entities']} entities, {stats['edges']} edges")
"""

import ast
import re
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.memory.graph import KGEdge

logger = logging.getLogger(__name__)


@dataclass
class CodeElement:
    """Represents a code element (function, class, etc.)"""
    type: str  # "function", "class", "import"
    name: str
    line: int
    docstring: Optional[str] = None
    parameters: Optional[List[str]] = None
    return_type: Optional[str] = None


@dataclass
class Comment:
    """Represents a code comment"""
    type: str  # "NOTE", "TODO", "FIXME", "GENERAL"
    text: str
    line: int
    context: Optional[str] = None  # Nearby code for context


class WorkspaceSpinner:
    """
    Scans workspace and extracts code structure + comments.

    Features:
    - Respects .gitignore patterns
    - AST parsing for Python
    - Regex parsing for TypeScript/JavaScript
    - Comment extraction (NOTE/TODO/FIXME)
    - Import dependency tracking
    """

    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.md': 'markdown',
        '.txt': 'text'
    }

    # Default patterns to ignore (even if not in .gitignore)
    DEFAULT_IGNORE_PATTERNS = {
        'node_modules', '.git', '__pycache__', '.pytest_cache',
        'dist', 'build', 'out', '.venv', 'venv', '.cache',
        '*.pyc', '*.pyo', '*.so', '*.dylib', '*.dll',
        '.DS_Store', 'Thumbs.db'
    }

    def __init__(self, enable_enrichment: bool = False):
        """
        Initialize workspace spinner.

        Args:
            enable_enrichment: Use Ollama to enrich entities/motifs (slower)
        """
        self.enable_enrichment = enable_enrichment
        self.gitignore_patterns: Set[str] = set()

    async def spin_workspace(
        self,
        workspace_path: str | Path,
        languages: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[MemoryShard]:
        """
        Scan entire workspace and create memory shards.

        Args:
            workspace_path: Path to workspace root
            languages: Languages to index (e.g., ["python", "typescript"])
            exclude_patterns: Additional glob patterns to exclude

        Returns:
            List of MemoryShards containing code structure + comments

        Example:
            >>> spinner = WorkspaceSpinner()
            >>> shards = await spinner.spin_workspace("/path/to/project")
            >>> len(shards)
            47  # Number of files processed
        """
        workspace_path = Path(workspace_path)

        if not workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        # Load .gitignore patterns
        self._load_gitignore(workspace_path)

        # Add user exclude patterns
        if exclude_patterns:
            self.gitignore_patterns.update(exclude_patterns)

        # Scan workspace
        shards = []
        for file_path in self._scan_directory(workspace_path, languages):
            try:
                shard = await self._process_file(file_path, workspace_path)
                if shard:
                    shards.append(shard)
                    logger.info(f"Indexed: {file_path.relative_to(workspace_path)}")
            except Exception as e:
                logger.warning(f"Failed to process {file_path}: {e}")
                continue

        logger.info(f"Workspace scan complete: {len(shards)} files indexed")
        return shards

    async def _process_file(
        self,
        file_path: Path,
        workspace_root: Path
    ) -> Optional[MemoryShard]:
        """
        Process single file and extract structure + comments.

        Args:
            file_path: Path to file
            workspace_root: Workspace root (for relative paths)

        Returns:
            MemoryShard or None if file should be skipped
        """
        # Read file content
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Binary file, skip
            return None

        # Get file language
        extension = file_path.suffix
        language = self.SUPPORTED_EXTENSIONS.get(extension)

        if not language:
            return None

        # Extract code elements
        elements: List[CodeElement] = []
        comments: List[Comment] = []

        if language == 'python':
            elements, comments = self._parse_python(content)
        elif language in ('typescript', 'javascript'):
            elements, comments = self._parse_typescript(content)
        elif language == 'markdown':
            comments = self._parse_markdown(content)

        # Build entities and motifs
        entities = [elem.name for elem in elements]
        motifs = self._extract_motifs(elements, comments)

        # Create descriptive text
        relative_path = file_path.relative_to(workspace_root)
        text_parts = [
            f"File: {relative_path}",
            f"Language: {language}",
            ""
        ]

        # Add code structure
        if elements:
            text_parts.append("Code Structure:")
            for elem in elements[:20]:  # Limit to first 20 elements
                if elem.type == 'function':
                    params = ', '.join(elem.parameters or [])
                    text_parts.append(f"  - function {elem.name}({params})")
                elif elem.type == 'class':
                    text_parts.append(f"  - class {elem.name}")
                elif elem.type == 'import':
                    text_parts.append(f"  - import {elem.name}")
            text_parts.append("")

        # Add important comments
        if comments:
            important_comments = [c for c in comments if c.type in ('NOTE', 'TODO', 'FIXME')]
            if important_comments:
                text_parts.append("Important Comments:")
                for comment in important_comments[:10]:  # Limit to first 10
                    text_parts.append(f"  [{comment.type}] {comment.text} (line {comment.line})")
                text_parts.append("")

        text = "\n".join(text_parts)

        # Create memory shard
        return MemoryShard(
            id=f"file_{hash(str(relative_path))}",
            text=text,
            episode=f"workspace_{workspace_root.name}",
            entities=entities,
            motifs=motifs,
            metadata={
                "file_path": str(relative_path),
                "language": language,
                "element_count": len(elements),
                "comment_count": len(comments),
                "todo_count": len([c for c in comments if c.type == 'TODO']),
                "note_count": len([c for c in comments if c.type == 'NOTE']),
                "fixme_count": len([c for c in comments if c.type == 'FIXME']),
                "source": "workspace_scan"
            }
        )

    def _parse_python(self, content: str) -> Tuple[List[CodeElement], List[Comment]]:
        """Parse Python file using AST."""
        elements: List[CodeElement] = []
        comments: List[Comment] = []

        try:
            tree = ast.parse(content)

            # Extract functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    elements.append(CodeElement(
                        type='function',
                        name=node.name,
                        line=node.lineno,
                        docstring=ast.get_docstring(node),
                        parameters=[arg.arg for arg in node.args.args],
                        return_type=self._get_return_annotation(node)
                    ))
                elif isinstance(node, ast.ClassDef):
                    elements.append(CodeElement(
                        type='class',
                        name=node.name,
                        line=node.lineno,
                        docstring=ast.get_docstring(node)
                    ))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        elements.append(CodeElement(
                            type='import',
                            name=alias.name,
                            line=node.lineno
                        ))
        except SyntaxError as e:
            logger.warning(f"Python syntax error: {e}")

        # Extract comments
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                comment_text = stripped[1:].strip()
                comment_type = self._classify_comment(comment_text)
                comments.append(Comment(
                    type=comment_type,
                    text=comment_text,
                    line=i
                ))

        return elements, comments

    def _parse_typescript(self, content: str) -> Tuple[List[CodeElement], List[Comment]]:
        """Parse TypeScript/JavaScript using regex (simple approach)."""
        elements: List[CodeElement] = []
        comments: List[Comment] = []

        # Extract functions
        func_pattern = r'(?:function|const|let|var)\s+(\w+)\s*=?\s*(?:async\s*)?\(([^)]*)\)'
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            params = [p.strip() for p in match.group(2).split(',') if p.strip()]
            line = content[:match.start()].count('\n') + 1

            elements.append(CodeElement(
                type='function',
                name=name,
                line=line,
                parameters=params
            ))

        # Extract classes
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            line = content[:match.start()].count('\n') + 1

            elements.append(CodeElement(
                type='class',
                name=name,
                line=line
            ))

        # Extract imports
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            name = match.group(1)
            line = content[:match.start()].count('\n') + 1

            elements.append(CodeElement(
                type='import',
                name=name,
                line=line
            ))

        # Extract comments
        # Single-line comments
        single_comment_pattern = r'//\s*(.+?)$'
        for match in re.finditer(single_comment_pattern, content, re.MULTILINE):
            comment_text = match.group(1).strip()
            comment_type = self._classify_comment(comment_text)
            line = content[:match.start()].count('\n') + 1

            comments.append(Comment(
                type=comment_type,
                text=comment_text,
                line=line
            ))

        # Block comments
        block_comment_pattern = r'/\*\s*(.+?)\s*\*/'
        for match in re.finditer(block_comment_pattern, content, re.DOTALL):
            comment_text = match.group(1).strip()
            comment_type = self._classify_comment(comment_text)
            line = content[:match.start()].count('\n') + 1

            comments.append(Comment(
                type=comment_type,
                text=comment_text,
                line=line
            ))

        return elements, comments

    def _parse_markdown(self, content: str) -> List[Comment]:
        """Extract TODO/NOTE items from markdown."""
        comments: List[Comment] = []

        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # Check for TODO items (checkbox format)
            if '- [ ]' in line or '- [x]' in line:
                text = line.strip().replace('- [ ]', '').replace('- [x]', '').strip()
                comments.append(Comment(
                    type='TODO',
                    text=text,
                    line=i
                ))

        return comments

    def _classify_comment(self, text: str) -> str:
        """Classify comment type based on content."""
        text_upper = text.upper()

        if text_upper.startswith('NOTE:'):
            return 'NOTE'
        elif text_upper.startswith('TODO:'):
            return 'TODO'
        elif text_upper.startswith('FIXME:'):
            return 'FIXME'
        else:
            return 'GENERAL'

    def _extract_motifs(
        self,
        elements: List[CodeElement],
        comments: List[Comment]
    ) -> List[str]:
        """Extract motifs from code elements and comments."""
        motifs = set()

        # Element types are motifs
        for elem in elements:
            motifs.add(elem.type)

        # Comment types are motifs
        for comment in comments:
            if comment.type != 'GENERAL':
                motifs.add(comment.type.lower())

        # Language constructs
        if any(e.type == 'class' for e in elements):
            motifs.add('object_oriented')

        if any(e.type == 'function' for e in elements):
            motifs.add('functional')

        return list(motifs)

    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Get return type annotation from function."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _load_gitignore(self, workspace_path: Path):
        """Load .gitignore patterns."""
        gitignore_path = workspace_path / '.gitignore'

        # Always add default patterns
        self.gitignore_patterns = self.DEFAULT_IGNORE_PATTERNS.copy()

        if gitignore_path.exists():
            try:
                with open(gitignore_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            self.gitignore_patterns.add(line)
            except Exception as e:
                logger.warning(f"Failed to load .gitignore: {e}")

    def _scan_directory(
        self,
        directory: Path,
        languages: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Recursively scan directory for supported files.

        Args:
            directory: Directory to scan
            languages: Filter by languages (None = all)

        Returns:
            List of file paths to process
        """
        files = []

        for path in directory.rglob('*'):
            # Skip directories
            if path.is_dir():
                continue

            # Check if should be ignored
            if self._should_ignore(path, directory):
                continue

            # Check extension
            if path.suffix not in self.SUPPORTED_EXTENSIONS:
                continue

            # Check language filter
            if languages:
                file_language = self.SUPPORTED_EXTENSIONS[path.suffix]
                if file_language not in languages:
                    continue

            files.append(path)

        return files

    def _should_ignore(self, path: Path, workspace_root: Path) -> bool:
        """Check if path matches any ignore pattern."""
        relative = path.relative_to(workspace_root)
        path_str = str(relative)

        for pattern in self.gitignore_patterns:
            # Simple pattern matching (not full gitignore spec)
            if pattern.endswith('/'):
                # Directory pattern
                if path_str.startswith(pattern.rstrip('/')):
                    return True
            elif '*' in pattern:
                # Glob pattern (simplified)
                pattern_regex = pattern.replace('.', r'\.').replace('*', '.*')
                if re.match(pattern_regex, path_str):
                    return True
            else:
                # Exact match or directory name match
                if pattern in path.parts or path_str == pattern:
                    return True

        return False

    # =========================================================================
    # HoloLoom Integration (November 2025)
    # =========================================================================

    async def index_to_hololoom(
        self,
        workspace_path: str | Path,
        hololoom_instance,
        incremental: bool = True,
        languages: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Index workspace into HoloLoom's knowledge graph.

        This method creates a complete code knowledge graph by:
        1. Parsing code structure (functions, classes, imports)
        2. Creating entities in knowledge graph (with file paths, line numbers)
        3. Creating edges (PART_OF, USES, IS_A, MENTIONS, DEFINES)
        4. Storing code elements as memories via HoloLoom.experience()
        5. Supporting incremental updates (only re-index changed files)

        Args:
            workspace_path: Path to workspace root
            hololoom_instance: HoloLoom instance (from HoloLoom())
            incremental: If True, only re-index changed files (default: True)
            languages: Languages to index (e.g., ["python", "typescript"])
            exclude_patterns: Additional glob patterns to exclude

        Returns:
            Statistics dictionary:
                - files: Number of files indexed
                - entities: Number of entities created
                - edges: Number of edges created
                - skipped: Number of files skipped (unchanged)
                - errors: Number of files with errors

        Example:
            >>> from HoloLoom import HoloLoom
            >>> async with HoloLoom() as loom:
            ...     spinner = WorkspaceSpinner()
            ...     stats = await spinner.index_to_hololoom(
            ...         workspace_path="/path/to/project",
            ...         hololoom_instance=loom,
            ...         incremental=True
            ...     )
            ...     print(f"Indexed {stats['files']} files")
            ...     print(f"Created {stats['entities']} entities")
            ...     print(f"Created {stats['edges']} edges")
        """
        workspace_path = Path(workspace_path)

        if not workspace_path.exists():
            raise ValueError(f"Workspace path does not exist: {workspace_path}")

        # Load .gitignore patterns
        self._load_gitignore(workspace_path)

        # Add user exclude patterns
        if exclude_patterns:
            self.gitignore_patterns.update(exclude_patterns)

        # Initialize statistics
        stats = {
            "files": 0,
            "entities": 0,
            "edges": 0,
            "skipped": 0,
            "errors": 0,
            "start_time": datetime.now().isoformat(),
        }

        # Load index metadata (for incremental updates)
        index_metadata = self._load_index_metadata(workspace_path) if incremental else {}

        # Get files to index
        if incremental:
            files_to_index = self._get_changed_files(
                workspace_path,
                index_metadata,
                languages
            )
        else:
            files_to_index = self._scan_directory(workspace_path, languages)

        logger.info(f"Indexing {len(files_to_index)} files to HoloLoom knowledge graph")

        # Process each file
        for file_path in files_to_index:
            try:
                # Parse file
                shard = await self._process_file(file_path, workspace_path)

                if not shard:
                    stats["skipped"] += 1
                    continue

                # Store in HoloLoom (creates memory)
                memory = await hololoom_instance.experience(
                    content=shard.text,
                    context={
                        "source": "workspace_indexer",
                        "file_path": str(file_path.relative_to(workspace_path)),
                        "language": shard.metadata.get("language"),
                        "entities": shard.entities,
                        "motifs": shard.motifs,
                        "element_count": shard.metadata.get("element_count", 0),
                        "indexed_at": datetime.now().isoformat()
                    }
                )

                # Create entities in knowledge graph
                entities_created = await self._create_entities(
                    hololoom_instance,
                    shard,
                    file_path,
                    workspace_path
                )
                stats["entities"] += entities_created

                # Create edges in knowledge graph
                edges_created = await self._create_edges(
                    hololoom_instance,
                    shard,
                    file_path,
                    workspace_path
                )
                stats["edges"] += edges_created

                stats["files"] += 1

                # Update index metadata
                if incremental:
                    file_hash = self._compute_file_hash(file_path)
                    relative_path = str(file_path.relative_to(workspace_path))
                    index_metadata[relative_path] = {
                        "hash": file_hash,
                        "indexed_at": datetime.now().isoformat(),
                        "memory_id": memory.id,
                        "entities": entities_created,
                        "edges": edges_created
                    }

                logger.info(
                    f"Indexed {file_path.relative_to(workspace_path)}: "
                    f"{entities_created} entities, {edges_created} edges"
                )

            except Exception as e:
                logger.error(f"Failed to index {file_path}: {e}")
                stats["errors"] += 1
                continue

        # Save index metadata
        if incremental:
            self._save_index_metadata(workspace_path, index_metadata)

        # Remove entities for deleted files
        if incremental:
            deleted_count = await self._cleanup_deleted_files(
                workspace_path,
                index_metadata,
                hololoom_instance
            )
            stats["deleted_entities"] = deleted_count

        stats["end_time"] = datetime.now().isoformat()

        logger.info(
            f"Workspace indexing complete: {stats['files']} files, "
            f"{stats['entities']} entities, {stats['edges']} edges"
        )

        return stats

    async def _create_entities(
        self,
        hololoom,
        shard: MemoryShard,
        file_path: Path,
        workspace_root: Path
    ) -> int:
        """
        Create entities in knowledge graph from code elements.

        Entity types:
        - class: Class definitions
        - function: Function/method definitions
        - import: Import statements
        - file: File nodes (parent of all entities)

        Each entity includes metadata:
        - file_path: Relative file path
        - line_number: Line number in source
        - docstring: Docstring (if available)
        - language: Programming language
        - indexed_at: Timestamp

        Args:
            hololoom: HoloLoom instance
            shard: MemoryShard with code structure
            file_path: Path to source file
            workspace_root: Workspace root path

        Returns:
            Number of entities created
        """
        # Get knowledge graph from HoloLoom
        kg = hololoom.graph

        relative_path = file_path.relative_to(workspace_root)
        entities_created = 0

        # Create file node
        file_entity_id = f"file::{relative_path}"

        if file_entity_id not in kg.nodes:
            kg.add_node(
                file_entity_id,
                type="file",
                name=str(relative_path),
                file_path=str(relative_path),
                language=shard.metadata.get("language"),
                element_count=shard.metadata.get("element_count", 0),
                indexed_at=datetime.now().isoformat()
            )
            entities_created += 1

        # Parse elements from shard metadata (if available)
        # Elements are stored during _process_file but not in shard by default
        # We'll need to re-parse the file to get detailed element info

        # For now, create entities from shard.entities list
        # This is a simplified version - in production, we'd want more detail

        for entity_name in shard.entities:
            entity_id = f"{relative_path}::{entity_name}"

            if entity_id not in kg.nodes:
                kg.add_node(
                    entity_id,
                    type="entity",  # Generic type - could be class, function, etc.
                    name=entity_name,
                    file_path=str(relative_path),
                    language=shard.metadata.get("language"),
                    indexed_at=datetime.now().isoformat()
                )
                entities_created += 1

                # Create DEFINES edge (file defines entity)
                kg.add_edge(
                    file_entity_id,
                    entity_id,
                    type="DEFINES",
                    weight=1.0
                )

        return entities_created

    async def _create_edges(
        self,
        hololoom,
        shard: MemoryShard,
        file_path: Path,
        workspace_root: Path
    ) -> int:
        """
        Create edges in knowledge graph from code relationships.

        Edge types created:
        - DEFINES: File defines entity (file → entity)
        - PART_OF: Entity is part of another (method → class)
        - USES: Entity uses another (function → function, import)
        - IS_A: Entity inherits from another (class → base_class)
        - MENTIONS: File mentions/imports module (file → module)

        Args:
            hololoom: HoloLoom instance
            shard: MemoryShard with code structure
            file_path: Path to source file
            workspace_root: Workspace root path

        Returns:
            Number of edges created
        """
        # Get knowledge graph from HoloLoom
        kg = hololoom.graph

        relative_path = file_path.relative_to(workspace_root)
        edges_created = 0

        # Create MENTIONS edges for imports
        # This is a simplified version - in production, we'd parse AST for detailed relationships

        # Example: If shard has motifs like "import", create edges
        if "import" in shard.motifs:
            # In a full implementation, we'd parse imports from AST
            # For now, this is a placeholder
            pass

        # Note: Full edge creation requires storing CodeElement objects
        # in the shard metadata, which we'll add in the enhanced version

        return edges_created

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file contents.

        Used for incremental update detection - if hash hasn't changed,
        file doesn't need re-indexing.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of SHA256 hash
        """
        sha256 = hashlib.sha256()

        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception:
            # If we can't read file, return empty hash
            return ""

    def _get_changed_files(
        self,
        workspace_path: Path,
        index_metadata: Dict[str, Any],
        languages: Optional[List[str]] = None
    ) -> List[Path]:
        """
        Get files that have changed since last index.

        Compares file hashes from index_metadata with current hashes.
        Also includes new files (not in index_metadata).

        Args:
            workspace_path: Workspace root path
            index_metadata: Index metadata from previous run
            languages: Languages to filter (None = all)

        Returns:
            List of file paths that need re-indexing
        """
        changed_files = []
        all_files = self._scan_directory(workspace_path, languages)

        for file_path in all_files:
            relative_path = str(file_path.relative_to(workspace_path))

            # New file (not in index)
            if relative_path not in index_metadata:
                changed_files.append(file_path)
                continue

            # Check if hash changed
            current_hash = self._compute_file_hash(file_path)
            stored_hash = index_metadata[relative_path].get("hash", "")

            if current_hash != stored_hash:
                changed_files.append(file_path)

        return changed_files

    def _load_index_metadata(self, workspace_path: Path) -> Dict[str, Any]:
        """
        Load index metadata from .hololoom_index.json.

        Metadata structure:
        {
            "path/to/file.py": {
                "hash": "abc123...",
                "indexed_at": "2025-11-17T10:30:00",
                "memory_id": "mem_xyz",
                "entities": 15,
                "edges": 32
            },
            ...
        }

        Args:
            workspace_path: Workspace root path

        Returns:
            Index metadata dictionary (empty if file doesn't exist)
        """
        index_file = workspace_path / ".hololoom_index.json"

        if not index_file.exists():
            return {}

        try:
            with open(index_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load index metadata: {e}")
            return {}

    def _save_index_metadata(
        self,
        workspace_path: Path,
        index_metadata: Dict[str, Any]
    ) -> None:
        """
        Save index metadata to .hololoom_index.json.

        Args:
            workspace_path: Workspace root path
            index_metadata: Index metadata to save
        """
        index_file = workspace_path / ".hololoom_index.json"

        try:
            with open(index_file, 'w') as f:
                json.dump(index_metadata, f, indent=2)
            logger.info(f"Saved index metadata to {index_file}")
        except Exception as e:
            logger.error(f"Failed to save index metadata: {e}")

    async def _cleanup_deleted_files(
        self,
        workspace_path: Path,
        index_metadata: Dict[str, Any],
        hololoom
    ) -> int:
        """
        Remove entities for files that were deleted from workspace.

        Compares index_metadata with current filesystem state.
        Removes entities and edges for files that no longer exist.

        Args:
            workspace_path: Workspace root path
            index_metadata: Index metadata
            hololoom: HoloLoom instance

        Returns:
            Number of entities removed
        """
        kg = hololoom.graph
        entities_removed = 0

        # Find files in index but not on disk
        for relative_path in list(index_metadata.keys()):
            file_path = workspace_path / relative_path

            if not file_path.exists():
                # File was deleted - remove its entities
                file_entity_id = f"file::{relative_path}"

                if file_entity_id in kg.nodes:
                    # Remove all entities defined by this file
                    # Get all entities this file defines
                    neighbors = list(kg.neighbors(file_entity_id))

                    for neighbor in neighbors:
                        if kg.has_edge(file_entity_id, neighbor):
                            edge_data = kg.get_edge_data(file_entity_id, neighbor)
                            # Check if it's a DEFINES edge
                            for key, attrs in edge_data.items():
                                if attrs.get('type') == 'DEFINES':
                                    # Remove the defined entity
                                    if neighbor in kg.nodes:
                                        kg.remove_node(neighbor)
                                        entities_removed += 1

                    # Remove file entity
                    kg.remove_node(file_entity_id)
                    entities_removed += 1

                # Remove from index metadata
                del index_metadata[relative_path]

                logger.info(f"Removed entities for deleted file: {relative_path}")

        return entities_removed

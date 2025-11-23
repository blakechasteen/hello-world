"""
Workspace Scanner - Auto-index entire workspace into HoloLoom knowledge graph

Scans workspace directories and extracts:
- Code structure (functions, classes, imports)
- Comments (NOTE, TODO, FIXME)
- File relationships (imports, dependencies)

Respects .gitignore patterns for intelligent filtering.

Usage:
    from HoloLoom.spinningWheel import WorkspaceSpinner

    spinner = WorkspaceSpinner()
    shards = await spinner.spin_workspace("/path/to/workspace")

    # Shards contain code structure + comments as MemoryShards
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import logging

from HoloLoom.protocols.types import MemoryShard

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

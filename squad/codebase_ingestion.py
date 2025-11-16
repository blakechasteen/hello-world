#!/usr/bin/env python3
"""
Codebase Ingestion Module
==========================
Scan entire codebases, parse structure, and ingest into HoloLoom memory.

Capabilities:
- Recursive directory scanning
- Multi-language parsing (Python, TypeScript, JavaScript, etc.)
- Code structure extraction (classes, functions, imports)
- MemoryShard creation for HoloLoom
- Metadata extraction (file paths, line numbers, dependencies)

Author: Claude Code
Date: November 16, 2025
"""

import os
import ast
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LanguageType(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    UNKNOWN = "unknown"


@dataclass
class CodeEntity:
    """Extracted code entity (class, function, etc.)"""
    name: str
    type: str  # class, function, method, variable
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    dependencies: List[str] = None
    language: str = "unknown"

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class CodebaseMetadata:
    """Metadata about ingested codebase"""
    root_path: str
    total_files: int
    total_entities: int
    languages: Dict[str, int]
    file_tree: List[str]
    timestamp: str


class CodebaseIngestionEngine:
    """
    Engine for ingesting entire codebases into HoloLoom memory.

    Scans directories, parses code, extracts structure, creates memory shards.
    """

    # File extensions mapped to languages
    LANGUAGE_EXTENSIONS = {
        ".py": LanguageType.PYTHON,
        ".ts": LanguageType.TYPESCRIPT,
        ".tsx": LanguageType.TYPESCRIPT,
        ".js": LanguageType.JAVASCRIPT,
        ".jsx": LanguageType.JAVASCRIPT,
        ".java": LanguageType.JAVA,
        ".cs": LanguageType.CSHARP,
        ".go": LanguageType.GO,
        ".rs": LanguageType.RUST,
        ".cpp": LanguageType.CPP,
        ".cc": LanguageType.CPP,
        ".h": LanguageType.CPP,
        ".hpp": LanguageType.CPP,
    }

    # Directories to ignore
    IGNORE_DIRS = {
        ".git", ".venv", "venv", "env", "__pycache__",
        "node_modules", "dist", "build", "target",
        ".next", ".nuxt", "out", "coverage"
    }

    # File patterns to ignore
    IGNORE_FILES = {
        ".pyc", ".pyo", ".so", ".dll", ".dylib",
        ".min.js", ".bundle.js", ".map"
    }

    def __init__(self, max_file_size: int = 1_000_000):
        """
        Initialize codebase ingestion engine.

        Args:
            max_file_size: Maximum file size to parse (bytes)
        """
        self.max_file_size = max_file_size
        self.stats = {
            "files_scanned": 0,
            "files_parsed": 0,
            "entities_extracted": 0,
            "errors": 0
        }

    def ingest_codebase(
        self,
        root_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> tuple[List[CodeEntity], CodebaseMetadata]:
        """
        Ingest entire codebase from directory.

        Args:
            root_path: Root directory to scan
            include_patterns: File patterns to include (e.g., ["*.py", "*.ts"])
            exclude_patterns: File patterns to exclude

        Returns:
            Tuple of (entities, metadata)
        """
        logger.info(f"Ingesting codebase from: {root_path}")

        if not os.path.exists(root_path):
            raise ValueError(f"Path does not exist: {root_path}")

        # Scan directory
        files = self._scan_directory(root_path, include_patterns, exclude_patterns)
        logger.info(f"Found {len(files)} files to parse")

        # Parse files
        all_entities = []
        language_counts = {}

        for file_path in files:
            try:
                entities = self._parse_file(file_path, root_path)
                all_entities.extend(entities)

                # Count languages
                lang = self._detect_language(file_path)
                language_counts[lang.value] = language_counts.get(lang.value, 0) + 1

                self.stats["files_parsed"] += 1
                self.stats["entities_extracted"] += len(entities)

            except Exception as e:
                logger.warning(f"Failed to parse {file_path}: {e}")
                self.stats["errors"] += 1

        # Build metadata
        metadata = CodebaseMetadata(
            root_path=root_path,
            total_files=len(files),
            total_entities=len(all_entities),
            languages=language_counts,
            file_tree=[str(Path(f).relative_to(root_path)) for f in files[:100]],  # First 100
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"Ingested {len(all_entities)} entities from {len(files)} files")
        return all_entities, metadata

    def _scan_directory(
        self,
        root_path: str,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]]
    ) -> List[str]:
        """Recursively scan directory for code files"""
        files = []

        for root, dirs, filenames in os.walk(root_path):
            # Remove ignored directories
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS]

            for filename in filenames:
                # Check file extension
                ext = Path(filename).suffix
                if ext not in self.LANGUAGE_EXTENSIONS:
                    continue

                # Check ignore patterns
                if any(pattern in filename for pattern in self.IGNORE_FILES):
                    continue

                file_path = os.path.join(root, filename)

                # Check file size
                try:
                    if os.path.getsize(file_path) > self.max_file_size:
                        logger.warning(f"Skipping large file: {file_path}")
                        continue
                except OSError:
                    continue

                files.append(file_path)
                self.stats["files_scanned"] += 1

        return files

    def _detect_language(self, file_path: str) -> LanguageType:
        """Detect programming language from file extension"""
        ext = Path(file_path).suffix
        return self.LANGUAGE_EXTENSIONS.get(ext, LanguageType.UNKNOWN)

    def _parse_file(self, file_path: str, root_path: str) -> List[CodeEntity]:
        """Parse file and extract entities"""
        lang = self._detect_language(file_path)

        if lang == LanguageType.PYTHON:
            return self._parse_python(file_path, root_path)
        elif lang in [LanguageType.TYPESCRIPT, LanguageType.JAVASCRIPT]:
            return self._parse_typescript_javascript(file_path, root_path)
        else:
            # Generic parsing for other languages
            return self._parse_generic(file_path, root_path, lang)

    def _parse_python(self, file_path: str, root_path: str) -> List[CodeEntity]:
        """Parse Python file using AST"""
        entities = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            relative_path = str(Path(file_path).relative_to(root_path))

            # Extract imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entities.append(CodeEntity(
                        name=node.name,
                        type="class",
                        file_path=relative_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                        dependencies=imports,
                        language="python"
                    ))

                elif isinstance(node, ast.FunctionDef):
                    # Get function signature
                    args = [arg.arg for arg in node.args.args]
                    signature = f"{node.name}({', '.join(args)})"

                    entities.append(CodeEntity(
                        name=node.name,
                        type="function",
                        file_path=relative_path,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node),
                        signature=signature,
                        dependencies=imports,
                        language="python"
                    ))

        except Exception as e:
            logger.warning(f"Python parsing error in {file_path}: {e}")

        return entities

    def _parse_typescript_javascript(self, file_path: str, root_path: str) -> List[CodeEntity]:
        """Parse TypeScript/JavaScript file (basic regex-based)"""
        entities = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            relative_path = str(Path(file_path).relative_to(root_path))
            lines = content.split('\n')

            # Extract imports
            imports = []
            for line in lines:
                if line.strip().startswith('import '):
                    # Basic import extraction
                    if 'from' in line:
                        parts = line.split('from')
                        if len(parts) == 2:
                            module = parts[1].strip().strip(';').strip('"').strip("'")
                            imports.append(module)

            # Extract classes (basic regex)
            import re

            # Classes
            class_pattern = r'class\s+(\w+)'
            for match in re.finditer(class_pattern, content):
                # Find line number
                line_num = content[:match.start()].count('\n') + 1

                entities.append(CodeEntity(
                    name=match.group(1),
                    type="class",
                    file_path=relative_path,
                    line_start=line_num,
                    line_end=line_num,  # Approximate
                    dependencies=imports,
                    language="typescript" if file_path.endswith('.ts') else "javascript"
                ))

            # Functions
            function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)'
            for match in re.finditer(function_pattern, content):
                line_num = content[:match.start()].count('\n') + 1

                entities.append(CodeEntity(
                    name=match.group(1),
                    type="function",
                    file_path=relative_path,
                    line_start=line_num,
                    line_end=line_num,
                    dependencies=imports,
                    language="typescript" if file_path.endswith('.ts') else "javascript"
                ))

            # Arrow functions
            arrow_pattern = r'(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\('
            for match in re.finditer(arrow_pattern, content):
                line_num = content[:match.start()].count('\n') + 1

                entities.append(CodeEntity(
                    name=match.group(1),
                    type="function",
                    file_path=relative_path,
                    line_start=line_num,
                    line_end=line_num,
                    dependencies=imports,
                    language="typescript" if file_path.endswith('.ts') else "javascript"
                ))

        except Exception as e:
            logger.warning(f"TS/JS parsing error in {file_path}: {e}")

        return entities

    def _parse_generic(self, file_path: str, root_path: str, lang: LanguageType) -> List[CodeEntity]:
        """Generic parsing for unsupported languages (basic structure extraction)"""
        entities = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            relative_path = str(Path(file_path).relative_to(root_path))

            # Just create a single entity for the whole file
            entities.append(CodeEntity(
                name=Path(file_path).stem,
                type="file",
                file_path=relative_path,
                line_start=1,
                line_end=len(content.split('\n')),
                language=lang.value
            ))

        except Exception as e:
            logger.warning(f"Generic parsing error in {file_path}: {e}")

        return entities

    def entities_to_memory_shards(self, entities: List[CodeEntity]) -> List[Dict[str, Any]]:
        """
        Convert code entities to HoloLoom MemoryShard format.

        Returns list of shard dictionaries ready for MemoryShard creation.
        """
        shards = []

        for entity in entities:
            # Create text representation
            text_parts = [f"{entity.type} {entity.name}"]

            if entity.signature:
                text_parts.append(f"Signature: {entity.signature}")

            if entity.docstring:
                text_parts.append(f"Documentation: {entity.docstring}")

            text_parts.append(f"Location: {entity.file_path}:{entity.line_start}")

            if entity.dependencies:
                text_parts.append(f"Dependencies: {', '.join(entity.dependencies[:5])}")

            text = "\n".join(text_parts)

            # Create shard
            shard = {
                "id": f"code-{entity.file_path.replace('/', '-')}-{entity.name}-{entity.line_start}",
                "text": text,
                "entities": [entity.name] + entity.dependencies[:3],
                "motifs": [entity.type, entity.language, "code"],
                "metadata": {
                    "source": "codebase_ingestion",
                    "file_path": entity.file_path,
                    "line_start": entity.line_start,
                    "line_end": entity.line_end,
                    "entity_type": entity.type,
                    "language": entity.language,
                    "timestamp": datetime.now().isoformat()
                }
            }

            shards.append(shard)

        return shards

    def get_stats(self) -> Dict[str, int]:
        """Get ingestion statistics"""
        return self.stats.copy()

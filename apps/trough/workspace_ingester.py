#!/usr/bin/env python3
"""
Squad Workspace Ingester
=========================
Ingests VS Code workspace code into HoloLoom memory for context-aware responses.

This module walks a workspace directory, extracts code structure, and creates
MemoryShards that Squad can use to understand your codebase.

Usage:
    from squad.workspace_ingester import WorkspaceIngester

    ingester = WorkspaceIngester(workspace_path="/path/to/project")
    shards = await ingester.ingest()
    # Feed shards to HoloLoom orchestrator
"""

import os
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field

from hololoom.documentation.types import MemoryShard
from hololoom.spinning_wheel.code import CodeSpinner, CodeSpinnerConfig


@dataclass
class WorkspaceConfig:
    """Configuration for workspace ingestion"""
    # Directories to scan
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py", "**/*.ts", "**/*.js", "**/*.tsx", "**/*.jsx"])

    # Directories to ignore
    exclude_patterns: List[str] = field(default_factory=lambda: [
        "**/node_modules/**",
        "**/.git/**",
        "**/__pycache__/**",
        "**/dist/**",
        "**/build/**",
        "**/.venv/**",
        "**/venv/**"
    ])

    # Language-specific settings
    extract_docstrings: bool = True
    extract_types: bool = True
    extract_imports: bool = True
    extract_comments: bool = True

    # Chunking
    max_file_size_kb: int = 500  # Skip files larger than this
    chunk_by_function: bool = True  # Create shards per function/class

    # Episode tagging
    episode_prefix: str = "squad:workspace"


class WorkspaceIngester:
    """
    Ingests VS Code workspace into HoloLoom memory.

    Walks workspace directory, extracts code structure, and creates
    MemoryShards for each function/class/module.
    """

    def __init__(self, workspace_path: str, config: Optional[WorkspaceConfig] = None):
        self.workspace_path = Path(workspace_path)
        self.config = config or WorkspaceConfig()

        # Initialize code spinner
        spinner_config = CodeSpinnerConfig(
            chunk_by='function' if self.config.chunk_by_function else None,
            extract_imports=self.config.extract_imports,
            extract_entities=True,
            include_line_numbers=True
        )
        self.spinner = CodeSpinner(spinner_config)

        self.stats = {
            "files_scanned": 0,
            "files_ingested": 0,
            "files_skipped": 0,
            "shards_created": 0
        }

    async def ingest(self) -> List[MemoryShard]:
        """
        Ingest entire workspace into memory shards.

        Returns:
            List of MemoryShards representing the workspace
        """
        all_shards = []

        # Walk workspace
        for pattern in self.config.include_patterns:
            for file_path in self.workspace_path.glob(pattern):
                # Skip excluded paths
                if self._should_exclude(file_path):
                    continue

                self.stats["files_scanned"] += 1

                # Check file size
                if file_path.stat().st_size > self.config.max_file_size_kb * 1024:
                    self.stats["files_skipped"] += 1
                    continue

                # Ingest file
                try:
                    shards = await self._ingest_file(file_path)
                    all_shards.extend(shards)
                    self.stats["files_ingested"] += 1
                    self.stats["shards_created"] += len(shards)
                except Exception as e:
                    print(f"Error ingesting {file_path}: {e}")
                    self.stats["files_skipped"] += 1

        return all_shards

    async def _ingest_file(self, file_path: Path) -> List[MemoryShard]:
        """Ingest a single code file"""
        # Read file
        content = file_path.read_text(encoding='utf-8', errors='ignore')

        # Detect language
        language = self._detect_language(file_path)

        # Parse based on language
        if language == 'python':
            return await self._ingest_python_file(file_path, content)
        elif language in ['typescript', 'javascript', 'tsx', 'jsx']:
            return await self._ingest_typescript_file(file_path, content)
        else:
            # Generic ingestion
            return await self._ingest_generic_file(file_path, content, language)

    async def _ingest_python_file(self, file_path: Path, content: str) -> List[MemoryShard]:
        """Ingest Python file with AST parsing"""
        shards = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fall back to generic ingestion
            return await self._ingest_generic_file(file_path, content, 'python')

        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                shard = self._create_function_shard(file_path, node, content, 'python')
                shards.append(shard)
            elif isinstance(node, ast.ClassDef):
                shard = self._create_class_shard(file_path, node, content, 'python')
                shards.append(shard)

        # If no functions/classes, create file-level shard
        if not shards:
            shards.append(self._create_file_shard(file_path, content, 'python'))

        return shards

    async def _ingest_typescript_file(self, file_path: Path, content: str) -> List[MemoryShard]:
        """Ingest TypeScript/JavaScript file with regex parsing"""
        shards = []

        # Extract functions (simple regex - could use TS parser for better results)
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*(?::\s*[^{]+)?\s*{'
        for match in re.finditer(function_pattern, content):
            function_name = match.group(1)
            start_pos = match.start()

            # Find function body (simple brace matching)
            end_pos = self._find_closing_brace(content, match.end() - 1)
            if end_pos:
                function_code = content[start_pos:end_pos + 1]

                shard = MemoryShard(
                    id=f"{file_path.stem}_{function_name}_{hash(function_code) % 10000}",
                    text=function_code,
                    episode=f"{self.config.episode_prefix}:{file_path.parent}",
                    entities=[function_name, file_path.stem],
                    motifs=["function", "typescript", "code"],
                    metadata={
                        "type": "function",
                        "name": function_name,
                        "file": str(file_path.relative_to(self.workspace_path)),
                        "language": "typescript",
                        "ingested_at": datetime.now().isoformat()
                    }
                )
                shards.append(shard)

        # If no functions found, create file-level shard
        if not shards:
            shards.append(self._create_file_shard(file_path, content, 'typescript'))

        return shards

    async def _ingest_generic_file(self, file_path: Path, content: str, language: str) -> List[MemoryShard]:
        """Generic ingestion for any file type"""
        return [self._create_file_shard(file_path, content, language)]

    def _create_function_shard(self, file_path: Path, node: ast.FunctionDef, content: str, language: str) -> MemoryShard:
        """Create shard for a Python function"""
        # Extract function source
        lines = content.split('\n')
        function_lines = lines[node.lineno - 1:node.end_lineno]
        function_code = '\n'.join(function_lines)

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract decorators
        decorators = [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]

        # Extract parameter names
        params = [arg.arg for arg in node.args.args]

        return MemoryShard(
            id=f"{file_path.stem}_{node.name}_{node.lineno}",
            text=function_code,
            episode=f"{self.config.episode_prefix}:{file_path.parent}",
            entities=[node.name, file_path.stem] + params,
            motifs=["function", language, "code"] + decorators,
            metadata={
                "type": "function",
                "name": node.name,
                "file": str(file_path.relative_to(self.workspace_path)),
                "line": node.lineno,
                "language": language,
                "docstring": docstring,
                "parameters": params,
                "decorators": decorators,
                "ingested_at": datetime.now().isoformat()
            }
        )

    def _create_class_shard(self, file_path: Path, node: ast.ClassDef, content: str, language: str) -> MemoryShard:
        """Create shard for a Python class"""
        # Extract class source
        lines = content.split('\n')
        class_lines = lines[node.lineno - 1:node.end_lineno]
        class_code = '\n'.join(class_lines)

        # Extract docstring
        docstring = ast.get_docstring(node) or ""

        # Extract methods
        methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]

        # Extract base classes
        bases = [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases]

        return MemoryShard(
            id=f"{file_path.stem}_{node.name}_{node.lineno}",
            text=class_code,
            episode=f"{self.config.episode_prefix}:{file_path.parent}",
            entities=[node.name, file_path.stem] + methods,
            motifs=["class", language, "code"] + bases,
            metadata={
                "type": "class",
                "name": node.name,
                "file": str(file_path.relative_to(self.workspace_path)),
                "line": node.lineno,
                "language": language,
                "docstring": docstring,
                "methods": methods,
                "bases": bases,
                "ingested_at": datetime.now().isoformat()
            }
        )

    def _create_file_shard(self, file_path: Path, content: str, language: str) -> MemoryShard:
        """Create shard for entire file"""
        # Extract imports (simple regex)
        imports = re.findall(r'(?:from|import)\s+([a-zA-Z0-9_.]+)', content)

        # Extract top-level entities
        entities = re.findall(r'(?:class|function|def|const|let|var)\s+([a-zA-Z0-9_]+)', content)

        return MemoryShard(
            id=f"{file_path.stem}_{hash(content) % 10000}",
            text=content,
            episode=f"{self.config.episode_prefix}:{file_path.parent}",
            entities=[file_path.stem] + entities[:10],  # Limit entities
            motifs=["file", language, "code"],
            metadata={
                "type": "file",
                "file": str(file_path.relative_to(self.workspace_path)),
                "language": language,
                "imports": imports[:20],  # Limit imports
                "size_bytes": len(content),
                "ingested_at": datetime.now().isoformat()
            }
        )

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext = file_path.suffix.lower()

        language_map = {
            '.py': 'python',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.js': 'javascript',
            '.jsx': 'jsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp'
        }

        return language_map.get(ext, 'unknown')

    def _should_exclude(self, file_path: Path) -> bool:
        """Check if path should be excluded"""
        path_str = str(file_path)

        for pattern in self.config.exclude_patterns:
            # Simple pattern matching (could use fnmatch for better results)
            if re.search(pattern.replace('**/', '').replace('/**', ''), path_str):
                return True

        return False

    def _find_closing_brace(self, content: str, start_pos: int) -> Optional[int]:
        """Find matching closing brace (simple implementation)"""
        depth = 1
        pos = start_pos + 1

        while pos < len(content) and depth > 0:
            if content[pos] == '{':
                depth += 1
            elif content[pos] == '}':
                depth -= 1
            pos += 1

        return pos - 1 if depth == 0 else None

    def get_stats(self) -> Dict[str, int]:
        """Get ingestion statistics"""
        return self.stats.copy()


# ============================================================================
# Convenience Functions
# ============================================================================

async def ingest_workspace(workspace_path: str, config: Optional[WorkspaceConfig] = None) -> List[MemoryShard]:
    """
    Quick function to ingest a workspace.

    Usage:
        shards = await ingest_workspace("/path/to/project")
    """
    ingester = WorkspaceIngester(workspace_path, config)
    shards = await ingester.ingest()

    print(f"\nWorkspace Ingestion Complete:")
    print(f"  Files scanned: {ingester.stats['files_scanned']}")
    print(f"  Files ingested: {ingester.stats['files_ingested']}")
    print(f"  Files skipped: {ingester.stats['files_skipped']}")
    print(f"  Shards created: {ingester.stats['shards_created']}")

    return shards

# -*- coding: utf-8 -*-
"""
Code Spinner
=============
Input adapter for ingesting code files, git diffs, and repository structure as MemoryShards.

Converts code-related data into structured memory shards that can be processed
by the HoloLoom orchestrator.

Design Philosophy:
- Thin adapter that normalizes code data → MemoryShards
- Handles multiple code formats (files, diffs, commits, patches)
- Language-agnostic parsing with optional language-specific enrichment
- Repository structure mapping for contextual understanding
- Optional enrichment for dependency/import graph extraction

Usage:
    from HoloLoom.spinning_wheel.code import CodeSpinner, CodeSpinnerConfig

    config = CodeSpinnerConfig(
        chunk_by='function',  # or 'class', 'file', None
        extract_imports=True,
        enable_enrichment=True
    )

    spinner = CodeSpinner(config)

    # Ingest a code file
    shards = await spinner.spin({
        'type': 'file',
        'path': 'src/main.py',
        'content': "def hello(): pass",
        'language': 'python'
    })

    # Ingest a git diff
    shards = await spinner.spin({
        'type': 'diff',
        'diff': git_diff_output,
        'commit_sha': 'abc123',
        'author': 'user@example.com'
    })
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseSpinner, SpinnerConfig

# Import HoloLoom types
try:
    from HoloLoom.documentation.types import MemoryShard
except ImportError:
    # Fallback if types not available
    from dataclasses import dataclass as dc_dataclass

    @dc_dataclass
    class MemoryShard:
        id: str
        text: str
        episode: Optional[str] = None
        entities: List[str] = field(default_factory=list)
        motifs: List[str] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeSpinnerConfig(SpinnerConfig):
    """
    Configuration for Code spinner.

    Attributes:
        chunk_by: How to split code ('function', 'class', 'file', None)
                 If None, creates one shard per file
        extract_imports: Extract import/dependency statements
        extract_entities: Extract code entities (classes, functions, variables)
        include_line_numbers: Include line numbers in metadata
        language_filter: Only process specific languages (None = all)
        max_chunk_lines: Maximum lines per chunk
    """
    chunk_by: Optional[str] = None  # None = single shard per file
    extract_imports: bool = True
    extract_entities: bool = True
    include_line_numbers: bool = True
    language_filter: Optional[Set[str]] = None
    max_chunk_lines: int = 200


class CodeSpinner(BaseSpinner):
    """
    Spinner for code files, diffs, and repository structures.

    Converts code data into MemoryShards suitable for HoloLoom processing.
    """

    # Language detection patterns
    LANGUAGE_PATTERNS = {
        'python': ['.py', '.pyw'],
        'javascript': ['.js', '.jsx', '.mjs'],
        'typescript': ['.ts', '.tsx'],
        'java': ['.java'],
        'cpp': ['.cpp', '.hpp', '.cc', '.h'],
        'c': ['.c', '.h'],
        'rust': ['.rs'],
        'go': ['.go'],
        'ruby': ['.rb'],
        'php': ['.php'],
        'swift': ['.swift'],
        'kotlin': ['.kt', '.kts'],
        'scala': ['.scala'],
        'r': ['.r', '.R'],
        'sql': ['.sql'],
        'shell': ['.sh', '.bash', '.zsh'],
    }

    def __init__(self, config: CodeSpinnerConfig = None):
        if config is None:
            config = CodeSpinnerConfig()
        super().__init__(config)
        self.config: CodeSpinnerConfig = config

    async def spin(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """
        Convert code data → MemoryShards.

        Args:
            raw_data: Dict with keys:
                - 'type': Type of code data ('file', 'diff', 'repo')
                - For 'file':
                    - 'path': File path (required)
                    - 'content': File content (required)
                    - 'language': Programming language (optional, auto-detect)
                    - 'metadata': Additional metadata
                - For 'diff':
                    - 'diff': Git diff output (required)
                    - 'commit_sha': Commit SHA (optional)
                    - 'author': Commit author (optional)
                    - 'message': Commit message (optional)
                - For 'repo':
                    - 'root_path': Repository root (required)
                    - 'files': List of file paths (required)
                    - 'structure': Directory tree (optional)

        Returns:
            List of MemoryShard objects
        """
        data_type = raw_data.get('type', 'file')

        if data_type == 'file':
            return await self._spin_file(raw_data)
        elif data_type == 'diff':
            return await self._spin_diff(raw_data)
        elif data_type == 'repo':
            return await self._spin_repo(raw_data)
        else:
            raise ValueError(f"Unknown code data type: {data_type}")

    async def _spin_file(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """Process a single code file into shards."""
        path = raw_data.get('path')
        content = raw_data.get('content')

        if not path or content is None:
            raise ValueError("'path' and 'content' are required for file type")

        # Detect language
        language = raw_data.get('language') or self._detect_language(path)

        # Check language filter
        if self.config.language_filter and language not in self.config.language_filter:
            return []

        # Generate episode identifier
        episode = raw_data.get('episode', f"code_{Path(path).stem}")

        # Create shards based on chunking strategy
        if self.config.chunk_by:
            shards = self._create_chunked_shards(
                content, path, language, episode, raw_data
            )
        else:
            shards = self._create_single_shard(
                content, path, language, episode, raw_data
            )

        # Optional enrichment
        if self.config.enable_enrichment:
            shards = await self._enrich_shards(shards)

        return shards

    async def _spin_diff(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """Process a git diff into shards."""
        diff = raw_data.get('diff')
        if not diff:
            raise ValueError("'diff' is required for diff type")

        # Parse diff
        diff_data = self._parse_diff(diff)

        # Generate episode identifier
        commit_sha = raw_data.get('commit_sha', 'uncommitted')
        episode = f"commit_{commit_sha[:8]}"

        shards = []

        # Create a summary shard
        summary_text = self._create_diff_summary(diff_data, raw_data)
        summary_shard = MemoryShard(
            id=f"{episode}_summary",
            text=summary_text,
            episode=episode,
            entities=self._extract_diff_entities(diff_data),
            motifs=['CODE_CHANGE', 'GIT_COMMIT'],
            metadata={
                'type': 'diff_summary',
                'commit_sha': commit_sha,
                'author': raw_data.get('author'),
                'message': raw_data.get('message'),
                'files_changed': len(diff_data['files']),
                'insertions': diff_data['stats']['insertions'],
                'deletions': diff_data['stats']['deletions'],
            }
        )
        shards.append(summary_shard)

        # Create shards for each changed file
        for idx, file_change in enumerate(diff_data['files']):
            file_shard = MemoryShard(
                id=f"{episode}_file_{idx:03d}",
                text=file_change['diff_text'],
                episode=episode,
                entities=file_change.get('entities', []),
                motifs=['FILE_CHANGE'],
                metadata={
                    'type': 'file_diff',
                    'file_path': file_change['path'],
                    'change_type': file_change['change_type'],
                    'language': self._detect_language(file_change['path']),
                    'lines_added': file_change['lines_added'],
                    'lines_removed': file_change['lines_removed'],
                }
            )
            shards.append(file_shard)

        return shards

    async def _spin_repo(self, raw_data: Dict[str, Any]) -> List[MemoryShard]:
        """Process repository structure into shards."""
        root_path = raw_data.get('root_path')
        files = raw_data.get('files')

        if not root_path or not files:
            raise ValueError("'root_path' and 'files' are required for repo type")

        episode = f"repo_{Path(root_path).name}"

        # Create structure shard
        structure_text = self._create_structure_text(files, root_path)
        structure_shard = MemoryShard(
            id=f"{episode}_structure",
            text=structure_text,
            episode=episode,
            entities=[],
            motifs=['REPOSITORY_STRUCTURE'],
            metadata={
                'type': 'repo_structure',
                'root_path': str(root_path),
                'file_count': len(files),
                'languages': list(self._count_languages(files).keys()),
            }
        )

        return [structure_shard]

    def _create_single_shard(
        self,
        content: str,
        path: str,
        language: str,
        episode: str,
        raw_data: Dict[str, Any]
    ) -> List[MemoryShard]:
        """Create a single shard for an entire file."""
        # Extract entities
        entities = []
        if self.config.extract_entities:
            entities = self._extract_code_entities(content, language)

        # Extract imports
        imports = []
        if self.config.extract_imports:
            imports = self._extract_imports(content, language)

        # Build metadata
        metadata = {
            'type': 'code_file',
            'path': path,
            'language': language,
            'line_count': len(content.split('\n')),
            'char_count': len(content),
            'hash': hashlib.md5(content.encode()).hexdigest(),
        }

        if imports:
            metadata['imports'] = imports

        # Merge additional metadata
        if 'metadata' in raw_data:
            metadata.update(raw_data['metadata'])

        # Create text with context
        text = f"# File: {path}\n# Language: {language}\n\n{content}"

        shard = MemoryShard(
            id=f"{episode}_full",
            text=text,
            episode=episode,
            entities=entities,
            motifs=['CODE_FILE', f'LANG_{language.upper()}'],
            metadata=metadata
        )

        return [shard]

    def _create_chunked_shards(
        self,
        content: str,
        path: str,
        language: str,
        episode: str,
        raw_data: Dict[str, Any]
    ) -> List[MemoryShard]:
        """Split code into chunks based on configuration."""
        chunks = []

        if self.config.chunk_by == 'function':
            chunks = self._chunk_by_function(content, language)
        elif self.config.chunk_by == 'class':
            chunks = self._chunk_by_class(content, language)
        elif self.config.chunk_by == 'file':
            # Simple line-based chunking
            chunks = self._chunk_by_lines(content, self.config.max_chunk_lines)
        else:
            raise ValueError(f"Unknown chunk_by mode: {self.config.chunk_by}")

        shards = []
        for idx, (chunk_text, chunk_meta) in enumerate(chunks):
            entities = []
            if self.config.extract_entities:
                entities = self._extract_code_entities(chunk_text, language)

            metadata = {
                'type': 'code_chunk',
                'path': path,
                'language': language,
                'chunk_index': idx,
                'chunk_by': self.config.chunk_by,
                **chunk_meta
            }

            if 'metadata' in raw_data:
                metadata.update(raw_data['metadata'])

            shard = MemoryShard(
                id=f"{episode}_chunk_{idx:03d}",
                text=chunk_text,
                episode=episode,
                entities=entities,
                motifs=['CODE_CHUNK', f'LANG_{language.upper()}'],
                metadata=metadata
            )
            shards.append(shard)

        return shards

    def _detect_language(self, path: str) -> str:
        """Detect programming language from file extension."""
        path_lower = path.lower()
        for language, extensions in self.LANGUAGE_PATTERNS.items():
            if any(path_lower.endswith(ext) for ext in extensions):
                return language
        return 'unknown'

    def _extract_code_entities(self, content: str, language: str) -> List[str]:
        """Extract code entities (classes, functions, variables)."""
        entities = []

        if language == 'python':
            # Extract classes
            entities.extend(re.findall(r'class\s+(\w+)', content))
            # Extract functions
            entities.extend(re.findall(r'def\s+(\w+)', content))

        elif language in ['javascript', 'typescript']:
            # Extract classes
            entities.extend(re.findall(r'class\s+(\w+)', content))
            # Extract functions
            entities.extend(re.findall(r'function\s+(\w+)', content))
            entities.extend(re.findall(r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>', content))

        elif language == 'java':
            # Extract classes
            entities.extend(re.findall(r'class\s+(\w+)', content))
            # Extract methods
            entities.extend(re.findall(r'\w+\s+(\w+)\s*\([^)]*\)\s*\{', content))

        elif language == 'go':
            # Extract functions
            entities.extend(re.findall(r'func\s+(\w+)', content))
            # Extract types
            entities.extend(re.findall(r'type\s+(\w+)\s+(?:struct|interface)', content))

        elif language == 'rust':
            # Extract functions
            entities.extend(re.findall(r'fn\s+(\w+)', content))
            # Extract structs
            entities.extend(re.findall(r'struct\s+(\w+)', content))

        return list(set(entities))[:30]  # Dedupe and limit

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """Extract import/dependency statements."""
        imports = []

        if language == 'python':
            imports.extend(re.findall(r'from\s+([\w.]+)\s+import', content))
            imports.extend(re.findall(r'import\s+([\w.]+)', content))

        elif language in ['javascript', 'typescript']:
            imports.extend(re.findall(r"import.*from\s+['\"]([^'\"]+)['\"]", content))
            imports.extend(re.findall(r"require\(['\"]([^'\"]+)['\"]\)", content))

        elif language == 'java':
            imports.extend(re.findall(r'import\s+([\w.]+);', content))

        elif language == 'go':
            imports.extend(re.findall(r'import\s+\"([^\"]+)\"', content))

        elif language == 'rust':
            imports.extend(re.findall(r'use\s+([\w:]+)', content))

        return list(set(imports))

    def _chunk_by_function(self, content: str, language: str) -> List[tuple]:
        """Chunk code by functions/methods."""
        chunks = []

        if language == 'python':
            # Simple regex-based function extraction
            pattern = r'((?:async\s+)?def\s+\w+\([^)]*\):.*?)(?=\n(?:async\s+)?def\s+|\nclass\s+|\Z)'
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                func_text = match.group(1).strip()
                func_name = re.search(r'def\s+(\w+)', func_text)
                chunks.append((
                    func_text,
                    {'function_name': func_name.group(1) if func_name else 'unknown'}
                ))

        # Fallback: chunk by lines if language-specific chunking not implemented
        if not chunks:
            chunks = self._chunk_by_lines(content, self.config.max_chunk_lines)

        return chunks

    def _chunk_by_class(self, content: str, language: str) -> List[tuple]:
        """Chunk code by classes."""
        chunks = []

        if language == 'python':
            pattern = r'(class\s+\w+.*?)(?=\nclass\s+|\Z)'
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                class_text = match.group(1).strip()
                class_name = re.search(r'class\s+(\w+)', class_text)
                chunks.append((
                    class_text,
                    {'class_name': class_name.group(1) if class_name else 'unknown'}
                ))

        # Fallback
        if not chunks:
            chunks = self._chunk_by_lines(content, self.config.max_chunk_lines)

        return chunks

    def _chunk_by_lines(self, content: str, max_lines: int) -> List[tuple]:
        """Simple line-based chunking."""
        lines = content.split('\n')
        chunks = []

        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i + max_lines]
            chunk_text = '\n'.join(chunk_lines)
            chunks.append((
                chunk_text,
                {'start_line': i + 1, 'end_line': i + len(chunk_lines)}
            ))

        return chunks

    def _parse_diff(self, diff: str) -> Dict[str, Any]:
        """Parse git diff output."""
        files = []
        current_file = None
        stats = {'insertions': 0, 'deletions': 0}

        for line in diff.split('\n'):
            # File header
            if line.startswith('diff --git'):
                if current_file:
                    files.append(current_file)
                current_file = {
                    'diff_text': '',
                    'lines_added': 0,
                    'lines_removed': 0,
                    'entities': []
                }

            # File path
            elif line.startswith('+++'):
                path = line[6:].strip()
                if current_file:
                    current_file['path'] = path
                    current_file['change_type'] = 'modified'

            # New file
            elif line.startswith('new file'):
                if current_file:
                    current_file['change_type'] = 'added'

            # Deleted file
            elif line.startswith('deleted file'):
                if current_file:
                    current_file['change_type'] = 'deleted'

            # Additions
            elif line.startswith('+') and not line.startswith('+++'):
                if current_file:
                    current_file['lines_added'] += 1
                    current_file['diff_text'] += line + '\n'
                stats['insertions'] += 1

            # Deletions
            elif line.startswith('-') and not line.startswith('---'):
                if current_file:
                    current_file['lines_removed'] += 1
                    current_file['diff_text'] += line + '\n'
                stats['deletions'] += 1

            # Context
            elif current_file:
                current_file['diff_text'] += line + '\n'

        if current_file:
            files.append(current_file)

        return {'files': files, 'stats': stats}

    def _create_diff_summary(self, diff_data: Dict[str, Any], raw_data: Dict[str, Any]) -> str:
        """Create human-readable diff summary."""
        message = raw_data.get('message', 'No commit message')
        author = raw_data.get('author', 'Unknown')
        sha = raw_data.get('commit_sha', 'uncommitted')[:8]

        summary = f"# Commit: {sha}\n"
        summary += f"# Author: {author}\n"
        summary += f"# Message: {message}\n\n"
        summary += f"Files changed: {len(diff_data['files'])}\n"
        summary += f"Insertions: +{diff_data['stats']['insertions']}\n"
        summary += f"Deletions: -{diff_data['stats']['deletions']}\n\n"

        summary += "Changed files:\n"
        for file_change in diff_data['files']:
            change_type = file_change['change_type']
            path = file_change.get('path', 'unknown')
            summary += f"  {change_type}: {path}\n"

        return summary

    def _extract_diff_entities(self, diff_data: Dict[str, Any]) -> List[str]:
        """Extract entities from diff (file names, function names)."""
        entities = []
        for file_change in diff_data['files']:
            path = file_change.get('path', '')
            # Add file name as entity
            if path:
                entities.append(Path(path).name)
        return list(set(entities))[:20]

    def _create_structure_text(self, files: List[str], root_path: str) -> str:
        """Create tree-like structure representation."""
        structure = f"# Repository: {Path(root_path).name}\n\n"
        structure += "File tree:\n"

        # Sort files for cleaner output
        sorted_files = sorted(files)

        for file_path in sorted_files:
            # Just use the file path directly (might be relative already)
            path_obj = Path(file_path)
            indent = '  ' * (len(path_obj.parts) - 1)
            structure += f"{indent}{path_obj.name}\n"

        return structure

    def _count_languages(self, files: List[str]) -> Dict[str, int]:
        """Count files by language."""
        lang_counts = {}
        for file_path in files:
            lang = self._detect_language(file_path)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        return lang_counts

    async def _enrich_shards(self, shards: List[MemoryShard]) -> List[MemoryShard]:
        """
        Enrich shards using optional enrichment services.
        Delegates to parent class enrichment infrastructure.
        """
        for shard in shards:
            enrichment = await self.enrich(shard.text)

            # Merge enrichment results
            if enrichment:
                if 'ollama' in enrichment and 'entities' in enrichment['ollama']:
                    shard.entities.extend(enrichment['ollama']['entities'])
                    shard.entities = list(set(shard.entities))[:40]

                if 'ollama' in enrichment and 'motifs' in enrichment['ollama']:
                    shard.motifs.extend(enrichment['ollama']['motifs'])
                    shard.motifs = list(set(shard.motifs))

                if shard.metadata is None:
                    shard.metadata = {}
                shard.metadata['enrichment'] = enrichment

        return shards


# Convenience functions
async def spin_code_file(
    path: str,
    content: str,
    language: Optional[str] = None,
    chunk_by: Optional[str] = None,
    enable_enrichment: bool = False
) -> List[MemoryShard]:
    """
    Quick function to convert a code file into MemoryShards.

    Args:
        path: File path
        content: File content
        language: Programming language (auto-detect if None)
        chunk_by: How to chunk ('function', 'class', 'file', None)
        enable_enrichment: Enable Ollama enrichment

    Returns:
        List of MemoryShard objects
    """
    config = CodeSpinnerConfig(
        chunk_by=chunk_by,
        enable_enrichment=enable_enrichment
    )

    spinner = CodeSpinner(config)

    raw_data = {
        'type': 'file',
        'path': path,
        'content': content,
        'language': language
    }

    return await spinner.spin(raw_data)


async def spin_git_diff(
    diff: str,
    commit_sha: Optional[str] = None,
    author: Optional[str] = None,
    message: Optional[str] = None
) -> List[MemoryShard]:
    """
    Quick function to convert a git diff into MemoryShards.

    Args:
        diff: Git diff output
        commit_sha: Commit SHA
        author: Commit author
        message: Commit message

    Returns:
        List of MemoryShard objects
    """
    spinner = CodeSpinner()

    raw_data = {
        'type': 'diff',
        'diff': diff,
        'commit_sha': commit_sha,
        'author': author,
        'message': message
    }

    return await spinner.spin(raw_data)


async def spin_repository(
    root_path: str,
    files: List[str]
) -> List[MemoryShard]:
    """
    Quick function to convert repository structure into MemoryShards.

    Args:
        root_path: Repository root directory
        files: List of file paths in repository

    Returns:
        List of MemoryShard objects
    """
    spinner = CodeSpinner()

    raw_data = {
        'type': 'repo',
        'root_path': root_path,
        'files': files
    }

    return await spinner.spin(raw_data)

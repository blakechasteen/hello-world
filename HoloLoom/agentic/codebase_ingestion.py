#!/usr/bin/env python3
"""
Codebase Ingestion System
==========================
Indexes entire codebases into HoloLoom's knowledge graph for deep analysis.

Features:
- Multi-language code parsing (Python, TypeScript, JavaScript, Java, C++, Rust)
- Entity extraction (functions, classes, imports, variables)
- Knowledge graph construction with typed relationships
- Incremental updates (only re-index changed files)
- Semantic search over code entities

Usage:
    from HoloLoom.agentic.codebase_ingestion import CodebaseIndexer

    indexer = CodebaseIndexer(kg=knowledge_graph)
    await indexer.ingest_workspace("/path/to/project", language="python")

    # Search for entities
    results = await indexer.search_entity("authenticate", entity_type="function")
"""

import ast
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from HoloLoom.memory.graph import KG, KGEdge
from HoloLoom.documentation.types import MemoryShard


logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of code entities we track."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    MODULE = "module"
    INTERFACE = "interface"
    TYPE = "type"


class Language(str, Enum):
    """Supported languages."""
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, etc.)."""
    name: str
    entity_type: EntityType
    file_path: str
    line_number: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parameters: List[str] = None
    return_type: Optional[str] = None
    decorators: List[str] = None
    parent: Optional[str] = None  # For methods (parent = class name)
    references: List[str] = None  # Other entities this references

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = []
        if self.decorators is None:
            self.decorators = []
        if self.references is None:
            self.references = []


@dataclass
class CodeFile:
    """Represents a parsed code file."""
    path: str
    language: Language
    entities: List[CodeEntity]
    imports: List[str]
    module_docstring: Optional[str] = None


class PythonParser:
    """Parse Python source code to extract entities."""

    def parse_file(self, file_path: str, content: str) -> CodeFile:
        """Parse Python file and extract all entities."""
        try:
            tree = ast.parse(content, filename=file_path)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return CodeFile(path=file_path, language=Language.PYTHON, entities=[], imports=[])

        entities = []
        imports = []
        module_docstring = ast.get_docstring(tree)

        for node in ast.walk(tree):
            # Functions
            if isinstance(node, ast.FunctionDef):
                entity = self._parse_function(node, file_path)
                entities.append(entity)

            # Classes
            elif isinstance(node, ast.ClassDef):
                entity = self._parse_class(node, file_path)
                entities.append(entity)

                # Methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method = self._parse_method(item, node.name, file_path)
                        entities.append(method)

            # Imports
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.extend(self._parse_import(node))

        return CodeFile(
            path=file_path,
            language=Language.PYTHON,
            entities=entities,
            imports=imports,
            module_docstring=module_docstring
        )

    def _parse_function(self, node: ast.FunctionDef, file_path: str) -> CodeEntity:
        """Parse function definition."""
        return CodeEntity(
            name=node.name,
            entity_type=EntityType.FUNCTION,
            file_path=file_path,
            line_number=node.lineno,
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node),
            parameters=[arg.arg for arg in node.args.args],
            return_type=self._get_return_annotation(node),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            references=self._extract_references(node)
        )

    def _parse_class(self, node: ast.ClassDef, file_path: str) -> CodeEntity:
        """Parse class definition."""
        return CodeEntity(
            name=node.name,
            entity_type=EntityType.CLASS,
            file_path=file_path,
            line_number=node.lineno,
            docstring=ast.get_docstring(node),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            references=[base.id for base in node.bases if isinstance(base, ast.Name)]
        )

    def _parse_method(self, node: ast.FunctionDef, class_name: str, file_path: str) -> CodeEntity:
        """Parse method definition."""
        return CodeEntity(
            name=node.name,
            entity_type=EntityType.METHOD,
            file_path=file_path,
            line_number=node.lineno,
            signature=self._get_signature(node),
            docstring=ast.get_docstring(node),
            parameters=[arg.arg for arg in node.args.args],
            return_type=self._get_return_annotation(node),
            decorators=[self._get_decorator_name(d) for d in node.decorator_list],
            parent=class_name,
            references=self._extract_references(node)
        )

    def _parse_import(self, node) -> List[str]:
        """Parse import statement."""
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            return [f"{module}.{alias.name}" for alias in node.names]
        return []

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Get function signature as string."""
        params = ", ".join(arg.arg for arg in node.args.args)
        return f"{node.name}({params})"

    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Get return type annotation."""
        if node.returns:
            return ast.unparse(node.returns)
        return None

    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call):
            return ast.unparse(decorator.func)
        return ast.unparse(decorator)

    def _extract_references(self, node) -> List[str]:
        """Extract names referenced in function/method body."""
        references = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                references.add(child.id)
            elif isinstance(child, ast.Attribute):
                references.add(child.attr)
        return list(references)


class TypeScriptParser:
    """Parse TypeScript/JavaScript files (regex-based for now)."""

    def parse_file(self, file_path: str, content: str) -> CodeFile:
        """Parse TypeScript/JavaScript file."""
        entities = []
        imports = []

        # Extract imports
        import_pattern = r'import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]'
        imports = re.findall(import_pattern, content)

        # Extract functions
        func_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
        for match in re.finditer(func_pattern, content):
            name = match.group(1)
            params = [p.strip().split(':')[0].strip() for p in match.group(2).split(',') if p.strip()]
            line_num = content[:match.start()].count('\n') + 1

            entities.append(CodeEntity(
                name=name,
                entity_type=EntityType.FUNCTION,
                file_path=file_path,
                line_number=line_num,
                parameters=params,
                signature=f"{name}({', '.join(params)})"
            ))

        # Extract classes
        class_pattern = r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(class_pattern, content):
            name = match.group(1)
            extends = match.group(2)
            line_num = content[:match.start()].count('\n') + 1

            entities.append(CodeEntity(
                name=name,
                entity_type=EntityType.CLASS,
                file_path=file_path,
                line_number=line_num,
                references=[extends] if extends else []
            ))

        # Extract interfaces
        interface_pattern = r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+(\w+))?'
        for match in re.finditer(interface_pattern, content):
            name = match.group(1)
            extends = match.group(2)
            line_num = content[:match.start()].count('\n') + 1

            entities.append(CodeEntity(
                name=name,
                entity_type=EntityType.INTERFACE,
                file_path=file_path,
                line_number=line_num,
                references=[extends] if extends else []
            ))

        language = Language.TYPESCRIPT if file_path.endswith('.ts') else Language.JAVASCRIPT

        return CodeFile(
            path=file_path,
            language=language,
            entities=entities,
            imports=imports
        )


class CodebaseIndexer:
    """
    Indexes codebases into HoloLoom's knowledge graph.

    Provides semantic search over code entities and hallucination detection.
    """

    def __init__(self, kg: Optional[KG] = None):
        """
        Initialize indexer.

        Args:
            kg: Knowledge graph to store entities. Creates new if None.
        """
        self.kg = kg or KG()
        self.parsers = {
            Language.PYTHON: PythonParser(),
            Language.TYPESCRIPT: TypeScriptParser(),
            Language.JAVASCRIPT: TypeScriptParser(),
        }
        self.indexed_files: Set[str] = set()

    async def ingest_workspace(
        self,
        workspace_path: str,
        languages: Optional[List[Language]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Ingest entire workspace into knowledge graph.

        Args:
            workspace_path: Path to workspace root
            languages: Languages to index (all if None)
            exclude_patterns: Patterns to exclude (e.g., "**/node_modules/**")

        Returns:
            Statistics about ingestion (files processed, entities found, etc.)
        """
        if languages is None:
            languages = [Language.PYTHON, Language.TYPESCRIPT, Language.JAVASCRIPT]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**",
                "**/.venv/**",
                "**/venv/**",
                "**/__pycache__/**",
                "**/dist/**",
                "**/build/**",
                "**/.git/**"
            ]

        workspace = Path(workspace_path)
        stats = {
            "files_processed": 0,
            "entities_found": 0,
            "errors": 0,
            "by_language": {},
            "by_entity_type": {}
        }

        # Find files to process
        file_extensions = {
            Language.PYTHON: [".py"],
            Language.TYPESCRIPT: [".ts", ".tsx"],
            Language.JAVASCRIPT: [".js", ".jsx"]
        }

        for language in languages:
            extensions = file_extensions.get(language, [])
            files = []

            for ext in extensions:
                files.extend(workspace.rglob(f"*{ext}"))

            # Filter out excluded patterns
            files = [f for f in files if not any(f.match(pattern) for pattern in exclude_patterns)]

            logger.info(f"Indexing {len(files)} {language.value} files...")

            for file_path in files:
                try:
                    await self.ingest_file(str(file_path), language)
                    stats["files_processed"] += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    stats["errors"] += 1

            stats["by_language"][language.value] = len(files)

        # Count entities
        stats["entities_found"] = len(self.kg.G.nodes)

        # Count by entity type
        for node in self.kg.G.nodes:
            node_data = self.kg.G.nodes[node]
            entity_type = node_data.get("entity_type", "unknown")
            stats["by_entity_type"][entity_type] = stats["by_entity_type"].get(entity_type, 0) + 1

        logger.info(f"Ingestion complete: {stats}")
        return stats

    async def ingest_file(self, file_path: str, language: Language) -> CodeFile:
        """
        Ingest single file into knowledge graph.

        Args:
            file_path: Path to file
            language: Programming language

        Returns:
            Parsed CodeFile object
        """
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse file
        parser = self.parsers.get(language)
        if not parser:
            raise ValueError(f"No parser available for {language}")

        code_file = parser.parse_file(file_path, content)

        # Add entities to knowledge graph
        for entity in code_file.entities:
            self._add_entity_to_kg(entity, code_file)

        # Add imports as relationships
        for imp in code_file.imports:
            self.kg.add_edges([
                KGEdge(
                    source=file_path,
                    target=imp,
                    relation="IMPORTS",
                    weight=1.0
                )
            ])

        self.indexed_files.add(file_path)

        return code_file

    def _add_entity_to_kg(self, entity: CodeEntity, code_file: CodeFile):
        """Add code entity to knowledge graph with relationships."""
        # Add entity node
        node_id = f"{entity.file_path}:{entity.name}"

        self.kg.G.add_node(
            node_id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            file_path=entity.file_path,
            line_number=entity.line_number,
            signature=entity.signature,
            docstring=entity.docstring,
            parameters=entity.parameters,
            return_type=entity.return_type,
            decorators=entity.decorators,
            parent=entity.parent,
            language=code_file.language.value
        )

        # Add relationships
        edges = []

        # Parent relationship (method → class)
        if entity.parent:
            parent_id = f"{entity.file_path}:{entity.parent}"
            edges.append(KGEdge(
                source=node_id,
                target=parent_id,
                relation="MEMBER_OF",
                weight=1.0
            ))

        # Reference relationships
        for ref in entity.references:
            # Only add if reference exists in graph
            ref_id = f"{entity.file_path}:{ref}"
            if self.kg.G.has_node(ref_id):
                edges.append(KGEdge(
                    source=node_id,
                    target=ref_id,
                    relation="USES",
                    weight=1.0
                ))

        if edges:
            self.kg.add_edges(edges)

    async def search_entity(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
        fuzzy: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for code entities by name.

        Args:
            name: Entity name to search for
            entity_type: Filter by entity type (None = all types)
            fuzzy: Allow fuzzy matching

        Returns:
            List of matching entities with metadata
        """
        results = []

        for node_id in self.kg.G.nodes:
            node_data = self.kg.G.nodes[node_id]
            node_name = node_data.get("name", "")
            node_type = node_data.get("entity_type", "")

            # Type filter
            if entity_type and node_type != entity_type.value:
                continue

            # Name match
            if fuzzy:
                # Case-insensitive substring match
                if name.lower() in node_name.lower():
                    results.append({
                        "id": node_id,
                        "name": node_name,
                        "type": node_type,
                        "file": node_data.get("file_path"),
                        "line": node_data.get("line_number"),
                        "signature": node_data.get("signature"),
                        "docstring": node_data.get("docstring")
                    })
            else:
                # Exact match
                if node_name == name:
                    results.append({
                        "id": node_id,
                        "name": node_name,
                        "type": node_type,
                        "file": node_data.get("file_path"),
                        "line": node_data.get("line_number"),
                        "signature": node_data.get("signature"),
                        "docstring": node_data.get("docstring")
                    })

        return results

    async def find_similar_entities(
        self,
        name: str,
        entity_type: Optional[EntityType] = None,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find entities with similar names (for hallucination detection).

        Args:
            name: Entity name to find similar to
            entity_type: Filter by entity type
            k: Number of results to return

        Returns:
            List of (entity_name, similarity_score) tuples
        """
        from difflib import SequenceMatcher

        candidates = []

        for node_id in self.kg.G.nodes:
            node_data = self.kg.G.nodes[node_id]
            node_name = node_data.get("name", "")
            node_type = node_data.get("entity_type", "")

            # Type filter
            if entity_type and node_type != entity_type.value:
                continue

            # Calculate similarity
            similarity = SequenceMatcher(None, name.lower(), node_name.lower()).ratio()

            if similarity > 0.3:  # Threshold for "similar"
                candidates.append((node_name, similarity))

        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:k]

    def to_memory_shards(self) -> List[MemoryShard]:
        """
        Convert indexed codebase to MemoryShard format for HoloLoom.

        Returns:
            List of MemoryShard objects representing code entities
        """
        shards = []

        for node_id in self.kg.G.nodes:
            node_data = self.kg.G.nodes[node_id]

            # Create text representation
            name = node_data.get("name", "")
            entity_type = node_data.get("entity_type", "")
            signature = node_data.get("signature", "")
            docstring = node_data.get("docstring", "")
            file_path = node_data.get("file_path", "")

            text = f"{entity_type} {name}"
            if signature:
                text += f" - Signature: {signature}"
            if docstring:
                text += f"\nDocumentation: {docstring}"

            shard = MemoryShard(
                id=node_id,
                text=text,
                episode=f"codebase_{file_path}",
                entities=[name],
                motifs=[entity_type],
                metadata={
                    "source": "codebase_ingestion",
                    "file_path": file_path,
                    "line_number": node_data.get("line_number"),
                    "entity_type": entity_type,
                    "language": node_data.get("language")
                }
            )

            shards.append(shard)

        return shards

    def get_statistics(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        stats = {
            "total_entities": len(self.kg.G.nodes),
            "total_files": len(self.indexed_files),
            "total_relationships": len(self.kg.G.edges),
            "by_type": {}
        }

        for node_id in self.kg.G.nodes:
            node_data = self.kg.G.nodes[node_id]
            entity_type = node_data.get("entity_type", "unknown")
            stats["by_type"][entity_type] = stats["by_type"].get(entity_type, 0) + 1

        return stats

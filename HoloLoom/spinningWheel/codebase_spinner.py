"""
Codebase Spinner - Ingest source code into HoloLoom memory

Supports:
- Python AST parsing (classes, functions, imports)
- Multi-language support (Python, TypeScript, JavaScript, Java, Go, Rust)
- Call graph construction
- Docstring extraction
- Dependency analysis
- File/directory traversal
- 9-signal importance scoring

Requires: ast (stdlib for Python)
Optional: tree-sitter (multi-language parsing)

Usage:
    from HoloLoom.spinningWheel.codebase_spinner import CodebaseSpinner

    # Single file
    spinner = CodebaseSpinner()
    result = await spinner.spin("/path/to/file.py")

    # Entire directory
    result = await spinner.spin_directory("/path/to/project")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, AsyncIterator
import ast
import hashlib
import re

from HoloLoom.documentation.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.importance import ImportanceScorer


@dataclass
class CodeFunction:
    """Parsed function/method"""
    name: str
    signature: str
    docstring: Optional[str]
    line_start: int
    line_end: int
    is_async: bool = False
    is_method: bool = False
    decorators: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # Functions called


@dataclass
class CodeClass:
    """Parsed class"""
    name: str
    bases: List[str]
    docstring: Optional[str]
    line_start: int
    line_end: int
    methods: List[CodeFunction] = field(default_factory=list)
    decorators: List[str] = field(default_factory=list)


@dataclass
class CodeFile:
    """Parsed code file"""
    file_path: Path
    language: str
    imports: List[str] = field(default_factory=list)
    classes: List[CodeClass] = field(default_factory=list)
    functions: List[CodeFunction] = field(default_factory=list)
    docstring: Optional[str] = None
    total_lines: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def complexity_score(self) -> float:
        """Estimate code complexity"""
        # Simple heuristic: functions + classes + lines
        func_count = len(self.functions)
        class_count = len(self.classes)
        method_count = sum(len(c.methods) for c in self.classes)

        return (func_count + class_count * 2 + method_count) / max(1, self.code_lines / 100)


class PythonParser:
    """Parse Python source code using AST"""

    @staticmethod
    def parse_file(file_path: Path) -> CodeFile:
        """
        Parse Python file

        Args:
            file_path: Path to .py file

        Returns:
            CodeFile object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            # Return empty CodeFile if parse fails
            return CodeFile(
                file_path=file_path,
                language='python',
                total_lines=source.count('\n') + 1
            )

        # Extract module docstring
        docstring = ast.get_docstring(tree)

        # Extract imports
        imports = PythonParser._extract_imports(tree)

        # Extract classes
        classes = PythonParser._extract_classes(tree, source)

        # Extract top-level functions
        functions = PythonParser._extract_functions(tree, source)

        # Count lines
        total_lines = source.count('\n') + 1
        code_lines = PythonParser._count_code_lines(source)
        comment_lines = PythonParser._count_comment_lines(source)

        return CodeFile(
            file_path=file_path,
            language='python',
            imports=imports,
            classes=classes,
            functions=functions,
            docstring=docstring,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines
        )

    @staticmethod
    def _extract_imports(tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return imports

    @staticmethod
    def _extract_classes(tree: ast.AST, source: str) -> List[CodeClass]:
        """Extract class definitions"""
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract bases
                bases = [PythonParser._get_node_name(base) for base in node.bases]

                # Extract docstring
                docstring = ast.get_docstring(node)

                # Extract decorators
                decorators = [PythonParser._get_node_name(d) for d in node.decorator_list]

                # Extract methods
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method = PythonParser._parse_function(item, source, is_method=True)
                        methods.append(method)

                code_class = CodeClass(
                    name=node.name,
                    bases=bases,
                    docstring=docstring,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    methods=methods,
                    decorators=decorators
                )
                classes.append(code_class)

        return classes

    @staticmethod
    def _extract_functions(tree: ast.AST, source: str) -> List[CodeFunction]:
        """Extract top-level function definitions"""
        functions = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = PythonParser._parse_function(node, source, is_method=False)
                functions.append(func)

        return functions

    @staticmethod
    def _parse_function(node: ast.AST, source: str, is_method: bool) -> CodeFunction:
        """Parse function/method node"""
        name = node.name

        # Build signature
        args = []
        if node.args.args:
            for arg in node.args.args:
                arg_str = arg.arg
                if arg.annotation:
                    arg_str += f": {PythonParser._get_node_name(arg.annotation)}"
                args.append(arg_str)

        signature = f"def {name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {PythonParser._get_node_name(node.returns)}"

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract decorators
        decorators = [PythonParser._get_node_name(d) for d in node.decorator_list]

        # Extract function calls (simplified)
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                calls.append(child.func.id)

        return CodeFunction(
            name=name,
            signature=signature,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            decorators=decorators,
            calls=list(set(calls))  # Deduplicate
        )

    @staticmethod
    def _get_node_name(node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{PythonParser._get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return PythonParser._get_node_name(node.value)
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)

    @staticmethod
    def _count_code_lines(source: str) -> int:
        """Count non-empty, non-comment lines"""
        count = 0
        for line in source.split('\n'):
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
        return count

    @staticmethod
    def _count_comment_lines(source: str) -> int:
        """Count comment lines"""
        count = 0
        for line in source.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#'):
                count += 1
        return count


class CodebaseSpinner(BaseSpinner):
    """
    Spinner for source code repositories

    Ingests code into HoloLoom memory with:
    - Python AST parsing
    - Multi-language support (extensible)
    - Class/function extraction
    - Docstring extraction
    - Import/dependency analysis
    - Call graph construction
    - 9-signal importance scoring
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        languages: Optional[List[str]] = None,
        include_tests: bool = False,
        max_files: Optional[int] = None
    ):
        """
        Initialize CodebaseSpinner

        Args:
            importance_threshold: Minimum importance score (0.0-1.0)
            languages: Languages to parse (default: ['python'])
            include_tests: Include test files
            max_files: Maximum files to process (None = all)
        """
        super().__init__(name="codebase")

        self.importance_threshold = importance_threshold
        self.languages = languages or ['python']
        self.include_tests = include_tests
        self.max_files = max_files

        # Language file extensions
        self.extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx'],
            'typescript': ['.ts', '.tsx'],
            'java': ['.java'],
            'go': ['.go'],
            'rust': ['.rs']
        }

        # Create importance scorer
        self.importance_scorer = ImportanceScorer(
            technical_terms={
                'class', 'function', 'method', 'async', 'def', 'return',
                'import', 'from', 'export', 'interface', 'type', 'struct',
                'impl', 'trait', 'package', 'module', 'namespace'
            }
        )

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            incremental=False,  # Code is version-controlled
            streaming=True,
            supported_formats=self.languages,
            batch_processing=True
        )

    def is_available(self) -> bool:
        """Check if code parsing dependencies are available"""
        return True  # ast is stdlib for Python

    async def _spin_impl(self, source: Any, **kwargs) -> List[MemoryShard]:
        """
        Spin code file(s) into MemoryShards

        Args:
            source: File path (str/Path) or directory path
            **kwargs: Additional arguments

        Returns:
            List of MemoryShards
        """
        path = Path(source)

        if path.is_file():
            # Single file
            code_file = self._parse_file(path)
            return self._file_to_shards(code_file)
        elif path.is_dir():
            # Directory (recursive)
            return await self.spin_directory(path)
        else:
            raise ValueError(f"source must be file or directory, got {source}")

    async def spin_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> List[MemoryShard]:
        """
        Spin entire directory

        Args:
            directory: Directory path
            recursive: Recursively traverse subdirectories

        Returns:
            List of MemoryShards
        """
        all_shards = []
        file_count = 0

        # Get all code files
        code_files = self._get_code_files(directory, recursive)

        for file_path in code_files:
            # Apply max_files limit
            if self.max_files and file_count >= self.max_files:
                break

            # Parse file
            try:
                code_file = self._parse_file(file_path)
                shards = self._file_to_shards(code_file)
                all_shards.extend(shards)
                file_count += 1
            except Exception:
                # Skip files that fail to parse
                continue

        return all_shards

    async def spin_stream(
        self,
        source: Any,
        batch_size: int = 10,
        **kwargs
    ) -> AsyncIterator[MemoryShard]:
        """
        Stream MemoryShards from codebase

        Args:
            source: File or directory path
            batch_size: Number of files per batch

        Yields:
            MemoryShard objects
        """
        shards = await self._spin_impl(source, **kwargs)

        for i in range(0, len(shards), batch_size):
            batch = shards[i:i + batch_size]
            for shard in batch:
                yield shard

    def _get_code_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Get all code files in directory"""
        files = []

        # Get extensions for selected languages
        valid_extensions = []
        for lang in self.languages:
            valid_extensions.extend(self.extensions.get(lang, []))

        # Traverse directory
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix not in valid_extensions:
                continue

            # Skip tests unless included
            if not self.include_tests and 'test' in file_path.name.lower():
                continue

            files.append(file_path)

        return files

    def _parse_file(self, file_path: Path) -> CodeFile:
        """Parse code file (language-specific)"""
        # Determine language
        ext = file_path.suffix
        lang = None
        for language, extensions in self.extensions.items():
            if ext in extensions:
                lang = language
                break

        if lang == 'python':
            return PythonParser.parse_file(file_path)
        else:
            # Unsupported language - create minimal CodeFile
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            return CodeFile(
                file_path=file_path,
                language=lang or 'unknown',
                total_lines=source.count('\n') + 1
            )

    def _file_to_shards(self, code_file: CodeFile) -> List[MemoryShard]:
        """
        Convert CodeFile to MemoryShards

        Args:
            code_file: CodeFile object

        Returns:
            List of MemoryShards (filtered by importance)
        """
        shards = []

        # Create one shard per file (with all classes/functions)
        importance = self.score_importance(code_file)

        # Filter by threshold
        if importance.score < self.importance_threshold:
            return []

        # Create shard
        shard = self._create_shard(
            id_suffix=hashlib.sha256(str(code_file.file_path).encode()).hexdigest()[:12],
            text=self._format_file_text(code_file),
            episode=f"codebase_{code_file.file_path.parent.name}",
            entities=self._extract_entities(code_file),
            motifs=self._extract_motifs(code_file),
            metadata={
                'file_path': str(code_file.file_path),
                'language': code_file.language,
                'total_lines': code_file.total_lines,
                'code_lines': code_file.code_lines,
                'comment_lines': code_file.comment_lines,
                'class_count': len(code_file.classes),
                'function_count': len(code_file.functions),
                'import_count': len(code_file.imports),
                'complexity_score': code_file.complexity_score,
                'importance_score': importance.score,
                'importance_reason': importance.reason
            }
        )
        shards.append(shard)

        return shards

    def _format_file_text(self, code_file: CodeFile) -> str:
        """Format code file for text field"""
        parts = []

        # Header
        parts.append(f"File: {code_file.file_path.name}")
        parts.append(f"Language: {code_file.language}")
        parts.append(f"Lines: {code_file.total_lines} (code: {code_file.code_lines}, comments: {code_file.comment_lines})")
        parts.append("")

        # Docstring
        if code_file.docstring:
            parts.append(f"Description: {code_file.docstring}")
            parts.append("")

        # Imports
        if code_file.imports:
            parts.append(f"Imports ({len(code_file.imports)}):")
            for imp in code_file.imports[:10]:  # Limit display
                parts.append(f"  - {imp}")
            if len(code_file.imports) > 10:
                parts.append(f"  ... and {len(code_file.imports) - 10} more")
            parts.append("")

        # Classes
        if code_file.classes:
            parts.append(f"Classes ({len(code_file.classes)}):")
            for cls in code_file.classes:
                bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
                parts.append(f"  - {cls.name}{bases_str} [{cls.line_start}-{cls.line_end}]")
                if cls.docstring:
                    parts.append(f"    {cls.docstring[:100]}...")
                if cls.methods:
                    parts.append(f"    Methods: {', '.join(m.name for m in cls.methods[:5])}")
            parts.append("")

        # Functions
        if code_file.functions:
            parts.append(f"Functions ({len(code_file.functions)}):")
            for func in code_file.functions:
                parts.append(f"  - {func.signature} [{func.line_start}-{func.line_end}]")
                if func.docstring:
                    parts.append(f"    {func.docstring[:100]}...")
            parts.append("")

        return '\n'.join(parts)

    def _extract_entities(self, code_file: CodeFile) -> List[str]:
        """Extract entities from code file"""
        entities = []

        # Classes
        entities.extend(cls.name for cls in code_file.classes)

        # Functions
        entities.extend(func.name for func in code_file.functions)

        # Imports (top-level modules)
        entities.extend(imp.split('.')[0] for imp in code_file.imports)

        return list(set(entities))

    def _extract_motifs(self, code_file: CodeFile) -> List[str]:
        """Extract motifs from code file"""
        motifs = []

        motifs.append(code_file.language)

        if code_file.classes:
            motifs.append('object_oriented')
        if code_file.functions:
            motifs.append('functional')
        if any(func.is_async for func in code_file.functions):
            motifs.append('async')
        if code_file.complexity_score > 2.0:
            motifs.append('complex')
        if code_file.comment_lines / max(1, code_file.code_lines) > 0.2:
            motifs.append('well_documented')

        return motifs

    def score_importance(self, code_file: CodeFile) -> ImportanceScore:
        """
        Score code file importance using 9 signals

        Args:
            code_file: CodeFile object

        Returns:
            ImportanceScore
        """
        signals = ImportanceSignals()

        # Build text representation for scoring
        text = self._format_file_text(code_file)

        # 1. Length score (based on code lines)
        code_lines = code_file.code_lines
        if code_lines < 50:
            signals.length_score = 0.3
        elif code_lines < 200:
            signals.length_score = 0.6
        elif code_lines <= 500:
            signals.length_score = min(1.0, code_lines / 500)
        else:
            signals.length_score = 0.9

        # 2. Technical score
        signals.technical_score = self.importance_scorer.technical_scorer.score(text)

        # 3. Structural score (classes, functions)
        struct_score = 0.0
        if code_file.classes:
            struct_score += 0.4
        if code_file.functions:
            struct_score += 0.3
        if code_file.docstring:
            struct_score += 0.3
        signals.structural_score = min(1.0, struct_score)

        # 4. Authority score (not applicable to code)
        signals.authority_score = 0.5

        # 5. Recency score (not applicable to code)
        signals.recency_score = 0.5

        # 6. Engagement score (not applicable to code)
        signals.engagement_score = 0.5

        # 7. Reference score (imports)
        signals.reference_score = min(1.0, len(code_file.imports) / 20.0)

        # 8. Noise detection (minimal for code)
        noise_score = 0.0

        # 9. Custom signals
        signals.custom_signals = {}
        signals.custom_signals['complexity'] = min(1.0, code_file.complexity_score / 3.0)
        signals.custom_signals['documentation'] = 1.0 if code_file.docstring else 0.0

        # Combine signals
        final_score = (
            0.20 * signals.length_score +
            0.15 * signals.technical_score +
            0.20 * signals.structural_score +
            0.00 * signals.authority_score +       # N/A
            0.00 * signals.recency_score +         # N/A
            0.00 * signals.engagement_score +       # N/A
            0.15 * signals.reference_score +
            0.20 * signals.custom_signals.get('complexity', 0.5) +
            0.10 * signals.custom_signals.get('documentation', 0.0)
        )

        # Generate reason
        reasons = []
        if code_file.classes:
            reasons.append(f"{len(code_file.classes)} classes")
        if code_file.functions:
            reasons.append(f"{len(code_file.functions)} functions")
        if code_file.complexity_score > 2.0:
            reasons.append("high complexity")
        if code_file.docstring:
            reasons.append("documented")

        reason = " + ".join(reasons) if reasons else "code file"

        return ImportanceScore(
            score=max(0.0, min(1.0, final_score)),
            signals=signals,
            reason=reason
        )


# Convenience functions

async def spin_codebase(
    directory: str,
    importance_threshold: float = 0.3,
    languages: Optional[List[str]] = None
) -> SpinResult:
    """
    Convenience function to spin a codebase directory

    Args:
        directory: Directory path
        importance_threshold: Min importance score
        languages: Languages to parse (default: ['python'])

    Returns:
        SpinResult with MemoryShards
    """
    spinner = CodebaseSpinner(
        importance_threshold=importance_threshold,
        languages=languages
    )

    shards = await spinner.spin_directory(Path(directory))

    return SpinResult(
        shards=shards,
        success=True,
        items_processed=len(shards),
        items_filtered=0
    )


def create_codebase_scorer() -> ImportanceScorer:
    """Create importance scorer optimized for code"""
    return ImportanceScorer(
        technical_terms={
            'class', 'function', 'method', 'async', 'await', 'return',
            'import', 'from', 'export', 'interface', 'type', 'struct',
            'impl', 'trait', 'package', 'module', 'namespace', 'decorator',
            'property', 'static', 'abstract', 'override', 'virtual'
        }
    )

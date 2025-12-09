"""
Codebase Spinner - Ingest source code into HoloLoom memory

Enhanced Features (v2.0 - December 2025):
- Python AST parsing (classes, functions, imports, decorators)
- Multi-language support (Python, TypeScript, JavaScript, Java, Go, Rust)
- **CodebaseProject** - Full codebase analysis with inter-file relationships
- **Cross-file call graph** - Track function calls across modules
- **Dependency graph** - Visualize import relationships
- **Git integration** - File age, commit frequency, author expertise for scoring
- **Incremental updates** - Hash-based change detection
- Docstring extraction
- 9-signal importance scoring with git-enhanced signals

Requires: ast (stdlib for Python)
Optional: tree-sitter (multi-language parsing), gitpython (git integration)

Usage:
    from HoloLoom.spinningWheel.codebase_spinner import CodebaseSpinner, CodebaseProject

    # Single file
    spinner = CodebaseSpinner()
    result = await spinner.spin("/path/to/file.py")

    # Entire directory
    result = await spinner.spin_directory("/path/to/project")

    # Full codebase analysis with cross-file relationships
    project = await spinner.analyze_codebase("/path/to/project")
    print(project.dependency_graph)
    print(project.call_graph)
    print(project.get_most_connected_files())
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, AsyncIterator, Tuple
from collections import defaultdict
import ast
import hashlib
import re
import json
import time
import subprocess

from HoloLoom.protocols.types import MemoryShard
from HoloLoom.spinningWheel.protocol import (
    BaseSpinner,
    SpinResult,
    SpinnerCapabilities,
    SpinnerCheckpoint,
    ImportanceScore,
    ImportanceSignals
)
from HoloLoom.spinningWheel.importance import ImportanceScorer


@dataclass
class ComplexityMetrics:
    """Cyclomatic complexity metrics for a function/method."""
    cyclomatic_complexity: int  # Primary metric (McCabe): 1 + decision_points
    decision_points: int        # Number of branches
    nesting_depth: int          # Maximum nesting level
    line_count: int             # Lines in function body

    # Breakdown of decision points
    if_count: int = 0
    elif_count: int = 0
    for_count: int = 0
    while_count: int = 0
    except_count: int = 0
    with_count: int = 0
    and_or_count: int = 0       # Boolean operators
    comprehension_count: int = 0
    ternary_count: int = 0
    assert_count: int = 0

    @property
    def risk_category(self) -> str:
        """Classify complexity risk level."""
        if self.cyclomatic_complexity <= 5:
            return "low"          # Simple, low risk
        elif self.cyclomatic_complexity <= 10:
            return "moderate"     # Acceptable
        elif self.cyclomatic_complexity <= 20:
            return "high"         # Consider refactoring
        else:
            return "very_high"    # Refactor strongly recommended

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'cyclomatic_complexity': self.cyclomatic_complexity,
            'decision_points': self.decision_points,
            'nesting_depth': self.nesting_depth,
            'line_count': self.line_count,
            'risk_category': self.risk_category,
            'breakdown': {
                'if': self.if_count,
                'elif': self.elif_count,
                'for': self.for_count,
                'while': self.while_count,
                'except': self.except_count,
                'with': self.with_count,
                'and_or': self.and_or_count,
                'comprehension': self.comprehension_count,
                'ternary': self.ternary_count,
                'assert': self.assert_count
            }
        }


# =============================================================================
# API Surface Analysis
# =============================================================================

@dataclass
class APISymbol:
    """A symbol in the API surface."""
    name: str
    symbol_type: str            # "function", "method", "class", "constant"
    visibility: str             # "public", "protected", "private"
    line_number: int
    docstring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'type': self.symbol_type,
            'visibility': self.visibility,
            'line': self.line_number,
            'has_docstring': self.docstring is not None
        }


@dataclass
class APISurface:
    """API surface analysis for a file."""
    # Symbol classification
    public_symbols: List[APISymbol] = field(default_factory=list)
    protected_symbols: List[APISymbol] = field(default_factory=list)  # _prefix
    private_symbols: List[APISymbol] = field(default_factory=list)    # __prefix

    # __all__ tracking
    has_all_export: bool = False
    all_exports: List[str] = field(default_factory=list)

    @property
    def exposure_ratio(self) -> float:
        """Ratio of public symbols to total."""
        total = len(self.public_symbols) + len(self.protected_symbols) + len(self.private_symbols)
        if total == 0:
            return 0.0
        return len(self.public_symbols) / total

    @property
    def truly_public_count(self) -> int:
        """Symbols that are truly public (in __all__ or no underscore)."""
        if self.has_all_export:
            return len(self.all_exports)
        return len(self.public_symbols)

    @property
    def total_symbols(self) -> int:
        """Total number of symbols analyzed."""
        return len(self.public_symbols) + len(self.protected_symbols) + len(self.private_symbols)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'public_count': len(self.public_symbols),
            'protected_count': len(self.protected_symbols),
            'private_count': len(self.private_symbols),
            'total_symbols': self.total_symbols,
            'exposure_ratio': self.exposure_ratio,
            'has_all_export': self.has_all_export,
            'all_exports': self.all_exports,
            'truly_public_count': self.truly_public_count,
            'public_symbols': [s.to_dict() for s in self.public_symbols],
            'protected_symbols': [s.to_dict() for s in self.protected_symbols],
            'private_symbols': [s.to_dict() for s in self.private_symbols]
        }


# =============================================================================
# Module Cohesion Metrics
# =============================================================================

@dataclass
class CohesionMetrics:
    """
    Module cohesion metrics measuring internal vs external coupling.

    High cohesion (score close to 1.0) = module functions call each other (good)
    Low cohesion (score close to 0.0) = module functions call external modules (less ideal)

    Based on LCOM (Lack of Cohesion of Methods) principles.
    """
    internal_calls: int = 0   # Calls to functions within same module
    external_calls: int = 0   # Calls to functions in other modules
    total_calls: int = 0      # Total outgoing calls

    @property
    def cohesion_score(self) -> float:
        """
        Cohesion score from 0.0 to 1.0.

        Higher = more cohesive (internal calls dominate)
        Returns 1.0 if no calls (maximally cohesive by default)
        """
        if self.total_calls == 0:
            return 1.0  # No calls = maximally cohesive
        return self.internal_calls / self.total_calls

    @property
    def coupling_score(self) -> float:
        """
        Coupling score from 0.0 to 1.0.

        Higher = more coupled to external modules
        Inverse of cohesion score.
        """
        return 1.0 - self.cohesion_score

    @property
    def rating(self) -> str:
        """Classify cohesion level."""
        if self.cohesion_score >= 0.7:
            return "high"      # Good: mostly internal calls
        elif self.cohesion_score >= 0.4:
            return "moderate"  # Acceptable mix
        else:
            return "low"       # Consider refactoring

    def to_dict(self) -> Dict[str, Any]:
        return {
            'internal_calls': self.internal_calls,
            'external_calls': self.external_calls,
            'total_calls': self.total_calls,
            'cohesion_score': self.cohesion_score,
            'coupling_score': self.coupling_score,
            'rating': self.rating
        }


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
    complexity: Optional[ComplexityMetrics] = None  # Cyclomatic complexity metrics


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
    api_surface: Optional[APISurface] = None  # API visibility analysis
    cohesion: Optional[CohesionMetrics] = None  # Module cohesion metrics

    @property
    def complexity_score(self) -> float:
        """Estimate code complexity"""
        # Simple heuristic: functions + classes + lines
        func_count = len(self.functions)
        class_count = len(self.classes)
        method_count = sum(len(c.methods) for c in self.classes)

        return (func_count + class_count * 2 + method_count) / max(1, self.code_lines / 100)


# =============================================================================
# Git Integration - File history and author data
# =============================================================================

@dataclass
class GitFileInfo:
    """Git history information for a file"""
    file_path: Path
    first_commit_date: Optional[float] = None  # Unix timestamp
    last_commit_date: Optional[float] = None
    commit_count: int = 0
    authors: List[str] = field(default_factory=list)
    primary_author: Optional[str] = None
    lines_added: int = 0
    lines_deleted: int = 0

    @property
    def age_days(self) -> float:
        """File age in days since first commit"""
        if not self.first_commit_date:
            return 0.0
        return (time.time() - self.first_commit_date) / 86400

    @property
    def last_modified_days(self) -> float:
        """Days since last modification"""
        if not self.last_commit_date:
            return 0.0
        return (time.time() - self.last_commit_date) / 86400

    @property
    def author_count(self) -> int:
        """Number of unique authors"""
        return len(set(self.authors))


class GitAnalyzer:
    """Analyze git history for code files"""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self._git_available = self._check_git()

    def _check_git(self) -> bool:
        """Check if git is available and this is a git repo"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--git-dir'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def get_file_info(self, file_path: Path) -> GitFileInfo:
        """Get git history for a file"""
        if not self._git_available:
            return GitFileInfo(file_path=file_path)

        rel_path = file_path.relative_to(self.repo_path) if file_path.is_absolute() else file_path

        try:
            # Get commit history
            result = subprocess.run(
                ['git', 'log', '--follow', '--format=%H|%at|%an', '--', str(rel_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return GitFileInfo(file_path=file_path)

            lines = result.stdout.strip().split('\n')
            if not lines or not lines[0]:
                return GitFileInfo(file_path=file_path)

            commits = []
            authors = []
            for line in lines:
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        commits.append(int(parts[1]))
                        authors.append(parts[2])

            if not commits:
                return GitFileInfo(file_path=file_path)

            # Count author contributions
            author_counts = defaultdict(int)
            for author in authors:
                author_counts[author] += 1
            primary_author = max(author_counts, key=author_counts.get) if author_counts else None

            return GitFileInfo(
                file_path=file_path,
                first_commit_date=min(commits),
                last_commit_date=max(commits),
                commit_count=len(commits),
                authors=list(set(authors)),
                primary_author=primary_author
            )
        except (subprocess.SubprocessError, ValueError):
            return GitFileInfo(file_path=file_path)

    def get_file_blame_stats(self, file_path: Path) -> Dict[str, int]:
        """Get line count per author via git blame"""
        if not self._git_available:
            return {}

        rel_path = file_path.relative_to(self.repo_path) if file_path.is_absolute() else file_path

        try:
            result = subprocess.run(
                ['git', 'blame', '--line-porcelain', str(rel_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return {}

            author_lines = defaultdict(int)
            for line in result.stdout.split('\n'):
                if line.startswith('author '):
                    author = line[7:].strip()
                    author_lines[author] += 1

            return dict(author_lines)
        except subprocess.SubprocessError:
            return {}


# =============================================================================
# Codebase Project - Cross-file analysis
# =============================================================================

@dataclass
class IncrementalUpdateResult:
    """
    Result of incremental codebase analysis.

    Tracks which files were added, modified, or removed since last analysis.
    """
    added_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    unchanged_files: List[str] = field(default_factory=list)

    # Updated project (includes all files, but only changed ones were re-parsed)
    project: Optional['CodebaseProject'] = None

    @property
    def total_changed(self) -> int:
        """Total number of files that changed"""
        return len(self.added_files) + len(self.modified_files) + len(self.removed_files)

    @property
    def has_changes(self) -> bool:
        """Whether any files changed"""
        return self.total_changed > 0

    def summary(self) -> str:
        """Human-readable summary of changes"""
        return (
            f"Incremental Update: "
            f"+{len(self.added_files)} added, "
            f"~{len(self.modified_files)} modified, "
            f"-{len(self.removed_files)} removed, "
            f"={len(self.unchanged_files)} unchanged"
        )


@dataclass
class DependencyEdge:
    """Edge in dependency graph"""
    source_file: str  # File that imports
    target_file: str  # File being imported
    import_names: List[str] = field(default_factory=list)  # What's imported
    import_type: str = "import"  # "import", "from_import", "dynamic"


@dataclass
class CallEdge:
    """Edge in call graph"""
    caller_file: str
    caller_function: str
    callee_file: str
    callee_function: str
    call_count: int = 1


@dataclass
class CodebaseProject:
    """
    Represents an entire codebase with inter-file relationships.

    Provides:
    - Dependency graph (which files import which)
    - Call graph (which functions call which)
    - File metrics aggregation
    - Most connected/important file detection
    """
    root_path: Path
    files: Dict[str, CodeFile] = field(default_factory=dict)  # path -> CodeFile
    dependency_edges: List[DependencyEdge] = field(default_factory=list)
    call_edges: List[CallEdge] = field(default_factory=list)
    git_info: Dict[str, GitFileInfo] = field(default_factory=dict)  # path -> GitFileInfo

    # Codebase-level stats
    total_files: int = 0
    total_lines: int = 0
    total_code_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    languages: Set[str] = field(default_factory=set)

    @property
    def dependency_graph(self) -> Dict[str, List[str]]:
        """Adjacency list: file -> [files it imports from]"""
        graph = defaultdict(list)
        for edge in self.dependency_edges:
            if edge.target_file not in graph[edge.source_file]:
                graph[edge.source_file].append(edge.target_file)
        return dict(graph)

    @property
    def reverse_dependency_graph(self) -> Dict[str, List[str]]:
        """Reverse adjacency: file -> [files that import it]"""
        graph = defaultdict(list)
        for edge in self.dependency_edges:
            if edge.source_file not in graph[edge.target_file]:
                graph[edge.target_file].append(edge.source_file)
        return dict(graph)

    @property
    def call_graph(self) -> Dict[str, List[Tuple[str, str]]]:
        """Call graph: (file, func) -> [(file, func) it calls]"""
        graph = defaultdict(list)
        for edge in self.call_edges:
            key = f"{edge.caller_file}::{edge.caller_function}"
            target = (edge.callee_file, edge.callee_function)
            if target not in graph[key]:
                graph[key].append(target)
        return dict(graph)

    def get_import_count(self, file_path: str) -> int:
        """Number of files that import this file"""
        return len(self.reverse_dependency_graph.get(file_path, []))

    def get_dependency_count(self, file_path: str) -> int:
        """Number of files this file imports from (within project)"""
        return len(self.dependency_graph.get(file_path, []))

    def get_most_imported_files(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Files with most imports (most depended upon)"""
        import_counts = {
            path: self.get_import_count(path)
            for path in self.files.keys()
        }
        sorted_files = sorted(import_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:limit]

    def get_most_connected_files(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Files with highest connectivity score.

        Score = imported_by_count * 2 + imports_count + call_references
        """
        scores = {}
        for path in self.files.keys():
            imported_by = self.get_import_count(path)
            imports = self.get_dependency_count(path)

            # Count call references
            call_refs = sum(1 for e in self.call_edges if e.callee_file == path)

            scores[path] = imported_by * 2 + imports + call_refs

        sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:limit]

    def get_orphan_files(self) -> List[str]:
        """Files with no imports and not imported by anyone"""
        orphans = []
        for path in self.files.keys():
            if self.get_import_count(path) == 0 and self.get_dependency_count(path) == 0:
                orphans.append(path)
        return orphans

    def get_circular_dependencies(self) -> List[List[str]]:
        """Find circular import chains"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self.dependency_graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if cycle not in cycles and cycle[::-1] not in cycles:
                        cycles.append(cycle)

            rec_stack.remove(node)

        for node in self.files.keys():
            if node not in visited:
                dfs(node, [])

        return cycles

    def get_high_complexity_functions(self, threshold: int = 10) -> List[Tuple[str, str, int]]:
        """
        Get functions with cyclomatic complexity above threshold.

        Args:
            threshold: Minimum complexity to include (default: 10 = "high" risk)

        Returns:
            List of (file_path, function_name, complexity) sorted by complexity descending
        """
        results = []
        for file_path, code_file in self.files.items():
            # Check top-level functions
            for func in code_file.functions:
                if func.complexity and func.complexity.cyclomatic_complexity > threshold:
                    results.append((
                        file_path,
                        func.name,
                        func.complexity.cyclomatic_complexity
                    ))
            # Check class methods
            for cls in code_file.classes:
                for method in cls.methods:
                    if method.complexity and method.complexity.cyclomatic_complexity > threshold:
                        results.append((
                            file_path,
                            f"{cls.name}.{method.name}",
                            method.complexity.cyclomatic_complexity
                        ))

        return sorted(results, key=lambda x: x[2], reverse=True)

    def get_complexity_summary(self) -> Dict[str, Any]:
        """
        Get aggregated complexity statistics across the codebase.

        Returns:
            Dictionary with complexity stats:
            - total_functions: Total functions analyzed
            - avg_complexity: Average cyclomatic complexity
            - max_complexity: Highest complexity found
            - risk_distribution: Count by risk category
            - high_complexity_count: Functions above threshold (10)
        """
        complexities = []

        for code_file in self.files.values():
            for func in code_file.functions:
                if func.complexity:
                    complexities.append(func.complexity)
            for cls in code_file.classes:
                for method in cls.methods:
                    if method.complexity:
                        complexities.append(method.complexity)

        if not complexities:
            return {
                'total_functions': 0,
                'avg_complexity': 0.0,
                'max_complexity': 0,
                'risk_distribution': {'low': 0, 'moderate': 0, 'high': 0, 'very_high': 0},
                'high_complexity_count': 0
            }

        # Calculate stats
        cc_values = [c.cyclomatic_complexity for c in complexities]
        risk_dist = {'low': 0, 'moderate': 0, 'high': 0, 'very_high': 0}
        for c in complexities:
            risk_dist[c.risk_category] += 1

        return {
            'total_functions': len(complexities),
            'avg_complexity': sum(cc_values) / len(cc_values),
            'max_complexity': max(cc_values),
            'risk_distribution': risk_dist,
            'high_complexity_count': sum(1 for cc in cc_values if cc > 10)
        }

    def get_api_surface_summary(self) -> Dict[str, Any]:
        """
        Get aggregated API surface statistics across the codebase.

        Returns:
            Dictionary with API surface stats:
            - total_files_analyzed: Files with API surface data
            - total_public: Total public symbols
            - total_protected: Total protected symbols
            - total_private: Total private symbols
            - avg_exposure_ratio: Average exposure ratio (0-1)
            - files_with_all: Files that define __all__
            - most_exposed_files: Top 5 files by exposure ratio
        """
        surfaces = []
        file_exposures = []

        for file_path, code_file in self.files.items():
            if code_file.api_surface:
                surfaces.append(code_file.api_surface)
                file_exposures.append((
                    file_path,
                    code_file.api_surface.exposure_ratio,
                    code_file.api_surface.total_symbols
                ))

        if not surfaces:
            return {
                'total_files_analyzed': 0,
                'total_public': 0,
                'total_protected': 0,
                'total_private': 0,
                'avg_exposure_ratio': 0.0,
                'files_with_all': 0,
                'most_exposed_files': []
            }

        # Calculate stats
        total_public = sum(len(s.public_symbols) for s in surfaces)
        total_protected = sum(len(s.protected_symbols) for s in surfaces)
        total_private = sum(len(s.private_symbols) for s in surfaces)
        files_with_all = sum(1 for s in surfaces if s.has_all_export)

        # Average exposure ratio (only for files with symbols)
        valid_ratios = [s.exposure_ratio for s in surfaces if s.total_symbols > 0]
        avg_exposure = sum(valid_ratios) / len(valid_ratios) if valid_ratios else 0.0

        # Most exposed files (sorted by exposure ratio, descending)
        most_exposed = sorted(
            [(f, e, t) for f, e, t in file_exposures if t > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_files_analyzed': len(surfaces),
            'total_public': total_public,
            'total_protected': total_protected,
            'total_private': total_private,
            'avg_exposure_ratio': avg_exposure,
            'files_with_all': files_with_all,
            'most_exposed_files': [
                {'file': f, 'exposure_ratio': e, 'total_symbols': t}
                for f, e, t in most_exposed
            ]
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export project analysis to dictionary"""
        return {
            'root_path': str(self.root_path),
            'total_files': self.total_files,
            'total_lines': self.total_lines,
            'total_code_lines': self.total_code_lines,
            'total_classes': self.total_classes,
            'total_functions': self.total_functions,
            'languages': list(self.languages),
            'files': list(self.files.keys()),
            'dependency_graph': self.dependency_graph,
            'most_imported': self.get_most_imported_files(5),
            'most_connected': self.get_most_connected_files(5),
            'orphan_files': self.get_orphan_files(),
            'circular_dependencies': self.get_circular_dependencies(),
            'complexity_summary': self.get_complexity_summary(),
            'high_complexity_functions': self.get_high_complexity_functions(10)[:10],
            'api_surface_summary': self.get_api_surface_summary()
        }


# =============================================================================
# TypeScript/JavaScript Parser (Pattern-based)
# =============================================================================

class TypeScriptParser:
    """Parse TypeScript/JavaScript files using regex patterns"""

    # Patterns for extraction
    IMPORT_PATTERN = re.compile(
        r'''import\s+(?:(?:\{[^}]+\})|(?:\*\s+as\s+\w+)|(?:\w+))?\s*(?:,\s*(?:\{[^}]+\})|(?:\*\s+as\s+\w+))?\s*from\s+['"]([^'"]+)['"]''',
        re.MULTILINE
    )
    REQUIRE_PATTERN = re.compile(r'''require\s*\(\s*['"]([^'"]+)['"]\s*\)''')
    CLASS_PATTERN = re.compile(
        r'''(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?''',
        re.MULTILINE
    )
    FUNCTION_PATTERN = re.compile(
        r'''(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)(?:\s*:\s*(\w+))?''',
        re.MULTILINE
    )
    ARROW_PATTERN = re.compile(
        r'''(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*\w+\s*)?=>''',
        re.MULTILINE
    )
    INTERFACE_PATTERN = re.compile(
        r'''(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+([\w,\s]+))?''',
        re.MULTILINE
    )
    TYPE_PATTERN = re.compile(
        r'''(?:export\s+)?type\s+(\w+)\s*=''',
        re.MULTILINE
    )

    @staticmethod
    def parse_file(file_path: Path) -> CodeFile:
        """Parse TypeScript/JavaScript file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Determine language
        ext = file_path.suffix.lower()
        language = 'typescript' if ext in ['.ts', '.tsx'] else 'javascript'

        # Extract imports
        imports = []
        imports.extend(TypeScriptParser.IMPORT_PATTERN.findall(source))
        imports.extend(TypeScriptParser.REQUIRE_PATTERN.findall(source))

        # Extract classes
        classes = []
        for match in TypeScriptParser.CLASS_PATTERN.finditer(source):
            name = match.group(1)
            bases = []
            if match.group(2):
                bases.append(match.group(2))
            if match.group(3):
                bases.extend([x.strip() for x in match.group(3).split(',')])

            # Find class line number
            line_start = source[:match.start()].count('\n') + 1

            classes.append(CodeClass(
                name=name,
                bases=bases,
                docstring=None,  # JSDoc extraction would need more work
                line_start=line_start,
                line_end=line_start,  # Can't easily determine end
                methods=[],
                decorators=[]
            ))

        # Extract functions
        functions = []
        for match in TypeScriptParser.FUNCTION_PATTERN.finditer(source):
            name = match.group(1)
            args = match.group(2)
            return_type = match.group(3) or ''
            line_start = source[:match.start()].count('\n') + 1

            is_async = 'async' in source[max(0, match.start()-10):match.start()]

            functions.append(CodeFunction(
                name=name,
                signature=f"function {name}({args})" + (f": {return_type}" if return_type else ""),
                docstring=None,
                line_start=line_start,
                line_end=line_start,
                is_async=is_async,
                is_method=False,
                decorators=[],
                calls=[]
            ))

        # Arrow functions
        for match in TypeScriptParser.ARROW_PATTERN.finditer(source):
            name = match.group(1)
            line_start = source[:match.start()].count('\n') + 1

            is_async = 'async' in source[max(0, match.start()-10):match.end()]

            functions.append(CodeFunction(
                name=name,
                signature=f"const {name} = (...) =>",
                docstring=None,
                line_start=line_start,
                line_end=line_start,
                is_async=is_async,
                is_method=False,
                decorators=[],
                calls=[]
            ))

        # Count lines
        total_lines = source.count('\n') + 1
        code_lines = sum(1 for line in source.split('\n') if line.strip() and not line.strip().startswith('//'))
        comment_lines = sum(1 for line in source.split('\n') if line.strip().startswith('//'))

        # Add interfaces/types as metadata
        interfaces = TypeScriptParser.INTERFACE_PATTERN.findall(source)
        types = TypeScriptParser.TYPE_PATTERN.findall(source)

        return CodeFile(
            file_path=file_path,
            language=language,
            imports=imports,
            classes=classes,
            functions=functions,
            docstring=None,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            metadata={
                'interfaces': [i[0] if isinstance(i, tuple) else i for i in interfaces],
                'types': types
            }
        )


# =============================================================================
# Cyclomatic Complexity Calculator
# =============================================================================

class CyclomaticComplexityCalculator:
    """
    Calculate cyclomatic complexity using AST analysis.

    Formula: CC = E - N + 2P (edges - nodes + 2*connected_components)
    Simplified: CC = 1 + number_of_decision_points

    Decision points:
    - if, elif statements
    - for, while loops
    - except handlers
    - and/or in boolean expressions
    - assert statements
    - comprehensions with 'if' clauses
    - ternary expressions (x if cond else y)
    """

    @staticmethod
    def calculate(func_node: ast.FunctionDef) -> ComplexityMetrics:
        """
        Calculate cyclomatic complexity for a function/method AST node.

        Args:
            func_node: ast.FunctionDef or ast.AsyncFunctionDef node

        Returns:
            ComplexityMetrics with all complexity measurements
        """
        # Initialize counters
        counters = {
            'if_count': 0,
            'elif_count': 0,
            'for_count': 0,
            'while_count': 0,
            'except_count': 0,
            'with_count': 0,
            'and_or_count': 0,
            'comprehension_count': 0,
            'ternary_count': 0,
            'assert_count': 0
        }

        max_depth = [0]  # Use list to allow modification in nested function
        visited_ifs = set()  # Track If nodes we've already processed

        def visit(node: ast.AST, depth: int = 0) -> None:
            """Recursively visit AST nodes to count decision points."""
            max_depth[0] = max(max_depth[0], depth)

            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.If):
                    # Skip if we've already counted this If (as elif)
                    if id(child) in visited_ifs:
                        visit(child, depth)
                        continue

                    counters['if_count'] += 1
                    visited_ifs.add(id(child))

                    # Count elif chain
                    elif_node = child
                    while elif_node.orelse and len(elif_node.orelse) == 1 and isinstance(elif_node.orelse[0], ast.If):
                        counters['elif_count'] += 1
                        elif_node = elif_node.orelse[0]
                        visited_ifs.add(id(elif_node))

                    visit(child, depth + 1)

                elif isinstance(child, ast.For):
                    counters['for_count'] += 1
                    visit(child, depth + 1)

                elif isinstance(child, ast.While):
                    counters['while_count'] += 1
                    visit(child, depth + 1)

                elif isinstance(child, ast.ExceptHandler):
                    counters['except_count'] += 1
                    visit(child, depth)

                elif isinstance(child, ast.With):
                    counters['with_count'] += 1
                    visit(child, depth + 1)

                elif isinstance(child, ast.Assert):
                    counters['assert_count'] += 1
                    visit(child, depth)

                elif isinstance(child, ast.BoolOp):
                    # and/or count as decision points
                    # CC += number of and/or operators (len(values) - 1)
                    if isinstance(child.op, (ast.And, ast.Or)):
                        counters['and_or_count'] += len(child.values) - 1
                    visit(child, depth)

                elif isinstance(child, ast.IfExp):
                    # Ternary expression: x if cond else y
                    counters['ternary_count'] += 1
                    visit(child, depth)

                elif isinstance(child, (ast.ListComp, ast.SetComp,
                                        ast.DictComp, ast.GeneratorExp)):
                    # Comprehensions with 'if' clauses
                    for generator in child.generators:
                        counters['comprehension_count'] += len(generator.ifs)
                    visit(child, depth)

                else:
                    visit(child, depth)

        # Start traversal from function body
        visit(func_node)

        # Calculate total decision points
        decision_points = (
            counters['if_count'] +
            counters['elif_count'] +
            counters['for_count'] +
            counters['while_count'] +
            counters['except_count'] +
            counters['with_count'] +
            counters['and_or_count'] +
            counters['comprehension_count'] +
            counters['ternary_count'] +
            counters['assert_count']
        )

        # Calculate line count (handle missing end_lineno gracefully)
        line_count = 1
        if hasattr(func_node, 'end_lineno') and func_node.end_lineno and func_node.lineno:
            line_count = func_node.end_lineno - func_node.lineno + 1

        return ComplexityMetrics(
            cyclomatic_complexity=1 + decision_points,
            decision_points=decision_points,
            nesting_depth=max_depth[0],
            line_count=line_count,
            **counters
        )


# =============================================================================
# API Surface Analyzer
# =============================================================================

class APISurfaceAnalyzer:
    """
    Analyze API surface visibility for Python modules.

    Classifies symbols by Python naming conventions:
    - public: No underscore prefix (normal names)
    - protected: Single underscore prefix (_name)
    - private: Double underscore prefix (__name, but NOT dunder __name__)

    Also detects explicit __all__ exports.
    """

    @staticmethod
    def analyze(tree: ast.AST, source: str = "") -> APISurface:
        """
        Analyze API surface of a Python module.

        Args:
            tree: Parsed AST of the module
            source: Source code (optional, for additional context)

        Returns:
            APISurface with classified symbols
        """
        surface = APISurface()

        # First, look for __all__ assignment
        surface.has_all_export, surface.all_exports = APISurfaceAnalyzer._find_all_export(tree)

        # Analyze top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                symbol = APISurfaceAnalyzer._classify_symbol(
                    name=node.name,
                    symbol_type="class",
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node)
                )
                APISurfaceAnalyzer._add_to_surface(surface, symbol)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = APISurfaceAnalyzer._classify_symbol(
                    name=node.name,
                    symbol_type="function",
                    line_number=node.lineno,
                    docstring=ast.get_docstring(node)
                )
                APISurfaceAnalyzer._add_to_surface(surface, symbol)

            elif isinstance(node, ast.Assign):
                # Module-level constants (NAME = value)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Skip __all__, __version__, etc. (dunder attributes)
                        if not APISurfaceAnalyzer._is_dunder(target.id):
                            symbol = APISurfaceAnalyzer._classify_symbol(
                                name=target.id,
                                symbol_type="constant",
                                line_number=node.lineno,
                                docstring=None
                            )
                            APISurfaceAnalyzer._add_to_surface(surface, symbol)

            elif isinstance(node, ast.AnnAssign):
                # Annotated assignments: NAME: Type = value
                if isinstance(node.target, ast.Name):
                    if not APISurfaceAnalyzer._is_dunder(node.target.id):
                        symbol = APISurfaceAnalyzer._classify_symbol(
                            name=node.target.id,
                            symbol_type="constant",
                            line_number=node.lineno,
                            docstring=None
                        )
                        APISurfaceAnalyzer._add_to_surface(surface, symbol)

        return surface

    @staticmethod
    def _find_all_export(tree: ast.AST) -> Tuple[bool, List[str]]:
        """Find __all__ assignment and extract exported names."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == '__all__':
                        # Found __all__ assignment
                        exports = []
                        if isinstance(node.value, (ast.List, ast.Tuple)):
                            for elt in node.value.elts:
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                    exports.append(elt.value)
                                elif isinstance(elt, ast.Str):  # Python 3.7 compatibility
                                    exports.append(elt.s)
                        return True, exports
        return False, []

    @staticmethod
    def _classify_symbol(name: str, symbol_type: str, line_number: int,
                         docstring: Optional[str]) -> APISymbol:
        """Classify a symbol's visibility based on naming convention."""
        visibility = APISurfaceAnalyzer._get_visibility(name)
        return APISymbol(
            name=name,
            symbol_type=symbol_type,
            visibility=visibility,
            line_number=line_number,
            docstring=docstring
        )

    @staticmethod
    def _get_visibility(name: str) -> str:
        """Determine visibility from name."""
        if APISurfaceAnalyzer._is_dunder(name):
            return "public"  # Dunder methods are part of public API
        elif name.startswith('__'):
            return "private"  # __name (not dunder) is private
        elif name.startswith('_'):
            return "protected"  # _name is protected
        else:
            return "public"

    @staticmethod
    def _is_dunder(name: str) -> bool:
        """Check if name is a dunder (double underscore both sides)."""
        return name.startswith('__') and name.endswith('__') and len(name) > 4

    @staticmethod
    def _add_to_surface(surface: APISurface, symbol: APISymbol) -> None:
        """Add symbol to appropriate visibility list."""
        if symbol.visibility == "public":
            surface.public_symbols.append(symbol)
        elif symbol.visibility == "protected":
            surface.protected_symbols.append(symbol)
        elif symbol.visibility == "private":
            surface.private_symbols.append(symbol)


# =============================================================================
# Module Cohesion Calculator
# =============================================================================

class CohesionCalculator:
    """
    Calculate module cohesion based on call graph analysis.

    Cohesion measures how much functions within a module call each other
    (internal calls) vs calling external modules (external calls).

    High cohesion = good modularization (functions work together)
    Low cohesion = potential refactoring opportunity
    """

    @staticmethod
    def calculate_for_project(project: 'CodebaseProject') -> Dict[str, CohesionMetrics]:
        """
        Calculate cohesion metrics for all files in a project.

        Uses the call_edges from the project's call graph to determine
        internal vs external calls per module.

        Args:
            project: CodebaseProject with populated call_edges

        Returns:
            Dict mapping file_path -> CohesionMetrics
        """
        cohesion_by_file: Dict[str, CohesionMetrics] = {}

        # Get all functions per file for fast lookup
        functions_by_file: Dict[str, Set[str]] = {}
        for file_path, code_file in project.files.items():
            func_names = set()
            for func in code_file.functions:
                func_names.add(func.name)
            for cls in code_file.classes:
                for method in cls.methods:
                    func_names.add(f"{cls.name}.{method.name}")
            functions_by_file[file_path] = func_names

        # Count internal vs external calls per file
        for file_path in project.files:
            internal = 0
            external = 0

            # Look through call edges originating from this file
            for edge in project.call_edges:
                caller_file = edge.get('caller_file', '')
                callee_file = edge.get('callee_file', '')

                if caller_file == file_path:
                    if callee_file == file_path:
                        internal += 1
                    else:
                        external += 1

            total = internal + external
            cohesion_by_file[file_path] = CohesionMetrics(
                internal_calls=internal,
                external_calls=external,
                total_calls=total
            )

        return cohesion_by_file

    @staticmethod
    def calculate_for_file(code_file: CodeFile, all_functions_in_file: Set[str]) -> CohesionMetrics:
        """
        Calculate cohesion metrics for a single file based on function calls.

        Args:
            code_file: CodeFile with parsed functions
            all_functions_in_file: Set of all function names defined in this file

        Returns:
            CohesionMetrics for the file
        """
        internal = 0
        external = 0

        # Collect all calls from all functions
        for func in code_file.functions:
            for call in func.calls:
                if call in all_functions_in_file:
                    internal += 1
                else:
                    external += 1

        for cls in code_file.classes:
            for method in cls.methods:
                for call in method.calls:
                    # Check if call is to a method in same class or function in file
                    if call in all_functions_in_file or f"{cls.name}.{call}" in all_functions_in_file:
                        internal += 1
                    else:
                        external += 1

        return CohesionMetrics(
            internal_calls=internal,
            external_calls=external,
            total_calls=internal + external
        )


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

        # Analyze API surface
        api_surface = APISurfaceAnalyzer.analyze(tree, source)

        return CodeFile(
            file_path=file_path,
            language='python',
            imports=imports,
            classes=classes,
            functions=functions,
            docstring=docstring,
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            api_surface=api_surface
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

        # Calculate cyclomatic complexity
        complexity = CyclomaticComplexityCalculator.calculate(node)

        return CodeFunction(
            name=name,
            signature=signature,
            docstring=docstring,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            decorators=decorators,
            calls=list(set(calls)),  # Deduplicate
            complexity=complexity
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
    - Git history integration
    - 9-signal importance scoring with git-enhanced signals
    - Full codebase analysis with cross-file relationships
    """

    def __init__(
        self,
        importance_threshold: float = 0.3,
        languages: Optional[List[str]] = None,
        include_tests: bool = False,
        max_files: Optional[int] = None,
        enable_git: bool = True
    ):
        """
        Initialize CodebaseSpinner

        Args:
            importance_threshold: Minimum importance score (0.0-1.0)
            languages: Languages to parse (default: ['python'])
            include_tests: Include test files
            max_files: Maximum files to process (None = all)
            enable_git: Enable git history analysis for importance scoring
        """
        super().__init__(name="codebase")

        self.importance_threshold = importance_threshold
        self.languages = languages or ['python']
        self.include_tests = include_tests
        self.max_files = max_files
        self.enable_git = enable_git

        # Git analyzer (initialized lazily per directory)
        self._git_analyzer: Optional[GitAnalyzer] = None
        self._git_cache: Dict[str, GitFileInfo] = {}

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

        # File hash cache for incremental updates
        self._file_hashes: Dict[str, str] = {}

    def _get_git_info(self, file_path: Path) -> GitFileInfo:
        """
        Get git information for a file (with caching).

        Lazily initializes GitAnalyzer for the repository root.
        """
        if not self.enable_git:
            return GitFileInfo(file_path=file_path)

        path_str = str(file_path)

        # Check cache first
        if path_str in self._git_cache:
            return self._git_cache[path_str]

        # Initialize git analyzer if needed (find repo root)
        if self._git_analyzer is None:
            # Walk up to find .git directory
            repo_root = file_path.parent
            while repo_root != repo_root.parent:
                if (repo_root / '.git').exists():
                    self._git_analyzer = GitAnalyzer(repo_root)
                    break
                repo_root = repo_root.parent
            else:
                # No git repo found
                return GitFileInfo(file_path=file_path)

        # Get git info and cache it
        git_info = self._git_analyzer.get_file_info(file_path)
        self._git_cache[path_str] = git_info
        return git_info

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file contents for change detection."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (IOError, OSError):
            return ""

    def has_file_changed(self, file_path: Path) -> bool:
        """
        Check if file has changed since last spin (for incremental updates).

        Args:
            file_path: Path to check

        Returns:
            True if file is new or has changed, False if unchanged
        """
        path_str = str(file_path)
        current_hash = self._compute_file_hash(file_path)

        if path_str not in self._file_hashes:
            # New file
            self._file_hashes[path_str] = current_hash
            return True

        if self._file_hashes[path_str] != current_hash:
            # File changed
            self._file_hashes[path_str] = current_hash
            return True

        return False

    def save_hash_cache(self, path: Path) -> None:
        """
        Persist hash cache to disk for incremental updates across sessions.

        The hash cache stores SHA256 hashes of all processed files, enabling
        efficient detection of changed files without re-reading file contents.

        Args:
            path: Path to save the cache file (JSON format)

        Example:
            spinner.save_hash_cache(Path("./cache/file_hashes.json"))
            # Later session...
            spinner.load_hash_cache(Path("./cache/file_hashes.json"))
            result = await spinner.analyze_codebase_incremental(...)
        """
        import json

        cache_data = {
            'version': 1,
            'timestamp': time.time(),
            'hashes': self._file_hashes
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)

    def load_hash_cache(self, path: Path) -> bool:
        """
        Load hash cache from disk.

        Args:
            path: Path to the cache file

        Returns:
            True if cache was loaded successfully, False otherwise

        Example:
            if spinner.load_hash_cache(Path("./cache/file_hashes.json")):
                # Cache loaded, incremental updates will work
                result = await spinner.analyze_codebase_incremental(...)
            else:
                # No cache, full analysis needed
                project = await spinner.analyze_codebase(...)
        """
        import json

        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            # Validate cache format
            if cache_data.get('version') != 1:
                return False

            self._file_hashes = cache_data.get('hashes', {})
            return True

        except (json.JSONDecodeError, IOError, KeyError):
            return False

    def get_capabilities(self) -> SpinnerCapabilities:
        """Return spinner capabilities"""
        return SpinnerCapabilities(
            basic_processing=True,
            entity_extraction=True,
            motif_extraction=True,
            importance_scoring=True,
            incremental=True,  # Hash-based incremental updates supported
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

    async def analyze_codebase(
        self,
        directory: Path,
        recursive: bool = True
    ) -> CodebaseProject:
        """
        Perform full codebase analysis with cross-file relationships.

        Unlike spin_directory() which returns MemoryShards, this method
        returns a CodebaseProject with:
        - Parsed CodeFile objects for each file
        - Dependency graph (which files import which)
        - Call graph (which functions call which)
        - Git history for each file
        - Aggregated codebase statistics

        Args:
            directory: Root directory of the codebase
            recursive: Recursively traverse subdirectories

        Returns:
            CodebaseProject with full analysis

        Example:
            project = await spinner.analyze_codebase(Path("./my_project"))
            print(f"Files: {project.total_files}")
            print(f"Most imported: {project.get_most_imported_files(5)}")
            print(f"Circular deps: {project.get_circular_dependencies()}")
        """
        directory = Path(directory)
        project = CodebaseProject(root_path=directory)

        # Get all code files
        code_files = self._get_code_files(directory, recursive)

        # Phase 1: Parse all files and collect git info
        for file_path in code_files:
            # Apply max_files limit
            if self.max_files and project.total_files >= self.max_files:
                break

            try:
                # Parse file
                code_file = self._parse_file(file_path)

                # Compute and store hash for incremental updates
                file_hash = self._compute_file_hash(file_path)
                if file_hash:
                    self._file_hashes[str(file_path)] = file_hash

                # Store with relative path as key
                relative_path = str(file_path.relative_to(directory))
                project.files[relative_path] = code_file

                # Collect git info
                git_info = self._get_git_info(file_path)
                project.git_info[relative_path] = git_info

                # Update aggregate stats
                project.total_files += 1
                project.total_lines += code_file.total_lines
                project.total_code_lines += code_file.code_lines
                project.total_classes += len(code_file.classes)
                project.total_functions += len(code_file.functions)
                project.languages.add(code_file.language)

            except Exception:
                # Skip files that fail to parse
                continue

        # Phase 2: Build dependency graph
        self._build_dependency_graph(project, directory)

        # Phase 3: Build call graph
        self._build_call_graph(project)

        return project

    async def analyze_codebase_incremental(
        self,
        directory: Path,
        previous_project: Optional[CodebaseProject] = None,
        recursive: bool = True
    ) -> IncrementalUpdateResult:
        """
        Perform incremental codebase analysis, only processing changed files.

        Uses SHA256 file hashes to detect added, modified, and removed files.
        Only changed files are re-parsed, dramatically speeding up analysis
        for large codebases where only a few files change between runs.

        Args:
            directory: Root directory of the codebase
            previous_project: Previous analysis result (optional, uses hash cache if not provided)
            recursive: Recursively traverse subdirectories

        Returns:
            IncrementalUpdateResult with change tracking and updated CodebaseProject

        Example:
            # First analysis
            project = await spinner.analyze_codebase(Path("./my_project"))
            spinner.save_hash_cache(Path("./cache/hashes.json"))

            # Later, after some changes...
            spinner.load_hash_cache(Path("./cache/hashes.json"))
            result = await spinner.analyze_codebase_incremental(
                Path("./my_project"),
                previous_project=project
            )

            print(result.summary())
            # "Incremental Update: +2 added, ~3 modified, -1 removed, =50 unchanged"

            if result.has_changes:
                # Only process changed files
                for f in result.modified_files:
                    print(f"Modified: {f}")
        """
        directory = Path(directory)
        result = IncrementalUpdateResult()

        # Get current set of code files
        current_files = self._get_code_files(directory, recursive)
        current_file_set = {str(f.relative_to(directory)) for f in current_files}

        # Get previous file set (from previous_project or hash cache)
        if previous_project:
            previous_file_set = set(previous_project.files.keys())
        else:
            # Convert full paths in hash cache to relative paths for comparison
            previous_file_set = set()
            for hash_path in self._file_hashes.keys():
                try:
                    rel_path = str(Path(hash_path).relative_to(directory))
                    previous_file_set.add(rel_path)
                except ValueError:
                    pass  # Path not under directory, skip it

        # Detect file changes
        for file_path in current_files:
            relative_path = str(file_path.relative_to(directory))

            if relative_path not in previous_file_set:
                # New file
                result.added_files.append(relative_path)
            elif self.has_file_changed(file_path):
                # Modified file
                result.modified_files.append(relative_path)
            else:
                # Unchanged file
                result.unchanged_files.append(relative_path)

        # Detect removed files
        for previous_path in previous_file_set:
            if previous_path not in current_file_set:
                result.removed_files.append(previous_path)
                # Clean up hash cache for removed files
                full_path = str(directory / previous_path)
                if full_path in self._file_hashes:
                    del self._file_hashes[full_path]

        # Build updated project
        project = CodebaseProject(root_path=directory)

        # Copy unchanged files from previous project (if available)
        if previous_project:
            for unchanged_path in result.unchanged_files:
                if unchanged_path in previous_project.files:
                    project.files[unchanged_path] = previous_project.files[unchanged_path]
                    if unchanged_path in previous_project.git_info:
                        project.git_info[unchanged_path] = previous_project.git_info[unchanged_path]

        # Parse added and modified files
        files_to_parse = result.added_files + result.modified_files
        for relative_path in files_to_parse:
            file_path = directory / relative_path

            try:
                code_file = self._parse_file(file_path)
                project.files[relative_path] = code_file
                project.git_info[relative_path] = self._get_git_info(file_path)
            except Exception:
                # Skip files that fail to parse
                continue

        # Recalculate aggregate stats from all files
        project.total_files = len(project.files)
        for code_file in project.files.values():
            project.total_lines += code_file.total_lines
            project.total_code_lines += code_file.code_lines
            project.total_classes += len(code_file.classes)
            project.total_functions += len(code_file.functions)
            project.languages.add(code_file.language)

        # Rebuild dependency and call graphs (affected by changes)
        self._build_dependency_graph(project, directory)
        self._build_call_graph(project)

        result.project = project
        return result

    def _build_dependency_graph(
        self,
        project: CodebaseProject,
        root_dir: Path
    ) -> None:
        """
        Build dependency graph from imports.

        Resolves imports to project files where possible.
        """
        for source_path, code_file in project.files.items():
            for import_stmt in code_file.imports:
                # Try to resolve import to a project file
                resolved = self._resolve_import(import_stmt, source_path, project, root_dir)

                if resolved:
                    target_path, import_names = resolved

                    # Determine import type
                    if import_stmt.startswith('from '):
                        import_type = "from_import"
                    else:
                        import_type = "import"

                    edge = DependencyEdge(
                        source_file=source_path,
                        target_file=target_path,
                        import_names=import_names,
                        import_type=import_type
                    )
                    project.dependency_edges.append(edge)

    def _resolve_import(
        self,
        import_stmt: str,
        source_path: str,
        project: CodebaseProject,
        root_dir: Path
    ) -> Optional[Tuple[str, List[str]]]:
        """
        Resolve an import statement to a project file.

        Returns:
            Tuple of (target_path, imported_names) or None if external
        """
        # Parse the import statement
        import_names = []

        # Handle "from X import Y" style
        if import_stmt.startswith('from '):
            # from module import name1, name2
            parts = import_stmt.split(' import ')
            if len(parts) == 2:
                module_part = parts[0].replace('from ', '').strip()
                names_part = parts[1].strip()
                import_names = [n.strip() for n in names_part.split(',')]
            else:
                module_part = import_stmt.replace('from ', '').strip()
        else:
            # import module or import module as alias
            module_part = import_stmt.replace('import ', '').strip()
            if ' as ' in module_part:
                module_part = module_part.split(' as ')[0].strip()

        # Try to find matching project file
        # Convert module.submodule to module/submodule
        module_path = module_part.replace('.', '/')

        # Check possible file paths
        possible_paths = [
            f"{module_path}.py",
            f"{module_path}/__init__.py",
            f"{module_path}.ts",
            f"{module_path}.tsx",
            f"{module_path}.js",
            f"{module_path}.jsx",
            f"{module_path}/index.ts",
            f"{module_path}/index.js",
        ]

        # Handle relative imports (starting with .)
        if module_part.startswith('.'):
            source_dir = str(Path(source_path).parent)
            # Count leading dots for relative depth
            dots = len(module_part) - len(module_part.lstrip('.'))
            relative_base = module_part.lstrip('.')

            # Go up directories based on dot count
            base_path = Path(source_dir)
            for _ in range(dots - 1):
                base_path = base_path.parent

            relative_path = relative_base.replace('.', '/')
            possible_paths = [
                str(base_path / f"{relative_path}.py"),
                str(base_path / relative_path / "__init__.py"),
                str(base_path / f"{relative_path}.ts"),
                str(base_path / relative_path / "index.ts"),
            ]

        # Check if any possible path exists in project
        for possible in possible_paths:
            # Normalize path separators
            normalized = possible.replace('\\', '/')
            if normalized in project.files:
                return (normalized, import_names)

            # Also try with OS separators
            normalized_os = str(Path(possible))
            if normalized_os in project.files:
                return (normalized_os, import_names)

        # Import is external (not in project)
        return None

    def _build_call_graph(self, project: CodebaseProject) -> None:
        """
        Build call graph from function calls.

        Tracks which functions call which other functions across files.
        """
        # First, build a map of all functions/methods in the project
        function_map: Dict[str, str] = {}  # function_name -> file_path

        for file_path, code_file in project.files.items():
            # Top-level functions
            for func in code_file.functions:
                function_map[func.name] = file_path

            # Class methods
            for cls in code_file.classes:
                for method in cls.methods:
                    # Store as ClassName.method_name
                    full_name = f"{cls.name}.{method.name}"
                    function_map[full_name] = file_path

        # Now find calls within each file
        for file_path, code_file in project.files.items():
            # Analyze function bodies for calls
            for func in code_file.functions:
                self._find_calls_in_function(
                    project, file_path, func.name, func, function_map
                )

            # Analyze class methods
            for cls in code_file.classes:
                for method in cls.methods:
                    caller_name = f"{cls.name}.{method.name}"
                    self._find_calls_in_function(
                        project, file_path, caller_name, method, function_map
                    )

    def _find_calls_in_function(
        self,
        project: CodebaseProject,
        caller_file: str,
        caller_func: str,
        func: CodeFunction,
        function_map: Dict[str, str]
    ) -> None:
        """Find function calls within a function body"""
        # Look for calls in the function's calls list
        for call_name in func.calls:
            # Check if this call is to a known function in the project
            if call_name in function_map:
                callee_file = function_map[call_name]

                # Only track cross-file calls or significant internal calls
                edge = CallEdge(
                    caller_file=caller_file,
                    caller_function=caller_func,
                    callee_file=callee_file,
                    callee_function=call_name,
                    call_count=1
                )
                project.call_edges.append(edge)

            # Also check for method calls like "self.other_method"
            # or "ClassName.method"
            if '.' in call_name:
                parts = call_name.split('.')
                if len(parts) == 2:
                    # Could be Class.method or instance.method
                    class_or_instance, method = parts
                    # Try to find matching class method
                    full_method = f"{class_or_instance}.{method}"
                    if full_method in function_map:
                        callee_file = function_map[full_method]
                        edge = CallEdge(
                            caller_file=caller_file,
                            caller_function=caller_func,
                            callee_file=callee_file,
                            callee_function=full_method,
                            call_count=1
                        )
                        project.call_edges.append(edge)

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
        elif lang in ('javascript', 'typescript'):
            return TypeScriptParser.parse_file(file_path)
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
        Score code file importance using 9 signals + git-enhanced scoring

        Args:
            code_file: CodeFile object

        Returns:
            ImportanceScore
        """
        signals = ImportanceSignals()

        # Build text representation for scoring
        text = self._format_file_text(code_file)

        # Get git information for this file (cached)
        git_info = self._get_git_info(code_file.file_path)

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

        # 4. Authority score (git-enhanced: commit count + author diversity)
        # Files with many commits and multiple authors are likely important
        if git_info.commit_count > 0:
            # Commit count: 1-5 = 0.3, 5-20 = 0.6, 20+ = 0.9
            commit_score = min(1.0, git_info.commit_count / 25.0)
            # Author diversity: single author = 0.5, multiple = bonus
            author_score = min(1.0, 0.5 + git_info.author_count * 0.15)
            signals.authority_score = 0.6 * commit_score + 0.4 * author_score
        else:
            # No git history, use neutral score
            signals.authority_score = 0.5

        # 5. Recency score (git-enhanced: recently modified files are more relevant)
        # Score based on days since last modification
        if git_info.last_commit_date:
            days_since_modified = git_info.last_modified_days
            if days_since_modified < 7:
                # Modified in last week - very fresh
                signals.recency_score = 1.0
            elif days_since_modified < 30:
                # Modified in last month - fresh
                signals.recency_score = 0.8
            elif days_since_modified < 90:
                # Modified in last quarter
                signals.recency_score = 0.6
            elif days_since_modified < 365:
                # Modified in last year
                signals.recency_score = 0.4
            else:
                # Older than a year
                signals.recency_score = 0.2
        else:
            # No git history, use neutral score
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
        signals.custom_signals['git_commits'] = min(1.0, git_info.commit_count / 25.0)
        signals.custom_signals['git_authors'] = git_info.author_count
        signals.custom_signals['git_age_days'] = git_info.age_days

        # Combine signals (git signals now active)
        final_score = (
            0.15 * signals.length_score +
            0.10 * signals.technical_score +
            0.15 * signals.structural_score +
            0.15 * signals.authority_score +       # Git: commit count + author diversity
            0.15 * signals.recency_score +         # Git: days since last modification
            0.00 * signals.engagement_score +      # N/A for code
            0.10 * signals.reference_score +
            0.15 * signals.custom_signals.get('complexity', 0.5) +
            0.05 * signals.custom_signals.get('documentation', 0.0)
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

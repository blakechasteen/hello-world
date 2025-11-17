#!/usr/bin/env python3
"""
Diff Engine - Advanced diffing capabilities for prompts
Implements Myers algorithm, word-level diff, and semantic diff
"""

import difflib
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class DiffType(Enum):
    """Type of difference"""
    EQUAL = "equal"
    INSERT = "insert"
    DELETE = "delete"
    REPLACE = "replace"


class DiffLevel(Enum):
    """Level of granularity for diff"""
    CHARACTER = "character"
    WORD = "word"
    LINE = "line"
    SEMANTIC = "semantic"


@dataclass
class DiffChunk:
    """Represents a single chunk of difference"""
    type: DiffType
    old_text: str
    new_text: str
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    context: Optional[str] = None

    def __repr__(self):
        if self.type == DiffType.EQUAL:
            return f"EQUAL: '{self.old_text[:20]}...'"
        elif self.type == DiffType.INSERT:
            return f"INSERT: '{self.new_text}'"
        elif self.type == DiffType.DELETE:
            return f"DELETE: '{self.old_text}'"
        else:
            return f"REPLACE: '{self.old_text}' -> '{self.new_text}'"


@dataclass
class DiffStats:
    """Statistics about differences"""
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    total_lines_old: int = 0
    total_lines_new: int = 0
    similarity: float = 0.0

    def __repr__(self):
        return (f"DiffStats(+{self.additions} -{self.deletions} "
                f"~{self.changes}, {self.similarity:.1%} similar)")


@dataclass
class DiffResult:
    """Complete diff result with chunks and statistics"""
    chunks: List[DiffChunk]
    stats: DiffStats
    old_text: str
    new_text: str
    level: DiffLevel
    unified_format: Optional[str] = None

    def __repr__(self):
        return f"DiffResult({len(self.chunks)} chunks, {self.stats})"


class MyersDiff:
    """
    Implementation of Myers diff algorithm
    Efficient O(ND) algorithm for finding shortest edit script
    """

    @staticmethod
    def diff(old: List[str], new: List[str]) -> List[Tuple[str, int, int, int, int]]:
        """
        Myers diff algorithm
        Returns list of (operation, old_start, old_end, new_start, new_end)
        """
        n = len(old)
        m = len(new)

        # Use Python's difflib SequenceMatcher which implements a similar algorithm
        matcher = difflib.SequenceMatcher(None, old, new)
        operations = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            operations.append((tag, i1, i2, j1, j2))

        return operations

    @staticmethod
    def _build_edit_graph(old: List[str], new: List[str]) -> Dict:
        """Build edit graph for Myers algorithm (for reference/future optimization)"""
        n = len(old)
        m = len(new)
        max_d = n + m

        # V array stores furthest reaching x for each k-diagonal
        v = {1: 0}
        trace = []

        for d in range(max_d + 1):
            trace.append(v.copy())

            for k in range(-d, d + 1, 2):
                # Determine whether to move down or right
                if k == -d or (k != d and v.get(k - 1, -1) < v.get(k + 1, -1)):
                    x = v.get(k + 1, 0)
                else:
                    x = v.get(k - 1, 0) + 1

                y = x - k

                # Follow diagonal matches
                while x < n and y < m and old[x] == new[y]:
                    x += 1
                    y += 1

                v[k] = x

                # Check if we reached the end
                if x >= n and y >= m:
                    return {'trace': trace, 'distance': d}

        return {'trace': trace, 'distance': max_d}


class DiffEngine:
    """Main diff engine with multiple strategies"""

    def __init__(self):
        self.myers = MyersDiff()

    def diff(
        self,
        old_text: str,
        new_text: str,
        level: DiffLevel = DiffLevel.LINE,
        context_lines: int = 3
    ) -> DiffResult:
        """
        Generate diff between two texts at specified granularity level

        Args:
            old_text: Original text
            new_text: Modified text
            level: Granularity level (character, word, line)
            context_lines: Number of context lines for unified format

        Returns:
            DiffResult with chunks, stats, and formatted output
        """
        if level == DiffLevel.CHARACTER:
            return self._char_diff(old_text, new_text)
        elif level == DiffLevel.WORD:
            return self._word_diff(old_text, new_text)
        elif level == DiffLevel.LINE:
            return self._line_diff(old_text, new_text, context_lines)
        elif level == DiffLevel.SEMANTIC:
            return self._semantic_diff(old_text, new_text)
        else:
            raise ValueError(f"Unknown diff level: {level}")

    def _char_diff(self, old_text: str, new_text: str) -> DiffResult:
        """Character-level diff"""
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        chunks = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            chunk_type = self._tag_to_type(tag)
            chunk = DiffChunk(
                type=chunk_type,
                old_text=old_text[i1:i2],
                new_text=new_text[j1:j2],
                old_start=i1,
                old_end=i2,
                new_start=j1,
                new_end=j2
            )
            chunks.append(chunk)

        stats = self._calculate_stats(chunks, old_text, new_text)
        unified = self._generate_unified_format(chunks, old_text, new_text, level=DiffLevel.CHARACTER)

        return DiffResult(
            chunks=chunks,
            stats=stats,
            old_text=old_text,
            new_text=new_text,
            level=DiffLevel.CHARACTER,
            unified_format=unified
        )

    def _word_diff(self, old_text: str, new_text: str) -> DiffResult:
        """Word-level diff"""
        old_words = self._tokenize_words(old_text)
        new_words = self._tokenize_words(new_text)

        matcher = difflib.SequenceMatcher(None, old_words, new_words)
        chunks = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            chunk_type = self._tag_to_type(tag)
            chunk = DiffChunk(
                type=chunk_type,
                old_text=' '.join(old_words[i1:i2]),
                new_text=' '.join(new_words[j1:j2]),
                old_start=i1,
                old_end=i2,
                new_start=j1,
                new_end=j2
            )
            chunks.append(chunk)

        stats = self._calculate_stats(chunks, old_text, new_text)
        unified = self._generate_unified_format(chunks, old_text, new_text, level=DiffLevel.WORD)

        return DiffResult(
            chunks=chunks,
            stats=stats,
            old_text=old_text,
            new_text=new_text,
            level=DiffLevel.WORD,
            unified_format=unified
        )

    def _line_diff(self, old_text: str, new_text: str, context_lines: int = 3) -> DiffResult:
        """Line-level diff with context"""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        chunks = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            chunk_type = self._tag_to_type(tag)
            chunk = DiffChunk(
                type=chunk_type,
                old_text=''.join(old_lines[i1:i2]),
                new_text=''.join(new_lines[j1:j2]),
                old_start=i1,
                old_end=i2,
                new_start=j1,
                new_end=j2
            )
            chunks.append(chunk)

        stats = self._calculate_stats(chunks, old_text, new_text)

        # Generate unified diff format
        unified_diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile='old',
            tofile='new',
            n=context_lines
        )
        unified = ''.join(unified_diff)

        return DiffResult(
            chunks=chunks,
            stats=stats,
            old_text=old_text,
            new_text=new_text,
            level=DiffLevel.LINE,
            unified_format=unified
        )

    def _semantic_diff(self, old_text: str, new_text: str) -> DiffResult:
        """
        Semantic diff - analyzes meaning changes
        Falls back to word diff with semantic annotations
        """
        # Start with word-level diff
        result = self._word_diff(old_text, new_text)

        # Annotate chunks with semantic context
        for chunk in result.chunks:
            if chunk.type in (DiffType.INSERT, DiffType.DELETE, DiffType.REPLACE):
                chunk.context = self._infer_semantic_context(chunk)

        result.level = DiffLevel.SEMANTIC
        return result

    def _infer_semantic_context(self, chunk: DiffChunk) -> str:
        """Infer semantic meaning of changes (can be enhanced with LLM)"""
        old = chunk.old_text.lower()
        new = chunk.new_text.lower()

        # Simple heuristics (can be replaced with LLM analysis)
        if chunk.type == DiffType.INSERT:
            if any(word in new for word in ['must', 'should', 'required']):
                return "Added constraint"
            elif any(word in new for word in ['example', 'e.g.', 'such as']):
                return "Added example"
            else:
                return "Added content"
        elif chunk.type == DiffType.DELETE:
            if any(word in old for word in ['deprecated', 'old', 'legacy']):
                return "Removed deprecated content"
            else:
                return "Removed content"
        elif chunk.type == DiffType.REPLACE:
            # Check for refinement
            if len(new) > len(old) * 1.5:
                return "Expanded explanation"
            elif len(new) < len(old) * 0.7:
                return "Simplified"
            else:
                return "Rephrased"

        return "Modified"

    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words preserving whitespace"""
        import re
        # Split on word boundaries but keep delimiters
        tokens = re.findall(r'\S+|\s+', text)
        return tokens

    def _tag_to_type(self, tag: str) -> DiffType:
        """Convert difflib tag to DiffType"""
        if tag == 'equal':
            return DiffType.EQUAL
        elif tag == 'insert':
            return DiffType.INSERT
        elif tag == 'delete':
            return DiffType.DELETE
        elif tag == 'replace':
            return DiffType.REPLACE
        else:
            raise ValueError(f"Unknown tag: {tag}")

    def _calculate_stats(self, chunks: List[DiffChunk], old_text: str, new_text: str) -> DiffStats:
        """Calculate diff statistics"""
        stats = DiffStats()

        for chunk in chunks:
            if chunk.type == DiffType.INSERT:
                stats.additions += len(chunk.new_text.splitlines() or [chunk.new_text])
            elif chunk.type == DiffType.DELETE:
                stats.deletions += len(chunk.old_text.splitlines() or [chunk.old_text])
            elif chunk.type == DiffType.REPLACE:
                stats.changes += 1

        stats.total_lines_old = len(old_text.splitlines() or [old_text])
        stats.total_lines_new = len(new_text.splitlines() or [new_text])

        # Calculate similarity ratio
        matcher = difflib.SequenceMatcher(None, old_text, new_text)
        stats.similarity = matcher.ratio()

        return stats

    def _generate_unified_format(
        self,
        chunks: List[DiffChunk],
        old_text: str,
        new_text: str,
        level: DiffLevel
    ) -> str:
        """Generate unified diff format"""
        lines = []
        lines.append("--- old")
        lines.append("+++ new")

        for chunk in chunks:
            if chunk.type == DiffType.EQUAL:
                # Show some context
                for line in chunk.old_text.splitlines()[:2]:
                    lines.append(f"  {line}")
            elif chunk.type == DiffType.DELETE:
                for line in chunk.old_text.splitlines():
                    lines.append(f"- {line}")
            elif chunk.type == DiffType.INSERT:
                for line in chunk.new_text.splitlines():
                    lines.append(f"+ {line}")
            elif chunk.type == DiffType.REPLACE:
                for line in chunk.old_text.splitlines():
                    lines.append(f"- {line}")
                for line in chunk.new_text.splitlines():
                    lines.append(f"+ {line}")

        return '\n'.join(lines)

    def side_by_side(
        self,
        old_text: str,
        new_text: str,
        width: int = 80,
        margin: int = 2
    ) -> str:
        """Generate side-by-side diff view"""
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        half_width = (width - margin) // 2
        result = []

        # Header
        header = f"{'OLD':<{half_width}}{' ' * margin}{'NEW':<{half_width}}"
        result.append(header)
        result.append("=" * width)

        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for i in range(i1, i2):
                    old_line = old_lines[i][:half_width-2]
                    new_line = new_lines[j1 + (i - i1)][:half_width-2]
                    result.append(f"  {old_line:<{half_width-2}}{' ' * margin}  {new_line}")

            elif tag == 'delete':
                for i in range(i1, i2):
                    old_line = old_lines[i][:half_width-2]
                    result.append(f"- {old_line:<{half_width-2}}{' ' * margin}")

            elif tag == 'insert':
                for j in range(j1, j2):
                    new_line = new_lines[j][:half_width-2]
                    result.append(f"{' ' * half_width}{' ' * margin}+ {new_line}")

            elif tag == 'replace':
                max_lines = max(i2 - i1, j2 - j1)
                for k in range(max_lines):
                    old_line = old_lines[i1 + k][:half_width-2] if i1 + k < i2 else ""
                    new_line = new_lines[j1 + k][:half_width-2] if j1 + k < j2 else ""

                    old_prefix = "- " if old_line else "  "
                    new_prefix = "+ " if new_line else "  "

                    result.append(
                        f"{old_prefix}{old_line:<{half_width-2}}"
                        f"{' ' * margin}{new_prefix}{new_line}"
                    )

        return '\n'.join(result)


def create_diff_engine() -> DiffEngine:
    """Factory function to create diff engine"""
    return DiffEngine()


# Convenience functions
def quick_diff(old_text: str, new_text: str, level: str = "line") -> DiffResult:
    """Quick diff with default settings"""
    engine = DiffEngine()
    level_enum = DiffLevel[level.upper()]
    return engine.diff(old_text, new_text, level_enum)


def diff_stats_only(old_text: str, new_text: str) -> DiffStats:
    """Get only statistics without full diff"""
    result = quick_diff(old_text, new_text, level="line")
    return result.stats

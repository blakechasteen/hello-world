#!/usr/bin/env python3
"""
Hallucination Detector
======================
Detects when AI-generated code references non-existent functions, classes, or APIs.

Features:
- Reference extraction from code (function calls, imports, class usage)
- Existence verification against indexed codebase
- Similarity-based suggestions ("Did you mean X?")
- Multi-language support (Python, TypeScript, JavaScript)
- Confidence scoring for each detection

Usage:
    from HoloLoom.agentic.hallucination_detector import HallucinationDetector

    detector = HallucinationDetector(indexer=codebase_indexer)
    hallucinations = await detector.detect(code, language="python")

    for h in hallucinations:
        print(f"Line {h.line}: {h.reference} does not exist")
        print(f"  Suggestions: {h.suggestions}")
"""

import ast
import re
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from HoloLoom.agentic.codebase_ingestion import CodebaseIndexer, EntityType, Language


logger = logging.getLogger(__name__)


class HallucinationType(str, Enum):
    """Types of hallucinations we detect."""
    FUNCTION_NOT_FOUND = "function_not_found"
    CLASS_NOT_FOUND = "class_not_found"
    IMPORT_NOT_FOUND = "import_not_found"
    METHOD_NOT_FOUND = "method_not_found"
    ATTRIBUTE_NOT_FOUND = "attribute_not_found"
    WRONG_SIGNATURE = "wrong_signature"  # Function exists but wrong params
    WRONG_TYPE = "wrong_type"  # Using function as class or vice versa


@dataclass
class CodeReference:
    """Represents a reference to an entity in code."""
    name: str
    reference_type: str  # "function_call", "class_usage", "import", "attribute"
    line_number: int
    context: str  # Surrounding code for context
    expected_type: Optional[EntityType] = None


@dataclass
class Hallucination:
    """Represents a detected hallucination."""
    reference: str
    hallucination_type: HallucinationType
    line_number: int
    context: str
    confidence: float  # 0.0-1.0 (how sure we are it's a hallucination)
    reason: str
    suggestions: List[str]  # Suggested real entities
    fix_suggestion: Optional[str] = None  # Suggested code fix


class ReferenceExtractor:
    """Base class for extracting references from code."""

    def extract_references(self, code: str, file_path: str = "temp.py") -> List[CodeReference]:
        """Extract all references from code."""
        raise NotImplementedError


class PythonReferenceExtractor(ReferenceExtractor):
    """Extract references from Python code."""

    def extract_references(self, code: str, file_path: str = "temp.py") -> List[CodeReference]:
        """Extract function calls, class usage, imports, attributes."""
        references = []

        try:
            tree = ast.parse(code, filename=file_path)
        except SyntaxError as e:
            logger.warning(f"Syntax error while extracting references: {e}")
            return references

        for node in ast.walk(tree):
            # Function calls
            if isinstance(node, ast.Call):
                ref = self._extract_call(node, code)
                if ref:
                    references.append(ref)

            # Class instantiation
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Could be class usage - check if capitalized (convention)
                if node.id[0].isupper():
                    references.append(CodeReference(
                        name=node.id,
                        reference_type="class_usage",
                        line_number=node.lineno,
                        context=self._get_context(code, node.lineno),
                        expected_type=EntityType.CLASS
                    ))

            # Imports
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    references.append(CodeReference(
                        name=alias.name,
                        reference_type="import",
                        line_number=node.lineno,
                        context=self._get_context(code, node.lineno),
                        expected_type=EntityType.MODULE
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    references.append(CodeReference(
                        name=full_name,
                        reference_type="import",
                        line_number=node.lineno,
                        context=self._get_context(code, node.lineno),
                        expected_type=None  # Could be function, class, or module
                    ))

            # Attribute access
            elif isinstance(node, ast.Attribute):
                references.append(CodeReference(
                    name=node.attr,
                    reference_type="attribute",
                    line_number=node.lineno,
                    context=self._get_context(code, node.lineno),
                    expected_type=None
                ))

        return references

    def _extract_call(self, node: ast.Call, code: str) -> Optional[CodeReference]:
        """Extract function call reference."""
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name:
            return CodeReference(
                name=func_name,
                reference_type="function_call",
                line_number=node.lineno,
                context=self._get_context(code, node.lineno),
                expected_type=EntityType.FUNCTION
            )

        return None

    def _get_context(self, code: str, line_number: int, window: int = 2) -> str:
        """Get surrounding lines for context."""
        lines = code.split('\n')
        start = max(0, line_number - window - 1)
        end = min(len(lines), line_number + window)
        return '\n'.join(lines[start:end])


class TypeScriptReferenceExtractor(ReferenceExtractor):
    """Extract references from TypeScript/JavaScript code."""

    def extract_references(self, code: str, file_path: str = "temp.ts") -> List[CodeReference]:
        """Extract references using regex patterns."""
        references = []

        # Function calls: functionName(...)
        func_call_pattern = r'(\w+)\s*\('
        for match in re.finditer(func_call_pattern, code):
            name = match.group(1)
            line_num = code[:match.start()].count('\n') + 1

            # Skip common keywords
            if name in ['if', 'for', 'while', 'switch', 'return', 'throw', 'new']:
                continue

            references.append(CodeReference(
                name=name,
                reference_type="function_call",
                line_number=line_num,
                context=self._get_context(code, line_num),
                expected_type=EntityType.FUNCTION
            ))

        # Class usage: new ClassName(...)
        class_pattern = r'new\s+(\w+)\s*\('
        for match in re.finditer(class_pattern, code):
            name = match.group(1)
            line_num = code[:match.start()].count('\n') + 1

            references.append(CodeReference(
                name=name,
                reference_type="class_usage",
                line_number=line_num,
                context=self._get_context(code, line_num),
                expected_type=EntityType.CLASS
            ))

        # Imports: import { X } from 'module'
        import_pattern = r'import\s+\{([^}]+)\}\s+from'
        for match in re.finditer(import_pattern, code):
            imports = match.group(1).split(',')
            line_num = code[:match.start()].count('\n') + 1

            for imp in imports:
                name = imp.strip()
                references.append(CodeReference(
                    name=name,
                    reference_type="import",
                    line_number=line_num,
                    context=self._get_context(code, line_num),
                    expected_type=None
                ))

        # Method calls: object.method(...)
        method_pattern = r'\.(\w+)\s*\('
        for match in re.finditer(method_pattern, code):
            name = match.group(1)
            line_num = code[:match.start()].count('\n') + 1

            references.append(CodeReference(
                name=name,
                reference_type="method_call",
                line_number=line_num,
                context=self._get_context(code, line_num),
                expected_type=EntityType.METHOD
            ))

        return references

    def _get_context(self, code: str, line_number: int, window: int = 2) -> str:
        """Get surrounding lines for context."""
        lines = code.split('\n')
        start = max(0, line_number - window - 1)
        end = min(len(lines), line_number + window)
        return '\n'.join(lines[start:end])


class HallucinationDetector:
    """
    Detects hallucinations in AI-generated code.

    Uses indexed codebase to verify that referenced entities actually exist.
    """

    def __init__(self, indexer: CodebaseIndexer):
        """
        Initialize detector with codebase indexer.

        Args:
            indexer: CodebaseIndexer with indexed codebase
        """
        self.indexer = indexer
        self.extractors = {
            Language.PYTHON: PythonReferenceExtractor(),
            Language.TYPESCRIPT: TypeScriptReferenceExtractor(),
            Language.JAVASCRIPT: TypeScriptReferenceExtractor(),
        }

        # Common built-in functions/classes to ignore
        self.python_builtins = set([
            'print', 'len', 'range', 'str', 'int', 'float', 'dict', 'list', 'tuple',
            'set', 'open', 'isinstance', 'type', 'enumerate', 'zip', 'map', 'filter',
            'max', 'min', 'sum', 'sorted', 'reversed', 'any', 'all', 'abs', 'round',
            'Exception', 'ValueError', 'TypeError', 'KeyError', 'AttributeError',
            'IndexError', 'RuntimeError', 'StopIteration', 'None', 'True', 'False'
        ])

        self.js_builtins = set([
            'console', 'log', 'error', 'warn', 'Array', 'Object', 'String', 'Number',
            'Boolean', 'Date', 'Math', 'JSON', 'Promise', 'Error', 'parseInt',
            'parseFloat', 'setTimeout', 'setInterval', 'fetch', 'document', 'window'
        ])

    async def detect(
        self,
        code: str,
        language: Language,
        file_path: str = "temp",
        strict: bool = False
    ) -> List[Hallucination]:
        """
        Detect hallucinations in code.

        Args:
            code: Source code to analyze
            language: Programming language
            file_path: File path (for context)
            strict: If True, flag all unrecognized references (more false positives)

        Returns:
            List of detected hallucinations
        """
        # Extract references
        extractor = self.extractors.get(language)
        if not extractor:
            logger.warning(f"No extractor for language: {language}")
            return []

        references = extractor.extract_references(code, file_path)
        hallucinations = []

        # Check each reference
        for ref in references:
            # Skip builtins
            if self._is_builtin(ref.name, language):
                continue

            # Search in indexed codebase
            results = await self.indexer.search_entity(
                ref.name,
                entity_type=ref.expected_type,
                fuzzy=False  # Exact match first
            )

            if not results:
                # Not found - potential hallucination
                hallucination = await self._create_hallucination(ref, language, strict)
                if hallucination:
                    hallucinations.append(hallucination)

            elif ref.expected_type:
                # Found but check if type matches
                found_types = set(r['type'] for r in results)
                if ref.expected_type.value not in found_types:
                    hallucinations.append(Hallucination(
                        reference=ref.name,
                        hallucination_type=HallucinationType.WRONG_TYPE,
                        line_number=ref.line_number,
                        context=ref.context,
                        confidence=0.7,
                        reason=f"Entity '{ref.name}' exists but is not a {ref.expected_type.value}",
                        suggestions=[f"{r['name']} ({r['type']})" for r in results[:3]],
                        fix_suggestion=None
                    ))

        return hallucinations

    async def _create_hallucination(
        self,
        ref: CodeReference,
        language: Language,
        strict: bool
    ) -> Optional[Hallucination]:
        """Create hallucination object with suggestions."""
        # Find similar entities
        similar = await self.indexer.find_similar_entities(
            ref.name,
            entity_type=ref.expected_type,
            k=5
        )

        # Calculate confidence
        confidence = self._calculate_confidence(ref, similar, strict)

        if confidence < 0.5 and not strict:
            # Low confidence - might be external library or dynamic
            return None

        # Determine hallucination type
        halluc_type = self._determine_type(ref)

        # Generate suggestions
        suggestions = []
        fix_suggestion = None

        if similar:
            suggestions = [f"{name} (similarity: {score:.2f})" for name, score in similar]

            # If we have a high-similarity match, suggest a fix
            best_match, best_score = similar[0]
            if best_score > 0.7:
                fix_suggestion = self._generate_fix(ref, best_match, language)

        return Hallucination(
            reference=ref.name,
            hallucination_type=halluc_type,
            line_number=ref.line_number,
            context=ref.context,
            confidence=confidence,
            reason=f"{ref.reference_type.replace('_', ' ').title()} '{ref.name}' not found in codebase",
            suggestions=suggestions,
            fix_suggestion=fix_suggestion
        )

    def _is_builtin(self, name: str, language: Language) -> bool:
        """Check if name is a built-in function/class."""
        if language == Language.PYTHON:
            return name in self.python_builtins
        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            return name in self.js_builtins
        return False

    def _calculate_confidence(
        self,
        ref: CodeReference,
        similar: List[tuple],
        strict: bool
    ) -> float:
        """Calculate confidence that this is a hallucination."""
        base_confidence = 0.8 if strict else 0.6

        # Lower confidence if we have very similar entities
        if similar:
            best_similarity = similar[0][1]
            if best_similarity > 0.8:
                # Likely a typo, not hallucination
                base_confidence = 0.95
            elif best_similarity > 0.6:
                base_confidence = 0.7

        # Higher confidence for common patterns
        if ref.name.startswith('_'):
            # Private/internal - might be dynamic
            base_confidence *= 0.7

        return min(base_confidence, 1.0)

    def _determine_type(self, ref: CodeReference) -> HallucinationType:
        """Determine hallucination type from reference."""
        type_map = {
            "function_call": HallucinationType.FUNCTION_NOT_FOUND,
            "class_usage": HallucinationType.CLASS_NOT_FOUND,
            "import": HallucinationType.IMPORT_NOT_FOUND,
            "method_call": HallucinationType.METHOD_NOT_FOUND,
            "attribute": HallucinationType.ATTRIBUTE_NOT_FOUND
        }
        return type_map.get(ref.reference_type, HallucinationType.FUNCTION_NOT_FOUND)

    def _generate_fix(self, ref: CodeReference, suggestion: str, language: Language) -> str:
        """Generate a code fix suggestion."""
        if language == Language.PYTHON:
            if ref.reference_type == "function_call":
                return f"Replace '{ref.name}' with '{suggestion}'"
            elif ref.reference_type == "class_usage":
                return f"Use '{suggestion}' instead of '{ref.name}'"
            elif ref.reference_type == "import":
                return f"Change import to: from ... import {suggestion}"

        elif language in [Language.TYPESCRIPT, Language.JAVASCRIPT]:
            if ref.reference_type == "function_call":
                return f"Replace '{ref.name}()' with '{suggestion}()'"
            elif ref.reference_type == "class_usage":
                return f"Change 'new {ref.name}' to 'new {suggestion}'"

        return f"Use '{suggestion}' instead of '{ref.name}'"

    async def detect_and_explain(
        self,
        code: str,
        language: Language,
        file_path: str = "temp"
    ) -> Dict[str, Any]:
        """
        Detect hallucinations and generate human-readable explanation.

        Returns:
            Dict with hallucinations, summary, and suggested fixes
        """
        hallucinations = await self.detect(code, language, file_path)

        summary = {
            "total_hallucinations": len(hallucinations),
            "by_type": {},
            "high_confidence": [],
            "suggested_fixes": []
        }

        for h in hallucinations:
            # Count by type
            h_type = h.hallucination_type.value
            summary["by_type"][h_type] = summary["by_type"].get(h_type, 0) + 1

            # High confidence (>0.8)
            if h.confidence > 0.8:
                summary["high_confidence"].append({
                    "line": h.line_number,
                    "reference": h.reference,
                    "type": h_type,
                    "suggestions": h.suggestions
                })

            # Fixes available
            if h.fix_suggestion:
                summary["suggested_fixes"].append({
                    "line": h.line_number,
                    "reference": h.reference,
                    "fix": h.fix_suggestion
                })

        return {
            "hallucinations": [
                {
                    "reference": h.reference,
                    "type": h.hallucination_type.value,
                    "line": h.line_number,
                    "confidence": h.confidence,
                    "reason": h.reason,
                    "suggestions": h.suggestions,
                    "fix": h.fix_suggestion,
                    "context": h.context
                }
                for h in hallucinations
            ],
            "summary": summary
        }

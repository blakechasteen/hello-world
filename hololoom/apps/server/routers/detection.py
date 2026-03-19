"""
Detection Router
================

Code analysis and detection endpoints for AI slop, logic errors, and hallucinations.
Extracted from agentic_api.py as part of W2 (Monolithic Files) SWOT remediation.

Endpoints:
- POST /detect/slop - Comprehensive AI slop detection
- POST /detect/logic - ML-based logic error detection
- POST /detect/hallucinations - Hallucination detection
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Detection"])


# ============================================================================
# Dependencies
# ============================================================================

def get_server_state(request: Request):
    """
    Dependency to get server state from the app.

    The state is attached to the app during startup in agentic_api.py.
    """
    return request.app.state.server_state


def get_code_language_enum(language: str):
    """
    Map language string to CodeLanguage enum.

    Args:
        language: Language string (python, typescript, javascript)

    Returns:
        CodeLanguage enum value

    Raises:
        HTTPException: If language is not supported
    """
    try:
        from trough.detector import CodeLanguage
    except ImportError:
        # Fallback if trough not available
        raise HTTPException(
            status_code=500,
            detail="Code analysis module not available"
        )

    lang_map = {
        "python": CodeLanguage.PYTHON,
        "typescript": CodeLanguage.TYPESCRIPT,
        "javascript": CodeLanguage.JAVASCRIPT
    }

    lang_enum = lang_map.get(language.lower())
    if not lang_enum:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language: {language}. Supported: python, typescript, javascript"
        )

    return lang_enum


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/detect/slop")
async def detect_ai_slop(
    code: str,
    language: str,
    file_path: str = "temp",
    state=Depends(get_server_state)
):
    """
    Comprehensive AI slop detection (all common pitfalls).

    Detects:
    - Hallucinations (non-existent functions/classes)
    - Missing error handling
    - Hardcoded secrets/magic numbers
    - Resource leaks
    - Security issues (SQL injection, XSS, command injection)
    - Performance anti-patterns
    - Dead code
    - Naming inconsistencies
    - Missing documentation
    - Incomplete code (TODO, pass statements)
    - Off-by-one errors
    - Timezone issues
    - Copy-paste errors

    Args:
        code: Source code to analyze
        language: Programming language
        file_path: File path for context

    Returns:
        Comprehensive list of all detected issues with fixes

    Example:
        POST /detect/slop
        {
          "code": "def process(data):\\n    result = fetch_data()\\n    return result",
          "language": "python",
          "file_path": "process.py"
        }
    """
    try:
        if not state.ai_slop_detector:
            raise HTTPException(status_code=500, detail="AI slop detector not initialized")

        lang_enum = get_code_language_enum(language)

        # Detect all issues
        issues = await state.ai_slop_detector.detect_all(code, lang_enum, file_path)
        summary = await state.ai_slop_detector.get_summary(issues)

        return {
            "success": True,
            "language": language,
            "total_issues": len(issues),
            "summary": summary,
            "issues": [
                {
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "line": issue.line_number,
                    "column": issue.column,
                    "description": issue.description,
                    "context": issue.context[:200],  # Truncate context
                    "fix": issue.fix_suggestion,
                    "confidence": issue.confidence
                }
                for issue in issues
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI slop detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/logic")
async def detect_logic_errors(
    code: str,
    language: str,
    file_path: str = "temp",
    state=Depends(get_server_state)
):
    """
    ML-based logic error detection.

    Detects subtle logic errors that pattern matching can't catch:
    - Infinite loops (loops with no exit condition)
    - Unreachable code (code after return/break/continue)
    - Logic contradictions (if x and not x)
    - Null/None dereference
    - Division by zero
    - Array out of bounds
    - Missing return statements
    - Constant conditions (always true/false)
    - Wrong operators (= instead of ==)

    Uses hybrid approach:
    - Control flow graph analysis
    - Abstract interpretation for value tracking
    - Symbolic execution for proofs
    - ML model for pattern recognition (future)

    Args:
        code: Source code to analyze
        language: Programming language (python, typescript, javascript)
        file_path: File path for context

    Returns:
        List of logic errors with confidence scores and proofs

    Example:
        POST /detect/logic
        {
          "code": "def divide(a, b):\\n    return a / b",
          "language": "python",
          "file_path": "math_utils.py"
        }

        Response:
        {
          "success": true,
          "language": "python",
          "total_errors": 1,
          "summary": {
            "total_errors": 1,
            "by_type": {"division_by_zero": 1},
            "high_confidence": [
              {
                "type": "division_by_zero",
                "line": 2,
                "description": "Potential division by zero - b not checked"
              }
            ],
            "proven": []
          },
          "errors": [...]
        }
    """
    try:
        if not state.ml_logic_detector:
            raise HTTPException(status_code=500, detail="ML logic detector not initialized")

        lang_enum = get_code_language_enum(language)

        # Detect logic errors
        errors = await state.ml_logic_detector.detect(code, lang_enum, file_path)
        summary = await state.ml_logic_detector.get_summary(errors)

        return {
            "success": True,
            "language": language,
            "total_errors": len(errors),
            "summary": summary,
            "errors": [
                {
                    "type": error.error_type.value,
                    "line": error.line_number,
                    "column": error.column,
                    "description": error.description,
                    "context": error.context[:200],  # Truncate context
                    "confidence": error.confidence,
                    "fix": error.fix_suggestion,
                    "proof": error.proof
                }
                for error in errors
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML logic detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/hallucinations")
async def detect_hallucinations(
    code: str,
    language: str,
    file_path: str = "temp",
    strict: bool = False,
    state=Depends(get_server_state)
):
    """
    Detect hallucinations in AI-generated code.

    Args:
        code: Source code to analyze
        language: Programming language
        file_path: File path for context
        strict: If True, flag all unrecognized references

    Returns:
        List of hallucinations with suggestions

    Example:
        POST /detect/hallucinations
        {
          "code": "def main():\\n    result = nonexistent_function()\\n    return result",
          "language": "python",
          "strict": false
        }
    """
    try:
        if not state.hallucination_detector:
            raise HTTPException(status_code=500, detail="Hallucination detector not initialized")

        lang_enum = get_code_language_enum(language)

        result = await state.hallucination_detector.detect_and_explain(
            code,
            lang_enum,
            file_path
        )

        return {
            "success": True,
            "language": language,
            "hallucinations": result["hallucinations"],
            "summary": result["summary"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hallucination detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

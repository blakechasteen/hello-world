#!/usr/bin/env python3
"""
Squad FastAPI Server (Enhanced)
================================
Exposes HoloLoom's agentic reasoning + LLM code generation as REST API.

Features:
- Full code generation capabilities (generate, refactor, fix, test, review, explain)
- Multi-provider LLM support (Ollama/qwen2.5-coder, Anthropic, OpenAI)
- Modular, extensible architecture
- Safety alignment integration

Author: Claude Code
Date: November 16, 2025
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# HoloLoom imports
from HoloLoom.config import Config
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.alignment.audit_trail import AuditTrail

# Squad modules
from llm_providers import LLMClient, LLMProvider
from code_generator import CodeGenerationEngine, CodeTask, CodeContext as GenCodeContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Request/Response Models
# ============================================================================

class CodeContext(BaseModel):
    """Code context from VS Code"""
    currentFile: Optional[str] = None
    fileName: Optional[str] = None
    languageId: Optional[str] = None
    selection: Optional[str] = None
    diagnostics: Optional[List[Dict]] = None
    workspace: Optional[str] = None


class QueryRequest(BaseModel):
    """Query request from VS Code"""
    text: str
    context: Optional[CodeContext] = None
    mode: str = "verify"  # direct, verify, research, plan_execute
    max_steps: int = 5


class ChatRequest(BaseModel):
    """Chat request from VS Code"""
    message: str
    context: Optional[CodeContext] = None


class CodeGenerationRequest(BaseModel):
    """Code generation request"""
    description: str
    language: Optional[str] = None
    context: Optional[CodeContext] = None


class CodeRefactorRequest(BaseModel):
    """Code refactoring request"""
    code: str
    instructions: str
    language: Optional[str] = None


class CodeFixRequest(BaseModel):
    """Code fix request"""
    code: str
    error_message: Optional[str] = None
    diagnostics: Optional[List[Dict]] = None
    language: Optional[str] = None


class CodeTestRequest(BaseModel):
    """Test generation request"""
    code: str
    language: Optional[str] = None
    test_framework: Optional[str] = None


class CodeReviewRequest(BaseModel):
    """Code review request"""
    code: str
    language: Optional[str] = None


class CodeExplainRequest(BaseModel):
    """Code explanation request"""
    code: str
    language: Optional[str] = None
    question: Optional[str] = None


class ReasoningStep(BaseModel):
    """Single reasoning step"""
    type: str
    query: Optional[str] = None
    confidence: Optional[float] = None
    finding: Optional[str] = None
    completed: bool = True


class VerificationResult(BaseModel):
    """Verification result"""
    verified: bool
    confidence: float
    contradictions: List[str]
    supporting_evidence: List[str]
    suggested_refinements: List[str]


class QueryResponse(BaseModel):
    """Query response to VS Code"""
    response: str
    confidence: float
    reasoning_mode: str
    steps_taken: List[ReasoningStep]
    total_queries: int
    total_duration_ms: float
    verification: Optional[VerificationResult] = None


class CodeGenerationResponse(BaseModel):
    """Code generation response"""
    code: str
    explanation: str
    confidence: float
    language: Optional[str] = None
    diff: Optional[str] = None
    task_type: str


# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="Squad Server (Enhanced)", version="0.2.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
llm_client: Optional[LLMClient] = None
code_engine: Optional[CodeGenerationEngine] = None
config: Optional[Config] = None
audit_trail: Optional[AuditTrail] = None


# ============================================================================
# Lifecycle
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize Squad server with LLM and HoloLoom"""
    global llm_client, code_engine, config, audit_trail

    logger.info("Starting Squad server (Enhanced)...")

    # Initialize LLM client (auto-selects best provider)
    llm_client = LLMClient()
    provider_info = llm_client.get_provider_info()
    logger.info(f"LLM Provider: {provider_info['provider']} ({provider_info['model']})")

    # Initialize code generation engine
    code_engine = CodeGenerationEngine(llm_client)
    logger.info("Code generation engine initialized")

    # Create config
    config = Config.fast()
    config.enable_alignment = True

    # Create audit trail
    audit_trail = AuditTrail()

    logger.info("Squad server ready! 🚀")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on server shutdown"""
    logger.info("Squad server stopped")


# ============================================================================
# API Endpoints - Core
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "llm_ready": llm_client is not None,
        "code_engine_ready": code_engine is not None,
        "llm_provider": llm_client.get_provider_info() if llm_client else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    return {
        "llm_ready": llm_client is not None,
        "code_engine_ready": code_engine is not None,
        "provider_info": llm_client.get_provider_info() if llm_client else None,
        "config_mode": config.mode.value if config else "unknown"
    }


# ============================================================================
# API Endpoints - Code Generation
# ============================================================================

@app.post("/generate", response_model=CodeGenerationResponse)
async def generate_code(request: CodeGenerationRequest):
    """
    Generate code from natural language description.

    This is the primary code generation endpoint.
    Uses LLM to create production-ready code from descriptions.
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        # Convert context if provided
        context = None
        if request.context:
            context = GenCodeContext(
                language=request.context.languageId,
                file_name=request.context.fileName,
                selection=request.context.selection,
                diagnostics=request.context.diagnostics,
                workspace=request.context.workspace
            )

        # Generate code
        logger.info(f"Generating code: {request.description[:50]}...")
        result = await code_engine.generate_code(
            description=request.description,
            language=request.language,
            context=context
        )

        return CodeGenerationResponse(
            code=result.code,
            explanation=result.explanation,
            confidence=result.confidence,
            language=result.language,
            task_type=result.task_type.value
        )

    except Exception as e:
        logger.error(f"Code generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refactor", response_model=CodeGenerationResponse)
async def refactor_code(request: CodeRefactorRequest):
    """
    Refactor existing code with instructions.

    Returns refactored code with unified diff showing changes.
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        logger.info(f"Refactoring: {request.instructions[:50]}...")
        result = await code_engine.refactor_code(
            code=request.code,
            instructions=request.instructions,
            language=request.language
        )

        return CodeGenerationResponse(
            code=result.code,
            explanation=result.explanation,
            confidence=result.confidence,
            language=result.language,
            diff=result.diff,
            task_type=result.task_type.value
        )

    except Exception as e:
        logger.error(f"Refactoring error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fix", response_model=CodeGenerationResponse)
async def fix_code(request: CodeFixRequest):
    """
    Fix buggy code based on error messages.

    Analyzes errors and provides targeted fixes.
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        logger.info(f"Fixing code: {request.error_message[:50] if request.error_message else 'auto-detect'}...")
        result = await code_engine.fix_code(
            code=request.code,
            error_message=request.error_message,
            diagnostics=request.diagnostics,
            language=request.language
        )

        return CodeGenerationResponse(
            code=result.code,
            explanation=result.explanation,
            confidence=result.confidence,
            language=result.language,
            task_type=result.task_type.value
        )

    except Exception as e:
        logger.error(f"Code fix error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tests", response_model=CodeGenerationResponse)
async def generate_tests(request: CodeTestRequest):
    """
    Generate comprehensive tests for code.

    Creates unit tests with edge case coverage.
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        logger.info(f"Generating tests for {request.language or 'unknown'} code...")
        result = await code_engine.generate_tests(
            code=request.code,
            language=request.language,
            test_framework=request.test_framework
        )

        return CodeGenerationResponse(
            code=result.code,
            explanation=result.explanation,
            confidence=result.confidence,
            language=result.language,
            task_type=result.task_type.value
        )

    except Exception as e:
        logger.error(f"Test generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review", response_model=CodeGenerationResponse)
async def review_code(request: CodeReviewRequest):
    """
    Review code for issues, improvements, and security.

    Returns comprehensive code review with suggestions.
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        logger.info(f"Reviewing {request.language or 'unknown'} code...")
        result = await code_engine.review_code(
            code=request.code,
            language=request.language
        )

        return CodeGenerationResponse(
            code=result.code,
            explanation=result.explanation,
            confidence=result.confidence,
            language=result.language,
            task_type=result.task_type.value
        )

    except Exception as e:
        logger.error(f"Code review error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=CodeGenerationResponse)
async def explain_code(request: CodeExplainRequest):
    """
    Explain what code does.

    Returns detailed explanation with step-by-step breakdown.
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        logger.info(f"Explaining {request.language or 'unknown'} code...")
        result = await code_engine.explain_code(
            code=request.code,
            language=request.language,
            question=request.question
        )

        return CodeGenerationResponse(
            code=result.code,  # Original code
            explanation=result.explanation,
            confidence=result.confidence,
            language=result.language,
            task_type=result.task_type.value
        )

    except Exception as e:
        logger.error(f"Code explanation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# API Endpoints - Legacy (for backward compatibility)
# ============================================================================

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    """
    Legacy query endpoint (kept for backward compatibility).

    For code-specific tasks, use the specialized endpoints instead:
    - /generate - Generate new code
    - /refactor - Refactor existing code
    - /fix - Fix bugs
    - /tests - Generate tests
    - /review - Code review
    - /explain - Explain code
    """
    if not code_engine:
        raise HTTPException(status_code=503, detail="Code engine not initialized")

    try:
        # Route to appropriate code generation endpoint based on query
        query_lower = request.text.lower()

        # Simple routing logic
        if any(word in query_lower for word in ["generate", "create", "write"]) and request.context and request.context.selection:
            # Generate based on selection
            result = await code_engine.generate_code(
                description=request.text,
                language=request.context.languageId,
                context=GenCodeContext(
                    language=request.context.languageId,
                    file_name=request.context.fileName,
                    selection=request.context.selection
                )
            )
            response_text = f"{result.code}\n\n{result.explanation}"
            confidence = result.confidence

        elif any(word in query_lower for word in ["explain", "what does"]):
            if request.context and request.context.selection:
                result = await code_engine.explain_code(
                    code=request.context.selection,
                    language=request.context.languageId,
                    question=request.text
                )
                response_text = result.explanation
                confidence = result.confidence
            else:
                response_text = "Please select code to explain."
                confidence = 0.5

        elif any(word in query_lower for word in ["fix", "debug", "error"]):
            if request.context and request.context.selection:
                result = await code_engine.fix_code(
                    code=request.context.selection,
                    error_message=request.text,
                    diagnostics=request.context.diagnostics,
                    language=request.context.languageId
                )
                response_text = f"{result.code}\n\n{result.explanation}"
                confidence = result.confidence
            else:
                response_text = "Please select code to fix."
                confidence = 0.5

        elif any(word in query_lower for word in ["test", "unit test"]):
            if request.context and request.context.selection:
                result = await code_engine.generate_tests(
                    code=request.context.selection,
                    language=request.context.languageId
                )
                response_text = f"{result.code}\n\n{result.explanation}"
                confidence = result.confidence
            else:
                response_text = "Please select code to test."
                confidence = 0.5

        elif any(word in query_lower for word in ["review", "check", "improve"]):
            if request.context and request.context.selection:
                result = await code_engine.review_code(
                    code=request.context.selection,
                    language=request.context.languageId
                )
                response_text = result.explanation
                confidence = result.confidence
            else:
                response_text = "Please select code to review."
                confidence = 0.5

        else:
            # General query - use LLM directly
            response_text = await llm_client.generate(
                prompt=request.text,
                system_prompt="You are a helpful AI coding assistant."
            )
            confidence = 0.7

        # Create response
        return QueryResponse(
            response=response_text,
            confidence=confidence,
            reasoning_mode=request.mode,
            steps_taken=[
                ReasoningStep(
                    type="query",
                    query=request.text,
                    confidence=confidence,
                    completed=True
                )
            ],
            total_queries=1,
            total_duration_ms=0.0,  # Not tracked in legacy endpoint
            verification=None
        )

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def handle_chat(request: ChatRequest):
    """
    Conversational chat endpoint.

    For quick questions without full code generation.
    """
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    try:
        response = await llm_client.generate(
            prompt=request.message,
            system_prompt="You are a helpful AI coding assistant."
        )

        return {
            "response": response,
            "confidence": 0.8
        }

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

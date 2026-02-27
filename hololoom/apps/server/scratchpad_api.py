#!/usr/bin/env python3
"""
Live Scratchpad API Server
===========================

FastAPI server for live transcription scratchpad system.

Endpoints:
- POST /api/scratchpad/transcribe - Transcribe audio and populate template
- GET /api/scratchpad/template/{domain} - Get template structure
- POST /api/scratchpad/update - Update field manually
- POST /api/scratchpad/finalize - Finalize and save to memory
- GET /api/scratchpad/state - Get current state

Author: Claude Code
Date: November 2025
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
import asyncio

from hololoom.spinningWheel.live_scratchpad import LiveScratchpad

# Try to import Whisper
try:
    from hololoom.spinningWheel.whisper_spinner import WhisperSpinner
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[ScratchpadAPI] Warning: Whisper not available. Install with: pip install openai-whisper")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Live Scratchpad API",
    description="Real-time audio transcription with domain-specific templates",
    version="1.0.0"
)

# CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
scratchpad = LiveScratchpad()
whisper = WhisperSpinner(model_size="base") if WHISPER_AVAILABLE else None


# ============================================================================
# Request/Response Models
# ============================================================================

class TranscribeRequest(BaseModel):
    domain: Optional[str] = None


class UpdateRequest(BaseModel):
    domain: str
    field_name: str
    value: Any


class FinalizeRequest(BaseModel):
    domain: str
    data: Dict[str, Any]


class StartRecordingRequest(BaseModel):
    domain: str
    auto_detect: bool = False


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Live Scratchpad API",
        "version": "1.0.0",
        "whisper_available": WHISPER_AVAILABLE,
        "domains": list(scratchpad.templates.keys())
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "whisper_available": WHISPER_AVAILABLE,
        "active_domain": scratchpad.active_domain
    }


@app.post("/api/scratchpad/start")
async def start_recording(request: StartRecordingRequest):
    """
    Start a new recording session.

    Initializes template for the specified domain.
    """
    try:
        await scratchpad.start_recording(
            domain=request.domain,
            auto_detect=request.auto_detect
        )

        return {
            "success": True,
            "domain": scratchpad.active_domain,
            "state": scratchpad.get_current_state()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/scratchpad/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    domain: Optional[str] = None
):
    """
    Transcribe audio file and populate template.

    Accepts audio file upload, transcribes with Whisper,
    and automatically populates domain template.
    """
    if not WHISPER_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Whisper not available. Install with: pip install openai-whisper"
        )

    # Save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
        content = await audio.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        # Transcribe with Whisper
        result = await whisper.spin(temp_path)

        if not result.success or not result.shards:
            raise HTTPException(status_code=500, detail="Transcription failed")

        # Get transcription text
        transcription = result.shards[0].text

        # Detect domain if not provided
        detected_domain = domain
        confidence = 1.0

        if not detected_domain:
            # Simple domain detection based on keywords
            detected_domain, confidence = _detect_domain_from_text(transcription)

        # Update scratchpad
        if not scratchpad.active_domain:
            await scratchpad.start_recording(detected_domain or 'budget')

        await scratchpad.update_from_transcription(transcription, detected_domain)

        # Return current state
        state = scratchpad.get_current_state()

        return {
            "success": True,
            "transcription": transcription,
            "detected_domain": detected_domain,
            "confidence": confidence,
            **state
        }

    finally:
        # Clean up temp file
        temp_path.unlink(missing_ok=True)


@app.get("/api/scratchpad/template/{domain}")
async def get_template(domain: str):
    """
    Get template structure for a domain.

    Returns field definitions, suggestions, and defaults.
    """
    if domain not in scratchpad.templates:
        raise HTTPException(
            status_code=404,
            detail=f"Domain '{domain}' not found. Available: {list(scratchpad.templates.keys())}"
        )

    template = scratchpad.templates[domain]

    return {
        "name": template.name,
        "fields": [
            {
                'name': f.name,
                'type': f.field_type,
                'required': f.required,
                'default': f.default,
                'suggestions': f.suggestions,
                'validation_pattern': f.validation_pattern
            }
            for f in template.fields
        ],
        "defaults": template.to_dict()
    }


@app.post("/api/scratchpad/update")
async def update_field(request: UpdateRequest):
    """
    Update a single field manually.

    Used when user edits fields in the UI.
    """
    # Ensure domain is active
    if scratchpad.active_domain != request.domain:
        await scratchpad.start_recording(request.domain)

    # Update field
    scratchpad.update_field(request.field_name, request.value)

    return {
        "success": True,
        "state": scratchpad.get_current_state()
    }


@app.post("/api/scratchpad/finalize")
async def finalize(request: FinalizeRequest):
    """
    Finalize and save to memory.

    Validates template, creates MemoryShard, and ingests into HoloLoom.
    """
    # Ensure domain matches
    if scratchpad.active_domain != request.domain:
        raise HTTPException(status_code=400, detail="Domain mismatch")

    # Update data
    scratchpad.current_data = request.data

    try:
        # Finalize (validates and creates shard)
        shard = await scratchpad.finalize()

        # Ingest into HoloLoom (if available)
        try:
            from hololoom.hololoom import hololoom

            async with HoloLoom() as loom:
                await loom.experience([shard])

            ingested = True
        except Exception as e:
            print(f"[ScratchpadAPI] Warning: Could not ingest into HoloLoom: {e}")
            ingested = False

        # Clear scratchpad
        scratchpad.clear()

        return {
            "success": True,
            "shard_id": shard.id,
            "ingested": ingested,
            "message": "Entry saved successfully!"
        }

    except ValueError as e:
        # Validation failed
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/scratchpad/state")
async def get_state():
    """
    Get current scratchpad state.

    Returns domain, data, transcription, and validation status.
    """
    return scratchpad.get_current_state()


@app.post("/api/scratchpad/clear")
async def clear():
    """Clear current scratchpad session"""
    scratchpad.clear()
    return {"success": True}


# ============================================================================
# Helper Functions
# ============================================================================

def _detect_domain_from_text(text: str) -> tuple[Optional[str], float]:
    """
    Detect domain from transcription text.

    Returns (domain, confidence) tuple.
    """
    text_lower = text.lower()

    # Domain keywords
    domain_keywords = {
        'recipe': ['recipe', 'cook', 'bake', 'ingredient', 'cup', 'tablespoon', 'oven'],
        'budget': ['spent', 'bought', 'dollar', 'budget', 'expense', 'cost'],
        'expense': ['receipt', 'vendor', 'reimbursable', 'invoice'],
        'time_tracking': ['worked', 'project', 'hour', 'billable', 'task'],
        'bee_inspection': ['hive', 'colony', 'bee', 'brood', 'queen', 'varroa'],
        'sop': ['procedure', 'protocol', 'step', 'safety', 'sop'],
    }

    # Count keyword matches
    scores = {}
    for domain, keywords in domain_keywords.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > 0:
            scores[domain] = matches / len(keywords)

    if not scores:
        return None, 0.0

    # Return domain with highest score
    best_domain = max(scores, key=scores.get)
    confidence = min(scores[best_domain] * 2, 1.0)  # Scale to 0-1

    return best_domain, confidence


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("[ScratchpadAPI] Starting server on http://localhost:8002")
    print(f"[ScratchpadAPI] Whisper available: {WHISPER_AVAILABLE}")
    print(f"[ScratchpadAPI] Available domains: {list(scratchpad.templates.keys())}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )

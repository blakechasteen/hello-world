"""
AirLLM GPU Server — OpenAI-Compatible Inference Endpoint
=========================================================

Single FastAPI app, instanced 3x (one per RTX 3080).
Each binds a different CUDA device via CUDA_VISIBLE_DEVICES.

Launch:
    CUDA_VISIBLE_DEVICES=0 GPU_ID=0 uvicorn gpu_server:app --port 8001
    CUDA_VISIBLE_DEVICES=1 GPU_ID=1 uvicorn gpu_server:app --port 8002
    CUDA_VISIBLE_DEVICES=2 GPU_ID=2 uvicorn gpu_server:app --port 8003

Or use launch.sh for all three.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Configuration from environment ─────────────────────────────

GPU_ID = int(os.environ.get("GPU_ID", "0"))
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-108B")
COMPRESSION = os.environ.get("COMPRESSION", "4bit")
PORT = int(os.environ.get("PORT", "8001"))


# ── Pydantic models (OpenAI chat completions schema) ──────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: list[ChatMessage]
    max_tokens: int = Field(default=8192, le=16384)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class HealthResponse(BaseModel):
    gpu_id: int
    model: str
    status: str          # "ready" | "busy" | "loading"
    compression: str
    uptime_s: float


class MetaResponse(BaseModel):
    gpu_id: int
    model: str
    compression: str
    max_context: int
    estimated_tok_per_s: float


# ── Global state ───────────────────────────────────────────────

_model = None
_tokenizer = None
_busy = asyncio.Lock()
_start_time = time.time()
_status = "loading"


# ── Lifespan: load model once at startup ──────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _status

    logger.info(f"GPU {GPU_ID}: Loading {MODEL_ID} ({COMPRESSION})...")

    try:
        from airllm import AutoModel

        _model = AutoModel.from_pretrained(
            MODEL_ID,
            compression=COMPRESSION,
            device="cuda:0",  # always cuda:0 — CUDA_VISIBLE_DEVICES remaps
        )

        # AirLLM models have their own tokenizer
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        except Exception:
            _tokenizer = None
            logger.warning("Tokenizer load failed — token counts will be estimated")

        _status = "ready"
        logger.info(f"GPU {GPU_ID}: Model ready.")
    except Exception as e:
        _status = f"error: {e}"
        logger.error(f"GPU {GPU_ID}: Failed to load model: {e}")

    yield

    logger.info(f"GPU {GPU_ID}: Shutting down.")


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title=f"AirLLM GPU Server (cuda:{GPU_ID})",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        gpu_id=GPU_ID,
        model=MODEL_ID,
        status="busy" if _busy.locked() else _status,
        compression=COMPRESSION,
        uptime_s=time.time() - _start_time,
    )


@app.get("/meta", response_model=MetaResponse)
async def meta():
    return MetaResponse(
        gpu_id=GPU_ID,
        model=MODEL_ID,
        compression=COMPRESSION,
        max_context=131072,
        estimated_tok_per_s=1.5,  # conservative for AirLLM 108B Q4
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(req: ChatRequest):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    # One request at a time per GPU (AirLLM streams layers sequentially)
    if _busy.locked():
        raise HTTPException(429, "GPU busy — try another server")

    async with _busy:
        t0 = time.time()

        # Build prompt from messages
        prompt = _build_prompt(req.messages)

        # Tokenize
        if _tokenizer is not None:
            input_ids = _tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=req.max_tokens,
            )
        else:
            # Fallback: let AirLLM handle tokenization
            input_ids = prompt

        # Run inference (this is the slow part — minutes per response)
        try:
            output = _model.generate(
                input_ids=input_ids if _tokenizer is None else input_ids.input_ids,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )

            # Decode
            if _tokenizer is not None:
                text = _tokenizer.decode(output[0], skip_special_tokens=True)
                # Strip the prompt from the output
                text = text[len(prompt):].strip()
                prompt_tokens = len(input_ids.input_ids[0])
                completion_tokens = len(output[0]) - prompt_tokens
            else:
                text = str(output)
                prompt_tokens = len(prompt.split())  # rough estimate
                completion_tokens = len(text.split())

        except Exception as e:
            logger.error(f"GPU {GPU_ID}: Inference error: {e}")
            raise HTTPException(500, f"Inference failed: {e}")

        elapsed = time.time() - t0
        logger.info(
            f"GPU {GPU_ID}: Generated {completion_tokens} tokens in {elapsed:.1f}s "
            f"({completion_tokens / max(elapsed, 0.1):.1f} tok/s)"
        )

        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatChoice(
                    message=ChatMessage(role="assistant", content=text),
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )


def _build_prompt(messages: list[ChatMessage]) -> str:
    """Build a chat prompt string from messages.

    Uses Qwen chat template format. Falls back to simple concatenation
    if template isn't available.
    """
    if _tokenizer is not None and hasattr(_tokenizer, "apply_chat_template"):
        return _tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Simple fallback
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"<|system|>\n{m.content}")
        elif m.role == "user":
            parts.append(f"<|user|>\n{m.content}")
        elif m.role == "assistant":
            parts.append(f"<|assistant|>\n{m.content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

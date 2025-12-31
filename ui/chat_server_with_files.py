"""
Persistent Chat Server with File Ingestion
===========================================

FastAPI server with:
- File upload and ingestion via SpinningWheel
- Direct Ollama chat
- localStorage persistence (client-side)
- Multiple input modes: text, file, URL

Run: python ui/chat_server_with_files.py
Then open: http://localhost:8080
"""

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import asyncio
import tempfile

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(title="Learner Chat with Files")

# Enable CORS for Ollama
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory backend
memory_backend = None

# Try to import HoloLoom components
try:
    from HoloLoom.spinningWheel.auto import spin
    from HoloLoom.config import Config, MemoryBackend
    from HoloLoom.memory.backend_factory import create_memory_backend
    SPINNERS_AVAILABLE = True
    MEMORY_AVAILABLE = True
    print("✓ SpinningWheel available")
    print("✓ Memory backend available")
except Exception as e:
    SPINNERS_AVAILABLE = False
    MEMORY_AVAILABLE = False
    print(f"✗ HoloLoom not available: {e}")

# Try to import Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("✓ Ollama client available")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("✗ Ollama client not available")


async def get_memory():
    """Get or create memory backend"""
    global memory_backend
    if memory_backend is None and MEMORY_AVAILABLE:
        try:
            config = Config.fast()
            config.memory_backend = MemoryBackend.HYBRID  # Auto-falls back to INMEMORY
            memory_backend = await create_memory_backend(config)
            print("✓ Memory backend initialized (HYBRID with fallback)")
        except Exception as e:
            print(f"✗ Memory backend failed: {e}")
    return memory_backend


@app.get("/", response_class=HTMLResponse)
async def get_chat():
    """Serve the enhanced chat interface"""
    html_path = Path(__file__).parent / "persistent_chat_with_files.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<h1>Creating interface...</h1>")


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest uploaded file via SpinningWheel and store in memory"""
    if not SPINNERS_AVAILABLE:
        return JSONResponse(
            {"error": "SpinningWheel not available"},
            status_code=500
        )

    try:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Get memory backend
        memory = await get_memory()

        # Spin the file (returns shards)
        shards = await spin(tmp_path, return_shards=True)

        # Store shards in memory backend
        if memory and hasattr(shards, '__iter__'):
            for shard in shards:
                if hasattr(shard, 'content'):
                    await memory.add_memory(
                        content=shard.content,
                        metadata={
                            "source": "file_upload",
                            "filename": file.filename,
                            "timestamp": shard.metadata.get("timestamp", "unknown")
                        }
                    )

        # Clean up
        Path(tmp_path).unlink()

        # Extract text from shards for display
        if hasattr(shards, '__iter__'):
            texts = [shard.content for shard in shards if hasattr(shard, 'content')]
            summary = f"Ingested {file.filename}: {len(texts)} segments extracted and stored in memory"
            content = "\n\n".join(texts[:3])  # First 3 segments
            if len(texts) > 3:
                content += f"\n\n... and {len(texts) - 3} more segments"
        else:
            summary = f"Ingested {file.filename}"
            content = str(shards)

        return {
            "success": True,
            "filename": file.filename,
            "summary": summary,
            "content": content,
            "memory_stored": memory is not None
        }

    except Exception as e:
        return JSONResponse(
            {"error": f"Ingestion failed: {str(e)}"},
            status_code=500
        )


@app.post("/api/ingest/url")
async def ingest_url(url: str = Form(...)):
    """Ingest URL via SpinningWheel website scraper and store in memory"""
    if not SPINNERS_AVAILABLE:
        return JSONResponse(
            {"error": "SpinningWheel not available"},
            status_code=500
        )

    try:
        # Use website spinner directly for proper URL scraping
        from HoloLoom.spinning_wheel.website import WebsiteSpinner, WebsiteSpinnerConfig

        config = WebsiteSpinnerConfig(
            extract_images=False,  # Text only for now
            timeout=15  # 15 second timeout for slow sites
        )
        spinner = WebsiteSpinner(config)

        # Scrape URL and get shards
        shards = await spinner.spin({'url': url})

        # Get memory backend
        memory = await get_memory()

        # Store shards in memory backend
        if memory and hasattr(shards, '__iter__'):
            for shard in shards:
                # Extract content from shard
                content_text = None
                if hasattr(shard, 'content'):
                    content_text = shard.content
                elif hasattr(shard, 'text'):
                    content_text = shard.text

                if content_text:
                    await memory.add_memory(
                        content=content_text,
                        metadata={
                            "source": "url_ingestion",
                            "url": url,
                            "timestamp": shard.metadata.get("timestamp", "unknown") if hasattr(shard, 'metadata') else "unknown"
                        }
                    )

        # Extract text from shards for display
        texts = []
        if hasattr(shards, '__iter__'):
            for shard in shards:
                if hasattr(shard, 'content') and shard.content:
                    texts.append(shard.content)
                elif hasattr(shard, 'text') and shard.text:
                    texts.append(shard.text)

        summary = f"Ingested URL: {len(texts)} segments extracted and stored in memory"
        content = "\n\n".join(texts[:3]) if texts else "(No content extracted - site may require JavaScript or block scraping)"
        if len(texts) > 3:
            content += f"\n\n... and {len(texts) - 3} more segments"

        return {
            "success": True,
            "url": url,
            "summary": summary,
            "content": content,
            "memory_stored": memory is not None
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            {"error": f"URL ingestion failed: {str(e)}"},
            status_code=500
        )


@app.post("/api/chat")
async def chat(message: str = Form(...)):
    """Chat with Ollama using memory context"""
    if not OLLAMA_AVAILABLE:
        return JSONResponse(
            {"error": "Ollama not available. Install with: pip install ollama"},
            status_code=500
        )

    try:
        # Get memory backend
        memory = await get_memory()

        # Retrieve relevant context from memory
        context_text = ""
        context_count = 0
        if memory:
            try:
                # Retrieve top 5 relevant memories
                memories = await memory.retrieve(message, limit=5)
                if memories:
                    context_parts = []
                    for mem in memories:
                        # Extract content from memory node
                        if hasattr(mem, 'content'):
                            context_parts.append(mem.content)
                        elif hasattr(mem, 'text'):
                            context_parts.append(mem.text)

                    if context_parts:
                        context_count = len(context_parts[:3])
                        context_text = "\n\nRelevant context from memory:\n" + "\n---\n".join(context_parts[:3])
            except Exception as e:
                print(f"Memory retrieval failed: {e}")

        # Build prompt with context
        prompt = message
        if context_text:
            prompt = f"{message}{context_text}"

        response = ollama.chat(
            model='deepseek-r1:8b',
            messages=[{'role': 'user', 'content': prompt}]
        )

        response_text = response['message']['content']

        # Include context indicator in response
        if context_count > 0:
            response_text = f"[Using {context_count} memory fragments]\n\n{response_text}"

        return {
            "success": True,
            "content": response_text,
            "memory_context_used": context_count > 0
        }

    except Exception as e:
        return JSONResponse(
            {"error": f"Chat failed: {str(e)}"},
            status_code=500
        )


@app.get("/api/status")
async def status():
    """Check server status"""
    return {
        "spinners_available": SPINNERS_AVAILABLE,
        "ollama_available": OLLAMA_AVAILABLE,
        "features": {
            "file_upload": SPINNERS_AVAILABLE,
            "url_ingestion": SPINNERS_AVAILABLE,
            "chat": OLLAMA_AVAILABLE
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("  Learner Chat with File Ingestion")
    print("="*60)
    print()
    print("Starting server...")
    print("Open: http://localhost:8080")
    print()
    print("Features:")
    print(f"  • File Upload: {'✓' if SPINNERS_AVAILABLE else '✗'}")
    print(f"  • URL Ingestion: {'✓' if SPINNERS_AVAILABLE else '✗'}")
    print(f"  • Chat: {'✓' if OLLAMA_AVAILABLE else '✗'}")
    print()
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8080)

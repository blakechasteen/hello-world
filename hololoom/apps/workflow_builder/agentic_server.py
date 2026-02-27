#!/usr/bin/env python3
"""
Agentic Web Dashboard - FastAPI Server
========================================

WebSocket-based server for real-time chat with agentic intelligence.

Features:
- 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
- LLM-activated intelligent search
- Persistent memory (Neo4j + Qdrant)
- Complete provenance tracking
- Real-time visualization of reasoning steps

Endpoints:
- GET / - Dashboard HTML
- WS /ws - WebSocket for real-time chat
- GET /api/threads - List all threads
- GET /api/thread/{id} - Get specific thread
- GET /api/status - Server status

Usage:
    python hololoom/web_dashboard/agentic_server.py

Then open: http://localhost:8001
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
import sys
import logging
import tempfile
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add ffmpeg to PATH for Whisper (if installed via winget)
ffmpeg_path = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
if ffmpeg_path.exists():
    for pkg in ffmpeg_path.glob("Gyan.FFmpeg*/ffmpeg-*/bin"):
        if pkg.exists():
            os.environ["PATH"] = str(pkg) + os.pathsep + os.environ["PATH"]
            logger.info(f"Added ffmpeg to PATH: {pkg}")
            break

from hololoom.agentic import create_agentic_orchestrator, ReasoningMode
from hololoom.config import Config, MemoryBackend
from hololoom.protocols.types import Query, MemoryShard
from hololoom.alignment.audit_trail import OutcomeType, AuditTrail
from hololoom.apps.workflow_builder.conversation_manager import ConversationManager
from hololoom.apps.workflow_builder.promptly_bridge import PromptlyBridge, PROMPTLY_AVAILABLE

# MCTS Agent Pool for background learning
from hololoom.agents.background_learner import create_agent_pool

# Multi-threaded conversation manager with breakthrough sharing
from hololoom.apps.workflow_builder.conversation_thread_manager import create_conversation_thread_manager

# Spinners for content ingestion
from hololoom.spinningWheel.youtube_spinner import YouTubeSpinner, TRANSCRIPT_API_AVAILABLE as YOUTUBE_AVAILABLE
from hololoom.spinningWheel.whisper_spinner import WhisperSpinner, WHISPER_AVAILABLE
from hololoom.spinningWheel.spreadsheet_spinner import SpreadsheetSpinner, PANDAS_AVAILABLE as SPREADSHEET_AVAILABLE
from hololoom.spinningWheel.pdf_spinner import PDFSpinner, PDF_AVAILABLE, PDFPLUMBER_AVAILABLE
from hololoom.spinningWheel.email_spinner import EmailSpinner, HTML_AVAILABLE as EMAIL_AVAILABLE
from hololoom.spinningWheel.codebase_spinner import CodebaseSpinner
from hololoom.spinningWheel.git_spinner import GitSpinner
from hololoom.spinningWheel.matrix_spinner import MatrixSpinner, MATRIX_AVAILABLE
from hololoom.spinningWheel.url_spinner import URLSpinner, WEB_AVAILABLE as URL_AVAILABLE

# Voice integration for conversational dashboard
from hololoom.apps.workflow_builder.voice_integration import create_voice_integration
from hololoom.apps.workflow_builder.voice_endpoints import add_voice_endpoints

# Active WebSocket connections
active_connections: List[WebSocket] = []

# Conversation tracking: WebSocket -> conversation_id
websocket_conversations: Dict[WebSocket, int] = {}

# Agentic orchestrator (initialized on startup)
agentic_orchestrator = None
orchestrator_config = None
conversation_manager = None
promptly_bridge = None
knowledge_shards = []  # Global shard storage for uploaded content

# Spinners for content ingestion
youtube_spinner = None
whisper_spinner = None
spreadsheet_spinner = None
pdf_spinner = None
email_spinner = None
codebase_spinner = None
git_spinner = None
matrix_spinner = None
url_spinner = None

# Agent pool for background MCTS learning
agent_pool = None

# Voice integration for conversational dashboard
voice_integration = None

# Guard flag to prevent double initialization
_lifespan_initialized = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agentic orchestrator on startup"""
    global agentic_orchestrator, orchestrator_config, conversation_manager, promptly_bridge, knowledge_shards, youtube_spinner, whisper_spinner, spreadsheet_spinner, pdf_spinner, email_spinner, codebase_spinner, git_spinner, matrix_spinner, url_spinner, agent_pool, voice_integration

    logger.info("="*60)
    logger.info("Starting HoloLoom Agentic Dashboard")
    logger.info("="*60)

    # Initialize conversation manager (persistent SQLite storage)
    conversation_manager = ConversationManager(db_path="./data/conversations.db")
    stats = conversation_manager.get_stats()
    logger.info(f"Conversation database initialized")
    logger.info(f"  - Total conversations: {stats['total_conversations']}")
    logger.info(f"  - Total messages: {stats['total_messages']}")
    logger.info(f"  - Database: {stats['db_path']}")

    # Initialize Promptly integration (if available)
    if PROMPTLY_AVAILABLE:
        try:
            promptly_bridge = PromptlyBridge()
            logger.info("Promptly integration enabled")
        except Exception as e:
            logger.warning(f"Promptly integration failed: {e}")
            promptly_bridge = None
    else:
        logger.info("Promptly not available (optional)")

    # Create configuration
    orchestrator_config = Config.fast()
    orchestrator_config.memory_backend = MemoryBackend.HYBRID  # Neo4j + Qdrant with fallback
    orchestrator_config.enable_safety_guardrails = False  # Disable strict safety for dashboard

    # Create initial knowledge shards (can be loaded from files)
    knowledge_shards = create_demo_shards()
    logger.info(f"Loaded {len(knowledge_shards)} knowledge shards")

    # Create permissive audit trail for dashboard (auto-approve read-only actions)
    permissive_audit = AuditTrail(persist_path=None)  # In-memory only for dashboard

    # Create agentic orchestrator with persistent memory
    try:
        agentic_orchestrator = await create_agentic_orchestrator(
            config=orchestrator_config,
            shards=knowledge_shards,
            enable_verification=True,
            enable_goal_tracking=True,
            audit_trail=permissive_audit
        )

        # Check LLM status
        llm_status = "available" if agentic_orchestrator.llm else "unavailable"
        logger.info(f"Agentic orchestrator initialized")
        logger.info(f"  - LLM status: {llm_status}")
        logger.info(f"  - Alignment: enabled")
        logger.info(f"  - Memory backend: {orchestrator_config.memory_backend.value}")
        logger.info(f"  - Verification: enabled")
        logger.info(f"  - Goal tracking: enabled")

    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Initialize spinners for content ingestion
    logger.info("Initializing content spinners...")

    # YouTube spinner
    if YOUTUBE_AVAILABLE:
        youtube_spinner = YouTubeSpinner(
            importance_threshold=0.2,  # Lower threshold for dashboard
            languages=['en', 'auto'],
            chunk_duration=60.0  # 1-minute chunks
        )
        logger.info("  - YouTube spinner: enabled")
    else:
        logger.info("  - YouTube spinner: unavailable (install youtube-transcript-api)")

    # Whisper spinner
    if WHISPER_AVAILABLE:
        whisper_spinner = WhisperSpinner(
            model_size="base",  # Fast model for dashboard
            importance_threshold=0.2
        )
        logger.info("  - Whisper spinner: enabled (model: base)")
    else:
        logger.info("  - Whisper spinner: unavailable (install openai-whisper)")

    # Spreadsheet spinner
    if SPREADSHEET_AVAILABLE:
        spreadsheet_spinner = SpreadsheetSpinner(
            importance_threshold=0.2,
            chunk_mode='sheet'  # One shard per sheet
        )
        logger.info("  - Spreadsheet spinner: enabled (Excel, CSV, TSV, ODS)")
    else:
        logger.info("  - Spreadsheet spinner: unavailable (install pandas openpyxl)")

    # PDF spinner
    if PDF_AVAILABLE or PDFPLUMBER_AVAILABLE:
        pdf_spinner = PDFSpinner(
            importance_threshold=0.2,
            enable_ocr=False  # OCR disabled by default for speed
        )
        logger.info("  - PDF spinner: enabled (PyPDF2/pdfplumber)")
    else:
        logger.info("  - PDF spinner: unavailable (install PyPDF2 or pdfplumber)")

    # Email spinner
    if EMAIL_AVAILABLE:
        email_spinner = EmailSpinner(
            importance_threshold=0.2
        )
        logger.info("  - Email spinner: enabled (.eml, .msg)")
    else:
        logger.info("  - Email spinner: unavailable (install beautifulsoup4)")

    # Codebase spinner (always available - uses stdlib ast)
    codebase_spinner = CodebaseSpinner(
        importance_threshold=0.2
    )
    logger.info("  - Codebase spinner: enabled (Python, TypeScript, JavaScript, etc.)")

    # Git spinner (always available - uses subprocess)
    git_spinner = GitSpinner(
        importance_threshold=0.2
    )
    logger.info("  - Git spinner: enabled (git repositories)")

    # Matrix spinner
    if MATRIX_AVAILABLE:
        matrix_spinner = MatrixSpinner(
            importance_threshold=0.2
        )
        logger.info("  - Matrix spinner: enabled (Matrix chat export)")
    else:
        logger.info("  - Matrix spinner: unavailable (install matrix-client)")

    # URL spinner
    if URL_AVAILABLE:
        url_spinner = URLSpinner(
            importance_threshold=0.2
        )
        logger.info("  - URL spinner: enabled (web scraping)")
    else:
        logger.info("  - URL spinner: unavailable (install requests beautifulsoup4)")

    # Initialize Voice Integration
    logger.info("Initializing Voice Integration...")
    try:
        global voice_integration
        voice_integration = await create_voice_integration(
            tts_backend="pyttsx3",  # Natural voice with emotions (upgrade from pyttsx3)
            whisper_model="base",  # Fast transcription
            auto_speak=True  # Auto-speak responses for conversational mode
        )

        # Add voice endpoints to the app
        add_voice_endpoints(app, voice_integration)

        logger.info("✓ Voice Integration enabled")
        logger.info(f"  - TTS: {voice_integration.tts_backend} (natural voice)")
        logger.info(f"  - Whisper: {voice_integration.whisper_model}")
        logger.info(f"  - Auto-speak: {voice_integration.auto_speak} (conversational mode)")
        logger.info("  - Voice endpoints: /api/voice/* available")

    except Exception as e:
        logger.warning(f"Voice Integration failed: {e}")
        logger.warning("Dashboard will work without voice capabilities")
        voice_integration = None

    # Initialize MCTS Agent Pool for background learning
    logger.info("Initializing MCTS Agent Pool...")
    try:
        from hololoom.memory.graph import KG
        from hololoom.embedding.spectral import MatryoshkaEmbeddings

        # Get KG and embeddings from orchestrator
        kg = agentic_orchestrator.kg if hasattr(agentic_orchestrator, 'kg') else KG()
        emb = agentic_orchestrator.emb if hasattr(agentic_orchestrator, 'emb') else MatryoshkaEmbeddings()

        agent_pool = await create_agent_pool(
            kg=kg,
            emb=emb,
            persist_path=Path("./data/agent_patterns"),
            enable_background_learning=True,
            mcts_simulations=50,  # Moderate for production
            exploration_simulations=25
        )

        logger.info("Agent Pool initialized with background learning")
        logger.info("  - Persist path: ./data/agent_patterns")
        logger.info("  - MCTS simulations: 50")
        logger.info("  - Background exploration: 25 simulations")

    except Exception as e:
        logger.warning(f"Agent Pool initialization failed: {e}")
        logger.warning("  - Continuing without background learning")
        agent_pool = None

    logger.info("="*60)
    logger.info("Dashboard ready at http://localhost:8002")
    logger.info("="*60)

    yield  # Server is now running

    # Shutdown code
    logger.info("Shutting down HoloLoom Agentic Dashboard...")

    # Cleanup agent pool
    if agent_pool:
        logger.info("Closing agent pool...")
        await agent_pool.close()
        stats = agent_pool.stats()
        logger.info(f"  - Total queries: {stats['total_queries']}")
        logger.info(f"  - Patterns learned: {sum(stats['patterns_by_agent'].values())}")


# Create FastAPI app with lifespan
app = FastAPI(title="HoloLoom Agentic Dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve main dashboard HTML"""
    html_file = Path(__file__).parent / "agentic_dashboard.html"

    if html_file.exists():
        return FileResponse(html_file)
    else:
        # Return inline HTML
        return HTMLResponse(content=get_agentic_html())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time agentic chat"""
    await websocket.accept()
    active_connections.append(websocket)

    # Don't create conversation yet - wait for first message
    # This prevents empty conversations from accumulating on page refreshes
    websocket_conversations[websocket] = None

    logger.info(f"Client connected. Active: {len(active_connections)}")

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)

            action = message_data.get('action', 'reason')

            if action == 'new_conversation':
                # Create new conversation thread
                new_conv = conversation_manager.create_conversation()
                websocket_conversations[websocket] = new_conv.id

                await websocket.send_json({
                    'type': 'conversation_created',
                    'data': new_conv.to_dict()
                })
                logger.info(f"New conversation {new_conv.id} created")

            elif action == 'load_conversation':
                # Load existing conversation
                conversation_id = message_data.get('conversation_id')
                conversation = conversation_manager.get_conversation(conversation_id)

                if conversation:
                    websocket_conversations[websocket] = conversation_id
                    messages = conversation_manager.get_messages(conversation_id)

                    await websocket.send_json({
                        'type': 'conversation_loaded',
                        'data': {
                            'conversation': conversation.to_dict(),
                            'messages': [msg.to_dict() for msg in messages]
                        }
                    })
                    logger.info(f"Loaded conversation {conversation_id} with {len(messages)} messages")
                else:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': f'Conversation {conversation_id} not found'}
                    })

            elif action == 'list_conversations':
                # List conversations with pagination support
                limit = message_data.get('limit', 50)
                offset = message_data.get('offset', 0)

                conversations, has_more = conversation_manager.list_conversations(limit=limit, offset=offset)

                await websocket.send_json({
                    'type': 'conversations_list',
                    'data': {
                        'conversations': [conv.to_dict() for conv in conversations],
                        'has_more': has_more,
                        'offset': offset + len(conversations)
                    }
                })

            elif action == 'delete_conversation':
                # Delete a conversation
                conversation_id = message_data.get('conversation_id')
                conversation_manager.delete_conversation(conversation_id)

                # If it's the current conversation, create a new one
                if websocket_conversations.get(websocket) == conversation_id:
                    new_conv = conversation_manager.create_conversation()
                    websocket_conversations[websocket] = new_conv.id
                    await websocket.send_json({
                        'type': 'conversation_created',
                        'data': new_conv.to_dict()
                    })

                await websocket.send_json({
                    'type': 'conversation_deleted',
                    'data': {'conversation_id': conversation_id}
                })

                # Send updated conversation list
                conversations, has_more = conversation_manager.list_conversations(limit=50)
                await websocket.send_json({
                    'type': 'conversations_list',
                    'data': {
                        'conversations': [conv.to_dict() for conv in conversations],
                        'has_more': has_more
                    }
                })

            # [REMOVED DUPLICATE] create_project handler moved to line 684

            elif action == 'list_projects':
                # List all projects
                projects = conversation_manager.list_projects()

                await websocket.send_json({
                    'type': 'projects_list',
                    'data': {
                        'projects': [proj.to_dict() for proj in projects]
                    }
                })

            elif action == 'update_project':
                # Update project details
                project_id = message_data.get('project_id')
                name = message_data.get('name')
                color = message_data.get('color')
                icon = message_data.get('icon')

                conversation_manager.update_project(project_id, name, color, icon)

                # Send updated project list
                projects = conversation_manager.list_projects()
                await websocket.send_json({
                    'type': 'project_updated',
                    'data': {'project_id': project_id}
                })
                await websocket.send_json({
                    'type': 'projects_list',
                    'data': {
                        'projects': [proj.to_dict() for proj in projects]
                    }
                })

            elif action == 'delete_project':
                # Delete project
                project_id = message_data.get('project_id')
                delete_conversations = message_data.get('delete_conversations', False)

                conversation_manager.delete_project(project_id, delete_conversations)

                await websocket.send_json({
                    'type': 'project_deleted',
                    'data': {'project_id': project_id}
                })

                # Send updated lists
                projects = conversation_manager.list_projects()
                await websocket.send_json({
                    'type': 'projects_list',
                    'data': {
                        'projects': [proj.to_dict() for proj in projects]
                    }
                })

                conversations = conversation_manager.list_conversations(limit=50)
                await websocket.send_json({
                    'type': 'conversations_list',
                    'data': {
                        'conversations': [conv.to_dict() for conv in conversations]
                    }
                })

            elif action == 'move_to_project':
                # Move conversation to project
                conversation_id = message_data.get('conversation_id')
                project_id = message_data.get('project_id')  # None = uncategorized

                conversation_manager.move_to_project(conversation_id, project_id)

                await websocket.send_json({
                    'type': 'conversation_moved',
                    'data': {
                        'conversation_id': conversation_id,
                        'project_id': project_id
                    }
                })

                # Send updated lists
                projects = conversation_manager.list_projects()
                await websocket.send_json({
                    'type': 'projects_list',
                    'data': {
                        'projects': [proj.to_dict() for proj in projects]
                    }
                })

                conversations = conversation_manager.list_conversations(limit=50)
                await websocket.send_json({
                    'type': 'conversations_list',
                    'data': {
                        'conversations': [conv.to_dict() for conv in conversations]
                    }
                })

            elif action == 'rename_conversation':
                # Rename conversation
                conversation_id = message_data.get('conversation_id')
                new_title = message_data.get('title', '')

                if new_title and conversation_id:
                    conversation_manager.rename_conversation(conversation_id, new_title)

                    await websocket.send_json({
                        'type': 'conversation_renamed',
                        'data': {
                            'conversation_id': conversation_id,
                            'title': new_title
                        }
                    })

                    # Send updated conversation list
                    conversations = conversation_manager.list_conversations(limit=50)
                    await websocket.send_json({
                        'type': 'conversations_list',
                        'data': {
                            'conversations': [conv.to_dict() for conv in conversations]
                        }
                    })

            elif action == 'search_conversations':
                # Search conversations
                query = message_data.get('query', '')
                limit = message_data.get('limit', 20)

                results = conversation_manager.search_conversations(query, limit)

                await websocket.send_json({
                    'type': 'search_results',
                    'data': {
                        'query': query,
                        'results': [conv.to_dict() for conv in results]
                    }
                })

            elif action == 'toggle_favorite':
                # Toggle favorite status
                conversation_id = message_data.get('conversation_id')

                conversation_manager.toggle_favorite(conversation_id)

                await websocket.send_json({
                    'type': 'favorite_toggled',
                    'data': {'conversation_id': conversation_id}
                })

                # Send updated conversation list
                conversations, has_more = conversation_manager.list_conversations(limit=50)
                await websocket.send_json({
                    'type': 'conversations_list',
                    'data': {
                        'conversations': [conv.to_dict() for conv in conversations],
                        'has_more': has_more
                    }
                })

            elif action == 'update_tags':
                # Update conversation tags
                conversation_id = message_data.get('conversation_id')
                tags = message_data.get('tags', [])

                conversation_manager.update_tags(conversation_id, tags)

                await websocket.send_json({
                    'type': 'tags_updated',
                    'data': {
                        'conversation_id': conversation_id,
                        'tags': tags
                    }
                })

                # Send updated conversation list
                conversations, has_more = conversation_manager.list_conversations(limit=50)
                await websocket.send_json({
                    'type': 'conversations_list',
                    'data': {
                        'conversations': [conv.to_dict() for conv in conversations],
                        'has_more': has_more
                    }
                })

            elif action == 'reason':
                # Process agentic reasoning request
                user_query = message_data.get('query', '')
                reasoning_mode_str = message_data.get('mode', 'direct')
                max_steps = message_data.get('max_steps', 3)

                # Get current conversation ID
                conversation_id = websocket_conversations.get(websocket)
                if not conversation_id:
                    # Create new conversation on first message
                    new_conv = conversation_manager.create_conversation()
                    conversation_id = new_conv.id
                    websocket_conversations[websocket] = conversation_id

                    # Notify client
                    await websocket.send_json({
                        'type': 'conversation_created',
                        'data': new_conv.to_dict()
                    })

                # Save user message to database
                user_message = conversation_manager.add_message(
                    conversation_id=conversation_id,
                    role='user',
                    content=user_query,
                    metadata={'mode': reasoning_mode_str}
                )

                # Map string to ReasoningMode enum
                mode_map = {
                    'direct': ReasoningMode.DIRECT,
                    'verify': ReasoningMode.VERIFY,
                    'research': ReasoningMode.RESEARCH,
                    'plan_execute': ReasoningMode.PLAN_EXECUTE
                }
                reasoning_mode = mode_map.get(reasoning_mode_str, ReasoningMode.DIRECT)

                logger.info(f"\n{'='*60}")
                logger.info(f"Query: {user_query}")
                logger.info(f"Mode: {reasoning_mode.value.upper()}")
                logger.info(f"Conversation: {conversation_id}")
                logger.info(f"{'='*60}")

                # Send acknowledgment
                await websocket.send_json({
                    'type': 'reasoning_started',
                    'data': {
                        'query': user_query,
                        'mode': reasoning_mode.value,
                        'max_steps': max_steps
                    }
                })

                try:
                    # Execute agentic reasoning
                    result = await agentic_orchestrator.reason(
                        query=Query(text=user_query),
                        mode=reasoning_mode,
                        max_steps=max_steps
                    )

                    # Extract reasoning steps with LLM markers
                    reasoning_steps = []
                    for step in result.steps_taken:
                        step_info = {
                            'type': step.get('type', 'unknown'),
                            'llm_generated': step.get('llm_generated', False),
                            'confidence': step.get('confidence', 0.0),
                            'query': step.get('query', ''),
                            'findings': step.get('findings', ''),
                            'checks': step.get('checks', []),
                            'goals': step.get('goals', []),
                            'sources': step.get('sources', 0)
                        }
                        reasoning_steps.append(step_info)

                    # Extract audit trail
                    audit_logs = []
                    if agentic_orchestrator.audit_trail:
                        for log in agentic_orchestrator.audit_trail.logs[-5:]:  # Last 5
                            audit_logs.append({
                                'action': log.action_description or log.decision_type.value,
                                'outcome': log.outcome.value,
                                'risk': log.risk_level or 'unknown',
                                'confidence': log.confidence,
                                'timestamp': log.timestamp.isoformat()
                            })

                    # Build response
                    response_data = {
                        'query': user_query,
                        'mode': reasoning_mode.value,
                        'response': result.spacetime.response or "No response generated",
                        'confidence': result.spacetime.confidence,
                        'total_steps': len(result.steps_taken),
                        'total_queries': result.total_queries,
                        'duration_ms': result.total_duration_ms,
                        'reasoning_steps': reasoning_steps,
                        'audit_trail': audit_logs,
                        'llm_status': 'active' if agentic_orchestrator.llm else 'unavailable'
                    }

                    logger.info(f"Reasoning complete:")
                    logger.info(f"  - Steps: {len(result.steps_taken)}")
                    logger.info(f"  - Confidence: {result.spacetime.confidence:.2f}")
                    logger.info(f"  - Duration: {result.total_duration_ms:.1f}ms")

                    # Save assistant message to database
                    assistant_metadata = {
                        'confidence': result.spacetime.confidence,
                        'duration_ms': result.total_duration_ms,
                        'total_steps': len(result.steps_taken),
                        'mode': reasoning_mode.value,
                        'reasoning_steps': reasoning_steps,
                        'llm_status': 'active' if agentic_orchestrator.llm else 'unavailable'
                    }
                    assistant_message = conversation_manager.add_message(
                        conversation_id=conversation_id,
                        role='assistant',
                        content=result.spacetime.response or "No response generated",
                        metadata=assistant_metadata
                    )

                    # Update conversation title if this is the first exchange
                    conversation = conversation_manager.get_conversation(conversation_id)
                    if conversation and conversation.message_count == 2:  # User + assistant
                        # Generate title from first user query (limit to 50 chars)
                        title = user_query[:50] + "..." if len(user_query) > 50 else user_query
                        conversation_manager.update_conversation_title(conversation_id, title)

                    # Send response
                    await websocket.send_json({
                        'type': 'reasoning_complete',
                        'data': response_data
                    })

                    # Send updated conversation list
                    conversations = conversation_manager.list_conversations(limit=50)
                    await websocket.send_json({
                        'type': 'conversations_list',
                        'data': {
                            'conversations': [conv.to_dict() for conv in conversations]
                        }
                    })

                except Exception as e:
                    logger.error(f"✗ Reasoning failed: {e}")
                    import traceback
                    traceback.print_exc()

                    await websocket.send_json({
                        'type': 'reasoning_error',
                        'data': {
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        }
                    })

            elif action == 'get_preview':
                # Get conversation preview (first N messages)
                conversation_id = message_data.get('conversation_id')
                limit = message_data.get('limit', 3)

                # Get messages from conversation manager (no limit parameter)
                all_messages = conversation_manager.get_messages(conversation_id)
                messages = all_messages[:limit]  # Slice to limit

                await websocket.send_json({
                    'type': 'preview_messages',
                    'data': {
                        'conversation_id': conversation_id,
                        'messages': [msg.to_dict() for msg in messages]
                    }
                })

            elif action == 'save_as_template':
                # Save conversation as template
                conversation_id = message_data.get('conversation_id')
                template_name = message_data.get('name')
                description = message_data.get('description', '')
                icon = message_data.get('icon', '📄')

                # Get conversation messages
                messages = conversation_manager.get_messages(conversation_id)
                messages_data = [{'role': msg.role, 'content': msg.content} for msg in messages]

                # Create template
                template = conversation_manager.create_template(template_name, description, messages_data, icon)

                await websocket.send_json({
                    'type': 'template_created',
                    'data': template.to_dict()
                })

            elif action == 'list_templates':
                # List all templates
                templates = conversation_manager.list_templates()

                await websocket.send_json({
                    'type': 'templates_list',
                    'data': {
                        'templates': [t.to_dict() for t in templates]
                    }
                })

            elif action == 'load_template':
                # Load template and create new conversation from it
                template_id = message_data.get('template_id')
                template = conversation_manager.load_template(template_id)

                if template:
                    # Create new conversation from template
                    new_conv = conversation_manager.create_conversation(title=template.name)

                    # Add template messages
                    for msg in json.loads(template.messages):
                        conversation_manager.add_message(
                            conversation_id=new_conv.id,
                            role=msg['role'],
                            content=msg['content']
                        )

                    await websocket.send_json({
                        'type': 'conversation_created',
                        'data': new_conv.to_dict()
                    })

            elif action == 'delete_template':
                # Delete a template
                template_id = message_data.get('template_id')
                conversation_manager.delete_template(template_id)

                await websocket.send_json({
                    'type': 'template_deleted',
                    'data': {'template_id': template_id}
                })

            elif action == 'create_project':
                # Create a new project
                name = message_data.get('name')
                icon = message_data.get('icon', '📁')
                color = message_data.get('color', '#667eea')

                project = conversation_manager.create_project(name, icon, color)

                await websocket.send_json({
                    'type': 'project_created',
                    'data': project.to_dict()
                })

                # Send updated project list so UI refreshes
                projects = conversation_manager.list_projects()
                await websocket.send_json({
                    'type': 'projects_list',
                    'data': {
                        'projects': [proj.to_dict() for proj in projects]
                    }
                })

            elif action == 'get_analytics':
                # Get analytics data
                analytics_data = conversation_manager.get_analytics()

                await websocket.send_json({
                    'type': 'analytics_data',
                    'data': analytics_data
                })

            elif action == 'list_prompts':
                # List Promptly prompts
                if not promptly_bridge:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': 'Promptly not available'}
                    })
                else:
                    prompts = promptly_bridge.list_prompts()
                    await websocket.send_json({
                        'type': 'prompts_list',
                        'data': {'prompts': prompts}
                    })

            elif action == 'get_prompt':
                # Get specific Promptly prompt
                if not promptly_bridge:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': 'Promptly not available'}
                    })
                else:
                    prompt_name = message_data.get('name')
                    version = message_data.get('version')
                    prompt = promptly_bridge.get_prompt(prompt_name, version)
                    await websocket.send_json({
                        'type': 'prompt_data',
                        'data': prompt
                    })

            elif action == 'save_prompt':
                # Save prompt to Promptly
                if not promptly_bridge:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': 'Promptly not available'}
                    })
                else:
                    name = message_data.get('name')
                    content = message_data.get('content')
                    description = message_data.get('description', '')
                    tags = message_data.get('tags', [])

                    promptly_bridge.save_prompt(name, content, description, tags)

                    await websocket.send_json({
                        'type': 'prompt_saved',
                        'data': {'name': name}
                    })

            elif action == 'ingest_youtube':
                # Ingest YouTube video transcript
                if not youtube_spinner:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': 'YouTube spinner not available. Install: pip install youtube-transcript-api'}
                    })
                else:
                    url = message_data.get('url', '')
                    logger.info(f"Ingesting YouTube: {url}")

                    try:
                        # Save permalink (raw "wool")
                        wool_dir = Path("./data/wool/youtube")
                        wool_dir.mkdir(parents=True, exist_ok=True)

                        video_id = youtube_spinner._extract_video_id(url)
                        permalink_file = wool_dir / f"{video_id}_permalink.txt"
                        with open(permalink_file, 'w') as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Video ID: {video_id}\n")
                            f.write(f"Ingested: {asyncio.get_event_loop().time()}\n")

                        # Transcribe YouTube video
                        result = await youtube_spinner.spin(url)

                        if result.success:
                            # Add shards to orchestrator memory
                            for shard in result.shards:
                                await agentic_orchestrator.add_shard(shard)

                            await websocket.send_json({
                                'type': 'youtube_ingested',
                                'data': {
                                    'video_id': result.metadata.get('video_id'),
                                    'title': result.metadata.get('title'),
                                    'shard_count': len(result.shards),
                                    'duration': result.metadata.get('duration'),
                                    'language': result.metadata.get('language')
                                }
                            })
                            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {url}")
                        else:
                            await websocket.send_json({
                                'type': 'error',
                                'data': {'error': result.error_message}
                            })

                    except Exception as e:
                        logger.error(f"  ✗ YouTube ingestion failed: {e}")
                        await websocket.send_json({
                            'type': 'error',
                            'data': {'error': str(e)}
                        })

            elif action == 'ingest_git':
                # Ingest Git repository
                if not git_spinner:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': 'Git spinner not available (should always be available)'}
                    })
                else:
                    repo_path = message_data.get('path', '')
                    logger.info(f"Ingesting Git repo: {repo_path}")

                    try:
                        # Validate path exists
                        if not Path(repo_path).exists():
                            await websocket.send_json({
                                'type': 'error',
                                'data': {'error': f'Path does not exist: {repo_path}'}
                            })
                            return

                        # Save permalink (raw "wool")
                        wool_dir = Path("./data/wool/git")
                        wool_dir.mkdir(parents=True, exist_ok=True)

                        repo_name = Path(repo_path).name
                        permalink_file = wool_dir / f"{repo_name}_permalink.txt"
                        with open(permalink_file, 'w') as f:
                            f.write(f"Path: {repo_path}\n")
                            f.write(f"Ingested: {asyncio.get_event_loop().time()}\n")

                        # Ingest Git repository
                        result = await git_spinner.spin(repo_path)

                        if result.success:
                            # Add shards to orchestrator memory
                            for shard in result.shards:
                                await agentic_orchestrator.add_shard(shard)

                            await websocket.send_json({
                                'type': 'git_ingested',
                                'data': {
                                    'repo_path': repo_path,
                                    'shard_count': len(result.shards),
                                    'commit_count': result.metadata.get('commit_count', 0),
                                    'author_count': result.metadata.get('author_count', 0)
                                }
                            })
                            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {repo_path}")
                        else:
                            await websocket.send_json({
                                'type': 'error',
                                'data': {'error': result.error_message}
                            })

                    except Exception as e:
                        logger.error(f"  ✗ Git ingestion failed: {e}")
                        await websocket.send_json({
                            'type': 'error',
                            'data': {'error': str(e)}
                        })

            elif action == 'ingest_url':
                # Ingest web URL
                if not url_spinner:
                    await websocket.send_json({
                        'type': 'error',
                        'data': {'error': 'URL spinner not available. Install: pip install requests beautifulsoup4'}
                    })
                else:
                    url = message_data.get('url', '')
                    logger.info(f"Ingesting URL: {url}")

                    try:
                        # Save permalink (raw "wool")
                        wool_dir = Path("./data/wool/url")
                        wool_dir.mkdir(parents=True, exist_ok=True)

                        # Create filename from URL (sanitized)
                        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                        permalink_file = wool_dir / f"{url_hash}_permalink.txt"
                        with open(permalink_file, 'w') as f:
                            f.write(f"URL: {url}\n")
                            f.write(f"Ingested: {asyncio.get_event_loop().time()}\n")

                        # Scrape URL
                        result = await url_spinner.spin(url)

                        if result.success:
                            # Add shards to orchestrator memory
                            for shard in result.shards:
                                await agentic_orchestrator.add_shard(shard)

                            await websocket.send_json({
                                'type': 'url_ingested',
                                'data': {
                                    'url': url,
                                    'shard_count': len(result.shards),
                                    'title': result.metadata.get('title', 'Unknown'),
                                    'word_count': result.metadata.get('word_count', 0)
                                }
                            })
                            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {url}")
                        else:
                            await websocket.send_json({
                                'type': 'error',
                                'data': {'error': result.error_message}
                            })

                    except Exception as e:
                        logger.error(f"  ✗ URL ingestion failed: {e}")
                        await websocket.send_json({
                            'type': 'error',
                            'data': {'error': str(e)}
                        })

            elif action == 'get_status':
                # Return system status
                status = {
                    'orchestrator_ready': agentic_orchestrator is not None,
                    'llm_available': agentic_orchestrator.llm is not None if agentic_orchestrator else False,
                    'memory_backend': orchestrator_config.memory_backend.value if orchestrator_config else 'unknown',
                    'alignment_enabled': True,
                    'active_connections': len(active_connections)
                }

                await websocket.send_json({
                    'type': 'status_response',
                    'data': status
                })

    except WebSocketDisconnect:
        active_connections.remove(websocket)
        if websocket in websocket_conversations:
            del websocket_conversations[websocket]
        logger.info(f"Client disconnected. Active connections: {len(active_connections)}")


@app.get("/api/status")
async def get_status():
    """Get server status"""
    return {
        'status': 'running',
        'orchestrator_ready': agentic_orchestrator is not None,
        'llm_available': agentic_orchestrator.llm is not None if agentic_orchestrator else False,
        'memory_backend': orchestrator_config.memory_backend.value if orchestrator_config else 'unknown',
        'alignment_enabled': True,
        'active_connections': len(active_connections),
        'youtube_available': youtube_spinner is not None,
        'whisper_available': whisper_spinner is not None,
        'spreadsheet_available': spreadsheet_spinner is not None,
        'pdf_available': pdf_spinner is not None,
        'email_available': email_spinner is not None,
        'codebase_available': codebase_spinner is not None,
        'git_available': git_spinner is not None,
        'matrix_available': matrix_spinner is not None,
        'url_available': url_spinner is not None,
        'agent_pool_available': agent_pool is not None
    }


# ============================================================================
# Agent Pool Endpoints (MCTS Background Learning)
# ============================================================================

@app.post("/api/agent/query")
async def agent_query(data: dict):
    """
    Query an agent with MCTS background learning.

    Request body:
        {
            "agent": "budget" | "architecture" | "research" | "code_review" | "planning" | "general",
            "query": "What is Q4 revenue?",
            "use_mcts": true,  # Optional, default true
            "feedback": {  # Optional, for learning
                "helpful": true,
                "confidence": 0.85
            }
        }

    Returns:
        {
            "response": "...",
            "confidence": 0.85,
            "agent": "budget",
            "mcts_used": true,
            "pattern_count": 5
        }
    """
    if not agent_pool:
        return JSONResponse(
            status_code=503,
            content={'error': 'Agent pool not available'}
        )

    agent_name = data.get('agent', 'general')
    query_text = data.get('query')
    use_mcts = data.get('use_mcts', True)
    feedback = data.get('feedback')

    if not query_text:
        return JSONResponse(
            status_code=400,
            content={'error': 'query is required'}
        )

    try:
        query = Query(text=query_text)

        # Process query
        result = await agent_pool.query(agent_name, query, use_mcts=use_mcts)

        # Provide feedback if given
        if feedback:
            await agent_pool.feedback(agent_name, query, result, feedback)

        # Get pattern count
        stats = agent_pool.stats()
        pattern_count = stats['patterns_by_agent'].get(agent_name, 0)

        return {
            'response': result.response,
            'confidence': result.confidence,
            'agent': agent_name,
            'mcts_used': use_mcts,
            'pattern_count': pattern_count
        }

    except Exception as e:
        logger.error(f"Agent query error: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.get("/api/agent/stats")
async def agent_stats():
    """
    Get agent pool statistics.

    Returns:
        {
            "total_queries": 42,
            "active_agents": ["budget", "research"],
            "query_count_by_agent": {"budget": 30, "research": 12},
            "patterns_by_agent": {"budget": 5, "research": 3},
            "learning_queue": {
                "queue_size": 2,
                "total_added": 42,
                "total_processed": 40,
                "pending": 2
            },
            "background_learner": {
                "running": true,
                "learning_cycles": 40,
                "patterns_learned": 8,
                "total_exploration_time_s": 120.5,
                "avg_time_per_cycle_ms": 3012.5
            }
        }
    """
    if not agent_pool:
        return JSONResponse(
            status_code=503,
            content={'error': 'Agent pool not available'}
        )

    try:
        stats = agent_pool.stats()
        return stats
    except Exception as e:
        logger.error(f"Agent stats error: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/api/agent/feedback")
async def agent_feedback(data: dict):
    """
    Provide feedback on an agent query (for background learning).

    Request body:
        {
            "agent": "budget",
            "query": "What is Q4 revenue?",
            "outcome": { ... },  # Spacetime result (optional)
            "feedback": {
                "helpful": true,
                "confidence": 0.85,
                "latency_ms": 150
            }
        }

    Returns:
        {
            "status": "queued",
            "queue_size": 3
        }
    """
    if not agent_pool:
        return JSONResponse(
            status_code=503,
            content={'error': 'Agent pool not available'}
        )

    agent_name = data.get('agent')
    query_text = data.get('query')
    outcome = data.get('outcome')
    feedback = data.get('feedback')

    if not agent_name or not query_text or not feedback:
        return JSONResponse(
            status_code=400,
            content={'error': 'agent, query, and feedback are required'}
        )

    try:
        query = Query(text=query_text)
        await agent_pool.feedback(agent_name, query, outcome, feedback)

        stats = agent_pool.stats()
        queue_size = stats['learning_queue']['queue_size']

        return {
            'status': 'queued',
            'queue_size': queue_size
        }

    except Exception as e:
        logger.error(f"Agent feedback error: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/api/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    """
    Upload and transcribe audio file.
    Saves raw audio (wool) and creates memory shards.
    """
    if not whisper_spinner:
        return JSONResponse(
            status_code=503,
            content={'error': 'Whisper not available. Install: pip install openai-whisper'}
        )

    try:
        # Save raw audio file (wool)
        wool_dir = Path("./data/wool/audio")
        wool_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = wool_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Transcribing audio: {file.filename}")

        # Transcribe (spin() returns SpinResult wrapper)
        result = await whisper_spinner.spin(file_path)

        # Check if successful
        if not result.success:
            return JSONResponse(
                status_code=400,
                content={'error': result.error_message}
            )

        # Add shards to global knowledge store
        knowledge_shards.extend(result.shards)

        logger.info(f"  ✓ Transcribed {len(result.shards)} shards from {file.filename}")

        # Extract metadata from first shard
        metadata = result.shards[0].metadata if result.shards else {}
        full_text = ' '.join(s.content for s in result.shards[:3])  # First 3 shards

        return {
            'success': True,
            'filename': file.filename,
            'shard_count': len(result.shards),
            'duration': metadata.get('duration'),
            'language': metadata.get('language'),
            'full_text': full_text[:200]  # Preview
        }

    except Exception as e:
        logger.error(f"  ✗ Audio transcription failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/api/upload_spreadsheet")
async def upload_spreadsheet(file: UploadFile = File(...)):
    """
    Upload and parse spreadsheet file.
    Saves raw file (wool) and creates memory shards from tables.

    Supports: Excel (.xlsx, .xls), CSV, TSV, LibreOffice (.ods)
    """
    if not spreadsheet_spinner:
        return JSONResponse(
            status_code=503,
            content={'error': 'Spreadsheet parser not available. Install: pip install pandas openpyxl'}
        )

    try:
        # Validate file type
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ['.xlsx', '.xls', '.csv', '.tsv', '.ods']:
            return JSONResponse(
                status_code=400,
                content={'error': f'Unsupported file format: {suffix}. Supported: .xlsx, .xls, .csv, .tsv, .ods'}
            )

        # Save raw spreadsheet file (wool)
        wool_dir = Path("./data/wool/spreadsheet")
        wool_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = wool_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Parsing spreadsheet: {file.filename}")

        # Parse spreadsheet
        result = await spreadsheet_spinner.spin(file_path)

        if result.success:
            # Add shards to orchestrator memory
            for shard in result.shards:
                await agentic_orchestrator.add_shard(shard)

            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {file.filename}")

            return {
                'success': True,
                'filename': file.filename,
                'shard_count': len(result.shards),
                'file_format': result.metadata.get('file_format'),
                'sheet_count': result.metadata.get('sheet_count'),
                'total_rows': result.metadata.get('total_rows')
            }
        else:
            return JSONResponse(
                status_code=400,
                content={'error': result.error_message}
            )

    except Exception as e:
        logger.error(f"  ✗ Spreadsheet parsing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/api/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and parse PDF file.
    Saves raw PDF (wool) and creates memory shards from pages.

    Supports: .pdf
    """
    if not pdf_spinner:
        return JSONResponse(
            status_code=503,
            content={'error': 'PDF parser not available. Install: pip install PyPDF2 or pdfplumber'}
        )

    try:
        # Validate file type
        suffix = Path(file.filename).suffix.lower()
        if suffix != '.pdf':
            return JSONResponse(
                status_code=400,
                content={'error': f'Unsupported file format: {suffix}. Expected: .pdf'}
            )

        # Save raw PDF file (wool)
        wool_dir = Path("./data/wool/pdf")
        wool_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = wool_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Parsing PDF: {file.filename}")

        # Parse PDF
        result = await pdf_spinner.spin(file_path)

        if result.success:
            # Add shards to orchestrator memory
            for shard in result.shards:
                await agentic_orchestrator.add_shard(shard)

            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {file.filename}")

            return {
                'success': True,
                'filename': file.filename,
                'shard_count': len(result.shards),
                'page_count': result.metadata.get('page_count'),
                'title': result.metadata.get('title', 'Unknown'),
                'author': result.metadata.get('author', 'Unknown')
            }
        else:
            return JSONResponse(
                status_code=400,
                content={'error': result.error_message}
            )

    except Exception as e:
        logger.error(f"  ✗ PDF parsing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/api/upload_email")
async def upload_email(file: UploadFile = File(...)):
    """
    Upload and parse email file.
    Saves raw email (wool) and creates memory shards from email content.

    Supports: .eml, .msg
    """
    if not email_spinner:
        return JSONResponse(
            status_code=503,
            content={'error': 'Email parser not available. Install: pip install beautifulsoup4'}
        )

    try:
        # Validate file type
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ['.eml', '.msg']:
            return JSONResponse(
                status_code=400,
                content={'error': f'Unsupported file format: {suffix}. Supported: .eml, .msg'}
            )

        # Save raw email file (wool)
        wool_dir = Path("./data/wool/email")
        wool_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = wool_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Parsing email: {file.filename}")

        # Parse email
        result = await email_spinner.spin(file_path)

        if result.success:
            # Add shards to orchestrator memory
            for shard in result.shards:
                await agentic_orchestrator.add_shard(shard)

            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {file.filename}")

            return {
                'success': True,
                'filename': file.filename,
                'shard_count': len(result.shards),
                'subject': result.metadata.get('subject', 'Unknown'),
                'from': result.metadata.get('from', 'Unknown'),
                'date': result.metadata.get('date', 'Unknown')
            }
        else:
            return JSONResponse(
                status_code=400,
                content={'error': result.error_message}
            )

    except Exception as e:
        logger.error(f"  ✗ Email parsing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/api/upload_code")
async def upload_code(file: UploadFile = File(...)):
    """
    Upload and parse code file.
    Saves raw code (wool) and creates memory shards from functions/classes.

    Supports: .py, .ts, .tsx, .js, .jsx, .java, .go, .rs, etc.
    """
    if not codebase_spinner:
        return JSONResponse(
            status_code=503,
            content={'error': 'Codebase parser not available (should always be available)'}
        )

    try:
        # Validate file type (common code extensions)
        suffix = Path(file.filename).suffix.lower()
        valid_extensions = ['.py', '.ts', '.tsx', '.js', '.jsx', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.hpp']
        if suffix not in valid_extensions:
            return JSONResponse(
                status_code=400,
                content={'error': f'Unsupported file format: {suffix}. Supported: {", ".join(valid_extensions)}'}
            )

        # Save raw code file (wool)
        wool_dir = Path("./data/wool/code")
        wool_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = wool_dir / file.filename
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        logger.info(f"Parsing code: {file.filename}")

        # Parse code
        result = await codebase_spinner.spin(file_path)

        if result.success:
            # Add shards to orchestrator memory
            for shard in result.shards:
                await agentic_orchestrator.add_shard(shard)

            logger.info(f"  ✓ Ingested {len(result.shards)} shards from {file.filename}")

            return {
                'success': True,
                'filename': file.filename,
                'shard_count': len(result.shards),
                'language': result.metadata.get('language', 'unknown'),
                'total_lines': result.metadata.get('total_lines', 0),
                'class_count': result.metadata.get('class_count', 0),
                'function_count': result.metadata.get('function_count', 0)
            }
        else:
            return JSONResponse(
                status_code=400,
                content={'error': result.error_message}
            )

    except Exception as e:
        logger.error(f"  ✗ Code parsing failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


def create_demo_shards() -> List[MemoryShard]:
    """Create demo knowledge shards"""
    return [
        MemoryShard(
            id="shard_thompson_1",
            text=(
                "Thompson Sampling is a Bayesian approach to the multi-armed bandit problem. "
                "It maintains a probability distribution (typically Beta) for each arm's reward. "
                "At each step, it samples from these distributions and selects the arm with the "
                "highest sample. This naturally balances exploration and exploitation through "
                "the uncertainty in the posterior distributions."
            ),
            metadata={"topic": "reinforcement_learning", "confidence": 0.95}
        ),
        MemoryShard(
            id="shard_thompson_2",
            text=(
                "Compared to epsilon-greedy and UCB, Thompson Sampling often converges faster. "
                "UCB uses deterministic upper confidence bounds while Thompson Sampling uses "
                "probabilistic sampling. Epsilon-greedy has fixed exploration rate, while "
                "Thompson Sampling adapts based on posterior uncertainty."
            ),
            metadata={"topic": "algorithm_comparison", "confidence": 0.90}
        ),
        MemoryShard(
            id="shard_transformers",
            text=(
                "Transformers revolutionized NLP through self-attention mechanisms. Unlike RNNs, "
                "they process sequences in parallel, using attention to weigh the importance of "
                "different positions. Key innovation: attention scores computed as "
                "softmax(QK^T/√d_k)V."
            ),
            metadata={"topic": "deep_learning", "confidence": 0.92}
        )
    ]


def get_agentic_html() -> str:
    """Return agentic dashboard HTML"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>HoloLoom Chat</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            height: 100vh;
            display: flex;
            flex-direction: row;
            overflow: hidden;
        }

        /* Main Content Area (everything except sidebar) */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Sidebar */
        .sidebar {
            width: 280px;
            background: rgba(0, 0, 0, 0.4);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .new-chat-btn {
            width: 100%;
            padding: 12px 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 10px;
            color: #fff;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }

        .new-chat-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .conversations-list {
            flex: 1;
            overflow-y: auto;
            padding: 10px;
        }

        .conversation-item {
            padding: 12px 14px;
            margin-bottom: 6px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            position: relative;
        }

        .conversation-item:hover {
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(102, 126, 234, 0.4);
        }

        .conversation-item.active {
            background: rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }

        .conversation-title {
            font-size: 0.9em;
            color: #e0e0e0;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .conversation-meta {
            font-size: 0.75em;
            color: #888;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .conversation-delete {
            position: absolute;
            top: 8px;
            right: 8px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: rgba(231, 76, 60, 0.2);
            border: none;
            color: #e74c3c;
            cursor: pointer;
            opacity: 0;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8em;
        }

        .conversation-item:hover .conversation-delete {
            opacity: 1;
        }

        .conversation-delete:hover {
            background: rgba(231, 76, 60, 0.4);
        }

        /* Header */
        .header {
            background: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title {
            font-size: 1.3em;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-status {
            display: flex;
            align-items: center;
            gap: 15px;
            color: #aaa;
            font-size: 0.9em;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #2ecc71;
            box-shadow: 0 0 5px #2ecc71;
        }

        .status-dot.offline {
            background: #e74c3c;
            box-shadow: none;
        }

        /* Mode Selector */
        .mode-bar {
            background: rgba(255, 255, 255, 0.03);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            display: flex;
            gap: 10px;
        }

        .mode-chip {
            padding: 6px 12px;
            background: rgba(102, 126, 234, 0.2);
            border: 1px solid rgba(102, 126, 234, 0.4);
            border-radius: 20px;
            color: #aaa;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.85em;
        }

        .mode-chip:hover {
            background: rgba(102, 126, 234, 0.3);
            color: #fff;
        }

        .mode-chip.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-color: #764ba2;
            color: #fff;
        }

        /* Chat Area */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .message {
            display: flex;
            gap: 10px;
            max-width: 85%;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2em;
            flex-shrink: 0;
        }

        .message.user .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .message.assistant .message-avatar {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .message-content {
            flex: 1;
        }

        .message-bubble {
            padding: 12px 16px;
            border-radius: 18px;
            line-height: 1.5;
            word-wrap: break-word;
        }

        .message.user .message-bubble {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #fff;
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-bubble {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-bottom-left-radius: 4px;
        }

        .message-meta {
            font-size: 0.75em;
            color: #666;
            margin-top: 4px;
            padding: 0 8px;
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .message.user .message-meta {
            justify-content: flex-end;
        }

        /* Reasoning Steps (Collapsible) */
        .reasoning-toggle {
            color: #667eea;
            cursor: pointer;
            font-size: 0.85em;
            margin-top: 8px;
            padding: 6px 12px;
            background: rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            display: inline-block;
            user-select: none;
        }

        .reasoning-toggle:hover {
            background: rgba(102, 126, 234, 0.2);
        }

        .reasoning-steps {
            margin-top: 10px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            border-left: 3px solid #667eea;
            display: none;
        }

        .reasoning-steps.expanded {
            display: block;
        }

        .reasoning-step {
            padding: 8px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.9em;
        }

        .reasoning-step:last-child {
            border-bottom: none;
        }

        .step-title {
            color: #667eea;
            font-weight: 500;
        }

        .step-detail {
            color: #aaa;
            margin-top: 4px;
        }

        /* Typing Indicator */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 12px;
        }

        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #667eea;
            animation: typing 1.4s infinite;
        }

        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                opacity: 0.3;
                transform: scale(0.8);
            }
            30% {
                opacity: 1;
                transform: scale(1);
            }
        }

        /* Input Area */
        .input-container {
            background: rgba(255, 255, 255, 0.05);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px 20px;
        }

        .input-wrapper {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }

        .input-field {
            flex: 1;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 24px;
            padding: 12px 20px;
            color: #fff;
            font-size: 0.95em;
            font-family: inherit;
            resize: none;
            max-height: 120px;
            overflow-y: auto;
        }

        .input-field:focus {
            outline: none;
            border-color: #667eea;
        }

        .send-button {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: #fff;
            font-size: 1.3em;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }

        .send-button:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .send-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        /* Welcome Message */
        .welcome-message {
            text-align: center;
            padding: 40px 20px;
            color: #aaa;
        }

        .welcome-message h2 {
            color: #667eea;
            margin-bottom: 10px;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.2);
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(102, 126, 234, 0.5);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(102, 126, 234, 0.8);
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <button class="new-chat-btn" onclick="newChat()">
                <span>+</span>
                <span>New Chat</span>
            </button>
        </div>
        <div class="conversations-list" id="conversationsList">
            <!-- Conversation items will be added here dynamically -->
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Header -->
        <div class="header">
            <div class="header-title">HoloLoom Chat</div>
            <div class="header-status">
                <span class="status-dot" id="statusDot"></span>
                <span id="statusText">Connecting...</span>
            </div>
        </div>

        <!-- Mode Selector -->
        <div class="mode-bar">
            <div class="mode-chip active" data-mode="direct">DIRECT</div>
            <div class="mode-chip" data-mode="verify">VERIFY</div>
            <div class="mode-chip" data-mode="research">RESEARCH</div>
            <div class="mode-chip" data-mode="plan_execute">PLAN</div>
        </div>

        <!-- Chat Messages -->
        <div class="chat-container" id="chatContainer">
            <div class="welcome-message">
                <h2>Welcome to HoloLoom</h2>
                <p>Ask me anything. I'll use agentic reasoning to find the best answer.</p>
            </div>
        </div>

        <!-- Input Area -->
        <div class="input-container">
            <div class="input-wrapper">
                <textarea
                    class="input-field"
                    id="messageInput"
                    placeholder="Type your message..."
                    rows="1"
                ></textarea>
                <button class="send-button" id="sendButton">
                    <span>&#10148;</span>
                </button>
            </div>
        </div>
    </div>

    <script>
        // State
        let ws = null;
        let selectedMode = 'direct';
        let messageHistory = [];
        let typingMessageId = null;
        let currentConversationId = null;
        let conversations = [];

        // Connect to WebSocket
        function connect() {
            ws = new WebSocket('ws://localhost:8002/ws');

            ws.onopen = () => {
                console.log('Connected to server');
                document.getElementById('statusDot').className = 'status-dot';
                document.getElementById('statusText').textContent = 'Connected';
                ws.send(JSON.stringify({ action: 'get_status' }));
                ws.send(JSON.stringify({ action: 'list_conversations' }));
            };

            ws.onclose = () => {
                console.log('Disconnected from server');
                document.getElementById('statusDot').className = 'status-dot offline';
                document.getElementById('statusText').textContent = 'Reconnecting...';
                setTimeout(connect, 3000);
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                handleMessage(message);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        // Handle incoming messages
        function handleMessage(message) {
            if (message.type === 'status_response') {
                const status = message.data;
                document.getElementById('statusText').textContent =
                    `LLM: ${status.llm_available ? 'Active' : 'N/A'} | ${status.memory_backend}`;
            }
            else if (message.type === 'reasoning_started') {
                showTypingIndicator();
            }
            else if (message.type === 'reasoning_complete') {
                removeTypingIndicator();
                addAssistantMessage(message.data);
                document.getElementById('sendButton').disabled = false;
                document.getElementById('messageInput').value = '';
                autoResize(document.getElementById('messageInput'));
            }
            else if (message.type === 'reasoning_error') {
                removeTypingIndicator();
                addAssistantMessage({
                    response: 'Sorry, an error occurred: ' + message.data.error,
                    confidence: 0,
                    duration_ms: 0,
                    total_steps: 0,
                    reasoning_steps: []
                });
                document.getElementById('sendButton').disabled = false;
            }
            else if (message.type === 'conversation_created') {
                currentConversationId = message.data.id;
                console.log('New conversation created:', currentConversationId);
                ws.send(JSON.stringify({ action: 'list_conversations' }));
            }
            else if (message.type === 'conversation_loaded') {
                const { conversation, messages } = message.data;
                currentConversationId = conversation.id;
                console.log('Loaded conversation:', conversation.id);

                // Clear chat area
                const chatContainer = document.getElementById('chatContainer');
                chatContainer.innerHTML = '';
                messageHistory = [];

                // Restore messages
                messages.forEach(msg => {
                    if (msg.role === 'user') {
                        addUserMessage(msg.content);
                    } else {
                        addAssistantMessage({
                            response: msg.content,
                            confidence: msg.metadata.confidence || 0,
                            duration_ms: msg.metadata.duration_ms || 0,
                            total_steps: msg.metadata.total_steps || 0,
                            reasoning_steps: msg.metadata.reasoning_steps || []
                        });
                    }
                });

                updateConversationList();
            }
            else if (message.type === 'conversations_list') {
                conversations = message.data.conversations;
                updateConversationList();
            }
        }

        // Add user message to chat
        function addUserMessage(text) {
            const chatContainer = document.getElementById('chatContainer');

            // Remove welcome message if exists
            const welcome = chatContainer.querySelector('.welcome-message');
            if (welcome) welcome.remove();

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';
            messageDiv.innerHTML = `
                <div class="message-avatar">👤</div>
                <div class="message-content">
                    <div class="message-bubble">${escapeHtml(text)}</div>
                    <div class="message-meta">
                        <span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}</span>
                        <span>${selectedMode.toUpperCase()}</span>
                    </div>
                </div>
            `;

            chatContainer.appendChild(messageDiv);
            scrollToBottom();
        }

        // Add assistant message to chat
        function addAssistantMessage(data) {
            const chatContainer = document.getElementById('chatContainer');
            const messageId = 'msg-' + Date.now();

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant';
            messageDiv.id = messageId;

            let reasoningHtml = '';
            if (data.reasoning_steps && data.reasoning_steps.length > 0) {
                const stepCount = data.reasoning_steps.length;
                const toggleId = `toggle-${messageId}`;
                const stepsId = `steps-${messageId}`;

                reasoningHtml = `
                    <div class="reasoning-toggle" onclick="toggleReasoning('${stepsId}')">
                        Show reasoning (${stepCount} steps)
                    </div>
                    <div class="reasoning-steps" id="${stepsId}">
                        ${data.reasoning_steps.map((step, i) => `
                            <div class="reasoning-step">
                                <div class="step-title">Step ${i + 1}: ${step.type}</div>
                                <div class="step-detail">
                                    ${step.query ? `Query: ${escapeHtml(step.query)}` : ''}
                                    ${step.confidence !== undefined ? ` (${(step.confidence * 100).toFixed(0)}% confidence)` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }

            messageDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-bubble">${escapeHtml(data.response || 'No response')}</div>
                    ${reasoningHtml}
                    <div class="message-meta">
                        <span>${new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}</span>
                        <span>${(data.confidence * 100).toFixed(0)}% confidence</span>
                        <span>${data.duration_ms.toFixed(0)}ms</span>
                    </div>
                </div>
            `;

            chatContainer.appendChild(messageDiv);
            scrollToBottom();
        }

        // Show typing indicator
        function showTypingIndicator() {
            const chatContainer = document.getElementById('chatContainer');
            const typingDiv = document.createElement('div');
            typingDiv.className = 'message assistant';
            typingDiv.id = 'typing-indicator';
            typingDiv.innerHTML = `
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="typing-indicator">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>
            `;
            chatContainer.appendChild(typingDiv);
            scrollToBottom();
        }

        // Remove typing indicator
        function removeTypingIndicator() {
            const typing = document.getElementById('typing-indicator');
            if (typing) typing.remove();
        }

        // Toggle reasoning steps
        function toggleReasoning(stepsId) {
            const steps = document.getElementById(stepsId);
            if (steps) {
                steps.classList.toggle('expanded');
            }
        }

        // Send message
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message || !ws || ws.readyState !== WebSocket.OPEN) {
                return;
            }

            // Add user message to chat
            addUserMessage(message);

            // Disable send button
            document.getElementById('sendButton').disabled = true;

            // Send to server
            ws.send(JSON.stringify({
                action: 'reason',
                query: message,
                mode: selectedMode,
                max_steps: 3
            }));
        }

        // Auto-resize textarea
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
        }

        // Scroll to bottom
        function scrollToBottom() {
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Escape HTML
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Mode selection
        document.querySelectorAll('.mode-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                document.querySelectorAll('.mode-chip').forEach(c => c.classList.remove('active'));
                chip.classList.add('active');
                selectedMode = chip.getAttribute('data-mode');
            });
        });

        // Send button click
        document.getElementById('sendButton').addEventListener('click', sendMessage);

        // Input field handling
        const messageInput = document.getElementById('messageInput');
        messageInput.addEventListener('input', function() {
            autoResize(this);
        });

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        // Conversation management
        function newChat() {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                console.error('WebSocket not connected');
                return;
            }

            // Clear current chat
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.innerHTML = `
                <div class="welcome-message">
                    <h2>Welcome to HoloLoom</h2>
                    <p>Ask me anything. I'll use agentic reasoning to find the best answer.</p>
                </div>
            `;
            messageHistory = [];

            // Request new conversation
            ws.send(JSON.stringify({ action: 'new_conversation' }));
        }

        function loadConversation(conversationId) {
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                console.error('WebSocket not connected');
                return;
            }

            ws.send(JSON.stringify({
                action: 'load_conversation',
                conversation_id: conversationId
            }));
        }

        function deleteConversation(conversationId, event) {
            if (event) {
                event.stopPropagation();
            }

            if (!ws || ws.readyState !== WebSocket.OPEN) {
                console.error('WebSocket not connected');
                return;
            }

            if (confirm('Delete this conversation?')) {
                ws.send(JSON.stringify({
                    action: 'delete_conversation',
                    conversation_id: conversationId
                }));
            }
        }

        function updateConversationList() {
            const listContainer = document.getElementById('conversationsList');

            if (conversations.length === 0) {
                listContainer.innerHTML = `
                    <div style="padding: 20px; text-align: center; color: #666; font-size: 0.85em;">
                        No conversations yet.<br>Start chatting!
                    </div>
                `;
                return;
            }

            listContainer.innerHTML = conversations.map(conv => {
                const isActive = conv.id === currentConversationId;
                const timestamp = formatTimestamp(conv.updated_at);
                const msgCount = conv.message_count || 0;

                return `
                    <div class="conversation-item ${isActive ? 'active' : ''}"
                         onclick="loadConversation(${conv.id})">
                        <button class="conversation-delete"
                                onclick="deleteConversation(${conv.id}, event)">
                            ×
                        </button>
                        <div class="conversation-title">${escapeHtml(conv.title)}</div>
                        <div class="conversation-meta">
                            <span>${timestamp}</span>
                            <span>${msgCount} msgs</span>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function formatTimestamp(timestamp) {
            const date = new Date(timestamp);
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / 60000);
            const diffHours = Math.floor(diffMs / 3600000);
            const diffDays = Math.floor(diffMs / 86400000);

            if (diffMins < 1) return 'Just now';
            if (diffMins < 60) return `${diffMins}m ago`;
            if (diffHours < 24) return `${diffHours}h ago`;
            if (diffDays < 7) return `${diffDays}d ago`;

            return date.toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric'
            });
        }

        // Connect on load
        connect();
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("  >> Starting HoloLoom Agentic Dashboard")
    print("="*60)
    print("\n  Open your browser to: http://localhost:8002\n")

    uvicorn.run("hololoom.apps.workflow_builder.agentic_server:app", host="0.0.0.0", port=8002, log_level="info")

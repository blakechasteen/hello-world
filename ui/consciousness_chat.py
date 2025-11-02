"""
Learner Chat Interface

Natural conversational UI for the consciousness stack with:
- Chat history with messages
- Streaming responses (simulated)
- Context inspector showing what's happening under the hood
- Settings panel for configuration
- Much more intuitive than form-based UI

Run: python ui/consciousness_chat.py
"""

import gradio as gr
import asyncio
import json
from datetime import datetime
from pathlib import Path
import sys
import time
import uuid
import pickle
from typing import Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Session storage for chat persistence
SESSIONS_DIR = Path(__file__).parent / ".chat_sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

from ui.consciousness_ui_simple import (
    DemoMemoryBackend,
    SimpleAwareness,
    SimpleMemoryFusion,
    SimpleContextPacker,
    SimpleDualStream,
)

# Try to import real LLM components (directly, avoiding heavy imports)
LLM_AVAILABLE = False
OllamaLLM = None

try:
    # Import just the LLM integration without full HoloLoom stack
    import importlib.util
    
    # Load llm_integration module
    llm_spec = importlib.util.spec_from_file_location(
        "llm_integration", 
        Path(__file__).parent.parent / "HoloLoom" / "awareness" / "llm_integration.py"
    )
    llm_module = importlib.util.module_from_spec(llm_spec)
    llm_spec.loader.exec_module(llm_module)
    OllamaLLM = llm_module.OllamaLLM
    LLM_AVAILABLE = True
    print("OK: LLM integration loaded")
except Exception as e:
    print(f"WARNING: LLM integration unavailable: {e}")


# Session persistence helpers
def get_session_id(request: gr.Request = None) -> str:
    """Get or create session ID from request"""
    # Use Gradio's session hash if available
    if request and hasattr(request, 'session_hash'):
        return request.session_hash
    # Fallback to creating new session
    session_id = str(uuid.uuid4())
    print(f"Created new session ID: {session_id}")
    return session_id


def save_chat_session(session_id: str, history: List, settings: Dict):
    """Save chat history and settings to disk"""
    session_file = SESSIONS_DIR / f"{session_id}.pkl"
    try:
        with open(session_file, 'wb') as f:
            pickle.dump({
                'history': history,
                'settings': settings,
                'timestamp': datetime.now().isoformat()
            }, f)
        print(f"Saved session {session_id} with {len(history)} messages")
    except Exception as e:
        print(f"WARNING: Failed to save session: {e}")


def load_chat_session(session_id: str) -> tuple:
    """Load chat history and settings from disk"""
    session_file = SESSIONS_DIR / f"{session_id}.pkl"
    if session_file.exists():
        try:
            with open(session_file, 'rb') as f:
                data = pickle.load(f)
                history = data.get('history', [])
                settings = data.get('settings', {})
                print(f"Loaded session {session_id} with {len(history)} messages")
                return history, settings
        except Exception as e:
            print(f"WARNING: Failed to load session: {e}")
    else:
        print(f"No existing session file for {session_id}")
    return [], {}


class ChatSession:
    """
    Manages chat session with consciousness stack.

    Orchestrates the complete consciousness pipeline:
    1. Awareness analysis (domain classification, confidence scoring)
    2. Memory retrieval (fusion or simple threshold-based)
    3. Context packing (intelligent token budget management)
    4. Response generation (LLM or template-based fallback)

    Attributes:
        backend: Memory backend for knowledge retrieval
        awareness: Awareness analyzer for query understanding
        history: Chat message history
        complexity: Processing complexity level (LITE/FAST/FULL/RESEARCH)
        use_fusion: Enable multi-pass memory fusion
        max_memories: Maximum memories to retrieve
        token_budget: Token budget for context packing
        llm: Optional LLM instance (Ollama) for generation

    Notes:
        - Auto-detects LLM availability at initialization
        - Falls back gracefully to templates if LLM unavailable
        - Supports runtime configuration changes
    """

    def __init__(self):
        """
        Initialize chat session with default settings.

        Sets up:
        - Demo memory backend (in-memory)
        - Simple awareness analyzer
        - Default complexity settings
        - Optional LLM connection (Ollama)

        Notes:
            - LLM initialization is non-fatal (falls back to templates)
            - Prints status messages for LLM availability
        """
        self.backend = DemoMemoryBackend()
        self.awareness = SimpleAwareness()
        self.history = []

        # Default settings
        self.complexity = "FULL"
        self.use_fusion = True
        self.max_memories = 10
        self.token_budget = 4000

        # ============================================================
        # LLM Initialization (Optional)
        # ============================================================
        self.llm = None
        if LLM_AVAILABLE and OllamaLLM:
            try:
                # Available models: llama3.2:3b (fast), deepseek-r1:8b (reasoning), 
                # blakechasteen/slama:latest (custom), gpt-oss:120b (massive)
                self.llm = OllamaLLM(model="deepseek-r1:8b")
                if self.llm.is_available():
                    print("OK: Ollama LLM connected! Using deepseek-r1:8b (reasoning model)")
                else:
                    self.llm = None
                    print("WARNING: Ollama not responding, using templates")
            except Exception as e:
                self.llm = None
                print(f"WARNING: LLM setup failed: {e}, using templates")

    def _build_fusion_details(self, memories, enabled: bool, max_passes: int) -> dict:
        """
        Build fusion details dictionary for metadata.

        Args:
            memories: Retrieved memory nodes
            enabled: Whether fusion was enabled
            max_passes: Maximum fusion passes used

        Returns:
            dict: Fusion metadata with statistics

        Notes:
            - Handles both MemoryNode objects and dict items
            - Computes max depth and average score
        """
        return {
            "enabled": enabled,
            "memories_retrieved": len(memories),
            "max_depth": max((getattr(n, 'retrieval_depth', 0) for n in memories), default=0),
            "avg_score": sum(getattr(n, 'composite_score', getattr(n, 'relevance', 0)) for n in memories) / len(memories) if memories else 0,
            "passes": max_passes
        }

    async def _generate_response(self, message: str, packed_context: str):
        """
        Generate response using LLM or template fallback.

        Args:
            message: User's message
            packed_context: Packed context text

        Returns:
            tuple: (internal_reasoning, external_response, generation_time_ms)

        Notes:
            - Tries LLM first if available
            - Falls back to templates on any error
            - Logs generation method used
        """
        generator = SimpleDualStream()

        # Try LLM if available
        if self.llm:
            try:
                # Build prompt with context
                prompt = f"""Based on the following context, answer the user's question naturally and conversationally.

Context:
{packed_context[:2000]}

Question: {message}

Provide a helpful, friendly response. Be conversational like you're chatting with someone."""

                llm_response = await self.llm.generate(
                    prompt=prompt,
                    system_prompt="You are a helpful AI assistant. Be friendly, conversational, and concise.",
                    max_tokens=1500,  # Increased from 500 to allow complete answers
                    temperature=0.7
                )

                # DeepSeek-R1 model includes reasoning in the response
                # Try to extract thinking process if present
                full_response = llm_response.content

                # Check if response has <think> tags (some reasoning models use this)
                if '<think>' in full_response and '</think>' in full_response:
                    # Extract thinking and final answer
                    think_start = full_response.find('<think>') + 7
                    think_end = full_response.find('</think>')
                    thinking = full_response[think_start:think_end].strip()
                    answer = full_response[think_end + 8:].strip()
                    internal_reasoning = f"DeepSeek-R1 Reasoning:\n{thinking}"
                    external_answer = answer
                else:
                    # No explicit tags, use full response as both (reasoning model shows work)
                    # For DeepSeek-R1, the full output IS the reasoning
                    internal_reasoning = f"DeepSeek-R1 Full Response (includes reasoning):\n{full_response}"
                    external_answer = full_response

                return (
                    internal_reasoning,
                    external_answer,
                    0  # TODO: Track actual time
                )
            except Exception as e:
                print(f"WARNING: LLM generation failed: {e}, using templates")

        # Fallback to templates
        internal, external, gen_time = await generator.generate(message, packed_context)
        return internal, external, gen_time

    def _build_response_metadata(self, awareness_ctx, fusion_details, packed, start_time, gen_time) -> dict:
        """
        Build complete response metadata.

        Args:
            awareness_ctx: Awareness analysis context
            fusion_details: Memory fusion details
            packed: Packed context result
            start_time: Pipeline start time (perf_counter)
            gen_time: Generation time in ms

        Returns:
            dict: Complete metadata with awareness, memory, packing, performance

        Notes:
            - Calculates total pipeline time
            - Includes token usage percentage
            - All timing in milliseconds
        """
        total_time = (time.perf_counter() - start_time) * 1000

        return {
            "awareness": {
                "confidence": awareness_ctx.confidence,
                "domain": f"{awareness_ctx.domain}/{awareness_ctx.subdomain}",
                "is_question": awareness_ctx.is_question
            },
            "memory": fusion_details,
            "packing": {
                "total_tokens": packed.total_tokens,
                "budget": self.token_budget,
                "usage_pct": (packed.total_tokens / (self.token_budget * 0.65)) * 100,
                "elements_included": packed.elements_included,
                "elements_compressed": packed.elements_compressed,
                "avg_importance": packed.avg_importance
            },
            "performance": {
                "total_ms": total_time,
                "packing_ms": packed.packing_time_ms,
                "generation_ms": gen_time
            }
        }

    async def process_message(self, message: str):
        """
        Process user message through complete consciousness pipeline.

        Args:
            message: User's chat message text

        Returns:
            dict: Response with message, internal_reasoning, and metadata

        Pipeline Steps:
            1. Awareness Analysis - Classify domain, assess confidence
            2. Memory Retrieval - Fusion (multipass) or simple threshold
            3. Context Packing - Optimize for token budget
            4. Response Generation - LLM (if available) or templates

        Response Format:
            {
                'message': str (external response),
                'internal_reasoning': str (thought process),
                'metadata': {
                    'awareness': {...},
                    'memory': {...},
                    'packing': {...},
                    'performance': {...}
                }
            }

        Notes:
            - Uses current session settings (complexity, fusion, budget)
            - LLM generation with automatic template fallback
            - Tracks performance metrics at each stage
        """

        # ============================================================
        # Validation
        # ============================================================
        if not message or not isinstance(message, str):
            raise ValueError("ERROR: Message must be a non-empty string")

        if not message.strip():
            raise ValueError("ERROR: Message cannot be whitespace only")

        # Validate complexity setting
        valid_complexity = ["LITE", "FAST", "FULL", "RESEARCH"]
        if self.complexity not in valid_complexity:
            print(f"WARNING: Invalid complexity '{self.complexity}', using 'FULL'")
            self.complexity = "FULL"

        start_time = time.perf_counter()

        # ============================================================
        # 1. Awareness Analysis
        # ============================================================
        try:
            awareness_ctx = await self.awareness.analyze(message)
        except Exception as e:
            print(f"WARNING: Awareness analysis failed: {e}, using defaults")
            # Create minimal awareness context
            from dataclasses import dataclass
            @dataclass
            class MinimalAwareness:
                confidence: float = 0.5
                domain: str = "general"
                subdomain: str = "unknown"
                is_question: bool = "?" in message
            awareness_ctx = MinimalAwareness()

        # ============================================================
        # 2. Memory Retrieval (Fusion or Simple)
        # ============================================================
        memories = []
        fusion_details = {}

        try:
            if self.use_fusion:
                max_passes = {"LITE": 1, "FAST": 2, "FULL": 3, "RESEARCH": 4}.get(self.complexity, 3)
                fusion = SimpleMemoryFusion(self.backend, max_passes=max_passes)
                memories = await fusion.retrieve_with_fusion(message, max_results=self.max_memories)
                fusion_details = self._build_fusion_details(memories, enabled=True, max_passes=max_passes)
            else:
                # Simple threshold-based retrieval
                items = await self.backend.retrieve_with_threshold(message, threshold=0.6, limit=self.max_memories)
                from ui.consciousness_ui_simple import MemoryNode
                memories = [
                    MemoryNode(
                        content=item['content'],
                        relevance=item['relevance'],
                        composite_score=item['relevance'],
                        retrieval_depth=0,
                        timestamp=item['timestamp'],
                        item_id=item.get('id', '')  # Include item_id
                    )
                    for item in items
                ]
                fusion_details = self._build_fusion_details(memories, enabled=False, max_passes=1)
        except Exception as e:
            print(f"WARNING: Memory retrieval failed: {e}, continuing without memories")
            fusion_details = self._build_fusion_details([], enabled=self.use_fusion, max_passes=1)

        # ============================================================
        # 3. Context Packing
        # ============================================================
        try:
            packer = SimpleContextPacker(token_budget=self.token_budget)
            packed = await packer.pack(message, awareness_ctx, memories)
        except Exception as e:
            print(f"WARNING: Context packing failed: {e}, using minimal context")
            # Create minimal packed context
            from dataclasses import dataclass
            @dataclass
            class MinimalPacked:
                formatted_text: str = message
                total_tokens: int = len(message.split())
                elements_included: int = 0
                elements_compressed: int = 0
                avg_importance: float = 0.0
                packing_time_ms: float = 0.0
            packed = MinimalPacked()

        # ============================================================
        # 4. Response Generation (LLM or Templates)
        # ============================================================
        try:
            internal, external, gen_time = await self._generate_response(message, packed.formatted_text)
        except Exception as e:
            print(f"ERROR: Response generation failed: {e}")
            internal = f"Error during generation: {str(e)}"
            external = "I apologize, but I encountered an error generating a response. Please try again."
            gen_time = 0.0

        # ============================================================
        # 5. Build Response with Metadata
        # ============================================================
        try:
            metadata = self._build_response_metadata(
                awareness_ctx, fusion_details, packed, start_time, gen_time
            )
        except Exception as e:
            print(f"WARNING: Metadata building failed: {e}, using minimal metadata")
            metadata = {
                "awareness": {"confidence": 0.0, "domain": "unknown/unknown", "is_question": False},
                "memory": {"enabled": False, "memories_retrieved": 0, "max_depth": 0, "avg_score": 0.0, "passes": 0},
                "packing": {"total_tokens": 0, "budget": self.token_budget, "usage_pct": 0.0, "elements_included": 0, "elements_compressed": 0, "avg_importance": 0.0},
                "performance": {"total_ms": (time.perf_counter() - start_time) * 1000, "packing_ms": 0.0, "generation_ms": gen_time}
            }

        return {
            "message": external,
            "internal_reasoning": internal,
            "metadata": metadata
        }


# Global session
session = ChatSession()


async def chat_async(message, history, complexity, use_fusion, max_memories, token_budget, session_id):
    """
    Process chat message through consciousness stack with session persistence.

    Args:
        message: User's message text
        session_id: Unique session identifier for persistence
        history: Chat history (list of message dicts)
        complexity: Processing complexity (LITE/FAST/FULL/RESEARCH)
        use_fusion: Enable multi-pass memory fusion
        max_memories: Maximum memories to retrieve
        token_budget: Token budget for context packing

    Returns:
        tuple: (updated_history, context_info_markdown)

    Notes:
        - Updates session settings dynamically
        - Returns empty context if message is empty
        - Catches and logs all processing errors
    """

    # Validation
    if not message or not message.strip():
        return history, ""

    # Update session settings
    session.complexity = complexity
    session.use_fusion = use_fusion
    session.max_memories = max_memories
    session.token_budget = token_budget

    # Add user message to history
    history.append({"role": "user", "content": message})

    # Process through consciousness stack
    try:
        response = await session.process_message(message)
    except Exception as e:
        print(f"ERROR: Chat processing failed: {e}")
        # Return error message to user
        error_response = {
            "message": f"I encountered an error: {str(e)}. Please try again.",
            "internal_reasoning": f"Error: {str(e)}",
            "metadata": {
                "awareness": {"confidence": 0.0, "domain": "error/error", "is_question": False},
                "memory": {"enabled": False, "memories_retrieved": 0, "max_depth": 0, "avg_score": 0.0, "passes": 0},
                "packing": {"total_tokens": 0, "budget": token_budget, "usage_pct": 0.0, "elements_included": 0, "elements_compressed": 0, "avg_importance": 0.0},
                "performance": {"total_ms": 0.0, "packing_ms": 0.0, "generation_ms": 0.0}
            }
        }
        response = error_response

    # Add assistant response to history
    history.append({
        "role": "assistant",
        "content": response["message"],
        "metadata": response["metadata"]
    })
    
    # Format context inspector
    meta = response["metadata"]
    context_info = f"""### 🔍 Under the Hood

**Awareness**
- Confidence: {meta['awareness']['confidence']:.0%}
- Domain: {meta['awareness']['domain']}
- Question: {'Yes' if meta['awareness']['is_question'] else 'No'}

**Memory** {'(Fusion Enabled)' if meta['memory']['enabled'] else '(Simple Retrieval)'}
- Retrieved: {meta['memory']['memories_retrieved']} memories
- Max Depth: {meta['memory']['max_depth']} hops
- Avg Score: {meta['memory']['avg_score']:.3f}
- Passes: {meta['memory']['passes']}

**Context Packing**
- Tokens: {meta['packing']['total_tokens']}/{meta['packing']['budget']} ({meta['packing']['usage_pct']:.1f}%)
- Included: {meta['packing']['elements_included']}, Compressed: {meta['packing']['elements_compressed']}
- Avg Importance: {meta['packing']['avg_importance']:.0%}

**Performance**
- Total: {meta['performance']['total_ms']:.2f}ms
- Packing: {meta['performance']['packing_ms']:.2f}ms
- Generation: {meta['performance']['generation_ms']:.2f}ms

---

**Internal Reasoning**
```
{response['internal_reasoning'][:300]}...
```
"""

    # Save session to disk for persistence across refreshes
    save_chat_session(session_id, history, {
        'complexity': complexity,
        'use_fusion': use_fusion,
        'max_memories': max_memories,
        'token_budget': token_budget
    })

    return history, context_info


def chat(message, history, complexity, use_fusion, max_memories, token_budget, session_id):
    """Sync wrapper"""
    return asyncio.run(chat_async(message, history, complexity, use_fusion, max_memories, token_budget, session_id))


def load_session(session_id):
    """Load chat session on page load"""
    history, settings = load_chat_session(session_id)

    # Return history and settings
    return (
        history,
        settings.get('complexity', 'FULL'),
        settings.get('use_fusion', True),
        settings.get('max_memories', 10),
        settings.get('token_budget', 4000)
    )


def clear_chat(session_id):
    """Clear chat history and delete session"""
    session_file = SESSIONS_DIR / f"{session_id}.pkl"
    if session_file.exists():
        session_file.unlink()  # Delete session file
    # Return new session ID along with cleared history
    new_session_id = str(uuid.uuid4())
    return [], "", new_session_id


# Build Gradio Chat UI
with gr.Blocks(
    title="Learner", 
    theme=gr.themes.Soft(),
    css="""
        /* Force full viewport height */
        body, #root, .gradio-container {
            height: 100vh !important;
            max-height: 100vh !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* Main content fills screen */
        .gradio-container > .contain {
            height: 100vh !important;
            display: flex !important;
            flex-direction: column !important;
        }
        
        /* Chat window takes maximum space and scrolls */
        [class*="chatbot"] {
            min-height: 70vh !important;
            max-height: 70vh !important;
            overflow-y: auto !important;
        }

        /* Ensure chat messages container scrolls */
        [class*="chatbot"] > div {
            overflow-y: auto !important;
        }

        /* Fix scrolling in chat container */
        .gradio-chatbot {
            overflow-y: auto !important;
        }
    """
) as demo:
    # Session state - persists across page loads via Gradio
    session_id_state = gr.State(value=lambda: str(uuid.uuid4()))

    with gr.Row():
        # Left column: Chat (main area)
        with gr.Column(scale=3):
            # Avatar options:
            # 1. Emoji: "🐻‍❄️" or "🧠" 
            # 2. Local file: "./ui/assets/avatar.jpg" (save image there)
            # 3. URL: "https://your-url.com/image.jpg"
            chatbot = gr.Chatbot(
                label="💬 Learner",
                height=700,  # Tall but scrollable
                type="messages",
                avatar_images=(None, "🐻‍❄️"),  # Polar bear emoji!
                show_copy_button=True,
                autoscroll=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask me anything... (Enter or Ctrl+Enter to send)",
                    scale=4,
                    lines=2,
                    show_label=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", size="sm")
                toggle_settings_btn = gr.Button("⚙️ Settings", size="sm", variant="secondary")
                gr.Markdown("*⌨️ Enter or Ctrl+Enter to send*", elem_classes="hint")
        
        # Right column: Collapsible Settings + Context
        with gr.Column(scale=1, visible=True) as settings_panel:
            gr.Markdown("### ⚙️ Settings")
            
            complexity_input = gr.Radio(
                choices=["LITE", "FAST", "FULL", "RESEARCH"],
                value="FULL",
                label="Complexity",
                info="LITE: Fast, RESEARCH: Deep"
            )
            
            use_fusion_input = gr.Checkbox(
                label="Enable Memory Fusion",
                value=True,
                info="Multipass graph crawling"
            )
            
            max_memories_input = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="Max Memories"
            )
            
            token_budget_input = gr.Slider(
                minimum=2000,
                maximum=8000,
                value=4000,
                step=500,
                label="Token Budget"
            )
            
            gr.Markdown("---")
            gr.Markdown("### 🔍 Context Inspector")
            
            context_output = gr.Markdown(
                value="*Send a message to see what's happening under the hood...*",
                label="Pipeline Details"
            )
    
    # Example questions
    gr.Markdown("---")
    gr.Markdown("### 💡 Try These Questions")
    
    gr.Examples(
        examples=[
            ["What are the applications of quantum computing?"],
            ["How does quantum entanglement work?"],
            ["Explain quantum teleportation in simple terms"],
            ["What are the main challenges in building quantum computers?"],
            ["Can quantum computers break current encryption?"],
        ],
        inputs=[msg],
    )
    
    # Event handlers
    def respond(message, history, complexity, use_fusion, max_memories, token_budget, session_id):
        if not message.strip():
            return history, "", context_output.value

        # Call async function
        new_history, context = chat(message, history, complexity, use_fusion, max_memories, token_budget, session_id)
        return new_history, "", context
    
    send_btn.click(
        fn=respond,
        inputs=[msg, chatbot, complexity_input, use_fusion_input, max_memories_input, token_budget_input, session_id_state],
        outputs=[chatbot, msg, context_output]
    )

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, complexity_input, use_fusion_input, max_memories_input, token_budget_input, session_id_state],
        outputs=[chatbot, msg, context_output]
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[session_id_state],
        outputs=[chatbot, context_output, session_id_state]
    )
    
    # Toggle settings panel visibility
    settings_visible = gr.State(True)
    
    def toggle_settings(visible):
        """Toggle visibility of settings panel"""
        return not visible, gr.update(visible=not visible)
    
    toggle_settings_btn.click(
        fn=toggle_settings,
        inputs=[settings_visible],
        outputs=[settings_visible, settings_panel]
    )

    # Load session on page load
    demo.load(
        fn=load_session,
        inputs=[session_id_state],
        outputs=[chatbot, complexity_input, use_fusion_input, max_memories_input, token_budget_input]
    )

    gr.Markdown("""
    ---
    
    **Features**:
    - 💬 Natural conversation flow with full-screen option
    - 🔍 Collapsible context inspection
    - ⚙️ Live configuration changes
    - 🧠 Full consciousness stack integration
    - 📊 Performance metrics per message
    
    **How it works**:
    1. Your message → Awareness analysis
    2. Memory fusion retrieves relevant knowledge
    3. Context packing optimizes for token budget
    4. Dual-stream generation creates response
    5. Toggle settings to see all details →
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LEARNER CHAT INTERFACE".center(80))
    print("=" * 60 + "\n")
    print("Starting chat server...")
    print("Open your browser to have a conversation!\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

"""
HoloLoom Lite - Desktop App

A Gradio-based desktop application for HoloLoom Lite.

Requirements:
    pip install gradio

Usage:
    python -m HoloLoom.lite desktop

Features:
    - Chat interface with history
    - Mode selector (Direct/Verify/Research/Plan-Execute)
    - Metrics panel
    - Memory browser
    - Export/Import functionality

Date: December 2025
"""

import asyncio

# Global loom instance
_loom = None


def start_desktop(share: bool = False, **kwargs) -> None:
    """Start the Gradio desktop app."""
    try:
        import gradio as gr
    except ImportError:
        print("Desktop app requires 'gradio' package.")
        print("Install with: pip install gradio")
        return

    print("\nStarting HoloLoom Lite Desktop App...")
    print("Press Ctrl+C to stop\n")

    # Initialize loom synchronously
    asyncio.run(_init_loom())

    # Create and launch app
    app = create_app()
    app.launch(share=share, **kwargs)


async def _init_loom():
    """Initialize the global loom instance."""
    global _loom
    from hololoom.lite import HoloLoomLite
    _loom = HoloLoomLite()
    await _loom._initialize()


def _run_async(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If there's a running loop, create a new one
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)


def create_app():
    """Create the Gradio application."""
    import gradio as gr

    # Wrapper functions that handle async
    def chat(message: str, history: list[tuple[str, str]], mode: str) -> tuple[str, list[tuple[str, str]]]:
        """Handle chat message."""
        global _loom
        if _loom is None:
            return "HoloLoom not initialized", history

        async def _process():
            # Store experience
            await _loom.experience(message, context={"type": "query", "source": "desktop"})

            # Reason based on mode
            result = await _loom.reason(message, mode=mode.lower().replace("-", "_"))
            return result

        try:
            result = _run_async(_process())
            response = f"{result.text}\n\n[Confidence: {result.confidence:.0%}]"
            history = history + [(message, response)]
            return "", history
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history = history + [(message, error_msg)]
            return "", history

    def get_metrics() -> str:
        """Get system metrics."""
        global _loom
        if _loom is None:
            return "Not initialized"

        metrics = _loom.get_metrics()
        return f"""Memories: {metrics['n_memories']}
Connections: {metrics['n_connections']}
Initialized: {metrics['initialized']}"""

    def get_memories() -> str:
        """Get recent memories."""
        global _loom
        if _loom is None:
            return "Not initialized"

        async def _get():
            memories = await _loom.recall("", limit=20)
            if not memories:
                return "No memories yet"
            return "\n\n".join([
                f"**{m.id[:8]}...**\n{m.text}"
                for m in memories
            ])

        try:
            return _run_async(_get())
        except Exception as e:
            return f"Error: {str(e)}"

    def store_memory(text: str) -> str:
        """Store explicit memory."""
        global _loom
        if _loom is None:
            return "Not initialized"
        if not text.strip():
            return "Enter some text to store"

        async def _store():
            mem = await _loom.experience(text)
            return f"Stored: {mem.id}"

        try:
            return _run_async(_store())
        except Exception as e:
            return f"Error: {str(e)}"

    # Build the UI
    with gr.Blocks(
        title="HoloLoom Lite",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
            .main-header { text-align: center; margin-bottom: 20px; }
            .metrics-panel { background: var(--background-fill-secondary); padding: 12px; border-radius: 8px; }
        """
    ) as app:
        # Header
        gr.Markdown(
            """
            # 🧠 HoloLoom Lite
            *Simplified AI with memory*
            """,
            elem_classes=["main-header"]
        )

        with gr.Row():
            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=400,
                    show_copy_button=True
                )

                with gr.Row():
                    mode = gr.Radio(
                        choices=["Direct", "Verify", "Research", "Plan-Execute"],
                        value="Direct",
                        label="Reasoning Mode",
                        info="Choose how HoloLoom processes your query"
                    )

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask anything...",
                        label="Message",
                        scale=4,
                        show_label=False
                    )
                    submit = gr.Button("Send", variant="primary", scale=1)

            # Sidebar
            with gr.Column(scale=1):
                with gr.Tab("Metrics"):
                    metrics_display = gr.Textbox(
                        label="System Metrics",
                        lines=5,
                        interactive=False
                    )
                    refresh_btn = gr.Button("Refresh Metrics", size="sm")

                with gr.Tab("Memories"):
                    memories_display = gr.Markdown(label="Recent Memories")
                    refresh_mem_btn = gr.Button("Refresh Memories", size="sm")

                with gr.Tab("Store"):
                    store_input = gr.Textbox(
                        label="New Memory",
                        placeholder="Enter text to store as memory...",
                        lines=3
                    )
                    store_btn = gr.Button("Store Memory", size="sm")
                    store_result = gr.Textbox(
                        label="Result",
                        interactive=False,
                        lines=1
                    )

        # Mode descriptions
        gr.Markdown("""
        **Modes:**
        - **Direct**: Quick single-pass answer (~150ms)
        - **Verify**: Answer with verification (~600ms)
        - **Research**: Multi-query exploration (~900ms)
        - **Plan-Execute**: Goal decomposition (~750ms)
        """)

        # Event handlers
        msg.submit(chat, [msg, chatbot, mode], [msg, chatbot])
        submit.click(chat, [msg, chatbot, mode], [msg, chatbot])
        refresh_btn.click(get_metrics, outputs=metrics_display)
        refresh_mem_btn.click(get_memories, outputs=memories_display)
        store_btn.click(store_memory, inputs=store_input, outputs=store_result)

        # Load initial metrics
        app.load(get_metrics, outputs=metrics_display)

    return app


if __name__ == "__main__":
    start_desktop()

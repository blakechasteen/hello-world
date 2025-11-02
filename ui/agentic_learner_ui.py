#!/usr/bin/env python3
"""
Agentic Learner UI - Gradio Interface
======================================
Beautiful chat interface with 4 agentic reasoning modes.

Features:
- Real-time chat with LLM
- 4 reasoning modes (DIRECT, VERIFY, RESEARCH, PLAN_EXECUTE)
- Verification inspector showing contradictions
- Memory browser showing what the system knows
- Settings panel for configuration

Run:
    python ui/agentic_learner_ui.py

Then open: http://localhost:7860
"""

import gradio as gr
import asyncio
import requests
import json
from datetime import datetime
from typing import List, Tuple, Dict
import time

# Server configuration
SERVER_URL = "http://localhost:8001"  # Updated to match start_agentic_server.py


# ============================================================================
# API Client
# ============================================================================

def check_server_health():
    """Check if server is running."""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def query_server(text: str, mode: str = "verify", max_steps: int = 5) -> Dict:
    """Send query to server."""
    try:
        response = requests.post(
            f"{SERVER_URL}/query",
            json={
                "text": text,
                "mode": mode.lower(),
                "max_steps": max_steps
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {
            "error": str(e),
            "response": f"Error: {str(e)}",
            "confidence": 0.0
        }


def add_memory(text: str, entities: List[str], motifs: List[str]):
    """Add memory to server."""
    try:
        response = requests.post(
            f"{SERVER_URL}/memories/add",
            json={
                "text": text,
                "entities": entities,
                "motifs": motifs
            }
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_stats():
    """Get server statistics."""
    try:
        response = requests.get(f"{SERVER_URL}/stats")
        return response.json()
    except:
        return {}


# ============================================================================
# Chat Interface
# ============================================================================

def chat(
    message: str,
    history: List[Tuple[str, str]],
    mode: str,
    max_steps: int
) -> Tuple[List[Tuple[str, str]], Dict, str]:
    """
    Process chat message with agentic reasoning.

    Returns:
        (updated_history, verification_info, status_message)
    """
    if not message.strip():
        return history, {}, "Please enter a message"

    # Show thinking status
    status = f"🤔 Processing with {mode} mode..."

    # Send to server
    result = query_server(message, mode, max_steps)

    if "error" in result:
        error_msg = f"❌ Server error: {result['error']}"
        history.append((message, f"Error: {result['error']}"))
        return history, {}, error_msg

    # Extract response
    response_text = result.get("response", "No response")
    confidence = result.get("confidence", 0.0)
    reasoning_mode = result.get("reasoning_mode", mode)
    total_queries = result.get("total_queries", 1)
    duration_ms = result.get("total_duration_ms", 0)
    llm_provider = result.get("llm_provider", "unknown")
    llm_model = result.get("llm_model", "unknown")

    # Format response with metadata
    formatted_response = f"{response_text}\n\n"
    formatted_response += f"_Confidence: {confidence:.2%} | "
    formatted_response += f"Mode: {reasoning_mode} | "
    formatted_response += f"Steps: {total_queries} | "
    formatted_response += f"Time: {duration_ms:.0f}ms_\n"
    formatted_response += f"_LLM: {llm_provider}/{llm_model}_"

    # Add to history
    history.append((message, formatted_response))

    # Build verification info
    verification = result.get("verification", {})
    verification_info = {
        "verified": verification.get("verified", True) if verification else True,
        "contradictions": verification.get("contradictions", []) if verification else [],
        "supporting_evidence": verification.get("supporting_evidence", []) if verification else [],
        "mode": reasoning_mode,
        "confidence": confidence
    }

    # Status message
    status = f"✓ Complete ({duration_ms:.0f}ms, {total_queries} steps, confidence: {confidence:.2%})"

    return history, verification_info, status


def format_verification_info(info: Dict) -> str:
    """Format verification info for display."""
    if not info:
        return "No verification data available"

    verified = info.get("verified", True)
    mode = info.get("mode", "unknown")
    confidence = info.get("confidence", 0.0)

    output = f"### Verification Report ({mode} mode)\n\n"
    output += f"**Status**: {'✅ Verified' if verified else '⚠️ Not Verified'}\n"
    output += f"**Confidence**: {confidence:.2%}\n\n"

    contradictions = info.get("contradictions", [])
    if contradictions:
        output += "**⚠️ Contradictions Found:**\n"
        for i, c in enumerate(contradictions, 1):
            output += f"{i}. {c}\n"
        output += "\n"

    supporting = info.get("supporting_evidence", [])
    if supporting:
        output += "**✓ Supporting Evidence:**\n"
        for i, s in enumerate(supporting[:3], 1):
            output += f"{i}. {s[:100]}...\n"

    return output


def get_server_status() -> str:
    """Get server status as formatted string."""
    if not check_server_health():
        return "❌ Server not running\n\nStart with:\n`python HoloLoom/server/agentic_api_integrated.py`"

    stats = get_stats()

    status = "✅ Server Running\n\n"
    status += f"**LLM**: {stats.get('llm_model', 'None')} "
    status += f"({'available' if stats.get('llm_available') else 'unavailable'})\n"
    status += f"**Memory**: {stats.get('memory_backend', 'unknown')} "
    status += f"({stats.get('memories_loaded', 0)} memories)\n"
    status += f"**Audit Trail**: {stats.get('audit_trail_entries', 0)} entries\n"

    return status


# ============================================================================
# Gradio UI
# ============================================================================

def build_ui():
    """Build Gradio interface."""

    with gr.Blocks(title="HoloLoom Agentic Learner", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 HoloLoom Agentic Learner

        Intelligent chat with 4 reasoning modes:
        - **DIRECT** (~150ms): Fast single-pass answers
        - **VERIFY** (~600ms): Checks for contradictions
        - **RESEARCH** (~900ms): Multi-query exploration
        - **PLAN_EXECUTE** (~750ms): Goal decomposition
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=True
                )

                with gr.Row():
                    message_box = gr.Textbox(
                        placeholder="Ask me anything...",
                        show_label=False,
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                status_box = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=False
                )

                # Settings
                with gr.Accordion("⚙️ Settings", open=False):
                    mode_dropdown = gr.Dropdown(
                        choices=["DIRECT", "VERIFY", "RESEARCH", "PLAN_EXECUTE"],
                        value="VERIFY",
                        label="Reasoning Mode",
                        info="How the system should process queries"
                    )

                    max_steps_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Max Steps",
                        info="Maximum reasoning steps for multi-step modes"
                    )

            with gr.Column(scale=1):
                # Verification inspector
                gr.Markdown("### 🔍 Verification Inspector")
                verification_display = gr.Markdown(
                    value="_No verification data yet_",
                    label="Verification Info"
                )

                # Server status
                gr.Markdown("### 📊 Server Status")
                server_status = gr.Markdown(
                    value=get_server_status(),
                    label="Status"
                )

                refresh_btn = gr.Button("🔄 Refresh Status", size="sm")

                # Memory browser
                with gr.Accordion("💾 Memory Browser", open=False):
                    gr.Markdown("View and add memories")

                    add_memory_text = gr.Textbox(
                        label="Memory Text",
                        placeholder="The system will remember this..."
                    )

                    add_memory_entities = gr.Textbox(
                        label="Entities (comma-separated)",
                        placeholder="concept1, concept2"
                    )

                    add_memory_btn = gr.Button("Add Memory", size="sm")
                    memory_status = gr.Textbox(label="Status", interactive=False)

        # State
        verification_state = gr.State({})

        # Event handlers
        def send_message(msg, history, mode, steps):
            new_history, verification, status = chat(msg, history, mode, steps)
            verification_md = format_verification_info(verification)
            return new_history, "", verification_md, status

        send_btn.click(
            send_message,
            inputs=[message_box, chatbot, mode_dropdown, max_steps_slider],
            outputs=[chatbot, message_box, verification_display, status_box]
        )

        message_box.submit(
            send_message,
            inputs=[message_box, chatbot, mode_dropdown, max_steps_slider],
            outputs=[chatbot, message_box, verification_display, status_box]
        )

        refresh_btn.click(
            get_server_status,
            outputs=server_status
        )

        def add_memory_handler(text, entities):
            if not text.strip():
                return "Please enter memory text"

            entity_list = [e.strip() for e in entities.split(",") if e.strip()]
            result = add_memory(text, entity_list, [])

            if "error" in result:
                return f"Error: {result['error']}"
            else:
                return f"✓ Added memory: {result.get('id')}"

        add_memory_btn.click(
            add_memory_handler,
            inputs=[add_memory_text, add_memory_entities],
            outputs=memory_status
        )

        # Examples
        gr.Examples(
            examples=[
                ["What is Thompson Sampling?", "DIRECT"],
                ["Is Thompson Sampling always optimal?", "VERIFY"],
                ["Compare Thompson Sampling vs UCB", "RESEARCH"],
                ["Implement a multi-armed bandit", "PLAN_EXECUTE"],
            ],
            inputs=[message_box, mode_dropdown],
        )

    return demo


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("HoloLoom Agentic Learner UI")
    print("="*80)
    print("\nChecking server...")

    if not check_server_health():
        print("\n❌ Server not running!")
        print("\nPlease start the server first:")
        print("  python HoloLoom/server/agentic_api_integrated.py")
        print("\nThen run this UI again.\n")
        import sys
        sys.exit(1)

    print("✅ Server is running!\n")

    stats = get_stats()
    print(f"LLM: {stats.get('llm_model', 'None')}")
    print(f"Memory: {stats.get('memory_backend', 'unknown')} ({stats.get('memories_loaded', 0)} memories)")
    print(f"\nStarting UI...")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

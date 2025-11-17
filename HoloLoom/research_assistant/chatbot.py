"""
Research Assistant Chatbot
==========================

Interactive Streamlit interface for the Personal Research Assistant.

**Features**:
- PDF upload and ingestion
- Chat-based Q&A across papers
- Pattern/Contradiction/Gap visualization
- Source attribution
- LLM-powered response generation

**Usage**:
    streamlit run HoloLoom/research_assistant/chatbot.py

Author: HoloLoom Team
Date: 2025-11-17
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import warnings

# Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    warnings.warn("Streamlit not available. Install: pip install streamlit")

# Research Assistant components
from HoloLoom.research_assistant.pdf_ingestion import (
    PDFIngestionPipeline,
    ResearchPaper,
    quick_ingest
)
from HoloLoom.research_assistant.paper_memory import PaperMemorySystem
from HoloLoom.research_assistant.synthesis_engine import ResearchSynthesisEngine

# LLM integration (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# ============================================================================
# Streamlit Configuration
# ============================================================================

if STREAMLIT_AVAILABLE:
    st.set_page_config(
        page_title="Personal Research Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'memory_system' not in st.session_state:
        st.session_state.memory_system = None

    if 'synthesis_engine' not in st.session_state:
        st.session_state.synthesis_engine = None

    if 'papers' not in st.session_state:
        st.session_state.papers = []

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'initialized' not in st.session_state:
        st.session_state.initialized = False


async def initialize_systems():
    """Initialize memory and synthesis systems."""
    if not st.session_state.initialized:
        # Create systems
        memory = PaperMemorySystem()
        await memory.initialize()

        synthesis = ResearchSynthesisEngine()
        await synthesis.start()

        st.session_state.memory_system = memory
        st.session_state.synthesis_engine = synthesis
        st.session_state.initialized = True


# ============================================================================
# PDF Upload and Ingestion
# ============================================================================

async def ingest_pdf(uploaded_file) -> Dict[str, Any]:
    """
    Ingest uploaded PDF file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Dict with ingestion statistics
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Ingest PDF
    pipeline = PDFIngestionPipeline()
    paper = pipeline.ingest(temp_path)

    # Store in memory
    memory = st.session_state.memory_system
    stats = await memory.ingest_paper(paper)

    # Register with synthesis engine
    synthesis = st.session_state.synthesis_engine
    for section in paper.sections:
        synthesis.register_paper_content(paper.title, section.content)

    # Run synthesis
    kg = memory.get_knowledge_graph()
    await synthesis.synthesize(kg)

    # Track paper
    st.session_state.papers.append(paper)

    return stats


# ============================================================================
# Query Processing
# ============================================================================

async def process_query(query: str) -> Dict[str, Any]:
    """
    Process user query and generate response.

    Args:
        query: User question

    Returns:
        Dict with response and metadata
    """
    memory = st.session_state.memory_system
    synthesis = st.session_state.synthesis_engine

    # Retrieve relevant passages
    results = await memory.query_papers(query, max_results=5)

    # Get synthesis insights
    insights = synthesis.get_insights_for_query(query)

    # Generate response
    if OLLAMA_AVAILABLE:
        response = await generate_llm_response(query, results, insights)
    else:
        response = generate_simple_response(query, results, insights)

    return {
        'response': response,
        'sources': results,
        'insights': insights
    }


def generate_simple_response(
    query: str,
    results: List[Dict[str, Any]],
    insights: Dict[str, List[str]]
) -> str:
    """
    Generate response without LLM (fallback).

    Args:
        query: User query
        results: Retrieved passages
        insights: Synthesis insights

    Returns:
        Formatted response string
    """
    response_parts = []

    # Add retrieved passages
    if results:
        response_parts.append("**Relevant passages:**\n")
        for i, result in enumerate(results[:3], 1):
            response_parts.append(
                f"{i}. From **{result['paper']}**, {result['section']}:\n"
                f"   {result['snippet']}\n"
            )

    # Add patterns
    if insights['patterns']:
        response_parts.append("\n**Discovered patterns:**")
        for pattern in insights['patterns'][:2]:
            response_parts.append(f"- {pattern}")

    # Add contradictions
    if insights['contradictions']:
        response_parts.append("\n**Note: Contradictions detected:**")
        for contradiction in insights['contradictions'][:2]:
            response_parts.append(f"- {contradiction}")

    # Add gaps
    if insights['gaps']:
        response_parts.append("\n**Knowledge gaps:**")
        for gap in insights['gaps'][:2]:
            response_parts.append(f"- {gap}")

    if not response_parts:
        return "No relevant information found in the uploaded papers."

    return "\n".join(response_parts)


async def generate_llm_response(
    query: str,
    results: List[Dict[str, Any]],
    insights: Dict[str, List[str]]
) -> str:
    """
    Generate response using Ollama LLM.

    Args:
        query: User query
        results: Retrieved passages
        insights: Synthesis insights

    Returns:
        LLM-generated response
    """
    # Build context from retrieved passages
    context_parts = []
    for result in results[:5]:
        context_parts.append(
            f"[{result['paper']}, {result['section']}]: {result['content'][:500]}"
        )
    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are a research assistant helping to answer questions about uploaded papers.

Context from papers:
{context}

Discovered patterns:
{chr(10).join(f"- {p}" for p in insights['patterns'][:3])}

Contradictions:
{chr(10).join(f"- {c}" for c in insights['contradictions'][:3])}

Knowledge gaps:
{chr(10).join(f"- {g}" for g in insights['gaps'][:3])}

User question: {query}

Provide a clear, concise answer based on the context. Cite sources when possible."""

    # Call Ollama
    try:
        response = ollama.generate(
            model='llama3.2:3b',
            prompt=prompt
        )
        return response['response']
    except Exception as e:
        warnings.warn(f"LLM generation failed: {e}. Falling back to simple response.")
        return generate_simple_response(query, results, insights)


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render sidebar with PDF upload and paper list."""
    with st.sidebar:
        st.header("📚 Research Papers")

        # Upload section
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a research paper",
            type="pdf",
            help="Upload a PDF research paper to add to your collection"
        )

        if uploaded_file:
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                stats = asyncio.run(ingest_pdf(uploaded_file))
                st.success(f"✅ {stats['paper_title']}")
                st.caption(f"Stored {stats['sections_stored']} sections, "
                          f"{stats['memories_created']} memories")

        # Paper list
        st.subheader("Papers Loaded")
        if st.session_state.papers:
            for i, paper in enumerate(st.session_state.papers, 1):
                with st.expander(f"{i}. {paper.title}", expanded=False):
                    st.caption(f"**Authors:** {', '.join(paper.authors)}")
                    st.caption(f"**Sections:** {len(paper.sections)}")
                    st.caption(f"**Citations:** {len(paper.citations)}")
                    if paper.abstract:
                        st.text(paper.abstract[:200] + "...")
        else:
            st.info("No papers loaded yet. Upload a PDF to get started!")

        # Statistics
        if st.session_state.memory_system and st.session_state.papers:
            st.divider()
            stats = st.session_state.memory_system.get_statistics()
            st.metric("Total Papers", stats['total_papers'])
            st.metric("Total Sections", stats['total_sections'])
            st.metric("Avg Citations", f"{stats['avg_citations_per_paper']:.1f}")


def render_chat_interface():
    """Render main chat interface."""
    st.header("💬 Ask Questions")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📖 Sources"):
                    for source in message["sources"][:3]:
                        st.caption(f"**{source['paper']}** ({source['section']})")
                        st.text(source['snippet'])

    # Chat input
    if prompt := st.chat_input("Ask about your papers..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = asyncio.run(process_query(prompt))

                st.markdown(result['response'])

                # Show sources
                if result['sources']:
                    with st.expander("📖 Sources"):
                        for source in result['sources'][:3]:
                            st.caption(f"**{source['paper']}** ({source['section']})")
                            st.text(source['snippet'])

        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['response'],
            "sources": result['sources']
        })


def render_synthesis_tabs():
    """Render tabs for patterns/contradictions/gaps."""
    st.header("🔍 Synthesis Insights")

    if not st.session_state.synthesis_engine:
        st.info("Upload papers to see synthesis insights")
        return

    tab1, tab2, tab3 = st.tabs(["📊 Patterns", "⚠️ Contradictions", "🕳️ Gaps"])

    synthesis = st.session_state.synthesis_engine

    with tab1:
        st.subheader("Discovered Patterns")
        patterns = synthesis.get_patterns()

        if patterns:
            for i, pattern in enumerate(patterns, 1):
                with st.expander(f"Pattern {i}: {pattern.content[:60]}...", expanded=True):
                    st.write(pattern.content)
                    st.caption(f"**Sources:** {len(pattern.source_memories)} queries")
                    st.caption(f"**Confidence:** {pattern.confidence:.2%}")
        else:
            st.info("No patterns discovered yet. Add more papers or ask questions.")

    with tab2:
        st.subheader("Contradictions Detected")
        contradictions = synthesis.get_contradictions()

        if contradictions:
            for i, c in enumerate(contradictions, 1):
                with st.expander(f"Contradiction {i}: {c.type.value}", expanded=True):
                    st.warning(c.explanation)
                    st.caption(f"**Score:** {c.score:.2f}")
                    st.caption(f"**Temporal gap:** {c.temporal_gap_days} days")
        else:
            st.info("No contradictions detected. Papers appear consistent.")

    with tab3:
        st.subheader("Knowledge Gaps Identified")
        gaps = synthesis.get_gaps()

        if gaps:
            for i, gap in enumerate(gaps, 1):
                with st.expander(f"Gap {i}: {gap.bridge_concept}", expanded=True):
                    st.write(f"**Missing concepts:** {', '.join(gap.missing_concepts[:5])}")
                    st.write(f"**Importance:** {gap.importance:.2%}")
                    st.write(f"**Type:** {gap.gap_type.value}")

                    if gap.suggested_queries:
                        st.write("**Suggested queries:**")
                        for query in gap.suggested_queries[:3]:
                            st.caption(f"• {query}")
        else:
            st.info("No knowledge gaps identified. Collection appears complete.")


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    if not STREAMLIT_AVAILABLE:
        print("ERROR: Streamlit not available. Install with: pip install streamlit")
        return

    # Header
    st.title("📚 Personal Research Assistant")
    st.caption("*Your research papers remember each other*")

    # Initialize session state
    initialize_session_state()

    # Initialize systems
    if not st.session_state.initialized:
        with st.spinner("Initializing systems..."):
            asyncio.run(initialize_systems())

    # Render UI
    render_sidebar()

    # Main area with tabs
    main_tab, insights_tab = st.tabs(["💬 Chat", "🔍 Insights"])

    with main_tab:
        if st.session_state.papers:
            render_chat_interface()
        else:
            st.info("👈 Upload a PDF in the sidebar to get started!")

    with insights_tab:
        render_synthesis_tabs()

    # Footer
    st.divider()
    st.caption("Powered by HoloLoom • Week 4: Personal Research Assistant Demo")


if __name__ == "__main__":
    main()

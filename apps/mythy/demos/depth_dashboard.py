#!/usr/bin/env python3
"""
🎨 MYTHRL NARRATIVE DEPTH DASHBOARD
===================================
Interactive Streamlit dashboard for real-time narrative intelligence visualization.

Features:
- Live depth analysis with progressive gate unlocking animation
- Character detection highlighting
- Campbell stage journey mapping
- Archetypal resonance radar chart
- Cosmic truth revelation with effects
- Performance metrics and cache statistics
- Example texts from Odyssey, LOTR, Harry Potter

Usage:
    streamlit run demos/narrative_depth_dashboard.py
    
    Or with custom port:
    streamlit run demos/narrative_depth_dashboard.py --server.port 8501
"""

import asyncio
import time
import sys
sys.path.insert(0, '.')

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Import mythRL components
from hololoom_narrative._cache import CachedMatryoshkaDepth, NarrativeCache
from hololoom_narrative.matryoshka_depth import DepthLevel


# Page configuration
st.set_page_config(
    page_title="mythRL Narrative Depth Dashboard",
    page_icon="🪆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .cosmic-truth {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        font-size: 1.2rem;
        text-align: center;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    .gate-unlocked {
        color: #10b981;
        font-weight: bold;
    }
    .gate-locked {
        color: #6b7280;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
</style>
""", unsafe_allow_html=True)


# Example texts
EXAMPLE_TEXTS = {
    "Simple Observation": "The man walked down the street. It was a sunny day.",
    
    "Telemachus' Call": """Telemachus sits idle while suitors ravage his home. Athena appears as Mentes, 
    stirring the young prince to action. "Seek your father," she counsels. "The journey 
    will make you a man." The call to adventure rings clear, though Telemachus fears to answer.""",
    
    "Odysseus & Athena": """Odysseus met Athena at the crossroads, her owl eyes seeing through all 
    deception. 'The journey inward is harder than any odyssey,' she said. 'To find home, 
    you must first lose yourself completely.'""",
    
    "Frodo's Sacrifice": """As Frodo cast the Ring into Mount Doom, he understood: the treasure was never 
    the Ring, but the self he discovered in seeking to destroy it. In that moment of absolute 
    sacrifice, the finite hobbit touched the infinite, and the darkness consuming Middle-earth 
    dissolved into light. Death and rebirth were not opposites but one eternal breath.""",
    
    "Harry's Mentor": """Harry Potter stood in Dumbledore's office, the old wizard's eyes twinkling with 
    ancient wisdom. "It is our choices, Harry, that show what we truly are, far more than our 
    abilities," Dumbledore said softly. The young wizard felt the weight of destiny and the 
    lightness of free will simultaneously.""",
    
    "Land of the Dead": """In Hades, Odysseus speaks with the dead. His mother Anticlea reveals she died 
    of grief, waiting for his return. Agamemnon warns of treacherous wives. Achilles declares 
    he would rather be a slave on earth than king among the dead. Tiresias prophecies the path 
    home - and the journey that comes after. Death and rebirth: the ultimate transformation."""
}


# Initialize cache (singleton pattern)
@st.cache_resource
def get_depth_analyzer():
    """Initialize and cache the depth analyzer."""
    cache = NarrativeCache(max_size=100, ttl_seconds=3600)
    return CachedMatryoshkaDepth(cache)


def create_gate_visualization(gates_unlocked, total_gates=5):
    """Create visual representation of unlocked gates."""
    gate_names = ["SURFACE", "SYMBOLIC", "ARCHETYPAL", "MYTHIC", "COSMIC"]
    
    cols = st.columns(5)
    for i, (col, name) in enumerate(zip(cols, gate_names)):
        with col:
            if i < gates_unlocked:
                st.markdown(f"### 🔓 {name}")
                st.markdown('<p class="gate-unlocked">✓ UNLOCKED</p>', unsafe_allow_html=True)
            else:
                st.markdown(f"### 🔒 {name}")
                st.markdown('<p class="gate-locked">○ LOCKED</p>', unsafe_allow_html=True)


def create_complexity_gauge(complexity):
    """Create a gauge chart for complexity score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=complexity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Complexity Score"},
        delta={'reference': 0.5},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.5], 'color': "lightyellow"},
                {'range': [0.5, 0.7], 'color': "lightgreen"},
                {'range': [0.7, 0.85], 'color': "orange"},
                {'range': [0.85, 1], 'color': "purple"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.85
            }
        }
    ))
    fig.update_layout(height=300)
    return fig


def create_archetypal_radar(archetypes):
    """Create radar chart for archetypal resonance."""
    if not archetypes:
        return None
    
    # Get top 6 archetypes
    top_archetypes = dict(list(archetypes.items())[:6])
    
    categories = list(top_archetypes.keys())
    values = list(top_archetypes.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Archetypal Resonance',
        line_color='rgb(102, 126, 234)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        title="Archetypal Resonance Pattern"
    )
    
    return fig


def create_depth_progression_chart(result):
    """Create chart showing complexity progression through depth levels."""
    depths = []
    complexities = []
    
    for depth, complexity in result.complexity_progression:
        depths.append(depth.name)
        complexities.append(complexity)
    
    df = pd.DataFrame({
        'Depth Level': depths,
        'Complexity': complexities
    })
    
    fig = px.bar(
        df,
        x='Depth Level',
        y='Complexity',
        title='Complexity Progression Through Depth Levels',
        color='Complexity',
        color_continuous_scale='Purples'
    )
    
    fig.update_layout(height=300)
    return fig


async def analyze_text_async(text, analyzer):
    """Async wrapper for text analysis."""
    return await analyzer.analyze_depth(text)


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🪆 mythRL Narrative Depth Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Progressive depth gating for narrative intelligence** - From surface observation to cosmic truth revelation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Example selection
        example_name = st.selectbox(
            "Choose Example Text:",
            ["Custom"] + list(EXAMPLE_TEXTS.keys()),
            index=3  # Default to Frodo's Sacrifice
        )
        
        st.markdown("---")
        st.header("📊 About")
        st.markdown("""
        This dashboard visualizes mythRL's **Matryoshka Narrative Depth Analysis**:
        
        - 🪆 **5 Progressive Depth Levels**
        - 🎭 **30+ Universal Characters**
        - 📖 **17 Campbell Hero's Journey Stages**
        - 🏛️ **22 Character Archetypes**
        - ⚡ **Sub-2ms Analysis** (with caching)
        """)
        
        st.markdown("---")
        st.header("🎯 Depth Levels")
        st.markdown("""
        1. **SURFACE** - Literal meaning
        2. **SYMBOLIC** - Metaphor & symbolism
        3. **ARCHETYPAL** - Universal patterns
        4. **MYTHIC** - Eternal truths
        5. **COSMIC** - Ultimate meaning
        """)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Analysis", "📊 Statistics", "💡 Examples"])
    
    with tab1:
        # Text input
        if example_name == "Custom":
            text_input = st.text_area(
                "Enter text to analyze:",
                height=200,
                placeholder="Enter your narrative text here...",
                help="Paste any narrative text - stories, myths, hero's journey excerpts, etc."
            )
        else:
            text_input = st.text_area(
                f"Analyzing: {example_name}",
                value=EXAMPLE_TEXTS[example_name],
                height=200
            )
        
        analyze_button = st.button("🔍 Analyze Narrative Depth", type="primary", use_container_width=True)
        
        if analyze_button and text_input:
            # Initialize analyzer
            analyzer = get_depth_analyzer()
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Analysis with progress
            status_text.text("🔄 Initializing analysis...")
            progress_bar.progress(20)
            time.sleep(0.1)
            
            status_text.text("🧠 Analyzing narrative complexity...")
            progress_bar.progress(40)
            
            # Run async analysis
            start_time = time.perf_counter()
            result = asyncio.run(analyze_text_async(text_input, analyzer))
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            status_text.text("🪆 Unlocking Matryoshka gates...")
            progress_bar.progress(60)
            time.sleep(0.1)
            
            status_text.text("🎭 Detecting characters and archetypes...")
            progress_bar.progress(80)
            time.sleep(0.1)
            
            status_text.text("✨ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.3)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.header("🎯 Analysis Results")
            
            # Key metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Max Depth",
                    result.max_depth_achieved.name,
                    delta=f"Level {result.max_depth_achieved.value}"
                )
            
            with col2:
                st.metric(
                    "Complexity",
                    f"{result.total_complexity:.3f}",
                    delta="High" if result.total_complexity > 0.7 else "Medium"
                )
            
            with col3:
                gates_unlocked = len([g for g in result.gates_unlocked if g])
                st.metric(
                    "Gates Unlocked",
                    f"{gates_unlocked}/5",
                    delta=f"{(gates_unlocked/5)*100:.0f}%"
                )
            
            with col4:
                st.metric(
                    "Confidence",
                    f"{result.bayesian_confidence:.3f}",
                    delta=f"{duration_ms:.1f}ms"
                )
            
            # Gate visualization
            st.markdown("### 🪆 Matryoshka Gates")
            create_gate_visualization(gates_unlocked)
            
            # Charts in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Complexity gauge
                fig_gauge = create_complexity_gauge(result.total_complexity)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                # Depth progression
                fig_prog = create_depth_progression_chart(result)
                st.plotly_chart(fig_prog, use_container_width=True)
            
            # Archetypal radar (if available)
            if result.archetypal_layer and result.archetypal_layer.archetypal_resonance:
                st.markdown("### 🏛️ Archetypal Resonance")
                fig_radar = create_archetypal_radar(result.archetypal_layer.archetypal_resonance)
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Cosmic truth (if achieved)
            if result.cosmic_truth:
                st.markdown("### 🌌 Cosmic Truth Revealed")
                st.markdown(f'<div class="cosmic-truth">{result.cosmic_truth}</div>', unsafe_allow_html=True)
            
            # Meaning transformation journey
            st.markdown("### 🎭 Transformation Journey")
            for i, step in enumerate(result.transformation_journey, 1):
                st.markdown(f"**{i}.** {step}")
            
            # Symbolic elements (if present)
            if result.symbolic_layer and result.symbolic_layer.symbolic_elements:
                st.markdown("### 🔣 Symbolic Elements")
                symbols_df = pd.DataFrame([
                    {"Symbol": symbol, "Interpretation": meaning}
                    for symbol, meaning in result.symbolic_layer.symbolic_elements.items()
                ])
                st.dataframe(symbols_df, use_container_width=True, hide_index=True)
            
            # Mythic truths (if present)
            if result.mythic_layer and result.mythic_layer.universal_truths:
                st.markdown("### ⚡ Mythic Truths")
                for truth in result.mythic_layer.universal_truths:
                    st.info(truth)
    
    with tab2:
        st.header("📊 Performance Statistics")
        
        # Get cache stats
        analyzer = get_depth_analyzer()
        cache_stats = asyncio.run(analyzer.cache.get_stats())
        
        # Display cache metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Cache Size", f"{cache_stats['size']}/{cache_stats['max_size']}")
        
        with col2:
            st.metric("Hit Rate", f"{cache_stats['hit_rate']*100:.1f}%")
        
        with col3:
            st.metric("Total Requests", cache_stats['total_requests'])
        
        # Cache performance chart
        if cache_stats['total_requests'] > 0:
            cache_df = pd.DataFrame({
                'Metric': ['Hits', 'Misses', 'Evictions'],
                'Count': [cache_stats['hits'], cache_stats['misses'], cache_stats['evictions']]
            })
            
            fig = px.pie(
                cache_df,
                values='Count',
                names='Metric',
                title='Cache Performance Distribution',
                color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Hot entries
        st.markdown("### 🔥 Most Accessed Entries")
        hot_entries = asyncio.run(analyzer.cache.get_hot_entries(limit=5))
        
        if hot_entries:
            hot_df = pd.DataFrame(hot_entries)
            st.dataframe(hot_df, use_container_width=True, hide_index=True)
        else:
            st.info("No cache entries yet. Analyze some texts to see statistics!")
    
    with tab3:
        st.header("💡 Example Analyses")
        
        st.markdown("""
        ### Pre-analyzed Examples
        
        These examples demonstrate different depth levels:
        """)
        
        example_results = {
            "Simple Observation": {"depth": "SYMBOLIC", "complexity": 0.310, "gates": 2},
            "Telemachus' Call": {"depth": "COSMIC", "complexity": 1.000, "gates": 5},
            "Odysseus & Athena": {"depth": "COSMIC", "complexity": 0.993, "gates": 5},
            "Frodo's Sacrifice": {"depth": "COSMIC", "complexity": 0.984, "gates": 5},
            "Harry's Mentor": {"depth": "ARCHETYPAL", "complexity": 0.625, "gates": 3},
            "Land of the Dead": {"depth": "COSMIC", "complexity": 0.959, "gates": 5}
        }
        
        for name, stats in example_results.items():
            with st.expander(f"📖 {name}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Depth", stats["depth"])
                with col2:
                    st.metric("Complexity", f"{stats['complexity']:.3f}")
                with col3:
                    st.metric("Gates", f"{stats['gates']}/5")
                
                st.code(EXAMPLE_TEXTS[name][:200] + "...")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏛️ ODYSSEY 244D SEMANTIC ANALYSIS
==================================
Deep narrative analysis using full 244-dimensional semantic space.

This demonstrates:
1. Research-mode 244D semantic calculus on Homer's Odyssey
2. Visualization of dominant narrative dimensions
3. Comparison of different epic moments in semantic space
4. Full semantic trajectory analysis with mythological axes

The 244 dimensions include:
- 16 Standard (Warmth, Formality, etc.)
- 16 Narrative (Heroism, Quest, Destiny, etc.)
- 16 Emotional Depth (Grief, Awe, Longing, etc.)
- 16 Relational (Intimacy, Betrayal, Forgiveness, etc.)
- 16 Archetypal (Hero, Shadow, Mentor, etc.)
- 16 Philosophical (Freedom, Meaning, Authenticity, etc.)
- 16 Transformation (Crisis, Awakening, Rebirth, etc.)
- 16 Moral/Ethical (Justice, Virtue, Mercy, etc.)
- 16 Creative (Beauty, Harmony, Inspiration, etc.)
- 16 Cognitive (Nuance, Paradox, Insight, etc.)
- 16 Temporal/Narrative (Suspense, Climax, Reversal, etc.)
- 12 Spatial (Light, Wilderness, Home, etc.)
- 12 Character (Intelligence, Cunning, Piety, etc.)
- 12 Plot (Irony, Hubris, Nemesis, etc.)
- 12 Theme (War/Peace, Fate/Free-Will, etc.)
- 4 Style/Voice (Epic, Lyric, Tragic, Sublime)
"""

# Force UTF-8 encoding for Windows console
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Import semantic calculus with 244D support
from HoloLoom.semantic_calculus import (
    SemanticFlowCalculus,
    SemanticSpectrum,
    EXTENDED_244_DIMENSIONS,
)
from HoloLoom.semantic_calculus.analyzer import create_semantic_analyzer
from HoloLoom.semantic_calculus.config import SemanticCalculusConfig
from HoloLoom.embedding.spectral import create_embedder


# Key Odyssey passages for analysis
ODYSSEY_PASSAGES = {
    "Book 1 - Call to Adventure": """
    Sing to me of the man, Muse, the man of twists and turns
    driven time and again off course, once he had plundered
    the hallowed heights of Troy. Many cities of men he saw
    and learned their minds, many pains he suffered, heartsick
    on the open sea, fighting to save his life and bring his
    comrades home.
    """,

    "Book 9 - Cyclops Ordeal": """
    Nobody—that's my name. Nobody—so my mother and father call me,
    all my friends. But he boomed back at me from his ruthless heart,
    'Nobody? I'll eat Nobody last of all his friends—
    I'll eat the others first! That's my gift to you!'
    Lurching up, he lunged out with his hands toward my men
    and snatching two at once, rapping them on the ground
    he knocked them dead like pups.
    """,

    "Book 11 - Underworld Descent": """
    There I saw the ghost of great Achilles, and Patroclus,
    and Antilochus, and Ajax, best of men after the son of Peleus.
    The ghost of swift-footed Achilles knew me and addressed me
    in words of sorrow: 'Royal son of Laertes, Odysseus,
    man of exploits, what greater feat will you contrive?
    How did you dare to come down to the house of Hades
    where the dead live on as images, phantoms of mortals?'
    """,

    "Book 16 - Father-Son Recognition": """
    Then throwing his arms around this marvel of a father
    Telemachus began to weep. Salt tears rose from the wells
    of longing in both men, and cries burst from both
    as keen and fluttering as those of the great taloned hawk,
    whose nestlings farmers take before they fly.
    So helplessly they cried, pouring out tears.
    """,

    "Book 23 - Marriage Reunion": """
    Now from his breast into his eyes the ache
    of longing mounted, and he wept at last,
    his dear wife, clear and faithful, in his arms,
    longed for as the sunwarmed earth is longed for by a swimmer
    spent in rough water where his ship went down.
    Joy, joy, like the joy of shipwrecked sailors
    when they see land—land that Poseidon has brought them through.
    """
}


def analyze_odyssey_passage_244d(
    passage: str,
    title: str,
    config: SemanticCalculusConfig
) -> Dict:
    """
    Analyze a passage using 244D semantic calculus.

    Returns:
        Dictionary with trajectory, dimensions, and analysis
    """
    print(f"\n{'='*80}")
    print(f"📖 {title}")
    print(f"{'='*80}")

    # Create embedder
    embed_model = create_embedder(sizes=[384])
    embed_fn = lambda words: embed_model.encode(words) if isinstance(words, list) else embed_model.encode([words])[0]

    # Create 244D analyzer
    print(f"🔬 Initializing {config.dimensions}D semantic analyzer...")
    analyzer = create_semantic_analyzer(embed_fn, config=config)

    # Analyze passage
    words = passage.split()[:30]  # First 30 words for focused analysis
    print(f"📝 Analyzing {len(words)} words...")

    result = analyzer.analyze_text(" ".join(words))

    # Extract results
    trajectory = result['trajectory']
    semantic_forces = result['semantic_forces']

    print(f"\n📊 TRAJECTORY METRICS:")
    print(f"   Total distance: {trajectory.total_distance():.3f}")
    print(f"   Average speed: {np.mean([s.speed for s in trajectory.states]):.3f}")
    print(f"   Max curvature: {max([trajectory.curvature(i) for i in range(len(trajectory.states))]):.3f}")

    print(f"\n🎯 TOP 10 DOMINANT DIMENSIONS (by velocity):")
    dominant = semantic_forces['dominant_velocity'][:10]
    for i, (dim_name, velocity) in enumerate(dominant, 1):
        # Determine category
        category = get_dimension_category(dim_name)
        print(f"   {i:2d}. {dim_name:<25} {velocity:>6.3f}  [{category}]")

    print(f"\n⚡ TOP 5 DIMENSIONS WITH STRONGEST FORCES (acceleration):")
    force_dominant = semantic_forces['dominant_force'][:5]
    for i, (dim_name, force) in enumerate(force_dominant, 1):
        category = get_dimension_category(dim_name)
        print(f"   {i}. {dim_name:<25} {force:>6.3f}  [{category}]")

    return {
        'title': title,
        'trajectory': trajectory,
        'semantic_forces': semantic_forces,
        'words': words,
        'analyzer': analyzer,
    }


def get_dimension_category(dim_name: str) -> str:
    """Determine which category a dimension belongs to."""
    # Map dimension names to categories
    narrative = ["Heroism", "Transformation", "Conflict", "Mystery", "Sacrifice",
                 "Wisdom", "Courage", "Redemption", "Destiny", "Honor", "Loyalty",
                 "Quest", "Transcendence", "Shadow", "Initiation", "Rebirth"]

    emotional = ["Authenticity", "Vulnerability", "Trust", "Hope", "Grief", "Shame",
                 "Compassion", "Rage", "Longing", "Awe", "Jealousy", "Guilt",
                 "Pride", "Disgust", "Ecstasy", "Dread"]

    archetypal = ["Hero-Archetype", "Mentor-Archetype", "Shadow-Archetype",
                  "Trickster-Archetype", "Mother-Archetype", "Father-Archetype",
                  "Child-Archetype", "Anima-Animus", "Self-Archetype",
                  "Threshold-Guardian", "Herald", "Ally", "Shapeshifter",
                  "Oracle", "Ruler", "Lover"]

    philosophical = ["Freedom", "Meaning", "Being", "Essence", "Absurdity",
                     "Authenticity-Existential", "Authenticity-Choice", "Bad-Faith",
                     "Thrownness", "Care", "Dasein", "Truth-Aletheia", "Anxiety",
                     "Responsibility", "Time-Consciousness", "Death-Awareness"]

    theme = ["Love-Hate", "War-Peace", "Civilization-Barbarism", "Individual-Society",
             "Nature-Culture", "Mortality-Immortality", "Knowledge-Ignorance",
             "Fate-Free-Will", "Appearance-Reality", "Order-Chaos", "Youth-Age",
             "Memory-Forgetting"]

    plot = ["Irony", "Hubris", "Nemesis", "Hamartia", "Catastrophe", "Comedy",
            "Complication", "Coincidence", "Necessity", "Causality",
            "Dramatic-Irony", "Deus-Ex-Machina"]

    if dim_name in narrative:
        return "Narrative"
    elif dim_name in emotional:
        return "Emotional"
    elif dim_name in archetypal:
        return "Archetypal"
    elif dim_name in philosophical:
        return "Philosophical"
    elif dim_name in theme:
        return "Theme"
    elif dim_name in plot:
        return "Plot"
    else:
        return "Core/Other"


def create_dimension_heatmap(results: List[Dict], output_dir: Path):
    """Create heatmap showing top dimensions across all passages."""
    print(f"\n{'='*80}")
    print("📊 Creating 244D dimension activation heatmap...")
    print(f"{'='*80}")

    # Collect top 15 dimensions from each passage
    passage_names = []
    num_passages = len(results)

    # First pass: collect all passage names and all dimensions
    for result in results:
        passage_names.append(result['title'].split(' - ')[0])

    # Build complete matrix: dimensions x passages
    all_dims = {}
    for passage_idx, result in enumerate(results):
        dominant = result['semantic_forces']['dominant_velocity'][:15]

        for dim_name, velocity in dominant:
            if dim_name not in all_dims:
                # Initialize with zeros for all passages
                all_dims[dim_name] = [0.0] * num_passages
            all_dims[dim_name][passage_idx] = velocity

    # Sort by total activation
    sorted_dims = sorted(all_dims.items(),
                         key=lambda x: sum(x[1]),
                         reverse=True)[:20]

    dim_names = [name for name, _ in sorted_dims]
    values = np.array([vals for _, vals in sorted_dims])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(values, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    # Labels
    ax.set_yticks(range(len(dim_names)))
    ax.set_yticklabels(dim_names, fontsize=9)
    ax.set_xticks(range(len(passage_names)))
    ax.set_xticklabels(passage_names, rotation=45, ha='right', fontsize=9)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Velocity Magnitude', rotation=270, labelpad=20)

    # Title
    ax.set_title('Top 20 Semantic Dimensions Across Odyssey Passages\n(244D Research Mode)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add values as text
    for i in range(len(dim_names)):
        for j in range(len(passage_names)):
            text = ax.text(j, i, f'{values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    plt.tight_layout()
    output_path = output_dir / "odyssey_244d_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved heatmap: {output_path}")

    return fig


def create_category_distribution(results: List[Dict], output_dir: Path):
    """Show which dimension categories are most active."""
    print(f"\n📊 Analyzing dimension category distribution...")

    category_counts = {
        'Narrative': 0,
        'Emotional': 0,
        'Archetypal': 0,
        'Philosophical': 0,
        'Theme': 0,
        'Plot': 0,
        'Core/Other': 0,
    }

    # Count activations by category
    for result in results:
        dominant = result['semantic_forces']['dominant_velocity'][:20]
        for dim_name, velocity in dominant:
            category = get_dimension_category(dim_name)
            category_counts[category] += velocity

    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = list(category_counts.keys())
    values = list(category_counts.values())
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#2ecc71', '#1abc9c', '#95a5a6']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Labels
    ax.set_xlabel('Dimension Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Velocity Magnitude', fontsize=12, fontweight='bold')
    ax.set_title('244D Semantic Category Distribution Across Odyssey\n(Sum of top-20 velocities per passage)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    output_path = output_dir / "odyssey_244d_categories.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved category chart: {output_path}")

    return fig


def main():
    """Run full 244D Odyssey analysis."""
    print("🏛️ ODYSSEY 244D SEMANTIC ANALYSIS")
    print("="*80)
    print("Analyzing Homer's Odyssey with full 244-dimensional semantic space")
    print("="*80)

    # Create output directory
    output_dir = Path("demos/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create research configuration (244D)
    config = SemanticCalculusConfig.research()
    print(f"\n✨ Configuration: {config}")
    print(f"   Dimensions: {config.dimensions}")
    print(f"   Cache size: {config.cache_size}")
    print(f"   Trajectory: {config.compute_trajectory}")
    print(f"   Ethics: {config.compute_ethics}")

    # Analyze each passage
    results = []
    for title, passage in ODYSSEY_PASSAGES.items():
        result = analyze_odyssey_passage_244d(passage, title, config)
        results.append(result)

    # Create visualizations
    print(f"\n{'='*80}")
    print("🎨 Creating visualizations...")
    print(f"{'='*80}")

    create_dimension_heatmap(results, output_dir)
    create_category_distribution(results, output_dir)

    # Summary
    print(f"\n{'='*80}")
    print("✅ 244D ODYSSEY ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"📚 Passages analyzed: {len(results)}")
    print(f"🎯 Total dimensions: 244")
    print(f"📊 Visualizations: 2")
    print(f"💾 Output directory: {output_dir.absolute()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

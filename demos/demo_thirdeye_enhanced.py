"""
ThirdEye Enhanced Visualization Demo
====================================

Demonstrates all 4 extension systems:
1. Conversation Memory - Entity persistence across messages
2. Semantic Extractor - 228D → visual parameters
3. Visual Styles - 6 aesthetic systems
4. Layer Composer - Multi-scene stacking

Run: PYTHONPATH=. python demos/demo_thirdeye_enhanced.py

Created: 2025-12-01
"""

import asyncio
import json
from pathlib import Path

# Enhanced composer (integrates all systems)
from HoloLoom.thirdeye.visualizers.enhanced_composer import (
    EnhancedSceneComposer,
    ComposerConfig,
    get_enhanced_composer,
    compose_enhanced_scene,
)

# Individual components for detailed demos
from HoloLoom.thirdeye.visualizers.conversation_memory import (
    ConversationMemory,
    EntityType,
    RelationType,
    get_conversation_memory,
)
from HoloLoom.thirdeye.visualizers.semantic_extractor import (
    SemanticExtractor,
    get_semantic_extractor,
)
from HoloLoom.thirdeye.visualizers.visual_styles import (
    StyleName,
    get_style,
    get_style_for_mood,
    STYLES,
)
from HoloLoom.thirdeye.visualizers.layer_composer import (
    LayerComposer,
    LayerType,
)


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


async def demo_conversation_memory():
    """Demo 1: Conversation Memory - Entity Persistence"""
    print_header("Demo 1: Conversation Memory")

    memory = ConversationMemory()

    # Simulate a conversation about building a system
    messages = [
        "Let's design a user authentication service",
        "The AuthService connects to a PostgreSQL database",
        "Users can also authenticate via OAuth through Google",
        "We need a Redis cache for session management",
    ]

    print("\nSimulating conversation about system architecture...")

    for i, msg in enumerate(messages, 1):
        print(f"\n[Message {i}]: \"{msg}\"")

        # Extract entities (simplified - real extraction would be smarter)
        if "AuthService" in msg or "authentication service" in msg:
            entity = memory.remember_entity(
                name="AuthService",
                entity_type=EntityType.SERVICE,
                attributes={"description": "Handles user authentication"}
            )
            print(f"  ->Remembered: {entity.name} ({entity.entity_type.value})")

        if "PostgreSQL" in msg:
            entity = memory.remember_entity(
                name="PostgreSQL",
                entity_type=EntityType.DATABASE,
                attributes={"type": "relational"}
            )
            memory.add_relationship("AuthService", "PostgreSQL", RelationType.CONNECTS_TO)
            print(f"  ->Remembered: {entity.name} ({entity.entity_type.value})")
            print(f"  ->Relationship: AuthService --CONNECTS_TO--> PostgreSQL")

        if "Google" in msg or "OAuth" in msg:
            entity = memory.remember_entity(
                name="GoogleOAuth",
                entity_type=EntityType.API,
                attributes={"provider": "Google"}
            )
            memory.add_relationship("AuthService", "GoogleOAuth", RelationType.CONNECTS_TO)
            print(f"  ->Remembered: {entity.name} ({entity.entity_type.value})")

        if "Redis" in msg:
            entity = memory.remember_entity(
                name="RedisCache",
                entity_type=EntityType.DATABASE,
                attributes={"type": "cache"}
            )
            memory.add_relationship("AuthService", "RedisCache", RelationType.CONNECTS_TO)
            print(f"  ->Remembered: {entity.name} ({entity.entity_type.value})")

    # Show all remembered entities
    print_subheader("Memory State")
    active = memory.get_active_entities()
    print(f"Total entities remembered: {len(active)}")
    for entity in active:
        print(f"  *{entity.name} ({entity.entity_type.value}) - importance: {entity.importance:.2f}")

    print(f"\nRelationships tracked: {len(memory.relationships)}")
    for rel in memory.relationships:
        print(f"  *{rel.source_id} --{rel.relation_type.value}--> {rel.target_id}")


async def demo_semantic_extraction():
    """Demo 2: Semantic Extractor - 228D to Visual Parameters"""
    print_header("Demo 2: Semantic Extraction")

    extractor = SemanticExtractor()  # Uses heuristic fallback (no HoloLoom)

    # Different text samples with varying semantics
    samples = [
        ("URGENT: Server is down! Need immediate help!", "urgent_technical"),
        ("Once upon a time, in a quiet village nestled in the mountains...", "calm_narrative"),
        ("The quarterly revenue increased by 23% compared to last year.", "neutral_data"),
        ("In the shadows of the ancient temple, something stirred...", "mysterious"),
        ("Error 500: Database connection failed. Check credentials.", "technical_urgent"),
    ]

    print("\nAnalyzing text semantics and deriving visual parameters...")

    for text, label in samples:
        print(f"\n[{label}]")
        print(f"  Text: \"{text[:50]}...\"" if len(text) > 50 else f"  Text: \"{text}\"")

        profile = await extractor.analyze(text)

        print(f"  Mood: {profile.mood.value}")
        print(f"  Density: {profile.density.value}")
        print(f"  Color Temperature: {profile.color_temperature}")
        print(f"  Animation Speed: {profile.animation_speed:.2f}")
        print(f"  Lighting Intensity: {profile.lighting_intensity:.2f}")

        # Show key semantic dimensions
        print(f"  Key axes: urgency={profile.urgency:.2f}, emotionality={profile.emotionality:.2f}, technicality={profile.technicality:.2f}")


async def demo_visual_styles():
    """Demo 3: Visual Styles - 6 Aesthetic Systems"""
    print_header("Demo 3: Visual Styles")

    print(f"\nAvailable styles: {len(STYLES)}")

    for style_name in StyleName:
        style = get_style(style_name)
        print(f"\n[{style_name.value.upper()}]")
        print(f"  Colors:")
        print(f"    Primary: {style.colors.primary}")
        print(f"    Background: {style.colors.background}")
        print(f"    Accent: {style.colors.accent}")
        print(f"  Materials:")
        print(f"    Wireframe: {style.materials.wireframe}")
        print(f"    Metalness: {style.materials.metalness}")
        print(f"    Emissive: {style.materials.emissive_intensity}")
        print(f"  Animation:")
        print(f"    Duration: {style.animation.duration}s")
        print(f"    Entrance: {style.animation.entrance}")
        print(f"  Best for: ", end="")
        if style_name == StyleName.BLUEPRINT:
            print("Technical diagrams, architecture")
        elif style_name == StyleName.STORYBOARD:
            print("Narratives, character scenes")
        elif style_name == StyleName.DASHBOARD:
            print("Data visualization, metrics")
        elif style_name == StyleName.SHOWROOM:
            print("Product displays, objects")
        elif style_name == StyleName.CINEMATIC:
            print("Dramatic scenes, stories")
        elif style_name == StyleName.NEON:
            print("Cyberpunk, futuristic themes")


async def demo_layer_composition():
    """Demo 4: Layer Composer - Multi-Scene Stacking"""
    print_header("Demo 4: Layer Composition")

    composer = LayerComposer()

    # Create sample scenes from different modes
    ui_scene = {
        "mode": "ui_design",
        "elements": [
            {"type": "ui_panel", "name": "LoginForm", "position": {"x": 0, "y": 0, "z": 0}},
            {"type": "label", "name": "Username", "position": {"x": -1, "y": 1, "z": 0}},
        ]
    }

    narrative_scene = {
        "mode": "narrative",
        "elements": [
            {"type": "character", "name": "User", "position": {"x": -3, "y": 0, "z": 2}},
            {"type": "dialogue", "name": "SpeechBubble", "position": {"x": -2, "y": 2, "z": 2}},
        ]
    }

    arch_scene = {
        "mode": "architecture",
        "elements": [
            {"type": "service", "name": "AuthAPI", "position": {"x": 2, "y": 0, "z": -2}},
            {"type": "database", "name": "UserDB", "position": {"x": 4, "y": 0, "z": -2}},
            {"type": "connection", "name": "APItoDB", "position": {"x": 3, "y": 0, "z": -2}},
        ]
    }

    print("\nComposing 3 scenes into layers...")
    print("  *UI Design scene (2 elements)")
    print("  *Narrative scene (2 elements)")
    print("  *Architecture scene (3 elements)")

    composed = composer.compose(
        scenes=[ui_scene, narrative_scene, arch_scene],
        focus_mode="ui_design"
    )

    print_subheader("Layer Assignment")
    for layer_type in LayerType:
        elements = composed.layers.get(layer_type, [])
        if elements:
            print(f"\n  {layer_type.value.upper()} layer ({len(elements)} elements):")
            for elem in elements:
                print(f"    *{elem.get('name', 'unnamed')} ({elem.get('type', 'unknown')})")

    print_subheader("Layer Configuration")
    for layer_type, config in composed.layer_configs.items():
        if composed.layers.get(layer_type):
            print(f"  {layer_type.value}: z={config.z_offset}, blur={config.blur}, opacity={config.opacity}")


async def demo_enhanced_composer():
    """Demo 5: Enhanced Composer - Full Integration"""
    print_header("Demo 5: Enhanced Composer (Full Integration)")

    # Create composer with all features enabled
    config = ComposerConfig(
        enable_memory=True,
        enable_semantic=True,
        enable_styles=True,
        enable_layers=True,
        auto_style_from_mood=True,
    )
    composer = EnhancedSceneComposer(config)

    # Test messages that exercise different modes
    test_cases = [
        (
            "Design a login screen with email and password fields, plus a 'Remember Me' checkbox",
            None, None
        ),
        (
            "The detective walked into the dimly lit room. 'We need to talk,' she said to the shadow in the corner.",
            None, "cinematic"
        ),
        (
            "The API Gateway routes requests to the AuthService, which queries the UserDatabase",
            None, "blueprint"
        ),
        (
            "ALERT: CPU usage at 95%! Memory critical! 3 services unhealthy!",
            None, None
        ),
    ]

    print("\nComposing enhanced scenes with all systems integrated...\n")

    for text, force_mode, force_style in test_cases:
        print(f"Input: \"{text[:60]}...\"" if len(text) > 60 else f"Input: \"{text}\"")

        scene = await composer.compose_async(
            text=text,
            force_mode=force_mode,
            force_style=force_style,
        )

        print(f"  Mode: {scene.mode}")
        print(f"  Style: {scene.style.name.value if scene.style else 'none'}")
        print(f"  Elements: {len(scene.elements)}")
        if scene.semantic_profile:
            print(f"  Mood: {scene.semantic_profile.mood.value}")
        print(f"  Persistent entities: {len(scene.persistent_entities)}")
        print()

    # Show accumulated memory
    print_subheader("Accumulated Memory State")
    memory_state = composer.get_memory_state()
    print(f"Entities remembered: {memory_state.get('entity_count', 0)}")
    for entity in memory_state.get('entities', [])[:5]:
        print(f"  *{entity['name']} ({entity['type']})")


async def demo_output_html():
    """Demo 6: Generate HTML Output for Viewing"""
    print_header("Demo 6: Generate Viewable HTML")

    # Create a rich narrative scene
    text = """
    In the heart of the enchanted forest, Princess Aurora discovered an ancient stone archway.
    Mysterious runes glowed softly along its edges. Beyond the archway, she could see a
    floating castle suspended among the clouds, its towers reaching toward an aurora-filled sky.
    """

    print(f"\nGenerating scene for: \"{text[:50]}...\"")

    scene = compose_enhanced_scene(text, force_style="cinematic")

    # Generate HTML visualization
    html = generate_demo_html(scene, text)

    # Save to file
    output_dir = Path("demos/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "thirdeye_enhanced_demo.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nGenerated: {output_path}")
    print("Open in browser to view the scene data!")

    # Also save the raw scene JSON
    json_path = output_dir / "thirdeye_scene.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, default=str)
    print(f"Scene JSON: {json_path}")


def generate_demo_html(scene: dict, original_text: str) -> str:
    """Generate an HTML page showing the scene data."""

    style_info = scene.get('style', {})
    semantic_info = scene.get('semantic', {})
    elements = scene.get('elements', [])

    # Get colors from style
    bg_color = style_info.get('colors', {}).get('background', '#1a1a2e')
    primary_color = style_info.get('colors', {}).get('primary', '#60a5fa')
    text_color = style_info.get('colors', {}).get('text', '#e2e8f0')

    elements_html = ""
    for elem in elements[:10]:  # Show first 10 elements
        elem_type = elem.get('type', 'unknown')
        elem_name = elem.get('label', elem.get('name', 'unnamed'))
        pos = elem.get('position', {})

        elements_html += f"""
        <div class="element">
            <span class="element-type">{elem_type}</span>
            <span class="element-name">{elem_name}</span>
            <span class="element-pos">({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}, {pos.get('z', 0):.1f})</span>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ThirdEye Enhanced Demo</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: {bg_color};
            color: {text_color};
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{
            color: {primary_color};
            font-size: 2rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid {primary_color}40;
            padding-bottom: 0.5rem;
        }}
        h2 {{
            color: {primary_color};
            font-size: 1.25rem;
            margin: 1.5rem 0 0.75rem;
        }}
        .input-text {{
            background: {primary_color}15;
            border-left: 3px solid {primary_color};
            padding: 1rem;
            margin: 1rem 0;
            font-style: italic;
            line-height: 1.6;
        }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }}
        .card {{
            background: {primary_color}10;
            border: 1px solid {primary_color}30;
            border-radius: 8px;
            padding: 1rem;
        }}
        .card-title {{
            color: {primary_color};
            font-weight: 600;
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .stat {{
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            border-bottom: 1px solid {primary_color}15;
        }}
        .stat-label {{ opacity: 0.7; }}
        .stat-value {{ font-weight: 500; }}
        .element {{
            display: flex;
            gap: 0.5rem;
            padding: 0.5rem;
            background: {primary_color}08;
            border-radius: 4px;
            margin-bottom: 0.25rem;
            font-size: 0.85rem;
        }}
        .element-type {{
            background: {primary_color}30;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            font-size: 0.75rem;
        }}
        .element-name {{ flex: 1; font-weight: 500; }}
        .element-pos {{ opacity: 0.5; font-family: monospace; }}
        .badge {{
            display: inline-block;
            background: {primary_color};
            color: {bg_color};
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ThirdEye Enhanced Scene</h1>

        <div class="input-text">
            {original_text}
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Scene Info</div>
                <div class="stat">
                    <span class="stat-label">Mode</span>
                    <span class="badge">{scene.get('mode', 'unknown')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Title</span>
                    <span class="stat-value">{scene.get('title', 'Untitled')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Elements</span>
                    <span class="stat-value">{len(elements)}</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Visual Style</div>
                <div class="stat">
                    <span class="stat-label">Style</span>
                    <span class="stat-value">{style_info.get('name', 'none')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Primary Color</span>
                    <span class="stat-value">{style_info.get('colors', {}).get('primary', 'N/A')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Animation</span>
                    <span class="stat-value">{style_info.get('animation', {}).get('entrance', 'N/A')}</span>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Semantic Analysis</div>
                <div class="stat">
                    <span class="stat-label">Mood</span>
                    <span class="badge">{semantic_info.get('mood', 'unknown')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Density</span>
                    <span class="stat-value">{semantic_info.get('density', 'N/A')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Color Temperature</span>
                    <span class="stat-value">{semantic_info.get('color_temperature', 'N/A')}</span>
                </div>
                <div class="stat">
                    <span class="stat-label">Animation Speed</span>
                    <span class="stat-value">{semantic_info.get('animation_speed', 0):.2f}</span>
                </div>
            </div>
        </div>

        <h2>Scene Elements</h2>
        <div class="card">
            {elements_html}
            {f'<div style="opacity:0.5;padding:0.5rem">...and {len(elements)-10} more</div>' if len(elements) > 10 else ''}
        </div>

        <h2>Raw Scene Data</h2>
        <div class="card">
            <pre style="overflow-x:auto;font-size:0.8rem;line-height:1.4;">{json.dumps(scene, indent=2, default=str)[:3000]}...</pre>
        </div>
    </div>
</body>
</html>"""


async def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  THIRDEYE ENHANCED VISUALIZATION SYSTEM - DEMO")
    print("=" * 60)

    await demo_conversation_memory()
    await demo_semantic_extraction()
    await demo_visual_styles()
    await demo_layer_composition()
    await demo_enhanced_composer()
    await demo_output_html()

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
    print("\nAll 4 extension systems demonstrated:")
    print("  1. Conversation Memory - Entity persistence")
    print("  2. Semantic Extractor - 228D -> visual params")
    print("  3. Visual Styles - 6 aesthetic systems")
    print("  4. Layer Composer - Multi-scene stacking")
    print("  5. Enhanced Composer - Full integration")
    print("\nCheck demos/output/ for HTML visualization!")


if __name__ == "__main__":
    asyncio.run(main())

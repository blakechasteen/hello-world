"""
ThirdEye + Image Generation Demo
================================

Demonstrates:
1. Creating scenes with ThirdEye enhanced composer
2. Converting scenes to optimized image prompts
3. Generating actual 2D images (if backend available)
4. Graceful fallback when no backend

Requirements (for actual generation):
- ComfyUI running on port 8188, OR
- AUTOMATIC1111 running on port 7860 with --api flag

Without a backend, the demo still shows prompt generation.

Created: 2025-12-01
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hololoom.thirdeye.visualizers.enhanced_composer import (
    EnhancedSceneComposer,
    ComposerConfig,
)
from hololoom.thirdeye.generators import (
    ImageGenerator,
    GenerationConfig,
    SceneToPrompt,
    build_prompt_from_scene,
)


# ============================================================
# Demo Scenes
# ============================================================

DEMO_TEXTS = [
    # Narrative scene
    """
    In the heart of the enchanted forest, Princess Aurora discovered an ancient stone archway.
    Mysterious runes glowed softly along its edges. Beyond the archway, she could see a
    floating castle suspended among the clouds, its towers reaching toward an aurora-filled sky.
    """,

    # UI design scene
    """
    Design a modern dashboard for a fitness tracking app. The main screen shows a circular
    progress ring for daily steps, heart rate monitor card, sleep quality chart, and
    a bottom navigation with icons for Home, Activity, Heart, and Profile.
    """,

    # Architecture scene
    """
    Our microservices architecture has three main services: AuthService handles OAuth2
    authentication, ProductService manages the catalog with Elasticsearch, and OrderService
    processes transactions through Stripe. All services communicate via RabbitMQ message queue
    and store data in PostgreSQL and Redis cache.
    """,
]


async def demo_prompt_building():
    """Demo 1: Scene to Prompt conversion."""
    print("\n" + "=" * 60)
    print("DEMO 1: Scene to Prompt Conversion")
    print("=" * 60)

    # Create enhanced composer
    config = ComposerConfig(
        enable_memory=True,
        enable_semantic=False,  # Disable async semantic for simpler demo
        enable_styles=True,
        enable_layers=True,
    )
    composer = EnhancedSceneComposer(config)
    prompt_builder = SceneToPrompt()

    for i, text in enumerate(DEMO_TEXTS, 1):
        print(f"\n--- Scene {i} ---")
        print(f"Input: {text[:100].strip()}...")

        # Create scene
        scene = composer.compose(text.strip())
        mode = scene.get('mode', 'abstract')
        print(f"Detected Mode: {mode}")

        # Convert to prompts
        components = prompt_builder.convert(scene)
        positive = components.to_positive_prompt()
        negative = components.to_negative_prompt()

        print(f"\nPositive Prompt:")
        print(f"  {positive[:200]}...")

        print(f"\nNegative Prompt:")
        print(f"  {negative[:100]}...")


async def demo_backend_check():
    """Demo 2: Check for available backends."""
    print("\n" + "=" * 60)
    print("DEMO 2: Backend Availability Check")
    print("=" * 60)

    generator = ImageGenerator()

    print("\nChecking for image generation backends...")
    status = await generator.check_backend()

    print(f"\nAvailable: {status['available']}")
    print(f"Backend: {status['backend']}")
    print(f"Message: {status['message']}")

    await generator.close()
    return status['available']


async def demo_image_generation(backend_available: bool):
    """Demo 3: Actual image generation (if backend available)."""
    print("\n" + "=" * 60)
    print("DEMO 3: Image Generation")
    print("=" * 60)

    if not backend_available:
        print("\nSkipping actual generation - no backend available.")
        print("To enable generation, start one of:")
        print("  - ComfyUI: python -m ComfyUI.main --listen")
        print("  - AUTOMATIC1111: python launch.py --api")
        print("\nShowing what would be generated...")

    # Create scene
    config = ComposerConfig(
        enable_memory=True,
        enable_semantic=False,
        enable_styles=True,
    )
    composer = EnhancedSceneComposer(config)
    scene = composer.compose(DEMO_TEXTS[0].strip())  # Fantasy narrative

    # Build prompt
    positive, negative = build_prompt_from_scene(scene)

    print(f"\nScene Mode: {scene.get('mode')}")
    print(f"Style: {scene.get('style', {}).get('name', 'default')}")
    print(f"\nPrompt for generation:")
    print(f"  {positive}")

    if backend_available:
        # Configure generation
        gen_config = GenerationConfig(
            width=512,
            height=512,
            steps=20,
            cfg_scale=7.0,
            sampler="euler",
        )

        generator = ImageGenerator(gen_config)

        print("\nGenerating image...")
        result = await generator.generate_from_scene(scene)

        if isinstance(result, dict) and 'error' in result:
            print(f"Error: {result['error']}")
        else:
            # Save the image
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "thirdeye_generated.png"

            result.save(output_path)
            print(f"\nImage saved to: {output_path}")
            print(f"Dimensions: {result.width}x{result.height}")
            print(f"Seed: {result.seed}")
            print(f"Backend: {result.metadata.get('backend')}")

        await generator.close()
    else:
        # Just show the prompt info
        print("\n[Would generate 512x512 image with above prompt]")


async def demo_style_variations():
    """Demo 4: Style-based prompt variations."""
    print("\n" + "=" * 60)
    print("DEMO 4: Style Variations")
    print("=" * 60)

    from hololoom.thirdeye.visualizers.visual_styles import StyleName, get_style

    base_text = "A futuristic city skyline at sunset"

    print(f"\nBase text: {base_text}")
    print("\nStyle variations:\n")

    config = ComposerConfig(enable_styles=True)
    composer = EnhancedSceneComposer(config)

    for style_name in [StyleName.CINEMATIC, StyleName.BLUEPRINT, StyleName.NEON]:
        scene = composer.compose(base_text, force_style=style_name.value)

        positive, _ = build_prompt_from_scene(scene)

        print(f"  {style_name.value.upper()}:")
        print(f"    {positive[:150]}...")
        print()


async def main():
    """Run all demos."""
    print("=" * 60)
    print("ThirdEye Image Generation Integration")
    print("=" * 60)

    # Demo 1: Prompt building (always works)
    await demo_prompt_building()

    # Demo 2: Check backend
    backend_available = await demo_backend_check()

    # Demo 3: Image generation
    await demo_image_generation(backend_available)

    # Demo 4: Style variations
    await demo_style_variations()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    if not backend_available:
        print("\nNote: To generate actual images, install and run:")
        print("  Option 1: ComfyUI (recommended)")
        print("    - https://github.com/comfyanonymous/ComfyUI")
        print("    - Run: python main.py --listen")
        print("")
        print("  Option 2: AUTOMATIC1111")
        print("    - https://github.com/AUTOMATIC1111/stable-diffusion-webui")
        print("    - Run: python launch.py --api")


if __name__ == "__main__":
    asyncio.run(main())

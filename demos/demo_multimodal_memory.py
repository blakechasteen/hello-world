#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multimodal Memory Demo
======================

Demonstrates HoloLoom's complete multimodal memory system:
- experience() → Store text memories
- remember_photo() → Store visual memories
- recall(include_photos=True) → Multimodal retrieval
- link_photo_to_memory() → Semantic linking

This is the full experience → recall cycle with photos.

Requirements:
    pip install Pillow openai-clip torch numpy

Usage:
    python demos/demo_multimodal_memory.py

What this demonstrates:
    1. Text memory creation
    2. Photo memory creation
    3. Automatic entity linking (photo → entities)
    4. Manual memory linking (photo → text)
    5. Multimodal search (query retrieves both text AND photos)
    6. Knowledge graph integration
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    print("❌ PIL not available. Install: pip install Pillow")
    PIL_AVAILABLE = False
    sys.exit(1)

from HoloLoom import HoloLoom


def create_test_image(width=400, height=300, color=(255, 0, 0), text="Test"):
    """Create a synthetic test image (colored rectangle with text)."""
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = ((width - text_width) // 2, (height - text_height) // 2)
    text_color = (255 - color[0], 255 - color[1], 255 - color[2])
    draw.text(position, text, fill=text_color, font=font)

    return np.array(img)


async def main():
    """Run multimodal memory demo."""

    print("=" * 80)
    print("Multimodal Memory Demo - Full Experience → Recall Cycle")
    print("=" * 80)
    print()

    async with HoloLoom() as loom:
        print("✓ HoloLoom initialized")
        print()

        # =====================================================================
        # Phase 1: Experience Text Memories
        # =====================================================================
        print("-" * 80)
        print("Phase 1: Experience Text Memories")
        print("-" * 80)
        print()

        text_memories = []

        # Store text about architecture
        print("📝 Experiencing: 'We discussed the system architecture'")
        mem1 = await loom.experience("We discussed the system architecture at the meeting")
        text_memories.append(mem1)
        print(f"   Memory ID: {mem1.id}")
        print()

        # Store text about diagrams
        print("📝 Experiencing: 'The architecture diagram shows the components'")
        mem2 = await loom.experience("The architecture diagram shows all the components and connections")
        text_memories.append(mem2)
        print(f"   Memory ID: {mem2.id}")
        print()

        # Store text about meetings
        print("📝 Experiencing: 'Team meeting covered design decisions'")
        mem3 = await loom.experience("Team meeting covered important design decisions and tradeoffs")
        text_memories.append(mem3)
        print(f"   Memory ID: {mem3.id}")
        print()

        # =====================================================================
        # Phase 2: Remember Photos
        # =====================================================================
        print("-" * 80)
        print("Phase 2: Remember Photos")
        print("-" * 80)
        print()

        # Create test images
        arch_diagram = create_test_image(600, 400, (100, 150, 255), "Architecture\nDiagram")
        component_diagram = create_test_image(600, 400, (255, 150, 100), "Component\nView")
        whiteboard_photo = create_test_image(600, 400, (150, 255, 150), "Whiteboard\nNotes")

        # Store architecture diagram
        print("📸 Remembering photo: Architecture diagram")
        photo1 = await loom.remember_photo(
            arch_diagram,
            caption="System architecture diagram from the meeting",
            tags=["architecture", "diagram", "system"],
            link_to_memory=mem1.id  # Link to first text memory
        )
        print(f"   Photo ID: {photo1.token_id}")
        print(f"   Caption: {photo1.caption}")
        print(f"   Tags: {', '.join(photo1.tags)}")
        print(f"   Linked to memory: {mem1.id}")
        print()

        # Store component diagram
        print("📸 Remembering photo: Component diagram")
        photo2 = await loom.remember_photo(
            component_diagram,
            caption="Component interaction diagram showing connections",
            tags=["component", "diagram", "connections"]
        )
        print(f"   Photo ID: {photo2.token_id}")
        print(f"   Caption: {photo2.caption}")
        print(f"   Tags: {', '.join(photo2.tags)}")
        print()

        # Store whiteboard photo
        print("📸 Remembering photo: Whiteboard notes")
        photo3 = await loom.remember_photo(
            whiteboard_photo,
            caption="Whiteboard notes from design discussion",
            tags=["whiteboard", "notes", "design"],
            link_to_memory=mem3.id  # Link to team meeting memory
        )
        print(f"   Photo ID: {photo3.token_id}")
        print(f"   Caption: {photo3.caption}")
        print(f"   Tags: {', '.join(photo3.tags)}")
        print(f"   Linked to memory: {mem3.id}")
        print()

        # =====================================================================
        # Phase 3: Multimodal Recall (Text Only - Baseline)
        # =====================================================================
        print("-" * 80)
        print("Phase 3: Multimodal Recall - Text Only (Baseline)")
        print("-" * 80)
        print()

        query1 = "What did we discuss about architecture?"
        print(f"Query: '{query1}'")
        print()

        text_only = await loom.recall(query1, limit=5, include_photos=False)
        print(f"Results ({len(text_only)} text memories):")
        for i, mem in enumerate(text_only, 1):
            print(f"  {i}. {mem.text[:60]}...")
        print()

        # =====================================================================
        # Phase 4: Multimodal Recall (Text + Photos)
        # =====================================================================
        print("-" * 80)
        print("Phase 4: Multimodal Recall - Text + Photos")
        print("-" * 80)
        print()

        query2 = "Show me the architecture diagram"
        print(f"Query: '{query2}'")
        print()

        results = await loom.recall(query2, limit=5, include_photos=True)

        print(f"Text Results ({len(results['text'])}):")
        for i, mem in enumerate(results['text'], 1):
            print(f"  {i}. {mem.text[:60]}...")
        print()

        print(f"Photo Results ({len(results['photos'])}):")
        for i, photo in enumerate(results['photos'], 1):
            score = results['scores']['photos'][i-1] if i-1 < len(results['scores']['photos']) else 0.0
            print(f"  {i}. {photo.caption}")
            print(f"     Tags: {', '.join(photo.tags)}")
            print(f"     Score: {score:.3f}")
        print()

        # =====================================================================
        # Phase 5: Query by Tags
        # =====================================================================
        print("-" * 80)
        print("Phase 5: Query Photos by Tag")
        print("-" * 80)
        print()

        tag = "diagram"
        print(f"Tag: '{tag}'")
        print()

        diagrams = await loom.get_photos_by_tag(tag)
        print(f"Found {len(diagrams)} photos with tag '{tag}':")
        for i, photo in enumerate(diagrams, 1):
            print(f"  {i}. {photo.caption}")
            print(f"     Tags: {', '.join(photo.tags)}")
        print()

        # =====================================================================
        # Phase 6: Visual Similarity Search
        # =====================================================================
        print("-" * 80)
        print("Phase 6: Find Visually Similar Photos")
        print("-" * 80)
        print()

        # Create a query image (similar color to architecture diagram)
        query_image = create_test_image(600, 400, (120, 160, 250), "Query")

        print("Query: Bluish diagram (similar to architecture diagram)")
        print()

        similar = await loom.find_similar_photos(query_image, k=3)
        print(f"Found {len(similar)} similar photos:")
        for i, photo in enumerate(similar, 1):
            score = photo.metadata.get('score', 0.0)
            print(f"  {i}. {photo.caption} (similarity: {score:.3f})")
            print(f"     Tags: {', '.join(photo.tags)}")
        print()

        # =====================================================================
        # Phase 7: Manual Linking
        # =====================================================================
        print("-" * 80)
        print("Phase 7: Manual Photo-Memory Linking")
        print("-" * 80)
        print()

        print(f"Linking photo '{photo2.caption}' to memory '{mem2.id}'")
        await loom.link_photo_to_memory(
            photo2.token_id,
            mem2.id,
            relationship="ILLUSTRATES"
        )
        print("✓ Link created: photo ILLUSTRATES memory")
        print()

        # =====================================================================
        # Phase 8: System Metrics
        # =====================================================================
        print("-" * 80)
        print("Phase 8: System Metrics")
        print("-" * 80)
        print()

        metrics = loom.get_metrics()
        print("Memory System:")
        print(f"  Total memories: {metrics['n_memories']}")
        print(f"  Connections: {metrics['n_connections']}")
        print(f"  Active: {metrics['n_active']}")
        print()

        # Photo memory stats
        if hasattr(loom, '_photo_memory'):
            photo_stats = loom._photo_memory.get_stats()
            print("Photo Memory:")
            print(f"  Total photos: {photo_stats['total_tokens']}")
            print(f"  Total queries: {photo_stats['total_queries']}")
            print(f"  CLIP enabled: {photo_stats['clip_enabled']}")
        print()

        # =====================================================================
        # Summary
        # =====================================================================
        print("=" * 80)
        print("Demo Complete!")
        print("=" * 80)
        print()

        print("What we demonstrated:")
        print("  ✓ Text memory creation (experience)")
        print("  ✓ Photo memory creation (remember_photo)")
        print("  ✓ Automatic entity linking (photo → entities)")
        print("  ✓ Manual memory linking (photo → text)")
        print("  ✓ Multimodal search (text + photos)")
        print("  ✓ Tag-based filtering")
        print("  ✓ Visual similarity search (CLIP)")
        print("  ✓ Knowledge graph integration")
        print()

        print("Key Innovations:")
        print("  • Photos as first-class memories")
        print("  • CLIP embeddings for semantic image-text matching")
        print("  • Automatic knowledge graph integration")
        print("  • Unified recall() API (backward compatible)")
        print("  • Multimodal retrieval (text OR photos OR both)")
        print()

        print("Next Steps:")
        print("  1. Add auto-captioning (BLIP model)")
        print("  2. Integrate OCR (DeepSeek-OCR)")
        print("  3. Face detection and recognition")
        print("  4. Video token support")
        print()


if __name__ == "__main__":
    asyncio.run(main())

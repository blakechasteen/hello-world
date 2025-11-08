#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photo Memory System Demo
=========================

Demonstrates HoloLoom's multimodal visual memory:
1. Store photos with captions and tags
2. Retrieve by text query (CLIP text-image matching)
3. Retrieve by similar image (CLIP image-image matching)
4. Retrieve by tags (categorical filtering)

Requirements:
    pip install Pillow openai-clip torch numpy

Usage:
    python demos/demo_photo_memory.py

What this demo shows:
    - Creating synthetic test images (colored rectangles)
    - Storing images with metadata
    - Text → Image retrieval ("find red images")
    - Image → Image retrieval (find visually similar)
    - Tag-based filtering
    - Performance metrics
"""

import asyncio
import time
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    print("❌ PIL not available. Install: pip install Pillow")
    PIL_AVAILABLE = False
    sys.exit(1)

from HoloLoom.memory.photo_tokens import PhotoTokenMemory, PhotoToken


def create_test_image(
    width: int = 400,
    height: int = 300,
    color: tuple = (255, 0, 0),
    text: str = "Test"
) -> np.ndarray:
    """
    Create a synthetic test image (colored rectangle with text).

    Args:
        width: Image width
        height: Image height
        color: RGB color tuple
        text: Text to draw

    Returns:
        RGB numpy array
    """
    # Create colored rectangle
    img = Image.new('RGB', (width, height), color=color)

    # Draw text
    draw = ImageDraw.Draw(img)

    # Use default font
    try:
        # Try to use a nicer font if available
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    position = (
        (width - text_width) // 2,
        (height - text_height) // 2
    )

    # Draw text with contrasting color
    text_color = (255 - color[0], 255 - color[1], 255 - color[2])
    draw.text(position, text, fill=text_color, font=font)

    return np.array(img)


async def main():
    """Run photo memory demo."""

    print("=" * 80)
    print("Photo Memory System Demo")
    print("=" * 80)
    print()

    # Create memory storage
    storage_path = "./demo_photo_memory"
    print(f"📁 Storage path: {storage_path}")
    print()

    async with PhotoTokenMemory(storage_path=storage_path) as memory:
        print(f"✓ PhotoTokenMemory initialized")
        print(f"  CLIP enabled: {memory.enable_clip}")
        print()

        # Create test images
        print("-" * 80)
        print("Creating Test Images")
        print("-" * 80)
        print()

        test_images = [
            {
                'image': create_test_image(400, 300, (255, 0, 0), "Red"),
                'caption': "A red rectangle",
                'tags': ['red', 'rectangle', 'simple'],
                'name': 'red_rect'
            },
            {
                'image': create_test_image(400, 300, (0, 0, 255), "Blue"),
                'caption': "A blue square",
                'tags': ['blue', 'square', 'simple'],
                'name': 'blue_square'
            },
            {
                'image': create_test_image(600, 200, (0, 255, 0), "Green Wide"),
                'caption': "A green wide rectangle",
                'tags': ['green', 'rectangle', 'wide'],
                'name': 'green_wide'
            },
            {
                'image': create_test_image(400, 300, (255, 255, 0), "Yellow"),
                'caption': "A yellow box",
                'tags': ['yellow', 'rectangle', 'simple'],
                'name': 'yellow_box'
            },
            {
                'image': create_test_image(400, 300, (255, 0, 255), "Magenta"),
                'caption': "A magenta rectangle",
                'tags': ['magenta', 'rectangle', 'simple'],
                'name': 'magenta_rect'
            },
        ]

        # Store images
        tokens = []
        for img_data in test_images:
            start = time.perf_counter()

            token = await memory.store(
                image=img_data['image'],
                caption=img_data['caption'],
                tags=img_data['tags'],
                source='demo'
            )

            elapsed = (time.perf_counter() - start) * 1000

            print(f"✓ Stored: {img_data['name']}")
            print(f"  Token ID: {token.token_id}")
            print(f"  Caption: {token.caption}")
            print(f"  Tags: {', '.join(token.tags)}")
            print(f"  Dimensions: {token.dimensions[0]}×{token.dimensions[1]}")
            print(f"  Latency: {elapsed:.1f}ms")
            print()

            tokens.append(token)

        # Show statistics
        stats = memory.get_stats()
        print(f"📊 Memory Statistics:")
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Storage path: {stats['storage_path']}")
        print()

        # Test 1: Retrieve by text query
        print("-" * 80)
        print("Test 1: Retrieve by Text Query")
        print("-" * 80)
        print()

        queries = [
            "find red images",
            "show me blue",
            "green rectangle",
            "yellow or magenta"
        ]

        for query in queries:
            print(f"Query: '{query}'")

            start = time.perf_counter()
            results = await memory.retrieve_by_text(query, k=3)
            elapsed = (time.perf_counter() - start) * 1000

            print(f"Latency: {elapsed:.1f}ms")
            print(f"Results ({len(results)}):")

            for i, (token, score) in enumerate(results, 1):
                print(f"  {i}. {token.caption} (score: {score:.3f})")
                print(f"     Tags: {', '.join(token.tags)}")

            print()

        # Test 2: Retrieve by similar image
        print("-" * 80)
        print("Test 2: Retrieve by Similar Image")
        print("-" * 80)
        print()

        # Create a query image (reddish color)
        query_image = create_test_image(400, 300, (200, 50, 50), "Query")

        print("Query: Reddish image (200, 50, 50)")
        start = time.perf_counter()
        results = await memory.retrieve_by_image(query_image, k=3)
        elapsed = (time.perf_counter() - start) * 1000

        print(f"Latency: {elapsed:.1f}ms")
        print(f"Results ({len(results)}):")

        for i, (token, score) in enumerate(results, 1):
            print(f"  {i}. {token.caption} (score: {score:.3f})")
            print(f"     Tags: {', '.join(token.tags)}")

        print()

        # Test 3: Retrieve by tags
        print("-" * 80)
        print("Test 3: Retrieve by Tags")
        print("-" * 80)
        print()

        tag_queries = [
            (['simple'], True),  # Match all
            (['red', 'rectangle'], True),
            (['red', 'blue'], False),  # Match any
        ]

        for tags, match_all in tag_queries:
            mode = "AND" if match_all else "OR"
            print(f"Tags: {tags} ({mode})")

            results = await memory.retrieve_by_tags(tags, k=5, match_all=match_all)

            print(f"Results ({len(results)}):")
            for i, token in enumerate(results, 1):
                print(f"  {i}. {token.caption}")
                print(f"     Tags: {', '.join(token.tags)}")

            print()

        # Test 4: Performance summary
        print("-" * 80)
        print("Performance Summary")
        print("-" * 80)
        print()

        final_stats = memory.get_stats()
        print(f"Total tokens stored: {final_stats['total_tokens']}")
        print(f"Total queries: {final_stats['total_queries']}")
        print(f"CLIP enabled: {final_stats['clip_enabled']}")
        print()

        print("✅ Demo complete!")
        print()
        print(f"Stored images can be found in: {storage_path}/images/")
        print(f"Metadata saved to: {storage_path}/metadata.json")
        print(f"Embeddings saved to: {storage_path}/embeddings.npz")


if __name__ == "__main__":
    asyncio.run(main())

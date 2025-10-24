"""
Multimodal Webpage Ingestion Example
=====================================

Demonstrate ingesting webpages with BOTH text and images.

This example shows:
1. Extract text content from webpage
2. Extract meaningful images (skip logos, icons, ads)
3. Download images locally
4. Store with rich metadata (captions, alt text, context)
5. Query across both text and images

Use cases:
- Visual tutorials (code + screenshots)
- Recipe sites (instructions + food images)
- Product reviews (text + product photos)
- Scientific articles (text + diagrams)
"""

import asyncio
import logging
from pathlib import Path

# Setup path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from HoloLoom.spinningWheel.website import WebsiteSpinner, WebsiteSpinnerConfig
from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def ingest_multimodal_webpage(url: str, storage_dir: str = "./images"):
    """
    Ingest a webpage with both text and images.

    Args:
        url: Webpage URL
        storage_dir: Where to save downloaded images
    """
    logger.info("=" * 60)
    logger.info("Multimodal Webpage Ingestion")
    logger.info("=" * 60)
    logger.info(f"URL: {url}")
    logger.info(f"Image storage: {storage_dir}")
    logger.info("")

    # Configure spinner with multimodal options
    config = WebsiteSpinnerConfig(
        # Text options
        chunk_by="paragraph",
        chunk_size=500,

        # Image options (MULTIMODAL!)
        extract_images=True,          # Enable image extraction
        download_images=True,          # Download images locally
        image_storage_dir=storage_dir, # Where to save
        max_images=10,                 # Max images per page
        min_image_width=200,           # Filter small images
        min_image_height=200
    )

    spinner = WebsiteSpinner(config)

    # Spin webpage
    logger.info("Scraping webpage...")
    shards = await spinner.spin({'url': url})

    if not shards:
        logger.error("Failed to process webpage")
        return

    logger.info(f"✓ Created {len(shards)} text chunks")
    logger.info("")

    # Check for images in first shard
    first_shard = shards[0]
    images = first_shard.metadata.get('images', [])

    if images:
        logger.info(f"✓ Extracted {len(images)} images:")
        logger.info("")

        for i, img in enumerate(images, 1):
            logger.info(f"Image {i}:")
            logger.info(f"  URL: {img['url'][:80]}")
            logger.info(f"  Alt text: {img['alt_text']}")
            logger.info(f"  Caption: {img['caption']}")
            logger.info(f"  Dimensions: {img['width']}x{img['height']}")
            logger.info(f"  Format: {img['format']}")
            logger.info(f"  Size: {img['size_bytes']} bytes")
            logger.info(f"  Relevance: {img['relevance_score']:.2f}")
            logger.info(f"  Local path: {img['local_path']}")
            logger.info(f"  Context: {img['context'][:100]}...")
            logger.info("")
    else:
        logger.info("No meaningful images found")
        logger.info("")

    # Store in memory
    logger.info("Storing in memory system...")
    memory = await create_unified_memory(
        user_id="blake",
        enable_mem0=False,
        enable_neo4j=True,
        enable_qdrant=False
    )

    memories = shards_to_memories(shards)
    for mem in memories:
        mem.user_id = "blake"
        mem.tags = ["multimodal", "webpage"]

    ids = await memory.store_many(memories)
    logger.info(f"✓ Stored {len(ids)} memories")
    logger.info("")

    # Demonstrate retrieval
    logger.info("Testing semantic search...")
    from HoloLoom.memory.protocol import Strategy

    # Search text
    results = await memory.recall(
        "main content",
        strategy=Strategy.SEMANTIC,
        limit=3
    )

    logger.info(f"Found {len(results.memories)} relevant chunks:")
    for i, (mem, score) in enumerate(zip(results.memories, results.scores), 1):
        logger.info(f"  {i}. [score: {score:.3f}] {mem.text[:80]}...")

        # Show if this chunk has images
        if 'images' in mem.metadata:
            logger.info(f"     → Contains {mem.metadata['image_count']} images")

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Multimodal ingestion complete!")
    logger.info("=" * 60)


async def compare_text_vs_multimodal():
    """
    Compare text-only vs multimodal ingestion.
    """
    logger.info("=" * 60)
    logger.info("Text-Only vs Multimodal Comparison")
    logger.info("=" * 60)
    logger.info("")

    url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

    # 1. Text-only ingestion
    logger.info("1. TEXT-ONLY INGESTION")
    logger.info("-" * 40)

    config_text_only = WebsiteSpinnerConfig(
        extract_images=False  # Disable images
    )

    spinner_text = WebsiteSpinner(config_text_only)
    shards_text = await spinner_text.spin({'url': url})

    logger.info(f"Chunks: {len(shards_text)}")
    logger.info(f"Images: 0 (disabled)")
    logger.info("")

    # 2. Multimodal ingestion
    logger.info("2. MULTIMODAL INGESTION")
    logger.info("-" * 40)

    config_multimodal = WebsiteSpinnerConfig(
        extract_images=True,
        download_images=True,
        image_storage_dir="./images",
        max_images=5
    )

    spinner_multi = WebsiteSpinner(config_multimodal)
    shards_multi = await spinner_multi.spin({'url': url})

    images = shards_multi[0].metadata.get('images', [])

    logger.info(f"Chunks: {len(shards_multi)}")
    logger.info(f"Images: {len(images)}")
    logger.info("")

    # 3. Show what's different
    logger.info("3. WHAT MULTIMODAL ADDS")
    logger.info("-" * 40)

    if images:
        logger.info("Extracted visual content:")
        for img in images[:3]:
            logger.info(f"  • {img['alt_text']}")
            logger.info(f"    Caption: {img['caption']}")
            logger.info(f"    Context: {img['context'][:60]}...")
            logger.info("")

        logger.info("This visual metadata enables:")
        logger.info("  ✓ Image-based search ('show me the diagram')")
        logger.info("  ✓ Visual context for text chunks")
        logger.info("  ✓ Richer semantic understanding")
        logger.info("  ✓ Better question answering")
    else:
        logger.info("No images found on this page")

    logger.info("")
    logger.info("=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest webpages with text AND images (multimodal)"
    )
    parser.add_argument(
        'url',
        nargs='?',
        default='https://en.wikipedia.org/wiki/Beekeeping',
        help='URL to ingest (default: Wikipedia beekeeping article)'
    )
    parser.add_argument(
        '--storage',
        type=str,
        default='./images',
        help='Where to save images (default: ./images)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run text-only vs multimodal comparison'
    )

    args = parser.parse_args()

    if args.compare:
        asyncio.run(compare_text_vs_multimodal())
    else:
        asyncio.run(ingest_multimodal_webpage(args.url, args.storage))


if __name__ == "__main__":
    main()

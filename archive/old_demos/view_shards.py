#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
View detailed shard information from web crawling.
"""

import asyncio
import json
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, str(Path(__file__).parent))

from HoloLoom.spinningWheel.website import WebsiteSpinner, WebsiteSpinnerConfig


async def view_shards_from_url(url: str, show_full_text: bool = False):
    """
    Scrape a URL and display detailed shard information.

    Args:
        url: URL to scrape
        show_full_text: If True, shows full text; if False, shows preview
    """
    print("=" * 80)
    print(f"Scraping: {url}")
    print("=" * 80)

    config = WebsiteSpinnerConfig(
        extract_images=True,
        max_images=3,
        chunk_by='paragraph',
        chunk_size=500,
        timeout=15
    )

    spinner = WebsiteSpinner(config)

    try:
        shards = await spinner.spin({'url': url})

        print(f"\n✓ Generated {len(shards)} shard(s)\n")

        for idx, shard in enumerate(shards, 1):
            print("=" * 80)
            print(f"SHARD {idx} of {len(shards)}")
            print("=" * 80)

            # Basic info
            print(f"\nID: {shard.id}")
            print(f"Episode: {shard.episode}")
            print(f"Text Length: {len(shard.text)} characters")
            print(f"Word Count: {len(shard.text.split())} words")

            # Entities
            print(f"\nEntities ({len(shard.entities)}):")
            if shard.entities:
                for entity in shard.entities[:10]:  # Show first 10
                    print(f"  - {entity}")
                if len(shard.entities) > 10:
                    print(f"  ... and {len(shard.entities) - 10} more")
            else:
                print("  (none)")

            # Motifs
            print(f"\nMotifs ({len(shard.motifs)}):")
            if shard.motifs:
                for motif in shard.motifs[:10]:
                    print(f"  - {motif}")
                if len(shard.motifs) > 10:
                    print(f"  ... and {len(shard.motifs) - 10} more")
            else:
                print("  (none)")

            # Metadata
            print(f"\nMetadata:")
            for key, value in shard.metadata.items():
                if key == 'images':
                    print(f"  {key}: {len(value)} image(s)")
                    for img_idx, img in enumerate(value, 1):
                        print(f"    Image {img_idx}:")
                        print(f"      - src: {img.get('src', 'N/A')[:80]}...")
                        print(f"      - alt: {img.get('alt', 'N/A')}")
                        print(f"      - context: {img.get('context', 'N/A')[:60]}...")
                        if img.get('local_path'):
                            print(f"      - saved to: {img['local_path']}")
                elif key == 'tags':
                    print(f"  {key}: {', '.join(value) if isinstance(value, list) else value}")
                elif isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")

            # Text content
            print(f"\nText Content:")
            print("-" * 80)
            if show_full_text:
                print(shard.text)
            else:
                # Show first 500 chars
                preview = shard.text[:500]
                print(preview)
                if len(shard.text) > 500:
                    print(f"\n... [{len(shard.text) - 500} more characters]")
            print("-" * 80)
            print()

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        total_chars = sum(len(s.text) for s in shards)
        total_words = sum(len(s.text.split()) for s in shards)
        total_entities = sum(len(s.entities) for s in shards)
        total_motifs = sum(len(s.motifs) for s in shards)

        print(f"Total Shards: {len(shards)}")
        print(f"Total Characters: {total_chars:,}")
        print(f"Total Words: {total_words:,}")
        print(f"Total Entities: {total_entities}")
        print(f"Total Motifs: {total_motifs}")

        # Images
        total_images = sum(
            len(s.metadata.get('images', []))
            for s in shards
            if isinstance(s.metadata.get('images'), list)
        )
        print(f"Total Images: {total_images}")

        return shards

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


async def main():
    """Main entry point."""

    # Default URL - Claude docs best practices
    url = "https://docs.claude.com/en/docs/agents-and-tools/agent-skills/best-practices"

    # Check for command line arguments
    if len(sys.argv) > 1:
        url = sys.argv[1]

    show_full = '--full' in sys.argv

    if show_full:
        print("\n(Showing FULL text content)\n")
    else:
        print("\n(Showing text PREVIEW - use --full flag for complete text)\n")

    await view_shards_from_url(url, show_full_text=show_full)

    print("\n" + "=" * 80)
    print("Usage:")
    print(f"  python {Path(__file__).name} [URL] [--full]")
    print("\nExamples:")
    print(f"  python {Path(__file__).name}")
    print(f"  python {Path(__file__).name} https://example.com/article")
    print(f"  python {Path(__file__).name} --full")
    print("=" * 80)


if __name__ == '__main__':
    asyncio.run(main())

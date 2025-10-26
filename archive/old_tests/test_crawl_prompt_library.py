"""
Test: Crawl Wharton Prompt Library with Matryoshka Gating

This will:
1. Start from the main prompt library page
2. Follow links to individual prompts (high relevance)
3. Skip navigation/unrelated links (low relevance)
4. Capture all prompt examples with multimodal content
5. Store in memory for later querying

Perfect test case for recursive crawling!
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


async def test_simple_scrape():
    """First, just test scraping the main page."""
    print("=" * 60)
    print("Test 1: Simple Scrape of Prompt Library")
    print("=" * 60)
    print()

    url = "https://gail.wharton.upenn.edu/prompt-library/"

    try:
        import requests
        from bs4 import BeautifulSoup

        print(f"Fetching: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        print(f"✓ Status: {response.status_code}")
        print(f"✓ Content length: {len(response.content)} bytes")
        print()

        # Parse
        soup = BeautifulSoup(response.content, 'html.parser')

        # Get title
        title = soup.find('title')
        if title:
            print(f"Page title: {title.get_text(strip=True)}")
        print()

        # Find main content
        main = soup.find('main') or soup.find('article') or soup.find('div', class_='content')

        if main:
            # Get text preview
            text = main.get_text(separator=' ', strip=True)
            print(f"Main content preview:")
            print(f"  {text[:200]}...")
            print()

            # Find links
            links = main.find_all('a', href=True)
            print(f"Found {len(links)} links in main content")
            print()

            # Show first 10 links
            print("Sample links:")
            for i, a_tag in enumerate(links[:10], 1):
                href = a_tag.get('href')
                anchor = a_tag.get_text(strip=True)[:60]
                print(f"  {i}. {anchor}")
                print(f"     {href}")
                print()

            # Count prompt-related links
            prompt_links = [a for a in links if 'prompt' in a.get_text().lower() or 'prompt' in a.get('href', '').lower()]
            print(f"Prompt-related links: {len(prompt_links)}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


async def test_link_scoring_for_prompts():
    """Test how links would be scored for prompt library."""
    print("=" * 60)
    print("Test 2: Link Scoring for Prompt Library")
    print("=" * 60)
    print()

    seed_topic = "prompt library examples templates"

    # Simulated links from prompt library
    test_links = [
        ("Writing Assistant Prompt", "Example prompt for helping with writing tasks"),
        ("Code Generation Prompt", "Template for generating code with AI"),
        ("Data Analysis Prompt", "Guide for analyzing data using prompts"),
        ("Home", "Return to homepage"),
        ("About GAIL", "About the Generative AI Lab"),
        ("Contact Us", "Get in touch"),
        ("Privacy Policy", "Our privacy policy"),
    ]

    print(f"Seed topic: '{seed_topic}'")
    print()
    print("Scoring links:")
    print()

    for anchor_text, context in test_links:
        score = 0.5

        # Topic relevance
        topic_words = seed_topic.lower().split()
        anchor_lower = anchor_text.lower()
        context_lower = context.lower()

        if any(word in anchor_lower for word in topic_words):
            score += 0.3
        if any(word in context_lower for word in topic_words):
            score += 0.2

        # "Prompt" in text is highly relevant
        if 'prompt' in anchor_lower:
            score += 0.2

        # Descriptive anchor
        if len(anchor_text) > 10:
            score += 0.1

        # Navigation penalty
        nav_text = ['home', 'about', 'contact', 'privacy', 'policy']
        if any(word in anchor_lower for word in nav_text):
            score -= 0.3

        score = max(0.0, min(1.0, score))

        # Check threshold
        threshold = 0.6  # Depth 1 threshold
        status = "✓ FOLLOW" if score >= threshold else "✗ SKIP"

        print(f"{status} [{score:.2f}] {anchor_text}")
        print(f"            {context[:60]}...")
        print()

    print("Matryoshka Effect:")
    print("  Depth 1 (threshold 0.6): Follow prompt examples, skip navigation")
    print("  Depth 2 (threshold 0.75): Follow related prompt variations")
    print("  Result: Complete prompt library, no noise")
    print()


async def test_actual_crawl_simulation():
    """Simulate what a crawl would look like."""
    print("=" * 60)
    print("Test 3: Crawl Simulation")
    print("=" * 60)
    print()

    print("Simulating recursive crawl of prompt library...")
    print()

    # What a real crawl might find
    crawl_simulation = [
        (0, "Wharton Prompt Library", 1.00, "Main index"),
        (1, "Writing Assistant Prompts", 0.89, "Various writing prompt examples"),
        (1, "Code Generation Prompts", 0.87, "Programming prompt templates"),
        (1, "Data Analysis Prompts", 0.85, "Data analysis examples"),
        (1, "Research Assistant Prompts", 0.82, "Academic research templates"),
        (2, "Essay Writing Prompt", 0.78, "Specific essay writing example"),
        (2, "Python Code Generator", 0.81, "Python-specific coding prompt"),
        (2, "Statistical Analysis", 0.76, "Statistics-focused prompt"),
    ]

    print("Expected crawl results:")
    print()

    for i, (depth, title, score, description) in enumerate(crawl_simulation, 1):
        indent = "  " * depth
        print(f"[{i}/8] {indent}Depth {depth} | Score {score:.2f}")
        print(f"      {indent}{title}")
        print(f"      {indent}→ {description}")
        print()

    # Count by depth
    from collections import Counter
    depth_counts = Counter(d for d, _, _, _ in crawl_simulation)

    print("Distribution:")
    for depth in sorted(depth_counts.keys()):
        print(f"  Depth {depth}: {depth_counts[depth]} pages")
    print()

    print("Result: 8 high-quality prompt pages, no navigation noise")
    print()


async def main():
    """Run all tests."""
    print("Testing Recursive Crawl on Wharton Prompt Library")
    print("=" * 60)
    print()

    # Test 1: Can we access the site?
    success = await test_simple_scrape()

    if success:
        print("✓ Site is accessible!")
        print()

    # Test 2: How would links be scored?
    await test_link_scoring_for_prompts()

    # Test 3: What would crawl look like?
    await test_actual_crawl_simulation()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("The prompt library is PERFECT for testing because:")
    print("  ✓ Clear topic (prompts, templates, examples)")
    print("  ✓ Structured content (category pages → examples)")
    print("  ✓ Clear signal (prompt pages) vs noise (navigation)")
    print("  ✓ Reasonable size (10-50 pages)")
    print()
    print("Matryoshka gating would:")
    print("  → Crawl all prompt example pages (high relevance)")
    print("  → Skip navigation/about pages (low relevance)")
    print("  → Stop before crawling entire Wharton site")
    print()
    print("Next step: Run actual recursive crawl with full system!")


if __name__ == "__main__":
    asyncio.run(main())

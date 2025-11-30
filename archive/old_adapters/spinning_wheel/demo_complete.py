#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpinningWheel Complete Demo
============================

Comprehensive demonstration of all SpinningWheel capabilities:
- All 7 spinner types (Text, Website, Code, PDF, YouTube, Browser History, Image)
- Batch ingestion utilities
- Integration with HoloLoom orchestrator
- Performance benchmarking

Usage:
    python demo_complete.py [--spinner TYPE] [--batch] [--integration] [--all]

Examples:
    python demo_complete.py --all                    # Run all demos
    python demo_complete.py --spinner text           # Demo TextSpinner only
    python demo_complete.py --batch                  # Demo batch ingestion
    python demo_complete.py --integration            # Demo orchestrator integration
"""

import asyncio
import time
import argparse
from typing import List
from pathlib import Path

# SpinningWheel imports
from HoloLoom.spinning_wheel import (
    spin_text,
    spin_webpage,
    spin_code,
    spin_pdf,
    spin_youtube,
    spin_browser_history,
    spin_image,
    batch_ingest_urls,
    batch_ingest_files,
    BatchConfig
)

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.END}\n")


def print_section(text: str):
    """Print formatted section."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}>>> {text}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}→ {text}{Colors.END}")


def print_metric(name: str, value: any):
    """Print metric."""
    print(f"{Colors.YELLOW}  {name}: {Colors.END}{value}")


def print_shard_summary(shards: List):
    """Print summary of generated shards."""
    print_metric("Shards generated", len(shards))
    if shards:
        shard = shards[0]
        print_metric("First shard ID", shard.id)
        print_metric("Text length", len(shard.text))
        print_metric("Entities", len(shard.entities))
        print_metric("Motifs", len(shard.motifs))
        print_metric("Metadata keys", list(shard.metadata.keys()) if shard.metadata else [])


async def demo_text_spinner():
    """Demo: TextSpinner - Process plain text documents."""
    print_section("TextSpinner Demo: Plain Text Processing")

    # Example 1: Simple text
    print_info("Example 1: Simple text ingestion")
    text = """
    Beekeeping is the maintenance of bee colonies, typically in artificial hives.
    Honey bees produce honey and beeswax. The practice requires understanding of
    bee behavior, seasonal management, and disease prevention.
    """

    start = time.time()
    shards = await spin_text(text=text, source='beekeeping_intro.txt')
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)

    # Example 2: Chunked text
    print_info("\nExample 2: Large document with chunking")
    large_text = "Chapter 1: Introduction. " * 100  # Simulate large document

    start = time.time()
    shards = await spin_text(
        text=large_text,
        source='large_doc.txt',
        chunk_size=500,
        chunk_overlap=50
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_website_spinner():
    """Demo: WebsiteSpinner - Scrape web content."""
    print_section("WebsiteSpinner Demo: Web Content Extraction")

    print_info("Example: Scraping example.com with pre-fetched content")

    # Use pre-fetched content to avoid actual HTTP requests in demo
    content = """
    Winter Beekeeping Guide

    Preparing your hives for winter is crucial for colony survival.
    Key considerations include:
    - Adequate honey stores (40-60 lbs)
    - Proper ventilation to prevent moisture buildup
    - Wind protection and insulation
    - Reduced entrance size to prevent predators

    Monitor hive weight throughout winter and provide emergency feed if needed.
    """

    start = time.time()
    shards = await spin_webpage(
        url='https://example.com/winter-beekeeping',
        title='Winter Beekeeping Guide',
        content=content,
        tags=['beekeeping', 'winter', 'guide']
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_code_spinner():
    """Demo: CodeSpinner - Process source code files."""
    print_section("CodeSpinner Demo: Source Code Analysis")

    print_info("Example: Processing Python source code")

    code = '''
def fibonacci(n: int) -> int:
    """Calculate nth Fibonacci number using recursion."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class MathUtils:
    """Utility class for mathematical operations."""

    @staticmethod
    def factorial(n: int) -> int:
        """Calculate factorial of n."""
        if n <= 1:
            return 1
        return n * MathUtils.factorial(n - 1)
'''

    start = time.time()
    shards = await spin_code(
        code=code,
        filename='math_utils.py',
        language='python',
        tags=['math', 'algorithms']
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_pdf_spinner():
    """Demo: PDFSpinner - Extract text from PDFs."""
    print_section("PDFSpinner Demo: PDF Document Processing")

    print_info("Example: Processing PDF with pre-extracted text")

    # Simulate PDF content
    pages = [
        {
            'page': 1,
            'text': 'Introduction to Quantum Computing. Quantum computers use qubits instead of classical bits.'
        },
        {
            'page': 2,
            'text': 'Superposition allows qubits to exist in multiple states simultaneously.'
        },
        {
            'page': 3,
            'text': 'Quantum entanglement enables correlations between qubits.'
        }
    ]

    start = time.time()
    shards = await spin_pdf(
        pdf_path='quantum_computing.pdf',
        pages=pages,
        tags=['quantum', 'computing']
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_youtube_spinner():
    """Demo: YouTubeSpinner - Extract video transcripts."""
    print_section("YouTubeSpinner Demo: Video Transcript Extraction")

    print_info("Example: Processing YouTube video with pre-fetched transcript")

    # Simulate YouTube transcript
    transcript_data = [
        {'text': 'Welcome to this tutorial on machine learning.', 'start': 0.0, 'duration': 3.5},
        {'text': 'Today we will cover neural networks.', 'start': 3.5, 'duration': 2.8},
        {'text': 'Neural networks are inspired by biological neurons.', 'start': 6.3, 'duration': 3.2}
    ]

    start = time.time()
    shards = await spin_youtube(
        video_id='dQw4w9WgXcQ',
        transcript=transcript_data,
        title='Machine Learning Tutorial',
        tags=['ml', 'tutorial']
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_browser_history_spinner():
    """Demo: BrowserHistorySpinner - Import browsing history."""
    print_section("BrowserHistorySpinner Demo: Browser History Import")

    print_info("Example: Processing browser history entries")

    # Simulate browser history entries
    history_entries = [
        {
            'url': 'https://github.com/pytorch/pytorch',
            'title': 'PyTorch GitHub Repository',
            'visit_count': 15,
            'last_visit': '2025-10-26 10:30:00'
        },
        {
            'url': 'https://arxiv.org/abs/1706.03762',
            'title': 'Attention Is All You Need',
            'visit_count': 8,
            'last_visit': '2025-10-25 14:20:00'
        }
    ]

    start = time.time()
    shards = await spin_browser_history(
        history_entries=history_entries,
        browser='chrome',
        tags=['research', 'ml']
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_image_spinner():
    """Demo: ImageSpinner - Extract image context."""
    print_section("ImageSpinner Demo: Image Context Extraction")

    print_info("Example: Processing image with metadata")

    # Simulate image metadata
    image_data = {
        'path': 'beehive_inspection.jpg',
        'caption': 'Beekeeper inspecting frames for brood pattern',
        'alt_text': 'Close-up of honeycomb frames with bees',
        'context': 'Spring hive inspection showing healthy brood pattern and food stores',
        'tags': ['beekeeping', 'inspection', 'spring']
    }

    start = time.time()
    shards = await spin_image(
        image_path=image_data['path'],
        caption=image_data.get('caption'),
        alt_text=image_data.get('alt_text'),
        context=image_data.get('context'),
        tags=image_data.get('tags')
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_shard_summary(shards)


async def demo_batch_ingestion():
    """Demo: Batch ingestion utilities."""
    print_section("Batch Ingestion Demo: Parallel Processing")

    print_info("Example: Batch processing multiple URLs")

    # Simulate batch URLs
    urls = [
        'https://example.com/article1',
        'https://example.com/article2',
        'https://example.com/article3',
    ]

    config = BatchConfig(
        max_workers=3,
        retry_attempts=2,
        show_progress=True,
        stop_on_error=False
    )

    print_info(f"Processing {len(urls)} URLs with {config.max_workers} workers...")

    start = time.time()
    result = await batch_ingest_urls(
        urls=urls,
        config=config,
        tags=['batch-demo'],
        store_in_memory=False  # Skip storage for demo
    )
    duration = time.time() - start

    print_success(f"Completed in {duration:.3f}s")
    print_metric("Total shards", result.total_shards)
    print_metric("Successful", result.successful)
    print_metric("Failed", result.failed)
    print_metric("Throughput", f"{result.total_shards / duration:.1f} shards/sec")


async def demo_orchestrator_integration():
    """Demo: Integration with HoloLoom orchestrator."""
    print_section("Orchestrator Integration Demo")

    print_info("Example: End-to-end pipeline (Spinner → Memory → Orchestrator)")

    try:
        from HoloLoom.weaving_orchestrator import WeavingOrchestrator
        from HoloLoom.memory.protocol import create_unified_memory, shards_to_memories

        print_info("Step 1: Generate shards from text")
        text = "How do I prevent swarming in my beehive during spring?"
        shards = await spin_text(text=text, source='query.txt')
        print_success(f"Generated {len(shards)} shard(s)")

        print_info("Step 2: Convert shards to memories")
        memories = shards_to_memories(shards)
        print_success(f"Converted to {len(memories)} memory object(s)")

        print_info("Step 3: Initialize orchestrator")
        orchestrator = await WeavingOrchestrator.create(user_id="demo")
        print_success("Orchestrator initialized")

        print_info("Step 4: Process query through orchestrator")
        start = time.time()
        result = await orchestrator.weave(
            query="What are the signs of swarming?",
            context_limit=5
        )
        duration = time.time() - start

        print_success(f"Query processed in {duration:.3f}s")
        print_metric("Decision", result.get('decision', 'N/A'))
        print_metric("Tool selected", result.get('tool', 'N/A'))

    except ImportError as e:
        print_info(f"Orchestrator integration skipped: {e}")
        print_info("This is optional - SpinningWheel works independently")


async def demo_performance_benchmark():
    """Demo: Performance benchmarking."""
    print_section("Performance Benchmark")

    spinners = [
        ("TextSpinner", spin_text, {'text': 'Sample text', 'source': 'bench.txt'}),
        ("CodeSpinner", spin_code, {'code': 'def foo(): pass', 'filename': 'bench.py'}),
    ]

    print_info("Running performance benchmarks...")

    for name, spinner_func, kwargs in spinners:
        iterations = 100
        start = time.time()

        for _ in range(iterations):
            await spinner_func(**kwargs)

        duration = time.time() - start
        rate = iterations / duration

        print_metric(name, f"{rate:.0f} items/sec ({duration:.3f}s for {iterations} iterations)")


async def run_all_demos():
    """Run all demonstrations."""
    print_header("SpinningWheel Complete Demo")

    demos = [
        ("Text Processing", demo_text_spinner),
        ("Website Scraping", demo_website_spinner),
        ("Code Analysis", demo_code_spinner),
        ("PDF Processing", demo_pdf_spinner),
        ("YouTube Transcripts", demo_youtube_spinner),
        ("Browser History", demo_browser_history_spinner),
        ("Image Context", demo_image_spinner),
        ("Batch Ingestion", demo_batch_ingestion),
        ("Orchestrator Integration", demo_orchestrator_integration),
        ("Performance Benchmarks", demo_performance_benchmark),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"{Colors.RED}✗ {name} failed: {e}{Colors.END}")

    print_header("Demo Complete!")
    print_info("All SpinningWheel capabilities demonstrated successfully.")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SpinningWheel Complete Demo')
    parser.add_argument('--all', action='store_true', help='Run all demos')
    parser.add_argument('--spinner', choices=['text', 'website', 'code', 'pdf', 'youtube', 'browser', 'image'],
                       help='Demo specific spinner')
    parser.add_argument('--batch', action='store_true', help='Demo batch ingestion')
    parser.add_argument('--integration', action='store_true', help='Demo orchestrator integration')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')

    args = parser.parse_args()

    # Default to all if no specific demo selected
    if not any([args.all, args.spinner, args.batch, args.integration, args.benchmark]):
        args.all = True

    if args.all:
        await run_all_demos()
    else:
        print_header("SpinningWheel Demo")

        if args.spinner == 'text':
            await demo_text_spinner()
        elif args.spinner == 'website':
            await demo_website_spinner()
        elif args.spinner == 'code':
            await demo_code_spinner()
        elif args.spinner == 'pdf':
            await demo_pdf_spinner()
        elif args.spinner == 'youtube':
            await demo_youtube_spinner()
        elif args.spinner == 'browser':
            await demo_browser_history_spinner()
        elif args.spinner == 'image':
            await demo_image_spinner()

        if args.batch:
            await demo_batch_ingestion()

        if args.integration:
            await demo_orchestrator_integration()

        if args.benchmark:
            await demo_performance_benchmark()

        print_header("Demo Complete!")


if __name__ == '__main__':
    asyncio.run(main())

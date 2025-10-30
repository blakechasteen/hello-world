#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Ultra Dashboard Generator - No Interactive Prompts
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from HoloLoom.warp.math_pipeline_elegant import ElegantMathPipeline
from HoloLoom.warp.math_dashboard_ultra import generate_ultra_dashboard


async def generate_demo_dashboard():
    """Generate ultra dashboard with demo data."""
    print("Generating ultra-polished dashboard...")
    print()

    # Create pipeline and run queries
    async with (ElegantMathPipeline()
        .fast()
        .enable_rl()
    ) as pipeline:

        # Run multiple queries to generate data
        queries = [
            "Find similar documents to AI",
            "Find similar documents to ML",
            "Optimize search performance",
            "Optimize retrieval speed",
            "Analyze graph topology",
            "Verify metric properties",
            "Transform feature space"
        ]

        print("Running queries to generate data...")
        results = []
        for i, query in enumerate(queries, 1):
            result = await pipeline.analyze(query)
            if result:
                results.append({
                    "execution_time_ms": result.execution_time_ms,
                    "confidence": result.confidence,
                    "total_cost": result.total_cost,
                    "operations_used": result.operations_used
                })
                print(f"  [{i}/{len(queries)}] {query[:50]}... ✓")

        # Get statistics
        stats = pipeline.statistics()

        print()
        print("Generating ultra dashboard...")

        # Generate ultra dashboard
        output_path = generate_ultra_dashboard(stats, results)

        print()
        print("=" * 80)
        print("✨ ULTRA DASHBOARD GENERATED!")
        print("=" * 80)
        print()
        print(f"File: {Path(output_path).absolute()}")
        print()
        print("Features:")
        print("  • Modern design system (CSS variables)")
        print("  • Professional typography (Inter + JetBrains Mono)")
        print("  • Animated gradient background")
        print("  • 30 floating particles")
        print("  • Enhanced glassmorphism")
        print("  • Smooth micro-interactions")
        print("  • Fade-in stagger animations")
        print("  • Medal glow effects")
        print("  • Custom Plotly.js dark theme")
        print()
        print(f"Open in browser: file:///{Path(output_path).absolute()}")
        print()


if __name__ == "__main__":
    asyncio.run(generate_demo_dashboard())

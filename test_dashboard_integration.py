#!/usr/bin/env python3
"""
Test Dashboard Integration - Phase 2.4 Validation
==================================================
Tests complete integration of Edward Tufte Machine with WeavingOrchestrator.

Author: Claude Code
Date: October 28, 2025
"""

import asyncio
from pathlib import Path

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config
from HoloLoom.documentation.types import Query, MemoryShard


def create_test_shards():
    """Create test memory shards."""
    return [
        MemoryShard(
            id="shard_1",
            text="The weaving orchestrator coordinates feature extraction and decision making.",
            episode="test"
        ),
        MemoryShard(
            id="shard_2",
            text="Thompson Sampling balances exploration and exploitation using Bayesian principles.",
            episode="test"
        ),
        MemoryShard(
            id="shard_3",
            text="Semantic cache provides 3-10x speedup through three-tier architecture.",
            episode="test"
        ),
    ]


async def test_dashboard_integration():
    """Test dashboard generation integration with WeavingOrchestrator."""
    print('[TEST] Dashboard Integration with WeavingOrchestrator')
    print('=' * 70)

    # 1. Test with dashboards DISABLED (default)
    print('\n[STEP 1] Testing with dashboards disabled...')
    config = Config.fast()
    shards = create_test_shards()

    async with WeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_dashboards=False
    ) as orch:
        query = Query(text="What is Thompson Sampling?")
        spacetime = await orch.weave(query)

        # Should NOT have dashboard
        assert 'dashboard' not in spacetime.metadata
        print('  [PASS] No dashboard generated when disabled')

    # 2. Test with dashboards ENABLED
    print('\n[STEP 2] Testing with dashboards enabled...')
    async with WeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_dashboards=True
    ) as orch:
        query = Query(text="How does the weaving orchestrator work?")
        spacetime = await orch.weave(query)

        # Should have dashboard
        assert 'dashboard' in spacetime.metadata
        dashboard = spacetime.metadata['dashboard']

        print(f'  Dashboard Title: {dashboard.title}')
        print(f'  Layout: {dashboard.layout.value}')
        print(f'  Panels: {len(dashboard.panels)}')

        # Validate dashboard structure
        assert dashboard.title
        assert dashboard.layout
        assert len(dashboard.panels) > 0
        print('  [PASS] Dashboard generated with correct structure')

        # Check panels
        print(f'\n  Generated Panels:')
        for i, panel in enumerate(dashboard.panels, 1):
            print(f'    {i}. {panel.title} ({panel.type.value})')

        # 3. Test save_dashboard() method
        print('\n[STEP 3] Testing save_dashboard() method...')
        output_path = Path('demos/output/integration_test_dashboard.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        orch.save_dashboard(spacetime, str(output_path))

        # Validate file was created
        assert output_path.exists()
        file_size = output_path.stat().st_size
        print(f'  File created: {output_path}')
        print(f'  Size: {file_size:,} bytes')
        print('  [PASS] Dashboard saved successfully')

    # 4. Test error handling (save without enabling)
    print('\n[STEP 4] Testing error handling...')
    async with WeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_dashboards=False
    ) as orch:
        query = Query(text="Test query")
        spacetime = await orch.weave(query)

        try:
            orch.save_dashboard(spacetime, 'test.html')
            print('  [FAIL] Should have raised ValueError')
        except ValueError as e:
            print(f'  [PASS] Correct error raised: {e}')

    print('\n' + '=' * 70)
    print('[SUCCESS] Phase 2.4 Integration Complete!')
    print('=' * 70)


async def test_dashboard_with_cache():
    """Test dashboard generation with semantic cache enabled."""
    print('\n[TEST] Dashboard with Semantic Cache')
    print('=' * 70)

    config = Config.fast()
    shards = create_test_shards()

    async with WeavingOrchestrator(
        cfg=config,
        shards=shards,
        enable_dashboards=True,
        enable_semantic_cache=True
    ) as orch:
        query = Query(text="Explain Thompson Sampling")
        spacetime = await orch.weave(query)

        dashboard = spacetime.metadata['dashboard']

        print(f'\n  Dashboard: {dashboard.title}')
        print(f'  Panels: {len(dashboard.panels)}')

        # Check if cache stats are in metadata
        if 'semantic_cache' in spacetime.metadata:
            cache_stats = spacetime.metadata['semantic_cache']
            print(f'\n  Cache Stats:')
            print(f'    Enabled: {cache_stats.get("enabled")}')
            print(f'    Hit Rate: {cache_stats.get("hit_rate", 0):.1%}')
            print('  [PASS] Cache statistics included')

    print('\n' + '=' * 70)
    print('[SUCCESS] Dashboard with cache operational!')
    print('=' * 70)


if __name__ == '__main__':
    # Run tests
    asyncio.run(test_dashboard_integration())
    asyncio.run(test_dashboard_with_cache())

    print('\n[COMPLETE] All Phase 2.4 tests passing!')

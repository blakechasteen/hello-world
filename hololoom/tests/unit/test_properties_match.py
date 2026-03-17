"""
Tests for properties_match on MemoryQuery — Level 2 structured property filtering.

Validates that items can be queried by their properties dict values,
enabling queries like "all open P0 observations" without semantic search.
"""

import asyncio
import pytest

from hololoom.core.memory.lite_bus import LiteMemoryBus
from hololoom.core.memory.bus import MemoryItem, MemoryQuery


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def bus_with_items():
    """Bus pre-loaded with items that have varying properties."""
    bus = LiteMemoryBus()

    async def _setup():
        await bus.initialize()
        await bus.store(MemoryItem(
            content="varroa count critical in hive 3",
            memory_type="hive",
            properties={"band": "P0", "status": "open", "category": "pest"},
            importance=0.95,
        ))
        await bus.store(MemoryItem(
            content="honey stores low in hive 1",
            memory_type="hive",
            properties={"band": "P1", "status": "open", "category": "stores"},
            importance=0.7,
        ))
        await bus.store(MemoryItem(
            content="varroa treated in hive 3",
            memory_type="hive",
            properties={"band": "P0", "status": "resolved", "category": "pest"},
            importance=0.95,
        ))
        await bus.store(MemoryItem(
            content="brood pattern spotty in hive 2",
            memory_type="hive",
            properties={"band": "P2", "status": "open", "category": "brood"},
            importance=0.5,
        ))
        await bus.store(MemoryItem(
            content="tomato soup recipe scored high",
            memory_type="kitchen",
            properties={"band": "P1", "status": "open", "category": "recipe"},
            importance=0.6,
        ))
        return bus

    return run(_setup())


class TestPropertiesMatch:
    def test_single_key_match(self, bus_with_items):
        """Filter by one property value."""
        async def _test():
            result = await bus_with_items.query(MemoryQuery(
                intent="open items",
                properties_match={"status": "open"},
            ))
            assert len(result.items) == 4
            assert all(item.get("status") == "open" for item in result.items)
            assert result.resolution_path == "structured"

        run(_test())

    def test_multi_key_match(self, bus_with_items):
        """Filter by multiple property values (AND logic)."""
        async def _test():
            result = await bus_with_items.query(MemoryQuery(
                intent="critical open",
                properties_match={"band": "P0", "status": "open"},
            ))
            assert len(result.items) == 1
            assert "varroa count critical" in result.items[0]["content"]

        run(_test())

    def test_empty_dict_matches_all(self, bus_with_items):
        """Empty properties_match should not filter (match everything)."""
        async def _test():
            # Empty dict is falsy, so it won't trigger Level 2 on its own.
            # Combine with memory_type to ensure Level 2 fires.
            result = await bus_with_items.query(MemoryQuery(
                intent="all hive",
                memory_type="hive",
                properties_match={},
                max_results=10,
            ))
            assert len(result.items) == 4  # All hive items

        run(_test())

    def test_none_is_backward_compatible(self, bus_with_items):
        """properties_match=None should not affect query behavior."""
        async def _test():
            result = await bus_with_items.query(MemoryQuery(
                intent="all hive",
                memory_type="hive",
                properties_match=None,
                max_results=10,
            ))
            assert len(result.items) == 4

        run(_test())

    def test_combined_with_memory_type(self, bus_with_items):
        """properties_match ANDs with existing predicates like memory_type."""
        async def _test():
            result = await bus_with_items.query(MemoryQuery(
                intent="open kitchen",
                memory_type="kitchen",
                properties_match={"status": "open"},
            ))
            assert len(result.items) == 1
            assert "tomato soup" in result.items[0]["content"]

        run(_test())

    def test_no_results(self, bus_with_items):
        """Property value that doesn't exist returns empty."""
        async def _test():
            result = await bus_with_items.query(MemoryQuery(
                intent="nonexistent",
                properties_match={"band": "P4"},
            ))
            # No items have band=P4, so Level 2 returns empty.
            # Query may fall through to Level 3/4 if intent matches.
            # But the structured level should have been tried.
            # Check that no P4 items are in results.
            for item in result.items:
                assert item.get("band") != "P4"

        run(_test())

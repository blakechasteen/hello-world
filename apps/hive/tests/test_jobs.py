"""Tests for hive jobs — scoring pipeline integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import asyncio
import pytest

from hololoom.core.memory.observation_reflector import ObservationReflector
from hololoom.core.memory.bus import MemoryQuery

from hive.bus_client import HiveBusClient
from hive.models import SAMPLE_HIVES
from hive.jobs import job_inspect, job_varroa, job_stores


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestJobInspect:
    def test_scores_all_sample_hives(self, tmp_path):
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",  # Force local fallback
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            assert client.mode == "local"

            results = await job_inspect(client, reflector, SAMPLE_HIVES, mode="spring")
            await client.close()
            return results

        results = run(_test())
        assert len(results) == 3  # 3 sample hives
        # Each result has expected shape
        for r in results:
            assert "hive_id" in r
            assert "band" in r
            assert r["band"] in ("P0", "P1", "P2", "P3")
            assert "composite" in r
            assert r["node_id"] is not None

    def test_dry_run_does_not_store(self, tmp_path):
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            results = await job_inspect(client, reflector, SAMPLE_HIVES, dry_run=True)
            # Bus should be empty
            assert len(client._bus._items) == 0
            await client.close()
            return results

        results = run(_test())
        assert len(results) == 3
        assert all(r["node_id"] is None for r in results)

    def test_single_hive_filter(self, tmp_path):
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            results = await job_inspect(client, reflector, SAMPLE_HIVES, hive_id="hive-2")
            await client.close()
            return results

        results = run(_test())
        assert len(results) == 1
        assert results[0]["hive_id"] == "hive-2"

    def test_troubled_hive_scores_higher(self, tmp_path):
        """Clover (hive-2) has varroa, no queen, disease — should outscore others."""
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            results = await job_inspect(client, reflector, SAMPLE_HIVES)
            await client.close()
            return results

        results = run(_test())
        scores = {r["hive_id"]: r["composite"] for r in results}
        assert scores["hive-2"] > scores["hive-1"]
        assert scores["hive-2"] > scores["hive-3"]

    def test_findings_queryable_via_properties_match(self, tmp_path):
        """Stored findings should be queryable by band and status."""
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            await job_inspect(client, reflector, SAMPLE_HIVES)

            result = await client.query(MemoryQuery(
                intent="open findings",
                memory_type="hive",
                properties_match={"status": "open"},
                max_results=10,
            ))
            await client.close()
            return result

        result = run(_test())
        assert len(result.items) >= 1  # At least some findings queryable
        assert all(item.get("status") == "open" for item in result.items if "status" in item)


class TestJobVarroa:
    def test_flags_hives_with_varroa(self, tmp_path):
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            results = await job_varroa(client, reflector, SAMPLE_HIVES)
            await client.close()
            return results

        results = run(_test())
        # hive-1 has varroa 2 (below threshold), hive-2 has 6, hive-3 has 1
        hive_ids = [r["hive_id"] for r in results]
        assert "hive-2" in hive_ids  # varroa 6.0 → scores > 0


class TestJobStores:
    def test_flags_low_stores(self, tmp_path):
        reflector = ObservationReflector()

        async def _test():
            client = HiveBusClient(
                api_url="http://localhost:1",
                snapshot_path=tmp_path / "bus.json",
            )
            await client.connect()
            results = await job_stores(client, reflector, SAMPLE_HIVES)
            await client.close()
            return results

        results = run(_test())
        # hive-2 has "light" stores → should be flagged
        hive_ids = [r["hive_id"] for r in results]
        assert "hive-2" in hive_ids

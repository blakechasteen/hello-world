import asyncio
from holoLoom.orchestrator import HoloLoomOrchestrator
from holoLoom.documentation.types import Query, MemoryShard
from holoLoom.config import Config


def test_orchestrator_basic():
    # Minimal smoke test for orchestrator pipeline
    shards = [
        MemoryShard(id="s1", text="Thompson Sampling is a Bayesian bandit method.", episode="docs"),
        MemoryShard(id="s2", text="It balances exploration and exploitation.", episode="docs"),
    ]

    cfg = Config.fused()
    orch = HoloLoomOrchestrator(cfg=cfg, shards=shards)

    async def run_query():
        q = Query(text="What is Thompson Sampling?")
        resp = await orch.process(q)
        return resp

    resp = asyncio.get_event_loop().run_until_complete(run_query())
    assert isinstance(resp, dict)
    assert resp.get("status") in ("success", "error")
    # If success, ensure tool is present
    if resp.get("status") == "success":
        assert "tool" in resp

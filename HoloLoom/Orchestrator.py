#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HoloLoom Orchestrator - Compatibility Adapter
==============================================
MIGRATION NOTE (2025-10-27):
This file provides backward compatibility during Phase 1 refactor.
The main orchestrator is now weaving_orchestrator.py (9-step cycle).

For new code, import WeavingOrchestrator directly:
    from HoloLoom.weaving_orchestrator import WeavingOrchestrator

This adapter preserves the old .process() API for:
- autospin.py, conversational.py, check_holoLoom.py
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict, dataclass

from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.Documentation.types import Query, MemoryShard
from HoloLoom.config import Config


@dataclass
class ProcessResult:
    """Legacy result object for backward compatibility with demos."""
    status: str
    query: str
    tool: str
    output: str
    confidence: float
    context_shards: int
    metadata: Dict[str, Any]
    trace: Dict[str, Any]
    
    @property
    def response(self) -> str:
        """Alias for output."""
        return self.output
    
    @property
    def context(self):
        """Mock context object for demos."""
        class MockContext:
            def __init__(self, count):
                self.shards = [None] * count
        return MockContext(self.context_shards) if self.context_shards > 0 else None


class HoloLoomOrchestrator:
    """
    Compatibility adapter wrapping WeavingOrchestrator with old .process() API.
    
    Maps old simple orchestrator's process() method to new weave() method.
    """
    
    def __init__(self, cfg: Config, shards: List[MemoryShard]):
        """Initialize adapter."""
        self.cfg = cfg
        self.shards = shards
        self.logger = logging.getLogger(__name__)
        self._weaver = WeavingOrchestrator(cfg=cfg, shards=shards)
    
    async def process(self, query: Query) -> ProcessResult:
        """
        Process query (legacy API).
        
        Calls weave() and converts Spacetime to legacy ProcessResult object.
        """
        try:
            spacetime = await self._weaver.weave(query)
            
            return ProcessResult(
                status="success",
                query=query.text,
                tool=spacetime.tool_used or "answer",
                output=spacetime.response,
                confidence=spacetime.confidence,
                context_shards=spacetime.trace.context_shards_count if spacetime.trace else 0,
                metadata={
                    "pattern": spacetime.metadata.get("pattern", "unknown"),
                    "execution_time_ms": spacetime.metadata.get("duration_ms", 0),
                },
                trace=asdict(spacetime.trace) if spacetime.trace else {}
            )
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return ProcessResult(
                status="error",
                query=query.text,
                tool="error",
                output=str(e),
                confidence=0.0,
                context_shards=0,
                metadata={},
                trace={}
            )
    
    async def __aenter__(self):
        await self._weaver.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return await self._weaver.__aexit__(exc_type, exc_val, exc_tb)
    
    async def close(self):
        await self._weaver.close()


async def run_query(query_text: str, config: Optional[Config] = None, shards: Optional[List[MemoryShard]] = None) -> Dict[str, Any]:
    """Legacy helper function."""
    if config is None:
        config = Config.fast()
    if shards is None:
        shards = []
    
    orchestrator = HoloLoomOrchestrator(cfg=config, shards=shards)
    try:
        return await orchestrator.process(Query(text=query_text))
    finally:
        await orchestrator.close()


__all__ = ['HoloLoomOrchestrator', 'WeavingOrchestrator', 'run_query']

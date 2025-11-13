"""
Simple RAG - Retrieval-Augmented Generation API
================================================

Zero-config RAG wrapper for HoloLoom.

Quick start:
    from HoloLoom.rag import SimpleRAG

    async with SimpleRAG() as rag:
        # Ingest
        await rag.ingest("Thompson Sampling balances exploration/exploitation")

        # Query
        result = await rag.query("What is Thompson Sampling?")

        # Result
        print(result.response)      # LLM-generated answer
        print(result.sources)       # Retrieved sources
        print(result.confidence)    # 0.0-1.0
"""

from HoloLoom.rag.simple_rag import SimpleRAG, RAGResult
from HoloLoom.rag.streaming import StreamToken
from HoloLoom.rag.sql_integration import (
    SQLRAGMixin,
    SQLRAGResult,
    SQLAdapter,
    TextToSQLTranslator,
    QueryIntent,
    SQLQueryMode
)
from HoloLoom.rag.multihop_reasoning import (
    MultiHopRAGMixin,
    MultiHopRAGResult,
    ReasoningPath,
)
from HoloLoom.rag.multiagent_rag import (
    MultiAgentRAG,
    MultiAgentRAGResult,
    AgentResponse,
    ConsensusMethod,
)

__all__ = [
    "SimpleRAG",
    "RAGResult",
    "StreamToken",
    "SQLRAGMixin",
    "SQLRAGResult",
    "SQLAdapter",
    "TextToSQLTranslator",
    "QueryIntent",
    "SQLQueryMode",
    "MultiHopRAGMixin",
    "MultiHopRAGResult",
    "ReasoningPath",
    "MultiAgentRAG",
    "MultiAgentRAGResult",
    "AgentResponse",
    "ConsensusMethod",
]

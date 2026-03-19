"""
Simple RAG - Retrieval-Augmented Generation API
================================================

Zero-config RAG wrapper for HoloLoom.

Quick start:
    from hololoom.rag import SimpleRAG

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

from hololoom.rag.multiagent_rag import (
    AgentResponse,
    ConsensusMethod,
    MultiAgentRAG,
    MultiAgentRAGResult,
)
from hololoom.rag.multihop_reasoning import (
    MultiHopRAGMixin,
    MultiHopRAGResult,
    ReasoningPath,
)
from hololoom.rag.simple_rag import RAGResult, SimpleRAG
from hololoom.rag.sql_integration import (
    QueryIntent,
    SQLAdapter,
    SQLQueryMode,
    SQLRAGMixin,
    SQLRAGResult,
    TextToSQLTranslator,
)
from hololoom.rag.streaming import StreamToken

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

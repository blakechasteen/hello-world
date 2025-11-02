"""
CRM Natural Language Query Service

Natural language query processing using HoloLoom's WeavingOrchestrator.
Enables queries like "find hot leads in fintech" or "contacts like Alice".

Features:
- Full HoloLoom weaving cycle integration
- Pattern detection and semantic routing
- Multi-modal query understanding
- Context-aware result ranking
"""

from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass

from crm_app.models import Contact, Company, Deal, Activity
from crm_app.protocols import CRMService
from crm_app.embedding_service import CRMEmbeddingService
from crm_app.similarity_service import SimilarityService

# HoloLoom imports
from HoloLoom.documentation.types import Query, MemoryShard
from HoloLoom.weaving_orchestrator import WeavingOrchestrator
from HoloLoom.config import Config


@dataclass
class NLQueryResult:
    """
    Natural language query result with semantic intelligence.

    Attributes:
        query: Original natural language query
        intent: Detected intent (similarity, lead_filter, deal_filter, etc.)
        entities: Matching CRM entities (Contact, Company, Deal, Activity)
        relevance_scores: Relevance score for each entity (0-1)
        confidence: Confidence in intent detection (0-1)
        metadata: Processing metadata (timing, method, fallback status)
        trace: Optional execution trace for debugging
    """
    query: str
    intent: str
    entities: List[Any]  # Contacts, Deals, Companies
    relevance_scores: List[float]
    confidence: float  # Confidence in intent detection (0-1)
    metadata: Dict[str, Any]
    trace: Optional[str] = None  # Execution trace for debugging


class NaturalLanguageQueryService:
    """
    Natural language query processing for CRM

    This service integrates HoloLoom's full weaving cycle to process
    natural language queries like:
    - "Find contacts similar to Alice Johnson"
    - "Show me hot leads in the technology sector"
    - "Which deals haven't been contacted in 2 weeks?"
    - "Enterprise companies with open deals over $100k"

    Architecture:
    - Uses WeavingOrchestrator for pattern detection and semantic routing
    - Integrates with CRM service for data access
    - Uses embedding service for semantic matching
    - Returns structured results with provenance
    """

    def __init__(
        self,
        crm_service: CRMService,
        embedding_service: CRMEmbeddingService,
        similarity_service: SimilarityService,
        config: Optional[Config] = None
    ):
        """
        Initialize NL query service

        Args:
            crm_service: CRM data access
            embedding_service: Embedding generation
            similarity_service: Similarity search
            config: HoloLoom config (default: FAST mode)
        """
        self.crm = crm_service
        self.embeddings = embedding_service
        self.similarity = similarity_service
        self.config = config or Config.fast()  # Use FAST mode for balance

        # Orchestrator will be initialized when needed
        self._orchestrator = None
        self._shards_cache = None
        self._cache_timestamp = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_orchestrator()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Clean up resources
        self._orchestrator = None
        return False

    # ========================================================================
    # Core Query Processing
    # ========================================================================

    async def query(
        self,
        natural_language: str,
        max_results: int = 10,
        use_orchestrator: bool = True
    ) -> NLQueryResult:
        """
        Process natural language query

        Args:
            natural_language: User's natural language query
            max_results: Maximum number of results to return
            use_orchestrator: Use full HoloLoom orchestrator (vs simple fallback)

        Returns:
            NLQueryResult with entities and metadata

        Example:
            ```python
            async with NaturalLanguageQueryService(crm, embeddings, similarity) as nl:
                result = await nl.query("find hot leads in fintech")

                for entity in result.entities:
                    print(f"{entity.name}: {result.relevance_scores[i]}")
            ```
        """
        if use_orchestrator:
            return await self._query_with_orchestrator(natural_language, max_results)
        else:
            return await self._query_simple_fallback(natural_language, max_results)

    async def _query_with_orchestrator(
        self,
        natural_language: str,
        max_results: int
    ) -> NLQueryResult:
        """
        Query using full HoloLoom WeavingOrchestrator

        This is the full integration that leverages:
        - Pattern detection (motifs)
        - Semantic routing (multi-scale retrieval)
        - Knowledge graph traversal
        - Decision engine for ranking
        """
        # Ensure orchestrator is initialized
        await self._ensure_orchestrator()

        # Create HoloLoom query
        query = Query(
            text=natural_language,
            metadata={
                "source": "crm_nl_query",
                "max_results": max_results
            }
        )

        try:
            # Process with full weaving cycle
            spacetime = await self._orchestrator.weave(query)

            # Extract entities from spacetime
            entities, scores = self._extract_entities_from_spacetime(spacetime)

            # Determine intent from spacetime
            intent = self._infer_intent(spacetime, natural_language)

            return NLQueryResult(
                query=natural_language,
                intent=intent,
                entities=entities[:max_results],
                relevance_scores=scores[:max_results],
                metadata={
                    "used_orchestrator": True,
                    "pattern_card": getattr(spacetime, "pattern_card", "FAST"),
                    "execution_time": getattr(spacetime, "execution_time", None)
                },
                trace=str(getattr(spacetime, "trace", ""))
            )

        except Exception as e:
            # Fallback to simple processing on error
            print(f"[NL Query] Orchestrator failed, using fallback: {e}")
            return await self._query_simple_fallback(natural_language, max_results)

    async def _query_simple_fallback(
        self,
        natural_language: str,
        max_results: int
    ) -> NLQueryResult:
        """
        Simple fallback query processing (no orchestrator)

        Uses basic intent detection and semantic search.
        """
        nl_lower = natural_language.lower()

        # Simple intent detection
        if "similar" in nl_lower or "like" in nl_lower:
            return await self._handle_similarity_query(natural_language, max_results)

        elif "hot" in nl_lower or "warm" in nl_lower or "cold" in nl_lower:
            return await self._handle_lead_filter_query(natural_language, max_results)

        elif any(ind in nl_lower for ind in ["technology", "fintech", "finance", "saas", "enterprise"]):
            return await self._handle_industry_query(natural_language, max_results)

        elif "deal" in nl_lower or "pipeline" in nl_lower:
            return await self._handle_deal_query(natural_language, max_results)

        else:
            # Default: semantic search
            return await self._handle_semantic_search(natural_language, max_results)

    # ========================================================================
    # Intent Handlers (Fallback Mode)
    # ========================================================================

    async def _handle_similarity_query(self, query: str, limit: int) -> NLQueryResult:
        """Handle 'find contacts like X' queries"""
        # Extract entity name (simple pattern matching)
        words = query.split()
        name_idx = -1

        for i, word in enumerate(words):
            if word.lower() in ["like", "similar"]:
                name_idx = i + 1
                break

        if name_idx > 0 and name_idx < len(words):
            name = words[name_idx]

            # Find contact by name
            contacts = self.crm.contacts.list()
            target = next((c for c in contacts if name.lower() in c.name.lower()), None)

            if target:
                similar = self.similarity.find_similar_contacts(target.id, limit=limit)

                return NLQueryResult(
                    query=query,
                    intent="similarity",
                    entities=[r.entity for r in similar],
                    relevance_scores=[r.similarity for r in similar],
                    metadata={"target": target.name, "target_id": target.id}
                )

        # Fallback to semantic search
        return await self._handle_semantic_search(query, limit)

    async def _handle_lead_filter_query(self, query: str, limit: int) -> NLQueryResult:
        """Handle lead engagement filtering"""
        # Determine threshold
        if "hot" in query.lower():
            threshold = 0.75
            engagement = "hot"
        elif "warm" in query.lower():
            threshold = 0.50
            engagement = "warm"
        elif "cold" in query.lower():
            threshold = 0.25
            engagement = "cold"
        else:
            threshold = 0.0
            engagement = "all"

        # Filter contacts
        contacts = [
            c for c in self.crm.contacts.list()
            if c.lead_score and c.lead_score >= threshold
        ]

        # Sort by score
        contacts.sort(key=lambda c: c.lead_score or 0, reverse=True)

        scores = [c.lead_score or 0.0 for c in contacts]

        return NLQueryResult(
            query=query,
            intent="lead_filter",
            entities=contacts[:limit],
            relevance_scores=scores[:limit],
            metadata={"engagement": engagement, "threshold": threshold}
        )

    async def _handle_industry_query(self, query: str, limit: int) -> NLQueryResult:
        """Handle industry-based queries"""
        # Extract industry
        industries = ["technology", "fintech", "finance", "saas", "enterprise"]
        query_lower = query.lower()

        matched = next((ind for ind in industries if ind in query_lower), None)

        if matched:
            # Find companies in industry
            companies = [
                c for c in self.crm.companies.list()
                if matched in c.industry.lower()
            ]

            # Get contacts at those companies
            contacts = []
            for company in companies:
                contacts.extend(self.crm.contacts.list({"company_id": company.id}))

            # Relevance based on lead score
            scores = [c.lead_score or 0.5 for c in contacts]

            # Sort by score
            sorted_pairs = sorted(zip(contacts, scores), key=lambda x: x[1], reverse=True)
            contacts, scores = zip(*sorted_pairs) if sorted_pairs else ([], [])

            return NLQueryResult(
                query=query,
                intent="industry_filter",
                entities=list(contacts[:limit]),
                relevance_scores=list(scores[:limit]),
                metadata={"industry": matched, "company_count": len(companies)}
            )

        return await self._handle_semantic_search(query, limit)

    async def _handle_deal_query(self, query: str, limit: int) -> NLQueryResult:
        """Handle deal-related queries"""
        deals = self.crm.deals.list({"open_only": True})

        # Parse filters from query
        if "over" in query.lower() or ">" in query:
            # Value filter (simple)
            import re
            amounts = re.findall(r'\$?(\d+)k', query.lower())
            if amounts:
                min_value = int(amounts[0]) * 1000
                deals = [d for d in deals if d.value >= min_value]

        # Sort by value * probability
        deals.sort(key=lambda d: d.value * d.probability, reverse=True)
        scores = [d.value * d.probability / 100000 for d in deals]  # Normalize

        return NLQueryResult(
            query=query,
            intent="deal_filter",
            entities=deals[:limit],
            relevance_scores=scores[:limit],
            metadata={"deal_count": len(deals)}
        )

    async def _handle_semantic_search(self, query: str, limit: int) -> NLQueryResult:
        """Handle generic semantic search"""
        # Use similarity service
        results = self.similarity.search_by_text(query, entity_type="contact", limit=limit)

        return NLQueryResult(
            query=query,
            intent="semantic_search",
            entities=[r.entity for r in results],
            relevance_scores=[r.similarity for r in results],
            metadata={"search_type": "semantic"}
        )

    # ========================================================================
    # HoloLoom Integration Helpers
    # ========================================================================

    async def _ensure_orchestrator(self):
        """Ensure orchestrator is initialized with current CRM data"""
        # Check if we need to refresh (cache for 5 minutes)
        import time
        current_time = time.time()

        if (self._orchestrator is None or
            self._cache_timestamp is None or
            (current_time - self._cache_timestamp) > 300):  # 5 minutes

            # Get fresh memory shards from CRM
            shards = self.crm.get_memory_shards()

            # Create orchestrator
            self._orchestrator = WeavingOrchestrator(
                cfg=self.config,
                shards=shards
            )

            self._shards_cache = shards
            self._cache_timestamp = current_time

    def _extract_entities_from_spacetime(self, spacetime) -> tuple[List[Any], List[float]]:
        """
        Extract CRM entities from spacetime result

        This maps HoloLoom's MemoryShards back to CRM domain entities.
        """
        entities = []
        scores = []

        # Check if spacetime has retrieved shards
        if hasattr(spacetime, 'retrieved_shards'):
            for shard in spacetime.retrieved_shards[:20]:  # Top 20
                entity = self._shard_to_entity(shard)
                if entity:
                    entities.append(entity)
                    scores.append(shard.metadata.get('relevance', 1.0))

        # Fallback: check if spacetime has direct results
        elif hasattr(spacetime, 'results'):
            for result in spacetime.results[:20]:
                if hasattr(result, 'entity'):
                    entities.append(result.entity)
                    scores.append(getattr(result, 'score', 1.0))

        return entities, scores

    def _shard_to_entity(self, shard: MemoryShard) -> Optional[Any]:
        """Convert MemoryShard back to CRM entity"""
        entity_type = shard.metadata.get('entity_type')
        entity_id = shard.metadata.get('id') or shard.id

        if not entity_type or not entity_id:
            return None

        if entity_type == 'contact':
            return self.crm.contacts.get(entity_id)
        elif entity_type == 'company':
            return self.crm.companies.get(entity_id)
        elif entity_type == 'deal':
            return self.crm.deals.get(entity_id)
        elif entity_type == 'activity':
            return self.crm.activities.get(entity_id)

        return None

    def _infer_intent(self, spacetime, query_text: str) -> str:
        """Infer query intent from spacetime and query text"""
        # Check motifs detected by HoloLoom
        if hasattr(spacetime, 'motifs'):
            motifs = [m.pattern for m in spacetime.motifs]

            if any(m in ["similar", "like", "related"] for m in motifs):
                return "similarity"
            elif any(m in ["hot", "warm", "qualified"] for m in motifs):
                return "lead_filter"
            elif any(m in ["deal", "pipeline", "sales"] for m in motifs):
                return "deal_query"

        # Fallback to text analysis
        query_lower = query_text.lower()
        if "similar" in query_lower or "like" in query_lower:
            return "similarity"
        elif "deal" in query_lower:
            return "deal_query"
        elif any(x in query_lower for x in ["hot", "warm", "cold", "lead"]):
            return "lead_filter"

        return "general_search"


# ============================================================================
# Convenience Functions
# ============================================================================

def create_nl_query_service(
    crm_service: CRMService,
    embedding_service: Optional[CRMEmbeddingService] = None,
    similarity_service: Optional[SimilarityService] = None,
    config: Optional[Config] = None
) -> NaturalLanguageQueryService:
    """
    Factory function to create NL query service

    Args:
        crm_service: CRM data access service
        embedding_service: Optional embedding service
        similarity_service: Optional similarity service
        config: Optional HoloLoom config

    Returns:
        Configured NaturalLanguageQueryService
    """
    if embedding_service is None:
        from crm_app.embedding_service import create_embedding_service
        embedding_service = create_embedding_service()

    if similarity_service is None:
        from crm_app.similarity_service import create_similarity_service
        similarity_service = create_similarity_service(crm_service, embedding_service)

    return NaturalLanguageQueryService(
        crm_service,
        embedding_service,
        similarity_service,
        config
    )

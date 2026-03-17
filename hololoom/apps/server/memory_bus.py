"""
Memory Bus API — Cross-Agent Observation Store
===============================================

Lightweight memory layer for agent observations. Agents write observations
after each response; any agent can search/read observations from any other.

Storage: SQLite (memory_bus.db in data/).
Search: Keyword-based (basic) + TF-IDF with temporal decay (semantic).

Created: 2026-03-13
"""

import math
import os
import re
import sqlite3
import logging
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory-bus"])

# ============================================================================
# Database
# ============================================================================

MEMORY_DB_PATH = os.environ.get(
    "MEMORY_BUS_DB",
    str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "memory_bus.db"),
)


class ObservationStore:
    """SQLite-backed observation store for cross-agent memory."""

    def __init__(self, db_path: str = MEMORY_DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_observations (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                room_id TEXT,
                content TEXT NOT NULL,
                tags TEXT,
                created_at TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_agent "
            "ON memory_observations(agent)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_created "
            "ON memory_observations(created_at)"
        )
        self._conn.commit()
        count = self._conn.execute(
            "SELECT COUNT(*) FROM memory_observations"
        ).fetchone()[0]
        logger.info("Memory bus: %s (%d observations)", db_path, count)

    def write(
        self,
        agent: str,
        content: str,
        tags: List[str] | None = None,
        room_id: str | None = None,
        relevance_score: float = 1.0,
    ) -> str:
        """Store an observation. Returns the generated ID."""
        obs_id = str(uuid.uuid4())[:12]
        self._conn.execute(
            "INSERT INTO memory_observations "
            "(id, agent, room_id, content, tags, created_at, relevance_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                obs_id,
                agent,
                room_id,
                content,
                ",".join(tags) if tags else None,
                datetime.now().isoformat(),
                relevance_score,
            ),
        )
        self._conn.commit()
        return obs_id

    def search(
        self,
        query: str,
        agent: str | None = None,
        limit: int = 10,
    ) -> List[dict]:
        """Keyword search with tag boost (3x weight)."""
        terms = query.lower().split()
        if not terms:
            return []

        # Fetch candidates (optionally filtered by agent)
        if agent:
            rows = self._conn.execute(
                "SELECT * FROM memory_observations WHERE agent = ? "
                "ORDER BY created_at DESC LIMIT 500",
                (agent,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_observations "
                "ORDER BY created_at DESC LIMIT 500",
            ).fetchall()

        scored = []
        for row in rows:
            content_lower = row["content"].lower()
            tags_lower = (row["tags"] or "").lower()
            score = 0.0
            for term in terms:
                if term in content_lower:
                    score += 1.0
                if term in tags_lower:
                    score += 3.0  # Tag matches worth 3x
            if score > 0:
                scored.append((score * row["relevance_score"], dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]

    def recent(
        self,
        agent: str | None = None,
        limit: int = 20,
    ) -> List[dict]:
        """Get most recent observations, optionally filtered by agent."""
        if agent:
            rows = self._conn.execute(
                "SELECT * FROM memory_observations WHERE agent = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (agent, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_observations "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def count(self, agent: str | None = None) -> int:
        if agent:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memory_observations WHERE agent = ?",
                (agent,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM memory_observations"
            ).fetchone()
        return row[0] if row else 0

    # -- TF-IDF semantic search (Phase 2) --

    def search_semantic(
        self,
        query: str,
        agent: str | None = None,
        limit: int = 10,
        decay_half_life: float = 30.0,
    ) -> List[dict]:
        """TF-IDF search with bigrams, temporal decay, and tag boost."""
        query_terms = _tokenize(query)
        if not query_terms:
            return []

        # Fetch candidates
        if agent:
            rows = self._conn.execute(
                "SELECT * FROM memory_observations WHERE agent = ? "
                "ORDER BY created_at DESC LIMIT 2000",
                (agent,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_observations "
                "ORDER BY created_at DESC LIMIT 2000",
            ).fetchall()

        if not rows:
            return []

        # Build TF-IDF index
        scorer = TFIDFScorer()
        docs = [(row["id"], row["content"] + " " + (row["tags"] or "")) for row in rows]
        scorer.fit(docs)

        now = datetime.now()
        scored = []
        for row in rows:
            doc_id = row["id"]
            tfidf_score = scorer.score(query_terms, doc_id)
            if tfidf_score <= 0:
                continue

            # Tag boost: 2x for tag matches
            tags_lower = (row["tags"] or "").lower()
            tag_boost = 1.0
            for term in query_terms:
                if "_" not in term and term in tags_lower:
                    tag_boost += 1.0  # +1x per matching tag term

            # Temporal decay: exp(-age_days / half_life)
            try:
                created = datetime.fromisoformat(row["created_at"])
                age_days = (now - created).total_seconds() / 86400
            except (ValueError, TypeError):
                age_days = 0.0
            decay = math.exp(-age_days / decay_half_life) if decay_half_life > 0 else 1.0

            final_score = tfidf_score * tag_boost * decay * row["relevance_score"]
            scored.append((final_score, dict(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:limit]]


# ============================================================================
# TF-IDF Scorer (pure Python, no external deps)
# ============================================================================

_STOPWORDS = frozenset(
    "a an the is it in on at to for of and or but not with by from as be this "
    "that was were are been has had do does did will would can could should may "
    "might shall into than so if its my your his her we they them our what which "
    "who when where how all each every some any no more most other than also just "
    "about over after before between through during".split()
)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into unigrams + bigrams, stripping stopwords."""
    words = [w for w in re.split(r'\W+', text.lower()) if len(w) >= 2 and w not in _STOPWORDS]
    # Generate bigrams
    bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
    return words + bigrams


class TFIDFScorer:
    """In-memory TF-IDF scorer with bigram support."""

    def __init__(self):
        self._doc_freq: Counter = Counter()  # term -> num docs containing it
        self._doc_terms: dict[str, Counter] = {}  # doc_id -> term frequencies
        self._num_docs: int = 0

    def fit(self, documents: List[Tuple[str, str]]) -> None:
        """Build index from (doc_id, text) pairs."""
        self._doc_freq.clear()
        self._doc_terms.clear()
        self._num_docs = len(documents)

        for doc_id, text in documents:
            terms = _tokenize(text)
            term_counts = Counter(terms)
            self._doc_terms[doc_id] = term_counts
            # Document frequency: count each term once per doc
            for term in set(terms):
                self._doc_freq[term] += 1

    def score(self, query_terms: List[str], doc_id: str) -> float:
        """Score a document against query terms using TF-IDF."""
        if doc_id not in self._doc_terms or self._num_docs == 0:
            return 0.0

        doc_terms = self._doc_terms[doc_id]
        total_terms = sum(doc_terms.values()) or 1
        score = 0.0

        for term in query_terms:
            tf = doc_terms.get(term, 0) / total_terms
            df = self._doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(self._num_docs / df)
            score += tf * idf

        return score


def _extract_keywords(context: str, top_k: int = 5) -> List[str]:
    """Extract top keywords from a context string (unigrams only)."""
    words = [w for w in re.split(r'\W+', context.lower()) if len(w) >= 2 and w not in _STOPWORDS]
    counts = Counter(words)
    return [word for word, _ in counts.most_common(top_k)]


# Singleton
_store: Optional[ObservationStore] = None


def get_store() -> ObservationStore:
    global _store
    if _store is None:
        _store = ObservationStore()
    return _store


# ============================================================================
# Request / Response Models
# ============================================================================

class WriteRequest(BaseModel):
    agent: str = Field(..., min_length=1, max_length=50)
    content: str = Field(..., min_length=1, max_length=10000)
    tags: List[str] = Field(default_factory=list)
    room_id: Optional[str] = None


class WriteResponse(BaseModel):
    id: str
    agent: str
    total_observations: int


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1, max_length=500)
    agent: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=100)


class ObservationOut(BaseModel):
    id: str
    agent: str
    room_id: Optional[str]
    content: str
    tags: Optional[str]
    created_at: str
    relevance_score: float


class SearchResponse(BaseModel):
    observations: List[ObservationOut]
    total: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/write", response_model=WriteResponse)
async def write_observation(request: WriteRequest):
    """Store an agent observation."""
    store = get_store()
    obs_id = store.write(
        agent=request.agent,
        content=request.content,
        tags=request.tags,
        room_id=request.room_id,
    )
    return WriteResponse(
        id=obs_id,
        agent=request.agent,
        total_observations=store.count(request.agent),
    )


@router.get("/search", response_model=SearchResponse)
async def search_observations(
    q: str,
    agent: Optional[str] = None,
    limit: int = 10,
):
    """Search observations by keyword."""
    store = get_store()
    results = store.search(query=q, agent=agent, limit=limit)
    return SearchResponse(
        observations=[ObservationOut(**r) for r in results],
        total=len(results),
    )


@router.get("/recent", response_model=SearchResponse)
async def recent_observations(
    agent: Optional[str] = None,
    limit: int = 20,
):
    """Get most recent observations."""
    store = get_store()
    results = store.recent(agent=agent, limit=limit)
    return SearchResponse(
        observations=[ObservationOut(**r) for r in results],
        total=len(results),
    )


@router.get("/relevant", response_model=SearchResponse)
async def relevant_observations(
    context: str,
    agent: Optional[str] = None,
    limit: int = 5,
    decay_half_life: float = 30.0,
):
    """Find relevant observations by auto-extracting keywords from context."""
    store = get_store()
    # Auto-extract keywords then search semantically
    keywords = _extract_keywords(context)
    if not keywords:
        return SearchResponse(observations=[], total=0)
    query = " ".join(keywords)
    results = store.search_semantic(
        query=query, agent=agent, limit=limit, decay_half_life=decay_half_life,
    )
    return SearchResponse(
        observations=[ObservationOut(**r) for r in results],
        total=len(results),
    )


@router.get("/status")
async def memory_status():
    """Memory bus status."""
    store = get_store()
    return {
        "total_observations": store.count(),
        "agents": {
            agent: store.count(agent)
            for agent in ["promptly", "elle", "coz"]
        },
        "db_path": store.db_path,
    }

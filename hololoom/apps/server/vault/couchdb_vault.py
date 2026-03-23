"""
CouchDB vault adapter — Obsidian Live Sync compatible.

Gives agents their own persistent PARA vault synced via CouchDB.
Documents are stored in Live Sync format so they're browsable in Obsidian.

Design Principle 1: Protocol First — AgentVault is the contract.
Design Principle 2: Graceful Degradation — works without CouchDB.

Usage:
    vault = CouchDBVault(url, database, username, password)
    await vault.connect()
    await vault.write_note("01_Projects/research.md", "# Research\n\n...")
    results = await vault.search_notes("research")
    await vault.disconnect()
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)

# Live Sync constants
MAX_DOC_SIZE = 1000          # chars per chunk
CHUNK_ID_PREFIX = "h:"


# --- Protocol ---

@runtime_checkable
class AgentVault(Protocol):
    """Protocol for agent-owned vault backends."""

    async def connect(self) -> bool: ...
    async def disconnect(self) -> None: ...
    async def write_note(self, rel_path: str, content: str) -> bool: ...
    async def read_note(self, rel_path: str) -> str | None: ...
    async def delete_note(self, rel_path: str) -> bool: ...
    async def search_notes(self, query: str, top_k: int = 5) -> list[dict]: ...
    async def list_notes(self) -> list[dict]: ...
    def is_connected(self) -> bool: ...


# --- Live Sync helpers ---

def _path_to_id(rel_path: str) -> str:
    """Convert path to CouchDB doc ID (lowercase, Live Sync convention)."""
    normalized = rel_path.replace("\\", "/").lower()
    if normalized.startswith("_"):
        normalized = "/" + normalized
    return normalized


def _id_to_path(doc_id: str) -> str:
    """Convert CouchDB doc ID back to path."""
    path = doc_id
    if path.startswith("/"):
        path = path[1:]
    return path


def _chunk_content(content: str) -> list[tuple[str, str]]:
    """Split content into (chunk_id, chunk_data) pairs."""
    if not content:
        return []
    chunks = []
    for i in range(0, len(content), MAX_DOC_SIZE):
        data = content[i:i + MAX_DOC_SIZE]
        content_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]
        chunks.append((f"{CHUNK_ID_PREFIX}{content_hash}", data))
    return chunks


def _quote_id(doc_id: str) -> str:
    """URL-encode document ID for CouchDB REST API."""
    return doc_id.replace("/", "%2F").replace(":", "%3A")


# --- Implementation ---

class CouchDBVault:
    """CouchDB vault backend with Obsidian Live Sync compatible format.

    Each note is stored as a CouchDB document with chunked content.
    Documents use the Live Sync format so Obsidian can browse the vault.
    """

    def __init__(
        self,
        url: str,
        database: str,
        username: str | None = None,
        password: str | None = None,
    ):
        self._url = url.rstrip("/")
        self._db = database
        self._auth = aiohttp.BasicAuth(username, password) if HAS_AIOHTTP and username and password else None
        self._session: aiohttp.ClientSession | None = None
        self._connected = False

    @property
    def _db_url(self) -> str:
        return f"{self._url}/{self._db}"

    async def connect(self) -> bool:
        """Connect to CouchDB and verify database access."""
        if not HAS_AIOHTTP:
            logger.warning("aiohttp not installed — CouchDB vault disabled")
            return False

        try:
            self._session = aiohttp.ClientSession(auth=self._auth)

            async with self._session.get(self._url) as resp:
                if resp.status != 200:
                    logger.warning(f"CouchDB not reachable at {self._url}: {resp.status}")
                    await self._session.close()
                    self._session = None
                    return False

            async with self._session.get(self._db_url) as resp:
                if resp.status == 200:
                    pass
                elif resp.status == 404:
                    logger.warning(f"Database '{self._db}' not found")
                    await self._session.close()
                    self._session = None
                    return False
                elif resp.status == 401:
                    logger.warning("CouchDB auth failed")
                    await self._session.close()
                    self._session = None
                    return False

            self._connected = True
            logger.info(f"Connected to CouchDB vault: {self._db_url}")
            return True

        except aiohttp.ClientError as e:
            logger.warning(f"CouchDB connection failed: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            return False

    async def disconnect(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._session is not None

    async def write_note(self, rel_path: str, content: str) -> bool:
        """Write a note to CouchDB in Live Sync format."""
        if not self.is_connected():
            return False

        try:
            doc_id = _path_to_id(rel_path)
            chunks = _chunk_content(content)

            # Upsert chunks (content-addressable, idempotent)
            for chunk_id, chunk_data in chunks:
                await self._upsert_chunk(chunk_id, chunk_data)

            # Build note document
            now_ms = int(time.time() * 1000)
            note_doc = {
                "_id": doc_id,
                "path": rel_path.replace("\\", "/"),
                "type": "plain",
                "ctime": now_ms,
                "mtime": now_ms,
                "size": len(content.encode("utf-8")),
                "children": [cid for cid, _ in chunks],
                "eden": {},
            }

            # Fetch existing rev
            existing = await self._get_doc(doc_id)
            if existing:
                note_doc["_rev"] = existing["_rev"]
                note_doc["ctime"] = existing.get("ctime", now_ms)

            async with self._session.put(
                f"{self._db_url}/{_quote_id(doc_id)}",
                json=note_doc,
            ) as resp:
                if resp.status in (201, 202):
                    logger.debug(f"Wrote: {rel_path}")
                    return True
                else:
                    body = await resp.text()
                    logger.warning(f"Write failed for {rel_path}: {resp.status} {body}")
                    return False

        except Exception as e:
            logger.warning(f"Write error for {rel_path}: {e}")
            return False

    async def read_note(self, rel_path: str) -> str | None:
        """Read a note from CouchDB, reassembling chunks."""
        if not self.is_connected():
            return None

        try:
            doc_id = _path_to_id(rel_path)
            doc = await self._get_doc(doc_id)
            if not doc or doc.get("deleted", False):
                return None

            children = doc.get("children", [])
            if not children:
                return ""

            parts = []
            for chunk_id in children:
                chunk = await self._get_doc(chunk_id)
                if chunk:
                    parts.append(chunk.get("data", ""))

            return "".join(parts)

        except Exception as e:
            logger.warning(f"Read error for {rel_path}: {e}")
            return None

    async def delete_note(self, rel_path: str) -> bool:
        """Soft-delete a note in CouchDB."""
        if not self.is_connected():
            return False

        try:
            doc_id = _path_to_id(rel_path)
            doc = await self._get_doc(doc_id)
            if not doc:
                return True

            doc["deleted"] = True
            doc["mtime"] = int(time.time() * 1000)

            async with self._session.put(
                f"{self._db_url}/{_quote_id(doc_id)}",
                json=doc,
            ) as resp:
                return resp.status in (201, 202)

        except Exception as e:
            logger.warning(f"Delete error for {rel_path}: {e}")
            return False

    async def search_notes(self, query: str, top_k: int = 5) -> list[dict]:
        """Keyword search over vault notes in CouchDB.

        Fetches all note documents, reassembles content, scores by term frequency.
        """
        if not self.is_connected():
            return []

        try:
            terms = [t.lower() for t in re.split(r'\W+', query) if len(t) >= 2]
            if not terms:
                return []

            # Fetch all note docs
            async with self._session.get(
                f"{self._db_url}/_all_docs",
                params={"include_docs": "true"},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            scored = []
            for row in data.get("rows", []):
                doc = row.get("doc", {})
                doc_type = doc.get("type", "")
                doc_id = doc.get("_id", "")

                if doc_type not in ("plain", "newnote", "notes"):
                    continue
                if doc_id.startswith("_") or doc_id.startswith(CHUNK_ID_PREFIX):
                    continue
                if doc.get("deleted", False):
                    continue

                path = doc.get("path", _id_to_path(doc_id))
                title = path.rsplit("/", 1)[-1].replace(".md", "").replace("-", " ")

                # Reassemble content for scoring
                children = doc.get("children", [])
                content = ""
                for chunk_id in children:
                    chunk = await self._get_doc(chunk_id)
                    if chunk:
                        content += chunk.get("data", "")

                content_lower = content.lower()
                score = 0
                for term in terms:
                    if term in title.lower():
                        score += 3
                    score += content_lower.count(term)

                if score > 0:
                    snippet = content[:200] if content else ""
                    scored.append({
                        "path": path,
                        "title": title,
                        "score": score,
                        "snippet": snippet,
                    })

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        except Exception as e:
            logger.warning(f"Search error: {e}")
            return []

    async def list_notes(self) -> list[dict]:
        """List all notes in the vault."""
        if not self.is_connected():
            return []

        try:
            async with self._session.get(
                f"{self._db_url}/_all_docs",
                params={"include_docs": "true"},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            notes = []
            for row in data.get("rows", []):
                doc = row.get("doc", {})
                doc_type = doc.get("type", "")
                doc_id = doc.get("_id", "")

                if doc_type not in ("plain", "newnote", "notes"):
                    continue
                if doc_id.startswith("_") or doc_id.startswith(CHUNK_ID_PREFIX):
                    continue
                if doc.get("deleted", False):
                    continue

                path = doc.get("path", _id_to_path(doc_id))
                notes.append({
                    "path": path,
                    "title": path.rsplit("/", 1)[-1].replace(".md", "").replace("-", " "),
                    "mtime": doc.get("mtime", 0),
                })

            return notes

        except Exception as e:
            logger.warning(f"List error: {e}")
            return []

    # --- Internal ---

    async def _upsert_chunk(self, chunk_id: str, data: str) -> bool:
        try:
            async with self._session.head(
                f"{self._db_url}/{_quote_id(chunk_id)}"
            ) as resp:
                if resp.status == 200:
                    return True  # Already exists (content-addressed)

            doc = {"_id": chunk_id, "type": "leaf", "data": data}
            async with self._session.put(
                f"{self._db_url}/{_quote_id(chunk_id)}",
                json=doc,
            ) as resp:
                return resp.status in (201, 202)

        except Exception as e:
            logger.warning(f"Chunk upsert error {chunk_id}: {e}")
            return False

    async def _get_doc(self, doc_id: str) -> dict | None:
        try:
            async with self._session.get(
                f"{self._db_url}/{_quote_id(doc_id)}"
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception:
            pass
        return None


# --- Singleton factory ---

_vault: CouchDBVault | None = None
_vault_init_attempted = False


async def get_agent_vault() -> CouchDBVault | None:
    """Get the agent vault singleton. Returns None if not configured.

    Graceful degradation: if CouchDB is unavailable, returns None.
    Callers must check before using.
    """
    global _vault, _vault_init_attempted

    if _vault is not None:
        return _vault if _vault.is_connected() else None

    if _vault_init_attempted:
        return None  # Don't retry if we already failed

    _vault_init_attempted = True

    url = os.environ.get("PROMPTLY_COUCHDB_URL")
    if not url:
        logger.debug("PROMPTLY_COUCHDB_URL not set — agent vault disabled")
        return None

    _vault = CouchDBVault(
        url=url,
        database=os.environ.get("PROMPTLY_COUCHDB_DB", "promptly-vault"),
        username=os.environ.get("PROMPTLY_COUCHDB_USER"),
        password=os.environ.get("PROMPTLY_COUCHDB_PASS"),
    )

    connected = await _vault.connect()
    if not connected:
        logger.warning("Agent vault (CouchDB) unavailable — working without it")
        _vault = None
        return None

    return _vault
